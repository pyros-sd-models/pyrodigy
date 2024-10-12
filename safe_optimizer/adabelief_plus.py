import math

import torch
from loguru import logger
from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class adabelief_plus(BaseOptimizer):
    """Customized AdaBelief with additional logging for debugging."""

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = True,
        ams_bound: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-16,
        **kwargs,
    ):
        # Logging parameters at initialization
        logger.debug(
            f"Initializing AdaBeliefPlus with lr={lr}, betas={betas}, weight_decay={weight_decay}, "
            f"weight_decouple={weight_decouple}, fixed_decay={fixed_decay}, rectify={rectify}, "
            f"n_sma_threshold={n_sma_threshold}, degenerated_to_sgd={degenerated_to_sgd}, "
            f"ams_bound={ams_bound}, r={r}, adanorm={adanorm}, adam_debias={adam_debias}, eps={eps}"
        )

        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

        defaults: DEFAULTS = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "rectify": rectify,
            "ams_bound": ams_bound,
            "adanorm": adanorm,
            "adam_debias": adam_debias,
            "eps": eps,
        }
        if adanorm:
            defaults.update({"r": r})

        super().__init__(params, defaults)

    @staticmethod
    def apply_weight_decay(
        p, grad, lr, weight_decay, weight_decouple, fixed_decay, ratio=None
    ):
        """
        Apply weight decay in a way that avoids in-place operations on leaf tensors.

        :param p: torch.Tensor. parameter.
        :param grad: torch.Tensor. gradient.
        :param lr: float. learning rate.
        :param weight_decay: float. weight decay.
        :param weight_decouple: bool. whether to decouple weight decay.
        :param fixed_decay: bool. whether to fix weight decay.
        :param ratio: Optional[float]. scale weight decay.
        """
        if weight_decouple:
            # Use out-of-place operation to update `p`
            decay_factor = 1.0 - weight_decay * (1.0 if fixed_decay else lr) * (
                ratio if ratio is not None else 1.0
            )
            p = p * decay_factor  # Assign to `p` without modifying it in-place
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)
        return p

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            logger.debug(
                f"Processing parameter group with lr={group['lr']} and betas={group['betas']}"
            )

            group["step"] = group.get("step", 0) + 1
            beta1, beta2 = group["betas"]
            bias_correction1 = self.debias(beta1, group["step"])
            bias_correction2_sq = math.sqrt(self.debias(beta2, group["step"]))

            logger.debug(
                f"Step {group['step']}: bias_correction1={bias_correction1}, bias_correction2_sq={bias_correction2_sq}"
            )

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group["rectify"],
                step=group["step"],
                lr=group["lr"],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )
            logger.debug(f"Step {group['step']}: step_size={step_size}, n_sma={n_sma}")

            step_size = self.apply_adam_debias(
                group["adam_debias"], step_size, bias_correction1
            )
            logger.debug(f"Step {group['step']}: adjusted step_size={step_size}")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                logger.debug(
                    f"Processing parameter with grad norm {torch.linalg.norm(grad)}"
                )

                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_var"] = torch.zeros_like(p)
                    if group["adanorm"]:
                        state["exp_grad_norm"] = torch.zeros(
                            (1,), dtype=grad.dtype, device=grad.device
                        )
                    if group["ams_bound"]:
                        state["max_exp_avg_var"] = torch.zeros_like(p)
                    logger.debug("Initialized state for parameter")

                p = self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    weight_decouple=group["weight_decouple"],
                    fixed_decay=group["fixed_decay"],
                )

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group["adanorm"],
                    exp_grad_norm=state.get("exp_grad_norm", None),
                    r=group.get("r", None),
                )
                logger.debug(
                    f"Step {group['step']}: s_grad norm {torch.linalg.norm(s_grad)}"
                )

                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1.0 - beta2
                ).add_(group["eps"])
                logger.debug(
                    f"Step {group['step']}: exp_avg norm {torch.linalg.norm(exp_avg)}, exp_avg_var norm {torch.linalg.norm(exp_avg_var)}"
                )

                de_nom = self.apply_ams_bound(
                    ams_bound=group["ams_bound"],
                    exp_avg_sq=exp_avg_var,
                    max_exp_avg_sq=state.get("max_exp_avg_var", None),
                    eps=group["eps"],
                )

                if not group["rectify"]:
                    de_nom.div_(bias_correction2_sq)
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                elif n_sma >= self.n_sma_threshold:
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                elif step_size > 0:
                    p.add_(exp_avg, alpha=-step_size)

                logger.debug(f"Step {group['step']}: parameter updated successfully")

        return loss

    @torch.no_grad()
    def reset(self):
        # Implement reset functionality here
        for group in self.param_groups:
            group["step"] = 0
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_var"] = torch.zeros_like(p)
                if group["adanorm"]:
                    state["exp_grad_norm"] = torch.zeros(
                        (1,), dtype=p.dtype, device=p.device
                    )
                if group["ams_bound"]:
                    state["max_exp_avg_var"] = torch.zeros_like(p)
