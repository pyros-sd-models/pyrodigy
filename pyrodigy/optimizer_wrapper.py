# wrapper.py
import importlib

import torch
from loguru import logger
from pytorch_optimizer import load_optimizer

from pyrodigy.cli import record_history


class OptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, params, optimizer_name, config_name, lr=0.001, **kwargs):
        # Load the configuration dynamically
        logger.debug(
            f"Loading configuration for optimizer '{optimizer_name}' with config '{config_name}'"
        )
        config = self.load_config(optimizer_name, config_name)
        config.update(kwargs)  # Override with any additional params

        # Remove 'lr' from config if it’s already provided explicitly
        if "lr" in config:
            del config["lr"]

        # Load the optimizer class
        optimizer_class = self.get_optimizer_class(optimizer_name)

        # Check if the optimizer accepts an `lr` parameter
        if "lr" in optimizer_class.__init__.__code__.co_varnames:
            self.optimizer = optimizer_class(params, lr=lr, **config)
        else:
            self.optimizer = optimizer_class(params, **config)
            logger.info(
                f"Optimizer '{optimizer_name}' does not require 'lr'. Initializing without 'lr'."
            )

        # Log optimizer details
        self.log_optimizer_details(optimizer_name, config_name, lr, config)

        # Confirm optimizer initialization
        logger.success(
            f"Optimizer '{optimizer_name}' initialized successfully with config '{config_name}'."
        )

        record_history(
            optimizer_name=optimizer_name,
            config_name=config_name,
            params={**config, "lr": lr}
            if "lr" in optimizer_class.__init__.__code__.co_varnames
            else config,
        )

    @staticmethod
    def load_config(optimizer_name, config_name):
        try:
            config_module = importlib.import_module(f"config.{optimizer_name}_config")
            config = config_module.get_config(config_name)
            logger.debug(f"Configuration loaded: {config}")
            return config
        except ModuleNotFoundError:
            logger.error(
                f"Configuration file for optimizer '{optimizer_name}' not found."
            )
            raise ValueError(f"No configuration found for optimizer '{optimizer_name}'")

    @staticmethod
    def get_optimizer_class(optimizer_name):
        """
        Returns the optimizer class for the given optimizer name using pytorch_optimizer's load_optimizer.
        """
        try:
            optimizer_class = load_optimizer(optimizer=optimizer_name)
            logger.debug(
                f"Optimizer class '{optimizer_class.__name__}' loaded for '{optimizer_name}'"
            )
            return optimizer_class
        except ValueError:
            logger.error(
                f"Optimizer '{optimizer_name}' not available in pytorch_optimizer."
            )
            raise ValueError(
                f"Optimizer '{optimizer_name}' not found in pytorch_optimizer."
            )

    def log_optimizer_details(self, optimizer_name, config_name, lr, config):
        """
        Logs detailed information about the optimizer, its configuration, and parameters.
        """
        logger.info(f"\n{'=' * 50}")
        logger.info("🚀 Using PYRO's Optimizer Wrapper")
        logger.info(f"\n{'=' * 50}")
        logger.info(f"🔧 Optimizer Name: {optimizer_name}")
        logger.info(f"⚙️  Configuration: {config_name}")
        logger.info(f"\n{'=' * 50}")
        # Log `lr` only if it's relevant to the optimizer
        if "lr" in self.optimizer.__class__.__init__.__code__.co_varnames:
            logger.info(f"💡 Learning Rate: {lr}")

        logger.info(f"🔧 Additional Config Parameters: {config}")
        logger.info(f"{'=' * 50}\n")

    def step(self, *args, **kwargs):
        logger.debug("Optimizer step started.")
        self.optimizer.step(*args, **kwargs)
        logger.debug("Optimizer step completed.")

    def zero_grad(self, *args, **kwargs):
        logger.debug("Optimizer gradients zeroed.")
        self.optimizer.zero_grad(*args, **kwargs)
