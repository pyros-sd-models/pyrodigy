"""
optimizer_wrapper.py

This module provides an OptimizerWrapper class that serves as a wrapper around PyTorch optimizers.
It enables dynamic loading of optimizer configurations and detailed logging for tracking optimizer
usage, configurations, and parameters.

Classes:
    OptimizerWrapper: Wraps a PyTorch optimizer with dynamic configuration and logging features.

"""

import importlib

import torch
from loguru import logger
from pytorch_optimizer import load_optimizer

from config.config_utils import get_config
from pyrodigy.cli import record_history


class OptimizerWrapper(torch.optim.Optimizer):
    """
    A wrapper class for PyTorch optimizers that dynamically loads configurations and provides
    logging capabilities. This class also records optimizer usage history.

    Attributes:
        optimizer (torch.optim.Optimizer): The actual optimizer instance from PyTorch or pytorch_optimizer.

    Methods:
        load_config(optimizer_name, config_name): Loads optimizer configuration dynamically.
        get_optimizer_class(optimizer_name): Retrieves the optimizer class from pytorch_optimizer.
        log_optimizer_details(optimizer_name, config_name, lr, config): Logs optimizer details and parameters.
        step(``*args`, ``**kwargs``): Performs an optimization step.
        zero_grad(``*args`, *``**kwargs``): Zeros the optimizer gradients.
    """

    def __init__(
        self, params, optimizer_name, config_name="consumer", lr=0.001, **kwargs
    ):
        """
        Initializes the OptimizerWrapper with specified parameters, optimizer, and configuration.

        Args:
            params (iterable): Model parameters to optimize.
            optimizer_name (str): Name of the optimizer to use.
            config_name (str): Name of the configuration to load.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            ``**kwargs``: Additional keyword arguments to override config parameters.

        Raises:
            ValueError: If the specified configuration or optimizer is not found.

        """
        logger.debug(
            f"Loading configuration for optimizer '{optimizer_name}' with config '{config_name}'"
        )
        config = self.load_config(optimizer_name, config_name)
        config.update(kwargs)  # Override with any additional params

        if "lr" in config and "lr" not in kwargs:
            config.pop("lr", None)  # Use explicit lr if provided, otherwise remove

        optimizer_class = self.get_optimizer_class(optimizer_name)
        self.optimizer = self._initialize_optimizer(optimizer_class, params, lr, config)

        self.log_optimizer_details(optimizer_name, config_name, lr, config)

        logger.success(
            f"Optimizer '{optimizer_name}' initialized successfully with config '{config_name}'"
        )

        record_history(
            optimizer_name=optimizer_name,
            config_name=config_name,
            params={**config, "lr": lr}
            if "lr" in optimizer_class.__init__.__code__.co_varnames
            else config,
        )

    def _initialize_optimizer(self, optimizer_class, params, lr, config):
        """
        Initializes the optimizer with given parameters and configuration.

        Args:
            optimizer_class (type): The optimizer class to instantiate.
            params (iterable): Parameters to optimize.
            lr (float): Learning rate, if applicable.
            config (dict): Additional configuration parameters.

        Returns:
            torch.optim.Optimizer: An instance of the optimizer class.
        """
        try:
            if "lr" in optimizer_class.__init__.__code__.co_varnames:
                return optimizer_class(params, lr=lr, **config)
            return optimizer_class(params, **config)
        except TypeError as e:
            logger.error(
                f"Failed to initialize optimizer '{optimizer_class.__name__}' with error: {e}"
            )
            raise ValueError(
                f"Error initializing optimizer '{optimizer_class.__name__}' with given parameters."
            ) from e

    @staticmethod
    def load_config(optimizer_name, config_name):
        """
        Loads the configuration for the specified optimizer from a configuration module.

        Args:
            optimizer_name (str): The name of the optimizer.
            config_name (str): The name of the configuration.

        Returns:
            dict: The loaded configuration parameters.

        Raises:
            ValueError: If the configuration file is not found or is missing the required configuration.
        """
        try:
            config = get_config(optimizer_name, config_name=config_name)
            logger.debug(f"Configuration loaded: {config}")
            return config
        except ModuleNotFoundError:
            logger.error(
                f"Configuration file for optimizer '{optimizer_name}' not found."
            )
            raise ValueError(f"No configuration found for optimizer '{optimizer_name}'")
        except AttributeError:
            logger.error(
                f"Config '{config_name}' not found in module '{optimizer_name}_config'."
            )
            raise ValueError(
                f"Configuration '{config_name}' is missing in '{optimizer_name}_config'"
            )

    @staticmethod
    def get_optimizer_class(optimizer_name):
        """
        Returns the optimizer class for the given optimizer name using pytorch_optimizer's load_optimizer.

        Args:
            optimizer_name (str): The name of the optimizer.

        Returns:
            class: The optimizer class.

        Raises:
            ValueError: If the optimizer is not available in pytorch_optimizer.
        """
        try:
            optimizer_class = load_optimizer(optimizer=optimizer_name)
            logger.debug(
                f"Optimizer class '{optimizer_class.__name__}' loaded for '{optimizer_name}'"
            )
            return optimizer_class
        except ValueError as e:
            logger.error(
                f"Optimizer '{optimizer_name}' not available in pytorch_optimizer."
            )
            raise ValueError(
                f"Optimizer '{optimizer_name}' not found in pytorch_optimizer."
            ) from e

    def log_optimizer_details(self, optimizer_name, config_name, lr, config):
        """
        Logs detailed information about the optimizer, its configuration, and parameters.

        Args:
            optimizer_name (str): The name of the optimizer.
            config_name (str): The configuration name.
            lr (float): The learning rate for the optimizer.
            config (dict): Additional configuration parameters.
        """
        logger.info(f"\n{'=' * 50}")
        logger.info("🚀 Using PYRO's Optimizer Wrapper")
        logger.info(f"🔧 Optimizer Name: {optimizer_name}")
        logger.info(f"⚙️  Configuration: {config_name}")
        logger.info(
            f"💡 Learning Rate: {lr if 'lr' in self.optimizer.__class__.__init__.__code__.co_varnames else 'N/A'}"
        )
        logger.info(f"🔧 Additional Config Parameters: {config}")
        logger.info(f"{'=' * 50}\n")

    def step(self, *args, **kwargs):
        """
        Performs an optimization step using the wrapped optimizer.

        Args:
            ``*args`: Positional arguments for the optimizer step.
            ``**kwargs``: Keyword arguments for the optimizer step.
        """
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """
        Zeros the gradients of all optimized parameters.

        Args:
            ``*args`: Positional arguments for zeroing gradients.
            ``**kwargs``: Keyword arguments for zeroing gradients.
        """
        self.optimizer.zero_grad(*args, **kwargs)
