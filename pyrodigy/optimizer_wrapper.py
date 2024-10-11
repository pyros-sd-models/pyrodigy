"""
wrapper.py

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

from pyrodigy.cli import record_history


class OptimizerWrapper(torch.optim.Optimizer):
    """
    A wrapper class for PyTorch optimizers that dynamically loads configurations and provides
    logging capabilities. This class also records optimizer usage history.

    Attributes:
        optimizer: The actual optimizer instance from PyTorch or pytorch_optimizer.

    Methods:
        load_config(optimizer_name, config_name): Loads optimizer configuration dynamically.
        get_optimizer_class(optimizer_name): Retrieves the optimizer class from pytorch_optimizer.
        log_optimizer_details(optimizer_name, config_name, lr, config): Logs optimizer details and parameters.
        step(*args, **kwargs): Performs an optimization step.
        zero_grad(*args, **kwargs): Zeros the optimizer gradients.

    """

    def __init__(self, params, optimizer_name, config_name, lr=0.001, **kwargs):
        """
        Initializes the OptimizerWrapper with specified parameters, optimizer, and configuration.

        Args:
            params: Model parameters to optimize.
            optimizer_name (str): Name of the optimizer to use.
            config_name (str): Name of the configuration to load.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            **kwargs: Additional keyword arguments to override config parameters.

        Raises:
            ValueError: If the specified configuration or optimizer is not found.

        """
        logger.debug(
            f"Loading configuration for optimizer '{optimizer_name}' with config '{config_name}'"
        )
        config = self.load_config(optimizer_name, config_name)
        config.update(kwargs)  # Override with any additional params

        # Remove 'lr' from config if it’s already provided explicitly
        if "lr" in config:
            del config["lr"]

        optimizer_class = self.get_optimizer_class(optimizer_name)

        if "lr" in optimizer_class.__init__.__code__.co_varnames:
            self.optimizer = optimizer_class(params, lr=lr, **config)
        else:
            self.optimizer = optimizer_class(params, **config)
            logger.info(
                f"Optimizer '{optimizer_name}' does not require 'lr'. Initializing without 'lr'."
            )

        self.log_optimizer_details(optimizer_name, config_name, lr, config)

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
        """
        Loads the configuration for the specified optimizer from a configuration module.

        Args:
            optimizer_name (str): The name of the optimizer.
            config_name (str): The name of the configuration.

        Returns:
            dict: The loaded configuration parameters.

        Raises:
            ValueError: If the configuration file is not found.

        """
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

        Args:
            optimizer_name (str): The name of the optimizer.
            config_name (str): The configuration name.
            lr (float): The learning rate for the optimizer.
            config (dict): Additional configuration parameters.

        """
        logger.info(f"\n{'=' * 50}")
        logger.info("🚀 Using PYRO's Optimizer Wrapper")
        logger.info(f"\n{'=' * 50}")
        logger.info(f"🔧 Optimizer Name: {optimizer_name}")
        logger.info(f"⚙️  Configuration: {config_name}")
        logger.info(f"\n{'=' * 50}")
        if "lr" in self.optimizer.__class__.__init__.__code__.co_varnames:
            logger.info(f"💡 Learning Rate: {lr}")

        logger.info(f"🔧 Additional Config Parameters: {config}")
        logger.info(f"{'=' * 50}\n")

    def step(self, *args, **kwargs):
        """
        Performs an optimization step using the wrapped optimizer.

        Args:
            *args: Positional arguments for the optimizer step.
            **kwargs: Keyword arguments for the optimizer step.

        """
        logger.debug("Optimizer step started.")
        self.optimizer.step(*args, **kwargs)
        logger.debug("Optimizer step completed.")

    def zero_grad(self, *args, **kwargs):
        """
        Zeros the gradients of all optimized parameters.

        Args:
            *args: Positional arguments for zeroing gradients.
            **kwargs: Keyword arguments for zeroing gradients.

        """
        logger.debug("Optimizer gradients zeroed.")
        self.optimizer.zero_grad(*args, **kwargs)
