# config_utils.py
"""
Utility Module for Configuration Retrieval
------------------------------------------

Provides a generic method to retrieve configurations from any optimizer
configuration module.
"""

from importlib import import_module


def get_config(optimizer, config_name):
    """
    Retrieves the configuration for the specified optimizer and configuration name.

    Args:
        optimizer (str | type | dict): The optimizer name as a string,
                                       the optimizer class, or a user-defined configuration dictionary.
        config_name (str): The name of the configuration to retrieve.

    Returns:
        dict: The configuration settings.

    Raises:
        ValueError: If the configuration is not found or the input format is incorrect.
    """
    if isinstance(optimizer, str):
        # If optimizer is a string, assume it’s a module name and load the config from file
        try:
            config_module = import_module(f"config.{optimizer}_config")
            return config_module.use_case_configs.get(config_name)
        except ModuleNotFoundError:
            raise ValueError(
                f"Configuration module for optimizer '{optimizer}' not found."
            )
        except AttributeError:
            raise ValueError(
                f"Configuration '{config_name}' not found for optimizer '{optimizer}'."
            )
    elif isinstance(optimizer, type):
        # If optimizer is a class, derive the module name from the class name
        optimizer_name = optimizer.__name__.lower()
        return get_config(optimizer_name, config_name)
    elif isinstance(optimizer, dict):
        # If optimizer is a dictionary, treat it as the user config dictionary
        return optimizer.get(config_name, None)
    else:
        raise ValueError(
            "Optimizer must be a string, class, or dictionary containing configurations."
        )
