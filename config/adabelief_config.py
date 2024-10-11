"""
AdaBelief Configuration Module
------------------------------

This module defines configurations for the AdaBelief optimizer, which adapts the
learning rate based on both the magnitude of the gradients and the variance in
the gradients, making it particularly effective for non-convex optimization.

The AdaBelief optimizer offers several configuration presets tailored to different
use cases:

Configurations:
    - **low_memory**: Optimized for environments with limited memory resources.
      This preset keeps memory usage low by avoiding advanced features like
      rectification and AMS bound.

    - **consumer**: Provides a balanced configuration suitable for general use cases.
      Includes rectification, which can help improve convergence, but avoids
      memory-intensive features like AMS bound.

    - **high_memory**: Designed for high-memory environments where maximum performance
      is prioritized. This configuration enables all available options, including AMS bound
      and AdaNorm, which can improve convergence and stability.

Functions:
    - get_config(config_name): Retrieve the specified configuration dictionary.
"""

use_case_configs = {
    "low_memory": {
        "optimizer": "AdaBelief",
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "rectify": False,
        "ams_bound": False,
        "adanorm": False,
    },
    "consumer": {
        "optimizer": "AdaBelief",
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "rectify": True,
        "ams_bound": False,
        "adanorm": False,
    },
    "high_memory": {
        "optimizer": "AdaBelief",
        "lr": 5e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "rectify": True,
        "ams_bound": True,
        "adanorm": True,
        "r": 0.95,
    },
}
