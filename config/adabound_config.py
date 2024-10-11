"""
AdaBound Configuration Module
-----------------------------

This module provides configuration presets for the AdaBound optimizer, which is
designed to combine the benefits of adaptive optimization methods and stochastic
gradient descent (SGD). AdaBound dynamically adjusts the learning rate bounds
during training, allowing for fast convergence and improved generalization.

Configurations:
    - **low_memory**: Suitable for low-memory environments. This configuration uses
      a lower `final_lr` and conservative settings to ensure stability and minimize
      resource usage.

    - **consumer**: Balanced configuration for general use cases. It maintains standard
      settings that are effective for a wide range of applications without incurring
      high memory costs.

    - **high_memory**: Designed for environments with ample memory and resources.
      This configuration enables additional features like `ams_bound` and adjusts
      `gamma` to take full advantage of the hardware capabilities, potentially
      enhancing performance at the expense of higher memory usage.

Functions:
    - get_config(config_name): Retrieve the configuration dictionary based on the
      provided configuration name.
"""

use_case_configs = {
    "low_memory": {
        "optimizer": "adabound",
        "lr": 1e-4,
        "final_lr": 0.01,
        "betas": (0.9, 0.999),
        "gamma": 1e-3,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "ams_bound": False,
        "adam_debias": False,
        "eps": 1e-8,
    },
    "consumer": {
        "optimizer": "adabound",
        "lr": 1e-4,
        "final_lr": 0.1,
        "betas": (0.9, 0.999),
        "gamma": 1e-3,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "ams_bound": False,
        "adam_debias": False,
        "eps": 1e-8,
    },
    "high_memory": {
        "optimizer": "adabound",
        "lr": 5e-5,
        "final_lr": 0.1,
        "betas": (0.9, 0.999),
        "gamma": 1e-4,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "ams_bound": True,
        "adam_debias": False,
        "eps": 1e-8,
    },
}
