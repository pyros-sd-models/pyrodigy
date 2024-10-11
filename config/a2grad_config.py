"""
A2Grad Configuration Module
---------------------------

This module provides configuration presets for the A2Grad optimizer, which is
designed for adaptively adjusting learning rates based on gradient values.
The optimizer offers different variants to manage memory and computational
requirements according to the user's needs.

Configurations:
    - **low_memory**: Suitable for low-memory environments. This configuration uses
      higher beta and lips values, focusing on stability over speed and with a 'uni'
      variant that provides uniform adaptation.

    - **consumer**: Balanced configuration for general use cases. It uses moderate
      beta and lips values and the 'inc' variant, which offers incremental updates,
      making it suitable for a variety of applications without high memory costs.

    - **high_memory**: Designed for environments with ample memory and resources.
      It uses lower lips values and an 'exp' variant, enabling exponential updates
      with finer granularity for potentially faster convergence at the expense of
      higher memory usage. The `rho` parameter is also set for improved stability.

Functions:
    - get_config(config_name): Retrieve the configuration dictionary based on the
      provided configuration name.
"""

use_case_configs = {
    "low_memory": {"optimizer": "a2grad", "beta": 10.0, "lips": 10.0, "variant": "uni"},
    "consumer": {"optimizer": "a2grad", "beta": 5.0, "lips": 5.0, "variant": "inc"},
    "high_memory": {
        "optimizer": "a2grad",
        "beta": 5.0,
        "lips": 1.0,
        "rho": 0.9,
        "variant": "exp",
    },
}
