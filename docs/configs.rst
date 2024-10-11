Configuration Modules
=====================

.. automodule:: config.a2grad_config
   :members:
   :undoc-members:
   :show-inheritance:

Code:

.. code-block:: python

       use_case_configs = {
           "low_memory": {"optimizer": "a2grad", "beta": 10.0, "lips": 10.0, "variant": "uni"},
           "consumer": {"optimizer": "a2grad", "beta": 5.0, "lips": 5.0, "variant": "inc"},
           "high_memory": {"optimizer": "a2grad", "beta": 5.0, "lips": 1.0, "rho": 0.9, "variant": "exp"}
       }




.. automodule:: config.adabelief_config
   :members:
   :undoc-members:
   :show-inheritance:

Code:

.. code-block:: python

       use_case_configs = {
           "low_memory": {"optimizer": "AdaBelief", "lr": 1e-4, "betas": (0.9, 0.999), "weight_decay": 0.01},
           "consumer": {"optimizer": "AdaBelief", "lr": 1e-4, "betas": (0.9, 0.999), "weight_decay": 0.01, "rectify": True},
           "high_memory": {"optimizer": "AdaBelief", "lr": 5e-5, "betas": (0.9, 0.999), "weight_decay": 0.01, "ams_bound": True}
       }

.. automodule:: config.adabound_config
   :members:
   :undoc-members:
   :show-inheritance:

Code:

.. code-block:: python

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
