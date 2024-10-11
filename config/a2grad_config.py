# a2grad_config.py
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


def get_config(config_name):
    if config_name not in use_case_configs:
        raise ValueError(f"Config {config_name} not found")
    return use_case_configs[config_name]
