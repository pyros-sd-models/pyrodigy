# adabelief_config.py
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


def get_config(config_name):
    if config_name not in use_case_configs:
        raise ValueError(f"Config '{config_name}' not found")
    return use_case_configs[config_name]
