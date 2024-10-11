__version__ = "0.1.0"

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Please install PyTorch manually to ensure CUDA support, "
        "following the instructions in the README."
    )


from .optimizer_wrapper import OptimizerWrapper as OptimizerWrapper
