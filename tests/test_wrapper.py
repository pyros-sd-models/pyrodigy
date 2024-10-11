import unittest
from unittest.mock import patch
from loguru import logger
import torch
from pytorch_optimizer import AdamP
from pyrodigy.optimizer_wrapper import OptimizerWrapper


# Sample model for testing
class SampleModel(torch.nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class TestWrapper(unittest.TestCase):
    def setUp(self):
        # Redirect loguru output to a string for testing
        self.logger_output = []
        logger.add(self.logger_output.append, format="{message}")

        # Sample model parameters for optimizer
        self.model = SampleModel()
        self.params = self.model.parameters()

    @patch("pyrodigy.optimizer_wrapper.load_optimizer")
    @patch("pyrodigy.optimizer_wrapper.OptimizerWrapper.load_config")
    def test_wrapper_initialization(self, mock_load_config, mock_load_optimizer):
        # Mock configuration
        mock_config = {"weight_decay": 0.01, "lr": 0.001}
        mock_load_config.return_value = mock_config

        # Mock optimizer class returned by load_optimizer
        mock_load_optimizer.return_value = AdamP

        # Initialize the Wrapper
        optimizer_name = "AdamP"
        config_name = "flux_high"
        wrapper = OptimizerWrapper(self.params, optimizer_name, config_name, lr=0.001)

        # Assertions
        mock_load_config.assert_called_once_with(optimizer_name, config_name)
        mock_load_optimizer.assert_called_once_with(optimizer=optimizer_name)

        # Check if the optimizer is initialized correctly
        self.assertIsInstance(wrapper.optimizer, AdamP)

        # Verify if expected logs are generated
        log_content = "".join(self.logger_output)
        self.assertIn("Using PYRO's Optimizer Wrapper", log_content)
        self.assertIn(f"Optimizer Name: {optimizer_name}", log_content)
        self.assertIn(f"Configuration: {config_name}", log_content)
        self.assertIn("Learning Rate: 0.001", log_content)

    def tearDown(self):
        # Remove all handlers added by logger during the test
        logger.remove()


if __name__ == "__main__":
    unittest.main()
