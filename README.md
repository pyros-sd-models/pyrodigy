
# 🔥 pyrodigy 🔥

*ATTENTION - WORK IN PROGRESS - TEST DEPLOYMENT - NOT COMPLETE*

**pyrodigy** is a Python wrapper around more than 70 optimizers from [pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer), along with some additional custom optimizers. Designed for flexibility, pyrodigy offers easy configuration management, history tracking, and a CLI for convenience.

## Features
- **Access to 70+ Optimizers**: Use a variety of optimizers, from well-known ones to niche algorithms.
- **Config Management**: View, add, set, or remove optimizer configurations directly from the CLI.
- **History Tracking**: Track optimizer instantiations with detailed history, including timestamps, parameters, and caller information.
- **Customizable TTL**: Automatically clear history entries older than a specified time-to-live (TTL).
- **Rich CLI Interface**: Manage configurations, view documentation, and explore history—all from the command line.

## Installation

With pip
```bash
pip install pyrodigy
```

or

Clone the repo and install **pyrodigy** using [Poetry](https://python-poetry.org/):
```bash
poetry install
```

### Dependencies

**Note:** Pyrodigy requires [PyTorch](https://pytorch.org/get-started/locally/) to be installed separately. You can install it based on your specific environment (CPU or GPU) and operating system. Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions.


## Usage
### CLI Commands
The CLI commands allow you to list optimizers, manage configurations, and handle history entries. 

#### List Available Optimizers
Displays a list of optimizers for which a wrapper exist
```bash
pyrodigy list
```

#### Show Optimizer Documentation
Prints the Markdown documentation for the specified optimizer:
```bash
pyrodigy show <optimizer_name>
```

### Configuration Management
Manage optimizer configurations using the `config` command with `get`, `set`, `add`, and `rm` actions.

- **View Configuration**:
  ```bash
  pyrodigy config <optimizer_name> get
  ```
- **Set Configuration**: Update an existing configuration with new values (JSON format).
  ```bash
  pyrodigy config <optimizer_name> set '{"default": {"lr": 0.01, "beta": 0.9}}'
  ```
- **Add New Configuration**: Add a new named configuration (JSON format).
  ```bash
  pyrodigy config <optimizer_name> add <config_name> '{"lr": 0.02, "beta": 0.95}'
  ```
- **Remove Configuration**: Remove a named configuration.
  ```bash
  pyrodigy config <optimizer_name> rm <config_name>
  ```

### History Management
Each time an optimizer is instantiated, an entry is created in its history. You can review or clear history and apply a TTL to automatically remove old entries.

- **Show History**: View the history for an optimizer. Specify a TTL to filter entries within a certain timeframe.
  ```bash
  pyrodigy history <optimizer_name> show --TTL 30d
  ```
  Example with TTL:
  ```bash
  pyrodigy history a2grad show --TTL 60d
  ```
- **Clear History**: Remove all history entries for the optimizer.
  ```bash
  pyrodigy history <optimizer_name> clear
  ```

## Example: Using pyrodigy in Code

### General usage

Instantiate an optimizer with pyrodigy’s `OptimizerWrapper`, which logs the creation details to the optimizer's history.

```python
from pyrodigy import OptimizerWrapper

# Define model parameters and optimizer configuration
params = model.parameters()
optimizer_name = "AdamP"
config_name = "default"
lr = 0.001

# Initialize the optimizer
optimizer = OptimizerWrapper(params, optimizer_name=optimizer_name, config_name=config_name, lr=lr)
```


### Implementing into [Kohya's training framework]([http://bla](https://github.com/kohya-ss/sd-scripts))

Instantiate an optimizer with pyrodigy’s `OptimizerWrapper`, which logs the creation details to the optimizer's history.

```python
# get optimizer in train_utils.py

if optimizer_type.lower().startswith("Pyro-Wrapper".lower()):
  try:
      from pyrodigy import OptimizerWrapper

      # Extract necessary information from optimizer_kwargs
      optimizer_name = optimizer_kwargs.get("id", "adabelief")
      optimizer_default_config = optimizer_kwargs.get("cfg", "low_memory")

      # Define optimizer_class using the optimizer name
      optimizer_class = OptimizerWrapper(optimizer=optimizer_name)

      # Initialize the optimizer using the Wrapper
      optimizer = optimizer_class(
          trainable_params,
          optimizer_name=optimizer_name,
          config_name=optimizer_default_config,
          lr=lr,
          **optimizer_kwargs,
      )

  except ImportError:
      raise ImportError(
          "No pytorch_optimizer"
      )
  except Exception as e:
      logger.error(
          "An error occurred while loading the optimizer:", exc_info=True
      )
      raise RuntimeError(
          f"Failed to initialize optimizer '{optimizer_name}' with type '{optimizer_type}'. "
          "Please check the optimizer name, configuration, and installation."
      ) from e

#....
```

### History Entries
Every time you create an optimizer instance, the following details are saved:
- **Optimizer Name**: The name of the optimizer.
- **Config Name**: The configuration used, if provided.
- **Parameters**: Any additional parameters such as learning rate.
- **Caller Information**: File, line number, and function name where the optimizer was instantiated.

## License
Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! If you find any issues or have suggestions, feel free to open an issue or submit a pull request.

## Support
For questions or support, please open an issue on GitHub.
