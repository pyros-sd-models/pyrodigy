Configuration Retrieval
=======================

The `get_config` function in pyrodigy provides a flexible and generalized way to retrieve optimizer configurations. This function supports multiple input types, allowing you to specify the optimizer in various ways.

Overview
--------

`get_config` enables you to retrieve configurations for an optimizer by specifying:

- The optimizer **name** as a string (e.g., `"a2grad"`).
- The optimizer **class** directly (e.g., `AdamP` from `pytorch_optimizer`).
- A **dictionary** containing custom configurations defined by the user.

Function Definition
-------------------

.. code-block:: python

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

Usage Examples
--------------

1. **Using an Optimizer Name (String)**

   You can retrieve a configuration by passing the optimizer's name as a string:

   .. code-block:: python

       config = get_config("a2grad", "low_memory")

2. **Using an Optimizer Class**

   You can also use the optimizer class directly to fetch the configuration:

   .. code-block:: python

       from pytorch_optimizer import AdamP
       config = get_config(AdamP, "low_memory")

3. **Using a Custom Configuration Dictionary**

   If you have a custom configuration dictionary, you can pass it directly to `get_config`:

   .. code-block:: python

       user_configs = {
           "low_memory": {"optimizer": "custom", "param1": 1},
           "high_memory": {"optimizer": "custom", "param1": 2}
       }
       config = get_config(user_configs, "high_memory")

Explanation of Input Types
--------------------------

- **String**: This is the name of the optimizer as defined in pyrodigy. The function will look up the configuration from predefined config files.
- **Class**: You can directly provide the optimizer class, and `get_config` will fetch its configuration by using the class's default name or alias.
- **Dictionary**: For more control, pass a custom dictionary containing configurations. This is useful for user-defined configurations not covered in the default config files.

Error Handling
--------------

If `get_config` encounters an unknown configuration or an unsupported input type, it will raise a `ValueError`. Ensure that the optimizer name or configuration dictionary contains valid entries.

