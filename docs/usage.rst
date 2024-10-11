Usage
=====

CLI Commands
------------

The CLI commands allow you to list optimizers, manage configurations, and handle history entries.

**List Available Optimizers**

.. code-block:: bash

   pyrodigy list

**Show Optimizer Documentation**

.. code-block:: bash

   pyrodigy show <optimizer_name>

Configuration Management
------------------------

Manage optimizer configurations using the `config` command:

**View Configuration:**

.. code-block:: bash

   pyrodigy config <optimizer_name> get

**Set Configuration:**

.. code-block:: bash

   pyrodigy config <optimizer_name> set '{"default": {"lr": 0.01, "beta": 0.9}}'

History Management
------------------

**Show History:**

.. code-block:: bash

   pyrodigy history <optimizer_name> show --TTL 30d

**Clear History:**

.. code-block:: bash

   pyrodigy history <optimizer_name> clear
