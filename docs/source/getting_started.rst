
Getting Started
================================================

Installation
--------------------------------------------------
Installing ``mlvern`` is straightforward using pip. You can install it directly from PyPI with the following command:

.. code-block:: python

   pip install mlvern

For the latest development version, you can clone the repository from GitHub and install it using:

.. code-block:: bash

   git clone https://github.com/mlvern/mlvern.git
   cd mlvern   
   pip install .

Make sure you have ``Python 3.10 or higher installed``, as mlvern requires it to function properly. You can verify your Python version by running:

.. code-block:: bash

   python --version

Once installed, you can verify the installation by importing mlvern in a Python shell:

.. code-block:: python

   import mlvern
   print(mlvern.__version__)

Example Usage
--------------------------------------------------

Here is a simple example to get you started with mlvern. This example confirms 

.. code-block:: python

   from mlvern import MlVern

   # Initialize a new mlvern repository
   mlvern_repo = MlVern()
   mlvern_repo.init()

   # Track a new model version
   model_path = 'path/to/your/model.pkl'
   version = 'v1.0'
   mlvern_repo.track(model_path, version)

   # List all tracked model versions
   versions = mlvern_repo.list_versions()
   print("Tracked model versions:", versions)

