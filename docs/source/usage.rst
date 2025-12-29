usage
====================================================================

Usage Guide
===================

This section provides detailed instructions on how to use the mlvern library for version control in machine learning projects.
Getting Started
-------------------
To get started with mlvern, first ensure that you have it installed. If you haven't installed it yet, please refer to the Installation section.
Once installed, you can import mlvern in your Python scripts as follows:
.. code-block:: python

   import mlvern
Basic Commands
-------------------
Here are some of the basic commands you can use with mlvern:
- Initialize a new mlvern repository:
  
  .. code-block:: bash

     mlvern init
- Track a new model version:
    .. code-block:: bash
    
         mlvern track --model model.pkl --version v1.0
- List all tracked model versions:
    .. code-block:: bash
    
         mlvern list    
For more advanced usage and features, please refer to the API Reference section.
