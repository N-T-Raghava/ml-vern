API
==========================================================

API Reference
===================

This section provides a comprehensive reference for the mlvern library's API, detailing the available classes, methods, and functions for version control in machine learning projects.

Module Overview
----------------------------------------------------------------------------------------------------------------------------------------

The mlvern library is designed over a single class ``Forge``. It is the entry point for interacting with the mlvern system, consisting of several modules that facilitate 
inspection, handeling, analysis of data and also version control for machine learning models. 


We can briefly divide the entire class into these modules:

**Dataset Lifecycle Management:**
  Provides methods for registering datasets, tracking dataset versions,
  and managing associated metadata.

**Experiment & Run Management:**
  Provides mechanisms for logging experiments, tracking execution runs,
  and comparing results across iterations.

**Model Registry & Access:**
  Enables model versioning, retrieval, and controlled deployment of
  registered models.

**Training & Execution:**
  Includes utilities for model training, performance evaluation, and
  analysis of version-level differences.

**Evaluation & Inference:**
  Contains helper functions for data preprocessing, transformation,
  and inference execution.

**Project Analytics & Statistics:**
  Provides statistical analysis and reporting capabilities for model
  versions, experiments, and overall project performance.

**Cleanup & Retention Management:**
  Offers mechanisms for cleaning up obsolete artifacts, enforcing
  retention policies, and managing storage utilization.


Class and Method Reference
---------------------------------------------------------------------------------------------------------------------------------------------------

Let's dive into the detailed API references for all the methods of the only class ``Forge`` in the ``mlvern`` library.

1. Dataset Lifecycle Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

APIs responsible for registering, storing, retrieving, and inspecting datasets.

1.1 Dataset Registration & Persistence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: register_dataset(data_frame, target, mlvern_dir) |rarrow| tuple(dict, bool)

  Registers a new dataset by creating a hash-identified directory used to 
  persist dataset metadata, analytical reports, and visualization artifacts.

  This function computes a deterministic fingerprint for the given dataset
  and creates a hash-based directory to store dataset metadata, validation
  reports, statistical summaries, and exploratory analysis artifacts.

  Heavy data inspection, statistical computation, and risk analysis are
  executed only once per unique dataset fingerprint. If the dataset has
  already been registered, no additional analysis is performed.

  :Parameters:
      **data_frame** : Input dataset as a pandas DataFrame.

      **target** : The name of the target column in the DataFrame.

      **mlvern_dir** : Path to the mlvern project directory where dataset metadata will be stored.

  :Returns:
      A tuple containing the dataset fingerprint and a boolean flag
      indicating whether the dataset was newly registered.

  :Return type:
      ``tuple(dict, bool)``

  :Notes:
      - A unique dataset hash is generated using the dataset schema and
        target column.
      - Exploratory data analysis plots are generated and stored under
        the dataset's ``plots/`` directory.
      - Dataset schema is persisted as ``schema.json``.
      - Dataset metadata is recorded in the project registry.

  :Example:
      .. code-block:: python

          fp, created = register_dataset(df, target="label", mlvern_dir=".mlvern")

          if created:
              print("Dataset registered successfully")
          else:
              print("Dataset already exists")


.. function:: save_dataset(data_frame, dataset_hash, name=None, tags=None) |rarrow| dict

  Persist a DataFrame to an existing dataset directory.

  This method saves the provided DataFrame to the dataset directory
  identified by the given dataset hash. In addition to storing the
  dataset contents, it records dataset-level metadata such as a
  human-readable name, user-defined tags, dataset shape, and the
  timestamp of persistence.

  :Parameters:
      **data_frame** :
          Dataset to be persisted as a pandas DataFrame.

      **dataset_hash** :
          Unique hash identifying the dataset directory where the
          DataFrame should be stored.

      **name** :
          Optional human-readable name for the dataset. If not provided,
          the dataset hash is used as the default name.

      **tags** :
          Optional dictionary of user-defined tags associated with
          the dataset.

  :Returns:
      A dictionary containing information about the saved dataset,
      including its storage path and persisted metadata.

  :Return type:
      ``dict``

  :Notes:
      - Dataset contents are saved to the resolved dataset path.
      - Dataset metadata is persisted separately and includes dataset
        shape, tags, and a UTC timestamp.
      - This method assumes that the dataset has already been registered.

  :Example:
      .. code-block:: python

          info = forge.save_dataset(
              df=data,
              dataset_hash=ds_hash,
              name="training_data_v1",
              tags={"source": "csv", "split": "train"},
          )

          print(info["path"])

1.2 Dataset Discovery & Enumeration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Additional Resources
------------------- 
For more detailed examples and usage patterns, please refer to the Usage Guide section of the documentation. If you have any questions or need further assistance, feel free to reach out to the mlvern community or check the GitHub repository for issues and discussions.






























