Quickstart
==========

This is a quick 5-minute beginner guide to getting used to mlvern for managing your machine learning models and experiments.

Initializing the Project
-------------------------------------
Create a new Python script or Jupyter notebook (note: Preferebly in a new directory) and run the following code to initialize your mlvern project:

.. code-block:: python

   from mlvern import Forge

   forge = Forge(project="your_project_name", base_dir="your_base_directory")
   forge.init()

This will create the necessary directory structure for your project. 
You can see newly created folder **.mlvern_your_project_name** in your base directory, which contains sub-directories for **datasets**, **models**, **runs**, and a **registry**.

.. code-block:: text

   your_base_directory/
   |
   ├── .mlvern_your_project_name/              ## Default Folder Created by mlvern
   │   ├── datasets/                           ## Preprocessed Datasets and EDA Reports
   │   ├── models/                             ## Trained Models            
   │   ├── runs/                               ## Experiment Runs and Metrics
   │   └── registry.json                       ## Universal Registry File for the Project
   |
   └── your_script (or) notebook.py

Registering the Dataset
---------------------------
Here comes the beautiful part! You can register your dataset using the following code snippet. 
For demonstration purposes, we will use the popular Iris dataset from scikit-learn.

Now when you register a dataset, mlvern automatically does the data inspection, Exploratory Data Analysis (EDA) using pandas-profiling and saves reports and plots under the **datasets/** sub-directory.

.. code-block:: python

   from sklearn.datasets import load_iris

   data = load_iris(as_frame=True)
   df = data.frame
   target = "target"

   dataset_fp, is_new = forge.register_dataset(df, target)

This registers the Iris dataset within you project, and you can clearly see the registered dataset is stored in the datasets folder along with data_inspection report, statistics_report and necessary plots.

.. code-block:: text

   datasets/
   └── dataset_hash/                ## Unique Folder for the Registered Dataset
         ├── plots/                 ## Data Distribution and Visualization Plots
         ├── reports/               ## EDA and Data Inspection Reports
         └── schema.json            ## Dataset Schema and Metadata

Training the Model
---------------------------
Now that you have registered your dataset, you can proceed to train your machine learning model. Here is an example of training a Logistic Regression model on the Iris dataset.


.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from mlvern import ModelTrainer

   X = df.drop(columns=[target])
   y = df[target]

   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

   lr_model = LogisticRegression(max_iter=200, random_state=42)
   
   config_lr = {
        "model_type": "LogisticRegression",
        "max_iter": 200,
        "solver": "lbfgs",
        "random_state": 42,
    }

    run_id_1, metrics_1 = forge.run(lr_model, X_train, y_train, X_val, y_val, config_lr, dataset_fp)


This trains the Logistic Regression model and logs the experiment run, including model configuration and evaluation metrics, under the **runs/** sub-directory.
You can explore the **runs/** folder to see the saved model, metrics, and configuration files.

.. code-block:: text

   runs/
   └── run_id_1/                    ## Unique Folder for the Experiment Run
         ├── model.pkl              ## Saved Trained Model
         ├── metrics.json           ## Evaluation Metrics
         ├── config.json            ## Model Configuration
         └── logs/                  ## Training Logs

You can repeat the training process with different models or configurations. For example, let's train a Random Forest Classifier on the same dataset.

.. code-block:: python

   rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
   
   config_rf = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
    }

    run_id_2, metrics_2 = forge.run(rf_model, X_train, y_train, X_val, y_val, config_rf, dataset_fp)

This trains the Random Forest model and logs the experiment run similarly under the **runs/** sub-directory.
You can now compare the metrics of both models to see which one performed better.

Comparing Models
---------------------------
You can easily compare the performance of different models using the logged metrics. Here is an example of how to compare the Logistic Regression and Random Forest models trained earlier.

.. code-block:: python

   from mlvern import ModelComparator

   comparator = ModelComparator(forge)

   comparison_df = comparator.compare_models([run_id_1, run_id_2])
   print(comparison_df)

This will display a comparison table of the evaluation metrics for both models, allowing you to easily identify which model performed better on the validation set.

Congratulations! You have successfully initialized an mlvern project, registered a dataset, trained multiple models, and compared their performance.

See the :doc:`Tutorials <tutorials/index>` and :doc:`API documentation <api>` for more advanced features and functionalities of mlvern.