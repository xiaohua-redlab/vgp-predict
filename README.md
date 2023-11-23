vgp-predict
==============================

The forecasting branch of the VGP project, used to predict future road congestion in différent city.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# VGP PREDICT - Environment Setup

This README provides instructions on how to set up the necessary conda environment for this project. The conda environment file (`environment.yml`) contains all the dependencies required to run the project.

## Prerequisites

Before you proceed with setting up the environment, make sure you have the following prerequisites installed:

- [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Environment Setup

Follow the steps below to create and activate the conda environment:

1. Clone the repository (if you haven't already):
```bash
git clone https://github.com/xiaohua-redlab/vgp-predict
cd your-repo
```

3. Create the conda environment using the provided `environment.yml` file:
`conda env create -f environment.yml`
This will create a new conda environment with all the required packages.

Or you can create a new conda environment and use the requirements in the `requirements.txt` file, with following command:
`conda create --name new_env_name --file requirements.txt`

3. Activate the newly created environment:
`conda activate my_env`
Replace `my_env` with the name you specified in the `environment.yml` file. VGP dataset environments are setted in `environment.yml`.

4. You are now inside the conda environment and can run your project using the installed dependencies.
When you're done working within the conda environment, you can deactivate it by running:
`conda deactivate`

## Updating the Environment

If you need to update the environment with new packages or versions, you can do so by modifying mannuelly the `environment.yml` file.
Then running the following command:
`conda env update -f environment.yml --prune`

In this project, we need set up the environmental variables for VGP dataset, like: 
VGP_HOST, VGP_USER, VGP_PWD, VGP_PORT (you can always add other variables) and these variables are already in `environment.yml` file.

## Some indications for `scratch_data_process.ipynb`
This is the notebook where we load and and first process the data from VGP dataset. At the end, we save the data for later use (training phase).

1. Make sure that all environmental variables are setted. And read carefully the code comments lines in the notebook.(that will help you understand the code and intention)

In this project, all the data should be saved at `/vgp-predict/data`.

For raw data : `/vgp-predict/data/external`

For processed data : `/vgp-predict/data/process`

2. Change to your path
In Step6 : Save processed data, you can save your processed data, don't forget switch to your path!


## Some indications for `predict_baseline_mlflow.ipynb`
This is the notebook where we transform the processed data and train the model. The MLFlow is required so that we can track the experiments, models and parameters in MLFlow UI Server.

1. Make sure that MLFlow is installed in your system (it will be automatically installed if you create the conda environment using the provided `environment.yml` or `requirements.txt` file). 
Just test it run well by running:

    ``` bash
    mlflow ui --default-artifact-root=file:mlruns
    ``` 

    P.S. you can add `--backend-store-uri=sqlite:///mlflow.db` in the above command to setup a remote backend-store (recommanded for collaborating)

    You should  have a message like: Serving on http://127.0.0.1:5000 and you can open the server.

3. Make sure that the processed data is saved at `./data/processed/filtered_data_city1.csv` (e.g.)

4. To launch MLFlow, you should set up some config, like: artifacts_dir, experiment_name, run_name, registered_model_name, remote_server_uri(e.g. http://127.0.0.1:5000).

    You can change them as you like, otherwise they are already setted in the code.

5. For the training part in the notebook, for now it's simple, but you should apply a 5_fold_CrossValidation_train to train and evalue a model. 

### 5-Fold Cross-Validation
- Introduction
5-Fold Cross-Validation (CV) is a widely used technique in machine learning to assess the performance of a predictive model. It helps to evaluate how well a model generalizes to unseen data and provides a more robust estimate of its performance compared to a single train-test split.

Here's a step-by-step guide on how to perform 5-fold cross-validation for training and evaluating a machine learning model:

- Data Preparation:

Ensure your dataset is properly preprocessed and divided into features (X) and target labels (y).

- Initialize the Model:

Choose a machine learning algorithm or model that you want to evaluate.

- Cross-Validation Split:

Divide your dataset into five approximately equal-sized subsets or "folds." Each fold represents a separate training and testing set.

- Iteration:
Perform the following steps for each of the five folds:
Use four folds for training the model.
Use the remaining one fold for testing (validation).
Train the model on the training folds.
Evaluate the model's performance on the validation fold using a chosen evaluation metric (e.g., accuracy, mean squared error, etc.).

- Aggregation:

Calculate the average performance metric (e.g., mean accuracy or mean squared error) across all five folds. This gives you a more stable and reliable estimate of the model's performance.

- Example Code:
Here's an example python code snippet in Python using scikit-learn to perform 5-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Initialize your model
model = RandomForestClassifier()

# Specify the number of folds
n_folds = 5

# Create a cross-validation object (KFold)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Perform 5-fold cross-validation and specify the scoring metric (e.g., accuracy)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Calculate the mean and standard deviation of the scores
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f'Mean Accuracy: {mean_accuracy}')
print(f'Standard Deviation: {std_accuracy}')
```

## Some indications for Model Selection Part
In this part, there is no prepared notebook, but here are some suggestions.

Idea : Replace the old model with the new one if the new one performs better.

```python
## predict with test set
old_y_pred = old_model.predict(X_test)
## get model score
old_mae = mean_absolute_error(y_test, old_y_pred)

## predict with test set
new_y_pred = new_model.predict(X_test)
## get model score
new_mae = mean_absolute_error(y_test, new_y_pred)

## if new_mae is better, retrain a new model with all new data
## mae is better when it is smaller(tend to 0)
if mean(new_mae) < mean(old_mae):
    final_model_new = ModelClass()
    final_model_new.train(all_new_data)
    mlflow.log_metrics(final_model_new)
else:
    final_model_new = old_model
```


