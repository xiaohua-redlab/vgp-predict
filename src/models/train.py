import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from urllib.parse import urlparse

import yaml
import argparse


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config_path = parsed_args.config
    config = read_params(config_path)
    
    ############################
    # Get params.yaml
    ############################
    
    external_data_path = config["external_data_config"]["external_data_csv"]
    city_id = config["data_process_config"]["city_id"]
    last_hours = config["data_process_config"]["last_hours"]
    train_test_split_ratio = config["data_process_config"]["train_test_split_ratio"]
    random_state = config["data_process_config"]["random_state"]
    
    experiment_name = config["mlflow_config"]["experiment_name"]
    artifacts_dir = config["mlflow_config"]["artifacts_dir"]
    run_name = config["mlflow_config"]["run_name"]
    registered_model_name = config["mlflow_config"]["registered_model_name"]
    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    
    fit_intercept = config["linear_regression"]["fit_intercept"]
    
    ############################
    # Get data
    ############################
    
    # get external data
    selected_df = pd.read_csv(external_data_path, sep='\t', header=0) 
    #"C:/_mlops/vgp_predict_lab/vgp-predict/data/external/filtered_data_city1.csv"
    selected_df = selected_df[selected_df["city_id"]==city_id]


    ############################
    # Data Processing
    ############################

    # create unix timestamp
    selected_df['unix_ts'] = pd.to_datetime(selected_df.timestamp).apply(lambda x: int(x.timestamp())).astype(int)

    # datetime from 2023/7/11
    selected_df = selected_df[selected_df['unix_ts']>=1689082398].reset_index(drop=True)

    # create hour index
    selected_df['hour_index'] = selected_df['unix_ts']//3600
    selected_df['hour_index'] = selected_df['hour_index'] - selected_df['hour_index'].min()

    # hard coded for knowing which day is the day according to initial hour_index
    df = selected_df[['hour_index', 'value']]

    # calculate mean congestion length per hour
    df = df.groupby(['hour_index']).mean().reset_index()

    # padding the missing hours (hours no congestion)
    pad_df = []
    for h in np.arange(0, df.hour_index.max()): # possible hour index
        matching_row = df[df['hour_index'] == h]
        v = matching_row['value'].values[0] if not matching_row.empty else 0
        day = ((h - 11) // 24 + 2) % 7 + 1 # calculate what day with hard coded formular according to initial hour
        pad_df.append([h, v, day]) # hour and value
    pad_df = pd.DataFrame(pad_df, columns = ['hour_index', 'mean_value', 'day'])

    # Use last 24h data (mean value) to predict actual mean value 
    feat_df = pad_df.copy()
    for i in range(1, last_hours+1):
        new_column_name = f'value_shift_{i}'
        feat_df[new_column_name] = feat_df['mean_value'].shift(i)
    feat_df = feat_df.dropna().reset_index()

    # Convert categorical variable (days) into dummy/indicator variables (as many 0/1 variables).
    feat_df = pd.get_dummies(feat_df, columns=['day'], dtype=int)

    ############################
    # Get train, test dataset
    ############################

    X = feat_df.iloc[:, 3:].values
    y = feat_df['mean_value'].values
    print("X shape : (%d, %d), y shape : (%d,)"%(X.shape[0], X.shape[1], y.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=random_state)

    ############################
    # Setup MLFlow experiment
    ############################

    try:
        mlflow.create_experiment(experiment_name)
    except:
        print("Expriment existes: %s"%experiment_name)
    
    mlflow.set_experiment(experiment_name)
    print(mlflow.tracking.MlflowClient().get_experiment_by_name(experiment_name))
    mlflow.set_tracking_uri(remote_server_uri)

    ############################
    # Start training with MLFlow
    ############################

    with mlflow.start_run(run_name=run_name):
        model = LinearRegression(fit_intercept=fit_intercept)  # fit_intercept=True == bias
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error on Test Set : %.4f \n"%mae)

        # stock models metadata in mlflow ui
        mlflow.log_param("fit_intercept", fit_intercept)
        mlflow.log_metric("mae", mae)

        # get the URI of the MLflow storage and parse out its protocol type (e.g., "file" for local file system, "s3" for Amazon S3 storage).
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store == "file":
            # if type is file system, save the trained linear regression model to the "model" directory of MLflow with a name
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=registered_model_name)
        else:
            mlflow.sklearn.load_model(model, "model")

    # mlflow.end_run()