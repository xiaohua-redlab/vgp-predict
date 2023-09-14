import joblib
import mlflow
import argparse
from pprint import pprint
from train import read_params
from mlflow.tracking import MlflowClient


if __name__ == "__main__":
    ############################
    # Get configs
    ############################
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config_path = parsed_args.config
    config = read_params(config_path)
    
    experiment_name = config["mlflow_config"]["experiment_name"]
    registered_model_name = config["mlflow_config"]["registered_model_name"]
    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    min_mae_run_id = config["production_config"]["min_mae_run_id"]
    model_dir = config["production_config"]["model_dir"]
    
    ############################
    # Get and Save the best model
    ############################
    
    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    min_mae = list(runs[runs["run_id"] == min_mae_run_id]["metrics.mae"])[0]
    print(min_mae)
    
    ## Move the model version from the "Staging" phase to the "Production" phase and save the model locally according to some conditions.
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == min_mae_run_id:
            # Gets the version number of the current model version.
            current_version = mv["version"]
            # Gets current model's path or URI.
            logged_model = mv["source"]
            pprint(mv, indent=4)
            
            # Set the stage of the model version to "Production".
            client.transition_model_version_stage(
                name=registered_model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=registered_model_name,
                version=current_version,
                stage="Staging"
            )        
    
    # Loading models with MLflow
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)

    
    
    
    