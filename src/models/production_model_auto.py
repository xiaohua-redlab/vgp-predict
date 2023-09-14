import joblib
import mlflow
import argparse
from pprint import pprint
from train import read_params
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    
    model_name = "linear_regression_model"
    model_dir = "models/model.joblib"
    remote_server_uri = "http://localhost:5000"
    experiment_name = "my_experiment"
    
    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    min_mae = min(runs["metrics.mae"])
    min_mae_run_id = list(runs[runs["metrics.mae"] == min_mae]["run_id"])[0]
    print(min_mae)
    
    ## Move the model version from the "Staging" phase to the "Production" phase and save the model locally according to some conditions.
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == min_mae_run_id:
            # Gets the version number of the current model version.
            current_version = mv["version"]
            # Gets current model's path or URI.
            logged_model = mv["source"]
            pprint(mv, indent=4)
            
            # Set the stage of the model version to "Production".
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )        
    
    # Loading models with MLflow
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)

    
    
    
    