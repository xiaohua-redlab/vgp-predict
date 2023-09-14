import pandas as pd
import numpy as np
import joblib
import yaml
import argparse
from train import read_params

if __name__ == "__main__":
    
    ############################
    # Get configs
    ############################
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config_path = parsed_args.config
    config = read_params(config_path)
    
    external_prod_data_path = config["external_data_config"]["external_prod_data_csv"]
    city_id = config["data_process_config"]["city_id"]
    last_hours = config["data_process_config"]["last_hours"]
    model_dir = config["production_config"]["model_dir"]
    
    ############################
    # Load model
    ############################
    
    loaded_model = joblib.load(model_dir)
    
    ############################
    # Get data
    ############################
    
    # get external data
    selected_df = pd.read_csv(external_prod_data_path, sep='\t', header=0) 
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
    # Get X
    ############################

    X = feat_df.iloc[:, 3:].values
    print("X shape : (%d, %d)"%(X.shape[0], X.shape[1]))
    
    # Predict on a Pandas DataFrame.
    y = loaded_model.predict(pd.DataFrame(X))
    print("y shape : (%d,)"%(y.shape[0]))