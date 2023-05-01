import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Text
import math
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def evaluate(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    print('Loading validation data..')
    df = pd.read_parquet(config['evaluate']['path_data'])
    print(df.head())

    print('\nProcessing and structuring validation dataset')
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= config['preprocess']['min_duration']) & (df.duration <= config['preprocess']['max_duration'])]

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    df[categorical] = df[categorical].astype(str)
    eval_dicts = df[categorical + numerical].to_dict(orient='records')

    print("\nMaking predictions on validation data...")
    with open(config['model']['model_path'], 'rb') as f_in:
        (dv, model)= pickle.load(f_in)
    
    X_eval = dv.transform(eval_dicts)
    y_eval = df[config['train']['target_column']].values
    
    y_pred = model.predict(X_eval)
    rmse = mean_squared_error(y_eval, y_pred, squared=False)

    print(f'RMSE score on validation set: {rmse}')

#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)