import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Text


def preprocess(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    print('Loading data..')
    df = pd.read_parquet(config['load_data']['path_data'])
    print(df.head())
    print('Data load complete!')

    print('\nFiltering taxi trips based on their duration. Only trips that last between 1 and 60 minutes are used!')
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= config['preprocess']['min_duration']) & (df.duration <= config['preprocess']['max_duration'])]

    print('\nConverting pickup and dropoff locations into categorical variables..')
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    df[categorical] = df[categorical].astype(str)
    print('\nSaving preprocessed data..')
    df.to_parquet(config['preprocess']['preprocessed_data'])

#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess(config_path=args.config)