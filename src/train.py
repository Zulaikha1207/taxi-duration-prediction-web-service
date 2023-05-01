import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Text
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, './src/train')

from train import *

def train_model(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    print('Loading data..')
    df = pd.read_parquet(config['preprocess']['preprocessed_data'])
    print(df.head())

    print('Converting the selected categorcial and numerical columns into a sparse matrix representation..')
    #The categorical columns are transformed using one-hot encoding, while the numerical column(s) are left unchanged.
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    
    print("\nConverting input features into sparse matric representations..")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df[config['train']['target_column']].values

    print('Get estimator name')
    estimator_name = config['model']['estimator_name']
    
    print('Fitting model..')
    model = training(input_features=X_train,
                 target=y_train,
                 estimator_name=estimator_name,
                 param_grid= config['model']['estimators'][estimator_name]['param_grid'],
                 cv=config['model']['cv'])
    
    print(f'Best RMSE score: {model.best_score_}')
    
    print('Saving model..')
    #model_path= config["model"]["model_path"]
    #joblib.dump(model, model_path)


#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)