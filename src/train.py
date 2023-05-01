import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Text
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def train(config_path: Text) -> None:
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



    print(y_train.dtype)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    rmse = mean_squared_error(y_train, y_pred, squared=False)
    print(rmse)


#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)