import argparse
import numpy as np
import pandas as pd
import yaml
from typing import Text
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


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

    # Get estimator names
    estimator_name = config['model']['estimators'].keys()
    
    rmse_scores = {}
    best_estimator_name = None
    best_rmse_score = float('inf')
    best_estimator = None

    for estimator_names in estimator_name:
        print(f'Fitting {estimator_names} model..')
        estimator = training(input_features=X_train,
                            target=y_train,
                            estimator_name=estimator_names,
                            param_grid=config['model']['estimators'][estimator_names]['param_grid'],
                            cv=config['model']['cv'])
        # Calculate cross-validation RMSE scores
        cv_scores = np.sqrt(-cross_val_score(estimator.best_estimator_, X_train, y_train, 
                                            scoring='neg_mean_squared_error', cv=config['model']['cv']))
        rmse_scores[estimator_names] = cv_scores
    
    # Update best RMSE score and estimator name
    if np.mean(cv_scores) < best_rmse_score:
        best_rmse_score = np.mean(cv_scores)
        best_estimator_name = estimator_names
        best_estimator = estimator.best_estimator_

    print(f'Best RMSE score: {best_rmse_score} for estimator {best_estimator_name}')

    print("Plotting RMSE scores for each estimator..")
    # Plot RMSE scores
    fig, ax = plt.subplots()
    for estimator_names in estimator_name:
        ax.plot(rmse_scores[estimator_names], label=estimator_names)

    ax.set(title='RMSE score comparison for different estimators',
        xlabel='Number of cross-validation folds',
        ylabel='RMSE score')
    ax.legend()
    plt.savefig(config['model']['rmse_comparison_plot'])

    # Save best estimator
    print('Saving best model..')
    with open(config['model']['model_path'], 'wb') as f_out:
        pickle.dump((dv, best_estimator), f_out)

#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)