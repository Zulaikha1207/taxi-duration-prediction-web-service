import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from typing import Dict, Text
from xgboost import XGBRegressor


class UnsupportedClassifier(Exception):

    def __init__(self, estimator_name):
        self.msg = f'Unsupported estimator {estimator_name}'
        super().__init__(self.msg)

    
def get_supported_estimator() -> Dict:
    """
    Returns:
        Dict: supported classifiers
    """

    return {
        'linreg': LinearRegression,
        'lassoreg': Lasso,
        'xgb': XGBRegressor
    }


def training(input_features: float, target: float,
        estimator_name: Text, param_grid: Dict, cv: int):
        """
        Train model:
            Args:
                df {pandas.dataframe}: dataset
                target_column {Text}: target column name
                estimator_name {Text}: estimator name
                param_grid {Dict}: grid parameters
                cv {int}: cross-validation value
            Returns:
                trained model
        """

        estimators = get_supported_estimator()

        if estimator_name not in estimators.keys():
            raise UnsupportedClassifier(estimator_name)
        
        estimator = estimators[estimator_name]()
        rmse_scorer = make_scorer(mean_squared_error, squared = False)
        clf = GridSearchCV(estimator= estimator,
                           param_grid = param_grid,
                           cv =cv,
                           verbose=1,
                           scoring=rmse_scorer)
        
        #Get X and y
        y_train = target
        X_train = input_features
        clf.fit(X_train, y_train)

        return clf