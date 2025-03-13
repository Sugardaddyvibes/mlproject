import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomExecption
from src.logger import logging
from src.utlis import save_object
from src.utlis import evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()
    
    def intiate_model_trainer(self,train_array,test_array):
        """
        this function is responsible for data modelling
        """
        try:
            logging.info("split trainning and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "xgboost":XGBRegressor(),
                "CatBoost":CatBoostRegressor()
                }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,
                                            X_test=X_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomExecption("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomExecption(e, sys)
            
