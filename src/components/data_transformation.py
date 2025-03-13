import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomExecption
from src.logger import logging
import pandas as pd
from sklearn.preprocessing  import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

from src.exception import CustomExecption
from src.logger import logging
from src.utlis import save_object,evaluate_model



@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_tranformed(self):
        """
        this function is responsible for data transformation
        """
        try:
            numerical_features=['reading score', 'writing score']
            cat_features=['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']
            num_pipeline = Pipeline(
                steps=[
                     ("imputer",SimpleImputer(strategy=("median"))),
                     ("StandardScaler",StandardScaler())
                ]
                )
            cat_Pipelines = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()), 
                    ("StandardScaler", StandardScaler(with_mean=False))
                      ]
                )
            logging.info(f"Numerical columns:{numerical_features}")
            logging.info(f"Categorical columns:{cat_features}")
            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_Pipeline",cat_Pipelines,cat_features) 
                ]
                )
            return preprocessor
        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomExecption(e, sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed") 
            logging.info("obtaining preprocessing object")


            preproccessing_obj=self.get_data_tranformed()

            target_column_names='math score'
            numerical_features=['reading score', 'writing score']
            
            
            input_feature_train_df=train_df.drop(columns=[ target_column_names],axis=1)
            target_feature_train_df=train_df[target_column_names]

            input_feature_test_df=test_df.drop(columns=[ target_column_names],axis=1)
            target_feature_test_df=test_df[target_column_names]


            logging.info(
             f"Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr=preproccessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preproccessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")


            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preproccessing_obj)
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )


        except Exception as e:
            print(f"Error occurred: {e}")
            raise CustomExecption(e, sys)




