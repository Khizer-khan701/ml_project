import os
import sys
from src.mlProject.exception import CustomException
from src.mlProject.logger import logging
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.mlProject.utils import save_object,evaluate_model

@dataclass
class DataTrainerConfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.data_trainer_config=DataTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
        
            }
            params={
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"]
                }
                ,
                "Random Forest":{
                    "n_estimators":[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    "learning_rate":[0.1,0.01,0.05,0.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    "learning_rate":[0.1,0.01,0.05,0.001],
                    "n_estimators":[8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    "depth":[6,8,10],
                    "learning_rate":[0.1,0.05,0.01]
                },
                "AdaBoost Regressor":{
                    "learning_rate":[0.1,0.01,0.05],
                    "n_estimators":[8,16,32,64,128,256]
                },
                
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params)
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and test dataset")
            save_object(
                file_path=self.data_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            R2_score=r2_score(y_test,predicted)
            return R2_score


        except Exception as e:
            raise CustomException(e,sys)