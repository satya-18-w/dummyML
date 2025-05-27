import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestRegressor
import mlflow
import sys
from hyperopt import hp,fmin,tpe,STATUS_OK,space_eval,Trials

from xgboost import XGBRegressor
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
import pathlib


def find_best_model_with_params(X_train,y_train,X_test,y_test):

    
    
    hyperparameters = {
        "RandomForestRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
        "XGBRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "learning_rate": hp.uniform("learning_rate", 0.03, 0.3),
        },
    }
    
    
    
    
    
    
    def evaluate_model(hyperopt_params):
        params=hyperopt_params
        if "max_depth" in params: params["max_depth"]=int(params["max_depth"])# Hyperopt supplies value as float we need tp convet it into int
        if "min_child_weight" in params: params["min_child_weight"]=int(params["min_child_weight"])
        if "max_delta_step" in params: params["max_delta_step"]=int(params["max_delta_step"])
        
        model=XGBRegressor(**params)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        rmse=root_mean_squared_error(y_test,y_pred=y_pred)
        mlflow.log_metric("rmse",rmse)
        return {
            "loss":rmse,"status": STATUS_OK,
        }
        
    space=hyperparameters["XGBRegressor"]
    with mlflow.start_run(run_name="XGBRegressor") :
        argmin=fmin(
            fn=evaluate_model,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            trials=Trials(),
            verbose=True
        )
        
    
    run_ids=[]
    with mlflow.start_run(run_name="Xgb-final-model") as run:
        run_id=run.info.run_id
        run_name=run.data.tags["mlflow.runName"]
        run_ids+=[(run_id,run_name)]
        params=space_eval(space,argmin)
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
        if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])  
        mlflow.log_params(params)
        model=XGBRegressor(**params)
        model.fit(X_train,y_train)
        mlflow.sklearn.log_model(model,"model")
        
    return model


def save_model(model,output_path):
    joblib.dump(model,output_path+"/model.joiblib")
    
    
    
    
def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    input_file=sys.argv[1]
    data_path=home_dir.as_posix()+input_file
    output_path=home_dir.as_posix()+"/models"
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    
    TARGET="trip_duration"
    df=pd.read_csv(data_path+"/train.csv")
    X=df.drop(columns=[TARGET],axis=1)
    y=df[TARGET]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
    trained_model=find_best_model_with_params(X_train,y_train,X_test,y_test)
    save_model(trained_model,output_path)
    
    
    
if __name__ == "__main__":
    main()
    
        