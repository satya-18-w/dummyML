import pathlib
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging


def load_data(data_path):
    df=pd.read_csv(data_path)
    return df

def train_test(data,test_size,seed):
    train,test=train_test_split(data, test_size=test_size, random_state=42,shuffle=True)
    return train,test

def set_target_data(train,test,output_path):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(output_path+"/train.csv",index=False)
    test.to_csv(output_path+"/test.csv",index=False)
    return


def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_path=home_dir.as_posix()+"/params.yaml"
    #file=sys.argv[1]
    file="/data/raw/train.csv"
    params=yaml.safe_load(open(params_path))["make_dataset"]
    data_path=home_dir.as_posix()+file
    df=load_data(data_path)
    train,test=train_test(df,test_size=params["test_size"],seed=params["random_state"])
    output_dir=home_dir.as_posix()+params["output_dir"]
    pathlib.Path(output_dir).mkdir(exist_ok=True,parents=True)
    set_target_data(train,test,output_dir)
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    