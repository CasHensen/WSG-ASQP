import numpy as np
import json
from pydantic.v1  import BaseModel, Field, Extra
from typing import List, Optional
import time
import os


def save_file(file, args, name):
    """
    Function to save obtained variables

    input: 
    - file: the variable which needs to be saved
    - args: the arguments containing which technique to use, but also the thresholds and other parameters
    - name: the name of the variable for example train, val, test, or vocabulary
    """
    directory = f"outputs/{args.dataset}/{name}/{args.labeling_approach}_{args.labeling}"
    
    if not os.path.exists(f"outputs/{args.dataset}/{name}"):
        os.mkdir(f"outputs/{args.dataset}/{name}")

    with open(directory, 'w') as fp:
            json.dump(file, fp, default=json_serialize)


def json_serialize(obj):
    """
    Make sure the variable can be json dumped

    input: 
    - obj: the object to be transformed s.t. it can be json dumped

    output: 
    - obj: the transformed object
    """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


def write_to_file(scores, args, data_type, labeling_time = 0):
    """
    Write the scores to a file s.t. scores of different runs can be easily compared

    input: 
    - scores: the obtained F1 score, precision, and recall for the dataset
    - args: the arguments containing which technique to use, but also the thresholds and other parameters
    - data_type: shows if it is the train data, validation data, or test data
    - labeling_time: time to label the dataset. default set to zero, in case you check a saved dataset without labeling time 
    """
    if data_type == 'test': 
        log_file_path = f"results_log/{args.dataset}/{data_type}/{args.labeling_approach}_{args.labeling}_{args.generative_approach}.txt"
    else:
        log_file_path = f"results_log/{args.dataset}/{data_type}/{args.labeling_approach}_{args.labeling}.txt"
    
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Datset={args.dataset}_{data_type}; Labeling approach={args.labeling_approach}_{args.labeling}; ; Generate approach={args.generative_approach}; Time to label={labeling_time:.4f}"
    exp_results = f"Total:    F1 = {scores['total']['f1']:.4f} & Precision = {scores['total']['precision']:.4f} & Recall = {scores['total']['recall']:.4f} \n" 
    exp_results += f"Aspect:   F1 = {scores['aspect']['f1']:.4f} & Precision = {scores['aspect']['precision']:.4f} & Recall = {scores['aspect']['recall']:.4f} \n"
    exp_results += f"Category: F1 = {scores['category']['f1']:.4f} & Precision = {scores['category']['precision']:.4f} & Recall = {scores['category']['recall']:.4f} \n"
    exp_results += f"Opinion:  F1 = {scores['opinion']['f1']:.4f} & Precision = {scores['opinion']['precision']:.4f} & Recall = {scores['opinion']['recall']:.4f} \n"
    exp_results += f"Sentiment:  F1 = {scores['sentiment']['f1']:.4f} & Precision = {scores['sentiment']['precision']:.4f} & Recall = {scores['sentiment']['recall']:.4f} \n"
    
    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    if not os.path.exists(f"results_log/{args.dataset}"):
        os.mkdir(f"results_log/{args.dataset}")
    
    if not os.path.exists(f"results_log/{args.dataset}/{data_type}"):
        os.mkdir(f"results_log/{args.dataset}/{data_type}")
        

    with open(log_file_path, "a+") as f:
        f.write(log_str)       
