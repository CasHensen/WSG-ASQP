import pandas as pd
import numpy as np
import time 

import GPT
from BERT_approach import bert_approach

import data_utils
import save_utils
import eval_utils

class Labeler():
    """
    The class calls upon the right labeling approach and evaluates the automtically labeled data

    attributes: 
    - args: the arguments containing the technique to use, but also the thresholds and other parameters
    - category_seeds: a dictionary containing the aspect categories and its seeds words
    - sentiment_seeds: a dictionary containing the sentiment polarities and its seeds words 
    - labeling_approach: the labeling approach used, either the PLM BERT or GPT is used 
    """
    def __init__(self, args):
        """
        The intialization function to assign variables to the class attributes.

        input: 
        - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
        """ 
        self.args = args
        self.category_seeds = data_utils.get_category_seeds(args)
        self.sentiment_seeds = data_utils.get_sentiment_seeds(args)
        self.labeling_approach = args.labeling_approach
    
    def __call__(self):
        """
        The call function of the class calls upon the correct labeling approach

        The training and validation data automatically labeled using the correct labeling approach. If possible the automatically labeled datasets are evaluated using the F1 score, precision, and recall

        output: 
        - labels_train: the automatically labeled train data. The format is a list of dictionaries, with each dictionary containing an entry for the sentence, the automatically obtained label, and the gold label if present. 
        - labels_val: the automatically labeled validation data. The format is a list of dictionaries, with each dictionary containing an entry for the sentence, the automatically obtained label, and the gold label if present. 
        """
        start = time.time()
        if self.args.labeled_train_data_available and self.labeling_approach == 'gpt':
            labels_train, labels_val = data_utils.load_train(self.args)
        elif self.labeling_approach == 'labeled':
            labels_train = data_utils.extract_labeled_data(self.args, 'train')
            labels_val = data_utils.extract_labeled_data(self.args, 'val')
        elif self.labeling_approach == 'gpt':
            train, val = data_utils.load_train(self.args)
            gpt = GPT.gpt(self.args, self.category_seeds, self.sentiment_seeds)
            labels_train = gpt(train)
            labels_val = gpt(val)
        else:
            train, val = data_utils.load_train(self.args)
            bert = bert_approach(self.args, self.category_seeds, self.sentiment_seeds)
            labels_train, labels_val = bert(train, val)
        self.label_time = time.time() - start 

        if self.args.dataset == 'rest15' or self.args.dataset == 'rest16':
            self.evaluate(labels_train, 'train')
            self.evaluate(labels_val, 'val')

        save_utils.save_file(labels_train, self.args, 'train')
        save_utils.save_file(labels_val, self.args, 'val')

        return labels_train, labels_val

    def evaluate(self, pred, data_path):
        """
        The function to evaluate automatically labeled dataset

        inputs: 
        - pred: the automatically labeled dataset
        - data_path: the name of the dataset, either train, validation, or test
        """
        scores, gold, perd = eval_utils.compute_scores(pred, self.args, data_path, self.label_time)

    
        
    
