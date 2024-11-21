import pandas as pd
import numpy as np
import time 

import GPT
from BERT_approach import bert_approach

import data_utils
import save_utils
import eval_utils

class Labeler():
    def __init__(self, args):
        self.args = args
        self.category_seeds = data_utils.get_category_seeds(args)
        self.sentiment_seeds = data_utils.get_sentiment_seeds(args)
        self.labeling_approach = args.labeling_approach
    
    def __call__(self):
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
        scores, gold, perd = eval_utils.compute_scores(pred, self.args, data_path, self.label_time)

    
        
    
