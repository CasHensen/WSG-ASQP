import pandas as pd
import numpy as np
import time 

from Paraphrase import paraphrase
import GPT
import data_utils
import save_utils
import eval_utils


class generative_model():
    def __init__(self, args, train_labels, val_labels):
        self.args = args
        if args.generative_approach == 'paraphrase':
            self.para = paraphrase(args, train_labels, val_labels)
        elif args.generative_approach == 'gpt':
            self.category_seeds = data_utils.get_category_seeds(args)
            self.sentiment_seeds = data_utils.get_sentiment_seeds(args)
            self.gpt = GPT.gpt(self.args, self.category_seeds, self.sentiment_seeds)
            
                   
    def evaluate(self):
        dataset = data_utils.load_test_data(self.args, 'test')  
        start = time.time()   
        if self.args.generative_approach == 'paraphrase':
            targets = self.para.evaluate(dataset)
        elif self.args.generative_approach == 'gpt':
            targets = self.gpt(dataset)
        label_time = time.time() - start
        
        ##evaluate function
        scores, gold, perd = eval_utils.compute_scores(targets, self.args, 'test', label_time)

        save_utils.save_file(targets, self.args, 'test')

        return scores, targets


    def inference(self, data):
        if self.args.generative_approach == 'paraphrase':
            inference = self.para.inference(data)
        elif self.args.generative_approach == 'gpt':
            targets = gpt(data)


