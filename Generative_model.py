import pandas as pd
import numpy as np
import time 

from Paraphrase import paraphrase
import GPT
import data_utils
import save_utils
import eval_utils


class generative_model():
    """
    The class carries out the generative phase of the WG-ASQP. 

    The class either uses the GPT approach to obtain inference from the data or trains the Paraphrase model on the automatically labeled data 

    attributes: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - para: the trained paraphrase method, if this is the approached used 
    - category_seeds: a dictionary containing the aspect categories and its seeds words, but only the aspect categories are used 
    - sentiment_seeds: a dictionary containing the sentiment polarities and its seeds words, but only the sentiment polarities are used 
    - gpt: the file containing the environment variables for the GPT model if this is the approach used
    """
    def __init__(self, args, train_labels, val_labels):
         """
        The intialization function to assign variables to the class attributes.

        input: 
        - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
        - train_labels: the train data labels, only used to train the paraphrase model, not used for the GPT approach
        - val_labels: the val data labels, only used to train the paraphrase model, not used for the GPT approach
        """ 
        self.args = args
        if args.generative_approach == 'paraphrase':
            self.para = paraphrase(args, train_labels, val_labels)
        elif args.generative_approach == 'gpt':
            self.category_seeds = data_utils.get_category_seeds(args)
            self.sentiment_seeds = data_utils.get_sentiment_seeds(args)
            self.gpt = GPT.gpt(self.args, self.category_seeds, self.sentiment_seeds)
            
                   
    def evaluate(self):
        """
        The function evaluates the chosen method, so the test data should contain labels

        output: 
        - scores: the F1 score, precision, and recall score for the chosen generative approach
        - targets: the obtained classification for each sentence in the form of a list of dictionaries containing the sentence and quadurplets per sentence. 
        """
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
        """
        The function to get the inference from the chosen method, labels are needed. 

        input: 
        - data: the dataset which needs to be classified, needs to be a list of dictionaries with the sentence and an empty label. 

        output: 
        - targets: the classified dataset containing the sentence and the obtained label
        """ 
        if self.args.generative_approach == 'paraphrase':
            targets = self.para.inference(data)
        elif self.args.generative_approach == 'gpt':
            targets = gpt(data)

        return targets


