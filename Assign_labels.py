import pandas as pd
import numpy as np

import torch

import data_utils

class assign_labels():
"""
This class assigns the labels to the unlabeled dataset using the obtained overlap scores. 

The assignment of labels is done using one of three techniques for the BERT approach. The technique is specified in the arguments. 

attributes: 
- args: the arguments containing which technique to use, but also the thresholds and other parameters. 
- device: the device shows whether it is run on CPU or on GPU
- threshold: the threshold to assign labels 
- threshold_attn: the threshold idicating wether two words are linked via attention 
- bert: the BERT model used to find replacements for the masked words 
- tokenizer: the tokenizer used to tokenize the input for BERT
- nlp: the natural language processor for POS tagging and dependency parsing
"""
    
    def __init__(self, args, bert, tokenizer, nlp):
        """
        The initialization function to assign variables to the class attributes

        input: 
        - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
        - bert: the BERT model used to find replacements for the masked words 
        - tokenizer: the tokenizer used to tokenize the input for BERT
        - nlp: the natural language processor for POS tagging and dependency parsing
        """
        
        self.args = args
        self.device = args.device

        self.threshold = args.threshold
        self.threshold_attn = args.threshold_attn
        
        self.bert = bert
        self.tokenizer = tokenizer
        self.nlp = nlp


    def __call__(self, scores):
        """
        The call function of the class initiates the assignment of labels using the overlap scores. 

        input:
        - scores: the list of dictionaries, where each dictionary contains the sentence, the potential aspect category, the potential opinion terms, the individual overlap scores, and the sentence overlap scores

        output:
        - targets: the list of dictionaries, where each dictionary contains the sentence and the assigned labels 
        """
        if self.args.labeling == 'one' :
            labels = self.assign_one(scores)
        elif self.args.labeling == 'multiple_pos':
            labels = self.assign_pos(scores)
        elif self.args.labeling == 'multiple_attention':
            labels = self.assign_attention(scores)
        else:
            raise Exception("no valid labeling approach is chosen")
        
        targets = self.create_target(labels)

        return targets


    def create_target(self, labels):
        """
        transform the labels extracted from the sentences into the correct output format

        input: 
        - labels: a list of dictionaries, where each dictionary contains the aspect terms, opinion terms, aspect categories, and sentiment polarities extracted from the sentence

        output: 
        - targets: a list of dictionaries, where each dictionary contains the sentence and the quadruplets
        """
        targets = []
        for label in labels: 
            quads = []
            target = {'sentence': label['sentence'], 'quads': '', 'label': label['label']}
            if label['aspect']:
                for i in range(len(label['aspect'])):
                    a = label['aspect'][i]
                    c = label['category'][i]
                    o = label['opinion'][i]
                    s = label['sentiment'][i]

                    quad = [a, c, s, o]
                    quads.append(quad)
            
            target['quads'] = quads
            targets.append(target)
        return targets

    def assign_one(self, scores):
        """
        The function for the one quadruplet technique, where one quadruplet per sentence is assigned using the sentence overlap scores. 

        input: 
        - scores: a list of dictionaries, where each dictionary contains the sentence, potential aspects, the potential opinions, the individual scores and the sentence scores

        output: 
        - labels 
        """
        labels = []
        for score in scores:
            label =  {'sentence': score['sentence'], 'aspect': [], 'category': [], 'opinion': [], 'sentiment': [], 'label': score['label']}
            if score["aspect"]:
                category = score["sentence_category_scores"].idxmax(axis=1)
                sentiment = score["sentence_sentiment_scores"].idxmax(axis=1)

                if score["sentence_category_scores"][category].iloc[0].item() > self.threshold and score["sentence_sentiment_scores"][sentiment].iloc[0].item() > self.threshold:                       
                    label['category'].append(category.item())
                    label['sentiment'].append(sentiment.item())
                    
                    aspect = 'NULL'
                    aspect_score = 0
                    for i in range(len(score['aspect'])):
                        temp = score['category_scores'][i][category.item()].item()
                        if temp > aspect_score:
                            aspect_score = temp
                            aspect = score['aspect'][i]
                    
                    opinion = 'NULL'
                    opinion_score = 0
                    for j in range(len(score['opinion'])):
                        temp = score['sentiment_scores'][j][sentiment.item()].item()
                        if temp > opinion_score:
                            opinion_score = temp
                            opinion = score['opinion'][j]

                    label['aspect'].append(aspect) 
                    label['opinion'].append(opinion)
            
            labels.append(label)
        return labels

                    
                        

    def assign_pos(self, scores):
        labels = []
        for score in scores:
            label =  {'sentence': score['sentence'], 'aspect': [], 'category': [], 'opinion': [], 'sentiment': [], 'label': score['label']}
            if score["aspect"]:
                for i in range(len(score["aspect"])):
                    category = score["category_scores"][i].idxmax(axis=1)
                    sentiment = score["sentiment_scores"][i].idxmax(axis=1)   
                    if score["category_scores"][i][category].iloc[0].item() > self.threshold and score["sentiment_scores"][i][sentiment].iloc[0].item() > self.threshold:                       
                        label['category'].append(category.item())
                        label['sentiment'].append(sentiment.item())
                        label['aspect'].append(score["aspect"][i])
                        label['opinion'].append(score["opinion"][i])
            labels.append(label)

        return labels


    def assign_attention(self, scores):
        labels = []
        for score in scores:
            label =  {'sentence': score['sentence'], 'aspect': [], 'category': [], 'opinion': [], 'sentiment': [], 'label': score['label']}
            potential_categories = []
            if score['aspect']:
                attentions, tokens = self.get_attentions(score)
                for column in score['sentence_category_scores'].columns:
                    if score['sentence_category_scores'][column].item() > self.threshold:
                        potential_categories.append(column)
            
                for i in range(len(score["aspect"])):
                    category = ''
                    score_cat = 0
                    for cat in potential_categories:
                        if score["category_scores"][i][cat].item() > self.threshold and score["category_scores"][i][cat].item() > score_cat:
                            category = cat
                            score_cat = score["category_scores"][i][cat].item()
                    
                    if category:
                        token_t = self.tokenizer(score["aspect"][i], return_tensors='pt', truncation=True)['input_ids'].to(self.device)
                        token = np.array(self.tokenizer.convert_ids_to_tokens(token_t[0])[1:-1])
                        asp_index = []
                        for a in token:
                            temp = np.where(tokens == a)[0]
                            if temp.any():
                                asp_index.append(temp[0])

                        for j in range(len(score['opinion'])):
                            token_t = self.tokenizer(score["opinion"][j], return_tensors='pt', truncation=True)['input_ids'].to(self.device)
                            token = np.array(self.tokenizer.convert_ids_to_tokens(token_t[0])[1:-1])
                            op_index = []
                            for o in token:
                                temp = np.where(tokens == o)[0]
                                if temp.any():
                                    op_index.append(temp[0])

                            attention_1 = 0
                            attention_2 = 0
                            for a in asp_index:
                                for o in op_index:
                                    attention_1 += attentions[a][o]
                                    attention_2 += attentions[o][a]
                            
                            #if op_index and asp_index: 
                            #    attention_1 = attention_1/(len(asp_index)*len(op_index))
                            #    attention_2 = attention_2/(len(asp_index)*len(op_index))

                            if attention_1 > self.threshold_attn and attention_2 > self.threshold_attn:
                                sentiment = score["sentiment_scores"][j].idxmax(axis=1)
                                if score["sentiment_scores"][j][sentiment].iloc[0].item() > self.threshold:
                                    label['aspect'].append(score['aspect'][i])
                                    label['category'].append(category)
                                    label['opinion'].append(score['opinion'][j])
                                    label['sentiment'].append(sentiment.item())

            labels.append(label)
            
        return labels
                        
                    


    def get_attentions(self, score):
        sentence = score['sentence']
        sentence_t = self.tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids'].to(self.device)
        attentions = self.bert(sentence_t, output_attentions=True,)[1]
        tokens = self.tokenizer.convert_ids_to_tokens(sentence_t[0])

        word_attention_summed = sum(attentions)/12
        word_attention_avg = torch.sum(word_attention_summed[0], 0)/12
        word_attn = [word_attention_avg[i,1:len(tokens)-1]/ sum(word_attention_avg[i,1:len(tokens)-1]) for i in range(len(tokens))]

        return word_attn, np.array(tokens[1:-1])


    
