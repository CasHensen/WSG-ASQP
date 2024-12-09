import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import collections
import re

import data_utils
import save_utils

import threading

class vocabulary_generator():
    """
    This class carries out the generation of the vocabularies for the aspect categories and sentiment polarities based on the seed words

    attributes: 
    - nlp: the natural language processor for POS tagging and dependency parsing
    - bert: the BERT model used to find replacements for the masked words 
    - tokenizer: the tokenizer used to tokenize the input for BERT
    - category_seeds: the seed words for the aspect categories
    - sentiment_seeds: the seed words for the sentiment polarities
    - num_threads: the number of threads used to thread the function 
    - device: the device shows whether it is run on CPU or on GPU
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - mask: the mask token
    - mask_token: the tokenized mask token
    - category_vocabulary: variable to store the aspect category vocabularies 
    - sentiment_vocabulary: variable to store the sentiment polarity vocabularies
    - k: the numnber of replacemtns found for masked words
    - the size of the vocabularies
    """
    def __init__(self, args, train, category_seeds, sentiment_seeds, bert, tokenizer, nlp):  
        """
        The intialization function to assign variables to the class attributes and to generate the vocabularies

        input: 
        - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
        - train: the unlabeled train data used to generate the vocabularies
        - category_seeds: the seed words for the aspect categories
        - sentiment_seeds: the seed words for the sentiment polarities
        - nlp: the natural language processor for POS tagging and dependency parsing
        - bert: the BERT model used to find replacements for the masked words 
        - tokenizer: the tokenizer used to tokenize the input for BERT
        """
        self.nlp = nlp
        self.bert = bert
        self.tokenizer = tokenizer

        self.category_seeds  = category_seeds
        self.sentiment_seeds = sentiment_seeds
        self.num_threads = args.num_threads
        self.device = args.device

        self.args = args

        self.mask = self.tokenizer.mask_token
        self.mask_token = self.tokenizer(self.mask).input_ids[1:-1]

        self.category_vocabulary = {}
        self.sentiment_vocabulary = {}

        self.k = args.k
        self.m = args.m

        self.threading_for_vocabularies(train)


    def get_vocabularies(self):
        """
        Function to obtain the aspect category and sentiment polarity vocabularies 

        output: 
        - category_vocabulary: the aspect category vocabularies 
        - sentiment_vocabulary: the sentiment polarity vocabularies
        """
        return self.category_vocabulary, self.sentiment_vocabulary


    def threading_for_vocabularies(self, train):
        """
        The generation of vocabularies is threaded over multiple threads for faster computation 

        input: 
        - train: the train data used to generate the vocabularies
        """
        seed_words = self.sentiment_seeds | self.category_seeds
        freq_table = {}
        for cat in seed_words:
            freq_table[cat] = {}
        split_sentences = np.array_split(train, self.num_threads)
        threads = []
        for i in range(self.num_threads):
            threads.append(threading.Thread(target=self.create_vocabularies, args=(pd.Series(split_sentences[i]), seed_words, freq_table)))
            threads[i].start()
            
        for i in range(self.num_threads):
            threads[i].join()

        vocabularies = self.filter_vocabularies(freq_table)

        for key in self.sentiment_seeds:
            self.sentiment_vocabulary[key] = vocabularies.get(key)

        for key in self.category_seeds:
            self.category_vocabulary[key] = vocabularies.get(key)

        save_utils.save_file(self.sentiment_vocabulary, self.args, 'sentiment_vocab')
        save_utils.save_file(self.category_vocabulary, self.args, 'category_vocab')


    def create_vocabularies(self, sentences, seed_words, vocabularies):
        """
        function to create generate the vocabularies fot the aspect categories and sentiment polarities

        input: 
        - sentences: the dataset containing the sentences used to create the vocabularies 
        - seed_words: the seed words used to guide the creation of the vocabularies
        - vocabularies: the variable to store the created vocabularies

        output: 
        - vocabularies: the creaeted vocabularies
        """
        words_in_sentence = sentences.str.split()
        for key, values in seed_words.items():
            for sentence in words_in_sentence:
                temp = np.array(sentence)
                to_mask = np.isin(temp, values)
                if any(to_mask):
                    temp[to_mask] = self.mask
                    masked_sentence = " ".join(temp)
                    
                    tokenized_sentence = self.tokenizer(masked_sentence, return_tensors='pt', padding='max_length', max_length=128).to(self.device)
                    
                    replacements = self.get_k_replacements(tokenized_sentence, self.k) 
                    self.update_freq_table(vocabularies, key, replacements)

    def update_freq_table(self, freq_table, cat, replacements):
        """
        Function to update the frequency table with the newly obtained replacements 

        input: 
        - freq_table: the frequency table storing the replacements for all aspect categories and sentiment polarities
        - cat: the aspect category or sentiment polarity for which replacements are added to the frequency table
        - replacements: the replacement which need to be added to the frequency table
        """
        for replacement in replacements:
            freq_table[cat][replacement] = freq_table[cat].get(replacement, 0) + 1

    def filter_vocabularies(self, freq_table):
        """
        Function to filter the vocabulary on overlapping words and taking the top-m frequent words

        input: 
        - freq_table: the frequency table containing all replacements per aspect category and sentiment polarity and their frequency 

        output: 
        - vocabularies: the filtered vocabularies on overlapping words with a maximum of m words
        """
        for category in freq_table:
            for key in freq_table[category]:
                for cat in freq_table:
                    if freq_table[cat].get(key) != None and freq_table[cat][key] < freq_table[category][key]:
                        del freq_table[cat][key]
        
        vocabularies = {}

        for category in freq_table:
            words = []
            size = 0
            sorted_dict = collections.OrderedDict(freq_table[category])
            for key in sorted_dict:
                words.append(key)
                size += 1
                if size >= self.m:
                    break
            vocabularies[category] = words
        
        return vocabularies

        
    def get_k_replacements(self, tokenized_sentence, k):
        """
        Obtain the top-k replacements for the masked word in the sentence

        input: 
        - tokenized_sentence: the tokenized sentence for which replacements need to be found 
        - k: the number of replacements needing to be found

        output: 
        - replacements: the top-k replacements for the masked word in the sentence 
        """
        mask_location = np.reshape(np.isin(tokenized_sentence.input_ids.cpu().numpy(), self.mask_token), (-1))
        output = self.bert(**tokenized_sentence)[0]
        word_scores, word_ids = torch.topk(output, k, -1)
        top_k_indices = word_ids.squeeze(0)[mask_location]
        replacements = pd.Series()
        for top_k in top_k_indices:
            decoded_top_k = np.array(self.tokenizer.batch_decode(top_k))
            filtered_replacements = self.filter_replacements(decoded_top_k)
            replacements = pd.concat([replacements, filtered_replacements], ignore_index=True)
        return replacements

    def filter_replacements(self, decoded_top_k):
        """
        Function to filter the replacements on stopwords, punctuation, and partial words

        input: 
        - decoded_top_k: the decoded top-k replacements needing to be filtered

        output: 
        - pd.Series(filtered_words): the filtered replacements 
        """
        removed_padding = []
        for word in decoded_top_k:
            if '##' in word:
                continue
            temp = word.replace(" ", "")
            temp = re.sub("[A-Z]", "", temp)
            temp = re.sub(r'[^\w\s]', "", temp)
            if len(temp) >= 3:
                removed_padding.append(temp)
        
        filtered_words = []
        if removed_padding:
            removed_padding = " ".join(removed_padding)
            doc = self.nlp(removed_padding)
            filtered_words = [token.text for token in doc if not token.is_stop]
        
        return pd.Series(filtered_words)

    

    

        

