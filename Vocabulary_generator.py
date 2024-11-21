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
    def __init__(self, args, train, category_seeds, sentiment_seeds, bert, tokenizer, nlp):
        self.category_seeds = category_seeds
        self.sentiment_seeds = sentiment_seeds
        
        self.nlp = nlp
        self.bert = bert
        self.tokenizer = tokenizer

        self.category_seeds  = category_seeds
        self.sentiment_seeds = sentiment_seeds
        self.num_threads = args.num_threads
        self.device = args.device

        self.args = args

        self.bert = bert
        self.tokenizer = tokenizer 
        self.mask = self.tokenizer.mask_token
        self.mask_token = self.tokenizer(self.mask).input_ids[1:-1]

        self.category_vocabulary = {}
        self.sentiment_vocabulary = {}

        self.k = args.k
        self.m = args.m

        self.threading_for_vocabularies(train)


    def get_vocabularies(self):
        return self.category_vocabulary, self.sentiment_vocabulary


    def threading_for_vocabularies(self, train):
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
        for replacement in replacements:
            freq_table[cat][replacement] = freq_table[cat].get(replacement, 0) + 1

    def filter_vocabularies(self, freq_table):
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

    

    

        

