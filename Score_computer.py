import pandas as pd
import numpy as np

import threading
import torch
import re

class score_computer():
    def __init__(self, args, bert, tokenizer, nlp, category_vocab, sentiment_vocab):
        self.k_2 = args.k_2
        self.device = args.device
        
        self.category_vocab = category_vocab
        self.sentiment_vocab = sentiment_vocab

        self.bert = bert
        self.tokenizer = tokenizer
        self.nlp = nlp

        self.num_threads = args.num_threads

        self.args = args

        self.mask = self.tokenizer.mask_token
        self.mask_token = self.tokenizer(self.mask).input_ids[1:-1]

        self.category_scores = pd.DataFrame(columns = list(self.category_vocab.keys()))
        self.sentiment_scores = pd.DataFrame(columns = list(self.sentiment_vocab.keys()))

    def __call__(self, train):
        labels = self.threading_for_overlapscores(train)
        return labels


    def threading_for_overlapscores(self, sentences):
        potentials = []
        split_sentences = np.array_split(sentences, self.num_threads)
        threads = []
        for i in range(self.num_threads):
            threads.append(threading.Thread(target=self.extract_scores, args=(split_sentences[i], potentials)))
            threads[i].start()
         
        for i in range(self.num_threads):
            threads[i].join()

        labels = self.normalize_scores(potentials)
        return labels


    def extract_scores(self, sentences, potentials):
        if self.args.labeling == 'one' or self.args.labeling == 'multiple_attention':
            self.extract_per_sentence(sentences, potentials)
        elif self.args.labeling == 'multiple_pos':
            self.extract_pos_tagging(sentences, potentials)
        else:
            self.extract_per_sentence(sentences, potentials)


    def extract_per_sentence(self, sentences, potentials):
        for sentence in sentences:
            label = {'sentence': sentence['sentence'], 'aspect': [], 'category_scores': [], 'category': [], 'opinion': [], 'sentiment_scores': [], 'sentiment': [], 'sentence_category_scores': [], 'sentence_sentiment_scores': [], 'label': sentence['label']}
            sentence = sentence['sentence']
            doc = self.nlp(sentence)
            words_in_sentence = [token.text for token in doc]
            noun_present = False
            nouns = []
            adjective_present = False
            adjectives = []
            for token in doc:
                if token.pos_ == 'NOUN':
                    noun_present = True
                    nouns.append(token)
                if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
                    adjective_present = True
                    adjectives.append(token)

            if adjective_present and noun_present: 
                for noun in nouns: 
                    label['aspect'].append(noun.text)
                    label['category_scores'].append(self.compute_overlap_scores(words_in_sentence.copy(), noun, self.category_vocab))
                for adjective in adjectives:
                    label['opinion'].append(adjective.text)
                    label['sentiment_scores'].append(self.compute_overlap_scores(words_in_sentence.copy(), adjective, self.sentiment_vocab))

                sentence_cat_scores = pd.DataFrame(columns = list(self.category_vocab.keys()))
                sentence_sent_scores =  pd.DataFrame(columns = list(self.sentiment_vocab.keys()))
                
                for i in range(len(label['category_scores'])):
                    sentence_cat_scores = pd.concat([sentence_cat_scores, label['category_scores'][i]], ignore_index = True)
                label['sentence_category_scores'] = pd.DataFrame(sentence_cat_scores.mean()).T
                
                for i in range(len(label['sentiment_scores'])):
                    sentence_sent_scores = pd.concat([sentence_sent_scores, label['sentiment_scores'][i]], ignore_index = True)
                label['sentence_sentiment_scores'] = pd.DataFrame(sentence_sent_scores.mean()).T
                    
                self.category_scores = pd.concat([self.category_scores, label['sentence_category_scores']], ignore_index = True)
                self.sentiment_scores = pd.concat([self.sentiment_scores, label['sentence_sentiment_scores']], ignore_index = True)
            
            potentials.append(label)

    def extract_pos_tagging(self, sentences, potentials):
        for sentence in sentences:
            label = {'sentence': sentence['sentence'], 'aspect': [], 'category_scores': [], 'category': [], 'opinion': [], 'sentiment_scores': [], 'sentiment': [], 'sentence_category_scores': [], 'sentence_sentiment_scores': [], 'label': sentence['label']}
            sentence_cat_scores = pd.DataFrame(columns = list(self.category_vocab.keys()))
            sentence_sent_scores =  pd.DataFrame(columns = list(self.sentiment_vocab.keys()))

            sentence = sentence['sentence']
            doc = self.nlp(sentence)
            words_in_sentence = [token.text for token in doc]
            for token in doc:
                #Case 1: Noun is linked to adverb or adjective as one of its childeren 
                if token.pos_ == "NOUN":
                    for child in token.children:
                        if child.pos_ == "ADJ" or child.pos_ == "ADV":
                            #first find overlap for aspect categories
                            label['aspect'].append(token.text)
                            label['category_scores'].append(self.compute_overlap_scores(words_in_sentence.copy(), token, self.category_vocab))
                            label['opinion'].append(child.text)
                            label['sentiment_scores'].append(self.compute_overlap_scores(words_in_sentence.copy(), child, self.sentiment_vocab))
                        
                    #Case 2: Noun is linked to adverb or adjective via a verb
                    if token.head.pos_ == "VERB":
                        for verb_child in token.head.children:
                            if verb_child.pos_ == "ADJ" or verb_child.pos_ == "ADV":                       
                                #first find overlap for aspect categories
                                label['aspect'].append(token.text)
                                label['category_scores'].append(self.compute_overlap_scores(words_in_sentence.copy(), token, self.category_vocab))
                                label['opinion'].append(verb_child.text)
                                label['sentiment_scores'].append(self.compute_overlap_scores(words_in_sentence.copy(), verb_child, self.sentiment_vocab))
                                
            if label['aspect']:                
                for i in range(len(label['category_scores'])):
                    sentence_cat_scores = pd.concat([sentence_cat_scores, label['category_scores'][i]], ignore_index = True)
                    sentence_sent_scores = pd.concat([sentence_sent_scores, label['sentiment_scores'][i]], ignore_index = True)

                label['sentence_category_scores'] = pd.DataFrame(sentence_cat_scores.mean()).T
                label['sentence_sentiment_scores'] = pd.DataFrame(sentence_sent_scores.mean()).T

                self.category_scores = pd.concat([self.category_scores, label['sentence_category_scores']], ignore_index = True)
                self.sentiment_scores = pd.concat([self.sentiment_scores, label['sentence_sentiment_scores']], ignore_index = True)

            potentials.append(label)



                
    def normalize_scores(self, labels):
        category_mean = self.category_scores.mean()
        category_std = self.category_scores.std()
        sentiment_mean = self.sentiment_scores.mean()
        sentiment_std = self.sentiment_scores.std()

        for label in labels: 
            if label["aspect"]:
                try:
                    for i in range(len(label["category_scores"])):
                        label["category_scores"][i] = (label["category_scores"][i].subtract(category_mean)).div(category_std)
                    
                    for j in range(len(label["sentiment_scores"])):
                        label["sentiment_scores"][j] = (label["sentiment_scores"][j].subtract(sentiment_mean)).div(sentiment_std)

                    label["sentence_category_scores"] = (label["sentence_category_scores"].subtract(category_mean)).div(category_std)
                    label["sentence_sentiment_scores"] = (label["sentence_sentiment_scores"].subtract(sentiment_mean)).div(sentiment_std)
                except Exception as e:
                    print(e)

        return labels                
                



    def compute_overlap_scores(self, words_in_sentence, token, vocabulary):   
        try:
            words_in_sentence[token.i] = self.mask
        except Exception as e:
            print(token)
            print(words_in_sentence)
            print(token.i)
            print(e)

        masked_sentence = " ".join(words_in_sentence)
        
        tokenized_sentence = self.tokenizer(masked_sentence, return_tensors='pt', padding='max_length', max_length=128).to(self.device)
        replacements = self.get_k_replacements(tokenized_sentence, self.k_2)

        overlap_score_dict = {}
        
        for key in vocabulary:
            overlap_score = sum(np.isin(replacements, vocabulary[key]) * 1)
            overlap_score_dict[key] = [overlap_score]
        
        values = pd.DataFrame.from_dict(overlap_score_dict)
        
        return values


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
