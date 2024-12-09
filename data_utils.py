import numpy as np
import pandas as pd
import json
from pydantic.v1  import BaseModel, Field, Extra
from typing import List, Optional
import re
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import eval_utils



def get_sentiment_seeds(args):
    """
    Returns the sentiment polarity seed words for the dataset 
    
    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 

    output: 
    - sentiment_dict: the sentiment polarity seed words for the dataset
    """
    sentiment_dict = {}
    if args.dataset == 'rest15' or args.dataset == 'rest16':   
        sentiment_dict = {"negative" : ["never", "small", "bad", "overpriced", "arrogant"],
                  "positive" : ["great", "best", "delicious", "excellent", "friendly"]}
    elif args.dataset == 'odido':
        sentiment_dict = {"positief" : ["opgelost", "tevreden", "juiste", "goed", "snel"], 
             "negatief" : ["niet", "probleem", "gefrustreerd", "blokkade", "ontevreden"]}
    
    return sentiment_dict
    
    

def get_sentiment(args):
    """
    Get the dictionary to reformulate the sentiment polarities to for instance natural language

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 

    output: 
    - senttag2opinion: reformulate a "POS" or "NEG" tag to natural langauge 
    - sentword2opinion: reformulate a "positive" or "negative" tag to natural language
    - opinion2word: reformulate the natural language format back to the "positive" or "negative" tag
    """
    if args.dataset == 'rest15' or args.dataset == 'rest16':
        senttag2opinion = {'POS': 'great', 'NEG': 'bad'}
        sentword2opinion = {'positive': 'great', 'negative': 'bad'}
        opinion2word = {'great': 'positive', 'bad': 'negative'}
        return senttag2opinion, sentword2opinion, opinion2word
    elif args.dataset == 'odido':
        senttag2opinion = {'POS': 'goed', 'NEG': 'slecht'}
        sentword2opinion = {'positief': 'goed', 'negatief': 'slecht'}
        opinion2word = {'goed': 'positief', 'slecht': 'negatief'}
        return senttag2opinion, sentword2opinion, opinion2word

    

def get_category_seeds(args):
    """
    Returns the aspect category seed words for the dataset 
    
    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 

    output: 
    - category_dict: the aspect category seed words for the dataset
    """
    category_dict = {}
    if args.dataset == 'rest15' or args.dataset == 'rest16':
        category_dict = {"ambience" : ["experience", "atmosphere", "decor","ambience", "setting"], 
            "drinks" : ["wine", "drinks", "beer" "glass", "drink"], 
            "food" : ["food", "pizza", "dessert", "dinner","dishes"] , 
            "location" : ["place", "street", "spot", "neighbourhood", "location"], 
            "restaurant" : ["restaurant", "seating", "room", "garden", "interior" ], 
            "service" : ["service", "staff", "waitress", "bartender", "hostess"], 
            "prices" : ["price", "money", "value", "cost", "offer"], 
            "miscellaneous" : [ "time", "portion", "delivery", "noise", "entertainment"]}
    elif args.dataset == 'odido':
        category_dict = {}
    return category_dict


def load_fine_tune(args):
    """
    Load the dataset to post train the BERT model

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 

    output: 
    - dataset: the dataset used to post train the BERT model
    """
    if args.dataset == 'odido':
        directory =  f'data/{args.dataset}/2024-samples-all-from-may.csv'
        fine_tune = pd.read_csv(directory)
        train, val = train_test_split(fine_tune, test_size= 500, random_state=args.seed) #500
        train_dataset = extract_sentences(train)
        train = []
        for i in train_dataset:
            train.append(i['sentence'])

        val_dataset = extract_sentences(val)
        val = []
        for i in val_dataset:
            val.append(i['sentence'])
    else:
        raise Exception("Post training only for Odido dataset; restaurant datasets already have a post trained model")
        train = None
        val = None 
    return train, val


def load_test_data(args, data_path):
    """
    Load the test dataset 

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - data_path: name of the test data

    output: 
    - the test dataset 
    """
    data_path = f'data/{args.dataset}/{data_path}.txt'
    with open(data_path, 'r') as file:
            test = file.read()
    dataset = extract_sentences(test)
    return dataset


def extract_labeled_data(args, data_path):
    """
    Load the dataset together with its labels 

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - data_path: name of the test data

    output: 
    - the dataset together with the labels
    """
    data_path = f'data/{args.dataset}/{data_path}.txt'
    with open(data_path, 'r') as file:
            df = file.read()
    sentences, labels = extract_matrices(df)
    sentences = clean_sentences(sentences)

    dataset = []
    for i in range(len(sentences)): 
        target = {'sentence': "", 'label': ''}
        target['sentence'] = sentences[i]
    
        target['quads'] = labels[i]
        target['label'] = labels[i]
        
        dataset.append(target)

    return dataset

def load_labeled_data(args, set):
    """
    Load the automatically labeled dataset and compute the scores for the automatically labeled data

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - set: the automatically labeled dataset which needs to loaded: train, val, test

    output: 
    - scores: the F1 score, precision, and recall for the 
    """
    data_path = f"outputs/{args.dataset}/{set}/{args.labeling_approach}_{args.labeling}"
    with open(data_path, 'r') as file:
        df = file.read()
    dataset = eval(df)

    scores, all_labels, all_preds = eval_utils.compute_scores(dataset, args, set, individual_comparissons = True)

    return scores

def load_train(args):
    """
    Load the sentences for the train and validation data

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 

    output: 
    - train: the train data
    - val: the validation data
    """
    if args.labeled_train_data_available:
        data_path = f"outputs/{args.dataset}/train/{args.labeling_approach}_{args.labeling}"
        with open(data_path, 'r') as file:
            df = file.read()
        train = eval(df)
        data_path = f"outputs/{args.dataset}/val/{args.labeling_approach}_{args.labeling}"
        with open(data_path, 'r') as file:
            df = file.read()
        val = eval(df)

        return train, val
    
    elif args.dataset == 'odido':
        directory =  f'data/{args.dataset}/2024-samples-all-from-june.csv'
        june = pd.read_csv(directory).sample(n = 2500, random_state = args.seed) #2500
        train, val = train_test_split(june, test_size= 500, random_state=args.seed) #500

        train = extract_sentences(train)
        val = extract_sentences(val)

        return train, val

    elif args.dataset == 'rest15' or args.dataset == 'rest16':
        directory_train =  f'data/{args.dataset}/train.txt'
        directory_val =  f'data/{args.dataset}/val.txt'
        with open(directory_train, 'r') as file:
            train = file.read()
        with open(directory_val, 'r') as file:
            val = file.read()

        train = extract_sentences(train)
        val = extract_sentences(val)

        return train, val



def extract_matrices(content):
    """
    extract the sentences and their corresponding labels from the content

    input: 
    - content: the lines read from the text file

    output: 
    - sentences: the sentences in the dataset
    - labels: the labels in the dataste
    """
    pattern = r"(.*?)####(.*?)\n"
    matches = re.findall(pattern, content, re.DOTALL)
    sentences = []  
    labels = []

    for match in matches:
        sentences.append(match[0].strip())
        quads = eval(match[1])
        label = []
        for i in quads:
            quad = list(map(str.lower,i))
            label.append(quad)
        
        labels.append(label)

    return pd.Series(sentences), labels

def clean_sentences(sentences):
    """
    clean sentences such that they are in a uniform format

    input: 
    - sentences: the sentences which need to be cleaned

    output: 
    - sentences: the cleaned sentences
    """
    sentences = sentences.apply(lambda x : x.strip())
    sentences = sentences.str.lower()
    sentences.replace(to_replace=r'[^\w\s]', value="", regex=True, inplace=True)
    sentences.replace(" ", None, inplace = True)
    sentences = pd.Series(filter(None, sentences))
    sentences = sentences.str.replace(' +', ' ', regex=True)
    return sentences

def extract_sentences(df):
    """
    Load the dataset in the format of a list containing dictionaries with as entries the sentences and its corresponding label if possible

    input: 
    - df: the content from which the dataset needs to be extracted in the correct format

    output: 
    - dataset: the dataset in the correct format
    """
    if isinstance(df, pd.DataFrame):
        sentences = df["gpt_summary"].str.split(".").explode()
    else:
        sentences, labels = extract_matrices(df)

    sentences = clean_sentences(sentences)

    dataset = []
    for i in range(len(sentences)): 
        target = {'sentence': "", 'label': ''}
        target['sentence'] = sentences[i]
        if isinstance(df, pd.DataFrame):
            target['label'] = [[]]
        else:
            target['label'] = labels[i]
        
        dataset.append(target)

    return dataset

def get_inference_dataset_paraphrase(args, sentences, tokenizer):
    """
    create the inference dataset for the Paraphrase method

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - sentences: the sentences for the inference
    - tokenizer: the tokenizer used to tokenize sentences for the paraphrase model

    output: 
    - ABSADataset: The dataset class used for the Paraphrase model
    """
    return ABSADataset(tokenizer, sentences, 'inference', max_len = args.max_seq_length)

def get_evaluation_dataset_paraphrase(args, dataset, tokenizer):
    """
    create the evaluation dataset for the Paraphrase method

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - dataset: The dataset containing sentences and the gold labels
    - tokenizer: the tokenizer used to tokenize sentences for the paraphrase model

    output: 
    - ABSADataset: The dataset class used for the Paraphrase model
    """
    sentences = []
    labels = []
    for i in range(len(dataset)):
        sentences.append(dataset[i]['sentence'])
        labels.append(dataset[i]['label'])

    return ABSADataset(tokenizer, sentences, 'evaluate', labels, max_len = args.max_seq_length)

def get_train_dataset_paraphrase(args, labels, tokenizer):
    """
    create the train dataset for the Paraphrase method

    input: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - labels: the automatically labeled datasets with labels and sentences
    - tokenizer: the tokenizer used to tokenize sentences for the paraphrase model

    output: 
    - ABSADataset: The dataset class used for the Paraphrase model
    """
    inputs = []
    targets = []
    senttag2opinion, sentword2opinion, opinion2word = get_sentiment(args)
    for label in labels:
        inputs.append(label['sentence'])
        all_quad_sentences = []
        for quad in label['quads']:
            if quad:
                a, c, s, o = quad
                
                try:    
                    s_change = sentword2opinion[s]  # 'POS' -> 'good'
                except: 
                    s_change = s 
                
                if a == 'NULL':  # for implicit aspect term
                    if args.dataset == 'rest15' or args.dataset == "rest16":
                        a = 'it'
                    elif args.dataset == "odido":
                        a = 'het'

                if args.dataset == 'rest15' or args.dataset == "rest16":  
                    one_quad_sentence = f"{c} is {s_change} because {a} is {o}"
                elif args.dataset == "odido":
                    one_quad_sentence = f"{c} is {s_change} omdat {a} is {o}"

                all_quad_sentences.append(one_quad_sentence)
        
        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    
    return ABSADataset(tokenizer, inputs, 'train' , targets, args.max_seq_length)
        
class ABSADataset(Dataset):
    """
    The class used in the Paraphrase method to handle the data from the code for the Paraphrase paper 

    The class is adjusted a bit to be implemented in the context of this thesis, by specifying the task for the paraphrase either train, evaluate, or inference. The different tasks cause different targets to be stored
    """
    def __init__(self, tokenizer, inputs, task, targets = None, max_len=128):
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = inputs
        self.inputs_t = []
        
        self.targets = targets
        if targets != None and task == 'train':
            self.targets_t = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs_t)

    def __getitem__(self, index):
        source_ids = self.inputs_t[index]["input_ids"].squeeze()
        src_mask = self.inputs_t[index]["attention_mask"].squeeze()  # might need to squeeze

        if self.targets != None and self.task == 'train':
            target_ids = self.targets_t[index]["input_ids"].squeeze()
            target_mask = self.targets_t[index]["attention_mask"].squeeze()  # might need to squeeze
            return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}
        
        elif self.targets != None and self.task == 'evaluate':
            return{"source_ids": source_ids, "source_mask": src_mask, 
                "label": self.targets}
        
        else:
            return {"source_ids": source_ids, "source_mask": src_mask}

        
    def _build_examples(self):

        for i in range(len(self.inputs)):
            # change input and target to two strings
            
            tokenized_input = self.tokenizer.batch_encode_plus(
              [self.inputs[i]], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            self.inputs_t.append(tokenized_input)
            
            if self.targets != None and self.task == 'train':
                target = self.targets[i]
                tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
                )
                self.targets_t.append(tokenized_target)

            
            
        








class Quad(BaseModel):
    aspect: str = Field(description="the aspect term of the quadruplet quadruplet, which is a term mentioned in the sentence")
    category: str = Field(description="the aspect category derived from the aspect term, which is part of the aspect category list")
    opinion: str = Field(description="the optinion term in the quadruplet, which is a term mentioned in the sentence")
    sentiment: str = Field(description="the sentiment polarity derived from the opinion term, which is part of the sentiment polarity list")
    #score: int = Field(description="The certainty score over its predictions on a scale of 1 to 5")

    class Config:
        extra = Extra.forbid  # Forbid extra fields not defined in the model

class Quadruplets(BaseModel):
    sentence: str = Field(description="the sentence for which the quadruplets are extracted")
    quads: List[Quad] = Field(description="A list of the quadruplets in the sentence")
    analysis: Optional[str] = Field(None, description="optional explanation")
