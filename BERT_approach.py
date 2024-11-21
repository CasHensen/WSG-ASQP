import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import spacy

from Vocabulary_generator import vocabulary_generator
from Score_computer import score_computer
from Assign_labels import assign_labels
import data_utils


class bert_approach():
    def __init__(self, args, category_seeds, sentiment_seeds):
        self.category_seeds = category_seeds
        self.sentiment_seeds = sentiment_seeds
        self.device = args.device
        self.args = args 

        if args.BERT_is_fine_tuned:
            BertForMaskedLM.from_pretrained(f'{args.fine_tuned_BERT}/model').to(self.device)
        
        self.bert = BertForMaskedLM.from_pretrained(args.BERT).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.BERT)
        self.nlp = spacy.load(self.args.spacy)

        self.bert_masking = self.tokenizer.mask_token

    def __call__(self, train, val):
        if self.args.fine_tune_bert and not self.args.BERT_is_fine_tuned:
            self.bert = self.fine_tune()
            self.bert.to(self.device)
        
        sentences = []
        for i in train:
            sentences.append(i['sentence'])

        vocabularies = vocabulary_generator(self.args, sentences, self.category_seeds, self.sentiment_seeds, self.bert, self.tokenizer, self.nlp)

        self.category_vocab, self.sentiment_vocab  = vocabularies.get_vocabularies()
        
        scorer = score_computer(self.args, self.bert, self.tokenizer, self.nlp, self.category_vocab, self.sentiment_vocab)
        scores_train = scorer(train)
        scores_val = scorer(val)

        labeler = assign_labels(self.args, self.bert, self.tokenizer, self.nlp)
        labels_train = labeler(scores_train)
        labels_val = labeler(scores_val)

        return labels_train, labels_val

    
    
    def fine_tune(self, masking_percentage = 0.15, learning_rate = 0.0001, batch_size = 64, epochs = 15):
        train, val = data_utils.load_fine_tune(self.args)
        train = pd.DataFrame(data = train, columns = ['sentences'])
        val = pd.DataFrame(data = val, columns = ['sentences'])
        train = Dataset.from_pandas(train)
        val = Dataset.from_pandas(val)
        train = train.map(
            self.tokenize_function, batched=True, remove_columns=[]
            )
        val = val.map(
            self.tokenize_function, batched=True, remove_columns=[]
            )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=masking_percentage)
        
        training_args = TrainingArguments(
            output_dir=self.args.fine_tuned_BERT,
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="no",
            num_train_epochs=epochs,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.bert,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model(f'{self.args.fine_tuned_BERT}/model')
        
        return self.bert


    def tokenize_function(self, sentences):
        result = self.tokenizer(sentences["sentences"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
