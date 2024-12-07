import pandas as pd
import numpy as np
import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import time

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

import data_utils
import eval_utils


class paraphrase():
    """
    The class carries out the Paraphrase generative approach 

    The paraphrase post trains the PLM T5 for the ASQP task. The ASQP is executed by formatting the targets as a natural language sentence

    attributes: 
    - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
    - device: the device shows whether it is run on CPU or on GPU
    - tfm_model: the genertive T5 model
    - tokenizer: the tokenizer associated with the T5 model
    - model: the post trained T5 model on the trian and validation data
    """
    def __init__(self, args, train_data, val_data):
         """
        The intialization function to assign variables to the class attributes.
        
        input: 
        - args: the arguments containing which technique to use, but also the thresholds and other parameters. 
        - train_data: the training data to post train the T5 model
        - val_data: the validation data to post train the T5 model
        """
        self.args = args
        self.device = args.device
        directory_T5 = f'{self.args.fine_tuned_T5}/T5_{self.args.labeling_approach}_{self.args.labeling}'
        directory_Tokenizer = f'{self.args.fine_tuned_T5}/tok_t5_{self.args.labeling_approach}_{self.args.labeling}'
        if args.T5_is_fine_tuned:
            self.tfm_model = T5ForConditionalGeneration.from_pretrained(directory_T5).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(directory_Tokenizer)
            self.model = T5FineTuner(self.args, self.tfm_model, self.tokenizer, train_data, val_data)
        else: 
            self.tfm_model = T5ForConditionalGeneration.from_pretrained(self.args.T5).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.T5)
            self.model = T5FineTuner(self.args, self.tfm_model, self.tokenizer, train_data, val_data)
            self.do_training()
        
    
    def do_training(self):
        """
        Starts the post training of the T5 model
        """
        print("\n****** Conduct Training ******")
        # prepare for trainer
        train_params = dict(
            default_root_dir=self.args.output_dir,
            accumulate_grad_batches=self.args.gradient_accumulation_steps,
            gpus=self.args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=self.args.num_train_epochs,
            callbacks=[LoggingCallback()],
        )

        trainer = pl.Trainer(**train_params, weights_summary=None)
        trainer.fit(self.model)

        #self.model.model.save_pretrained(f'{self.args.fine_tuned_T5}/T5_{self.args.labeling_approach}_{self.args.labeling}')
        #self.tokenizer.save_pretrained(f'{self.args.fine_tuned_T5}/tok_t5_{self.args.labeling_approach}_{self.args.labeling}')

        print("Finish training and saving the model!")
    
    def evaluate(self, dataset):
        """
        Evaluates the T5 model on a test dataset using the F1 score, precision, and recall

        input: 
        - dataset: the test dataset used to evaluate the T5 model

        output: 
        - targets: the obtained targets for the test data as a list of dictionaries, where each dictionary has an entry for the sentence, the extracted quadruplets, and the gold labels
        """
        evaluation_dataset = data_utils.get_evaluation_dataset_paraphrase(self.args, dataset, self.tokenizer)
        evaluation_loader =  DataLoader(evaluation_dataset, batch_size=32, num_workers=4)

        outputs = []
        label = []
        for batch in tqdm(evaluation_loader):
            outs = self.model.model.generate(input_ids=batch['source_ids'].to(self.args.device), 
                                        attention_mask=batch['source_mask'].to(self.args.device), 
                                        max_length=128)  # num_beams=8, early_stopping=True)
            dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            outputs.extend(dec)
            label.extend(batch['label'])
        targets = []
        gold = []
        for i in range(len(outputs)):
            quad = self.extract_spans_para(outputs[i])
            target = {'sentence': dataset[i]['sentence'], 'quads': quad, 'label': dataset[i]['label']}
            targets.append(target)
        return targets

    
    def inference(self, sentences):
        """
        The function returns the quadruplets extracted from a dataset

        input: 
        - sentence: the sentences from which the quadruplets need to be extracted

        output: 
        - targets: the obtained targets for the test data as a list of dictionaries, where each dictionary has an entry for the sentence and the extracted quadruplets
        """
        test_dataset = data_utils.get_inference_dataset_paraphrase(self.args, sentences, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

        outputs = []
        for batch in tqdm(test_loader):
            outs = self.model.model.generate(input_ids=batch['source_ids'].to(self.args.device), 
                                        attention_mask=batch['source_mask'].to(self.args.device), 
                                        max_length=128)  # num_beams=8, early_stopping=True)
            dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            outputs.extend(dec)
            
        
        targets = []
        for i in range(len(outputs)):
            quad = self.extract_spans_para(outputs[i])
            target = {'sentence': sentences[i], 'quads': quad}
            targets.append(target)
            
        return targets



    def extract_spans_para(self, seq):
        """
        The function extracts the quadruplets from the natural language sentence generated by the T5 model

        input: 
        - seq: the sequence from which the quadruplets need to be extracted

        output: 
        - quads: the extracted quadruplets from the generated sequence
        """
        labels = [i.strip() for i in seq.split('[SSEP]')]
        quads = []
        for label in labels:
            quad = []
            if self.args.dataset == 'odido':
                try:
                    ac_sp, at_ot = label.split(' omdat ')
                    ac, sp = ac_sp.split(' is ')
                    at, ot = at_ot.split(' is ')

                    # if the aspect term is implicit
                    if at.lower() == 'het':
                        at = 'NULL'
                    quad = [at.lower(), ac.lower(), sp.lower(), ot.lower()]
                except ValueError:
                    ac, at, sp, ot = '', '', '', ''

            else:
                try:
                    ac_sp, at_ot = label.split(' because ')
                    ac, sp = ac_sp.split(' is ')
                    at, ot = at_ot.split(' is ')

                    # if the aspect term is implicit
                    if at.lower() == 'it':
                        at = 'NULL'
                    
                    quad = [at.lower(), ac.lower(), sp.lower(), ot.lower()]
                except ValueError:
                    ac, at, sp, ot = '', '', '', ''
                    
            quads.append(quad)
        return quads





class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        #for key in sorted(metrics):
        #    if key not in ["log", "progress_bar"]:
        #        logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        #output_test_results_file = os.path.join(pl_module.args.output_dir, "test_results.txt")
        #with open(output_test_results_file, "w") as writer:
        #    for key in sorted(metrics):
        #        if key not in ["log", "progress_bar"]:
        #            logger.info("{} = {}\n".format(key, str(metrics[key])))
        #            writer.write("{} = {}\n".format(key, str(metrics[key])))


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    Code from the Paraphrase paper, adjusted a bit for the implementation of my thesis 
    """
    def __init__(self, args, tfm_model, tokenizer, train, val):
        super(T5FineTuner, self).__init__()
        self.train_data = train
        self.val_data = val
        self.args = args
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = data_utils.get_train_dataset_paraphrase(self.args, self.train_data, tokenizer=self.tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
            // self.args.gradient_accumulation_steps
            * float(self.args.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = data_utils.get_train_dataset_paraphrase(self.args, self.val_data, tokenizer=self.tokenizer)
        return DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=4)
