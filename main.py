import argparse
import os
import torch
import shutil

from Labeler import Labeler
from Generative_model import generative_model
import data_utils
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_args():
    parser = argparse.ArgumentParser()
    # Select the dataset
    parser.add_argument("--dataset", default='odido', type=str, required=False,
                            help="The name of the dataset, selected from: [rest15, rest16, odido]")

    # Select the approaches
    parser.add_argument("--labeling_approach", default='gpt', 
                        help="The labeling approach, selected from: [bert, gpt, labeled]")
    parser.add_argument("--generative_approach", default='paraphrase', 
                        help="The labeling approach, selected from: [paraphrase, gpt, uniform]")
    
    args = parser.parse_args()

    #determine if BERT is already fine tuned
    if os.path.exists(f"outputs/{args.dataset}/BERT/model"):
        parser.add_argument("--BERT_is_fine_tuned", default = True)
    else: 
        parser.add_argument("--BERT_is_fine_tuned", default = False)

    #select the approach for labeling
   

    if args.labeling_approach == 'bert':
        parser.add_argument("--labeling", default='one', 
                        help="The labeling approach for the bert labeling, selected from: [one, multiple_pos, multiple_attention]")
    elif args.labeling_approach == 'gpt':
        parser.add_argument("--labeling", default='evaluator', 
                        help="The labeling approach for the GPT labeling, selected from: [evaluator, no_evaluator]")
    else:
        parser.add_argument("--labeling", default='', 
                        help="The labeling approach for the GPT labeling, selected from:")
    
    args = parser.parse_args()
    if os.path.exists(f'outputs/{args.dataset}/train/{args.labeling_approach}_{args.labeling}'):
        parser.add_argument("--labeled_train_data_available", default=True, 
                        help="The labeling approach, selected from: [True, False]")
    else:
        parser.add_argument("--labeled_train_data_available", default=False, 
                        help="The labeling approach, selected from: [True, False]")

    # Dataset parameters
    args = parser.parse_args()
    if args.dataset == 'odido':
        parser.add_argument("--T5", default='yhavinga/t5-base-dutch', type=str,
                        help="Path to pre-trained model or shortcut name, selected from: [yhavinga/t5-base-dutch, t5-base]")
        parser.add_argument("--BERT", default='GroNLP/bert-base-dutch-cased', type=str,
                        help="Path to pre-trained model or shortcut name for bert, selected from: [activebus/BERT-DK_rest, GroNLP/bert-base-dutch-cased]")
        parser.add_argument("--spacy", default='nl_core_news_sm', 
                        help="The pre trained spacy model, selected from: [en_core_web_sm, nl_core_news_sm]")
        parser.add_argument("--fine_tune_bert", default=True, 
                        help="The pre trained spacy model, selected from: [en_core_web_sm, nl_core_news_sm]")
    else:
        parser.add_argument("--T5", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name, selected from: [yhavinga/t5-base-dutch, t5-base]")
        parser.add_argument("--BERT", default='activebus/BERT-DK_rest', type=str,
                        help="Path to pre-trained model or shortcut name for bert, selected from: [activebus/BERT-DK_rest, GroNLP/bert-base-dutch-cased]")
        parser.add_argument("--spacy", default='en_core_web_sm', 
                        help="The pre trained spacy model, selected from: [en_core_web_sm, nl_core_news_sm]")
        parser.add_argument("--fine_tune_bert", default=False, 
                        help="The pre trained spacy model, selected from: [en_core_web_sm, nl_core_news_sm]")
        args.BERT_is_fine_tuned = True
        

    parser.add_argument("--gpt", default='sweden-gpt-35.env', 
                        help="The environment for the gpt model")

    #labeling parameters changable
    parser.add_argument('--k', type=int, default=30,
                        help="the number of replacements found for vocabulary creation")
    parser.add_argument('--k_2', type=int, default=30,
                        help="the number of replacements found")
    parser.add_argument('--m', type=int, default=75,
                        help="the size of the vocabularies")
    parser.add_argument('--threshold', type=int, default=0.0,
                        help="the size of the vocabularies")
    parser.add_argument('--threshold_attn', type=int, default=0.1,
                        help="the size of the vocabularies")
    parser.add_argument('--gpt_threshold', type=int, default=7,
                        help="the size of the vocabularies")

    # Model parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform. default = 30, set to 5 for demo")
    parser.add_argument('--seed', type=int, default=52,
                        help="random seed for initialization")
    parser.add_argument("--num_threads", default=10, type=int, 
                        help="Total number of threads.")

    # Environment parameters
    parser.add_argument("--n_gpu", default=torch.cuda.device_count())
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    # training parameters
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    fine_tuned_T5 = f"outputs/{args.dataset}/T5"
    if not os.path.exists(fine_tuned_T5):
        os.mkdir(fine_tuned_T5)
    args.fine_tuned_T5 = fine_tuned_T5

    fine_tuned_BERT = f"outputs/{args.dataset}/BERT"
    if not os.path.exists(fine_tuned_BERT):
        os.mkdir(fine_tuned_BERT)
    args.fine_tuned_BERT = fine_tuned_BERT

    # Tasks on/off for fine tuning and labeling
    if os.path.exists(f'{args.fine_tuned_T5}/T5_{args.labeling_approach}_{args.labeling}'):
        args.T5_is_fine_tuned = True
    else: 
        args.T5_is_fine_tuned = False
    

    return args

args = init_args()
execute = True

if execute:
    print("\n", "="*30, f"NEW EXP: WG_ASQP on {args.dataset}", "="*30, "\n")
    print("\n", "="*11, f"With labeling approach: {args.labeling_approach} and generative approach: {args.generative_approach}", "="*11, "\n")

    labeler = Labeler(args)
    train, val = labeler()

    model = generative_model(args, train, val)
    scores, targets = model.evaluate()

    shutil.rmtree(f"outputs/{args.dataset}/lightning_logs")
else:
    data_utils.load_labeled_data(args, 'test')




    
