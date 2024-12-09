import pandas as pd
import numpy as np
import re

import save_utils

def compute_scores(pred_seqs, args, data_type, label_time):
    """
    Compute the F1 score, precision, and recall for the predicted sequences for both the whole quadruplets and the individual elements. The ouput is stored in the output file

    input: 
    - pred_seq: a list of dictionaries containing the sentence, predicted quadrupelts, adn the gold label
    - args: the arguments containing which technique to use, but also the thresholds and other parameters
    - data_type: the type of dataset either trian, validation, or test
    - label_time: the time to obtain the quadruplets for the sentence

    output: 
    - scores: the F1 score, precision, and recall for the whole quadruplets
    - all_labels: list of all labels in the dataset
    - all_preds: list of all predictions in the dataset
    """
    num_samples = len(pred_seqs)

    all_labels, all_preds = [], []

    label_cat, pred_cat = [], []
    label_asp, pred_asp = [], []
    label_ot, pred_ot = [], []
    label_st, pred_st = [], []

    for i in range(num_samples):
        gold = pred_seqs[i]['label']
        pred = pred_seqs[i]['quads']
        all_labels.append(gold)
        all_preds.append(pred)
        aspects = []
        categories = []
        opinions = []
        sentiments = []
        for i in gold:
            try:
                a, c, s, o = i
                aspects.append([a])
                categories.append([c])
                opinions.append([o])
                sentiments.append([s])
            except:
                aspects.append([''])
                categories.append([''])
                opinions.append([''])
                sentiments.append([''])
        
        label_asp.append(aspects)
        label_cat.append(categories)
        label_ot.append(opinions)
        label_st.append(sentiments)
        
        aspects = []
        categories = []
        opinions = []
        sentiments = []
        for j in pred:
            try:
                a, c, s, o = j
                aspects.append([a])
                categories.append([c])
                opinions.append([o])
                sentiments.append([s])
            except:
                aspects.append([''])
                categories.append([''])
                opinions.append([''])
                sentiments.append([''])

        pred_asp.append(aspects)
        pred_cat.append(categories)
        pred_ot.append(opinions)
        pred_st.append(sentiments)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels, args)
    print(scores)

    scores_asp = compute_f1_scores(pred_asp, label_asp, args)
    scores_cat = compute_f1_scores(pred_cat, label_cat, args)
    scores_ot = compute_f1_scores(pred_ot, label_ot, args)
    scores_st = compute_f1_scores(pred_st, label_st, args)

    dictionary = {'total': scores, 'aspect': scores_asp, 'category': scores_cat, 'opinion': scores_ot, 'sentiment': scores_st}
    
    save_utils.write_to_file(dictionary, args, data_type, label_time)

    return scores, all_labels, all_preds

def compute_f1_scores(pred_pt, gold_pt, args, individual_comparissons = False):
    """
    Function to compute F1 scores, precision, and recall with predicted elements and gold labels. 

    input: 
    - pred_pt: a list of the predicted elements 
    - gold_pt: a list of the gold labels
    - args: the arguments containing which technique to use, but also the thresholds and other parameters
    - individual_comparison: print the predicted labels and gold labels next to each other to compare them individually and see the differences. The default value is set to False

    output: 
    - scores: the obtained F1 score, precision, and recall metrics for the predicted sequence 
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(gold_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        if individual_comparissons:
            print(f'{gold_pt[i]} VS {pred_pt[i]}')
            
            
        temp = list(gold_pt[i].copy())
        for t in pred_pt[i]:
            if t in temp:
                n_tp += 1
                temp.remove(t)

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

