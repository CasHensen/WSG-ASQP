import pandas as pd
import numpy as np
import re

import save_utils

def compute_scores(pred_seqs, args, data_type, label_time = -1, individual_comparissons = False):
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
    scores = compute_f1_scores(all_preds, all_labels, args, individual_comparissons)
    print(scores)

    scores_asp = compute_f1_scores(pred_asp, label_asp, args)
    scores_cat = compute_f1_scores(pred_cat, label_cat, args)
    scores_ot = compute_f1_scores(pred_ot, label_ot, args)
    scores_st = compute_f1_scores(pred_st, label_st, args)

    dictionary = {'total': scores, 'aspect': scores_asp, 'category': scores_cat, 'opinion': scores_ot, 'sentiment': scores_st}
    
    if not individual_comparissons:
        save_utils.write_to_file(dictionary, args, data_type, label_time)

    return scores, all_labels, all_preds

def compute_f1_scores(pred_pt, gold_pt, args, individual_comparissons = False):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
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
                #if individual_comparissons:
                #    print(f'{gold_pt[i]} VS {pred_pt[i]}')
                n_tp += 1
                temp.remove(t)

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

