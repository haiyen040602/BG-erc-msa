import json
import logging
import re
import copy

import editdistance
import numpy as np

from constants import *
from sklearn.metrics import classification_report
from itertools import chain
from data_utils import (extract_iemocap_from_extraction_universal,
                        extract_moseii_from_extraction_universal, 
                        extract_meld_from_extraction_universal)

logger = logging.getLogger(__name__)

sentiment_word_list = ['positive', 'negative', 'neutral']

def extract_spans_extraction(task, seq, io_format):
    if "moseii" in task:
        if io_format:
            return extract_moseii_from_extraction_universal(seq)
        else:
            raise NotImplementedError
    elif "meld" in task:
        if io_format:
            return extract_meld_from_extraction_universal(seq)
        else:
            raise NotImplementedError
    elif "iemocap" in task:
        if io_format:
            return extract_iemocap_from_extraction_universal(seq)
        else:
            raise NotImplementedError



def fix_preds_moseii(preds, truths):
    moseii_dicts = ['neu', 'neg', 'pos']
    wrong_cnt = 0
    new_preds = preds.copy()
    new_truths = truths.copy()
    for i in range(len(truths)):
        if preds[i][0] not in moseii_dicts:
            wrong_cnt = wrong_cnt + 1
            print("WRONG PREDICTION: ", preds[i])
            new_preds.pop(i-wrong_cnt+1)
            new_truths.pop(i-wrong_cnt+1)
            # print(truths)
            # print(new_truths)
            # print(preds)
            # print(new_preds)
    return wrong_cnt, new_preds, new_truths


def fix_preds_meld(preds, truths):
    meld_dicts = ['neutral', 'surprise', 'anger', 'disgust', 'fear', 'joy', 'sadness']
    wrong_cnt = 0
    new_preds = preds.copy()
    new_truths = truths.copy()
    for i in range(len(truths)):
        if preds[i][0] not in meld_dicts:
            wrong_cnt = wrong_cnt + 1
            print("WRONG PREDICTION: ", preds[i])
            new_preds.pop(i-wrong_cnt+1)
            new_truths.pop(i-wrong_cnt+1)
            # print(truths)
            # print(new_truths)
            # print(preds)
            # print(new_preds)
    return wrong_cnt, new_preds, new_truths


def fix_preds_iemocap(preds, truths):
    iemocap_dicts = ['neutral', 'excited', 'angry', 'joy', 'sadness', 'frustrated']
    wrong_cnt = 0
    new_preds = preds.copy()
    new_truths = truths.copy()
    for i in range(len(truths)):
        if preds[i][0] not in iemocap_dicts:
            wrong_cnt = wrong_cnt + 1
            print("WRONG PREDICTION: ", preds[i])
            new_preds.pop(i-wrong_cnt+1)
            new_truths.pop(i-wrong_cnt+1)
            # print(truths)
            # print(new_truths)
            # print(preds)
            # print(new_preds)
    return wrong_cnt, new_preds, new_truths

def remove_error_predictions(all_predictions, all_labels, task):
    if "moseii" in task:
        fixed_preds = fix_preds_moseii(all_predictions, all_labels)
    elif "meld" in task:
        wrong_cnt, fixed_preds, fixed_truths = fix_preds_meld(all_predictions, all_labels)
    elif "iemocap" in task:
        wrong_cnt, fixed_preds, fixed_truths = fix_preds_iemocap(all_predictions, all_labels)
    else:
        logger.info("*** Unimplemented Error ***")
        wrong_cnt = 0
        fixed_preds = all_predictions
        fixed_truths = all_labels

    return wrong_cnt, fixed_preds, fixed_truths


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0
    gold_pt = copy.deepcopy(gold_pt)

    for i in range(len(pred_pt)):
        
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                # to prevent generate same correct answer and get recall larger than 1
                gold_pt[i].remove(t)
                n_tp += 1
    
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

def eval_emotionlines(results, truths):
    truths = list(chain(*truths))
    # cm = confusion_matrix(results, truths)
    report = classification_report(truths, results)
    print(report)

def compute_scores(pred_seqs, gold_seqs, sents, paradigm, task, verbose=False):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []
    score_truths, score_preds = [], []

    for i in range(num_samples):
        if "extraction" in paradigm:
            gold_list = extract_spans_extraction(task, gold_seqs[i], paradigm)
            pred_list = extract_spans_extraction(task, pred_seqs[i], paradigm)
            if "moseii" in task:
                score_truths.append(float(gold_list[1][0]))
                score_preds.append(float(pred_list[1][0]))
                all_labels.append(gold_list[0])
                all_predictions.append(pred_list[0])
            else:
                all_labels.append(gold_list)
                all_predictions.append(pred_list)

    score_preds = np.array(score_preds)
    score_truths = np.array(score_truths)
    mae = 0
    if "moseii" in task:
        mae = np.mean(np.absolute(score_preds - score_truths))

    raw_scores = compute_f1_scores(all_predictions, all_labels)
    raw_scores["mae"] = mae
    # fix the issues due to generation
    wrong_cnt, fixed_preds, fixed_truths = remove_error_predictions(all_predictions, all_labels, task)
    fixed_scores = compute_f1_scores(fixed_preds, fixed_truths)
    fixed_scores["mae"] = mae

    if verbose:
        # for i in range(3):   
        #     logger.info(f"Gold: {gold_seqs[i]}")
        #     logger.info(f"Gold list: {all_labels[i]}")
        #     logger.info(f"Pred: {pred_seqs[i]}")
        #     logger.info(f"Pred list: {all_predictions[i]}")
        logger.info("MAE of raw output")
        logger.info(str(mae))
        # logger.info("Results of fixed output")
        # logger.info(str(fixed_scores))
        logger.info("Number of error predictions: ")
        logger.info(str(wrong_cnt))
        eval_emotionlines(all_predictions, all_labels)
        eval_emotionlines(fixed_preds, fixed_truths)

    return raw_scores, fixed_scores, all_labels, all_predictions, all_predictions


def avg_n_seeds_by_pair(output_dir, dirs, decode_txt, n_runs):
    score_avg_dict = {}
    score_type_list = ["raw_scores", "fixed_scores"]
    metric_list = ["precision", "recall", "f1", "mae"]
    pairs = []

    # collect value
    for dir_ in dirs:
        for score_type in score_type_list:
            if score_type not in score_avg_dict:
                score_avg_dict[score_type] = {}
            pair = dir_.split('/')[-1]
            src, tgt = pair.split('-')
            if pair not in score_avg_dict[score_type]:
                score_avg_dict[score_type][pair] = {}
            score_dict_i = json.load(open(f"{dir_}/score/test_{decode_txt}_score.json","r"))

            for metric in metric_list:
                if metric not in score_avg_dict[score_type][pair]:
                    score_avg_dict[score_type][pair][metric] = []
                score_avg_dict[score_type][pair][metric].append(score_dict_i[score_type][tgt][metric])

    # get all value
    for score_type in score_type_list:
        all_mat_dict = {k: [] for k in metric_list}
        for pair in score_avg_dict[score_type]:
            for metric in metric_list:
                f1_list_by_seed = score_avg_dict[score_type][pair][metric]
                all_mat_dict[metric].append(f1_list_by_seed)
        for metric in metric_list:
            if "all" not in score_avg_dict[score_type]:
                score_avg_dict[score_type]["all"] = {}
            score_avg_dict[score_type]["all"][metric] = np.mean(all_mat_dict[metric], axis=0)

    # avg value
    for score_type in score_type_list:
        for pair in score_avg_dict[score_type]:
            for metric in metric_list:
                mean = np.mean(score_avg_dict[score_type][pair][metric])
                std = np.std(score_avg_dict[score_type][pair][metric])
                score_avg_dict[score_type][pair][metric] = (mean, std)

    # visual result
    for score_type in score_type_list:
        logger.info('@'*100)
        logger.info(f"Avged {n_runs} runs {score_type}")
        logger.info('\t'.join(list(score_avg_dict[score_type].keys())))
        f1_list = [i["f1"][0] for i in list(score_avg_dict[score_type].values())]
        logger.info('\t'.join([f"{i*100:.2f}" for i in f1_list]))
        f1_std_list = [i["f1"][1] for i in list(score_avg_dict[score_type].values())]
        logger.info('\t'.join([f"{i*100:.2f}" for i in f1_std_list]))

    json.dump(score_avg_dict, open(output_dir+f"/score_{decode_txt}_avg.json", "w"), indent=2)