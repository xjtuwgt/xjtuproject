import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from time import time
import pandas as pd
import swifter
from pandas import DataFrame
from multihopr.hotpotqaIOUtils import HOTPOT_DevData_Distractor
from transformers import LongformerTokenizer
from multihopQA.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
from multihopQA.hotpot_evaluate_v1 import f1_score, exact_match_score

def load_data_frame(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def load_data_frame_align_with_dev(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    orig_dev_data, _ = HOTPOT_DevData_Distractor()
    assert orig_dev_data.shape[0] == data_frame.shape[0]
    data_frame = add_row_idx(data_frame)
    orig_dev_data = add_row_idx(orig_dev_data)
    data = data_frame.merge(orig_dev_data, left_on='row_idx', right_on='row_idx')
    print('Merging {} in {:.4f} seconds'.format(data.shape, time() - start_time))
    return data

def get_all_json_files(file_path: str, extension: str = '.json'):
    file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith(extension)]
    return file_names

def update_sp_score(prediction, gold):
    cur_sp_pred = set(prediction)
    gold_sp_pred = set(gold)
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1

def update_sp_sent(prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1

def evaluation(data: DataFrame, tokenizer: LongformerTokenizer, joint_evalution=True):
    # 'aty_pred': answer_type_pred_results, 'aty_true': answer_type_true_results,
    # 'sd_pred': support_doc_pred_results, 'sd_true': support_doc_true_results,
    # 'ss_pred': support_sent_pred_results, 'ss_true': support_sent_true_results,
    # 'sps_pred': span_pred_start_results, 'spe_pred': span_pred_end_results,
    # 'sps_true': span_true_start_results, 'spe_true': span_true_end_results,
    # 'encode_ids': encode_id_results, 'ss_score': sport_sent_prediction_score, 'ss_ds_pair': doc_sent_fact_pair
    def row_evaluation(row):
        answer_type_prediction, answer_type_true = row['aty_pred'], row['aty_true']
        supp_doc_prediction, supp_doc_true = row['sd_pred'], row['sd_true']
        supp_sent_prediciton, supp_sent_true = row['ss_pred'], row['ss_true']
        span_prediction_start, span_prediction_end = row['sps_pred'], row['spe_pred']
        # span_true_start, span_true_end = row['sps_true'][0], row['spe_true'][0]
        span_true_start, span_true_end = row['sps_true'], row['spe_true']
        # print(span_true_start, span_true_end)
        encode_ids = row['encode_ids']


        if answer_type_prediction > 0:
            span_prediction = 'yes' if answer_type_prediction == 1 else 'no'
        else:
            span_prediction = tokenizer.decode(encode_ids[span_prediction_start:(span_prediction_end+1)], skip_special_tokens=True)

        if answer_type_true > 0:
            span_gold = 'yes' if answer_type_true == 1 else 'no'
        else:
            span_gold = tokenizer.decode(encode_ids[span_true_start:(span_true_end+1)], skip_special_tokens=True)

        span_em = exact_match_score(span_prediction, span_gold)
        span_f1, span_prec, span_recall = f1_score(span_prediction, span_gold)
        sp_doc_em, sp_doc_prec, sp_doc_recall, sp_doc_f1 = update_sp_score(supp_doc_prediction, supp_doc_true)
        sp_sent_em, sp_sent_prec, sp_sent_recall, sp_sent_f1 = update_sp_score(supp_sent_prediciton, supp_sent_true)
        answer_true = 1 if answer_type_prediction == answer_type_true else 0

        if joint_evalution:
            joint_prec = span_prec * sp_sent_prec
            joint_recall = span_recall * sp_sent_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = span_em * sp_sent_em
        else:
            joint_prec, joint_recall, joint_f1, joint_em = 0,  0, 0, 0
        return answer_true, span_em, span_f1, span_prec, span_recall, sp_doc_em, sp_doc_f1, sp_doc_prec, \
               sp_doc_recall, sp_sent_em, sp_sent_f1, sp_sent_prec, sp_sent_recall, joint_em, joint_f1, joint_prec, joint_recall

    res_names = ['answer_type_pred', 'span_em', 'span_f1', 'span_prec', 'span_recall',
                 'sp_doc_em', 'sp_doc_f1', 'sp_doc_prec', 'sp_doc_recall', 'sp_sent_em', 'sp_sent_f1', 'sp_sent_prec', 'sp_sent_recall',
                 'joint_em', 'joint_f1', 'joint_prec', 'joint_recall']
    data[res_names] = data.swifter.apply(lambda row: pd.Series(row_evaluation(row)), axis=1)
    return data, res_names

def evaluation2(data: DataFrame, tokenizer: LongformerTokenizer, joint_evalution=True):
    # 'aty_pred': answer_type_pred_results, 'aty_true': answer_type_true_results,
    # 'sd_pred': support_doc_pred_results, 'sd_true': support_doc_true_results,
    # 'ss_pred': support_sent_pred_results, 'ss_true': support_sent_true_results,
    # 'sps_pred': span_pred_start_results, 'spe_pred': span_pred_end_results,
    # 'sps_true': span_true_start_results, 'spe_true': span_true_end_results,
    # 'encode_ids': encode_id_results
    def row_evaluation(row):
        answer_type_prediction, answer_type_true = row['aty_pred'], row['aty_true']
        supp_doc_prediction, supp_doc_true = row['sd_pred'], row['sd_true']
        supp_sent_prediction, supp_sent_true = row['ss_pred'], row['ss_true']
        supp_sent_score, supp_sent_fact_pair = row['ss_score'], row['ss_ds_pair']
        #####
        supp_fact_pair_prediction = [supp_sent_fact_pair[i] for i in supp_sent_prediction]
        ctx_documents = row['context']
        support_facts = row['supporting_facts']
        support_facts_gold = [(x[0], x[1]) for x in support_facts]
        support_facts_prediction = []
        for doc_idx, sent_idx in supp_fact_pair_prediction:
            support_facts_prediction.append((ctx_documents[doc_idx][0], sent_idx))
        # print(support_facts_prediction)
        # print(support_facts_gold)
        sp_sent_pair_em, sp_sent_pair_prec, sp_sent_pair_recall, sp_sent_pair_f1 = update_sp_sent(support_facts_prediction, support_facts_gold)
        #####
        span_prediction_start, span_prediction_end = row['sps_pred'], row['spe_pred']
        span_true_start, span_true_end = row['sps_true'], row['spe_true']
        encode_ids = row['encode_ids']

        if answer_type_prediction > 0:
            span_prediction = 'yes' if answer_type_prediction == 1 else 'no'
        else:
            span_prediction = tokenizer.decode(encode_ids[span_prediction_start:(span_prediction_end+1)], skip_special_tokens=True)

        if answer_type_true > 0:
            span_gold = 'yes' if answer_type_true == 1 else 'no'
        else:
            span_gold = tokenizer.decode(encode_ids[span_true_start:(span_true_end+1)], skip_special_tokens=True)

        span_em = exact_match_score(span_prediction, span_gold)
        span_f1, span_prec, span_recall = f1_score(span_prediction, span_gold)
        sp_doc_em, sp_doc_prec, sp_doc_recall, sp_doc_f1 = update_sp_score(supp_doc_prediction, supp_doc_true)
        sp_sent_em, sp_sent_prec, sp_sent_recall, sp_sent_f1 = update_sp_score(supp_sent_prediction, supp_sent_true)
        answer_true = 1 if answer_type_prediction == answer_type_true else 0

        if joint_evalution:
            joint_prec = span_prec * sp_sent_prec
            joint_recall = span_recall * sp_sent_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = span_em * sp_sent_em
        else:
            joint_prec, joint_recall, joint_f1, joint_em = 0,  0, 0, 0

        #++++++++++++++++++++++++
        if joint_evalution:
            pair_joint_prec = span_prec * sp_sent_pair_prec
            pair_joint_recall = span_recall * sp_sent_pair_recall
            if pair_joint_prec + pair_joint_recall > 0:
                pair_joint_f1 = 2 * pair_joint_prec * pair_joint_recall / (pair_joint_prec + pair_joint_recall)
            else:
                pair_joint_f1 = 0.
            pair_joint_em = span_em * sp_sent_pair_em
        else:
            pair_joint_prec, pair_joint_recall, pair_joint_f1, pair_joint_em = 0, 0, 0, 0
        #++++++++++++++++++++++++
        return answer_true, span_em, span_f1, span_prec, span_recall, sp_doc_em, sp_doc_f1, sp_doc_prec, \
               sp_doc_recall, sp_sent_em, sp_sent_f1, sp_sent_prec, sp_sent_recall, joint_em, joint_f1, joint_prec, joint_recall, \
               sp_sent_pair_em, sp_sent_pair_f1, sp_sent_pair_prec, sp_sent_pair_recall, pair_joint_em, pair_joint_f1, pair_joint_prec, \
               pair_joint_recall

    res_names = ['answer_type_pred', 'span_em', 'span_f1', 'span_prec', 'span_recall',
                 'sp_doc_em', 'sp_doc_f1', 'sp_doc_prec', 'sp_doc_recall', 'sp_sent_em', 'sp_sent_f1', 'sp_sent_prec', 'sp_sent_recall',
                 'joint_em', 'joint_f1', 'joint_prec', 'joint_recall',
                 'sp_sent_pair_em', 'sp_sent_pair_f1', 'sp_sent_pair_prec', 'sp_sent_pair_recall',
                 'pair_joint_em', 'pair_joint_f1', 'pair_joint_prec', 'pair_joint_recall']
    data[res_names] = data.swifter.apply(lambda row: pd.Series(row_evaluation(row)), axis=1)
    return data, res_names


def add_row_idx(data: DataFrame):
    data['row_idx'] = range(0, len(data))
    return data

def performance_evaluation(data_frame: DataFrame, res_names):
    res = dict()
    for idx, col_name in enumerate(res_names):
        metric = data_frame[col_name].mean()
        res[col_name] = metric
    return res

def print_metrics(metric_dict: dict, res_file_name: str, file_idx: int, total_num: int):
    print('*' * 75)
    for key, value in metric_dict.items():
        print('{}/{}: {}: {}\t= {:.5f}'.format(file_idx, total_num, res_file_name, key, value))
    print('*' * 75)

def performance_collection(folder_name):
    print('Loading tokenizer')
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    json_file_names = get_all_json_files(file_path=folder_name)
    json_file_names = [x for x in json_file_names if x != 'config.json']
    total_json_num = len(json_file_names)
    print('{} json files have been found'.format(len(json_file_names)))
    max_sp_sent_f1 = 0
    max_metric_res = None
    max_json_file_name = None
    for idx, json_file_name in enumerate(json_file_names):
        if json_file_name != 'config.json':
            data_frame = load_data_frame(file_path=folder_name, json_fileName=json_file_name)
            col_names_set = set([col for col in data_frame.columns])
            # for col in data_frame.columns:
            #     print(col)
            if 'ss_ds_pair' in col_names_set:
                data_frame, res_names = evaluation2(data=data_frame, tokenizer=tokenizer)
            else:
                data_frame, res_names = evaluation(data=data_frame, tokenizer=tokenizer)
            metric_res = performance_evaluation(data_frame=data_frame, res_names=res_names)
            print_metrics(metric_dict=metric_res, res_file_name=json_file_name, file_idx=idx + 1, total_num=total_json_num)
            if max_sp_sent_f1 < metric_res['sp_sent_f1']:
                max_sp_sent_f1 = metric_res['sp_sent_f1']
                max_metric_res = metric_res
                max_json_file_name = json_file_name
    print('*' * 75)
    print('-' * 35 + 'MAX METRIC' + '-'*30)
    print_metrics(metric_dict=max_metric_res, res_file_name=max_json_file_name, file_idx=-1, total_num=total_json_num)
    print('*' * 75)
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def performance_collection2(folder_name):
    print('Loading tokenizer')
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    json_file_names = get_all_json_files(file_path=folder_name)
    json_file_names = [x for x in json_file_names if x != 'config.json']
    total_json_num = len(json_file_names)
    print('{} json files have been found'.format(len(json_file_names)))
    max_sp_sent_f1 = 0
    max_metric_res = None
    max_json_file_name = None
    for idx, json_file_name in enumerate(json_file_names):
        if json_file_name != 'config.json':
            data_frame = load_data_frame_align_with_dev(file_path=folder_name, json_fileName=json_file_name)
            col_names_set = set([col for col in data_frame.columns])
            # for col in data_frame.columns:
            #     print(col)
            if 'ss_ds_pair' in col_names_set:
                data_frame, res_names = evaluation2(data=data_frame, tokenizer=tokenizer)
            else:
                data_frame, res_names = evaluation(data=data_frame, tokenizer=tokenizer)
            # data_frame, res_names = evaluation2(data=data_frame, tokenizer=tokenizer)
            metric_res = performance_evaluation(data_frame=data_frame, res_names=res_names)
            print_metrics(metric_dict=metric_res, res_file_name=json_file_name, file_idx=idx + 1, total_num=total_json_num)
            if max_sp_sent_f1 < metric_res['sp_sent_f1']:
                max_sp_sent_f1 = metric_res['sp_sent_f1']
                max_metric_res = metric_res
                max_json_file_name = json_file_name
    print('*' * 75)
    print('-' * 35 + 'MAX METRIC' + '-'*30)
    print_metrics(metric_dict=max_metric_res, res_file_name=max_json_file_name, file_idx=-1, total_num=total_json_num)
    print('*' * 75)

def evaluation_consistent_checker(res_folder_name: str, json_res_file_name: str, orig_folder_name: str, orig_json_file_name: str):
    print('Loading tokenizer')
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    print('Loading prediction result...')
    res_data_frame = load_data_frame(file_path=res_folder_name, json_fileName=json_res_file_name)
    print('Loading original data...')
    orig_data_frame = load_data_frame(file_path=orig_folder_name, json_fileName=orig_json_file_name)

    assert res_data_frame.shape[0] == orig_data_frame.shape[0]

    def row_evaluation(row):
        answer_type_prediction, answer_type_true = row['aty_pred'], row['aty_true']
        span_prediction_start, span_prediction_end = row['sps_pred'], row['spe_pred']
        span_true_start, span_true_end = row['sps_true'], row['spe_true']
        encode_ids = row['encode_ids']

        if answer_type_prediction > 0:
            span_prediction = 'yes' if answer_type_prediction == 1 else 'no'
        else:
            span_prediction = tokenizer.decode(encode_ids[span_prediction_start:(span_prediction_end+1)], skip_special_tokens=True)

        if answer_type_true > 0:
            span_gold = 'yes' if answer_type_true == 1 else 'no'
        else:
            span_gold = tokenizer.decode(encode_ids[span_true_start:(span_true_end+1)], skip_special_tokens=True)
        return span_gold, span_prediction, answer_type_true, answer_type_prediction

    res_names = ['span_gold_res', 'span_pred_res', 'answer_type_gold_res', 'answer_type_pred_res']
    res_data_frame[res_names] = res_data_frame.swifter.apply(lambda row: pd.Series(row_evaluation(row)), axis=1)

    type_error_count = 0
    answer_error_count = 0
    span_answer_count = 0
    for row_idx, row in res_data_frame.iterrows():
        span_gold, span_pred, answer_type_gold, answer_type_pred = row['span_gold_res'], row['span_pred_res'], row['answer_type_gold_res'], row['answer_type_pred_res']
        orig_row = orig_data_frame.iloc[row_idx]
        orig_answer, orig_type = orig_row['answer'], orig_row['type']

        # print('{}:\t Predicted type: {}, gold type: {}, orig type: {}'.format(row_idx, answer_type_pred, answer_type_gold, orig_type))
        # print('{}:\t Predicted answer: {}, gold answer: {}, orig answer: {}'.format(row_idx, span_pred, span_gold, orig_answer))
        if answer_type_pred != answer_type_gold:
            # print('{}:\t Predicted answer: {}, gold answer: {}, orig answer: {}'.format(row_idx, span_pred, span_gold,
            #                                                                             orig_answer))
            type_error_count = type_error_count + 1
        if orig_answer not in ['yes', 'no']:
            span_answer_count = span_answer_count + 1

            if span_pred != span_gold:
                print('{}:\t Predicted answer: {}, gold answer: {}, orig answer: {}'.format(row_idx, span_pred, span_gold,
                                                                                            orig_answer))
                answer_error_count = answer_error_count + 1
    print('Type prediction error = {}'.format(type_error_count))
    print('Answer prediction error = {}'.format(answer_error_count))
    print(span_answer_count, answer_error_count, (span_answer_count - answer_error_count) *1.0/span_answer_count)

def prediction_res_consistent_checker_example():
    result_folder_name = '../model/'
    orig_data_folder_name = '../data/hotpotqa/'
    result_json_file_name = 'Nov_04_2020_03_22_07_44000_acc_0.9784.json'
    data_json_file_name = 'hotpot_dev_distractor_v1.json'
    evaluation_consistent_checker(res_folder_name=result_folder_name, json_res_file_name=result_json_file_name,
                                  orig_folder_name=orig_data_folder_name, orig_json_file_name=data_json_file_name)

###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
   # x = get_all_json_files('../data/hotpotqa/')
   result_folder_name = '../model/'
   performance_collection2(folder_name=result_folder_name)
   # prediction_res_consistent_checker_example()
   print()