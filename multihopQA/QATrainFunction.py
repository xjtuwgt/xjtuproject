import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopr.hotpotqaIOUtils import save_check_point, load_check_point
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import os
import pandas as pd
from time import time
import torch
import swifter
from torch import Tensor as T
from multihopQA.hotpot_evaluate_v1 import json_eval
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer
from multihopQA.hotpotQAdataloader import HotpotDataset, HotpotDevDataset
from multihopQA.longformerQAUtils import LongformerQATensorizer, LongformerEncoder
from multihopQA.UnifiedQAModel import LongformerHotPotQAModel
from datetime import date, datetime

def read_train_dev_data_frame(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    logging.info('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {} at step {}: {:.4f}'.format(mode, metric, step, metrics[metric]))

def get_date_time():
    today = date.today()
    str_today = today.strftime('%b_%d_%Y')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    date_time_str = str_today + '_' + current_time
    return date_time_str

def get_train_data_loader(args):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.train_data_name)
    shuffle = True
    batch_size = args.batch_size
    #####################################################
    training_data_shuffle = args.training_shuffle == 1
    #####################################################
    data_size = data_frame.shape[0]
    if args.train_data_filtered == 1:
        data_frame = data_frame[data_frame['level'] != 'easy']
        logging.info('Filtered data by removing easy case {} to {}'.format(data_size, data_frame.shape[0]))
    elif args.train_data_filtered == 2:
        data_frame = data_frame[data_frame['level'] == 'hard']
        logging.info(
            'Filtered data by removing easy and medium case {} to {}'.format(data_size, data_frame.shape[0]))
    else:
        logging.info('Using all training data {}'.format(data_size))

    data_size = data_frame.shape[0]
    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataloader = DataLoader(
        HotpotDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer, pad_neg_doc_num=args.pad_neg_samp_size,
                      max_sent_num=args.max_sent_num,
                      global_mask_type=args.global_mask_type, training_shuffle=training_data_shuffle),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDataset.collate_fn
    )
    return dataloader, data_size


def get_dev_data_loader(args):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.dev_data_name)
    batch_size = args.test_batch_size
    data_size = data_frame.shape[0]
    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataloader = DataLoader(
        HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer, max_sent_num=args.max_sent_num,
                      global_mask_type=args.global_mask_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDevDataset.collate_fn
    )
    return dataloader, data_size

def get_model(args):
    start_time = time()
    longEncoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                 hidden_dropout=args.input_drop, attn_dropout=args.attn_drop,
                                                 seq_project=args.seq_project)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.frozen_layer_num > 0:
        modules = [longEncoder.embeddings, *longEncoder.encoder.layer[:args.frozen_layer_num]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info('Frozen the first {} layers'.format(args.frozen_layer_num))
    logging.info('Loading encoder takes {:.4f}'.format(time() - start_time))
    model = LongformerHotPotQAModel(longformer=longEncoder, num_labels=args.num_labels, args=args)
    logging.info('Constructing model completes in {:.4f}'.format(time() - start_time))
    return model

def get_check_point(args):
    start_time = time()
    longEncoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                 hidden_dropout=args.input_drop, attn_dropout=args.attn_drop,
                                                 seq_project=args.seq_project)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.frozen_layer_num > 0:
        modules = [longEncoder.embeddings, *longEncoder.encoder.layer[:args.frozen_layer_num]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info('Frozen the first {} layers'.format(args.frozen_layer_num))
    logging.info('Loading encoder takes {:.4f}'.format(time() - start_time))
    model = LongformerHotPotQAModel(longformer=longEncoder, num_labels=args.num_labels, args=args)
    logging.info('Constructing model completes in {:.4f}'.format(time() - start_time))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model_path = args.save_path
    model_file_name = args.init_checkpoint
    hotpot_qa_model_name = os.path.join(model_path, model_file_name)
    model, optimizer, _, _, _ = load_check_point(model=model, optimizer=optimizer, PATH=hotpot_qa_model_name)
    return model, optimizer

def training_warm_up(model, optimizer, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warm_up_steps
    start_time = time()
    step = 0
    training_logs = []
    logging.info('Starting warm up...')
    logging.info('*' * 75)
    for batch_idx, train_sample in enumerate(train_dataloader):
        log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
        step = step + 1
        training_logs.append(log)
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('Training average', step, metrics)
            logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, 'warm_up', batch_idx + 1,
                                                                                     time() - start_time))
            training_logs = []
        if step >= warm_up_steps:
            logging.info('Warm up completed in {:.4f} seconds'.format(time() - start_time))
            logging.info('*' * 75)
            break
    logging.info('Evaluating on Valid Dataset...')
    metric_dict = model_evaluation(model=model, dev_data_loader=dev_dataloader, args=args)
    log_metrics('Valid', 'warm up', metric_dict['metrics'])
    logging.info('Answer type prediction accuracy: {}'.format(metric_dict['answer_type_acc']))
    logging.info('*' * 75)

def train_all_steps(model, optimizer, train_dataloader, dev_dataloader, args):
    assert args.save_checkpoint_steps % args.valid_steps == 0
    warm_up_steps = args.warm_up_steps
    if warm_up_steps > 0:
        training_warm_up(model=model, optimizer=optimizer, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, args=args)
        logging.info('*' * 75)
        current_learning_rate = optimizer.param_groups[-1]['lr']
        # learning_rate = args.learning_rate * 0.5
        learning_rate = current_learning_rate * 0.5
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        logging.info('Update learning rate from {} to {}'.format(current_learning_rate, learning_rate))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_steps, eta_min=1e-12)
    start_time = time()
    train_loss = 0.0
    eval_metric = None
    max_sent_pred_f1 = 0.0
    step = 0
    training_logs = []
    for epoch in range(1, args.epoch + 1):
        for batch_idx, train_sample in enumerate(train_dataloader):
            log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
            # ##+++++++++++++++++++++++++++++++++++++++++++++++
            scheduler.step()
            # ##+++++++++++++++++++++++++++++++++++++++++++++++
            step = step + 1
            training_logs.append(log)
            ##+++++++++++++++++++++++++++++++++++++++++++++++
            if step % args.save_checkpoint_steps == 0:
                save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss, eval_metric=eval_metric, args=args)
                logging.info('Saving the mode in {}'.format(save_path))
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch, batch_idx + 1, time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('*' * 75)
                logging.info('Evaluating on Valid Dataset...')
                metric_dict = model_evaluation(model=model, dev_data_loader=dev_dataloader, args=args)
                answer_type_acc = metric_dict['answer_type_acc']
                eval_metric = answer_type_acc
                sent_pred_f1 = metric_dict['metrics']['sp_f1']
                logging.info('*' * 75)
                log_metrics('Valid', step, metric_dict['metrics'])
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                dev_data_frame = metric_dict['res_dataframe']
                date_time_str = get_date_time()
                dev_result_name = os.path.join(args.save_path,
                                               date_time_str + '_' + str(step) + "_acc_" + answer_type_acc + '.json')
                dev_data_frame.to_json(dev_result_name, orient='records')
                logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                if max_sent_pred_f1 < sent_pred_f1:
                    max_sent_pred_f1 = sent_pred_f1
                    save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss,
                                                 eval_metric=max_sent_pred_f1, args=args)
                    logging.info('Saving the mode in {} with current best metric = {:.4f}'.format(save_path, max_sent_pred_f1))
    logging.info('Training completed...')

def train_single_step(model, optimizer, train_sample, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    if args.cuda:
        sample = dict()
        for key, value in train_sample.items():
            sample[key] = value.cuda()
    else:
        sample = train_sample
    loss_output = model(sample)
    yn_loss, span_loss, supp_doc_loss, supp_sent_loss = loss_output['yn_loss'], \
                                                        loss_output['span_loss'], \
                                                        loss_output['doc_loss'], loss_output['sent_loss']
    supp_doc_pair_loss = loss_output['doc_pair_loss']
    if args.do_retrieval:
        loss = supp_doc_loss + supp_sent_loss + supp_doc_pair_loss * args.pair_score_weight
    else:
        loss = supp_doc_loss + supp_sent_loss + span_loss * args.span_weight + yn_loss + supp_doc_pair_loss * args.pair_score_weight

    loss.sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    torch.cuda.empty_cache()
    log = {
        'al_loss': loss.sum().item(),
        'an_loss': span_loss.sum().item(),
        'sd_loss': supp_doc_loss.sum().item(),
        'pd_loss': supp_doc_pair_loss.sum().item(),
        'ss_loss': supp_sent_loss.sum().item(),
        'yn_loss': yn_loss.sum().item()
    }
    return log

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def model_evaluation(model, dev_data_loader, args):
    '''
            Evaluate the model on test or valid datasets
    '''
    start_time = time()
    step = 0
    N = 0
    total_steps = len(dev_data_loader)
    # **********************************************************
    answer_type_predicted = []
    answer_span_predicted = []
    supp_sent_predicted = []
    supp_doc_predicted = []
    answer_type_acc = 0.0
    ###########################################################
    model.eval()
    ###########################################################
    with torch.no_grad():
        for dev_sample in dev_data_loader:
            if args.cuda:
                sample = dict()
                for key, value in dev_sample.items():
                    sample[key] = value.cuda()
            else:
                sample = dev_sample
            output = model(sample)
            N = N + sample['ctx_encode'].shape[0]
            # ++++++++++++++++++
            answer_type_res = output['yn_score']
            if len(answer_type_res.shape) > 1:
                answer_type_res = answer_type_res.squeeze(dim=-1)
            answer_types = torch.argmax(answer_type_res, dim=-1)
            answer_type_true = sample['yes_no']
            if len(answer_type_true.shape) > 1:
                answer_type_true = answer_type_true.squeeze(dim=-1)
            correct_answer_type = (answer_types == answer_type_true).sum().data.item()
            answer_type_acc += correct_answer_type
            answer_type_predicted += answer_types.detach().tolist()
            # +++++++++++++++++++
            start_logits, end_logits = output['span_score']
            predicted_span_start = torch.argmax(start_logits, dim=-1)
            predicted_span_end = torch.argmax(end_logits, dim=-1)
            predicted_span_start = predicted_span_start.detach().tolist()
            predicted_span_end = predicted_span_end.detach().tolist()
            predicted_span_pair = list(zip(predicted_span_start, predicted_span_end))
            answer_span_predicted += predicted_span_pair
            # ++++++++++++++++++
            supp_doc_res = output['doc_score']
            doc_lens = sample['doc_lens']
            doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
            supp_doc_pred_i = supp_doc_prediction(scores=supp_doc_res, mask=doc_mask, pred_num=2)
            supp_doc_predicted += supp_doc_pred_i
            # ++++++++++++++++++
            supp_sent_res = output['sent_score']
            sent_lens = sample['sent_lens']
            sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
            sent_fact_doc_idx, sent_fact_sent_idx = sample['fact_doc'], sample['fact_sent']
            supp_sent_pred_i = supp_sent_prediction(scores=supp_sent_res, mask=sent_mask, doc_fact=sent_fact_doc_idx,
                                                    sent_fact=sent_fact_sent_idx, pred_num=2,
                                                    threshold=args.sent_threshold)
            supp_sent_predicted += supp_sent_pred_i
            # +++++++++++++++++++
            step = step + 1
            if step % args.test_log_steps == 0:
                logging.info('Evaluating the model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    logging.info('Evaluation is completed over {} examples in {:.4f} seconds'.format(N, time() - start_time))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    answer_type_acc = '{:.4f}'.format(answer_type_acc * 1.0/N)
    logging.info('Loading tokenizer')
    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    logging.info('Loading preprocessed data...')
    data = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.test_data_name)
    data['answer_prediction'] = answer_type_predicted
    data['answer_span_prediction'] = answer_span_predicted
    data['supp_doc_prediction'] = supp_doc_predicted
    data['supp_sent_prediction'] = supp_sent_predicted
    def row_process(row):
        answer_prediction = row['answer_prediction']
        answer_span_predicted = row['answer_span_prediction']
        span_start, span_end = answer_span_predicted
        encode_ids = row['ctx_encode']
        if answer_prediction > 0:
            predicted_answer = 'yes' if answer_prediction == 1 else 'no'
        else:
            predicted_answer = tokenizer.decode(encode_ids[span_start:(span_end + 1)], skip_special_tokens=True)

        ctx_contents = row['context']
        supp_doc_prediction = row['supp_doc_prediction']
        supp_doc_titles = [ctx_contents[idx][0] for idx in supp_doc_prediction]
        supp_sent_prediction = row['supp_sent_prediction']
        supp_sent_pairs = [(ctx_contents[pair_idx[0]][0], pair_idx[1]) for pair_idx in supp_sent_prediction]
        return predicted_answer, supp_doc_titles, supp_sent_pairs

    pred_names = ['answer', 'sp_doc', 'sp']
    data[pred_names] = data.swifter.progress_bar(True).apply(lambda row: pd.Series(row_process(row)), axis=1)
    res_names = ['_id', 'answer', 'sp_doc', 'sp']
    ###++++++++++++++++++++
    predicted_data_json = data[res_names].to_json()
    golden_data_json = read_train_dev_data_frame(file_path=args.orig_data_path, json_fileName=args.orig_dev_data_name)
    metrics = json_eval(prediction=predicted_data_json, gold=golden_data_json)
    ###++++++++++++++++++++
    res = {'metrics': metrics, 'answer_type_acc': answer_type_acc, 'res_dataframe': data}
    return res

def supp_doc_prediction(scores: T, mask: T, pred_num=2):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    supp_facts_predicted = []
    for idx in range(batch_size):
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        supp_facts_predicted.append(pred_labels_i)
    return supp_facts_predicted

def supp_sent_prediction(scores: T, mask: T, doc_fact: T, sent_fact: T, pred_num=2, threshold=0.9):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    supp_facts_predicted = []
    for idx in range(batch_size):
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        for i in range(pred_num, sample_size):
            if masked_scores[idx, pred_idxes_i[i]] >= threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
                pred_labels_i.append(pred_idxes_i[i])
        #################################################
        doc_fact_i = doc_fact[idx].detach().tolist()
        sent_fact_i = sent_fact[idx].detach().tolist()
        doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i))  ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
        #################################################
        doc_sent_idx_pair_i = []
        for pred_idx in pred_labels_i:
            doc_sent_idx_pair_i.append(doc_sent_pair_i[pred_idx])
        supp_facts_predicted.append(doc_sent_idx_pair_i)
    return supp_facts_predicted
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++