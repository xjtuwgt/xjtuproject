import logging
import os
from multihopr.hotpotqaIOUtils import save_check_point
from time import time
import pandas as pd
from multihopr.dataLoader import HotpotTrainDataset, HotpotDevDataset
from multihopr.longformerUtils import LongformerTensorizer, LongformerEncoder
from multihopr.twintowerRetriver import TwinTowerRetriver
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer
import torch

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {} at step {}: {:.5f}'.format(mode, metric, step, metrics[metric]))

def read_train_dev_data_frame(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    logging.info('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def get_data_loader(args, train=True):
    if train:
        if args.gpu_num > 0:
            data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.train_data_name)
        else:
            data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.dev_data_name)
        shuffle = True
        batch_size = args.batch_size
    else:
        data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.dev_data_name)
        batch_size = args.test_batch_size
        shuffle = False
    data_size = data_frame.shape[0]
    if args.train_data_filtered == 1:
        data_frame = data_frame[data_frame['level'] != 'easy']
        logging.info('Filtered data by removing easy case {} to {}'.format(data_size, data_frame.shape[0]))
    elif args.train_data_filtered == 2:
        data_frame = data_frame[data_frame['level'] == 'hard']
        logging.info('Filtered data by removing easy and medium case {} to {}'.format(data_size, data_frame.shape[0]))
    else:
        logging.info('Using all training data {}'.format(data_size))

    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    query_tensorizer = LongformerTensorizer(tokenizer=tokenizer, max_length=args.max_query_len)  ## for padding
    document_tensorizer = LongformerTensorizer(tokenizer=tokenizer, max_length=args.max_doc_len)  ## for padding
    if train:
        dataloader = DataLoader(
            HotpotTrainDataset(train_data_frame=data_frame, query_tensorizer=query_tensorizer,
                               doc_tensorizer=document_tensorizer,
                               negative_sample_size=args.neg_samp_size, mode=args.tri_mode),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=HotpotTrainDataset.collate_fn)
    else:
        dataloader = DataLoader(
            HotpotDevDataset(dev_data_frame=data_frame, query_tensorizer=query_tensorizer,
                             doc_tensorizer=document_tensorizer),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=HotpotDevDataset.collate_fn
        )
    return dataloader

def frozon_layer(model: LongformerEncoder, frozen_layer_num: int):
    if frozen_layer_num > 0:
        modules = [model.embeddings, *model.encoder.layer[:frozen_layer_num]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info('Frozen the first {} layers'.format(frozen_layer_num))

def get_model(args):
    start_time = time()
    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    query_encoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                   seq_project=args.seq_project)
    query_encoder.resize_token_embeddings(len(tokenizer))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    frozon_layer(model=query_encoder, frozen_layer_num=args.query_frozen_layer_num)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Loading query encoder takes {:.4f}'.format(time() - start_time))
    start_time = time()
    logging.info('Loading document encoder...')
    document_encoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                      seq_project=args.seq_project)
    document_encoder.resize_token_embeddings(len(tokenizer))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    frozon_layer(model=document_encoder, frozen_layer_num=args.doc_frozen_layer_num)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Loading document encoder takes {:.4f}'.format(time() - start_time))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    model = TwinTowerRetriver(model_name=args.model_name, hop_model_name=args.hop_model_name,
                              query_encoder=query_encoder, document_encoder=document_encoder, args=args)
    logging.info('Constructing model completes in {:.4f}'.format(time() - start_time))
    return model

def train_all_steps(model, optimizer, scheduler, train_data_loader, dev_data_loader, args):
    assert args.save_checkpoint_steps % args.valid_steps == 0
    start_time = time()
    train_loss = 0.0
    eval_metric = 0.0
    step = 0
    training_logs = []
    for epoch in range(1, args.epochs + 1):
        for batch_idx, train_sample in enumerate(train_data_loader):
            log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
            training_logs.append(log)
            scheduler.step()
            step = step + 1
            if step % args.save_checkpoint_steps == 0:
                save_variable_dict = {
                    'step': step,
                    'loss': train_loss,
                    'metric': eval_metric
                }
                # save_check_point(model=model, optimizer=optimizer, save_variable_list=save_variable_dict, args=args)
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} steps takes {:.4f} seconds'.format(step, time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = test_all_steps(model=model, test_data_loader=dev_data_loader, args=args)
                eval_metric = metrics['HITS@4']
                log_metrics('Valid', step, metrics)

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
            if key == 'mode':
                sample[key] = value
            else:
                sample[key] = value.cuda()
    else:
        sample = train_sample
    pair_loss, ht_loss, triple_loss = model(sample)
    # print(pair_loss, ht_loss, triple_loss)
    if args.margin > 0:
        loss = pair_loss + ht_loss + triple_loss
    else:
        loss = pair_loss + triple_loss
    # print(loss, pair_loss, ht_loss, triple_loss)
    loss.sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    # print(pair_scores.shape, triple_scores.shape)
    log = {
        'al_loss': loss.sum().item(),
        'pi_loss': pair_loss.sum().item(),
        'ht_loss': ht_loss.sum().item() if args.margin > 0 is not None else 0.0,
        'tr_loss': triple_loss.sum().item()
    }
    return log

def test_all_steps(model, test_data_loader, args):
    '''
            Evaluate the model on test or valid datasets
            '''
    start_time = time()
    model.eval()
    test_dataset = test_data_loader
    logs = []
    step = 0
    total_steps = len(test_dataset)
    # print(total_steps)
    with torch.no_grad():
        for test_sample in test_dataset:
            if args.cuda:
                sample = dict()
                for key, value in test_sample.items():
                    if key == 'mode':
                        sample[key] = value
                    else:
                        sample[key] = value.cuda()
            else:
                sample = test_sample

            pair_scores = model(sample)
            # print(pair_scores, '\n', sample['class'])
            argsort = torch.argsort(pair_scores, dim=1, descending=True)
            batch_size = pair_scores.shape[0]
            for i in range(batch_size):
                ranking_0 = torch.nonzero((argsort[i, :] == 0))
                ranking_1 = torch.nonzero((argsort[i, :] == 1))
                assert ranking_0.size(0) == 1
                assert ranking_1.size(0) == 1
                ranking_0 = ranking_0.item() + 1
                ranking_1 = ranking_1.item() + 1

                hit1 = 1.0 if ranking_0 <= 2 else 0.0
                hit2 = 1.0 if ranking_0 <= 2 and ranking_1 <= 2 else 0.0
                hit3 = 1.0 if ranking_0 <= 3 and ranking_1 <= 3 else 0.0
                hit4 = 1.0 if ranking_0 <= 4 and ranking_1 <= 4 else 0.0
                hit5 = 1.0 if ranking_0 <= 5 and ranking_1 <= 5 else 0.0
                hit6 = 1.0 if ranking_0 <= 6 and ranking_1 <= 6 else 0.0
                hit7 = 1.0 if ranking_0 <= 7 and ranking_1 <= 7 else 0.0
                hit8 = 1.0 if ranking_0 <= 8 and ranking_1 <= 8 else 0.0
                logs.append({
                    'HITS@1': hit1,
                    'HITS@2': hit2,
                    'HITS@3': hit3,
                    'HITS@4': hit4,
                    'HITS@5': hit5,
                    'HITS@6': hit6,
                    'HITS@7': hit7,
                    'HITS@8': hit8
                })
            step += 1
            if step % args.test_log_steps == 0:
                logging.info('Evaluating the model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics