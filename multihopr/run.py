import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import argparse
import logging
import os
import pandas as pd
from time import time
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from multihopr.gpu_utils import gpu_setting, set_seeds
from multihopr.hotpotTrainFunction import train_all_steps, test_all_steps, log_metrics, get_data_loader, get_model
PTRE_TAINED_LONFORMER_BASE = 'allenai/longformer-base-4096'

###=====================Training data statistics=====================###
# id = 0 Split min	Query: 8	Pos_doc: 20	Neg_doc: 17	Pos_doc_sent: 1	Neg_doc_sent: 1
# id = 1 Split 95	Query: 55	Pos_doc: 224	Neg_doc: 299	Pos_doc_sent: 6	Neg_doc_sent: 9
# id = 2 Split 97.5	Query: 69	Pos_doc: 253	Neg_doc: 351	Pos_doc_sent: 7	Neg_doc_sent: 10
# id = 3 Split 99.5	Query: 95	Pos_doc: 322	Neg_doc: 509	Pos_doc_sent: 9	Neg_doc_sent: 15
# id = 4 Split 99.75	Query: 105	Pos_doc: 353	Neg_doc: 604	Pos_doc_sent: 10	Neg_doc_sent: 19
# id = 5 Split 99.9	Query: 116	Pos_doc: 403	Neg_doc: 795	Pos_doc_sent: 12	Neg_doc_sent: 23
# id = 6 Split 99.99	Query: 143	Pos_doc: 552	Neg_doc: 1545	Pos_doc_sent: 16	Neg_doc_sent: 42
# id = 7 Split max	Query: 157	Pos_doc: 919	Neg_doc: 2095	Pos_doc_sent: 26	Neg_doc_sent: 85
###=====================Training data statistics=====================###
MAX_QUERY_PAD_LEN = 116
MAX_DOCUMENT_PAD_LEN = 403
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Retrieval Models',
        usage='train.py [<args>] [-h | --help]')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--do_debug', default=False, action='store_true', help='whether')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/fullwiki_qa')
    parser.add_argument('--train_data_name', type=str, default='hotpot_train_full_wiki_tokenized.json')
    parser.add_argument('--train_data_filtered', type=int, default=1)
    parser.add_argument('--dev_data_name', type=str, default='hotpot_dev_full_wiki_tokenized.json')
    parser.add_argument('--model', default='TwinTowerRetriever', type=str)
    parser.add_argument('--tri_mode', default='head-batch', type=str)
    parser.add_argument('--pretrained_cfg_name', default=PTRE_TAINED_LONFORMER_BASE, type=str)
    num_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parser.add_argument('--gpu_num', default=num_gpu, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int, help='valid/test batch size')
    parser.add_argument('--gamma', default=2.0, type=float, help='parameter for focal loss')
    parser.add_argument('--alpha', default=1.0, type=float, help='parameter for focal loss')
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--model_name', default='BiLinear', type=str) #'DotProduct'
    parser.add_argument('--hop_model_name', default='DistMult', type=str)
    parser.add_argument('--query_frozen_layer_num', default=10, type=int, help='number of layers for query encoder frozen during training')
    parser.add_argument('--doc_frozen_layer_num', default=10, type=int, help='number of layers for document encoder frozen during training')
    parser.add_argument('--neg_samp_size', default=8, type=int)
    parser.add_argument('--project_dim', default=128, type=int)
    parser.add_argument('--margin', default=0.0000001, type=float)
    parser.add_argument('--input_drop', default=0.1, type=float)
    parser.add_argument('--attn_drop', default=0.25, type=float)
    parser.add_argument('--max_query_len', default=MAX_QUERY_PAD_LEN, type=int)
    parser.add_argument('--max_doc_len', default=MAX_DOCUMENT_PAD_LEN, type=int)
    parser.add_argument('--weight_decay', default=01E-9, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ##+++++++++++++++++++++++++Parameter for multi-head co-attention++++++++++++++++++++++++++++++++++
    parser.add_argument('--do_co_attn', default=False, action='store_true', help='whether perform later interaction')
    parser.add_argument('--heads', default=8, type=int, help='number of heads for co-attentions')
    parser.add_argument('--layers', default=2, type=int, help='number layers')
    parser.add_argument('--seq_project', default=True, action='store_true', help='whether perform sequence projection')
    parser.add_argument('--triple_score', default=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--grad_clip_value', default=10.0, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=12, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='../model', type=str)
    parser.add_argument('--max_steps', default=80000, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=50, type=int, help='valid/test log every xx steps')
    parser.add_argument('--rand_seed', default=1234, type=int, help='random seed')

    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    set_seeds(args.rand_seed)
    start_time = time()
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)
    logging.info('Model: {}'.format(args.model))
    logging.info('Data Path: {}'.format(args.data_path))
    if args.cuda:
        if args.gpu_num > 1:
            device_ids, used_memory = gpu_setting(args.gpu_num)
        else:
            device_ids, used_memory = gpu_setting()
        if used_memory > 100:
            logging.info('Using memory = {}'.format(used_memory))
        if device_ids is not None:
            device = torch.device('cuda:{}'.format(device_ids[0]))
        else:
            device = torch.device('cuda:0')
        logging.info('Set the cuda with idxes = {}'.format(device_ids))
        logging.info('Cuda setting {}'.format(device))
    else:
        device = torch.device('cpu')
        device_ids = None
        logging.info('CPU setting')

    logging.info('Loading training data...')
    train_data_loader = get_data_loader(args=args, train=True)
    logging.info('Loading training data completed in {:.4f} seconds'.format(time() - start_time))
    dev_data_loader = get_data_loader(args=args, train=False)
    logging.info('Loading dev data completed in {:.4f} seconds'.format(time() - start_time))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Loading data completed')
    logging.info('*'*120)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.do_train:
        # Set training configuration
        start_time = time()
        logging.info('Loading retrieval model...')
        model = get_model(args=args).to(device)
        logging.info('Loading retrieval model completed in {} seconds'.format(time() - start_time))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_steps, eta_min=1e-10)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if device_ids is not None:
            model = DataParallel(model, device_ids=device_ids)
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
        logging.info('*' * 150)
        logging.info("Model hype-parameter information...")
        for key, value in vars(args).items():
            logging.info('Hype-parameter\t{} = {}'.format(key, value))
        logging.info('*' * 150)
        logging.info('Start Training...')
        logging.info('batch_size = {}'.format(args.batch_size))
        logging.info('projection_dim = {}'.format(args.project_dim))
        logging.info('learning_rate = {}'.format(args.learning_rate))
        train_all_steps(model=model, scheduler=scheduler, optimizer=optimizer, train_data_loader=train_data_loader,
                        dev_data_loader=dev_data_loader, args=args)
        logging.info('Completed training in {:.4f} seconds'.format(time() - start_time))
        logging.info('Evaluating on Valid Dataset...')
        metrics = test_all_steps(model=model, test_data_loader=dev_data_loader, args=args)
        log_metrics('Valid', 'all step', metrics)


if __name__ == '__main__':
    main(parse_args())