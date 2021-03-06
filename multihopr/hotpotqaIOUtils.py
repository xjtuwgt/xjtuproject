import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import pandas as pd
from pandas import DataFrame
import torch
import json
from torch.optim import Adam
from multihopr.twintowerRetriver import TwinTowerRetriver
from time import time

hotpot_path = '../data/hotpotqa/'
print(os.path.abspath(hotpot_path))
hotpot_train_data = 'hotpot_train_v1.1.json'  # _id;answer;question;supporting_facts;context;type;level
hotpot_dev_fullwiki = 'hotpot_dev_fullwiki_v1.json'  # _id;answer;question;supporting_facts;context;type;level
hotpot_test_fullwiki = 'hotpot_test_fullwiki_v1.json'  # _id; question; context
hotpot_dev_distractor = 'hotpot_dev_distractor_v1.json'  # _id;answer;question;supporting_facts;context;type;level

def loadWikiData(PATH, json_fileName)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(os.path.join(PATH, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def HOTPOT_TrainData(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_train_data)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_DevData_FullWiki(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_dev_fullwiki)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_DevData_Distractor(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_dev_distractor)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_Test_FullWiki(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_test_fullwiki)
    column_names = [col for col in data.columns]
    return data, column_names

def save_data_frame_to_json(df: DataFrame, file_name: str):
    df.to_json(file_name, orient='records')
    print('Save {} data in json file'.format(df.shape))

###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_check_point(model, optimizer: Adam, loss, eval_metric, step, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    save_path = os.path.join(args.save_path, str(step) + '_' + str(loss) + '_'+ str(eval_metric) + '.pt')
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module
    torch.save({
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'eval': eval_metric
    }, save_path)
    return save_path

# def save_check_point(model, optimizer: Adam, save_variable_list, args):
#     '''
#     Save the parameters of the model and the optimizer,
#     as well as some other variables such as step and learning_rate
#     '''
#     argparse_dict = vars(args)
#     with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
#         json.dump(argparse_dict, fjson)
#
#     torch.save({
#         **save_variable_list,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()},
#         os.path.join(args.save_path, 'checkpoint')
#     )

def load_check_point(model, optimizer: Adam, PATH: str):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        checkpoint = torch.load(PATH, device)
    else:
        checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    eval_metric = checkpoint['eval']
    return model, optimizer, step, loss, eval_metric

def load_model(model, PATH: str):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        checkpoint = torch.load(PATH, device)
    else:
        checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    eval_metric = checkpoint['eval']
    return model, step, loss, eval_metric

