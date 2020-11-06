from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from time import time
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
from torch.utils.data import Dataset
from multihopr.longformerUtils import LongformerTensorizer
from multihopr.longformerUtils import PRE_TAINED_LONFORMER_BASE

class HotpotDevDataset(Dataset):
    def __init__(self, dev_data_frame: DataFrame, query_tensorizer: LongformerTensorizer,
                 doc_tensorizer: LongformerTensorizer, max_doc_num=10, max_sent_num=85, sent_global_mask=False):
        self.len = dev_data_frame.shape[0]
        self.dev_data = dev_data_frame
        self.max_doc_num = max_doc_num
        self.max_query_len = query_tensorizer.max_length
        self.max_doc_len = doc_tensorizer.max_length
        self.query_tensorizer = query_tensorizer
        self.doc_tensorizer = doc_tensorizer
        self.max_sent_num = max_sent_num
        self.sent_global_mask = sent_global_mask

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.dev_data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        pos_ctx_encode, pos_ctx_lens, neg_ctx_encode, neg_ctx_lens = example['p_ctx_encode'], example['p_ctx_lens'], \
                                                                     example['n_ctx_encode'], example['n_ctx_lens']
        pos_max_ctx_len, neg_max_ctx_len = example['pc_max_len'], example['nc_max_len']
        ctx_len = pos_max_ctx_len if pos_max_ctx_len > neg_max_ctx_len else neg_max_ctx_len
        if ctx_len > self.max_doc_len:
            ctx_len = self.max_doc_len
        if query_len > self.max_query_len:
            query_len = self.max_query_len
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_type = example['answer_type']
        order_mask = torch.tensor([0], dtype=torch.bool) if answer_type else torch.tensor([1], dtype=torch.bool)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_id_tensor = self.query_tensorizer.token_ids_to_tensor(token_ids=query_encode)
        query_attn_mask = self.query_tensorizer.get_attn_mask(token_ids_tensor=query_id_tensor)
        query_global_attn_mask = self.query_tensorizer.get_global_attn_mask(tokens_ids_tensor=query_id_tensor)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_encode_tensor_list = []
        ctx_encode_att_mask_list = []
        ctx_encode_global_att_mask_list = []
        ctx_sent_num_list = [] ## record the number sentences in each document++++
        ctx_sent_position_list = [] ## record the number sentences in each document++++
        pos_id_list = []
        for doc_idx, doc_tup in enumerate(pos_ctx_encode):
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, _, title_len = doc_tup
            assert len(doc_encode_ids) == pos_ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i
            doc_encode_id_tensor = self.doc_tensorizer.token_ids_to_tensor(token_ids=doc_encode_ids)
            doc_encode_att_mask = self.doc_tensorizer.get_attn_mask(token_ids_tensor=doc_encode_id_tensor)
            ##+++++++++++++++++++++++++
            if self.sent_global_mask:
                global_mask_idxs = [0, 1] + [x[0] for x in sent_start_end_pair]
            else:
                global_mask_idxs = None
            doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(tokens_ids_tensor=doc_encode_id_tensor, gobal_mask_idxs=global_mask_idxs)
            doc_sent_num = len(sent_start_end_pair)
            sent_positions = torch.zeros(2*self.max_sent_num, dtype=torch.long)
            for s_idx, position_pair in enumerate(sent_start_end_pair):
                sent_positions[2*s_idx] = position_pair[0]
                sent_positions[2*s_idx + 1] = position_pair[1]
            ##+++++++++++++++++++++++++
            pos_id_list.append((doc_encode_id_tensor, doc_encode_att_mask, doc_encode_global_att_mask, doc_weight, doc_sent_num, sent_positions))
        assert len(pos_id_list) == 2
        if pos_id_list[0][3] > pos_id_list[1][3]:
            pos_ids, pos_att_mask, pos_global_att_mask, pos_doc_sent_nums, pos_sent_position = [pos_id_list[0][0], pos_id_list[1][0]], \
                                                         [pos_id_list[0][1], pos_id_list[1][1]], \
                                                         [pos_id_list[0][2], pos_id_list[1][2]], [pos_id_list[0][4], pos_id_list[1][4]], [pos_id_list[0][5], pos_id_list[1][5]]
        else:
            pos_ids, pos_att_mask, pos_global_att_mask, pos_doc_sent_nums, pos_sent_position = [pos_id_list[1][0], pos_id_list[0][0]], \
                                                         [pos_id_list[1][1], pos_id_list[0][1]], \
                                                         [pos_id_list[1][2], pos_id_list[0][2]], [pos_id_list[1][4], pos_id_list[0][4]], [pos_id_list[1][5], pos_id_list[0][5]]
        ctx_encode_tensor_list = ctx_encode_tensor_list + pos_ids
        ctx_encode_att_mask_list = ctx_encode_att_mask_list + pos_att_mask
        ctx_encode_global_att_mask_list = ctx_encode_global_att_mask_list + pos_global_att_mask
        ctx_sent_num_list = ctx_sent_num_list + pos_doc_sent_nums
        ctx_sent_position_list = ctx_sent_position_list + pos_sent_position
        pos_doc_num = len(ctx_encode_tensor_list)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for doc_idx, doc_tup in enumerate(neg_ctx_encode):
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, _, title_len = doc_tup
            assert len(doc_encode_ids) == neg_ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i
            doc_encode_id_tensor = self.doc_tensorizer.token_ids_to_tensor(token_ids=doc_encode_ids)
            doc_encode_att_mask = self.doc_tensorizer.get_attn_mask(token_ids_tensor=doc_encode_id_tensor)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            doc_sent_num = len(sent_start_end_pair)
            if self.sent_global_mask:
                global_mask_idxs = [0, 1] + [x[0] for x in sent_start_end_pair]
            else:
                global_mask_idxs = None
            doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(tokens_ids_tensor=doc_encode_id_tensor, gobal_mask_idxs=global_mask_idxs)
            sent_positions = torch.zeros(2*self.max_sent_num, dtype=torch.long)
            for s_idx, position_pair in enumerate(sent_start_end_pair):
                sent_positions[2*s_idx] = position_pair[0]
                sent_positions[2*s_idx + 1] = position_pair[1]
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(tokens_ids_tensor=doc_encode_id_tensor)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_encode_tensor_list.append(doc_encode_id_tensor)
            ctx_encode_att_mask_list.append(doc_encode_att_mask)
            ctx_encode_global_att_mask_list.append(doc_encode_global_att_mask)
            ctx_sent_num_list.append(doc_sent_num)
            ctx_sent_position_list.append(sent_positions)
        ctx_doc_num = len(ctx_encode_tensor_list)
        if ctx_doc_num < self.max_doc_num: #append the last negative documents to max doc number
            ctx_encode_tensor_list = ctx_encode_tensor_list + [ctx_encode_tensor_list[-1]] * (self.max_doc_num - ctx_doc_num)
            ctx_encode_att_mask_list = ctx_encode_att_mask_list + [ctx_encode_att_mask_list[-1]] * (self.max_doc_num - ctx_doc_num)
            ctx_encode_global_att_mask_list = ctx_encode_global_att_mask_list + [ctx_encode_global_att_mask_list[-1]] * (self.max_doc_num - ctx_doc_num)
            ctx_sent_num_list = ctx_sent_num_list + [ctx_sent_num_list[-1]] * (self.max_doc_num - ctx_doc_num)
            ctx_sent_position_list = ctx_sent_position_list + [ctx_sent_position_list[-1]] * (self.max_doc_num - ctx_doc_num)

        assert self.max_doc_num == len(ctx_encode_tensor_list) and self.max_doc_num == len(ctx_encode_att_mask_list) \
               and self.max_doc_num == len(ctx_encode_global_att_mask_list) and self.max_doc_num == len(ctx_sent_num_list) \
               and self.max_doc_num == len(ctx_sent_position_list)

        class_label = torch.LongTensor([1] * pos_doc_num + [0] * (self.max_doc_num - pos_doc_num))
        ctx_id_tensor = torch.stack(ctx_encode_tensor_list, dim=0)
        ctx_attn_mask = torch.stack(ctx_encode_att_mask_list, dim=0)
        ctx_global_attn_mask = torch.stack(ctx_encode_global_att_mask_list, dim=0)
        ctx_sent_nums = torch.LongTensor(ctx_sent_num_list)
        ctx_max_sent_num = max(ctx_sent_num_list)
        ctx_sent_position = torch.stack(ctx_sent_position_list, dim=0)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return query_id_tensor, query_attn_mask, query_global_attn_mask, ctx_id_tensor, ctx_attn_mask, \
               ctx_global_attn_mask, class_label, order_mask, ctx_sent_nums, ctx_sent_position, query_len, ctx_len, ctx_max_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_query_len = max([_[10] for _ in data])
        batch_max_doc_len = max([_[11] for _ in data])
        batch_max_sent_num = max([_[12] for _ in data])
        # print(batch_max_doc_len, batch_max_query_len)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_query_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_query_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_query_global_sample = torch.stack([_[2] for _ in data], dim=0)
        # print('query', batch_query_sample.shape, batch_query_mask_sample.shape, batch_query_global_sample.shape)

        batch_query_sample = batch_query_sample[:, range(0,batch_max_query_len)]
        batch_query_mask_sample = batch_query_mask_sample[:, range(0,batch_max_query_len)]
        batch_query_global_sample = batch_query_global_sample[:, range(0,batch_max_query_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_sample = torch.stack([_[3] for _ in data], dim=0)
        batch_doc_mask_sample = torch.stack([_[4] for _ in data], dim=0)
        batch_doc_global_mask_sample = torch.stack([_[5] for _ in data], dim=0)
        # print('doc', batch_doc_sample.shape, batch_doc_mask_sample.shape, batch_doc_global_mask_sample.shape)

        batch_doc_sample = batch_doc_sample[:,:,range(0,batch_max_doc_len)]
        batch_doc_mask_sample = batch_doc_mask_sample[:,:,range(0,batch_max_doc_len)]
        batch_doc_global_mask_sample = batch_doc_global_mask_sample[:,:,range(0,batch_max_doc_len)]
        # print('filtered doc', batch_doc_sample.shape, batch_doc_mask_sample.shape, batch_doc_global_mask_sample.shape)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_class = torch.stack([_[6] for _ in data], dim=0)
        batch_doc_order_mask = torch.stack([_[7] for _ in data], dim=0)
        batch_doc_sent_num = torch.stack([_[8] for _ in data], dim=0)
        batch_doc_sent_positions = torch.stack([_[9] for _ in data], dim=0)
        batch_doc_sent_positions = batch_doc_sent_positions[:, :, range(0, 2 * batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'query': batch_query_sample, 'query_attn_mask': batch_query_mask_sample, 'query_global_mask': batch_query_global_sample,
               'ctx_doc': batch_doc_sample, 'ctx_attn_mask': batch_doc_mask_sample, 'ctx_global_mask': batch_doc_global_mask_sample,
               'class': batch_class, 'order_mask': batch_doc_order_mask, 'sent_num': batch_doc_sent_num, 'sent_position': batch_doc_sent_positions}
        return res
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class HotpotTrainDataset(Dataset):
    def __init__(self, train_data_frame: DataFrame, query_tensorizer: LongformerTensorizer,
                 doc_tensorizer: LongformerTensorizer, negative_sample_size: int,  mode: str, max_sent_num: int = 85, sent_global_mask=False):
        self.len = train_data_frame.shape[0]
        self.train_data = train_data_frame
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.max_query_len = query_tensorizer.max_length
        self.max_doc_len = doc_tensorizer.max_length
        self.query_tensorizer = query_tensorizer
        self.doc_tensorizer = doc_tensorizer
        self.max_sent_num = max_sent_num
        self.sent_global_mask = sent_global_mask

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.train_data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        pos_ctx_encode, pos_ctx_lens, neg_ctx_encode, neg_ctx_lens = example['p_ctx_encode'], example['p_ctx_lens'], \
                                                                     example['n_ctx_encode'], example['n_ctx_lens']
        pos_max_ctx_len, neg_max_ctx_len = example['pc_max_len'], example['nc_max_len']
        ctx_len = pos_max_ctx_len if pos_max_ctx_len > neg_max_ctx_len else neg_max_ctx_len
        if ctx_len > self.max_doc_len:
            ctx_len = self.max_doc_len
        if query_len > self.max_query_len:
            query_len = self.max_query_len
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_type = example['answer_type']
        order_mask = torch.tensor([0], dtype=torch.bool) if answer_type else torch.tensor([1], dtype=torch.bool)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_id_tensor = self.query_tensorizer.token_ids_to_tensor(token_ids=query_encode)
        query_attn_mask = self.query_tensorizer.get_attn_mask(token_ids_tensor=query_id_tensor)
        query_global_attn_mask = self.query_tensorizer.get_global_attn_mask(tokens_ids_tensor=query_id_tensor)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_encode_tensor_list = []
        ctx_encode_att_mask_list = []
        ctx_encode_global_att_mask_list = []
        ctx_sent_num_list = [] ## record the number sentences in each document++++
        ctx_sent_position_list = [] ## record the number sentences in each document++++
        pos_id_list = []
        for doc_idx, doc_tup in enumerate(pos_ctx_encode):
            # doc_encode_ids, doc_weight, doc_len_i = doc_tup
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, _, _ = doc_tup
            assert len(doc_encode_ids) == pos_ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i
            doc_encode_id_tensor = self.doc_tensorizer.token_ids_to_tensor(token_ids=doc_encode_ids)
            doc_encode_att_mask = self.doc_tensorizer.get_attn_mask(token_ids_tensor=doc_encode_id_tensor)
            ##+++++++++++++++++++++++++
            if self.sent_global_mask:
                global_mask_idxs = [0, 1] + [_[0] for _ in sent_start_end_pair]
            else:
                global_mask_idxs = None
            doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(
                tokens_ids_tensor=doc_encode_id_tensor, gobal_mask_idxs=global_mask_idxs)
            doc_sent_num = len(sent_start_end_pair)
            sent_positions = torch.zeros(2 * self.max_sent_num, dtype=torch.long)
            for s_idx, position_pair in enumerate(sent_start_end_pair):
                sent_positions[2 * s_idx] = position_pair[0]
                sent_positions[2 * s_idx + 1] = position_pair[1]
            ##+++++++++++++++++++++++++
            # doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(tokens_ids_tensor=doc_encode_id_tensor)
            pos_id_list.append((doc_encode_id_tensor, doc_encode_att_mask, doc_encode_global_att_mask, doc_weight, doc_sent_num, sent_positions))
        assert len(pos_id_list) == 2
        if pos_id_list[0][3] > pos_id_list[1][3]:
            pos_ids, pos_att_mask, pos_global_att_mask, pos_doc_sent_nums, pos_sent_position = [pos_id_list[0][0], pos_id_list[1][0]], \
                                                         [pos_id_list[0][1], pos_id_list[1][1]], \
                                                         [pos_id_list[0][2], pos_id_list[1][2]], [pos_id_list[0][4], pos_id_list[1][4]], [pos_id_list[0][5], pos_id_list[1][5]]
        else:
            pos_ids, pos_att_mask, pos_global_att_mask, pos_doc_sent_nums, pos_sent_position = [pos_id_list[1][0], pos_id_list[0][0]], \
                                                         [pos_id_list[1][1], pos_id_list[0][1]], \
                                                         [pos_id_list[1][2], pos_id_list[0][2]], [pos_id_list[1][4], pos_id_list[0][4]], [pos_id_list[1][5], pos_id_list[0][5]]
        ctx_encode_tensor_list = ctx_encode_tensor_list + pos_ids
        ctx_encode_att_mask_list = ctx_encode_att_mask_list + pos_att_mask
        ctx_encode_global_att_mask_list = ctx_encode_global_att_mask_list + pos_global_att_mask
        ctx_sent_num_list = ctx_sent_num_list + pos_doc_sent_nums
        ctx_sent_position_list = ctx_sent_position_list + pos_sent_position

        pos_doc_num = len(ctx_encode_tensor_list)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        total_neg_doc_hot_pot = len(neg_ctx_encode)
        if self.negative_sample_size < total_neg_doc_hot_pot:
            neg_sample_idxs = np.random.choice(total_neg_doc_hot_pot, self.negative_sample_size, replace=False)
        else:
            neg_sample_idxs = np.random.choice(total_neg_doc_hot_pot, self.negative_sample_size, replace=True)

        for neg_idx in neg_sample_idxs:
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, _, _ = neg_ctx_encode[neg_idx]
            # doc_encode_ids, doc_weight, doc_len_i = neg_ctx_encode[neg_idx]
            assert len(doc_encode_ids) == doc_len_i
            doc_encode_id_tensor = self.doc_tensorizer.token_ids_to_tensor(token_ids=doc_encode_ids)
            doc_encode_att_mask = self.doc_tensorizer.get_attn_mask(token_ids_tensor=doc_encode_id_tensor)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.sent_global_mask:
                global_mask_idxs = [0, 1] + [x[0] for x in sent_start_end_pair]
            else:
                global_mask_idxs = None
            doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(
                tokens_ids_tensor=doc_encode_id_tensor, gobal_mask_idxs=global_mask_idxs)
            doc_sent_num = len(sent_start_end_pair)
            sent_positions = torch.zeros(2 * self.max_sent_num, dtype=torch.long)
            for s_idx, position_pair in enumerate(sent_start_end_pair):
                sent_positions[2 * s_idx] = position_pair[0]
                sent_positions[2 * s_idx + 1] = position_pair[1]
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # doc_encode_global_att_mask = self.doc_tensorizer.get_global_attn_mask(tokens_ids_tensor=doc_encode_id_tensor)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_encode_tensor_list.append(doc_encode_id_tensor)
            ctx_encode_att_mask_list.append(doc_encode_att_mask)
            ctx_encode_global_att_mask_list.append(doc_encode_global_att_mask)
            ctx_sent_num_list.append(doc_sent_num)
            ctx_sent_position_list.append(sent_positions)
        assert len(ctx_encode_tensor_list) == (pos_doc_num + self.negative_sample_size)

        class_label = torch.LongTensor([1] * pos_doc_num + [0] * self.negative_sample_size)
        ctx_id_tensor = torch.stack(ctx_encode_tensor_list, dim=0)
        ctx_attn_mask = torch.stack(ctx_encode_att_mask_list, dim=0)
        ctx_global_attn_mask = torch.stack(ctx_encode_global_att_mask_list, dim=0)
        ctx_sent_nums = torch.LongTensor(ctx_sent_num_list)
        ctx_max_sent_num = max(ctx_sent_num_list)
        ctx_sent_position = torch.stack(ctx_sent_position_list, dim=0)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return query_id_tensor, query_attn_mask, query_global_attn_mask, ctx_id_tensor, ctx_attn_mask, ctx_global_attn_mask, \
               class_label, order_mask, ctx_sent_nums, ctx_sent_position, self.mode, query_len, ctx_len, ctx_max_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_query_len = max([_[11] for _ in data])
        batch_max_doc_len = max([_[12] for _ in data])
        batch_max_sent_num = max([_[13] for _ in data])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_query_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_query_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_query_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_query_sample = batch_query_sample[:, range(0,batch_max_query_len)]
        batch_query_mask_sample = batch_query_mask_sample[:, range(0,batch_max_query_len)]
        batch_query_global_sample = batch_query_global_sample[:, range(0,batch_max_query_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_sample = torch.stack([_[3] for _ in data], dim=0)
        batch_doc_mask_sample = torch.stack([_[4] for _ in data], dim=0)
        batch_doc_global_mask_sample = torch.stack([_[5] for _ in data], dim=0)

        batch_doc_sample = batch_doc_sample[:,:,range(0,batch_max_doc_len)]
        batch_doc_mask_sample = batch_doc_mask_sample[:,:,range(0,batch_max_doc_len)]
        batch_doc_global_mask_sample = batch_doc_global_mask_sample[:,:,range(0,batch_max_doc_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_class = torch.stack([_[6] for _ in data], dim=0)
        batch_doc_order_mask = torch.stack([_[7] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_sent_num = torch.stack([_[8] for _ in data], dim=0)
        batch_doc_sent_positions = torch.stack([_[9] for _ in data], dim=0)
        batch_doc_sent_positions = batch_doc_sent_positions[:, :, range(0, 2 * batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mode = data[0][10]
        res = {'query': batch_query_sample, 'query_attn_mask': batch_query_mask_sample, 'query_global_mask': batch_query_global_sample,
               'ctx_doc': batch_doc_sample, 'ctx_attn_mask': batch_doc_mask_sample, 'ctx_global_mask': batch_doc_global_mask_sample,
               'class': batch_class, 'order_mask': batch_doc_order_mask, 'mode': mode, 'sent_num': batch_doc_sent_num, 'sent_position': batch_doc_sent_positions}
        return res

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
def read_train_dev_data_frame(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    file_path = '../data/hotpotqa/fullwiki_qa'
    dev_file_name = 'hotpot_dev_full_wiki_tokenized.json'

    from transformers.configuration_longformer import LongformerConfig
    from transformers import LongformerTokenizer

    config = LongformerConfig()
    # print(config.hidden_size)

    data_frame = read_train_dev_data_frame(file_path=file_path, json_fileName=dev_file_name)
    longtokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    query_tensorizer = LongformerTensorizer(tokenizer=longtokenizer, max_length=128)
    # # print(query_tensorizer.get_pad_id())
    # # print(longtokenizer.pad_token_id)
    document_tensorizer = LongformerTensorizer(tokenizer=longtokenizer, max_length=701)

    start_time = time()
    dev_dataloader = DataLoader(
        HotpotDevDataset(dev_data_frame=data_frame, query_tensorizer=query_tensorizer, doc_tensorizer=document_tensorizer, max_doc_num=10),
        batch_size=4,
        shuffle=False,
        num_workers=6,
        collate_fn=HotpotDevDataset.collate_fn
    )
    for x in dev_dataloader:
        y= x
        # print(y['query'].shape, y['query_attn_mask'].shape, y['query_global_mask'].shape,
        #       y['ctx_doc'].shape, y['ctx_attn_mask'].shape, y['ctx_global_mask'].shape)
        # break
        # print(y)
    print('Dev Runtime = {}'.format(time() -start_time))

    start_time = time()
    train_file_name = 'hotpot_train_full_wiki_tokenized.json'
    data_frame = read_train_dev_data_frame(file_path=file_path, json_fileName=train_file_name)
    train_dataloader = DataLoader(
        HotpotTrainDataset(train_data_frame=data_frame,  query_tensorizer=query_tensorizer,
                           doc_tensorizer=document_tensorizer, negative_sample_size=8, mode='head-batch'),
        batch_size=4,
        shuffle=False,
        num_workers=6,
        collate_fn=HotpotTrainDataset.collate_fn
    )
    for x in train_dataloader:
        y= x
    print('Train Runtime = {}'.format(time() -start_time))