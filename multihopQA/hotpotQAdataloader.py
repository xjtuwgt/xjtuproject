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
import torch.nn.functional as F
from scipy.linalg import block_diag
from multihopQA.longformerQAUtils import LongformerQATensorizer

class HotpotDataset(Dataset): ##for training data loader
    def __init__(self, data_frame: DataFrame, hotpot_tensorizer: LongformerQATensorizer,
                 pad_neg_doc_num=8, max_sent_num=150, training=True, training_shuffle=False,
                 global_mask_type: str = 'query_doc'):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_len = hotpot_tensorizer.max_length
        self.hotpot_tensorizer = hotpot_tensorizer
        self.training = training
        self.pad_neg_doc_num = pad_neg_doc_num if pad_neg_doc_num > 0 else 0
        assert self.pad_neg_doc_num <= 8
        if training:
            self.max_doc_num = self.pad_neg_doc_num + 2
        else:
            self.max_doc_num = 10
        self.max_sent_num = max_sent_num
        self.global_mask_type = global_mask_type # query, query_doc, query_doc_sent
        self.training_shuffle = training_shuffle

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        pos_ctx_encode, pos_ctx_lens, neg_ctx_encode, neg_ctx_lens = example['p_ctx_encode'], example['p_ctx_lens'], \
                                                                     example['n_ctx_encode'], example['n_ctx_lens']
        norm_answer = example['norm_answer']
        if norm_answer.strip() in ['yes', 'no', 'noanswer']: ## yes: 1, no/noanswer: 2, span = 0
            yes_no_label = torch.LongTensor([1]) if norm_answer.strip() == 'yes' else torch.LongTensor([2])
        else:
            yes_no_label = torch.LongTensor([0])
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        yes_no_question = example['yes_no']
        not_found_answer = example['no_found']
        if not_found_answer:
            yes_no_question = True
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.pad_neg_doc_num <= 0:
            ctx_enocde = pos_ctx_encode
            ctx_lens = pos_ctx_lens
            doc_labels = [1] * len(pos_ctx_encode)
            doc_num = len(doc_labels)
            ctx_position = [_[4] for _ in example['p_ctx']]
            ctx_weights = [_[2] for _ in example['p_ctx']]
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            orig_orders = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(ctx_position))]
            ctx_enocde = [ctx_enocde[orig_orders[i]] for i in range(doc_num)]
            ctx_lens = [ctx_lens[orig_orders[i]] for i in range(doc_num)]
            doc_labels = [doc_labels[orig_orders[i]] for i in range(doc_num)]
            ctx_weights = [ctx_weights[orig_orders[i]] for i in range(doc_num)]
        else:
            pos_position = [_[4] for _ in example['p_ctx']]
            neg_position = [_[4] for _ in example['n_ctx']]
            pos_weights = [_[2] for _ in example['p_ctx']]
            neg_weights = [_[2] for _ in example['n_ctx']]
            if self.pad_neg_doc_num <= len(neg_ctx_encode):
                neg_ctx_encode = neg_ctx_encode[:self.pad_neg_doc_num]
                neg_ctx_lens = neg_ctx_lens[:self.pad_neg_doc_num]
                neg_position = neg_position[:self.pad_neg_doc_num]
                neg_weights = neg_weights[:self.pad_neg_doc_num]

            ctx_enocde = pos_ctx_encode + neg_ctx_encode
            ctx_lens = pos_ctx_lens + neg_ctx_lens
            doc_labels = [1] * len(pos_ctx_encode) + [0] * len(neg_ctx_encode)
            ctx_weights = pos_weights + neg_weights
            ctx_orig_position = pos_position + neg_position
            doc_num = len(doc_labels)
            assert len(ctx_enocde) == len(ctx_lens) and len(ctx_enocde) == len(doc_labels) and doc_num >=2
            ##++++++++++++++++++++++++++++++++
            if not self.training_shuffle:
                orig_orders = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(ctx_orig_position))]
                ctx_enocde = [ctx_enocde[orig_orders[i]] for i in range(doc_num)]
                ctx_lens = [ctx_lens[orig_orders[i]] for i in range(doc_num)]
                doc_labels = [doc_labels[orig_orders[i]] for i in range(doc_num)]
                ctx_weights = [ctx_weights[orig_orders[i]] for i in range(doc_num)]
            ##++++++++++++++++++++++++++++++++
            else:
                shuffle_ctx_index = np.random.choice(doc_num, doc_num, replace=False)
                ctx_enocde = [ctx_enocde[shuffle_ctx_index[i]] for i in range(doc_num)]
                ctx_lens = [ctx_lens[shuffle_ctx_index[i]] for i in range(doc_num)]
                doc_labels = [doc_labels[shuffle_ctx_index[i]] for i in range(doc_num)]
                ctx_weights = [ctx_weights[shuffle_ctx_index[i]] for i in range(doc_num)]
            ##++++++++++++++++++++++++++++++++
        ####################################################
        ctx_label_weight = [(i, doc_labels[i], ctx_weights[i]) for i in range(len(doc_labels)) if doc_labels[i] == 1]
        assert len(ctx_label_weight) == 2
        if ctx_label_weight[0][2] < ctx_label_weight[1][2]:
            head_doc_idx, tail_doc_idx = torch.LongTensor([ctx_label_weight[0][0]]), torch.LongTensor([ctx_label_weight[1][0]])
        else:
            head_doc_idx, tail_doc_idx = torch.LongTensor([ctx_label_weight[1][0]]), torch.LongTensor([ctx_label_weight[0][0]])
        ####################################################

        concat_encode = query_encode
        concat_len = query_len
        doc_start_end_pair_list = []
        sent_start_end_pair_list = []
        sent_lens = []
        ###############################
        sent_nums = []
        ###############################
        supp_sent_labels_list = []
        answer_position_list = []
        previous_len = query_len
        concat_sent_num = 0
        for doc_idx, doc_tup in enumerate(ctx_enocde):
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, _ = doc_tup
            concat_sent_num = concat_sent_num + len(sent_start_end_pair)
            sent_lens = sent_lens + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
            sent_nums = sent_nums + [len(sent_start_end_pair)]
            assert len(doc_encode_ids) == ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i \
                   and doc_len_i == ctx_lens[doc_idx] and len(sent_start_end_pair) == len(supp_sent_labels)
            concat_encode = concat_encode + doc_encode_ids
            concat_len = concat_len + doc_len_i
            doc_start_end_pair_list.append((previous_len, previous_len + doc_len_i - 1))
            sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
            sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
            supp_sent_labels_list = supp_sent_labels_list + supp_sent_labels
            if len(answer_positions) > 0:
                for a_idx, answer_pos in enumerate(answer_positions):
                    sent_a_idx, a_start, a_end = answer_pos
                    sent_off_set = sent_start_end_pair_i[sent_a_idx][0]
                    temp_position = (sent_off_set + a_start, sent_off_set + a_end)
                    answer_position_list.append(temp_position)
            previous_len = previous_len + doc_len_i

        assert doc_start_end_pair_list[-1][1] + 1 == concat_len
        assert len(doc_labels) == len(doc_start_end_pair_list)
        assert previous_len == len(concat_encode) and previous_len == concat_len
        assert len(supp_sent_labels_list) == len(sent_start_end_pair_list) and concat_sent_num == len(sent_start_end_pair_list)
        # if not yes_no_question:
        #     answer = example['norm_answer']
        #     assert len(answer_position_list) > 0, 'yes_no: {} {}, {}'.format(yes_no_question, len(answer_position_list), answer)

        def position_filter(start_end_pair_list: list):
            filtered_positions = []
            filtered_lens = []
            for pos_pair in start_end_pair_list:
                p_st, p_en = pos_pair
                if p_st >= self.max_len:
                    p_st, p_en = 0, 0
                else:
                    p_st = p_st
                    p_en = self.max_len - 1 if p_en >=self.max_len else p_en
                if p_en == 0:
                    filtered_lens.append(0)
                else:
                    filtered_lens.append(p_en - p_st + 1)

                filtered_positions.append((p_st, p_en))
            return filtered_positions, filtered_lens

        # print('ctx before', ctx_lens)
        doc_start_end_pair_list, ctx_lens = position_filter(doc_start_end_pair_list)
        # print('ctx after', ctx_lens)
        if doc_num < self.max_doc_num:
            pad_doc_num = self.max_doc_num - doc_num
            doc_start_end_pair_list = doc_start_end_pair_list + [(0, 0)] * pad_doc_num
            ctx_lens = ctx_lens + [0] * pad_doc_num
            doc_labels = doc_labels + [0] * pad_doc_num
            sent_nums = sent_nums + [0] * pad_doc_num
        ###############################################################################################################
        # print(len(sent_nums))
        def mask_generation(sent_num_docs: list, max_sent_num: int):
            assert len(sent_num_docs) > 0 and sent_num_docs[0] > 0
            ss_attn_mask = np.ones((sent_num_docs[0], sent_num_docs[0]))
            sd_attn_mask = np.ones((1, sent_num_docs[0]))
            doc_pad_num = 0
            for idx in range(1, len(sent_num_docs)):
                sent_num_i = sent_num_docs[idx]
                if sent_num_i > 0:
                    ss_mask_i = np.ones((sent_num_i, sent_num_i))
                    ss_attn_mask = block_diag(ss_attn_mask, ss_mask_i)
                    sd_mask_i = np.ones((1, sent_num_i))
                    sd_attn_mask = block_diag(sd_attn_mask, sd_mask_i)
                else:
                    doc_pad_num = doc_pad_num + 1

            sent_num_sum = sum(sent_num_docs)
            assert sent_num_sum <= max_sent_num, '{}, max {}'.format(sent_num_sum, max_sent_num)
            ss_attn_mask = torch.from_numpy(ss_attn_mask).type(torch.bool)
            sd_attn_mask = torch.from_numpy(sd_attn_mask).type(torch.bool)
            sent_pad_num = max_sent_num - sent_num_sum
            if sent_pad_num > 0:
                ss_attn_mask = F.pad(ss_attn_mask, [0, sent_pad_num, 0, sent_pad_num], 'constant', False)
                sd_attn_mask = F.pad(sd_attn_mask, [0, sent_pad_num, 0, 0], 'constant', False)
            if doc_pad_num > 0:
                sd_attn_mask = F.pad(sd_attn_mask, [0, 0, 0, doc_pad_num], 'constant', False)
            # print(ss_attn_mask.shape)
            # print(sd_attn_mask.shape)
            return ss_attn_mask, sd_attn_mask
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=sent_nums, max_sent_num=self.max_sent_num)
        ###############################################################################################################
        # print('sent before', sent_lens)
        sent_start_end_pair_list, sent_lens = position_filter(sent_start_end_pair_list)
        # print('sent after', sent_lens)
        # if not yes_no_question:
        #     test = concat_encode[answer_position_list[0][0]: (answer_position_list[0][1] + 1)]
        #     encode_test = hotpot_tensorizer.to_string(test)
        #     print('encode {}\npos_do {}\n {}'.format(encode_test, example['norm_answer'], encode_test == example['norm_answer']))
        #     print('{}\n {} \n{}'.format(sent_start_end_pair_list[-1][1], doc_start_end_pair_list[-1][1], concat_len))
        if concat_sent_num < self.max_sent_num:
            pad_sent_num = self.max_sent_num - concat_sent_num
            sent_start_end_pair_list = sent_start_end_pair_list + [(0, 0)] * pad_sent_num
            sent_lens = sent_lens + [0] * pad_sent_num
            supp_sent_labels_list = supp_sent_labels_list + [0] * pad_sent_num

        if not yes_no_question:
            rand_answer_idx = np.random.randint(len(answer_position_list))
            answer_position = answer_position_list[rand_answer_idx]
        else:
            answer_position = (0, 0)

        cat_doc_encodes = self.hotpot_tensorizer.token_ids_to_tensor(token_ids=concat_encode)
        cat_doc_attention_mask = self.hotpot_tensorizer.get_attn_mask(token_ids_tensor=cat_doc_encodes)
        if self.global_mask_type == 'query':
            query_mask_idxes = [x for x in range(query_len)]
        elif self.global_mask_type == 'query_doc':
            query_mask_idxes = [x for x in range(query_len)] + [x[0]  for x in doc_start_end_pair_list] + [x[1] for x in doc_start_end_pair_list]
        elif self.global_mask_type == 'query_doc_sent':
            query_mask_idxes = [x for x in range(query_len)] + [x[0]  for x in doc_start_end_pair_list] + [x[1] for x in doc_start_end_pair_list] + \
                               [x[0]  for x in sent_start_end_pair_list] + [x[1] for x in sent_start_end_pair_list]
        else:
            query_mask_idxes = [x for x in range(query_len)]
        cat_doc_global_attn_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes, gobal_mask_idxs=query_mask_idxes)

        doc_start_idxes = torch.LongTensor([x[0] for x in doc_start_end_pair_list])
        doc_end_idexes = torch.LongTensor([x[1] for x in doc_start_end_pair_list])
        sent_start_idxes = torch.LongTensor([x[0] for x in sent_start_end_pair_list])
        sent_end_idxes = torch.LongTensor([x[1] for x in sent_start_end_pair_list])
        answer_start_idx, answer_end_idx = torch.LongTensor([answer_position[0]]), torch.LongTensor([answer_position[1]])
        doc_lens = torch.LongTensor(ctx_lens)
        doc_labels = torch.LongTensor(doc_labels)
        sent_lens = torch.LongTensor(sent_lens)
        supp_sent_labels = torch.LongTensor(supp_sent_labels_list)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if concat_len > self.max_len:
            concat_len = self.max_len
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, doc_end_idexes, sent_start_idxes, \
               sent_end_idxes, answer_start_idx, \
               answer_end_idx, doc_lens, doc_labels, sent_lens, supp_sent_labels, yes_no_label, head_doc_idx, \
               tail_doc_idx, ss_attn_mask, sd_attn_mask, concat_len, concat_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[18] for _ in data])
        batch_max_sent_num = max([_[19] for _ in data])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        batch_doc_ends = torch.stack([_[4] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_sent_ends = torch.stack([_[6] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_answer_starts = torch.stack([_[7] for _ in data], dim=0)
        batch_answer_ends = torch.stack([_[8] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[9] for _ in data], dim=0)
        batch_doc_labels = torch.stack([_[10] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[11] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        batch_sent_labels = torch.stack([_[12] for _ in data], dim=0)
        batch_sent_labels = batch_sent_labels[:, range(0, batch_max_sent_num)]
        batch_yes_no = torch.stack([_[13] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_head_idx = torch.stack([_[14] for _ in data], dim=0)
        batch_tail_idx = torch.stack([_[15] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[16] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[17] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts,
               'doc_end': batch_doc_ends, 'sent_start': batch_sent_starts,
               'sent_end': batch_sent_ends, 'ans_start': batch_answer_starts, 'ans_end': batch_answer_ends,
               'doc_lens': batch_doc_lens, 'doc_labels': batch_doc_labels, 'sent_lens': batch_sent_lens,
               'sent_labels': batch_sent_labels, 'yes_no': batch_yes_no, 'head_idx': batch_head_idx,
               'tail_idx': batch_tail_idx, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask}
        return res
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def read_train_dev_data_frame(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame
##**********************************************************************************************************************

class HotpotDevDataset(Dataset): ##for dev dataloader
    def __init__(self, data_frame: DataFrame, hotpot_tensorizer: LongformerQATensorizer,
                 max_doc_num=10, max_sent_num=150, global_mask_type: str = 'query_doc'):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_len = hotpot_tensorizer.max_length
        self.hotpot_tensorizer = hotpot_tensorizer
        self.max_doc_num = max_doc_num
        self.max_sent_num = max_sent_num
        self.global_mask_type = global_mask_type # query, query_doc, query_doc_sent

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        pos_ctx_encode, pos_ctx_lens, neg_ctx_encode, neg_ctx_lens = example['p_ctx_encode'], example['p_ctx_lens'], \
                                                                     example['n_ctx_encode'], example['n_ctx_lens']
        norm_answer = example['norm_answer']
        if norm_answer.strip() in ['yes', 'no', 'noanswer']: ## yes: 1, no/noanswer: 2, span = 0
            yes_no_label = torch.LongTensor([1]) if norm_answer.strip() == 'yes' else torch.LongTensor([2])
        else:
            yes_no_label = torch.LongTensor([0])
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        yes_no_question = example['yes_no']
        not_found_answer = example['no_found']
        if not_found_answer:
            yes_no_question = True
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_enocde = pos_ctx_encode + neg_ctx_encode
        ctx_lens = pos_ctx_lens + neg_ctx_lens
        doc_labels = [1] * len(pos_ctx_encode) + [0] * len(neg_ctx_encode)
        doc_num = len(doc_labels)
        pos_position = [_[4] for _ in example['p_ctx']]
        neg_position = [_[4] for _ in example['n_ctx']]
        pos_weights = [_[2] for _ in example['p_ctx']]
        neg_weights = [_[2] for _ in example['n_ctx']]
        ctx_weights = pos_weights + neg_weights
        ctx_orig_position = pos_position + neg_position
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        orig_orders = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(ctx_orig_position))]
        ctx_enocde = [ctx_enocde[orig_orders[i]] for i in range(doc_num)]
        ctx_lens = [ctx_lens[orig_orders[i]] for i in range(doc_num)]
        doc_labels = [doc_labels[orig_orders[i]] for i in range(doc_num)]
        ctx_weights = [ctx_weights[orig_orders[i]] for i in range(doc_num)]
        ####################################################
        ctx_label_weight = [(i, doc_labels[i], ctx_weights[i]) for i in range(len(doc_labels)) if doc_labels[i] == 1]
        assert len(ctx_label_weight) == 2
        if ctx_label_weight[0][2] < ctx_label_weight[1][2]:
            head_doc_idx, tail_doc_idx = torch.LongTensor([ctx_label_weight[0][0]]), torch.LongTensor([ctx_label_weight[1][0]])
        else:
            head_doc_idx, tail_doc_idx = torch.LongTensor([ctx_label_weight[1][0]]), torch.LongTensor([ctx_label_weight[0][0]])
        ####################################################
        concat_encode = query_encode
        concat_len = query_len
        doc_start_end_pair_list = []
        sent_start_end_pair_list = []
        sent_lens = []
        ###############################
        sent_nums = []
        supp_fact_doc_idx_list = []
        supp_fact_sent_idx_list = []
        ###############################
        supp_sent_labels_list = []
        answer_position_list = []
        previous_len = query_len
        concat_sent_num = 0
        for doc_idx, doc_tup in enumerate(ctx_enocde):
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, _ = doc_tup
            # =======================================
            supp_fact_doc_idx_list = supp_fact_doc_idx_list + [doc_idx] * len(sent_start_end_pair)
            supp_fact_sent_idx_list = supp_fact_sent_idx_list + [x for x in range(len(sent_start_end_pair))]
            # =======================================
            concat_sent_num = concat_sent_num + len(sent_start_end_pair)
            sent_lens = sent_lens + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
            sent_nums = sent_nums + [len(sent_start_end_pair)]
            assert len(doc_encode_ids) == ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i \
                   and doc_len_i == ctx_lens[doc_idx] and len(sent_start_end_pair) == len(supp_sent_labels)
            concat_encode = concat_encode + doc_encode_ids
            concat_len = concat_len + doc_len_i
            doc_start_end_pair_list.append((previous_len, previous_len + doc_len_i - 1))
            sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
            sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
            supp_sent_labels_list = supp_sent_labels_list + supp_sent_labels
            if len(answer_positions) > 0:
                for a_idx, answer_pos in enumerate(answer_positions):
                    sent_a_idx, a_start, a_end = answer_pos
                    sent_off_set = sent_start_end_pair_i[sent_a_idx][0]
                    temp_position = (sent_off_set + a_start, sent_off_set + a_end)
                    answer_position_list.append(temp_position)
            previous_len = previous_len + doc_len_i

        # +++++++++++++++++++++++++++
        assert len(supp_fact_doc_idx_list) == len(supp_fact_sent_idx_list) and len(supp_fact_doc_idx_list) == len(supp_sent_labels_list)
        # +++++++++++++++++++++++++++
        assert doc_start_end_pair_list[-1][1] + 1 == concat_len
        assert len(doc_labels) == len(doc_start_end_pair_list)
        assert previous_len == len(concat_encode) and previous_len == concat_len
        assert len(supp_sent_labels_list) == len(sent_start_end_pair_list) and concat_sent_num == len(sent_start_end_pair_list)
        # if not yes_no_question:
        #     answer = example['norm_answer']
        #     assert len(answer_position_list) > 0, 'yes_no: {} {}, {}'.format(yes_no_question, len(answer_position_list), answer)
        def position_filter(start_end_pair_list: list):
            filtered_positions = []
            filtered_lens = []
            for pos_pair in start_end_pair_list:
                p_st, p_en = pos_pair
                if p_st >= self.max_len:
                    p_st, p_en = 0, 0
                else:
                    p_st = p_st
                    p_en = self.max_len - 1 if p_en >=self.max_len else p_en
                if p_en == 0:
                    filtered_lens.append(0)
                else:
                    filtered_lens.append(p_en - p_st + 1)

                filtered_positions.append((p_st, p_en))
            return filtered_positions, filtered_lens

        # print('ctx before', ctx_lens)
        doc_start_end_pair_list, ctx_lens = position_filter(doc_start_end_pair_list)
        # print('ctx after', ctx_lens)
        if doc_num < self.max_doc_num:
            pad_doc_num = self.max_doc_num - doc_num
            doc_start_end_pair_list = doc_start_end_pair_list + [(0, 0)] * pad_doc_num
            ctx_lens = ctx_lens + [0] * pad_doc_num
            doc_labels = doc_labels + [0] * pad_doc_num
            sent_nums = sent_nums + [0] * pad_doc_num
        ###############################################################################################################
        # print(len(sent_nums))
        def mask_generation(sent_num_docs: list, max_sent_num: int):
            assert len(sent_num_docs) > 0 and sent_num_docs[0] > 0
            ss_attn_mask = np.ones((sent_num_docs[0], sent_num_docs[0]))
            sd_attn_mask = np.ones((1, sent_num_docs[0]))
            doc_pad_num = 0
            for idx in range(1, len(sent_num_docs)):
                sent_num_i = sent_num_docs[idx]
                if sent_num_i > 0:
                    ss_mask_i = np.ones((sent_num_i, sent_num_i))
                    ss_attn_mask = block_diag(ss_attn_mask, ss_mask_i)
                    sd_mask_i = np.ones((1, sent_num_i))
                    sd_attn_mask = block_diag(sd_attn_mask, sd_mask_i)
                else:
                    doc_pad_num = doc_pad_num + 1

            sent_num_sum = sum(sent_num_docs)
            assert sent_num_sum <= max_sent_num, '{}, max {}'.format(sent_num_sum, max_sent_num)
            ss_attn_mask = torch.from_numpy(ss_attn_mask).type(torch.bool)
            sd_attn_mask = torch.from_numpy(sd_attn_mask).type(torch.bool)
            sent_pad_num = max_sent_num - sent_num_sum
            if sent_pad_num > 0:
                ss_attn_mask = F.pad(ss_attn_mask, [0, sent_pad_num, 0, sent_pad_num], 'constant', False)
                sd_attn_mask = F.pad(sd_attn_mask, [0, sent_pad_num, 0, 0], 'constant', False)
            if doc_pad_num > 0:
                sd_attn_mask = F.pad(sd_attn_mask, [0, 0, 0, doc_pad_num], 'constant', False)
            # print(ss_attn_mask.shape)
            # print(sd_attn_mask.shape)
            return ss_attn_mask, sd_attn_mask
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=sent_nums, max_sent_num=self.max_sent_num)
        ###############################################################################################################
        # print('sent before', sent_lens)
        sent_start_end_pair_list, sent_lens = position_filter(sent_start_end_pair_list)
        # print('sent after', sent_lens)
        # if not yes_no_question:
        #     test = concat_encode[answer_position_list[0][0]: (answer_position_list[0][1] + 1)]
        #     encode_test = hotpot_tensorizer.to_string(test)
        #     print('encode {}\npos_do {}\n {}'.format(encode_test, example['norm_answer'], encode_test == example['norm_answer']))
        #     print('{}\n {} \n{}'.format(sent_start_end_pair_list[-1][1], doc_start_end_pair_list[-1][1], concat_len))
        if concat_sent_num < self.max_sent_num:
            pad_sent_num = self.max_sent_num - concat_sent_num
            sent_start_end_pair_list = sent_start_end_pair_list + [(0, 0)] * pad_sent_num
            sent_lens = sent_lens + [0] * pad_sent_num
            supp_sent_labels_list = supp_sent_labels_list + [0] * pad_sent_num
            ##+++++++++++++++
            supp_fact_doc_idx_list = supp_fact_doc_idx_list + [0] * pad_sent_num
            supp_fact_sent_idx_list = supp_fact_sent_idx_list + [0] * pad_sent_num
            ##+++++++++++++++

        if not yes_no_question:
            rand_answer_idx = np.random.randint(len(answer_position_list))
            answer_position = answer_position_list[rand_answer_idx]
        else:
            answer_position = (0, 0)

        cat_doc_encodes = self.hotpot_tensorizer.token_ids_to_tensor(token_ids=concat_encode)
        cat_doc_attention_mask = self.hotpot_tensorizer.get_attn_mask(token_ids_tensor=cat_doc_encodes)
        if self.global_mask_type == 'query':
            query_mask_idxes = [x for x in range(query_len)]
        elif self.global_mask_type == 'query_doc':
            query_mask_idxes = [x for x in range(query_len)] + [x[0]  for x in doc_start_end_pair_list] + [x[1] for x in doc_start_end_pair_list]
        elif self.global_mask_type == 'query_doc_sent':
            query_mask_idxes = [x for x in range(query_len)] + [x[0]  for x in doc_start_end_pair_list] + [x[1] for x in doc_start_end_pair_list] + \
                               [x[0]  for x in sent_start_end_pair_list] + [x[1] for x in sent_start_end_pair_list]
        else:
            query_mask_idxes = [x for x in range(query_len)]
        cat_doc_global_attn_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes, gobal_mask_idxs=query_mask_idxes)

        doc_start_idxes = torch.LongTensor([x[0] for x in doc_start_end_pair_list])
        doc_end_idexes = torch.LongTensor([x[1] for x in doc_start_end_pair_list])
        sent_start_idxes = torch.LongTensor([x[0] for x in sent_start_end_pair_list])
        sent_end_idxes = torch.LongTensor([x[1] for x in sent_start_end_pair_list])
        answer_start_idx, answer_end_idx = torch.LongTensor([answer_position[0]]), torch.LongTensor([answer_position[1]])
        doc_lens = torch.LongTensor(ctx_lens)
        doc_labels = torch.LongTensor(doc_labels)
        sent_lens = torch.LongTensor(sent_lens)
        supp_sent_labels = torch.LongTensor(supp_sent_labels_list)
        #+++++++++++++++++++++++++++++++++++++++++
        supp_fact_doc_idx = torch.LongTensor(supp_fact_doc_idx_list)
        supp_fact_sent_idx = torch.LongTensor(supp_fact_sent_idx_list)
        #+++++++++++++++++++++++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if concat_len > self.max_len:
            concat_len = self.max_len
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, doc_end_idexes, sent_start_idxes, \
               sent_end_idxes, answer_start_idx, \
               answer_end_idx, doc_lens, doc_labels, sent_lens, supp_sent_labels, yes_no_label, head_doc_idx, \
               tail_doc_idx, ss_attn_mask, sd_attn_mask, supp_fact_doc_idx, supp_fact_sent_idx, concat_len, concat_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[20] for _ in data])
        batch_max_sent_num = max([_[21] for _ in data])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        batch_doc_ends = torch.stack([_[4] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_sent_ends = torch.stack([_[6] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_answer_starts = torch.stack([_[7] for _ in data], dim=0)
        batch_answer_ends = torch.stack([_[8] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[9] for _ in data], dim=0)
        batch_doc_labels = torch.stack([_[10] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[11] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        batch_sent_labels = torch.stack([_[12] for _ in data], dim=0)
        batch_sent_labels = batch_sent_labels[:, range(0, batch_max_sent_num)]
        batch_yes_no = torch.stack([_[13] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_head_idx = torch.stack([_[14] for _ in data], dim=0)
        batch_tail_idx = torch.stack([_[15] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[16] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[17] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_supp_fact_doc_idx = torch.stack([_[18] for _ in data], dim=0)
        batch_supp_fact_sent_idx = torch.stack([_[19] for _ in data], dim=0)
        batch_supp_fact_doc_idx = batch_supp_fact_doc_idx[:, range(0, batch_max_sent_num)]
        batch_supp_fact_sent_idx = batch_supp_fact_sent_idx[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts,
               'doc_end': batch_doc_ends, 'sent_start': batch_sent_starts,
               'sent_end': batch_sent_ends, 'ans_start': batch_answer_starts, 'ans_end': batch_answer_ends,
               'doc_lens': batch_doc_lens, 'doc_labels': batch_doc_labels, 'sent_lens': batch_sent_lens,
               'sent_labels': batch_sent_labels, 'yes_no': batch_yes_no, 'head_idx': batch_head_idx, 'fact_doc': batch_supp_fact_doc_idx,
               'fact_sent': batch_supp_fact_sent_idx, 'tail_idx': batch_tail_idx,
               'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask}
        return res

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HotpotTestDataset(Dataset): # for test data loader
    def __init__(self, data_frame: DataFrame, hotpot_tensorizer: LongformerQATensorizer, max_doc_num=10, max_sent_num=150,
                 global_mask_type: str = 'query_doc'):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_len = hotpot_tensorizer.max_length
        self.hotpot_tensorizer = hotpot_tensorizer
        self.max_doc_num = max_doc_num
        self.max_sent_num = max_sent_num
        self.global_mask_type = global_mask_type # query, query_doc, query_doc_sent

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        ctx_encode, ctx_lens = example['ctx_encode'], example['ctx_lens']
        doc_num = len(ctx_encode)

        concat_encode = query_encode
        concat_len = query_len
        doc_start_end_pair_list = []
        sent_start_end_pair_list = []
        sent_lens = []
        ###############################
        sent_nums = []
        supp_fact_doc_idx_list = []
        supp_fact_sent_idx_list = []
        ###############################
        previous_len = query_len
        concat_sent_num = 0
        for doc_idx, doc_tup in enumerate(ctx_encode):
            doc_encode_ids, doc_len_i, sent_start_end_pair, _, _ = doc_tup
            concat_sent_num = concat_sent_num + len(sent_start_end_pair)
            sent_lens = sent_lens + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
            sent_nums = sent_nums + [len(sent_start_end_pair)]
            # =======================================
            supp_fact_doc_idx_list = supp_fact_doc_idx_list + [doc_idx] * len(sent_start_end_pair)
            supp_fact_sent_idx_list = supp_fact_sent_idx_list + [x for x in range(len(sent_start_end_pair))]
            # =======================================
            assert len(doc_encode_ids) == ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i \
                   and doc_len_i == ctx_lens[doc_idx]
            concat_encode = concat_encode + doc_encode_ids
            concat_len = concat_len + doc_len_i
            doc_start_end_pair_list.append((previous_len, previous_len + doc_len_i - 1))
            sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
            sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
            previous_len = previous_len + doc_len_i

        # +++++++++++++++++++++++++++
        assert len(supp_fact_doc_idx_list) == len(supp_fact_sent_idx_list) and len(supp_fact_sent_idx_list) == len(sent_lens)
        # +++++++++++++++++++++++++++
        assert doc_start_end_pair_list[-1][1] + 1 == concat_len
        assert previous_len == len(concat_encode) and previous_len == concat_len
        # if not yes_no_question:
        #     answer = example['norm_answer']
        #     assert len(answer_position_list) > 0, 'yes_no: {} {}, {}'.format(yes_no_question, len(answer_position_list), answer)

        def position_filter(start_end_pair_list: list):
            filtered_positions = []
            filtered_lens = []
            for pos_pair in start_end_pair_list:
                p_st, p_en = pos_pair
                if p_st >= self.max_len:
                    p_st, p_en = 0, 0
                else:
                    p_st = p_st
                    p_en = self.max_len - 1 if p_en >= self.max_len else p_en
                if p_en == 0:
                    filtered_lens.append(0)
                else:
                    filtered_lens.append(p_en - p_st + 1)

                filtered_positions.append((p_st, p_en))
            return filtered_positions, filtered_lens

        # print('ctx before', ctx_lens)
        doc_start_end_pair_list, ctx_lens = position_filter(doc_start_end_pair_list)
        # print('ctx after', ctx_lens)
        if doc_num < self.max_doc_num:
            pad_doc_num = self.max_doc_num - doc_num
            doc_start_end_pair_list = doc_start_end_pair_list + [(0, 0)] * pad_doc_num
            ctx_lens = ctx_lens + [0] * pad_doc_num
            sent_nums = sent_nums + [0] * pad_doc_num
        ###############################################################################################################
        # print(len(sent_nums))
        def mask_generation(sent_num_docs: list, max_sent_num: int):
            assert len(sent_num_docs) > 0 and sent_num_docs[0] > 0
            ss_attn_mask = np.ones((sent_num_docs[0], sent_num_docs[0]))
            sd_attn_mask = np.ones((1, sent_num_docs[0]))
            doc_pad_num = 0
            for idx in range(1, len(sent_num_docs)):
                sent_num_i = sent_num_docs[idx]
                if sent_num_i > 0:
                    ss_mask_i = np.ones((sent_num_i, sent_num_i))
                    ss_attn_mask = block_diag(ss_attn_mask, ss_mask_i)
                    sd_mask_i = np.ones((1, sent_num_i))
                    sd_attn_mask = block_diag(sd_attn_mask, sd_mask_i)
                else:
                    doc_pad_num = doc_pad_num + 1

            sent_num_sum = sum(sent_num_docs)
            assert sent_num_sum <= max_sent_num, '{}, max {}'.format(sent_num_sum, max_sent_num)
            ss_attn_mask = torch.from_numpy(ss_attn_mask).type(torch.bool)
            sd_attn_mask = torch.from_numpy(sd_attn_mask).type(torch.bool)
            sent_pad_num = max_sent_num - sent_num_sum
            if sent_pad_num > 0:
                ss_attn_mask = F.pad(ss_attn_mask, [0, sent_pad_num, 0, sent_pad_num], 'constant', False)
                sd_attn_mask = F.pad(sd_attn_mask, [0, sent_pad_num, 0, 0], 'constant', False)
            if doc_pad_num > 0:
                sd_attn_mask = F.pad(sd_attn_mask, [0, 0, 0, doc_pad_num], 'constant', False)
            # print(ss_attn_mask.shape)
            # print(sd_attn_mask.shape)
            return ss_attn_mask, sd_attn_mask

        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=sent_nums, max_sent_num=self.max_sent_num)
        ###############################################################################################################
        # print('sent before', sent_lens)
        sent_start_end_pair_list, sent_lens = position_filter(sent_start_end_pair_list)
        # print('sent after', sent_lens)
        # if not yes_no_question:
        #     test = concat_encode[answer_position_list[0][0]: (answer_position_list[0][1] + 1)]
        #     encode_test = hotpot_tensorizer.to_string(test)
        #     print('encode {}\npos_do {}\n {}'.format(encode_test, example['norm_answer'], encode_test == example['norm_answer']))
        #     print('{}\n {} \n{}'.format(sent_start_end_pair_list[-1][1], doc_start_end_pair_list[-1][1], concat_len))
        if concat_sent_num < self.max_sent_num:
            pad_sent_num = self.max_sent_num - concat_sent_num
            sent_start_end_pair_list = sent_start_end_pair_list + [(0, 0)] * pad_sent_num
            sent_lens = sent_lens + [0] * pad_sent_num
            ##+++++++++++++++
            supp_fact_doc_idx_list = supp_fact_doc_idx_list + [0] * pad_sent_num
            supp_fact_sent_idx_list = supp_fact_sent_idx_list + [0] * pad_sent_num
            ##+++++++++++++++

        cat_doc_encodes = self.hotpot_tensorizer.token_ids_to_tensor(token_ids=concat_encode)
        cat_doc_attention_mask = self.hotpot_tensorizer.get_attn_mask(token_ids_tensor=cat_doc_encodes)
        if self.global_mask_type == 'query':
            query_mask_idxes = [x for x in range(query_len)]
        elif self.global_mask_type == 'query_doc':
            query_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                          doc_start_end_pair_list]
        elif self.global_mask_type == 'query_doc_sent':
            query_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[1] for x in
                                                                                                          doc_start_end_pair_list] + \
                               [x[0] for x in sent_start_end_pair_list] + [x[1] for x in sent_start_end_pair_list]
        else:
            query_mask_idxes = [x for x in range(query_len)]
        cat_doc_global_attn_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes,
                                                                               gobal_mask_idxs=query_mask_idxes)

        doc_start_idxes = torch.LongTensor([x[0] for x in doc_start_end_pair_list])
        doc_end_idexes = torch.LongTensor([x[1] for x in doc_start_end_pair_list])
        sent_start_idxes = torch.LongTensor([x[0] for x in sent_start_end_pair_list])
        sent_end_idxes = torch.LongTensor([x[1] for x in sent_start_end_pair_list])

        #+++++++++++++++++++++++++++++++++++++++++
        supp_fact_doc_idx = torch.LongTensor(supp_fact_doc_idx_list)
        supp_fact_sent_idx = torch.LongTensor(supp_fact_sent_idx_list)
        #+++++++++++++++++++++++++++++++++++++++++

        doc_lens = torch.LongTensor(ctx_lens)
        sent_lens = torch.LongTensor(sent_lens)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if concat_len > self.max_len:
            concat_len = self.max_len
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, doc_end_idexes, sent_start_idxes, \
               sent_end_idxes, doc_lens, sent_lens, ss_attn_mask, sd_attn_mask, supp_fact_doc_idx, supp_fact_sent_idx, concat_len, concat_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[13] for _ in data])
        batch_max_sent_num = max([_[14] for _ in data])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        batch_doc_ends = torch.stack([_[4] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_sent_ends = torch.stack([_[6] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[7] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[8] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[9] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[10] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_supp_fact_doc_idx = torch.stack([_[11] for _ in data], dim=0)
        batch_supp_fact_sent_idx = torch.stack([_[12] for _ in data], dim=0)
        batch_supp_fact_doc_idx = batch_supp_fact_doc_idx[:, range(0, batch_max_sent_num)]
        batch_supp_fact_sent_idx = batch_supp_fact_sent_idx[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts,
               'doc_end': batch_doc_ends, 'sent_start': batch_sent_starts,
               'sent_end': batch_sent_ends, 'doc_lens': batch_doc_lens, 'sent_lens': batch_sent_lens,
               'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask, 'fact_doc': batch_supp_fact_doc_idx,
               'fact_sent': batch_supp_fact_sent_idx}
        return res
##**********************************************************************************************************************
##****************************************************Data loader testing***********************************************
##**********************************************************************************************************************
def data_loader_consistent_checker():
    file_path = '../data/hotpotqa/distractor_qa'
    dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
    from torch.utils.data import DataLoader
    from transformers import LongformerTokenizer
    from multihopQA.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
    batch_size = 2

    data_frame = read_train_dev_data_frame(file_path=file_path, json_fileName=dev_file_name)
    longtokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    dev_dataloader = DataLoader(
        HotpotDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer, training=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=HotpotDataset.collate_fn
    )

    head_two = data_frame.head(batch_size)
    print(type(head_two))
    for idx, row in head_two.iterrows():
        context = row['context']
        supp_fact_filtered = row['supp_facts_filtered']
        for supp, sen_idx in supp_fact_filtered:
            print('Support doc: {}, sent id: {}'.format(supp, sen_idx))
            print('-' * 70)
        print()
        for doc_idx, doc in enumerate(context):
            print('doc {}: title = {} \n text = {}'.format(doc_idx + 1, doc[0], ' '.join(doc[1])))
            print('-' * 70)
        print('*' * 70)
        print()
        print('Original answer = {}'.format(row['norm_answer']))
        print('=' * 70)
    print('+' * 70)
    print('\n'*3)
    for batch_idx, sample in enumerate(dev_dataloader):
        ctx_encode = sample['ctx_encode']
        doc_start = sample['doc_start'].squeeze(dim=-1)
        doc_end = sample['doc_end'].squeeze(dim=-1)
        answer_start = sample['ans_start'].squeeze(dim=-1)
        answer_end = sample['ans_end'].squeeze(dim=-1)
        head_idx = sample['head_idx'].squeeze(dim=-1)
        tail_idx = sample['tail_idx'].squeeze(dim=-1)
        doc_num = doc_start.shape[1]
        print(doc_start.shape, doc_end.shape)
        for idx in range(ctx_encode.shape[0]):
            ctx_i = ctx_encode[idx]
            doc_start_i = doc_start[idx]
            doc_end_i = doc_end[idx]
            head_i = head_idx[idx].data.item()
            tail_i = tail_idx[idx].data.item()
            ans_start_i = answer_start[idx].data.item()
            ans_end_i = answer_end[idx].data.item()
            # text = hotpot_tensorizer.to_string(ctx_i)
            print('Decoded answer = {}'.format(hotpot_tensorizer.to_string(ctx_i[ans_start_i:(ans_end_i + 1)])))
            for k in range(doc_num):
                doc_k = hotpot_tensorizer.to_string(ctx_i[doc_start_i[k]:(doc_end_i[k] + 1)])
                print('Supp doc {}: text = {}'.format(k+1, doc_k))
                if k == head_i:
                    print('=' * 70)
                    print('Head positive doc {}: text: {}'.format(head_i + 1, doc_k))
                    print('=' * 70)
                if k == tail_i:
                    print('=' * 70)
                    print('Tail positive doc {}: text: {}'.format(tail_i + 1, doc_k))
                    print('=' * 70)
                print('-'*70)
            print('*' * 70)
            print()
        # print(ctx_encode.shape)
        break
    print('Runtime = {}'.format(time() - start_time))

def data_loader_checker():
    file_path = '../data/hotpotqa/distractor_qa'
    dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
    from torch.utils.data import DataLoader
    from transformers import LongformerTokenizer
    from multihopQA.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
    batch_size = 2

    data_frame = read_train_dev_data_frame(file_path=file_path, json_fileName=dev_file_name)
    longtokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    dev_dataloader = DataLoader(
        HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=HotpotDevDataset.collate_fn
    )

    for batch_idx, sample in enumerate(dev_dataloader):
        ans_start = sample['ans_start']
        ans_end = sample['ans_end']
        print(ans_start, ans_end)
        # break
    print('Runtime = {}'.format(time() - start_time))

def test_data_loader_checker():
    file_path = '../data/hotpotqa/distractor_qa'
    dev_file_name = 'hotpot_test_distractor_wiki_tokenized.json'
    from torch.utils.data import DataLoader
    from transformers import LongformerTokenizer
    from multihopQA.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
    batch_size = 4
    data_frame = read_train_dev_data_frame(file_path=file_path, json_fileName=dev_file_name)
    longtokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    test_dataloader = DataLoader(
        HotpotTestDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=HotpotTestDataset.collate_fn
    )
    for batch_idx, sample in enumerate(test_dataloader):
        sd_mask = sample['sd_mask']
        # print(sd_mask[0])
        # print(sd_mask.shape)
        break
    print('Runtime = {}'.format(time() - start_time))

if __name__ == '__main__':
    data_loader_consistent_checker()
    # data_loader_checker()
    # test_data_loader_checker()
    print()