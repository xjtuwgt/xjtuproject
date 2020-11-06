import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopr.hotpotqaIOUtils import *
full_wiki_path = '../data/hotpotqa/fullwiki_qa'
abs_full_wiki_path = os.path.abspath(full_wiki_path)
from pandas import DataFrame
from time import time
from multihopr.longformerUtils import LongformerTensorizer
from multihopr.longformerUtils import PRE_TAINED_LONFORMER_BASE
from transformers import LongformerTokenizer
import itertools
import operator
import numpy as np
import re
import swifter
SPECIAL_QUERY_TOKEN = '<unk>'
SPECIAL_DOCUMENT_TOKEN = '<unk>'
START_SENTENCE_TOKEN = '<s>'
CLS_TOKEN = '<s>'
END_SENTENCE_TOKEN = '</s>'
SEP_TOKEN = '</s>'

def Hotpot_TrainDev_Data_Preprocess(data: DataFrame, tokenizer: LongformerTensorizer):
    """
    Supporting_facts: pair of (title, sentence index) --> (str, int)
    Level: {easy, medium, hard}
    Question: query --> str
    Context: list of pair (title, text) --> (str, list(str))
    Answer: str
    Type: {comparison, bridge}
    """
    def pos_neg_context_split(row):
        question, supporting_facts, contexts, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
        doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
        supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                     if supp_sent_idx < doc_title2doc_len[supp_title]]
        positive_titles = set([x[0] for x in supporting_facts_filtered])
        norm_answer = normalize_text(text=answer)
        yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
        norm_question = normalize_question(question.lower())
        not_found_flag = False
        ################################################################################################################
        pos_doc_num = len(positive_titles)
        pos_ctxs, neg_ctxs = [], []
        for ctx in contexts:
            title, text = ctx[0], ctx[1]
            text_lower = [normalize_text(text=sent) for sent in text]
            if title in positive_titles:
                count = 1
                supp_sent_flags = []
                for supp_title, supp_sent_idx in supporting_facts_filtered:
                    if title == supp_title:
                        supp_sent = text_lower[supp_sent_idx]
                        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
                            new_supp_sent, find_idx = answer_span_replace(norm_answer.strip(), supp_sent)
                            if find_idx >= 0:
                                count = count + 1
                                supp_sent_flags.append((supp_sent_idx, True))
                                text_lower[supp_sent_idx] = new_supp_sent
                            else:
                                supp_sent_flags.append((supp_sent_idx, False))
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                pos_ctxs.append([title.lower(), text_lower, count, supp_sent_flags])  ## Identify the support document with answer
            else:
                neg_ctxs.append([title.lower(), text_lower, 1, []])
        neg_doc_num = len(neg_ctxs)
        pos_counts = [x[2] for x in pos_ctxs]
        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
            if sum(pos_counts) == 2:
                not_found_flag = True
        assert len(pos_counts) == 2
        if (pos_counts[0] > 1 and pos_counts[1] > 1) or (pos_counts[0] <= 1 and pos_counts[1] <= 1):
            answer_type = False
        else:
            answer_type = True
        return norm_question, norm_answer, pos_ctxs, neg_ctxs, supporting_facts_filtered, answer_type, pos_doc_num, neg_doc_num, yes_no_flag, not_found_flag
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['norm_query', 'norm_answer', 'p_ctx', 'n_ctx_H', 'supp_facts_filtered', 'answer_type', 'p_doc_num',
          'n_doc_num', 'yes_no', 'no_found']] = data.swifter\
        .progress_bar(True).apply(lambda row: pd.Series(pos_neg_context_split(row)), axis=1)
    not_found_num = data[data['no_found']].shape[0]
    print('Splitting positive samples from negative samples takes {:.4f} seconds, answer not found = {}'.format(time() - start_time, not_found_num))
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, pos_ctxs, neg_ctxs, norm_answer = row['norm_query'], row['p_ctx'], row['n_ctx_H'], row['norm_answer']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_res = CLS_TOKEN + SPECIAL_QUERY_TOKEN + SEP_TOKEN + norm_question + SEP_TOKEN #
        query_encode_ids = tokenizer.text_encode(text=query_res, add_special_tokens=False) # Query encode
        query_len = len(query_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if norm_answer in ['yes', 'no', 'noanswer']:
            answer_encode_ids = [-1]
        else:
            answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
        answer_len = len(answer_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def document_encoder(title: str, doc_sents: list):
            title_res = CLS_TOKEN + SPECIAL_DOCUMENT_TOKEN + SEP_TOKEN + title + SEP_TOKEN
            title_encode_ids = tokenizer.text_encode(text=title_res, add_special_tokens=False)
            title_len = len(title_encode_ids)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            encode_id_lens = []
            encode_id_lens.append(title_len)
            doc_encode_id_list = []
            doc_encode_id_list.append(title_encode_ids)
            for sent_idx, sent_text in enumerate(doc_sents):
                sent_text_res = START_SENTENCE_TOKEN + sent_text + END_SENTENCE_TOKEN
                sent_encode_ids = tokenizer.text_encode(text=sent_text_res, add_special_tokens=False)
                doc_encode_id_list.append(sent_encode_ids)
                sent_len = len(sent_encode_ids)
                encode_id_lens.append(sent_len)
            doc_sent_len_cum_list = list(itertools.accumulate(encode_id_lens, operator.add))
            sent_start_end_pair = [(doc_sent_len_cum_list[i], doc_sent_len_cum_list[i+1] - 1) for i in range(len(encode_id_lens) - 1)]
            doc_encode_ids = list(itertools.chain.from_iterable(doc_encode_id_list))
            assert len(doc_encode_ids) == doc_sent_len_cum_list[-1]
            return doc_encode_ids, sent_start_end_pair, len(doc_encode_ids), title_len
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        positive_ctx_encode_ids = []
        max_positive_doc_len = 0
        positive_ctx_lens = []
        for p_idx, content in enumerate(pos_ctxs):
            p_title, p_doc, p_doc_weight, supp_sent_flags = content
            p_doc_encode_ids, sent_start_end_pair, p_doc_len_i, p_title_len = document_encoder(title=p_title, doc_sents=p_doc)
            # #########################
            if max_positive_doc_len < p_doc_len_i:
                max_positive_doc_len = p_doc_len_i
            #########################
            assert len(p_doc) == len(sent_start_end_pair)
            positive_ctx_lens.append(p_doc_len_i)
            supp_sent_labels = [0] * len(p_doc)
            ctx_with_answer = False
            answer_positions = [] ## answer position
            for sup_sent_idx, supp_sent_flag in supp_sent_flags:
                supp_sent_labels[sup_sent_idx] = 1
                if supp_sent_flag:
                    start_id, end_id = sent_start_end_pair[sup_sent_idx]
                    supp_sent_labels[sup_sent_idx] = 2
                    supp_sent_encode_ids = p_doc_encode_ids[start_id:end_id]
                    answer_start_idx = find_sub_list(target=answer_encode_ids, source=supp_sent_encode_ids)
                    assert answer_start_idx >= 0, "supp sent {} \n answer={} \n p_doc = {} \n answer={} \n {} \n {}".format(tokenizer.tokenizer.decode(supp_sent_encode_ids),
                                                                                     tokenizer.tokenizer.decode(answer_encode_ids), p_doc[sup_sent_idx], norm_answer, supp_sent_encode_ids, answer_encode_ids)
                    ctx_with_answer = True
                    answer_positions.append((sup_sent_idx, answer_start_idx, answer_start_idx + answer_len - 1))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            p_tuple = (p_doc_encode_ids, p_doc_weight, p_doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len)
            positive_ctx_encode_ids.append(p_tuple)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        negative_ctx_encode_ids = []
        negative_ctx_lens = []
        max_negative_doc_len = 0
        for n_idx, content in enumerate(neg_ctxs):
            n_title, n_doc, n_doc_weight, _ = content
            n_doc_encode_ids, sent_start_end_pair, n_doc_len_i, n_title_len = document_encoder(title=n_title, doc_sents=n_doc)
            negative_ctx_lens.append(n_doc_len_i)
            if max_negative_doc_len < n_doc_len_i:
                max_negative_doc_len = n_doc_len_i
            n_tuple = (n_doc_encode_ids, n_doc_weight, n_doc_len_i, sent_start_end_pair, [0] * len(n_doc), False, [], n_title_len)
            negative_ctx_encode_ids.append(n_tuple)
        return query_encode_ids, query_len, answer_encode_ids, answer_len, positive_ctx_encode_ids, positive_ctx_lens, max_positive_doc_len, \
               negative_ctx_encode_ids, negative_ctx_lens, max_negative_doc_len
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['ques_encode', 'ques_len', 'answer_encode', 'answer_len', 'p_ctx_encode', 'p_ctx_lens', 'pc_max_len', 'n_ctx_encode', 'n_ctx_lens', 'nc_max_len']] = \
        data.swifter.progress_bar(True).apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    data = data[data['n_doc_num'] > 0]
    print('Number of filtered data = {}'.format(data.shape))
    return data

def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question

def normalize_text(text: str) -> str:
    text = ' ' + text.lower().strip()
    return text

def remove_multi_spaces(text: str) -> str:
    text = re.sub("\s\s+" , " ", text)
    return text

def answer_span_replace(answer, sentence):
    find_idx = sentence.find(answer)
    if find_idx < 0:
        return sentence, find_idx
    if find_idx == 0:
        new_sentence = ' ' + sentence
    else:
        new_sentence = sentence[0:find_idx] + ' ' + answer + ' ' + sentence[(find_idx + len(answer)):]
    return new_sentence, find_idx

def find_sub_list(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    def equal_list(a_list, b_list):
        for j in range(len(a_list)):
            if a_list[j] != b_list[j]:
                return False
        return True
    for i in range(len(source) - len(target)):
        temp = source[i:(i+t_len)]
        is_equal = equal_list(target, temp)
        if is_equal:
            return i
    return -1

def Hotpot_Test_Data_Preprocess(data: DataFrame, tokenizer: LongformerTensorizer):
    # ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def query_encoder(row):
        question = row['question']
        norm_question = normalize_question(question.lower())
        query_res = CLS_TOKEN + SPECIAL_QUERY_TOKEN + SEP_TOKEN + norm_question + SEP_TOKEN  #
        query_encode_ids = tokenizer.text_encode(text=query_res, add_special_tokens=False)  # Query encode
        query_len = len(query_encode_ids)
        return norm_question, query_encode_ids, query_len
    start_time = time()
    data[['norm_question', 'q_encode_ids', 'q_len']] = \
        data.swifter.apply(lambda row: pd.Series(query_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data = {}'.format(data.shape))
    return data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hotpotqa_preprocess_example():
    start_time = time()
    train_data, _ = HOTPOT_TrainData()
    dev_data, _ = HOTPOT_DevData_Distractor()
    test_data, _ = HOTPOT_Test_FullWiki()
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    longformer_tokenizer = LongformerTensorizer(tokenizer=tokenizer, max_length=-1)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dev_data = Hotpot_TrainDev_Data_Preprocess(data=dev_data, tokenizer=longformer_tokenizer)
    print('Get {} dev records'.format(dev_data.shape[0]))
    dev_data.to_json(os.path.join(abs_full_wiki_path, 'hotpot_dev_full_wiki_tokenized.json'))
    train_data = Hotpot_TrainDev_Data_Preprocess(data=train_data, tokenizer=longformer_tokenizer)
    print('Get {} training records'.format(train_data.shape[0]))
    train_data.to_json(os.path.join(abs_full_wiki_path, 'hotpot_train_full_wiki_tokenized.json'))
    test_data = Hotpot_Test_Data_Preprocess(data=test_data, tokenizer=longformer_tokenizer)
    test_data.to_json(os.path.join(abs_full_wiki_path, 'hotpot_test_full_wiki_tokenized.json'))
    print('Get {} testing records'.format(test_data.shape[0]))
    print('Runtime = {:.4f} seconds'.format(time() - start_time))

def hotpotqa_train_len_statistics_example():
    train_data = loadWikiData(PATH=abs_full_wiki_path, json_fileName='hotpot_train_full_wiki_tokenized.json')
    query_lens, pos_ctx_lens, neg_ctx_lens, pos_ctx_sent_nums, neg_ctx_sent_nums = [], [], [], [], []

    for idx, row in train_data.iterrows():
        q_len_i, p_ctx_lens_i, n_ctx_lens_i = row['ques_len'], row['p_ctx_lens'], row['n_ctx_lens']
        pos_ctx, neg_ctx = row['p_ctx'], row['n_ctx_H']
        for p_tex in pos_ctx:
            pos_ctx_sent_nums.append(len(p_tex[1]))
        for n_tex in neg_ctx:
            neg_ctx_sent_nums.append(len(n_tex[1]))
        query_lens.append(q_len_i)
        pos_ctx_lens = pos_ctx_lens + p_ctx_lens_i
        neg_ctx_lens = neg_ctx_lens + n_ctx_lens_i
        if idx % 5000 == 0:
            print('Loading {} records'.format(idx))
    q_lens = np.array(query_lens)
    p_ctx_lens = np.array(pos_ctx_lens)
    n_ctx_lens = np.array(neg_ctx_lens)
    pos_ctx_sent_nums = np.array(pos_ctx_sent_nums)
    neg_ctx_sent_nums = np.array(neg_ctx_sent_nums)

    percitile_array = [95, 97.5, 99.5, 99.75, 99.9, 99.99]
    min_values = ('min', np.min(q_lens), np.min(pos_ctx_lens), np.min(neg_ctx_lens), np.min(pos_ctx_sent_nums), np.min(neg_ctx_sent_nums))
    max_values = ('max', np.max(q_lens), np.max(pos_ctx_lens), np.max(neg_ctx_lens), np.max(pos_ctx_sent_nums), np.max(neg_ctx_sent_nums))
    pc_list = []
    pc_list.append(min_values)
    for pc in percitile_array:
        pc_list.append((pc, np.percentile(q_lens, pc), np.percentile(p_ctx_lens, pc), np.percentile(n_ctx_lens, pc),
                        np.percentile(pos_ctx_sent_nums, pc), np.percentile(neg_ctx_sent_nums, pc)))
    pc_list.append(max_values)

    for idx, v in enumerate(pc_list):
        print('id = {} Split {}\tQuery: {}\tPos_doc: {}\tNeg_doc: {}\tPos_doc_sent: {}\tNeg_doc_sent: {}'.format(idx, v[0], int(v[1]+0.5),
                                                                           int(v[2] + 0.5), int(v[3] + 0.5), int(v[4] + 0.5), int(v[5] + 0.5)))
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def train_data_hardness_analysis():
    train_data, col_names = HOTPOT_TrainData()
    def hard_pos_neg_context_split(row):
        supporting_facts, contexts, answer = row['supporting_facts'], row['context'], row['answer']
        doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
        supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                     if supp_sent_idx < doc_title2doc_len[supp_title]]
        positive_titles = set([x[0] for x in supporting_facts_filtered])
        pos_doc_num = len(positive_titles)
        assert pos_doc_num == 2
        pos_ctxs, neg_ctxs = [], []
        for ctx in contexts:
            title, text = ctx[0], ctx[1]
            text_strip = [sent.strip() for sent in text]
            if title in positive_titles:
                count = 1
                for supp_title, supp_sent_idx in supporting_facts_filtered:
                    if title == supp_title:
                        supp_sent = text[supp_sent_idx]
                        if answer.strip() in supp_sent:
                            count = count + 1
                pos_ctxs.append([title, text_strip, count])  ## Identify the support document with answer
            else:
                neg_ctxs.append([title, text_strip, 1])
        pos_answer_counts = tuple([x[2] for x in pos_ctxs])
        if pos_answer_counts[0] >= 2 and pos_answer_counts[1] >=2:
            single_answer = 0
        else:
            single_answer = 1
        return single_answer

    train_data['single_hop'] = train_data.swifter.progress_bar(True).apply(lambda row: pd.Series(hard_pos_neg_context_split(row)), axis=1)
    for key, sub_data in train_data.groupby(['level']):
        # print(key, sub_data.shape)
        print(key, sub_data['single_hop'].sum()/sub_data.shape[0])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def data_consistent_checker():
    from random import seed
    from random import random
    seed(1)
    train_data = loadWikiData(PATH=abs_full_wiki_path, json_fileName='hotpot_dev_full_wiki_tokenized.json')
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    longformer_tokenizer = LongformerTensorizer(tokenizer=tokenizer, max_length=-1)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ooidx_count = 0
    answer_in_consist_count = 0
    record_with_multi_answer = 0
    for idx, row in train_data.iterrows():
        question, supporting_facts, contexts, answer = row['norm_query'], row['supporting_facts'], row['context'], row['norm_answer']
        supporting_facts_filtered = row['supp_facts_filtered']

        pos_ctx_encode, pos_ctx_lens, neg_ctx_encode, neg_ctx_lens = row['p_ctx_encode'], row['p_ctx_lens'], \
                                                                     row['n_ctx_encode'], row['n_ctx_lens']
        pos_ctx = row['p_ctx']
        if random() > 0.9:
            for doc_idx, doc_tup in enumerate(pos_ctx_encode):
                doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len = doc_tup
                assert doc_len_i == len(doc_encode_ids)
                assert len(sent_start_end_pair) == len(supp_sent_labels)
                doc_title_encode_ids = doc_encode_ids[:p_title_len]
                doc_title = pos_ctx[doc_idx][0]
                doc_title_decode_text = longformer_tokenizer.to_string(doc_title_encode_ids)
                print(doc_title)
                print(doc_title_decode_text)
                print('*'*100)
                # if len(answer_positions) > 0:
                #     for sent_idx, start_idx, end_idx in answer_positions:
                #         sent_pair_idx = sent_start_end_pair[sent_idx]
                #         sent_encode_ids = doc_encode_ids[sent_pair_idx[0]:sent_pair_idx[1]]
                #         answer_encode_ids = sent_encode_ids[start_idx:end_idx]
                #         sent_text = pos_ctx[doc_idx][1][sent_idx]
                #         sent_decode_text = longformer_tokenizer.to_string(sent_encode_ids)
                #         answer_decode_text = longformer_tokenizer.to_string(answer_encode_ids)
                #         print(sent_text)
                #         print(sent_decode_text)
                #         print(answer_decode_text, answer)
                #         print('*' * 100)
                    # if len(answer_positions) > 1:
                    #     print(answer_positions)
                    #     record_with_multi_answer = record_with_multi_answer + 1


                # if ctx_with_answer:
                #     if doc_weight > 2:
                #         print(doc_weight, ctx_with_answer)
                # print(len(supp_sent_labels), len(ctx_with_answer))
                # print(doc_len_i, len(sent_start_end_pair), len(supp_sent_labels))

        # pos_ctx = row['p_ctx']
        # ++++++++++++++++++++
        # answer_encode_ids = row['answer_encode']
        # answer_decode = longformer_tokenizer.to_string(token_ids=answer_encode_ids)
        # if answer_decode != answer:
        #     print(answer_decode)
        #     print(answer)
        #     print('*' * 100)
        #     answer_in_consist_count = answer_in_consist_count + 1
        #++++++++++++++++++++
        # if len(supporting_facts) != len(supporting_facts_filtered):
        #     print(supporting_facts_filtered)
        #     print(supporting_facts)
        #     for p_ctx in pos_ctx:
        #         title, content, _, _ = p_ctx
        #         print(title, len(content))
        #     print('*' * 100)
        #     ooidx_count = ooidx_count + 1
        if idx % 10000 == 0:
            print('{} records have been scanned'.format(idx))

    print('Number of records out of index = {}'.format(ooidx_count))
    print('Number of answer inconsistent count = {}'.format(answer_in_consist_count))
    print('Number of records with multiple answers = {}'.format(record_with_multi_answer))

if __name__ == '__main__':
    # hotpotqa_preprocess_example()
    # hotpotqa_train_len_statistics_example()
    # train_data_hardness_analysis()
    # data_consistent_checker()
    test_data, x = HOTPOT_Test_FullWiki()
    print(x)
    print()