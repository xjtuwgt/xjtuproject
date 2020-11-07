import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopr.hotpotqaIOUtils import *
distractor_wiki_path = '../data/hotpotqa/distractor_qa'
abs_distractor_wiki_path = os.path.abspath(distractor_wiki_path)
from pandas import DataFrame
from time import time
from multihopQA.longformerQAUtils import LongformerQATensorizer
from multihopQA.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
from transformers import LongformerTokenizer
import itertools
import operator
import re
import numpy as np
import swifter
SPECIAL_QUERY_TOKEN = '<unk>'
SPECIAL_DOCUMENT_TOKEN = '<unk>'
START_SENTENCE_TOKEN = '<s>'
CLS_TOKEN = '<s>'
END_SENTENCE_TOKEN = '</s>'
SEP_TOKEN = '</s>'

def Hotpot_Train_Data_Preprocess(data: DataFrame, tokenizer: LongformerQATensorizer):
    # ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
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
                                     if supp_sent_idx < doc_title2doc_len[supp_title]] ##some supporting facts are out of sentence index
        positive_titles = set([x[0] for x in supporting_facts_filtered]) ## get postive document titles
        norm_answer = normalize_text(text=answer) ## normalize the answer (add a space between the answer)
        norm_question = normalize_question(question.lower()) ## normalize the question by removing the question mark
        not_found_flag = False ## some answer might be not founded in supporting sentence
        ################################################################################################################
        pos_doc_num = len(positive_titles) ## number of positive documents
        pos_ctxs, neg_ctxs = [], []
        for ctx_idx, ctx in enumerate(contexts): ## Original ctx index, record the original index order
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
                pos_ctxs.append([title.lower(), text_lower, count, supp_sent_flags, ctx_idx])  ## Identify the support document with answer
            else:
                neg_ctxs.append([title.lower(), text_lower, 0, [], ctx_idx])
        neg_doc_num = len(neg_ctxs)
        pos_counts = [x[2] for x in pos_ctxs]
        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
            if sum(pos_counts) == 2:
                not_found_flag = True
        assert len(pos_counts) == 2
        if not_found_flag:
            norm_answer = 'noanswer'
        if (pos_counts[0] > 1 and pos_counts[1] > 1) or (pos_counts[0] <= 1 and pos_counts[1] <= 1):
            answer_type = False
        else:
            answer_type = True
        yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
        return norm_question, norm_answer, pos_ctxs, neg_ctxs, supporting_facts_filtered, answer_type, pos_doc_num, neg_doc_num, yes_no_flag, not_found_flag
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['norm_query', 'norm_answer', 'p_ctx', 'n_ctx', 'supp_facts_filtered', 'answer_type', 'p_doc_num',
          'n_doc_num', 'yes_no', 'no_found']] = data.swifter.apply(lambda row: pd.Series(pos_neg_context_split(row)), axis=1)
    not_found_num = data[data['no_found']].shape[0]
    print('Splitting positive samples from negative samples takes {:.4f} seconds, answer not found = {}'.format(time() - start_time, not_found_num))
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, pos_ctxs, neg_ctxs, norm_answer = row['norm_query'], row['p_ctx'], row['n_ctx'], row['norm_answer']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_res = CLS_TOKEN + SPECIAL_QUERY_TOKEN + SEP_TOKEN + norm_question + SEP_TOKEN #
        query_encode_ids = tokenizer.text_encode(text=query_res, add_special_tokens=False) # Query encode
        query_len = len(query_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if norm_answer.strip() in ['yes', 'no', 'noanswer']:
            answer_encode_ids = [-1]
        else:
            answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
        answer_len = len(answer_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def document_encoder(title: str, doc_sents: list):
            title_res = SPECIAL_DOCUMENT_TOKEN + SEP_TOKEN + title + SEP_TOKEN ##++++++++ Different from full setting
            title_encode_ids = tokenizer.text_encode(text=title_res, add_special_tokens=False)
            title_len = len(title_encode_ids)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            encode_id_lens = []
            encode_id_lens.append(title_len)
            doc_encode_id_list = []
            doc_encode_id_list.append(title_encode_ids)
            doc_len = len(doc_sents)
            for sent_idx, sent_text in enumerate(doc_sents):
                if sent_idx == doc_len - 1:
                    sent_text_res = START_SENTENCE_TOKEN + sent_text + END_SENTENCE_TOKEN + SEP_TOKEN
                else:
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
            p_title, p_doc, p_doc_weight, supp_sent_flags, _ = content
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
                    supp_sent_encode_ids = p_doc_encode_ids[start_id:(end_id+1)]
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
            n_title, n_doc, n_doc_weight, _, _ = content
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
    data[['ques_encode', 'ques_len', 'answer_encode', 'answer_len', 'p_ctx_encode', 'p_ctx_lens', 'pc_max_len',
          'n_ctx_encode', 'n_ctx_lens', 'nc_max_len']] = \
        data.swifter.progress_bar(True).apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data = {}'.format(data.shape))
    return data

#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Hotpot_Dev_Test_Data_PreProcess(data: DataFrame, tokenizer: LongformerQATensorizer):
    # ['question', 'context', '_id']
    """
    Supporting_facts: pair of (title, sentence index) --> (str, int)
    Level: {easy, medium, hard}
    Question: query --> str
    Context: list of pair (title, text) --> (str, list(str))
    Answer: str
    Type: {comparison, bridge}
    """
    def norm_context(row):
        question, contexts = row['question'], row['context']
        norm_question = normalize_question(question.lower())
        hotpot_ctxs = []
        ################################################################################################################
        for ctx_idx, ctx in enumerate(contexts):  ## Original ctx index
            title, text = ctx[0], ctx[1]
            text_lower = [normalize_text(text=sent) for sent in text]
            hotpot_ctxs.append([title.lower(), text_lower, ctx_idx])
        return norm_question, hotpot_ctxs

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['norm_query', 'norm_ctx']] = data.swifter.apply(lambda row: pd.Series(norm_context(row)), axis=1)
    print('Normalizing samples takes {:.4f} seconds'.format(time() - start_time))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, norm_ctxs = row['norm_query'], row['norm_ctx']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_res = CLS_TOKEN + SPECIAL_QUERY_TOKEN + SEP_TOKEN + norm_question + SEP_TOKEN  #
        query_encode_ids = tokenizer.text_encode(text=query_res, add_special_tokens=False)  # Query encode
        query_len = len(query_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def document_encoder(title: str, doc_sents: list):
            title_res = SPECIAL_DOCUMENT_TOKEN + SEP_TOKEN + title + SEP_TOKEN  ##++++++++ Different from full setting
            title_encode_ids = tokenizer.text_encode(text=title_res, add_special_tokens=False)
            title_len = len(title_encode_ids)
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            encode_id_lens = []
            encode_id_lens.append(title_len)
            doc_encode_id_list = []
            doc_encode_id_list.append(title_encode_ids)
            doc_len = len(doc_sents)
            for sent_idx, sent_text in enumerate(doc_sents):
                if sent_idx == doc_len - 1:
                    sent_text_res = START_SENTENCE_TOKEN + sent_text + END_SENTENCE_TOKEN + SEP_TOKEN
                else:
                    sent_text_res = START_SENTENCE_TOKEN + sent_text + END_SENTENCE_TOKEN
                sent_encode_ids = tokenizer.text_encode(text=sent_text_res, add_special_tokens=False)
                doc_encode_id_list.append(sent_encode_ids)
                sent_len = len(sent_encode_ids)
                encode_id_lens.append(sent_len)
            doc_sent_len_cum_list = list(itertools.accumulate(encode_id_lens, operator.add))
            sent_start_end_pair = [(doc_sent_len_cum_list[i], doc_sent_len_cum_list[i + 1] - 1) for i in
                                   range(len(encode_id_lens) - 1)]
            doc_encode_ids = list(itertools.chain.from_iterable(doc_encode_id_list))
            assert len(doc_encode_ids) == doc_sent_len_cum_list[-1]
            return doc_encode_ids, sent_start_end_pair, len(doc_encode_ids), title_len
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_encode_ids = []
        max_doc_len = 0
        ctx_lens = []
        for ctx_idx, content in enumerate(norm_ctxs):
            title, doc_sents, _ = content
            doc_encode_ids, sent_start_end_pair, doc_len_i, title_len = document_encoder(title=title, doc_sents=doc_sents)
            # #########################
            if max_doc_len < doc_len_i:
                max_doc_len = doc_len_i
            #########################
            assert len(doc_sents) == len(sent_start_end_pair)
            ctx_lens.append(doc_len_i)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_tuple = (doc_encode_ids, doc_len_i, sent_start_end_pair, title_len, ctx_idx)
            ctx_encode_ids.append(ctx_tuple)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return query_encode_ids, query_len, ctx_encode_ids, ctx_lens, max_doc_len
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['ques_encode', 'ques_len', 'ctx_encode', 'ctx_lens', 'ctx_max_len']] = \
        data.swifter.progress_bar(True).apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data = {}'.format(data.shape))
    return data
#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question

def normalize_text(text: str) -> str:
    text = ' ' + text.lower().strip() ###adding the ' ' is important to make the consist encoder
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
########################################################################################################################
def hotpotqa_preprocess_example():
    start_time = time()
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    longformer_tokenizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=-1)
    dev_data, _ = HOTPOT_DevData_Distractor()
    dev_test_data = Hotpot_Dev_Test_Data_PreProcess(data=dev_data, tokenizer=longformer_tokenizer)
    print('Get {} dev-test records'.format(dev_test_data.shape[0]))
    dev_test_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_test_distractor_wiki_tokenized.json'))
    print('*' * 75)
    dev_data, _ = HOTPOT_DevData_Distractor()
    dev_data = Hotpot_Dev_Test_Data_PreProcess(data=dev_data, tokenizer=longformer_tokenizer)
    print('Get {} dev records'.format(dev_data.shape[0]))
    dev_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_dev_distractor_wiki_tokenized.json'))
    print('*' * 75)
    train_data, _ = HOTPOT_TrainData()
    train_data = Hotpot_Train_Data_Preprocess(data=train_data, tokenizer=longformer_tokenizer)
    print('Get {} training records'.format(train_data.shape[0]))
    train_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_train_distractor_wiki_tokenized.json'))
    print('Runtime = {:.4f} seconds'.format(time() - start_time))
    print('*' * 75)


def hotpot_data_analysis():
    # data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_dev_distractor_wiki_tokenized.json')
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    col_list = []
    for col in data_frame.columns:
        col_list.append(col)
    print(col_list)
    max_len = 4096
    padding_num = 0.0
    max_doc_len = 0
    for idx, row in data_frame.iterrows():
        p_ctx_len, n_ctx_len = row['p_ctx_lens'], row['n_ctx_lens']
        q_len = row['ques_len']
        ctx_lens = p_ctx_len + n_ctx_len
        ctx_len_sum = sum(ctx_lens) + q_len
        if max_doc_len < ctx_len_sum:
            max_doc_len = ctx_len_sum
        if ctx_len_sum < max_len:
            padding_num = padding_num + 1
        else:
            print(ctx_len_sum)
    print(padding_num/data_frame.shape[0], padding_num, data_frame.shape[0])
    print(max_doc_len)

def num_of_support_sents():
    # data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_dev_distractor_wiki_tokenized.json')
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    supp_sent_num_list = []
    supp_sent_array = np.zeros(100)
    for idx, row in data_frame.iterrows():
        supp_facts = row['supp_facts_filtered']
        supp_sent_num_list.append(len(supp_facts))
        supp_sent_array[len(supp_facts)] = supp_sent_array[len(supp_facts)] + 1

    percitile_array = [95, 97.5, 99.5, 99.75, 99.9, 99.99]
    supp_sent_num = np.array(supp_sent_num_list)
    min_values = ('min', np.min(supp_sent_num))
    max_values = ('max', np.max(supp_sent_num))
    pc_list = []
    pc_list.append(min_values)
    for pc in percitile_array:
        pc_list.append((pc, np.percentile(supp_sent_num, pc)))
    pc_list.append(max_values)
    print(pc_list)
    for i in range(100):
        if supp_sent_array[i] > 0:
            print('{}\t{}'.format(i, supp_sent_array[i]))


def answer_type_analysis():
    # data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_dev_distractor_wiki_tokenized.json')
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    yes_num = 0
    no_num = 0
    no_ans_num = 0
    span_num = 0
    for idx, row in data_frame.iterrows():
        answer = row['norm_answer']
        if answer.strip() == 'yes':
            yes_num = yes_num + 1
        elif answer.strip() == 'no':
            no_num = no_num + 1
        elif answer.strip() == 'noanswer':
            no_ans_num = no_ans_num + 1
        else:
            span_num = span_num + 1
    print('yes\t{}\nno\t{}\n no_ans\t{}\n span_num\t{}'.format(yes_num, no_num, no_ans_num, span_num))

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def data_consistent_checker():
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    def supp_fact_check(row):
        support_facts, filtered_support_facts = row['supporting_facts'], row['supp_facts_filtered']
        for x in support_facts:
            print('supp {}'.format(x))

        for x in filtered_support_facts:
            print('filtered supp {}'.format(x))

    def answer_check(row):
        answer_encode_id = row['answer_encode']
        answer_norm = row['norm_answer']
        orig_answer = row['answer']
        print('Decode = {}\nnorm = {}\norig = {}'.format(tokenizer.decode(answer_encode_id, skip_special_tokens=True), answer_norm, orig_answer))

    def support_sentence_checker(row):
        filtered_support_facts = row['supp_facts_filtered']
        for x in filtered_support_facts:
            print(x)
        print('=' * 100)
        p_ctx = row['p_ctx']
        for idx, context in enumerate(p_ctx):
            print(context[0])
            print('context={}\nnum sents={}'.format(context[1], len(context[1])))
            print('*'*75)
        print('+'*100)
        p_ctx_encode = row['p_ctx_encode']
        # print(len(p_ctx_encode), len(p_ctx))
        for idx, context in enumerate(p_ctx_encode):
            p_doc_encode_ids, p_doc_weight, p_doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len = context
            print('encode {}\nwith len {}\nstore len {}'.format(p_doc_encode_ids, len(p_doc_encode_ids), p_doc_len_i))
            print('sent pair = {}\nnum sents ={}'.format(sent_start_end_pair, len(sent_start_end_pair)))
            print('sent labels = {}'.format(supp_sent_labels))
            print('context len = {}'.format(len(context)))
            print('context with answer = {}'.format(ctx_with_answer))
            print('title = {}'.format(tokenizer.decode(p_doc_encode_ids[:p_title_len], skip_special_tokens=True)))
            print('answer position = {}'.format(answer_positions))
            if len(answer_positions) > 0:
                sent_start, sent_end = sent_start_end_pair[answer_positions[0][0]]
                support_sentence = tokenizer.decode(p_doc_encode_ids[sent_start:(sent_end + 1)], skip_special_tokens=True)
                print('sentence idx={}, Decode sentence = {}'.format(answer_positions[0][0], support_sentence))
                sentence_ids = p_doc_encode_ids[sent_start:(sent_end + 1)]
                decode_answer = tokenizer.decode(sentence_ids[answer_positions[0][1]:(answer_positions[0][2]+1)], skip_special_tokens=True)
                print('decode answer = {}, orig answer = {}'.format(decode_answer, row['norm_answer']))
            print(context[1])
            print('*'*75)
        print('+' * 100)

        print('p_ctx_lens', row['p_ctx_lens'])

    def doc_order_checker(row):
        pos_docs = row['p']

    '''
    _id, answer, question, supporting_facts, context, type, level, norm_query, norm_answer, p_ctx, n_ctx, supp_facts_filtered,
    answer_type, p_doc_num, n_doc_num, yes_no, no_found, ques_encode, ques_len, answer_encode, answer_len, p_ctx_encode,
    p_ctx_lens, pc_max_len, n_ctx_encode, n_ctx_lens, nc_max_len
    :return:
    '''
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path,
                              json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    print('Data frame size = {}'.format(data_frame.shape))
    record_num = data_frame.shape[0]
    row_num = 2
    random_idx = np.random.choice(record_num, row_num, replace=False)
    for idx in range(row_num):
        row_i = data_frame.loc[random_idx[idx], :]
        # supp_fact_check(row=row_i)
        # answer_check(row=row_i)
        support_sentence_checker(row=row_i)
        print('$' * 90)

def data_statistic():
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path,
                              json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    supp_sent_num = 0
    cand_2_sent_num = 0
    max_2_sent_num = 0
    cand_10_sent_num = 0
    max_10_sent_num = 0
    count = 0
    for idx, row in data_frame.iterrows():
        support_sents = row['supp_facts_filtered']
        sent_num_i = len(support_sents)
        p_ctxs, n_ctxs = row['p_ctx'], row['n_ctx']
        p_sent_num_i = sum([len(_[1]) for _ in p_ctxs])
        n_sent_num_i = sum([len(_[1]) for _ in n_ctxs])
        supp_sent_num = supp_sent_num + sent_num_i
        cand_2_sent_num = cand_2_sent_num + p_sent_num_i
        if max_2_sent_num < p_sent_num_i:
            max_2_sent_num = p_sent_num_i
        cand_10_sent_num = cand_10_sent_num + p_sent_num_i + n_sent_num_i
        if max_10_sent_num < (p_sent_num_i + n_sent_num_i):
            max_10_sent_num = p_sent_num_i + n_sent_num_i
        count = count + 1

    print(supp_sent_num/count, cand_2_sent_num/count, cand_10_sent_num/count)
    print(max_2_sent_num)
    print(max_10_sent_num)

if __name__ == '__main__':
    hotpotqa_preprocess_example()
    # x = '100 bc  ad 400'
    # y = remove_multi_spaces(x)
    # print(len(x))
    # print(len(y))
    # hotpot_data_analysis()
    # num_of_support_sents()
    # answer_type()
    # data_consistent_checker()
    # data_statistic()
    print()