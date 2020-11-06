from torch import nn
import torch
from torch import Tensor as T
from typing import Tuple
from multihopr.longformerUtils import LongformerEncoder
from multihopr.CoFormer import CoFormerEncoder
from multihopr.hotpotlossUtils import PairBCERetrievalFocalLoss

class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)

    def forward(self, head_emb: T, query_emb: T, tail_emb: T):
        h_embedded = self.inp_drop(head_emb)
        q_embedded = self.inp_drop(query_emb)
        scores = torch.matmul(h_embedded * q_embedded, tail_emb.transpose(-1, -2))
        return scores

class DotProduct(nn.Module):
    def __init__(self, args):
        super(DotProduct, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        d_embed = self.inp_drop(doc_emb)
        scores = (q_embed * d_embed).sum(dim=-1)
        return scores

class BiLinear(nn.Module):
    def __init__(self, args):
        super(BiLinear, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.bilinear_map = nn.Bilinear(in1_features=args.project_dim, in2_features=args.project_dim, out_features=1, bias=False)

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        doc_embed = self.inp_drop(doc_emb)
        scores = self.bilinear_map(doc_embed, q_embed).squeeze(dim=-1)
        return scores

class TwinTowerRetriver(nn.Module):
    def __init__(self, hop_model_name, model_name, query_encoder: LongformerEncoder, document_encoder: LongformerEncoder,
                 args, fix_query_encoder=False, fix_document_encoder=False):
        super(TwinTowerRetriver, self).__init__()
        self.model_name = model_name
        self.hop_model_name = hop_model_name
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.fix_query_encoder = fix_query_encoder
        self.fix_document_encoder = fix_document_encoder
        self.do_co_attn = args.do_co_attn
        assert query_encoder.get_out_size() == document_encoder.get_out_size()
        d_model = query_encoder.get_out_size()
        if self.do_co_attn:
            heads, attn_drop, input_drop = args.heads, args.attn_drop, args.input_drop
            layer_num = args.layers
            self.co_encoder = CoFormerEncoder(num_layers=layer_num, d_model=d_model, heads=heads,
                                            attn_drop=attn_drop, input_drop=input_drop)

        if hop_model_name not in ['DistMult', 'ComplEx']:
            raise ValueError('model %s not supported' % model_name)
        else:
            self.distmult = DistMult(args=args) if hop_model_name == 'DistMult' else None
        if model_name not in ['DotProduct', 'BiLinear']:
            raise ValueError('model %s not supported' % model_name)
        else:
            self.dotproduct = DotProduct(args=args) if model_name == 'DotProduct' else None
            self.bilinear = BiLinear(args=args) if model_name == 'BiLinear' else None

        self.retrieval_loss_fct = PairBCERetrievalFocalLoss(alpha=args.alpha, gamma=args.gamma, margin=args.margin)

    @staticmethod
    def get_representation(sub_model: LongformerEncoder, ids: T, attn_mask: T, global_attn_mask: T, fix_encoder: bool = False) -> (
            T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model.forward(input_ids=ids, attention_mask=attn_mask,
                                                                                  global_attention_mask=global_attn_mask)
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model.forward(input_ids=ids, attention_mask=attn_mask,
                                                                                  global_attention_mask=global_attn_mask)
        return sequence_output, pooled_output, hidden_states

    def encode_question_ctx(self, question_ids: T, question_attn_mask: T, question_global_attn_mask: T,  ctx_ids: T,
                ctx_attn_mask: T, ctx_global_attn_mask: T) -> Tuple[T, T, T, T]:
        q_sequence_out, q_pooled_out, _ = self.get_representation(self.query_encoder, question_ids,
                                                                  question_attn_mask, question_global_attn_mask, self.fix_query_encoder)
        ctx_sequence_out, ctx_pooled_out, _ = self.get_representation(self.document_encoder, ctx_ids,
                                                                        ctx_attn_mask, ctx_global_attn_mask, self.fix_document_encoder)
        return q_sequence_out, q_pooled_out, ctx_sequence_out, ctx_pooled_out


    def pre_trained_encoder(self, sample: dict):
        q_input_ids, q_attn_mask, q_global_attn_mask = sample['query'], sample['query_attn_mask'], sample[
            'query_global_mask']
        ctx_input_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_doc'], sample['ctx_attn_mask'], sample[
            'ctx_global_mask']
        batch_size, sample_size, _ = ctx_input_ids.shape
        ctx_input_ids, ctx_attn_mask, ctx_global_attn_mask = ctx_input_ids.view(batch_size * sample_size, -1), \
                                                             ctx_attn_mask.view(batch_size * sample_size, -1), \
                                                             ctx_global_attn_mask.view(batch_size * sample_size, -1)
        _, q_embed, _, ctx_embed = self.encode_question_ctx(question_ids=q_input_ids,
                                                                              question_attn_mask=q_attn_mask,
                                                                              question_global_attn_mask=q_global_attn_mask,
                                                                              ctx_ids=ctx_input_ids,
                                                                              ctx_attn_mask=ctx_attn_mask,
                                                                              ctx_global_attn_mask=ctx_global_attn_mask)
        ctx_embed = ctx_embed.view(batch_size, sample_size, -1)
        q_embed = q_embed.unsqueeze(dim=1).repeat([1,sample_size,1])
        return q_embed, ctx_embed, sample_size


    def co_attention_encoder(self, sample: dict):
        q_input_ids, q_attn_mask, q_global_attn_mask = sample['query'], sample['query_attn_mask'], sample[
            'query_global_mask']
        ctx_input_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_doc'], sample['ctx_attn_mask'], sample[
            'ctx_global_mask']
        batch_size, sample_size, ctx_len = ctx_input_ids.shape
        _, q_len = q_input_ids.shape
        ctx_input_ids, ctx_attn_mask, ctx_global_attn_mask = ctx_input_ids.view(batch_size * sample_size, -1), \
                                                             ctx_attn_mask.view(batch_size * sample_size, -1), \
                                                             ctx_global_attn_mask.view(batch_size * sample_size, -1)
        q_seq_embed, _, ctx_seq_embed, _ = self.encode_question_ctx(question_ids=q_input_ids,
                                                                              question_attn_mask=q_attn_mask,
                                                                              question_global_attn_mask=q_global_attn_mask,
                                                                              ctx_ids=ctx_input_ids,
                                                                              ctx_attn_mask=ctx_attn_mask,
                                                                              ctx_global_attn_mask=ctx_global_attn_mask)
        q_seq_embed, ctx_seq_emb = self.co_encoder.forward(query=q_seq_embed, ctx=ctx_seq_embed, q_mask=q_attn_mask, ctx_mask=ctx_attn_mask)
        ctx_seq_emb = ctx_seq_emb.view(batch_size, sample_size, ctx_len, -1)
        q_seq_embed = q_seq_embed.view(batch_size, sample_size, q_len, -1)
        q_embed = q_seq_embed[:, :, 0, :]
        ctx_embed = ctx_seq_emb[:, :, 0, :]
        return q_embed, ctx_embed, sample_size

    def forward(self, sample):
        """
        :param sample:
        :param train_mode: 0 --> train, 1--> valid, 2 --> test
        :return:
        """
        if self.do_co_attn:
            q_embed, ctx_embed, sample_size = self.co_attention_encoder(sample=sample)
        else:
            q_embed, ctx_embed, sample_size = self.pre_trained_encoder(sample=sample)
        ####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pair_score = self.pair_score_computation(q_embed=q_embed, ctx_embed=ctx_embed)
        # print(pair_score.shape)
        if self.training:
            pair_loss, ht_loss = self.pair_loss_computation(pair_score=pair_score, sample=sample)
            triple_score = self.triple_score_computation(q_embed=q_embed, ctx_embed=ctx_embed, sample_size=sample_size, mode=sample['mode'])
            # print(triple_score.shape)
            triple_loss = self.triple_loss_computation(triple_score=triple_score)
            return pair_loss, ht_loss, triple_loss
        else:
            return pair_score

    def DistMult(self, head_ctx_emb: T, q_emb: T, tail_ctx_emb: T) -> T:
        score = self.distmult.forward(head_emb=head_ctx_emb, query_emb=q_emb, tail_emb=tail_ctx_emb)
        return score

    def DotProduct(self, q_emb: T, ctx_emb: T) -> T:
        score = self.dotproduct.forward(query_emb=q_emb, doc_emb=ctx_emb)
        return score

    def BiLinear(self, q_emb: T, ctx_emb: T) -> T:
        score = self.bilinear.forward(query_emb=q_emb, doc_emb=ctx_emb)
        return score

    def pair_score_computation(self, q_embed: T, ctx_embed: T):
        model_func = {'DotProduct': self.DotProduct, 'BiLinear': self.BiLinear}
        if self.model_name in model_func:
            pair_score = model_func[self.model_name](q_embed, ctx_embed)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        pair_score = pair_score.squeeze(dim=1)
        return pair_score

    def triple_score_computation(self, q_embed: T, ctx_embed: T, sample_size: int, mode: str):
        if mode == 'head-batch':
            head_ctx_emb = ctx_embed[:, 0, :].unsqueeze(dim=1)
            trip_q_emb = q_embed[:, 0, :].unsqueeze(dim=1)
            tail_idxs = torch.arange(start=1, end=sample_size, step=1)
            tail_ctx_emb = ctx_embed[:, tail_idxs, :]
        elif mode == 'tail-batch':
            head_ctx_emb = ctx_embed[:, 1, :].unsqueeze(dim=1)
            trip_q_emb = q_embed[:, 1, :].unsqueeze(dim=1)
            tail_idxs = [0] + torch.arange(start=2, end=sample_size, step=1).tolist()
            tail_ctx_emb = ctx_embed[:, tail_idxs, :]
        else:
            raise ValueError('mode %s not supported' % mode)
        hop_model_func = {'DistMult': self.DistMult}
        if self.hop_model_name in hop_model_func:
            triple_score = hop_model_func[self.hop_model_name](head_ctx_emb, trip_q_emb, tail_ctx_emb)
        else:
            raise ValueError('model %s not supported' % self.hop_model_name)
        triple_score = triple_score.squeeze(dim=1)
        return triple_score

    def pair_loss_computation(self, pair_score: T, sample):
        class_label, order_mask = sample['class'], sample['order_mask']
        pair_loss, ht_loss = self.retrieval_loss_fct.forward(scores=pair_score, score_mask=order_mask, pair_or_not=True)
        return pair_loss, ht_loss

    def triple_loss_computation(self, triple_score):
        triple_loss = self.retrieval_loss_fct.forward(scores=triple_score, pair_or_not=False)
        return triple_loss
