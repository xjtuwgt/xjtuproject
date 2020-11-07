from torch import Tensor as T
from torch import nn
import torch
from multihopQA.longformerQAUtils import LongformerEncoder
from multihopQA.hotpotQAlossUtils import MultiClassFocalLoss, PairwiseCEFocalLoss, TriplePairwiseCEFocalLoss
from multihopQA.Transformer import Transformer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_input, d_mid, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_input, d_mid)
        self.w_2 = nn.Linear(d_mid, d_out)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.w_1.weight.data)
        nn.init.kaiming_uniform_(self.w_2.weight.data)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class BiLinear(nn.Module):
    def __init__(self, project_dim: int, args):
        super(BiLinear, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.bilinear_map = nn.Bilinear(in1_features=project_dim, in2_features=project_dim, out_features=1, bias=False)

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        doc_embed = self.inp_drop(doc_emb)
        scores = self.bilinear_map(doc_embed, q_embed).squeeze(dim=-1)
        return scores

class DotProduct(nn.Module):
    def __init__(self, args, transpose=False):
        super(DotProduct, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.transpose = transpose

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        d_embed = self.inp_drop(doc_emb)
        if not self.transpose:
            scores = (q_embed * d_embed).sum(dim=-1)
        else:
            scores = torch.matmul(q_embed, d_embed.transpose(-1,-2))
        return scores

class CosineSimilar(nn.Module):
    def __init__(self, args, transpose=False):
        super(CosineSimilar, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.transpose = transpose
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        d_embed = self.inp_drop(doc_emb)
        scores = self.cos(q_embed, d_embed)
        return scores
########################################################################################################################
########################################################################################################################
class LongformerHotPotQAModel(nn.Module):
    def __init__(self, longformer: LongformerEncoder, num_labels: int, args, fix_encoder=False):
        super().__init__()
        self.num_labels = num_labels
        self.longformer = longformer
        self.hidden_size = longformer.get_out_size()
        self.yn_outputs = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=3) ## yes, no, span question score
        self.qa_outputs = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=num_labels) ## span prediction score
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.score_model_name = args.model_name ## supp doc score/supp sent score
        self.hop_model_name = args.hop_model_name ## triple score

        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.with_graph = args.with_graph
        if self.with_graph:
            self.transformer_layer = Transformer(d_model=self.hidden_size, heads=args.heads)
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.score_model_name not in ['MLP']:
            raise ValueError('model %s not supported' % self.score_model_name)
        else:
            self.doc_mlp = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) if self.score_model_name == 'MLP' else None
            self.sent_mlp = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) if self.score_model_name == 'MLP' else None

        if self.hop_model_name not in ['DotProduct', 'BiLinear', 'Cosine']:
            self.hop_model_name = None
        else:
            self.hop_doc_dotproduct = DotProduct(args=args) if self.hop_model_name == 'DotProduct' else None
            self.hop_doc_cosine = CosineSimilar(args=args) if self.hop_model_name == 'Cosine' else None
            self.hop_doc_bilinear = BiLinear(args=args, project_dim=self.hidden_size) if self.hop_model_name == 'BiLinear' else None

    @staticmethod
    def get_representation(sub_model: LongformerEncoder, ids: T, attn_mask: T, global_attn_mask: T,
                           fix_encoder: bool = False) -> (
            T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model.forward(input_ids=ids,
                                                                                      attention_mask=attn_mask,
                                                                                      global_attention_mask=global_attn_mask)
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model.forward(input_ids=ids,
                                                                                  attention_mask=attn_mask,
                                                                                  global_attention_mask=global_attn_mask)
        return sequence_output, pooled_output, hidden_states

    def forward(self, sample):
        ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_encode'], sample['ctx_attn_mask'], sample['ctx_global_mask']
        doc_start_positions, doc_end_positions = sample['doc_start'], sample['doc_end']
        sent_start_positions, sent_end_positions = sample['sent_start'], sample['sent_end']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (self.hop_model_name is not None) and self.training:
            head_doc_positions, tail_doc_positions = sample['head_idx'], sample['tail_idx']
            head_tail_pair = (head_doc_positions, tail_doc_positions)
        else:
            head_tail_pair = None
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sequence_output, _, _ = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        yn_scores = self.yes_no_prediction(sequence_output=sequence_output)
        start_logits, end_logits = self.span_prediction(sequence_output=sequence_output)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_logits = start_logits.masked_fill(ctx_attn_mask == 0, -1e11)
        end_logits = end_logits.masked_fill(ctx_attn_mask == 0, -1e11)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.with_graph:
            sent_sent_mask, doc_sent_mask = sample['ss_mask'], sample['sd_mask']
            supp_sent_scores, supp_doc_scores, supp_head_tail_scores = self.supp_doc_sent_prediction(sequence_output=sequence_output,
                                                                                                     doc_start_points=doc_start_positions,
                                                                                                     doc_end_points=doc_end_positions,
                                                                                                     sent_start_points=sent_start_positions,
                                                                                                     sent_end_points=sent_end_positions,
                                                                                                     sent_sent_mask=sent_sent_mask,
                                                                                                     doc_sent_mask=doc_sent_mask,
                                                                                                     head_tail_pair=head_tail_pair)
        else:
            supp_doc_scores, supp_head_tail_scores = self.supp_doc_prediction(sequence_output=sequence_output,
                                                                              start_points=doc_start_positions,
                                                                              end_points=doc_end_positions,
                                                                              head_tail_pair=head_tail_pair)
            supp_sent_scores = self.supp_sent_prediction(sequence_output=sequence_output,
                                                         start_points=sent_start_positions,
                                                         end_points=sent_end_positions)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output = {'yn_score': yn_scores, 'span_score': (start_logits, end_logits), 'doc_score': (supp_doc_scores, supp_head_tail_scores), 'sent_score': supp_sent_scores}
        if self.training:
            loss_res = self.loss_computation(output=output, sample=sample)
            return loss_res
        else:
            return output

    def yes_no_prediction(self, sequence_output: T):
        cls_emb = sequence_output[:, 0, :]
        scores = self.yn_outputs(cls_emb).squeeze(dim=-1)
        return scores

    def span_prediction(self, sequence_output: T):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def supp_doc_sent_prediction(self, sequence_output, doc_start_points, doc_end_points, doc_sent_mask,
                                 sent_start_points, sent_end_points, sent_sent_mask, head_tail_pair=None):
        ##++++++++++++++++++++++++++++++++++++
        batch_size, sent_num = sent_start_points.shape
        batch_idx = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, sent_num).to(sequence_output.device)
        sent_start_embed = sequence_output[batch_idx, sent_start_points]
        sent_end_embed = sequence_output[batch_idx, sent_end_points]
        sent_embed = (sent_start_embed + sent_end_embed) / 2.0
        sent_embed = self.transformer_layer.forward(query=sent_embed, key=sent_embed, value=sent_embed, x_mask=sent_sent_mask)
        #####++++++++++++++++++++
        sent_model_func = {'MLP': self.MLP}
        if self.score_model_name in sent_model_func:
            sent_pair_score = sent_model_func[self.score_model_name](sent_embed, mode='sentence').squeeze(dim=-1)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        ####+++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++
        batch_size, doc_num = doc_start_points.shape
        batch_idx = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, doc_num).to(sequence_output.device)
        doc_start_embed = sequence_output[batch_idx, doc_start_points]
        doc_end_embed = sequence_output[batch_idx, doc_end_points]
        doc_embed = (doc_start_embed + doc_end_embed) / 2.0
        doc_embed = self.transformer_layer.forward(query=doc_embed, key=sent_embed, value=sent_embed, x_mask=doc_sent_mask)
        ##++++++++++++++++++++++++++++++++++++
        #####++++++++++++++++++++
        doc_model_func = {'MLP': self.MLP}
        if self.score_model_name in doc_model_func:
            doc_pair_score = doc_model_func[self.score_model_name](doc_embed, mode='document').squeeze(dim=-1)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        #####++++++++++++++++++++
        #####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        head_tail_score = None
        if head_tail_pair is not None:
            head_position, tail_position = head_tail_pair
            ######++++++++++++++++++++++++++++++++++++++
            query_emb = sequence_output[:, 1, :]
            query_emb = query_emb.unsqueeze(dim=1).repeat([1, doc_num, 1])
            # ######++++++++++++++++++++++++++++++++++++++
            if len(head_position.shape) > 1:
                head_position = head_position.squeeze(dim=-1)
            p_batch_idx = torch.arange(0, batch_size).to(sequence_output.device)
            head_emb = doc_embed[p_batch_idx, head_position].unsqueeze(dim=1).repeat([1, doc_num, 1])
            ###################
            head_emb = head_emb * query_emb
            ###################
            hop_model_func = {'DotProduct': self.Hop_DotProduct, 'BiLinear': self.Hop_BiLinear, 'Cosine': self.Hop_Cosine}
            if self.hop_model_name in hop_model_func:
                head_tail_score = hop_model_func[self.hop_model_name](head_emb, doc_embed).squeeze(dim=-1)
            else:
                raise ValueError('model %s not supported' % self.hop_model_name)
        #####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return sent_pair_score, doc_pair_score, head_tail_score

    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def Hop_DotProduct(self, head_emb: T, tail_emb: T) -> T:
        score = self.hop_doc_dotproduct.forward(head_emb, tail_emb)
        # print('pair score = {}'.format(score))
        return score

    def Hop_BiLinear(self, head_emb: T, tail_emb: T) -> T:
        score = self.hop_doc_bilinear.forward(head_emb, tail_emb)
        return score

    def Hop_Cosine(self, head_emb: T, tail_emb: T) -> T:
        score = self.hop_doc_cosine.forward(head_emb, tail_emb)
        return score
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def MLP(self, ctx_emb: T, mode: str) -> T:
        if mode == 'document':
            query_ctx_emb = ctx_emb
            score = self.doc_mlp.forward(query_ctx_emb)
        elif mode == 'sentence':
            query_ctx_emb = ctx_emb
            score = self.sent_mlp.forward(query_ctx_emb)
        else:
            raise ValueError('model %s not supported' % mode)
        return score

    def score_label_pair(self, output_scores, sample):
        yn_scores = output_scores['yn_score']
        start_logits, end_logits = output_scores['span_score']
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(start_logits.shape, end_logits.shape, sample['ctx_attn_mask'].shape, sample['ctx_attn_mask'].sum(dim=1), sample['doc_lens'].sum(dim=1))
        answer_start_positions, answer_end_positions, yn_labels = sample['ans_start'], sample['ans_end'], sample['yes_no']
        if len(yn_labels.shape) > 0:
            yn_labels = yn_labels.squeeze(dim=-1)
        yn_num = (yn_labels > 0).sum().data.item()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        supp_doc_scores, supp_head_tail_scores = output_scores['doc_score']
        supp_sent_scores = output_scores['sent_score']
        # ******************************************************************************************************************
        # ******************************************************************************************************************
        doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
        sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
        supp_head_position, supp_tail_position = sample['head_idx'], sample['tail_idx']
        doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
        sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if len(answer_start_positions.size()) > 1:
            answer_start_positions = answer_start_positions.squeeze(-1)
        if len(answer_end_positions.size()) > 1:
            answer_end_positions = answer_end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        answer_start_positions.clamp_(0, ignored_index)
        answer_end_positions.clamp_(0, ignored_index)
        if yn_num > 0:
            ans_batch_idx = (yn_labels > 0).nonzero().squeeze()
            start_logits[ans_batch_idx, answer_start_positions[ans_batch_idx]] = 1e11
            end_logits[ans_batch_idx, answer_end_positions[ans_batch_idx]] = 1e11
        # ******************************************************************************************************************
        # ******************************************************************************************************************
        return {'yn': (yn_scores, yn_labels),
                'span': ((start_logits, end_logits), (answer_start_positions, answer_end_positions), ignored_index),
                'doc': (supp_doc_scores, doc_label, doc_mask),
                'doc_pair': (supp_head_tail_scores, supp_head_position, supp_tail_position),
                'sent': (supp_sent_scores, sent_label, sent_mask)}

    def loss_computation(self, output, sample):
        predict_label_pair = self.score_label_pair(output_scores=output, sample=sample)
        ##+++++++++++++
        yn_score, yn_label = predict_label_pair['yn']
        yn_loss_fct = MultiClassFocalLoss(num_class=3)
        yn_loss = yn_loss_fct.forward(yn_score, yn_label)
        ##+++++++++++++
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_doc_scores, doc_label, doc_mask = predict_label_pair['doc']
        supp_doc_loss = supp_loss_fct.forward(scores=supp_doc_scores, targets=doc_label, target_len=doc_mask)
        ##+++++++++++++
        ##+++++++++++++
        supp_pair_doc_scores, head_position, tail_position = predict_label_pair['doc_pair']
        if supp_pair_doc_scores is None:
            supp_doc_pair_loss = torch.tensor(0.0).to(head_position.device)
        else:
            supp_pair_loss_fct = TriplePairwiseCEFocalLoss()
            supp_doc_pair_loss = supp_pair_loss_fct.forward(scores=supp_pair_doc_scores,
                                                            head_position=head_position,
                                                            tail_position=tail_position,
                                                            score_mask=doc_mask)
        ##+++++++++++++
        supp_sent_scores, sent_label, sent_mask = predict_label_pair['sent']
        supp_sent_loss = supp_loss_fct.forward(scores=supp_sent_scores, targets=sent_label, target_len=sent_mask)
        ##+++++++++++++
        span_logits, span_position, ignored_index = predict_label_pair['span']
        start_logits, end_logits = span_logits
        start_positions, end_positions = span_position
        # ++++++++++++++++++++++++++++++++++++++++++++++++
        span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = span_loss_fct(start_logits, start_positions)
        end_loss = span_loss_fct(end_logits, end_positions)
        # ++++++++++++++++++++++++++++++++++++++++++++++++
        span_loss = (start_loss + end_loss) / 2
        return {'yn_loss': yn_loss, 'span_loss': span_loss, 'doc_loss': supp_doc_loss, 'doc_pair_loss': supp_doc_pair_loss,
                'sent_loss': supp_sent_loss}