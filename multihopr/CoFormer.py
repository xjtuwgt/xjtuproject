from torch import nn
import torch
import copy
from torch import Tensor as T
import torch.nn.functional as F
import math
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadCoAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, attn_drop: float = 0.1):
        super(MultiHeadCoAttention, self).__init__()
        self.d_model = d_model
        self.head_num = heads
        assert self.d_model % self.head_num == 0
        self.d_k = self.d_model // self.head_num
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.linears = clones(nn.Linear(self.d_model, self.d_model), 6)
        self.init()

    def init(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)

    def forward(self, query: T, context: T, q_attn_mask: T = None, ctx_attn_mask: T = None) -> (T, T):
        if q_attn_mask is not None:
            q_attn_mask = q_attn_mask.unsqueeze(dim=1)
        if ctx_attn_mask is not None:
            ctx_attn_mask = ctx_attn_mask.unsqueeze(dim=1)

        batch_size = query.shape[0]
        query, context, query_value, context_value = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
                                                      for l, x in zip(self.linears, (query, context, query, context))]

        q_res, _, ctx_res, _ = coattention(query=query, context=context, query_value=query_value, context_value=context_value,
                                             q_attn_mask=q_attn_mask, ctx_attn_mask=ctx_attn_mask, dropout=self.attn_dropout)

        q_res = q_res.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)
        ctx_res = ctx_res.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)
        q_res = self.linears[4](q_res)
        ctx_res = self.linears[5](ctx_res)
        return q_res, ctx_res


def coattention(query: T, context: T, query_value: T, context_value: T, q_attn_mask: T = None,
                    ctx_attn_mask: T = None, dropout=None):
    d_k = query.shape[-1]
    scores = torch.matmul(query, context.transpose(-2, -1))/math.sqrt(d_k)
    if ctx_attn_mask:
        q_scores = scores.masked_fill(ctx_attn_mask==0, -1e9)
    else:
        q_scores = scores
    if q_attn_mask:
        ctx_scores = scores.transpose(-1,-2).masked_fill(q_attn_mask==0, -1e9)
    else:
        ctx_scores = scores.transpose(-1,-2)

    q_attn_p = F.softmax(q_scores, dim=-1)
    ctx_attn_p = F.softmax(ctx_scores, dim=-1)
    if dropout:
        q_attn_p = dropout(q_attn_p)
        ctx_attn_p = dropout(ctx_attn_p)
    q_res = torch.matmul(q_attn_p, context_value)
    ctx_res = torch.matmul(ctx_attn_p, query_value)
    return q_res, q_attn_p, ctx_res, ctx_attn_p

######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class CoFormer(nn.Module):
    "Encoder is made up of co-attn and feed forward (defined below)"
    def __init__(self, d_model: int, heads: int, attn_drop: float = 0.1, input_drop: float = 0.1):
        super(CoFormer, self).__init__()
        self.co_attn = MultiHeadCoAttention(d_model=d_model, heads=heads, attn_drop=attn_drop)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=input_drop)
        self.co_attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(input_drop)


    def forward(self, x: T, y: T, x_mask: T, y_mask: T):
        "Follow Figure 1 (left) for connections."
        x_res, y_res = self.co_attn.forward(query=x, context=y, q_attn_mask=x_mask, ctx_attn_mask=y_mask)
        x_res = x_res + self.dropout(self.co_attn_norm(x_res))
        y_res = y_res + self.dropout(self.co_attn_norm(y_res))

        x_res = x_res + self.dropout(self.ff_norm(self.feed_forward(x_res)))
        y_res = y_res + self.dropout(self.ff_norm(self.feed_forward(y_res)))
        return x_res, y_res


class CoFormerEncoder(nn.Module):
    def __init__(self, num_layers:int, d_model: int, heads: int, attn_drop: float = 0.1, input_drop: float = 0.1):
        super(CoFormerEncoder, self).__init__()
        self.coformer_models = nn.ModuleList()
        for l in range(0, num_layers):
            self.coformer_models.append(CoFormer(d_model=d_model, heads=heads, attn_drop=attn_drop, input_drop=input_drop))

    def forward(self, query: T, ctx: T, q_mask: T, ctx_mask: T):
        h_q, h_ctx = query, ctx
        for coformer in self.coformer_models:
            h_q, h_ctx = coformer(h_q, h_ctx, q_mask, ctx_mask)
        return h_q, h_ctx