from torch import nn
import torch
from torch import Tensor as T
import torch.nn.functional as F

class PairBCERetrievalFocalLoss(nn.Module):
    def __init__(self, alpha=1.0,  gamma=2.0, margin=0.000001, reduction='mean'):
        super(PairBCERetrievalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.margin = margin
        self.smooth = 1e-6
        self.reduction = reduction

    def forward(self, scores: T, score_mask=None, pair_or_not=True):
        if pair_or_not:
            batch_size, neg_sample_size = scores.shape[0], scores.shape[1] - 2
            head_scores = scores[:, 0].view(batch_size, 1, 1).repeat([1, neg_sample_size, 1])
            tail_scores = scores[:, 1].view(batch_size, 1, 1).repeat([1, neg_sample_size, 1])
            neg_scores = scores[:, torch.arange(2, scores.shape[1])].unsqueeze(dim=-1)
            head_scores = torch.cat([head_scores, neg_scores], dim=-1)
            tail_scores = torch.cat([tail_scores, neg_scores], dim=-1)
            head_loss, tail_loss = self.focal_loss(scores=head_scores), self.focal_loss(scores=tail_scores)
            pair_loss = head_loss + tail_loss
            if score_mask is not None:
                head_tail_score = scores[:, [0, 1]]
                head_tail_score = F.softmax(head_tail_score, dim=-1)
                ht_loss = F.relu(head_tail_score[:, 1] - head_tail_score[:, 0] - self.margin)
                ht_loss = ht_loss.masked_fill(score_mask.squeeze(dim=-1), 0)
                if self.reduction == 'mean':
                    ht_loss = torch.sum(ht_loss)
                    none_zero_sum = score_mask.shape.numel() - score_mask.sum()
                    if none_zero_sum > 0:
                        ht_loss = ht_loss / none_zero_sum
                else:
                    ht_loss = ht_loss.sum()
            else:
                ht_loss = None
            return pair_loss, ht_loss
        else:
            batch_size, neg_sample_size = scores.shape[0], scores.shape[1] - 1
            triple_scores = scores[:, 0].view(batch_size, 1, 1).repeat([1, neg_sample_size, 1])
            neg_scores = scores[:, torch.arange(1, scores.shape[1])].unsqueeze(dim=-1)
            triple_scores = torch.cat([triple_scores, neg_scores], dim=-1)
            loss = self.focal_loss(scores=triple_scores)
            return loss

    def focal_loss(self, scores: T):
        logpt = F.log_softmax(scores, dim=-1).to(scores.device)
        logpt = logpt[:, :, 0] if len(scores.shape) == 3 else logpt[:, 0]
        pt = torch.exp(logpt).to(scores.device)
        pt = torch.clamp(pt, self.smooth, 1.0 - self.smooth)
        loss = -self.alpha * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

if __name__ == '__main__':
    scores = torch.randn(4, 200)
    print(scores)
    scores[:,[0,1]] = torch.max(scores) + 0.1
    print(scores)

    pbceloss = PairBCERetrievalFocalLoss()
    y = pbceloss.forward(scores=scores)
    print(y)