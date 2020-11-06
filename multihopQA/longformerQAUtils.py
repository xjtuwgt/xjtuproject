from transformers import LongformerModel, LongformerTokenizer
from torch import Tensor as T
from torch import nn
import torch
from transformers.configuration_longformer import LongformerConfig
PRE_TAINED_LONFORMER_BASE = 'allenai/longformer-base-4096'

class QATensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """
    def text_to_tensor(self, text: str, add_special_tokens: bool = True):
        raise NotImplementedError

    def text_encode(self, text: str, add_special_tokens: bool = True):
        raise NotImplementedError

    def token_ids_padding(self, token_ids):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def token_ids_to_tensor(self, token_ids):
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, token_ids_tensor: T):
        raise NotImplementedError

    def get_global_attn_mask(self, token_ids_tensor: T, gobal_mask_idxs = None):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

class LongformerQATensorizer(QATensorizer):
    def __init__(self, tokenizer: LongformerTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, add_special_tokens: bool = True):
        text_tokens = self.tokenizer.tokenize(text=text)
        token_ids = self.tokenizer.encode(text_tokens, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id
        return torch.tensor(token_ids)

    def text_encode(self, text: str, add_special_tokens: bool = True):
        text_tokens = self.tokenizer.tokenize(text=text)
        token_ids = self.tokenizer.encode(text_tokens, add_special_tokens=add_special_tokens,
                                              pad_to_max_length=False, truncation=True)
        return token_ids

    def token_ids_padding(self, token_ids):
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id
        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_special_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.unk_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def token_ids_to_tensor(self, token_ids):
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id
        return torch.tensor(token_ids)

    def get_attn_mask(self, token_ids_tensor: T) -> T:
        attention_mask = torch.ones(token_ids_tensor.shape, dtype=torch.long, device=token_ids_tensor.device)
        attention_mask[token_ids_tensor == self.get_pad_id()] = 0
        return attention_mask

    def get_global_attn_mask(self, tokens_ids_tensor: T, gobal_mask_idxs = None) -> T:
        global_attention_mask = torch.zeros(tokens_ids_tensor.shape, dtype=torch.long, device=tokens_ids_tensor.device)
        if gobal_mask_idxs is None:
            first_sep_idx = torch.nonzero((tokens_ids_tensor == self.get_special_separator_ids()), as_tuple=True)[0][0].item()
            global_attention_mask[torch.arange(0, first_sep_idx)] = 1
        else:
            global_attention_mask[gobal_mask_idxs] = 1
        return global_attention_mask

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


class LongformerEncoder(LongformerModel):
    def __init__(self, config, project_dim: int = 0, seq_project=True):
        LongformerModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.seq_project = seq_project
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, attn_dropout: float = 0.1,
                     hidden_dropout: float = 0.1, seq_project=False, **kwargs) -> LongformerModel:
        cfg = LongformerConfig.from_pretrained(cfg_name if cfg_name else PRE_TAINED_LONFORMER_BASE)
        if attn_dropout != 0:
            cfg.attention_probs_dropout_prob = attn_dropout
        if hidden_dropout !=0:
            cfg.hidden_dropout_prob = hidden_dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, seq_project=seq_project, **kwargs)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            attention_mask=attention_mask,
                                                                            global_attention_mask=global_attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             global_attention_mask=global_attention_mask)
        pooled_output = sequence_output[:, 0, :] ### get the first element [CLS], the second is the adding new token
        # print(pooled_output.shape, sequence_output.shape)
        if self.encode_proj:
            if self.seq_project:
                sequence_output = self.encode_proj(sequence_output)
                pooled_output = sequence_output[:, 0, :]
            else:
                pooled_output = self.encode_proj(pooled_output)
        # print(pooled_output.shape, sequence_output.shape)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

# class LongformerHotPotQAModel(LongformerModel):
#     def __init__(self, config, num_labels: int):
#         super().__init__(config)
#         self.num_labels = num_labels
#         self.longformer = LongformerModel(config, add_pooling_layer=False)
#         self.qa_outputs = nn.Linear(config.hidden_size, num_labels)
#         self.yn_outputs = nn.Linear(config.hidden_size, 2) ## span OR yes/no question
#
#         self.init_weights()
#
#     @classmethod
#     def init_encoder(cls, cfg_name: str, num_labels: int, dropout: float = 0.1, **kwargs) -> LongformerModel:
#         cfg = LongformerConfig.from_pretrained(cfg_name if cfg_name else 'allenai/longformer-base-4096')
#         if dropout != 0:
#             cfg.attention_probs_dropout_prob = dropout
#             cfg.hidden_dropout_prob = dropout
#         return cls.from_pretrained(cfg_name, config=cfg, num_label=num_labels, **kwargs)
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             global_attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             inputs_embeds=None,
#             start_positions=None,
#             end_positions=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.longformer(
#             input_ids,
#             attention_mask=attention_mask,
#             global_attention_mask=global_attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#
#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)
#
#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#
#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output
#
#         return QuestionAnsweringModelOutput(
#             loss=total_loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#
#
# # class LongformerEncoder(LongformerModel):
# #     def __init__(self, config, project_dim: int = 0, seq_project=True):
# #         LongformerModel.__init__(self, config)
# #         assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
# #         self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
# #         self.seq_project = seq_project
# #         self.init_weights()
# #
# #     @classmethod
# #     def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, seq_project=False, **kwargs) -> LongformerModel:
# #         cfg = LongformerConfig.from_pretrained(cfg_name if cfg_name else TRIVIAQA_MODEL_NAME_LARGE)
# #         if dropout != 0:
# #             cfg.attention_probs_dropout_prob = dropout
# #             cfg.hidden_dropout_prob = dropout
# #         return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, seq_project=seq_project, **kwargs)
# #
# #     def forward(self,
# #         input_ids=None,
# #         attention_mask=None,
# #         global_attention_mask=None,
# #         token_type_ids=None,
# #         position_ids=None,
# #         inputs_embeds=None,
# #         output_attentions=None,
# #         output_hidden_states=None,
# #         return_dict=None):
# #
# #         if self.config.output_hidden_states:
# #             sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
# #                                                                             attention_mask=attention_mask,
# #                                                                             global_attention_mask=global_attention_mask)
# #         else:
# #             hidden_states = None
# #             sequence_output, pooled_output = super().forward(input_ids=input_ids,
# #                                                              attention_mask=attention_mask,
# #                                                              global_attention_mask=global_attention_mask)
# #         pooled_output = sequence_output[:, 0, :] ### get the first element [CLS], the second is the adding new token
# #         # print(pooled_output.shape, sequence_output.shape)
# #         if self.encode_proj:
# #             if self.seq_project:
# #                 sequence_output = self.encode_proj(sequence_output)
# #                 pooled_output = sequence_output[:, 0, :]
# #             else:
# #                 pooled_output = self.encode_proj(pooled_output)
# #         # print(pooled_output.shape, sequence_output.shape)
# #         return sequence_output, pooled_output, hidden_states
# #
# #     def get_out_size(self):
# #         if self.encode_proj:
# #             return self.encode_proj.out_features
# #         return self.config.hidden_size


if __name__ == '__main__':
    from multihopr.twintowerRetriver import TwinTowerRetriver
    # cfg = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
    # print(cfg.attention_mode)
    # print(cfg.attention_window)
    # print(cfg)
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    print(len(tokenizer))
    # prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    # choice0 = "It is eaten with a fork and a knife."
    # choice1 = "It is eaten while held in the hand."
    # encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
    #
    # print(tokenizer.decode(encoding['input_ids'][1]))