from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.luke import LukeModel, LukePreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class EntitySpanClusteringOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class FeedForwardLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dense = nn.Linear(input_size, hidden_size)
        self.output_dense = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden_dense(x)
        h = self.activation(h)
        h = self.dropout(h)
        y = self.output_dense(h)
        return y


class LukeForEntitySpanClustering(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)
        classifier_hidden_size = getattr(config, "classifier_hidden_size", None)
        if classifier_hidden_size is None:
            classifier_hidden_size = config.hidden_size
        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = config.hidden_dropout_prob
        self.scorer = FeedForwardLayer(
            config.hidden_size * 3 * 3, classifier_hidden_size, 1, classifier_dropout
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.LongTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        entity_start_positions: Optional[torch.LongTensor] = None,
        entity_end_positions: Optional[torch.LongTensor] = None,
        num_sequences: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EntitySpanClusteringOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert entity_ids is not None
        assert entity_attention_mask is not None
        num_mentions = (entity_attention_mask.view(len(entity_ids), -1) == 1).sum(-1).tolist()

        def _unfold(x):
            return torch.cat([v[:n] for v, n in zip(x, num_sequences)]) if x is not None else x

        input_ids = _unfold(input_ids)
        attention_mask = _unfold(attention_mask)
        token_type_ids = _unfold(token_type_ids)
        position_ids = _unfold(position_ids)
        entity_ids = _unfold(entity_ids)
        entity_attention_mask = _unfold(entity_attention_mask)
        entity_token_type_ids = _unfold(entity_token_type_ids)
        entity_position_ids = _unfold(entity_position_ids)
        entity_start_positions = _unfold(entity_start_positions)
        entity_end_positions = _unfold(entity_end_positions)
        head_mask = _unfold(head_mask)
        inputs_embeds = _unfold(inputs_embeds)

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_size = outputs.last_hidden_state.size(-1)

        assert entity_start_positions is not None
        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        if entity_start_positions.device != outputs.last_hidden_state.device:
            entity_start_positions = entity_start_positions.to(outputs.last_hidden_state.device)
        start_states = torch.gather(outputs.last_hidden_state, -2, entity_start_positions)

        assert entity_end_positions is not None
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        if entity_end_positions.device != outputs.last_hidden_state.device:
            entity_end_positions = entity_end_positions.to(outputs.last_hidden_state.device)
        end_states = torch.gather(outputs.last_hidden_state, -2, entity_end_positions)

        entity_vectors = torch.cat(
            [start_states, end_states, outputs.entity_last_hidden_state], dim=2
        )
        entity_vectors = entity_vectors[torch.nonzero(entity_attention_mask, as_tuple=True)]

        m = len(entity_vectors)
        left = torch.arange(m).expand(m, m).T.ravel()
        right = torch.arange(m).expand(m, m).ravel()
        left_vectors = entity_vectors[left]
        right_vectors = entity_vectors[right]
        feature_vectors = torch.cat(
            [left_vectors, right_vectors, left_vectors * right_vectors], dim=1
        )

        mask = torch.zeros(m, m, dtype=torch.bool)
        offset = 0
        for n in num_mentions:
            for k in range(n):
                mask[offset + k, offset : offset + k] = 1
            offset += n

        scores = self.scorer(feature_vectors).view(m, m)
        scores = scores.masked_fill(torch.logical_not(mask.to(scores.device)), float("-inf"))
        scores = scores.masked_fill(torch.eye(m, dtype=torch.bool, device=scores.device), 0)

        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(scores, dim=-1)
            target = torch.zeros(m, m, dtype=torch.bool)
            ignore_index = -100

            mention_idx = -1
            offset = 0
            for n, multi_labels in zip(num_mentions, labels.cpu().numpy()):
                for label in multi_labels[:n]:
                    mention_idx += 1
                    for idx in label:
                        if idx == ignore_index:
                            break
                        target[mention_idx, offset + idx] = 1
                offset += n
            assert mention_idx == m - 1

            loss = -log_probs.masked_select(target.to(log_probs.device)).sum() / m

        shape = (len(num_mentions), max(num_mentions), max(num_mentions))
        logits = torch.full(shape, float("-inf"), dtype=scores.dtype)
        offset = 0
        for i, n in enumerate(num_mentions):
            logits[i, :n, :n] = scores[offset : offset + n, offset : offset + n]
            offset += n
        logits = logits.to(scores.device)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    outputs.hidden_states,
                    outputs.entity_hidden_states,
                    outputs.attentions,
                ]
                if v is not None
            )

        return EntitySpanClusteringOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )
