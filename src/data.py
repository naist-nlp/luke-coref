import logging
from typing import Any, Dict, List, Optional, TypedDict

import torch
from transformers import (
    BatchEncoding,
    DataCollatorWithPadding,
    LukeTokenizer,
    MLukeTokenizer,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


class Entity(TypedDict):
    start: int
    end: int
    entity_id: str


class Example(TypedDict):
    text: str
    entities: List[Entity]


class Preprocessor:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, max_document_length: Optional[int] = None
    ):
        if not isinstance(tokenizer, (LukeTokenizer, MLukeTokenizer)):
            raise RuntimeError(
                "Only `LukeTokenizer` and `MLukeTokenizer` are currently supported,"
                f" but got `{type(tokenizer).__name__}`."
            )

        self.tokenizer = tokenizer
        self.tokenizer.task = "entity_span_classification"
        self.max_document_length = max_document_length

    def __call__(self, document: List[Example]) -> Dict[str, Any]:
        if self.max_document_length is not None and len(document) > self.max_document_length:
            logger.warning(
                f"document (len={len(document)}) is truncated to {self.max_document_length} sequences."
            )
            document = document[: self.max_document_length]

        batch_encoding: Dict[str, List[Any]] = {}
        antecedents: List[List[int]] = []

        clusters: Dict[str, List[int]] = {}
        mention_idx = -1
        for example in document:
            text = example["text"]
            entities = []
            entity_spans = []

            for ent in example["entities"]:
                mention_idx += 1
                entities.append(text[ent["start"] : ent["end"]])
                entity_spans.append((ent["start"], ent["end"]))

                entity_id = ent["entity_id"]
                assert entity_id
                if entity_id not in clusters:
                    clusters[entity_id] = []
                antecedents.append(clusters[entity_id].copy())
                clusters[entity_id].append(mention_idx)

            self.tokenizer.task = None
            sequence = self.tokenizer._create_input_sequence(
                text, entities=entities, entity_spans=entity_spans
            )
            self.tokenizer.task = "entity_span_classification"
            encoding = self.tokenizer.prepare_for_model(*sequence)

            for k, v in encoding.items():
                if k not in batch_encoding:
                    batch_encoding[k] = []
                batch_encoding[k].append(v)

        output = BatchEncoding(batch_encoding)
        # insert self-reference as a dummy antecedent, meaning no antecedent
        output["labels"] = [labels if labels else [idx] for idx, labels in enumerate(antecedents)]
        output["num_sequences"] = len(batch_encoding["input_ids"])

        return output


class Collator(DataCollatorWithPadding):
    max_label_size = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.return_tensors != "pt":
            raise RuntimeError(f"return_tensors='{self.return_tensors}' is not supported.")

        batch_features: List[Dict[str, Any]] = []
        batch_labels: List[List[List[int]]] = []  # {documents} x {mentions} x {multi-label}
        num_sequences: List[int] = []

        for f in features:
            f = f.copy()
            n = f.pop("num_sequences")
            num_sequences.append(n)
            batch_labels.append(f.pop("labels"))
            batch_features.extend({k: v[i] for k, v in f.items()} for i in range(n))

        batch = super().__call__(batch_features)
        batch_merged: Dict[str, Any] = {k: [] for k in batch.keys()}

        offset = 0
        max_num_sequences = max(num_sequences)
        for n in num_sequences:
            # merge sequences as a single document
            for k, v in batch.items():
                v_seqs = v[offset : offset + n]
                if n < max_num_sequences:
                    padding = torch.full_like(v_seqs[0], -100)[None]
                    v_seqs = torch.cat([v_seqs] + [padding] * (max_num_sequences - n), dim=0)
                batch_merged[k].append(v_seqs)
            offset += n

        batch_merged = {k: torch.stack(v) for k, v in batch_merged.items()}
        batch = BatchEncoding(batch_merged)

        max_num_mentions = max(len(labels) for labels in batch_labels)
        for i in range(len(batch_labels)):
            labels = [
                label + [-100] * (self.max_label_size - len(label)) for label in batch_labels[i]
            ]
            labels.extend([[-100] * self.max_label_size] * (max_num_mentions - len(labels)))
            batch_labels[i] = labels

        batch["labels"] = torch.tensor(batch_labels, dtype=torch.int64)
        batch["num_sequences"] = torch.tensor(num_sequences, dtype=torch.int64)

        return batch
