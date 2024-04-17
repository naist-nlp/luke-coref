import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import Collator, Preprocessor
from models import LukeForEntitySpanClustering
from training_utils import LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    model: str = "studio-ousia/mluke-base"
    cache_dir: Optional[str] = None
    max_document_length: Optional[int] = None


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    data_files = {k: getattr(args, f"{k}_file") for k in ["train", "validation", "test"]}
    data_files = {k: v for k, v in data_files.items() if v is not None}
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    preprocessor = Preprocessor(tokenizer, args.max_document_length)

    def preprocess(document):
        return preprocessor(document["examples"])

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, remove_columns=column_names)

    model = LukeForEntitySpanClustering.from_pretrained(args.model, config=config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits.get("train"),
        eval_dataset=splits.get("validation"),
        data_collator=Collator(tokenizer),
        compute_metrics=_compute_metrics,
        preprocess_logits_for_metrics=_preprocess_logits,
    )
    trainer.add_callback(LoggerCallback(logger))

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        logger.info(f"eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        result = trainer.predict(splits["test"])
        logger.info(f"test metrics: {result.metrics}")
        trainer.log_metrics("predict", result.metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("predict", result.metrics)

        if trainer.is_world_process_zero():
            predictions = predict(result.predictions, raw_datasets["test"])
            output_file = Path(training_args.output_dir).joinpath("test_predictions.jsonl")
            with open(output_file, mode="w") as f:
                dump(f, raw_datasets["test"], predictions, entity_id_format="E{}")


def _preprocess_logits(logits, labels, max_label_size=128, padding_value=-float("inf")):
    return torch.nn.functional.pad(
        logits, (0, max_label_size - logits.shape[2]), value=padding_value
    )


def _compute_metrics(p: EvalPrediction, max_label_size=128, padding_index=-100):
    preds = p.predictions.argmax(axis=-1).ravel()
    labels = p.label_ids.reshape(-1, max_label_size)
    mask = labels[:, 0] != padding_index
    preds = preds[mask]
    labels = labels[mask]
    accuracy = ((labels - preds[:, None]) == 0).sum() / len(labels)
    return {"accuracy": accuracy}


def predict(logits, dataset):
    outputs = []
    for antecedents, document in zip(logits.argmax(axis=-1), dataset):
        num_mentions = sum(len(example["mentions"]) for example in document["examples"])
        links = [(idx, antecedent) for idx, antecedent in enumerate(antecedents[:num_mentions])]
        clusters = []
        while len(links) > 0:
            remaining = []
            for link in links:
                if link[1] == link[0]:  # no antecedents
                    clusters.append({link[0]})
                    continue
                for cluster in clusters:
                    if link[1] in cluster:
                        cluster.add(link[0])
                        break
                else:
                    remaining.append(link)
            links = remaining
        outputs.append(clusters)
    return outputs


def dump(writer, dataset, predictions: List[List[Set[int]]], entity_id_format="{}"):
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))

    for document, clusters in zip(dataset, predictions):
        outputs: List[Dict] = []

        mentions: List[Dict] = []
        for example in document["examples"]:
            output = {"id": example["id"], "text": example["text"], "mentions": []}
            for mention in example["mentions"]:
                mention = dict(mention, entity_id=None)
                mentions.append(mention)
                output["mentions"].append(mention)
            outputs.append(output)

        for i, mention_ids in enumerate(sorted(clusters, key=lambda c: min(c))):
            for mention_id in mention_ids:
                mentions[mention_id]["entity_id"] = entity_id_format.format(i + 1)
        assert all(mention["entity_id"] is not None for mention in mentions)

        writer.write(encoder.encode({"id": document["id"], "examples": outputs}))
        writer.write("\n")


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "default.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    if args.validation_file is None:
        training_args.evaluation_strategy = "no"
    main(args, training_args)
