# LUKE-Coref

This repository is for coreference resolution training/inference using [LUKE](https://github.com/studio-ousia/luke).

## Usage

### Installation

```sh
$ git clone https://github.com/naist-nlp/luke-coref.git
$ cd luke-coref
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

### Dataset preparation

Datasets must be in the JSON Lines format, where each line represents a **document** that consists of examples, as exemplified below:

```json
{
  "id": "doc-001",
  "examples": [
    {
      "id": "s1",
      "text": "She graduated from NAIST.",
      "mentions": [
        {
          "start": 19,
          "end": 24,
          "entity_id": "E1"
        }
      ]
    },
    {
      "id": "s2",
      "text": "The university is located in Ikoma, Nara.",
      "mentions": [
        {
          "start": 0,
          "end": 14,
          "entity_id": "E1"
        },
        {
          "start": 29,
          "end": 34,
          "entity_id": "E2"
        },
        {
          "start": 36,
          "end": 40,
          "entity_id": "E3"
        },
      ]
    }
  ]
}
```

Note that <ins>this implementation does not provide an end-to-end system</ins>. Thus, mention recognition must be performed in advance.

### Fine-tuning

```py
torchrun --nproc_per_node 4 src/main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file data/train.jsonl \
    --validation_file data/dev.jsonl \
    --test_file data/test.jsonl \
    --model "studio-ousia/luke-large" \
    --output_dir ./output/ \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --save_strategy epoch
```

### Evaluation/Prediction

```py
torchrun --nproc_per_node 4 src/main.py \
    --do_eval \
    --do_predict \
    --validation_file data/dev.jsonl \
    --test_file data/test.jsonl \
    --model PATH_TO_YOUR_MODEL \
    --output_dir ./output/ \
    --per_device_eval_batch_size 4
```
