# FlexAttention for Efficient High-Resolution Vision-Language Models

[[Project Page](https://vis-www.cs.umass.edu/flexattention/)] [[Paper](https://vis-www.cs.umass.edu/flexattention/)]

## Overview

![overview](assets/overview.jpg)

This repository contains the official code for FlexAttention for Efficient High-Resolution Vision-Language Models.

## News

* July 2024: Open-source codebase and evaluation.
* July 2024: Accepted by ECCV'2024!

## Installation

```bash
conda create -n flexattention python=3.9
conda activate flexattention
pip install -e .
pip install -e ".[train]"
pip install -e ./transformers
```

## Checkpoint

You can download our 7B model checkpoint from [huggingface]() and put it into `checkpoints` folder.

## Evaluation

### TextVQA

1. Follow this [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#textvqa) to download the textvqa evaluaton images and annotation, and extract to `datasets/eval/textvqa`.
2. Run the multi-gpu inference:
```bash
torchrun --nproc_per_node 3 scripts/evaluation/eval_textvqa.py --dist --model-path checkpoints/llava-v1.5-7b-flexattn --id llava-v1.5-7b-flexattn
```
It will generate a file similar to `answer_textvqa_llava-v1.5-7b-flexattn_xxx.jsonl` on the folder root.

3. Run the evaluation script:
```bash
bash scripts/evaluation/get_textvqa_score.sh ANSWER_FILE
```

### V* Bench

1. Download the dataset from huggingface.

```bash
git lfs install
git clone https://huggingface.co/datasets/craigwu/vstar_bench
```

2. Run the multi-gpu inference:
```bash
# Attribute
torchrun --nproc_per_node 3 scripts/evaluation/eval_vbench.py --dist --model-path checkpoints/llava-v1.5-7b-flexattn --id llava-v1.5-7b-flexattn --subset direct_attributes

# Spatial
torchrun --nproc_per_node 3 scripts/evaluation/eval_vbench.py --dist --model-path checkpoints/llava-v1.5-7b-flexattn --id llava-v1.5-7b-flexattn --subset relative_position
```

### MagnifierBench

1. Download the dataset from [here](https://drive.google.com/file/d/1DE5PBkhHMdVNOpDg6GtfzO73ZFrK9ltZ/view?usp=sharing), and extract it to `datasets/eval/`.

2. Run the multi-gpu inference:
```bash
torchrun --nproc_per_node 3 scripts/evaluation/eval_magnifier.py --dist --model-path checkpoints/llava-v1.5-7b-flexattn --id llava-v1.5-7b-flexattn
```


## Training

Coming soon.

## Acknowledgement

[LLaVA](https://github.com/haotian-liu/LLaVA): the codebase that our project build on. Thanks for their amazing code and model.

## Citation

If our work is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```

```