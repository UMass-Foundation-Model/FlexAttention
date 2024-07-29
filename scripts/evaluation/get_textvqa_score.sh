#!/bin/bash
python -m llava.eval.eval_textvqa \
    --annotation-file datasets/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $1
