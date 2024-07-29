#!/bin/bash
python -m llava.eval.eval_textvqa \
    --annotation-file /gpfs/u/home/LMCG/LMCGljnn/scratch/code/hdvlm/datasets/textvqa/TextVQA_0.5.1_val.json \
    --result-file $1
