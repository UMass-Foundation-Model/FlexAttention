import argparse
import torch
import json
import shortuuid
import os
import time
import torch.distributed as dist
import string

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, expand2square
from tqdm import tqdm
from utils.textvqa import TextVQADataset
from utils.docvqa import DocVQADataset
from utils.chartqa import ChartQADataset
import numpy as np
from llava.distributed import world_info_from_env, init_distributed_device
import pickle as pkl
import random
from copy import deepcopy


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def eval_textvqa(model, tokenizer, image_processor, args, dataset_cls=TextVQADataset):
    eval_dataset = dataset_cls()
    model.eval()
    ans_strs = []
    for ib, batch in enumerate(tqdm(eval_dataset, desc="Running inference", disable=(args.rank != 0))):
        if ib % args.world_size != args.rank:
            continue
        image = batch["image"].convert("RGB")
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{batch['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        ori_sizes = [[image.width, image.height]]
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=None,
                max_new_tokens=args.max_new_tokens,
                ori_sizes=ori_sizes,
                use_cache=True)
        new_predictions = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        ans_id = shortuuid.uuid()
        idx = os.path.basename(batch["img_path"]).split(".")[0] if isinstance(dataset_cls, TextVQADataset) else os.path.basename(batch["img_path"])
        ans_str = json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": new_predictions,
                                   "answer_id": ans_id,
                                   "model_id": "llava-v1.5",
                                   "metadata": {}}) + "\n"
        ans_strs.append([ib, ans_str])

    pkl.dump(ans_strs, open(f"temp_{eval_dataset.vqa_dataset}_{args.id}_{args.rank}.pkl", "wb"))
    time.sleep(1)
    dist.barrier()
    if dist.get_rank() == 0:
        uuid = str(int(time.time())) + shortuuid.uuid()[:4]
        merged_list = []
        for i in range(args.world_size):
            filename = f"temp_{eval_dataset.vqa_dataset}_{args.id}_{i}.pkl"
            merged_list.extend(pkl.load(open(filename, "rb")))
            os.remove(filename)
        merged_list = sorted(merged_list, key=lambda x: x[0])
        ans_file = open(f"answer_{eval_dataset.vqa_dataset}_{args.id}_{uuid}.jsonl", "w")
        ans_file.writelines([x[1] for x in merged_list])
        ans_file.close()


def eval_docvqa(model, tokenizer, image_processor, args):
    eval_dataset = DocVQADataset()
    conv = conv_templates[args.conv_mode].copy()
    model.eval()
    ans_strs = []
    for ib, batch in enumerate(tqdm(eval_dataset, desc="Running inference", disable=(args.rank != 0))):
        if ib % args.world_size != args.rank:
            continue
        image = batch["image"].convert("RGB")
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{batch['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        new_predictions = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        ans_str = deepcopy(batch)
        del ans_str["image"]
        ans_str["prediction"] = new_predictions
        ans_strs.append([ib, ans_str])

    pkl.dump(ans_strs, open(f"temp_docvqa_{args.id}_{args.rank}.pkl", "wb"))
    time.sleep(1)
    dist.barrier()
    if dist.get_rank() == 0:
        punctuation_string = string.punctuation
        uuid = str(int(time.time())) + shortuuid.uuid()[:4]
        merged_list = []
        for i in range(args.world_size):
            filename = f"temp_docvqa_{args.id}_{i}.pkl"
            merged_list.extend(pkl.load(open(filename, "rb")))
            os.remove(filename)
        merged_list = sorted(merged_list, key=lambda x: x[0])
        ans_file = open(f"answer_docvqa_{args.id}_{uuid}.jsonl", "w")
        json.dump([x[1] for x in merged_list], ans_file, indent=1)
        ans_file.close()
        ans_file = json.load(open(f"answer_docvqa_{args.id}_{uuid}.jsonl"))
        scores = []
        for sample in ans_file:
            answer = sample["answers"][0].lower()
            prediction = sample["prediction"].lower()
            for i in punctuation_string:
                answer = answer.replace(i, '')
                prediction = prediction.replace(i, '')
            answer = answer.strip()
            prediction = prediction.strip()
            if answer in prediction:
                scores.append(1.0)
            else:
                scores.append(0.0)
        print("score:", np.mean(scores))


def eval_chartqa(model, tokenizer, image_processor, args):
    eval_textvqa(model, tokenizer, image_processor, args, dataset_cls=ChartQADataset)


def main(args):
    seed_everything(0)
    # Model
    disable_torch_init()
    world_info_from_env()
    init_distributed_device(args)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map=f"cuda:{args.local_rank}")
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1" # this is chosen
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    if args.rank == 0:
        print("Evaluate on", args.task)
    if "textvqa" == args.task:
        eval_textvqa(model, tokenizer, image_processor, args)
    elif "docvqa" == args.task:
        eval_docvqa(model, tokenizer, image_processor, args)
    elif "chartqa" == args.task:
        eval_chartqa(model, tokenizer, image_processor, args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--task", type=str, default="textvqa")
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--dist",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
