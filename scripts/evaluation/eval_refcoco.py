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
from PIL import Image
from io import BytesIO
import base64


def get_iou(box1, box2):
    # box1 and box2 should be in the format [x1, y1, x2, y2]
    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                   max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    iou = intersection / union if union > 0 else 0
    return iou


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def eval_refcoco(model, tokenizer, image_processor, args):
    world_size = args.world_size
    rank = args.rank
    dataset_name = "refcoco"
    id = args.id
    with open("/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/raw/refcocog/refcocog_val.tsv", "r") as f:
        lines = f.readlines()
        pbar = tqdm(lines, disable=(rank != 0))
        correct = 0
        total = 0
        ious = []
        for ii, line in enumerate(pbar):
            if ii % world_size != rank:
                continue
            line = line.rstrip()
            uniq_id, image_id, text, region_coord, image = line.split("\t")

            image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
            gt_box = np.array(list(map(float, region_coord.split(","))))
            width = image.width
            height = image.height
            gt_box = gt_box / np.array([width, height, width, height])
            size = max(width, height)
            image = image.resize((size, size))

            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nPlease provide the bounding box coordinate of the region this sentence describes: {text}. ASSISTANT:"
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
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
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            new_predictions = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            try:
                pred_box = eval(new_predictions)
            except:
                pred_box = [0, 0, 0, 0]
            iou = get_iou(pred_box, gt_box)
            if iou >= 0.5:
                correct += 1
            total += 1
            ious.append(iou)
            pbar.set_description(f"iou: {iou:.2f} score: {correct / total:.4f}")

    with open(f"{dataset_name}_results_part{rank}_{id}.json", "w") as f:
        f.write(json.dumps([total, correct, ious]))
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        total = 0
        correct = 0
        ious = []
        print(f"evaluate on rank {rank}. world size is {world_size}")
        for rank_i in range(world_size):
            [total_part, correct_part, ious_part] = json.load(open(f"{dataset_name}_results_part{rank_i}_{id}.json"))
            os.remove(f"{dataset_name}_results_part{rank_i}_{id}.json")
            total += total_part
            correct += correct_part
            ious.extend(ious_part)
        score = correct / total
        print(np.mean(ious))
        print("score:", score)
    else:
        score = 0.0
    if world_size > 1:
        torch.distributed.barrier()


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
    if "refcoco" == args.task:
        eval_refcoco(model, tokenizer, image_processor, args)
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
