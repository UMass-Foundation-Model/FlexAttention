import argparse
import torch
import json
import shortuuid
import os
import time
import torch.distributed as dist
import pickle as pkl
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from tqdm import tqdm
from utils.vbench import VBenchDataset
import numpy as np
from llava.distributed import world_info_from_env, init_distributed_device
import random


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def eval_vbench(model, tokenizer, image_processor, args):
    eval_dataset = VBenchDataset(subset=args.subset)
    model.train()
    correct = 0
    total = 0
    for ib, batch in enumerate(tqdm(eval_dataset, desc="Running inference", disable=(args.rank != 0))):
        if ib % args.world_size != args.rank:
            continue
        image = batch["image"].convert("RGB")
        ori_sizes = [[image.width, image.height]]
        options = batch["options"]
        assert len(options) <= 4
        losses = []
        for option in options:
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{batch['question']}\nAnswer the question using a single word or phrase. ASSISTANT: {option}"
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            image_tensor = process_images([image], image_processor, args)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            attention_mask = torch.ones_like(input_ids, device=input_ids.device).bool()
            labels = input_ids.clone()
            answer_start_id = (labels == 13566).nonzero()[0,1].item() + 2
            assert labels[0, answer_start_id-1] == 29901
            labels[:, :answer_start_id] = -100
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    images=image_tensor,
                    attention_mask=attention_mask,
                    ori_sizes=ori_sizes,
                    labels=labels,
                )
            loss = output.loss.cpu().detach().numpy()
            losses.append(loss)
        if np.argmin(losses) == 0:
            correct += 1
        total += 1

    pkl.dump([correct, total], open(f"temp_{eval_dataset.vqa_dataset}_{args.id}_{args.rank}.pkl", "wb"))
    dist.barrier()
    if dist.get_rank() == 0:
        uuid = str(int(time.time()))[-4:] + shortuuid.uuid()[:2]
        correct, total = 0, 0
        for i in range(args.world_size):
            filename = f"temp_{eval_dataset.vqa_dataset}_{args.id}_{i}.pkl"
            [pc, pt] = pkl.load(open(filename, "rb"))
            correct += pc
            total += pt
            os.remove(filename)
        print(correct/total)
        with open(f"vbench_{args.subset}_{args.id}_{uuid}_{round(correct / total, 3)}.txt", "w") as f:
            f.write(f"{args.subset} {correct} {total} {correct / total}\n")


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
    if "vbench" == args.task:
        eval_vbench(model, tokenizer, image_processor, args)
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
    parser.add_argument("--task", type=str, default="vbench")
    parser.add_argument("--id", type=str, default="default")
    parser.add_argument("--subset", type=str, choices=['direct_attributes', 'GPT4V-hard', 'OCR', 'relative_position'], required=True)
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
