import argparse
import torch
import json
import shortuuid
import os
import time
import torch.distributed as dist
from io import BytesIO
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, expand2square
from tqdm import tqdm
from utils.vbench import VBenchDataset
from utils.magnifier import MagnifierDataset
import numpy as np
from llava.distributed import world_info_from_env, init_distributed_device
import pickle as pkl
import random
from copy import deepcopy
from calflops import calculate_flops


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_mb(model, tokenizer, image_processor, args):
    eval_dataset = MagnifierDataset()
    model.eval()
    max_new_tokens = 64
    max_new_tokens -= 1
    total_time = []
    total_token_num = []
    total_flops = []

    for warmup_iter in range(1, 11):
        batch = eval_dataset[-warmup_iter]
        image = BytesIO(batch["images"][0])
        image = Image.open(image).convert("RGB")
        ori_sizes = [[image.width, image.height]]
        batch['instruction'] = batch['instruction'].replace(" A.", "\nA.").replace(", B.", "\nB.").replace(", C.", "\nC.").replace(", D.", "\nD.")
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{batch['instruction']}\nAnswer with the option's letter from the given choices directly. ASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        with torch.no_grad():
            torch.cuda.synchronize()
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=None,
                max_new_tokens=max_new_tokens,
                ori_sizes=ori_sizes,
                use_cache=True)
            torch.cuda.synchronize()

    for ib, batch in enumerate(tqdm(eval_dataset, desc="Running inference")):
        image = BytesIO(batch["images"][0])
        image = Image.open(image).convert("RGB")
        ori_sizes = [[image.width, image.height]]
        batch['instruction'] = batch['instruction'].replace(" A.", "\nA.").replace(", B.", "\nB.").replace(", C.", "\nC.").replace(", D.", "\nD.")
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{batch['instruction']}\nAnswer with the option's letter from the given choices directly. ASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=None,
                max_new_tokens=max_new_tokens,
                ori_sizes=ori_sizes,
                use_cache=True)
            torch.cuda.synchronize()
            end = time.time()
            new_predictions = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            token_num = output_ids[0].shape[0]
            tqdm.write(f"{token_num} {new_predictions}")
            total_time.append(end - start)
            total_token_num.append(token_num)
            inputs = {}
            inputs["inputs"] = input_ids
            inputs["images"] = image_tensor
            inputs["max_new_tokens"] = torch.tensor(max_new_tokens)
            # inputs["ori_sizes"] = torch.tensor(ori_sizes).long()
            with torch.no_grad():
                flops, macs, params = calculate_flops(
                    model=model,
                    kwargs=inputs,
                    print_results=False,
                    print_detailed=False,
                    forward_mode="generate"
                )
            flops = float(flops.split(" ")[0])
            total_flops.append(flops)
    throughput = np.sum(total_token_num) / np.sum(total_time)
    print("throughput (token/s):", round(throughput, 4))
    print("Avg TFLOPs:", np.mean(total_flops))
    print("Avg token num:", np.mean(total_token_num))


def main(args):
    seed_everything(0)
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map="cuda:0")
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

    run_mb(model, tokenizer, image_processor, args)


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
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
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
