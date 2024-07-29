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
from utils.magnifier import MagnifierDataset
import numpy as np
from llava.distributed import world_info_from_env, init_distributed_device
import pickle as pkl
import random
from copy import deepcopy


from PIL import Image
import re
from io import BytesIO
import requests

API_KEY = "YOUR_API_KEY"

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_chat_response(promot, api_key, model="gpt-4-0613", temperature=0, max_tokens=256, n=1, patience=5, sleep_time=5):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Your task is to judge whether the model response is correct to answer the given question or not."},
        {"role": "user", "content": promot},
    ]

    payload = {"model": model, "messages": messages}

    while patience > 0:
        patience -= 1
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            response.raise_for_status()
            response_data = response.json()

            prediction = response_data["choices"][0]["message"]["content"].strip()
            if prediction != "" and prediction is not None:
                return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)
            time.sleep(sleep_time)

    return ""

def parse_pred_ans(pred_ans, question):
    match = re.search(r"The answer is ([A-D])", pred_ans)
    if match:
        return match.group(1)
    choices = ["A", "B", "C", "D"]
    for selection in choices:
        if selection in pred_ans:
            return selection
    pattern = "A\\. (.+?), B\\. (.+?), C\\. (.+?), D\\. (.+)"
    matches = re.search(pattern, question)
    if matches:
        options = {"A": matches.group(1), "B": matches.group(2), "C": matches.group(3), "D": matches.group(4)}
        for c, option in options.items():
            option = option.strip()
            if option.endswith(".") or option.endswith(",") or option.endswith("?"):
                option = option[:-1]
            if option.upper() in pred_ans.upper():
                return c
    for selection in choices:
        if selection in pred_ans.upper():
            return selection
    return "other"

def prepare_query(freeform_q,freeform_ans, ans,api_key = API_KEY):
    freeform_question = freeform_q
    freeform_response = freeform_ans
    correct_answer = ans

    # Formulating the prompt for ChatGPT
    prompt = f"Question: {freeform_question}\nModel Response: {freeform_response}\nGround Truth: {correct_answer}\nWill the model response be considered correct? You should only answer yes or no."

    # Querying ChatGPT
    chat_response = get_chat_response(prompt, api_key)

    return prompt,chat_response
def eval_magnifier(model, tokenizer, image_processor, args, dataset_cls=MagnifierDataset):
    eval_dataset = dataset_cls()
    model.eval()
    ans_strs = []
    num_data = 0
    score = 0
    # ff_score = 0
    for ib, batch in enumerate(tqdm(eval_dataset, desc="Running inference", disable=(args.rank != 0))):
        if ib % args.world_size != args.rank:
            continue
        image = BytesIO(batch["images"][0])
        image = Image.open(image).convert("RGB")
        ori_sizes = [[image.width, image.height]]
        if args.type == "freeform":
            freeform_question = (batch['instruction'].split("?")[0] + "?").strip()
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{freeform_question} ASSISTANT:"
            options = batch['instruction'].split("?")[1]
            answer_option = batch["answer"]
            for single_opt in options.split(","):
                single_opt = single_opt.strip()
                if single_opt.startswith(answer_option.upper()):
                    freeform_answer = single_opt.split(".")[1].strip()
                    break
        else:
            batch['instruction'] = batch['instruction'].replace(" A.", "\nA.").replace(", B.", "\nB.").replace(", C.", "\nC.").replace(", D.", "\nD.")
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{batch['instruction']}\nAnswer with the option's letter from the given choices directly. ASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)
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
        num_data += 1
        new_predictions = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if args.type == "freeform":
            gpt_prompt,gpt_response = prepare_query(freeform_question,new_predictions, freeform_answer)
            if gpt_response.lower() == "yes":
                score += 1
            elif gpt_response.lower() == "no":
                score += 0
            ans_id = shortuuid.uuid()
            # {
            #     "id": sample["id"],
            #     "instruction": sample["instruction"],
            #     "answer": sample["answer"],
            #     "images": sample["images"],
            #     "image_ids": sample["image_ids"],
            #     "related_instructions": sample["related_instructions"],
            # }
            ans_str = json.dumps({"question_id": batch["id"],
                                  "prompt": prompt,
                                  "text": new_predictions,
                                  "gt": batch["answer"],
                                  "gpt_prompt":gpt_prompt,
                                  "gpt_response":gpt_response,
                                  "ff_ans": new_predictions,
                                  "model_id": "llava-v1.5",
                                  "metadata": {}})+ "\n"
            ans_strs.append([ib, ans_str])
            print(f"rank:{args.rank} num:{num_data} ffscore:{score}")
        else:
            multi_ans = parse_pred_ans(pred_ans=new_predictions,question=batch['instruction'])
            if multi_ans == batch["answer"]:
                score += 1
            ans_id = shortuuid.uuid()
            ans_str = json.dumps({"question_id": batch["id"],
                                       "prompt": prompt,
                                       "text": new_predictions,
                                       "gt":batch["answer"],
                                       "prased_ans":multi_ans,
                                       "model_id": "llava-v1.5",
                                       "metadata": {}})
            ans_strs.append([ib, ans_str])
            # print(f"rank:{args.rank} num:{num_data} score:{score}")
    # print(f"rank:{args.rank} num:{num_data} ffscore:{score}")

    pkl.dump([num_data, score], open(f"temp_{eval_dataset.vqa_dataset}_{args.id}_{args.rank}.pkl", "wb"))
    time.sleep(1)
    dist.barrier()
    if dist.get_rank() == 0:
        total, score = 0, 0
        for i in range(args.world_size):
            filename = f"temp_{eval_dataset.vqa_dataset}_{args.id}_{i}.pkl"
            pt, ps = pkl.load(open(filename, "rb"))
            total += pt
            score += ps
            os.remove(filename)
        print(score / total, score, total)


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
    if "magnifier" == args.task:
        eval_magnifier(model, tokenizer, image_processor, args)
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
    parser.add_argument("--task", type=str, default="magnifier")
    parser.add_argument("--type", type=str, default="multi")
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
