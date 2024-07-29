import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from llava.distributed import world_info_from_env, init_distributed_device
import matplotlib.pyplot as plt
from PIL import Image
import math
import warnings
warnings.filterwarnings("ignore")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def eval_model(args):
    # Model
    disable_torch_init()
    init_distributed_device(args)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map=f"cuda:{args.local_rank}")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    for ib, ((input_ids, image_tensor, image_sizes), line) in enumerate(tqdm(zip(data_loader, questions), total=len(questions), disable=(args.rank != 0))):
        if ib % args.world_size != args.rank:
            continue
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_attentions=True,
                use_cache=True)
        output_ids = outputs["sequences"]
        attentions = outputs["attentions"]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        image = Image.open(os.path.join(args.image_folder, line["image"])).convert("RGB")
        input_image = expand2square(image, (0, 0, 0))
        for idx in range(len(attentions) - 1):
            if idx != 0:
                continue
            plt.figure(figsize=(30, 30))
            size = 24
            for ii in range(len(attentions[0])):
                # mean_attn = attentions[idx][ii][:, :, 611:].sum(0).sum(0).sum(0).detach().float().cpu()
                mean_attn = attentions[idx][ii][:, :, -1:].sum(0).sum(0).sum(0).detach().float().cpu()
                image_attn = mean_attn[35: 35 + 576]
                thr = image_attn.topk(int(size * size * 0.2)).values[-1]
                image_attn = torch.where(image_attn >= thr, image_attn, 0)
                image_attn = (image_attn - image_attn.min()) / (image_attn.max() - image_attn.min())
                image_attn = ((image_attn * 255) > 64) * 1.0
                image_attn = image_attn.reshape(size, size).unsqueeze(0).unsqueeze(0)
                image_attn = torch.nn.functional.adaptive_max_pool2d(image_attn, (8, 8))
                image_attn = torch.nn.functional.interpolate(image_attn, size=input_image.size, mode='nearest').squeeze()
                plt.subplot(6, 6, ii + 1)
                plt.imshow(input_image)
                plt.imshow(image_attn, alpha=0.7, cmap='rainbow')
                plt.title(str(ii)+"|"+str(round(image_attn.float().mean().item(), 3)))
            plt.tight_layout()
            plt.savefig(f"examples/A_{str(ib).zfill(6)}_{idx}.png")
            plt.close()
        cur_prompt = cur_prompt.replace("\n", "##")
        print(ib, cur_prompt, " | ", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    args = parser.parse_args()

    eval_model(args)
