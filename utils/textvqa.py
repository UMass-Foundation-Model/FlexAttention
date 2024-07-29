from torch.utils.data import Dataset
import json
import os
from PIL import Image


class TextVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/hdvlm/datasets/textvqa/train_images",
        annotations_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/hdvlm/datasets/textvqa/TextVQA_0.5.1_val.json",
    ):
        annotations = json.load(open(annotations_path))["data"]
        self.questions = []
        self.answers = []
        self.image_paths = []
        self.question_ids = []
        for anno in annotations:
            question_id = anno["question_id"]
            question = anno["question"]
            imageId = anno["image_id"]
            answer = anno["answers"]
            self.questions.append(question)
            self.answers.append(answer)
            self.image_paths.append(os.path.join(image_dir_path, "{}.jpg".format(imageId)))
            self.question_ids.append(question_id)
        self.vqa_dataset = "textvqa"

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        question_id = self.question_ids[idx]
        answer = self.answers[idx]
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        return {
            "image": image,
            "question": question,
            "answers": answer,
            "question_id": question_id,
            "img_path": img_path,
        }


if __name__ == "__main__":
    dataset = TextVQADataset()
    for sample in dataset:
        print(sample)
