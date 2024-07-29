from torch.utils.data import Dataset
import json
import os
from PIL import Image


class DocVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/hdvlm/datasets/docvqa/images",
        annotations_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/hdvlm/datasets/docvqa/qas/val_v1.0_withQT.json",
    ):
        annotations = json.load(open(annotations_path))["data"]
        self.questions = []
        self.answers = []
        self.image_paths = []
        self.question_ids = []
        for anno in annotations:
            question_id = anno["questionId"]
            question = anno["question"]
            imageId = os.path.basename(anno["image"])
            answers = anno["answers"]
            self.questions.append(question)
            self.answers.append(answers)
            self.image_paths.append(os.path.join(image_dir_path, imageId))
            self.question_ids.append(question_id)
        self.vqa_dataset = "docvqa"

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
    dataset = DocVQADataset()
    for sample in dataset:
        print(sample)
