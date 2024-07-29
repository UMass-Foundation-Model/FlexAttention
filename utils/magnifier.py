from torch.utils.data import Dataset
import json
import os
from PIL import Image
import base64

class MagnifierDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/cdl/hdvlm/data/MagnifierBench/images.json",
        annotations_path="/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/cdl/hdvlm/data/MagnifierBench/data_instructions.json",
    ):
        image_path = image_dir_path
        with open(image_path, "r") as f:
            images_data = json.load(f)
        images_data = {id: base64.b64decode(img) for id, img in images_data.items()}
        filepath = annotations_path
        with open(filepath, "r") as f:
            data = json.load(f)
        self.magnifier_dataset = []
        for id, row in data["data"].items():
            images = []
            for img_id in row["image_ids"]:
                if img_id in images_data:
                    images.append(images_data[img_id])
            self.magnifier_dataset.append({
                "id": id,
                "instruction": row["instruction"],
                "answer": row["answer"],
                "images": images,
                "image_ids": row["image_ids"],
                "related_instructions": row["rel_ins_ids"],
            })
        # annotations = json.load(open(annotations_path))["data"]
        # self.questions = []
        # self.answers = []
        # self.image_paths = []
        # self.question_ids = []
        # for anno in annotations:
        #     question_id = anno["question_id"]
        #     question = anno["question"]
        #     imageId = anno["image_id"]
        #     answer = anno["answers"]
        #     self.questions.append(question)
        #     self.answers.append(answer)
        #     self.image_paths.append(os.path.join(image_dir_path, "{}.jpg".format(imageId)))
        #     self.question_ids.append(question_id)
        self.vqa_dataset = "magnifier"

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        sample = self.magnifier_dataset[idx]
        # question_id = self.question_ids[idx]
        # answer = self.answers[idx]
        # img_path = self.image_paths[idx]
        # image = Image.open(img_path)
        return {
            "id": sample["id"],
            "instruction": sample["instruction"],
            "answer": sample["answer"],
            "images": sample["images"],
            "image_ids": sample["image_ids"],
            "related_instructions": sample["related_instructions"],
        }


if __name__ == "__main__":
    dataset = MagnifierDataset()
    for sample in dataset:
        print(sample)
