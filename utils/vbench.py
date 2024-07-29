from torch.utils.data import Dataset
import json
import os
from PIL import Image
import glob


class VBenchDataset(Dataset):
    def __init__(
        self,
        subset="direct_attributes",
        root="datasets/eval/vstar_bench",
    ):
        annotations = glob.glob(os.path.join(root, subset, "*.json"))
        self.questions = []
        self.answers = []
        self.image_paths = []
        self.question_ids = []
        self.options = []
        for anno in annotations:
            data = json.load(open(anno))
            for suffix in ["jpg", "JPG", "webp", "png", "jpeg"]:
                image_path = os.path.join(anno.replace("json", suffix))
                if os.path.exists(image_path):
                    break
            question_id = os.path.basename(anno)[:-5]
            self.questions.append(data["question"])
            self.image_paths.append(image_path)
            self.question_ids.append(question_id)
            self.options.append(data["options"])
        self.vqa_dataset = "vbench"

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        options = self.options[idx]
        question_id = self.question_ids[idx]
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        return {
            "image": image,
            "question": question,
            "options": options,
            "question_id": question_id,
            "img_path": img_path,
        }


if __name__ == "__main__":
    dataset = VBenchDataset()
    for sample in dataset:
        pass
