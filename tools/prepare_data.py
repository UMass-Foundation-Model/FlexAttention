import json
import os
from copy import deepcopy
from tqdm import tqdm
import random

if __name__ == "__main__":
    data = json.load(open("playground/llava_v1_5_mix665k/llava_v1_5_mix665k.json"))
    new_data = []
    for d in tqdm(data):
        if "image" in d:
            convs = d["conversations"]
            if not convs[0]["value"].startswith("<image>"):
                convs[0]["value"] = "<image>\n" + convs[0]["value"]
            convs[0]["value"] = convs[0]["value"].rstrip("\n<image>")
            assert "<image>" in convs[0]["value"] and convs[0]["value"].count("<image>") == 1
        new_data.append(d)
    random.shuffle(new_data)
    data = new_data
    ok_data = []
    for d in tqdm(data):
        if "image" not in d:
            ok_data.append(d)
            continue
        image = os.path.join(root, d["image"])
        if not os.path.exists(image):
            done = False
            uuid = d["image"].split(".")[0]
            for suffix in ["png", "jpg", "jpeg", "gif"]:
                trial = uuid + "." + suffix
                path = os.path.join(root, trial)
                if os.path.exists(path):
                    d["image"] = trial
                    done = True
                    break
            if done:
                tqdm.write(f"updated: {d['image']}")
                assert os.path.exists(os.path.join(root, d["image"]))
                ok_data.append(d)
            else:
                tqdm.write(f">>>>>>>>>>>>>>>>>>> {image}")
        else:
            assert os.path.exists(os.path.join(root, d["image"]))
            ok_data.append(d)
    print(len(data))
    print(len(ok_data))
    json.dump(ok_data, open("playground/llava_v1_5_mix665k/llava_v1_5_mix665k_clean_ok.json", "w"), indent=1)
