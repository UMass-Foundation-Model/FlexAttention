import torch
import sys
import json
from safetensors import safe_open
from safetensors.torch import save_file

if __name__ == "__main__":
    filename = sys.argv[-1]
    if filename.endswith(".json"):
        data = json.load(open(filename))
        keys = list(data["weight_map"].keys())
        for key in keys:
            if "vision_tower" in key:
                continue
            elif "k_proj" in key:
                new_key = key.replace("k_proj", "k_proj_hd")
                data["weight_map"][new_key] = data["weight_map"][key]
            elif "v_proj" in key:
                new_key = key.replace("v_proj", "v_proj_hd")
                data["weight_map"][new_key] = data["weight_map"][key]
        json.dump(data, open(filename, "w"), indent=2)
    elif filename.endswith(".pt"):
        weights = torch.load(filename)
        keys = list(weights.keys())
        for key in keys:
            if "k_proj" in key or "v_proj" in key:
                new_key = key[:-7] + "_hd" + key[-7:]
                weights[new_key] = weights[key].clone()
                print(key, "----->", new_key)
        torch.save(weights, filename)
        print("done")
    elif filename.endswith("safetensors"):
        weights = {}
        with safe_open(filename, framework="pt", device=0) as f:
            metadata = f.metadata()
            for k in f.keys():
                weights[k] = f.get_tensor(k)
        keys = list(weights.keys())
        for key in keys:
            if "vision_tower" in key:
                continue
            elif "k_proj" in key or "v_proj" in key:
                new_key = key[:-7] + "_hd" + key[-7:]
                weights[new_key] = weights[key].clone()
                print(key, "----->", new_key)
        save_file(weights, filename, metadata=metadata)
        print("done")
