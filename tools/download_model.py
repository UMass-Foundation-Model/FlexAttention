from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="liuhaotian/llava-v1.5-7b", local_dir="./llava-v1.5-7b", local_dir_use_symlinks=False))
