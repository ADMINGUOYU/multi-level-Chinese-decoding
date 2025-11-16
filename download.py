import os
from huggingface_hub import snapshot_download

# Set HuggingFace mirror source for China (no VPN needed)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 确认环境变量已设置
print(f"当前 HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
# Download the entire repository
# The function returns the path to the cached downloaded directory
# set local_dir_use_symlinks to auto for downloading (cached)
# AND set the references
local_data_dir = "./"
local_pretrains_dir = "./"

# Set custom cache directory to current project folder instead of /root/.cache
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
os.makedirs(cache_dir, exist_ok=True)

# Assuming repo_id is defined earlier in the context (e.g., repo_id = "your/repo_name")
# You must define 'repo_id' before running this code.

# This line in the image seems like a partial/commented out/placeholder call
# snapshot_download(repo_id = repo_id, repo_type = "dataset")
repo_id = 'liulab-repository/Du-IN'
# Download 'data/*' patterns
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["data/*"],
    local_dir=local_data_dir,
    local_dir_use_symlinks=True,
    cache_dir=cache_dir
)

# Download 'pretrains/*' patterns
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["pretrains/*"],
    local_dir=local_pretrains_dir,
    local_dir_use_symlinks=True,
    cache_dir=cache_dir
)

print(f"Dataset downloaded to: {local_data_dir} and {local_pretrains_dir}")