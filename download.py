from huggingface_hub import snapshot_download

# Download the entire repository
# The function returns the path to the cached downloaded directory
# set local_dir_use_symlinks to auto for downloading (cached)
# AND set the references
local_data_dir = "./"
local_pretrains_dir = "./"

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
    local_dir_use_symlinks=True
)

# Download 'pretrains/*' patterns
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["pretrains/*"],
    local_dir=local_pretrains_dir,
    local_dir_use_symlinks=True
)

print(f"Dataset downloaded to: {local_data_dir} and {local_pretrains_dir}")