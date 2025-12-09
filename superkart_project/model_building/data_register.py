import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Dataset repo ID — MUST be lowercase
repo_id = "DataWiz-6939/superkart-project-dataset"
repo_type = "dataset"

# Read token
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is not set. Run:  export HF_TOKEN='your_token_here'")

api = HfApi(token=HF_TOKEN)

# Check if dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Repo created successfully.")

# Upload the dataset folder inside the project
api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print(" Dataset uploaded successfully to Hugging Face.")
