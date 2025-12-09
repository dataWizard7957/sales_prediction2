import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Dataset repo ID — MUST be lowercase
repo_id = "DataWiz-6939/superkart-project-dataset"
repo_type = "dataset"

# Read token
HF_TOKEN = os.getenv("HF_TOKEN")

# Add debug print and robust check
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
    print(f"HF_TOKEN length: {len(HF_TOKEN)}") # This will print length, but not value in CI logs

if not HF_TOKEN: # Check for None or empty string after stripping
    raise ValueError("HF_TOKEN is not set or is empty. Please ensure it is provided and has no leading/trailing whitespace.")

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
