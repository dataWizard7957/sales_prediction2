from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "DataWiz-6939/Sales-Prediction"  

api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="."  # Upload to root of Space
)

