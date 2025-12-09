# for data manipulation
import pandas as pd
import numpy as np
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/DataWiz-6939/superkart-project-dataset/superkart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Feature Engineering

# 1. Extract Product Category from Product_Id
df['Product_Category'] = df['Product_Id'].apply(lambda x: x[:2])
product_category_map = {
    'DR': 'Drinks',
    'NC': 'Non-Consumable',
    'FD': 'Food & Veg'
}
df['Product_Category'] = df['Product_Category'].map(product_category_map).fillna('Other')
# Drop original Product_Id
df.drop(columns=['Product_Id'], inplace=True)

# 2. Compute Store Age
current_year = 2024 # Assuming current year for calculation
df['Store_Age'] = current_year - df['Store_Establishment_Year']
# Drop original Store_Establishment_Year
df.drop(columns=['Store_Establishment_Year'], inplace=True)

# 3. Classify Food Type into Perishable category
perishable_types = ['Meat', 'Dairy', 'Fruits and Vegetables', 'Breakfast', 'Seafood'] # Based on common knowledge and data dictionary examples
df['Food_Type'] = df['Product_Type'].apply(lambda x: 'Perishable' if x in perishable_types else 'Non-Perishable')

# --- End Feature Engineering ---

# Target column name
target_col = 'Product_Store_Sales_Total'

# Split into X  and y
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)


Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="DataWiz-6939/superkart-project-dataset",
        repo_type="dataset",
    )



print("Data preprocessing and feature engineering complete. Split data uploaded to Hugging Face.")
