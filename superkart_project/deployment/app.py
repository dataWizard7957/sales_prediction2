import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import numpy as np


# Download and load the model
# Ensure the HF token is available in the environment if the repo is private/gated
model_path = hf_hub_download(repo_id= "DataWiz-6939/sales-prediction-model", filename="best_sales_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Superkart Sales Prediction
st.title("Superkart Sales Prediction App")
st.write("""
This application predicts the total sales for a given product in a specific store.
Please enter the product and store details below to get a sales forecast.
""")

# User input fields based on the dataset features
st.header("Product Details")
product_id_prefix = st.selectbox("Product Category Prefix (from Product_Id)", ['DR', 'NC', 'FD'])
product_id_num = st.number_input("Product ID Number (e.g., 001 for Px001)", min_value=0, max_value=999, value=np.random.randint(0,100))
product_id_dummy = f"{product_id_prefix}{product_id_num:03d}"

product_weight = st.number_input("Product Weight", min_value=1.0, max_value=50.0, value=10.0, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ['low sugar', 'regular', 'no sugar'])
product_allocated_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
product_type = st.selectbox("Product Type", [
    'Dairy', 'Snack Foods', 'Household', 'Frozen Foods', 'Fruits and Vegetables',
    'Meat', 'Breakfast', 'Seafood', 'Hard Drinks', 'Canned', 'Soft Drinks',
    'Health and Hygiene', 'Baking Goods', 'Bread', 'Starchy Foods', 'Others'
])
product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=10.0, max_value=1000.0, value=150.0, step=1.0)

st.header("Store Details")
store_id = st.text_input("Store ID (e.g., S001)", 'S001') # Can be string as it's one-hot encoded
store_establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2023, value=2000)
store_size = st.selectbox("Store Size", ['High', 'Medium', 'Low'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type 1', 'Departmental Store', 'Supermarket Type 2', 'Food Mart'])

# Assemble input into DataFrame (raw features)
input_df = pd.DataFrame([{
    'Product_Id': product_id_dummy, # Will be engineered
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_Type': product_type,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}])

# Replicate Feature Engineering (MUST match prep.py)

# 1. Extract Product Category from Product_Id
input_df['Product_Category'] = input_df['Product_Id'].apply(lambda x: x[:2])
product_category_map = {
    'DR': 'Drinks',
    'NC': 'Non-Consumable',
    'FD': 'Food & Veg'
}
input_df['Product_Category'] = input_df['Product_Category'].map(product_category_map).fillna('Other')
# Drop original Product_Id (as it's engineered)
input_df.drop(columns=['Product_Id'], inplace=True)

# 2. Compute Store Age
current_year = 2024 # Must match year used in prep.py
input_df['Store_Age'] = current_year - input_df['Store_Establishment_Year']
# Drop original Store_Establishment_Year
input_df.drop(columns=['Store_Establishment_Year'], inplace=True)

# 3. Classify Food Type into Perishable category
perishable_types = ['Meat', 'Dairy', 'Fruits and Vegetables', 'Breakfast', 'Seafood'] # Must match prep.py
input_df['Food_Type'] = input_df['Product_Type'].apply(lambda x: 'Perishable' if x in perishable_types else 'Non-Perishable')

# End Feature Engineering

if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result:")
    st.success(f"The predicted total sales for this product in the store is: **${prediction:,.2f}**")
