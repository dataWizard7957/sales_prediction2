# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Explicitly set MLflow to log to a local directory named 'mlruns'
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/DataWiz-6939/superkart-project-dataset/Xtrain.csv"
Xtest_path = "hf://datasets/DataWiz-6939/superkart-project-dataset/Xtest.csv"
ytrain_path = "hf://datasets/DataWiz-6939/superkart-project-dataset/ytrain.csv"
ytest_path = "hf://datasets/DataWiz-6939/superkart-project-dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Define features
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age'
]
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Id',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Category',
    'Food_Type'
]


# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Regressor model
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid for XGBoost Regressor (reduced for faster tuning)
param_grid = {
    'xgbregressor__n_estimators': [75, 100],
    'xgbregressor__max_depth': [4, 5],
    'xgbregressor__learning_rate': [0.05, 0.1],
    'xgbregressor__subsample': [0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.8, 1.0],
    'xgbregressor__alpha': [0.1, 0.5] # L1 regularization
}


# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run(run_name="GridSearchCV_Hyperparameter_Tuning"):
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error') # Using negative MSE for GridSearchCV
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = -results['mean_test_score'][i] # Convert back to positive MSE
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True, run_name=f"Params_{i}"):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_validation_mse", mean_score)
            mlflow.log_metric("std_validation_mse", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Calculate regression metrics
    train_mse = mean_squared_error(ytrain, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_r2 = r2_score(ytrain, y_pred_train)

    test_mse = mean_squared_error(ytest, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2
    })

    # Save the model locally
    model_path = "best_sales_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "DataWiz-6939/sales-prediction-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_sales_prediction_model_v1.joblib",
        path_in_repo="best_sales_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
