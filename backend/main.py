from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
import re

import time
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import torch
from transformers import BertTokenizer, AutoTokenizer
from transformers import AutoModelForSequenceClassification

# print(np.__version__)
import xgboost as xgb

app = FastAPI()

def tokenize(row):
    if row is None or row is '':
        tokens = ""
    else:
        tokens = row.split(" ")
    return tokens

def remove_reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', " ", token)
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens

def assemble_bag_single(data):
    """
    Generate a bag-of-words feature vector for a single input string.
    
    Args:
        data (list of tokens): Preprocessed tokens of the email.
    
    Returns:
        dict: A dictionary representing the token frequency.
    """
    unique_tokens = {}
    
    for token in data:
        if token in unique_tokens:
            unique_tokens[token] += 1
        else:
            unique_tokens[token] = 1
    
    return unique_tokens

def prepare_features_single(email_text, model_columns):
    """
    Prepares feature vector for a single email.

    Args:
        email_text (str): The raw email content as a single string.
        model_columns (list): The feature columns expected by the trained model.
    
    Returns:
        pd.DataFrame: A single-row DataFrame with aligned features.
    """
    # Preprocess the input email
    tokens = tokenize(email_text)  # Tokenize the single string
    tokens = remove_reg_expressions(tokens)  # Remove unwanted patterns
    
    # Generate the bag-of-words representation
    bag = assemble_bag_single(tokens)
    
    # Convert dictionary to DataFrame with one row
    feature_df = pd.DataFrame([bag]).fillna(0)
    
    # Reindex to match model columns, filling missing columns with 0
    feature_df = feature_df.reindex(columns=model_columns, fill_value=0)
    
    # Ensure the data type is correct for model input
    # feature_df = feature_df.astype(np.float32)
    
    return feature_df

# Load models from saved files
def load_xgboost_models(base_path, filename_prefix="xgboost_model"):
    """
    Load XGBoost models from specified folder.
    
    Args:
        base_path (str): The base directory where models are stored.
        filename_prefix (str): The common prefix for model filenames.
    
    Returns:
        dict: A dictionary of loaded models keyed by folder name.
    """
    models = {}
    print("Loading XGBoost models...")
    try:
        start_time = time.time()  # Start timing
        for folder in os.listdir(base_path):
            model_path = os.path.join(base_path, folder)
            if os.path.exists(model_path):
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                models[folder] = model
                print(f"Loaded model for folder: {folder}")
            else:
                print(f"Model file not found: {model_path}")
        end_time = time.time()  # End timing
        print("XGBoost Model loaded succesfully time:", end_time - start_time)
    except Exception as e:
        print(f"Error loading models: {e}")
    # print("Loaded models:", list(models.keys()))
    return models

# Load the model in FastAPI
def load_model_bert(path=r".\Models\bert_model.pth"):
    start_time = time.time()  # Start timing
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=9)
    
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    end_time = time.time()  # End timing
    print("BERT Model loaded successfully. time: ", end_time - start_time)
    return model

def load_models():
    # Load Logistic Regression Model
    start_time = time.time()  # Start timing
    log_regression_models = joblib.load(".\Models\model_log_regression.pkl")
    end_time = time.time()  # End timing
    print("Lg Models loaded successfully, Time:", end_time - start_time)
    
    start_time = time.time()  # Start timing
    rf_models = joblib.load(".\Models\model_RF.pkl")
    end_time = time.time()  # End timing
    print("RF Models loaded successfully, Time:", end_time - start_time)
    
    xg_models = load_xgboost_models(".\Models\XGBoost")
    
    return log_regression_models, rf_models, xg_models

log_regression_models, rf_models, xg_models = load_models()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = load_model_bert()


class_le = joblib.load("label_encoder.pkl")
filtered_folders = joblib.load("filtered_folders.pkl")


# Modify predict function to load the feature columns from the model
def predict(email_text, log_regression_models, filtered_folders, model_columns):
    """
    Generates predictions for a single email text input.

    Args:
        email_text (str): The raw email content as a single string.
        log_regression_models (dict): Loaded logistic regression models.
        filtered_folders (list): List of folder names for classification.
        model_columns (list): The feature columns expected by the trained model.
    
    Returns:
        str: The predicted class label.
    """
    # Generate features with correct columns
    X_test = prepare_features_single(email_text, model_columns)
    
    # Predict probabilities
    testing_probs = pd.DataFrame(columns=filtered_folders)
    for folder in filtered_folders:
        # Compute probability for each folder
        testing_probs[folder] = log_regression_models[folder].predict_proba(X_test)[:, 1]
    
    # Select the folder with the highest probability
    y_test_pred = testing_probs.idxmax(axis=1)
    return y_test_pred

def xg_predict(email_text, models, filtered_folders, model_columns):
    """
    Generates predictions using XGBoost models for a single email text.

    Args:
        email_text (str): The raw email content as a single string.
        models (dict): Loaded XGBoost models for each class.
        filtered_folders (list): List of folder names for classification.
        model_columns (list): The feature columns expected by the trained models.
    
    Returns:
        str: The predicted class label.
    """
    # Debugging logs
    # print("Model keys:", list(models.keys()))
    # print("Filtered folders:", filtered_folders)
    
    # Preprocess and generate features for the input email
    X_test = prepare_features_single(email_text, model_columns)
    
    # Rename the columns of X_test to match the training columns
    X_test.columns = [f'feature_{i}' for i in range(X_test.shape[1])]
    # print("Generated X_test columns:", X_test.columns)

    # Predict probabilities for each class
    testing_probs = pd.DataFrame(columns=filtered_folders)
    for folder in filtered_folders:
        try:

            # Compute probabilities using the folder-specific model
            testing_probs[folder] = models[f"xgboost_model_{folder}.json"].predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Error predicting for folder {folder}: {e}")
            testing_probs[folder] = 0  # Default to 0 if an error occurs

    # Select the folder with the highest probability
    y_test_pred = testing_probs.idxmax(axis=1)
    return y_test_pred

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

# email_text = """Good Luck 
#                 XYZ
#                 noo
#              """
    
# Load model columns from a trained model for reindexing
log_reg_model_columns = log_regression_models[list(log_regression_models.keys())[0]].feature_names_in_
rf_model_columns = rf_models[list(rf_models.keys())[0]].feature_names_in_

# Generate features
# features = prepare_features_single(email_text)


# Set CORS policy to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],             # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all headers
)

@app.post("/classifyemail")
async def classify_email(emailContent: dict):
    # Your email content
    # print(emailContent['emailContent'])

    y_hat = predict_bert(emailContent['emailContent'])

    category = class_le.inverse_transform([y_hat])[0]
    print("Prediction Done, The class is:",y_hat, " ", category)
    return {"category": category}  # tag/class

@app.post("/classifyemailensemble")
async def classify_email_ensemble(emailContent: dict):
    
    email_text = emailContent['emailContent']
    
    y_hat_bert = predict_bert(email_text)
    y_hat_reg = predict(email_text, log_regression_models, filtered_folders, rf_model_columns)
    y_hat_rf = predict(email_text, rf_models, filtered_folders, rf_model_columns)
    y_hat_xg = xg_predict(email_text, xg_models, filtered_folders, rf_model_columns)
    
    y_hat_bert = class_le.inverse_transform([y_hat_bert])[0]
    y_hat_reg = class_le.inverse_transform([y_hat_reg])[0]
    y_hat_rf = class_le.inverse_transform([y_hat_rf])[0]
    y_hat_xg = class_le.inverse_transform([y_hat_xg])[0]
    
    analysis = {"Classification": "Supervised Models",
                "Logistic_Regression_Probability": "72%",
                "Logistic_Regression_Class": y_hat_reg,
                
                "Random_Forest_Probability": "74%",
                "Random_Forest_Class": y_hat_rf,
                
                "XGBoost_Probability": "72%",
                "XGBoost_Class": y_hat_xg,
                
                "BERT_Probability": "77%",
                "BERT_Class": y_hat_bert,
                }
    return analysis

@app.get("/test")
async def test():
    return {"message": "Test success!"}


