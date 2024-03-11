import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def segregate_data(data):
    A = data.iloc[:, 1:4]
    C = data.iloc[:, 4]
    return A, C

def calculate_pseudo_inverse(A):
    return np.linalg.pinv(A)

def calculate_model_vector(A_inv, C):
    return np.dot(A_inv, C)

def calculate_evaluation_metrics(C, predicted_C):
    mse = mean_squared_error(C, predicted_C)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((C - predicted_C) / C)) * 100
    r2 = r2_score(C, predicted_C)
    return mse, rmse, mape, r2

def print_evaluation_metrics(mse, rmse, mape, r2):
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R-squared (R2) Score:", r2)

# Load the data from the Excel file
file_path = "D:\python\Machine_Learning\Lab Session1 Data.xlsx"
sheet_name = "Purchase data"
purchase_data = load_data(file_path, sheet_name)

# Segregate data into matrices A and C
A, C = segregate_data(purchase_data)

# Calculate the pseudo-inverse of A
A_inv = calculate_pseudo_inverse(A)

# Calculate the model vector X
X = calculate_model_vector(A_inv, C)

# Calculate evaluation metrics
predicted_C = np.dot(A, X)
mse, rmse, mape, r2 = calculate_evaluation_metrics(C, predicted_C)

# Print evaluation metrics
print_evaluation_metrics(mse, rmse, mape, r2)
