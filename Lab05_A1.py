import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier 

def load_dataset(file_path):
    """Load the dataset using Pandas."""
    df = pd.read_csv(file_path)
    return df

def train_model(X_train, y_train):
    """Train the model using RandomForestClassifier."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1-Score."""
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1

# Loading the dataset
file_path = "C:\\Users\\ashis\\Downloads\\archive (6)\\Data\\features_30_sec.csv"
df = load_dataset(file_path)

# Choose the target variable
target_variable = 'label'

# Assuming 'spectral_bandwidth_mean' and 'spectral_bandwidth_var' are your feature columns
X = df[['spectral_bandwidth_mean', 'spectral_bandwidth_var']]
y = df[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Obtain predictions for both training and test sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics for training data
precision_train, recall_train, f1_train = calculate_metrics(y_train, y_pred_train)

# Calculate metrics for test data
precision_test, recall_test, f1_test = calculate_metrics(y_test, y_pred_test)

# Print the results
print("Training Precision: {:.4f}".format(precision_train))
print("Training Recall: {:.4f}".format(recall_train))
print("Training F1-Score: {:.4f}".format(f1_train))
print("\nTest Precision: {:.4f}".format(precision_test))
print("Test Recall: {:.4f}".format(recall_test))
print("Test F1-Score: {:.4f}".format(f1_test))
