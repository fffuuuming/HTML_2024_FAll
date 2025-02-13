import pandas as pd

# Load preprocessed datasets
X_train_encoded = pd.read_csv('encoded_imputed_X_train.csv')
X_val_encoded = pd.read_csv('encoded_imputed_X_val.csv')
y_train = pd.read_csv('encoded_y_train.csv').squeeze()
y_val = pd.read_csv('encoded_y_val.csv').squeeze()

# Combine training and validation sets for feature selection and final training
X_combined = pd.concat([X_train_encoded, X_val_encoded], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)