import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('train_data.csv')

# Convert 'date' column to datetime (adjusted format)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Sort the DataFrame by 'date' from earliest to latest
df = df.sort_values(by='date').reset_index(drop=True)

# Map True to 1 and False to -1 for 'is_night_game' and 'home_team_win'
df['is_night_game'] = df['is_night_game'].map({True: 1, False: -1})
df['home_team_win'] = df['home_team_win'].map({True: 1, False: -1})

# Extract the year from the 'date' column
df['season'] = df['date'].dt.year

# Split the data into training (2016-2021) and testing (2022-2023)
train_df = df[df['season'].between(2016, 2021)].copy()
test_df = df[df['season'].between(2022, 2023)].copy()

# Check if the datasets are not empty
if train_df.empty or test_df.empty:
    raise ValueError("Training or testing data is empty. Please check the years in your dataset.")

# Drop the 'date' and 'year' columns as they are not used in training
train_df = train_df.drop(['date'], axis=1)
test_df = test_df.drop(['date'], axis=1)

# Separate features and target variable
X_train = train_df.drop('home_team_win', axis=1)
y_train = train_df['home_team_win']


X_test = test_df.drop('home_team_win', axis=1)
y_test = test_df['home_team_win']

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
# Identify numerical columns with missing values
num_cols_with_missing = X_train.select_dtypes(include=[np.number]).columns[
    X_train.select_dtypes(include=[np.number]).isnull().any()
].tolist()

# Initialize the median imputer
median_imputer = SimpleImputer(strategy='median')

# Fit on training data
X_train[num_cols_with_missing] = median_imputer.fit_transform(X_train[num_cols_with_missing])

# Transform test data
X_test[num_cols_with_missing] = median_imputer.transform(X_test[num_cols_with_missing])

# Identify categorical columns with missing values
cat_cols_with_missing = X_train[categorical_cols].columns[
    X_train[categorical_cols].isnull().any()
].tolist()

# Initialize the most frequent imputer
mode_imputer = SimpleImputer(strategy='most_frequent')

# Fit on training data
X_train[cat_cols_with_missing] = mode_imputer.fit_transform(X_train[cat_cols_with_missing])

# Transform test data
X_test[cat_cols_with_missing] = mode_imputer.transform(X_test[cat_cols_with_missing])

# Initialize the Target Encoder
target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=1)

# Fit the encoder on the training data
X_train_encoded = X_train.copy()
X_train_encoded[categorical_cols] = target_encoder.fit_transform(X_train[categorical_cols], y_train)

# Transform the test data
X_test_encoded = X_test.copy()
X_test_encoded[categorical_cols] = target_encoder.transform(X_test[categorical_cols])

# Initialize the model