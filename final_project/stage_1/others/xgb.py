import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import category_encoders as ce

# # Load the filled datasets
# df = pd.read_csv('train_data_filled.csv')
#
# # Convert 'date' column to datetime and extract the season/year
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['season'] = df['date'].dt.year
#
# # Drop the 'date' column since it's no longer needed for training
# df = df.drop(['date'], axis=1)
# df = df.drop(['id'], axis=1)
# # Separate the features and target variable
# X = df.drop('home_team_win', axis=1)
# y = df['home_team_win']
#
# # Convert target values from [-1, 1] to [0, 1]
# y = y.map({-1: 0, 1: 1})
#
# # Split data into training (2016-2021) and validation sets (2022-2023)
# X_train = X[df['season'].between(2016, 2021)].copy()
# y_train = y[df['season'].between(2016, 2021)].copy()
#
# X_val = X[df['season'].between(2022, 2023)].copy()
# y_val = y[df['season'].between(2022, 2023)].copy()
#
# # Ensure y_train and y_val are in the correct format
# y_train = y_train.astype(int)
# y_val = y_val.astype(int)
#
# # Ensure training and validation sets have the same columns
# X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
#
# # Load the test dataset
# df_test = pd.read_csv('same_season_test_data.csv')
# df_test = df_test.drop(['id'], axis=1)
# df_test['is_night_game'] = df_test['is_night_game'].map({True: 1, False: -1})
# # Ensure test set has the same columns as training set
# # Fill NaN values in 'is_night_game' column with 1
# df_test['is_night_game'] = df_test['is_night_game'].fillna(1).copy()
# df_test = df_test.reindex(columns=X_train.columns, fill_value=0)
#
# # Encode categorical features in the training, validation, and test sets using the same encoder
# categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
#
# # Use Target Encoder for categorical columns
# target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=1)
# X_train_encoded = target_encoder.fit_transform(X_train, y_train)
# X_val_encoded = target_encoder.transform(X_val)
# df_test_encoded = target_encoder.transform(df_test)
#
# # Convert categorical columns to numeric types to avoid errors in prediction
# for col in categorical_cols:
#     X_train_encoded[col] = X_train_encoded[col].astype(float)
#     X_val_encoded[col] = X_val_encoded[col].astype(float)
#     df_test_encoded[col] = df_test_encoded[col].astype(float)
#
# # Fill missing values in numerical columns using XGBoost Regressor
# num_cols_with_missing = X_train_encoded.select_dtypes(include=[np.number]).columns[
#     X_train_encoded.select_dtypes(include=[np.number]).isnull().any()
# ].tolist()
#
# for col in num_cols_with_missing:
#     # Define the features to be used for imputing the missing column
#     X_train_not_missing = X_train_encoded.loc[X_train_encoded[col].notnull()].drop(columns=[col])
#     y_train_not_missing = X_train_encoded.loc[X_train_encoded[col].notnull(), col]
#
#     X_train_missing = X_train_encoded.loc[X_train_encoded[col].isnull()].drop(columns=[col])
#
#     # Ensure columns match between training and missing sets
#     X_train_missing = X_train_missing.reindex(columns=X_train_not_missing.columns, fill_value=0)
#
#     # Train XGBoost Regressor to predict the missing values
#     if not X_train_missing.empty:
#         xgb_reg = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
#         xgb_reg.fit(X_train_not_missing, y_train_not_missing)
#
#         # Predict and fill missing values
#         X_train_encoded.loc[X_train_encoded[col].isnull(), col] = xgb_reg.predict(X_train_missing)
#
# # Do the same for the validation set and test set, ensuring feature names match
# for col in num_cols_with_missing:
#     X_val_missing = X_val_encoded.loc[X_val_encoded[col].isnull()].drop(columns=[col])
#     df_test_missing = df_test_encoded.loc[df_test_encoded[col].isnull()].drop(columns=[col])
#
#     # Ensure columns match between training and missing sets
#     X_val_missing = X_val_missing.reindex(columns=X_train_not_missing.columns, fill_value=0)
#     df_test_missing = df_test_missing.reindex(columns=X_train_not_missing.columns, fill_value=0)
#
#     if not X_val_missing.empty:
#         # Use the same trained regressor to predict missing values in the validation set
#         X_val_encoded.loc[X_val_encoded[col].isnull(), col] = xgb_reg.predict(X_val_missing)
#
#     if not df_test_missing.empty:
#         # Use the same trained regressor to predict missing values in the test set
#         df_test_encoded.loc[df_test_encoded[col].isnull(), col] = xgb_reg.predict(df_test_missing)
#
# # Save the encoded and imputed datasets to CSV files
# X_train_encoded.to_csv('encoded_imputed_X_train.csv', index=False)
# X_val_encoded.to_csv('encoded_imputed_X_val.csv', index=False)
# df_test_encoded.to_csv('encoded_imputed_X_test.csv', index=False)
# y_train.to_csv('encoded_y_train.csv', index=False)
# y_val.to_csv('encoded_y_val.csv', index=False)

X_train_encoded = pd.read_csv('encoded_imputed_X_train.csv')
X_val_encoded = pd.read_csv('encoded_imputed_X_val.csv')
y_train = pd.read_csv('encoded_y_train.csv').squeeze()
y_val = pd.read_csv('encoded_y_val.csv').squeeze()
df_test_encoded = pd.read_csv('encoded_imputed_X_test.csv')

# Set best parameters
# best_params = {'learning_rate': 0.014678042392570766, 'max_depth': 9, 'subsample': 0.6494212466301128,
#                'colsample_bytree': 0.6207175381052339, 'gamma': 0.0487736525239055, 'min_child_weight': 6}
best_params = {'learning_rate': 0.015322902367403895, 'max_depth': 10, 'subsample': 0.8644131755994242, 'colsample_bytree': 0.6253225895393616, 'gamma': 0.25891556173348174, 'min_child_weight': 1, 'reg_alpha': 0.529685114593104, 'reg_lambda': 0.1718234235777999}

# Combine training and validation sets for final training
X_combined = pd.concat([X_train_encoded, X_val_encoded], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)

# Train the final model with the best hyperparameters
best_model = XGBClassifier(**best_params)
best_model.fit(X_combined, y_combined)

# Predict on the validation set
y_train_pred = best_model.predict(X_train_encoded)

# Calculate final validation accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Predict on the combined training set
y_combined_pred = best_model.predict(X_combined)

# Calculate training accuracy
Total_accuracy = accuracy_score(y_combined, y_combined_pred)
print(f"Total Accuracy: {Total_accuracy:.4f}")

# Predict on the validation set
y_val_pred = best_model.predict(X_val_encoded)

# Calculate final validation accuracy
final_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Final Validation Accuracy: {final_accuracy:.4f}")

# Print the top 50 important features
# feature_importances = pd.Series(best_model.feature_importances_, index=X_combined.columns)
# top_50_features = feature_importances.sort_values(ascending=False).head(50)
# print("Top 50 Important Features:")
# print(top_50_features)

# Predict on the test set
y_test_pred = best_model.predict(df_test_encoded)

# Map predictions back to 'TRUE' and 'FALSE'
y_test_pred_mapped = np.where(y_test_pred == 1, 'TRUE', 'FALSE')

# Create the output DataFrame with 'id' and 'home_team_win'
output_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred_mapped)),
    'home_team_win': y_test_pred_mapped
})

# Save the predictions to 'predictions.csv'
output_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
