import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.sparse import load_npz
import joblib
from sklearn.metrics import accuracy_score

# Load datasets
X_train_encoded = pd.read_csv('encoded_imputed_X_train.csv')
X_val_encoded = pd.read_csv('encoded_imputed_X_val.csv')
y_train = pd.read_csv('encoded_y_train.csv').squeeze()
y_val = pd.read_csv('encoded_y_val.csv').squeeze()
X_test_encoded = pd.read_csv('full_encoded_imputed_2024_X_test.csv')

# Combine training and validation sets for final training
X_combined = pd.concat([X_train_encoded, X_val_encoded], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)

# Load pre-trained models
print("Loading pre-trained models...")
try:
    xgb_model = joblib.load('final_xgb_model.joblib')
    print("Loaded pre-trained XGBoost model.")
except FileNotFoundError:
    raise FileNotFoundError("XGBoost model not found. Please train and save the model as 'final_xgb_model.joblib'.")

try:
    rf_model = joblib.load('final_random_forest_model.joblib')
    print("Loaded pre-trained Random Forest model.")
except FileNotFoundError:
    raise FileNotFoundError("Random Forest model not found. Please train and save the model as 'final_random_forest_model.joblib'.")

try:
    catboost_model = joblib.load('final_catboost_model.joblib')
    print("Loaded pre-trained CatBoost model.")
except FileNotFoundError:
    raise FileNotFoundError("CatBoost model not found. Please train and save the model as 'final_catboost_model.joblib'.")

try:
    log_reg_model = joblib.load('final_logistic_model_poly.joblib')
    print("Loaded pre-trained Logistic Regression model.")
except FileNotFoundError:
    raise FileNotFoundError("Logistic Regression model not found. Please train and save the model as 'final_logistic_model_poly.joblib'.")

# Load precomputed polynomial features for Logistic Regression
print("Loading precomputed polynomial features...")
try:
    X_test_poly = load_npz('X_test_poly.npz')
    X_val_poly = load_npz('X_val_poly.npz')
    print("Loaded precomputed polynomial features.")
except FileNotFoundError:
    raise FileNotFoundError("Precomputed polynomial features not found. Please ensure 'X_test_poly.npz' and 'X_val_poly.npz' exist.")

# Get predictions
print("Generating predictions...")
X_test_rf = X_test_encoded.drop(columns=['is_night_game'])
X_val_rf = X_val_encoded.drop(columns=['is_night_game'])
print("'is_night_game' column dropped from X_combined.")
# Initial CatBoost model for feature selection
feature_selector_model = CatBoostClassifier(iterations=100, random_seed=42, task_type="GPU", logging_level="Silent")
feature_selector_model.fit(X_combined, y_combined)

# Identify important features
feature_importances = pd.Series(feature_selector_model.feature_importances_, index=X_combined.columns)
important_features = feature_importances[feature_importances > 0.001].index  # Set a threshold for feature importance

print(f"Selected {len(important_features)} important features from {X_combined.shape[1]}.")
print("Important features selected for the model:")
print(important_features.tolist())

# Keep only the important features
X_test_cb = X_test_encoded[important_features]
X_val_cb = X_val_encoded[important_features]

xgb_pred = xgb_model.predict_proba(X_test_encoded)[:, 1]
rf_pred = rf_model.predict_proba(X_test_rf)[:, 1]
catboost_pred = catboost_model.predict_proba(X_test_cb)[:, 1]
log_reg_pred = log_reg_model.predict_proba(X_test_poly)[:, 1]

# Blending: Weighted Average
blend_weights = [0.35, 0.35, 0.3, 0]  # Adjust weights as needed
final_pred_prob = (
    blend_weights[0] * xgb_pred +
    blend_weights[1] * rf_pred +
    blend_weights[2] * catboost_pred +
    blend_weights[3] * log_reg_pred
)

# Thresholding to get final predictions
final_pred = (final_pred_prob >= 0.5).astype(int)

# Map predictions back to 'TRUE' and 'FALSE'
y_test_pred_mapped = np.where(final_pred == 1, 'TRUE', 'FALSE')

# Create the output DataFrame with 'id' and 'home_team_win'
output_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred_mapped)),
    'home_team_win': y_test_pred_mapped
})

# Save the predictions to 'predictions.csv'
output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")

# Evaluate the blending model on validation set
xgb_val_pred = xgb_model.predict_proba(X_val_encoded)[:, 1]
rf_val_pred = rf_model.predict_proba(X_val_rf)[:, 1]
catboost_val_pred = catboost_model.predict_proba(X_val_cb)[:, 1]
log_reg_val_pred = log_reg_model.predict_proba(X_val_poly)[:, 1]

val_pred_prob = (
    blend_weights[0] * xgb_val_pred +
    blend_weights[1] * rf_val_pred +
    blend_weights[2] * catboost_val_pred +
    blend_weights[3] * log_reg_val_pred
)

val_pred = (val_pred_prob >= 0.5).astype(int)
validation_accuracy = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy of Blended Model: {validation_accuracy:.4f}")
