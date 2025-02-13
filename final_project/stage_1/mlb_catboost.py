import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


import warnings
warnings.filterwarnings('ignore')

# from ray import tune
# # from ray.tune.suggest.hyperopt import HyperOptSearch ------> deprecated
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.schedulers import ASHAScheduler


X_train = pd.read_csv('./combined_X.csv', low_memory=False)
y_train = pd.read_csv('./combined_y.csv', low_memory=False)
X_test = pd.read_csv('./encoded_imputed_X_test.csv', low_memory=False)


# approach 3 : optuna

# X_sub_train, X_val, y_sub_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def objective(trial):
    # Suggest values for hyperparameters
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }

    # Train an XGBoost classifier
    model = XGBClassifier(
        **param, 
        use_label_encoder=False, 
        objective='binary:logistic', 
        eval_metric='error'
    )
    
    # Cross-validate the model
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()


# Perform Bayesian Optimization with Optuna
study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize')  # We aim to maximize the AUC score
study.optimize(objective, n_trials=300)

# Display the best parameters and the best score
print("Best Parameters:", study.best_params)
print("Best Accuracy:", study.best_value)

best_params = study.best_params

# Retrain the model with the best parameters on the full training data
best_model = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    objective='binary:logistic',
    eval_metric='error'
)
best_model.fit(X_train, y_train)

# Predict probabilities and binary labels on the full training data
y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_train_pred = best_model.predict(X_train)

# Calculate metrics
print("Training AUC:", roc_auc_score(y_train, y_train_proba))
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

print("Classification Report:\n", classification_report(y_train, y_train_pred))

y_test_pred = best_model.predict(X_test)
y_test_pred_mapped = np.where(y_test_pred == 1, 'TRUE', 'FALSE')

output_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred_mapped)),
    'home_team_win': y_test_pred_mapped
})


output_df.to_csv('predictions_catboost.csv', index=False)
print("Predictions saved to 'predictions.csv'")


# other's best_param
best_params = {
    'learning_rate': 0.015322902367403895,
    'max_depth': 10, 'subsample': 0.8644131755994242, 
    'colsample_bytree': 0.6253225895393616, 
    'gamma': 0.25891556173348174, 
    'min_child_weight': 1, 
    'reg_alpha': 0.529685114593104, 
    'reg_lambda': 0.1718234235777999
}