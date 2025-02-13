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

# import logging
# optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train, y_train):
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
        eval_metric='error',
        # verbosity=0,
        random_state=42
    )
    
    # Cross-validate the model
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

def train_xgb_with_optuna(X_train, y_train, n_trials):

    # Perform Bayesian Optimization with Optuna
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize')  # We aim to maximize the AUC score
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    # print("Best Parameters:", study.best_params)
    # print("Best Accuracy:", study.best_value)

    best_params = study.best_params


    # Retrain the model with the best parameters on the full training data
    final_model = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        objective='binary:logistic',
        eval_metric='error'
    )
    final_model.fit(X_train, y_train)

    # Predict probabilities and binary labels on the full training data
    # y_train_proba = final_model.predict_proba(X_train)[:, 1]
    y_train_pred = final_model.predict(X_train)

    return final_model, accuracy_score(y_train, y_train_pred)
    # # Calculate metrics
    # print("Training AUC:", roc_auc_score(y_train, y_train_proba))
    # print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

    # print("Classification Report:\n", classification_report(y_train, y_train_pred))

def predict_xgb(xgb, X_test):

    y_test_pred = xgb.predict(X_test)
    y_test_pred_mapped = np.where(y_test_pred == 1, 'TRUE', 'FALSE')

    output_df = pd.DataFrame({
        'id': np.arange(len(y_test_pred_mapped)),
        'home_team_win': y_test_pred_mapped
    })

    output_df.to_csv('predictions_xgb.csv', index=False)

def main():
    X_train = pd.read_csv('../training_data/combined_X.csv', low_memory=False)
    y_train = pd.read_csv('../training_data/combined_y.csv', low_memory=False)
    X_test = pd.read_csv('./encoded_imputed_X_test.csv', low_memory=False)

    n_trials = 300
    xgb_model, accuracy = train_xgb_with_optuna(X_train, y_train, n_trials)

    print(f'accuracy: {accuracy}')  

    predict_xgb(xgb_model, X_test)

if __name__ == "__main__":
    main()
