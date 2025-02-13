import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train, y_train):

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("num_leaves", 2, 25),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 10),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'n_estimators': trial.suggest_int("n_estimators", 100, 500),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 8.0),
        'verbosity': 0
    }

    callback = [lgb.log_evaluation(0), lgb.early_stopping(stopping_rounds=10)]
    
    traind_data = lgb.Dataset(X_train, label=y_train)

    cv_results = lgb.cv(
        params,
        traind_data,
        nfold=3,
        callbacks=callback,
        seed=42,
    )

    return cv_results['valid binary_error-mean'][-1]

def train_lgb_with_optuna(X_train, y_train, n_trials):

    # Perform Bayesian Optimization with Optuna
    study = optuna.create_study(direction="minimize") 
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    # print("Best Parameters:", study.best_params)
    # print("Best error:", study.best_value)

    best_params = study.best_params

    train_data = lgb.Dataset(X_train, label=y_train)

    final_model = LGBMClassifier( 
        **best_params,
        objective='binary', 
        random_state=42,
    )
    final_model.fit(X_train, y_train)
    
    # Predict on the train set
    y_train_pred = final_model.predict(X_train)
    y_train_pred_class = (y_train_pred > 0.5).astype(int)

    return final_model, accuracy_score(y_train, y_train_pred_class)


def predict_lgb(lgb, X_test):

    y_test_pred = lgb.predict(X_test)
    y_test_pred_class = (y_test_pred > 0.5).astype(int)
    y_test_pred_mapped = np.where(y_test_pred_class == 1, 'TRUE', 'FALSE')

    output_df = pd.DataFrame({
        'id': np.arange(len(y_test_pred_mapped)),
        'home_team_win': y_test_pred_mapped
    })

    output_df.to_csv('predictions_lgb.csv', index=False)
    # print("Predictions saved to 'predictions_lgb.csv'")

def main():
    X_train = pd.read_csv('../training_data/combined_X.csv', low_memory=False)
    y_train = pd.read_csv('../training_data/combined_y.csv', low_memory=False)
    X_test = pd.read_csv('./encoded_imputed_X_test.csv', low_memory=False)

    y_train = y_train.squeeze()

    n_trials = 200
    lgb_model, accuracy = train_lgb_with_optuna(X_train, y_train, n_trials)

    print(f'accuracy: {accuracy}')  
    
    predict_lgb(lgb_model, X_test)


if __name__ == "__main__":
    main()
