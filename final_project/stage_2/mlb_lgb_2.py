import numpy as np
import pickle
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import time

import logging
# optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train, y_train):

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'n_estimators': trial.suggest_int("n_estimators", 50, 200),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("num_leaves", 10, 40),
        'min_child_samples': trial.suggest_int("min_child_samples", 15, 100),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10.0),
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

    print("Best Parameters:", study.best_params)
    print("Best cv Accuracy:", float(1) - study.best_value)

    best_params = study.best_params

    # train_data = lgb.Dataset(X_train, label=y_train)

    final_model = LGBMClassifier( 
        **best_params,
        objective='binary', 
        random_state=42,
    )
    final_model.fit(X_train, y_train)

    with open('lgb_model.pkl', 'wb') as file:
        pickle.dump(final_model, file)
    print("Model saved as lgb_model.pkl")
    
    # Predict on the train set
    y_train_pred = final_model.predict(X_train)
    y_train_pred_class = (y_train_pred > 0.5).astype(int)

    return final_model, accuracy_score(y_train, y_train_pred_class)


def predict_lgb(lgb, X_test):
    return lgb.predict(X_test)

def save_predictions(y_test_pred, name):

    y_test_pred_mapped = np.where(y_test_pred == 1, 'TRUE', 'FALSE')

    output_df = pd.DataFrame({
        'id': np.arange(len(y_test_pred_mapped)),
        'home_team_win': y_test_pred_mapped
    })

    output_df.to_csv(f'{name}.csv', index=False)
    print(f"Predictions saved to '{name}.csv'")

def benchmark(X_train, y_train, X_test):

    traind_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': 0
    }

    cv_results = lgb.cv(
        params,
        traind_data,
        nfold=5,
        seed=42,
    )

    print(f'cv accuracy : {float(1) - cv_results["valid binary_error-mean"][-1]}')

    lgbm_clf = LGBMClassifier(**params)

    lgbm_clf.fit(X_train, y_train)
    
    y_test_pred = predict_lgb(lgbm_clf, X_train)
    print(y_test_pred)
    print(f'final train accuracy : {accuracy_score(y_train, y_test_pred)}') 

    return lgbm_clf

def main():
    X_train = pd.read_csv('../training_data/combined_X.csv', low_memory=False)
    y_train = pd.read_csv('../training_data/combined_y.csv', low_memory=False)
    X_test_stage_1 = pd.read_csv('../stage_1/full_encoded_imputed_X_test.csv', low_memory=False)
    X_test_stage_2 = pd.read_csv('./full_encoded_imputed_2024_X_test.csv', low_memory=False)


    y_train = y_train.squeeze()
    
    # benchmark
    # base_model = benchmark(X_train, y_train, X_test)
    # base_pred = predict_lgb(base_model, X_test)
    # save_predictions(base_pred, 'predictions_lgb_base_stage_2')

    start_time = time.time()

    n_trials = 200
    lgb_model, accuracy = train_lgb_with_optuna(X_train, y_train, n_trials)

    end_time = time.time()

    print(f'time taken: {end_time - start_time}')

    print(f'accuracy: {accuracy}')
    
    y_test_pred_stage_1 = predict_lgb(lgb_model, X_test_stage_1)
    save_predictions(y_test_pred_stage_1, 'predictions_lgb_stage_1')

    y_test_pred_stage_2 = predict_lgb(lgb_model, X_test_stage_2)
    save_predictions(y_test_pred_stage_2, 'predictions_lgb_stage_2')





if __name__ == "__main__":
    main()
