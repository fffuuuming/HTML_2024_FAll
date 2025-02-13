import numpy as np
import pandas as pd
import optuna
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.datasets import make_classification

import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def objective(trial, X_train, y_train):

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("num_leaves", 2, 25),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 50),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'n_estimators': trial.suggest_int("n_estimators", 100, 300),
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


if __name__ == "__main__":
    
    start_time = time.time()

    X_train = pd.read_csv('../training_data/combined_X.csv', low_memory=False)
    y_train = pd.read_csv('../training_data/combined_y.csv', low_memory=False)
    X_test_1 = pd.read_csv('../stage_1/full_encoded_imputed_X_test.csv', low_memory=False)
    X_test_2 = pd.read_csv('./full_encoded_imputed_2024_X_test.csv', low_memory=False)

    y_train = y_train.squeeze()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)

    best_params = study.best_params

    lgbm_clf = LGBMClassifier(
        **best_params,
        random_state=42
    )

    X_sub_train, X_val, y_sub_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    bagging_clf = BaggingClassifier(
        estimator=lgbm_clf, 
        n_estimators=500, 
        max_samples=0.8, 
        bootstrap=True,
        random_state=42,
        # n_jobs=-1
    )
    bagging_clf.fit(X_sub_train, y_sub_train)

    # Evaluate the model
    y_pred = bagging_clf.predict(X_val)

    print('train on full training data')
    # train on full traing data
    bagging_clf = BaggingClassifier(
        estimator=lgbm_clf, 
        n_estimators=500, 
        max_samples=0.8, 
        bootstrap=True,
        random_state=42,
        # n_jobs=-1
    )
    bagging_clf.fit(X_train, y_train)
    
    joblib.dump(bagging_clf, "custom_lgb_bagging_model.joblib")
    print("fit completed")

    # predict stage_1 data
    predictions = np.array([estimator.predict(X_test_1) for estimator in bagging_clf.estimators_])
    mean_predictions = np.mean(predictions, axis=0)
    print(f"Mean Predictions : {mean_predictions}")

    output_df = pd.DataFrame({
        'id': np.arange(len(X_test_1)),
        'probability': mean_predictions
    })

    output_df.to_csv('bagclassifier_lgb_mean_stage_1.csv', index=False)
    print("Predictions saved to 'bagclassifier_lgb_mean_stage_1.csv'")

    # predict stage_2 data
    predictions = np.array([estimator.predict(X_test_2) for estimator in bagging_clf.estimators_])
    mean_predictions = np.mean(predictions, axis=0)
    print(f"Mean Predictions : {mean_predictions}")

    output_df = pd.DataFrame({
        'id': np.arange(len(X_test_2)),
        'probability': mean_predictions
    })

    output_df.to_csv('bagclassifier_lgb_mean_stage_2.csv', index=False)
    print("Predictions saved to 'bagclassifier_lgb_mean_stage_2.csv'")


    print(f"Evaluation : Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("Optuna best Parameters:", study.best_params)
    print("Optuna best error:", study.best_value)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
