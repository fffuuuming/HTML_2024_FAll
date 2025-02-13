import numpy as np
import pandas as pd
import optuna
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.datasets import make_classification

import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

class CustomBaggingClassifier:
    def __init__(self, 
            base_estimator, 
            n_estimators=10, 
            max_samples=1.0, 
            random_state=None
        ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.models = []
        self.predictions = []
        self.mean_predictions = 0

    def fit(self, X_train, y_train):
        # reinitialize models
        self.models = []

        n_samples = int(self.max_samples * X_train.shape[0])
        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            X_sample, y_sample = resample(X_train, y_train, n_samples=n_samples, random_state=rng.integers(0, 1e6))
            model = self.base_estimator()
            model.fit(X_sample, y_sample)
            self.models.append(model)
    
    def predict(self, X):

        predictions = np.array([model.predict(X) for model in self.models])
        self.predictions = predictions
        self.mean_predictions = np.mean(predictions, axis=0)

        print(f'mean predictions : {self.mean_predictions}')

        # Use majority voting
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote

    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)


def objective(trial, X_train, y_train):

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("num_leaves", 5, 25),
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

    print("Best Parameters:", study.best_params)
    print("Best error:", study.best_value)

    best_params = study.best_params

    # single model
    single_model = LGBMClassifier(
        **best_params,
        objective='binary',
        random_state=42
    )
    single_model.fit(X_train, y_train)

    y_train_pred = single_model.predict(X_train)

    print(f"Single Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")

    X_sub_train, X_val, y_sub_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    bagging_clf = CustomBaggingClassifier(base_estimator=lambda: LGBMClassifier(**best_params,objective='binary', random_state=42),
                                          n_estimators=500, max_samples=0.8, random_state=42)
    bagging_clf.fit(X_sub_train, y_sub_train)


    # Evaluate the model
    y_pred = bagging_clf.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")

    bagging_clf = CustomBaggingClassifier(base_estimator=lambda: LGBMClassifier(**best_params,objective='binary', random_state=42),
                                          n_estimators=500, max_samples=0.8, random_state=42)
    bagging_clf.fit(X_train, y_train)

    # predict stage_1 data
    y_pred = bagging_clf.predict(X_test_1)

    bagging_mean = bagging_clf.mean_predictions

    output_df = pd.DataFrame({
        'id': np.arange(len(X_test_1)),
        'probability': bagging_mean
    })

    output_df.to_csv('bagging_lgb_mean_stage_1.csv', index=False)
    print("Predictions saved to 'bagging_lgb_mean_stage_1.csv'")

    # predict stage_2 data
    y_pred = bagging_clf.predict(X_test_2)

    bagging_mean = bagging_clf.mean_predictions

    output_df = pd.DataFrame({
        'id': np.arange(len(X_test_2)),
        'home_team_win': bagging_mean
    })

    output_df.to_csv('bagging_lgb_mean_stage_2.csv', index=False)
    print("Predictions saved to 'bagging_lgb_mean_stage_2.csv'")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
