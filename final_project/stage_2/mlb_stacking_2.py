import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score

# Import relevant libraries for Stacking Ensembles
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier

# Import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
# from math import sqrt

from mlb_xgb_2 import train_xgb_with_optuna
from mlb_lgb_2 import train_lgb_with_optuna


def evaluate_model(model, X_test, y_test):

    # # Training data evaluation
    # train_pred = model.predict(X_train)
    # print('----------------------------------------------\n')
    # print('Train Accuracy: ', round(accuracy_score(y_true=y_train, y_pred=train_pred), 6))
    # print('Train Precision: ', round(precision_score(y_true=y_train, y_pred=train_pred, average='weighted'), 6))
    # print('Train Recall: ', round(recall_score(y_true=y_train, y_pred=train_pred, average='weighted'), 6))
    # print('Train F1-Score: ', round(f1_score(y_true=y_train, y_pred=train_pred, average='weighted'), 6))
    # print('----------------------------------------------\n')

    # Testing data evaluation
    test_pred = model.predict(X_test)
    print('Test Accuracy: ', round(accuracy_score(y_true=y_test, y_pred=test_pred), 6))
    print('Test Precision: ', round(precision_score(y_true=y_test, y_pred=test_pred, average='weighted'), 6))
    print('Test Recall: ', round(recall_score(y_true=y_test, y_pred=test_pred, average='weighted'), 6))
    print('Test F1-Score: ', round(f1_score(y_true=y_test, y_pred=test_pred, average='weighted'), 6))
    print('----------------------------------------------\n')

    # Confusion Matrix
    print('Confusion Matrix:\n', confusion_matrix(y_true=y_test, y_pred=test_pred))

X_train = pd.read_csv('./combined_X.csv', low_memory=False)
y_train = pd.read_csv('./combined_y.csv', low_memory=False)
X_test = pd.read_csv('./encoded_imputed_2024_X_test.csv', low_memory=False)


xgb_trials = 50
xgb_model, accuracy_xgb = train_xgb_with_optuna(X_train, y_train, xgb_trials)
print(f'xgb accuracty : {accuracy_xgb}\n')

lgb_trials = 50
lgb_model, accuracy_lgb = train_lgb_with_optuna(X_train, y_train, lgb_trials)
print(f'lgb accuracy: {accuracy_lgb}\n')

# check diversity
# print('Feature importance for XGB:\n', xgb_model.feature_importances_)
# print('Feature importance for LGB:\n', light_gbm.feature_importances_)
# print('Feature importance for LGB:\n', cat_model.feature_importances_)

# preds_xgb = xgb_model.predict(X_val)
# preds_lgb = light_gbm.predict(X_val)
# preds_cat = cat_model.predict(X_val)

# correlation = np.corrcoef(preds_xgb, preds_lgb)
# print('Correlation between XGB and LGB predictions:', correlation[0, 1])


# Store all the base models in a list
estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    # ('cat', cat_model),
]

X_sub_train, X_val, y_sub_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=101)
y_sub_train = y_sub_train.squeeze()
y_val = y_val.squeeze()

# Create the stacked model with the base models and Elastic Net as the meta-model
print('stacking\n')
stack = StackingClassifier(
    estimators=estimators, 
    final_estimator=HistGradientBoostingClassifier(),
    passthrough=True
)

cv_scores = cross_val_score(stack, X_train, y_train, cv=5, scoring='accuracy')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')

stack.fit(X_train, y_train)

y_test_pred = stack.predict(X_test)
y_test_pred_mapped = np.where(y_test_pred == 1, 'TRUE', 'FALSE')

output_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred_mapped)),
    'home_team_win': y_test_pred_mapped
})

# construct actual predictions on test data
output_df.to_csv('predictions_stacking_stage_2.csv', index=False)
print("Predictions saved to 'predictions_stacking_stage_2.csv'")
