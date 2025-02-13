def grid_search(X, y, estimator, param_grid):

    # set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1
    )

    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['Disbursed'].values)
        cvresult = xgb.cv(
            xgb_param,
            xgtrain,
            num_boost_round=alg.get_params()['n_estimators'],
            nfold=cv_folds,
            metrics='auc',
            early_stopping_rounds=early_stopping_rounds,
            show_progress=False
        )
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report
    print("\nModel Report")
    print(f"Accuracy : {metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions):.4g}")
    print(f"AUC Score (Train): {metrics.roc_auc_score(dtrain['Disbursed'].values, dtrain_predprob):f}")

    # Feature importance
    feat_imp = pd.Series(alg.booster_.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()



    # approach 1 : grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01],
    'max_depth': [2, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.8, 1.0]
}

# best_param, best_acc = grid_search(X, y, XGBClassifier(), param_grid)
# print(f'best parameters: {best_param}, best accuracy: {best_acc}\n')


# approach 2 : csdn

xgb1 = XGBClassifier(
    learning_rate=0.01,
    n_estimators=500,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    seed=27 
)

# modelfit(xgb1, X, y)