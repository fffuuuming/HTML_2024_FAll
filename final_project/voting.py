import pandas as pd
import numpy as np


# 500 iterations
lgb_stage_1 = pd.read_csv('bagclassifier_lgb_mean_stage_1.csv')
lgb_stage_2 = pd.read_csv('bagclassifier_lgb_mean_stage_2.csv')

# 500 iterations
xgb_stage_1 = pd.read_csv('XGB_predictions_with_probabilities_stage_1.csv')
xgb_stage_2 = pd.read_csv('XGB_predictions_with_probabilities_stage_2.csv')

# 500 iterations
rf_stage_1 = pd.read_csv('rf_predictions_with_probabilities_stage_1.csv')
rf_stage_2 = pd.read_csv('rf_predictions_with_probabilities_stage_2.csv')

# 1000 iterations
tree_stage_1 = pd.read_csv('stage1_1000bag_tree_voting.csv')
tree_stage_2 = pd.read_csv('stage2_voting_tree.csv')

lgb_stage_1_prob = lgb_stage_1['probability']
lgb_stage_2_prob = lgb_stage_2['probability']

xgb_stage_1_prob = xgb_stage_1['probability']
xgb_stage_2_prob = xgb_stage_2['probability']

rf_stage_1_prob = rf_stage_1['probability']
rf_stage_2_prob = rf_stage_2['probability']

tree_stage_1_prob = tree_stage_1['voting']
tree_stage_2_prob = tree_stage_2['voting']

w1 = 500
w2 = 1000

stage_1_prob = ((lgb_stage_1_prob + xgb_stage_1_prob + rf_stage_1_prob) * w1 + tree_stage_1_prob * w2) / 2500
stage_2_prob = ((lgb_stage_2_prob + xgb_stage_2_prob + rf_stage_2_prob) * w1 + tree_stage_2_prob * w2) / 2500

# Transform probabilities to binary predictions
stage_1_predictions = stage_1_prob >= 0.5
stage_2_predictions = stage_2_prob >= 0.5

print(stage_1_predictions)
print(stage_2_predictions)
print(stage_1_prob)
print(stage_2_prob)

stage_1_output_df = pd.DataFrame({
    'id': np.arange(len(lgb_stage_1_prob)),
    'home_team_win': stage_1_predictions
})
stage_1_output_df.to_csv('voting_predictions_stage_1.csv', index=False)
print("Predictions saved to 'voting_predictions_stage_1.csv'")


stage_2_output_df = pd.DataFrame({
    'id': np.arange(len(lgb_stage_2_prob)),
    'home_team_win': stage_2_predictions
})
stage_2_output_df.to_csv('voting_predictions_stage_2.csv', index=False)
print("Predictions saved to 'voting_predictions_stage_2.csv'")

