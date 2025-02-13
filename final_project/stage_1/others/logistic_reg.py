import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the original dataset
df = pd.read_csv('train_data_filled.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Drop the 'date' column as it is not used in training
df = df.drop(['date'], axis=1)
df = df.drop(['id'], axis=1)
# Combine the original training and testing data into one dataset
combined_df = df.copy()

# Separate features and target variable from the combined data
X_combined = combined_df.drop('home_team_win', axis=1)
y_combined = combined_df['home_team_win']

# Identify categorical columns
categorical_cols = X_combined.select_dtypes(include=['object']).columns.tolist()

# Impute missing values in numerical columns
num_cols_with_missing = X_combined.select_dtypes(include=[np.number]).columns[
    X_combined.select_dtypes(include=[np.number]).isnull().any()
].tolist()

median_imputer = SimpleImputer(strategy='median')

# Fit on combined data
X_combined[num_cols_with_missing] = median_imputer.fit_transform(X_combined[num_cols_with_missing])

# Impute missing values in categorical columns
cat_cols_with_missing = X_combined[categorical_cols].columns[
    X_combined[categorical_cols].isnull().any()
].tolist()

mode_imputer = SimpleImputer(strategy='most_frequent')

# Fit on combined data
X_combined[cat_cols_with_missing] = mode_imputer.fit_transform(X_combined[cat_cols_with_missing])

# Initialize the Target Encoder
target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=1)

# Fit the encoder on the combined training data
X_combined_encoded = X_combined.copy()
X_combined_encoded[categorical_cols] = target_encoder.fit_transform(X_combined[categorical_cols], y_combined)

# Identify numerical columns after encoding
numerical_cols = X_combined_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the combined training data
X_combined_encoded[numerical_cols] = scaler.fit_transform(X_combined_encoded[numerical_cols])

# Load the test dataset
df_test = pd.read_csv('same_season_test_data.csv')
df_test = df_test.drop(['id'], axis=1)
# Count the values in the 'season' column (including NaN values)
season_counts = df_test['season'].value_counts(dropna=False)

# Count the number of games each team played in every season
game_counts_home = df_test.groupby(['season', 'home_team_abbr']).size().reset_index(name='home_games')
game_counts_away = df_test.groupby(['season', 'away_team_abbr']).size().reset_index(name='away_games')

game_counts = pd.merge(game_counts_home, game_counts_away, how='outer', left_on=['season', 'home_team_abbr'], right_on=['season', 'away_team_abbr']).fillna(0)
game_counts['total_games'] = game_counts['home_games'] + game_counts['away_games']

df_test['is_night_game'] = df_test['is_night_game'].map({True: 1, False: -1})

# Fill NaN values in 'is_night_game' column with 1
df_test['is_night_game'] = df_test['is_night_game'].fillna(1).copy()

# Calculate the number of games each team should play per season (average games per season)
expected_games_per_team = game_counts.groupby('season')['total_games'].mean().to_dict()

# Fill NaN values in 'season' column with a season such that each team plays a similar number of games
def fill_season_nan(row):
    if pd.isna(row['season']):
        team = row['home_team_abbr'] if pd.notna(row['home_team_abbr']) else row['away_team_abbr']
        for season, expected_games in expected_games_per_team.items():
            current_games = game_counts[(game_counts['season'] == season) & ((game_counts['home_team_abbr'] == team) | (game_counts['away_team_abbr'] == team))]['total_games'].sum()
            if current_games < expected_games:
                return season
        return np.random.randint(2016, 2024)  # Fallback in case all seasons are filled
    return row['season']

df_test['season'] = df_test.apply(fill_season_nan, axis=1).copy()

# Identify numerical columns with missing values
num_cols_with_missing = df_test.select_dtypes(include=[np.number]).columns[
    df_test.select_dtypes(include=[np.number]).isnull().any()
].tolist()

# Initialize the median imputer for numerical columns
median_imputer = SimpleImputer(strategy='median')

# Fit and transform on numerical columns with missing values
df_test[num_cols_with_missing] = median_imputer.fit_transform(df_test[num_cols_with_missing])

# Identify categorical columns with missing values
cat_cols_with_missing = df_test.select_dtypes(include=['object']).columns[
    df_test.select_dtypes(include=['object']).isnull().any()
].tolist()

# Initialize the most frequent imputer for categorical columns
mode_imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform on categorical columns with missing values
df_test[cat_cols_with_missing] = mode_imputer.fit_transform(df_test[cat_cols_with_missing])

# Transform the test data
X_test_encoded = df_test.copy()
X_test_encoded[categorical_cols] = target_encoder.transform(df_test[categorical_cols])

# Identify numerical columns after encoding
numerical_cols = X_test_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Transform the test data
X_test_encoded[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])

# Initialize the Logistic Regression model
model = LogisticRegression(
    solver='liblinear',
    penalty='l1',  # Use 'l1' for Lasso or 'l2' for Ridge
    C=0.05,        # Smaller C increases regularization strength
    max_iter=1000,
    random_state=42
)

# Optionally, create polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
X_combined_poly = poly.fit_transform(X_combined_encoded)
X_test_poly = poly.transform(X_test_encoded)

# Fit the model on the combined training data
model.fit(X_combined_poly, y_combined)

# Predict on the combined training data
y_train_pred = model.predict(X_combined_poly)

# Calculate training accuracy
train_accuracy = accuracy_score(y_combined, y_train_pred)
print(f"Train Set Accuracy: {train_accuracy:.2f}")

# Predict on the new test data
y_test_pred = model.predict(X_test_poly)

# Map predictions back to 'TRUE' and 'FALSE'
y_test_pred_mapped = np.where(y_test_pred == 1, 'TRUE', 'FALSE')

# Create the output DataFrame with 'id' and 'home_team_win'
output_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred_mapped)),
    'home_team_win': y_test_pred_mapped
})

# Save the predictions to 'predictions.csv'
output_df.to_csv('predictions.csv', index=False)
