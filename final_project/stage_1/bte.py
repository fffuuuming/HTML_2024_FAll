import gc
import numpy as np 
import pandas as pd
# import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


"""
Initialize : Betaencoder(target_category_column)
fit        : fit(df, target)
transform  : transform(df, statistics, N_min)
"""
class BetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta =  beta_prior + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value


cat_cols = [
    'home_team_abbr', 'away_team_abbr','home_team_season', 'away_team_season'
]

target = 'home_team_win'

df = pd.read_csv('./sorted_team_train_data.csv', usecols=cat_cols+[target])

df['home_team_abbr'] = df['home_team_abbr'].astype(str)
df['away_team_abbr'] = df['away_team_abbr'].astype(str)
df['home_team_season'] = df['home_team_season'].astype(str)
df['away_team_season'] = df['away_team_season'].astype(str)
df.fillna('', inplace=True)

print(df)

# label encoding
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])

print(df)

N_min = 1000
feature_cols = []

# encode variables
for c in cat_cols:

    # fit encoder
    be = BetaEncoder(c)
    be.fit(df, target)

    # mean
    feature_name = f'{c}_mean'
    df[feature_name] = be.transform(df, 'mean', N_min)
    feature_cols.append(feature_name)

    # mode
    feature_name = f'{c}_mode'
    df[feature_name] = be.transform(df, 'mode', N_min)
    feature_cols.append(feature_name)
    
    # median
    feature_name = f'{c}_median'
    df[feature_name] = be.transform(df, 'median', N_min)
    feature_cols.append(feature_name)    

    # var
    feature_name = f'{c}_var'
    df[feature_name] = be.transform(df, 'var', N_min)
    feature_cols.append(feature_name)        
    
    # skewness
    feature_name = f'{c}_skewness'
    df[feature_name] = be.transform(df, 'skewness', N_min)
    feature_cols.append(feature_name)    
    
    # kurtosis
    feature_name = f'{c}_kurtosis'
    df[feature_name] = be.transform(df, 'kurtosis', N_min)
    feature_cols.append(feature_name)

print(df[[target]+feature_cols].head())

df.to_csv('bte_test.csv')


print(df.describe())

has_nan_columns = df.isnull().any()
print(has_nan_columns)

has_inf = np.isinf(df).any().any()
print(f"DataFrame has inf values: {has_inf}")
