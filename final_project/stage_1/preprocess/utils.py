import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer
from datetime import datetime, timedelta

def unique_team_season(df, column):
    team_abbr = df[f'{column}'].str.split('_').str[0]
    unique_values = team_abbr.unique()
    unique_values_count = team_abbr.nunique()
    # unique_values = df[f'{column}'].unique()
    # print(unique_values)
    return unique_values, unique_values_count


def unique_column(df, column):

    unique_values_count = df[f'{column}'].nunique()
    unique_values = df[f'{column}'].unique()

    return unique_values_count, unique_values

def sort_home_away():
    df = pd.read_csv('updated_train_data.csv')

    # sort by host/away, date
    sorted_host_df = df.sort_values(by=['host_team_abbr', 'date'])
    sorted_away_df = df.sort_values(by=['away_team_abbr', 'date'])

    sorted_host_df.to_csv('sorted_host_train_data.csv', index=False)
    sorted_away_df.to_csv('sorted_away_train_data.csv', index=False)
    
    return sorted_host_df, sorted_away_df

def sort_teams_date(h_df, a_df):
    h_col = ['home_team_abbr', 'home_pitcher', 'home_pitcher_rest', 'date', 'season']
    a_col = ['away_team_abbr', 'away_pitcher', 'away_pitcher_rest', 'date', 'season']

    ht = h_df[h_col]
    at = a_df[a_col]

    ht.columns = ['team', 'pitcher', 'pitcher_rest', 'date', 'season']
    at.columns = ['team', 'pitcher', 'pitcher_rest', 'date', 'season']

    team_df = pd.concat([ht, at])

    sorted_team_df = team_df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    return sorted_team_df

def fill_back(df):
    pass

def main():

    df = pd.read_csv('sorted_team_date.csv')
    # print(df.shape)

    # fill missed pitcher_rest with 0
    df['pitcher_rest'] = df['pitcher_rest'].fillna(0)

    _, unique_teams = unique_column(df, 'team') # 30
    _, unique_seasons = unique_column(df, 'season')

    processed_df = pd.DataFrame(columns=df.columns)
    missed_df = pd.DataFrame(columns=df.columns)

    max_range = 10

    # by team
    for i, t in enumerate(unique_teams):

        team_df = df[(df['team'] == t)]

        missed_col = team_df[team_df['pitcher'].isna()]

        if missed_df.empty:
            missed_df = missed_col
        else:
            missed_df = pd.concat([missed_df, missed_col], ignore_index=False)


        for index, row in missed_col.iterrows():

            missed_date = row['date']
            rest_days = row['pitcher_rest']

            # both pitcher & pitcher_rest
            if(rest_days == 0):
                for i in range(max_range):
                    if (index + i) not in team_df.index: 
                        continue

                    p = team_df.loc[index + i, 'pitcher']
                    d = team_df.loc[index + i, 'date']
                    r = team_df.loc[index + i, 'pitcher_rest']
                    if pd.notna(p) and pd.notna(d) and (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=r)).strftime("%Y-%m-%d") == missed_date:
                        team_df.loc[index, 'pitcher'] = p
                        missed_df.loc[index, 'pitcher'] = p
                        break

                continue

            last_pitch_date = (datetime.strptime(missed_date, "%Y-%m-%d") - timedelta(days=rest_days)).strftime("%Y-%m-%d")
            last_pitch = team_df[team_df['date'] == last_pitch_date]

            candidate_pitchers = last_pitch['pitcher'].tolist()

            # double header
            if last_pitch.shape[0] == 2: continue

            # non double header
            team_df.loc[index, 'pitcher'] = candidate_pitchers[0]
            missed_df.loc[index, 'pitcher'] = candidate_pitchers[0]

        if processed_df.empty:
            processed_df = team_df
        else:
            processed_df = pd.concat([processed_df, team_df], ignore_index=True)


    pd.set_option('display.max_rows', None)
    # print(f'\n missed_df :\n{missed_df}\n')
    # print(f'\n missed_df :\n{missed_df}\n')
    # pd.reset_option('display.max_rows')

    if not os.path.exists('fine_processed.csv'):
        processed_df.to_csv('fine_processed.csv', index=False)

    if not os.path.exists('fine_missed.csv'):
        missed_df.to_csv('fine_missed.csv', index=False)
    
    

if __name__ == "__main__":
    main()