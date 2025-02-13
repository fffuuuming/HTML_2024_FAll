import pandas as pd

""""
orig_df : dataset which needs to be filled, needs to have 'date' column
missed_completed_df : use this to fill the missed pitchers
"""
def fill_pitcher(orig_df, missed_completed_df):

    filled_df = orig_df.copy()
    
    for _, row in missed_completed_df.iterrows():
        t = row['team']
        d = row['date']
        p = row['pitcher']

        condition_home = (filled_df['home_team_abbr'] == t) & (filled_df['date'] == d)
        condition_away = (filled_df['away_team_abbr'] == t) & (filled_df['date'] == d)

        if not filled_df[condition_home].empty:
            filled_df.loc[condition_home, 'home_pitcher'] = p

        elif not filled_df[condition_away].empty:
            filled_df.loc[condition_away, 'away_pitcher'] = p

    return filled_df

def main():

    orig_df = pd.read_csv('dataset_you_want_to_fill.csv')
    missed_completed_df = pd.read_csv('fine_missed_completed.csv')

    filled_orig_df = fill_pitcher(orig_df, missed_completed_df)
    
    # use this to check the correctness, expect : (29, 30)
    print(missed_completed_df['pitcher'].isna().sum())
    print(filled_orig_df['home_pitcher'].isna().sum() + filled_orig_df['away_pitcher'].isna().sum())

    filled_orig_df.to_csv('updated_filled_pitcher.csv', index=False)


if __name__ == "__main__":
    main()