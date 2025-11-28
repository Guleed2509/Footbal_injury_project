import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, duplicate rows, normalize types.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)

    # Categorical encoding proof-of-concept
    if 'position' in df: df['position'] = df['position'].astype('category')
    if 'team_name' in df: df['team_name'] = df['team_name'].astype('category')

    return df
