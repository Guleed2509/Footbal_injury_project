import pandas as pd

def add_features(df: pd.DataFrame)->pd.DataFrame:
    df = df.copy()

    # Workload example features
    df["minutes_30d"] = df.groupby("player_id")["minutes_played"].rolling(30).sum().reset_index(level=0,drop=True)
    df["games_last_30d"] = df.groupby("player_id")["match_id"].rolling(30).count().reset_index(level=0,drop=True)

    df["intensity_score"] = (
        df["tackles"]*0.4 +
        df["duels"]*0.3 +
        df["distance_covered_km"]*0.3
    )

    # Past injury risk indicator
    df["injury_score_history"] = (
        df["injuries_total"]*0.7 +
        df["days_missed_total"]*0.3
    )

    df.fillna(0,inplace=True)
    return df
