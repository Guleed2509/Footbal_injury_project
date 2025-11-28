import joblib, pandas as pd
from preprocess import preprocess
from features import add_features

def predict(csv_input:str, model_path="models/model.joblib"):

    df = pd.read_csv(csv_input)
    df = preprocess(df)
    df = add_features(df)

    model = joblib.load(model_path)
    df["risk"] = model.predict_proba(df)[:,1]

    return df[["player_id","risk","position","team_name"]].sort_values("risk",ascending=False)
