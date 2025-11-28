import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess
from features import add_features
import pandas as pd

def train_model(input_csv:str, save_path="models/model.joblib"):
    df = pd.read_csv(input_csv)

    df = preprocess(df)
    df = add_features(df)

    target = "injured"   # <-- pas aan naar juiste doeldimensie
    X = df.drop(columns=[target])
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=22,
        class_weight="balanced"
    )
    model.fit(X_train,y_train)

    joblib.dump(model, save_path)
    print("✔ Model saved →",save_path)

if __name__=="__main__":
    train_model("data/clean_players.csv")
