import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# 1. Config
# -------------------------
TARGET_COL = "injured_next_season"
MODEL_PATH = "models/best_model_lightgbm.joblib"
DATA_PATH = "data/processed/model_df.parquet"

# -------------------------
# 2. Load model & data (cached)
# -------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_parquet(path)

model = load_model()
df = load_data()

# -------------------------
# 3. Helper: predict risk for one player
# -------------------------
def predict_next_match_risk(model, dataframe, player_id):
    player_rows = dataframe[dataframe["player_id"] == player_id]

    if len(player_rows) == 0:
        return None, f"Player {player_id} not found in dataset."

    # Use most recent season row
    row = player_rows.sort_values("season_year").iloc[-1]

    # Drop target if present
    if TARGET_COL in row.index:
        X = row.drop(TARGET_COL).to_frame().T
    else:
        X = row.to_frame().T

    # Predict probability
    prob = model.predict_proba(X)[0][1]

    if prob > 0.7:
        level = "HIGH"
        color = "🔴"
    elif prob > 0.4:
        level = "MEDIUM"
        color = "🟠"
    else:
        level = "LOW"
        color = "🟢"

    result = {
        "player_id": int(row["player_id"]),
        "team_name": row.get("team_name", "Unknown"),
        "position": row.get("position", "Unknown"),
        "season_year": int(row["season_year"]),
        "injury_risk_next_match": float(prob),
        "risk_level": level,
        "risk_icon": color,
    }
    return result, None

# -------------------------
# 4. Streamlit layout
# -------------------------
st.set_page_config(
    page_title="Football Injury Risk Dashboard",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Injury Risk Dashboard")
st.markdown(
    """
    This dashboard uses a machine learning model to estimate **injury risk for football players**
    before the next match.

    **What you can do:**
    - Inspect injury risk for a single player
    - See the top players with highest predicted risk
    - Explore the underlying dataset
    """
)

# Sidebar
st.sidebar.header("Navigation")
view = st.sidebar.radio(
    "Select view",
    ["🔍 Single Player Risk", "📈 Top Risk Players", "📊 Dataset Overview"]
)

# -------------------------
# View 1: Single Player
# -------------------------
if view == "🔍 Single Player Risk":
    st.subheader("🔍 Single Player Injury Risk")

    unique_players = sorted(df["player_id"].unique())
    player_id = st.sidebar.selectbox("Select a player_id", unique_players)

    if st.sidebar.button("Predict risk"):
        result, error = predict_next_match_risk(model, df, player_id)
        if error:
            st.error(error)
        else:
            st.markdown(
                f"""
                ### Player {result['player_id']} — {result['team_name']}
                **Position:** {result['position']}
                **Season:** {result['season_year']}

                **Predicted Injury Risk (next match):**
                {result['risk_icon']} **{result['risk_level']}**
                _({result['injury_risk_next_match']:.2%} probability)_
                """
            )

# -------------------------
# View 2: Top Risk Players
# -------------------------
elif view == "📈 Top Risk Players":
    st.subheader("📈 Top Predicted Risk Players")

    n_top = st.sidebar.slider("Number of players to show", 5, 50, 10)

    # Take latest season per player
    df_latest = df.sort_values("season_year").groupby("player_id").tail(1)

    # Drop target if present
    if TARGET_COL in df_latest.columns:
        X_latest = df_latest.drop(columns=[TARGET_COL])
    else:
        X_latest = df_latest.copy()

    probs = model.predict_proba(X_latest)[:, 1]
    df_latest = df_latest.copy()
    df_latest["injury_risk_next_match"] = probs

    def label_risk(p):
        if p > 0.7:
            return "HIGH"
        elif p > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    df_latest["risk_level"] = df_latest["injury_risk_next_match"].apply(label_risk)

    df_top = df_latest.sort_values("injury_risk_next_match", ascending=False).head(n_top)

    st.dataframe(
        df_top[["player_id", "team_name", "position", "season_year",
                "injury_risk_next_match", "risk_level"]]
        .reset_index(drop=True)
        .style.format({"injury_risk_next_match": "{:.2%}"})
    )

# -------------------------
# View 3: Dataset Overview
# -------------------------
elif view == "📊 Dataset Overview":
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Unique players", df["player_id"].nunique())
    with col3:
        st.metric("Seasons", df["season_year"].nunique())

    st.write("Columns:")
    st.write(list(df.columns))

    st.write("Sample (first 20 rows):")
    st.dataframe(df.head(20))
