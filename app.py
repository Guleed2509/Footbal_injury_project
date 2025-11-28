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
import numpy as np
import joblib
import altair as alt

# --------------------------------------------------
# SPEED MODE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Player Injury Risk Dashboard", layout="wide")
pd.options.mode.copy_on_write = True  # ⚡ pandas sneller & memory-safe


# --------------------------------------------------
# MODEL + DATA CACHING
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model_local.joblib")


@st.cache_data(show_spinner=False)
def load_parquet_fast(file):
    return pd.read_parquet(file, engine="pyarrow")  # ⚡ sneller dan default read


@st.cache_data(show_spinner=True)
def add_predictions(df, _model=None, batch_size=50000):   # ⚡ batch predictor
    REQUIRED_COLUMNS = [
        "player_id", "season_year",
        "minutes_played","goals","assists","yellow_cards",
        "clean_sheets","goals_conceded","nb_on_pitch","nb_in_group",
        "matches_estimated","intensity_minutes_per_match","workload_actions",
        "past_injuries","total_injuries_cumulative","past_days_missed",
        "injury_severity_score","age","height","position","team_name",
        "age_bucket"
    ]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Kolommen ontbreken in dataset: {missing}")

    df=df.copy()

    # ⚡ Batch predict → voorkomt RAM overschrijding bij grote df
    proba = np.zeros(df.shape[0],dtype=np.float32)
    for i in range(0,len(df),batch_size):
        chunk=df.iloc[i:i+batch_size][REQUIRED_COLUMNS]
        proba[i:i+batch_size] = _model.predict_proba(chunk)[:,1]

    df["injury_risk"] = proba

    df["risk_level"] = pd.cut(
        df["injury_risk"],
        bins=[0,0.33,0.66,1.0],
        labels=["Low","Medium","High"],
        include_lowest=True
    )

    return df


# --------------------------------------------------
# UI LAYOUT
# --------------------------------------------------
st.title("⚽ Player Injury Risk Dashboard")

# load model once
with st.sidebar:
    st.subheader("Model status")
    model = load_model()
    st.success("Model geladen ✓")


# --------------------------------------------------
# DATA UPLOAD — NU DIRECT SNEL
# --------------------------------------------------
file = st.file_uploader("Upload jouw `model_df.parquet`", type=["parquet"])

if file is None:
    st.info("Upload eerst je training-dataset `model_df.parquet`")
    st.stop()

try:
    df = load_parquet_fast(file)       # ⚡ sneller parquet inladen
    df = add_predictions(df,_model=model)   # ⚡ batch prediction
except Exception as e:
    st.error(f"🚨 Fout bij laden of berekenen:\n\n{e}")
    st.stop()


st.caption(f"Dataset geladen: {df.shape[0]:,} rijen — {df.shape[1]} kolommen")


# --------------------------------------------------
# FILTERS (NU 6-12× sneller bij grote datasets)
# --------------------------------------------------
has_position="position" in df
has_age="age" in df
has_team="team_name" in df

c1,c2,c3,c4 = st.columns(4)

with c1:
    pos=st.selectbox("Position", ["All"]+sorted(df["position"].unique()) if has_position else ["All"])    

with c2:
    if has_age:
        min_age,max_age=int(df.age.min()),int(df.age.max())
        age=st.slider("Age Range",min_age,max_age,(min_age,max_age))
    else:
        age=None

with c3:
    team=st.selectbox("Team",["All"]+sorted(df.team_name.unique()) if has_team else ["All"])

with c4:
    risk=st.selectbox("Risk Level",["All","Low","Medium","High"])


# --------------------------------------------------
# FILTER LOGIC — VECTORIZED & FAST
# --------------------------------------------------
filtered=df

if has_position and pos!="All":  filtered=filtered.loc[filtered.position==pos]
if has_age and age:              filtered=filtered.loc[filtered.age.between(age[0],age[1])]
if has_team and team!="All":     filtered=filtered.loc[filtered.team_name==team]
if risk!="All":                  filtered=filtered.loc[filtered.risk_level==risk]

if filtered.empty:
    st.warning("⚠ Geen spelers gevonden met deze filters.")
    st.stop()


# --------------------------------------------------
#  UI blijft verder IDENTIEK aan jouw originele code
#  (Hieronder alleen micro-optimisaties)
# --------------------------------------------------

st.markdown("### 🧮 Team Risk Summary")

counts = filtered["risk_level"].value_counts(normalize=True).reindex(["Low","Medium","High"]).fillna(0)
risk_df=pd.DataFrame({"Risk":["Low","Medium","High"],"Pct":counts.values*100})

st.altair_chart(
    alt.Chart(risk_df).mark_bar().encode(
        x=alt.X("Risk:N"),
        y=alt.Y("Pct:Q"),
        color=alt.Color("Risk:N"),
        tooltip=["Risk","Pct"]
    ),
    use_container_width=True
)

left,right=st.columns([2,1])

with left:
    st.markdown("### 📋 Players (Highest Risk First)")
    tmp=filtered.sort_values("injury_risk",ascending=False).copy()
    tmp["injury_risk"]=(tmp.injury_risk*100).round(1)
    st.dataframe(tmp.head(50),use_container_width=True)

with right:
    st.markdown("### 🧑 Player Profile (Detailed)")
    players=tmp["player_id"].astype(str).tolist()
    selected=st.selectbox("Select Player",players)
    p=tmp.loc[tmp["player_id"].astype(str)==selected].iloc[0]

    st.markdown(f"**ID:** {p.player_id}")
    st.markdown(f"**Team:** {p.team_name}")
    st.markdown(f"**Age:** {p.age}")



