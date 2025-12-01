# ==============================
# ⚽ Football Injury Risk Dashboard (Enhanced UI + English)
# ==============================

import os
import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st


# --------------------------------------------------
# BASIC CONFIG
# --------------------------------------------------
MODEL_PATH = "models/best_model_local.joblib"

st.set_page_config(
    page_title="Player Injury Risk Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

pd.options.mode.copy_on_write = True



# --------------------------------------------------
# MODEL + DATA HELPERS
# --------------------------------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_fast_parquet(file):
    return pd.read_parquet(file, engine="pyarrow")


@st.cache_data(show_spinner=True)
def add_predictions(df, _model, batch_size=50_000):

    required_cols = [
        "player_id","season_year","minutes_played","goals","assists","yellow_cards",
        "clean_sheets","goals_conceded","nb_on_pitch","nb_in_group","matches_estimated",
        "intensity_minutes_per_match","workload_actions","past_injuries",
        "total_injuries_cumulative","past_days_missed","injury_severity_score",
        "age","height","position","team_name","age_bucket"
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    proba = np.zeros(len(df))

    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i: i+batch_size][required_cols]
        proba[i:i+batch_size] = model.predict_proba(chunk)[:,1]

    df["injury_risk"] = proba * 100
    df["risk_level"] = pd.cut(df["injury_risk"],
                             bins=[0,33,66,100],
                             labels=["Low","Medium","High"])
    return df



# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model & Data Control")
    try:
        model = load_model()
        st.success("Model loaded successfully ✓")
    except Exception as e:
        st.error(e); st.stop()

    st.write("---")
    st.markdown("Upload your processed data file below.")

    uploaded_file = st.file_uploader("📁 Upload `model_df.parquet`", type=["parquet"])



# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='color:#0077ff;'>⚽ Player Injury Risk Dashboard</h1>", unsafe_allow_html=True)
st.write(
"""
Machine learning-powered injury risk evaluation for football athletes.

**The percentage displayed = predicted chance of injury within the next match.**
Use as decision support — not medical replacement.
"""
)



# --------------------------------------------------
# LOAD & APPLY MODEL
# --------------------------------------------------
if uploaded_file is None:
    st.info("Upload your dataset to continue.")
    st.stop()

try:
    df_raw = load_fast_parquet(uploaded_file)
    df = add_predictions(df_raw, _model = model)
except Exception as e:
    st.error(e); st.stop()

st.success(f"Dataset loaded: **{df.shape[0]:,} rows / {df.shape[1]} columns**")
st.caption("Model operates on global dataset — not limited to one squad.")



# --------------------------------------------------
# FILTER PANEL — BEAUTIFIED
# --------------------------------------------------
st.markdown("## 🎛 Player Filtering Panel")
st.write("Filter the dataset to view groups, squads, and performance risk groups.")

col1,col2,col3,col4 = st.columns(4)

with col1:
    pos = st.selectbox("Position", ["All"] + sorted(df.position.unique()))

with col2:
    age = st.slider("Age Range",16,40,(18,34))

with col3:
    team = st.selectbox("Team",["All"] + sorted(df.team_name.unique()))

with col4:
    rfilter = st.selectbox("Risk Level",["All","Low","Medium","High"])


filtered = df.copy()
if pos!="All": filtered = filtered[df.position==pos]
if team!="All": filtered = filtered[df.team_name==team]
filtered = filtered[filtered.age.between(age[0],age[1])]
if rfilter!="All": filtered = filtered[filtered.risk_level==rfilter]

if filtered.empty:
    st.warning("No players match selected filters."); st.stop()



# --------------------------------------------------
# TEAM RISK DISTRIBUTION – MORE COLORFUL BAR
# --------------------------------------------------
st.markdown("## 📊 Risk Distribution Overview")
st.write("Shows injury-risk composition of the selected player group.")

colA,colB = st.columns([1.4,0.8])

with colA:
    risk_counts = filtered.risk_level.value_counts(normalize=True).reindex(["Low","Medium","High"]).fillna(0)*100
    chart_df = pd.DataFrame({"Risk":["Low","Medium","High"],"Percent":risk_counts.values})

    chart = alt.Chart(chart_df).mark_bar(size=60,cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(
        x=alt.X("Risk:N",title="Risk Classification"),
        y=alt.Y("Percent:Q",title="Percent of Players",scale=alt.Scale(domain=[0,100])),
        color=alt.Color("Risk:N", scale=alt.Scale(
            domain=["Low","Medium","High"],
            range=["#00e676","#f4d03f","#ff4d4d"]  # VIBRANT COLORS 🔥
        )),
        tooltip=["Risk","Percent"]
    )
    st.altair_chart(chart,use_container_width=True)

with colB:
    st.metric("Total Players",len(filtered))
    st.metric("High Risk",(filtered.risk_level=="High").sum())
    st.metric("Medium Risk",(filtered.risk_level=="Medium").sum())
    st.metric("Low Risk",(filtered.risk_level=="Low").sum())



# --------------------------------------------------
# RANK TABLE (TOP 50)
# --------------------------------------------------
st.markdown("## 🏆 Ranked Risk Table (Top 50)")
ranked = filtered.sort_values("injury_risk",ascending=False)
ranked["injury_risk"] = ranked.injury_risk.round(1)

st.dataframe(
    ranked[["player_id","team_name","position","age","height","injury_risk","risk_level"]].head(50),
    use_container_width=True
)



# --------------------------------------------------
# PLAYER PROFILE
# --------------------------------------------------
st.markdown("## 👤 Individual Player Report")

L,R = st.columns([1.5,1])

with L:
    selected = st.selectbox("Select Player",ranked.player_id.astype(str))
    p = ranked[ranked.player_id.astype(str)==selected].iloc[0]

    st.markdown(f"### 🏷 Player `{p.player_id}` — {p.team_name}")
    st.write(f"**Position:** {p.position}")
    st.write(f"**Age:** {p.age} | **Height:** {p.height} cm")
    st.write(f"**Age Class:** {p.age_bucket}")

    st.write("---")
    st.markdown("### 🔍 Risk Contributors (Key Features) ")
    st.write(f"• Past injuries: **{p.past_injuries}**")
    st.write(f"• Total injury days: **{p.past_days_missed}**")
    st.write(f"• Workload actions: **{p.workload_actions}**")
    st.write(f"• High-intensity minutes: **{p.intensity_minutes_per_match}**")
    st.caption("Higher workload + past injury load increases risk profile.")


with R:
    risk = p.injury_risk
    st.subheader("🧠 Injury Risk Prediction")

    st.metric("Estimated Injury Probability",f"{risk:.1f} %")
    if risk>=70: st.error("🔴 VERY HIGH RISK — Reduce load + medical evaluation recommended.")
    elif risk>=40: st.warning("🟠 ELEVATED RISK — Monitor intensity and recovery.")
    else: st.success("🟢 LOW RISK — Player appears fit & healthy.")

    st.caption("This probability refers to the upcoming match window — not long-term risk.")
