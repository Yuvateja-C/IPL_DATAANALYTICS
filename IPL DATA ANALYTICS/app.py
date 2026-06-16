import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    st.error("Plotly is not installed. Run: pip install plotly")
    st.stop()

from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="IPL Analytics & Prediction Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

def inject_custom_css():
    theme_css = f"""
    <style>
    :root {{
        --bg: {"#09090b" if IS_DARK else "#ffffff"};
        --card: {"#0c0c0f" if IS_DARK else "#ffffff"};
        --border: {"#1e1e24" if IS_DARK else "#e4e4e7"};
        --text: {"#fafafa" if IS_DARK else "#09090b"};
        --text-muted: #71717a;
        --accent: #2563eb;
    }}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }}
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text);
    }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

inject_custom_css()

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        matches = pd.read_csv(os.path.join(base_dir, "data", "matches.csv"))
        deliveries = pd.read_csv(os.path.join(base_dir, "data", "deliveries.zip"))
    except Exception:
        try:
            matches = pd.read_csv("data/matches.csv")
            deliveries = pd.read_csv("data/deliveries.zip")
        except Exception as e:
            st.error(f"Error loading datasets: {e}")
            return pd.DataFrame(), pd.DataFrame()

    if "season" in matches.columns:
        matches["season"] = pd.to_numeric(matches["season"].astype(str).str[:4], errors="coerce")
        matches = matches.dropna(subset=["season", "winner"]).copy()
        matches["season"] = matches["season"].astype(int)

    return matches, deliveries

matches, deliveries = load_data()

TEAM_COLORS = {
    "Chennai Super Kings": "#FBC02D",
    "Mumbai Indians": "#0D47A1",
    "Royal Challengers Bangalore": "#D32F2F",
    "Royal Challengers Bengaluru": "#D32F2F",
    "Kolkata Knight Riders": "#4A148C",
    "Sunrisers Hyderabad": "#FF5722",
    "Rajasthan Royals": "#E91E63",
    "Kings XI Punjab": "#C62828",
    "Punjab Kings": "#C62828",
    "Delhi Capitals": "#1976D2",
    "Delhi Daredevils": "#1976D2",
    "Gujarat Titans": "#1A237E",
    "Lucknow Super Giants": "#00BCD4",
    "Deccan Chargers": "#006064",
    "Kochi Tuskers Kerala": "#FF9800",
    "Pune Warriors": "#607D8B",
    "Rising Pune Supergiant": "#E040FB",
    "Rising Pune Supergiants": "#E040FB",
    "Gujarat Lions": "#FFB300",
}

teams = [
    "Chennai Super Kings", "Deccan Chargers", "Delhi Capitals", "Delhi Daredevils",
    "Gujarat Lions", "Gujarat Titans", "Kings XI Punjab", "Kochi Tuskers Kerala",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians", "Pune Warriors",
    "Punjab Kings", "Rajasthan Royals", "Rising Pune Supergiant", "Rising Pune Supergiants",
    "Royal Challengers Bangalore", "Royal Challengers Bengaluru", "Sunrisers Hyderabad"
]

toss_decisions = ["bat", "field"]

historical_venues = sorted([
    "Arun Jaitley Stadium", "Arun Jaitley Stadium, Delhi", "Barabati Stadium",
    "Barsapara Cricket Stadium, Guwahati", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Brabourne Stadium", "Brabourne Stadium, Mumbai", "Buffalo Park", "De Beers Diamond Oval",
    "Dr DY Patil Sports Academy", "Dr DY Patil Sports Academy, Mumbai",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Dubai International Cricket Stadium", "Eden Gardens", "Eden Gardens, Kolkata",
    "Feroz Shah Kotla", "Green Park", "Himachal Pradesh Cricket Association Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala", "Holkar Cricket Stadium",
    "JSCA International Stadium Complex", "Kingsmead", "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru", "M.Chinnaswamy Stadium", "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium, Chepauk, Chennai",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Maharashtra Cricket Association Stadium", "Maharashtra Cricket Association Stadium, Pune",
    "Narendra Modi Stadium, Ahmedabad", "Nehru Stadium", "New Wanderers Stadium", "Newlands",
    "OUTsurance Oval", "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Punjab Cricket Association Stadium, Mohali", "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal", "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Sardar Patel Stadium, Motera", "Saurashtra Cricket Association Stadium",
    "Sawai Mansingh Stadium", "Sawai Mansingh Stadium, Jaipur",
    "Shaheed Veer Narayan Singh International Stadium", "Sharjah Cricket Stadium",
    "Sheikh Zayed Stadium", "St George's Park", "Subrata Roy Sahara Stadium",
    "SuperSport Park", "Vidarbha Cricket Association Stadium, Jamtha", "Wankhede Stadium",
    "Wankhede Stadium, Mumbai", "Zayed Cricket Stadium, Abu Dhabi"
])

venue_mapping = {v: v for v in historical_venues}
venue_mapping.update({
    "ACA-VDCA Cricket Stadium (Visakhapatnam)": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Arun Jaitley Stadium (Delhi)": "Arun Jaitley Stadium",
    "Barabati Stadium (Cuttack)": "Barabati Stadium",
    "Barsapara Cricket Stadium (Guwahati)": "Barsapara Cricket Stadium, Guwahati",
    "Brabourne Stadium (Mumbai)": "Brabourne Stadium",
    "Buffalo Park (East London)": "Buffalo Park",
    "De Beers Diamond Oval (Kimberley)": "De Beers Diamond Oval",
    "Dr DY Patil Sports Academy (Mumbai)": "Dr DY Patil Sports Academy",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
    "Eden Gardens (Kolkata)": "Eden Gardens",
    "Ekana Cricket Stadium (Lucknow)": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Feroz Shah Kotla (Delhi)": "Feroz Shah Kotla",
    "Green Park (Kanpur)": "Green Park",
    "HPCA Stadium (Dharamsala)": "Himachal Pradesh Cricket Association Stadium",
    "Holkar Cricket Stadium (Indore)": "Holkar Cricket Stadium",
    "IS Bindra PCA Stadium (Mohali)": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "JSCA International Stadium Complex (Ranchi)": "JSCA International Stadium Complex",
    "Kingsmead (Durban)": "Kingsmead",
    "M Chinnaswamy Stadium (Bengaluru)": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium (Chennai)": "MA Chidambaram Stadium",
    "Maharaja Yadavindra Singh Stadium (Mullanpur)": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Maharashtra Cricket Association Stadium (Pune)": "Maharashtra Cricket Association Stadium",
    "Narendra Modi Stadium (Ahmedabad)": "Narendra Modi Stadium, Ahmedabad",
    "Nehru Stadium (Kochi)": "Nehru Stadium",
    "New Wanderers Stadium (Johannesburg)": "New Wanderers Stadium",
    "Newlands (Cape Town)": "Newlands",
    "OUTsurance Oval (Bloemfontein)": "OUTsurance Oval",
    "Rajiv Gandhi International Stadium (Hyderabad)": "Rajiv Gandhi International Stadium",
    "Sardar Patel Stadium, Motera (Ahmedabad)": "Sardar Patel Stadium, Motera",
    "Saurashtra Cricket Association Stadium (Rajkot)": "Saurashtra Cricket Association Stadium",
    "Sawai Mansingh Stadium (Jaipur)": "Sawai Mansingh Stadium",
    "Shaheed Veer Narayan Singh Stadium (Raipur)": "Shaheed Veer Narayan Singh International Stadium",
    "Sheikh Zayed Stadium (Abu Dhabi)": "Sheikh Zayed Stadium",
    "St George's Park (Gqeberha)": "St George's Park",
    "Subrata Roy Sahara Stadium (Pune)": "Subrata Roy Sahara Stadium",
    "SuperSport Park (Centurion)": "SuperSport Park",
    "Vidarbha Cricket Association Stadium (Nagpur)": "Vidarbha Cricket Association Stadium, Jamtha",
    "Wankhede Stadium (Mumbai)": "Wankhede Stadium",
})

le_team = LabelEncoder().fit(teams)
le_venue = LabelEncoder().fit(historical_venues)
le_toss = LabelEncoder().fit(toss_decisions)

try:
    model = joblib.load("winner_predictor.pkl")
except Exception:
    model = None

st.title("IPL Analytics & Winner Prediction")

if matches.empty or deliveries.empty:
    st.warning("Could not load match or delivery data.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Team Analysis", "Player Performance", "Match Predictor"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Matches Played</div><div class='metric-value'>{len(matches):,}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>Seasons Played</div><div class='metric-value'>{matches['season'].nunique()}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-label'>Teams Participated</div><div class='metric-value'>{pd.concat([matches['team1'], matches['team2']]).nunique()}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-label'>Total Runs Scored</div><div class='metric-value'>{int(deliveries['total_runs'].sum()):,}</div></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        toss_counts = matches["toss_decision"].value_counts().reset_index()
        toss_counts.columns = ["Decision", "Count"]
        fig = px.pie(toss_counts, values="Count", names="Decision", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        toss_impact = (matches["toss_winner"] == matches["winner"]).value_counts().reset_index()
        toss_impact.columns = ["Won Toss And Match", "Count"]
        toss_impact["Won Toss And Match"] = toss_impact["Won Toss And Match"].map({True: "Won Match", False: "Lost Match"})
        fig = px.bar(toss_impact, x="Won Toss And Match", y="Count")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    wins_count = matches["winner"].value_counts().reset_index()
    wins_count.columns = ["Team", "Wins"]
    wins_count = wins_count.sort_values("Wins")
    st.plotly_chart(px.bar(wins_count, y="Team", x="Wins", orientation="h", color="Team", color_discrete_map=TEAM_COLORS), use_container_width=True)

    played_total = matches["team1"].value_counts().add(matches["team2"].value_counts(), fill_value=0)
    wins = matches["winner"].value_counts()
    win_rate = ((wins / played_total) * 100).round(1).reset_index()
    win_rate.columns = ["Team", "Win Rate (%)"]
    win_rate = win_rate.sort_values("Win Rate (%)")
    st.plotly_chart(px.bar(win_rate, y="Team", x="Win Rate (%)", orientation="h", color="Team", color_discrete_map=TEAM_COLORS), use_container_width=True)

with tab3:
    top_batsmen = deliveries.groupby("batter")["batsman_runs"].sum().sort_values().tail(10).reset_index()
    st.plotly_chart(px.bar(top_batsmen, y="batter", x="batsman_runs", orientation="h"), use_container_width=True)

    wickets = deliveries[deliveries["is_wicket"] == 1]
    top_bowlers = wickets["bowler"].value_counts().head(10).reset_index()
    top_bowlers.columns = ["bowler", "wickets"]
    top_bowlers = top_bowlers.sort_values("wickets")
    st.plotly_chart(px.bar(top_bowlers, y="bowler", x="wickets", orientation="h"), use_container_width=True)

with tab4:
    if model is None:
        st.error("Prediction model could not be loaded.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            team1_sel = st.selectbox("Select Team 1", teams)
            team2_sel = st.selectbox("Select Team 2", [t for t in teams if t != team1_sel])
            toss_winner_sel = st.selectbox("Select Toss Winner", [team1_sel, team2_sel])
        with col2:
            toss_decision_sel = st.selectbox("Select Toss Decision", toss_decisions)
            venue_sel = st.selectbox("Select Match Venue", list(venue_mapping.keys()))
            season_sel = st.number_input("Select Season Year", min_value=2008, max_value=2026, value=2024)

        if st.button("Predict Winner"):
            mapped_venue = venue_mapping[venue_sel]
            input_df = pd.DataFrame([{
                "team1_enc": le_team.transform([team1_sel])[0],
                "team2_enc": le_team.transform([team2_sel])[0],
                "toss_winner_enc": le_team.transform([toss_winner_sel])[0],
                "toss_decision_enc": le_toss.transform([toss_decision_sel])[0],
                "venue_enc": le_venue.transform([mapped_venue])[0],
                "season": season_sel,
            }])

            if hasattr(model, "feature_names_in_"):
                input_df = input_df[model.feature_names_in_]

            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            winner = team1_sel if pred in [1, le_team.transform([team1_sel])[0]] else team2_sel

            st.success(f"Predicted Winner: {winner}")
            st.write(f"{team1_sel}: {round(probs[1] * 100, 2)}%")
            st.write(f"{team2_sel}: {round(probs[0] * 100, 2)}%")