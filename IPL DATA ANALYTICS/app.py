import streamlit as st
import joblib
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    st.error("Plotly is not installed. Run: pip install plotly")
    st.stop()

from sklearn.preprocessing import LabelEncoder

# 1. PAGE CONFIG
st.set_page_config(
    page_title="IPL Analytics & Prediction Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. THEME TOGGLE PATTERN
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS DESIGN SYSTEM
def inject_custom_css():
    theme_css = f"""
    <style>
    :root {{
        --bg: {"#09090b" if IS_DARK else "#ffffff"};
        --bg-subtle: {"#0c0c0f" if IS_DARK else "#f9fafb"};
        --card: {"#0c0c0f" if IS_DARK else "#ffffff"};
        --card-hover: {"#131316" if IS_DARK else "#f4f4f5"};
        --border: {"#1e1e24" if IS_DARK else "#e4e4e7"};
        --border-subtle: {"#16161a" if IS_DARK else "#f0f0f2"};
        --text: {"#fafafa" if IS_DARK else "#09090b"};
        --text-muted: #71717a;
        --text-dim: {"#52525b" if IS_DARK else "#a1a1aa"};
        --accent: #2563eb;
        --accent-muted: #1d4ed8;
        --green: {"#22c55e" if IS_DARK else "#16a34a"};
        --green-muted: {"rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"};
        --red: {"#ef4444" if IS_DARK else "#dc2626"};
        --red-muted: {"rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"};
        --amber: {"#f59e0b" if IS_DARK else "#d97706"};
        --amber-muted: {"rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"};
        --shadow: {"none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"};
        --radius: 12px;
    }}
    
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 1.5rem 2rem 2.5rem !important;
        max-width: 1400px !important;
    }}
    
    /* Tabs (pill-style) */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: var(--text) !important;
        background: var(--card-hover) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--text) !important;
        background: var(--card) !important;
        border-color: var(--border) !important;
        box-shadow: var(--shadow) !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 8px !important;
        background: var(--bg-subtle) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        margin-bottom: 1.5rem !important;
        width: fit-content !important;
    }}
    
    /* Columns spacing */
    [data-testid="stHorizontalBlock"] {{
        gap: 1.5rem !important;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: var(--accent);
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-bottom: 0.25rem;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
    }}
    
    /* Chart Wrap */
    .chart-wrap {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }}
    .chart-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.15rem;
    }}
    .chart-subtitle {{
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }}
    
    /* Data Table */
    .table-wrap {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
        box-shadow: var(--shadow);
        overflow-x: auto;
        margin-bottom: 1.5rem;
    }}
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.75rem 1rem;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border);
    }}
    .data-table td {{
        padding: 0.75rem 1rem;
        color: var(--text);
        border-bottom: 1px solid var(--border-subtle);
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Badges */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-align: center;
    }}
    .badge-green {{ color: var(--green); background: var(--green-muted); }}
    .badge-red {{ color: var(--red); background: var(--red-muted); }}
    .badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
    .badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}
    
    /* Customize Streamlit widgets */
    div[data-baseweb="select"] > div {{
        background-color: var(--card) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
    }}
    input {{
        background-color: var(--card) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="input"] {{
        background-color: var(--card) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
    }}
    
    /* Target primary and secondary buttons */
    .stButton > button {{
        background-color: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{
        background-color: var(--accent-muted) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.2);
    }}
    
    /* Prediction Card */
    .prediction-card {{
        background: linear-gradient(135deg, rgba(37,99,235,0.04) 0%, rgba(29,78,216,0.04) 100%);
        border: 1px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: var(--shadow);
    }}
    
    .prediction-header {{
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 1rem;
    }}
    
    .probability-bar-container {{
        width: 100%;
        background-color: var(--border);
        border-radius: 10px;
        height: 16px;
        overflow: hidden;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        display: flex;
    }}
    
    .probability-bar-team1 {{
        height: 100%;
        background-color: var(--accent);
        transition: width 0.5s ease-in-out;
    }}
    .probability-bar-team2 {{
        height: 100%;
        background-color: var(--text-muted);
        transition: width 0.5s ease-in-out;
    }}
    
    /* Brand Header style */
    .brand-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
    }}
    .brand-logo-text {{
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .brand-desc {{
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.1rem;
    }}
    
    /* Theme toggle button styling override */
    .theme-toggle-btn > button {{
        background-color: var(--bg-subtle) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        width: auto !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 0.8rem !important;
    }}
    .theme-toggle-btn > button:hover {{
        background-color: var(--card-hover) !important;
        transform: none !important;
        box-shadow: none !important;
    }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

inject_custom_css()

# 4. PLOTLY THEME CONFIG
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#fafafa", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#71717a" if not IS_DARK else "#a1a1aa"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#71717a" if not IS_DARK else "#a1a1aa"),
    ),
)

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
    "Gujarat Lions": "#FFB300"
}

# 5. CORE SYSTEM VARIABLES (Required for ML model encoding)
teams = [
    "Chennai Super Kings", "Deccan Chargers", "Delhi Capitals", "Delhi Daredevils", 
    "Gujarat Lions", "Gujarat Titans", "Kings XI Punjab", "Kochi Tuskers Kerala", 
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians", "Pune Warriors", 
    "Punjab Kings", "Rajasthan Royals", "Rising Pune Supergiant", "Rising Pune Supergiants", 
    "Royal Challengers Bangalore", "Royal Challengers Bengaluru", "Sunrisers Hyderabad"
]

historical_venues = [
    "Arun Jaitley Stadium", "Arun Jaitley Stadium, Delhi", "Barabati Stadium", 
    "Barsapara Cricket Stadium, Guwahati", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow", 
    "Brabourne Stadium", "Brabourne Stadium, Mumbai", "Buffalo Park", "De Beers Diamond Oval", 
    "Dr DY Patil Sports Academy", "Dr DY Patil Sports Academy, Mumbai", "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium", 
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam", "Dubai International Cricket Stadium", 
    "Eden Gardens", "Eden Gardens, Kolkata", "Feroz Shah Kotla", "Green Park", 
    "Himachal Pradesh Cricket Association Stadium", "Himachal Pradesh Cricket Association Stadium, Dharamsala", 
    "Holkar Cricket Stadium", "JSCA International Stadium Complex", "Kingsmead", 
    "M Chinnaswamy Stadium", "M Chinnaswamy Stadium, Bengaluru", "M.Chinnaswamy Stadium", 
    "MA Chidambaram Stadium", "MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium, Chepauk, Chennai", 
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur", "Maharashtra Cricket Association Stadium", 
    "Maharashtra Cricket Association Stadium, Pune", "Narendra Modi Stadium, Ahmedabad", "Nehru Stadium", 
    "New Wanderers Stadium", "Newlands", "OUTsurance Oval", "Punjab Cricket Association IS Bindra Stadium", 
    "Punjab Cricket Association IS Bindra Stadium, Mohali", "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh", 
    "Punjab Cricket Association Stadium, Mohali", "Rajiv Gandhi International Stadium", "Rajiv Gandhi International Stadium, Uppal", 
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad", "Sardar Patel Stadium, Motera", "Saurashtra Cricket Association Stadium", 
    "Sawai Mansingh Stadium", "Sawai Mansingh Stadium, Jaipur", "Shaheed Veer Narayan Singh International Stadium", 
    "Sharjah Cricket Stadium", "Sheikh Zayed Stadium", "St George's Park", "Subrata Roy Sahara Stadium", 
    "SuperSport Park", "Vidarbha Cricket Association Stadium, Jamtha", "Wankhede Stadium", 
    "Wankhede Stadium, Mumbai", "Zayed Cricket Stadium, Abu Dhabi"
]

toss_decisions = ["bat", "field"]

venue_mapping = {
    'ACA-VDCA Cricket Stadium (Visakhapatnam)': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
    'Arun Jaitley Stadium (Delhi)': 'Arun Jaitley Stadium',
    'Barabati Stadium (Cuttack)': 'Barabati Stadium',
    'Barsapara Cricket Stadium (Guwahati)': 'Barsapara Cricket Stadium, Guwahati',
    'Brabourne Stadium (Mumbai)': 'Brabourne Stadium',
    'Buffalo Park (East London)': 'Buffalo Park',
    'De Beers Diamond Oval (Kimberley)': 'De Beers Diamond Oval',
    'Dr DY Patil Sports Academy (Mumbai)': 'Dr DY Patil Sports Academy',
    'Dubai International Cricket Stadium': 'Dubai International Cricket Stadium',
    'Eden Gardens (Kolkata)': 'Eden Gardens',
    'Ekana Cricket Stadium (Lucknow)': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Feroz Shah Kotla (Delhi)': 'Feroz Shah Kotla',
    'Green Park (Kanpur)': 'Green Park',
    'HPCA Stadium (Dharamsala)': 'Himachal Pradesh Cricket Association Stadium',
    'Holkar Cricket Stadium (Indore)': 'Holkar Cricket Stadium',
    'IS Bindra PCA Stadium (Mohali)': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'JSCA International Stadium Complex (Ranchi)': 'JSCA International Stadium Complex',
    'Kingsmead (Durban)': 'Kingsmead',
    'M Chinnaswamy Stadium (Bengaluru)': 'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium (Chennai)': 'MA Chidambaram Stadium',
    'Maharaja Yadavindra Singh Stadium (Mullanpur)': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Maharashtra Cricket Association Stadium (Pune)': 'Maharashtra Cricket Association Stadium',
    'Narendra Modi Stadium (Ahmedabad)': 'Narendra Modi Stadium, Ahmedabad',
    'Nehru Stadium (Kochi)': 'Nehru Stadium',
    'New Wanderers Stadium (Johannesburg)': 'New Wanderers Stadium',
    'Newlands (Cape Town)': 'Newlands',
    'OUTsurance Oval (Bloemfontein)': 'OUTsurance Oval',
    'Rajiv Gandhi International Stadium (Hyderabad)': 'Rajiv Gandhi International Stadium',
    'Sardar Patel Stadium, Motera (Ahmedabad)': 'Sardar Patel Stadium, Motera',
    'Saurashtra Cricket Association Stadium (Rajkot)': 'Saurashtra Cricket Association Stadium',
    'Sawai Mansingh Stadium (Jaipur)': 'Sawai Mansingh Stadium',
    'Shaheed Veer Narayan Singh Stadium (Raipur)': 'Shaheed Veer Narayan Singh International Stadium',
    'Sharjah Cricket Stadium': 'Sharjah Cricket Stadium',
    'Sheikh Zayed Stadium (Abu Dhabi)': 'Sheikh Zayed Stadium',
    "St George's Park (Gqeberha)": "St George's Park",
    'Subrata Roy Sahara Stadium (Pune)': 'Subrata Roy Sahara Stadium',
    'SuperSport Park (Centurion)': 'SuperSport Park',
    'Vidarbha Cricket Association Stadium (Nagpur)': 'Vidarbha Cricket Association Stadium, Jamtha',
    'Wankhede Stadium (Mumbai)': 'Wankhede Stadium'
}

clean_venues_list = sorted(list(venue_mapping.keys()))

# 6. INITIALIZE & FIT ENCODERS
le_team = LabelEncoder().fit(teams)
le_venue = LabelEncoder().fit(historical_venues)
le_toss = LabelEncoder().fit(toss_decisions)

# 7. LOAD ML MODEL
try:
    model = joblib.load("winner_predictor.pkl")
except FileNotFoundError:
    st.error("Could not find 'winner_predictor.pkl'. Please ensure it is in the same directory.")
    model = None

# 8. DATA LOADING PIPELINE (With Caching)
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("data/matches.csv")
        deliveries = pd.read_csv("data/deliveries.csv")
        
        # Clean seasons
        matches["season"] = matches["season"].astype(str).str[:4]
        matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
        matches.dropna(subset=["season", "winner"], inplace=True)
        matches["season"] = matches["season"].astype(int)
        
        return matches, deliveries
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_matches, df_deliveries = load_data()

# 9. UI HELPERS
def metric_card(label, value, delta=None, delta_type="up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def chart_wrap(title, subtitle=""):
    st.markdown(f"""
    <div class="chart-wrap">
        <div class="chart-title">{title}</div>
        <div class="chart-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)

def chart_wrap_end():
    st.markdown("</div>", unsafe_allow_html=True)

# 10. APP LAYOUT
# Header
head_left, head_right = st.columns([5, 1])
with head_left:
    st.markdown(f"""
    <div class="brand-container">
        <div>
            <div class="brand-logo-text">🏏 IPL Analytics & Winner Prediction</div>
            <div class="brand-desc">Explore historical IPL datasets and predict upcoming match winners using Machine Learning</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    st.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
    theme_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    st.button(theme_label, on_click=toggle_theme, key="theme_toggle")
    st.markdown('</div>', unsafe_allow_html=True)

if df_matches.empty or df_deliveries.empty:
    st.warning("Data files matches.csv or deliveries.csv could not be loaded. Please check data directory.")
else:
    # Setup Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Team Analysis", "Player Performance", "Match Predictor"])
    
    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Matches Played", f"{len(df_matches):,}")
        with c2:
            metric_card("Seasons Played", str(df_matches["season"].nunique()))
        with c3:
            metric_card("Teams Participated", str(pd.concat([df_matches["team1"], df_matches["team2"]]).nunique()))
        with c4:
            metric_card("Total Runs Scored", f"{df_deliveries['total_runs'].sum():,}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            chart_wrap("Toss Decision Selection Breakdown", "Percentage of matches where teams chose to bat vs field")
            toss_counts = df_matches["toss_decision"].value_counts().reset_index()
            toss_counts.columns = ["Decision", "Count"]
            
            fig_toss = px.pie(
                toss_counts, 
                values="Count", 
                names="Decision", 
                hole=0.4,
                color="Decision",
                color_discrete_map={"field": "#2563eb", "bat": "#f59e0b"}
            )
            fig_toss.update_layout(**PLOT_LAYOUT)
            fig_toss.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_toss, width="stretch", config={"displayModeBar": False})
            chart_wrap_end()
            
        with col_chart2:
            chart_wrap("Toss Win vs Match Win Impact", "Analysis of whether winning the toss increases the probability of winning the match")
            toss_impact = (df_matches["toss_winner"] == df_matches["winner"]).value_counts().reset_index()
            toss_impact.columns = ["Toss Winner Won", "Count"]
            toss_impact["Toss Winner Won"] = toss_impact["Toss Winner Won"].map({True: "Won Match", False: "Lost Match"})
            
            fig_impact = px.bar(
                toss_impact,
                x="Toss Winner Won",
                y="Count",
                color="Toss Winner Won",
                color_discrete_map={"Won Match": "#22c55e", "Lost Match": "#ef4444"}
            )
            fig_impact.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_impact, width="stretch", config={"displayModeBar": False})
            chart_wrap_end()
            
    # ==========================================
    # TAB 2: TEAM PERFORMANCE
    # ==========================================
    with tab2:
        team_subtab1, team_subtab2 = st.tabs(["Performance Rankings", "Head-to-Head Tool"])
        
        with team_subtab1:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                chart_wrap("Overall Wins by Team", "Total number of matches won historically")
                wins_count = df_matches["winner"].value_counts().reset_index()
                wins_count.columns = ["Team", "Wins"]
                wins_count = wins_count.sort_values(by="Wins", ascending=True)
                
                fig_wins = px.bar(
                    wins_count,
                    y="Team",
                    x="Wins",
                    orientation="h",
                    color="Team",
                    color_discrete_map=TEAM_COLORS
                )
                fig_wins.update_layout(**PLOT_LAYOUT)
                fig_wins.update_layout(showlegend=False)
                st.plotly_chart(fig_wins, width="stretch", config={"displayModeBar": False})
                chart_wrap_end()
                
            with col_t2:
                chart_wrap("Win Rate Percentage by Team", "Percentage of wins relative to total matches played")
                played1 = df_matches["team1"].value_counts()
                played2 = df_matches["team2"].value_counts()
                played_total = played1.add(played2, fill_value=0)
                wins = df_matches["winner"].value_counts()
                
                win_rate = ((wins / played_total) * 100).round(1).reset_index()
                win_rate.columns = ["Team", "Win Rate (%)"]
                win_rate = win_rate.sort_values(by="Win Rate (%)", ascending=True)
                
                fig_wr = px.bar(
                    win_rate,
                    y="Team",
                    x="Win Rate (%)",
                    orientation="h",
                    color="Team",
                    color_discrete_map=TEAM_COLORS
                )
                fig_wr.update_layout(**PLOT_LAYOUT)
                fig_wr.update_layout(showlegend=False)
                st.plotly_chart(fig_wr, width="stretch", config={"displayModeBar": False})
                chart_wrap_end()
                
        with team_subtab2:
            st.markdown("<div style='margin-bottom: 1rem;'>Select two teams to compare their head-to-head match stats:</div>", unsafe_allow_html=True)
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                all_teams = sorted(list(df_matches["team1"].unique()))
                team_a = st.selectbox("Select Team A", all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0)
            with col_sel2:
                other_teams = [t for t in all_teams if t != team_a]
                team_b = st.selectbox("Select Team B", other_teams, index=other_teams.index("Chennai Super Kings") if "Chennai Super Kings" in other_teams else 0)
                
            h2h_matches = df_matches[
                ((df_matches["team1"] == team_a) & (df_matches["team2"] == team_b)) |
                ((df_matches["team1"] == team_b) & (df_matches["team2"] == team_a))
            ]
            
            total_h2h = len(h2h_matches)
            
            if total_h2h > 0:
                team_a_wins = len(h2h_matches[h2h_matches["winner"] == team_a])
                team_b_wins = len(h2h_matches[h2h_matches["winner"] == team_b])
                
                c_h1, c_h2, c_h3 = st.columns(3)
                with c_h1:
                    metric_card(f"{team_a} Wins", str(team_a_wins))
                with c_h2:
                    metric_card("Total Head-to-Head Matches", str(total_h2h))
                with c_h3:
                    metric_card(f"{team_b} Wins", str(team_b_wins))
                    
                pct_a = round((team_a_wins / total_h2h) * 100)
                pct_b = round((team_b_wins / total_h2h) * 100)
                
                st.markdown(f"""
                <div style="margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 600; font-size: 0.9rem; display: flex; justify-content: space-between;">
                    <span>{team_a} ({pct_a}%)</span>
                    <span>{team_b} ({pct_b}%)</span>
                </div>
                <div class="probability-bar-container">
                    <div class="probability-bar-team1" style="width: {pct_a}%; background-color: {TEAM_COLORS.get(team_a, '#2563eb')};"></div>
                    <div class="probability-bar-team2" style="width: {pct_b}%; background-color: {TEAM_COLORS.get(team_b, '#71717a')};"></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br><h4>Recent Encounters</h4>", unsafe_allow_html=True)
                recent_df = h2h_matches.sort_values(by="season", ascending=False).head(5)
                
                rows = ""
                for idx, row in recent_df.iterrows():
                    winner_badge = f"<span class='badge badge-green'>{row['winner']}</span>"
                    toss_winner_badge = f"<span class='badge badge-blue'>{row['toss_winner']} ({row['toss_decision']})</span>"
                    rows += f"""
                    <tr>
                        <td>{row['season']}</td>
                        <td>{row['team1']} vs {row['team2']}</td>
                        <td>{toss_winner_badge}</td>
                        <td>{row['venue']}</td>
                        <td>{winner_badge}</td>
                    </tr>
                    """
                    
                st.markdown(f"""
                <div class="table-wrap">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Season</th>
                                <th>Matchup</th>
                                <th>Toss Choice</th>
                                <th>Venue</th>
                                <th>Winner</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"No historical head-to-head records found between {team_a} and {team_b}.")
                
    # ==========================================
    # TAB 3: PLAYER PERFORMANCE
    # ==========================================
    with tab3:
        player_subtab1, player_subtab2 = st.tabs(["Leaderboards", "Player Profile Finder"])
        
        with player_subtab1:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                chart_wrap("Orange Cap: Top 10 Batsmen", "Total historical runs scored by batter")
                top_batsmen = df_deliveries.groupby("batter")["batsman_runs"].sum().sort_values(ascending=True).tail(10).reset_index()
                fig_batsmen = px.bar(
                    top_batsmen,
                    y="batter",
                    x="batsman_runs",
                    orientation="h",
                    labels={"batter": "Player", "batsman_runs": "Runs"},
                    color="batsman_runs",
                    color_continuous_scale="Oranges"
                )
                fig_batsmen.update_layout(**PLOT_LAYOUT)
                fig_batsmen.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_batsmen, width="stretch", config={"displayModeBar": False})
                chart_wrap_end()
                
            with col_p2:
                chart_wrap("Purple Cap: Top 10 Bowlers", "Total wickets taken historically")
                wickets = df_deliveries[df_deliveries["is_wicket"] == 1]
                top_bowlers = wickets["bowler"].value_counts().head(10).reset_index()
                top_bowlers.columns = ["bowler", "wickets"]
                top_bowlers = top_bowlers.sort_values(by="wickets", ascending=True)
                
                fig_bowlers = px.bar(
                    top_bowlers,
                    y="bowler",
                    x="wickets",
                    orientation="h",
                    labels={"bowler": "Player", "wickets": "Wickets"},
                    color="wickets",
                    color_continuous_scale="Purples"
                )
                fig_bowlers.update_layout(**PLOT_LAYOUT)
                fig_bowlers.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_bowlers, width="stretch", config={"displayModeBar": False})
                chart_wrap_end()
                
        with player_subtab2:
            st.markdown("<div style='margin-bottom: 1rem;'>Search for any player to generate their IPL career report card:</div>", unsafe_allow_html=True)
            all_players = sorted(list(set(df_deliveries["batter"].unique()).union(set(df_deliveries["bowler"].unique()))))
            selected_player = st.selectbox("Search / Select Player Name", all_players, index=all_players.index("Virat Kohli") if "Virat Kohli" in all_players else 0)
            
            player_bat_data = df_deliveries[df_deliveries["batter"] == selected_player]
            player_bowl_data = df_deliveries[df_deliveries["bowler"] == selected_player]
            
            is_batsman = len(player_bat_data) > 0
            is_bowler = len(player_bowl_data) > 0
            
            if is_batsman:
                bat_runs = player_bat_data["batsman_runs"].sum()
                balls_faced = player_bat_data[player_bat_data["extras_type"] != "wides"].shape[0]
                dismissals = df_deliveries[(df_deliveries["player_dismissed"] == selected_player)].shape[0]
                bat_avg = round(bat_runs / dismissals, 2) if dismissals > 0 else bat_runs
                bat_sr = round((bat_runs / balls_faced) * 100, 1) if balls_faced > 0 else 0
                
                match_runs = player_bat_data.groupby("match_id")["batsman_runs"].sum()
                high_score = match_runs.max() if not match_runs.empty else 0
                fifties = ((match_runs >= 50) & (match_runs < 100)).sum()
                hundreds = (match_runs >= 100).sum()
            
            if is_bowler:
                bowl_wickets = player_bowl_data[player_bowl_data["is_wicket"] == 1].shape[0]
                runs_conceded = player_bowl_data[~player_bowl_data["extras_type"].isin(["legbyes", "byes"])]["total_runs"].sum()
                balls_bowled = player_bowl_data[~player_bowl_data["extras_type"].isin(["wides"])].shape[0]
                
                bowl_econ = round((runs_conceded / balls_bowled) * 6, 2) if balls_bowled > 0 else 0
                bowl_avg = round(runs_conceded / bowl_wickets, 2) if bowl_wickets > 0 else 0
                
                match_wickets = player_bowl_data[player_bowl_data["is_wicket"] == 1].groupby("match_id").size()
                match_runs_bowl = player_bowl_data[~player_bowl_data["extras_type"].isin(["legbyes", "byes"])].groupby("match_id")["total_runs"].sum()
                
                bowling_df = pd.DataFrame({"wickets": match_wickets, "runs": match_runs_bowl}).fillna({"wickets": 0, "runs": 0})
                if not bowling_df.empty:
                    best_perf = bowling_df.sort_values(by=["wickets", "runs"], ascending=[False, True]).iloc[0]
                    best_fig = f"{int(best_perf['wickets'])}/{int(best_perf['runs'])}"
                else:
                    best_fig = "N/A"
                    
            if is_batsman or is_bowler:
                col_b, col_w = st.columns(2)
                with col_b:
                    if is_batsman:
                        st.markdown(f"""
                        <div class="metric-card" style="margin-bottom: 1.5rem;">
                            <h4 style="margin: 0 0 1rem; color: var(--accent);">🏏 Batting Statistics</h4>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.25rem;">
                                <div>
                                    <div class="metric-label">IPL Career Runs</div>
                                    <div class="metric-value">{bat_runs:,}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Batting Average</div>
                                    <div class="metric-value">{bat_avg}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Strike Rate</div>
                                    <div class="metric-value">{bat_sr}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Highest Score</div>
                                    <div class="metric-value">{high_score}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Hundreds (100s)</div>
                                    <div class="metric-value">{hundreds}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Fifties (50s)</div>
                                    <div class="metric-value">{fifties}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No career batting statistics found.")
                        
                with col_w:
                    if is_bowler:
                        st.markdown(f"""
                        <div class="metric-card" style="margin-bottom: 1.5rem;">
                            <h4 style="margin: 0 0 1rem; color: #a855f7;">⚾ Bowling Statistics</h4>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.25rem;">
                                <div>
                                    <div class="metric-label">Career Wickets</div>
                                    <div class="metric-value">{bowl_wickets}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Economy Rate</div>
                                    <div class="metric-value">{bowl_econ}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Bowling Average</div>
                                    <div class="metric-value">{bowl_avg}</div>
                                </div>
                                <div>
                                    <div class="metric-label">Best Bowling Figures</div>
                                    <div class="metric-value">{best_fig}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No career bowling statistics found.")
            else:
                st.warning("No performance statistics found for the selected player.")
                
    # ==========================================
    # TAB 4: MATCH PREDICTOR
    # ==========================================
    with tab4:
        st.markdown("### Match Winner Prediction Engine")
        st.write("Input match parameters below and compute the predicted winner using our trained Random Forest model.")
        
        if model is not None:
            col_inp1, col_inp2 = st.columns(2)
            
            with col_inp1:
                team1_sel = st.selectbox("Select Team 1", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 10, key="pred_t1")
                remaining_teams_pred = [t for t in teams if t != team1_sel]
                team2_sel = st.selectbox("Select Team 2", remaining_teams_pred, index=0, key="pred_t2")
                
                toss_winner_sel = st.selectbox("Select Toss Winner", [team1_sel, team2_sel], key="pred_tw")
                
            with col_inp2:
                toss_decision_sel = st.selectbox("Select Toss Decision", toss_decisions, key="pred_td")
                selected_clean_venue_pred = st.selectbox("Select Match Venue", clean_venues_list, index=clean_venues_list.index("Wankhede Stadium (Mumbai)") if "Wankhede Stadium (Mumbai)" in clean_venues_list else 38, key="pred_v")
                season_sel = st.number_input("Select Season Year", min_value=2008, max_value=2026, value=2024, key="pred_s")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Predict Winner", key="predict_btn"):
                try:
                    matched_historical_venue = venue_mapping[selected_clean_venue_pred]
                    
                    team1_encoded = le_team.transform([team1_sel])[0]
                    team2_encoded = le_team.transform([team2_sel])[0]
                    toss_winner_encoded = le_team.transform([toss_winner_sel])[0]
                    toss_decision_encoded = le_toss.transform([toss_decision_sel])[0]
                    venue_encoded = le_venue.transform([matched_historical_venue])[0]
                    
                    input_dict = {
                        'team1_enc': team1_encoded,
                        'team2_enc': team2_encoded,
                        'toss_winner_enc': toss_winner_encoded,
                        'toss_decision_enc': toss_decision_encoded,
                        'venue_enc': venue_encoded,
                        'season': season_sel
                    }
                    
                    input_df = pd.DataFrame([input_dict])
                    
                    if hasattr(model, "feature_names_in_"):
                        input_df = input_df[model.feature_names_in_]
                    
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
                    if prediction == 1 or prediction == team1_encoded:
                        predicted_winner = team1_sel
                    else:
                        predicted_winner = team2_sel
                    
                    prob_team1 = round(probabilities[1] * 100)
                    prob_team2 = round(probabilities[0] * 100)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-header">
                            🏆 Predicted Match Winner: <span style="color: {TEAM_COLORS.get(predicted_winner, '#2563eb')}; font-weight: 800;">{predicted_winner}</span>
                        </div>
                        <div style="font-size: 0.95rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                            <span>{team1_sel}: <strong>{prob_team1}%</strong></span>
                            <span>{team2_sel}: <strong>{prob_team2}%</strong></span>
                        </div>
                        <div class="probability-bar-container">
                            <div class="probability-bar-team1" style="width: {prob_team1}%; background-color: {TEAM_COLORS.get(team1_sel, '#2563eb')};"></div>
                            <div class="probability-bar-team2" style="width: {prob_team2}%; background-color: {TEAM_COLORS.get(team2_sel, '#71717a')};"></div>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
                            *Prediction calculated using a Random Forest Classifier trained on historical IPL match features.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    h2h_pred_matches = df_matches[
                        ((df_matches["team1"] == team1_sel) & (df_matches["team2"] == team2_sel)) |
                        ((df_matches["team1"] == team2_sel) & (df_matches["team2"] == team1_sel))
                    ]
                    total_h2h_pred = len(h2h_pred_matches)
                    if total_h2h_pred > 0:
                        team1_wins_pred = len(h2h_pred_matches[h2h_pred_matches["winner"] == team1_sel])
                        team2_wins_pred = len(h2h_pred_matches[h2h_pred_matches["winner"] == team2_sel])
                        
                        st.markdown("<br><h5>Historical Head-to-Head Statistics</h5>", unsafe_allow_html=True)
                        st.markdown(f"""
                        Historically, these two teams have faced each other <strong>{total_h2h_pred} times</strong>. 
                        <strong>{team1_sel}</strong> won <strong>{team1_wins_pred} times</strong>, while <strong>{team2_sel}</strong> won <strong>{team2_wins_pred} times</strong>.
                        """)
                    else:
                        st.markdown("<br><h5>Historical Head-to-Head Statistics</h5>", unsafe_allow_html=True)
                        st.markdown("There are no historical matchups recorded between these two teams in the dataset.")
                        
                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")
        else:
            st.error("Prediction Engine is disabled. Could not load ML model 'winner_predictor.pkl'.")