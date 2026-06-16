# ============================================================
# IPL DATA ANALYTICS + MATCH WINNER PREDICTION SYSTEM
# COMPLETE END-TO-END PROJECT
# ============================================================

# ================= IMPORT LIBRARIES =================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

# ============================================================
# 1. LOAD DATASET
# ============================================================

# Download from Kaggle:
# matches.csv
# deliveries.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
matches = pd.read_csv(os.path.join(BASE_DIR, "data", "matches.csv"))
deliveries = pd.read_csv(os.path.join(BASE_DIR, "data", "deliveries.zip", "deliveries.csv"))

print("\n================ DATASET LOADED ================\n")

print("Matches Shape:", matches.shape)
print("Deliveries Shape:", deliveries.shape)

print("\nMatches Columns:\n")
print(matches.columns)

# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n================ DATA CLEANING ================\n")

# Select important columns

matches = matches[
    [
        "season",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "winner"
    ]
]

# Convert season into numeric values

matches["season"] = matches["season"].astype(str)

matches["season"] = matches["season"].str[:4]

matches["season"] = pd.to_numeric(
    matches["season"],
    errors="coerce"
)

# Remove invalid rows

matches.dropna(inplace=True)

matches["season"] = matches["season"].astype(int)

# Remove null values

matches.dropna(inplace=True)

# Remove matches with no result

matches = matches[matches["winner"].notna()]

print("Cleaned Dataset Shape:", matches.shape)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n================ EDA ANALYSIS ================\n")

# ------------------------------------------------------------
# TEAM WIN ANALYSIS
# ------------------------------------------------------------

plt.figure(figsize=(12,6))

matches["winner"].value_counts().plot(
    kind="bar",
    color="skyblue"
)

plt.title("IPL Team Wins Analysis")
plt.xlabel("Teams")
plt.ylabel("Number of Wins")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# TOSS IMPACT ANALYSIS
# ------------------------------------------------------------

matches["toss_match_win"] = (
    matches["toss_winner"] == matches["winner"]
)

plt.figure(figsize=(6,5))

sns.countplot(
    x="toss_match_win",
    data=matches
)

plt.title("Toss Winner vs Match Winner")
plt.xlabel("Toss Winner Won Match")
plt.ylabel("Count")

plt.show()

# ------------------------------------------------------------
# TOP 10 VENUES
# ------------------------------------------------------------

plt.figure(figsize=(12,6))

matches["venue"].value_counts().head(10).plot(
    kind="bar",
    color="orange"
)

plt.title("Top IPL Venues")
plt.xlabel("Venue")
plt.ylabel("Matches Hosted")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# PLAYER ANALYSIS
# ------------------------------------------------------------

# TOP RUN SCORERS

top_batsmen = deliveries.groupby("batter")["batsman_runs"] \
    .sum() \
    .sort_values(ascending=False) \
    .head(10)

plt.figure(figsize=(12,6))

top_batsmen.plot(
    kind="bar",
    color="green"
)

plt.title("Top 10 Run Scorers in IPL")
plt.xlabel("Player")
plt.ylabel("Runs")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# TOP WICKET TAKERS
# ------------------------------------------------------------

wickets = deliveries[deliveries["is_wicket"] == 1]

top_bowlers = wickets["bowler"].value_counts().head(10)

plt.figure(figsize=(12,6))

top_bowlers.plot(
    kind="bar",
    color="red"
)

plt.title("Top 10 Wicket Takers in IPL")
plt.xlabel("Bowler")
plt.ylabel("Wickets")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n================ FEATURE ENGINEERING ================\n")

# Encode categorical columns

team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
toss_encoder = LabelEncoder()

matches["team1_enc"] = team_encoder.fit_transform(matches["team1"])

matches["team2_enc"] = team_encoder.transform(matches["team2"])

matches["toss_winner_enc"] = team_encoder.transform(
    matches["toss_winner"]
)

matches["venue_enc"] = venue_encoder.fit_transform(
    matches["venue"]
)

matches["toss_decision_enc"] = toss_encoder.fit_transform(
    matches["toss_decision"]
)

# Target Variable

matches["target"] = (
    matches["winner"] == matches["team1"]
).astype(int)

# ============================================================
# 5. MACHINE LEARNING MODEL
# ============================================================

print("\n================ MODEL TRAINING ================\n")

features = [
    "team1_enc",
    "team2_enc",
    "toss_winner_enc",
    "venue_enc",
    "toss_decision_enc",
    "season"
]

X = matches[features]
y = matches["target"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------------------------------

lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)

# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)

# ============================================================
# 6. MODEL EVALUATION
# ============================================================

print("\n================ MODEL EVALUATION ================\n")

# ------------------------------------------------------------
# LOGISTIC REGRESSION RESULTS
# ------------------------------------------------------------

lr_pred = lr.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====\n")

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print("ROC-AUC:",
      roc_auc_score(
          y_test,
          lr.predict_proba(X_test)[:,1]
      ))

print("\nClassification Report:\n")

print(classification_report(y_test, lr_pred))

# ------------------------------------------------------------
# RANDOM FOREST RESULTS
# ------------------------------------------------------------

rf_pred = rf.predict(X_test)

print("\n===== RANDOM FOREST =====\n")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("ROC-AUC:",
      roc_auc_score(
          y_test,
          rf.predict_proba(X_test)[:,1]
      ))

print("\nClassification Report:\n")

print(classification_report(y_test, rf_pred))

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.show()

# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(rf, "winner_predictor.pkl")

print("\nModel Saved Successfully!")
print("Saved File: winner_predictor.pkl")

# ============================================================
# 10. MATCH PREDICTION SYSTEM
# ============================================================

print("\n================ MATCH PREDICTION ================\n")

def predict_match_winner(
    team1,
    team2,
    toss_winner,
    toss_decision,
    venue,
    season
):

    sample = pd.DataFrame({
        "team1_enc": [
            team_encoder.transform([team1])[0]
        ],

        "team2_enc": [
            team_encoder.transform([team2])[0]
        ],

        "toss_winner_enc": [
            team_encoder.transform([toss_winner])[0]
        ],

        "venue_enc": [
            venue_encoder.transform([venue])[0]
        ],

        "toss_decision_enc": [
            toss_encoder.transform([toss_decision])[0]
        ],

        "season": [season]
    })

    prediction = rf.predict(sample)[0]

    probability = rf.predict_proba(sample)[0]

    if prediction == 1:
        winner = team1
        win_prob = probability[1]
    else:
        winner = team2
        win_prob = probability[0]

    print("\n===================================")
    print("TEAM 1:", team1)
    print("TEAM 2:", team2)
    print("VENUE :", venue)
    print("-----------------------------------")
    print("PREDICTED WINNER:", winner)
    print("WIN PROBABILITY:", round(win_prob * 100, 2), "%")
    print("===================================\n")

# ============================================================
# 11. SAMPLE PREDICTION
# ============================================================

predict_match_winner(
    team1="Mumbai Indians",
    team2="Chennai Super Kings",
    toss_winner="Mumbai Indians",
    toss_decision="field",
    venue="Wankhede Stadium",
    season=2024
)

# ============================================================
# 12. FINAL PROJECT OUTPUTS
# ============================================================

print("\n================ FINAL OUTPUTS ================\n")

print("1. IPL Data Analytics Completed")
print("2. Team & Player Visualizations Generated")
print("3. Match Winner Prediction System Built")
print("4. ML Model Evaluation Completed")
print("5. Model Saved for Streamlit Deployment")

print("\nPROJECT COMPLETED SUCCESSFULLY")
