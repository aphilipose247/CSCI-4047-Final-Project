import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Put this file in the same folder as your Phase 2 notebook,
# so it can find the PNG and CSV files the notebook saved.

st.set_page_config(page_title="NFL Combine Dashboard", layout="wide")

# ---------------- Load the results saved in Phase 2 ----------------
q1_results = pd.read_csv("q1_by_position.csv")
q1_importance = pd.read_csv("q1_importance_by_position.csv", index_col=0)
q2_results = pd.read_csv("q2_by_position.csv")
q2_importance = pd.read_csv("q2_importance_by_position.csv", index_col=0)
compare = pd.read_csv("q1_vs_q2_comparison.csv")
q3_results = pd.read_csv("q3_results.csv", index_col=0)
q3_importance = pd.read_csv("q3_feature_importance.csv", index_col=0)

# QB left out: only 7 quarterbacks completed every drill
q1_results = q1_results[q1_results["Position"] != "QB"]
q2_results = q2_results[q2_results["Position"] != "QB"]
compare = compare[compare["Position"] != "QB"]
q1_importance = q1_importance.drop(index="QB", errors="ignore")
q2_importance = q2_importance.drop(index="QB", errors="ignore")

q3_accuracy = q3_results.loc["accuracy", "precision"]   # classification_report stores accuracy here

# ---------------- Title and description ----------------
st.title("NFL Combine Dashboard: What Gets a Player Drafted?")
st.markdown(
    "We analyzed **NFL Scouting Combine results from 2010 to 2023** (Kaggle, user *thomassshaw*): "
    "4,741 players with their height, weight, and six drills. Our models use the **2,024 players who "
    "completed every drill**, grouped into position groups (QB excluded, since only 7 completed every drill)."
)

c1, c2, c3 = st.columns(3)
c1.metric("Players at the combine", "4,741")
c2.metric("Completed every drill", "2,024")
c3.metric("Position prediction accuracy", f"{q3_accuracy * 100:.0f}%")

# ---------------- Dropdown ----------------
question = st.selectbox(
    "Choose a question",
    ["1. Do physical measurements determine if a player gets drafted?",
     "2. Which combine stats are the best predictors of whether a player gets drafted?",
     "3. Can a player's position group be predicted from his combine results?"],
)

# ---------------- Question 1 ----------------
if question.startswith("1"):
    st.header(question)
    st.image("q1_by_position.png", caption="Model accuracy vs guessing 'drafted' for everyone")

    st.subheader("Results by position group")
    st.dataframe(q1_results, hide_index=True, use_container_width=True)

    st.subheader("Height vs weight importance")
    st.dataframe(q1_importance, use_container_width=True)

    st.info(
        "**Takeaway:** No. Height and weight alone don't determine whether a player gets drafted. "
        "No position group's model beat simply guessing \"drafted\" by more than about one point. "
        "Where size mattered at all, weight mattered more than height for every group except TE."
    )

# ---------------- Question 2 ----------------
elif question.startswith("2"):
    st.header(question)
    st.image("q2_importance_by_position.png", caption="Feature importance for each position group")

    # Second dropdown: pick a position to see its stats ranked
    group = st.selectbox("Pick a position group", list(q2_importance.index))
    imp = q2_importance.loc[group].sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp.index, imp.values,
            color=["#FFC72C" if s == imp.idxmax() else "#00053E" for s in imp.index])
    ax.set_title(f"Most Important Combine Stats for {group} Players")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Combine Stat")
    st.pyplot(fig)
    st.caption(f"Top stat for {group}: **{imp.idxmax()}**")

    st.subheader("Does adding the drills help? (Q1 size only vs Q2 all stats)")
    st.dataframe(compare, hide_index=True, use_container_width=True)

    st.info(
        "**Takeaway:** The 40-yard dash is the best predictor. It was the most important stat for 6 of 7 "
        "position groups, and bench press mattered most for offensive linemen. Adding the drills helped most "
        "for running backs (accuracy 58% → 70%) and wide receivers (51% → 74%). Speed predicts being drafted "
        "better than size."
    )

# ---------------- Question 3 ----------------
else:
    st.header(question)
    left, right = st.columns(2)
    left.image("q3_confusion_matrix.png", caption="Predicted vs actual position group")
    right.image("q3_feature_importance.png", caption="Which stats identify a player's position")

    st.subheader("How well each position group was identified")
    st.dataframe(q3_results.drop(index=["accuracy", "macro avg", "weighted avg", "QB"], errors="ignore"),
                 use_container_width=True)

    st.info(
        f"**Takeaway:** Yes. The model predicted position group with {q3_accuracy * 100:.0f}% accuracy, "
        "compared with about 12% for random guessing. Offensive linemen were easiest to identify (87% recall), "
        "and wide receivers (46%) and tight ends (59%) were hardest. Weight and height did most of the work."
    )

# ---------------- Key results ----------------
st.header("Key Results")
st.success(
    "- **Size alone doesn't get you drafted.** Height and weight predicted the draft no better than guessing.\n"
    "- **Speed matters most.** The 40-yard dash was the top predictor for 6 of 7 position groups.\n"
    f"- **Size reveals position.** Combine stats predicted position group with {q3_accuracy * 100:.0f}% "
    "accuracy, mostly from weight and height."
)