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

# ---------------- Slide picker ----------------
FIRST = "First"
INTRO = "Introduction"
ANALYZE = "How We Analyzed the Data"
Q1 = "1. Do physical measurements determine if a player gets drafted?"
Q2 = "2. Which combine stats are the best predictors of whether a player gets drafted?"
Q3 = "3. Can a player's position group be predicted from his combine results?"
CONCLUSION = "Conclusion"
THX = "Thanks"

slides = [FIRST, INTRO, ANALYZE, Q1, Q2, Q3, CONCLUSION, THX]
if "i" not in st.session_state:
    st.session_state.i = 0

back, nxt = st.columns(2)
if back.button("← Back", disabled=st.session_state.i == 0, use_container_width=True):
    st.session_state.i -= 1
    st.rerun()
if nxt.button("Next →", disabled=st.session_state.i == len(slides) - 1, use_container_width=True):
    st.session_state.i += 1
    st.rerun()

slide = slides[st.session_state.i]
st.progress((st.session_state.i + 1) / len(slides))
# ---------------- Intro slides ----------------

if slide == FIRST:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 4rem;">CSCI 4047 FINAL PROJECT</h1>
            <p style="font-size: 2rem;">By: Alan Philipose and Sylar Jolly</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif slide == INTRO:
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

    st.subheader("Questions we answer")
    st.markdown(
        f"1. {Q1[3:]}\n"
        f"2. {Q2[3:]}\n"
        f"3. {Q3[3:]}\n\n"
    )
# ------------------------- Analyzing the Data Page -------------------------------------
elif slide == ANALYZE:
    st.title("Classification w/ Decision Trees")
    st.markdown(
        """
        <ul style="font-size: 1.5rem;">
            <li>We chose to do classification because every question has a category in its answer</li>
            <li>We didn't use regression since none of our answers are a continuous number</li>
            <li>Trained a separate tree for each position group</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    left.image("Screenshot 2026-11-18 181953.png", caption="Q1 Decision Tree")
    right.image("Screenshot 2026-11-18 182248.png", caption="Q2 Decision Tree")
# ---------------- Question 1 ----------------
elif slide == Q1:
    st.header(Q1)
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
elif slide == Q2:
    st.header(Q2)
    st.image("q2_importance_by_position.png", caption="Feature importance for each position group")

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
elif slide == Q3:
    st.header(Q3)
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

# ---------------- Conclusion slide ----------------
elif slide == CONCLUSION:
    st.title("Conclusion")
    st.success(
        "- **Size alone doesn't get you drafted.** Height and weight predicted the draft no better than guessing.\n"
        "- **Speed matters most.** The 40-yard dash was the top predictor for 6 of 7 position groups.\n"
        f"- **Size reveals position.** Combine stats predicted position group with {q3_accuracy * 100:.0f}% "
        "accuracy, mostly from weight and height."
    )
# ---------------- Thank you slide ----------------

else:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 6rem;">THANK YOU</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
