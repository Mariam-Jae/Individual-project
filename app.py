import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# Text cleaning 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

#confidence scores
def get_model_score(resume_text, model):
    cleaned = clean_text(resume_text)
    X = vectorizer.transform([cleaned])
    prob = model.predict_proba(X)
    score = np.max(prob)
    return score

# Load models
biased_model = joblib.load("biased_model.pkl")
debiased_model = joblib.load("debiased_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load dataset 1
df1 = pd.read_excel("Resumes.xlsx")
#dataset 2
df2 = pd.read_excel("CV.xlsx")

@st.cache_data
def load_resumes():
    df1 = pd.read_excel("Resumes.xlsx")
    df2 = pd.read_excel("CV.xlsx")

    df2 = df2.rename(columns={"output": "resume_text"})

    resumes_df = pd.concat([df1, df2], ignore_index=True)

    # Filter out empty columns
    resumes_df = resumes_df[resumes_df['resume_text'].notna()].reset_index(drop=True)

    return resumes_df

resumes_df = load_resumes()
df2 = df2.rename(columns={
    "output": "resume_text"
})
resumes_df = pd.concat([df1, df2], ignore_index=True)

st.title("Automated AI Hiring")
st.write("comparing predictions from a biased and debiased model.")


st.write("Total resumes available:", len(resumes_df))

#predicted professions dictionary
profession_map = {
0: "accountant",
1: "architect",
2: "attorney",
3: "chiropractor",
4: "comedian",
5: "composer",
6: "dentist",
7: "dietitian",
8: "dj",
9: "filmmaker",
10: "interior designer",
11: "journalist",
12: "model",
13: "nurse",
14: "painter",
15: "paralegal",
16: "pastor",
17: "personal trainer",
18: "photographer",
19: "physician",
20: "poet",
21: "professor",
22: "psychologist",
23: "rapper",
24: "software engineer",
25: "surgeon",
26: "teacher",
27: "yoga teacher"
}
# Keep the rows that arent empty only
resumes_df = resumes_df[resumes_df['resume_text'].notna()].reset_index(drop=True)

# Button to load random CV
st.subheader("CV input")
if st.button("Load Random CV"):
    random_resume = resumes_df.sample(1).iloc[0]["resume_text"]
    st.session_state["resume_text"] = random_resume

# Text box 
user_input = st.text_area(
    "Paste or edit CV text:",
    value=st.session_state.get("resume_text", ""),
    height=200
)
resume_text = st.session_state.get("resume_text", user_input)

if st.button("Analyse CV"):
    if resume_text.strip() == "":
        st.warning("Please enter a CV.")
    else:
        cleaned = clean_text(resume_text)
        X = vectorizer.transform([cleaned])

        biased_pred = biased_model.predict(X)[0]
        debiased_pred = debiased_model.predict(X)[0]

        biased_prob = np.max(biased_model.predict_proba(X))
        debiased_prob = np.max(debiased_model.predict_proba(X))

        st.subheader("Results")

        col1, col2 = st.columns(2)

#biased model A
        with col1:
            st.write("### Model A")
            st.write(f"Predicted Profession: **{profession_map.get(biased_pred, 'Unknown')}**")
            st.write(f"Confidence: **{biased_prob:.2f}**")

#debiased model B
        with col2:
            st.write("### Model B")
            st.write(f"Predicted Profession: **{profession_map.get(debiased_pred, 'Unknown')}**")
            st.write(f"Confidence: **{debiased_prob:.2f}**")

        st.info(
            "Try changing pronouns (he/she) "
            "and see how predictions change."
        )

st.divider()
st.header("Compare Two Candidates for the Same Job")

# Select profession dropdown for candidate comparison feature
target_prof = st.selectbox(
    "Select a profession to compare candidates for:",
    options=list(profession_map.values())
)

# predicted professions using biased model
resumes_df["pred_biased_prof"] = [
    profession_map[biased_model.predict(vectorizer.transform([clean_text(r)]))[0]]
    for r in resumes_df["resume_text"]
]

# Filter resumes relevant to the selected profession
relevant_resumes = resumes_df[resumes_df["pred_biased_prof"] == target_prof]

if len(relevant_resumes) < 2:
    st.warning("Not enough relevant CVs to compare for this profession.")
else:
    # Button to load two random compatible CVs
    if st.button("Load Two Random CVs"):
        sample = relevant_resumes.sample(2)
        st.session_state["cv_a"] = sample.iloc[0]["resume_text"]
        st.session_state["cv_b"] = sample.iloc[1]["resume_text"]


    # Show CVs side by side
    if "cv_a" in st.session_state and "cv_b" in st.session_state:
        col1, col2 = st.columns(2)



        with col1:
            st.subheader("Candidate A")
            st.session_state["cv_a"] = st.text_area(
                "Edit Candidate A CV:",
                value=st.session_state["cv_a"],
                height=200
            )

        with col2:
            st.subheader("Candidate B")
            st.session_state["cv_b"] = st.text_area(
                "Edit Candidate B CV:",
                value=st.session_state["cv_b"],
                height=200
            )

        # Evaluate candidates for profession
        if st.button("Evaluate Candidates"):
            cv_a = st.session_state["cv_a"]
            cv_b = st.session_state["cv_b"]

            # Scores from biased model (how much the candidate matches the job role)
            biased_a = get_model_score(cv_a, biased_model)
            biased_b = get_model_score(cv_b, biased_model)

            # Scores from debiased model
            debiased_a = get_model_score(cv_a, debiased_model)
            debiased_b = get_model_score(cv_b, debiased_model)

            st.subheader("Model Recommendations")
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Model A")
                winner = "Candidate A" if biased_a > biased_b else "Candidate B"
                st.write(f"Recommended: **{winner}**")
                st.write(f"Candidate A: {profession_map[biased_model.predict(vectorizer.transform([clean_text(cv_a)]))[0]]} (Score: {biased_a:.2f})")
                st.write(f"Candidate B: {profession_map[biased_model.predict(vectorizer.transform([clean_text(cv_b)]))[0]]} (Score: {biased_b:.2f})")

            with col2:
                st.write("### Model B")
                winner = "Candidate A" if debiased_a > debiased_b else "Candidate B"
                st.write(f"Recommended: **{winner}**")
                st.write(f"Candidate A: {profession_map[debiased_model.predict(vectorizer.transform([clean_text(cv_a)]))[0]]} (Score: {debiased_a:.2f})")
                st.write(f"Candidate B: {profession_map[debiased_model.predict(vectorizer.transform([clean_text(cv_b)]))[0]]} (Score: {debiased_b:.2f})")
