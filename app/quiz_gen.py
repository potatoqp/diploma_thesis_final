import json
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


# calculate BERTSCORE if available
try:
    from bert_score import score as bertscore_score
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False


DEFAULT_DATASET_PATH = "data/final/cf_grounded_pruned_salvagedv4.jsonl"
N_QUESTIONS_DEFAULT = 10
RANDOM_SEED_DEFAULT = 42


# ROUGE-1 Precision helpers
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9%\s\.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    return normalize_text(s).split()

def rouge1_precision(ref: str, hyp: str) -> float:
    r_toks = tokenize(ref)
    h_toks = tokenize(hyp)
    if not r_toks and not h_toks:
        return 1.0
    if not h_toks:
        return 0.0
    from collections import defaultdict
    r_counts = defaultdict(int)
    for t in r_toks:
        r_counts[t] += 1
    h_counts = defaultdict(int)
    for t in h_toks:
        h_counts[t] += 1
    overlap = sum(min(c, r_counts.get(t, 0)) for t, c in h_counts.items())
    prec = overlap / len(h_toks)
    return prec


# BERTScore Precision helper
def bertscore_precision_single(ref: str, hyp: str) -> Optional[float]:
    if not BERTSCORE_AVAILABLE:
        return None
    try:
        P, R, F = bertscore_score([hyp], [ref], lang="en", rescale_with_baseline=True)
        return float(P[0])
    except Exception:
        return None


# data
@st.cache_data(show_spinner=False)
def load_dataset(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    df = pd.DataFrame(rows)

    mask = (df.get("answerable", True)) & (df.get("answer", "").astype(str).str.strip() != "")
    df = df[mask].copy()

    if "source_id" in df.columns and "question_ix" in df.columns:
        df["qid"] = df["source_id"].astype(str) + ":" + df["question_ix"].astype(str)
    else:
        df["qid"] = df.index.astype(str)

    keep = ["qid", "grounded_question", "answer"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    return df[keep]

def sample_questions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

# round to nearest 0.5 
def grade_from_mean_precision(mean_prec: float) -> float:
    raw = mean_prec * 10.0
    half_step = round(raw * 2.0) / 2.0
    return max(1.0, min(10.0, half_step))


# UI
st.set_page_config(page_title="Test About Cystic Fibrosis", layout="centered")

st.title("Test About Cystic Fibrosis")
st.write(
    "Test your knowledge! Answer the randomly selected questions and after submitting all answers, "
    "press 'Show Grade' to see your overall score (1–10)."
)

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Dataset path (.jsonl):", value=DEFAULT_DATASET_PATH)
    n_questions = st.number_input("Number of questions", min_value=N_QUESTIONS_DEFAULT, max_value=N_QUESTIONS_DEFAULT, value=N_QUESTIONS_DEFAULT, step=1, disabled=True)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=RANDOM_SEED_DEFAULT, step=1)
    if st.button("New set of questions"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

#load dataset
data_file = Path(data_path)
if not data_file.exists():
    st.error(f"File not found: {data_file.resolve()}")
    st.stop()

df_all = load_dataset(str(data_file))
if len(df_all) == 0:
    st.warning("No answerable entries found in the dataset.")
    st.stop()

if "sampled" not in st.session_state:
    st.session_state.sampled = sample_questions(df_all, int(n_questions), int(seed))
sampled = st.session_state.sampled

#store latest answer & scores
if "answers" not in st.session_state:
    st.session_state.answers = {}  # qid -> {"user": str, "rouge1_p": float, "bertscore_p": Optional[float]}

st.write(f"Loaded {len(df_all)} answerable items. Evaluation sample: {len(sampled)} questions.")

# render questions
for idx, row in sampled.iterrows():
    qid = row["qid"]
    question_text = row["grounded_question"] or "[no text]"
    prev_ans = st.session_state.answers.get(qid, {}).get("user", "")
    is_locked = qid in st.session_state.answers  # lock after first submit

    with st.form(key=f"form_{qid}", clear_on_submit=False):
        st.subheader(f"Question {idx+1}")
        st.write(question_text)

        user_ans = st.text_input(
            "Your answer:",
            key=f"input_{qid}",
            value=prev_ans,
            disabled=is_locked,  # disable editing once locked
        )
        submitted = st.form_submit_button("Submit", disabled=is_locked)

        if is_locked:
            st.info("Answer already submitted for this question.")
        elif submitted:
            model_answer = row["answer"] or ""
            r1p = rouge1_precision(model_answer, user_ans or "")
            bsp = bertscore_precision_single(model_answer, user_ans or "")
            st.session_state.answers[qid] = {"user": user_ans, "rouge1_p": r1p, "bertscore_p": bsp}
            st.success("Answer submitted!")

st.markdown("---")

# completion status
answered = [qid for qid in st.session_state.answers.keys()]
num_answered = len(answered)
num_total = len(sampled)
st.write(f"Answered {num_answered}/{num_total} questions.")

all_answered = (num_answered == num_total)
if not all_answered:
    missing = []
    for idx, row in sampled.iterrows():
        if row["qid"] not in st.session_state.answers:
            missing.append(str(idx + 1))
    if missing:
        st.info("Please answer all questions before showing your grade. Remaining: " + ", ".join(missing))

col1, col2 = st.columns(2)
with col1:
    if st.button("Show Grade", disabled=not all_answered):
        vals = [st.session_state.answers[q]["rouge1_p"] for q in st.session_state.answers]
        mean_r1p = sum(vals) / len(vals)
        grade = grade_from_mean_precision(mean_r1p)
        st.success(f"Your grade: {grade:.1f}/10")

with col2:
    if st.button("Download results as CSV", disabled=not all_answered):
        rows = []
        for _, row in sampled.iterrows():
            qid = row["qid"]
            rec = {
                "qid": qid,
                "question": row["grounded_question"],
                "user_answer": st.session_state.answers.get(qid, {}).get("user", ""),
                "rouge1_precision": st.session_state.answers.get(qid, {}).get("rouge1_p", None),
                "bertscore_precision": st.session_state.answers.get(qid, {}).get("bertscore_p", None),
            }
            rows.append(rec)
        out_df = pd.DataFrame(rows)
        st.download_button(
            "Download CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="qa_eval_session.csv",
            mime="text/csv",
        )



