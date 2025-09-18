
import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from bert_score import score as bertscore_score
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False



#text utils + ROUGE

def normalize_text(s: str) -> str:
    """Lowercase, strip, keep only [a-z0-9 .-%] to make overlap stable on short answers."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9%\s\.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return normalize_text(s).split()

def rouge_1_p_r_f1(ref: str, hyp: str) -> Tuple[float, float, float]:
    r_toks = tokenize(ref)
    h_toks = tokenize(hyp)
    if not r_toks and not h_toks:
        return 1.0, 1.0, 1.0
    if not r_toks or not h_toks:
        return 0.0, 0.0, 0.0
    r_counts = defaultdict(int)
    for t in r_toks:
        r_counts[t] += 1
    h_counts = defaultdict(int)
    for t in h_toks:
        h_counts[t] += 1
    overlap = sum(min(c, r_counts.get(t, 0)) for t, c in h_counts.items())
    prec = overlap / len(h_toks)
    rec = overlap / len(r_toks)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1

def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        ai = a[i-1]
        for j in range(1, n+1):
            if ai == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l_f1(ref: str, hyp: str) -> float:
    r_toks = tokenize(ref)
    h_toks = tokenize(hyp)
    if not r_toks and not h_toks:
        return 1.0
    if not r_toks or not h_toks:
        return 0.0
    L = _lcs(r_toks, h_toks)
    rec = L / len(r_toks)
    prec = L / len(h_toks)
    return 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)



#BERTScore wrapper
def run_bertscore(refs: List[str], hyps: List[str], lang: str = "en") -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    if not BERTSCORE_AVAILABLE:
        return [None]*len(refs), [None]*len(refs), [None]*len(refs)
    try:
        P, R, F = bertscore_score(hyps, refs, lang=lang, rescale_with_baseline=True)
        return P.tolist(), R.tolist(), F.tolist()
    except Exception:
        return [None]*len(refs), [None]*len(refs), [None]*len(refs)



#parsing human evaluations, with my structure

def parse_human_file(txt: str) -> List[Dict]:
    parts = re.split(r'(?m)^\s*(\d+\*?\))', txt.strip())
    entries = []
    for i in range(1, len(parts), 2):
        label = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        entries.append((label, content))

    items = []
    for label, content in entries:
        mpass = re.search(r'passage\s*:\s*(\d+)', content, re.IGNORECASE)
        src = int(mpass.group(1)) if mpass else None
        mqi = re.search(r'"question_ix"\s*:\s*(\d+)', content)
        qix = int(mqi.group(1)) if mqi else 0
        answers = []
        for ans_match in re.finditer(r'answer(?:\s+from\s+(\d+))?\s*:\s*(.+)', content, re.IGNORECASE):
            from_doc = ans_match.group(1)
            ans_text = ans_match.group(2).strip()
            ans_text = re.sub(r'\(.*?\)\s*$', '', ans_text).strip()
            label2 = f"from_{from_doc}" if from_doc else "ref"
            answers.append((label2, ans_text))
        notes = []
        for line in content.splitlines():
            if "bm25 got" in line.lower():
                notes.append(line.strip())
        if answers:
            items.append({
                "entry": label,
                "source_id": src,
                "question_ix": qix,
                "answers": answers,
                "notes": " | ".join(notes)
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="Path to my-evaluations.txt")
    ap.add_argument("--model", required=True, help="Path to cf_grounded_pruned_salvagedv4.jsonl")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--bertscore_lang", default="en", help="Language code for BERTScore (default: en)")
    args = ap.parse_args()

    with open(args.human, "r", encoding="utf-8") as f:
        human_txt = f.read()
    items = parse_human_file(human_txt)

    model_map: Dict[Tuple[int, int], Dict] = {}
    with open(args.model, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (int(obj.get("source_id")), int(obj.get("question_ix")))
                model_map[key] = obj
            except Exception:
                continue

    rows = []
    missing_pairs = []
    for it in items:
        key = (it["source_id"], it["question_ix"]) if it["source_id"] is not None else None
        model_ans = ""
        model_found = False
        if key and key in model_map:
            model_ans = model_map[key].get("answer", "") or ""
            model_found = True
        else:
            missing_pairs.append((it["entry"], it["source_id"], it["question_ix"]))
        for label2, ref_text in it["answers"]:
            rows.append({
                "entry": it["entry"],
                "source_id": it["source_id"],
                "question_ix": it["question_ix"],
                "ref_variant": label2,
                "reference": ref_text,
                "model_answer": model_ans,
                "model_found": model_found,
                "notes": it["notes"]
            })

    df = pd.DataFrame(rows)

    r1_p, r1_r, r1_f, rl_f = [], [], [], []
    for _, row in df.iterrows():
        p, r, f = rouge_1_p_r_f1(row["reference"], row["model_answer"])
        r1_p.append(p); r1_r.append(r); r1_f.append(f)
        rl_f.append(rouge_l_f1(row["reference"], row["model_answer"]))
    df["rouge1_p"] = r1_p
    df["rouge1_r"] = r1_r
    df["rouge1_f1"] = r1_f
    df["rougeL_f1"] = rl_f

    P, R, F = run_bertscore(df["reference"].tolist(), df["model_answer"].tolist(), lang=args.bertscore_lang)
    df["bertscore_p"] = P
    df["bertscore_r"] = R
    df["bertscore_f1"] = F

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    if missing_pairs:
        print("Missing pairs (first 5):", missing_pairs[:5])


if __name__ == "__main__":
    main()
