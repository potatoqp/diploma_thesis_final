
import argparse, json, re
import pandas as pd
from rank_bm25 import BM25Okapi

WORD_RE = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str):
    return [t.lower() for t in WORD_RE.findall(text or "")]

def build_bm25(passages):
    tokenized = [tokenize(p) for p in passages]
    return BM25Okapi(tokenized), tokenized

def retrieve(bm25, tokenized_corpus, question: str, k: int = 5):
    q = tokenize(question)
    scores = bm25.get_scores(q)
    #top-k indices by score
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, float(scores[i])) for i in idxs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True, help="cf_passages.csv")
    ap.add_argument("--drafts", required=True, help="cf_drafts.jsonl")
    ap.add_argument("--out", required=True, help="write contexts here (jsonl)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.passages)             
    id_list = df["id"].tolist()
    passages = df["passage"].astype(str).tolist()
    bm25, tok = build_bm25(passages)

    with open(args.drafts, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            q_items = obj.get("items", [])
            for j, it in enumerate(q_items):
                qtext = it.get("question", "")
                hits = retrieve(bm25, tok, qtext, k=args.k)
                ctx = [{
                    "rank": r+1,
                    "doc_index": i,
                    "doc_id": id_list[i],
                    "score": score,
                    "passage": passages[i]
                } for r, (i, score) in enumerate(hits)]
                rec = {
                    "source_id": obj["source_id"],
                    "question_ix": j,
                    "question": qtext,
                    "draft_answer": it.get("answer"),
                    "draft_difficulty": it.get("difficulty"),
                    "contexts": ctx
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
