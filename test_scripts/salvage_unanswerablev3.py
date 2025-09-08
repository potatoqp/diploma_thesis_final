# salvage_unanswerable.py
import argparse, json, re, sys, os, csv
import pandas as pd
import requests
from datetime import datetime, timezone
from rank_bm25 import BM25Okapi

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"

# ---------- LLM prompts ----------
SYSTEM_PROMPT = """You are a careful, citation-faithful answer extractor.
You will receive:
- A GROUNDED_QUESTION.
- EVIDENCE_SENTENCES (from the same source passage).

Rules:
- Answer ONLY if the EVIDENCE_SENTENCES explicitly contain the answer.
- The answer must be a SHORT, DIRECT PHRASE (<= MAX_CHARS), preferably verbatim.
- Do NOT answer with "Yes"/"No" or paraphrase beyond what's present.
- If you cannot answer strictly from EVIDENCE_SENTENCES, return an EMPTY string.

Return STRICT JSON with keys:
  extracted_answer (string),
  used_quote (string),
  rationale (string)
"""

USER_TEMPLATE = """GROUNDED_QUESTION:
{question}

MAX_CHARS: {max_chars}

EVIDENCE_SENTENCES:
{evidence}

Return JSON now.
"""

# ---------- text utils ----------
WORD_RE = re.compile(r"[A-Za-z0-9']+")

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

STOPWORDS = set("""
the a an and or of to in on for with as by from at that this those these be is are was were has have had not no
what which who where when why how according study paper article report reports reported conclusion conclusions
""".split())

def _tokens(text: str):
    return [t.lower() for t in WORD_RE.findall(text or "")]

def _key_tokens(text: str):
    return [t for t in _tokens(text) if t not in STOPWORDS]

def _hard_sentence_split(text: str):
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def _window_slices(text: str, window_tokens=50, stride=25, max_slices=60):
    toks = _tokens(text)
    if not toks:
        return []
    out, i = [], 0
    while i < len(toks) and len(out) < max_slices:
        chunk = toks[i:i+window_tokens]
        if not chunk:
            break
        out.append(" ".join(chunk))
        i += stride
    if out:
        tail = " ".join(toks[-window_tokens:])
        if tail != out[-1]:
            out.append(tail)
    return out

def _sentences(text: str):
    hard = _hard_sentence_split(text)
    needs_windows = (len(hard) <= 1) or any(len(h) > 400 for h in hard)
    if not needs_windows:
        return hard
    hybrid = []
    for h in (hard or [text]):
        if len(h) <= 400 and len(h.split()) >= 6:
            hybrid.append(h.strip())
        else:
            for w in _window_slices(h, window_tokens=50, stride=25, max_slices=60):
                hybrid.append(w)
    seen, dedup = set(), []
    for s in hybrid:
        n = _norm(s)
        if n not in seen:
            seen.add(n)
            dedup.append(s)
    return dedup

def _shorten_wordsafe(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    cut = max(s.rfind(". ", 0, max_chars),
              s.rfind("; ", 0, max_chars),
              s.rfind(", ", 0, max_chars),
              s.rfind(" — ", 0, max_chars),
              s.rfind(" – ", 0, max_chars))
    if cut > 40:
        return s[:cut].rstrip()
    cut = s.rfind(" ", 0, max_chars)
    if cut > 40:
        return s[:cut].rstrip()
    m = re.match(r"^(.{0,%d})(?:\b|$)" % (max_chars-1), s)
    return (m.group(1) if m else s[:max_chars-1]).rstrip()

def _looks_like_yesno(ans: str) -> bool:
    return bool(re.fullmatch(r"\s*(yes|no)\s*", ans or "", flags=re.I))

def _preview(s: str, n=80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n-1].rstrip() + "…"

# ---------- support check (looser) ----------
def _supported(span: str, source_text: str, jaccard_thresh: float = 0.50) -> bool:
    if not span or not source_text:
        return False
    # exact normalized substring succeeds
    if _norm(span) in _norm(source_text):
        return True
    # token Jaccard across sentences/windows
    tt = set(_key_tokens(span))
    if not tt:
        return False
    for s in _sentences(source_text):
        st = set(_key_tokens(s))
        if not st:
            continue
        j = len(tt & st) / max(1, len(tt | st))
        if j >= jaccard_thresh:
            return True
    return False

# ---------- tiny BM25 over passage slices ----------
def _bm25_rank_slices(passage: str, question: str, topk: int = 10):
    slices = _sentences(passage)  # hybrid sentences/windows
    if not slices:
        return []
    # Build BM25 on the fly (corpus is tiny)
    corpus_tok = [_key_tokens(s) for s in slices]
    bm25 = BM25Okapi(corpus_tok)
    q = _key_tokens(question)
    scores = bm25.get_scores(q)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [(slices[i], float(scores[i])) for i in idxs]

def _top_evidence_blob(passage: str, question: str, bm25_topk: int, topn_sentences: int):
    # 1) grab top BM25 slices
    bm_hits = _bm25_rank_slices(passage, question, topk=bm25_topk)
    if not bm_hits:
        return []
    # 2) from those slices, break to sentences/windows again, rank by token overlap
    qtok = set(_key_tokens(question))
    pool = []
    for text, _sc in bm_hits:
        for s in _sentences(text):
            stok = set(_key_tokens(s))
            overlap = len(qtok & stok)
            if overlap > 0:
                pool.append((overlap, len(s), s))
    if not pool:
        # fallback: just use the raw BM25 slices
        uniq = []
        seen = set()
        for s, _ in bm_hits:
            n = _norm(s)
            if n not in seen:
                seen.add(n); uniq.append(s)
        return uniq[:topn_sentences]

    pool.sort(key=lambda x: (-x[0], x[1]))
    # dedupe by normalized string
    seen, picked = set(), []
    for _, _, s in pool:
        n = _norm(s)
        if n not in seen:
            seen.add(n)
            picked.append(s)
        if len(picked) >= topn_sentences:
            break
    return picked

# ---------- LLM plumbing ----------
def call_ollama(model: str, endpoint: str, system: str, user: str, timeout_s: int = 600) -> str:
    payload = {
        "model": model,
        "prompt": user,
        "system": system,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192},
        "stream": False,
    }
    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def parse_loose_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])
        raise

# ---------- salvage core ----------
def salvage_record(rec,
                   passage_text: str,
                   model: str,
                   endpoint: str,
                   max_chars: int,
                   bm25_topk: int,
                   evidence_sentences: int,
                   support_jaccard: float):
    q = rec.get("grounded_question") or rec.get("draft_question") or ""
    if not q or not passage_text:
        return None, "[salvage_failed:missing_q_or_passage]"

    # Build evidence blob from BM25 slices + overlap refinement
    picked = _top_evidence_blob(passage_text, q, bm25_topk=bm25_topk, topn_sentences=evidence_sentences)
    if not picked:
        return None, "[salvage_failed:no_overlap]"

    evidence_blob = "\n- " + "\n- ".join(picked[:evidence_sentences])
    user = USER_TEMPLATE.format(question=q, max_chars=max_chars, evidence=evidence_blob)

    try:
        raw = call_ollama(model, endpoint, SYSTEM_PROMPT, user)
        data = parse_loose_json(raw)
    except Exception as e:
        return None, f"[salvage_failed:llm_error:{e}]"

    ans = (data.get("extracted_answer") or "").strip()
    used = (data.get("used_quote") or "").strip()

    # Normalize output: no yes/no, length-safe, support check looser
    if ans and len(ans) > max_chars:
        ans = _shorten_wordsafe(ans, max_chars)
    if _looks_like_yesno(ans):
        ans = ""

    source_for_support = "\n".join(picked)
    if ans and _supported(ans, source_for_support, jaccard_thresh=support_jaccard):
        return {
            "answer": ans,
            "evidence": used if _supported(used, passage_text, support_jaccard) else " ".join(picked),
            "answerable": True,
            "rationale": (rec.get("rationale") or "") + " [salvaged_from_source]"
        }, "[salvaged_from_source]"

    # ---- fallback: take best sentence/phrase from picked, wordsafe clip ----
    best_sentence = picked[0]
    phrase = _shorten_wordsafe(best_sentence, max_chars)
    if phrase and _supported(phrase, passage_text, support_jaccard):
        return {
            "answer": phrase,
            "evidence": best_sentence,
            "answerable": True,
            "rationale": (rec.get("rationale") or "") + " [salvaged_fallback]"
        }, "[salvaged_fallback]"

    return None, "[salvage_failed:unsupported_or_empty]"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="cf_grounded_pruned.jsonl (input)")
    ap.add_argument("--passages", required=True, help="cf_passages.csv (with id, passage columns)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--max_chars", type=int, default=140)
    ap.add_argument("--bm25_topk", type=int, default=10, help="BM25 slices to consider from the passage")
    ap.add_argument("--evidence_sentences", type=int, default=6, help="Max sentences/windows sent to the LLM")
    ap.add_argument("--support_jaccard", type=float, default=0.50, help="Looser token Jaccard for support")
    ap.add_argument("--checkpoint", type=int, default=10, help="checkpoint every N records")
    ap.add_argument("--log_csv", default="", help="optional CSV to log salvaged rows")
    args = ap.parse_args()

    df = pd.read_csv(args.passages)
    if "id" not in df.columns or "passage" not in df.columns:
        print("ERROR: cf_passages.csv must have columns: id, passage", file=sys.stderr)
        sys.exit(1)
    id_to_passage = {str(r["id"]): str(r["passage"]) for _, r in df.iterrows()}

    total = 0
    candidates = 0
    salvaged = 0
    kept = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    log_writer = None
    log_fh = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        log_fh = open(args.log_csv, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_fh)
        log_writer.writerow(["source_id", "question_ix", "grounded_question", "tag", "answer_len", "answer_preview"])

    print(f"[start salvage] in={args.inp} out={args.out} bm25_topk={args.bm25_topk} evidence_sentences={args.evidence_sentences}")

    with open(args.inp, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            rec = json.loads(line)

            needs = (not rec.get("answerable", False)) or (not (rec.get("answer") or "").strip())
            if not needs:
                kept += 1
                rec["salvaged_at"] = datetime.now(timezone.utc).isoformat()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                candidates += 1
                src_id = rec.get("source_id")
                passage = id_to_passage.get(str(src_id), "")
                if not passage:
                    tag = "[salvage_failed:no_passage]"
                    rec["rationale"] = (rec.get("rationale") or "") + " " + tag
                else:
                    result, tag = salvage_record(
                        rec,
                        passage_text=passage,
                        model=args.model,
                        endpoint=args.endpoint,
                        max_chars=args.max_chars,
                        bm25_topk=args.bm25_topk,
                        evidence_sentences=args.evidence_sentences,
                        support_jaccard=args.support_jaccard,
                    )
                    if result:
                        rec.update(result)
                        salvaged += 1
                        ans_preview = _preview(rec.get("answer",""), 80)
                        print(f'\n[salvaged] sid={src_id} qix={rec.get("question_ix")} tag={tag} len={len(rec.get("answer",""))} "{ans_preview}"')
                        if log_writer:
                            log_writer.writerow([
                                src_id,
                                rec.get("question_ix"),
                                _preview(rec.get("grounded_question") or rec.get("draft_question") or "", 120),
                                tag,
                                len(rec.get("answer") or ""),
                                ans_preview
                            ])
                    else:
                        rec["rationale"] = (rec.get("rationale") or "") + " " + tag

                rec["salvaged_at"] = datetime.now(timezone.utc).isoformat()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # progress + checkpoint
            print(f"[progress] {total}", end="\r")
            if total % args.checkpoint == 0:
                fout.flush()
                os.fsync(fout.fileno())
                print(f"\n[checkpoint] saved after {total}")

    if log_fh:
        log_fh.close()

    print(f"\n[done salvage] wrote {args.out}")
    print(f"  total={total}  kept={kept}  candidates={candidates}  salvaged={salvaged}")
    if args.log_csv:
        print(f"  log: {args.log_csv}")

if __name__ == "__main__":
    main()
