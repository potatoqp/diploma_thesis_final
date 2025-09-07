
import argparse, json, re, sys, os, csv
import pandas as pd
import requests
from datetime import datetime, timezone

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"

# ---------------- LLM prompts ----------------

SYSTEM_PROMPT = """You are a careful, citation-faithful answer extractor.
You will receive:
- A GROUNDED_QUESTION.
- EVIDENCE_SENTENCES (from the *same* source passage).

Rules:
- Answer ONLY if the EVIDENCE_SENTENCES explicitly contain the answer.
- The answer must be a SHORT, DIRECT PHRASE (<= MAX_CHARS), preferably verbatim.
- Do NOT answer with "Yes"/"No" or paraphrase beyond what's present.
- If you cannot answer *strictly* from EVIDENCE_SENTENCES, return an EMPTY string.

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

# text utils 

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

def _window_slices(text: str, window_tokens=50, stride=25, max_slices=40):
    toks = _tokens(text)
    if not toks:
        return []
    out = []
    i = 0
    while i < len(toks) and len(out) < max_slices:
        chunk = toks[i:i+window_tokens]
        if not chunk:
            break
        out.append(" ".join(chunk))
        i += stride
    if out and " ".join(toks[-window_tokens:]) != out[-1]:
        out.append(" ".join(toks[-window_tokens:]))
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
            for w in _window_slices(h, window_tokens=50, stride=25, max_slices=40):
                hybrid.append(w)
    seen, dedup = set(), []
    for s in hybrid:
        n = _norm(s)
        if n not in seen:
            seen.add(n)
            dedup.append(s)
    return dedup

def _rank_sentences_by_overlap(passage: str, question: str, topn: int):
    qtok = set(_key_tokens(question))
    if not qtok:
        return []
    scored = []
    for s in _sentences(passage):
        stok = set(_key_tokens(s))
        score = len(qtok & stok)
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return [s for _, s in scored[:topn]]

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

def _supported(span: str, source_text: str) -> bool:
    if not span or not source_text:
        return False
    if _norm(span) in _norm(source_text):
        return True
    tt = set(_key_tokens(span))
    if not tt:
        return False
    for s in _sentences(source_text):
        st = set(_key_tokens(s))
        if not st:
            continue
        j = len(tt & st) / max(1, len(tt | st))
        if j >= 0.7:
            return True
    return False

def _looks_like_yesno(ans: str) -> bool:
    return bool(re.fullmatch(r"\s*(yes|no)\s*", ans or "", flags=re.I))

def _preview(s: str, n=80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n-1].rstrip() + "…"

#llm plumbing

def call_ollama(model: str, endpoint: str, system: str, user: str, timeout_s: int = 600) -> str:
    payload = {
        "model": model,
        "prompt": user,
        "system": system,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 8192
        },
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

# salvage core 

def salvage_record(rec, passage_text: str, model: str, endpoint: str, max_chars: int, topn: int):
    q = rec.get("grounded_question") or rec.get("draft_question") or ""
    best = _rank_sentences_by_overlap(passage_text, q, topn=topn)
    if not best:
        return None, "[salvage_failed:no_overlap]"
    evidence_blob = "\n- " + "\n- ".join(best)
    user = USER_TEMPLATE.format(question=q, max_chars=max_chars, evidence=evidence_blob)
    try:
        raw = call_ollama(model, endpoint, SYSTEM_PROMPT, user)
        data = parse_loose_json(raw)
    except Exception as e:
        return None, f"[salvage_failed:llm_error:{e}]"
    ans = (data.get("extracted_answer") or "").strip()
    used = (data.get("used_quote") or "").strip()
    if ans and len(ans) > max_chars:
        ans = _shorten_wordsafe(ans, max_chars)
    if _looks_like_yesno(ans):
        ans = ""
    if ans and _supported(ans, "\n".join(best)):
        return {
            "answer": ans,
            "evidence": used if _supported(used, passage_text) else " ".join(best),
            "answerable": True,
            "rationale": (rec.get("rationale") or "") + " [salvaged_from_source]"
        }, "[salvaged_from_source]"
    return None, "[salvage_failed:unsupported_or_empty]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="cf_grounded_pruned.jsonl (input)")
    ap.add_argument("--passages", required=True, help="cf_passages.csv (with id, passage columns)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--max_chars", type=int, default=140)
    ap.add_argument("--topn", type=int, default=4)
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--log_csv", default="", help="optional CSV path to log salvaged rows")
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
        log_writer.writerow(["source_id", "question_ix", "grounded_question", "answer_len", "answer_preview"])

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
                    rec["rationale"] = (rec.get("rationale") or "") + " [salvage_failed:no_passage]"
                else:
                    result, tag = salvage_record(rec, passage, args.model, args.endpoint, args.max_chars, args.topn)
                    if result:
                        rec.update(result)
                        salvaged += 1
                        # log for each successful salvage 
                        ans_preview = _preview(rec.get("answer",""), 80)
                        print(f'\n[salvaged] source_id={src_id} question_ix={rec.get("question_ix")} len={len(rec.get("answer",""))} "{ans_preview}"')
                        # --- optional CSV log ---
                        if log_writer:
                            log_writer.writerow([
                                src_id,
                                rec.get("question_ix"),
                                _preview(rec.get("grounded_question") or rec.get("draft_question") or "", 120),
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


