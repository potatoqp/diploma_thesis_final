
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import requests

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a strict, citation-faithful editor.
You will receive:
- GROUNDED_QUESTION
- CURRENT_ANSWER (may be too long)
- EVIDENCE (and sometimes PASSAGE), which are authoritative.

Rules:
- Rewrite CURRENT_ANSWER to be SHORT and DIRECT (<= MAX_CHARS characters).
- You MUST ONLY USE wording that is supported by EVIDENCE/PASSAGE. Prefer verbatim phrases.
- If the question is yes/no, return just "Yes" or "No" only if EVIDENCE clearly supports it.
- Do NOT add numbers, names, or facts not present in EVIDENCE/PASSAGE.
- If you CANNOT answer concisely using only EVIDENCE/PASSAGE, return an EMPTY string.

Return STRICT JSON ONLY with keys:
  pruned_answer (string),
  used_quote (string),   // short supporting span you relied on (can be equal to pruned_answer)
  rationale (string)     // brief note why you chose that span
"""

USER_TEMPLATE = """GROUNDED_QUESTION:
{question}

MAX_CHARS: {max_chars}

CURRENT_ANSWER:
{answer}

EVIDENCE:
{evidence}

PASSAGE (optional, may duplicate EVIDENCE):
{passage}

Return JSON now.
"""



def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _contains_norm(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return _norm(needle) in _norm(haystack or "")

def _sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text or "")
    return [p.strip() for p in parts if p.strip()]

STOPWORDS = set("""
the a an and or of to in on for with as by from at that this those these be is are was were has have had not no
""".split())

def _key_tokens(text: str):
    toks = re.findall(r"[A-Za-z0-9']+", text or "")
    return [t.lower() for t in toks if t.lower() not in STOPWORDS]

def _best_support_sentence(context_text: str, question_text: str, min_overlap: int = 1) -> str:
    qtok = set(_key_tokens(question_text))
    if not qtok:
        return ""
    best = ""
    best_score = 0
    for s in _sentences(context_text):
        stok = set(_key_tokens(s))
        score = len(qtok & stok)
        if score > best_score or (score == best_score and 0 < len(s) < len(best or s + "X")):
            best = s
            best_score = score
    return best if best_score >= min_overlap else ""

def _extract_yesno(sentence: str) -> Optional[str]:
    s = _norm(sentence)
    if re.search(r"\bno\b|\bnot\b|\bcannot\b|\bcan not\b|\bcould not\b|\bwas not\b|\bwere not\b|\bwithout\b|\bnot detected\b|\bcouldn'?t\b", s):
        return "No"
    if re.search(r"\bcorrelat\w*|\bassociat\w*|\blink\w*|\bwas\b|\bwere\b|\bis\b|\bare\b", s):
        return "Yes"
    return None

def _extract_age(sentence: str) -> Optional[str]:
    m = re.search(r"\b(by|at)\s+the\s+age\s+of\s+([^.,;]+)", sentence, flags=re.IGNORECASE)
    if m: return m.group(2).strip()
    m = re.search(r"\b(at|by)\s+(\d+\s*(?:to|-|–)\s*\d+\s*(?:months?|years?)|\d+\s*(?:months?|years?))\b", sentence, flags=re.IGNORECASE)
    if m: return m.group(2).strip()
    m = re.search(r"\b\d+\s*(?:to|-|–)\s*\d+\s*(?:months?|years?)\b", sentence, flags=re.IGNORECASE)
    if m: return m.group(0).strip()
    m = re.search(r"\b\d+\s*(?:months?|years?)\b", sentence, flags=re.IGNORECASE)
    if m: return m.group(0).strip()
    m = re.search(r"\bmonths?\s+to\s+year\b", sentence, flags=re.IGNORECASE)
    if m: return m.group(0).strip()
    return None

def _question_type(q: str) -> str:
    qn = _norm(q)
    first = (qn.split(" ", 1)[0] if qn else "")
    if qn.startswith(("is ","are ","was ","were ","do ","does ","did ","can ","could ","should ","has ","have ","had ")) or first in {"is","are","was","were","do","does","did","can","could","should","has","have","had"}:
        return "yesno"
    if any(k in qn for k in ["what age","at what age","how old","what year","what month","how many years","how many months"]):
        return "age"
    if any(k in qn for k in ["method","technique","test","assay"]):
        return "method"
    if any(k in qn for k in ["which","who","what ","name the","what is the","what was the"]):
        return "which"
    return "other"

def _extract_answer(question: str, sentence: str) -> Optional[str]:
    qtype = _question_type(question)
    if qtype == "yesno":  return _extract_yesno(sentence)
    if qtype == "age":    return _extract_age(sentence)
    # method/which left to LLM or fallback snippet
    snippet = sentence.strip()
    if len(snippet) > 160: snippet = snippet[:157] + "..."
    return snippet or None


@dataclass
class GenConfig:
    model: str
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 8192

def call_ollama(cfg: GenConfig, system: str, prompt: str, timeout_s: int = 600) -> str:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "system": system,
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_ctx": cfg.num_ctx,
        },
        "stream": False,
    }
    r = requests.post(cfg.endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def parse_loose_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])
        raise

#pruning logic 

def prune_one(rec: Dict[str, Any], cfg: GenConfig, max_chars: int, ctx_max_chars: int) -> Dict[str, Any]:
    """
    Returns a new (or updated) record with possibly updated:
      - answer
      - evidence
      - answerable
      - rationale (appended with prune note)
    """
    answerable = bool(rec.get("answerable", True))
    answer = rec.get("answer", "") or ""
    grounded_q = rec.get("grounded_question") or rec.get("draft_question") or ""
    evidence = rec.get("evidence", "") or ""
    contexts = rec.get("contexts", []) or []

    # quick pass-through
    if not answerable or len(answer) <= max_chars:
        return rec

    # build compact passage from contexts if present (helps the LLM, but we also verify after)
    compact_passage = ""
    if contexts:
        chunks = []
        for c in contexts[:4]:  # keep it light
            p = (c.get("passage", "") or "")
            if len(p) > ctx_max_chars:
                p = p[:ctx_max_chars] + " …[truncated]"
            chunks.append(p)
        compact_passage = "\n---\n".join(chunks)

    # LLM prune attempt
    user = USER_TEMPLATE.format(
        question=grounded_q,
        max_chars=max_chars,
        answer=answer,
        evidence=evidence if evidence else "(none provided)",
        passage=compact_passage if compact_passage else "(none provided)"
    )

    pruned_answer = ""
    used_quote = ""
    prune_rationale = ""
    llm_ok = False
    try:
        raw = call_ollama(cfg, SYSTEM_PROMPT, user)
        data = parse_loose_json(raw)
        pruned_answer = (data.get("pruned_answer") or "").strip()
        used_quote = (data.get("used_quote") or "").strip()
        prune_rationale = (data.get("rationale") or "").strip()
        # enforce length
        if pruned_answer and len(pruned_answer) <= max_chars:
            llm_ok = True
    except Exception as e:
        prune_rationale = f"PRUNE_FALLBACK: {type(e).__name__}: {e}"

    # verification pool & helpers 
    verify_pool = []
    if evidence:
        verify_pool.append(evidence)
    for c in contexts:
        verify_pool.append(c.get("passage", "") or "")
    verify_text = "\n".join(verify_pool)

    def supported(text: str) -> bool:
        return bool(text) and (_contains_norm(verify_text, text) or _contains_norm(evidence, text))

    evidence_supported = bool(evidence) and _contains_norm(verify_text, evidence)

    # accept pruned LLM answer if supported 
    if llm_ok and supported(pruned_answer):
        new_ev = used_quote if supported(used_quote) else (evidence or used_quote)
        note = f"[pruned:{len(answer)}→{len(pruned_answer)}]"
        rec["answer"] = pruned_answer
        rec["evidence"] = new_ev
        rec["rationale"] = (rec.get("rationale") or "") + f" {note}"
        rec["answerable"] = True
        return rec

    # deterministic fallback extraction 
    pool_text = evidence if evidence else verify_text
    best_sentence = _best_support_sentence(pool_text, grounded_q, min_overlap=1)
    fallback_ans = _extract_answer(grounded_q, best_sentence) if best_sentence else None

    if fallback_ans and len(fallback_ans) <= max_chars and supported(fallback_ans):
        rec["answer"] = fallback_ans
        rec["evidence"] = best_sentence or evidence
        rec["rationale"] = (rec.get("rationale") or "") + f" [pruned_fallback:{len(answer)}→{len(fallback_ans)}]"
        rec["answerable"] = True
        return rec

    # if original evidence is supported, keep answerable=True and clip evidence 
    if evidence_supported:
        snippet = evidence.strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars - 3].rstrip() + "..."
        if snippet:
            rec["answer"] = snippet
            rec["evidence"] = evidence
            rec["rationale"] = (rec.get("rationale") or "") + f" [pruned_clip:{len(answer)}→{len(snippet)}]"
            rec["answerable"] = True
            return rec

    # last resort: truly unsupported, flip to unanswerable 
    rec["answer"] = ""
    rec["answerable"] = False
    rec["rationale"] = (rec.get("rationale") or "") + " [pruner: could not find faithful concise span]"
    return rec




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL from grounding stage")
    ap.add_argument("--out", required=True, help="Output JSONL with pruned answers")
    ap.add_argument("--model", default="llama3.1", help="Ollama model tag")
    ap.add_argument("--max_chars", type=int, default=140, help="Max chars for answers")
    ap.add_argument("--ctx-max-chars", type=int, default=900, help="Truncate contexts fed to LLM")
    ap.add_argument("--checkpoint", type=int, default=10, help="fsync every N items")
    args = ap.parse_args()

    cfg = GenConfig(model=args.model)

    # pre-count
    try:
        total = sum(1 for _ in open(args.inp, "r", encoding="utf-8"))
    except Exception:
        total = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    processed = 0
    kept = 0
    pruned = 0
    flipped_false = 0

    print(f"[start prune] model={cfg.model} inp={args.inp} out={args.out} max_chars={args.max_chars}")

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            processed += 1
            rec = json.loads(line)

            before_ans = rec.get("answer","")
            before_len = len(before_ans)
            before_flag = bool(rec.get("answerable", True))

            # run pruner only when needed
            if before_flag and before_len > args.max_chars:
                rec = prune_one(rec, cfg, args.max_chars, args.ctx_max_chars)
                if rec.get("answer") and len(rec.get("answer")) <= args.max_chars and rec.get("answerable", True):
                    pruned += 1
                elif not rec.get("answerable", True):
                    flipped_false += 1
            else:
                kept += 1  # unchanged

            # stamp time
            rec["pruned_at"] = datetime.now(timezone.utc).isoformat()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # progress + checkpointing
            print(f"[progress] {processed}/{total or '?'}", end="\r")
            if processed % args.checkpoint == 0:
                fout.flush()
                os.fsync(fout.fileno())
                print(f"\n[checkpoint] saved {processed}/{total or '?'}")

    print(f"\n[done prune] wrote {args.out}")
    print(f"  total={processed}  kept={kept}  pruned={pruned}  flipped_to_unanswerable={flipped_false}")

if __name__ == "__main__":
    main()
