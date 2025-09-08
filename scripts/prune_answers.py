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
  used_quote (string),
  rationale (string)
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

#helpers 

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

def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", _norm(text or "")))

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
    negative_patterns = [
        r"\bno\s+significant\b", r"\bno\s+evidence\b", r"\bno\s+correlation\b", r"\bno\s+association\b",
        r"\bnot\s+significant\b", r"\bnot\s+associated\b", r"\bnot\s+correlated\b",
        r"\bwas\s+not\b", r"\bwere\s+not\b", r"\bis\s+not\b", r"\bare\s+not\b",
        r"\bdoes\s+not\b", r"\bdid\s+not\b", r"\bhad\s+no\b", r"\bhave\s+no\b",
        r"\bwithout\b", r"\babsence\s+of\b", r"\black\s+of\b", r"\bfailed\s+to\b",
        r"\bno\s+difference\b", r"\bno\s+effect\b", r"\bno\s+impact\b"
    ]
    for pattern in negative_patterns:
        if re.search(pattern, s):
            return "No"
    positive_patterns = [
        r"\bsignificant\s+correlation\b", r"\bsignificant\s+association\b",
        r"\bpositive\s+correlation\b", r"\bpositive\s+association\b",
        r"\bwas\s+associated\b", r"\bwere\s+associated\b", r"\bwas\s+correlated\b",
        r"\bwere\s+correlated\b", r"\bis\s+associated\b", r"\bare\s+associated\b",
        r"\bstrong\s+correlation\b", r"\bstrong\s+association\b",
        r"\bdirect\s+correlation\b", r"\bdirect\s+association\b",
        r"\bclear\s+correlation\b", r"\bclear\s+association\b"
    ]
    for pattern in positive_patterns:
        if re.search(pattern, s):
            return "Yes"
    if re.search(r"\bcorrelation\b|\bassociation\b", s):
        return "Yes"
    if re.search(r"\bnot\b|\bno\b|\bwithout\b|\babsence\b", s):
        return None
    if re.search(r"\bwas\b|\bwere\b|\bis\b|\bare\b|\bdetected\b|\bobserved\b|\bfound\b", s):
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
    first_word = qn.split(" ", 1)[0] if qn else ""
    yesno_starters = {
        "is", "are", "was", "were", "do", "does", "did", "can", "could", "should",
        "would", "will", "has", "have", "had", "must", "may", "might"
    }
    if first_word in yesno_starters:
        return "yesno"
    yesno_patterns = [
        r"^is\s+.+\?", r"^are\s+.+\?", r"^was\s+.+\?", r"^were\s+.+\?",
        r"^do\s+.+\?", r"^does\s+.+\?", r"^did\s+.+\?", r"^can\s+.+\?",
        r"^could\s+.+\?", r"^should\s+.+\?", r"^has\s+.+\?", r"^have\s+.+\?",
        r"^had\s+.+\?", r"^doesn'?t\s+.+\?", r"^isn'?t\s+.+\?", r"^aren'?t\s+.+\?",
        r"^wasn'?t\s+.+\?", r"^weren'?t\s+.+\?", r"^don'?t\s+.+\?", r"^didn'?t\s+.+\?",
        r"^can'?t\s+.+\?", r"^couldn'?t\s+.+\?", r"^shouldn'?t\s+.+\?"
    ]
    for pattern in yesno_patterns:
        if re.match(pattern, qn):
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
    snippet = sentence.strip()
    if len(snippet) > 160: snippet = snippet[:157].rstrip()
    return snippet or None

def _shorten_to_phrase(sent: str, max_chars: int) -> str:
    s = (sent or "").strip()
    if len(s) <= max_chars:
        return s
    cut = max(s.rfind(". ", 0, max_chars), s.rfind("; ", 0, max_chars), s.rfind(", ", 0, max_chars), s.rfind(" — ", 0, max_chars))
    if cut > 40:
        return s[:cut].rstrip()
    cut = s.rfind(" ", 0, max_chars)
    if cut > 40:
        return s[:cut].rstrip()
    m = re.match(r"^(.{0,%d})(?:\b|$)" % (max_chars-1), s)
    return (m.group(1) if m else s[:max_chars-1]).rstrip()

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

#prefix detection helpers 

def _raw_prefix_clip(text: str, max_chars: int) -> str:
    """Exact naive prefix: first max_chars-3 chars + '...' (no trimming)."""
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars - 3] + "..."

def _has_punctuation_before_cap(text: str, max_chars: int) -> bool:
    """
    True if a natural boundary (.[!?;,:]) exists near the end of the kept window,
    meaning we *could* stop earlier without a dumb prefix cut.
    """
    if not text or len(text) <= max_chars:
        return False
    window = text[:max_chars - 3]
    return bool(re.search(r"[\.!?;,:]\s*[^\s]*$", window[-30:]))

def _is_bare_prefix(ans: str, source: str, max_chars: int) -> bool:
    """
    Flags cases where `ans` is the *start* of source (no ellipsis) and likely truncated:
      - source starts with ans (normalized-space compare),
      - answer is near the cap (>= 85% of cap or within 8 chars),
      - AND (no terminal punctuation) OR (ends mid-word),
      - AND there wasn't an earlier punctuation boundary in the kept window.
    """
    if not ans or not source:
        return False
    a = _normalize_spaces(ans)
    s = _normalize_spaces(source)
    if not s.startswith(a):
        return False
    if len(a) < 20:
        return False  # avoid flagging tiny snippets
    near_cap = (len(ans) >= int(0.85 * max_chars)) or (max_chars - len(ans) <= 8)
    no_terminal = not re.search(r"[.!?]\s*$", ans)
    # mid-word cut if both last char of ans and next char in source are alnum
    next_char = s[len(a):len(a)+1]
    prev_char = a[-1:] if a else ""
    midword = bool(prev_char and prev_char[-1].isalnum() and next_char and next_char.isalnum())
    if near_cap and (no_terminal or midword) and not _has_punctuation_before_cap(source, max_chars):
        return True
    return False

def _is_prefix_clip_answer(ans: str, evidence: str, contexts: List[Dict[str, Any]], max_chars: int) -> bool:
    """
    True iff ans:
      - equals the raw prefix clip (with ... or …) of evidence/passage with no earlier punctuation boundary, OR
      - is a bare, near-cap prefix of the evidence/passage that ends mid-word or lacks terminal punctuation.
    """
    if not ans:
        return False

    norm_ans = _normalize_spaces(ans)

    def _match_source(src: str) -> bool:
        if not src:
            return False
        # Exact naive "..." prefix
        raw = _raw_prefix_clip(src, max_chars)
        if _normalize_spaces(raw) == norm_ans and not _has_punctuation_before_cap(src, max_chars):
            return True
        # Single ellipsis char variation
        if raw.endswith("...") and norm_ans == _normalize_spaces(raw[:-3] + "…") and not _has_punctuation_before_cap(src, max_chars):
            return True
        # Bare prefix (no ellipsis) near cap / mid-word
        if _is_bare_prefix(ans, src, max_chars):
            return True
        return False

    if evidence and _match_source(evidence):
        return True
    for c in (contexts or []):
        if _match_source((c.get("passage", "") or "")):
            return True
    return False

#llm plumbing

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

#pruning core 

def prune_one(rec: Dict[str, Any], cfg: GenConfig, max_chars: int, ctx_max_chars: int) -> Dict[str, Any]:
    answerable = bool(rec.get("answerable", True))
    answer = rec.get("answer", "") or ""
    grounded_q = rec.get("grounded_question") or rec.get("draft_question") or ""
    evidence = rec.get("evidence", "") or ""
    contexts = rec.get("contexts", []) or []

    #if already unanswerable, pass through
    if not answerable:
        return rec

    #attempt to salvage a clean phrase from best sentence
    def _salvage_from_best(current_rationale_suffix: str) -> Dict[str, Any]:
        # Build verify text
        verify_pool = []
        if evidence:
            verify_pool.append(evidence)
        for c in contexts:
            verify_pool.append(c.get("passage", "") or "")
        verify_text = "\n".join(verify_pool)

        def supported(text: str) -> bool:
            if not text:
                return False
            tnorm = _norm(text)
            if _contains_norm(verify_text, tnorm) or _contains_norm(evidence, tnorm):
                return True
            tt = _token_set(text)
            if not tt:
                return False
            for sent in _sentences(verify_text):
                st = _token_set(sent)
                if not st:
                    continue
                j = len(tt & st) / max(1, len(tt | st))
                if j >= 0.7:
                    return True
            return False

        pool_text = evidence if evidence else verify_text
        best_sentence = _best_support_sentence(pool_text, grounded_q, min_overlap=1)
        if best_sentence:
            phrase = _shorten_to_phrase(best_sentence, max_chars).strip()
            if phrase and phrase != rec.get("answer","") and supported(phrase) and not _is_prefix_clip_answer(phrase, evidence, contexts, max_chars):
                rec["answer"] = phrase
                rec["evidence"] = best_sentence
                rec["rationale"] = (rec.get("rationale") or "") + f" {current_rationale_suffix}"
                rec["answerable"] = True
                return rec

        # No salvage → flip to unanswerable
        rec["answer"] = ""
        rec["answerable"] = False
        rec["rationale"] = (rec.get("rationale") or "") + " [prefix-clip→unanswerable]"
        return rec

    # If the current answer is a prefix-clip of evidence/passage, try to salvage before flipping
    if _is_prefix_clip_answer(answer, evidence, contexts, max_chars):
        return _salvage_from_best("[prefix-salvaged]")

    # If the answer is short and not a prefix-stub, nothing to do
    if len(answer) <= max_chars:
        return rec

    # compact passages for LLM context
    compact_passage = ""
    if contexts:
        chunks = []
        for c in contexts[:4]:
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
        if pruned_answer and len(pruned_answer) <= max_chars:
            llm_ok = True
    except Exception as e:
        prune_rationale = f"PRUNE_FALLBACK: {type(e).__name__}: {e}"

    #build verify text
    verify_pool = []
    if evidence:
        verify_pool.append(evidence)
    for c in contexts:
        verify_pool.append(c.get("passage", "") or "")
    verify_text = "\n".join(verify_pool)

    #support check
    def supported(text: str) -> bool:
        if not text:
            return False
        tnorm = _norm(text)
        if _contains_norm(verify_text, tnorm) or _contains_norm(evidence, tnorm):
            return True
        tt = _token_set(text)
        if not tt:
            return False
        for sent in _sentences(verify_text):
            st = _token_set(sent)
            if not st:
                continue
            j = len(tt & st) / max(1, len(tt | st))
            if j >= 0.7:
                return True
        return False

    # accept LLM answer if supported
    if llm_ok and supported(pruned_answer):
        rec["answer"] = pruned_answer
        rec["evidence"] = used_quote if supported(used_quote) else (evidence or used_quote)
        rec["rationale"] = (rec.get("rationale") or "") + f" [pruned:{len(answer)}→{len(pruned_answer)}]"
        rec["answerable"] = True
        if _is_prefix_clip_answer(rec["answer"], evidence, contexts, max_chars):
            return _salvage_from_best("[prefix-salvaged]")
        return rec

    # deterministic fallback: best sentence → extract / shorten
    pool_text = evidence if evidence else verify_text
    best_sentence = _best_support_sentence(pool_text, grounded_q, min_overlap=1)
    fallback_ans = _extract_answer(grounded_q, best_sentence) if best_sentence else None

    if fallback_ans and len(fallback_ans) <= max_chars and supported(fallback_ans):
        rec["answer"] = fallback_ans
        rec["evidence"] = best_sentence or evidence
        rec["rationale"] = (rec.get("rationale") or "") + f" [pruned_fallback:{len(answer)}→{len(fallback_ans)}]"
        rec["answerable"] = True
        if _is_prefix_clip_answer(rec["answer"], evidence, contexts, max_chars):
            return _salvage_from_best("[prefix-salvaged]")
        return rec

    if best_sentence:
        phrase = _shorten_to_phrase(best_sentence, max_chars)
        if phrase and supported(phrase):
            rec["answer"] = phrase
            rec["evidence"] = best_sentence
            rec["rationale"] = (rec.get("rationale") or "") + " [pruned_phrase]"
            rec["answerable"] = True
            if _is_prefix_clip_answer(rec["answer"], evidence, contexts, max_chars):
                return _salvage_from_best("[prefix-salvaged]")
            return rec

    # evidence clipping path (word-safe)
    evidence_supported = bool(evidence) and supported(evidence)
    if evidence_supported:
        s = evidence.strip()
        if len(s) > max_chars:
            s = _shorten_to_phrase(s, max_chars)
            if len(s) < len(evidence.strip()):
                s = s + "..."
        if s:
            rec["answer"] = s
            rec["evidence"] = evidence
            rec["rationale"] = (rec.get("rationale") or "") + f" [pruned_clip:{len(answer)}→{len(s)}]"
            rec["answerable"] = True
            if _is_prefix_clip_answer(rec["answer"], evidence, contexts, max_chars):
                return _salvage_from_best("[prefix-salvaged]")
            return rec

    # fail: flip to unanswerable
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

            before_ans = rec.get("answer","") or ""
            before_flag = bool(rec.get("answerable", True))

            needs_prune = before_flag and (
                len(before_ans) > args.max_chars or
                _is_prefix_clip_answer(before_ans, rec.get("evidence","") or "", rec.get("contexts",[]) or [], args.max_chars)
            )

            if needs_prune:
                rec = prune_one(rec, cfg, args.max_chars, args.ctx_max_chars)
                if rec.get("answer") and len(rec.get("answer")) <= args.max_chars and rec.get("answerable", True):
                    pruned += 1
                elif not rec.get("answerable", True):
                    flipped_false += 1
            else:
                kept += 1

            rec["pruned_at"] = datetime.now(timezone.utc).isoformat()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[progress] {processed}/{total or '?'}", end="\r")
            if processed % args.checkpoint == 0:
                fout.flush()
                os.fsync(fout.fileno())
                print(f"\n[checkpoint] saved {processed}/{total or '?'}")

    print(f"\n[done prune] wrote {args.out}")
    print(f"  total={processed}  kept={kept}  pruned={pruned}  flipped_to_unanswerable={flipped_false}")

if __name__ == "__main__":
    main()




