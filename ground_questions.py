
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timezone

import requests

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a careful grader and question rewriter.
You receive: (1) a DRAFT question, (2) retrieved PASSAGES with doc_ids.

Rules:
- Produce a GROUNDED_QUESTION answerable ONLY from the passages.
- NEVER infer or guess missing details (numbers, names, dates). If a requested detail isn't explicitly in the passages, set answerable=false.
- If the draft asks for more precision than the passages contain, either
  (a) rewrite it to match what IS stated (coarser granularity), OR
  (b) set answerable=false.
- If answerable=true, also provide:
  - answer: a short verbatim phrase copied from the passages (do not paraphrase),
  - evidence: a short exact quote from the passages that justifies the answer.
- If answerable=false, answer and evidence must be empty strings.

Return STRICT JSON with keys:
  grounded_question (string),
  answer (string),
  rationale (string),
  used_doc_ids (array of integers),
  answerable (boolean),
  evidence (string)
Return ONLY JSON.
"""

USER_TEMPLATE = """DRAFT QUESTION:
{question}

RETRIEVED PASSAGES (doc_id :: text):
{contexts}

Instructions:
- If the draft is already grounded, keep it but tighten wording.
- If not fully supported, constrain it to facts present in the passages.
- Set answerable=false only if no rewrite can be supported from the passages.
"""


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _contains_norm(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)

def _sentences(text: str):
    # primary split on ., !, ? or newlines
    chunks = re.split(r'(?<=[\.\!\?])\s+|\n+', text or "")
    chunks = [c.strip() for c in chunks if c and c.strip()]

    # fallback: if it's basically one monster sentence, split further on commas/semicolons
    if len(chunks) == 1 and len(chunks[0]) > 400:
        chunks = re.split(r'[;,]\s+', chunks[0])
        chunks = [c.strip() for c in chunks if c and c.strip()]

    return chunks


STOPWORDS = set("""
the a an and or of to in on for with as by from at that this those these be is are was were has have had not no
""".split())

def _key_tokens(text: str):
    toks = re.findall(r"[A-Za-z0-9']+", text or "")
    return [t.lower() for t in toks if t.lower() not in STOPWORDS]

def _best_support_sentence(context_text: str, question_text: str, min_overlap: int = 1) -> str:
    """
    Pick the sentence in context with the highest token overlap with the (grounded) question.
    Overlap threshold is low (>=1) to be forgiving; prefer shorter sentence on ties.
    """
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


YESNO_PREFIXES = tuple([
    "is ", "are ", "was ", "were ", "do ", "does ", "did ",
    "can ", "could ", "should ", "has ", "have ", "had ",
    "isn", "aren", "wasn", "weren", "doesn", "don", "didn", "can’t", "cannot", "couldn", "hasn", "haven", "hadn"
])
YESNO_STARTS = (
    lambda q: _norm(q).startswith(YESNO_PREFIXES) or _norm(q).startswith("does ") or _norm(q).startswith("did ")
)

def _question_type(q: str) -> str:
    qn = _norm(q)
    if YESNO_STARTS(qn) or qn.endswith("?") and qn.split(" ", 1)[0] in {"is","are","was","were","do","does","did","can","could","should","has","have","had"}:
        return "yesno"
    if any(k in qn for k in ["what age", "at what age", "how old", "what year", "what month", "how many years", "how many months"]):
        return "age"
    if any(k in qn for k in ["which", "who", "what", "name the", "what is the", "what was the"]):
        return "which"
    if any(k in qn for k in ["method", "technique", "test", "assay"]):
        return "method"
    return "other"

def _extract_yesno(sentence: str, question: Optional[str] = None) -> Optional[str]:
    s = _norm(sentence)
    q = _norm(question or "")

    # special handling for correlation/association questions
    if "correlat" in q or "associat" in q:
        # explicit negatives around the keyword
        if re.search(r"\bno\s+(?:significant\s+)?(correlation|association)\b", s):
            return "No"
        if re.search(r"\b(?:not|non[-\s]?)\w*\s+(?:correlat\w*|associat\w*)", s):
            return "No"
        # positive mentions
        if re.search(r"\b(correlat\w*|associat\w*)\b", s):
            return "Yes"
        return None  # neither observed

    # generic yes/no (keep broad, but don't let stray 'no' elsewhere dominate)
    # If the question is yes/no but not correlation-specific, prefer explicit negations of the predicate
    if re.search(r"\b(not\s+|no\s+)(evidence|effect|difference|increase|decrease|benefit|protect\w*|detect\w*|present)\b", s):
        return "No"
    if re.search(r"\b(present|detected|observed|increased|decreased|elevated|associated|was|were|is|are|exists?)\b", s):
        return "Yes"
    # fallback: any plain 'no' as last resort
    if re.search(r"\bno\b|\bnot\b|\bcannot\b|\bcan\s+not\b|\bcould\s+not\b", s):
        return "No"
    return None


def _extract_age(sentence: str) -> Optional[str]:
    # common phrasing: "by the age of X", "at X months/years", "between X and Y months/years"
    m = re.search(r"\b(by|at)\s+the\s+age\s+of\s+([^.,;]+)", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip()
    m = re.search(r"\b(at|by)\s+(\d+\s*(?:to|-|–)\s*\d+\s*(?:months?|years?)|\d+\s*(?:months?|years?))\b", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip()
    # fallback: any X months/years phrase
    m = re.search(r"\b\d+\s*(?:to|-|–)\s*\d+\s*(?:months?|years?)\b", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    m = re.search(r"\b\d+\s*(?:months?|years?)\b", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    # non-numeric but useful span in your dataset ("months to year")
    m = re.search(r"\bmonths?\s+to\s+year\b", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None

def _extract_method(sentence: str) -> Optional[str]:
    # pick span after "by/using/with" up to punctuation
    m = re.search(r"\b(by|using|with)\s+([^.;:]+)", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip()
    # look for method-ish nouns near 'analysis', 'test', 'assay'
    m = re.search(r"\b([A-Za-z \-]*?(analysis|assay|test|technique)[^.;:]*)", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None

def _extract_which(sentence: str) -> Optional[str]:
    # try to grab the NP after copula or 'was/were ...'
    m = re.search(r"\b(?:was|were|is|are)\s+([^.;:,]+)", sentence, flags=re.IGNORECASE)
    if m:
        # trim trailing clutter
        ans = m.group(1).strip()
        ans = re.split(r",|;|\.|\(|\)|—|-", ans)[0].strip()
        return ans if ans else None
    # association phrasing: "associated with X"
    m = re.search(r"\bassociated with\s+([^.;:,]+)", sentence, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def _extract_answer(question: str, sentence: str) -> Optional[str]:
    qtype = _question_type(question)
    # NOTE: intentionally DO NOT handle yes/no here.
    if qtype == "age":
        return _extract_age(sentence)
    if qtype == "method":
        return _extract_method(sentence)
    if qtype == "which":
        return _extract_which(sentence)
    # fallback: short snippet from the evidence sentence
    snippet = sentence.strip()
    if len(snippet) > 180:
        snippet = snippet[:177] + "..."
    return snippet or None




@dataclass
class GenConfig:
    model: str
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.2 #reduce randomness
    top_p: float = 0.9
    num_ctx: int = 8192
    seed: Optional[int] = None

def call_ollama(cfg: GenConfig, prompt: str) -> str:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_ctx": cfg.num_ctx,
        },
        "system": SYSTEM_PROMPT,
        "stream": False,
    }
    r = requests.post(cfg.endpoint, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def compact(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + " …[truncated]"

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="Input JSONL (from bm25_retrieval.py), e.g. cf_drafts_with_ctx.jsonl")
    ap.add_argument("--out", required=True,
                    help="Output JSONL, e.g. cf_grounded.jsonl")
    ap.add_argument("--model", default="llama3.1",
                    help="Ollama model tag (default: llama3.1)")
    ap.add_argument("--maxctx", type=int, default=5,
                    help="Use top-N contexts per question (default: 5)")
    ap.add_argument("--ctx-max-chars", type=int, default=1200,
                    help="Truncate each context to this many chars (default: 1200)")
    args = ap.parse_args()

    cfg = GenConfig(model=args.model)

    # pre-count for progress
    try:
        total = sum(1 for _ in open(args.inp, "r", encoding="utf-8"))
    except Exception:
        total = 0

    processed = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[start] model={cfg.model} inp={args.inp} out={args.out} maxctx={args.maxctx}")

    with open(args.inp, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:

        for line in fin:
            processed += 1
            obj = json.loads(line)
            question = obj.get("question", "")
            contexts = obj.get("contexts", [])[:args.maxctx]

            # build contexts string; truncate per passage
            ctx_lines = []
            used_ids_default = []
            for c in contexts:
                doc_id = c.get("doc_id")
                used_ids_default.append(doc_id)
                passage = c.get("passage", "")
                ctx_lines.append(f"{doc_id} :: {compact(passage, args.ctx_max_chars)}")
            ctx_str = "\n".join(ctx_lines)

            prompt = USER_TEMPLATE.format(question=question, contexts=ctx_str)

            raw = None
            try:
                raw = call_ollama(cfg, prompt)
                data = parse_loose_json(raw)

                grounded = data.get("grounded_question", question)
                rationale = data.get("rationale", "")
                used_doc_ids = data.get("used_doc_ids", used_ids_default)
                answerable_model = bool(data.get("answerable", True))
                answer = data.get("answer", "")
                evidence = data.get("evidence", "")

            except Exception as e:
                grounded = question
                rationale = f"FALLBACK: {type(e).__name__} - {e}"
                used_doc_ids = used_ids_default
                answerable_model = True
                answer = ""
                evidence = ""

            # validator
            ctx_text = "\n".join([c.get("passage", "") for c in contexts])

            # if model didn't provide evidence or answer, try to find a best sentence
            best_sentence = _best_support_sentence(ctx_text, grounded or question, min_overlap=1)

            # decide answerable:
            answerable = answerable_model
            flipped = False

            if not answerable_model:
                # try to salvage: if we found a decent support sentence, mark answerable
                if best_sentence:
                    answerable = True
                    flipped = True

            # ensure evidence exists if answerable
            if answerable and (not evidence):
                evidence = best_sentence or ""

            # ensure answer exists if answerable
            if answerable and not answer:
                if evidence:
                    extracted = _extract_answer(grounded or question, evidence)
                else:
                    extracted = None
                answer = extracted or (evidence[:177] + "..." if evidence and len(evidence) > 180 else (evidence or ""))

            #forbid bare Yes/No answers — replace with an evidence snippet
            if answerable and re.fullmatch(r"\s*(yes|no)\s*", (answer or ""), flags=re.IGNORECASE):
              repl = (evidence or "").strip()
              if len(repl) > 180:
                repl = repl[:177] + "..."
              answer = repl

            # if still nothing usable, mark unanswerable
            if answerable and (not evidence or not answer):
                answerable = False

            # build rationale if empty
            if not rationale:
                used_ids_str = ", ".join(str(i) for i in (used_doc_ids or []))
                if answerable:
                    rationale = f"Supported by doc_id(s) [{used_ids_str}] ("
                    rationale += "model evidence verified" if evidence and _contains_norm(ctx_text, evidence) else "auto evidence from best sentence"
                    rationale += ")"
                else:
                    rationale = f"Marked unanswerable after validation on doc_id(s) [{used_ids_str}]"

            

            rec = {
                "source_id": obj.get("source_id"),
                "question_ix": obj.get("question_ix"),
                "draft_question": question,
                "grounded_question": grounded,
                "answer": answer,
                "rationale": rationale,
                "used_doc_ids": used_doc_ids,
                "answerable": answerable,
                "evidence": evidence,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": cfg.model
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[progress] {processed}/{total or '?'}", end="\r")

            if processed % 10 == 0:
                fout.flush()
                os.fsync(fout.fileno())
                print(f"\n[checkpoint] saved after {processed}/{total or '?'}")


    print(f"[done] Wrote {args.out}")

if __name__ == "__main__":
    main()

