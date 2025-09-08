
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import requests

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are an expert assessment designer.
Given a single passage, create high-quality QUESTION(s) that can be answered solely from that passage.
The question should be unambiguous, grounded, and require understanding of the passage (not external facts).
Return STRICT JSON with keys: question, answer, difficulty (easy|medium|hard), rationale.
Do NOT include anything else.
"""

USER_TEMPLATE = """PASSAGE:
\"\"\"
{passage}
\"\"\"

Write {num} question(s) that each can be answered ONLY using the passage above.
Aim for coverage and clarity.
Output a JSON list. Each element must have: question, answer, difficulty, rationale.
"""


@dataclass
class GenConfig:
    model: str
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 8192
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


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
    if cfg.seed is not None:
        payload["options"]["seed"] = cfg.seed
    if cfg.stop:
        payload["stop"] = cfg.stop

    resp = requests.post(cfg.endpoint, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def _read_df(path: str, input_col: str, id_col: Optional[str]) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl") or path.endswith(".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Unsupported input format. Use .csv or .jsonl")

    if input_col not in df.columns:
        raise ValueError(f"Missing required column '{input_col}' in {path}")

    if id_col and id_col not in df.columns:
        id_col = None

    if id_col is None:
        df = df.copy()
        df["__row_id"] = range(len(df))
        id_col = "__row_id"

    return df[[id_col, input_col]].rename(columns={id_col: "id", input_col: "passage"})


def parse_json_list(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    # Try straight JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except Exception:
        # Fallback: try to extract the JSON array between first '[' and last ']'
        if "[" in text and "]" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            snippet = text[start:end]
            return json.loads(snippet)
        raise
    raise ValueError("Unexpected JSON structure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV or JSONL with passages")
    ap.add_argument("--input-col", default="passage", help="Column/field containing passages")
    ap.add_argument("--id-col", default="id", help="Optional id column name")
    ap.add_argument("--model", default="llama3.1", help="Ollama model tag")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Ollama HTTP endpoint")
    ap.add_argument("--num", type=int, default=1, help="Questions per passage")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = GenConfig(
        model=args.model,
        endpoint=args.endpoint,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        seed=args.seed,
    )

    df = _read_df(args.input, args.input_col, args.id_col)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            passage = str(row["passage"]).strip()
            if not passage:
                continue
            print(f"Generating for passage ID {row['id']}...")
            prompt = USER_TEMPLATE.format(passage=passage, num=args.num)
            raw = None  # keeps linters happy if an exception occurs before assignment
            try:
                raw = call_ollama(cfg, prompt)
                print(f"Received response for passage ID {row['id']} ({len(raw)} characters).")
                items = parse_json_list(raw)
            except Exception as e:
                items = [{
                    "error": f"{type(e).__name__}: {e}",
                    "raw": raw
                }]
            record = {
                "source_id": row["id"],
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": cfg.model,
                "num_requested": args.num,
                "items": items,
                "passage": passage,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
