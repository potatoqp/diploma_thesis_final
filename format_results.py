import json
import pandas as pd

rows = []
with open("cf_drafts_small.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        for item in obj.get("items", []):
            rows.append({
                "source_id": obj["source_id"],
                "question": item.get("question"),
                "answer": item.get("answer"),
                "difficulty": item.get("difficulty"),
                "rationale": item.get("rationale")
            })

pd.DataFrame(rows).to_csv("cf_drafts_small_readable.csv", index=False)
