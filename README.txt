run commands:

python scripts/gen_questions_ollama.py --input data/raw/cf_passages.csv --input-col passage --model llama3.1 --out data/processed/cf_drafts.jsonl --num 2


python scripts/bm25_retrieval.py --passages data/raw/cf_passages.csv --drafts data/processed/cf_drafts.jsonl --out data/processed/cf_drafts_with_ctx.jsonl --k 5


python scripts/ground_questions.py --in data/processed/cf_drafts_with_ctx.jsonl --out data/processed/cf_grounded.jsonl --model llama3.1 --maxctx 5


python scripts/prune_answers.py --in data/processed/cf_grounded.jsonl --out data/processed/cf_grounded_pruned.jsonl --model llama3.1 --max_chars 140 --checkpoint 10


python scripts/salvage_unanswerable.py --in data/processed/cf_grounded_pruned.jsonl --passages data/raw/cf_passages.csv --out data/final/cf_grounded_pruned_salvaged.jsonl --model llama3.1 --max_chars 140 --topn 3 --checkpoint 10

python test_scripts/salvage_unanswerablev4.py --in data/processed/cf_grounded_pruned.jsonl --passages data/raw/cf_passages.csv --out data/final/cf_grounded_pruned_salvagedv4.jsonl --max_chars 140 --topn 4 --checkpoint 10

