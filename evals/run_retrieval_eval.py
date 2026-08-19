"""Run Recall@K and MRR against a small, versioned retrieval evaluation set."""

import argparse
import json
from pathlib import Path

from app.services import search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/qa_dataset.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("评测集为空")
    hits, reciprocal_ranks = 0, []
    details = []
    for row in rows:
        results = search(row["question"], row["task_id"], args.k)
        ranks = [index for index, item in enumerate(results, 1) if item.document == row["relevant_document"]]
        hit = bool(ranks)
        hits += hit
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
        details.append({"question": row["question"], "hit": hit, "rank": ranks[0] if ranks else None})
    print(json.dumps({"samples": len(rows), f"recall@{args.k}": hits / len(rows), "mrr": sum(reciprocal_ranks) / len(rows), "details": details}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
