"""
Build a sample subset of MedHall full-20260520 — 5 cases per hallucination type,
randomly drawn but seeded for reproducibility.

Output:
  benchmark/data/medhall/sample-20260520.jsonl       (25 cases, all categories merged)
  benchmark/data/medhall/factual-sample-20260520.jsonl    (5)
  benchmark/data/medhall/contextual-sample-20260520.jsonl (5)
  benchmark/data/medhall/citation-sample-20260520.jsonl   (5)
  benchmark/data/medhall/numerical-sample-20260520.jsonl  (5)
  benchmark/data/medhall/relational-sample-20260520.jsonl (5)

Sampling rule:
  - Seed 42 (deterministic)
  - 5 per category — try to span difficulty (one each from l3/l4/l5 first, then random
    fill if a category has fewer than 5 difficulties)
  - Preserve original record verbatim (no answer stripping happens here; that's
    the HF public-release uploader's job)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark" / "data" / "medhall"
DATE_SUFFIX = "20260520"
SAMPLE_PER_CAT = 5
SEED = 42

CATEGORIES = ["factual", "contextual", "citation", "numerical", "relational"]


def _cat_of(item: dict) -> str | None:
    for t in item.get("tags") or []:
        if isinstance(t, str) and t.startswith("hallu_type:"):
            return t.split(":", 1)[1]
    return None


def _diff_of(item: dict) -> str | None:
    for t in item.get("tags") or []:
        if isinstance(t, str) and t.startswith("difficulty:"):
            return t.split(":", 1)[1]
    return None


def main() -> None:
    full_path = DATA_DIR / f"full-{DATE_SUFFIX}.jsonl"
    items = [json.loads(ln) for ln in full_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        cat = _cat_of(it)
        if cat in CATEGORIES:
            by_cat[cat].append(it)

    rng = random.Random(SEED)
    picks: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        pool = by_cat[cat]
        if len(pool) <= SAMPLE_PER_CAT:
            picks[cat] = list(pool)
            continue
        # Stratify by difficulty when possible
        by_diff: dict[str, list[dict]] = defaultdict(list)
        for it in pool:
            by_diff[_diff_of(it) or "?"].append(it)
        chosen: list[dict] = []
        for diff in ("l3", "l4", "l5"):
            cand = by_diff.get(diff) or []
            if cand:
                chosen.append(rng.choice(cand))
        remaining = [it for it in pool if it not in chosen]
        rng.shuffle(remaining)
        while len(chosen) < SAMPLE_PER_CAT and remaining:
            chosen.append(remaining.pop())
        picks[cat] = chosen

    # Write per-category files + merged
    all_picked: list[dict] = []
    for cat, cases in picks.items():
        path = DATA_DIR / f"{cat}-sample-{DATE_SUFFIX}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for it in cases:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {path.name}: {len(cases)} cases")
        all_picked.extend(cases)

    merged_path = DATA_DIR / f"sample-{DATE_SUFFIX}.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for it in all_picked:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  {merged_path.name}: {len(all_picked)} cases")


if __name__ == "__main__":
    main()
