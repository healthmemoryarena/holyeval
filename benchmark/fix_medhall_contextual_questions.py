"""
After stripping reference_range / status / clinical_significance / etc from
user JSON files, the 16 contextual questions can no longer be answered the
same way (they ask for fields that no longer exist).

This script applies hand-curated rewrites that:
  - Drop "reference range / abnormal flag / above-or-below range" asks
  - Keep the data-retrieval test ("did AI correctly find the value")
  - Strip "(reference X-Y)" and "[abnormal]" tags from known_facts

Idempotent — safe to re-run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark" / "data" / "medhall"


# Hand-curated question rewrites (one entry per case ID).
# Each rewrite preserves: patient persona intro + date reference + indicator list.
# Drops: reference range / abnormal flag / above-below-range asks.
QUESTION_REWRITES: dict[str, str] = {
    "mh_20260520_co_0000": (
        "I'm a 34-year-old male Sales Manager and I stay pretty focused on preventive health "
        "because of my high-pressure work routine. I want to review a specific result from my "
        "last checkup.\n\n"
        "On my 2023-01-12 exam, what was my lipoprotein-associated phospholipase A2 result, "
        "including the unit?"
    ),
    "mh_20260520_co_0001": (
        "I'm a 29-year-old woman working as a marketing manager, and I usually stay on top of "
        "preventive health details because I try to balance job stress with good sleep and exercise.\n\n"
        "At my 2022-09-26 exam, what was my urine red blood cell count, including the unit?"
    ),
    "mh_20260520_co_0002": (
        "I'm a 28-year-old marketing specialist and generally very healthy, so I pay close "
        "attention to preventive lab results from my routine wellness visits.\n\n"
        "At my 2024-03-31 exam, what was my urine red blood cell count, including the unit?"
    ),
    "mh_20260520_co_0003": (
        "I'm a 32-year-old male software engineer and I keep close track of my health metrics "
        "because I spend a lot of time sitting at work. I like reviewing my results from past "
        "checkups to make sure my fitness and metabolic markers still line up with my "
        "preventative approach.\n\n"
        "From my exam on 2023-03-03, what were my recorded body fat percentage, insulin level, "
        "and ECG heart rate values?"
    ),
    "mh_20260520_co_0004": (
        "I'm a 34-year-old male software engineer and I track my health metrics closely, "
        "especially body composition and performance-related measures from my annual exams.\n\n"
        "From my 2022-07-02 exam, what were my body fat percentage, extracellular water, "
        "and muscle mass values?"
    ),
    "mh_20260520_co_0005": (
        "I'm a 34-year-old HR manager in generally excellent health, and I'm reviewing my "
        "routine preventive labs from my annual visit.\n\n"
        "From my exam on 2021-05-02, what were my CRP, high-sensitivity CRP, and ESR results, "
        "including the units?"
    ),
    "mh_20260520_co_0006": (
        "I'm a 32-year-old male software engineer who tracks performance and recovery closely, "
        "so I like to review the exact numbers from my checkups rather than just broad summaries.\n\n"
        "From my exam on 2021-07-12, what were my exact QT interval on the ECG, body fat "
        "percentage, and high-sensitivity CRP values?"
    ),
    "mh_20260520_co_0007": (
        "I'm a 32-year-old woman working in HR, and I track my health closely because I care "
        "a lot about fitness, recovery, and performance. I'm reviewing my latest exam.\n\n"
        "From my exam on 2022-09-25, what were the exact results for my urine microalbumin, "
        "total body water, and resting heart rate, including units?"
    ),
    "mh_20260520_co_0008": (
        "I'm a 32-year-old male software engineer who tracks my health data closely.\n\n"
        "From my 2022-08-05 exam, can you list the exact result for the inhalant allergen panel, "
        "C-peptide, and ESR, including units?"
    ),
    "mh_20260520_co_0009": (
        "I'm a 28-year-old male software engineer who tracks fitness and biometrics closely, "
        "so after my annual exam I want a precise readout of a few specific measurements.\n\n"
        "From my 2021-12-16 exam, give me the exact result and unit for each of these four "
        "indicators: resting heart rate (physical exam), microalbuminuria, urine red blood cells, "
        "and high-sensitivity CRP."
    ),
    "mh_20260520_co_0010": (
        "I'm a 28-year-old woman working in HR, and I track my health closely because I train "
        "regularly.\n\n"
        "From my 2022-04-20 exam, can you give me the exact numeric result for each of these "
        "four markers: progesterone, lipid arteriosclerosis index, body fat percentage, and "
        "urine red blood cell count? Please include units."
    ),
    "mh_20260520_co_0011": (
        "I'm a 52-year-old male Senior Civil Engineer managing hypertension carefully, and I like "
        "to review my lab data carefully after each visit.\n\n"
        "Looking at my exam from 2022-03-24, can you list the exact result and unit for the "
        "kidney-related urine tests recorded that day?"
    ),
    "mh_20260520_co_0012": (
        "I'm a 52-year-old male civil engineer managing diabetes and high blood pressure, and "
        "I like to audit my health data carefully after each visit.\n\n"
        "For my exam on 2021-11-06, can you pull the exact numbers for my LDL, oxidized LDL, "
        "arteriosclerosis index, and hip circumference, including units?"
    ),
    "mh_20260520_co_0013": (
        "I'm a 58-year-old male taxi driver with stage 3 chronic kidney disease and high blood "
        "pressure.\n\n"
        "From my 2022-10-16 exam, give me the exact result and unit for each of these four "
        "indicators: body fat percentage, blood urea nitrogen, microalbuminuria, and "
        "high-sensitivity CRP."
    ),
    "mh_20260520_co_0014": (
        "I'm a 28-year-old woman working in marketing, and I stay on top of my asthma and "
        "routine checkups.\n\n"
        "From my exam on 2022-09-08, can you pull the exact results for my ESR, serum amyloid A, "
        "resting heart rate, and urine casts, including units?"
    ),
    "mh_20260520_co_0015": (
        "I'm a 26-year-old graphic designer with hyperthyroidism, and I want to review my latest "
        "exam carefully.\n\n"
        "From my 2024-04-17 exam, what were the exact results for my eosinophil percentage, GGT, "
        "and reverse T3, including units?"
    ),
}


# Strip "(reference X-Y)" and "[abnormal/normal]" tags from known_facts strings
KF_REF_RE = re.compile(r"\s*\(reference[^)]*\)\s*", re.IGNORECASE)
KF_TAG_RE = re.compile(r"\s*\[(?:abnormal|normal|abnormal_high|abnormal_low|high|low)\]\s*", re.IGNORECASE)


def clean_known_fact(f: str) -> str:
    out = KF_REF_RE.sub(" ", f)
    out = KF_TAG_RE.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


def patch_file(path: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Returns (questions_changed, known_facts_cleaned, items_skipped)."""
    items = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    q_changed = 0
    kf_changed = 0
    skipped = 0
    for item in items:
        is_contextual = any(
            t.startswith("hallu_type:") and t.split(":")[1] == "contextual"
            for t in item.get("tags", [])
        )
        if not is_contextual:
            continue

        # Question rewrite
        new_q = QUESTION_REWRITES.get(item["id"])
        if new_q is None:
            skipped += 1
            continue
        old_q = item["user"]["strict_inputs"][0]
        if old_q != new_q:
            item["user"]["strict_inputs"][0] = new_q
            q_changed += 1
            # Update eval.context to match the question intro
            ev = item.get("eval") or {}
            if "context" in ev:
                ev["context"] = new_q.split("\n\n")[0]

        # known_facts cleanup
        ev = item.get("eval") or {}
        kf = ev.get("known_facts")
        if isinstance(kf, list):
            new_kf = [clean_known_fact(s) if isinstance(s, str) else s for s in kf]
            if new_kf != kf:
                ev["known_facts"] = new_kf
                kf_changed += sum(1 for a, b in zip(kf, new_kf) if a != b)

    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return q_changed, kf_changed, skipped


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for fname in ["contextual-20260520.jsonl", "full-20260520.jsonl"]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"!! missing {path}")
            continue
        q, kf, skip = patch_file(path, dry_run=args.dry_run)
        print(f"  {fname}: {q} questions rewritten, {kf} known_facts cleaned, {skip} skipped")


if __name__ == "__main__":
    main()
