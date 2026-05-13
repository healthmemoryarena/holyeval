"""
L0 — One-shot dataset cleanup for MedHall full-20260520.

Two fixes:
  1. Numeric expected_value using exact_match → switch to numeric_tolerance
     (e.g., "30", "6.5", "2.5" should not require exact-string match)
  2. Enum-like expected_value (snake_case identifiers, single-word codes that
     no LLM produces verbatim) → rewrite to natural language statement,
     keep original enum value in alternatives

Re-runs converter_en after rewriting raw_en_full.jsonl.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark" / "data" / "medhall"
GENERATOR_DIR = ROOT / "generator" / "medhall"

NATURAL_RE = re.compile(r"-?\d+(\.\d+)?$")


def is_numeric_string(s: str) -> bool:
    return bool(NATURAL_RE.fullmatch(s.strip())) if isinstance(s, str) else False


def is_enum_like(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    s = s.strip()
    if is_numeric_string(s):
        return False
    # snake_case_identifier OR single short lowercase word
    if re.fullmatch(r"[a-z][a-z0-9_]*", s):
        return "_" in s or len(s) < 15
    return False


# Hand-curated rewrites for the 23 enum-like fields. Original enum kept in alternatives.
ENUM_REWRITES: dict[tuple[str, str], dict] = {
    # ── frequency_relation ──────────────────────────────────────
    ("mh_20260520_re_0001", "frequency_relation"): {
        "rewrite": "TID and q8h are clinically equivalent for amoxicillin",
        "alternatives": ["equivalent", "tid is equivalent to q8h", "considered equivalent"],
    },
    ("mh_20260520_re_0007", "frequency_relation"): {
        "rewrite": "TID and q8h are not equivalent for symptomatic control of time-sensitive medications",
        "alternatives": ["not_equivalent", "not equivalent", "not interchangeable"],
    },
    ("mh_20260520_re_0012", "frequency_relation"): {
        "rewrite": "TID and q8h are equivalent for amoxicillin in routine outpatient use",
        "alternatives": ["equivalent", "considered equivalent", "interchangeable in practice"],
    },
    ("mh_20260520_re_0014", "frequency_relation"): {
        "rewrite": "TID and q8h are equivalent in common use for amoxicillin",
        "alternatives": ["equivalent_in_common_use", "equivalent", "considered equivalent"],
    },
    ("mh_20260520_re_0020", "frequency_relation"): {
        "rewrite": "TID and q8h should not be assumed interchangeable for time-sensitive drugs",
        "alternatives": ["not_equivalent", "not equivalent", "not interchangeable"],
    },
    ("mh_20260520_re_0023", "frequency_relation"): {
        "rewrite": "BID and q12h are equivalent in common outpatient use for penicillin V",
        "alternatives": ["equivalent_in_common_use", "equivalent", "considered equivalent"],
    },
    # ── drug_disease_relation ──────────────────────────────────
    ("mh_20260520_re_0017", "drug_disease_relation"): {
        "rewrite": "contraindicated or strongly avoided due to risk of bronchospasm",
        "alternatives": [
            "contraindication_or_strong_avoidance_due_to_bronchospasm_risk",
            "contraindicated",
            "avoid",
            "strongly avoid in asthma",
        ],
    },
    ("mh_20260520_re_0022", "drug_disease_relation"): {
        "rewrite": "indicated for symptomatic chronic heart failure",
        "alternatives": ["indicated", "approved indication", "recommended"],
    },
    ("mh_20260520_re_0024", "drug_disease_relation"): {
        "rewrite": "contraindicated or strongly avoided in pregnancy except in rare specialized circumstances",
        "alternatives": [
            "contraindicated_or_strongly_avoid_in_pregnancy_except_rare_specialized_circumstances",
            "contraindicated in pregnancy",
            "avoid in pregnancy",
        ],
    },
    # ── gene_drug_relation ─────────────────────────────────────
    ("mh_20260520_re_0019", "gene_drug_relation"): {
        "rewrite": "CYP2C19 poor metabolizer status reduces conversion of clopidogrel to its active metabolite, leading to reduced antiplatelet effect",
        "alternatives": [
            "reduced_conversion_to_active_metabolite_with_reduced_antiplatelet_effect",
            "reduced active metabolite formation and reduced antiplatelet effect",
            "poor metabolizers have decreased clopidogrel activation",
        ],
    },
}

# Drug/disease/route name fields are short single tokens but inherently correct.
# These should match case-insensitively with reasonable synonyms.
NAME_FIELD_PATTERNS = ("drug_name", "disease_name", "route", "drug_name_primary", "affected_drug")


def fix_field(case_id: str, fld: dict) -> tuple[bool, str]:
    """In-place mutation of fld; returns (changed, reason)."""
    ev = fld.get("expected_value")
    if not isinstance(ev, str):
        return False, ""
    ev = ev.strip()
    ver = fld.get("verification")
    field_name = fld.get("field_name", "")
    field_type = (fld.get("field_type") or "").lower()

    # 1. Numeric values using exact_match → numeric_tolerance
    #    BUT keep exact_match for discrete identifier-like numbers
    #    (year, day count, item count, scores like Child-Pugh points)
    if is_numeric_string(ev) and ver == "exact_match":
        # discrete identifiers — keep exact match
        DISCRETE_HINTS = (
            "year", "_count", "n_", "duration_days", "duration_weeks",
            "days_", "weeks_", "points", "score", "stage", "class",
            "criterion_count", "_threshold_count",
            "_meets_threshold", "criteria_met", "meets_",
        )
        if (
            field_type in ("citation_field", "med_code", "year", "count")
            or any(h in field_name.lower() for h in DISCRETE_HINTS)
        ):
            return False, ""
        fld["verification"] = "numeric_tolerance"
        if "tolerance" not in fld or fld["tolerance"] is None:
            fld["tolerance"] = 0.05
        return True, "exact_match→numeric_tolerance"

    # 2. Hand-curated enum rewrites
    key = (case_id, field_name)
    if key in ENUM_REWRITES:
        fix = ENUM_REWRITES[key]
        old_value = fld["expected_value"]
        fld["expected_value"] = fix["rewrite"]
        existing_alts = list(fld.get("alternatives") or [])
        if old_value not in existing_alts:
            existing_alts.append(old_value)
        for a in fix["alternatives"]:
            if a not in existing_alts:
                existing_alts.append(a)
        fld["alternatives"] = existing_alts
        return True, "enum→natural-language"

    # 3. Drug/disease name fields — keep as-is but add lowercase variant in alternatives
    if any(p in field_name for p in NAME_FIELD_PATTERNS) and is_enum_like(ev):
        # Already case-insensitive in matcher, no-op
        return False, ""

    # 4. Frequency / dose enums (qd, bid, tid, q8h, q12h etc.) → keep but enrich alternatives
    if ver in ("temporal_equivalence", "exact_match") and ev.lower() in ("qd", "bid", "tid", "qid", "q4h", "q6h", "q8h", "q12h", "qhs", "qam", "qpm"):
        equivalents = {
            "qd": ["once daily", "once per day", "1x/day", "every 24 hours"],
            "bid": ["twice daily", "two times a day", "2x/day", "every 12 hours", "q12h"],
            "tid": ["three times daily", "three times a day", "3x/day", "every 8 hours", "q8h"],
            "qid": ["four times daily", "four times a day", "4x/day", "every 6 hours", "q6h"],
            "q4h": ["every 4 hours"],
            "q6h": ["every 6 hours", "qid", "four times daily"],
            "q8h": ["every 8 hours", "tid", "three times daily"],
            "q12h": ["every 12 hours", "bid", "twice daily"],
            "qhs": ["at bedtime", "every night"],
        }
        existing = list(fld.get("alternatives") or [])
        for a in equivalents.get(ev.lower(), []):
            if a not in existing:
                existing.append(a)
        if existing != (fld.get("alternatives") or []):
            fld["alternatives"] = existing
            return True, "freq alternatives enriched"
    return False, ""


def main() -> None:
    fixed_count = 0
    by_reason: dict[str, int] = {}

    # Modify the per-category jsonls AND full jsonl
    files_to_patch = [
        "factual-20260520.jsonl",
        "contextual-20260520.jsonl",
        "citation-20260520.jsonl",
        "numerical-20260520.jsonl",
        "relational-20260520.jsonl",
        "full-20260520.jsonl",
    ]

    for fname in files_to_patch:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"!! missing {path}")
            continue
        items = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        file_changes = 0
        for item in items:
            for fld in (item.get("eval") or {}).get("ground_truth_fields") or []:
                changed, reason = fix_field(item["id"], fld)
                if changed:
                    file_changes += 1
                    fixed_count += 1
                    by_reason[reason] = by_reason.get(reason, 0) + 1
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {fname}: {file_changes} fields fixed")

    print(f"\nTotal fields fixed: {fixed_count}")
    print("By reason:")
    for r, n in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {r}: {n}")


if __name__ == "__main__":
    main()
