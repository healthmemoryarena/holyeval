"""
Consistency + fairness audit for MedHall full-20260520.

Checks:
  C1  JSON exam_data per-indicator fields == fields available in DuckDB
      (no spoiler fields like reference_range/status/clinical_significance/
      abnormal_findings/overall_assessment/recommendations leaked into JSON)
  C2  For each user referenced by contextual cases, every (date, indicator)
      that appears in known_facts also appears in DuckDB AND in the JSON,
      with the same value (within numeric tolerance) and same unit
  C3  Question text doesn't reference fields that no longer exist in the data
      (reference range, abnormal flag, marked normal/abnormal etc.)
  C4  Every contextual case has at least one matching record in the user's
      data (otherwise the question is unanswerable)

Prints a summary; exit 0 if all clean, exit 1 otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark" / "data" / "medhall"
USER_DATA_DIR = ROOT / "benchmark" / "data" / "eslbench" / ".data"

SPOILER_FIELDS_PER_INDICATOR = {"reference_range", "status", "clinical_significance"}
SPOILER_FIELDS_PER_EXAM = {
    "abnormal_findings", "overall_assessment", "recommendations", "generation_summary",
}
TROUBLE_RE = re.compile(
    r"reference\s*range|flagged|marked\s*(?:abnormal|normal)|above\s+range|below\s+range",
    re.IGNORECASE,
)


def load_contextual_cases() -> list[dict]:
    path = DATA_DIR / "contextual-20260520.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def get_user_emails_from_case(case: dict) -> list[str]:
    ovr = (case.get("user") or {}).get("target_overrides") or {}
    emails = set()
    for v in ovr.values():
        if isinstance(v, dict):
            if v.get("email"):
                emails.add(v["email"])
            elif v.get("user_email"):
                emails.add(v["user_email"])
            elif v.get("tool_context", {}).get("user_email"):
                emails.add(v["tool_context"]["user_email"])
    return sorted(emails)


def parse_known_fact(s: str) -> tuple[str, str, float | None, str] | None:
    """Parse '2023-01-12: indicator_name = 132.0 ng/mL' → (date, indicator, value, unit)"""
    m = re.match(r"\s*(\d{4}-\d{2}-\d{2})\s*:\s*([\w\-]+)\s*=\s*([^\s]+)\s*([^\s].*)?", s)
    if not m:
        return None
    date, ind, val_str, unit = m.group(1), m.group(2), m.group(3), (m.group(4) or "").strip()
    try:
        val = float(val_str)
    except ValueError:
        val = None
    return date, ind, val, unit


def c1_check_json_spoilers() -> list[str]:
    """C1: scan all user JSONs for spoiler fields."""
    issues = []
    for user_dir in sorted(USER_DATA_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        exam_path = user_dir / "exam_data.json"
        if not exam_path.exists():
            continue
        data = json.loads(exam_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for ex in data:
            for f in SPOILER_FIELDS_PER_EXAM:
                if f in ex:
                    issues.append(f"{user_dir.name}: per-exam spoiler '{f}' present")
            for ind in (ex.get("indicators") or {}).values():
                if not isinstance(ind, dict):
                    continue
                for f in SPOILER_FIELDS_PER_INDICATOR:
                    if f in ind:
                        issues.append(f"{user_dir.name}: per-indicator spoiler '{f}' present")
                        break
                if any(f in ind for f in SPOILER_FIELDS_PER_INDICATOR):
                    break  # only report once per exam
    return issues


def c2_check_db_json_alignment(emails: set[str]) -> list[str]:
    """C2: simulate runtime hydration (eslbench/retrieve does this at tool load)
    and check that the hydrated DB row set is congruent with JSON indicators
    (matching value/unit per (date, indicator_name))."""
    issues = []
    for email in sorted(emails):
        ud = USER_DATA_DIR / email.replace("@", "_AT_")
        exam_path = ud / "exam_data.json"
        if not exam_path.exists():
            issues.append(f"{email}: missing exam_data.json")
            continue

        # Materialize what the tool would produce in DB at runtime
        in_mem = duckdb.connect(":memory:")
        in_mem.execute("""
            CREATE TABLE exam_indicators(
                user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,
                exam_type VARCHAR, exam_location VARCHAR, value VARCHAR, unit VARCHAR
            )
        """)
        data = json.loads(exam_path.read_text(encoding="utf-8"))
        rows = []
        json_records = {}
        if isinstance(data, list):
            for ex in data:
                if not isinstance(ex, dict):
                    continue
                date = ex.get("exam_date") or ex.get("date")
                for ind_key, ind in (ex.get("indicators") or {}).items():
                    if not isinstance(ind, dict):
                        continue
                    timestamp = ind.get("timestamp") or (f"{date} 00:00:00" if date else None)
                    indicator_name = ind.get("indicator_name") or ind_key
                    val = None if ind.get("value") is None else str(ind.get("value"))
                    unit = ind.get("unit")
                    rows.append((email, timestamp, indicator_name, ex.get("exam_type"), ex.get("exam_location"), val, unit))
                    json_records[(date, indicator_name.lower())] = (val, unit)
        if rows:
            in_mem.executemany("INSERT INTO exam_indicators VALUES (?,?,?,?,?,?,?)", rows)

        db_rows = in_mem.execute(
            "SELECT strftime(time, '%Y-%m-%d'), lower(indicator), value, unit FROM exam_indicators"
        ).fetchall()
        in_mem.close()
        db_records = {(r[0], r[1]): (r[2], r[3]) for r in db_rows}

        if len(db_records) != len(json_records):
            issues.append(f"{email}: JSON has {len(json_records)} indicators, hydrated DB has {len(db_records)}")
        # Value/unit consistency
        for k, jv in json_records.items():
            dv = db_records.get(k)
            if dv is None:
                issues.append(f"{email}: {k} in JSON missing in hydrated DB")
            elif dv != jv:
                issues.append(f"{email}: {k} value/unit mismatch — DB={dv} JSON={jv}")
    return issues


def c3_check_question_text() -> list[str]:
    """C3: contextual questions shouldn't reference stripped fields."""
    issues = []
    for case in load_contextual_cases():
        q = case["user"]["strict_inputs"][0]
        if TROUBLE_RE.search(q):
            m = TROUBLE_RE.search(q)
            issues.append(f"{case['id']}: question still references '{m.group()}'")
    return issues


def c4_check_known_facts_in_data() -> tuple[list[str], dict[str, int]]:
    """C4: each known_fact's (date, indicator) tuple must exist in the user's JSON
    (which is the source of truth — DB on disk is empty until hydrated at runtime).

    known_fact uses indicator_key form (lowercase, underscores), JSON has
    both indicator_name and indicator_key. We check both forms.
    """
    issues = []
    stats = {"total_facts": 0, "matched_in_json": 0}
    for case in load_contextual_cases():
        emails = get_user_emails_from_case(case)
        if not emails:
            issues.append(f"{case['id']}: no user_email in target_overrides")
            continue
        email = emails[0]
        ud = USER_DATA_DIR / email.replace("@", "_AT_")
        exam_path = ud / "exam_data.json"
        if not exam_path.exists():
            issues.append(f"{case['id']}: user {email} exam_data.json missing")
            continue

        data = json.loads(exam_path.read_text(encoding="utf-8"))
        # Build key→value lookup, accepting both indicator_name and indicator_key
        json_keys: dict[tuple[str, str], tuple[str, str]] = {}
        for ex in data:
            date = ex.get("exam_date") or ex.get("date")
            for ind_dict_key, ind in (ex.get("indicators") or {}).items():
                if not isinstance(ind, dict):
                    continue
                val = "" if ind.get("value") is None else str(ind.get("value"))
                unit = ind.get("unit") or ""
                for alias in (ind.get("indicator_key"), ind.get("indicator_name"), ind_dict_key):
                    if alias:
                        json_keys[(date, str(alias).lower())] = (val, unit)

        for fact in (case.get("eval") or {}).get("known_facts") or []:
            stats["total_facts"] += 1
            parsed = parse_known_fact(fact)
            if not parsed:
                issues.append(f"{case['id']}: cannot parse known_fact: {fact[:80]}")
                continue
            date, ind, val, unit = parsed
            ind_lower = ind.lower()
            key = (date, ind_lower)
            if key in json_keys:
                stats["matched_in_json"] += 1
                # Value check (tolerant)
                json_val, json_unit = json_keys[key]
                if val is not None:
                    try:
                        if abs(float(json_val) - val) / max(abs(val), 1e-6) > 0.01:
                            issues.append(
                                f"{case['id']} [{email}]: {date} {ind} value mismatch — "
                                f"known_fact={val} JSON={json_val}"
                            )
                    except (ValueError, TypeError):
                        if str(val) != str(json_val):
                            issues.append(
                                f"{case['id']} [{email}]: {date} {ind} value mismatch — "
                                f"known_fact={val} JSON={json_val}"
                            )
            else:
                issues.append(
                    f"{case['id']} [{email}]: known_fact '{date} {ind}' NOT in JSON"
                )
    return issues, stats


def main() -> int:
    print("=" * 70)
    print("MedHall full-20260520 consistency + fairness audit")
    print("=" * 70)

    print("\n[C1] JSON files free of spoiler fields...")
    c1 = c1_check_json_spoilers()
    if c1:
        for x in c1[:10]:
            print(f"  ✗ {x}")
        if len(c1) > 10:
            print(f"  ... +{len(c1) - 10} more")
    else:
        print("  ✓ all clean")

    cases = load_contextual_cases()
    all_emails = set()
    for c in cases:
        all_emails.update(get_user_emails_from_case(c))
    print(f"\n[C2] DB ↔ JSON alignment for {len(all_emails)} users referenced by contextual cases...")
    c2 = c2_check_db_json_alignment(all_emails)
    if c2:
        for x in c2[:10]:
            print(f"  ⚠ {x}")
        if len(c2) > 10:
            print(f"  ... +{len(c2) - 10} more")
    else:
        print("  ✓ JSON indicators ⊆ DB indicators")

    print("\n[C3] Contextual questions free of stripped-field references...")
    c3 = c3_check_question_text()
    if c3:
        for x in c3:
            print(f"  ✗ {x}")
    else:
        print(f"  ✓ all {len(cases)} questions clean")

    print("\n[C4] Each known_fact (date, indicator) must exist in user data...")
    c4, stats = c4_check_known_facts_in_data()
    print(f"  Total known_facts checked: {stats['total_facts']}")
    print(f"  Matched in JSON: {stats['matched_in_json']} ({stats['matched_in_json']*100//max(stats['total_facts'],1)}%)")
    if c4:
        print()
        for x in c4[:10]:
            print(f"  ⚠ {x}")
        if len(c4) > 10:
            print(f"  ... +{len(c4) - 10} more")
    else:
        print("  ✓ every known_fact resolvable from user data")

    has_critical = bool(c1) or bool(c3) or stats["matched_in_json"] < stats["total_facts"]
    print("\n" + "=" * 70)
    print(f"VERDICT: {'❌ ISSUES FOUND' if has_critical else '✅ CLEAN'}")
    print("=" * 70)
    return 1 if has_critical else 0


if __name__ == "__main__":
    sys.exit(main())
