"""
Deterministic normalizers for MedHall field-level verification.

Each function returns (match: bool, reason: str). They handle canonicalization
that has no medical-judgment component (frequency canonicals, numeric+unit
splitting, code equivalence, structured-field key alignment).

These are wired into hallucination_eval_agent._match_fields as additional
verification types so the heavyweight LLM judge (semantic_match) is only
needed for genuine paraphrase cases.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# ============================================================
# 1. Frequency canonicalization
# ============================================================
# Cluster all forms (qd, once daily, every 24 hours, 1x/day, ...) into a
# canonical "FREQ_PER_DAY:N" representation. Two answers match if they map to
# the same canonical.

_FREQ_CANONICAL: dict[str, str] = {}


def _add_freq_aliases(canonical: str, aliases: Iterable[str]) -> None:
    for a in aliases:
        _FREQ_CANONICAL[a.lower()] = canonical


_add_freq_aliases("FREQ_PER_DAY:1", [
    "qd", "qday", "q.d.", "q_d", "q24h", "q.24h",
    "once daily", "once a day", "once per day", "1x/day", "1/day",
    "daily", "every day", "every 24 hours", "every 24h",
])
_add_freq_aliases("FREQ_PER_DAY:2", [
    "bid", "b.i.d.", "q12h", "q.12h",
    "twice daily", "twice a day", "two times a day", "two times daily",
    "2x/day", "2/day", "every 12 hours", "every 12h",
])
_add_freq_aliases("FREQ_PER_DAY:3", [
    "tid", "t.i.d.", "q8h", "q.8h",
    "three times daily", "three times a day",
    "3x/day", "3/day", "every 8 hours", "every 8h",
])
_add_freq_aliases("FREQ_PER_DAY:4", [
    "qid", "q.i.d.", "q6h", "q.6h",
    "four times daily", "four times a day",
    "4x/day", "4/day", "every 6 hours", "every 6h",
])
_add_freq_aliases("FREQ_PER_DAY:6", [
    "q4h", "every 4 hours", "every 4h", "six times daily", "six times a day",
])
_add_freq_aliases("FREQ_BEDTIME", [
    "qhs", "q.h.s.", "at bedtime", "every night", "nightly", "every night at bedtime",
])
_add_freq_aliases("FREQ_PRN", ["prn", "p.r.n.", "as needed", "when needed"])


def _canon_freq(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    s = text.strip().lower()
    if s in _FREQ_CANONICAL:
        return _FREQ_CANONICAL[s]
    # Strip surrounding punctuation
    s_clean = re.sub(r"[(),.\";:'\[\]]", " ", s).strip()
    s_clean = re.sub(r"\s+", " ", s_clean)
    if s_clean in _FREQ_CANONICAL:
        return _FREQ_CANONICAL[s_clean]
    # "every 8 hours" / "8 hours" pattern
    m = re.search(r"every\s+(\d+)\s*(?:hours?|hrs?|h)\b", s_clean)
    if m:
        h = int(m.group(1))
        if h > 0 and 24 % h == 0:
            return f"FREQ_PER_DAY:{24 // h}"
    # "N times daily / per day / a day / each day"
    m = re.search(r"(\d+)\s*(?:x|times)\s*(?:per|/|a|each)?\s*(?:daily|day|d)\b", s_clean)
    if m:
        return f"FREQ_PER_DAY:{m.group(1)}"
    # word-number form: "two times daily", "three times daily", etc
    word_to_num = {"once": 1, "twice": 2, "thrice": 3, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    m = re.search(r"\b(once|twice|thrice|two|three|four|five|six)\s*(?:times)?\s*(?:per|/|a|each)?\s*(?:daily|day|d)\b", s_clean)
    if m:
        n = word_to_num[m.group(1)]
        return f"FREQ_PER_DAY:{n}"
    return None


def frequency_match(expected: str, extracted: str, alternatives: list[str] | None = None) -> tuple[bool, str]:
    exp_canon = _canon_freq(expected)
    ext_canon = _canon_freq(extracted)
    if exp_canon and ext_canon and exp_canon == ext_canon:
        return True, ""
    # Fallback to alternatives matching
    for a in alternatives or []:
        if _canon_freq(a) == ext_canon and ext_canon is not None:
            return True, ""
    if exp_canon and ext_canon:
        return False, f"frequency_mismatch: expected={expected}({exp_canon}), got={extracted}({ext_canon})"
    return False, f"frequency_unparseable: expected={expected}, got={extracted}"


# ============================================================
# 2. Numeric+unit split match
# ============================================================
# For fields where expected is "500 mg" but extractor splits into value=500, unit=mg

_NUM_UNIT_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Zµ%/]+(?:[/·.]?[a-zA-Z0-9]+)*)?\s*$")


def _split_num_unit(s: str) -> tuple[float | None, str | None]:
    if not isinstance(s, str):
        return None, None
    m = _NUM_UNIT_RE.match(s.strip())
    if not m:
        return None, None
    val = float(m.group(1))
    unit = (m.group(2) or "").strip() or None
    return val, unit


_UNIT_EQUIV: dict[str, set[str]] = {
    "mg": {"mg", "milligrams", "milligram"},
    "mcg": {"mcg", "µg", "ug", "micrograms", "microgram"},
    "g": {"g", "grams", "gram"},
    "kg": {"kg", "kilograms", "kilogram"},
    "ml": {"ml", "milliliters", "milliliter", "cc"},
    "l": {"l", "liters", "liter"},
    "iu": {"iu", "international units"},
    "mg/dl": {"mg/dl", "mg/dL"},
    "mmol/l": {"mmol/l", "mmol/L"},
    "mg/day": {"mg/day", "mg/d"},
    "mmhg": {"mmhg"},
    "%": {"%"},
    "bpm": {"bpm", "beats/min", "beats per minute"},
    "ng/ml": {"ng/ml"},
    "iu/l": {"iu/l", "u/l", "units/l"},
}


def _unit_canonical(u: str | None) -> str | None:
    if not u:
        return None
    u = u.lower().strip()
    for canon, members in _UNIT_EQUIV.items():
        if u in members or u == canon:
            return canon
    return u


def numeric_with_unit_match(
    expected: str, extracted: str, expected_unit: str | None,
    tolerance: float = 0.05, alternatives: list[str] | None = None,
) -> tuple[bool, str]:
    """Match "500 mg" against expected "500" + unit "mg" or vice versa."""
    exp_val, exp_unit_inline = _split_num_unit(expected)
    ext_val, ext_unit_inline = _split_num_unit(extracted)
    if exp_val is None and ext_val is None:
        return False, f"non_numeric: expected={expected}, got={extracted}"

    # Prefer inline unit, else expected_unit field
    eu = _unit_canonical(exp_unit_inline) or _unit_canonical(expected_unit)
    xu = _unit_canonical(ext_unit_inline) or None  # if extractor split unit out, it's not in here

    if exp_val is None or ext_val is None:
        return False, f"could_not_parse_numeric: expected={expected}, got={extracted}"

    # Tolerance-based numeric match
    if exp_val == 0:
        num_match = abs(ext_val) < tolerance
    else:
        num_match = abs(ext_val - exp_val) / abs(exp_val) <= tolerance

    if not num_match:
        return False, f"numeric_deviation: expected={exp_val}, got={ext_val}"

    # Unit match: if extractor didn't pull unit out, accept (unit handled separately)
    if eu and xu and eu != xu:
        return False, f"unit_mismatch: expected={eu}, got={xu}"

    return True, ""


# ============================================================
# 3. Structured / dict match
# ============================================================
# expected like {"drug": "tetracycline", "dose": "500 mg", "frequency": "qid"}
# extracted may be a string dump or partial dict.

_KEY_ALIASES: dict[str, set[str]] = {
    "drug": {"drug", "drug_name", "medication", "medication_name", "name", "drug_name_primary"},
    "dose": {"dose", "dosage", "amount", "strength"},
    "unit": {"unit", "units"},
    "route": {"route", "by_mouth", "administration", "way_of_administration"},
    "frequency": {"frequency", "freq", "schedule", "interval", "dosing_frequency"},
    "duration": {"duration", "course", "length"},
}

_VALUE_NORMS: dict[str, dict[str, str]] = {
    "route": {
        "po": "oral", "by mouth": "oral", "p.o.": "oral", "orally": "oral",
        "iv": "intravenous", "i.v.": "intravenous",
        "im": "intramuscular", "i.m.": "intramuscular",
        "sc": "subcutaneous", "s.c.": "subcutaneous", "subq": "subcutaneous",
    },
}


def _normalize_dict(d: dict[str, Any]) -> dict[str, str]:
    """Map keys to canonical names + normalize string values."""
    norm = {}
    for k, v in d.items():
        canon_k = next((ck for ck, members in _KEY_ALIASES.items() if k.lower() in members), k.lower())
        sv = str(v).strip().lower() if v is not None else ""
        # Apply value normalizers
        if canon_k in _VALUE_NORMS:
            sv = _VALUE_NORMS[canon_k].get(sv, sv)
        # Inline unit consolidation: "500 mg" stays as is; we don't split here
        norm[canon_k] = sv
    return norm


_ROUTE_TOKENS = {
    "oral": "oral", "orally": "oral", "po": "oral", "by mouth": "oral",
    "iv": "intravenous", "intravenous": "intravenous", "intravenously": "intravenous",
    "im": "intramuscular", "intramuscular": "intramuscular",
    "sc": "subcutaneous", "subq": "subcutaneous", "subcutaneous": "subcutaneous",
    "sublingual": "sublingual", "sl": "sublingual",
}


def _parse_regimen(s: str) -> dict[str, str] | None:
    """Parse natural-language regimen strings like 'tetracycline 500 mg by mouth four times daily'."""
    if not isinstance(s, str):
        return None
    s_lower = s.strip().lower()

    # Dose (number + unit)
    m_dose = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|µg|ug)\b", s_lower)
    dose = None
    if m_dose:
        dose = f"{m_dose.group(1)} {m_dose.group(2)}"

    # Frequency: try canonicalize, prefer rightmost match (regimens put freq at end)
    freq_canon = _canon_freq(s_lower)
    if freq_canon is None:
        # Try sub-spans from the end (last 6 tokens, last 5, ...)
        tokens = s_lower.split()
        for i in range(max(0, len(tokens) - 6), len(tokens)):
            sub = " ".join(tokens[i:])
            cand = _canon_freq(sub)
            if cand:
                freq_canon = cand
                break

    # Route
    route = None
    for token, canon in _ROUTE_TOKENS.items():
        if re.search(rf"\b{re.escape(token)}\b", s_lower):
            route = canon
            break

    # Drug: drop dose/freq/route, leftover word(s) at start
    drug = None
    if m_dose:
        before_dose = s_lower[: m_dose.start()].strip()
        # remove trailing punctuation
        before_dose = re.sub(r"[,;:.\-\s]+$", "", before_dose)
        if before_dose:
            drug = before_dose

    out = {}
    if drug:
        out["drug"] = drug
    if dose:
        out["dose"] = dose
    if route:
        out["route"] = route
    if freq_canon:
        out["frequency_canonical"] = freq_canon

    return out if out else None


def _parse_to_dict(s: str) -> dict[str, str] | None:
    """Best-effort parse of "drug: tetracycline, dose: 500 mg, freq: qid" or JSON."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return _normalize_dict(d)
        except Exception:
            pass
    # "key: value, key: value" / "key=value; key=value"
    pairs = re.findall(r"([A-Za-z_]+)\s*[:=]\s*([^,;]+)", s)
    if pairs:
        return _normalize_dict({k: v.strip() for k, v in pairs})
    # Natural-language regimen
    return _parse_regimen(s)


def structured_match(expected: str, extracted: str) -> tuple[bool, str]:
    """For when expected is a structured dict-like string."""
    exp_d = _parse_to_dict(expected)
    ext_d = _parse_to_dict(extracted)
    if exp_d is None:
        return False, f"expected_not_structured: {expected}"
    if ext_d is None:
        # Fall back to substring check: ALL expected values should appear in extracted text
        ext_lower = extracted.lower()
        missing = [k for k, v in exp_d.items() if v and v not in ext_lower]
        if not missing:
            return True, ""
        return False, f"missing_fields_in_extracted: {missing}"
    # Allow drug/route to be missing in extracted (often inferred from question / oral by default)
    SOFT_KEYS = {"drug", "route"}
    for k, v in exp_d.items():
        if not v:
            continue
        ext_v = ext_d.get(k, "")
        if not ext_v and k in SOFT_KEYS:
            continue  # drug name often inferred from question
        if not ext_v:
            return False, f"missing_field[{k}]: expected={v}"
        # Use numeric+unit logic for dose/duration
        if k in ("dose", "duration"):
            ok, _ = numeric_with_unit_match(v, ext_v, expected_unit=None)
            if not ok:
                return False, f"field[{k}]: expected={v}, got={ext_v}"
        elif k in ("frequency", "frequency_canonical"):
            # frequency_canonical is the canonical form; just compare directly
            if k == "frequency_canonical":
                ext_canon = ext_d.get("frequency_canonical") or _canon_freq(ext_v)
                if ext_canon != v:
                    return False, f"field[{k}]: expected={v}, got={ext_canon}"
            else:
                ok, _ = frequency_match(v, ext_v)
                if not ok:
                    return False, f"field[{k}]: expected={v}, got={ext_v}"
        elif k == "route":
            # Already normalized by _parse_regimen / _normalize_dict
            ext_norm = _VALUE_NORMS.get("route", {}).get(ext_v.lower(), ext_v.lower())
            if v.lower() != ext_norm:
                return False, f"field[{k}]: expected={v}, got={ext_v}"
        else:
            if v not in ext_v and ext_v not in v:
                return False, f"field[{k}]: expected={v}, got={ext_v}"
    return True, ""


# ============================================================
# 4. String-set / token-bag match
# ============================================================
# For short answers where token order doesn't matter:
#   "carbidopa-levodopa immediate-release" ≡ "immediate-release carbidopa-levodopa"

_STOPWORDS = {"a", "an", "the", "of", "and", "or", "with", "by"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[\s\-/,;()]+", s.lower()) if t and t not in _STOPWORDS}


def string_set_match(expected: str, extracted: str, alternatives: list[str] | None = None) -> tuple[bool, str]:
    exp_tokens = _tokens(expected)
    ext_tokens = _tokens(extracted)
    if not exp_tokens:
        return False, "expected_empty"
    if exp_tokens.issubset(ext_tokens) or ext_tokens.issubset(exp_tokens):
        return True, ""
    for a in alternatives or []:
        a_tokens = _tokens(a)
        if a_tokens and (a_tokens.issubset(ext_tokens) or ext_tokens.issubset(a_tokens)):
            return True, ""
    return False, f"token_set_mismatch: expected={expected}, got={extracted}"


# ============================================================
# 5. Code equivalence
# ============================================================
# Same medical concept, different code system or different specificity:
#   - LOINC 6556-5 ≡ 626-2 (both rapid strep antigen)
#   - ATC A10BK01 (specific drug) ⊂ A10BK (drug class)
#   - SNOMED 230572002 ≡ 422088007 (diabetic polyneuropathy variants)
#
# Maintained as a hand-curated equivalence table seeded from the cases that
# triggered code_mismatch. Extend over time.

_CODE_EQUIV_TABLE: dict[str, set[str]] = {
    # LOINC: rapid strep antigen — both valid
    "6556-5": {"6556-5", "626-2", "78012-2"},
    "626-2": {"6556-5", "626-2", "78012-2"},
    # SNOMED CT: diabetic polyneuropathy concepts
    "230572002": {"230572002", "422088007"},
    "422088007": {"230572002", "422088007"},
    # SNOMED CT: diabetic kidney disease
    "127013003": {"127013003", "709147005"},
    "709147005": {"127013003", "709147005"},
    "420279001": {"420279001", "709044004"},
    "709044004": {"420279001", "709044004"},
}


def _atc_parent_match(expected: str, extracted: str) -> bool:
    """ATC: A10BK01 (full) ↔ A10BK (4-char class) — accept parent match."""
    e = expected.upper().strip()
    x = extracted.upper().strip()
    if not (re.fullmatch(r"[A-Z]\d{2}[A-Z]{1,2}\d{0,2}", e) and re.fullmatch(r"[A-Z]\d{2}[A-Z]{1,2}\d{0,2}", x)):
        return False
    # Parent-child: longer code starts with shorter code
    return e.startswith(x) or x.startswith(e)


def code_equivalence(
    expected: str, extracted: str, code_system: str | None = None,
    alternatives: list[str] | None = None,
) -> tuple[bool, str]:
    e = expected.upper().strip()
    x = extracted.upper().strip()

    if e == x:
        return True, ""

    # Alternatives
    for a in alternatives or []:
        if x == a.upper().strip():
            return True, ""

    # Equivalence table
    eq_set = _CODE_EQUIV_TABLE.get(expected.strip())
    if eq_set and extracted.strip() in eq_set:
        return True, ""
    # Try lowercase keys too
    eq_set = _CODE_EQUIV_TABLE.get(e)
    if eq_set and x in {s.upper() for s in eq_set}:
        return True, ""

    # ATC parent/child
    if (code_system or "").upper() == "ATC" and _atc_parent_match(expected, extracted):
        return True, ""

    return False, f"code_mismatch: expected={expected}({code_system}), got={extracted}"
