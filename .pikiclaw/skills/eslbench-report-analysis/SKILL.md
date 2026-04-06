---
name: eslbench-report-analysis
description: Use when analyzing ESL-Bench evaluation reports - extracting scores by difficulty dimension, comparing methods, checking fail rates, token costs, duration per query. Triggers on keywords like eslbench report, benchmark results, group by difficulty.
---

# ESL-Bench Report Analysis

Extract and compare evaluation results from HolyEval benchmark report JSON files.

## Report Location

```
~/xiaotong/holyeval/benchmark/report/eslbench/
  {dataset}_{target_type}_{model}_{date}_{time}.json
```

Example: `sample50-20260331_llm_api_gpt-5.4_20260331_223735.json`

## Report JSON Structure

```python
report = {
    "benchmark_name": "eslbench",
    "dataset_name": "sample50-20260331",
    "runtime_target": {...},
    "cases": [...],           # Per-case results
    "pass_count": 0,          # Cases scoring >= threshold
    "fail_count": 3,          # Cases with eval errors (not low scores)
    "pass_rate": 0.0,         # pass_count / total (NOTE: often 0 due to strict threshold)
    "avg_score": 0.512,       # Mean score across all cases (0-1 scale)
    "total_duration_seconds": 8492
}
```

### Per-Case Structure

```python
case = {
    "id": "user5022_AT_demo_Q001",
    "title": "[Lookup] ...",
    "tags": ["difficulty:Trend", "answer_type:numeric_value"],
    "eval": {
        "result": "scored",    # or "error"
        "score": 0.85,         # 0.0 - 1.0
        "feedback": "..."
    },
    "cost": {
        "target": {
            "gpt-5.4": {
                "input_tokens": 8621,
                "output_tokens": 314,
                "total_tokens": 8935,
                "input_token_details": {"cache_read": 7168},
                "output_token_details": {"reasoning": 0}
            }
        }
    },
    "start": "...", "end": "..."
}
```

### Key Fields

| Field | Where | Notes |
|---|---|---|
| Difficulty dimension | `case.tags` → `"difficulty:Lookup"` | 5 values: Lookup, Trend, Comparison, Anomaly, Explanation |
| Score | `case.eval.score` | 0.0-1.0, multiply by 100 for percentage |
| Answer type | `case.tags` → `"answer_type:text"` | text, numeric_value, list, boolean |
| Fail | `report.fail_count` | Eval errors, not low scores |
| Token usage | `case.cost.target.{model}.input_tokens` | May be None/empty if not tracked |
| Duration | `report.total_duration_seconds` | Wall-clock for entire run |

## Resolving Input Files

Input can be any of:
- **Glob pattern**: `~/xiaotong/holyeval/benchmark/report/eslbench/sample50-20260331_*.json`
- **Directory**: scan all `*.json` inside
- **Explicit file list**: one or more JSON file paths
- **Zip URL**: download, extract, scan `*.json` inside
- **Zip file path**: extract, scan `*.json` inside

The analysis output MUST show the **full resolved file path** for each report processed.

Default report location (if no input specified): `~/xiaotong/holyeval/benchmark/report/eslbench/`

## Analysis Script Template

```python
import json, glob, os, re, zipfile, tempfile
from collections import defaultdict

DIMS = ["Lookup", "Trend", "Comparison", "Anomaly", "Explanation"]

def resolve_report_files(inputs):
    """Resolve various input forms to a sorted list of JSON file paths.
    inputs: list of paths, glob patterns, directories, or zip files."""
    files = []
    for inp in inputs:
        inp = os.path.expanduser(inp)
        if inp.endswith(".zip") and os.path.isfile(inp):
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(inp) as z:
                z.extractall(tmpdir)
            files.extend(sorted(glob.glob(f"{tmpdir}/**/*.json", recursive=True)))
        elif os.path.isdir(inp):
            files.extend(sorted(glob.glob(f"{inp}/**/*.json", recursive=True)))
        elif "*" in inp or "?" in inp:
            files.extend(sorted(glob.glob(inp)))
        elif os.path.isfile(inp):
            files.append(inp)
    # Filter out __MACOSX etc
    return [f for f in files if "/__MACOSX/" not in f and f.endswith(".json")]

def analyze_reports(inputs=None):
    if inputs is None:
        inputs = [os.path.expanduser("~/xiaotong/holyeval/benchmark/report/eslbench")]
    files = resolve_report_files(inputs)
    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            report = json.load(f)
        cases = report.get("cases", [])
        n = len(cases)
        if n == 0: continue
        fail_n = report.get("fail_count", 0)
        dur = report.get("total_duration_seconds", 0)

        by_diff = defaultdict(list)
        in_tok = out_tok = 0
        for c in cases:
            diff = next((t.split(":",1)[1] for t in c.get("tags",[]) if t.startswith("difficulty:")), None)
            score = c.get("eval",{}).get("score")
            if diff and score is not None:
                by_diff[diff].append(score)
            for model_usage in (c.get("cost",{}).get("target") or {}).values():
                if isinstance(model_usage, dict):
                    in_tok += model_usage.get("input_tokens",0) or 0
                    out_tok += model_usage.get("output_tokens",0) or 0

        all_scores = [s for scores in by_diff.values() for s in scores]
        rows.append({
            "fpath": fpath,  # full resolved path
            "label": fname,  # filename for display
            "n": n,
            "dims": {d: (sum(by_diff[d])/len(by_diff[d])*100 if by_diff[d] else None) for d in DIMS},
            "total": sum(all_scores)/len(all_scores)*100 if all_scores else 0,
            "fail": f"{fail_n}/{n}", "fail_n": fail_n,
            "dur": dur,
            "per_q": dur/n if n else 0,
            "in_tok": in_tok, "out_tok": out_tok,
        })
    return rows

def fmt_tok(n):
    """Format token count: 1234567 -> '1.2M', 51245 -> '51K', 0 -> '-'."""
    if n == 0: return "-"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n//1_000}K"
    return str(n)

def print_table(rows):
    """Print score overview table. Always list resolved file paths first."""
    # Show resolved file paths
    print("**Processed report files:**\n")
    for i, r in enumerate(rows, 1):
        print(f"{i}. `{r['fpath']}`")
    print()
    # Score table
    def f(v): return f"{v:5.1f}" if v is not None else "  N/A"
    print(f"| {'Report (file)':<52} | {'Look':>5} | {'Trend':>5} | {'Comp':>5} | {'Anom':>5} | {'Expl':>5} | {'Avg%':>5} | {'Fail':>5} | {'Dur':>6} | {'s/q':>4} | {'InTok':>6} | {'OutTok':>6} |")
    print(f"|{'-'*54}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*6}|{'-'*8}|{'-'*8}|")
    for r in rows:
        d = r["dims"]
        print(f"| {r['label']:<52} | {f(d['Lookup'])} | {f(d['Trend'])} | {f(d['Comparison'])} | {f(d['Anomaly'])} | {f(d['Explanation'])} | {r['total']:5.1f} | {r['fail']:>5} | {r['dur']:5.0f}s | {r['per_q']:3.0f}s | {fmt_tok(r['in_tok']):>6} | {fmt_tok(r['out_tok']):>6} |")
```

### After Score Overview: Write a Summary

After printing the score table, **always write a free-form summary** covering:

1. **Best overall performer** — which report/model has highest Avg%, noting fail rate trade-off
2. **Per-dimension winners** — which model leads each dimension (Look/Trend/Comp/Anom/Expl)
3. **Weakest dimension** — which dimension is universally low across all methods and why
4. **Efficiency** — token usage vs. score (cost-effectiveness), duration per query
5. **Reliability** — fail rates, runs that should be discarded (high fail = unreliable)
6. **Notable surprises** — unexpected results (e.g. structured API outperforming LLM, score jumps between runs)

Format as a `## Summary` section with bullet points. Be specific with numbers, don't just say "good" or "bad".

## Fail Reason Analysis

Cases with `eval.result == "fail"` are execution errors, not low scores. Classify by extracting error from `eval.feedback` traceback.

```python
from collections import Counter

def analyze_fails(report):
    """Classify fail reasons from a report. Returns Counter of reason -> count."""
    reasons = Counter()
    for c in report.get("cases", []):
        ev = c.get("eval", {})
        if ev.get("result") != "fail":
            continue
        fb = ev.get("feedback", "")
        if "TimeoutError" in fb or "CancelledError" in fb:
            reasons["Timeout/CancelledError"] += 1
        elif "orchestrator.py" in fb:
            reasons["Orchestrator error (model API failure)"] += 1
        elif "asyncio/tasks.py" in fb:
            reasons["Async task error"] += 1
        else:
            reasons[fb[:80] if fb else "unknown"] += 1
    return reasons
```

### Common Fail Categories

| Category | Cause | Fix |
|---|---|---|
| Timeout/CancelledError | Query took too long, asyncio cancelled | Increase timeout or reduce concurrency |
| Orchestrator model error | Target model API returned error (rate limit, bad response) | Check API key/quota, retry |
| Async task error | General asyncio failure | Check logs for root cause |

## Per-Agent Per-Dimension Error Analysis

Classify WHY each agent scores low on each dimension. Combines fail cases and low-score cases into one view.

### Feedback Classification

`eval.feedback` format: `[answer_type] score=X.XXX — detail`

| Category | Pattern in feedback | Meaning |
|---|---|---|
| `no_data_found` | "no_data", "no relevant" | Agent couldn't retrieve data |
| `numeric_mismatch` | "Number match 0/" | Computed number wrong |
| `partial_list_match` | "F1=" | Incomplete list returned |
| `rubric_low_score` | "LLM judge", "Rubric" | LLM judge scored low on rubric |
| `exact_match_fail` | "exact" + "0/" | Text/boolean exact match failed |
| `FAIL_execution_error` | `eval.result == "fail"` | Execution error (timeout/API) |

### Analysis Script

```python
import json, glob, os, re
from collections import Counter

REPORT_DIR = os.path.expanduser("~/xiaotong/holyeval/benchmark/report/eslbench")
DIMS = ["Lookup", "Trend", "Comparison", "Anomaly", "Explanation"]

def classify_feedback(fb):
    """Classify a single eval.feedback string into an error category."""
    m = re.match(r'\[(\w+)\]\s*score=([\d.]+)\s*[—-]\s*(.*)', fb)
    if not m:
        return fb[:50] or "unknown"
    _, _, detail = m.groups()
    dl = detail.lower()
    if "no_data" in dl or "no relevant" in dl:
        return "no_data_found"
    if "number match 0/" in dl:
        return "numeric_mismatch"
    if "f1=" in dl:
        return "partial_list_match"
    if "llm judge" in dl or "rubric" in dl:
        return "rubric_low_score"
    if "exact" in dl and "0/" in dl:
        return "exact_match_fail"
    return detail[:50]

def analyze_report_by_dim(report, threshold=0.8):
    """Per-dimension: avg score, low-score count, top error reasons, sample case IDs."""
    results = {}
    for dim in DIMS:
        reasons = Counter()
        reason_cases = defaultdict(list)  # reason -> [case_id, ...]
        scores = []
        for c in report.get("cases", []):
            diff = next((t.split(":",1)[1] for t in c.get("tags",[])
                         if t.startswith("difficulty:")), None)
            if diff != dim:
                continue
            ev = c.get("eval", {})
            cid = c.get("id", "?")
            if ev.get("result") == "fail":
                reasons["FAIL_execution_error"] += 1
                reason_cases["FAIL_execution_error"].append(cid)
                continue
            score = ev.get("score")
            if score is None:
                continue
            scores.append(score)
            if score < threshold:
                cat = classify_feedback(ev.get("feedback", ""))
                reasons[cat] += 1
                reason_cases[cat].append(cid)
        results[dim] = {
            "avg": sum(scores)/len(scores)*100 if scores else 0,
            "scored": len(scores),
            "low": sum(reasons.values()),
            "reasons": reasons,
            "reason_cases": reason_cases,
        }
    return results

def print_full_analysis(inputs=None, show_cases=3):
    """Print per-report-file per-dimension error breakdown with sample case IDs.
    inputs: same as analyze_reports(). show_cases: max case IDs per error category (0 to hide)."""
    if inputs is None:
        inputs = [os.path.expanduser("~/xiaotong/holyeval/benchmark/report/eslbench")]
    files = resolve_report_files(inputs)
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            report = json.load(f)
        fail_n = report.get("fail_count", 0)
        n = len(report.get("cases", []))
        print(f"\n{'='*70}")
        print(f"  {fpath}")
        print(f"  fails={fail_n}/{n}, avg={report.get('avg_score',0):.3f}")
        print(f"{'='*70}")
        results = analyze_report_by_dim(report)
        for dim in DIMS:
            r = results[dim]
            top3 = r["reasons"].most_common(3)
            top_str = ", ".join(f"{k}:{v}" for k,v in top3) if top3 else "all good"
            print(f"  {dim:<12} avg={r['avg']:5.1f}%  low={r['low']}/{r['scored']}  | {top_str}")
            if show_cases > 0:
                for reason, count in top3:
                    ids = r["reason_cases"][reason][:show_cases]
                    suffix = f" +{count-show_cases} more" if count > show_cases else ""
                    print(f"    {reason}: {', '.join(ids)}{suffix}")
```

### Cross-Agent Dimension Patterns (Observed)

| Dimension | LLM+retrieval | RAG (k=10/50) | Structured API |
|---|---|---|---|
| **Lookup** | `no_data_found`, `partial_list_match` | `rubric_low_score`, `no_data_found` | `numeric_mismatch` |
| **Trend** | `no_data_found`, `rubric_low_score` | `numeric_mismatch`, `rubric_low_score` | `numeric_mismatch` |
| **Comparison** | `partial_list_match` (dominant), `rubric_low_score` | `partial_list_match`, `rubric_low_score` | `rubric_low_score`, `partial_list_match` |
| **Anomaly** | Generally high; `partial_list_match` rare | `numeric_mismatch`, `partial_list_match` | `numeric_mismatch` |
| **Explanation** | `no_data_found`, `rubric_low_score` (hardest) | `numeric_mismatch`, `rubric_low_score` | `numeric_mismatch`, `rubric_low_score` |

Key insights:
- **LLM+retrieval** fails on Comparison (can't enumerate cross-event overlaps) and Explanation (missing causal chains)
- **RAG methods** suffer `numeric_mismatch` across all dimensions — retrieval noise degrades computation
- **Structured API at 0%** is connectivity issue, not capability — all errors are format/match failures on empty responses
- **Comparison** is universally the weakest dimension across all agent types
- **Anomaly** discriminates most: LLM+retrieval scores 80%+ while RAG drops to 1-25%

## Interpretation Notes

- **`pass_rate`**: Often 0.0 because it uses a strict binary threshold; use `avg_score` instead for meaningful comparison
- **`fail_count`**: Eval-level errors (API timeout, parse failure), NOT low-scoring answers. High fail = unreliable run
- **`theta_api_expert` at 0%**: Usually means API connectivity issue, not model failure
- **Token cost = 0**: Means token tracking not enabled for that target type (e.g. mem0_rag, theta_api)
- **Multiple runs of same model**: Take the latest or lowest-fail run; early runs may have config bugs
- **`s/q` (seconds per query)**: Dominated by retrieval + generation time; <30s usually means cached/fast path

## Output to File

When saving analysis to a markdown file, output to `docs/eslbench_report_analysis-YYYYMMDD.md` and **always** print the mdpreview link:

```
http://10.241.13.122:22086/preview.html?path={path-relative-to-/home/fat/}
```

Example: if saved to `/home/fat/caill/thetagendata/docs/eslbench_report_analysis-20260401.md`, output:

```
http://10.241.13.122:22086/preview.html?path=caill/thetagendata/docs/eslbench_report_analysis-20260401.md
```
