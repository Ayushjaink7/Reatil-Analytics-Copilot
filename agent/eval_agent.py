"""
eval_agent.py - measure the SQL agent's execution accuracy on a fixed question set.

For each question in eval_questions.json, the agent writes its own SQL. We run that and the
hand-written gold SQL, then compare the RESULTS (not the SQL text, since many queries are
correct). A question passes when:
  - both return the same number of rows, and
  - every gold column has a matching column in the agent's result (same values after
    rounding to 2 decimals; row order ignored).
Extra columns in the agent's answer are allowed; column names don't matter.

    python agent/eval_agent.py                 # uses GROQ_API_KEY and the default model
    GROQ_MODEL=qwen/qwen3.8-27b python agent/eval_agent.py

Writes agent/logs/eval_results.json with the per-question details.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_agent import DEFAULT_DB, GroqLLM, SQLAgent, run_readonly_query  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "eval_questions.json"
RESULTS = HERE / "logs" / "eval_results.json"


def _normalise(col: pd.Series) -> list:
    """Round numbers and stringify, so 12.0 == 12 and 0.1234 == 0.12."""
    out = []
    for v in col.tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append(None)
        elif isinstance(v, (int, float)):
            out.append(round(float(v), 2))
        else:
            out.append(str(v))
    return out


def results_match(pred: pd.DataFrame, gold: pd.DataFrame, ordered: bool) -> bool:
    if pred is None or len(pred) != len(gold):
        return False
    pred_cols = [_normalise(pred[c]) for c in pred.columns]
    for g in gold.columns:
        g_vals = _normalise(gold[g])
        key = (lambda xs: xs) if ordered else (lambda xs: sorted(xs, key=lambda x: (x is None, str(x))))
        if not any(key(p) == key(g_vals) for p in pred_cols):
            return False
    return True


def evaluate(agent: SQLAgent, questions: list[dict]) -> dict:
    rows = []
    for item in questions:
        gold, _ = run_readonly_query(agent.con, item["gold_sql"], max_rows=10_000)
        result = agent.ask(item["question"])
        # compare as sets of values: a correct answer sorted differently still passes
        passed = result.ok and results_match(result.data, gold, ordered=False)
        rows.append({"id": item["id"], "question": item["question"], "passed": passed,
                     "attempts": result.attempts, "latency_s": result.latency_s,
                     "agent_sql": result.sql, "error": result.error})
        print(f"{'PASS' if passed else 'FAIL'}  {item['id']}  ({result.attempts} attempt(s), "
              f"{result.latency_s}s)  {item['question']}")
    n_pass = sum(r["passed"] for r in rows)
    summary = {
        "model": getattr(agent.llm, "model", "custom"),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_accuracy": round(n_pass / len(rows), 3),
        "passed": n_pass, "total": len(rows),
        "first_try_success": sum(r["attempts"] == 1 and r["passed"] for r in rows),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "details": rows,
    }
    print(f"\nExecution accuracy: {n_pass}/{len(rows)} = {summary['execution_accuracy']:.0%}  "
          f"| passed on first try: {summary['first_try_success']}  "
          f"| avg latency: {summary['avg_latency_s']}s")
    return summary


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE.parent / ".env")
    except ImportError:
        pass
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    agent = SQLAgent(DEFAULT_DB, llm=GroqLLM(), explain=False)
    summary = evaluate(agent, questions)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved details to {RESULTS}")
