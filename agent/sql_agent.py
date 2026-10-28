"""
sql_agent.py - Retail Analytics Copilot: business question -> read-only SQL -> answer.

Pipeline for one question
  1. Build a prompt with the database schema, business definitions and a few examples.
  2. Ask an LLM on Groq to write ONE SQLite SELECT query.
  3. Check the query with a static validator (one statement, SELECT/WITH only,
     no write or admin keywords).
  4. Run it on a read-only connection that also has an authorizer and a time limit.
  5. If validation or execution fails, send the error back to the LLM and retry.
  6. Ask the LLM to explain the result in plain English, using only the returned rows.

Safety is layered, so one layer failing doesn't open the database:
  layer 1  prompt rules           (the model is told to write SELECT only)
  layer 2  validate_sql()         (rejects anything that isn't a single SELECT/WITH)
  layer 3  read-only connection   (mode=ro URI + PRAGMA query_only)
  layer 4  sqlite authorizer      (denies every action except reading tables and functions)
  plus     a 10-second timeout and a 200-row cap on what comes back

Usage
  export GROQ_API_KEY=gsk_...                 # or put it in a .env file
  python agent/sql_agent.py "Which 5 states have the highest late-delivery rate?"
  python agent/sql_agent.py                   # interactive mode
  python agent/sql_agent.py --list-models     # see which Groq models your key can use
  GROQ_MODEL=qwen/qwen3.8-27b python agent/sql_agent.py "..."   # try another model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "olist.db"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "queries.jsonl"

# Groq retired llama-3.3-70b-versatile in Aug 2026; gpt-oss-120b is its listed replacement.
# Model names change, so it can be overridden with the GROQ_MODEL environment variable.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_ROWS = 200          # rows returned to the user / LLM
QUERY_TIMEOUT_S = 10    # abort queries that run longer than this
MAX_ATTEMPTS = 3        # 1 try + 2 repairs

# ---------------------------------------------------------------------------
# Business context: things the schema alone doesn't tell the model
# ---------------------------------------------------------------------------
BUSINESS_NOTES = """
Business definitions (always follow these):
- Revenue = SUM(order_items.price) over orders whose order_status NOT IN ('canceled', 'unavailable').
  GMV = SUM(order_items.price + order_items.freight_value) with the same filter.
- A customer (a person) is customers.customer_unique_id. customers.customer_id is a new id for every
  order, so never count customer_id when the question is about customers or repeat buyers.
- Late delivery: a delivered order is late when date(order_delivered_customer_date) >
  date(order_estimated_delivery_date). The view v_order_delivery already has is_late (0/1),
  delivery_days and promised_days for every delivered order - prefer it for delivery questions.
- Product categories are Portuguese in products.product_category_name. Join category_translation
  to show product_category_name_english, using COALESCE with the Portuguese name if missing.
- Reviews: review_score is 1-5. An order can have more than one review row.
- Timestamps are TEXT in 'YYYY-MM-DD HH:MM:SS'. Use strftime('%Y-%m', col) for months,
  date(col) for days and julianday(a) - julianday(b) for durations in days.
- Data covers Sep 2016 - Oct 2018. Complete months are Jan 2017 - Aug 2018.
- States are 2-letter Brazilian codes (SP = Sao Paulo, RJ = Rio de Janeiro, MG = Minas Gerais).
- Money is in Brazilian reais (BRL).
""".strip()

RULES = """
Rules:
- Write exactly ONE SQLite query that starts with SELECT or WITH. Never modify data.
- Use only the tables, views and columns listed in the schema.
- Give every computed column a short snake_case alias.
- Round money and rates with ROUND(x, 2). Express rates as percentages when the question says "rate" or "share".
- Add ORDER BY for rankings and LIMIT for "top N" questions. Return at most 200 rows.
- If the question cannot be answered from this data, return: SELECT 'not answerable from this data' AS note
- Reply with the SQL inside one ```sql code block and nothing else.
""".strip()

# A few worked examples teach the model the house style and the definitions above.
FEW_SHOT = [
    (
        "What is the late-delivery rate by month in 2018?",
        """SELECT strftime('%Y-%m', order_purchase_timestamp) AS month,
       COUNT(*) AS delivered_orders,
       ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct
FROM v_order_delivery
WHERE order_purchase_timestamp >= '2018-01-01' AND order_purchase_timestamp < '2019-01-01'
GROUP BY month
ORDER BY month;""",
    ),
    (
        "Which 3 product categories earned the most revenue?",
        """SELECT COALESCE(t.product_category_name_english, p.product_category_name) AS category,
       ROUND(SUM(oi.price), 2) AS revenue
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
LEFT JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY category
ORDER BY revenue DESC
LIMIT 3;""",
    ),
]

# ---------------------------------------------------------------------------
# Layer 2: static SQL validation
# ---------------------------------------------------------------------------
FORBIDDEN = [
    r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b", r"\bcreate\b",
    r"\breplace\s+into\b", r"\battach\b", r"\bdetach\b", r"\bpragma\b", r"\bvacuum\b",
    r"\breindex\b", r"\btruncate\b", r"\bload_extension\b", r"\bwritefile\b",
]


def _strip_comments_and_strings(sql: str) -> str:
    """Remove comments and string literals so keyword checks don't trip on data values."""
    sql = re.sub(r"--[^\n]*", " ", sql)                 # -- line comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)    # /* block comments */
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)          # 'string literals'
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)          # "quoted identifiers"
    return sql


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, reason). Accepts a single read-only SELECT/WITH statement only."""
    if not sql or not sql.strip():
        return False, "empty query"
    body = _strip_comments_and_strings(sql).strip().rstrip(";").strip()
    if ";" in body:
        return False, "only one statement is allowed"
    if not re.match(r"^(select|with)\b", body, flags=re.I):
        return False, "query must start with SELECT or WITH"
    for pattern in FORBIDDEN:
        if re.search(pattern, body, flags=re.I):
            keyword = pattern.replace(r"\b", "").replace(r"\s+", " ")
            return False, f"forbidden keyword: {keyword.upper()}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Layers 3 and 4: read-only connection, authorizer, timeout
# ---------------------------------------------------------------------------
_ALLOWED_ACTIONS = {
    getattr(sqlite3, name) for name in ("SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION",
                                        "SQLITE_RECURSIVE") if hasattr(sqlite3, name)
}


def _authorizer(action, arg1, arg2, db_name, trigger):
    """SQLite calls this for every operation while compiling a query. Allow reads only."""
    if action in _ALLOWED_ACTIONS:
        # block the few functions that could touch the filesystem
        if action == getattr(sqlite3, "SQLITE_FUNCTION", -1) and (arg2 or "").lower() in {
            "load_extension", "readfile", "writefile", "fts3_tokenizer"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def connect_readonly(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the database so that it physically cannot be written through this connection."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Build it first:  python build_db.py --data-dir data --db {db_path}")
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    con.set_authorizer(_authorizer)
    return con


def run_readonly_query(con: sqlite3.Connection, sql: str, max_rows: int = MAX_ROWS,
                       timeout_s: float = QUERY_TIMEOUT_S) -> tuple[pd.DataFrame, bool]:
    """Validate and run a query. Returns (rows, truncated). Raises ValueError / sqlite3.Error."""
    ok, reason = validate_sql(sql)
    if not ok:
        raise ValueError(f"Rejected by validator: {reason}")

    deadline = time.monotonic() + timeout_s
    # the progress handler runs every 10k VM steps; returning 1 aborts the query
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise sqlite3.OperationalError(f"query took longer than {timeout_s}s and was stopped") from e
        raise
    finally:
        con.set_progress_handler(None, 0)
    truncated = len(rows) > max_rows
    return pd.DataFrame(rows[:max_rows], columns=cols), truncated


# ---------------------------------------------------------------------------
# Schema description for the prompt
# ---------------------------------------------------------------------------
TABLE_NOTES = {
    "orders": "one row per order; status and the 5 timestamps of the order lifecycle",
    "order_items": "one row per item in an order (order_item_id = 1, 2, ...); price and freight per item",
    "customers": "one row per order-customer; customer_unique_id identifies the person",
    "payments": "one row per payment method used on an order",
    "reviews": "customer review (score 1-5, optional comment) per order",
    "products": "product attributes; category name in Portuguese",
    "sellers": "seller location",
    "category_translation": "Portuguese -> English category names",
    "geolocation": "zip-code prefix -> latitude/longitude (many rows per prefix)",
    "v_order_delivery": "VIEW: delivered orders with delivery_days, promised_days and is_late (0/1)",
}

# low-cardinality columns whose real values help the model write correct filters
VALUE_HINT_COLUMNS = {
    "orders": ["order_status"],
    "payments": ["payment_type"],
    "customers": ["customer_state"],
}


def describe_schema(con: sqlite3.Connection) -> str:
    """Build a compact schema description: tables, columns, types and example values."""
    con.set_authorizer(None)  # schema introspection needs PRAGMA; re-enabled below
    try:
        objects = con.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
        lines = []
        for name, kind in objects:
            cols = con.execute(f'PRAGMA table_info("{name}")').fetchall()
            col_text = ", ".join(f"{c[1]} {c[2] or ''}".strip() for c in cols)
            note = TABLE_NOTES.get(name, "")
            lines.append(f"{kind.upper()} {name}  -- {note}\n  columns: {col_text}")
            for col in VALUE_HINT_COLUMNS.get(name, []):
                values = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{col}" FROM "{name}" ORDER BY 1 LIMIT 30')]
                lines.append(f"  values of {col}: {', '.join(map(str, values))}")
        return "\n".join(lines)
    finally:
        con.set_authorizer(_authorizer)


# ---------------------------------------------------------------------------
# LLM client (Groq). Anything with .complete(messages) -> str works, which keeps
# the agent testable offline with a fake client.
# ---------------------------------------------------------------------------
class GroqLLM:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("Install the Groq SDK first:  pip install groq") from e
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY (environment variable or .env file). "
                               "Get a key at https://console.groq.com/keys")
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0)
        return resp.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    question: str
    sql: str | None = None
    data: pd.DataFrame | None = None
    truncated: bool = False
    answer: str | None = None
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list)  # every attempt: sql + error

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None


def extract_sql(text: str) -> str:
    """Pull the SQL out of the model reply (```sql block, any ``` block, or bare SELECT/WITH)."""
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.S | re.I) or \
        re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(with|select)\b.*", text, flags=re.S | re.I)
    return m.group(0).strip() if m else text.strip()


class SQLAgent:
    def __init__(self, db_path: str | Path = DEFAULT_DB, llm=None, explain: bool = True,
                 log: bool = True):
        self.con = connect_readonly(db_path)
        self.schema = describe_schema(self.con)
        self.llm = llm if llm is not None else GroqLLM()
        self.explain = explain
        self.log = log

    def system_prompt(self) -> str:
        examples = "\n\n".join(f"Question: {q}\n```sql\n{s}\n```" for q, s in FEW_SHOT)
        return (
            "You are a senior analytics engineer for Olist, a Brazilian e-commerce marketplace. "
            "You translate business questions into SQLite queries.\n\n"
            f"DATABASE SCHEMA\n{self.schema}\n\n{BUSINESS_NOTES}\n\n{RULES}\n\n"
            f"EXAMPLES\n{examples}"
        )

    def ask(self, question: str) -> AgentResult:
        result = AgentResult(question=question)
        start = time.monotonic()
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": f"Question: {question}"}]

        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts = attempt
            try:
                reply = self.llm.complete(messages)
            except Exception as e:  # network, auth, rate limit...
                result.error = f"LLM call failed: {e}"
                break
            sql = extract_sql(reply)
            try:
                data, truncated = run_readonly_query(self.con, sql)
                result.sql, result.data, result.truncated, result.error = sql, data, truncated, None
                result.trace.append({"attempt": attempt, "sql": sql, "error": None})
                break
            except (ValueError, sqlite3.Error) as e:
                # self-repair: show the model its own query and the error, then retry
                result.sql, result.error = sql, str(e)
                result.trace.append({"attempt": attempt, "sql": sql, "error": str(e)})
                messages += [
                    {"role": "assistant", "content": f"```sql\n{sql}\n```"},
                    {"role": "user", "content": f"That query failed with this error:\n{e}\n"
                                                "Fix it and reply with the corrected SQL only."},
                ]

        if result.ok and self.explain:
            result.answer = self._explain(question, result)
        result.latency_s = round(time.monotonic() - start, 2)
        if self.log:
            self._log(result)
        return result

    def _explain(self, question: str, result: AgentResult) -> str:
        preview = result.data.head(30).to_csv(index=False)
        messages = [
            {"role": "system", "content":
                "You explain query results to a business stakeholder in 2-4 sentences. "
                "Use only numbers that appear in the result. If the result is empty or "
                "says the question is not answerable, say so plainly. No SQL, no markdown tables."},
            {"role": "user", "content":
                f"Question: {question}\n\nSQL used:\n{result.sql}\n\n"
                f"Result ({len(result.data)} rows{', truncated' if result.truncated else ''}):\n{preview}"},
        ]
        try:
            return self.llm.complete(messages).strip()
        except Exception as e:
            return f"(could not generate an explanation: {e})"

    def _log(self, result: AgentResult) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "question": result.question,
                  "sql": result.sql, "ok": result.ok, "error": result.error,
                  "attempts": result.attempts, "rows": None if result.data is None else len(result.data),
                  "latency_s": result.latency_s, "model": getattr(self.llm, "model", "custom")}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def print_result(r: AgentResult) -> None:
    print("\n" + "=" * 80)
    print(f"Q: {r.question}")
    print("-" * 80)
    print(f"SQL (attempt {r.attempts}):\n{r.sql}")
    print("-" * 80)
    if r.ok:
        with pd.option_context("display.max_rows", 20, "display.width", 120):
            print(r.data.to_string(index=False, max_rows=20))
        if r.truncated:
            print(f"... showing first {MAX_ROWS} rows")
        if r.answer:
            print("-" * 80 + f"\nAnswer: {r.answer}")
    else:
        print(f"Failed: {r.error}")
    print(f"({r.latency_s}s)")


def main() -> None:
    try:
        from dotenv import load_dotenv  # optional: read GROQ_API_KEY from a .env file
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Ask business questions about the Olist data.")
    parser.add_argument("question", nargs="*", help="question in plain English (omit for interactive mode)")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="path to olist.db")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model id")
    parser.add_argument("--no-explain", action="store_true", help="skip the plain-English answer")
    parser.add_argument("--list-models", action="store_true", help="list Groq models for your key")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(GroqLLM(model=args.model).list_models()))
        return

    agent = SQLAgent(args.db, llm=GroqLLM(model=args.model), explain=not args.no_explain)
    if args.question:
        print_result(agent.ask(" ".join(args.question)))
        return

    print(f"Retail Analytics Copilot ({args.model}). Ask a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            print_result(agent.ask(q))


if __name__ == "__main__":
    sys.exit(main())
