# Retail Analytics Copilot

SQL, machine learning, statistics and a GenAI text-to-SQL agent on ~99k real e-commerce orders from
[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), a Brazilian marketplace (Sep 2016 – Oct 2018).

| Part | Question | What's in it |
|---|---|---|
| 1 | How is the business doing, and who are the customers? | SQLite, CTEs, window functions; RFM segmentation with K-Means |
| 2 | Can we predict a late delivery when the order is placed? | Logistic Regression vs Random Forest vs Gradient Boosting, leakage-free features, rolling time-series CV |
| 3 | Why does the model flag an order? | Permutation importance, LR coefficients, exact linear SHAP reason codes, partial dependence |
| 4 | Do late deliveries hurt reviews? How big must a retention A/B test be? | Mann-Whitney U, Welch t-test, chi-square, Mantel-Haenszel, power analysis + simulation |
| 5 | Can a manager ask questions in plain English? | Groq LLM agent that writes read-only SQL, with 4 safety layers and an evaluation set |
| 6 | What should the business do? | 4 recommendations with estimated monthly impact |

## Results at a glance

- **Growth:** Jan–Aug GMV grew **140%** year over year (R$3.6M → R$8.6M). Only **3.0%** of customers ever order twice.
- **Delivery:** orders arrive in **12.6 days** on average against **23.7 days** promised, yet **6.8%** are still late,
  concentrated in the North/Northeast and at sellers slow to hand over to the carrier.
- **Late-delivery model** (strict out-of-time test, May–Aug 2018, 4.4% late): Logistic Regression, selected by rolling CV,
  **ROC-AUC 0.72**, PR-AUC 0.10 (2.2× the base rate). Flagging the riskiest 10% of orders catches **25%** of late
  deliveries (2.5× lift); the riskiest 30% catch 60%.
- **Leakage check:** a random split would have overstated tree-model AUC by **0.09–0.15**.
- **Reviews:** late orders get **1-2 stars 62%** of the time vs 9% on time (relative risk **6.7**, 95% CI 6.5–6.9);
  mean 2.3 vs 4.3 stars (Mann-Whitney p < 1e-300, rank-biserial r = −0.64). Holds within states (Mantel-Haenszel OR 15.8).
- **A/B test:** the 90-day repeat rate is only 1.3%, so detecting a 20% lift needs **33,245 customers per group**
  (~45 weeks of new customers); a 50% lift needs 6,032 (~8 weeks). Verified by simulation (power 79%, false positives 4.8%).

Full write-up: [REPORT.md](REPORT.md). Every number above is produced by the notebook and saved in `results/summary.json`.

## Repository layout

```
retail_analytics_copilot.ipynb   the analysis, with outputs (Parts 1-6)
build_db.py                      loads the 9 CSVs into SQLite (olist.db) + indexes + v_order_delivery view
agent/sql_agent.py               text-to-SQL agent (Groq), read-only guardrails, self-repair, CLI
agent/eval_agent.py              execution-accuracy evaluation on 12 questions
agent/eval_questions.json        the 12 questions with hand-written gold SQL
figures/                         charts used in the report
results/summary.json             key numbers written by the notebook
REPORT.md                        short business report
requirements.txt, .env.example
data/                            put the Kaggle CSVs here (not committed)
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Data: download the Kaggle dataset and unzip the 9 CSVs into data/
python build_db.py                 # builds olist.db (~170 MB, ~5 seconds)

# 2. Notebook: Run All takes about 7 minutes (the rolling CV fits 54 models)
jupyter notebook retail_analytics_copilot.ipynb

# 3. Agent: needs a free Groq API key from https://console.groq.com/keys
cp .env.example .env               # then paste your key into .env
python agent/sql_agent.py "Which 5 states have the highest late-delivery rate?"
python agent/sql_agent.py          # interactive mode
python agent/eval_agent.py         # accuracy on the 12-question test set
python agent/sql_agent.py --list-models
```

The agent defaults to `openai/gpt-oss-120b` (Groq retired `llama-3.3-70b-versatile` in August 2026). Set
`GROQ_MODEL` in `.env` to use any other model your key can access.

## Key design decisions

- **No leakage in the model.** Only information available at checkout: the promised date, basket, payment, addresses,
  and point-in-time history (a seller's past late rate, hand-over speed and workload, and recent delivery conditions,
  using only deliveries that finished before the order was placed). Training labels are restricted to orders already
  delivered at each cut-off.
- **Time-based evaluation.** Rolling-origin CV over 6 monthly folds for model selection, then a single untouched
  test on the last four months. Purchase month and platform order volume are excluded because tree models used them
  to memorise periods.
- **Top-10% operating point** instead of a probability threshold, because the late rate shifts between periods
  (1%–19% by month) and class weights make raw probabilities uncalibrated.
- **Effect sizes over p-values.** With ~96k orders, everything is "significant"; the report leads with relative risk,
  mean differences with confidence intervals, and rank-biserial correlation.
- **Agent safety is layered.** Prompt rules → static validator (single SELECT/WITH, no write/admin keywords) →
  read-only SQLite connection (`mode=ro`, `query_only`) → authorizer that denies every non-read operation, plus a
  10-second timeout and 200-row cap. Failed queries are sent back to the model with the error for up to 2 repairs.

## Limitations

- Observational data: the review analysis shows a strong, robust association, not proof of causation.
- The test period ends in Aug 2018; orders still undelivered at the data snapshot are missing, which can hide some
  very late orders from the last weeks.
- Recommendation impacts are estimates under stated assumptions, and estimates 1–3 overlap.
- SHAP for the tree model (notebook 3.5) needs `pip install shap`; it couldn't be installed where the outputs were
  generated, so permutation importance and exact linear SHAP are the explanations shown.
- The agent's accuracy on the 12-question set has to be measured with your own Groq key (`agent/eval_agent.py`).
