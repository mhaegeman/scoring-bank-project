# Credit Risk Scoring Dashboard

> An interactive tool for bank advisors to assess client loan default risk, explore similar client profiles, and make informed credit decisions — powered by a LightGBM classifier.

---

## What does it do?

When a client applies for a loan, a bank advisor opens this dashboard, enters the client's ID, and instantly sees:

- **A default probability score** — how likely is this client to miss a payment?
- **A clear accept / monitor recommendation** based on a configurable risk threshold
- **Similar client profiles** from the historical database for comparison
- **Demographic comparisons** — how does this client compare to people with the same education level, income bracket, marital status, or employer type?
- **Interactive radar and bar charts** for visual explanation to the client
- **Model performance metrics** (confusion matrix, ROC curve, precision-recall curve) for advisors who want to understand model reliability

---

## Dashboard walkthrough

### 1 — Client lookup & risk score

Enter a client ID (range 100 002 – 112 188) and set the maximum acceptable default probability with the threshold slider.

![Client lookup and risk score](docs/images/01_client_score.png)

The dashboard immediately shows:
- Default probability as a percentage
- ✅ **Low risk** — credit can be considered
- ⚠️ **High risk** — client requires closer review
- Whether the client has a prior default history

---

### 2 — Client profile card

The right panel shows the client's key financial indicators in plain language:

| Indicator | What it means |
|---|---|
| **Loan Duration** | Length of the requested loan in months |
| **Annuity** | Monthly repayment amount (€) |
| **Age** | Client's age at application |
| **Employment Start** | Years since the client started their current job |
| **Annuity / Income ratio** | Monthly repayment as a share of monthly income |

![Client profile card](docs/images/02_client_profile.png)

---

### 3 — Similar client profiles

The model finds the 5 most similar clients in the historical database (using nearest-neighbour search) and displays their profiles and outcomes. This helps advisors understand how comparable cases were handled.

![Similar clients table](docs/images/03_similar_clients.png)

---

### 4 — Demographic comparison charts

Select a demographic dimension — gender, education level, income bracket, company type, or marital status — to see:

- A **bar chart** showing how many clients in the same group repaid vs. defaulted
- A **radar chart** overlaying the client's profile against the average good-payer and average defaulter in their group

![Radar and bar charts](docs/images/04_comparison_charts.png)

This gives advisors a visual story they can share directly with the client.

---

### 5 — Model performance metrics

Expand the metrics panel to inspect the classifier's historical performance:

- **Confusion matrix** — true/false positive and negative rates
- **ROC curve** — overall discrimination ability (AUC)
- **Precision-Recall curve** — performance on the minority (default) class

![Model metrics](docs/images/05_model_metrics.png)

---

## Quick start

### Prerequisites

- Python 3.10 or higher
- Git

### Install

```bash
git clone https://github.com/mhaegeman/scoring-bank-project.git
cd scoring-bank-project
pip install -r requirements.txt
```

### Run the dashboard

```bash
PYTHONPATH=src streamlit run src/scoring_bank/dashboard/app.py
```

Open your browser at `http://localhost:8501`.

---

## Project structure

```
scoring-bank-project/
├── src/
│   └── scoring_bank/           # Installable Python package
│       ├── config.py           # Centralised file paths & constants
│       ├── features/
│       │   └── engineering.py  # Feature engineering & EDA utilities
│       ├── models/
│       │   ├── scorer.py       # LightGBM prediction helpers
│       │   └── similarity.py   # Nearest-neighbour client lookup
│       ├── data/
│       │   └── loader.py       # Data loading functions
│       └── dashboard/
│           ├── app.py          # Streamlit application entry point
│           └── visualizations.py  # Radar charts & bar plots
├── models/                     # Pre-trained model files (.pkl)
├── data/
│   └── data_api.csv            # 10 000 client records (130+ features)
├── notebooks/                  # Archived exploration notebooks
│   ├── 01_scoring_pipeline.ipynb
│   └── 02_api_data_prep.ipynb
├── tests/                      # pytest test suite (43 tests, 64% coverage)
├── .github/workflows/ci.yml    # GitHub Actions CI (lint + test on every push)
├── pyproject.toml              # Package config, ruff, black, pytest, coverage
├── requirements.txt            # Pinned runtime dependencies
└── requirements-dev.txt        # Dev/test dependencies
```

---

## How the model works

The credit scoring model is a **LightGBM classifier** trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. It combines features from seven source tables:

| Source table | What it contributes |
|---|---|
| Application | Demographics, income, employment, credit amount |
| Bureau | Previous loans from other institutions |
| Bureau Balance | Monthly status of those loans |
| POS Cash | Point-of-sale & cash loan history |
| Credit Card | Credit card balance history |
| Installments | Previous loan repayment behaviour |
| Previous Applications | Past Home Credit loan applications |

The model outputs a probability between 0 and 1. A bank advisor sets their own **acceptance threshold** — the maximum default probability they are willing to accept — depending on the portfolio risk appetite.

---

## Development

### Run tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest tests/ -v
```

### Run linting & formatting

```bash
ruff check src/scoring_bank/ tests/
black src/scoring_bank/ tests/
```

### CI

Every push triggers the GitHub Actions pipeline which runs:

1. `ruff` — pyflakes + pycodestyle checks
2. `black --check` — formatting consistency
3. `pytest` with coverage (≥ 60% required on testable code)

---

## Adding screenshots

To populate the images above, take screenshots of the running dashboard and save them to `docs/images/`:

```
docs/images/
├── 01_client_score.png
├── 02_client_profile.png
├── 03_similar_clients.png
├── 04_comparison_charts.png
└── 05_model_metrics.png
```

---

## License

This project is for educational and demonstration purposes.
