# Sales Forecasting & Demand Intelligence System

End-to-end project: EDA → time series decomposition → 3-model forecasting → segment-level
forecasting → anomaly detection → product clustering → Streamlit dashboard → executive report.

## ⚠️ Important — data files not included

`train.csv` (Superstore Sales, from Kaggle: `rohitsahoo/sales-forecasting`) and `vgsales.csv`
(Video Game Sales, from Kaggle: `gregorut/videogamesales`) are **not bundled in this ZIP** —
Kaggle's terms require you to download them yourself. Grab both and drop them in this folder
before running anything.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`prophet` can be finicky to install on Windows — if it fails, install `pystan`/build tools first,
or run the notebook in Google Colab instead (Colab has most of these preinstalled).

## Running the analysis

```bash
jupyter notebook analysis.ipynb
```

Run all cells top to bottom. It covers Tasks 1–6: EDA, decomposition, ADF stationarity testing,
SARIMA/Prophet/XGBoost forecasting with a model comparison table, category/region-level
forecasts, anomaly detection (Isolation Forest + Z-score), and K-Means product segmentation
with PCA visualization. Charts are saved to `charts/` as you go.

**A few things you'll need to fill in as you run it** (flagged in the notebook itself):
- The exact SARIMA `(p,d,q)(P,D,Q,m)` the grid search lands on — it's chosen automatically by
  AIC, but read the printed summary and sanity-check it.
- The written observations after the decomposition plot and the anomaly section are templates —
  replace the bracketed reasoning with what your actual data shows.

## Running the dashboard

```bash
streamlit run app.py
```

Opens locally at `http://localhost:8501`. Four pages: Sales Overview, Forecast Explorer,
Anomaly Report, Product Demand Segments — matching Task 7 exactly.

## Deploying to Streamlit Community Cloud (free)

1. Push this folder (including `train.csv`) to a **public** GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app".
3. Point it at your repo, branch, and `app.py`. Deploy.
4. Copy the live `*.streamlit.app` URL for your submission.

## Files in this folder

| File | Purpose |
|---|---|
| `analysis.ipynb` | Tasks 1–6, fully coded |
| `app.py` | Task 7 — Streamlit dashboard |
| `requirements.txt` | Task 7/8 — dependency list for redeployment |
| `summary_report.docx` | Task 8 — executive report template (fill in bracketed numbers after running the notebook) |
| `charts/` | Output folder — populated when you run the notebook |

## Submission checklist

- [ ] Downloaded real `train.csv` and `vgsales.csv` into this folder
- [ ] Ran `analysis.ipynb` end-to-end, no errors
- [ ] Filled in the bracketed numbers in `summary_report.docx` with real results
- [ ] Exported `charts/` (auto-populated by the notebook)
- [ ] Deployed `app.py` to Streamlit Community Cloud, grabbed the live link
- [ ] Pushed everything to a public GitHub repo
- [ ] Zipped the folder as `SalesForecasting_[YourName].zip`
- [ ] Submitted via the Google Form with Colab link + Streamlit link
