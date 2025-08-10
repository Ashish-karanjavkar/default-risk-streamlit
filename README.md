# Default-Risk Predictor — Streamlit App

A simple Streamlit app that trains a model on the **LendingClub** training CSV and produces a **two-column (no headers)** `submission.csv` with `[row_id, probability_of_repayment]`.

## How it works
- Upload `loans_d_training_set.csv` (must include `repaid_loan`) and `loans_d_test_set.csv` (must include `row_id`).
- Click **Train & Predict**. The app trains an ensemble (XGBoost + MLP; LightGBM optional) and gives you a **Download** button for `submission.csv`.
- A **Baseline** button can generate `0.5` for all rows (for setup testing only).

## Deploy on Streamlit Cloud
1. Create a **new GitHub repo** and add:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - (Optional) `.gitignore` with `*.csv` so you don’t commit the big datasets.
2. Go to https://share.streamlit.io → **New app** → connect your GitHub, pick the repo and `app.py` as the main file → **Deploy**.
3. Open the app URL, upload the two CSVs, and download `submission.csv`.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Do **not** commit the course CSVs to GitHub; just upload them at runtime in the app.
- For the assignment, you still need to submit your **single Python script** and your **1000-word report** per the module rules.
