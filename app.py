# app.py — with progress bar
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

# Optional: LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

SEED = 42
np.random.seed(SEED)
st.set_page_config(page_title="Default Risk Predictor", page_icon="⏳", layout="centered")
st.title("⏳ Default Risk Predictor (with progress)")

st.write("Upload your **training** and **test** CSVs and click **Train & Predict**. "
         "A progress bar shows the stages so users know what’s happening.")

# ---- UI ----
train_file = st.file_uploader("Training CSV (must include 'repaid_loan')", type=["csv"])
test_file  = st.file_uploader("Test CSV (must include 'row_id')", type=["csv"])
go = st.button("🚀 Train & Predict", type="primary", disabled=not (train_file and test_file))
baseline = st.button("🧪 Baseline (0.5 for all rows)", disabled=not test_file)

def preprocess(X, X_test):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Numeric
    X[numeric_cols]      = X[numeric_cols].fillna(X[numeric_cols].median())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X[numeric_cols].median())

    # Categorical
    for col in categorical_cols:
        mode_val = X[col].mode()[0] if not X[col].mode().empty else "NA"
        X[col] = X[col].fillna(mode_val).astype(str)
        X_test[col] = X_test[col].fillna(mode_val).astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([X[col], X_test[col]], axis=0))
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
    return X, X_test

# ---- XGB progress callback ----
class StreamlitProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, pbar, start=10, end=70, total_rounds=1200):
        self.pbar = pbar
        self.start = start
        self.end = end
        self.total = max(1, total_rounds)

    def after_iteration(self, model, epoch, evals_log):
        # Update every 10 rounds to reduce UI spam
        if (epoch + 1) % 10 == 0 or (epoch + 1) == self.total:
            frac = min(1.0, (epoch + 1) / self.total)
            value = int(self.start + (self.end - self.start) * frac)
            self.pbar.progress(value, text=f"Training XGBoost… {int(frac*100)}%")
        return False

def train_and_predict(train_df, test_df, pbar):
    if "repaid_loan" not in train_df.columns:
        raise KeyError("Training data must contain 'repaid_loan'.")
    if "row_id" not in test_df.columns:
        raise KeyError("Test data must contain 'row_id'.")

    pbar.progress(1, text="Reading & validating data…")

    y = train_df["repaid_loan"].astype(int).copy()
    X = train_df.drop(columns=["repaid_loan", "row_id"], errors="ignore").copy()
    X_test = test_df.drop(columns=["row_id"], errors="ignore").copy()
    row_ids = test_df["row_id"].astype(int).values

    # Preprocess
    pbar.progress(5, text="Preprocessing (imputing & encoding)…")
    X, X_test = preprocess(X, X_test)

    # Split
    pbar.progress(8, text="Creating train/validation split…")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    spw = max(1e-6, neg / max(1, pos))

    # XGBoost
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    dte = xgb.DMatrix(X_test)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 10.0,
        "lambda": 2.0,
        "gamma": 0.05,
        "random_state": SEED,
        "tree_method": "hist",
        "scale_pos_weight": spw,
    }
    xgb_rounds = 1200
    xgb_es = 100
    pbar.progress(10, text="Training XGBoost… 0%")
    xgb_booster = xgb.train(
        params=xgb_params,
        dtrain=dtr,
        num_boost_round=xgb_rounds,
        evals=[(dva, "valid")],
        early_stopping_rounds=xgb_es,
        verbose_eval=False,
        callbacks=[StreamlitProgressCallback(pbar, start=10, end=70, total_rounds=xgb_rounds)]
    )
    xgb_best_iter = xgb_booster.best_iteration or xgb_rounds
    xgb_va = xgb_booster.predict(dva, iteration_range=(0, xgb_best_iter + 1))
    xgb_test = xgb_booster.predict(dte, iteration_range=(0, xgb_best_iter + 1))
    xgb_auc = roc_auc_score(y_va, xgb_va)

    # LightGBM (optional)
    lgb_test = None; lgb_auc = None
    if LGB_AVAILABLE:
        pbar.progress(72, text="Training LightGBM…")
        lgbm = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=10.0, reg_lambda=2.0,
            random_state=SEED, n_jobs=-1, force_col_wise=True, scale_pos_weight=spw, verbose=-1
        )
        # light progress: bump the bar a few times during fit
        # (LightGBM Python API doesn't expose per-iter callback cleanly like XGB here)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                 callbacks=[lgb.early_stopping(100, verbose=False)])
        pbar.progress(80, text="LightGBM finishing…")
        lgb_va = lgbm.predict_proba(X_va)[:, 1]
        lgb_test = lgbm.predict_proba(X_test)[:, 1]
        lgb_auc = roc_auc_score(y_va, lgb_va)
    else:
        pbar.progress(72, text="Skipping LightGBM (not installed)…")

    # MLP
    pbar.progress(82, text="Training MLP…")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(160, 80), activation="relu", solver="adam",
                        max_iter=400, random_state=SEED, early_stopping=True, validation_fraction=0.1, verbose=False)
    mlp.fit(X_tr_s, y_tr)
    mlp_va = mlp.predict_proba(X_va_s)[:, 1]
    mlp_test = mlp.predict_proba(X_te_s)[:, 1]
    mlp_auc = roc_auc_score(y_va, mlp_va)

    # Blend
    pbar.progress(92, text="Blending predictions…")
    if LGB_AVAILABLE and lgb_auc is not None:
        aucs = np.array([xgb_auc, lgb_auc, mlp_auc]); aucs = np.clip(aucs, 1e-6, None)
        w = aucs / aucs.sum(); w_xgb, w_lgb, w_mlp = w.tolist()
        p_repaid = w_xgb * xgb_test + w_lgb * lgb_test + w_mlp * mlp_test
        weights = {"xgb": w_xgb, "lgb": w_lgb, "mlp": w_mlp}
    else:
        w_xgb, w_mlp = 0.8, 0.2; p_repaid = w_xgb * xgb_test + w_mlp * mlp_test
        weights = {"xgb": w_xgb, "mlp": w_mlp}

    p_repaid = np.clip(p_repaid, 0.0, 1.0)

    pbar.progress(97, text="Preparing download…")
    metrics = {"xgb_auc": float(xgb_auc), "mlp_auc": float(mlp_auc)}
    if lgb_auc is not None: metrics["lgb_auc"] = float(lgb_auc)

    return row_ids, p_repaid, metrics, weights

def to_semicolon_bytes(row_ids, probs):
    lines = [f"{int(r)}; {p}" for r, p in zip(row_ids, probs)]
    return ("\n".join(lines)).encode("utf-8")

if baseline and test_file:
    df = pd.read_csv(test_file)
    if "row_id" not in df.columns:
        st.error("Test data must contain 'row_id'.")
    else:
        rid = df["row_id"].astype(int).values
        probs = np.full_like(rid, 0.5, dtype=float)
        st.download_button("Download final_predictions_FINAL.csv",
                           data=to_semicolon_bytes(rid, probs),
                           file_name="final_predictions_FINAL.csv", mime="text/csv")

if go and (train_file and test_file):
    pbar = st.progress(0, text="Starting…")
    try:
        train_df = pd.read_csv(train_file)
        test_df  = pd.read_csv(test_file)
        row_ids, probs, metrics, weights = train_and_predict(train_df, test_df, pbar)
        pbar.progress(100, text="Done ✅")
        st.markdown("**Validation AUCs:**")
        for k, v in metrics.items(): st.write(f"- {k}: `{v:.4f}`")
        st.markdown("**Blend Weights:**")
        for k, v in weights.items(): st.write(f"- {k}: `{v:.3f}`")
        st.download_button("⬇️ Download final_predictions_FINAL.csv",
                           data=to_semicolon_bytes(row_ids, probs),
                           file_name="final_predictions_FINAL.csv", mime="text/csv")
    except Exception as e:
        pbar.progress(0, text="Error")
        st.error(f"{e}")
