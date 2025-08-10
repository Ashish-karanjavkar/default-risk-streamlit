# app.py — Risk Predictor Calculator (shield icon + optional logo + fixed checklist)
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
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

SEED = 42
np.random.seed(SEED)

# 🛡️ New page icon
st.set_page_config(page_title="Risk Predictor Calculator", page_icon="🛡️", layout="centered")

# --- Pastel styling ---
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f6fbff 0%, #fffdf7 100%); }
.title-card { background:#ffffffCC;border:1px solid #e6eef8;border-radius:16px;padding:18px 20px;box-shadow:0 8px 30px rgba(16,24,40,.06);margin-bottom:6px; }
.subtitle { color:#475569;font-size:.95rem;margin-top:6px; }
.block-container { padding-top: 1.2rem; }
.card { background:#ffffffEE;border:1px solid #edf2f7;border-radius:14px;padding:16px 18px;box-shadow:0 4px 18px rgba(0,0,0,.05); }
.footer { text-align:center;margin-top:18px;color:#6b7280;font-size:.95rem; }
.stButton>button { border-radius:10px;padding:.6rem 1rem; }
.progress-wrap{ margin-top:12px; }
.logo-wrap { display:flex; gap:12px; align-items:center; }
.logo-wrap img { border-radius:12px; border:1px solid #edf2f7; }
</style>
""", unsafe_allow_html=True)

# --- Optional local logo.png in repo root ---
logo_html = ""
try:
    from PIL import Image
    img = Image.open("logo.png")
    st.image(img, width=72)
except Exception:
    # If no logo.png, render a cute SVG badge instead
    st.markdown("""
    <div class="logo-wrap">
      <div style="width:72px;height:72px;border-radius:12px;border:1px solid #edf2f7;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#E6F3FF,#FDF6E8)">
        <span style="font-size:34px">🛡️</span>
      </div>
      <div>
        <h1 style="margin:0">Risk Predictor Calculator</h1>
        <div class="subtitle">Upload LendingClub CSVs, train, and download a submission file.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.subheader("Run settings")
    fast = st.checkbox("Fast demo (subsample + fewer rounds)", value=True)
    skip_mlp = st.checkbox("Skip MLP (faster & stable)", value=True)
    sample_frac = st.slider("Training sample fraction", 0.1, 1.0, 0.5, 0.1,
                            help="Use only this fraction of training rows when Fast demo is ON.")

# --- Uploaders ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Upload data")
train_file = st.file_uploader("Training CSV (must include `repaid_loan`)", type=["csv"])
test_file  = st.file_uploader("Test CSV (must include `row_id`)", type=["csv"])
go = st.button("🚀 Train & Predict", type="primary", disabled=not (train_file and test_file))
st.markdown('</div>', unsafe_allow_html=True)

# --- Checklist logic ---
TASKS = [
    "Reading & validating data",
    "Preprocessing (imputing & encoding)",
    "Creating train/validation split",
    "Training XGBoost",
    "Training LightGBM",
    "Training MLP",
    "Blending predictions",
    "Preparing download"
]

def checklist_md(status_map):
    icon = {"queued":"🟡", "running":"⏳", "done":"✅"}
    lines = [f"- {icon.get(status_map.get(t, 'queued'), '🟡')} {t}" for t in TASKS]
    return "\n".join(lines)

def to_bytes_semicolon(row_ids, probs):
    lines = [f"{int(r)}; {p}" for r, p in zip(row_ids, probs)]
    return ("\n".join(lines)).encode("utf-8")

def preprocess(X, X_test):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    X[numeric_cols]      = X[numeric_cols].fillna(X[numeric_cols].median())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X[numeric_cols].median())
    for col in categorical_cols:
        mode_val = X[col].mode()[0] if not X[col].mode().empty else "NA"
        X[col] = X[col].fillna(mode_val).astype(str)
        X_test[col] = X_test[col].fillna(mode_val).astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([X[col], X_test[col]], axis=0))
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
    return X, X_test

# --- Status UI ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Run status")
pbar = st.progress(0, text="Waiting to start…")
status_placeholder = st.empty()
status_placeholder.markdown(checklist_md({t:"queued" for t in TASKS}))
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">Created by: <strong>Vivek Maharaj</strong></div>', unsafe_allow_html=True)

# --- Run pipeline ---
if go and (train_file and test_file):
    status = {t:"queued" for t in TASKS}

    def set_status(name, state, p=None, label=None):
        status[name] = state
        if p is not None:
            pbar.progress(p, text=label or "")
        status_placeholder.markdown(checklist_md(status))

    try:
        # 1) Read
        set_status("Reading & validating data", "running", 5, "Reading & validating data…")
        train_df = pd.read_csv(train_file)
        test_df  = pd.read_csv(test_file)
        if "repaid_loan" not in train_df.columns: st.error("Training data missing `repaid_loan`"); st.stop()
        if "row_id" not in test_df.columns: st.error("Test data missing `row_id`"); st.stop()
        if fast and 0 < sample_frac < 1.0:
            train_df = train_df.sample(frac=sample_frac, random_state=SEED, replace=False)
        y = train_df["repaid_loan"].astype(int).copy()
        X = train_df.drop(columns=["repaid_loan", "row_id"], errors="ignore").copy()
        X_test = test_df.drop(columns=["row_id"], errors="ignore").copy()
        row_ids = test_df["row_id"].astype(int).values
        set_status("Reading & validating data", "done")

        # 2) Preprocess
        set_status("Preprocessing (imputing & encoding)", "running", 15, "Preprocessing (imputing & encoding)…")
        X, X_test = preprocess(X, X_test)
        set_status("Preprocessing (imputing & encoding)", "done")

        # 3) Split
        set_status("Creating train/validation split", "running", 22, "Creating train/validation split…")
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        pos = int((y == 1).sum()); neg = int((y == 0).sum())
        spw = max(1e-6, neg / max(1, pos))
        set_status("Creating train/validation split", "done")

        # Speeds
        if fast:
            xgb_rounds = 300; xgb_eta = 0.1; xgb_depth = 4
            lgb_rounds = 400; mlp_max_iter = 100; mlp_layers = (64, 32)
        else:
            xgb_rounds = 1200; xgb_eta = 0.03; xgb_depth = 6
            lgb_rounds = 3000; mlp_max_iter = 300; mlp_layers = (160, 80)

        # 4) XGBoost
        set_status("Training XGBoost", "running", 35, "Training XGBoost…")
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        dte = xgb.DMatrix(X_test)
        xgb_params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "eta": xgb_eta, "max_depth": xgb_depth, "min_child_weight": 2,
            "subsample": 0.8, "colsample_bytree": 0.8, "alpha": 10.0, "lambda": 2.0, "gamma": 0.05,
            "random_state": SEED, "tree_method": "hist", "scale_pos_weight": spw,
        }
        xgb_booster = xgb.train(params=xgb_params, dtrain=dtr, num_boost_round=xgb_rounds,
                                evals=[(dva, "valid")], early_stopping_rounds=50, verbose_eval=False)
        xgb_best_iter = xgb_booster.best_iteration or xgb_rounds
        xgb_va = xgb_booster.predict(dva, iteration_range=(0, xgb_best_iter + 1))
        xgb_test = xgb_booster.predict(dte, iteration_range=(0, xgb_best_iter + 1))
        xgb_auc = roc_auc_score(y_va, xgb_va)
        set_status("Training XGBoost", "done")

        # 5) LightGBM (skip in fast mode)
        lgb_test = None; lgb_auc = None
        if LGB_AVAILABLE and not fast:
            set_status("Training LightGBM", "running", 55, "Training LightGBM…")
            lgbm = lgb.LGBMClassifier(n_estimators=lgb_rounds, learning_rate=0.03, num_leaves=63,
                                      subsample=0.8, colsample_bytree=0.8, reg_alpha=10.0, reg_lambda=2.0,
                                      random_state=SEED, n_jobs=-1, force_col_wise=True, scale_pos_weight=spw, verbose=-1)
            lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                     callbacks=[lgb.early_stopping(100, verbose=False)])
            lgb_va = lgbm.predict_proba(X_va)[:, 1]; lgb_test = lgbm.predict_proba(X_test)[:, 1]
            lgb_auc = roc_auc_score(y_va, lgb_va)
            set_status("Training LightGBM", "done")
        else:
            set_status("Training LightGBM", "done", 55, "Skipping LightGBM (Fast mode)…")

        # 6) MLP (optional)
        mlp_test = None; mlp_auc = None
        if not skip_mlp:
            set_status("Training MLP", "running", 70, "Training MLP…")
            scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr); X_va_s = scaler.transform(X_va); X_te_s = scaler.transform(X_test)
            mlp = MLPClassifier(hidden_layer_sizes=mlp_layers, activation="relu", solver="adam",
                                max_iter=mlp_max_iter, random_state=SEED, early_stopping=True, validation_fraction=0.1, verbose=False)
            mlp.fit(X_tr_s, y_tr); mlp_va = mlp.predict_proba(X_va_s)[:, 1]; mlp_test = mlp.predict_proba(X_te_s)[:, 1]
            mlp_auc = roc_auc_score(y_va, mlp_va)
            set_status("Training MLP", "done")
        else:
            set_status("Training MLP", "done", 70, "Skipping MLP (disabled)…")

        # 7) Blend
        set_status("Blending predictions", "running", 85, "Blending predictions…")
        preds = []; weights = []
        preds.append(xgb_test); weights.append(0.75)
        if lgb_test is not None: preds.append(lgb_test); weights.append(0.15)
        if mlp_test is not None: preds.append(mlp_test); weights.append(0.10)
        w = np.array(weights, dtype=float); w = w / w.sum()
        P = np.zeros_like(preds[0])
        for pi, wi in zip(preds, w): P += wi * pi
        p_repaid = np.clip(P, 0.0, 1.0)
        set_status("Blending predictions", "done")

        # 8) Prepare download
        set_status("Preparing download", "running", 95, "Preparing download…")
        st.markdown("**Validation AUCs:**")
        st.write(f"- xgb_auc: `{xgb_auc:.4f}`")
        if lgb_auc is not None: st.write(f"- lgb_auc: `{lgb_auc:.4f}`")
        if mlp_auc is not None: st.write(f"- mlp_auc: `{mlp_auc:.4f}`")
        st.download_button("⬇️ Download final_predictions_FINAL.csv",
                           data=to_bytes_semicolon(row_ids, p_repaid),
                           file_name="final_predictions_FINAL.csv", mime="text/csv")
        set_status("Preparing download", "done", 100, "Done ✅")

    except Exception as e:
        pbar.progress(0, text="Error")
        status_placeholder.markdown(checklist_md({t:"queued" for t in TASKS}))
        st.error(f"{e}")
        st.stop()
