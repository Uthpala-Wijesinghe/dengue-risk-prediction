import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
EXAMPLE_PATH = os.path.join(BASE_DIR, "models", "holdout_sample.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "results.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "model_ready.csv")

ZONE_COLOURS = {"Safety": "#3F9142", "Warning": "#D9963A", "Epidemic": "#C0433A"}
ZONE_ADVICE = {
    "Safety": "Cases are at or below this district's normal level. Continue routine surveillance.",
    "Warning": "Cases are above normal for this district. Step up larval surveys, clean-up campaigns and public awareness.",
    "Epidemic": "Cases are far above normal (over mean + 2 SD). Prepare hospital capacity and start intensive vector control.",
}
KNOWN_TARGETS = ["zone_name", "is_epidemic", "cases"]
MODEL_CHOICES = ["Decision Tree", "Logistic / Ridge Regression", "Random Forest", "XGBoost",
                 "Soft Voting (RF + XGB)", "Stacking (RF + XGB)"]


# the pickled pipeline uses this function (defined in the notebook), so it must exist before unpickling
def shifted_log1p(x):
    return np.log1p(np.asarray(x, dtype=float) + 100.0)


@st.cache_resource
def load_bundle():
    import __main__
    __main__.shifted_log1p = shifted_log1p
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_example():
    if os.path.exists(EXAMPLE_PATH):
        return pd.read_csv(EXAMPLE_PATH).iloc[0].to_dict()
    return None


@st.cache_data
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def read_csv(source):
    return pd.read_csv(source)


st.set_page_config(page_title="Dengue Risk Dashboard - Sri Lanka", layout="wide")

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("models/model.pkl not found. Run the notebook first to train and save the model.")
    st.stop()

st.title("Sri Lanka Dengue Risk Dashboard")
st.caption("Random Forest vs XGBoost and their ensembles on district-level dengue data.")

tab_predict, tab_train, tab_results = st.tabs(["Predict", "Train & Compare", "Notebook Results"])

# ------------------------------------------------------------------ tab 1: predict
with tab_predict:
    example = load_example()
    reference = bundle["district_reference"]
    st.markdown(
        f"Deployed model: **{bundle['model_names']['zone']}**, trained on {bundle['trained_on']}. "
        "Enter last month's situation for a district to predict this month's risk zone, "
        "Epidemic probability and expected case count."
    )

    defaults = {
        "district": "Colombo", "season": bundle["seasons"][0],
        "prev_month_incidence": 100.0, "prev_year_same_month_incidence": 100.0, "rolling_3month_avg": 100.0,
        "mom_change_pct": 0.0, "neighboring_district_cases_lag1": 100.0,
        "rainfall_3_months_prior": 150.0, "temperature_mean_3_months_prior": 27.0,
        "humidity_mean_1_month_prior": 80.0, "humidity_mean_2_months_prior": 80.0, "wind_speed_mean": 10.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if example is not None and st.button(f"Fill in the held-out example ({example['district']}, "
                                         f"{int(example['year'])}-{int(example['month']):02d})"):
        for key in defaults:
            value = example[key]
            st.session_state[key] = value if isinstance(value, str) else float(value)

    with st.form("inputs"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("District and season")
            district = st.selectbox("District", bundle["districts"], key="district")
            season = st.selectbox("Season", bundle["seasons"], key="season",
                                  help="NE_Monsoon: Dec-Feb, Inter_Monsoon: Mar-Apr and Oct-Nov, SW_Monsoon: May-Sep")
            ref = reference.loc[district]
            st.markdown("**District profile** (filled in from the census)")
            elevation = st.number_input("Elevation (m)", value=float(ref["elevation"]))
            population_density = st.number_input("Population density (per km²)", value=float(ref["population_density"]))
            urban_pct = st.number_input("Urban population (%)", value=float(ref["urban_pct"]), min_value=0.0, max_value=100.0)
            youth_pct = st.number_input("Youth share (%)", value=float(ref["youth_pct"]), min_value=0.0, max_value=100.0)

        with col2:
            st.subheader("Recent cases")
            prev_month = st.number_input("Cases last month", min_value=0.0, step=1.0, key="prev_month_incidence")
            prev_year = st.number_input("Cases in the same month last year", min_value=0.0, step=1.0,
                                        key="prev_year_same_month_incidence")
            rolling = st.number_input("Average cases over the last 3 months", min_value=0.0, key="rolling_3month_avg")
            mom = st.number_input("Change from 2 months ago to last month (%)", min_value=-100.0, key="mom_change_pct")
            neighbours = st.number_input("Average cases last month in the 3 nearest districts", min_value=0.0,
                                         key="neighboring_district_cases_lag1")

        with col3:
            st.subheader("Weather")
            rain3 = st.number_input("Rainfall 3 months ago (mm)", min_value=0.0, key="rainfall_3_months_prior")
            temp3 = st.number_input("Mean temperature 3 months ago (°C)", key="temperature_mean_3_months_prior")
            hum1 = st.number_input("Mean humidity last month (%)", min_value=0.0, max_value=100.0,
                                   key="humidity_mean_1_month_prior")
            hum2 = st.number_input("Mean humidity 2 months ago (%)", min_value=0.0, max_value=100.0,
                                   key="humidity_mean_2_months_prior")
            wind = st.number_input("Mean wind speed this month (km/h)", min_value=0.0, key="wind_speed_mean")

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "district": district, "season": season,
            "rainfall_3_months_prior": rain3, "temperature_mean_3_months_prior": temp3,
            "humidity_mean_1_month_prior": hum1, "humidity_mean_2_months_prior": hum2,
            "wind_speed_mean": wind, "elevation": elevation,
            "prev_month_incidence": prev_month, "prev_year_same_month_incidence": prev_year,
            "rolling_3month_avg": rolling, "mom_change_pct": mom,
            "neighboring_district_cases_lag1": neighbours,
            "population_density": population_density, "urban_pct": urban_pct, "youth_pct": youth_pct,
        }])[bundle["feature_columns"]]

        zone_proba = bundle["zone_classifier"].predict_proba(row)[0]
        zone = bundle["zone_names"][int(np.argmax(zone_proba))]
        epidemic_proba = bundle["epidemic_classifier"].predict_proba(row)[0, 1]
        cases = max(0.0, float(bundle["case_regressor"].predict(row)[0]))

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.markdown(
            f"<div style='padding:1rem;border-radius:8px;background:{ZONE_COLOURS[zone]};color:white'>"
            f"<div style='font-size:0.9rem'>Predicted risk zone</div>"
            f"<div style='font-size:2rem;font-weight:700'>{zone}</div></div>",
            unsafe_allow_html=True,
        )
        r2.metric("Epidemic alarm probability", f"{epidemic_proba:.0%}",
                  help="From the separate Epidemic-vs-not model. Above 50% counts as an alarm. This is a risk score, not a calibrated probability: on the 2024-2026 test years the model overestimated (notebook Section 8).")
        r3.metric("Expected cases this month", f"{cases:,.0f}")

        st.info(ZONE_ADVICE[zone])
        st.markdown("**Zone probabilities**")
        st.bar_chart(pd.DataFrame({"probability": zone_proba}, index=bundle["zone_names"]))

        if example is not None and district == example["district"] and np.isclose(prev_month, example["prev_month_incidence"]):
            st.caption(f"Held-out example, actual outcome: {example['zone_name']} with {int(example['cases'])} cases.")


# ------------------------------------------------------------------ tab 2: train & compare
def build_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def make_models(task, chosen, rf_cfg, xgb_cfg, n_classes):
    rf_common = dict(n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth"],
                     min_samples_split=2 * rf_cfg["min_samples_leaf"], min_samples_leaf=rf_cfg["min_samples_leaf"],
                     max_features=rf_cfg["max_features"], max_samples=0.8, bootstrap=True, n_jobs=-1, random_state=42)
    xgb_common = dict(n_estimators=xgb_cfg["n_estimators"], learning_rate=xgb_cfg["learning_rate"],
                      max_depth=xgb_cfg["max_depth"], min_child_weight=3, subsample=xgb_cfg["subsample"],
                      colsample_bytree=0.7, reg_lambda=xgb_cfg["reg_lambda"], reg_alpha=0.1, gamma=0.1,
                      tree_method="hist", n_jobs=-1, random_state=42)
    if task == "Classification":
        rf = RandomForestClassifier(criterion="gini", **rf_common)
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        xgb = XGBClassifier(objective=objective, **xgb_common)
        all_models = {
            "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=40,
                                                    min_samples_leaf=10, random_state=42),
            "Logistic / Ridge Regression": LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=2000),
            "Random Forest": rf,
            "XGBoost": xgb,
            "Soft Voting (RF + XGB)": VotingClassifier([("rf", rf), ("xgb", xgb)], voting="soft"),
            "Stacking (RF + XGB)": StackingClassifier([("rf", rf), ("xgb", xgb)],
                                                      final_estimator=LogisticRegression(max_iter=2000),
                                                      stack_method="predict_proba", cv=KFold(5, shuffle=False)),
        }
    else:
        rf = RandomForestRegressor(criterion="squared_error", **rf_common)
        xgb = XGBRegressor(objective="reg:squarederror", **xgb_common)
        all_models = {
            "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42),
            "Logistic / Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": rf,
            "XGBoost": xgb,
            "Soft Voting (RF + XGB)": VotingRegressor([("rf", rf), ("xgb", xgb)]),
            "Stacking (RF + XGB)": StackingRegressor([("rf", rf), ("xgb", xgb)], final_estimator=Ridge(alpha=1.0),
                                                     cv=KFold(5, shuffle=False)),
        }
    return {name: all_models[name] for name in chosen}


with tab_train:
    st.markdown(
        "Upload a CSV (or use the dengue modelling table), choose what to predict and which models to train, "
        "then compare them on held-out data. Single models (weak learners) are listed first, then the ensembles."
    )
    source = st.radio("Data", ["Dengue modelling table (data/model_ready.csv)", "Upload a CSV"], horizontal=True)
    data = None
    if source.startswith("Dengue"):
        if os.path.exists(DATA_PATH):
            data = read_csv(DATA_PATH)
        else:
            st.warning("data/model_ready.csv not found. Run the notebook first, or upload a CSV.")
    else:
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded is not None:
            data = read_csv(uploaded)

    if data is not None:
        st.write(f"**{data.shape[0]:,} rows, {data.shape[1]} columns.** First rows:")
        st.dataframe(data.head(), width="stretch")
        nulls = data.isna().sum()
        if nulls.sum() > 0:
            with st.expander(f"Null values found in {int((nulls > 0).sum())} columns (median / most-frequent imputation is applied)"):
                st.dataframe(nulls[nulls > 0].rename("missing").to_frame())

        c1, c2 = st.columns(2)
        with c1:
            default_target = "zone_name" if "zone_name" in data.columns else data.columns[-1]
            target = st.selectbox("Target to predict", list(data.columns), index=list(data.columns).index(default_target))
            guess = "Regression" if pd.api.types.is_numeric_dtype(data[target]) and data[target].nunique() > 15 else "Classification"
            task = st.radio("Task", ["Classification", "Regression"], index=["Classification", "Regression"].index(guess), horizontal=True)
        with c2:
            exclude = {target} | {t for t in KNOWN_TARGETS if t in data.columns} | {c for c in ["year", "month"] if c in data.columns}
            default_features = [c for c in data.columns if c not in exclude]
            features = st.multiselect("Features", [c for c in data.columns if c != target], default=default_features)

        if "year" in data.columns:
            years = sorted(data["year"].dropna().unique().astype(int))
            split_mode = st.radio("Train/test split", ["By time (train on earlier years)", "Random 80/20"], horizontal=True)
            last_train_year = st.select_slider("Last training year", options=years[:-1],
                                               value=2023 if 2023 in years[:-1] else years[-2]) if split_mode.startswith("By time") else None
        else:
            split_mode, last_train_year = "Random 80/20", None

        chosen = st.multiselect("Models", MODEL_CHOICES, default=["Decision Tree", "Logistic / Ridge Regression",
                                                                  "Random Forest", "XGBoost", "Soft Voting (RF + XGB)"])

        with st.expander("Random Forest and XGBoost settings (defaults are the values designed in the notebook)"):
            h1, h2 = st.columns(2)
            with h1:
                st.markdown("**Random Forest**")
                rf_cfg = {
                    "n_estimators": st.slider("n_estimators (trees)", 50, 1000, 500, 50),
                    "max_depth": st.slider("max_depth", 2, 30, 10),
                    "min_samples_leaf": st.slider("min_samples_leaf", 1, 30, 5),
                    "max_features": st.select_slider("max_features (m, share of columns per split)",
                                                     options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], value=0.4),
                }
            with h2:
                st.markdown("**XGBoost**")
                xgb_cfg = {
                    "n_estimators": st.slider("n_estimators (boosting rounds)", 50, 1000, 143, 1),
                    "learning_rate": st.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3], value=0.03),
                    "max_depth": st.slider("max_depth ", 2, 10, 4),
                    "reg_lambda": st.select_slider("reg_lambda (L2)", options=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0], value=2.0),
                    "subsample": st.select_slider("subsample", options=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], value=0.8),
                }

        if st.button("Train and compare", type="primary", disabled=not (features and chosen)):
            frame = data.dropna(subset=[target])
            X, y = frame[features], frame[target]
            categorical = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
            numeric = [c for c in features if c not in categorical]

            label_encoder = None
            if task == "Classification":
                label_encoder = LabelEncoder().fit(y.astype(str))
                y = pd.Series(label_encoder.transform(y.astype(str)), index=y.index)
            elif not pd.api.types.is_numeric_dtype(y):
                st.error("Regression needs a numeric target.")
                st.stop()

            if last_train_year is not None:
                train_mask = frame["year"] <= last_train_year
                X_tr, X_te, y_tr, y_te = X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
            else:
                stratify = y if task == "Classification" and y.value_counts().min() >= 2 else None
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

            n_classes = y.nunique() if task == "Classification" else 0
            models = make_models(task, chosen, rf_cfg, xgb_cfg, n_classes)
            rows, fitted = [], {}
            progress = st.progress(0.0, text="Training...")
            for i, (name, model) in enumerate(models.items(), start=1):
                pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", model)]).fit(X_tr, y_tr)
                fitted[name] = pipe
                pred = pipe.predict(X_te)
                if task == "Classification":
                    rows.append({"Model": name,
                                 "Train accuracy": accuracy_score(y_tr, pipe.predict(X_tr)),
                                 "Test accuracy": accuracy_score(y_te, pred),
                                 "Balanced accuracy": balanced_accuracy_score(y_te, pred),
                                 "Macro F1": f1_score(y_te, pred, average="macro")})
                else:
                    rows.append({"Model": name,
                                 "Train R2": r2_score(y_tr, pipe.predict(X_tr)),
                                 "Test MAE": mean_absolute_error(y_te, pred),
                                 "Test RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                                 "Test R2": r2_score(y_te, pred)})
                progress.progress(i / len(models), text=f"Trained {name}")
            progress.empty()

            results = pd.DataFrame(rows).set_index("Model")
            st.session_state["train_output"] = (task, results, fitted, X_te, y_te, label_encoder, len(X_tr), len(X_te))

        if "train_output" in st.session_state:
            task_done, results, fitted, X_te, y_te, label_encoder, n_tr, n_te = st.session_state["train_output"]
            st.subheader("Results")
            st.caption(f"Trained on {n_tr:,} rows, tested on {n_te:,} rows.")
            st.dataframe(results.style.format("{:.3f}"), width="stretch")
            metric = "Macro F1" if task_done == "Classification" else "Test MAE"
            st.markdown(f"**{metric} by model** ({'higher' if task_done == 'Classification' else 'lower'} is better)")
            st.bar_chart(results[metric])

            g1, g2 = st.columns(2)
            with g1:
                if task_done == "Classification":
                    pick = st.selectbox("Confusion matrix for", list(fitted))
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(confusion_matrix(y_te, fitted[pick].predict(X_te)),
                                           display_labels=label_encoder.classes_).plot(ax=ax, colorbar=False, cmap="Blues")
                    ax.set_title(pick)
                    st.pyplot(fig)
            with g2:
                if "Random Forest" in fitted:
                    rf_pipe = fitted["Random Forest"]
                    names = [n.split("__", 1)[1] for n in rf_pipe.named_steps["prep"].get_feature_names_out()]
                    imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names)
                    imp = imp.sort_values(ascending=False).head(15)
                    st.markdown("**Random Forest feature importance (top 15)**")
                    st.bar_chart(imp)

# ------------------------------------------------------------------ tab 3: notebook results
with tab_results:
    results = load_results()
    if results is None:
        st.warning("models/results.json not found. Run the notebook first.")
    else:
        st.markdown("Results from the notebook (trained on 2015-2023, tested on 2024-2026).")
        sections = [
            ("Risk zone: cross-validation on yearly folds (used for choosing)", "zone_cv"),
            ("Risk zone: test years", "zone_test"),
            ("Epidemic alarm: test years", "epidemic_test"),
            ("Case count: test years", "cases_test"),
            ("Comparison with the published approach (Fan et al., 2025)", "paper_comparison"),
        ]
        for title, key in sections:
            st.subheader(title)
            st.dataframe(pd.DataFrame(results[key]).set_index("Model"), width="stretch")
        st.subheader("Random Forest feature importance")
        st.bar_chart(pd.Series(results["rf_importance"]))
