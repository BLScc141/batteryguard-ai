from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TARGET = "battery_failure"
ID_COLUMNS = ["vehicle_id", "battery_serial"]
LEAKAGE_COLUMNS = [
    "predicted_remaining_life_cycles", "thermal_runaway_risk",
    "battery_stress_index", "aging_score", "thermal_health_score",
    "charging_quality_score", "driving_stress_score",
]
SEED = 42

def build_ohe_pipeline(model, numeric_cols, categorical_cols, scale_numeric=False):
    num = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num.append(("scaler", StandardScaler()))
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", Pipeline(num), numeric_cols),
        ("categorical", cat, categorical_cols),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])

def build_smotenc_pipeline(model, numeric_cols, categorical_cols):
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline

    cat_encoder = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("categorical", cat_encoder, categorical_cols),
    ], sparse_threshold=0.0)
    cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
    sampler = SMOTENC(categorical_features=cat_indices, random_state=SEED)
    post = ColumnTransformer([
        ("numeric", "passthrough", list(range(len(numeric_cols)))),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_indices),
    ])
    return ImbPipeline([
        ("preprocess", prep),
        ("smotenc", sampler),
        ("postprocess", post),
        ("model", model),
    ])

def positive_probability(model, X):
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]

def select_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities, pos_label=1)
    if len(thresholds) == 0:
        return 0.5, 0.0
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])

def metric_row(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_positive": float(precision_score(y, predictions, zero_division=0)),
        "recall_positive": float(recall_score(y, predictions, zero_division=0)),
        "f1_positive": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "predictions": predictions,
    }

def main():
    parser = argparse.ArgumentParser(description="Compara modelos para battery_failure.")
    parser.add_argument("--data", default="data/processed/ev_battery_failure_clean.csv")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    data_path = (ROOT / args.data).resolve()
    out_root = (ROOT / args.output).resolve()
    models_dir, results_dir = out_root / "models", out_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando dataset: {args.data}")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("battery_failure debe contener exactamente las clases 0 y 1")
    print(f"[INFO] Dataset: {len(df):,} filas, {len(df.columns)} columnas")
    print(f"[INFO] Clases: {df[TARGET].value_counts().sort_index().to_dict()}")

    excluded = set([TARGET, *ID_COLUMNS, *LEAKAGE_COLUMNS]).intersection(df.columns)
    features = [c for c in df.columns if c not in excluded]
    X, y = df[features].copy(), df[TARGET].copy()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in features if c not in numeric_cols]
    print(f"[INFO] Predictores: {len(features)} ({len(numeric_cols)} numéricos, {len(categorical_cols)} categóricos)")
    print(f"[INFO] Excluidas: {sorted(excluded)}")

    row_index = np.arange(len(y))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, row_index, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"[INFO] Split estratificado: train={len(X_train):,} test={len(X_test):,}; positivos test={int(y_test.sum())}, negativos test={int(len(y_test)-y_test.sum())}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    rf_regularized = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20, max_features="sqrt",
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    rf_smotenc = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20, max_features="sqrt",
        class_weight=None, random_state=SEED, n_jobs=-1,
    )
    specs = [
        ("Regresión Logística balanceada", LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1500, solver="lbfgs", random_state=SEED
        ), "ohe_scaled"),
        ("Random Forest balanceado", rf_base, "ohe"),
        ("Random Forest regularizado", rf_regularized, "ohe"),
        ("Random Forest regularizado + SMOTENC", rf_smotenc, "smotenc"),
    ]

    records, models, thresholds, oof_scores = [], {}, {}, {}
    for name, estimator, kind in specs:
        pipe = (
            build_smotenc_pipeline(estimator, numeric_cols, categorical_cols)
            if kind == "smotenc"
            else build_ohe_pipeline(estimator, numeric_cols, categorical_cols, scale_numeric=(kind == "ohe_scaled"))
        )
        print(f"[INFO] Validación 5-fold: {name}")
        started = time.perf_counter()
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_seconds = time.perf_counter() - started
        oof_prob = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)
        oof_prob = oof_prob[:, 1]
        threshold, oof_tuned_f1 = select_f1_threshold(y_train.to_numpy(), oof_prob)
        oof_scores[name] = oof_tuned_f1
        thresholds[name] = threshold

        fit_started = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started

        test_started = time.perf_counter()
        test_prob = positive_probability(pipe, X_test)
        infer_ms_per_row = (time.perf_counter() - test_started) * 1000 / len(X_test)
        test_default = metric_row(y_test, test_prob, 0.5)
        test_tuned = metric_row(y_test, test_prob, threshold)
        train_prob = positive_probability(pipe, X_train)
        train_tuned_f1 = float(f1_score(y_train, (train_prob >= threshold).astype(int), zero_division=0))
        gap = train_tuned_f1 - test_tuned["f1_positive"]
        diagnosis = "brecha baja" if gap <= 0.05 else ("brecha moderada" if gap <= 0.15 else "posible sobreajuste")
        models[name] = pipe
        records.append({
            "model": name,
            "parameters": json.dumps(estimator.get_params(), default=str),
            "threshold_selected_oof": threshold,
            "cv_f1_default_mean": float(cv_scores.mean()),
            "cv_f1_default_std": float(cv_scores.std()),
            "cv_oof_f1_tuned": oof_tuned_f1,
            "accuracy_test_default": test_default["accuracy"],
            "precision_test_default": test_default["precision_positive"],
            "recall_test_default": test_default["recall_positive"],
            "f1_test_default": test_default["f1_positive"],
            "accuracy_test_tuned": test_tuned["accuracy"],
            "precision_test_tuned": test_tuned["precision_positive"],
            "recall_test_tuned": test_tuned["recall_positive"],
            "f1_test_tuned": test_tuned["f1_positive"],
            "roc_auc_test": test_tuned["roc_auc"],
            "train_f1_tuned": train_tuned_f1,
            "train_test_f1_gap": gap,
            "train_seconds": fit_seconds,
            "inference_ms_per_row": infer_ms_per_row,
            "cv_seconds": cv_seconds,
            "fit_diagnosis": diagnosis,
        })
        print(f"[RESULT] {name}: CV F1@0.5={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}); "
              f"OOF threshold={threshold:.4f}, OOF F1={oof_tuned_f1:.4f}; "
              f"test F1@0.5={test_default['f1_positive']:.4f}, test F1@threshold={test_tuned['f1_positive']:.4f}, "
              f"precision={test_tuned['precision_positive']:.4f}, recall={test_tuned['recall_positive']:.4f}; "
              f"train={fit_seconds:.2f}s, inference={infer_ms_per_row:.4f}ms/row, {diagnosis}")

    results = pd.DataFrame(records)
    results.to_csv(results_dir / "experiments.csv", index=False, encoding="utf-8")
    selected_name = max(oof_scores, key=oof_scores.get)
    selected, selected_threshold = models[selected_name], thresholds[selected_name]
    test_prob = positive_probability(selected, X_test)
    selected_default = metric_row(y_test, test_prob, 0.5)
    selected_tuned = metric_row(y_test, test_prob, selected_threshold)
    cm = confusion_matrix(y_test, selected_tuned["predictions"], labels=[0, 1])
    pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(results_dir / "confusion_matrix.csv")
    report = classification_report(y_test, selected_tuned["predictions"], output_dict=True, zero_division=0)
    (results_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    baseline = {
        "accuracy_test": float(accuracy_score(y_test, dummy_pred)),
        "precision_positive_test": float(precision_score(y_test, dummy_pred, zero_division=0)),
        "recall_positive_test": float(recall_score(y_test, dummy_pred, zero_division=0)),
        "f1_positive_test": float(f1_score(y_test, dummy_pred, zero_division=0)),
    }
    (results_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    joblib.dump(selected, models_dir / "modelo_candidato.joblib")
    if "preprocess" in selected.named_steps:
        preprocessor = selected.named_steps["preprocess"]
    else:
        preprocessor = selected.named_steps["preprocess"]
    joblib.dump(preprocessor, models_dir / "preprocesador.joblib")

    last_transform = selected.named_steps.get("postprocess", preprocessor)
    feature_names = last_transform.get_feature_names_out().tolist()
    model_step = selected.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        values = model_step.feature_importances_
        method = "feature_importances_"
    else:
        values = np.abs(model_step.coef_[0])
        method = "absolute_coefficient_not_comparable_across_scales"
    pd.DataFrame({"feature": feature_names, "value": values, "method": method}).sort_values(
        "value", ascending=False
    ).to_csv(results_dir / "feature_importance.csv", index=False, encoding="utf-8")

    raw = pd.read_csv(data_path, low_memory=False)
    train_ids = raw.iloc[train_idx]["vehicle_id"].astype(str).head(12).tolist()
    presets = {}
    for pid in train_ids:
        row = raw.loc[raw["vehicle_id"].astype(str).eq(pid)].iloc[0]
        presets[pid] = {
            col: (None if pd.isna(row[col]) else row[col].item() if hasattr(row[col], "item") else row[col])
            for col in features
        }
    (models_dir / "presets.json").write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "target": TARGET, "features": features, "numeric_features": numeric_cols,
        "categorical_features": categorical_cols, "excluded_columns": sorted(excluded),
        "selected_model": selected_name, "threshold": selected_threshold,
        "threshold_selection": "max F1 from 5-fold out-of-fold probabilities within train; test untouched",
        "random_state": SEED, "train_rows": len(X_train), "test_rows": len(X_test),
        "test_positive": int(y_test.sum()), "test_negative": int(len(y_test) - y_test.sum()),
        "risk_level_note": "Binary classifier output; mapping to 3 operational levels remains an E4 calibration task.",
    }
    (models_dir / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    thresholds_out = results[["model", "threshold_selected_oof", "cv_oof_f1_tuned", "f1_test_default", "f1_test_tuned"]]
    thresholds_out.to_csv(results_dir / "threshold_comparison.csv", index=False, encoding="utf-8")
    summary = {
        "selected_model_by_oof_f1": selected_name,
        "selected_threshold": selected_threshold,
        "test_default_threshold_metrics": {k: v for k, v in selected_default.items() if k != "predictions"},
        "test_selected_threshold_metrics": {k: v for k, v in selected_tuned.items() if k != "predictions"},
        "baseline_test": baseline, "test_confusion_matrix": cm.tolist(),
        "random_state": SEED, "split": "80/20 stratified; test untouched for threshold selection",
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Selección por F1 OOF: {selected_name}; threshold={selected_threshold:.6f}")
    print(f"[INFO] Test threshold seleccionado: F1={selected_tuned['f1_positive']:.4f}, "
          f"precision={selected_tuned['precision_positive']:.4f}, recall={selected_tuned['recall_positive']:.4f}")
    print(f"[INFO] Baseline test: {baseline}")
    print(f"[INFO] Matriz de confusión final: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print("[SUCCESS] Modelo serializado: models/modelo_candidato.joblib")
    print("[SUCCESS] Preprocesador serializado: models/preprocesador.joblib")
    print("[SUCCESS] Esquema y presets: models/schema.json; models/presets.json")
    print("[SUCCESS] Métricas/logs: results/")

if __name__ == "__main__":
    main()



