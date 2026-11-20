from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.train import build_ohe_pipeline


TARGET = "battery_failure"
ID_COLUMNS = {"vehicle_id", "battery_serial"}
DERIVED_COLUMNS = {
    "predicted_remaining_life_cycles",
    "thermal_runaway_risk",
    "battery_stress_index",
    "aging_score",
    "thermal_health_score",
    "charging_quality_score",
    "driving_stress_score",
}
AMBIGUOUS_COLUMNS = {
    "previous_faults",
    "BMS_warning_count",
    "sensor_fault_count",
    "abnormal_voltage_events",
    "state_of_health",
    "battery_health_percent",
    "capacity_loss_percent",
    "remaining_capacity",
}
SEED = 42
EXPERIMENT_COLUMNS = [
    "model",
    "parameters",
    "threshold_selected_oof",
    "cv_f1_default_mean",
    "cv_f1_default_std",
    "cv_oof_f1_tuned",
    "accuracy_test_default",
    "precision_test_default",
    "recall_test_default",
    "f1_test_default",
    "accuracy_test_tuned",
    "precision_test_tuned",
    "recall_test_tuned",
    "f1_test_tuned",
    "roc_auc_test",
    "train_f1_tuned",
    "train_test_f1_gap",
    "train_seconds",
    "inference_ms_per_row",
    "cv_seconds",
    "fit_diagnosis",
]


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    index = int(np.argmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])


def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def evaluate_scenario(
    name: str,
    features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, object]:
    X_train_scenario = X_train[features]
    X_test_scenario = X_test[features]
    numeric_columns = X_train_scenario.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features if column not in numeric_columns]
    estimator = LogisticRegression(
        class_weight=None,
        C=0.1,
        max_iter=1500,
        solver="lbfgs",
        random_state=SEED,
    )
    pipeline = build_ohe_pipeline(
        estimator,
        numeric_columns,
        categorical_columns,
        scale_numeric=True,
    )

    cv_started = time.perf_counter()
    cv_scores = cross_val_score(
        pipeline, X_train_scenario, y_train, cv=cv, scoring="f1", n_jobs=1
    )
    cv_seconds = time.perf_counter() - cv_started
    oof_probabilities = cross_val_predict(
        pipeline,
        X_train_scenario,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    # Each cloned pipeline uses classes [0, 1], so the positive score is column 1.
    oof_scores = oof_probabilities[:, 1]
    threshold, oof_f1 = select_f1_threshold(y_train.to_numpy(), oof_scores)

    fit_started = time.perf_counter()
    pipeline.fit(X_train_scenario, y_train)
    train_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    test_scores = positive_scores(pipeline, X_test_scenario)
    inference_ms_per_row = (
        (time.perf_counter() - inference_started) * 1000 / len(X_test_scenario)
    )
    default_predictions = (test_scores >= 0.5).astype(int)
    tuned_predictions = (test_scores >= threshold).astype(int)
    train_scores = positive_scores(pipeline, X_train_scenario)
    train_f1 = float(
        f1_score(y_train, (train_scores >= threshold).astype(int), zero_division=0)
    )
    test_f1 = float(f1_score(y_test, tuned_predictions, zero_division=0))
    gap = train_f1 - test_f1
    diagnosis = "brecha baja" if gap <= 0.05 else ("brecha moderada" if gap <= 0.15 else "posible sobreajuste")
    matrix = confusion_matrix(y_test, tuned_predictions, labels=[0, 1]).tolist()

    print(
        f"[RESULT] {name}: features={len(features)}, threshold OOF={threshold:.6f}, "
        f"F1 OOF={oof_f1:.4f}, test precision={precision_score(y_test, tuned_predictions, zero_division=0):.4f}, "
        f"recall={recall_score(y_test, tuned_predictions, zero_division=0):.4f}, "
        f"F1={test_f1:.4f}, AUC={roc_auc_score(y_test, test_scores):.4f}, CM={matrix}"
    )

    return {
        "model": name,
        "parameters": json.dumps(estimator.get_params(), default=str),
        "threshold_selected_oof": threshold,
        "cv_f1_default_mean": float(cv_scores.mean()),
        "cv_f1_default_std": float(cv_scores.std()),
        "cv_oof_f1_tuned": oof_f1,
        "accuracy_test_default": float(accuracy_score(y_test, default_predictions)),
        "precision_test_default": float(precision_score(y_test, default_predictions, zero_division=0)),
        "recall_test_default": float(recall_score(y_test, default_predictions, zero_division=0)),
        "f1_test_default": float(f1_score(y_test, default_predictions, zero_division=0)),
        "accuracy_test_tuned": float(accuracy_score(y_test, tuned_predictions)),
        "precision_test_tuned": float(precision_score(y_test, tuned_predictions, zero_division=0)),
        "recall_test_tuned": float(recall_score(y_test, tuned_predictions, zero_division=0)),
        "f1_test_tuned": test_f1,
        "roc_auc_test": float(roc_auc_score(y_test, test_scores)),
        "train_f1_tuned": train_f1,
        "train_test_f1_gap": gap,
        "train_seconds": train_seconds,
        "inference_ms_per_row": inference_ms_per_row,
        "cv_seconds": cv_seconds,
        "fit_diagnosis": diagnosis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara una ablación completa y otra conservadora sin cambiar el split."
    )
    parser.add_argument("--data", default="data/processed/ev_battery_failure_clean.csv")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    data_path = (ROOT / args.data).resolve()
    output_root = (ROOT / args.output).resolve()
    results_path = output_root / "results" / "experiments.csv"
    output_root.joinpath("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False).dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    excluded = ID_COLUMNS | DERIVED_COLUMNS | {TARGET}
    all_features = [column for column in df.columns if column not in excluded]
    missing_ambiguous = AMBIGUOUS_COLUMNS - set(all_features)
    if missing_ambiguous:
        raise ValueError(f"No se encontraron las columnas de ablación: {sorted(missing_ambiguous)}")
    conservative_features = [column for column in all_features if column not in AMBIGUOUS_COLUMNS]

    X, y = df[all_features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(
        f"[INFO] split estratificado seed={SEED}: train={len(X_train):,}, "
        f"test={len(X_test):,}; test positives={int(y_test.sum())}"
    )
    print(f"[INFO] ambiguas excluidas solo en escenario conservador: {sorted(AMBIGUOUS_COLUMNS)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = [
        evaluate_scenario(
            "Ablation Logistic Regression C=0.1 (60 complete features)",
            all_features,
            X_train,
            X_test,
            y_train,
            y_test,
            cv,
        ),
        evaluate_scenario(
            "Ablation Logistic Regression C=0.1 (52 conservative features)",
            conservative_features,
            X_train,
            X_test,
            y_train,
            y_test,
            cv,
        ),
    ]

    new_rows = pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS)
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[~existing["model"].isin(new_rows["model"])]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(results_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Experimentos de ablación registrados en {results_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
