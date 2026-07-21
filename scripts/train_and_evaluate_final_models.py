#!/usr/bin/env python3
"""Fit the locked ADR1D parameter models and evaluate the test split once."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DATA = ROOT / "data"
BASE_TABLE_PATH = DATA / "adr1d_modeling_table.csv"
DECAY_TABLE_PATH = DATA / "adr1d_decay_detectability_table.csv"
BASELINE_SUMMARY_PATH = RESULTS / "baseline_validation_summary.json"
DECAY_SUMMARY_PATH = RESULTS / "decay_detectability_validation_summary.json"
PROTOCOL_PATH = RESULTS / "final_model_protocol.json"
PREDICTIONS_PATH = RESULTS / "final_test_predictions.csv"
METRICS_PATH = RESULTS / "final_test_metrics.json"
MODEL_PATH = MODELS / "adr1d_parameter_models.joblib"
MANIFEST_PATH = MODELS / "model_manifest.json"

EXPECTED_PROTOCOL_SHA256 = (
    "56555a235dd6610a5bd3d6376cbe1123490fc50d28b6a0a0c0e8a0c342fdc2d3"
)


@contextmanager
def guarded_linear_algebra() -> Iterable[None]:
    """Suppress known matrix warnings while retaining finite-value checks."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            yield


def require_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise FloatingPointError(f"Non-finite values found in {label}")


def validate_fitted_estimator(estimator: Any, label: str) -> None:
    model = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    for attribute in ("coef_", "intercept_", "feature_importances_"):
        if hasattr(model, attribute):
            require_finite(np.asarray(getattr(model, attribute)), label + " " + attribute)


def regression_metrics(
    actual_log: np.ndarray,
    predicted_log: np.ndarray,
    actual_physical: np.ndarray,
    predicted_physical: np.ndarray,
) -> dict[str, float]:
    absolute_percentage = np.abs(predicted_physical - actual_physical) / np.abs(
        actual_physical
    )
    return {
        "rmse_log10": float(math.sqrt(mean_squared_error(actual_log, predicted_log))),
        "mae_log10": float(mean_absolute_error(actual_log, predicted_log)),
        "r2_log10": float(r2_score(actual_log, predicted_log)),
        "rmse_physical": float(
            math.sqrt(mean_squared_error(actual_physical, predicted_physical))
        ),
        "mae_physical": float(mean_absolute_error(actual_physical, predicted_physical)),
        "r2_physical": float(r2_score(actual_physical, predicted_physical)),
        "mean_absolute_percentage_error": float(absolute_percentage.mean()),
        "median_absolute_percentage_error": float(np.median(absolute_percentage)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol(protocol: dict[str, Any]) -> None:
    if sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("The locked final protocol has changed")
    paths = {
        "adr1d_modeling_table.csv": BASE_TABLE_PATH,
        "adr1d_decay_detectability_table.csv": DECAY_TABLE_PATH,
        "baseline_validation_summary.json": BASELINE_SUMMARY_PATH,
        "decay_detectability_validation_summary.json": DECAY_SUMMARY_PATH,
    }
    for name, path in paths.items():
        expected = protocol["input_artifacts"][name]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Locked input changed: {name}")


def tree_regression_pipeline(parameters: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(**parameters)),
        ]
    )


def classifier_pipeline(parameters: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**parameters)),
        ]
    )


def by_regime_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for regime, frame in predictions.groupby("diagnostic_regime"):
        velocity_ape = np.abs(
            frame["predicted_effective_velocity_m_s"]
            - frame["actual_effective_velocity_m_s"]
        ) / frame["actual_effective_velocity_m_s"]
        dispersion_ape = np.abs(
            frame["predicted_effective_dispersion_m2_s"]
            - frame["actual_effective_dispersion_m2_s"]
        ) / frame["actual_effective_dispersion_m2_s"]
        result[str(regime)] = {
            "rows": int(len(frame)),
            "velocity_median_absolute_percentage_error": float(
                np.median(velocity_ape)
            ),
            "dispersion_median_absolute_percentage_error": float(
                np.median(dispersion_ape)
            ),
        }
    return result


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    verify_protocol(protocol)
    base = pd.read_csv(BASE_TABLE_PATH)
    decay = pd.read_csv(DECAY_TABLE_PATH)

    if not base[["scenario_id", "split"]].equals(
        decay[["scenario_id", "split"]]
    ):
        raise RuntimeError("Base and decay tables are not aligned")
    development_mask = base["split"].isin(protocol["development_splits"])
    test_mask = base["split"].eq(protocol["test_split"])
    if int(development_mask.sum()) != 255 or int(test_mask.sum()) != 45:
        raise RuntimeError("Unexpected development or test row count")

    base_features = [name for name in base if name.startswith("feature_")]
    decay_features = [name for name in decay if name.startswith("feature_")]
    compact_features = protocol["models"]["decay_rate_resolvable"][
        "feature_columns"
    ]
    if len(base_features) != 69 or len(decay_features) != 86:
        raise RuntimeError("Locked feature counts do not match")

    velocity_model = tree_regression_pipeline(
        protocol["models"]["effective_velocity"]["parameters"]
    )
    dispersion_model = tree_regression_pipeline(
        protocol["models"]["effective_dispersion"]["parameters"]
    )
    decay_classifier = classifier_pipeline(
        protocol["models"]["decay_resolvability"]["parameters"]
    )
    decay_regressor = tree_regression_pipeline(
        protocol["models"]["decay_rate_resolvable"]["parameters"]
    )

    development_base = base.loc[development_mask]
    test_base = base.loc[test_mask]
    development_decay = decay.loc[development_mask]
    test_decay = decay.loc[test_mask]
    resolvable_development = development_decay.loc[
        development_decay["target_decay_resolvable"] == 1
    ]

    with guarded_linear_algebra():
        velocity_model.fit(
            development_base[base_features],
            development_base["target_log10_effective_velocity"],
        )
        dispersion_model.fit(
            development_base[base_features],
            development_base["target_log10_effective_dispersion"],
        )
        decay_classifier.fit(
            development_decay[decay_features],
            development_decay["target_decay_resolvable"],
        )
        decay_regressor.fit(
            resolvable_development[compact_features],
            resolvable_development["target_log10_decay_rate_resolvable"],
        )

        predicted_velocity_log = velocity_model.predict(test_base[base_features])
        predicted_dispersion_log = dispersion_model.predict(
            test_base[base_features]
        )
        decay_probabilities = decay_classifier.predict_proba(
            test_decay[decay_features]
        )
        predicted_decay_log = decay_regressor.predict(
            test_decay[compact_features]
        )

    for name, estimator in {
        "effective velocity": velocity_model,
        "effective dispersion": dispersion_model,
        "decay resolvability": decay_classifier,
        "resolvable decay rate": decay_regressor,
    }.items():
        validate_fitted_estimator(estimator, name)

    predicted_velocity = np.power(10.0, predicted_velocity_log)
    predicted_dispersion = np.power(10.0, predicted_dispersion_log)
    predicted_decay_if_resolvable = np.power(10.0, predicted_decay_log)
    positive_index = int(np.where(decay_classifier.classes_ == 1)[0][0])
    decay_probability = decay_probabilities[:, positive_index]
    decision_threshold = float(
        protocol["models"]["decay_resolvability"]["decision_threshold"]
    )
    predicted_resolvable = (decay_probability >= decision_threshold).astype(int)

    for label, values in {
        "velocity predictions": predicted_velocity,
        "dispersion predictions": predicted_dispersion,
        "decay probabilities": decay_probability,
        "decay-rate predictions": predicted_decay_if_resolvable,
    }.items():
        require_finite(values, label)

    predictions = test_decay[
        [
            "scenario_id",
            "split",
            "diagnostic_regime",
            "diagnostic_decay_state",
            "diagnostic_damkohler_number",
        ]
    ].copy()
    predictions["actual_effective_velocity_m_s"] = test_base[
        "target_effective_velocity_m_s"
    ].to_numpy(dtype=float)
    predictions["predicted_effective_velocity_m_s"] = predicted_velocity
    predictions["actual_effective_dispersion_m2_s"] = test_base[
        "target_effective_dispersion_m2_s"
    ].to_numpy(dtype=float)
    predictions["predicted_effective_dispersion_m2_s"] = predicted_dispersion
    predictions["actual_decay_resolvable"] = test_decay[
        "target_decay_resolvable"
    ].to_numpy(dtype=int)
    predictions["predicted_decay_resolvable"] = predicted_resolvable
    predictions["predicted_decay_resolvable_probability"] = decay_probability
    predictions["actual_decay_rate_s_1"] = test_decay[
        "target_decay_rate_s_1"
    ].to_numpy(dtype=float)
    predictions["predicted_decay_rate_if_resolvable_s_1"] = (
        predicted_decay_if_resolvable
    )
    predictions["reported_decay_rate_s_1"] = np.where(
        predicted_resolvable == 1,
        predicted_decay_if_resolvable,
        np.nan,
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False, float_format="%.12g")

    velocity_scores = regression_metrics(
        test_base["target_log10_effective_velocity"].to_numpy(dtype=float),
        predicted_velocity_log,
        test_base["target_effective_velocity_m_s"].to_numpy(dtype=float),
        predicted_velocity,
    )
    dispersion_scores = regression_metrics(
        test_base["target_log10_effective_dispersion"].to_numpy(dtype=float),
        predicted_dispersion_log,
        test_base["target_effective_dispersion_m2_s"].to_numpy(dtype=float),
        predicted_dispersion,
    )

    actual_resolvable = predictions["actual_decay_resolvable"].to_numpy(dtype=int)
    classification_scores = {
        "accuracy": float(accuracy_score(actual_resolvable, predicted_resolvable)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_resolvable, predicted_resolvable)
        ),
        "precision": float(
            precision_score(actual_resolvable, predicted_resolvable, zero_division=0)
        ),
        "recall": float(
            recall_score(actual_resolvable, predicted_resolvable, zero_division=0)
        ),
        "f1": float(
            f1_score(actual_resolvable, predicted_resolvable, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(actual_resolvable, decay_probability)),
        "log_loss": float(log_loss(actual_resolvable, decay_probabilities)),
        "decision_threshold": decision_threshold,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            actual_resolvable, predicted_resolvable
        ).ravel().astype(int).tolist(),
    }

    resolvable_mask = actual_resolvable == 1
    actual_decay = predictions.loc[
        resolvable_mask, "actual_decay_rate_s_1"
    ].to_numpy(dtype=float)
    predicted_decay = predicted_decay_if_resolvable[resolvable_mask]
    actual_decay_log = np.log10(actual_decay)
    predicted_decay_log_resolvable = np.log10(predicted_decay)
    decay_ape = np.abs(predicted_decay - actual_decay) / actual_decay
    decay_regression_scores = {
        "rows": int(resolvable_mask.sum()),
        "rmse_log10": float(
            math.sqrt(
                mean_squared_error(
                    actual_decay_log, predicted_decay_log_resolvable
                )
            )
        ),
        "mae_log10": float(
            mean_absolute_error(actual_decay_log, predicted_decay_log_resolvable)
        ),
        "r2_log10": float(
            r2_score(actual_decay_log, predicted_decay_log_resolvable)
        ),
        "rmse_physical": float(
            math.sqrt(mean_squared_error(actual_decay, predicted_decay))
        ),
        "mae_physical": float(mean_absolute_error(actual_decay, predicted_decay)),
        "median_absolute_percentage_error": float(np.median(decay_ape)),
        "mean_absolute_percentage_error": float(decay_ape.mean()),
    }

    metrics = {
        "status": "final_test_complete",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "development_rows": int(development_mask.sum()),
        "development_resolvable_decay_rows": int(len(resolvable_development)),
        "test_rows": int(test_mask.sum()),
        "test_decay_state_counts": {
            key: int(value)
            for key, value in predictions[
                "diagnostic_decay_state"
            ].value_counts().to_dict().items()
        },
        "effective_velocity": velocity_scores,
        "effective_dispersion": dispersion_scores,
        "decay_resolvability": classification_scores,
        "decay_rate_conditional_on_resolvable": decay_regression_scores,
        "by_regime": by_regime_metrics(predictions),
        "post_test_tuning_performed": False,
        "software": {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle = {
        "bundle_version": "1.0.0",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "models": {
            "effective_velocity": velocity_model,
            "effective_dispersion": dispersion_model,
            "decay_resolvability": decay_classifier,
            "decay_rate_resolvable": decay_regressor,
        },
        "feature_columns": {
            "effective_parameters": base_features,
            "decay_resolvability": decay_features,
            "decay_rate_resolvable": compact_features,
        },
        "decision_threshold": decision_threshold,
        "target_contract": {
            "effective_velocity": "v/R in m/s",
            "effective_dispersion": "D/R in m2/s",
            "decay_resolvability": "1 when 1-exp(-Da) is at least 0.03",
            "decay_rate_resolvable": "lambda in 1/s, reported only when resolvable",
        },
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    manifest = {
        "bundle_version": "1.0.0",
        "model_file": MODEL_PATH.name,
        "model_sha256": sha256(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "protocol_file": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "predictions_sha256": sha256(PREDICTIONS_PATH),
        "metrics_sha256": sha256(METRICS_PATH),
        "training_script_sha256": sha256(Path(__file__)),
        "development_rows": int(development_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "post_test_tuning_performed": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
