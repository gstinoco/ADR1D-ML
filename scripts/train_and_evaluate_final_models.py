"""
================================================================================
ADR1D-ML: Locked Model Training and Evaluation
================================================================================

This module reconstructs the four ADR1D-ML pipelines from the protocol fixed
before final testing. It verifies every locked input, fits the models on the
combined training and validation scenarios, evaluates the reserved test split,
and serializes the model bundle and reproducibility artifacts.

Main Operations
---------------
1. Verify the locked protocol and all input SHA-256 digests.
2. Fit effective-velocity, effective-dispersion, and decay pipelines.
3. Compute the unchanged final-test metrics and scenario predictions.
4. Serialize the bundle and write its machine-readable manifest.

Authors
-------
Gerardo Tinoco-Guerrero
Francisco J. Domínguez-Mota
J. Alberto Guzmán-Torres

Universidad Michoacana de San Nicolás de Hidalgo, Morelia, Mexico.
Contact: gerardo.tinoco@umich.mx

Funding & Institutional Support
-------------------------------
This work received institutional and financial support from:
- Secretariat of Science, Humanities, Technology and Innovation (SECIHTI),
  Mexico.
- Coordination of Scientific Research, Universidad Michoacana de San Nicolás
  de Hidalgo (CIC-UMSNH), Mexico.
- SIIIA MATH: Soluciones en Ingeniería.
- International Centre for Numerical Methods in Engineering (CIMNE).
- Aula CIMNE Morelia.

Revision History
----------------
- Initial release: December 2024.
- Last update: July 2026.
================================================================================
"""

# Standard library
import hashlib
import json
import math
import warnings
from contextlib import contextmanager
from pathlib import Path

# Third-party libraries
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

ROOT                  = Path(__file__).resolve().parents[1]
RESULTS               = ROOT / "results"
MODELS                = ROOT / "models"
DATA                  = ROOT / "data"
BASE_TABLE_PATH       = DATA / "adr1d_modeling_table.csv"
DECAY_TABLE_PATH      = DATA / "adr1d_decay_detectability_table.csv"
BASELINE_SUMMARY_PATH = RESULTS / "baseline_validation_summary.json"
DECAY_SUMMARY_PATH    = RESULTS / "decay_detectability_validation_summary.json"
PROTOCOL_PATH         = RESULTS / "final_model_protocol.json"
PREDICTIONS_PATH      = RESULTS / "final_test_predictions.csv"
METRICS_PATH          = RESULTS / "final_test_metrics.json"
MODEL_PATH            = MODELS / "adr1d_parameter_models.joblib"
MANIFEST_PATH         = MODELS / "model_manifest.json"

EXPECTED_PROTOCOL_SHA256 = "56555a235dd6610a5bd3d6376cbe1123490fc50d28b6a0a0c0e8a0c342fdc2d3"


@contextmanager
def _guarded_linear_algebra():
    """
    Suppress expected matrix warnings during fitting and prediction.

    Yields
    ------
    None
        Control returns to the numerical block while warnings are scoped
        locally. Explicit finite-value checks are performed afterward.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            yield


def _require_finite(values, label):
    """
    Require every value in a numeric array to be finite.

    Parameters
    ----------
    values : numpy.ndarray
        Values produced during fitting or prediction.
    label : str
        Human-readable quantity name used in the exception message.

    Raises
    ------
    FloatingPointError
        If at least one value is `NaN` or infinite.

    """
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise FloatingPointError(f"Non-finite values found in {label}")


def _validate_fitted_estimator(estimator, label):
    """
    Check finite learned parameters on a fitted estimator or pipeline.

    Parameters
    ----------
    estimator : object
        Fitted scikit-learn estimator or pipeline.
    label : str
        Model name used in diagnostic messages.

    Raises
    ------
    FloatingPointError
        If coefficients, intercepts, or feature importances are non-finite.

    """
    model = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    for attribute in ("coef_", "intercept_", "feature_importances_"):
        if hasattr(model, attribute):
            _require_finite(np.asarray(getattr(model, attribute)), label + " " + attribute)


def _regression_metrics(actual_log, predicted_log, actual_physical, predicted_physical):
    """
    Compute locked regression metrics in logarithmic and physical scales.

    Parameters
    ----------
    actual_log : numpy.ndarray
        Reference values transformed with base-10 logarithms.
    predicted_log : numpy.ndarray
        Predicted values in base-10 logarithmic space.
    actual_physical : numpy.ndarray
        Positive reference values in physical units.
    predicted_physical : numpy.ndarray
        Positive predictions in physical units.

    Returns
    -------
    dict of str to float
        RMSE, MAE, and R2 in both scales, together with mean and median
        absolute percentage errors.

    """
    absolute_percentage = np.abs(predicted_physical - actual_physical) / np.abs(actual_physical)
    return {
        "rmse_log10": float(math.sqrt(mean_squared_error(actual_log, predicted_log))),
        "mae_log10": float(mean_absolute_error(actual_log, predicted_log)),
        "r2_log10": float(r2_score(actual_log, predicted_log)),
        "rmse_physical": float(math.sqrt(mean_squared_error(actual_physical, predicted_physical))),
        "mae_physical": float(mean_absolute_error(actual_physical, predicted_physical)),
        "r2_physical": float(r2_score(actual_physical, predicted_physical)),
        "mean_absolute_percentage_error": float(absolute_percentage.mean()),
        "median_absolute_percentage_error": float(np.median(absolute_percentage)),
    }


def _sha256(path):
    """
    Compute the SHA-256 digest of a file in bounded-memory blocks.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_protocol(protocol):
    """
    Verify the locked protocol and all model-development inputs.

    Parameters
    ----------
    protocol : dict
        Parsed final model protocol containing trusted input digests.

    Raises
    ------
    RuntimeError
        If the protocol or any locked input differs from its recorded digest.

    """
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("The locked final protocol has changed")
    paths = {
        "adr1d_modeling_table.csv": BASE_TABLE_PATH,
        "adr1d_decay_detectability_table.csv": DECAY_TABLE_PATH,
        "baseline_validation_summary.json": BASELINE_SUMMARY_PATH,
        "decay_detectability_validation_summary.json": DECAY_SUMMARY_PATH,
    }
    for name, path in paths.items():
        expected = protocol["input_artifacts"][name]
        actual   = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Locked input changed: {name}")


def _tree_regression_pipeline(parameters):
    """
    Construct a median-imputed Extra Trees regression pipeline.

    Parameters
    ----------
    parameters : dict
        Keyword arguments fixed for `ExtraTreesRegressor`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted imputation and tree-regression pipeline.

    """
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(**parameters))])


def _classifier_pipeline(parameters):
    """
    Construct the decay-resolvability classification pipeline.

    Parameters
    ----------
    parameters : dict
        Keyword arguments fixed for `LogisticRegression`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted median-imputation, standardization, and logistic pipeline.

    """
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(**parameters))])


def _by_regime_metrics(predictions):
    """
    Summarize effective-parameter errors by ADR1D physical regime.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Final test predictions with reference and predicted effective values.

    Returns
    -------
    dict of str to dict
        Test-row counts and median absolute percentage errors for each regime.

    """
    result = {}
    for regime, frame in predictions.groupby("diagnostic_regime"):
        velocity_ape   = np.abs(frame["predicted_effective_velocity_m_s"] - frame["actual_effective_velocity_m_s"]) / frame["actual_effective_velocity_m_s"]
        dispersion_ape = np.abs(frame["predicted_effective_dispersion_m2_s"] - frame["actual_effective_dispersion_m2_s"]) / frame["actual_effective_dispersion_m2_s"]
        result[str(regime)] = {
            "rows": int(len(frame)),
            "velocity_median_absolute_percentage_error": float(np.median(velocity_ape)),
            "dispersion_median_absolute_percentage_error": float(np.median(dispersion_ape)),
        }
    return result


def main():
    """
    Reconstruct, evaluate, and serialize the locked ADR1D-ML release.

    Returns
    -------
    None
        Predictions, metrics, model bundle, and manifest are written to their
        repository paths. Final metrics are printed as JSON.

    Raises
    ------
    RuntimeError
        If locked inputs, table alignment, row counts, or feature counts differ
        from the protocol.

    Notes
    -----
    Version 1.0.0 uses the already disclosed test split only to reproduce the
    evaluation fixed before publication. It must not be retuned against that
    split while retaining the original performance claim.

    """
    # Verify immutable development decisions before fitting any estimator.
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _verify_protocol(protocol)
    base  = pd.read_csv(BASE_TABLE_PATH)
    decay = pd.read_csv(DECAY_TABLE_PATH)

    # Enforce scenario alignment and the locked development/test partition.
    if not base[["scenario_id", "split"]].equals(decay[["scenario_id", "split"]]):
        raise RuntimeError("Base and decay tables are not aligned")
    development_mask = base["split"].isin(protocol["development_splits"])
    test_mask        = base["split"].eq(protocol["test_split"])
    if int(development_mask.sum()) != 255 or int(test_mask.sum()) != 45:
        raise RuntimeError("Unexpected development or test row count")

    base_features    = [name for name in base if name.startswith("feature_")]
    decay_features   = [name for name in decay if name.startswith("feature_")]
    compact_features = protocol["models"]["decay_rate_resolvable"]["feature_columns"]
    if len(base_features) != 69 or len(decay_features) != 86:
        raise RuntimeError("Locked feature counts do not match")

    # Reconstruct the four pipelines from protocol parameters only.
    velocity_model   = _tree_regression_pipeline(protocol["models"]["effective_velocity"]["parameters"])
    dispersion_model = _tree_regression_pipeline(protocol["models"]["effective_dispersion"]["parameters"])
    decay_classifier = _classifier_pipeline(protocol["models"]["decay_resolvability"]["parameters"])
    decay_regressor  = _tree_regression_pipeline(protocol["models"]["decay_rate_resolvable"]["parameters"])

    development_base       = base.loc[development_mask]
    test_base              = base.loc[test_mask]
    development_decay      = decay.loc[development_mask]
    test_decay             = decay.loc[test_mask]
    resolvable_development = development_decay.loc[development_decay["target_decay_resolvable"] == 1]

    # Fit on development scenarios and evaluate the reserved scenarios once.
    with _guarded_linear_algebra():
        velocity_model.fit(development_base[base_features], development_base["target_log10_effective_velocity"])
        dispersion_model.fit(development_base[base_features], development_base["target_log10_effective_dispersion"])
        decay_classifier.fit(development_decay[decay_features], development_decay["target_decay_resolvable"])
        decay_regressor.fit(resolvable_development[compact_features], resolvable_development["target_log10_decay_rate_resolvable"])

        predicted_velocity_log   = velocity_model.predict(test_base[base_features])
        predicted_dispersion_log = dispersion_model.predict(test_base[base_features])
        decay_probabilities      = decay_classifier.predict_proba(test_decay[decay_features])
        predicted_decay_log      = decay_regressor.predict(test_decay[compact_features])

    for name, estimator in {"effective velocity": velocity_model, "effective dispersion": dispersion_model, "decay resolvability": decay_classifier, "resolvable decay rate": decay_regressor}.items():
        _validate_fitted_estimator(estimator, name)

    predicted_velocity            = np.power(10.0, predicted_velocity_log)
    predicted_dispersion          = np.power(10.0, predicted_dispersion_log)
    predicted_decay_if_resolvable = np.power(10.0, predicted_decay_log)
    positive_index                = int(np.where(decay_classifier.classes_ == 1)[0][0])
    decay_probability             = decay_probabilities[:, positive_index]
    decision_threshold            = float(protocol["models"]["decay_resolvability"]["decision_threshold"])
    predicted_resolvable          = (decay_probability >= decision_threshold).astype(int)

    for label, values in {"velocity predictions": predicted_velocity, "dispersion predictions": predicted_dispersion, "decay probabilities": decay_probability, "decay-rate predictions": predicted_decay_if_resolvable}.items():
        _require_finite(values, label)

    predictions = test_decay[["scenario_id", "split", "diagnostic_regime", "diagnostic_decay_state", "diagnostic_damkohler_number"]].copy()
    predictions["actual_effective_velocity_m_s"] = test_base["target_effective_velocity_m_s"].to_numpy(dtype=float)
    predictions["predicted_effective_velocity_m_s"] = predicted_velocity
    predictions["actual_effective_dispersion_m2_s"] = test_base["target_effective_dispersion_m2_s"].to_numpy(dtype=float)
    predictions["predicted_effective_dispersion_m2_s"] = predicted_dispersion
    predictions["actual_decay_resolvable"] = test_decay["target_decay_resolvable"].to_numpy(dtype=int)
    predictions["predicted_decay_resolvable"] = predicted_resolvable
    predictions["predicted_decay_resolvable_probability"] = decay_probability
    predictions["actual_decay_rate_s_1"] = test_decay["target_decay_rate_s_1"].to_numpy(dtype=float)
    predictions["predicted_decay_rate_if_resolvable_s_1"] = predicted_decay_if_resolvable
    predictions["reported_decay_rate_s_1"] = np.where(predicted_resolvable == 1, predicted_decay_if_resolvable, np.nan)
    predictions.to_csv(PREDICTIONS_PATH, index=False, float_format="%.12g")

    # Compute the unchanged final-test metrics in both numerical scales.
    velocity_scores   = _regression_metrics(test_base["target_log10_effective_velocity"].to_numpy(dtype=float), predicted_velocity_log, test_base["target_effective_velocity_m_s"].to_numpy(dtype=float), predicted_velocity)
    dispersion_scores = _regression_metrics(test_base["target_log10_effective_dispersion"].to_numpy(dtype=float), predicted_dispersion_log, test_base["target_effective_dispersion_m2_s"].to_numpy(dtype=float), predicted_dispersion)

    actual_resolvable     = predictions["actual_decay_resolvable"].to_numpy(dtype=int)
    classification_scores = {
        "accuracy": float(accuracy_score(actual_resolvable, predicted_resolvable)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_resolvable, predicted_resolvable)),
        "precision": float(precision_score(actual_resolvable, predicted_resolvable, zero_division=0)),
        "recall": float(recall_score(actual_resolvable, predicted_resolvable, zero_division=0)),
        "f1": float(f1_score(actual_resolvable, predicted_resolvable, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual_resolvable, decay_probability)),
        "log_loss": float(log_loss(actual_resolvable, decay_probabilities)),
        "decision_threshold": decision_threshold,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(actual_resolvable, predicted_resolvable).ravel().astype(int).tolist(),
    }

    resolvable_mask                = actual_resolvable == 1
    actual_decay                   = predictions.loc[resolvable_mask, "actual_decay_rate_s_1"].to_numpy(dtype=float)
    predicted_decay                = predicted_decay_if_resolvable[resolvable_mask]
    actual_decay_log               = np.log10(actual_decay)
    predicted_decay_log_resolvable = np.log10(predicted_decay)
    decay_ape                      = np.abs(predicted_decay - actual_decay) / actual_decay
    decay_regression_scores        = {
        "rows": int(resolvable_mask.sum()),
        "rmse_log10": float(math.sqrt(mean_squared_error(actual_decay_log, predicted_decay_log_resolvable))),
        "mae_log10": float(mean_absolute_error(actual_decay_log, predicted_decay_log_resolvable)),
        "r2_log10": float(r2_score(actual_decay_log, predicted_decay_log_resolvable)),
        "rmse_physical": float(math.sqrt(mean_squared_error(actual_decay, predicted_decay))),
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
        "test_decay_state_counts": {key: int(value) for key, value in predictions["diagnostic_decay_state"].value_counts().to_dict().items()},
        "effective_velocity": velocity_scores,
        "effective_dispersion": dispersion_scores,
        "decay_resolvability": classification_scores,
        "decay_rate_conditional_on_resolvable": decay_regression_scores,
        "by_regime": _by_regime_metrics(predictions),
        "post_test_tuning_performed": False,
        "software": {"joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Persist only standard scikit-learn objects and explicit column contracts.
    bundle = {
        "bundle_version": "1.0.0",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "models": {
            "effective_velocity": velocity_model,
            "effective_dispersion": dispersion_model,
            "decay_resolvability": decay_classifier,
            "decay_rate_resolvable": decay_regressor,
        },
        "feature_columns": {"effective_parameters": base_features, "decay_resolvability": decay_features, "decay_rate_resolvable": compact_features},
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

    # Bind the model, protocol, predictions, metrics, and training code by hash.
    manifest = {
        "bundle_version": "1.0.0",
        "model_file": MODEL_PATH.name,
        "model_sha256": _sha256(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "protocol_file": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "predictions_sha256": _sha256(PREDICTIONS_PATH),
        "metrics_sha256": _sha256(METRICS_PATH),
        "training_script_sha256": _sha256(Path(__file__)),
        "development_rows": int(development_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "post_test_tuning_performed": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
