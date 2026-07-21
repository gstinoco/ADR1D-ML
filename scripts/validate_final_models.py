#!/usr/bin/env python3
"""
================================================================================
ADR1D-ML: Independent Model-Bundle Validation
================================================================================

This module independently verifies the distributed ADR1D-ML artifacts. It
checks cryptographic digests, loads the trusted bundle, reproduces the locked
test predictions, recomputes selected metrics, and verifies that unresolved
decay is not reported as a numerical rate.

Main Operations
---------------
1. Verify model, protocol, metrics, and prediction digests.
2. Reproduce five prediction arrays from the serialized pipelines.
3. Recompute four representative final-test metrics.
4. Write a machine-readable validation report.

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
- Initial release: July 2026.
- Last update: July 2026.
================================================================================
"""

from __future__ import annotations

# Standard library
import hashlib
import json
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

# Third-party libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DATA = ROOT / "data"
BASE_TABLE_PATH = DATA / "adr1d_modeling_table.csv"
DECAY_TABLE_PATH = DATA / "adr1d_decay_detectability_table.csv"
PROTOCOL_PATH = RESULTS / "final_model_protocol.json"
PREDICTIONS_PATH = RESULTS / "final_test_predictions.csv"
METRICS_PATH = RESULTS / "final_test_metrics.json"
MODEL_PATH = MODELS / "adr1d_parameter_models.joblib"
MANIFEST_PATH = MODELS / "model_manifest.json"
REPORT_PATH = RESULTS / "final_model_validation.json"


@contextmanager
def _guarded_linear_algebra() -> Iterable[None]:
    """
    Suppress expected matrix warnings during independent prediction.

    Yields
    ------
    None
        Control returns to the numerical block while warnings are scoped
        locally. Explicit finite-value checks are performed afterward.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            yield


def _require_finite(values: np.ndarray, label: str) -> None:
    """
    Require every value in a numeric array to be finite.

    Parameters
    ----------
    values : numpy.ndarray
        Values reproduced from the serialized models.
    label : str
        Quantity name used in the exception message.

    Raises
    ------
    FloatingPointError
        If at least one value is `NaN` or infinite.

    """
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise FloatingPointError(f"Non-finite values found in {label}")


def _sha256(path: Path) -> str:
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


def _assert_close(actual: float, expected: float, label: str) -> None:
    """
    Compare two scalar values with the locked numerical tolerance.

    Parameters
    ----------
    actual : float
        Recomputed value.
    expected : float
        Published reference value.
    label : str
        Quantity name used in the assertion message.

    Raises
    ------
    AssertionError
        If the two values differ beyond relative `1e-9` and absolute `1e-12`
        tolerances.

    """
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise AssertionError(f"Mismatch in {label}: {actual} != {expected}")


def _assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
) -> None:
    """
    Compare two prediction arrays with the locked numerical tolerance.

    Parameters
    ----------
    actual : numpy.ndarray
        Reproduced predictions.
    expected : numpy.ndarray
        Published predictions.
    label : str
        Quantity name used in the assertion message.

    Raises
    ------
    AssertionError
        If shape-compatible arrays differ beyond relative `1e-9` and absolute
        `1e-12` tolerances.

    """
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise AssertionError(f"Mismatch in {label}")


def main() -> None:
    """
    Validate the serialized model and locked final-test artifacts.

    Returns
    -------
    None
        A JSON validation report is written to `results/` and printed to
        standard output.

    Raises
    ------
    AssertionError
        If an artifact digest, prediction, metric, row contract, or decay
        reporting rule differs from the published release.
    FloatingPointError
        If any reproduced numeric output is non-finite.

    """
    # Load machine-readable references without trusting the model yet.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTIONS_PATH)
    base = pd.read_csv(BASE_TABLE_PATH)
    decay = pd.read_csv(DECAY_TABLE_PATH)

    # Verify all locked files before deserializing the Joblib bundle.
    if manifest["model_sha256"] != _sha256(MODEL_PATH):
        raise AssertionError("Model bundle hash mismatch")
    if manifest["protocol_sha256"] != _sha256(PROTOCOL_PATH):
        raise AssertionError("Protocol hash mismatch")
    if manifest["metrics_sha256"] != _sha256(METRICS_PATH):
        raise AssertionError("Metrics hash mismatch")
    if manifest["predictions_sha256"] != _sha256(PREDICTIONS_PATH):
        raise AssertionError("Predictions hash mismatch")
    if manifest["post_test_tuning_performed"] is not False:
        raise AssertionError("Manifest reports post-test tuning")

    # Load the verified bundle and reconstruct the exact reserved test rows.
    bundle = joblib.load(MODEL_PATH)
    if bundle["protocol_sha256"] != manifest["protocol_sha256"]:
        raise AssertionError("Bundle protocol hash mismatch")
    test_mask = base["split"].eq("test")
    test_base = base.loc[test_mask]
    test_decay = decay.loc[test_mask]
    if len(test_base) != 45 or len(predictions) != 45:
        raise AssertionError("Expected 45 test rows")
    if not predictions["split"].eq("test").all():
        raise AssertionError("Predictions contain non-test rows")
    if set(predictions["scenario_id"]) != set(test_base["scenario_id"]):
        raise AssertionError("Prediction IDs do not match test IDs")

    columns = bundle["feature_columns"]
    models = bundle["models"]
    # Reproduce all model outputs independently of the training script.
    with _guarded_linear_algebra():
        velocity = np.power(
            10.0,
            models["effective_velocity"].predict(
                test_base[columns["effective_parameters"]]
            ),
        )
        dispersion = np.power(
            10.0,
            models["effective_dispersion"].predict(
                test_base[columns["effective_parameters"]]
            ),
        )
        probabilities = models["decay_resolvability"].predict_proba(
            test_decay[columns["decay_resolvability"]]
        )
        decay_rate = np.power(
            10.0,
            models["decay_rate_resolvable"].predict(
                test_decay[columns["decay_rate_resolvable"]]
            ),
        )
    positive_index = int(np.where(models["decay_resolvability"].classes_ == 1)[0][0])
    positive_probability = probabilities[:, positive_index]
    predicted_resolvable = (
        positive_probability >= float(bundle["decision_threshold"])
    ).astype(int)

    for label, values in {
        "velocity": velocity,
        "dispersion": dispersion,
        "decay probability": positive_probability,
        "decay rate": decay_rate,
    }.items():
        _require_finite(values, label)

    # Compare every prediction array with the published CSV artifact.
    _assert_array_close(
        velocity,
        predictions["predicted_effective_velocity_m_s"].to_numpy(dtype=float),
        "velocity predictions",
    )
    _assert_array_close(
        dispersion,
        predictions["predicted_effective_dispersion_m2_s"].to_numpy(dtype=float),
        "dispersion predictions",
    )
    _assert_array_close(
        positive_probability,
        predictions["predicted_decay_resolvable_probability"].to_numpy(dtype=float),
        "decay probabilities",
    )
    if not np.array_equal(
        predicted_resolvable,
        predictions["predicted_decay_resolvable"].to_numpy(dtype=int),
    ):
        raise AssertionError("Resolvable labels do not reproduce")
    _assert_array_close(
        decay_rate,
        predictions["predicted_decay_rate_if_resolvable_s_1"].to_numpy(dtype=float),
        "decay-rate predictions",
    )

    actual_velocity = predictions["actual_effective_velocity_m_s"].to_numpy(dtype=float)
    actual_dispersion = predictions["actual_effective_dispersion_m2_s"].to_numpy(
        dtype=float
    )
    velocity_rmse_log = math.sqrt(
        mean_squared_error(np.log10(actual_velocity), np.log10(velocity))
    )
    dispersion_rmse_log = math.sqrt(
        mean_squared_error(np.log10(actual_dispersion), np.log10(dispersion))
    )
    # Recompute representative metrics from the reproduced arrays.
    _assert_close(
        velocity_rmse_log,
        metrics["effective_velocity"]["rmse_log10"],
        "velocity test RMSE",
    )
    _assert_close(
        dispersion_rmse_log,
        metrics["effective_dispersion"]["rmse_log10"],
        "dispersion test RMSE",
    )

    balanced_accuracy = balanced_accuracy_score(
        predictions["actual_decay_resolvable"],
        predicted_resolvable,
    )
    _assert_close(
        balanced_accuracy,
        metrics["decay_resolvability"]["balanced_accuracy"],
        "decay resolvability test score",
    )
    resolvable = predictions["actual_decay_resolvable"].eq(1)
    actual_decay_log = np.log10(
        predictions.loc[resolvable, "actual_decay_rate_s_1"].to_numpy(dtype=float)
    )
    predicted_decay_log = np.log10(decay_rate[resolvable.to_numpy()])
    decay_rmse_log = math.sqrt(
        mean_squared_error(actual_decay_log, predicted_decay_log)
    )
    _assert_close(
        decay_rmse_log,
        metrics["decay_rate_conditional_on_resolvable"]["rmse_log10"],
        "decay-rate test RMSE",
    )

    unresolved = predictions["predicted_decay_resolvable"].eq(0)
    if predictions.loc[unresolved, "reported_decay_rate_s_1"].notna().any():
        raise AssertionError("A decay rate was reported below resolution")

    # Record a compact validation result for external audits.
    report = {
        "status": "ok",
        "model_bundle_loaded": True,
        "model_sha256": _sha256(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "test_rows": int(len(predictions)),
        "reproduced_prediction_arrays": 5,
        "recomputed_metric_checks": 4,
        "post_test_tuning_performed": False,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
