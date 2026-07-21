#!/usr/bin/env python3
"""
================================================================================
ADR1D-ML: Public Integration Validation
================================================================================

This module validates the complete inference path from raw source and sensor
CSV files to model-ready features and transport-parameter predictions. It
complements the locked-test validator by exercising the public API used by
external numerical workflows.

Main Operations
---------------
1. Rebuild the 86 predictors for three unlabeled example scenarios.
2. Compare extracted features with the distributed reference table.
3. Load the digest-verified model through the public API.
4. Reproduce all eight public prediction columns.

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
import json
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd

# Local modules
from extract_sensor_features import build_feature_table, validate_inputs
from predict_parameters import load_verified_bundle, predict_feature_table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _assert_frame_close(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
    label: str,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """
    Compare selected numeric columns in two aligned tables.

    Parameters
    ----------
    actual : pandas.DataFrame
        Recomputed table.
    expected : pandas.DataFrame
        Distributed reference table.
    columns : list of str
        Numeric columns compared in their given order.
    label : str
        Quantity name used in the assertion message.
    rtol : float, optional
        Relative tolerance passed to `numpy.allclose`.
    atol : float, optional
        Absolute tolerance passed to `numpy.allclose`.

    Raises
    ------
    AssertionError
        If any selected value differs beyond tolerance.

    """
    if not np.allclose(
        actual[columns].to_numpy(dtype=float),
        expected[columns].to_numpy(dtype=float),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        raise AssertionError(f"Mismatch in {label}")


def main() -> None:
    """
    Reproduce the public example from raw sensors through model predictions.

    Returns
    -------
    None
        A compact JSON validation summary is printed to standard output.

    Raises
    ------
    AssertionError
        If scenario order, extracted features, predictions, or status labels
        differ from the distributed references.
    ValueError
        If the example violates the public source or sensor contract.

    """
    # Load raw examples and their independently stored reference outputs.
    sources = pd.read_csv(DATA / "example_sources.csv")
    observations = pd.read_csv(DATA / "example_sensor_observations.csv")
    expected_features = pd.read_csv(DATA / "example_features.csv")
    expected_predictions = pd.read_csv(RESULTS / "example_predictions.csv")

    # Rebuild the complete 86-feature representation.
    validate_inputs(sources, observations)
    actual_features = build_feature_table(sources, observations)
    if (
        actual_features["scenario_id"].tolist()
        != expected_features["scenario_id"].tolist()
    ):
        raise AssertionError("Feature scenario order changed")
    feature_columns = [
        column for column in actual_features if column.startswith("feature_")
    ]
    _assert_frame_close(
        actual_features,
        expected_features,
        feature_columns,
        "extracted features",
        rtol=1e-9,
        atol=1e-7,
    )

    # Exercise the same verified in-memory API offered to numerical solvers.
    bundle, manifest = load_verified_bundle()
    actual_predictions = predict_feature_table(actual_features, bundle)
    if (
        actual_predictions["scenario_id"].tolist()
        != expected_predictions["scenario_id"].tolist()
    ):
        raise AssertionError("Prediction scenario order changed")
    numeric_prediction_columns = [
        "effective_velocity_m_s",
        "effective_dispersion_m2_s",
        "decay_resolvable_probability",
        "decay_resolvable",
        "decay_rate_if_resolvable_s_1",
        "reported_decay_rate_s_1",
    ]
    _assert_frame_close(
        actual_predictions,
        expected_predictions,
        numeric_prediction_columns,
        "example predictions",
    )
    if not actual_predictions["decay_status"].equals(
        expected_predictions["decay_status"]
    ):
        raise AssertionError("Decay status labels changed")

    # Emit a compact integration-test record without generating new files.
    report = {
        "status": "ok",
        "example_scenarios": int(len(actual_features)),
        "validated_feature_columns": int(len(feature_columns)),
        "validated_prediction_columns": int(len(actual_predictions.columns)),
        "model_sha256": manifest["model_sha256"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
