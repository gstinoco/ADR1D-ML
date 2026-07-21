#!/usr/bin/env python3
"""Validate the public raw-sensor-to-parameter integration workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_sensor_features import build_feature_table, validate_inputs
from predict_parameters import load_verified_bundle, predict_feature_table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def assert_frame_close(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
    label: str,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    if not np.allclose(
        actual[columns].to_numpy(dtype=float),
        expected[columns].to_numpy(dtype=float),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        raise AssertionError(f"Mismatch in {label}")


def main() -> None:
    sources = pd.read_csv(DATA / "example_sources.csv")
    observations = pd.read_csv(DATA / "example_sensor_observations.csv")
    expected_features = pd.read_csv(DATA / "example_features.csv")
    expected_predictions = pd.read_csv(RESULTS / "example_predictions.csv")

    validate_inputs(sources, observations)
    actual_features = build_feature_table(sources, observations)
    if actual_features["scenario_id"].tolist() != expected_features[
        "scenario_id"
    ].tolist():
        raise AssertionError("Feature scenario order changed")
    feature_columns = [
        column for column in actual_features if column.startswith("feature_")
    ]
    assert_frame_close(
        actual_features,
        expected_features,
        feature_columns,
        "extracted features",
        rtol=1e-9,
        atol=1e-7,
    )

    bundle, manifest = load_verified_bundle()
    actual_predictions = predict_feature_table(actual_features, bundle)
    if actual_predictions["scenario_id"].tolist() != expected_predictions[
        "scenario_id"
    ].tolist():
        raise AssertionError("Prediction scenario order changed")
    numeric_prediction_columns = [
        "effective_velocity_m_s",
        "effective_dispersion_m2_s",
        "decay_resolvable_probability",
        "decay_resolvable",
        "decay_rate_if_resolvable_s_1",
        "reported_decay_rate_s_1",
    ]
    assert_frame_close(
        actual_predictions,
        expected_predictions,
        numeric_prediction_columns,
        "example predictions",
    )
    if not actual_predictions["decay_status"].equals(
        expected_predictions["decay_status"]
    ):
        raise AssertionError("Decay status labels changed")

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
