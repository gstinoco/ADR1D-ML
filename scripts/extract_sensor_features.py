#!/usr/bin/env python3
"""Convert ADR1D-format sensor series into the 86 model input features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SENSOR_POSITIONS_M = {
    "S01": 100.0,
    "S02": 250.0,
    "S03": 400.0,
    "S04": 600.0,
    "S05": 800.0,
    "S06": 1000.0,
}
EXPECTED_TIMES_PER_SENSOR = 49
EXPECTED_TIMES_S = np.arange(0.0, 86400.0 + 1800.0, 1800.0)
SOURCE_COLUMNS = [
    "scenario_id",
    "source_concentration_mg_L",
    "source_start_s",
    "source_duration_s",
]
OBSERVATION_COLUMNS = [
    "scenario_id",
    "sensor_id",
    "x_m",
    "time_s",
    "concentration_observed_mg_L",
    "is_below_detection_limit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-csv", required=True, type=Path)
    parser.add_argument("--observations-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def parse_boolean(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not valid.all():
        invalid = sorted(normalized.loc[~valid].unique())
        raise ValueError(f"Invalid detection-limit flags: {invalid}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def trapezoidal_integral(values: np.ndarray, times: np.ndarray) -> float:
    increments = np.diff(times)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * increments))


def cumulative_mass_time(
    signal: np.ndarray,
    times: np.ndarray,
    fraction: float,
) -> float:
    segment_mass = 0.5 * (signal[:-1] + signal[1:]) * np.diff(times)
    total_mass = float(segment_mass.sum())
    if total_mass <= 0.0:
        return math.nan

    target = fraction * total_mass
    cumulative = 0.0
    for index, mass in enumerate(segment_mass):
        next_cumulative = cumulative + float(mass)
        if next_cumulative >= target and mass > 0.0:
            local_fraction = (target - cumulative) / float(mass)
            return float(
                times[index]
                + local_fraction * (times[index + 1] - times[index])
            )
        cumulative = next_cumulative
    return float(times[-1])


def sensor_features(
    frame: pd.DataFrame,
    source_concentration: float,
    source_start: float,
    source_duration: float,
) -> dict[str, float]:
    frame = frame.sort_values("time_s")
    if len(frame) != EXPECTED_TIMES_PER_SENSOR:
        raise ValueError(
            f"Expected {EXPECTED_TIMES_PER_SENSOR} observations per sensor, "
            f"found {len(frame)}"
        )
    if frame["time_s"].duplicated().any():
        raise ValueError("Sensor time values must be unique")

    times = frame["time_s"].to_numpy(dtype=float)
    observed = frame["concentration_observed_mg_L"].to_numpy(dtype=float)
    below_limit = parse_boolean(frame["is_below_detection_limit"])
    detected = ~below_limit
    if not np.isfinite(times).all():
        raise ValueError("Sensor times must be finite")
    if not np.isfinite(observed[detected]).all():
        raise ValueError("Detected concentrations must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Sensor times must be strictly increasing")

    signal = np.where(detected, observed, 0.0)
    detected_times = times[detected]
    area = trapezoidal_integral(signal, times)
    peak_value = float(signal.max())
    peak_time = (
        float(times[int(np.argmax(signal))]) if peak_value > 0.0 else math.nan
    )

    if area > 0.0:
        first_moment = trapezoidal_integral(signal * times, times) / area
        variance = trapezoidal_integral(
            signal * np.square(times - first_moment), times
        ) / area
        centroid = float(first_moment - source_start)
        spread = float(math.sqrt(max(variance, 0.0)))
    else:
        centroid = math.nan
        spread = math.nan

    return {
        "detection_fraction": float(detected.mean()),
        "peak_ratio": peak_value / source_concentration,
        "peak_time_since_source_s": peak_time - source_start,
        "first_detection_since_source_s": (
            float(detected_times[0] - source_start)
            if detected_times.size
            else math.nan
        ),
        "last_detection_since_source_s": (
            float(detected_times[-1] - source_start)
            if detected_times.size
            else math.nan
        ),
        "normalized_area": area / (source_concentration * source_duration),
        "centroid_since_source_s": centroid,
        "temporal_spread_s": spread,
        "t10_since_source_s": cumulative_mass_time(signal, times, 0.10)
        - source_start,
        "t50_since_source_s": cumulative_mass_time(signal, times, 0.50)
        - source_start,
        "t90_since_source_s": cumulative_mass_time(signal, times, 0.90)
        - source_start,
    }


def line_slope_and_r2(values: np.ndarray) -> tuple[float, float]:
    positions = np.asarray(list(SENSOR_POSITIONS_M.values()), dtype=float)
    if not np.isfinite(values).all():
        return math.nan, math.nan
    centered_x = positions - positions.mean()
    centered_y = values - values.mean()
    slope = float(np.sum(centered_x * centered_y) / np.sum(centered_x**2))
    fitted = values.mean() + slope * centered_x
    total = float(np.sum(centered_y**2))
    residual = float(np.sum((values - fitted) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 1.0
    return slope, float(np.clip(r2, 0.0, 1.0))


def positive_inverse(value: float) -> float:
    return 1.0 / value if value > 0.0 else math.nan


def physics_features(row: dict[str, object]) -> dict[str, float]:
    labels = [sensor_id.lower() for sensor_id in SENSOR_POSITIONS_M]

    def values(suffix: str) -> np.ndarray:
        return np.asarray(
            [row[f"feature_{label}_{suffix}"] for label in labels],
            dtype=float,
        )

    peak_ratio = values("peak_ratio")
    normalized_area = values("normalized_area")
    centroid = values("centroid_since_source_s")
    peak_time = values("peak_time_since_source_s")
    t50 = values("t50_since_source_s")
    temporal_variance = values("temporal_spread_s") ** 2

    log_peak_slope, log_peak_r2 = line_slope_and_r2(
        np.log(np.clip(peak_ratio, 1e-12, None))
    )
    log_area_slope, log_area_r2 = line_slope_and_r2(
        np.log(np.clip(normalized_area, 1e-12, None))
    )
    centroid_slope, centroid_r2 = line_slope_and_r2(centroid)
    peak_time_slope, peak_time_r2 = line_slope_and_r2(peak_time)
    t50_slope, t50_r2 = line_slope_and_r2(t50)
    variance_slope, variance_r2 = line_slope_and_r2(temporal_variance)

    velocity_centroid = positive_inverse(centroid_slope)
    velocity_t50 = positive_inverse(t50_slope)
    dispersion_proxy = (
        max(0.0, 0.5 * variance_slope * velocity_centroid**3)
        if math.isfinite(velocity_centroid)
        else math.nan
    )

    def decay_proxy(attenuation_slope: float) -> float:
        if not math.isfinite(velocity_centroid):
            return math.nan
        estimate = (
            dispersion_proxy * attenuation_slope**2
            - velocity_centroid * attenuation_slope
        )
        return max(0.0, estimate)

    return {
        "feature_physics_log_peak_slope_per_m": log_peak_slope,
        "feature_physics_log_peak_fit_r2": log_peak_r2,
        "feature_physics_log_area_slope_per_m": log_area_slope,
        "feature_physics_log_area_fit_r2": log_area_r2,
        "feature_physics_centroid_slope_s_per_m": centroid_slope,
        "feature_physics_centroid_fit_r2": centroid_r2,
        "feature_physics_peak_time_slope_s_per_m": peak_time_slope,
        "feature_physics_peak_time_fit_r2": peak_time_r2,
        "feature_physics_t50_slope_s_per_m": t50_slope,
        "feature_physics_t50_fit_r2": t50_r2,
        "feature_physics_temporal_variance_slope_s2_per_m": variance_slope,
        "feature_physics_temporal_variance_fit_r2": variance_r2,
        "feature_physics_velocity_centroid_proxy_m_s": velocity_centroid,
        "feature_physics_velocity_t50_proxy_m_s": velocity_t50,
        "feature_physics_dispersion_proxy_m2_s": dispersion_proxy,
        "feature_physics_decay_area_proxy_s_1": decay_proxy(log_area_slope),
        "feature_physics_decay_peak_proxy_s_1": decay_proxy(log_peak_slope),
    }


def validate_inputs(sources: pd.DataFrame, observations: pd.DataFrame) -> None:
    require_columns(sources, SOURCE_COLUMNS, "Source table")
    require_columns(observations, OBSERVATION_COLUMNS, "Observation table")
    if sources["scenario_id"].duplicated().any():
        raise ValueError("Source scenario identifiers must be unique")
    if set(sources["scenario_id"]) != set(observations["scenario_id"]):
        raise ValueError("Source and observation scenario identifiers differ")

    expected_sensors = set(SENSOR_POSITIONS_M)
    for scenario_id, frame in observations.groupby("scenario_id", sort=False):
        if set(frame["sensor_id"]) != expected_sensors:
            raise ValueError(f"{scenario_id} does not contain all six sensors")
        for sensor_id, expected_position in SENSOR_POSITIONS_M.items():
            sensor = frame.loc[frame["sensor_id"] == sensor_id]
            positions = sensor["x_m"].to_numpy(dtype=float)
            if len(sensor) != EXPECTED_TIMES_PER_SENSOR:
                raise ValueError(
                    f"{scenario_id}/{sensor_id} must contain "
                    f"{EXPECTED_TIMES_PER_SENSOR} rows"
                )
            if not np.allclose(positions, expected_position, rtol=0.0, atol=1e-9):
                raise ValueError(f"Unexpected position for {scenario_id}/{sensor_id}")
            times = np.sort(sensor["time_s"].to_numpy(dtype=float))
            if not np.allclose(times, EXPECTED_TIMES_S, rtol=0.0, atol=1e-9):
                raise ValueError(f"Unexpected time grid for {scenario_id}/{sensor_id}")


def build_feature_table(
    sources: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    observation_groups = {
        scenario_id: frame
        for scenario_id, frame in observations.groupby("scenario_id", sort=False)
    }
    rows: list[dict[str, object]] = []

    for source in sources.itertuples(index=False):
        concentration = float(source.source_concentration_mg_L)
        start = float(source.source_start_s)
        duration = float(source.source_duration_s)
        if concentration <= 0.0 or duration <= 0.0:
            raise ValueError("Source concentration and duration must be positive")

        row: dict[str, object] = {"scenario_id": source.scenario_id}
        if hasattr(source, "split"):
            row["split"] = source.split
        row.update(
            {
                "feature_source_concentration_mg_L": concentration,
                "feature_source_start_s": start,
                "feature_source_duration_s": duration,
            }
        )

        scenario_observations = observation_groups[source.scenario_id]
        for sensor_id in SENSOR_POSITIONS_M:
            frame = scenario_observations.loc[
                scenario_observations["sensor_id"] == sensor_id,
                [
                    "time_s",
                    "concentration_observed_mg_L",
                    "is_below_detection_limit",
                ],
            ]
            features = sensor_features(frame, concentration, start, duration)
            prefix = f"feature_{sensor_id.lower()}_"
            row.update({prefix + name: value for name, value in features.items()})

        row.update(physics_features(row))
        rows.append(row)

    output = pd.DataFrame(rows)
    feature_columns = [column for column in output if column.startswith("feature_")]
    if len(feature_columns) != 86:
        raise RuntimeError(f"Expected 86 model features, found {len(feature_columns)}")
    return output


def main() -> None:
    args = parse_args()
    sources = pd.read_csv(args.sources_csv)
    observations = pd.read_csv(args.observations_csv)
    validate_inputs(sources, observations)
    output = build_feature_table(sources, observations)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.12g")
    print(
        json.dumps(
            {
                "status": "ok",
                "scenarios": int(len(output)),
                "feature_columns": 86,
                "output": str(args.output_csv),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
