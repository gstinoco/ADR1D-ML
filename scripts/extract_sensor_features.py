"""
================================================================================
ADR1D-ML: Sensor Feature Extraction
================================================================================

This module converts ADR1D source metadata and concentration histories into
the 86 predictors required by the parameter-inference model. It validates the
six-sensor geometry and temporal grid used during training, summarizes each
sensor history, and derives transport and attenuation descriptors across the
observation network.

Main Operations
---------------
1. Validate source and observation tables.
2. Extract eleven descriptors from each of six sensor histories.
3. Derive seventeen physics-motivated cross-sensor features.
4. Export one model-ready row per scenario.

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
import argparse
import json
import math
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd


SENSOR_POSITIONS_M        = {"S01": 100.0, "S02": 250.0, "S03": 400.0, "S04": 600.0, "S05": 800.0, "S06": 1000.0}
EXPECTED_TIMES_PER_SENSOR = 49
EXPECTED_TIMES_S          = np.arange(0.0, 86400.0 + 1800.0, 1800.0)
SOURCE_COLUMNS            = ["scenario_id", "source_concentration_mg_L", "source_start_s", "source_duration_s"]
OBSERVATION_COLUMNS       = ["scenario_id", "sensor_id", "x_m", "time_s", "concentration_observed_mg_L", "is_below_detection_limit"]


def _parse_args():
    """
    Parse command-line paths for source, observation, and output tables.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with `sources_csv`, `observations_csv`,
        and `output_csv` attributes represented as paths.

    """
    parser = argparse.ArgumentParser(description="Extract the 86 ADR1D-ML predictors from source metadata and six-sensor concentration histories.")
    parser.add_argument("--sources-csv", required=True, type=Path)
    parser.add_argument("--observations-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def _require_columns(frame, columns, label):
    """
    Verify that a table contains every required column.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table whose schema is being validated.
    columns : list of str
        Required column names.
    label : str
        Human-readable table name used in error messages.

    Raises
    ------
    ValueError
        If one or more required columns are missing.

    """
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _parse_boolean(values):
    """
    Convert accepted textual or numeric Boolean flags into a Boolean array.

    Parameters
    ----------
    values : pandas.Series
        Values encoded as `true`, `false`, `1`, or `0`, without regard to
        capitalization or surrounding whitespace.

    Returns
    -------
    numpy.ndarray
        One-dimensional Boolean array in the original row order.

    Raises
    ------
    ValueError
        If the series contains an unsupported flag.

    """
    normalized = values.astype(str).str.strip().str.lower()
    valid      = normalized.isin({"true", "false", "1", "0"})

    if not valid.all():
        invalid = sorted(normalized.loc[~valid].unique())
        raise ValueError(f"Invalid detection-limit flags: {invalid}")

    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def _trapezoidal_integral(values, times):
    """
    Integrate a sampled signal with the composite trapezoidal rule.

    Parameters
    ----------
    values : numpy.ndarray
        Signal values ordered by time.
    times : numpy.ndarray
        Strictly increasing sample times in seconds.

    Returns
    -------
    float
        Approximate time integral of the sampled signal.

    """
    increments = np.diff(times)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * increments))


def _cumulative_mass_time(signal, times, fraction):
    """
    Locate the time at which a fraction of integrated signal mass is reached.

    Parameters
    ----------
    signal : numpy.ndarray
        Non-negative concentration signal ordered by time.
    times : numpy.ndarray
        Strictly increasing sample times in seconds.
    fraction : float
        Requested cumulative fraction between zero and one.

    Returns
    -------
    float
        Interpolated time in seconds, or `NaN` when total signal mass is zero.

    """
    segment_mass = 0.5 * (signal[:-1] + signal[1:]) * np.diff(times)
    total_mass   = float(segment_mass.sum())

    if total_mass <= 0.0:
        return math.nan

    target     = fraction * total_mass
    cumulative = 0.0

    for index, mass in enumerate(segment_mass):
        next_cumulative = cumulative + float(mass)
        if next_cumulative >= target and mass > 0.0:
            local_fraction = (target - cumulative) / float(mass)
            return float(times[index] + local_fraction * (times[index + 1] - times[index]))
        cumulative = next_cumulative

    return float(times[-1])


def _sensor_features(frame, source_concentration, source_start, source_duration):
    """
    Summarize one sensor concentration history with eleven descriptors.

    Parameters
    ----------
    frame : pandas.DataFrame
        One sensor history with time, observed concentration, and censoring
        columns. Exactly 49 unique samples are required.
    source_concentration : float
        Inlet pulse concentration in mg/L.
    source_start : float
        Pulse start time in seconds.
    source_duration : float
        Positive pulse duration in seconds.

    Returns
    -------
    dict of str to float
        Detection, peak, integral, centroid, spread, and cumulative-time
        descriptors. Time descriptors are measured relative to source start.

    Raises
    ------
    ValueError
        If the history has the wrong size, duplicated or unordered times,
        invalid censoring flags, or non-finite detected concentrations.

    """
    frame = frame.sort_values("time_s")

    if len(frame) != EXPECTED_TIMES_PER_SENSOR:
        raise ValueError(f"Expected {EXPECTED_TIMES_PER_SENSOR} observations per sensor, found {len(frame)}")
    if frame["time_s"].duplicated().any():
        raise ValueError("Sensor time values must be unique")

    times       = frame["time_s"].to_numpy(dtype=float)
    observed    = frame["concentration_observed_mg_L"].to_numpy(dtype=float)
    below_limit = _parse_boolean(frame["is_below_detection_limit"])
    detected    = ~below_limit

    if not np.isfinite(times).all():
        raise ValueError("Sensor times must be finite")
    if not np.isfinite(observed[detected]).all():
        raise ValueError("Detected concentrations must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Sensor times must be strictly increasing")

    signal         = np.where(detected, observed, 0.0)
    detected_times = times[detected]
    area           = _trapezoidal_integral(signal, times)
    peak_value     = float(signal.max())
    peak_time      = float(times[int(np.argmax(signal))]) if peak_value > 0.0 else math.nan

    if area > 0.0:
        first_moment = _trapezoidal_integral(signal * times, times) / area
        variance     = _trapezoidal_integral(signal * np.square(times - first_moment), times) / area
        centroid     = float(first_moment - source_start)
        spread       = float(math.sqrt(max(variance, 0.0)))
    else:
        centroid = math.nan
        spread   = math.nan

    return {
        "detection_fraction":             float(detected.mean()),
        "peak_ratio":                     peak_value / source_concentration,
        "peak_time_since_source_s":       peak_time - source_start,
        "first_detection_since_source_s": (float(detected_times[0] - source_start) if detected_times.size else math.nan),
        "last_detection_since_source_s":  (float(detected_times[-1] - source_start) if detected_times.size else math.nan),
        "normalized_area":                area / (source_concentration * source_duration),
        "centroid_since_source_s":        centroid,
        "temporal_spread_s":              spread,
        "t10_since_source_s":             _cumulative_mass_time(signal, times, 0.10) - source_start,
        "t50_since_source_s":             _cumulative_mass_time(signal, times, 0.50) - source_start,
        "t90_since_source_s":             _cumulative_mass_time(signal, times, 0.90) - source_start,
    }


def _line_slope_and_r2(values):
    """
    Fit a straight line against the six fixed sensor positions.

    Parameters
    ----------
    values : numpy.ndarray
        Six finite values ordered from sensor `S01` through `S06`.

    Returns
    -------
    tuple of float
        Least-squares slope and coefficient of determination. Both values are
        `NaN` when at least one input is non-finite.

    """
    positions = np.asarray(list(SENSOR_POSITIONS_M.values()), dtype=float)

    if not np.isfinite(values).all():
        return math.nan, math.nan

    centered_x = positions - positions.mean()
    centered_y = values - values.mean()
    slope      = float(np.sum(centered_x * centered_y) / np.sum(centered_x**2))
    fitted     = values.mean() + slope * centered_x
    total      = float(np.sum(centered_y**2))
    residual   = float(np.sum((values - fitted) ** 2))
    r2         = 1.0 - residual / total if total > 0.0 else 1.0

    return slope, float(np.clip(r2, 0.0, 1.0))


def _positive_inverse(value):
    """
    Return the inverse of a positive value.

    Parameters
    ----------
    value : float
        Candidate denominator.

    Returns
    -------
    float
        `1/value` for positive values and `NaN` otherwise.

    """
    return 1.0 / value if value > 0.0 else math.nan


def _physics_features(row):
    """
    Derive seventeen transport descriptors across the six sensors.

    Parameters
    ----------
    row : dict of str to object
        Scenario-level source and sensor descriptors generated by this module.

    Returns
    -------
    dict of str to float
        Attenuation slopes, travel-time slopes, fit coefficients, and
        physics-motivated velocity, dispersion, and decay proxies.

    Notes
    -----
    The proxy variables support machine-learning inference; they are not
    independent numerical estimates with guaranteed physical consistency.

    """
    labels = [sensor_id.lower() for sensor_id in SENSOR_POSITIONS_M]

    def values(suffix):
        """
        Collect one sensor descriptor in upstream spatial order.

        Parameters
        ----------
        suffix : str
            Feature suffix shared by the six sensor-specific columns.

        Returns
        -------
        numpy.ndarray
            Six descriptor values ordered from sensor `S01` through `S06`.

        """
        return np.asarray([row[f"feature_{label}_{suffix}"] for label in labels], dtype=float)

    peak_ratio        = values("peak_ratio")
    normalized_area   = values("normalized_area")
    centroid          = values("centroid_since_source_s")
    peak_time         = values("peak_time_since_source_s")
    t50               = values("t50_since_source_s")
    temporal_variance = values("temporal_spread_s") ** 2

    log_peak_slope, log_peak_r2   = _line_slope_and_r2(np.log(np.clip(peak_ratio, 1e-12, None)))
    log_area_slope, log_area_r2   = _line_slope_and_r2(np.log(np.clip(normalized_area, 1e-12, None)))
    centroid_slope, centroid_r2   = _line_slope_and_r2(centroid)
    peak_time_slope, peak_time_r2 = _line_slope_and_r2(peak_time)
    t50_slope, t50_r2             = _line_slope_and_r2(t50)
    variance_slope, variance_r2   = _line_slope_and_r2(temporal_variance)

    velocity_centroid = _positive_inverse(centroid_slope)
    velocity_t50      = _positive_inverse(t50_slope)
    dispersion_proxy  = max(0.0, 0.5 * variance_slope * velocity_centroid**3) if math.isfinite(velocity_centroid) else math.nan

    def decay_proxy(attenuation_slope):
        """
        Estimate non-negative decay from a fitted attenuation slope.

        Parameters
        ----------
        attenuation_slope : float
            Fitted logarithmic attenuation slope in inverse meters.

        Returns
        -------
        float
            Non-negative decay proxy in inverse seconds, or `NaN` when the
            centroid-based velocity proxy is unavailable.

        """
        if not math.isfinite(velocity_centroid):
            return math.nan

        estimate = dispersion_proxy * attenuation_slope**2 - velocity_centroid * attenuation_slope
        return max(0.0, estimate)

    return {
        "feature_physics_log_peak_slope_per_m":             log_peak_slope,
        "feature_physics_log_peak_fit_r2":                  log_peak_r2,
        "feature_physics_log_area_slope_per_m":             log_area_slope,
        "feature_physics_log_area_fit_r2":                  log_area_r2,
        "feature_physics_centroid_slope_s_per_m":           centroid_slope,
        "feature_physics_centroid_fit_r2":                  centroid_r2,
        "feature_physics_peak_time_slope_s_per_m":          peak_time_slope,
        "feature_physics_peak_time_fit_r2":                 peak_time_r2,
        "feature_physics_t50_slope_s_per_m":                t50_slope,
        "feature_physics_t50_fit_r2":                       t50_r2,
        "feature_physics_temporal_variance_slope_s2_per_m": variance_slope,
        "feature_physics_temporal_variance_fit_r2":         variance_r2,
        "feature_physics_velocity_centroid_proxy_m_s":      velocity_centroid,
        "feature_physics_velocity_t50_proxy_m_s":           velocity_t50,
        "feature_physics_dispersion_proxy_m2_s":            dispersion_proxy,
        "feature_physics_decay_area_proxy_s_1":             decay_proxy(log_area_slope),
        "feature_physics_decay_peak_proxy_s_1":             decay_proxy(log_peak_slope),
    }


def validate_inputs(sources, observations):
    """
    Validate source metadata and raw sensor observations against the model domain.

    Parameters
    ----------
    sources : pandas.DataFrame
        One row per scenario with identifier, source concentration, start time,
        and duration.
    observations : pandas.DataFrame
        Sensor histories containing all six trained locations and the fixed
        0--86,400 s grid at 1,800 s intervals.

    Raises
    ------
    ValueError
        If schemas, identifiers, sensors, positions, row counts, or time grids
        violate the public input contract.

    """
    _require_columns(sources, SOURCE_COLUMNS, "Source table")
    _require_columns(observations, OBSERVATION_COLUMNS, "Observation table")

    if sources["scenario_id"].duplicated().any():
        raise ValueError("Source scenario identifiers must be unique")
    if set(sources["scenario_id"]) != set(observations["scenario_id"]):
        raise ValueError("Source and observation scenario identifiers differ")

    expected_sensors = set(SENSOR_POSITIONS_M)

    for scenario_id, frame in observations.groupby("scenario_id", sort=False):
        if set(frame["sensor_id"]) != expected_sensors:
            raise ValueError(f"{scenario_id} does not contain all six sensors")

        for sensor_id, expected_position in SENSOR_POSITIONS_M.items():
            sensor    = frame.loc[frame["sensor_id"] == sensor_id]
            positions = sensor["x_m"].to_numpy(dtype=float)

            if len(sensor) != EXPECTED_TIMES_PER_SENSOR:
                raise ValueError(f"{scenario_id}/{sensor_id} must contain {EXPECTED_TIMES_PER_SENSOR} rows")
            if not np.allclose(positions, expected_position, rtol=0.0, atol=1e-9):
                raise ValueError(f"Unexpected position for {scenario_id}/{sensor_id}")

            times = np.sort(sensor["time_s"].to_numpy(dtype=float))

            if not np.allclose(times, EXPECTED_TIMES_S, rtol=0.0, atol=1e-9):
                raise ValueError(f"Unexpected time grid for {scenario_id}/{sensor_id}")


def build_feature_table(sources, observations):
    """
    Build one model-ready feature row per source scenario.

    Parameters
    ----------
    sources : pandas.DataFrame
        Validated source table with one row per scenario.
    observations : pandas.DataFrame
        Validated observations with six complete sensor histories per scenario.

    Returns
    -------
    pandas.DataFrame
        Scenario identifier, optional split label, and 86 ordered predictor
        columns. No transport target or analytical concentration is included.

    Raises
    ------
    ValueError
        If source concentration or duration is not positive.
    RuntimeError
        If the generated table does not contain exactly 86 predictors.

    Notes
    -----
    Call :func:`validate_inputs` before this function when processing external
    data. The command-line interface performs both stages automatically.

    """
    # Group once so each scenario can be processed without repeated filtering.
    observation_groups = {scenario_id: frame for scenario_id, frame in observations.groupby("scenario_id", sort=False)}
    rows               = []

    for source in sources.itertuples(index=False):
        concentration = float(source.source_concentration_mg_L)
        start         = float(source.source_start_s)
        duration      = float(source.source_duration_s)

        if concentration <= 0.0 or duration <= 0.0:
            raise ValueError("Source concentration and duration must be positive")

        # Source descriptors are known inputs, not inferred transport targets.
        row = {"scenario_id": source.scenario_id}

        if hasattr(source, "split"):
            row["split"] = source.split

        row.update({"feature_source_concentration_mg_L": concentration, "feature_source_start_s": start, "feature_source_duration_s": duration})

        scenario_observations = observation_groups[source.scenario_id]

        for sensor_id in SENSOR_POSITIONS_M:
            frame    = scenario_observations.loc[scenario_observations["sensor_id"] == sensor_id, ["time_s", "concentration_observed_mg_L", "is_below_detection_limit"]]
            features = _sensor_features(frame, concentration, start, duration)
            prefix   = f"feature_{sensor_id.lower()}_"
            row.update({prefix + name: value for name, value in features.items()})

        row.update(_physics_features(row))
        rows.append(row)

    output          = pd.DataFrame(rows)
    feature_columns = [column for column in output if column.startswith("feature_")]

    if len(feature_columns) != 86:
        raise RuntimeError(f"Expected 86 model features, found {len(feature_columns)}")

    return output


def main():
    """
    Validate input CSV files, extract features, and write the output table.

    Returns
    -------
    None
        Results are written to the path supplied through `--output-csv`; a
        compact JSON execution summary is printed to standard output.

    """
    args = _parse_args()

    # Load and validate the public raw-input contract.
    sources      = pd.read_csv(args.sources_csv)
    observations = pd.read_csv(args.observations_csv)
    validate_inputs(sources, observations)

    # Extract and persist one feature vector per scenario.
    output = build_feature_table(sources, observations)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.12g")

    print(json.dumps({"status": "ok", "scenarios": int(len(output)), "feature_columns": 86, "output": str(args.output_csv)}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
