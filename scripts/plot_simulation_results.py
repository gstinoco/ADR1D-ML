"""
================================================================================
ADR1D-ML: Simulation-Space Visualization
================================================================================

This module visualizes the ADR1D simulations that support parameter inference.
It reconstructs analytical concentration fields directly from the public
scenario tables, displays virtual-sensor breakthrough curves, and compares
reference and machine-learning-driven fields for correctly resolved test cases.

Main Operations
---------------
1. Reconstruct three benchmark fields spanning the modeled decay states.
2. Compare analytical sensor curves with noisy and censored observations.
3. Reconstruct all correctly classified resolvable test fields with ML outputs.
4. Export two publication-ready figures and simulation-space error metrics.

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
- Initial release: January 2025.
- Last update: July 2026.
================================================================================
"""

# Standard library
import json
import math
import os
import shutil
from pathlib import Path


ROOT  = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/matplotlib_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE)

# Third-party libraries. Matplotlib is imported after defining its writable
# cache because some restricted environments do not expose a user cache path.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.special import erfc  # noqa: E402


MODELING_TABLE_PATH         = ROOT / "data/adr1d_decay_detectability_table.csv"
EXAMPLE_SOURCES_PATH        = ROOT / "data/example_sources.csv"
EXAMPLE_OBSERVATIONS_PATH   = ROOT / "data/example_sensor_observations.csv"
TEST_PREDICTIONS_PATH       = ROOT / "results/final_test_predictions.csv"
EXAMPLE_FIGURE_PATH         = ROOT / "docs/simulation_examples.png"
RECONSTRUCTION_FIGURE_PATH  = ROOT / "docs/simulation_reconstructions.png"
RECONSTRUCTION_METRICS_PATH = ROOT / "results/simulation_reconstruction_metrics.csv"

DOMAIN_LENGTH_M     = 1000.0
FINAL_TIME_S        = 86400.0
SENSOR_POSITIONS_M  = np.asarray([100.0, 250.0, 400.0, 600.0, 800.0, 1000.0])
FIELD_POSITIONS_M   = np.linspace(0.0, DOMAIN_LENGTH_M, 201)
FIELD_TIMES_S       = np.linspace(0.0, FINAL_TIME_S, 121)
EXAMPLE_STATE_ORDER = ("zero", "below_resolution", "resolvable")
STATE_LABELS        = {"zero": "Zero decay", "below_resolution": "Below resolution", "resolvable": "Resolvable decay"}


def _step_response_grid(positions_m, elapsed_times_s, effective_velocity_m_s, effective_dispersion_m2_s, decay_rate_s_1):
    """
    Evaluate the reactive Ogata-Banks unit-step response on a space-time grid.

    Parameters
    ----------
    positions_m : numpy.ndarray
        One-dimensional spatial coordinates in meters.
    elapsed_times_s : numpy.ndarray
        One-dimensional times elapsed since step activation, in seconds.
        Non-positive values represent times before activation.
    effective_velocity_m_s : float
        Positive effective velocity `v/R` in meters per second.
    effective_dispersion_m2_s : float
        Positive effective dispersion `D/R` in square meters per second.
    decay_rate_s_1 : float
        Non-negative first-order decay rate in inverse seconds.

    Returns
    -------
    numpy.ndarray
        Dimensionless response with shape `(n_times, n_positions)`.

    Raises
    ------
    ValueError
        If a coordinate or physical parameter lies outside its valid domain.
    FloatingPointError
        If the analytical evaluation produces a non-finite value.

    Notes
    -----
    Dividing the ADR equation by `R` permits the analytical solution to be
    evaluated directly with `v/R`, `D/R`, and `lambda` by setting `R = 1`.

    """
    positions = np.asarray(positions_m, dtype=float)
    elapsed   = np.asarray(elapsed_times_s, dtype=float)
    if positions.ndim != 1 or elapsed.ndim != 1:
        raise ValueError("Positions and elapsed times must be one-dimensional")
    if np.any(positions < 0.0) or np.any(~np.isfinite(positions)):
        raise ValueError("Spatial coordinates must be finite and non-negative")
    if np.any(~np.isfinite(elapsed)):
        raise ValueError("Elapsed times must be finite")
    if effective_velocity_m_s <= 0.0 or effective_dispersion_m2_s <= 0.0:
        raise ValueError("Effective velocity and dispersion must be positive")
    if decay_rate_s_1 < 0.0 or not math.isfinite(decay_rate_s_1):
        raise ValueError("Decay rate must be finite and non-negative")

    response = np.zeros((elapsed.size, positions.size), dtype=float)
    positive = elapsed > 0.0
    if not positive.any():
        return response

    x_grid          = positions[np.newaxis, :]
    time_grid       = elapsed[positive, np.newaxis]
    transport_speed = math.sqrt(effective_velocity_m_s**2 + 4.0 * effective_dispersion_m2_s * decay_rate_s_1)
    denominator     = 2.0 * np.sqrt(effective_dispersion_m2_s * time_grid)
    first_exponent  = (effective_velocity_m_s - transport_speed) * x_grid / (2.0 * effective_dispersion_m2_s)
    second_exponent = (effective_velocity_m_s + transport_speed) * x_grid / (2.0 * effective_dispersion_m2_s)
    first_argument  = (x_grid - transport_speed * time_grid) / denominator
    second_argument = (x_grid + transport_speed * time_grid) / denominator

    with np.errstate(over="raise", invalid="raise"):
        positive_response = 0.5 * (np.exp(first_exponent) * erfc(first_argument) + np.exp(second_exponent) * erfc(second_argument))
    if not np.isfinite(positive_response).all():
        raise FloatingPointError("Non-finite analytical step response")
    response[positive, :] = np.clip(positive_response, 0.0, 1.0)
    return response


def _pulse_field(positions_m, times_s, source_concentration_mg_L, source_start_s, source_duration_s, effective_velocity_m_s, effective_dispersion_m2_s, decay_rate_s_1):
    """
    Evaluate a finite inlet pulse by analytical step-response superposition.

    Parameters
    ----------
    positions_m : numpy.ndarray
        One-dimensional spatial coordinates in meters.
    times_s : numpy.ndarray
        One-dimensional simulation times in seconds.
    source_concentration_mg_L : float
        Positive inlet concentration during the pulse, in milligrams per liter.
    source_start_s : float
        Non-negative pulse start time in seconds.
    source_duration_s : float
        Positive pulse duration in seconds.
    effective_velocity_m_s : float
        Positive effective velocity in meters per second.
    effective_dispersion_m2_s : float
        Positive effective dispersion in square meters per second.
    decay_rate_s_1 : float
        Non-negative first-order decay rate in inverse seconds.

    Returns
    -------
    numpy.ndarray
        Concentration field in milligrams per liter with shape
        `(n_times, n_positions)`.

    Raises
    ------
    ValueError
        If the source definition is invalid or the analytical field becomes
        negative beyond floating-point tolerance.

    """
    if source_concentration_mg_L <= 0.0:
        raise ValueError("Source concentration must be positive")
    if source_start_s < 0.0 or source_duration_s <= 0.0:
        raise ValueError("Source start and duration are outside their domains")

    positions    = np.asarray(positions_m, dtype=float)
    times        = np.asarray(times_s, dtype=float)
    response_on  = _step_response_grid(positions, times - source_start_s, effective_velocity_m_s, effective_dispersion_m2_s, decay_rate_s_1)
    response_off = _step_response_grid(positions, times - source_start_s - source_duration_s, effective_velocity_m_s, effective_dispersion_m2_s, decay_rate_s_1)
    field        = source_concentration_mg_L * (response_on - response_off)

    # Enforce the finite-pulse boundary exactly at x = 0.
    boundary = np.isclose(positions, 0.0)
    active   = (times >= source_start_s) & (times < source_start_s + source_duration_s)
    if boundary.any():
        field[:, boundary] = np.where(active[:, np.newaxis], source_concentration_mg_L, 0.0)

    tolerance = 1.0e-10 * max(1.0, source_concentration_mg_L)
    if float(field.min()) < -tolerance:
        raise ValueError("Analytical pulse produced a negative concentration")
    if not np.isfinite(field).all():
        raise FloatingPointError("Analytical pulse produced a non-finite value")
    return np.clip(field, 0.0, source_concentration_mg_L)


def _load_inputs():
    """
    Load and validate the public tables required by both visualizations.

    Returns
    -------
    tuple of pandas.DataFrame
        Modeling table, example source table, example observation table, and
        locked test predictions, in that order.

    Raises
    ------
    ValueError
        If identifiers are duplicated or the three example decay states are
        incomplete.

    """
    modeling     = pd.read_csv(MODELING_TABLE_PATH)
    sources      = pd.read_csv(EXAMPLE_SOURCES_PATH)
    observations = pd.read_csv(EXAMPLE_OBSERVATIONS_PATH)
    predictions  = pd.read_csv(TEST_PREDICTIONS_PATH)

    for label, frame in {"modeling table": modeling, "example sources": sources, "test predictions": predictions}.items():
        if frame["scenario_id"].duplicated().any():
            raise ValueError(f"Duplicate scenario identifiers in {label}")

    example_rows = modeling.loc[modeling["scenario_id"].isin(sources["scenario_id"])]
    states       = set(example_rows["diagnostic_decay_state"])
    if len(example_rows) != 3 or states != set(EXAMPLE_STATE_ORDER):
        raise ValueError("Examples must contain one scenario from each decay state")
    if set(observations["scenario_id"]) != set(sources["scenario_id"]):
        raise ValueError("Example observations and source metadata are not aligned")
    return modeling, sources, observations, predictions


def _plot_simulation_examples(modeling, sources, observations):
    """
    Plot benchmark fields and virtual-sensor curves for three decay states.

    Parameters
    ----------
    modeling : pandas.DataFrame
        Modeling table containing reference parameters and decay-state labels.
    sources : pandas.DataFrame
        Three validation-scenario pulse definitions.
    observations : pandas.DataFrame
        Noisy and censored histories at six virtual sensors for each scenario.

    Returns
    -------
    None
        The figure is written to `docs/simulation_examples.png`.

    Notes
    -----
    These validation scenarios illustrate benchmark behavior. They are part of
    the final development set and do not provide independent test evidence.

    """
    joined = sources.merge(
        modeling[
            [
                "scenario_id",
                "diagnostic_decay_state",
                "target_effective_velocity_m_s",
                "target_effective_dispersion_m2_s",
                "target_decay_rate_s_1",
                "feature_source_concentration_mg_L",
                "feature_source_start_s",
                "feature_source_duration_s",
            ]
        ],
        on="scenario_id",
        validate="one_to_one",
    )
    for source_name, feature_name in {"source_concentration_mg_L": "feature_source_concentration_mg_L", "source_start_s": "feature_source_start_s", "source_duration_s": "feature_source_duration_s"}.items():
        if not np.allclose(joined[source_name], joined[feature_name], rtol=0.0, atol=1e-9):
            raise ValueError(f"Example source mismatch in {source_name}")

    ordered = joined.set_index("diagnostic_decay_state").loc[list(EXAMPLE_STATE_ORDER)]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), layout="constrained", sharex="row", sharey="row")
    fig.suptitle("ADR1D simulation examples", fontsize=16, fontweight="bold")
    sensor_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.5, 6))
    heatmap       = None

    for column, (state, row) in enumerate(ordered.iterrows()):
        source_concentration = float(row["source_concentration_mg_L"])
        source_start         = float(row["source_start_s"])
        source_duration      = float(row["source_duration_s"])
        parameters           = (float(row["target_effective_velocity_m_s"]), float(row["target_effective_dispersion_m2_s"]), float(row["target_decay_rate_s_1"]))
        field                = _pulse_field(FIELD_POSITIONS_M, FIELD_TIMES_S, source_concentration, source_start, source_duration, *parameters)
        normalized           = field / source_concentration

        field_axis = axes[0, column]
        heatmap    = field_axis.pcolormesh(FIELD_POSITIONS_M, FIELD_TIMES_S / 3600.0, normalized, shading="auto", cmap="cividis", vmin=0.0, vmax=1.0)
        for sensor_position in SENSOR_POSITIONS_M:
            field_axis.axvline(sensor_position, color="white", linewidth=0.55, linestyle=":", alpha=0.75)
        field_axis.set_title(f"{STATE_LABELS[state]}\n{row['scenario_id']}", fontsize=11, fontweight="bold")
        field_axis.set_xlabel("Distance (m)")
        if column == 0:
            field_axis.set_ylabel("Time (h)")

        curve_axis            = axes[1, column]
        scenario_observations = observations.loc[observations["scenario_id"] == row["scenario_id"]]
        for sensor_index, sensor_position in enumerate(SENSOR_POSITIONS_M):
            color            = sensor_colors[sensor_index]
            analytical_curve = _pulse_field(np.asarray([sensor_position]), FIELD_TIMES_S, source_concentration, source_start, source_duration, *parameters)[:, 0]
            curve_axis.plot(FIELD_TIMES_S / 3600.0, analytical_curve / source_concentration, color=color, linewidth=1.45)
            sensor_id      = f"S{sensor_index + 1:02d}"
            sensor_rows    = scenario_observations.loc[scenario_observations["sensor_id"] == sensor_id]
            censored       = sensor_rows["is_below_detection_limit"].astype(str).str.lower().eq("true").to_numpy()
            observed       = sensor_rows["concentration_observed_mg_L"].to_numpy(dtype=float) / source_concentration
            observed_times = sensor_rows["time_s"].to_numpy(dtype=float) / 3600.0
            curve_axis.scatter(observed_times[~censored], observed[~censored], color=color, s=10, alpha=0.75, linewidth=0.0)
            curve_axis.scatter(observed_times[censored], observed[censored], color=color, marker="x", s=12, alpha=0.6, linewidth=0.7)
        curve_axis.axvspan(source_start / 3600.0, (source_start + source_duration) / 3600.0, color="#999999", alpha=0.12, linewidth=0.0)
        curve_axis.set_xlim(0.0, FINAL_TIME_S / 3600.0)
        curve_axis.set_ylim(-0.015, 1.08)
        curve_axis.set_xlabel("Time (h)")
        curve_axis.grid(True, color="#dddddd", linewidth=0.55)
        if column == 0:
            curve_axis.set_ylabel("Normalized concentration")

    if heatmap is None:
        raise RuntimeError("No simulation example was plotted")
    fig.colorbar(heatmap, ax=axes[0, :], label="Normalized concentration, C/C0", shrink=0.86, pad=0.015)
    sensor_handles = [Line2D([0], [0], color=sensor_colors[index], linewidth=1.6, label=f"S{index + 1:02d}: {position:.0f} m") for index, position in enumerate(SENSOR_POSITIONS_M)]
    sensor_handles.extend(
        [
            Line2D([0], [0], color="#555555", marker="o", linestyle="none", markersize=4, label="Observed"),
            Line2D([0], [0], color="#555555", marker="x", linestyle="none", markersize=5, label="Censored"),
        ]
    )
    fig.legend(handles=sensor_handles, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=4, frameon=False, fontsize=8.5)
    fig.savefig(EXAMPLE_FIGURE_PATH, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_test_reconstructions(modeling, predictions):
    """
    Compare reference and ML-driven fields for all true-positive test cases.

    Parameters
    ----------
    modeling : pandas.DataFrame
        Modeling table containing source definitions for every scenario.
    predictions : pandas.DataFrame
        Locked test predictions with reference and reported parameter values.

    Returns
    -------
    pandas.DataFrame
        Simulation-space error metrics for each reconstructed test scenario.

    Raises
    ------
    ValueError
        If the locked release does not contain exactly four correctly
        classified resolvable cases or if a reported decay rate is missing.

    Notes
    -----
    The four cases are the complete set of true-positive resolvable scenarios,
    not a visual-quality subset. Unresolved cases are excluded because the
    public model contract intentionally withholds a single decay-rate estimate.

    """
    source_columns = ["scenario_id", "feature_source_concentration_mg_L", "feature_source_start_s", "feature_source_duration_s"]
    cases          = predictions.loc[(predictions["actual_decay_resolvable"] == 1) & (predictions["predicted_decay_resolvable"] == 1)].merge(modeling[source_columns], on="scenario_id", validate="one_to_one")
    cases          = cases.sort_values("scenario_id").reset_index(drop=True)
    if len(cases) != 4:
        raise ValueError("The locked release must contain four resolvable true positives")
    if cases["reported_decay_rate_s_1"].isna().any():
        raise ValueError("A true-positive test case has no reported decay rate")

    reconstructions = []
    metric_rows     = []
    for _, case in cases.iterrows():
        source_concentration = float(case["feature_source_concentration_mg_L"])
        source_start         = float(case["feature_source_start_s"])
        source_duration      = float(case["feature_source_duration_s"])
        reference            = (
            _pulse_field(
                FIELD_POSITIONS_M,
                FIELD_TIMES_S,
                source_concentration,
                source_start,
                source_duration,
                float(case["actual_effective_velocity_m_s"]),
                float(case["actual_effective_dispersion_m2_s"]),
                float(case["actual_decay_rate_s_1"]),
            )
            / source_concentration
        )
        reconstructed = (
            _pulse_field(
                FIELD_POSITIONS_M,
                FIELD_TIMES_S,
                source_concentration,
                source_start,
                source_duration,
                float(case["predicted_effective_velocity_m_s"]),
                float(case["predicted_effective_dispersion_m2_s"]),
                float(case["reported_decay_rate_s_1"]),
            )
            / source_concentration
        )
        error = np.abs(reconstructed - reference)
        reconstructions.append((case, reference, reconstructed, error))
        metric_rows.append(
            {
                "scenario_id": case["scenario_id"],
                "diagnostic_regime": case["diagnostic_regime"],
                "actual_effective_velocity_m_s": case["actual_effective_velocity_m_s"],
                "predicted_effective_velocity_m_s": case["predicted_effective_velocity_m_s"],
                "actual_effective_dispersion_m2_s": case["actual_effective_dispersion_m2_s"],
                "predicted_effective_dispersion_m2_s": case["predicted_effective_dispersion_m2_s"],
                "actual_decay_rate_s_1": case["actual_decay_rate_s_1"],
                "reported_decay_rate_s_1": case["reported_decay_rate_s_1"],
                "field_rmse_normalized": float(np.sqrt(np.mean(error**2))),
                "field_mae_normalized": float(np.mean(error)),
                "field_max_abs_error_normalized": float(np.max(error)),
                "spatial_nodes": int(FIELD_POSITIONS_M.size),
                "time_nodes": int(FIELD_TIMES_S.size),
            }
        )

    error_limit = max(float(error.max()) for _, _, _, error in reconstructions)
    error_limit = max(error_limit, 0.01)
    fig, axes = plt.subplots(len(reconstructions), 3, figsize=(14.2, 12.8), layout="constrained", sharex=True, sharey=True, squeeze=False)
    fig.suptitle("ML-driven reconstruction of resolvable locked-test simulations", fontsize=15, fontweight="bold")
    concentration_image = None
    error_image         = None
    for row_index, (case, reference, reconstructed, error) in enumerate(reconstructions):
        for column_index, (field, cmap, upper) in enumerate(((reference, "cividis", 1.0), (reconstructed, "cividis", 1.0), (error, "magma", error_limit))):
            axis  = axes[row_index, column_index]
            image = axis.pcolormesh(FIELD_POSITIONS_M, FIELD_TIMES_S / 3600.0, field, shading="auto", cmap=cmap, vmin=0.0, vmax=upper)
            for sensor_position in SENSOR_POSITIONS_M:
                axis.axvline(sensor_position, color="white", linewidth=0.45, linestyle=":", alpha=0.6)
            if column_index < 2:
                concentration_image = image
            else:
                error_image = image
            if row_index == len(reconstructions) - 1:
                axis.set_xlabel("Distance (m)")
            if column_index == 0:
                axis.set_ylabel("Time (h)")

        regime = str(case["diagnostic_regime"]).replace("_", " ")
        axes[row_index, 0].text(
            0.02,
            0.96,
            f"{case['scenario_id']}\n{regime}",
            transform=axes[row_index, 0].transAxes,
            va="top",
            fontsize=8.2,
            color="white",
            bbox={"facecolor": "#222222", "edgecolor": "none", "alpha": 0.72},
        )
        axes[row_index, 1].text(
            0.02,
            0.96,
            (
                "reference -> ML\n"
                f"u: {case['actual_effective_velocity_m_s']:.3g} -> "
                f"{case['predicted_effective_velocity_m_s']:.3g} m/s\n"
                f"D: {case['actual_effective_dispersion_m2_s']:.3g} -> "
                f"{case['predicted_effective_dispersion_m2_s']:.3g} m2/s\n"
                f"lambda: {case['actual_decay_rate_s_1']:.2e} -> "
                f"{case['reported_decay_rate_s_1']:.2e} 1/s"
            ),
            transform=axes[row_index, 1].transAxes,
            va="top",
            fontsize=7.3,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
        )
        rmse = math.sqrt(float(np.mean(error**2)))
        axes[row_index, 2].text(
            0.98,
            0.96,
            f"RMSE = {rmse:.3f} C0\nMax = {float(error.max()):.3f} C0",
            transform=axes[row_index, 2].transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="white" if error_limit > 0.25 else "black",
            bbox={"facecolor": "#222222", "edgecolor": "none", "alpha": 0.65},
        )

    axes[0, 0].set_title("Reference field", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("ML-driven field", fontsize=11, fontweight="bold")
    axes[0, 2].set_title("Absolute field error", fontsize=11, fontweight="bold")
    if concentration_image is None or error_image is None:
        raise RuntimeError("No locked-test reconstruction was plotted")
    fig.colorbar(concentration_image, ax=axes[:, :2], label="Normalized concentration, C/C0", shrink=0.86, pad=0.012)
    fig.colorbar(error_image, ax=axes[:, 2], label="Absolute normalized error", shrink=0.86, pad=0.012)
    fig.savefig(RECONSTRUCTION_FIGURE_PATH, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(metric_rows)


def main():
    """
    Generate benchmark and ML-driven simulation visualizations.

    Returns
    -------
    None
        Two PNG figures and one CSV metric table are written to their public
        repository paths. A compact JSON summary is printed to standard output.

    """
    try:
        modeling, sources, observations, predictions = _load_inputs()
        _plot_simulation_examples(modeling, sources, observations)
        metrics = _plot_test_reconstructions(modeling, predictions)
        metrics.to_csv(RECONSTRUCTION_METRICS_PATH, index=False, float_format="%.12g")
        print(
            json.dumps(
                {
                    "example_figure": str(EXAMPLE_FIGURE_PATH.relative_to(ROOT)),
                    "example_scenarios": 3,
                    "reconstruction_figure": str(RECONSTRUCTION_FIGURE_PATH.relative_to(ROOT)),
                    "reconstruction_metrics": str(RECONSTRUCTION_METRICS_PATH.relative_to(ROOT)),
                    "reconstructed_test_scenarios": int(len(metrics)),
                    "status": "ok",
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
