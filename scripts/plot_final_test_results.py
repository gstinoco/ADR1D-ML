#!/usr/bin/env python3
"""
================================================================================
ADR1D-ML: Locked-Test Visualization
================================================================================

This module creates the four-panel diagnostic figure distributed with
ADR1D-ML. It visualizes effective-parameter agreement, decay-resolvability
probabilities, and conditional decay-rate predictions using only the locked
test prediction and metric artifacts.

Main Operations
---------------
1. Plot reference-versus-predicted effective velocity and dispersion.
2. Display decay probabilities by physical detectability state.
3. Plot conditional decay magnitude for resolvable test cases.
4. Export a deterministic, publication-ready PNG image.

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
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


PREDICTIONS_PATH = ROOT / "results/final_test_predictions.csv"
METRICS_PATH = ROOT / "results/final_test_metrics.json"
OUTPUT_PATH = ROOT / "docs/final_test_diagnostics.png"


def _identity_limits(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """
    Compute shared positive limits for a logarithmic identity plot.

    Parameters
    ----------
    actual : numpy.ndarray
        Positive reference values.
    predicted : numpy.ndarray
        Positive model predictions.

    Returns
    -------
    tuple of float
        Lower and upper plotting limits with fixed visual margins.

    """
    lower = float(min(actual.min(), predicted.min())) * 0.8
    upper = float(max(actual.max(), predicted.max())) * 1.25
    return lower, upper


def main() -> None:
    """
    Generate and save the locked four-panel diagnostic figure.

    Returns
    -------
    None
        The PNG image is written to `docs/final_test_diagnostics.png`; its path
        is printed to standard output.

    """
    # Load only locked test artifacts; no model fitting occurs here.
    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    fig.suptitle("ADR1D locked test evaluation", fontsize=15, fontweight="bold")

    panels = [
        (
            axes[0, 0],
            "actual_effective_velocity_m_s",
            "predicted_effective_velocity_m_s",
            "Effective velocity",
            "m/s",
            "#087e8b",
            metrics["effective_velocity"],
        ),
        (
            axes[0, 1],
            "actual_effective_dispersion_m2_s",
            "predicted_effective_dispersion_m2_s",
            "Effective dispersion",
            "m2/s",
            "#f28e2b",
            metrics["effective_dispersion"],
        ),
    ]
    for axis, actual_name, predicted_name, title, unit, color, scores in panels:
        actual = predictions[actual_name].to_numpy(dtype=float)
        predicted = predictions[predicted_name].to_numpy(dtype=float)
        lower, upper = _identity_limits(actual, predicted)
        axis.scatter(
            actual,
            predicted,
            s=32,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.plot(
            [lower, upper],
            [lower, upper],
            color="#333333",
            linestyle="--",
            linewidth=1.2,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel(f"Reference ({unit})")
        axis.set_ylabel(f"Prediction ({unit})")
        axis.text(
            0.04,
            0.95,
            (
                f"R2 = {scores['r2_physical']:.3f}\n"
                f"MdAPE = "
                f"{100 * scores['median_absolute_percentage_error']:.1f}%"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )
        axis.grid(True, which="both", color="#dddddd", linewidth=0.6)

    # Display classifier probabilities by the three physical decay states.
    probability_axis = axes[1, 0]
    state_order = ["zero", "below_resolution", "resolvable"]
    colors = ["#7f7f7f", "#59a14f", "#e15759"]
    rng = np.random.default_rng(20260720)
    for index, (state, color) in enumerate(zip(state_order, colors)):
        values = predictions.loc[
            predictions["diagnostic_decay_state"] == state,
            "predicted_decay_resolvable_probability",
        ].to_numpy(dtype=float)
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        probability_axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=34,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
    probability_axis.axhline(
        metrics["decay_resolvability"]["decision_threshold"],
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="Decision threshold",
    )
    probability_axis.set_xticks(range(3), ["Zero", "Below\nresolution", "Resolvable"])
    probability_axis.set_ylim(-0.03, 1.03)
    probability_axis.set_ylabel("Predicted probability")
    probability_axis.set_title("Decay resolvability", fontsize=11, fontweight="bold")
    probability_axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    probability_axis.legend(loc="upper left", frameon=True, fontsize=8)
    probability_axis.text(
        0.98,
        0.05,
        (
            "Balanced accuracy = "
            f"{metrics['decay_resolvability']['balanced_accuracy']:.3f}\n"
            f"ROC AUC = {metrics['decay_resolvability']['roc_auc']:.3f}"
        ),
        transform=probability_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )

    # Compare conditional decay magnitude only on physically resolvable cases.
    decay_axis = axes[1, 1]
    resolvable = predictions["actual_decay_resolvable"].eq(1)
    actual_decay = predictions.loc[resolvable, "actual_decay_rate_s_1"].to_numpy(
        dtype=float
    )
    predicted_decay = predictions.loc[
        resolvable, "predicted_decay_rate_if_resolvable_s_1"
    ].to_numpy(dtype=float)
    lower, upper = _identity_limits(actual_decay, predicted_decay)
    decay_axis.scatter(
        actual_decay,
        predicted_decay,
        s=46,
        color="#b07aa1",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
    )
    decay_axis.plot(
        [lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1.2
    )
    decay_axis.set_xscale("log")
    decay_axis.set_yscale("log")
    decay_axis.set_xlim(lower, upper)
    decay_axis.set_ylim(lower, upper)
    decay_axis.set_xlabel("Reference lambda (1/s)")
    decay_axis.set_ylabel("Prediction lambda (1/s)")
    decay_axis.set_title("Decay rate when resolvable", fontsize=11, fontweight="bold")
    decay_axis.grid(True, which="both", color="#dddddd", linewidth=0.6)
    decay_scores = metrics["decay_rate_conditional_on_resolvable"]
    decay_axis.text(
        0.04,
        0.95,
        (
            f"n = {decay_scores['rows']}\n"
            f"R2 log10 = {decay_scores['r2_log10']:.3f}\n"
            f"MdAPE = "
            f"{100 * decay_scores['median_absolute_percentage_error']:.1f}%"
        ),
        transform=decay_axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )

    # Persist the figure and remove the repository-local font cache.
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor="white")
    plt.close(fig)
    shutil.rmtree(CACHE, ignore_errors=True)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
