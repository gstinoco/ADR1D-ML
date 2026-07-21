#!/usr/bin/env python3
"""Infer transport parameters with the verified ADR1D-ML model bundle.

This module verifies and loads the serialized ADR1D-ML bundle, checks the
feature contract, and predicts identifiable transport parameters from an
in-memory table or a CSV file. It provides both the public Python API and the
command-line inference interface.

Main operations
---------------
1. Verify the model SHA-256 digest before deserialization.
2. Validate the required feature columns.
3. Predict effective velocity, effective dispersion, and decay resolvability.
4. Report conditional decay only when the classifier resolves it.

Authors and contributors
------------------------
Gerardo Tinoco-Guerrero, Francisco J. Domínguez-Mota,
J. Alberto Guzmán-Torres, Gabriela Pedraza-Jiménez, Eli Chagolla-Inzunza,
Jorge L. González-Figueroa, Christopher N. Magaña-Barocio, and
Maria Goretti Fraga-Lopez.

Universidad Michoacana de San Nicolás de Hidalgo, Morelia, Mexico.
Contact: gerardo.tinoco@umich.mx

Funding and institutional support
---------------------------------
SECIHTI, CIC-UMSNH, SIIIA MATH: Soluciones en Ingeniería, CIMNE, and
Aula CIMNE Morelia.

Revision history
----------------
- Initial release: July 2026.
- Last modification: July 2026.
"""

from __future__ import annotations

# Standard library
import argparse
import hashlib
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

# Third-party libraries
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/adr1d_parameter_models.joblib"
DEFAULT_MANIFEST = ROOT / "models/model_manifest.json"


@contextmanager
def _guarded_linear_algebra() -> Iterable[None]:
    """Suppress expected matrix warnings during model prediction.

    Yields
    ------
    None
        Control returns to the prediction block while NumPy warnings are
        scoped locally. Explicit finite-value checks remain active afterward.

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
    """Require every numeric result to be finite.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric values produced by a fitted estimator.
    label : str
        Human-readable quantity name used in the exception message.

    Raises
    ------
    FloatingPointError
        If at least one value is `NaN` or infinite.

    """
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise FloatingPointError(f"Non-finite values found in {label}")


def _sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file without loading it entirely in memory.

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


def _parse_args() -> argparse.Namespace:
    """Parse command-line inference options.

    Returns
    -------
    argparse.Namespace
        Input, output, model, manifest, split, and optional row-limit values.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="Optionally select rows when the input contains a split column.",
    )
    parser.add_argument("--limit", type=int, help="Optionally keep the first N rows.")
    return parser.parse_args()


def load_verified_bundle(
    model_path: Path = DEFAULT_MODEL,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and load the trusted model bundle and its manifest.

    Parameters
    ----------
    model_path : pathlib.Path, optional
        Serialized Joblib bundle. The repository model is used by default.
    manifest_path : pathlib.Path, optional
        JSON manifest containing the trusted model and protocol digests.

    Returns
    -------
    tuple of dict
        Loaded model bundle followed by its parsed manifest.

    Raises
    ------
    RuntimeError
        If the model digest differs from the manifest or if bundle and
        manifest reference different locked protocols.

    Notes
    -----
    Joblib files must come from a trusted source. Digest verification detects
    accidental or unauthorized file changes; it does not make arbitrary pickle
    files safe to load.

    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model_sha256"] != _sha256(model_path):
        raise RuntimeError("Model hash does not match the manifest")
    bundle = joblib.load(model_path)
    if bundle["protocol_sha256"] != manifest["protocol_sha256"]:
        raise RuntimeError("Bundle and manifest use different protocols")
    return bundle, manifest


def predict_feature_table(
    table: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Predict identifiable transport parameters from an in-memory feature table.

    Parameters
    ----------
    table : pandas.DataFrame
        One or more scenarios containing every feature declared by the bundle.
        Additional metadata columns are allowed.
    bundle : dict
        Verified ADR1D-ML bundle returned by :func:`load_verified_bundle`.

    Returns
    -------
    pandas.DataFrame
        Effective velocity, effective dispersion, decay probability, binary
        resolvability, status, conditional decay, and reported decay. The
        scenario identifier is propagated when present.

    Raises
    ------
    ValueError
        If the table is empty or required feature columns are missing.
    FloatingPointError
        If a model returns a non-finite prediction.

    Notes
    -----
    `reported_decay_rate_s_1` remains missing when decay is below modeled
    sensor resolution. That result must not be replaced automatically by zero.

    """
    if table.empty:
        raise ValueError("The feature table is empty")

    columns = bundle["feature_columns"]
    required = sorted({name for group in columns.values() for name in group})
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError(f"Missing {len(missing)} required feature columns")

    # Evaluate all four pipelines under the locked feature contracts.
    models = bundle["models"]
    with _guarded_linear_algebra():
        velocity = np.power(
            10.0,
            models["effective_velocity"].predict(
                table[columns["effective_parameters"]]
            ),
        )
        dispersion = np.power(
            10.0,
            models["effective_dispersion"].predict(
                table[columns["effective_parameters"]]
            ),
        )
        probabilities = models["decay_resolvability"].predict_proba(
            table[columns["decay_resolvability"]]
        )
        decay_rate = np.power(
            10.0,
            models["decay_rate_resolvable"].predict(
                table[columns["decay_rate_resolvable"]]
            ),
        )
    positive_index = int(np.where(models["decay_resolvability"].classes_ == 1)[0][0])
    probability = probabilities[:, positive_index]
    threshold = float(bundle["decision_threshold"])
    resolvable = (probability >= threshold).astype(int)
    for label, values in {
        "effective velocity": velocity,
        "effective dispersion": dispersion,
        "decay probability": probability,
        "decay rate": decay_rate,
    }.items():
        _require_finite(values, label)

    # Preserve the operational distinction between unresolved and zero decay.
    output = pd.DataFrame(index=table.index)
    if "scenario_id" in table:
        output["scenario_id"] = table["scenario_id"]
    output["effective_velocity_m_s"] = velocity
    output["effective_dispersion_m2_s"] = dispersion
    output["decay_resolvable_probability"] = probability
    output["decay_resolvable"] = resolvable
    output["decay_status"] = np.where(
        resolvable == 1,
        "resolvable",
        "below_sensor_resolution",
    )
    output["decay_rate_if_resolvable_s_1"] = decay_rate
    output["reported_decay_rate_s_1"] = np.where(
        resolvable == 1,
        decay_rate,
        np.nan,
    )
    return output


def main() -> None:
    """Run verified ADR1D-ML inference from a CSV feature table.

    Returns
    -------
    None
        Predictions are written to `--output-csv`; a compact JSON execution
        summary is printed to standard output.

    """
    args = _parse_args()

    # Verify the artifact before reading user-provided feature data.
    bundle, manifest = load_verified_bundle(args.model, args.manifest)
    table = pd.read_csv(args.input_csv)
    if args.split:
        if "split" not in table:
            raise ValueError("The input has no split column")
        table = table.loc[table["split"] == args.split].copy()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        table = table.head(args.limit).copy()
    if table.empty:
        raise ValueError("No rows remain after filtering")

    # Predict and persist the public output contract.
    output = predict_feature_table(table, bundle)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.12g")
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": int(len(output)),
                "resolvable_predictions": int(output["decay_resolvable"].sum()),
                "output": str(args.output_csv),
                "model_sha256": manifest["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
