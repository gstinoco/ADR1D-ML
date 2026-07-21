#!/usr/bin/env python3
"""Run the serialized ADR1D parameter models on a compatible feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/adr1d_parameter_models.joblib"
DEFAULT_MANIFEST = ROOT / "models/model_manifest.json"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
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
    """Verify and load the trusted model bundle and its manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model_sha256"] != sha256(model_path):
        raise RuntimeError("Model hash does not match the manifest")
    bundle = joblib.load(model_path)
    if bundle["protocol_sha256"] != manifest["protocol_sha256"]:
        raise RuntimeError("Bundle and manifest use different protocols")
    return bundle, manifest


def predict_feature_table(
    table: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Predict identifiable parameters for an in-memory feature table."""

    if table.empty:
        raise ValueError("The feature table is empty")

    columns = bundle["feature_columns"]
    required = sorted({name for group in columns.values() for name in group})
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError(f"Missing {len(missing)} required feature columns")

    models = bundle["models"]
    with guarded_linear_algebra():
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
    positive_index = int(
        np.where(models["decay_resolvability"].classes_ == 1)[0][0]
    )
    probability = probabilities[:, positive_index]
    threshold = float(bundle["decision_threshold"])
    resolvable = (probability >= threshold).astype(int)
    for label, values in {
        "effective velocity": velocity,
        "effective dispersion": dispersion,
        "decay probability": probability,
        "decay rate": decay_rate,
    }.items():
        require_finite(values, label)

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
    args = parse_args()
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

    output = predict_feature_table(table, bundle)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.12g")
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": int(len(output)),
                "resolvable_predictions": int(resolvable.sum()),
                "output": str(args.output_csv),
                "model_sha256": manifest["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
