# Reproducibility results

This directory contains the locked protocol, model-selection summaries, final
test predictions, final metrics, the independent validation record, and the
three-scenario inference example.

`final_model_protocol.json` was fixed before the test split was opened. The
models were subsequently fitted on the combined training and validation
partitions and evaluated once on the 45 reserved test scenarios. No model,
feature, threshold, or hyperparameter was changed after that evaluation.

Run `python scripts/validate_final_models.py` from the repository root to load
the bundle, reproduce five prediction arrays, recompute four reported metrics,
and verify the model, protocol, metrics, and prediction hashes.
