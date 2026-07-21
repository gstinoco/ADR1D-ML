# ADR1D-ML technical report

## Product

ADR1D-ML contains four components for estimating normalized transport
parameters from descriptors of six virtual sensor histories:

1. effective-velocity regression, `u_eff = v/R`;
2. effective-dispersion regression, `D_eff = D/R`;
3. classification of decay that is resolvable by the modeled sensors;
4. conditional regression of `lambda` when decay is resolvable.

The model configurations were fixed before the test split was opened through
`results/final_model_protocol.json`, whose SHA-256 digest is
`56555a235dd6610a5bd3d6376cbe1123490fc50d28b6a0a0c0e8a0c342fdc2d3`.
The four pipelines were fitted on 255 development scenarios and evaluated once
on 45 reserved test scenarios. No feature, threshold, estimator, or
hyperparameter was changed after that evaluation.

## Locked test results

| Output | Main metric | Result |
|---|---|---:|
| Effective velocity | MdAPE | 4.40% |
| Effective velocity | Physical-scale R2 | 0.9840 |
| Effective dispersion | MdAPE | 27.65% |
| Effective dispersion | Physical-scale R2 | 0.6916 |
| Decay resolvability | Balanced accuracy | 0.7564 |
| Decay resolvability | ROC AUC | 0.8120 |
| Decay resolvability | Recall | 0.6667 |
| Resolvable `lambda` | MdAPE | 69.01% |
| Resolvable `lambda` | Log10-scale R2 | 0.4250 |

Effective velocity remained close to its validation performance. Effective
dispersion degraded from 19.16% validation MdAPE to 27.65% test MdAPE and
should be treated as a moderate-precision estimate.

The test split contains 22 zero-decay cases, 17 positive cases below the
modeled resolution, and six resolvable cases. The classifier identified four
of the six resolvable cases and produced six false positives, giving the
confusion matrix `[[TN, FP], [FN, TP]] = [[33, 6], [2, 4]]`. Conditional decay
magnitude was evaluated on only six cases and retained high error. It must not
be presented as a precise measurement of `lambda` without additional data.

## Numerical integration

The normalized ADR1D equation is

$$
\frac{\partial C}{\partial t}
=D_{eff}\frac{\partial^2 C}{\partial x^2}
-u_{eff}\frac{\partial C}{\partial x}
-\lambda C.
$$

Effective velocity and dispersion predictions can be used directly as
`u_eff` and `D_eff` within the demonstrated ADR1D domain. If an independent
measurement supplies `R`, the dimensional coefficients can be reconstructed as
`v = R*u_eff` and `D = R*D_eff`.

When the classifier marks decay as resolvable, the fourth model supplies a
conditional `lambda` estimate. When it marks decay as unresolved, `lambda`
must not be replaced automatically by zero. Under the same benchmark
definition, the result indicates approximately

$$
\lambda L/u_{eff}<0.030459.
$$

The recommended response is to propagate a sensitivity interval between zero
and the corresponding upper relationship, or to obtain more informative
measurements before fixing the reactive term.

## Bundle and inference

`models/adr1d_parameter_models.joblib` contains the four pipelines and their
column contracts. `models/model_manifest.json` records the bundle digest,
size, protocol, and locked-result hashes. `scripts/predict_parameters.py`
accepts a compatible feature table without requiring labels. Raw sensor series
matching the trained geometry can first be transformed with
`scripts/extract_sensor_features.py`.

Independent validation loaded the bundle, reproduced five prediction arrays,
recomputed four metrics, and returned status `ok`. The model SHA-256 is
`6890df1b30f5572611e5fcdc0d80a4f923e3877f93594b642c0d084c3f361cea`.

## Domain and limitations

- The model covers the ranges sampled by ADR1D 1.0.0; extrapolation outside
  that distribution was not evaluated.
- Geometry, temporal window, resolution, noise, and censoring correspond to
  the six synthetic benchmark sensors.
- The modeled medium is one-dimensional and homogeneous, with constant
  parameters and a known pulse source.
- `v`, `D`, and `R` cannot be identified separately from concentration without
  independent information.
- Dispersion is less precise than velocity, and conditional decay magnitude is
  supported by a small test subset.
- Water Quality Portal records do not provide equivalent labels for field
  validation of these coefficients. Transfer to real observations remains
  outside the demonstrated scope.

## Reproduction

From the repository root:

```bash
python scripts/train_and_evaluate_final_models.py
python scripts/validate_final_models.py
python scripts/plot_final_test_results.py
```

The training command verifies the locked protocol and input digests before
fitting. Re-execution reproduces the published procedure; it does not justify
tuning against the public test split while retaining the original performance
claim.
