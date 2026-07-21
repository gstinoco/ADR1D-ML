# Model bundle

`adr1d_parameter_models.joblib` contains four scikit-learn pipelines:

1. effective velocity regression, `u_eff = v/R`;
2. effective dispersion regression, `D_eff = D/R`;
3. decay-resolvability classification;
4. conditional decay-rate regression for resolvable cases.

The bundle was fitted on 255 development scenarios and evaluated once on the
45 locked ADR1D test scenarios. Its interface, feature contracts, decision
threshold, training protocol, file size, and SHA-256 digest are recorded in
`model_manifest.json`.

The model must be loaded only from a trusted source. The public inference and
validation scripts verify its SHA-256 digest before calling `joblib.load`.
Exact dependency versions are pinned in the repository-level
`requirements.txt` because pickle-compatible model persistence is sensitive to
scikit-learn versions.

The bundle is licensed under the repository MIT License. The training tables
are distributed separately under CC BY 4.0.
