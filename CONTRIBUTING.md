# Contributing to ADR1D-ML

Contributions should preserve the scientific traceability, documentation style,
and deterministic behavior of the published model. Open an issue before
changing the feature contract, sensor geometry, target definitions, locked
protocol, or interpretation of unresolved decay.

## Python style

ADR1D-ML follows the recurring conventions used in the research group's Python
repositories, consolidated into one consistent standard for scientific code.

### Module documentation

Every executable or scientific module must begin with:

- a short title and description of its numerical or machine-learning role;
- a concise list of the operations implemented by the module;
- authorship and institutional affiliation;
- funding and institutional support;
- initial-release and last-modification dates.

The module docstring should describe scientific responsibility and scope. It
must not repeat implementation details that are clearer in function docstrings.

### Imports

Group imports in this order and identify the groups with short comments:

1. Python standard library;
2. third-party libraries;
3. local project modules.

Avoid wildcard imports and imports inside functions unless delayed loading is
needed for a documented reason.

### Naming and type annotations

- Use `snake_case` for functions, variables, and modules.
- Use `UPPER_CASE` for module constants.
- Prefix internal helpers with `_` only when they are not part of the public
  interface.
- Add type annotations to public functions and to internal functions whose
  contracts are not immediately evident.
- Use descriptive scientific names and include units in tabular column names.

### Function documentation

Use NumPy-style docstrings with the sections that apply:

```python
def example(value: float) -> float:
    """Return a transformed physical value.

    Parameters
    ----------
    value : float
        Input value in the documented physical unit.

    Returns
    -------
    float
        Transformed value in the documented physical unit.

    Raises
    ------
    ValueError
        If the input lies outside the supported domain.
    """
```

Document shapes, units, allowed values, side effects, generated files, and
scientific assumptions whenever they are part of the function contract.

### Comments and control flow

- Place short comments before meaningful stages such as validation, feature
  extraction, model fitting, prediction, and persistence.
- Explain scientific choices and non-obvious safeguards rather than narrating
  individual assignments.
- Prefer small functions with one responsibility and early validation of
  invalid inputs.
- Raise specific exceptions with actionable messages. Broad exception handling
  is reserved for command-line boundaries that report and re-raise failures.

### Reproducibility

- Preserve scenario-level train, validation, and test separation.
- Keep random seeds and estimator parameters explicit.
- Verify serialized artifacts before loading them.
- Do not tune against the public test split while retaining the locked
  version-1.0.0 performance claim.
- Record changes that alter predictions, metrics, hashes, dependencies, or the
  sensor contract.

## Required checks

Create a Python 3.12 environment and install the pinned dependencies before
submitting changes:

```bash
python -m pip install -r requirements.txt
python scripts/validate_final_models.py
python scripts/validate_release.py
```

Install the development-only formatter and linter separately from the runtime
dependencies:

```bash
python -m pip install ruff==0.12.3
ruff check scripts
ruff check --select D --ignore D203,D213 scripts
ruff check --select ANN scripts
ruff format --check scripts
```

Changes to training or persistence code also require:

```bash
python scripts/train_and_evaluate_final_models.py
python scripts/validate_final_models.py
python scripts/plot_final_test_results.py
```

The working tree should contain no caches, generated temporary files, absolute
local paths, or untracked model variants when a contribution is proposed.
