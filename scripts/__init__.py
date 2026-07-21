"""
================================================================================
ADR1D-ML: Scientific Python Package
================================================================================

This package groups the feature-extraction, parameter-inference, training,
validation, and visualization modules distributed with ADR1D-ML. The stable
in-memory interface is provided by `load_verified_bundle` and
`predict_feature_table` in `scripts.predict_parameters`.

Public Interface
----------------
1. Verify and load the serialized model bundle.
2. Infer identifiable transport parameters from a compatible feature table.

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
