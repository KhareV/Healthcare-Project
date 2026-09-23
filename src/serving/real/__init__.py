"""Stage-4 real (non-synthetic-fixture) serving composition.

Everything under this package binds the frozen Stage-3 outputs
(``artifacts/models/selected_models_v1.json`` plus its G3 freeze) into the
existing Pulkit serving seams. Nothing here trains, searches, calibrates, or
touches final-test data.
"""
