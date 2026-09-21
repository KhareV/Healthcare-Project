# Phase 13 Full Reproducibility Report

Status: **PASS**. A full 2,000-subject replay was generated after perturbing unrelated Python/NumPy global RNG state into an isolated audit tree. No production artifact was overwritten.

| Artifact | Exact bytes | Production SHA-256 | Replay SHA-256 |
|---|---|---|---|
| subjects | True | 3552746be9fff5bb6e202ea453e7c9e8fca84526311f8916000ef4cd775fc5da | 3552746be9fff5bb6e202ea453e7c9e8fca84526311f8916000ef4cd775fc5da |
| episodes | True | 955284f6a85cbd44eac77dea5e8e1f541f8f2ba801691eca4c1902ab8a6640ed | 955284f6a85cbd44eac77dea5e8e1f541f8f2ba801691eca4c1902ab8a6640ed |
| raw_events | True | 1b0c06fbc59a007571925c17fcc9d2dffb7f2eab06bb9b0cd77dc1c4bd777f10 | 1b0c06fbc59a007571925c17fcc9d2dffb7f2eab06bb9b0cd77dc1c4bd777f10 |
| support_intervals | True | b5bb9fd6ea6d9407da12847f42468b0d82b8cd16e781de372fbfceeda5e34a06 | b5bb9fd6ea6d9407da12847f42468b0d82b8cd16e781de372fbfceeda5e34a06 |
| retained_cohort | True | 766e455ff0505ef760123850b4ba10c082fee55a37f3e02db07692d23e1687aa | 766e455ff0505ef760123850b4ba10c082fee55a37f3e02db07692d23e1687aa |
| structural_index | True | 3456a79d0a1a5296f4004ec23a9b1c428f834e8e0e12ff16f10aa2a348b06e3b | 3456a79d0a1a5296f4004ec23a9b1c428f834e8e0e12ff16f10aa2a348b06e3b |
| canonical_timeline | True | 048e3e0898350ff7bbddb716e3184e1bf958b5e5de00b9d4d90f6ff1b7a83e51 | 048e3e0898350ff7bbddb716e3184e1bf958b5e5de00b9d4d90f6ff1b7a83e51 |
| canonical_statics | True | 2300ead9b497d773f17765f9f62f573a398a942142587712a995141a249ed482 | 2300ead9b497d773f17765f9f62f573a398a942142587712a995141a249ed482 |
| canonical_features | True | 98f75a368a480dd520a7a309f30a2a55f7dcd7a96cb9ea17a620dadd25be5049 | 98f75a368a480dd520a7a309f30a2a55f7dcd7a96cb9ea17a620dadd25be5049 |
| pre_split_package | True | 09e7b0a2aa9396c65d5f90193a689c499803659f76108b0bb5be7f8b62da52e6 | 09e7b0a2aa9396c65d5f90193a689c499803659f76108b0bb5be7f8b62da52e6 |
| phase9_qa | True | 960f42441279104faa57c4bb462d8e0fd570a442ae574c49c3851d71185650ef | 960f42441279104faa57c4bb462d8e0fd570a442ae574c49c3851d71185650ef |
| split | True | 3d6af5298f219f4608e5318ca09856642a1918a1154563ed0e8aec1785563cf6 | 3d6af5298f219f4608e5318ca09856642a1918a1154563ed0e8aec1785563cf6 |
| fit_subjects | True | a46c9bafedc2d8bcf49c06d171e4ca7660e3eb7eb26160a0888e650890e00370 | a46c9bafedc2d8bcf49c06d171e4ca7660e3eb7eb26160a0888e650890e00370 |
| preprocessor | True | c28bfdfbd83f08de96c30da94557a4da158ce28fedda254c1823be7b27fd6eb0 | c28bfdfbd83f08de96c30da94557a4da158ce28fedda254c1823be7b27fd6eb0 |

Environment: Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, PyTorch 2.8.0, XGBoost 2.1.4.
