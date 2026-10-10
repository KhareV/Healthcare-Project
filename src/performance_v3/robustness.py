"""Development-only stress tests for V3 (protocol F10).

Pure, seeded perturbations of canonical rows that stay semantically consistent with the frozen
schema: no future information, no pattern chosen from outcomes. Padding / sentinel encoding follows
the frozen canonical rows (padding_mask True, values None, mask False, TSLO = no-observation sentinel).

Known limitation (documented in the report): SOFA(t), SOFA trend and SOFA component inputs come from
the frozen scorer on the UNPERTURBED raw record and are not re-derived under these row-level perturbations.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

TSLO_SENTINEL = 54.0          # frozen "no genuine observation inside the 48h lookback" value
BIN_HOURS = 6.0
NOISE_CHANNELS = tuple(range(0, 15))   # clinical channels; support-state/rate channels 15-20 untouched
LAB_GROUP = ("pao2__latest", "fio2__latest", "platelet_count__latest", "bilirubin_total__latest", "creatinine__latest", "lactate__latest")
VITAL_GROUP = ("mean_arterial_pressure__latest", "heart_rate__latest", "systolic_blood_pressure__latest",
               "diastolic_blood_pressure__latest", "respiratory_rate__latest", "oxygen_saturation__latest", "temperature__latest")
OUTAGE_GROUPS = {"labs": LAB_GROUP, "vitals": VITAL_GROUP}


def row_rng(row: Mapping[str, object], seed: int, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{tag}|{row['stay_id']}|{row['prediction_time']}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def noise_row(row: Mapping[str, object], sd: float, seed: int) -> Dict[str, object]:
    """Multiplicative Gaussian noise (1 + sd*N(0,1)) on genuinely observed values of clinical channels only."""

    out = copy.deepcopy(dict(row))
    rng = row_rng(row, seed, f"noise{sd}")
    for b, values in enumerate(out["history_values"]):
        for c in NOISE_CHANNELS:
            if out["observation_mask"][b][c] and values[c] is not None:
                values[c] = float(values[c]) * float(1.0 + sd * rng.standard_normal())
    return out


def extra_padding_row(row: Mapping[str, object], k: int) -> Dict[str, object]:
    """Pad the k OLDEST bins (information removal only). TSLO of later bins whose last observation lay in
    a removed bin becomes the no-observation sentinel (TSLO is measured from the bin end)."""

    out = copy.deepcopy(dict(row))
    n_bins = len(out["history_values"])
    n_channels = len(out["temporal_feature_names"])
    for b in range(k):
        out["padding_mask"][b] = True
        for c in range(n_channels):
            out["history_values"][b][c] = None
            out["observation_mask"][b][c] = False
            out["tslo_hours"][b][c] = TSLO_SENTINEL
    for b in range(k, n_bins):
        for c in range(n_channels):
            if out["observation_mask"][b][c]:
                continue
            tslo = out["tslo_hours"][b][c]
            if tslo is None or tslo >= TSLO_SENTINEL:
                continue
            bins_back = int(np.ceil(float(tslo) / BIN_HOURS)) if tslo > 0 else 0
            if b - bins_back < k:   # last observation fell inside a removed bin
                out["tslo_hours"][b][c] = TSLO_SENTINEL
    return out


def outage_row(row: Mapping[str, object], names: Iterable[str]) -> Dict[str, object]:
    out = copy.deepcopy(dict(row))
    index = {n: i for i, n in enumerate(out["temporal_feature_names"])}
    for name in names:
        c = index[name]
        for b in range(len(out["history_values"])):
            out["history_values"][b][c] = None
            out["observation_mask"][b][c] = False
            out["tslo_hours"][b][c] = TSLO_SENTINEL
    return out


def cutoff_band(grid_index: int) -> str:
    return "early_0-1" if grid_index <= 1 else ("middle_2-4" if grid_index <= 4 else "late_5+")


def stay_subsample(rows: Sequence[Mapping[str, object]], n_stays: int, seed: int) -> List[Mapping[str, object]]:
    stays = sorted({r["stay_id"] for r in rows}, key=lambda s: hashlib.sha256(f"{seed}|{s}".encode()).hexdigest())
    keep = set(stays[:n_stays])
    return [r for r in rows if r["stay_id"] in keep]


def scenario_list() -> List[Dict[str, object]]:
    scenarios: List[Dict[str, object]] = [{"name": "unperturbed", "kind": "none"}]
    scenarios += [{"name": f"noise_{int(sd * 100)}pct", "kind": "noise", "sd": sd} for sd in (0.05, 0.10, 0.20)]
    scenarios += [{"name": f"extra_padding_oldest_{k}", "kind": "padding", "k": k} for k in (1, 2, 3)]
    scenarios += [{"name": f"outage_{g}", "kind": "outage", "group": g} for g in OUTAGE_GROUPS]
    return scenarios


def apply_scenario(row: Mapping[str, object], scenario: Mapping[str, object], seed: int) -> Mapping[str, object]:
    kind = scenario["kind"]
    if kind == "none":
        return row
    if kind == "noise":
        return noise_row(row, float(scenario["sd"]), seed)
    if kind == "padding":
        return extra_padding_row(row, int(scenario["k"]))
    if kind == "outage":
        return outage_row(row, OUTAGE_GROUPS[str(scenario["group"])])
    raise ValueError("unknown scenario kind " + repr(kind))
