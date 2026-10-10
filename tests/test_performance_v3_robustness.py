import copy

import numpy as np

from performance_v3 import robustness as R

NAMES = [
    "pao2__latest", "fio2__latest", "platelet_count__latest", "bilirubin_total__latest",
    "mean_arterial_pressure__latest", "glasgow_coma_scale__latest", "creatinine__latest",
    "urine_output_volume__sum", "heart_rate__latest", "systolic_blood_pressure__latest",
    "diastolic_blood_pressure__latest", "respiratory_rate__latest", "oxygen_saturation__latest",
    "temperature__latest", "lactate__latest", "vasopressor_on", "invasive_ventilation_on",
    "norepinephrine_rate", "epinephrine_rate", "dopamine_rate", "dobutamine_rate",
]


def make_row():
    values = [[None] * 21 for _ in range(8)]
    mask = [[False] * 21 for _ in range(8)]
    tslo = [[54.0] * 21 for _ in range(8)]
    for b in range(4, 8):
        for c in range(21):
            if (b + c) % 2 == 0:
                values[b][c] = 10.0 + c
                mask[b][c] = True
                tslo[b][c] = 1.0
            else:
                tslo[b][c] = 6.0 * (b - 4) + 3.0 if b > 4 else 54.0
    return {
        "stay_id": "S1", "prediction_time": "2100-01-01T00:00:00Z", "grid_index": 0,
        "temporal_feature_names": NAMES, "history_values": values, "observation_mask": mask, "tslo_hours": tslo,
        "padding_mask": [True] * 4 + [False] * 4, "delta_sofa_24": 1.0, "organ_support_label": 1.0,
    }


def test_noise_touches_only_observed_clinical_values_and_is_deterministic():
    row = make_row()
    before = copy.deepcopy(row)
    a, b = R.noise_row(row, 0.1, 1), R.noise_row(row, 0.1, 1)
    assert row == before                      # input untouched
    assert a == b                             # seeded
    assert R.noise_row(row, 0.1, 2) != a
    assert a["observation_mask"] == row["observation_mask"] and a["tslo_hours"] == row["tslo_hours"]
    assert a["delta_sofa_24"] == 1.0 and a["organ_support_label"] == 1.0
    for bi in range(8):
        for c in range(21):
            if row["history_values"][bi][c] is None:
                assert a["history_values"][bi][c] is None
            elif c >= 15:
                assert a["history_values"][bi][c] == row["history_values"][bi][c]
    changed = sum(a["history_values"][4][c] != row["history_values"][4][c] for c in range(15) if row["history_values"][4][c] is not None)
    assert changed > 0


def test_extra_padding_removes_oldest_observed_bins_and_resets_dependent_tslo():
    row = make_row()
    out = R.extra_padding_row(row, 2)
    assert out["padding_mask"] == row["padding_mask"]          # bins 0-3 were already padding
    assert R.extra_padding_row(row, 5)["padding_mask"] == [True] * 5 + [False] * 3
    for b in range(2):
        assert all(v is None for v in out["history_values"][b])
        assert not any(out["observation_mask"][b])
        assert all(t == R.TSLO_SENTINEL for t in out["tslo_hours"][b])
    # bins 2..3 were already empty padding; bins 4..7 untouched where the last observation lies inside kept bins
    assert out["history_values"][5] == row["history_values"][5]
    padded = R.extra_padding_row(row, 3)
    assert padded["history_values"][7] == row["history_values"][7]


def test_tslo_pointing_into_removed_bin_becomes_sentinel():
    row = make_row()
    row["observation_mask"][5][0] = False
    row["history_values"][5][0] = None
    row["tslo_hours"][5][0] = 8.0   # last obs 8h before bin-5 end -> inside bin 3 (bins_back=2 -> bin 3)
    out = R.extra_padding_row(row, 4)
    assert out["tslo_hours"][5][0] == R.TSLO_SENTINEL
    out2 = R.extra_padding_row(row, 3)
    assert out2["tslo_hours"][5][0] == 8.0


def test_outage_clears_whole_channels_only():
    row = make_row()
    out = R.outage_row(row, R.LAB_GROUP)
    idx = [NAMES.index(n) for n in R.LAB_GROUP]
    for b in range(8):
        for c in range(21):
            if c in idx:
                assert out["history_values"][b][c] is None and out["observation_mask"][b][c] is False
                assert out["tslo_hours"][b][c] == R.TSLO_SENTINEL
            else:
                assert out["history_values"][b][c] == row["history_values"][b][c]


def test_scenarios_bands_and_subsample_are_deterministic():
    names = [s["name"] for s in R.scenario_list()]
    assert names[0] == "unperturbed" and len(names) == len(set(names)) == 9
    assert [R.cutoff_band(g) for g in (0, 1, 2, 4, 5, 9)] == ["early_0-1"] * 2 + ["middle_2-4"] * 2 + ["late_5+"] * 2
    rows = [{"stay_id": f"S{i}", "prediction_time": "t"} for i in range(50)]
    a = R.stay_subsample(rows, 10, 3)
    assert a == R.stay_subsample(rows, 10, 3) and len(a) == 10
