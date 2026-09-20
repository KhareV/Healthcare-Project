import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from data.synthetic.config import RuntimeConfig, load_runtime_config
from data.synthetic.generator import generate
from data.synthetic.validation import parse_utc
from data.timestamps import RetainedICUStay, generate_prediction_timestamps

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/synthetic/fixtures/phase3_fixture_v1.json"


def loaded(): return load_runtime_config(CONFIG, ROOT)
def generated(config=None): return generate(config or loaded(), ROOT)


def test_generation_has_one_episode_per_subject_and_unique_ids():
    subjects, episodes, events, supports = generated()
    assert len(subjects) == len(episodes) == 4 and not supports
    assert len({x["subject_id"] for x in subjects}) == 4
    assert len({x["stay_id"] for x in episodes}) == 4
    assert len({x["event_id"] for x in events}) == len(events)


def test_all_phase2_concepts_and_units_are_generated():
    _, _, events, _ = generated()
    spec = yaml.safe_load((ROOT / "configs/synthetic/synthetic_generator_v1.yaml").read_text())
    inventory = spec["raw_variable_inventory"]
    assert {x["concept_code"] for x in events} == set(inventory)
    assert all(x["unit"] == inventory[x["concept_code"]]["unit"] for x in events)


def test_events_are_ordered_bounded_and_timezone_aware():
    _, episodes, events, _ = generated(); bounds = {x["stay_id"]:(parse_utc(x["intime"]),parse_utc(x["outtime"])) for x in episodes}
    keys = [(x["stay_id"],x["event_time"],x["event_id"]) for x in events]
    assert keys == sorted(keys)
    assert all(bounds[x["stay_id"]][0] <= parse_utc(x["event_time"]) <= bounds[x["stay_id"]][1] for x in events)


def test_no_exported_latent_target_split_or_prediction_fields():
    tables = generated()
    forbidden = {"latent_state","latent_state_vector","trajectory","target","delta_sofa_24","remaining_stay","support_label","split","prediction_time","grid_index"}
    assert all(not (set(row) & forbidden) for table in tables for row in table)


def test_interval_semantics_only_for_urine_output():
    _,_,events,_=generated()
    for row in events:
        if row["concept_code"] == "urine_output_volume": assert row["interval_start"] and row["interval_end"] == row["event_time"]
        else: assert row["interval_start"] is row["interval_end"] is None


def test_same_config_is_byte_content_deterministic_in_memory():
    assert generated() == generated()


def test_different_seed_changes_content_without_schema_change():
    config=loaded(); values=copy.deepcopy(config.values); values["primary_seed"] += 1
    changed=RuntimeConfig(config.path,values,"different")
    original=generated(config); alternative=generated(changed)
    assert original != alternative
    assert [set(r) for r in original[0]] == [set(r) for r in alternative[0]]


def test_subject_child_stream_is_stable_when_later_subject_added():
    config=loaded(); values=copy.deepcopy(config.values); values["n_subjects"] += 1
    larger=RuntimeConfig(config.path,values,"larger")
    small=generated(config); big=generated(larger)
    assert small[0] == big[0][:4] and small[1] == big[1][:4]
    assert small[2] == [r for r in big[2] if int(r["subject_id"][-8:]) <= 4]


def test_generation_does_not_mutate_numpy_global_rng():
    np.random.seed(912); expected=np.random.random(3); np.random.seed(912); generated(); observed=np.random.random(3)
    assert np.array_equal(expected,observed)


def test_vedant_timestamp_utility_accepts_generated_episodes():
    _,episodes,_,_=generated(); counts=[]
    for row in episodes:
        stay=RetainedICUStay(row["subject_id"],row["stay_id"],parse_utc(row["intime"]),parse_utc(row["outtime"]))
        counts.append(len(generate_prediction_timestamps([stay])))
    assert max(counts) > 0 and len(set(counts)) > 1


def test_raw_history_precondition_is_identity_and_event_time_explicit():
    _,_,events,_=generated()
    assert all({"subject_id","stay_id","event_time","event_id"} <= set(row) for row in events)


def test_no_exact_subject_trajectory_clones():
    subjects,_,events,_=generated(); fps=[]
    for subject in subjects:
        fps.append(json.dumps([(r["concept_code"],r["event_time"],r["value_numeric"]) for r in events if r["subject_id"]==subject["subject_id"]]))
    assert len(fps) == len(set(fps))
