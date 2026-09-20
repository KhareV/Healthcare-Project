from datetime import timedelta
import pytest
from synthetic_feature_helpers import T,builder,context,event

def hr_index(b): return b.feature_schema.feature_names.index("heart_rate__latest")

def test_exact_lookback_edges_and_all_right_closed_bin_boundaries():
    b=builder(); eps=timedelta(microseconds=1); rows=[event("heart_rate",999,T-timedelta(hours=48),"excluded"),event("heart_rate",10,T-timedelta(hours=48)+eps,"inside")]
    for index,hours in enumerate((42,36,30,24,18,12,6,0)):
        rows.append(event("heart_rate",20+index,T-timedelta(hours=hours),f"edge-{index}"))
    item=b.build(context(rows)); f=hr_index(b)
    assert [row[f] for row in item.history_values]==[20,21,22,23,24,25,26,27]
    assert all(row[f] for row in item.observation_mask)

def test_t_plus_epsilon_and_60h_old_extremes_have_exactly_zero_effect():
    b=builder(); baseline=b.build(context([event("heart_rate",70,T-timedelta(hours=1),"valid")]))
    hostile=[event("heart_rate",70,T-timedelta(hours=1),"valid"),event("heart_rate",250,T+timedelta(microseconds=1),"future"),event("heart_rate",250,T-timedelta(hours=60),"old")]
    assert b.build(context(hostile))==baseline

def test_earliest_cutoff_has_four_prefix_padding_bins_and_exact_intime_is_excluded():
    intime=T-timedelta(hours=24); b=builder(); item=b.build(context([event("heart_rate",80,intime,"at-intime")],intime=intime))
    assert item.padding_mask==(True,True,True,True,False,False,False,False)
    assert not any(any(row) for row in item.observation_mask)
    assert all(value==54.0 for row in item.tslo_hours for value in row)

def test_padding_differs_from_active_missingness():
    item=builder().build(context([],intime=T-timedelta(hours=24)))
    assert item.padding_mask[3] is True and item.padding_mask[4] is False
    assert item.history_values[3]==item.history_values[4]

def test_row_order_and_other_stay_subject_are_irrelevant():
    b=builder(); valid=[event("heart_rate",60,T-timedelta(hours=2),"a"),event("heart_rate",70,T-timedelta(hours=1),"b")]
    attack=[event("heart_rate",250,T-timedelta(hours=1),"x",stay="B",subject="X")]
    assert b.build(context(valid))==b.build(context(list(reversed(valid))+attack))

def test_latest_tie_break_and_urine_sum_are_deterministic():
    b=builder(); when=T-timedelta(hours=1); rows=[event("heart_rate",60,when,"a"),event("heart_rate",70,when,"z"),event("urine_output_volume",100,when,"u1"),event("urine_output_volume",150,when-timedelta(hours=1),"u2")]
    item=b.build(context(rows)); assert item.history_values[7][b.feature_schema.feature_names.index("heart_rate__latest")]==70
    assert item.history_values[7][b.feature_schema.feature_names.index("urine_output_volume__sum")]==250

def test_illegal_episode_relative_cutoff_identity_fails_closed():
    from dataclasses import replace
    from features.synthetic import FeatureBuilderError
    with pytest.raises(FeatureBuilderError,match="episode-relative"):
        builder().build(replace(context(),icu_elapsed_hours=66,grid_index=7))

def test_relevant_unknown_concept_and_wrong_unit_fail_closed():
    b=builder(); row=event("heart_rate",80,T-timedelta(hours=1),"x")
    unknown=dict(row,canonical_concept="latent_state")
    with pytest.raises(Exception,match="unsupported"):
        b.build(context([unknown]))
    wrong=dict(row,unit="future_units")
    with pytest.raises(Exception,match="unit/provenance"):
        b.build(context([wrong]))
