from datetime import timedelta
from synthetic_feature_helpers import T,builder,context,event

def test_no_observation_is_null_false_and_finite_sentinel_everywhere():
    b=builder(); item=b.build(context([])); f=b.feature_schema.feature_names.index("creatinine__latest")
    assert [row[f] for row in item.history_values]==[None]*8
    assert [row[f] for row in item.observation_mask]==[False]*8
    assert [row[f] for row in item.tslo_hours]==[54.0]*8

def test_tslo_uses_bin_end_and_progresses_without_value_carry_or_reset():
    b=builder(); item=b.build(context([event("heart_rate",80,T-timedelta(hours=47),"one")]))
    f=b.feature_schema.feature_names.index("heart_rate__latest")
    assert [row[f] for row in item.history_values]==[80.0,None,None,None,None,None,None,None]
    assert [row[f] for row in item.observation_mask]==[True,False,False,False,False,False,False,False]
    assert [row[f] for row in item.tslo_hours]==[5.0,11.0,17.0,23.0,29.0,35.0,41.0,47.0]

def test_prelookback_value_cannot_seed_tslo_or_values():
    b=builder(); item=b.build(context([event("heart_rate",250,T-timedelta(hours=60),"old")]))
    f=b.feature_schema.feature_names.index("heart_rate__latest")
    assert all(row[f] is None for row in item.history_values)
    assert all(row[f] is False for row in item.observation_mask)
    assert all(row[f]==54.0 for row in item.tslo_hours)
