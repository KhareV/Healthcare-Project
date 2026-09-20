from datetime import datetime,timedelta,timezone
import inspect,math
import pytest
from labels.icu_time import ICUTimeLabelError,remaining_episode_time_label

T=datetime(2100,1,1,tzinfo=timezone.utc)

@pytest.mark.parametrize("hours",[1,10,24,10.25])
def test_exact_hours_log1p_and_inverse(hours):
    result=remaining_episode_time_label(prediction_time=T,outtime=T+timedelta(hours=hours),icu_time_temporally_eligible=True)
    assert result.remaining_hours==hours
    assert result.log1p_remaining_hours==pytest.approx(math.log1p(hours))
    assert math.expm1(result.log1p_remaining_hours)==pytest.approx(hours)

def test_positive_eligibility_timezone_and_signature_contract():
    for hours in (0,-1):
        with pytest.raises(ICUTimeLabelError,match="positive"):
            remaining_episode_time_label(prediction_time=T,outtime=T+timedelta(hours=hours),icu_time_temporally_eligible=True)
    with pytest.raises(ICUTimeLabelError,match="eligible"):
        remaining_episode_time_label(prediction_time=T,outtime=T+timedelta(hours=6),icu_time_temporally_eligible=False)
    with pytest.raises(ICUTimeLabelError,match="timezone"):
        remaining_episode_time_label(prediction_time=T.replace(tzinfo=None),outtime=T+timedelta(hours=6),icu_time_temporally_eligible=True)
    assert set(inspect.signature(remaining_episode_time_label).parameters)=={"prediction_time","outtime","icu_time_temporally_eligible"}
