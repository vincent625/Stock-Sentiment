import pandas as pd

from qqq_sentiment.analysis import reaction_time_metrics


def test_reaction_time_metrics():
    curve = pd.DataFrame(
        {
            "horizon_minutes": [1, 5, 15, 30, 60],
            "spread_car": [0.001, 0.004, 0.007, 0.009, 0.010],
        }
    )
    result = reaction_time_metrics(curve)
    assert result["T50_min"] == 15
    assert result["T80_min"] == 30
