"""CSRankNormExempt：豁免列保原值，其余列与 qlib CSRankNorm 同款截面排名。"""
import numpy as np
import pandas as pd

from quant.features.handler import CSRankNormExempt

DATES = pd.to_datetime(["2024-01-05"] * 4 + ["2024-01-08"] * 4)
INST = ["A", "B", "C", "D"] * 2


def _panel():
    idx = pd.MultiIndex.from_arrays([DATES, INST], names=["datetime", "instrument"])
    return pd.DataFrame({
        ("feature", "f1"): [1.0, 2, 3, 4, 4, 3, 2, 1],
        ("feature", "mkt_vol_20"): [0.5] * 4 + [0.7] * 4,
        ("label", "LABEL0"): np.arange(8.0),
    }, index=idx)


def test_exempt_column_kept_raw_others_ranked():
    out = CSRankNormExempt(fields_group="feature", exempt=["mkt_vol_20"])(_panel())

    assert (out[("feature", "mkt_vol_20")].to_numpy() == [0.5] * 4 + [0.7] * 4).all()
    got = np.sort(out[("feature", "f1")].xs("2024-01-05", level="datetime").to_numpy())
    expect = np.sort((np.array([1, 2, 3, 4]) / 4 - 0.5) * 3.46)
    assert np.allclose(got, expect)
    assert (out[("label", "LABEL0")].to_numpy() == np.arange(8.0)).all()


def test_empty_exempt_equals_plain_csranknorm():
    from qlib.data.dataset.processor import CSRankNorm

    ours = CSRankNormExempt(fields_group="feature")(_panel())
    qlibs = CSRankNorm(fields_group="feature")(_panel())
    assert np.allclose(ours[("feature", "f1")], qlibs[("feature", "f1")])
