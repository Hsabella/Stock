"""neutralize_frame 的截面正确性：残差对市值/行业正交、缺数据不掉股票。"""
import numpy as np
import pandas as pd
import pytest

from quant.features.neutralize import neutralize_frame

DATES = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])


def _fake_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    inst = pd.Index([f"SH{600000 + i}" for i in range(n)], name="instrument")
    ind_map = pd.Series(np.take(["801080", "801150", "801780"], np.arange(n) % 3), index=inst)
    ind_effect = ind_map.map({"801080": 2.0, "801150": -1.0, "801780": 0.5})

    scores, sizes = [], []
    for dt in DATES:
        lm = pd.Series(rng.normal(10.0, 1.0, n), index=inst)
        y = ind_effect + 0.8 * lm + rng.normal(0, 0.1, n)
        idx = pd.MultiIndex.from_product([[dt], inst], names=["datetime", "instrument"])
        scores.append(pd.Series(y.to_numpy(), index=idx, name="score"))
        sizes.append(pd.Series(lm.to_numpy(), index=idx))
    return pd.concat(scores), ind_map, pd.concat(sizes)


def test_residual_orthogonal_to_size_and_industry():
    score, ind_map, log_mktcap = _fake_data()
    res = neutralize_frame(score, ind_map, log_mktcap)

    assert res.index.equals(score.index.sortlevel()[0])
    assert not res.isna().any()
    for dt in DATES:
        cross = res.xs(dt, level="datetime")
        lm = log_mktcap.xs(dt, level="datetime")
        assert abs(np.corrcoef(cross, lm)[0, 1]) < 0.05
        group_means = cross.groupby(ind_map).mean()
        assert group_means.abs().max() < 0.05


def test_missing_industry_and_size_keep_all_stocks():
    score, ind_map, log_mktcap = _fake_data()
    ind_map = ind_map.iloc[:-10]                      # 10 只无行业 → UNK 桶
    log_mktcap = log_mktcap.mask(                     # 每截面 10 只无市值 → 中位数填
        log_mktcap.index.get_level_values("instrument").isin(ind_map.index[:10]))

    res = neutralize_frame(score, ind_map, log_mktcap)
    assert len(res) == len(score)
    assert not res.isna().any()


def test_all_size_missing_degrades_to_industry_only():
    score, ind_map, log_mktcap = _fake_data()
    res = neutralize_frame(score, ind_map, log_mktcap * np.nan)
    assert not res.isna().any()
    cross = res.xs(DATES[0], level="datetime")
    assert cross.groupby(ind_map).mean().abs().max() < 0.05


def test_group_dispersion_equalized():
    score, ind_map, log_mktcap = _fake_data()
    ind_map.iloc[:40] = pd.NA                          # 40 只落 UNK 大桶
    res = neutralize_frame(score, ind_map, log_mktcap)
    for dt in DATES:
        cross = res.xs(dt, level="datetime")
        stds = cross.groupby(ind_map.reindex(cross.index).fillna("UNK")).std()
        assert stds.max() / stds.min() < 1.05          # 各组散度归一 → 无组能靠方差占领头部


def test_changes_ranking():
    score, ind_map, log_mktcap = _fake_data()
    res = neutralize_frame(score, ind_map, log_mktcap)
    raw_top = score.xs(DATES[0], level="datetime").nlargest(30).index
    new_top = res.xs(DATES[0], level="datetime").nlargest(30).index
    assert set(raw_top) != set(new_top)
