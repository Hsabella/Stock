"""财报 PIT 因子构建：累计→单季→TTM 的换算、同比负基数、披露截止日可见性。"""
import numpy as np
import pandas as pd

from quant.features.ext_features import _fin_one, build_financials

CAL = pd.bdate_range("2022-01-04", "2024-06-28")


def _cum_stock(periods_roe_profit):
    rows = []
    for period, roe, profit in periods_roe_profit:
        rows.append({"datetime": pd.Timestamp(period), "roe": roe, "net_profit": profit})
    df = pd.DataFrame(rows)
    df["instrument"] = "SH600000"
    return df.set_index(["instrument", "datetime"])


# 2022 年单季 ROE 恒为 2、单季利润恒为 10；2023 年单季 3 / 15；2024Q1 为 4 / 21
FULL = [
    ("2022-03-31", 2, 10), ("2022-06-30", 4, 20), ("2022-09-30", 6, 30), ("2022-12-31", 8, 40),
    ("2023-03-31", 3, 15), ("2023-06-30", 6, 30), ("2023-09-30", 9, 45), ("2023-12-31", 12, 60),
    ("2024-03-31", 4, 21),
]


def test_ttm_yoy_delta_values():
    feat = _fin_one(_cum_stock(FULL).droplevel(0))

    q4_23, q1_24 = pd.Timestamp("2023-12-31"), pd.Timestamp("2024-03-31")
    assert feat.loc[q4_23, "roe_ttm"] == 12          # 3+3+3+3
    assert feat.loc[q1_24, "roe_ttm"] == 13          # 3+3+3+4（跨年差分正确）
    assert np.isclose(feat.loc[q4_23, "profit_yoy"], (60 - 40) / 40)
    assert np.isclose(feat.loc[q1_24, "profit_yoy"], (21 - 15) / 15)
    assert feat.loc[q1_24, "roe_delta"] == 1         # 13 - 12


def test_negative_base_yoy_keeps_sign():
    rows = [("2022-03-31", 1, -10), ("2022-06-30", 2, -8), ("2022-09-30", 3, -5),
            ("2022-12-31", 4, -2), ("2023-03-31", 1, 5)]
    feat = _fin_one(_cum_stock(rows).droplevel(0))
    # 去年同期 -10 → 今年 5：扭亏为盈必须是正的同比
    assert feat.loc[pd.Timestamp("2023-03-31"), "profit_yoy"] == (5 - (-10)) / 10


def test_missing_quarter_gives_nan_not_wrong_value():
    rows = [r for r in FULL if r[0] != "2023-06-30"]
    feat = _fin_one(_cum_stock(rows).droplevel(0))
    assert np.isnan(feat.loc[pd.Timestamp("2023-12-31"), "roe_ttm"])


def test_pit_visibility_on_calendar():
    fin = _cum_stock(FULL)
    feat = build_financials(fin, CAL)
    s = feat.xs("SH600000", level="instrument")["roe_ttm"]

    # 2023 年报（roe_ttm=12）法定截止 2024-04-30，之前只能看到三季报口径（2+2+3+... = 视 Q3 数据）
    before = s.loc["2024-03-01":"2024-04-26"].dropna().unique()
    assert 12 not in before and 13 not in before
    # 4-30 之后年报与一季报（同为 4-30 截止）都已可见，取最新 → 13
    assert (s.loc["2024-05-06":"2024-06-28"] == 13).all()


def test_output_shape():
    feat = build_financials(_cum_stock(FULL), CAL)
    assert list(feat.columns) == ["roe_ttm", "profit_yoy", "roe_delta"]
    assert feat.index.names == ["datetime", "instrument"]
