"""universe.py 纯逻辑单测（不依赖 qlib/网络）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.data.universe import (  # noqa: E402
    count_active,
    filter_st,
    merge_intervals,
    read_instruments,
    to_qlib_code,
    trim_new_listings,
    union_universe,
    write_instruments,
)


def test_merge_overlapping_and_adjacent():
    ivs = [("2020-01-01", "2020-06-30"), ("2020-07-01", "2020-12-31"), ("2021-06-01", "2021-12-31")]
    assert merge_intervals(ivs) == [("2020-01-01", "2020-12-31"), ("2021-06-01", "2021-12-31")]


def test_merge_keeps_real_gap():
    ivs = [("2020-01-01", "2020-06-30"), ("2020-07-03", "2020-12-31")]
    assert merge_intervals(ivs) == ivs  # 中间空了 2 天，是真实退出


def test_union_seamless_index_switch():
    """同一只股票同日从 csi500 调入 csi300，并集应无缝合并。"""
    csi300 = {"SH600000": [("2021-06-15", "2023-06-14")]}
    csi500 = {"SH600000": [("2019-01-01", "2021-06-14")], "SZ000002": [("2019-01-01", "2023-12-31")]}
    u = union_universe([csi300, csi500])
    assert u["SH600000"] == [("2019-01-01", "2023-06-14")]
    assert count_active(u, "2021-06-15") == 2


def test_trim_new_listings_clips_head():
    calendar = [f"2020-01-{d:02d}" for d in range(1, 32)]  # 简化：连续 31 个"交易日"
    universe = {"SZ300001": [("2020-01-05", "2020-01-31")]}
    listing = {"SZ300001": "2020-01-01"}
    out = trim_new_listings(universe, listing, calendar, min_listed_days=10)
    assert out["SZ300001"] == [("2020-01-11", "2020-01-31")]  # calendar[0+10]


def test_trim_drops_interval_entirely_before_cutoff():
    calendar = [f"2020-01-{d:02d}" for d in range(1, 32)]
    universe = {"SZ300001": [("2020-01-02", "2020-01-05")]}
    listing = {"SZ300001": "2020-01-01"}
    assert trim_new_listings(universe, listing, calendar, min_listed_days=10) == {}


def test_trim_without_listing_info_keeps_asis():
    universe = {"SH600000": [("2020-01-05", "2020-01-31")]}
    out = trim_new_listings(universe, {}, ["2020-01-01"], min_listed_days=10)
    assert out == universe


def test_filter_st():
    u = {"SH600000": [("2020-01-01", "2020-12-31")], "SZ000003": [("2020-01-01", "2020-12-31")]}
    assert set(filter_st(u, {"SZ000003"})) == {"SH600000"}


def test_to_qlib_code():
    assert to_qlib_code("600000") == "SH600000"
    assert to_qlib_code("000001") == "SZ000001"
    assert to_qlib_code("300750") == "SZ300750"
    assert to_qlib_code("688981") == "SH688981"


def test_instruments_roundtrip(tmp_path):
    u = {"SH600000": [("2020-01-01", "2020-12-31"), ("2022-01-01", "2022-12-31")]}
    path = tmp_path / "u.txt"
    write_instruments(u, path)
    assert read_instruments(path) == u
