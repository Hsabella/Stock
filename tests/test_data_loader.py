# tests/test_data_loader.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import data_loader as dl


def test_list_scan_dates_desc():
    dates = dl.list_scan_dates()
    assert "20260529" in dates
    assert dates == sorted(dates, reverse=True)  # 倒序


def test_load_old_43col_file_ok():
    df = dl.load_partial_csv("20260510")  # 仅 43 列, 缺 sector_rank
    assert "decision" in df.columns
    assert "sector_rank" not in df.columns  # 容错: 不强造列


def test_forward_drops_junk_cols():
    df = dl.load_forward_csv("20260521")  # 含 '1'/'_no_data' 垃圾列
    assert "1" not in df.columns and "_no_data" not in df.columns
    assert "ret_T+1" in df.columns


def test_parse_csi_benchmark():
    val = dl.parse_csi_benchmark("20260528")  # md 头: T+1=-0.45%
    assert val is not None and abs(val - (-0.0045)) < 1e-6
