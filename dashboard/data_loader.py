# dashboard/data_loader.py
"""唯一读数据层。只读 results/ 与 watchlist.yaml, 绝不重算引擎。
非 streamlit 环境下 @st.cache_data 退化为普通函数, 便于单测。
"""
import glob
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEC_DIR = os.path.join(REPO, "results", "decisions")
FWD_DIR = os.path.join(REPO, "results", "forward")

try:
    import streamlit as st
    cache = st.cache_data
except Exception:  # 单测/无 streamlit 时
    def cache(func=None, **_):
        return func if func else (lambda f: f)

JUNK_COLS = {"1", "_no_data", "Unnamed: 0"}


@cache
def list_scan_dates():
    dates = []
    for p in glob.glob(os.path.join(DEC_DIR, "partial_*.csv")):
        m = re.search(r"partial_(\d{8})\.csv$", p)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@cache
def list_forward_dates():
    dates = []
    for p in glob.glob(os.path.join(FWD_DIR, "forward_*.csv")):
        m = re.search(r"forward_(\d{8})\.csv$", p)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@cache
def load_partial_csv(date):
    return pd.read_csv(os.path.join(DEC_DIR, f"partial_{date}.csv"))


@cache
def load_partial_json(date):
    with open(os.path.join(DEC_DIR, f"partial_{date}.json")) as f:
        return json.load(f)


@cache
def load_partial_md(date):
    p = os.path.join(DEC_DIR, f"partial_{date}.md")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


@cache
def load_forward_csv(date):
    df = pd.read_csv(os.path.join(FWD_DIR, f"forward_{date}.csv"))
    return df.drop(columns=[c for c in df.columns if c in JUNK_COLS], errors="ignore")


@cache
def parse_csi_benchmark(date, horizon=1):
    """从 forward md 头部抓 'T+1=-0.45%' 返回小数(-0.0045)。"""
    p = os.path.join(FWD_DIR, f"forward_{date}.md")
    if not os.path.exists(p):
        return None
    m = re.search(rf"T\+{horizon}=([-+]?\d+\.?\d*)%", open(p, encoding="utf-8").read())
    return float(m.group(1)) / 100 if m else None


@cache
def load_factor_ic():
    p = os.path.join(REPO, "results", "factor_ic.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["date", "factor", "ic", "n"])
