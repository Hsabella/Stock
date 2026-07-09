"""本地 parquet 数仓：qlib bin 之外的扩展数据（换手率/估值/资金流）统一落这里。

存储约定：每个 dataset 一个 parquet，长表格式，列含 instrument / datetime + 数据列。
append 语义：新数据与旧数据按 (instrument, datetime) 去重，保留最新一次写入。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.config import load_config


def warehouse_dir() -> Path:
    path = Path(load_config("data")["warehouse"]["path"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(name: str) -> Path:
    return warehouse_dir() / f"{name}.parquet"


def load(name: str) -> pd.DataFrame:
    """读 dataset；不存在返回空表。返回值以 (instrument, datetime) 为索引。"""
    path = dataset_path(name)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["instrument", "datetime"])
    return df.sort_index()


def append(name: str, new: pd.DataFrame) -> int:
    """追加写入并按索引去重（保留新值）。返回写入后的总行数。

    new 需含 instrument / datetime 列（或已是该 MultiIndex）。
    """
    if new.empty:
        return len(load(name))
    if not isinstance(new.index, pd.MultiIndex):
        new = new.copy()
        new["datetime"] = pd.to_datetime(new["datetime"])
        new = new.set_index(["instrument", "datetime"])

    old = load(name)
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(dataset_path(name))
    return len(merged)


def coverage(name: str) -> str:
    """人读的覆盖率摘要，用于拉数进度检查。"""
    df = load(name)
    if df.empty:
        return f"{name}: 空"
    n_inst = df.index.get_level_values(0).nunique()
    dts = df.index.get_level_values(1)
    return f"{name}: {n_inst} 只 × [{dts.min().date()} → {dts.max().date()}], {len(df):,} 行"
