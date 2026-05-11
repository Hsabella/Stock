#!/usr/bin/env python3
"""仅拉取 + 累积新闻流, 不跑完整决策引擎.

用途: 盘中每小时跑一次, 在 history.csv 里累积更丰富的新闻样本,
后续日终的 run_partial_engine 就能拿到 3 天滑窗内更密集的数据.

crontab 推荐: 0 10-15 * * 1-5 /path/to/python3 scripts/fetch_news_only.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from factors.news.data import fetch_global_news, append_to_history

if __name__ == "__main__":
    df = fetch_global_news()
    added = append_to_history(df)
    print(f"fetched={len(df)} new={added}")
