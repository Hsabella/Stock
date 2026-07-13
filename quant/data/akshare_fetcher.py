"""akshare 扩展数据批量拉取：断点续拉 + 限流退避 + 失败清单。

四个 dataset（拉取经验移植自 factors/*/data.py：东财限流 → 新浪/百度替代）：
- sina_daily      新浪日线全历史 → turnover(换手率) / outstanding_share(流通股本)
- baidu_valuation 百度估值全历史 → pe_ttm / pb（长周期为稀疏采样点，用时需 ffill）
- em_fundflow     东财个股资金流，**只有最近 ~120 天**——无法回测，本 dataset 仅做
                  滚动累积（每周跑一次 append 攒历史），资金流因子留待二批评估
- financials      财务摘要(ROE/归母净利润, v2 Step2)。datetime = **报告期**（每股唯一，
                  年报与次年一季报的披露截止日相同，用披露日做行键会互相覆盖）；
                  PIT 由 ext_features 构建时经 pit_available_date(报告期→法定披露
                  截止日：一季报/年报 4-30、半年报 8-31、三季报 10-31) 实现零前视
                  （比计划的 +30/60/90 更保守，年报 +90 早于法定截止会踩前视）

用法:
    python -m quant.data.akshare_fetcher --dataset sina_daily [--since 2018-01-01] [--limit 100]
    python -m quant.data.akshare_fetcher --dataset baidu_valuation
    python -m quant.data.akshare_fetcher --dataset em_fundflow --refresh
    python -m quant.data.akshare_fetcher --dataset financials

断点：warehouse 目录下 <dataset>.done.txt 记录已完成代码，重跑自动跳过（针对单次
中断续拉）。周期性增量数据集（em_fundflow/financials）定期重跑时要加 --refresh
清掉断点，否则全部代码被视作已完成而空转。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from quant.config import load_config
from quant.data import warehouse
from quant.data.universe import read_instruments

THROTTLE_SEC = 0.4
MAX_RETRY = 3


def universe_codes(since: str) -> list[str]:
    """csi_union 中成分区间与 [since, 今天] 有交集的代码（回测窗口内出现过的都要）。"""
    cfg = load_config("data")
    path = Path(cfg["qlib"]["provider_uri"]) / "instruments" / f"{cfg['universe']['output_name']}.txt"
    m = read_instruments(path)
    return sorted(code for code, ivs in m.items() if any(end >= since for _, end in ivs))


# ---------- 各数据源的单股拉取（返回带 instrument/datetime 列的长表） ----------

def fetch_sina_daily(code: str, since: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_a_daily(symbol=code.lower(), adjust="")
    df["datetime"] = pd.to_datetime(df["date"])
    df = df[df["datetime"] >= since]
    out = df[["datetime", "turnover", "outstanding_share", "close"]].copy()
    out.rename(columns={"close": "raw_close"}, inplace=True)
    out["instrument"] = code
    return out


def fetch_baidu_valuation(code: str, since: str) -> pd.DataFrame:
    import akshare as ak

    plain = code[2:]  # SH600000 -> 600000
    frames = {}
    for col, indicator in [("pe_ttm", "市盈率(TTM)"), ("pb", "市净率")]:
        df = ak.stock_zh_valuation_baidu(symbol=plain, indicator=indicator, period="全部")
        df["datetime"] = pd.to_datetime(df["date"])
        frames[col] = df.set_index("datetime")["value"].rename(col)
        time.sleep(THROTTLE_SEC)
    out = pd.concat(frames.values(), axis=1).reset_index()
    out = out[out["datetime"] >= since]
    out["instrument"] = code
    return out


def fetch_em_fundflow(code: str, since: str) -> pd.DataFrame:
    import akshare as ak

    plain, market = code[2:], code[:2].lower()
    df = ak.stock_individual_fund_flow(stock=plain, market=market)
    df["datetime"] = pd.to_datetime(df["日期"])
    out = df[["datetime", "主力净流入-净额", "主力净流入-净占比"]].copy()
    out.columns = ["datetime", "main_net_inflow", "main_net_inflow_pct"]
    out = out[out["datetime"] >= since]
    out["instrument"] = code
    return out


def pit_available_date(period: pd.Timestamp) -> pd.Timestamp:
    """报告期 → 法定披露截止日（该日起视为全市场可见）。"""
    if period.month == 3:
        return period.replace(month=4, day=30)
    if period.month == 6:
        return period.replace(month=8, day=31)
    if period.month == 9:
        return period.replace(month=10, day=31)
    return pd.Timestamp(year=period.year + 1, month=4, day=30)


def fetch_financials(code: str, since: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_financial_abstract(symbol=code[2:])
    idx_col = "指标" if "指标" in df.columns else df.columns[0]
    df = df.set_index(idx_col)
    periods = sorted(c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 8)
    if not periods:
        return pd.DataFrame()

    def pick(keywords: list[str]) -> pd.Series:
        for kw in keywords:
            hits = [i for i in df.index if isinstance(i, str) and kw in i]
            if hits:
                row = df.loc[hits[0]]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return pd.to_numeric(row[periods], errors="coerce")
        return pd.Series(index=periods, dtype=float)

    out = pd.DataFrame({
        "datetime": pd.to_datetime(periods),  # 行键 = 报告期
        "roe": pick(["净资产收益率(ROE)", "摊薄净资产收益率"]).to_numpy(),
        "net_profit": pick(["归母净利润", "归属于母公司股东的净利润"]).to_numpy(),
    })
    # 留 ~400 天余量：profit_yoy 同比要用上一年同期
    cutoff = pd.Timestamp(since) - pd.Timedelta(days=400)
    out = out[out["datetime"] >= cutoff].dropna(subset=["roe", "net_profit"], how="all")
    out["instrument"] = code
    return out


FETCHERS = {
    "sina_daily": fetch_sina_daily,
    "baidu_valuation": fetch_baidu_valuation,
    "em_fundflow": fetch_em_fundflow,
    "financials": fetch_financials,
}


# ---------- 批量循环：断点/退避/失败清单 ----------

def _done_file(dataset: str) -> Path:
    return warehouse.warehouse_dir() / f"{dataset}.done.txt"


def _load_done(dataset: str) -> set[str]:
    f = _done_file(dataset)
    return set(f.read_text().split()) if f.exists() else set()


def run(dataset: str, since: str, limit: int | None = None, refresh: bool = False) -> int:
    fetch_one = FETCHERS[dataset]
    codes = universe_codes(since)
    if refresh:
        _done_file(dataset).unlink(missing_ok=True)
        print(f"[fetch:{dataset}] --refresh: 断点已清，全池重拉增量")
    done = _load_done(dataset)
    todo = [c for c in codes if c not in done]
    if limit:
        todo = todo[:limit]
    print(f"[fetch:{dataset}] 全池 {len(codes)} 只, 已完成 {len(done)}, 本次 {len(todo)}")

    failed: list[str] = []
    buffer: list[pd.DataFrame] = []
    for i, code in enumerate(todo, 1):
        for attempt in range(1, MAX_RETRY + 1):
            try:
                df = fetch_one(code, since)
                if not df.empty:
                    buffer.append(df)
                done.add(code)
                break
            except Exception as e:
                if attempt == MAX_RETRY:
                    failed.append(f"{code}\t{type(e).__name__}: {str(e)[:80]}")
                else:
                    time.sleep(THROTTLE_SEC * 4 * attempt)  # 退避后重试
        time.sleep(THROTTLE_SEC)

        if i % 100 == 0 or i == len(todo):
            if buffer:
                total = warehouse.append(dataset, pd.concat(buffer, ignore_index=True))
                buffer.clear()
                _done_file(dataset).write_text("\n".join(sorted(done)))
                print(f"[fetch:{dataset}] 进度 {i}/{len(todo)}, 仓内 {total:,} 行, 失败 {len(failed)}", flush=True)

    if failed:
        fail_file = warehouse.warehouse_dir() / f"{dataset}.failed.txt"
        fail_file.write_text("\n".join(failed))
        print(f"[fetch:{dataset}] {len(failed)} 只失败, 清单见 {fail_file}（重跑本命令自动重试）")
    print(f"[fetch:{dataset}] 完成。{warehouse.coverage(dataset)}")
    return 0 if not failed else 1


def main() -> int:
    import socket

    socket.setdefaulttimeout(60)  # akshare 底层 requests 多不带 timeout，半死连接会挂住 cron
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(FETCHERS))
    parser.add_argument("--since", default="2018-01-01")
    parser.add_argument("--limit", type=int, default=None, help="只拉前 N 只（试跑用）")
    parser.add_argument("--refresh", action="store_true",
                        help="清掉断点全池重拉（em_fundflow/financials 周期性增量必加）")
    args = parser.parse_args()
    return run(args.dataset, args.since, args.limit, args.refresh)


if __name__ == "__main__":
    sys.exit(main())
