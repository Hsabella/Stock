"""数据仓质量检查：每次数据包更新后必跑（update.py 会调用）。

检查项：
1. 复权连续性 —— 复权价日收益超出涨跌停物理边界（|r|>21%）的异常点统计
2. 缺失率 —— csi_union 成员在其成分区间内的收盘价缺失比例
3. akshare 抽样比价 —— qlib 原始价（$close/$factor）vs 新浪不复权日线，误差<0.5%
4. PIT 抽查 —— csi300 必须存在真实的历史调出记录（防"当前快照"假历史）

退出码：0=通过；1=硬失败（比价超差/数据为空/无调出记录）。软异常只告警。

用法:
    python -m quant.data.checks [--sample 20] [--days 20] [--skip-akshare] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

from quant.config import load_config
from quant.data.universe import read_instruments

RET_LIMIT = 0.21          # 超过创业板/科创板 20% 涨跌停的物理边界即视为可疑跳变
CROSSCHECK_TOL = 0.005    # 与 akshare 比价的相对误差容忍


def _init_qlib(cfg: dict):
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=cfg["qlib"]["provider_uri"], region="cn")
    return D


def _sample_codes(cfg: dict, n: int, seed: int) -> list[str]:
    inst_dir = Path(cfg["qlib"]["provider_uri"]) / "instruments"
    union_file = inst_dir / f"{cfg['universe']['output_name']}.txt"
    src = union_file if union_file.exists() else inst_dir / "csi300.txt"
    codes = sorted(read_instruments(src))
    random.Random(seed).shuffle(codes)
    return codes[:n]


def check_continuity(D, codes: list[str], start: str, end: str) -> list[str]:
    """复权价日收益超界统计。返回告警列表（软异常）。"""
    warnings = []
    df = D.features(codes, ["$close"], start_time=start, end_time=end)
    for code, sub in df.groupby(level="instrument"):
        ret = sub["$close"].droplevel(0).pct_change(fill_method=None).abs()
        bad = ret[ret > RET_LIMIT]
        if len(bad):
            days = ", ".join(str(d.date()) for d in bad.index[:3])
            warnings.append(f"{code}: {len(bad)} 个交易日 |收益|>{RET_LIMIT:.0%} (如 {days})，需人工核对是否新股/重组")
    print(f"[checks] 复权连续性: {len(codes)} 只采样, {len(warnings)} 只存在可疑跳变")
    return warnings


def check_missing(D, codes: list[str], start: str, end: str) -> float:
    """采样股票在区间内的收盘价缺失率（相对交易日历）。"""
    cal = D.calendar(start_time=start, end_time=end, freq="day")
    df = D.features(codes, ["$close"], start_time=start, end_time=end)
    counts = df["$close"].dropna().groupby(level="instrument").size()
    # 停牌本身会缺数据，这里只报总量水位，不做硬判定
    rate = 1 - counts.reindex([c.upper() for c in codes]).fillna(0).sum() / (len(cal) * len(codes))
    print(f"[checks] 缺失率(含正常停牌): {rate:.2%}  采样 {len(codes)} 只 × {len(cal)} 交易日")
    return rate


def crosscheck_akshare(D, codes: list[str], days: int) -> tuple[int, int, list[str]]:
    """qlib 原始价 vs akshare 新浪不复权日线抽样比价。返回 (通过只数, 比对只数, 错误)。"""
    import akshare as ak

    errors, ok = [], 0
    checked = 0
    for code in codes:
        sym = code.lower()  # SH600000 -> sh600000，新浪格式
        try:
            ash = ak.stock_zh_a_daily(symbol=sym, adjust="")
            time.sleep(0.4)
        except Exception as e:  # 单只失败不阻断，akshare 偶发限流
            errors.append(f"{code}: akshare 拉取失败 {type(e).__name__}")
            continue
        ash = ash.set_index(pd.to_datetime(ash["date"]))["close"].tail(days)
        q = D.features([code], ["$close", "$factor"], start_time=str(ash.index[0].date()),
                       end_time=str(ash.index[-1].date()))
        if q.empty:
            errors.append(f"{code}: qlib 无对应区间数据")
            continue
        raw = (q["$close"] / q["$factor"]).droplevel(0)
        joined = pd.concat([raw.rename("qlib"), ash.rename("ak")], axis=1).dropna()
        if joined.empty:
            errors.append(f"{code}: 无重叠交易日")
            continue
        checked += 1
        err = (joined["qlib"] / joined["ak"] - 1).abs().median()
        if err <= CROSSCHECK_TOL:
            ok += 1
        else:
            errors.append(f"{code}: 与 akshare 中位相对误差 {err:.2%} 超过 {CROSSCHECK_TOL:.1%}")
    print(f"[checks] akshare 比价: {ok}/{checked} 通过（{len(codes) - checked} 只未能比对）")
    return ok, checked, errors


def check_pit(cfg: dict) -> list[str]:
    """csi300 必须有真实调出记录，展示样例供人工核对。返回硬错误列表。"""
    inst_dir = Path(cfg["qlib"]["provider_uri"]) / "instruments"
    m = read_instruments(inst_dir / "csi300.txt")
    max_end = max(e for ivs in m.values() for _, e in ivs)
    removed = {c: ivs for c, ivs in m.items() if all(e < max_end for _, e in ivs)}
    if not removed:
        return ["csi300 无任何调出记录，疑似当前快照假历史（幸存者偏差）"]
    samples = list(removed.items())[:5]
    print(f"[checks] PIT: csi300 历史调出 {len(removed)} 只，样例:")
    for code, ivs in samples:
        print(f"    {code}: {ivs}")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--skip-akshare", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config("data")
    D = _init_qlib(cfg)
    end = str(D.calendar(freq="day")[-1].date())
    codes = _sample_codes(cfg, args.sample, args.seed)

    hard_errors: list[str] = []
    hard_errors += check_pit(cfg)
    warnings = check_continuity(D, codes, args.start, end)
    check_missing(D, codes, args.start, end)

    if not args.skip_akshare:
        ok, checked, cc_errors = crosscheck_akshare(D, codes[: min(args.sample, 20)], args.days)
        if checked == 0:
            hard_errors.append("akshare 比价一只都没比成（网络/限流），需人工重试")
        elif ok / checked < 0.9:
            hard_errors += cc_errors

    for w in warnings:
        print(f"  [warn] {w}")
    if hard_errors:
        print("\n[checks] 硬失败:")
        for e in hard_errors:
            print(f"  - {e}")
        return 1
    print("\n[checks] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
