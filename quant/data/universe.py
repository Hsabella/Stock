"""PIT（point-in-time）股票池构建：沪深300 + 中证500 + 中证1000 成分并集。

核心原则：回测在任一历史时点只能"看到"当时真实的成分股名单（instruments
文件的 [调入日, 调出日] 区间就是事实表），从而避免幸存者偏差。

本模块职责：
1. 校验三个指数的 instruments 历史是否齐全（csi1000 缺失时给出降级提示）
2. 合成并集 csi_union（区间合并，同日 300↔500 切换无缝拼接）
3. 次新过滤（上市不足 N 个交易日的区间掐头）+ ST 过滤（当前名单，见文档局限）
4. 写回 qlib instruments 目录，供 D.instruments('csi_union') 使用

用法:
    python -m quant.data.universe          # 校验 + 构建 + 覆盖率报告
    python -m quant.data.universe --check-only
"""
from __future__ import annotations

import argparse
import sys
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

from quant.config import load_config

Interval = tuple[str, str]  # (start, end) ISO 日期字符串，含两端


# ---------- instruments 文件读写 ----------

def read_instruments(path: Path) -> dict[str, list[Interval]]:
    """读 qlib instruments 文件（CODE\\tSTART\\tEND 每行一个成分区间）。"""
    result: dict[str, list[Interval]] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code, start, end = parts[0].upper(), parts[1][:10], parts[2][:10]
        result.setdefault(code, []).append((start, end))
    return result


def write_instruments(universe: dict[str, list[Interval]], path: Path) -> None:
    lines = [
        f"{code}\t{start}\t{end}"
        for code in sorted(universe)
        for start, end in sorted(universe[code])
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------- 区间运算 ----------

def _next_day(iso: str) -> str:
    return (date.fromisoformat(iso) + timedelta(days=1)).isoformat()


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """合并重叠或首尾相接（自然日相邻）的区间；真实退出产生的空档保留。"""
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= _next_day(merged[-1][1]):
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def union_universe(index_maps: list[dict[str, list[Interval]]]) -> dict[str, list[Interval]]:
    combined: dict[str, list[Interval]] = {}
    for m in index_maps:
        for code, ivs in m.items():
            combined.setdefault(code, []).extend(ivs)
    return {code: merge_intervals(ivs) for code, ivs in combined.items()}


# ---------- 过滤 ----------

def trim_new_listings(
    universe: dict[str, list[Interval]],
    listing_dates: dict[str, str],
    calendar: list[str],
    min_listed_days: int,
) -> dict[str, list[Interval]]:
    """把每只股票上市后前 min_listed_days 个交易日从成分区间里掐掉。"""
    result: dict[str, list[Interval]] = {}
    for code, ivs in universe.items():
        listed = listing_dates.get(code)
        if listed is None:
            result[code] = ivs  # 无上市信息时不动，checks 层面另行统计
            continue
        idx = bisect_left(calendar, listed)
        cutoff_idx = idx + min_listed_days
        cutoff = calendar[cutoff_idx] if cutoff_idx < len(calendar) else "9999-12-31"
        kept = [(max(start, cutoff), end) for start, end in ivs if end >= cutoff]
        if kept:
            result[code] = kept
    return result


def filter_st(universe: dict[str, list[Interval]], st_codes: set[str]) -> dict[str, list[Interval]]:
    return {code: ivs for code, ivs in universe.items() if code not in st_codes}


def to_qlib_code(plain: str) -> str:
    """'600000' → 'SH600000'；0/2/3 开头 → SZ，4/8/9 开头（北交所/B股）原样带 BJ 标记。"""
    if plain.startswith(("6", "9", "5")):
        return f"SH{plain}"
    if plain.startswith(("0", "2", "3")):
        return f"SZ{plain}"
    return f"BJ{plain}"


def fetch_current_st_codes(cache_path: Path, refresh: bool = False) -> set[str]:
    """当前名称含 ST 的股票（akshare 全 A 名录，缓存到本地 CSV）。

    局限：历史时点的 ST 状态拿不到 point-in-time 数据，这里只能剔除"现在是
    ST"的股票。方向上属保守剔除，已在 data_dictionary.md 记录。
    """
    import pandas as pd

    if cache_path.exists() and not refresh:
        return set(pd.read_csv(cache_path)["code"].astype(str))

    import akshare as ak

    df = ak.stock_info_a_code_name()
    st = df[df["name"].str.contains("ST", na=False)]
    codes = {to_qlib_code(str(c).zfill(6)) for c in st["code"]}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": sorted(codes)}).to_csv(cache_path, index=False)
    return codes


# ---------- 校验与报告 ----------

def count_active(index_map: dict[str, list[Interval]], on: str) -> int:
    return sum(any(start <= on <= end for start, end in ivs) for ivs in index_map.values())


EXPECTED_SIZE = {"csi300": (270, 330), "csi500": (450, 550), "csi1000": (900, 1100)}


def validate_index_files(inst_dir: Path, indexes: list[str], probe_dates: list[str]) -> list[str]:
    """校验各指数 instruments 是否存在、规模是否符合预期。返回错误列表。"""
    errors: list[str] = []
    for name in indexes:
        path = inst_dir / f"{name}.txt"
        if not path.exists():
            errors.append(f"{name}: instruments 文件缺失 ({path})")
            continue
        m = read_instruments(path)
        starts = [s for ivs in m.values() for s, _ in ivs]
        ends = [e for ivs in m.values() for _, e in ivs]
        removed = sum(all(e < max(ends) for _, e in ivs) for ivs in m.values())
        print(f"[universe] {name}: {len(m)} 只历史成分, 区间 {min(starts)} → {max(ends)}, 已调出 {removed} 只")
        if removed == 0:
            errors.append(f"{name}: 无任何调出记录，疑似只有当前快照（非 PIT 历史），会引入幸存者偏差")
        lo, hi = EXPECTED_SIZE.get(name, (0, 10**6))
        for d in probe_dates:
            n = count_active(m, d)
            status = "OK" if lo <= n <= hi else "!! 异常"
            print(f"    {d} 时点成分数: {n} ({status}, 预期 {lo}-{hi})")
            if not lo <= n <= hi:
                errors.append(f"{name}: {d} 时点成分数 {n} 超出预期 [{lo},{hi}]")
    return errors


def build(cfg: dict, check_only: bool = False) -> int:
    provider_uri = Path(cfg["qlib"]["provider_uri"])
    inst_dir = provider_uri / "instruments"
    ucfg = cfg["universe"]
    probe_dates = ["2020-06-30", "2023-06-30", "2026-06-30"]

    errors = validate_index_files(inst_dir, ucfg["indexes"], probe_dates)
    if errors:
        print("\n[universe] 校验未通过:")
        for e in errors:
            print(f"  - {e}")
        print("  降级路径: ① qlib cn_index collector 解析中证官网公告 ② tushare index_weight"
              " ③ 从 data.yaml universe.indexes 移除该指数（如降级为 300+500）")
        return 1
    if check_only:
        print("[universe] 校验通过（check-only）")
        return 0

    index_maps = [read_instruments(inst_dir / f"{n}.txt") for n in ucfg["indexes"]]
    universe = union_universe(index_maps)
    print(f"[universe] 并集: {len(universe)} 只历史成分")

    calendar = [line[:10] for line in (provider_uri / "calendars" / "day.txt").read_text().splitlines() if line.strip()]
    listing = {code: min(s for s, _ in ivs) for code, ivs in read_instruments(inst_dir / "all.txt").items()} \
        if (inst_dir / "all.txt").exists() else {}
    universe = trim_new_listings(universe, listing, calendar, ucfg["min_listed_days"])
    print(f"[universe] 次新掐头后: {len(universe)} 只（上市信息覆盖 {len(listing)} 只）")

    if ucfg.get("exclude_st"):
        st = fetch_current_st_codes(Path(ucfg["st_cache"]))
        before = len(universe)
        universe = filter_st(universe, st)
        print(f"[universe] ST 剔除: {before - len(universe)} 只")

    out = inst_dir / f"{ucfg['output_name']}.txt"
    write_instruments(universe, out)
    for d in probe_dates:
        print(f"[universe] csi_union {d} 时点成分数: {count_active(universe, d)}")
    print(f"[universe] 已写入 {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只校验指数文件，不写 csi_union")
    args = parser.parse_args()
    return build(load_config("data"), check_only=args.check_only)


if __name__ == "__main__":
    sys.exit(main())
