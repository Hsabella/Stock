"""行业标签：股票 → 行业组，落 warehouse industry.parquet 快照（v2 Step1 中性化底座）。

口径与来源：
- 主体：申万一级。官网成分接口 2026-07 起对行业指数返回空（仅申万50 等发布指数
  有数据），保留 fetch_sw1_members 以待恢复；实际走旧引擎 SW2 缓存
  （cache/sector_momentum/symbol_to_sw2.csv，sector_momentum 维度每周刷新）
  + 二级→一级映射。2021 版申万重编码后"去尾归十"有 22 个例外，
  显式列在 SW2_TO_SW1_EXCEPTIONS（按板块名称逐一核对）。
- 补缺：SW2 缓存有 10 个一级行业整体缺席（银行/电力设备/煤炭等，当日板块拉取
  失败所致），用新浪行业(49 板块)给缺标签的股票兜底，组码前缀 S_ 与申万组
  独立共存——中性化只要求分组一致，不要求同一分类体系。
- 仍无标签的（主要是已退市股，快照源只覆盖当前上市）由 neutralize 归 UNK 桶。

行业归属只有当前快照、无法回溯——行业变更极少，用最近快照近似历史可接受；
重跑本命令累积快照，load_map 按 as-of 取目标日之前最近一份（无则用最早一份）。

用法: python -m quant.data.industry
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from quant.data import warehouse

DATASET = "industry"
SW2_CACHE = Path(__file__).resolve().parents[2] / "cache" / "sector_momentum" / "symbol_to_sw2.csv"
THROTTLE_SEC = 0.4
MAX_RETRY = 3

SW1_NAMES = {
    "801010": "农林牧渔", "801030": "基础化工", "801040": "钢铁", "801050": "有色金属",
    "801080": "电子", "801110": "家用电器", "801120": "食品饮料", "801130": "纺织服饰",
    "801140": "轻工制造", "801150": "医药生物", "801160": "公用事业", "801170": "交通运输",
    "801180": "房地产", "801200": "商贸零售", "801210": "社会服务", "801230": "综合",
    "801710": "建筑材料", "801720": "建筑装饰", "801730": "电力设备", "801740": "国防军工",
    "801750": "计算机", "801760": "传媒", "801770": "通信", "801780": "银行",
    "801790": "非银金融", "801880": "汽车", "801890": "机械设备", "801950": "煤炭",
    "801960": "石油石化", "801970": "环保", "801980": "美容护理",
}

SW2_TO_SW1_EXCEPTIONS = {
    "801092": "801880", "801093": "801880", "801095": "801880", "801096": "801880",  # 汽车
    "801072": "801890", "801074": "801890", "801076": "801890",                       # 机械设备
    "801077": "801890", "801078": "801890",
    "801191": "801790", "801193": "801790", "801194": "801790",                       # 非银金融
    "801101": "801750", "801103": "801750", "801104": "801750",                       # 计算机
    "801102": "801770", "801223": "801770",                                           # 通信
    "801991": "801170", "801992": "801170",                                           # 交通运输
    "801993": "801210", "801994": "801210",                                           # 社会服务
    "801995": "801760",                                                               # 传媒
}


def _to_instrument(symbol: str) -> str | None:
    if symbol.startswith("6"):
        return f"SH{symbol}"
    if symbol[0] in "03":
        return f"SZ{symbol}"
    return None  # 北交所等不在 csi_union 池内


def _sw2_parent(sw2_code: str) -> str:
    return SW2_TO_SW1_EXCEPTIONS.get(sw2_code, str(int(sw2_code) // 10 * 10))


def fetch_sw1_members() -> pd.DataFrame:
    """首选路径：申万官网逐板块拉成分。任一板块返回空视为接口不可用，抛错走降级。"""
    import akshare as ak

    boards = ak.sw_index_first_info()
    rows = []
    for _, b in boards.iterrows():
        code = str(b["行业代码"]).replace(".SI", "")
        name = str(b["行业名称"])
        cons = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                cons = ak.index_component_sw(symbol=code)
                break
            except Exception:
                if attempt < MAX_RETRY:
                    time.sleep(THROTTLE_SEC * 4 * attempt)
        if cons is None or cons.empty:
            raise RuntimeError(f"SW1 板块 {code} {name} 成分为空，接口不可用")
        for sym in cons["证券代码"].astype(str).str.zfill(6):
            inst = _to_instrument(sym)
            if inst:
                rows.append({"instrument": inst, "ind_code": code, "ind_name": name})
        time.sleep(THROTTLE_SEC)
    return pd.DataFrame(rows).drop_duplicates(subset="instrument", keep="first")


def build_from_sw2_cache() -> pd.DataFrame:
    """降级路径：SW2 缓存 → 一级映射。"""
    if not SW2_CACHE.exists():
        raise RuntimeError(f"SW2 缓存不存在: {SW2_CACHE}")
    sw2 = pd.read_csv(SW2_CACHE, dtype={"symbol": str, "sw2_code": str})
    sw2["symbol"] = sw2["symbol"].str.zfill(6)
    sw2["instrument"] = sw2["symbol"].map(_to_instrument)
    sw2["ind_code"] = sw2["sw2_code"].map(_sw2_parent)
    unknown = sw2[~sw2["ind_code"].isin(SW1_NAMES)]
    if not unknown.empty:
        raise RuntimeError(f"SW2→SW1 映射出现未知一级代码: {unknown['sw2_code'].unique()[:5]}")
    sw2["ind_name"] = sw2["ind_code"].map(SW1_NAMES)
    out = sw2.dropna(subset=["instrument"])[["instrument", "ind_code", "ind_name"]]
    return out.drop_duplicates(subset="instrument", keep="first")


def fetch_sina_sectors() -> pd.DataFrame:
    """补缺路径：新浪行业 49 板块全成分。单板块反复失败只打印，缺的股票留给 UNK。"""
    import akshare as ak

    boards = ak.stock_sector_spot(indicator="新浪行业")
    rows = []
    for _, b in boards.iterrows():
        label, name = b["label"], str(b["板块"])
        for attempt in range(1, MAX_RETRY + 1):
            try:
                cons = ak.stock_sector_detail(sector=label)
                for sym in cons["code"].astype(str).str.zfill(6):
                    inst = _to_instrument(sym)
                    if inst:
                        rows.append({"instrument": inst, "ind_code": f"S_{label}", "ind_name": name})
                break
            except Exception as e:
                if attempt == MAX_RETRY:
                    print(f"  [sina {name}] 拉取失败: {str(e)[:60]}")
                else:
                    time.sleep(THROTTLE_SEC * 4 * attempt)
        time.sleep(THROTTLE_SEC)
    return pd.DataFrame(rows).drop_duplicates(subset="instrument", keep="first")


def build(snapshot_date: pd.Timestamp | None = None) -> int:
    snap = (snapshot_date or pd.Timestamp.today()).normalize()
    try:
        base = fetch_sw1_members()
        source = "申万官网"
    except Exception as e:
        print(f"[industry] 申万接口不可用({e})，用 SW2 缓存映射")
        base = build_from_sw2_cache()
        source = f"SW2缓存({pd.Timestamp(SW2_CACHE.stat().st_mtime, unit='s').date()})+新浪补缺"
    sina = fetch_sina_sectors()
    fill = sina[~sina["instrument"].isin(base["instrument"])]
    df = pd.concat([base, fill], ignore_index=True)
    df["datetime"] = snap
    warehouse.append(DATASET, df)
    print(f"[industry] 快照 {snap.date()} 已入仓(来源: {source}), {len(base)}+{len(fill)} 只 → "
          f"{df['ind_code'].nunique()} 组, {warehouse.coverage(DATASET)}")
    return 0


def load_map(as_of: pd.Timestamp | str | None = None) -> pd.Series:
    """instrument → ind_code。取 as_of 之前最近一期快照；早于全部快照时用最早一期。"""
    df = warehouse.load(DATASET)
    if df.empty:
        return pd.Series(dtype=object)
    snaps = df.index.get_level_values("datetime").unique().sort_values()
    snap = snaps[-1]
    if as_of is not None:
        eligible = snaps[snaps <= pd.Timestamp(as_of)]
        snap = eligible[-1] if len(eligible) else snaps[0]
    return df.xs(snap, level="datetime")["ind_code"]


if __name__ == "__main__":
    sys.exit(build())
