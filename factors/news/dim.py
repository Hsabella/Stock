"""news_dim: 用关键词词典对新闻流做个股 / 板块情绪命中.

算法:
1. 个股直接命中: 标题/内容包含 股票名 / 股票简称 / 6 位代码
2. 板块间接命中: 命中所属 SW2 行业名 或对应题材短词
3. 情绪极性: 利好词 / 利空词出现次数差
4. 风险事件: 减持 / 解禁 / 立案 / 业绩预亏 等强信号词独立标记 → 直接 risk_tag
5. news_rank:
   - 默认 0.5 (中性, 没新闻不影响)
   - 有正面命中 → 0.0~0.4
   - 有风险事件 → 0.9~1.0
"""
from __future__ import annotations
import re
import pandas as pd

from engine.ranker import cross_section_rank

# ---- 词典 ----
BULLISH_KW = [
    "中标", "签约", "战略合作", "重大合同", "订单", "重大进展",
    "业绩超预期", "净利润大增", "同比增长", "创新高", "再创新高",
    "突破", "量产", "首次", "首发",
    "增持", "回购", "股权激励",
    "受益", "利好", "风口", "政策支持", "补贴", "评级上调",
    "目标价上调", "买入评级", "强推", "推荐评级",
]

BEARISH_KW = [
    "减持", "解禁", "限售解禁", "大股东减持",
    "业绩不及预期", "业绩预亏", "业绩预降", "净利润下滑", "亏损", "同比下降",
    "处罚", "立案调查", "退市风险", "警示函", "失控",
    "澄清", "否认", "辟谣",
    "限产", "停产", "故障", "召回", "事故",
    "商誉减值", "计提", "资产减值",
    "评级下调", "目标价下调", "下调评级",
    "运价下跌", "运价回落", "需求疲软",
]

# 风险事件: 强信号词, 命中即 risk_tag
RISK_EVENT_KW = {
    "RISK_NEWS_减持": ["减持公告", "大股东减持", "拟减持", "计划减持"],
    "RISK_NEWS_解禁": ["限售解禁", "解禁日"],
    "RISK_NEWS_立案": ["立案调查", "行政处罚", "警示函"],
    "RISK_NEWS_预亏": ["业绩预亏", "业绩预降", "商誉减值", "资产减值"],
    "RISK_NEWS_退市": ["退市风险", "*ST"],
}

# 板块关键词扩展 (SW2 名以外的常见称呼)
SECTOR_ALIASES = {
    "航运港口": ["航运", "油运", "运价", "BDI", "VLCC", "油轮", "原油价格", "布伦特"],
    "半导体": ["半导体", "芯片", "国产替代", "HBM", "存储", "晶圆", "光刻"],
    "能源金属": ["碳酸锂", "锂矿", "锂电"],
    "光伏设备": ["光伏", "硅料", "组件", "电池片"],
    "电池": ["动力电池", "储能"],
    "贵金属": ["黄金", "白银", "金价", "银价"],
    "工业金属": ["铜", "铝", "铅锌"],
    "自动化设备": ["机器人", "智能制造", "工业自动化"],
    "汽车零部件": ["新能源车", "电动车", "智能驾驶"],
    "通信设备": ["5G", "光模块", "算力"],
    "航天装备Ⅱ": ["航天", "卫星", "火箭"],
    "航空装备Ⅱ": ["大飞机", "C919", "军机"],
}


def _count_hits(text: str, kw_list: list[str]) -> int:
    if not text:
        return 0
    return sum(1 for k in kw_list if k in text)


def _matched_kws(text: str, kw_list: list[str]) -> list[str]:
    if not text:
        return []
    return [k for k in kw_list if k in text]


def compute_one(symbol: str, name: str, sw2_name: str | None,
                news_df: pd.DataFrame) -> dict:
    """对单只股票扫一遍新闻流."""
    out = {
        "symbol": symbol,
        "news_direct_hits": 0,
        "news_sector_hits": 0,
        "news_bull_score": 0,
        "news_bear_score": 0,
        "news_events": [],
        "news_risk_tags": [],
        "news_sample": "",
    }
    if news_df is None or news_df.empty:
        return out

    # 拼标题+内容统一搜
    text_blob = (news_df["title"].fillna("") + " " + news_df["content"].fillna("")).astype(str)

    # 个股命中: 名称 (>=2 字才匹配避免误命中) + 6 位代码
    name_hits = pd.Series(False, index=news_df.index)
    if name and len(name) >= 2:
        name_hits = text_blob.str.contains(re.escape(name), na=False)
    sym_hits = text_blob.str.contains(symbol, na=False)
    direct_mask = name_hits | sym_hits
    out["news_direct_hits"] = int(direct_mask.sum())

    # 板块命中: SW2 名 + alias
    sector_kw = []
    if sw2_name:
        sector_kw.append(sw2_name.replace("Ⅱ", ""))  # 去罗马数字尾
        sector_kw.extend(SECTOR_ALIASES.get(sw2_name, []))
    sector_mask = pd.Series(False, index=news_df.index)
    for kw in sector_kw:
        if kw and len(kw) >= 2:
            sector_mask = sector_mask | text_blob.str.contains(re.escape(kw), na=False)
    sector_mask = sector_mask & ~direct_mask  # 去重
    out["news_sector_hits"] = int(sector_mask.sum())

    # 情绪打分: 直接命中权重 1, 板块间接 0.4
    relevant_idx = direct_mask | sector_mask
    if relevant_idx.any():
        for i in news_df.index[relevant_idx]:
            txt = text_blob.iloc[i]
            weight = 1.0 if direct_mask.iloc[i] else 0.4
            out["news_bull_score"] += weight * _count_hits(txt, BULLISH_KW)
            out["news_bear_score"] += weight * _count_hits(txt, BEARISH_KW)
            for tag, kws in RISK_EVENT_KW.items():
                if direct_mask.iloc[i] and _count_hits(txt, kws):
                    out["news_risk_tags"].append(tag)
                    matched = _matched_kws(txt, kws)
                    out["news_events"].append(f"{tag.replace('RISK_NEWS_','')}: {','.join(matched)}")

    # 抽一条最相关样本作 driver 显示
    if direct_mask.any():
        sample_row = news_df[direct_mask].iloc[-1]  # 最新一条
        out["news_sample"] = str(sample_row["title"])[:60] or str(sample_row["content"])[:60]

    out["news_risk_tags"] = sorted(set(out["news_risk_tags"]))
    return out


def compose_dim(rows: list[dict]) -> pd.DataFrame:
    """汇总为 DataFrame 并算 news_rank.

    news_raw = direct*1.0 + sector*0.4 + (bull - bear)*0.6
    news_rank: 默认 0.5 (没新闻 = 中性), 有命中按 raw rank, 风险事件直接 0.95
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    raw = (df["news_direct_hits"].astype(float) * 1.0
           + df["news_sector_hits"].astype(float) * 0.4
           + (df["news_bull_score"] - df["news_bear_score"]).astype(float) * 0.6)
    df["news_raw"] = raw

    # 默认中性 0.5; 有命中(direct + sector > 0)的做横截面 rank
    has_hit = (df["news_direct_hits"] + df["news_sector_hits"]) > 0
    rank = pd.Series(0.5, index=df.index)
    if has_hit.any():
        sub_rank = cross_section_rank(raw[has_hit])  # 0 = 最好 (raw 最大)
        # 把 [0,1] 压到 [0.1, 0.7] 防止极值挤出无新闻样本
        rank[has_hit] = 0.1 + sub_rank * 0.6
    # 风险事件强压 0.95
    risk_mask = df["news_risk_tags"].apply(lambda t: bool(t))
    rank[risk_mask] = 0.95
    df["news_rank"] = rank
    return df
