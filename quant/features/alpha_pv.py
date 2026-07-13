"""量价因子库 v1：全部用 qlib expression 表达（零预计算）。

符号约定：按学术/业界常见方向书写（值越大 → 预期下周收益越高），但**方向只是
假设**，最终以 ic_report 的实测 IC 为准；LightGBM 本身不依赖方向。

数据字段来自 chenditc bin 包：$open/$high/$low/$close/$vwap 为后复权价，
$volume/$amount 为量额。资金流因子因东财只有 ~120 天历史，v1 缺席（见
akshare_fetcher.em_fundflow 的滚动累积说明），以 vwap_dev/intraday_pos 两个
量价代理近似"盘中资金行为"。
"""
from __future__ import annotations

# 短期反转：A 股周频最稳健的效应之一，近期涨多的下周倾向弱
# 注意：qlib 表达式不支持一元负号 -(...)，取反一律写成 0-... / 1-... 形式
EXPRESSIONS: dict[str, str] = {
    "rev_5": "1-$close/Ref($close,5)",
    "rev_10": "1-$close/Ref($close,10)",
    # 中期动量（跳过最近 5 天避开反转干扰）
    "mom_20_skip5": "Ref($close,5)/Ref($close,25)-1",
    "mom_120": "Ref($close,20)/Ref($close,120)-1",
    # 低波动/低彩票偏好
    "vol_20": "0-Std($close/Ref($close,1)-1,20)",
    "max_ret_20": "0-Max($close/Ref($close,1)-1,20)",
    # 价格在 60 日区间中的位置
    "pos_60": "($close-Min($low,60))/(Max($high,60)-Min($low,60)+1e-12)",
    # 量能：近期放量程度
    "vol_ratio_5_60": "Mean($volume,5)/(Mean($volume,60)+1e-12)",
    # Amihud 非流动性溢价（越难成交的股要求越高回报）
    "amihud_20": "Mean(Abs($close/Ref($close,1)-1)/($amount+1e-12),20)",
    # 均线偏离（移植自 factors/technical 的 bias）
    "bias_20": "$close/Mean($close,20)-1",
    # 量价配合度
    "corr_pv_10": "Corr($close, Log($volume+1), 10)",
    # 盘中资金行为代理：收盘持续强于当日均价 / 收在日内高位
    "vwap_dev_5": "Mean($close/$vwap-1,5)",
    "intraday_pos_5": "Mean(($close-$low)/($high-$low+1e-12),5)",
}

# 标签：T 日收盘出信号 → T+1 开盘买入 → 持一周到下周同日开盘卖出。
# 用开盘价规避"收盘价出信号又按收盘价成交"的前视偏差。
LABEL_EXPRESSION = "Ref($open,-6)/Ref($open,-1)-1"
LABEL_NAME = "LABEL0"

# parquet 数仓提供的扩展特征（ext_features.py 产出，名字在这里登记以便统一引用）。
# 只有登记在此的列会进模型（handler 按本表过滤 parquet 列）。
# v2 Step2 财报 PIT 因子按预注册门槛 |IC|≥0.015 裁决（factor_report_v2 实测）：
# roe_delta +0.0164 入模；profit_yoy +0.0121、roe_ttm +0.0064 未达标，
# 列仍在 parquet 里但不入模。v1 因子（含 size，本轮 IC 已衰减至 drop）原样保留，
# 保证 Step2 归因干净（只加了 roe_delta 一个变量），删减留给 S3 组装时统一决策。
# v2 Step3 风格切换特征（全截面同值，来自 index_daily 指数仓）：大盘 20 日波动/动量、
# 大小盘风向。**已实测证伪不入模**（2026-07 v3 实验：限池超额 +2.4%→-3.9%）——同值列
# 是"时间坐标"，树模型用它把训练窗切成时段逐段记忆标签，时间记忆过拟合。列仍产出在
# parquet（供研究），handler 的 CSRankNormExempt 保留（若以后以分桶/交互形式重试）。
REGIME_FEATURES = ["mkt_vol_20", "mkt_mom_20", "size_spread_20"]

EXT_FEATURES = ["turn_20", "turn_ratio_5_120", "size", "ep", "bp", "roe_delta"]

ALL_FEATURES = list(EXPRESSIONS) + EXT_FEATURES
