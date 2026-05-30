# dashboard/metrics.py
"""纯函数聚合: 输入 DataFrame, 输出展示用 DataFrame/标量, 无 IO。"""
import pandas as pd


def decision_group_counts(df):
    return df["decision"].value_counts().to_dict()


def forward_group_perf(df, ret_col="ret_T+1",
                       groups=("BUY", "CONTINUE_WATCH", "DROP", "AVOID", "HOLD", "REDUCE")):
    rows = []
    for g in groups:
        sub = df[df["decision"] == g][ret_col].dropna()
        if len(sub) == 0:
            continue
        rows.append({
            "decision": g, "count": len(sub),
            "hit_rate": (sub > 0).mean(),
            "mean_ret": sub.mean(), "median_ret": sub.median(),
            "worst": sub.min(), "best": sub.max(),
        })
    return pd.DataFrame(rows)


def buy_vs_drop(df, ret_col="ret_T+1"):
    buy = df[df["decision"] == "BUY"][ret_col].dropna()
    drop = df[df["decision"] == "DROP"][ret_col].dropna()
    if len(buy) == 0 or len(drop) == 0:
        return float("nan")
    return buy.mean() - drop.mean()


def factor_ic_summary(ic_df):
    """factor_ic.csv -> 每因子 平均IC/为正占比/天数, 按 IC 降序。"""
    if ic_df.empty:
        return ic_df
    g = ic_df.groupby("factor")["ic"]
    out = pd.DataFrame({
        "ic_mean": g.mean(),
        "pos_frac": g.apply(lambda s: (s > 0).mean()),
        "n_days": g.size(),
    }).reset_index().sort_values("ic_mean", ascending=False)
    return out
