# dashboard/app.py
"""A股决策驾驶舱。启动: streamlit run dashboard/app.py
只读 results/; 写操作仅改 watchlist.yaml/state, 非交易系统。"""
import os
import sys

# `streamlit run dashboard/app.py` 只把 dashboard/ 放进 sys.path, 这里补上仓库根,
# 让 `from dashboard import ...` 与 tests 一致可用。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data_loader as dl, metrics as mx, controls as ct  # noqa: E402

st.set_page_config(page_title="A股决策驾驶舱", layout="wide")
st.title("📊 A股决策驾驶舱")
st.caption("只读 results/ · 写操作仅改 watchlist/state · **非交易系统**")

tab1, tab2, tab3, tab4 = st.tabs(["今日决策", "Forward 兑现", "因子体检", "控制台"])

DECISION_EMOJI = {"BUY": "🟢", "REDUCE": "🟠", "HOLD": "🟡",
                  "CONTINUE_WATCH": "🟡", "DROP": "🔴", "AVOID": "⛔"}
RANK_COLS = ["fundamental_rank", "fund_flow_rank", "liquidity_rank", "tech_rank",
             "chips_rank", "regime_rank", "sector_rank", "news_rank"]

# ---------------- Tab ① 今日决策 ----------------
with tab1:
    dates = dl.list_scan_dates()
    if not dates:
        st.warning("results/decisions 下暂无 partial_*.csv")
    else:
        date = st.selectbox("决策日期", dates, key="t1_date")
        j = dl.load_partial_json(date)
        st.markdown(f"**大盘**: `{j.get('market_regime', '?')}` · "
                    f"**股票池** {len(j.get('decisions', []))} 只")
        df = dl.load_partial_csv(date)
        counts = mx.decision_group_counts(df)
        st.write(" · ".join(f"{DECISION_EMOJI.get(k, '')}{k} {v}" for k, v in counts.items()))

        decs = sorted(df["decision"].dropna().unique().tolist())
        pick = st.multiselect("筛选 decision", decs, default=decs, key="t1_pick")
        kw = st.text_input("搜索 symbol/name", key="t1_kw").strip()
        view = df[df["decision"].isin(pick)]
        if kw:
            view = view[view["symbol"].astype(str).str.contains(kw) |
                        view["name"].astype(str).str.contains(kw)]
        show_cols = [c for c in ["symbol", "name", "sector", "decision", "state",
                                 "confidence", "composite"] + RANK_COLS if c in view.columns]
        sort_col = "composite" if "composite" in view.columns else "confidence"
        cfg = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f")
               for c in RANK_COLS if c in view.columns}
        st.dataframe(view[show_cols].sort_values(sort_col, ascending=False),
                     width="stretch", hide_index=True, column_config=cfg)
        st.caption("提示: rank 0=最强(绿/满), 1=最弱。")

        sel = st.text_input("查看某 symbol 的 drivers/risks", key="t1_sel").strip()
        if sel:
            hit = next((d for d in j["decisions"] if str(d["symbol"]) == sel), None)
            if hit:
                st.json({k: hit.get(k) for k in ("decision", "confidence", "ranks",
                                                 "drivers", "risks", "gates_failed")})
            else:
                st.info("该 symbol 不在当日决策集")
        with st.expander("原始 .md 报告"):
            st.markdown(dl.load_partial_md(date) or "(无 md)")

# ---------------- Tab ② Forward 兑现 ----------------
with tab2:
    fdates = dl.list_forward_dates()
    if not fdates:
        st.warning("results/forward 下暂无 forward_*.csv")
    else:
        fdate = st.selectbox("决策日(看 T+N 兑现)", fdates, key="t2_date")
        fdf = dl.load_forward_csv(fdate)
        ret_cols = [c for c in fdf.columns if c.startswith("ret_T+")]
        ret_col = st.radio("周期", ret_cols, horizontal=True, key="t2_h") if ret_cols else "ret_T+1"
        csi = dl.parse_csi_benchmark(fdate, horizon=ret_col.split("+")[-1])
        if csi is not None:
            st.metric(f"沪深300 {ret_col} 基准", f"{csi:+.2%}")
        perf = mx.forward_group_perf(fdf, ret_col)
        if not perf.empty:
            st.dataframe(perf, width="stretch", hide_index=True,
                         column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                        for c in ["hit_rate", "mean_ret", "median_ret",
                                                  "worst", "best"]})
        spread = mx.buy_vs_drop(fdf, ret_col)
        st.metric("BUY − DROP 价差 (负=BUY跑输, 当前核心问题)", f"{spread:+.2%}")
        if not perf.empty:
            st.bar_chart(perf.set_index("decision")["mean_ret"])
        st.markdown("**个股明细**")
        cols = [c for c in ["symbol", "name", "sector", "decision", "confidence",
                            "composite", ret_col, "max_dd"] if c in fdf.columns]
        st.dataframe(fdf[cols].sort_values(ret_col, ascending=False),
                     width="stretch", hide_index=True)

# ---------------- Tab ③ 因子体检 ----------------
with tab3:
    st.warning("⚠ 14 天样本 · IC 统计上不显著 · 仅方向性参考")
    ic_df = dl.load_factor_ic()
    if ic_df.empty:
        st.info("尚无 results/factor_ic.csv, 先运行 `python3 compute_factor_ic.py`")
    else:
        summ = mx.factor_ic_summary(ic_df)
        st.markdown("**逐因子平均日 IC**(正=有预测力, 负=反向)")
        st.bar_chart(summ.set_index("factor")["ic_mean"])
        st.dataframe(summ, width="stretch", hide_index=True,
                     column_config={"ic_mean": st.column_config.NumberColumn(format="%.4f"),
                                    "pos_frac": st.column_config.NumberColumn(format="%.2f")})
        comp = summ[summ["factor"] == "composite"]
        if not comp.empty:
            st.metric("composite 基线 IC", f"{comp.iloc[0]['ic_mean']:+.4f}")
        st.markdown("**IC 时间序列**")
        facs = st.multiselect("选因子", summ["factor"].tolist(),
                              default=[f for f in ["fund_flow_rank", "sector_rank", "composite"]
                                       if f in summ["factor"].tolist()], key="t3_f")
        if facs:
            piv = ic_df[ic_df["factor"].isin(facs)].pivot_table(
                index="date", columns="factor", values="ic")
            st.line_chart(piv)
        st.info("诊断: penalty 系统性惩罚赢家 · technical/liquidity 接飞刀 · "
                "sector 反追高在动量市做反")

# ---------------- Tab ④ 控制台 ----------------
with tab4:
    st.warning("⚠ 非交易系统 · 写操作仅改 watchlist.yaml · 保存自动备份")
    rows = ct.load_watchlist_rows()
    edited = st.data_editor(
        pd.DataFrame(rows), num_rows="dynamic", width="stretch", key="t4_edit",
        column_config={"state": st.column_config.SelectboxColumn(
            options=["NONE", "WATCHING", "HELD", "EXITED"])})
    confirm = st.checkbox("我确认保存对 watchlist.yaml 的修改", key="t4_ok")
    if st.button("💾 保存 watchlist", disabled=not confirm):
        try:
            new_rows = [{k: v for k, v in r.items() if pd.notna(v)}
                        for r in edited.to_dict("records")]
            reloaded = ct.save_watchlist(new_rows)
            st.success(f"已保存并备份, 共 {len(reloaded)} 行")
        except ValueError as e:
            st.error(f"校验失败, 未写入: {e}")

    st.divider()
    if "daily_proc" not in st.session_state:
        st.session_state.daily_proc = None
    running = (st.session_state.daily_proc is not None
               and st.session_state.daily_proc.poll() is None)
    if st.button("▶ 一键 daily_run(约数分钟)", disabled=running):
        if st.session_state.get("t4_run_ok"):
            st.session_state.daily_proc = ct.run_daily()
            st.toast("已在后台启动 daily_run")
        else:
            st.warning("请先勾选下方确认框")
    st.checkbox("我确认触发全量 daily_run", key="t4_run_ok")
    st.caption("运行状态: " + ("🟢 进行中" if running else "⚪ 空闲"))
    with st.expander("最新运行日志尾部"):
        st.code(ct.latest_log_tail())
