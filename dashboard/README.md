# A股决策驾驶舱

只读 `results/` 的 Streamlit 看板 + 极简控制台。**非交易系统**:写操作仅改 `watchlist.yaml`。

## 启动
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py

## 标签页
- 今日决策:任一扫描日的 50 只票决策 + 8 维 rank + drivers/risks
- Forward 兑现:决策的 T+1/T+3/T+5 真实兑现、BUY−DROP 价差、沪深300 基准
- 因子体检:逐因子 IC(哪个维度有预测力/反向)、IC 时间序列
- 控制台:编辑 watchlist(自动备份)、一键 daily_run、看运行日志

数据由 `scripts/daily_run.sh` 产出;IC 由 `compute_factor_ic.py` 落盘 `results/factor_ic.csv`。
