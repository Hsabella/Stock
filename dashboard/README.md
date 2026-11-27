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

## 本地看盘台

在项目根目录运行以下命令可启动并打开浏览器。

```bash
. scripts/lib_no_proxy.sh
arch -arm64 .venv-quant/bin/python -m dashboard.market_terminal --open
```

默认地址为 <http://127.0.0.1:8767>，可用 `--port` 指定其他端口。服务只监听本机，关闭运行终端即停止。

- 首页优先列出最近收盘破线的持仓、盘中触线和卫星额度。止损沿用晨报引擎的棘轮口径。盘中报价每 15 秒更新，日 K 与现金按页面标注的数据日期读取。
- 抄底灯从晨报读取，支持今日开盘窗、结构较稳、全部和搜索。日 K 标注前低、反弹高点、回踩与量能，结构筛选仅用于观察。止损退出票按晨报的回补条件复核。
- 消息读取本地新闻流，按持仓、抄底灯和板块归类。X 链接使用公开摘要接口，长文以原帖为准。页面提醒随报价更新。
- 服务以只读方式使用 `watchlist.yaml`、`results/brief/`、`cache/kline/` 和 `cache/news/history.csv`，外部行情请求只包含证券代码。
