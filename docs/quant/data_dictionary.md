# 数据字典（M1：qlib 数据仓 + PIT 股票池）

> 更新方式：`.venv-quant/bin/python -m quant.data.bootstrap --force-download` 重拉数据包 →
> `python -m quant.data.universe` 重建股票池 → `python -m quant.data.checks` 质检。

## 数据源

| 项 | 说明 |
|---|---|
| 主数据包 | [chenditc/investment_data](https://github.com/chenditc/investment_data) 每日发布的 `qlib_bin.tar.gz`（qlib 官方 cn_data 已停更）；tushare+akshare 多源交叉校验 |
| 落盘位置 | `~/.qlib/qlib_data/cn_data`（项目外，不进 git；归档缓存在 `~/.qlib/downloads/`） |
| 覆盖范围 | 2000-01-04 起的全 A 日线，日更（当前到 2026-07-03） |
| venv | 项目根 `.venv-quant`（pyqlib 0.9.7 / lightgbm 4.6.0 / akshare）。lightgbm 的 arm64 libomp 已手工放入包内并加 rpath（brew 的 /usr/local 是 x86 遗留，不可用） |

## 字段（features/<code>/*.day.bin）

`open / high / low / close / vwap`（均为后复权价）、`factor`（复权因子，原始价 = close/factor）、
`volume / amount / change / adjclose`。qlib expression 中以 `$close` 形式引用。

## instruments（PIT 成分事实表）

每行 `CODE\tSTART\tEND` = 该股在该指数的一段成分区间。数据包自带：
`all / csiall / csi300 / csi500 / csi800 / csi1000`。

**csi_union**（`quant/data/universe.py` 生成）= csi300 ∪ csi500 ∪ csi1000，处理：
1. 区间合并（同日 300↔500 切换无缝拼接，真实退出的空档保留）
2. 次新掐头：上市（以 all.txt 首日为代理）不足 120 个交易日的区间头部裁掉
3. ST 剔除：当前名称含 ST 的 158 只整体移除

规模核对（构建时自动校验）：三指数在 2020/2023/2026 探测日成分数精确为 300/500/1000；
csi_union 活跃成分 1726→1798 只；csi300 历史调出 639 只（PIT 真实性证据）。

## 已知局限（有意接受，勿当 bug）

- **ST 过滤非 PIT**：历史时点的 ST 状态无免费数据源，只能按"当前是 ST"整体剔除。方向保守（可能多剔掉几只曾经正常的股票），对约 1800 只的池子影响 <1%。
- **停牌日无行情**：qlib 面板中表现为缺行/NaN，回测层会自动视为不可交易；采样缺失率 5.9%（含正常停牌）、2020-2026 面板整体 close 缺失 0.13%，属正常水位。
- **macOS 注意**：qlib 大面板 `D.features` 会开多进程（spawn），调用方脚本必须有 `if __name__ == "__main__"` 保护，不能用 stdin heredoc 跑。

## 质检基线（2026-07-09 首次全绿）

复权连续性 0 异常（20 只采样，|日收益|>21% 视为可疑）；akshare 新浪不复权日线比价 20/20 通过
（qlib `$close/$factor` vs akshare close，中位误差 <0.5%）；csi300 PIT 调出记录 639 只。
