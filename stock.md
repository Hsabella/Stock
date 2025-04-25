下面先给出核心思路和方法论，之后分别介绍 MACD 与 RSI 指标的含义、如何在 TradingView 中做筛选，以及如何利用 Python 脚本批量扫描，最后给出示例代码与示例标的（仅作演示，需实时验证）。

## 一、核心思路  
1. **指标含义**：  
   - MACD(6,12,5) 用于判断短期动能，重点关注“零线金叉”与“信号线金叉”信号。  
   - RSI(周期24) < 30 表示超卖，可配合 MACD 金叉进场。  
2. **技术筛选工具**：  
   - **TradingView 内置 Screener**：可实现零延迟的实时技术条件筛选([fx168news](https://www.fx168news.com/article/TradingView-829982?utm_source=chatgpt.com))([fintastic.trading](https://fintastic.trading/wisdom_box/%E5%B7%A5%E5%85%B7%E7%AE%B1/tradingview%E7%9A%84screener%E4%BB%8B%E7%B4%B9%E8%88%877%E5%80%8B%E5%A5%BD%E7%94%A8%E8%A8%AD%E5%AE%9A/?utm_source=chatgpt.com))。  
   - **Python 脚本 + 数据接口**：利用 `yfinance`、`pandas_ta` 等库下载标的历史行情并计算技术指标，适合自定义批量扫码([CSDN Blog](https://blog.csdn.net/qq_42022690/article/details/129200632?utm_source=chatgpt.com))。  

## 二、MACD 指标与“零线金叉”  
MACD（指数平滑异同移动平均线）由快线（EMA6−EMA12）与慢线（EMA 的信号线，周期5）构成。  
- **信号线金叉**：MACD 线自下向上穿越信号线，为动量转强的短线买入信号([Investopedia](https://www.investopedia.com/terms/m/macd.asp?utm_source=chatgpt.com))([Fidelity Investments](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd?utm_source=chatgpt.com))。  
- **零线金叉**：当 MACD 线及信号线同时从负区跨越至正区，表明短期均线上穿长期均线，潜在新一波上涨开始([StockCharts](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/macd-zero-line-crosses-with-swing-points?utm_source=chatgpt.com))([Mind Math Money](https://www.mindmathmoney.com/articles/understanding-the-macd-indicator-macd-line-signal-line-histogram-crossover-and-zero-line?utm_source=chatgpt.com))。  

## 三、RSI 指标与超卖区域  
RSI（相对强弱指数）常用周期为14，此处设为24来避免频繁抖动。当 RSI24 < 30 时，表示标的处于超卖区域，通常预示回升可能性增加([Zhihu](https://zhuanlan.zhihu.com/p/695780484?utm_source=chatgpt.com))。  
- RSI 低于 30 → 超卖／反弹机会。  
- 结合 MACD 零线金叉，可形成“低位金叉”高胜率入场条件([https://www.tempmail.us.com](https://www.tempmail.us.com/zh/screener/%E5%A6%82%E4%BD%95%E4%BB%8E%E7%89%B9%E5%AE%9A%E4%BA%A4%E6%98%93%E6%89%80%E7%AD%9B%E9%80%89%E8%AF%81%E5%88%B8%E4%BB%A5%E5%9C%A8-pine-%E8%84%9A%E6%9C%AC%E4%B8%AD%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E8%82%A1%E7%A5%A8%E7%AD%9B%E9%80%89%E5%99%A8?utm_source=chatgpt.com))。  

## 四、在 TradingView Screener 上实操  
1. 打开 TradingView，进入 **Stock Screener → Advanced**([fx168news](https://www.fx168news.com/article/TradingView-829982?utm_source=chatgpt.com))。  
2. 在 **Technical** 条件中：  
   - 添加 `MACD(6,12,5)` → 选择 **“MACD Line crosses above Signal Line”**（信号线金叉）或 **“Zero Line Cross (Bullish)”**（零线金叉）条件。  
   - 添加 `RSI(24)` → 选择 **“RSI < 30”** 条件。  
3. 可选：再加上 **“Price above 50-day MA”** 等条件过滤震荡高风险标的。  
4. 即可获得符合条件的实时标的列表；点击每只标的可进一步查看具体图表与回测。  

## 五、Python 批量脚本示例  
以下脚本示范如何用 `yfinance` 与 `pandas_ta` 批量筛选美国市场标的：  
```python
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# 示例标的列表
symbols = ["AAPL","MSFT","AMZN","TSLA","GOOGL"]

candidates = []
for sym in symbols:
    df = yf.download(sym, period="60d", interval="1d")
    # 计算 MACD(6,12,5)
    macd = ta.macd(df["Close"], fast=6, slow=12, signal=5)
    df = df.join(macd)
    # 计算 RSI(24)
    df["RSI24"] = ta.rsi(df["Close"], length=24)
    # 判断“昨日 MACD_diff<0 且今日 MACD_diff>0”（信号线金叉） & 今日 RSI24<30
    df["MACD_diff"] = df["MACD_6_12_5"] - df["MACDs_6_12_5"]
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    if (yesterday["MACD_diff"] < 0) and (today["MACD_diff"] > 0) and (today["RSI24"] < 30):
        candidates.append({
            "symbol": sym,
            "MACD_diff": round(today["MACD_diff"],4),
            "RSI24": round(today["RSI24"],2)
        })

print("筛选结果：", candidates)
```
运行后，`candidates` 中即是满足“低位金叉”且 RSI24<30 的标的（需实时执行以获取最新结果）。

## 六、示例结果（仅作演示）  
> **假设** 2025-04-25 收盘后扫描，脚本返回：  
| 代码    | 今日 MACD_diff | 今日 RSI24 |
| ------- | -------------: | ---------: |
| MSFT    |         0.0152 |      28.47 |
| TSLA    |         0.0089 |      26.12 |
| NFLX    |         0.0045 |      29.88 |

上述仅为示例数据，需在本地或云端进行实时回测与验证。

## 七、小结  
- **TradingView Screener**：最快捷、零延迟，适合实时监控；  
- **Python 脚本**：灵活可定制，适合批量扫描与策略回测；  
- 建议结合二者：日内用 Screener 跟踪，策略开发阶段用脚本回测。  
- 最终入场前请务必结合成交量、支撑阻力等多重确认，提高胜率。