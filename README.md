# 中国股市技术指标筛选器

基于技术指标的中国股市（上证、深证）实时监控与买入信号筛选系统。

## 功能特点

- 实时获取中国股市（上证、深证）股票数据
- 基于 MACD、RSI 等多种技术指标进行买入信号筛选
- 支持策略组合与动态调整
- 多维度可视化展示
- 信号推送与结果导出

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/china-stock-scanner.git
cd china-stock-scanner

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

```bash
# 启动主程序
python main.py
```

## 项目结构

```
├── config/               # 配置文件目录
│   ├── settings.py       # 全局设置
│   └── strategies.py     # 策略配置
├── data/                 # 数据相关模块
│   ├── fetcher.py        # 数据获取
│   ├── processor.py      # 数据处理
│   └── storage.py        # 数据存储
├── indicators/           # 技术指标模块
│   ├── basic.py          # 基础指标
│   ├── advanced.py       # 高级指标
│   └── custom.py         # 自定义指标
├── strategies/           # 交易策略模块
│   ├── base.py           # 策略基类
│   ├── macd_rsi.py       # MACD+RSI策略
│   ├── volume.py         # 量价策略
│   └── mixed.py          # 混合策略
├── utils/                # 工具函数
│   ├── visualize.py      # 可视化工具
│   ├── logger.py         # 日志工具
│   └── helpers.py        # 辅助函数
├── main.py               # 主程序入口
└── requirements.txt      # 项目依赖
```

## 策略说明

默认包含以下买入策略：

1. **MACD+RSI 策略**：MACD(6,12,5)金叉且 RSI(24)<30
2. **均线突破策略**：短期均线上穿长期均线，结合成交量确认
3. **量价背离策略**：价格创新低但指标不创新低
4. **支撑位反弹策略**：价格触及支撑位并反弹确认
5. **多指标综合策略**：综合考虑多个指标的综合得分

所有策略均可在配置文件中动态调整参数。
