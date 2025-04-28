#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全局设置配置文件
"""

# 数据获取设置
DATA_SOURCE = "akshare"  # 数据源: akshare, tushare, baostock
DATA_REFRESH_INTERVAL = 60  # 数据刷新间隔（秒）
STOCK_MARKETS = ["上证", "深证"]  # 扫描的股票市场
EXCLUDE_ST = True  # 是否排除ST股票
EXCLUDE_NEW = True  # 是否排除上市不足100天的新股

# 技术指标设置
DEFAULT_INDICATORS = {
    "MACD": {"fast_period": 6, "slow_period": 12, "signal_period": 5},
    "RSI": {"period": 24, "overbought": 70, "oversold": 30},
    "MA": {"short": 5, "mid": 20, "long": 60},
    "VOLUME": {"short": 5, "long": 20},
    "BOLL": {"window": 20, "std_dev": 2.0},
    "KDJ": {"n": 9, "m1": 3, "m2": 3}
}

# 扫描设置
SCAN_INTERVAL = 15 * 60  # 扫描间隔（秒），默认15分钟
MAX_STOCKS_TO_SCAN = 1000  # 最大扫描股票数量
HISTORY_DAYS = 120  # 回溯历史数据天数

# 结果设置
MAX_RESULTS = 50  # 最大展示结果数量
RESULT_SORT_BY = "score"  # 结果排序依据：score, price, volume, change_percent
OUTPUT_FORMAT = ["csv", "html"]  # 输出格式
OUTPUT_DIR = "./results"  # 结果输出目录

# 日志设置
LOG_LEVEL = "INFO"  # 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_FILE = "./logs/stock_scanner.log"  # 日志文件
LOG_ROTATION = "1 day"  # 日志轮转

# UI设置
ENABLE_GUI = True  # 是否启用图形界面
THEME = "dark"  # 主题：dark, light

# 通知设置
ENABLE_NOTIFICATION = False  # 是否启用通知
NOTIFICATION_TYPES = ["email"]  # 通知类型：email, sms, wechat, dingtalk
EMAIL_CONFIG = {
    "smtp_server": "smtp.example.com",
    "smtp_port": 465,
    "sender": "your_email@example.com",
    "password": "your_password",
    "receivers": ["receiver@example.com"]
}

# 高级设置
DEBUG_MODE = False  # 调试模式
ENABLE_PROXY = False  # 是否启用代理
PROXY_CONFIG = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

# 回测设置
BACKTEST_START = "2020-01-01"  # 回测起始日期
BACKTEST_END = "2023-12-31"  # 回测结束日期
INITIAL_CAPITAL = 1000000  # 初始资金
COMMISSION_RATE = 0.0003  # 手续费率
SLIPPAGE = 0.002  # 滑点率
