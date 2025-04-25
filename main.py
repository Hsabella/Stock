#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标股票筛选系统 - 主程序入口
"""

from strategies.macd_rsi import MacdRsiStrategy
from indicators.basic import BasicIndicator
from data.fetcher import data_fetcher
from config.strategies import STRATEGY_WEIGHTS, DYNAMIC_ADJUSTMENT
from config.settings import (
    DATA_SOURCE, STOCK_MARKETS, SCAN_INTERVAL, MAX_STOCKS_TO_SCAN,
    RESULT_SORT_BY, OUTPUT_FORMAT, OUTPUT_DIR, MAX_RESULTS,
    ENABLE_GUI, ENABLE_NOTIFICATION, LOG_LEVEL, LOG_FILE, LOG_ROTATION
)
import os
import sys
import time
import datetime
import threading
import schedule
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger
import traceback
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块

# 导入策略
# 后续可添加更多策略
# from strategies.ma_crossover import MaCrossoverStrategy
# from strategies.volume_price import VolumePriceStrategy
# ...


# 配置日志
def setup_logger():
    """配置日志系统"""
    # 创建日志目录
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 清除默认处理器
    logger.remove()

    # 添加控制台输出
    logger.add(sys.stderr, level=LOG_LEVEL)

    # 添加文件输出
    logger.add(
        LOG_FILE,
        rotation=LOG_ROTATION,
        retention="30 days",
        level=LOG_LEVEL,
        encoding="utf-8"
    )

    logger.info("日志系统初始化完成")


class StockScanner:
    """股票扫描器类"""

    def __init__(self):
        """初始化股票扫描器"""
        self.strategies = {}  # 策略字典
        self.running = False  # 运行状态
        self.market_status = "unknown"  # 市场状态：bull/bear/sideways
        self.last_scan_time = None  # 上次扫描时间
        self.last_scan_results = {}  # 上次扫描结果
        self.init_dirs()  # 初始化目录
        self.load_strategies()  # 加载策略

    def init_dirs(self):
        """初始化目录结构"""
        # 创建输出目录
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"创建输出目录: {OUTPUT_DIR}")

    def load_strategies(self):
        """加载交易策略"""
        # 清空当前策略
        self.strategies = {}

        # 加载MACD与RSI策略
        self.strategies["macd_rsi"] = MacdRsiStrategy()

        # 后续可添加更多策略
        # self.strategies["ma_crossover"] = MaCrossoverStrategy()
        # self.strategies["volume_price"] = VolumePriceStrategy()
        # ...

        # 设置策略权重
        for strategy_id, weight in STRATEGY_WEIGHTS.items():
            if strategy_id in self.strategies:
                self.strategies[strategy_id].set_weight(weight)

        # 统计已加载的策略
        enabled_strategies = [
            s for s in self.strategies.values() if s.is_enabled]
        logger.info(f"已加载{len(enabled_strategies)}/{len(self.strategies)}个策略")
        for strategy in enabled_strategies:
            logger.info(f"  - {strategy}")

    def adjust_strategies(self):
        """根据市场状态动态调整策略参数"""
        if not DYNAMIC_ADJUSTMENT["enabled"]:
            return

        logger.debug("正在执行策略动态调整...")

        # 判断市场状态
        self.detect_market_regime()

        # 根据市场状态调整参数
        if self.market_status in DYNAMIC_ADJUSTMENT["market_regimes"]:
            adjustments = DYNAMIC_ADJUSTMENT["market_regimes"][self.market_status]

            for param_path, value in adjustments.items():
                # 解析参数路径，如 "macd_rsi_strategy.rsi_oversold"
                parts = param_path.split(".")
                if len(parts) < 2:
                    continue

                strategy_id, param_name = parts[0], ".".join(parts[1:])

                # 调整权重
                if param_name == "weight" and strategy_id in self.strategies:
                    self.strategies[strategy_id].set_weight(value)
                    logger.info(f"已调整策略 '{strategy_id}' 权重为 {value}")

                # 调整其他参数
                elif "." in param_name and strategy_id in self.strategies:
                    # 嵌套参数，如 params.rsi_oversold
                    nested_parts = param_name.split(".")
                    if nested_parts[0] == "params" and len(nested_parts) > 1:
                        param_key = nested_parts[1]
                        self.strategies[strategy_id].set_params(
                            {param_key: value})
                        logger.info(
                            f"已调整策略 '{strategy_id}' 参数 '{param_key}' 为 {value}")

                # 直接参数
                elif strategy_id in self.strategies:
                    # 尝试直接设置属性
                    if hasattr(self.strategies[strategy_id], param_name):
                        setattr(
                            self.strategies[strategy_id], param_name, value)
                        logger.info(
                            f"已调整策略 '{strategy_id}' 属性 '{param_name}' 为 {value}")

    def detect_market_regime(self):
        """检测市场状态 (牛市/熊市/震荡市)"""
        try:
            # 获取指数数据（上证指数）
            index_data = data_fetcher.get_index_data("sh000001")
            if index_data.empty:
                logger.warning("无法获取指数数据，无法判断市场状态")
                return

            # 计算指标
            index_data = BasicIndicator.add_ma(
                index_data, short=20, mid=60, long=120)

            # 获取最新数据
            latest = index_data.iloc[-1]

            # 简单判断市场状态规则：
            # 1. 牛市：价格在所有均线之上，且短期>中期>长期均线
            if (latest["close"] > latest["MA_20"] and
                latest["close"] > latest["MA_60"] and
                latest["close"] > latest["MA_120"] and
                    latest["MA_20"] > latest["MA_60"] > latest["MA_120"]):
                new_status = "bull"
            # 2. 熊市：价格在所有均线之下，且短期<中期<长期均线
            elif (latest["close"] < latest["MA_20"] and
                  latest["close"] < latest["MA_60"] and
                  latest["close"] < latest["MA_120"] and
                  latest["MA_20"] < latest["MA_60"] < latest["MA_120"]):
                new_status = "bear"
            # 3. 震荡市：其他情况
            else:
                new_status = "sideways"

            # 状态变化时记录日志
            if new_status != self.market_status:
                logger.info(f"市场状态由 '{self.market_status}' 变为 '{new_status}'")
                self.market_status = new_status

            return self.market_status

        except Exception as e:
            logger.error(f"判断市场状态出错: {str(e)}")
            return self.market_status

    def scan_stocks(self):
        """扫描股票并执行策略"""
        self.last_scan_time = datetime.datetime.now()
        logger.info(
            f"开始扫描股票，当前时间: {self.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # 调整策略
            self.adjust_strategies()

            # 获取股票列表
            stocks = data_fetcher.get_stock_list(markets=STOCK_MARKETS)
            if stocks.empty:
                logger.error("获取股票列表失败")
                return

            total_stocks = len(stocks)
            logger.info(f"共获取到{total_stocks}只股票")

            # 限制扫描数量
            if total_stocks > MAX_STOCKS_TO_SCAN:
                logger.info(f"限制扫描数量为{MAX_STOCKS_TO_SCAN}只股票")
                stocks = stocks.sample(MAX_STOCKS_TO_SCAN)

            # 每种策略的结果
            all_results = []

            # 逐个扫描股票
            for i, (_, stock) in enumerate(stocks.iterrows()):
                symbol = stock["symbol"]
                name = stock["name"]

                # 获取K线数据
                data = data_fetcher.get_k_data(symbol, period="daily")
                if data.empty:
                    logger.warning(f"获取 {symbol}({name}) K线数据失败，跳过")
                    continue

                # 计算基础指标
                data = BasicIndicator.calculate_all(data)

                # 应用各个策略
                stock_results = []

                for strategy_id, strategy in self.strategies.items():
                    if not strategy.is_enabled:
                        continue

                    try:
                        # 执行策略扫描
                        strategy_result = strategy.scan(data)

                        # 如果最新数据有买入信号
                        latest = strategy_result.iloc[-1]
                        if latest["signal"] > 0:
                            # 记录结果
                            result = {
                                "symbol": symbol,
                                "name": name,
                                "strategy": strategy_id,
                                "score": latest["score"],
                                "weight": strategy.weight,
                                "weighted_score": latest["score"] * strategy.weight,
                                "date": latest.name,
                                "price": latest["close"],
                                "signal": latest["signal"],
                                "indicator_values": {
                                    "MACD": latest.get("MACD", None),
                                    "MACD_diff": latest.get("MACD_diff", None),
                                    f"RSI_{strategy.params.get('rsi_length', 24)}": latest.get(f"RSI_{strategy.params.get('rsi_length', 24)}", None)
                                }
                            }
                            stock_results.append(result)

                    except Exception as e:
                        logger.error(
                            f"执行策略 {strategy_id} 扫描 {symbol}({name}) 出错: {str(e)}")
                        logger.debug(traceback.format_exc())

                # 如果有策略发现了信号
                if stock_results:
                    all_results.extend(stock_results)

                # 进度报告
                if (i + 1) % 100 == 0 or (i + 1) == len(stocks):
                    logger.info(
                        f"已扫描 {i+1}/{len(stocks)} 只股票，发现 {len(all_results)} 个信号")

            # 处理结果
            if all_results:
                self.process_results(all_results)
            else:
                logger.info("未发现任何买入信号")

            return True

        except Exception as e:
            logger.error(f"扫描股票过程出错: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def process_results(self, results: List[Dict]):
        """处理扫描结果

        Args:
            results: 扫描结果列表
        """
        logger.info(f"处理扫描结果，共 {len(results)} 个信号")

        # 转换为DataFrame便于处理
        df = pd.DataFrame(results)

        # 按照加权得分排序
        if "weighted_score" in df.columns:
            df = df.sort_values("weighted_score", ascending=False)

        # 限制结果数量
        if len(df) > MAX_RESULTS:
            df = df.head(MAX_RESULTS)

        # 保存结果
        self.last_scan_results = df.to_dict("records")

        # 输出结果
        self.output_results(df)

        # 发送通知
        if ENABLE_NOTIFICATION:
            self.send_notification(df)

    def output_results(self, results: pd.DataFrame):
        """输出结果到文件

        Args:
            results: 结果DataFrame
        """
        # 创建时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保输出目录存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # CSV输出
        if "csv" in OUTPUT_FORMAT:
            csv_file = os.path.join(
                OUTPUT_DIR, f"stock_signals_{timestamp}.csv")
            results.to_csv(csv_file, index=False, encoding="utf-8-sig")
            logger.info(f"结果已保存到CSV文件: {csv_file}")

        # HTML输出
        if "html" in OUTPUT_FORMAT:
            html_file = os.path.join(
                OUTPUT_DIR, f"stock_signals_{timestamp}.html")

            # 创建更美观的HTML
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>股票买入信号 - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .high-score {{ background-color: #dff0d8; }}
                    .signal {{ font-weight: bold; color: #c00; }}
                </style>
            </head>
            <body>
                <h1>股票买入信号</h1>
                <p>扫描时间: {scan_time}</p>
                <p>市场状态: {market_status}</p>
                <table>
                    <tr>
                        <th>代码</th>
                        <th>名称</th>
                        <th>策略</th>
                        <th>得分</th>
                        <th>权重</th>
                        <th>加权得分</th>
                        <th>日期</th>
                        <th>价格</th>
                        <th>指标</th>
                    </tr>
            """.format(
                timestamp=timestamp,
                scan_time=self.last_scan_time.strftime("%Y-%m-%d %H:%M:%S"),
                market_status=self.market_status
            )

            # 添加数据行
            for _, row in results.iterrows():
                indicators = ""
                if "indicator_values" in row:
                    for name, value in row["indicator_values"].items():
                        if value is not None:
                            indicators += f"{name}: {value:.4f}<br>"

                css_class = "high-score" if row["weighted_score"] >= 80 else ""

                html += f"""
                <tr class="{css_class}">
                    <td>{row['symbol']}</td>
                    <td>{row['name']}</td>
                    <td>{row['strategy']}</td>
                    <td>{row['score']}</td>
                    <td>{row['weight']:.2f}</td>
                    <td class="signal">{row['weighted_score']:.2f}</td>
                    <td>{row['date'].strftime('%Y-%m-%d')}</td>
                    <td>{row['price']:.2f}</td>
                    <td>{indicators}</td>
                </tr>
                """

            html += """
                </table>
            </body>
            </html>
            """

            # 保存HTML
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info(f"结果已保存到HTML文件: {html_file}")

    def send_notification(self, results: pd.DataFrame):
        """发送结果通知

        Args:
            results: 结果DataFrame
        """
        # TODO: 实现实际的通知功能（邮件、微信、钉钉等）
        pass

    def start(self):
        """启动扫描服务"""
        if self.running:
            logger.warning("扫描服务已经在运行中")
            return

        self.running = True
        logger.info("启动股票扫描服务")

        # 立即执行一次扫描
        self.scan_stocks()

        # 定时任务
        schedule.every(SCAN_INTERVAL).seconds.do(self.scan_stocks)

        # 启动定时任务线程
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(1)

        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        logger.info(f"定时扫描已设置，间隔: {SCAN_INTERVAL}秒")

        # 启动WebUI（如果配置了）
        if ENABLE_GUI:
            self.start_gui()

    def stop(self):
        """停止扫描服务"""
        if not self.running:
            logger.warning("扫描服务未在运行")
            return

        self.running = False
        logger.info("停止股票扫描服务")

        # 清除所有定时任务
        schedule.clear()

    def start_gui(self):
        """启动图形界面"""
        # TODO: 实现WebUI界面
        logger.info("WebUI界面功能尚未实现")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="技术指标股票筛选系统")
    parser.add_argument("-s", "--scan", action="store_true", help="立即执行一次扫描")
    parser.add_argument("-r", "--run", action="store_true", help="启动定时扫描服务")
    parser.add_argument("-g", "--gui", action="store_true", help="启动图形界面")
    parser.add_argument("-d", "--debug", action="store_true", help="开启调试模式")
    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 调试模式
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.add(
            LOG_FILE,
            rotation=LOG_ROTATION,
            retention="30 days",
            level="DEBUG",
            encoding="utf-8"
        )
        logger.debug("调试模式已开启")

    # 创建扫描器实例
    scanner = StockScanner()

    # 根据参数决定操作
    if args.scan:
        # 执行一次扫描
        scanner.scan_stocks()
    elif args.run or args.gui:
        # 启动服务
        scanner.start()

        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止服务...")
            scanner.stop()
    else:
        # 默认执行一次扫描
        scanner.scan_stocks()


if __name__ == "__main__":
    main()
