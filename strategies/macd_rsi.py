#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MACD与RSI策略实现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from loguru import logger

from strategies.base import StrategyBase
from indicators.basic import BasicIndicator
from config.strategies import MACD_RSI_STRATEGY


class MacdRsiStrategy(StrategyBase):
    """MACD与RSI策略"""

    def __init__(self, params: Dict = None):
        """初始化MACD与RSI策略

        Args:
            params: 策略参数，覆盖默认参数
        """
        # 使用配置文件中的参数
        config = MACD_RSI_STRATEGY.copy()

        # 初始化基类
        super().__init__(
            name=config.get("name", "MACD与RSI超卖策略"),
            description=config.get("description", "MACD金叉与RSI低位共振策略"),
            params=config.get("params", {})
        )

        # 更新策略启用状态
        self.is_enabled = config.get("enabled", True)

        # 更新权重
        weight = config.get("weight", 1.0)
        if 0 <= weight <= 1:
            self.weight = weight

        # 更新最低分数
        min_score = config.get("min_score", 0)
        if 0 <= min_score <= 100:
            self.min_score = min_score

        # 条件逻辑设置，默认为AND（全部满足）
        self.logic_mode = config.get("logic", "AND")

        # 应用条件设置
        self.conditions = config.get("conditions", {
            "macd_crossover": True,
            "macd_zero_crossover": True,
            "rsi_oversold": True,
            "price_above_ma": False
        })

        # 覆盖用户提供的参数
        if params:
            self.set_params(params)

    def scan(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """执行策略扫描

        Args:
            data: 数据源DataFrame，应包含价格数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法执行策略")
            return pd.DataFrame()

        # 确保数据包含必要的指标，如果没有则计算
        if "MACD" not in data.columns or f"RSI_{self.params.get('rsi_length', 24)}" not in data.columns:
            logger.debug("数据中缺少必要的指标，正在计算...")
            data = self._ensure_indicators(data)

        # 生成信号
        result = self.generate_signals(data)

        # 计算策略得分
        result["score"] = self.calculate_score(result)

        # 根据得分生成最终信号
        result["signal"] = 0  # 默认无信号
        result.loc[result["score"] >= self.min_score, "signal"] = 1  # 买入信号

        # 记录最新的信号点
        if any(result["signal"] > 0):
            latest_signal = result[result["signal"] > 0].iloc[-1]
            logger.info(
                f"MACD与RSI策略发现买入信号: {latest_signal.name}, 分数: {latest_signal['score']:.2f}")

        return result

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于指标生成详细信号

        Args:
            data: 包含指标的DataFrame

        Returns:
            pd.DataFrame: 添加了详细信号的DataFrame
        """
        result = data.copy()

        # 提取参数
        rsi_length = self.params.get("rsi_length", 24)
        rsi_oversold = self.params.get("rsi_oversold", 30)

        # 计算各条件
        conditions = {}

        # 条件1: MACD金叉
        if self.conditions.get("macd_crossover", True):
            conditions["macd_crossover"] = result["MACD_golden_cross"]

        # 条件2: MACD零线金叉
        if self.conditions.get("macd_zero_crossover", True):
            conditions["macd_zero_crossover"] = result["MACD_zero_golden_cross"]

        # 条件3: RSI超卖
        if self.conditions.get("rsi_oversold", True):
            conditions[f"rsi_{rsi_length}_oversold"] = result[f"RSI_{rsi_length}"] < rsi_oversold

        # 条件4: 价格位于均线之上（可选）
        if self.conditions.get("price_above_ma", False):
            # 通常使用中期均线，如MA_20
            conditions["price_above_ma"] = result.get(
                "price_above_mid_ma", pd.Series(False, index=result.index))

        # 条件5: RSI向上拐头（额外条件）
        if self.conditions.get("rsi_turning_up", False):
            conditions["rsi_turning_up"] = result.get(
                f"RSI_{rsi_length}_rising", pd.Series(False, index=result.index))

        # 将条件结果添加到DataFrame
        for name, condition in conditions.items():
            result[f"condition_{name}"] = condition

        # 根据逻辑模式计算最终条件
        if self.logic_mode == "AND":
            # 所有条件必须满足
            result["conditions_met"] = pd.Series(True, index=result.index)
            for condition in conditions.values():
                result["conditions_met"] &= condition

        elif self.logic_mode == "OR":
            # 满足任一条件
            result["conditions_met"] = pd.Series(False, index=result.index)
            for condition in conditions.values():
                result["conditions_met"] |= condition

        else:  # 自定义逻辑
            # 默认为AND逻辑
            result["conditions_met"] = pd.Series(True, index=result.index)
            for condition in conditions.values():
                result["conditions_met"] &= condition

        return result

    def calculate_score(self, data: pd.DataFrame) -> pd.Series:
        """计算策略得分

        Args:
            data: DataFrame，应包含生成的信号

        Returns:
            pd.Series: 策略得分（0-100）
        """
        # 初始化得分
        score = pd.Series(0, index=data.index)

        # 提取参数
        rsi_length = self.params.get("rsi_length", 24)
        rsi_oversold = self.params.get("rsi_oversold", 30)

        # 1. 条件满足基础分: 60分
        mask_conditions_met = data.get(
            "conditions_met", pd.Series(False, index=data.index))
        score.loc[mask_conditions_met] += 60

        # 2. MACD金叉强度: 0-15分
        if "MACD_diff" in data.columns:
            # MACD金叉当天的差值，越大得分越高
            macd_cross_value = data.loc[data.get(
                "MACD_golden_cross", False), "MACD_diff"]
            if not macd_cross_value.empty:
                # 标准化到0-15分
                normalized = (macd_cross_value /
                              macd_cross_value.abs().max() * 15).clip(0, 15)
                score.loc[normalized.index] += normalized

        # 3. RSI超卖程度: 0-15分
        if f"RSI_{rsi_length}" in data.columns:
            # RSI值越低，分数越高（但不包括极端异常值）
            rsi_values = data[f"RSI_{rsi_length}"]
            # 只考虑低于超卖阈值的部分
            rsi_oversold_score = (
                (rsi_oversold - rsi_values) / rsi_oversold * 15).clip(0, 15)
            score += rsi_oversold_score

        # 4. 成交量确认: 0-10分
        if "VOLUME_EXPAND" in data.columns:
            # 成交量放大，加分
            score.loc[data["VOLUME_EXPAND"]] += 5
            if "VOLUME_MILD_EXPAND" in data.columns:
                # 温和放量，额外加分
                score.loc[data["VOLUME_MILD_EXPAND"]] += 5

        # 5. 价格位置确认: 0-10分（距离支撑位置远近）
        if "PRICE_BELOW_BOLL_LOWER" in data.columns:
            # 价格低于布林下轨，可能超跌
            score.loc[data["PRICE_BELOW_BOLL_LOWER"]] += 10

        # 四舍五入到整数
        return score.round().astype(int).clip(0, 100)

    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保数据包含必要的技术指标

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 添加了技术指标的数据
        """
        # 提取MACD参数
        macd_params = {
            "fast": self.params.get("macd_fast", 6),
            "slow": self.params.get("macd_slow", 12),
            "signal": self.params.get("macd_signal", 5)
        }

        # 提取RSI参数
        rsi_params = {
            "length": self.params.get("rsi_length", 24),
            "overbought": self.params.get("rsi_overbought", 70),
            "oversold": self.params.get("rsi_oversold", 30)
        }

        # 构建所需指标配置
        indicators_config = {
            "MACD": macd_params,
            "RSI": rsi_params,
            "MA": {"short": 5, "mid": 20, "long": 60},
            "VOLUME": {"short": 5, "long": 20},
            "BOLL": {"length": 20, "std": 2.0}
        }

        # 计算指标
        return BasicIndicator.calculate_all(data, indicators_config)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(".")  # 添加项目根目录到路径

    from data.fetcher import data_fetcher

    # 获取一支股票的数据
    stock_list = data_fetcher.get_stock_list()
    if not stock_list.empty:
        # 使用第一只股票进行测试
        symbol = stock_list.iloc[0]["symbol"]
        print(f"使用{symbol}测试MACD与RSI策略...")

        # 获取历史K线
        data = data_fetcher.get_k_data(symbol, period="daily")

        if not data.empty:
            # 初始化策略
            strategy = MacdRsiStrategy()
            print(f"策略信息: {strategy}")

            # 执行策略扫描
            result = strategy.scan(data)

            # 统计买入信号
            buy_signals = result[result["signal"] > 0]
            print(f"共发现{len(buy_signals)}个买入信号")

            if not buy_signals.empty:
                print("\n最近的买入信号:")
                latest_signals = buy_signals.tail(3)
                # 添加信号标签
                for date, row in latest_signals.iterrows():
                    rsi_col = f"RSI_{strategy.params.get('rsi_length', 24)}"
                    print(
                        f"日期: {date.strftime('%Y-%m-%d')}, 收盘价: {row['close']:.2f}, 得分: {row['score']}, MACD: {row['MACD']:.4f}, RSI: {row[rsi_col]:.2f}")

                # 简单回测
                backtest_result = strategy.backtest(data)
                print("\n回测结果:")
                for key, value in backtest_result.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
