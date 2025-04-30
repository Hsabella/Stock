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
        if "MACD_LINE" not in data.columns or "RSI" not in data.columns:
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
            conditions["macd_crossover"] = result["MACD_GOLDEN_CROSS"]

        # 条件2: MACD零线金叉
        if self.conditions.get("macd_zero_crossover", True):
            conditions["macd_zero_crossover"] = (
                (result["MACD_LINE"].shift(1).fillna(0) <= 0) &
                (result["MACD_LINE"].fillna(0) > 0)
            )

        # 条件3: RSI超卖
        if self.conditions.get("rsi_oversold", True):
            conditions[f"rsi_{rsi_length}_oversold"] = result["RSI"].fillna(
                0) < rsi_oversold

        # 条件4: 价格位于均线之上（可选）
        if self.conditions.get("price_above_ma", False):
            # 通常使用中期均线，如MA_20
            conditions["price_above_ma"] = result.get(
                "price_above_mid_ma", pd.Series(False, index=result.index))

        # 条件5: RSI向上拐头（额外条件）
        if self.conditions.get("rsi_turning_up", False):
            conditions["rsi_turning_up"] = (
                (result["RSI"].shift(1).fillna(0) < result["RSI"].fillna(0)) &
                (result["RSI"].shift(2).fillna(0) >
                 result["RSI"].shift(1).fillna(0))
            )

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
        """计算每个交易日的策略得分，用于评估买入时机的优劣

        Args:
            data: 包含技术指标的DataFrame

        Returns:
            pd.Series: 0-100的得分，分数越高代表买入信号越强
        """
        # 获取参数
        macd_fast = self.params.get("macd_fast", 12)
        macd_slow = self.params.get("macd_slow", 26)
        macd_signal = self.params.get("macd_signal", 9)
        rsi_length = self.params.get("rsi_length", 14)
        rsi_oversold = self.params.get("rsi_oversold", 35)

        # 初始化得分为0
        score = pd.Series(0, index=data.index)

        # 1. MACD金叉： 有金叉加20分 (增加权重)
        if "MACD_GOLDEN_CROSS" in data.columns:
            score.loc[data["MACD_GOLDEN_CROSS"]] += 20

        # 2. MACD零线以上： 15分 (增加权重)
        if "MACD_LINE" in data.columns:
            # MACD值为正数，加分
            score.loc[data["MACD_LINE"].fillna(0) > 0] += 15

            # 计算MACD上穿零线的强度
            zero_cross = (data["MACD_LINE"].shift(1).fillna(
                0) <= 0) & (data["MACD_LINE"].fillna(0) > 0)

            if zero_cross.any():
                # 找到MACD零线金叉的位置
                zero_cross_idx = data.index[zero_cross]

                # 计算零线金叉的强度（MACD值的绝对值）
                if not zero_cross_idx.empty:
                    macd_cross_value = data.loc[zero_cross_idx, "MACD_LINE"].abs(
                    )

                    # 标准化到0-15分，确保数值有效且处理NA值
                    if not macd_cross_value.empty and macd_cross_value.abs().max() > 0:
                        max_value = macd_cross_value.abs().max()
                        normalized = (macd_cross_value /
                                      max_value * 15).fillna(0).clip(0, 15)

                        # 安全地将浮点值转换为整数
                        for idx in normalized.index:
                            if idx in score.index:
                                score.loc[idx] = score.loc[idx] + \
                                    normalized.loc[idx]

        # 3. RSI超卖程度: 0-20分 (增加权重)
        if "RSI" in data.columns:
            # RSI值越低，分数越高（但不包括极端异常值）
            rsi_values = data["RSI"].fillna(50)  # 使用中性值填充空值

            # 只考虑低于超卖阈值的部分，计算分数
            oversold_mask = rsi_values < rsi_oversold
            if oversold_mask.any():
                # 只对超卖的数据计算分数
                rsi_oversold_values = rsi_values[oversold_mask]
                rsi_oversold_score = (
                    (rsi_oversold - rsi_oversold_values) / rsi_oversold * 20).clip(0, 20)

                # 安全地将分数加到score上
                for idx in rsi_oversold_score.index:
                    if idx in score.index:
                        score.loc[idx] = score.loc[idx] + \
                            rsi_oversold_score.loc[idx]

        # 3.1 RSI向上拐头: 15分 (新增)
        rsi_turning_up = (data["RSI"].shift(1).fillna(0) < data["RSI"].fillna(0)) & \
            (data["RSI"].shift(2).fillna(0) > data["RSI"].shift(1).fillna(0))
        score.loc[rsi_turning_up] += 15

        # 4. 成交量确认: 0-10分
        if "VOLUME_EXPAND" in data.columns:
            # 成交量放大，加分
            score.loc[data["VOLUME_EXPAND"].fillna(False)] += 5
            if "VOLUME_MILD_EXPAND" in data.columns:
                # 温和放量，额外加分
                score.loc[data["VOLUME_MILD_EXPAND"].fillna(False)] += 5

        # 5. 价格位置确认: 0-10分（距离支撑位置远近）
        if "BB_LOWER_TOUCH" in data.columns:
            # 价格低于布林下轨，可能超跌
            score.loc[data["BB_LOWER_TOUCH"].fillna(False)] += 10

        # 6. 均线支撑确认: 0-10分 (新增)
        if "price_above_mid_ma" in data.columns:
            # 价格位于中期均线之上，表明有支撑
            score.loc[data["price_above_mid_ma"].fillna(False)] += 5

        if "MA_short_cross_mid" in data.columns:
            # 短期均线上穿中期均线，加分
            score.loc[data["MA_short_cross_mid"].fillna(False)] += 5

        # 7. 趋势持续性: 0-10分 (新增)
        # 连续上涨（收盘价连续高于前一日）
        if "close" in data.columns:
            price_rising = data["close"].fillna(
                0) > data["close"].shift(1).fillna(0)
            # 连续2天上涨
            two_days_rising = price_rising & price_rising.shift(
                1).fillna(False)
            score.loc[two_days_rising] += 5

            # 连续3天上涨
            three_days_rising = two_days_rising & price_rising.shift(
                2).fillna(False)
            score.loc[three_days_rising] += 5

        # 确保分数是浮点数，然后四舍五入到整数
        return score.astype(float).round().astype('Int64').clip(0, 100)

    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保数据包含必要的技术指标

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 添加了技术指标的数据
        """
        # 提取MACD参数
        macd_params = {
            "fast_period": self.params.get("macd_fast", 6),
            "slow_period": self.params.get("macd_slow", 12),
            "signal_period": self.params.get("macd_signal", 5)
        }

        # 提取RSI参数
        rsi_params = {
            "period": self.params.get("rsi_length", 24),
            "overbought": self.params.get("rsi_overbought", 70),
            "oversold": self.params.get("rsi_oversold", 30)
        }

        # 构建所需指标配置
        indicators_config = {
            "MACD": macd_params,
            "RSI": rsi_params,
            "MA": {"short": 5, "mid": 20, "long": 60},
            "VOLUME": {"short": 5, "long": 20},
            "BOLL": {"window": 20, "std_dev": 2.0}
        }

        # 计算指标
        return BasicIndicator.calculate_all(data, indicators_config)

    def _calculate_score(self, row: pd.Series) -> float:
        """计算每一行的综合评分

        Args:
            row: DataFrame的一行

        Returns:
            float: 信号强度评分 (0.0-1.0)
        """
        score = 0.0
        total_weight = 0

        # MACD金叉
        if self.conditions.get("macd_crossover", True):
            weight = self.weights.get("macd_crossover", 10)
            if row.get("condition_macd_crossover", False):
                score += weight
            total_weight += weight

        # MACD零线金叉
        if self.conditions.get("macd_zero_crossover", True):
            weight = self.weights.get("macd_zero_crossover", 5)
            if row.get("condition_macd_zero_crossover", False):
                score += weight
            total_weight += weight

        # RSI超卖
        rsi_length = self.params.get("rsi_length", 24)
        if self.conditions.get("rsi_oversold", True):
            weight = self.weights.get("rsi_oversold", 8)
            if row.get(f"condition_rsi_{rsi_length}_oversold", False):
                score += weight
            total_weight += weight

        # 价格位于均线之上
        if self.conditions.get("price_above_ma", False):
            weight = self.weights.get("price_above_ma", 4)
            if row.get("condition_price_above_ma", False):
                score += weight
            total_weight += weight

        # RSI向上拐头
        if self.conditions.get("rsi_turning_up", False):
            weight = self.weights.get("rsi_turning_up", 6)
            if row.get("condition_rsi_turning_up", False):
                score += weight
            total_weight += weight

        # 计算归一化评分 (0.0-1.0)
        normalized_score = score / total_weight if total_weight > 0 else 0.0
        return round(normalized_score, 3)


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
                    print(
                        f"日期: {date.strftime('%Y-%m-%d')}, 收盘价: {row['close']:.2f}, 得分: {row['score']}, MACD: {row['MACD_LINE']:.4f}, RSI: {row['RSI']:.2f}")

                # 简单回测
                backtest_result = strategy.backtest(data)
                print("\n回测结果:")
                for key, value in backtest_result.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
