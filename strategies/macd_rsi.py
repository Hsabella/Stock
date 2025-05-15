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

        # 更新最低分数 - 提高到65分
        min_score = config.get("min_score", 0)
        if 0 <= min_score <= 100:
            self.min_score = max(min_score, 65)  # 确保最低分数不低于65

        # 条件逻辑设置，默认为OR（任一满足）
        self.logic_mode = config.get("logic", "OR")

        # 应用条件设置
        self.conditions = config.get("conditions", {
            "macd_crossover": True,
            "macd_zero_crossover": False,
            "rsi_oversold": True,
            "price_above_ma": False,
            "rsi_turning_up": True
        })

        # 设置指标权重
        self.weights = {
            "macd_crossover": 20,
            "macd_zero_crossover": 15,
            "rsi_oversold": 20,
            "price_above_ma": 10,
            "rsi_turning_up": 15,
            "kdj_golden_cross": 15,
            "bb_breakout": 15,
            "ma_alignment": 15,
            "w_bottom": 20,
            "rsi_bullish_divergence": 25
        }

        # 市场环境设置
        self.market_regime = "sideways"  # 默认为震荡市

        # 设置风险过滤参数
        self.risk_filter = {
            "enable_trend_filter": True,  # 启用趋势过滤
            "downtrend_period": 10,       # 下跌趋势判断周期
            "downtrend_threshold": -5,    # 下跌趋势阈值（百分比）
            "volume_filter": True,        # 启用成交量异常过滤
            "volatility_filter": True,    # 启用波动率风险过滤
            "max_continuous_down_days": 4  # 最大连续下跌天数
        }

        # 覆盖用户提供的参数
        if params:
            self.set_params(params)

    def scan(self, data: pd.DataFrame, market_regime: str = "sideways", **kwargs) -> pd.DataFrame:
        """执行策略扫描

        Args:
            data: 数据源DataFrame，应包含价格数据
            market_regime: 市场环境，可选值为"bull"（牛市）、"bear"（熊市）、"sideways"（震荡市）
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法执行策略")
            return pd.DataFrame()

        # 更新市场环境
        self.market_regime = market_regime
        self._adjust_params_by_market_regime()

        # 确保数据包含必要的指标，如果没有则计算
        if "MACD_LINE" not in data.columns or "RSI" not in data.columns:
            logger.debug("数据中缺少必要的指标，正在计算...")
            data = self._ensure_indicators(data)

        # 生成信号
        result = self.generate_signals(data)

        # 计算策略得分
        result["score"] = self.calculate_score(result)

        # 记录原始分数，用于调试和分析
        result["original_score"] = result["score"].copy()

        # 应用最终风险过滤与信号调整
        result = self._apply_risk_filtering(result)

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
        rsi_length = self.params.get("rsi_length", 14)
        rsi_oversold = self.params.get("rsi_oversold", 35)

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

        # 条件6: KDJ金叉（如果有KDJ指标）
        if self.conditions.get("kdj_golden_cross", True) and all(col in result.columns for col in ["KDJ_K", "KDJ_D"]):
            conditions["kdj_golden_cross"] = (
                (result["KDJ_K"].shift(1).fillna(0) <= result["KDJ_D"].shift(1).fillna(0)) &
                (result["KDJ_K"].fillna(0) > result["KDJ_D"].fillna(0))
            )

        # 条件7: 布林带突破（如果有布林带指标）
        if self.conditions.get("bb_breakout", True) and all(col in result.columns for col in ["close", "BB_LOWER"]):
            conditions["bb_breakout"] = (
                (result["close"].shift(1).fillna(0) <= result["BB_LOWER"].shift(1).fillna(0)) &
                (result["close"].fillna(0) > result["BB_LOWER"].fillna(0))
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
        score = pd.Series(0.0, index=data.index)  # 使用浮点数类型

        # 1. MACD金叉： 有金叉加20分 (增加权重)
        if "MACD_GOLDEN_CROSS" in data.columns:
            # 将布尔序列转换为浮点数
            macd_golden_cross = data["MACD_GOLDEN_CROSS"].astype(float)
            score = score.add(macd_golden_cross * 20.0, fill_value=0.0)

        # 2. MACD零线以上： 15分 (增加权重)
        if "MACD_LINE" in data.columns:
            # MACD值为正数，加分
            macd_positive = (data["MACD_LINE"].fillna(0) > 0).astype(float)
            score = score.add(macd_positive * 15.0, fill_value=0.0)

            # 计算MACD上穿零线的强度
            macd_prev = data["MACD_LINE"].shift(1).fillna(0)
            macd_curr = data["MACD_LINE"].fillna(0)
            zero_cross = (macd_prev <= 0) & (macd_curr > 0)

            if zero_cross.any():
                # 找到MACD零线金叉的位置
                zero_cross_idx = data.index[zero_cross]

                # 计算零线金叉的强度（MACD值的绝对值）
                if not zero_cross_idx.empty:
                    macd_cross_value = data.loc[zero_cross_idx, "MACD_LINE"].abs(
                    )

                    # 标准化到0-15分，确保数值有效且处理NA值
                    if not macd_cross_value.empty and macd_cross_value.max() > 0:
                        max_value = macd_cross_value.max()
                        normalized = (macd_cross_value /
                                      max_value * 15.0).fillna(0)

                        # 使用安全的加法操作
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
                    (rsi_oversold - rsi_oversold_values) / rsi_oversold * 20.0).clip(0, 20)

                # 使用安全的加法操作
                for idx in rsi_oversold_score.index:
                    if idx in score.index:
                        score.loc[idx] = score.loc[idx] + \
                            rsi_oversold_score.loc[idx]

        # 3.1 RSI向上拐头: 15分 (新增)
        if "RSI" in data.columns:
            rsi_curr = data["RSI"].fillna(0)
            rsi_prev1 = data["RSI"].shift(1).fillna(0)
            rsi_prev2 = data["RSI"].shift(2).fillna(0)
            rsi_turning_up = (rsi_prev1 < rsi_curr) & (rsi_prev2 > rsi_prev1)
            score = score.add(rsi_turning_up.astype(float)
                              * 15.0, fill_value=0.0)

        # 4. 成交量确认: 0-10分
        if "VOLUME_EXPAND" in data.columns:
            # 成交量放大，加分
            volume_expand = data["VOLUME_EXPAND"].fillna(False).astype(float)
            score = score.add(volume_expand * 5.0, fill_value=0.0)

            if "VOLUME_MILD_EXPAND" in data.columns:
                # 温和放量，额外加分
                volume_mild_expand = data["VOLUME_MILD_EXPAND"].fillna(
                    False).astype(float)
                score = score.add(volume_mild_expand * 5.0, fill_value=0.0)
        else:
            # 缺少成交量数据时的替代策略：使用价格范围作为波动性指标
            if "high" in data.columns and "low" in data.columns:
                # 计算当日价格范围相对于N日平均的比例
                price_range = data["high"] - data["low"]
                avg_range = price_range.rolling(window=10).mean()
                range_ratio = price_range / avg_range.replace(0, 0.001)

                # 价格范围扩大视为成交量放大的替代指标
                volume_expand_proxy = (range_ratio > 1.5).astype(float)
                score = score.add(volume_expand_proxy * 5.0, fill_value=0.0)

                # 区分突破性放量和衰竭性放量
                if "close" in data.columns and "open" in data.columns:
                    strong_up = ((data["close"] > data["open"]) & (
                        range_ratio > 1.5)).astype(float)
                    score = score.add(strong_up * 5.0, fill_value=0.0)

        # 5. 价格位置确认: 0-10分（距离支撑位置远近）
        if "BB_LOWER_TOUCH" in data.columns:
            # 价格低于布林下轨，可能超跌
            bb_lower_touch = data["BB_LOWER_TOUCH"].fillna(False).astype(float)
            score = score.add(bb_lower_touch * 10.0, fill_value=0.0)

        # 5.1 布林带突破: +15分 (新增)
        if "close" in data.columns and "BB_LOWER" in data.columns:
            # 价格从下轨向上突破 +15分
            close_curr = data["close"].fillna(0)
            close_prev = data["close"].shift(1).fillna(0)
            bb_lower_curr = data["BB_LOWER"].fillna(0)
            bb_lower_prev = data["BB_LOWER"].shift(1).fillna(0)

            bb_lower_breakout = (close_prev <= bb_lower_prev) & (
                close_curr > bb_lower_curr)
            score = score.add(bb_lower_breakout.astype(float)
                              * 15.0, fill_value=0.0)

        # 5.2 布林带收窄后扩张: +10分 (新增)
        if "BB_SQUEEZE" in data.columns:
            # 布林带收窄后扩张 (波动率即将增加的信号)
            squeeze_prev = data["BB_SQUEEZE"].shift(
                2).fillna(False).astype(bool)
            squeeze_curr = ~data["BB_SQUEEZE"].shift(
                1).fillna(False).astype(bool)
            bb_squeeze_release = (squeeze_prev & squeeze_curr).astype(float)
            score = score.add(bb_squeeze_release * 10.0, fill_value=0.0)

        # 6. 均线支撑确认: 0-10分 (新增)
        if "price_above_mid_ma" in data.columns:
            # 价格位于中期均线之上，表明有支撑
            price_above_ma = data["price_above_mid_ma"].fillna(
                False).astype(float)
            score = score.add(price_above_ma * 5.0, fill_value=0.0)

        if "MA_short_cross_mid" in data.columns:
            # 短期均线上穿中期均线，加分
            ma_cross = data["MA_short_cross_mid"].fillna(False).astype(float)
            score = score.add(ma_cross * 5.0, fill_value=0.0)

        # 6.1 均线多周期系统: 0-25分 (新增)
        # 检查均线多头排列 (5日>10日>20日>60日)
        if all(f"MA_{period}" in data.columns for period in [5, 20, 60]):
            if "MA_10" in data.columns:
                # 均线多头排列 (5>10>20>60)
                ma5 = data["MA_5"].fillna(0)
                ma10 = data["MA_10"].fillna(0)
                ma20 = data["MA_20"].fillna(0)
                ma60 = data["MA_60"].fillna(0)

                ma_bull_aligned = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
                score = score.add(ma_bull_aligned.astype(
                    float) * 15.0, fill_value=0.0)
            else:
                # 简化版多头排列 (5>20>60)
                ma5 = data["MA_5"].fillna(0)
                ma20 = data["MA_20"].fillna(0)
                ma60 = data["MA_60"].fillna(0)

                ma_bull_aligned = (ma5 > ma20) & (ma20 > ma60)
                score = score.add(ma_bull_aligned.astype(
                    float) * 10.0, fill_value=0.0)

            # 价格站上所有均线
            above_all_ma = (data["close"].fillna(
                0) > data["MA_5"].fillna(0)).astype(bool)
            for period in [20, 60]:
                above_all_ma = above_all_ma & (data["close"].fillna(
                    0) > data[f"MA_{period}"].fillna(0)).astype(bool)
            score = score.add(above_all_ma.astype(float)
                              * 10.0, fill_value=0.0)

        # 7. 趋势持续性: 0-10分 (新增)
        # 连续上涨（收盘价连续高于前一日）
        if "close" in data.columns:
            # 处理连续上涨
            price_rising = (data["close"].fillna(
                0) > data["close"].shift(1).fillna(0)).astype(bool)

            # 连续2天上涨
            two_days_rising = price_rising & price_rising.shift(
                1).fillna(False).astype(bool)
            score = score.add(two_days_rising.astype(float)
                              * 5.0, fill_value=0.0)

            # 连续3天上涨
            three_days_rising = two_days_rising & price_rising.shift(
                2).fillna(False).astype(bool)
            score = score.add(three_days_rising.astype(float)
                              * 5.0, fill_value=0.0)

        # 8. KDJ指标评分: 0-25分 (新增)
        if all(col in data.columns for col in ["KDJ_K", "KDJ_D", "KDJ_J"]):
            # KDJ金叉
            k_prev = data["KDJ_K"].shift(1).fillna(0)
            k_curr = data["KDJ_K"].fillna(0)
            d_prev = data["KDJ_D"].shift(1).fillna(0)
            d_curr = data["KDJ_D"].fillna(0)

            kdj_golden_cross = (k_prev <= d_prev) & (k_curr > d_curr)
            score = score.add(kdj_golden_cross.astype(float)
                              * 15.0, fill_value=0.0)

            # KDJ超卖区间反转
            j_prev = data["KDJ_J"].shift(1).fillna(0)
            j_curr = data["KDJ_J"].fillna(0)

            kdj_oversold_reverse = (j_prev < 20) & (j_curr > j_prev)
            score = score.add(kdj_oversold_reverse.astype(
                float) * 10.0, fill_value=0.0)

        # 9. RSI背离判断: 0-25分 (新增，更精确的背离检测)
        if "RSI" in data.columns and "close" in data.columns:
            lookback = min(20, len(data) - 1)  # 确保不超过数据长度

            # 对每个交易日进行检查
            for i in range(lookback, len(data)):
                current_idx = data.index[i]
                window = data.iloc[i-lookback:i+1]

                if len(window) < 5:  # 确保有足够的数据点
                    continue

                # 寻找窗口内的价格最低点
                if "low" in window.columns:
                    price_series = window["low"]  # 使用最低价检测底部背离
                else:
                    price_series = window["close"]

                # 如果当前价格接近窗口内的最低点（差距不超过3%）
                if not price_series.empty:
                    min_price = price_series.min()
                    min_price_idx = price_series.idxmin()
                    if len(price_series) > 0:
                        current_price = price_series.iloc[-1]

                        # 价格接近最低点
                        if current_price <= min_price * 1.03:
                            # 检查RSI是否高于对应的最低点RSI值
                            if min_price_idx in window.index and current_idx in window.index:
                                min_price_rsi = window.loc[min_price_idx, "RSI"]
                                current_rsi = window.loc[current_idx, "RSI"]

                                # 安全地检查RSI值
                                if pd.notna(min_price_rsi) and pd.notna(current_rsi):
                                    # 如果当前RSI高于最低点的RSI，认为出现看涨背离
                                    if current_rsi > min_price_rsi * 1.1:  # RSI高出10%以上
                                        score.loc[current_idx] += 25.0

        # 10. 下跌趋势过滤器 (新增) - 大幅降低强下跌趋势中的得分
        if self.risk_filter["enable_trend_filter"] and "close" in data.columns:
            downtrend_period = self.risk_filter["downtrend_period"]
            downtrend_threshold = self.risk_filter["downtrend_threshold"]

            if len(data) >= downtrend_period:
                # 计算N日价格变化百分比
                close_current = data["close"].fillna(0)
                close_past = data["close"].shift(downtrend_period).fillna(0)
                # 避免除以零
                close_past_safe = close_past.replace(0, np.nan)
                price_change_pct = (close_current / close_past_safe - 1) * 100
                price_change_pct = price_change_pct.fillna(0)  # 填充可能的NaN

                # 强下跌趋势条件：N日跌幅超过阈值
                strong_downtrend = price_change_pct < downtrend_threshold
                score = score.mask(strong_downtrend, score - 40.0)

                # 检查连续下跌天数
                max_down_days = self.risk_filter["max_continuous_down_days"]
                if max_down_days > 0:
                    # 计算最近的连续下跌天数
                    price_drop = data["close"] < data["close"].shift(1)
                    continuous_drop_days = pd.Series(0, index=data.index)

                    # 安全地计算连续下跌天数
                    for i in range(1, len(data)):
                        if i < len(price_drop) and price_drop.iloc[i]:
                            if i > 0:
                                continuous_drop_days.iloc[i] = continuous_drop_days.iloc[i-1] + 1

                    # 连续下跌超过阈值，大幅减分
                    excessive_drop_days = continuous_drop_days >= max_down_days
                    score = score.mask(excessive_drop_days, score - 30.0)

        # 11. 成交量异常过滤器 (新增)
        if self.risk_filter["volume_filter"] and "volume" in data.columns and "close" in data.columns:
            # 成交量异常放大但价格下跌
            volume_current = data["volume"].fillna(0)
            volume_mean = data["volume"].rolling(20).mean().fillna(0)
            volume_surge = volume_current > volume_mean * 2

            price_drop = data["close"] < data["close"].shift(1).fillna(0)
            volume_price_diverge = volume_surge & price_drop
            score = score.mask(volume_price_diverge, score - 30.0)

        # 12. 波动率风险过滤器 (新增)
        if self.risk_filter["volatility_filter"] and all(col in data.columns for col in ["high", "low", "close"]):
            # 计算真实波动幅度(ATR)
            true_range = pd.DataFrame(index=data.index)
            true_range["hl"] = data["high"] - data["low"]
            true_range["hc"] = (data["high"] - data["close"].shift(1)).abs()
            true_range["lc"] = (data["low"] - data["close"].shift(1)).abs()

            # 计算真实波动幅度
            tr = true_range.max(axis=1)
            atr14 = tr.rolling(14).mean().fillna(0)

            # 计算波动率相对于历史的水平
            atr30_mean = atr14.rolling(30).mean().fillna(0)
            # 避免除以零
            atr30_mean_safe = atr30_mean.replace(0, np.nan)
            atr_ratio = atr14 / atr30_mean_safe
            atr_ratio = atr_ratio.fillna(0)  # 填充可能的NaN

            # 波动率异常高时减分
            high_volatility = atr_ratio > 1.5
            score = score.mask(high_volatility, score - 20.0)

        # 确保分数范围在0-100之间
        score = score.clip(0.0, 100.0)

        # 四舍五入到整数
        return score.round().astype(int)

    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保数据包含必要的技术指标

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 添加了技术指标的数据
        """
        # 提取MACD参数
        macd_params = {
            "fast_period": self.params.get("macd_fast", 12),
            "slow_period": self.params.get("macd_slow", 26),
            "signal_period": self.params.get("macd_signal", 9)
        }

        # 提取RSI参数
        rsi_params = {
            "period": self.params.get("rsi_length", 14),
            "overbought": self.params.get("rsi_overbought", 70),
            "oversold": self.params.get("rsi_oversold", 35)
        }

        # 构建所需指标配置
        indicators_config = {
            "MACD": macd_params,
            "RSI": rsi_params,
            "MA": {"short": 5, "mid": 20, "long": 60},
            "VOLUME": {"short": 5, "long": 20},
            "BOLL": {"window": 20, "std_dev": 2.0},
            "KDJ": {"n": 9, "m1": 3, "m2": 3}  # 添加KDJ指标
        }

        # 计算指标
        try:
            return BasicIndicator.calculate_all(data, indicators_config)
        except Exception as e:
            logger.error(f"确保指标时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return data

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
            weight = self.weights.get("macd_crossover", 20)
            if row.get("condition_macd_crossover", False):
                score += weight
            total_weight += weight

        # MACD零线金叉
        if self.conditions.get("macd_zero_crossover", True):
            weight = self.weights.get("macd_zero_crossover", 15)
            if row.get("condition_macd_zero_crossover", False):
                score += weight
            total_weight += weight

        # RSI超卖
        rsi_length = self.params.get("rsi_length", 14)
        if self.conditions.get("rsi_oversold", True):
            weight = self.weights.get("rsi_oversold", 20)
            if row.get(f"condition_rsi_{rsi_length}_oversold", False):
                score += weight
            total_weight += weight

        # 价格位于均线之上
        if self.conditions.get("price_above_ma", False):
            weight = self.weights.get("price_above_ma", 10)
            if row.get("condition_price_above_ma", False):
                score += weight
            total_weight += weight

        # RSI向上拐头
        if self.conditions.get("rsi_turning_up", True):
            weight = self.weights.get("rsi_turning_up", 15)
            if row.get("condition_rsi_turning_up", False):
                score += weight
            total_weight += weight

        # KDJ金叉
        if self.conditions.get("kdj_golden_cross", True):
            weight = self.weights.get("kdj_golden_cross", 15)
            if row.get("condition_kdj_golden_cross", False):
                score += weight
            total_weight += weight

        # 布林带突破
        if self.conditions.get("bb_breakout", True):
            weight = self.weights.get("bb_breakout", 15)
            if row.get("condition_bb_breakout", False):
                score += weight
            total_weight += weight

        # 计算归一化评分 (0.0-1.0)
        normalized_score = score / total_weight if total_weight > 0 else 0.0
        return round(normalized_score, 3)

    def _adjust_params_by_market_regime(self):
        """根据市场环境调整策略参数"""
        if self.market_regime == "bull":
            # 牛市参数：重视突破和动量
            self.params["rsi_oversold"] = 40  # 提高超卖阈值
            self.weights["macd_crossover"] = 25  # 增加MACD金叉权重
            self.weights["ma_alignment"] = 20  # 增加均线多头排列权重
            self.min_score = max(60, self.min_score)  # 确保最低分数不低于60
            # 牛市中适当放宽下跌过滤条件
            self.risk_filter["downtrend_threshold"] = -8
            self.risk_filter["max_continuous_down_days"] = 5

        elif self.market_regime == "bear":
            # 熊市参数：重视超卖和背离
            self.params["rsi_oversold"] = 30  # 降低超卖阈值
            self.weights["rsi_oversold"] = 25  # 增加RSI超卖权重
            self.weights["rsi_bullish_divergence"] = 30  # 增加RSI背离权重
            self.min_score = max(70, self.min_score)  # 熊市中提高信号要求
            # 熊市中加强下跌过滤条件
            self.risk_filter["downtrend_threshold"] = -5
            self.risk_filter["max_continuous_down_days"] = 3

        else:  # sideways 震荡市
            # 使用默认参数
            self.params["rsi_oversold"] = 35
            self.min_score = max(65, self.min_score)  # 确保最低分数不低于65
            # 震荡市中使用适中的下跌过滤条件
            self.risk_filter["downtrend_threshold"] = -6
            self.risk_filter["max_continuous_down_days"] = 4

    def _apply_risk_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用风险过滤机制进行最终信号调整

        Args:
            data: 包含技术指标和初始分数的DataFrame

        Returns:
            pd.DataFrame: 调整后的DataFrame
        """
        result = data.copy()

        # 使用20日移动均线判断总体趋势
        if "MA_20" in result.columns and "close" in result.columns:
            # 检查价格是否处于中期均线之下，若是则降低分数
            below_mid_ma = result["close"] < result["MA_20"]
            score_float = result["score"].astype(float)  # 转为浮点数进行计算

            if self.market_regime == "bear":
                # 熊市中，价格在均线下方不大幅减分，因为可能是超跌反弹机会
                result.loc[below_mid_ma, "score"] = (
                    score_float.loc[below_mid_ma] * 0.9).round().astype(int)
            else:
                # 震荡或牛市中，价格在均线下方是较大风险
                result.loc[below_mid_ma, "score"] = (
                    score_float.loc[below_mid_ma] * 0.7).round().astype(int)

        # 检查当日价格有无大幅跳空低开
        if "open" in result.columns and "close" in result.columns:
            # 安全地计算跳空比例
            close_prev = result["close"].shift(1).fillna(0)
            # 避免除以零
            close_prev_safe = close_prev.replace(0, np.nan)
            gap_ratio = (result["open"] / close_prev_safe - 1)
            gap_ratio = gap_ratio.fillna(0)  # 填充可能的NaN

            # 大幅跳空低开（超过2%）
            gap_down = gap_ratio < -0.02
            score_float = result["score"].astype(float)  # 转为浮点数进行计算
            result.loc[gap_down, "score"] = (
                score_float.loc[gap_down] * 0.7).round().astype(int)

        # 市场状态调整
        if self.market_regime == "bear":
            # 熊市中需要更高的得分才能产生信号
            bear_market_mask = result["score"] < 85
            score_float = result["score"].astype(float)  # 转为浮点数进行计算
            result.loc[bear_market_mask, "score"] = (
                score_float.loc[bear_market_mask] * 0.8).round().astype(int)

        elif self.market_regime == "bull":
            # 牛市中对背离和反转信号的要求可以适当放宽
            bull_market_boost = result["score"] > 50
            score_float = result["score"].astype(float)  # 转为浮点数进行计算
            result.loc[bull_market_boost, "score"] = (
                score_float.loc[bull_market_boost] * 1.1).round().astype(int)

        # 确保分数在合理范围内
        result["score"] = result["score"].clip(0, 100)

        return result


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
