#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础技术指标计算模块
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from loguru import logger

from config.settings import DEFAULT_INDICATORS


class BasicIndicator:
    """基础技术指标计算类"""

    @staticmethod
    def calculate_all(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """计算所有基础指标

        Args:
            df: 原始数据
            config: 指标配置

        Returns:
            pd.DataFrame: 包含计算后的所有指标的DataFrame
        """
        # 创建计算器实例
        calculator = BasicIndicator()

        # 使用输入数据的拷贝
        result_df = df.copy()

        # 配置默认值
        default_config = {
            "MA": {"short": 5, "mid": 20, "long": 60},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "RSI": {"period": 14, "overbought": 70, "oversold": 30},
            "BOLL": {"window": 20, "std_dev": 2.0},
            "KDJ": {"n": 9, "m1": 3, "m2": 3},
            "VOLUME": {"short": 5, "long": 20}
        }

        # 合并用户配置
        if config:
            for key, value in config.items():
                if key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        # 检查是否有成交量数据
        has_volume = "volume" in result_df.columns and not result_df["volume"].isna(
        ).all()

        if not has_volume:
            # 尝试使用价格范围作为成交量的替代
            logger.info("使用价格波动范围作为成交量的替代")
            if all(col in result_df.columns for col in ["high", "low"]):
                result_df["volume"] = (
                    result_df["high"] - result_df["low"]) * 1000
                has_volume = True

        try:
            # 添加移动平均线
            if "MA" in default_config:
                ma_config = default_config["MA"]
                result_df = BasicIndicator.add_ma(
                    result_df,
                    price_col="close",
                    period_short=ma_config["short"],
                    period_mid=ma_config["mid"],
                    period_long=ma_config["long"]
                )

            # 添加MACD
            if "MACD" in default_config:
                macd_config = default_config["MACD"]
                result_df = BasicIndicator.add_macd(
                    result_df,
                    price_col="close",
                    fast_period=macd_config["fast_period"],
                    slow_period=macd_config["slow_period"],
                    signal_period=macd_config["signal_period"]
                )

            # 添加RSI
            if "RSI" in default_config:
                rsi_config = default_config["RSI"]
                result_df = BasicIndicator.add_rsi(
                    result_df,
                    price_col="close",
                    period=rsi_config["period"],
                    overbought=rsi_config["overbought"],
                    oversold=rsi_config["oversold"]
                )

            # 添加布林带
            if "BOLL" in default_config:
                boll_config = default_config["BOLL"]
                result_df = BasicIndicator.add_bollinger(
                    result_df,
                    price_col="close",
                    window=boll_config["window"],
                    std_dev=boll_config["std_dev"]
                )

            # 添加KDJ
            if "KDJ" in default_config:
                kdj_config = default_config["KDJ"]
                result_df = BasicIndicator.add_kdj(
                    result_df,
                    n=kdj_config["n"],
                    m1=kdj_config["m1"],
                    m2=kdj_config["m2"]
                )

            # 添加成交量指标
            if has_volume and "VOLUME" in default_config:
                vol_config = default_config["VOLUME"]
                result_df = BasicIndicator.add_volume_indicators(
                    result_df,
                    period_short=vol_config["short"],
                    period_long=vol_config["long"]
                )

            # 缺失值处理 - 填充技术指标的缺失值
            for col in result_df.columns:
                if col not in df.columns:  # 只处理新增的指标列
                    # 对于非布尔类型的列，使用0填充
                    if result_df[col].dtype != bool:
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        # 对于布尔列，使用False填充
                        result_df[col] = result_df[col].fillna(False)

        except Exception as e:
            logger.error(f"计算指标时发生错误: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

        return result_df

    @staticmethod
    def add_macd(df: pd.DataFrame, price_col: str = "close",
                 fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9) -> pd.DataFrame:
        """添加MACD指标

        Args:
            df: 原始数据
            price_col: 价格列名，默认为'close'
            fast_period: 快线周期，默认为12
            slow_period: 慢线周期，默认为26
            signal_period: 信号线周期，默认为9

        Returns:
            pd.DataFrame: 包含MACD指标的DataFrame
        """
        try:
            # 检查价格列是否存在
            if price_col not in df.columns:
                logger.error(f"MACD计算错误: 列 '{price_col}' 不存在于数据中")
                return df

            # 创建副本，避免警告
            df_copy = df.copy()

            # 确保价格数据完整，填充缺失值
            price = df_copy[price_col].ffill().bfill()

            # 计算EMA
            ema_fast = price.ewm(span=fast_period, adjust=False).mean()
            ema_slow = price.ewm(span=slow_period, adjust=False).mean()

            # 计算MACD线和信号线
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(
                span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            # 创建结果DataFrame
            result_df = df_copy.copy()
            result_df["MACD_LINE"] = macd_line
            result_df["MACD_SIGNAL"] = signal_line
            result_df["MACD_HIST"] = histogram

            # 检测MACD金叉和死叉
            result_df["MACD_GOLDEN_CROSS"] = (macd_line.shift(
                1) <= signal_line.shift(1)) & (macd_line > signal_line)
            result_df["MACD_DEATH_CROSS"] = (macd_line.shift(
                1) >= signal_line.shift(1)) & (macd_line < signal_line)

            # 检测MACD柱状图方向变化（拐点）
            hist_shift = histogram.shift(1)
            result_df["MACD_HIST_REVERSAL_UP"] = (
                hist_shift < 0) & (histogram > hist_shift)
            result_df["MACD_HIST_REVERSAL_DOWN"] = (
                hist_shift > 0) & (histogram < hist_shift)

            # 检测零轴金叉和死叉
            result_df["MACD_ZERO_GOLDEN_CROSS"] = (
                macd_line.shift(1) <= 0) & (macd_line > 0)
            result_df["MACD_ZERO_DEATH_CROSS"] = (
                macd_line.shift(1) >= 0) & (macd_line < 0)

            return result_df

        except Exception as e:
            logger.error(f"MACD计算错误: {str(e)}")
            return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, price_col: str = "close", period: int = 14,
                overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
        """添加RSI指标

        Args:
            df: 原始数据
            price_col: 价格列名，默认为'close'
            period: 计算周期，默认为14
            overbought: 超买阈值，默认为70
            oversold: 超卖阈值，默认为30

        Returns:
            pd.DataFrame: 包含RSI指标的DataFrame
        """
        try:
            # 检查价格列是否存在
            if price_col not in df.columns:
                logger.error(f"RSI计算错误: 列 '{price_col}' 不存在于数据中")
                return df

            # 创建副本，避免警告
            df_copy = df.copy()

            # 确保价格数据完整，填充缺失值
            price = df_copy[price_col].ffill().bfill()

            # 计算价格变化
            delta = price.diff()

            # 区分上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # 计算初始平均值
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # 计算相对强度
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # 避免除以零

            # 计算RSI
            rsi = 100 - (100 / (1 + rs))

            # 创建结果DataFrame
            result_df = df_copy.copy()
            result_df["RSI"] = rsi

            # 检测RSI金叉和死叉（相对于超买超卖线）
            result_df["RSI_GOLDEN_CROSS"] = (
                rsi.shift(1) <= oversold) & (rsi > oversold)
            result_df["RSI_DEATH_CROSS"] = (
                rsi.shift(1) >= overbought) & (rsi < overbought)

            # 添加超买超卖信号
            result_df["RSI_OVERBOUGHT"] = rsi > overbought
            result_df["RSI_OVERSOLD"] = rsi < oversold

            return result_df

        except Exception as e:
            logger.error(f"RSI计算错误: {str(e)}")
            return df

    @staticmethod
    def add_ma(df: pd.DataFrame, price_col: str = "close",
               period_short: int = 5, period_mid: int = 20,
               period_long: int = 60) -> pd.DataFrame:
        """添加移动平均线指标

        Args:
            df: 原始数据
            price_col: 计算列名，默认为close
            period_short: 短期均线周期，默认为5
            period_mid: 中期均线周期，默认为20
            period_long: 长期均线周期，默认为60

        Returns:
            pd.DataFrame: 添加了移动平均线指标的DataFrame
        """
        try:
            # 检查价格列是否存在
            if price_col not in df.columns:
                logger.error(f"MA计算错误: 列 '{price_col}' 不存在于数据中")
                return df

            # 创建副本，避免警告
            df_copy = df.copy()

            # 确保价格数据完整，填充缺失值
            price = df_copy[price_col].ffill().bfill()

            # 创建结果DataFrame
            result_df = df_copy.copy()

            # 短期均线
            result_df[f"MA_{period_short}"] = price.rolling(
                window=period_short).mean()

            # 中期均线
            result_df[f"MA_{period_mid}"] = price.rolling(
                window=period_mid).mean()

            # 长期均线
            result_df[f"MA_{period_long}"] = price.rolling(
                window=period_long).mean()

            # 价格位置相对于均线
            result_df["price_above_short_ma"] = price > result_df[f"MA_{period_short}"]
            result_df["price_above_mid_ma"] = price > result_df[f"MA_{period_mid}"]
            result_df["price_above_long_ma"] = price > result_df[f"MA_{period_long}"]

            # 判断短中均线交叉
            short_ma = result_df[f"MA_{period_short}"]
            mid_ma = result_df[f"MA_{period_mid}"]
            long_ma = result_df[f"MA_{period_long}"]

            result_df["MA_short_cross_mid"] = (
                (short_ma.shift(1) < mid_ma.shift(1)) &
                (short_ma > mid_ma)
            )
            result_df["MA_short_dead_cross_mid"] = (
                (short_ma.shift(1) > mid_ma.shift(1)) &
                (short_ma < mid_ma)
            )

            # 判断中长均线交叉
            result_df["MA_mid_cross_long"] = (
                (mid_ma.shift(1) < long_ma.shift(1)) &
                (mid_ma > long_ma)
            )
            result_df["MA_mid_dead_cross_long"] = (
                (mid_ma.shift(1) > long_ma.shift(1)) &
                (mid_ma < long_ma)
            )

            return result_df

        except Exception as e:
            logger.error(f"MA计算错误: {str(e)}")
            return df

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame, period_short: int = 5,
                              period_long: int = 20) -> pd.DataFrame:
        """添加成交量指标

        Args:
            df: 原始数据
            period_short: 短期均线周期，默认为5
            period_long: 长期均线周期，默认为20

        Returns:
            pd.DataFrame: 包含成交量指标的DataFrame
        """
        try:
            # 检查必要的列是否存在
            if "volume" not in df.columns:
                logger.error(f"成交量计算错误: 列 'volume' 不存在于数据中")
                return df

            # 创建副本，避免警告
            df_copy = df.copy()

            # 确保成交量数据完整，填充缺失值
            volume = df_copy["volume"].ffill().bfill()

            # 创建结果DataFrame
            result_df = df_copy.copy()

            # 短期成交量均线
            result_df["VOLUME_MA_SHORT"] = volume.rolling(
                window=period_short).mean()

            # 长期成交量均线
            result_df["VOLUME_MA_LONG"] = volume.rolling(
                window=period_long).mean()

            # 计算相对成交量（相对于长期均线）
            result_df["VOLUME_RATIO"] = volume / \
                result_df["VOLUME_MA_LONG"].replace(0, np.finfo(float).eps)

            # 判断放量
            # 成交量是长期均线的2倍以上
            result_df["VOLUME_EXPAND"] = result_df["VOLUME_RATIO"] > 2.0
            result_df["VOLUME_MILD_EXPAND"] = (result_df["VOLUME_RATIO"] > 1.5) & (
                result_df["VOLUME_RATIO"] <= 2.0)  # 温和放量

            # 判断缩量
            # 成交量不足长期均线的一半
            result_df["VOLUME_SHRINK"] = result_df["VOLUME_RATIO"] < 0.5

            # 成交量趋势
            result_df["VOLUME_TREND_UP"] = result_df["VOLUME_MA_SHORT"] > result_df["VOLUME_MA_LONG"]
            result_df["VOLUME_TREND_DOWN"] = result_df["VOLUME_MA_SHORT"] < result_df["VOLUME_MA_LONG"]

            # 成交量波动
            result_df["VOLUME_STD"] = volume.rolling(window=period_long).std()
            result_df["VOLUME_STD_RATIO"] = result_df["VOLUME_STD"] / \
                result_df["VOLUME_MA_LONG"].replace(0, np.finfo(float).eps)

            # 异常成交量（超过长期均值+2倍标准差）
            result_df["VOLUME_ANOMALY"] = volume > (
                result_df["VOLUME_MA_LONG"] + 2 * result_df["VOLUME_STD"])

            return result_df

        except Exception as e:
            logger.error(f"成交量计算错误: {str(e)}")
            return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, price_col: str = "close",
                      window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """添加布林带指标

        Args:
            df: 原始数据
            price_col: 价格列名，默认为'close'
            window: 计算周期，默认为20
            std_dev: 标准差倍数，默认为2.0

        Returns:
            pd.DataFrame: 包含布林带指标的DataFrame
        """
        try:
            # 检查价格列是否存在
            if price_col not in df.columns:
                logger.error(f"布林带计算错误: 列 '{price_col}' 不存在于数据中")
                return df

            # 创建副本，避免警告
            df_copy = df.copy()

            # 确保价格数据完整，填充缺失值
            price = df_copy[price_col].ffill().bfill()

            # 计算中轨线（简单移动平均）
            middle_band = price.rolling(window=window).mean()

            # 计算标准差
            std = price.rolling(window=window).std()

            # 计算上轨和下轨
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            # 计算宽度和带宽比率
            bandwidth = (upper_band - lower_band) / middle_band * 100

            # 计算%B指标（价格在带中的位置）
            percent_b = (price - lower_band) / (upper_band -
                                                lower_band).replace(0, np.finfo(float).eps)

            # 创建结果DataFrame
            result_df = df_copy.copy()
            result_df["BB_UPPER"] = upper_band
            result_df["BB_MIDDLE"] = middle_band
            result_df["BB_LOWER"] = lower_band
            result_df["BB_WIDTH"] = bandwidth
            result_df["BB_PERCENT_B"] = percent_b

            # 确定价格和布林带的关系
            result_df["BB_UPPER_TOUCH"] = price >= upper_band
            result_df["BB_LOWER_TOUCH"] = price <= lower_band

            # 检测突破
            # 从上轨向下突破
            result_df["BB_UPPER_BREAKOUT_DOWN"] = (
                (price.shift(1) >= upper_band.shift(1)) &
                (price < upper_band)
            )

            # 从下轨向上突破
            result_df["BB_LOWER_BREAKOUT_UP"] = (
                (price.shift(1) <= lower_band.shift(1)) &
                (price > lower_band)
            )

            # 计算布林带挤压指标（当带宽低于窗口期内的2个标准差时）
            bandwidth_std = bandwidth.rolling(window=50).std()
            bandwidth_mean = bandwidth.rolling(window=50).mean()
            squeeze_threshold = bandwidth_mean - (2 * bandwidth_std)
            result_df["BB_SQUEEZE"] = bandwidth < squeeze_threshold

            # 计算超买和超卖条件
            result_df["BB_OVERBOUGHT"] = percent_b > 1.0  # 价格高于上轨
            result_df["BB_OVERSOLD"] = percent_b < 0.0    # 价格低于下轨

            # 增加波动率突变指标
            std_ratio = std / std.rolling(window=50).mean()
            result_df["BB_VOL_EXPANSION"] = std_ratio > 1.5  # 波动率扩大

            return result_df

        except Exception as e:
            logger.error(f"布林带计算错误: {str(e)}")
            return df

    @staticmethod
    def add_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """添加KDJ指标

        Args:
            df: 原始数据
            n: 周期，默认为9
            m1: K值平滑因子，默认为3
            m2: D值平滑因子，默认为3

        Returns:
            pd.DataFrame: 包含KDJ指标的DataFrame
        """
        try:
            # 检查必要的列是否存在
            required_cols = ["high", "low", "close"]
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"KDJ计算错误: 列 '{col}' 不存在于数据中")
                    return df

            # 创建副本，避免警告
            df_copy = df.copy()

            # 确保数据完整，填充缺失值
            for col in required_cols:
                df_copy[col] = df_copy[col].ffill().bfill()

            # 计算RSV (Raw Stochastic Value)
            low_n = df_copy["low"].rolling(n).min()
            high_n = df_copy["high"].rolling(n).max()

            # 避免除以零
            denominator = (high_n - low_n).replace(0, np.finfo(float).eps)
            rsv = (df_copy["close"] - low_n) / denominator * 100

            # 计算K值 (默认情况下使用9天RSV的3日指数移动平均)
            k = rsv.ewm(alpha=1/m1, adjust=False).mean()

            # 计算D值 (默认情况下使用K值的3日指数移动平均)
            d = k.ewm(alpha=1/m2, adjust=False).mean()

            # 计算J值 (3 * K - 2 * D)
            j = 3 * k - 2 * d

            # 创建结果DataFrame
            result_df = df_copy.copy()
            result_df["KDJ_K"] = k
            result_df["KDJ_D"] = d
            result_df["KDJ_J"] = j

            # KDJ金叉和死叉
            result_df["KDJ_GOLDEN_CROSS"] = (
                k.shift(1) <= d.shift(1)) & (k > d)
            result_df["KDJ_DEATH_CROSS"] = (k.shift(1) >= d.shift(1)) & (k < d)

            # KDJ超买和超卖
            result_df["KDJ_OVERBOUGHT"] = j > 100
            result_df["KDJ_OVERSOLD"] = j < 0

            # 从超买区域向下突破
            result_df["KDJ_OVERBOUGHT_EXIT"] = (j.shift(1) >= 100) & (j < 100)

            # 从超卖区域向上突破
            result_df["KDJ_OVERSOLD_EXIT"] = (j.shift(1) <= 0) & (j > 0)

            return result_df

        except Exception as e:
            logger.error(f"KDJ计算错误: {str(e)}")
            return df


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(".")  # 添加项目根目录到路径，以便导入模块

    from data.fetcher import data_fetcher

    # 获取一支股票的K线数据
    stock_list = data_fetcher.get_stock_list()
    if not stock_list.empty:
        # 获取第一只股票的数据
        symbol = stock_list.iloc[0]["symbol"]
        print(f"获取{symbol}的K线数据并计算技术指标...")

        # 获取K线数据
        df = data_fetcher.get_k_data(symbol, period="daily")

        if not df.empty:
            # 计算指标
            result = BasicIndicator.calculate_all(df)

            # 查看结果
            print(f"原始数据列: {df.columns.tolist()}")
            print(f"计算指标后的列: {result.columns.tolist()}")
            print("\n最新的指标数据:")
            latest = result.tail(1)
            print(
                f"MACD: {latest['MACD'].values[0]:.4f}, 信号线: {latest['MACD_signal'].values[0]:.4f}")
            print(f"MACD差值: {latest['MACD_diff'].values[0]:.4f}")
            print(f"RSI(24): {latest['RSI_24'].values[0]:.2f}")

            if latest["MACD_golden_cross"].values[0]:
                print("MACD金叉信号！")
            if latest["MACD_zero_golden_cross"].values[0]:
                print("MACD零线金叉信号！")
            if latest[f"RSI_24_oversold"].values[0]:
                print("RSI超卖信号！")
