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
    def calculate_all(df: pd.DataFrame, indicators: Dict = None) -> pd.DataFrame:
        """计算所有配置的技术指标

        Args:
            df: 包含OHLCV数据的DataFrame
            indicators: 指标配置，默认使用settings中的DEFAULT_INDICATORS

        Returns:
            pd.DataFrame: 添加了技术指标的DataFrame
        """
        if df.empty:
            logger.warning("输入数据为空，无法计算指标")
            return df

        # 检查必要的列是否存在
        required_cols = ["close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"数据缺少必要列: {missing_cols}, 无法计算指标")
            logger.debug(f"可用列: {df.columns.tolist()}")
            return df

        # 复制数据，避免修改原始数据
        result_df = df.copy()

        # 使用默认配置或传入的配置
        indicators = indicators or DEFAULT_INDICATORS

        # 计算各类指标
        try:
            if "MACD" in indicators:
                result_df = BasicIndicator.add_macd(
                    result_df, **indicators["MACD"])

            if "RSI" in indicators:
                result_df = BasicIndicator.add_rsi(
                    result_df, **indicators["RSI"])

            if "MA" in indicators:
                result_df = BasicIndicator.add_ma(
                    result_df, **indicators["MA"])

            if "VOLUME" in indicators:
                # 检查volume列是否存在
                if "volume" in result_df.columns:
                    result_df = BasicIndicator.add_volume_indicators(
                        result_df, **indicators["VOLUME"])
                else:
                    logger.warning("数据中缺少volume列，跳过计算成交量指标")

            if "BOLL" in indicators:
                result_df = BasicIndicator.add_bollinger(
                    result_df, **indicators["BOLL"])

            if "KDJ" in indicators:
                # 检查高低价列是否存在
                if "high" in result_df.columns and "low" in result_df.columns:
                    result_df = BasicIndicator.add_kdj(
                        result_df, **indicators["KDJ"])
                else:
                    logger.warning("数据中缺少high/low列，跳过计算KDJ指标")

            return result_df
        except Exception as e:
            logger.error(f"计算指标时发生错误: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, price_col: str = "close") -> pd.DataFrame:
        """添加MACD指标

        Args:
            df: 数据源DataFrame
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            price_col: 价格列名

        Returns:
            pd.DataFrame: 添加了MACD指标的DataFrame
        """
        try:
            # 检查必要的列是否存在
            if price_col not in df.columns:
                logger.warning(f"数据中缺少{price_col}列，无法计算MACD指标")
                return df

            # 复制原始数据以避免警告
            df_copy = df.copy()

            # 填充价格数据中的缺失值
            price = df_copy[price_col].fillna(
                method='ffill').fillna(method='bfill')

            # 检查是否仍有NaN值
            if price.isna().any():
                logger.warning(f"价格数据中存在无法填充的缺失值，无法计算MACD指标")
                return df

            # 手动计算MACD
            exp1 = price.ewm(span=fast_period, adjust=False).mean()
            exp2 = price.ewm(span=slow_period, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(
                span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            # 创建结果DataFrame
            result_df = df.copy()
            result_df["MACD_LINE"] = macd_line
            result_df["MACD_SIGNAL"] = signal_line
            result_df["MACD_HIST"] = histogram

            # 添加金叉死叉信号
            result_df["MACD_GOLDEN_CROSS"] = (
                (result_df["MACD_LINE"].shift(1).fillna(0) <= result_df["MACD_SIGNAL"].shift(1).fillna(0)) &
                (result_df["MACD_LINE"].fillna(0) >
                 result_df["MACD_SIGNAL"].fillna(0))
            )

            result_df["MACD_DEATH_CROSS"] = (
                (result_df["MACD_LINE"].shift(1).fillna(0) >= result_df["MACD_SIGNAL"].shift(1).fillna(0)) &
                (result_df["MACD_LINE"].fillna(0) <
                 result_df["MACD_SIGNAL"].fillna(0))
            )

            # 添加零轴上下信号
            result_df["MACD_ABOVE_ZERO"] = result_df["MACD_LINE"].fillna(0) > 0
            result_df["MACD_BELOW_ZERO"] = result_df["MACD_LINE"].fillna(0) < 0

            # 添加柱状图方向变化信号
            result_df["MACD_HIST_TURN_POSITIVE"] = (
                (result_df["MACD_HIST"].shift(1).fillna(0) <= 0) &
                (result_df["MACD_HIST"].fillna(0) > 0)
            )

            result_df["MACD_HIST_TURN_NEGATIVE"] = (
                (result_df["MACD_HIST"].shift(1).fillna(0) >= 0) &
                (result_df["MACD_HIST"].fillna(0) < 0)
            )

            return result_df

        except Exception as e:
            import traceback
            logger.error(f"计算MACD指标出错: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close",
                overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
        """添加RSI指标

        Args:
            df: 数据源DataFrame
            period: 周期
            price_col: 价格列名
            overbought: 超买阈值
            oversold: 超卖阈值

        Returns:
            pd.DataFrame: 添加了RSI指标的DataFrame
        """
        try:
            # 检查必要的列是否存在
            if price_col not in df.columns:
                logger.warning(f"数据中缺少{price_col}列，无法计算RSI指标")
                return df

            # 复制原始数据以避免警告
            df_copy = df.copy()

            # 填充价格数据中的缺失值
            price = df_copy[price_col].fillna(
                method='ffill').fillna(method='bfill')

            # 检查是否仍有NaN值
            if price.isna().any():
                logger.warning(f"价格数据中存在无法填充的缺失值，无法计算RSI指标")
                return df

            # 手动计算RSI
            delta = price.diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # 防止除以零
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            rsi = 100 - (100 / (1 + rs))

            # 创建结果DataFrame
            result_df = df.copy()
            result_df["RSI"] = rsi

            # 添加超买超卖信号
            result_df["RSI_OVERBOUGHT"] = result_df["RSI"].fillna(
                0) > overbought
            result_df["RSI_OVERSOLD"] = result_df["RSI"].fillna(0) < oversold

            # 添加交叉中线信号
            result_df["RSI_ABOVE_50"] = result_df["RSI"].fillna(0) > 50
            result_df["RSI_BELOW_50"] = result_df["RSI"].fillna(0) < 50

            # 添加RSI背离信号（价格新高但RSI未创新高，或价格新低但RSI未创新低）
            price_high = price.rolling(window=period).max() == price
            price_low = price.rolling(window=period).min() == price
            rsi_high = rsi.rolling(window=period).max() == rsi
            rsi_low = rsi.rolling(window=period).min() == rsi

            result_df["RSI_BEARISH_DIVERGENCE"] = price_high & ~rsi_high
            result_df["RSI_BULLISH_DIVERGENCE"] = price_low & ~rsi_low

            return result_df

        except Exception as e:
            import traceback
            logger.error(f"计算RSI指标出错: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return df

    @staticmethod
    def add_ma(df: pd.DataFrame, short: int = 5, mid: int = 20,
               long: int = 60, column: str = "close") -> pd.DataFrame:
        """添加移动平均线指标

        Args:
            df: 数据源DataFrame
            short: 短期均线周期
            mid: 中期均线周期
            long: 长期均线周期
            column: 计算列名，默认为close

        Returns:
            pd.DataFrame: 添加了移动平均线指标的DataFrame
        """
        try:
            # 计算移动平均线
            result_df = df.copy()

            # 短期均线
            result_df[f"MA_{short}"] = ta.sma(df[column], length=short)

            # 中期均线
            result_df[f"MA_{mid}"] = ta.sma(df[column], length=mid)

            # 长期均线
            result_df[f"MA_{long}"] = ta.sma(df[column], length=long)

            # 判断短中均线交叉
            result_df["MA_short_cross_mid"] = (
                (result_df[f"MA_{short}"].shift(1).fillna(0) < result_df[f"MA_{mid}"].shift(1).fillna(0)) &
                (result_df[f"MA_{short}"].fillna(0)
                 > result_df[f"MA_{mid}"].fillna(0))
            )
            result_df["MA_short_dead_cross_mid"] = (
                (result_df[f"MA_{short}"].shift(1).fillna(0) > result_df[f"MA_{mid}"].shift(1).fillna(0)) &
                (result_df[f"MA_{short}"].fillna(0)
                 < result_df[f"MA_{mid}"].fillna(0))
            )

            # 判断中长均线交叉
            result_df["MA_mid_cross_long"] = (
                (result_df[f"MA_{mid}"].shift(1).fillna(0) < result_df[f"MA_{long}"].shift(1).fillna(0)) &
                (result_df[f"MA_{mid}"].fillna(0) >
                 result_df[f"MA_{long}"].fillna(0))
            )
            result_df["MA_mid_dead_cross_long"] = (
                (result_df[f"MA_{mid}"].shift(1).fillna(0) > result_df[f"MA_{long}"].shift(1).fillna(0)) &
                (result_df[f"MA_{mid}"].fillna(0) <
                 result_df[f"MA_{long}"].fillna(0))
            )

            # 判断价格与均线关系
            result_df["price_above_short_ma"] = result_df[column].fillna(
                0) > result_df[f"MA_{short}"].fillna(0)
            result_df["price_above_mid_ma"] = result_df[column].fillna(
                0) > result_df[f"MA_{mid}"].fillna(0)
            result_df["price_above_long_ma"] = result_df[column].fillna(
                0) > result_df[f"MA_{long}"].fillna(0)

            return result_df

        except Exception as e:
            logger.error(f"计算移动平均线指标出错: {str(e)}")
            return df

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame, short: int = 5,
                              long: int = 20) -> pd.DataFrame:
        """添加成交量指标

        Args:
            df: 数据源DataFrame
            short: 短期成交量均线周期
            long: 长期成交量均线周期

        Returns:
            pd.DataFrame: 添加了成交量指标的DataFrame
        """
        try:
            if "volume" not in df.columns:
                logger.warning("数据中缺少成交量数据，无法计算成交量指标")
                return df

            result_df = df.copy()

            # 计算成交量均线
            result_df[f"VOLUME_MA_{short}"] = ta.sma(
                df["volume"], length=short)
            result_df[f"VOLUME_MA_{long}"] = ta.sma(df["volume"], length=long)

            # 计算相对成交量
            result_df["VOLUME_RATIO_S"] = df["volume"].fillna(0) / \
                result_df[f"VOLUME_MA_{short}"].fillna(1)  # 防止除以0
            result_df["VOLUME_RATIO_L"] = df["volume"].fillna(0) / \
                result_df[f"VOLUME_MA_{long}"].fillna(1)  # 防止除以0

            # 判断放量
            result_df["VOLUME_EXPAND"] = result_df["VOLUME_RATIO_L"].fillna(
                0) > 1.5

            # 判断温和放量（成交量适度放大）
            result_df["VOLUME_MILD_EXPAND"] = (
                (result_df["VOLUME_RATIO_L"].fillna(0) > 1.2) &
                (result_df["VOLUME_RATIO_L"].fillna(0) < 2.0)
            )

            return result_df

        except Exception as e:
            logger.error(f"计算成交量指标出错: {str(e)}")
            return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, window: int = 20, std_dev: int = 2,
                      price_col: str = "close") -> pd.DataFrame:
        """添加布林带指标

        Args:
            df: 数据源DataFrame
            window: 移动窗口大小
            std_dev: 标准差系数
            price_col: 价格列名

        Returns:
            pd.DataFrame: 添加了布林带指标的DataFrame
        """
        try:
            if price_col not in df.columns:
                logger.warning(f"数据中缺少{price_col}列，无法计算布林带指标")
                return df

            # 创建结果DataFrame
            result_df = df.copy()

            # 处理价格列中的缺失值
            price_data = result_df[price_col].fillna(
                method='ffill').fillna(method='bfill')

            # 检查是否仍有NaN值
            if price_data.isna().any():
                logger.warning(f"价格数据中存在无法填充的缺失值，无法计算布林带指标")
                return df

            # 计算移动平均线
            middle_band = price_data.rolling(window=window).mean()

            # 计算标准差
            sigma = price_data.rolling(window=window).std()

            # 计算上下轨
            upper_band = middle_band + std_dev * sigma
            lower_band = middle_band - std_dev * sigma

            # 添加布林带指标
            result_df["BB_MIDDLE"] = middle_band
            result_df["BB_UPPER"] = upper_band
            result_df["BB_LOWER"] = lower_band

            # 计算价格与布林带的关系
            result_df["BB_UPPER_TOUCH"] = (
                (result_df[price_col].shift(1).fillna(0) < result_df["BB_UPPER"].shift(1).fillna(0)) &
                (result_df[price_col].fillna(0) >=
                 result_df["BB_UPPER"].fillna(0))
            )

            result_df["BB_LOWER_TOUCH"] = (
                (result_df[price_col].shift(1).fillna(0) > result_df["BB_LOWER"].shift(1).fillna(0)) &
                (result_df[price_col].fillna(0) <=
                 result_df["BB_LOWER"].fillna(0))
            )

            result_df["BB_SQUEEZE"] = (
                (result_df["BB_UPPER"].fillna(0) - result_df["BB_LOWER"].fillna(0)) /
                result_df["BB_MIDDLE"].replace(0, np.finfo(
                    float).eps).fillna(np.finfo(float).eps)
            )

            # 添加超买超卖判断
            result_df["BB_OVERBOUGHT"] = result_df[price_col].fillna(
                0) > result_df["BB_UPPER"].fillna(0)
            result_df["BB_OVERSOLD"] = result_df[price_col].fillna(
                0) < result_df["BB_LOWER"].fillna(0)

            # 计算%B指标（价格在布林带中的相对位置）
            bb_range = result_df["BB_UPPER"] - result_df["BB_LOWER"]
            bb_range = bb_range.replace(0, np.finfo(float).eps)  # 避免除零错误
            result_df["BB_PERCENT_B"] = (
                result_df[price_col] - result_df["BB_LOWER"]) / bb_range

            return result_df

        except Exception as e:
            import traceback
            logger.error(f"计算布林带指标出错: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return df

    @staticmethod
    def add_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """添加KDJ指标

        Args:
            df: 数据源DataFrame
            n: 计算周期
            m1: K值平滑系数
            m2: D值平滑系数

        Returns:
            pd.DataFrame: 添加了KDJ指标的DataFrame
        """
        try:
            # 检查必要的列是否存在
            required_columns = ["high", "low", "close"]
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"数据中缺少{', '.join(missing_columns)}列，无法计算KDJ指标")
                return df

            # 创建结果DataFrame和复制数据
            result_df = df.copy()
            df_copy = df.copy()

            # 填充缺失值
            for col in required_columns:
                df_copy[col] = df_copy[col].fillna(
                    method='ffill').fillna(method='bfill')

            # 检查是否仍有NaN值
            if df_copy[required_columns].isna().any().any():
                logger.warning(f"价格数据中存在无法填充的缺失值，无法计算KDJ指标")
                return df

            # 手动计算KDJ
            low_min = df_copy['low'].rolling(window=n).min()
            high_max = df_copy['high'].rolling(window=n).max()

            # 计算RSV，避免除零错误
            rsv_divisor = high_max - low_min
            rsv_divisor = rsv_divisor.replace(0, np.finfo(float).eps)
            rsv = 100 * ((df_copy['close'] - low_min) / rsv_divisor)

            # 使用EMA计算KDJ的三个值
            k = rsv.ewm(alpha=1/m1, adjust=False).mean()
            d = k.ewm(alpha=1/m2, adjust=False).mean()
            j = 3 * k - 2 * d

            # 添加KDJ指标到结果DataFrame
            result_df["KDJ_K"] = k
            result_df["KDJ_D"] = d
            result_df["KDJ_J"] = j

            # 添加金叉死叉判断
            result_df["KDJ_GOLDEN_CROSS"] = (
                (result_df["KDJ_K"].shift(1).fillna(0) <= result_df["KDJ_D"].shift(1).fillna(0)) &
                (result_df["KDJ_K"].fillna(0) > result_df["KDJ_D"].fillna(0))
            )

            result_df["KDJ_DEATH_CROSS"] = (
                (result_df["KDJ_K"].shift(1).fillna(0) >= result_df["KDJ_D"].shift(1).fillna(0)) &
                (result_df["KDJ_K"].fillna(0) < result_df["KDJ_D"].fillna(0))
            )

            # 添加超买超卖判断
            result_df["KDJ_OVERBOUGHT"] = result_df["KDJ_J"].fillna(0) > 100
            result_df["KDJ_OVERSOLD"] = result_df["KDJ_J"].fillna(0) < 0

            return result_df

        except Exception as e:
            import traceback
            logger.error(f"计算KDJ指标出错: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
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
