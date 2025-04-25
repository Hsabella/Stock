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
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                 signal: int = 9, column: str = "close") -> pd.DataFrame:
        """添加MACD指标

        Args:
            df: 数据源DataFrame
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            column: 计算列名，默认为close

        Returns:
            pd.DataFrame: 添加了MACD指标的DataFrame
        """
        try:
            # 使用pandas_ta计算MACD
            macd = ta.macd(df[column], fast=fast, slow=slow, signal=signal)

            # 重命名列，使命名更明确
            col_names = {
                f"MACD_{fast}_{slow}_{signal}": f"MACD",
                f"MACDh_{fast}_{slow}_{signal}": f"MACD_histogram",
                f"MACDs_{fast}_{slow}_{signal}": f"MACD_signal"
            }
            macd = macd.rename(columns=col_names)

            # 合并到原始数据
            result_df = pd.concat([df, macd], axis=1)

            # 计算MACD变化率（用于判断金叉银叉）
            result_df["MACD_diff"] = result_df["MACD"] - \
                result_df["MACD_signal"]

            # 判断金叉和死叉
            result_df["MACD_golden_cross"] = (
                result_df["MACD_diff"].shift(1) < 0) & (result_df["MACD_diff"] > 0)
            result_df["MACD_dead_cross"] = (
                result_df["MACD_diff"].shift(1) > 0) & (result_df["MACD_diff"] < 0)

            # 判断零线金叉和零线死叉
            result_df["MACD_zero_golden_cross"] = (
                result_df["MACD"].shift(1) < 0) & (result_df["MACD"] > 0)
            result_df["MACD_zero_dead_cross"] = (
                result_df["MACD"].shift(1) > 0) & (result_df["MACD"] < 0)

            return result_df

        except Exception as e:
            logger.error(f"计算MACD指标出错: {str(e)}")
            return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, length: int = 14, overbought: int = 70,
                oversold: int = 30, column: str = "close") -> pd.DataFrame:
        """添加RSI指标

        Args:
            df: 数据源DataFrame
            length: RSI周期
            overbought: 超买阈值
            oversold: 超卖阈值
            column: 计算列名，默认为close

        Returns:
            pd.DataFrame: 添加了RSI指标的DataFrame
        """
        try:
            # 计算RSI
            rsi = ta.rsi(df[column], length=length)

            # 重命名列
            rsi = rsi.rename(f"RSI_{length}")

            # 合并到原始数据
            result_df = pd.concat([df, rsi], axis=1)

            # 判断超买超卖
            result_df[f"RSI_{length}_overbought"] = result_df[f"RSI_{length}"] > overbought
            result_df[f"RSI_{length}_oversold"] = result_df[f"RSI_{length}"] < oversold

            # 判断拐点
            result_df[f"RSI_{length}_rising"] = result_df[f"RSI_{length}"] > result_df[f"RSI_{length}"].shift(
                1)
            result_df[f"RSI_{length}_falling"] = result_df[f"RSI_{length}"] < result_df[f"RSI_{length}"].shift(
                1)

            return result_df

        except Exception as e:
            logger.error(f"计算RSI指标出错: {str(e)}")
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
            result_df["price_above_short_ma"] = result_df[
                column] > result_df[f"MA_{short}"].fillna(0)
            result_df["price_above_mid_ma"] = result_df[
                column] > result_df[f"MA_{mid}"].fillna(0)
            result_df["price_above_long_ma"] = result_df[
                column] > result_df[f"MA_{long}"].fillna(0)

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
            result_df["VOLUME_RATIO_S"] = df["volume"] / \
                result_df[f"VOLUME_MA_{short}"]
            result_df["VOLUME_RATIO_L"] = df["volume"] / \
                result_df[f"VOLUME_MA_{long}"]

            # 判断放量
            result_df["VOLUME_EXPAND"] = result_df["VOLUME_RATIO_L"] > 1.5

            # 判断温和放量（成交量适度放大）
            result_df["VOLUME_MILD_EXPAND"] = (
                (result_df["VOLUME_RATIO_L"] > 1.2) &
                (result_df["VOLUME_RATIO_L"] < 2.0)
            )

            # 计算成交量变化率
            result_df["VOLUME_CHANGE"] = df["volume"] / \
                df["volume"].shift(1) - 1

            # OBV指标(On-Balance Volume)
            obv = ta.obv(df["close"], df["volume"])
            result_df["OBV"] = obv

            return result_df

        except Exception as e:
            logger.error(f"计算成交量指标出错: {str(e)}")
            return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, length: int = 20, std: float = 2.0,
                      column: str = "close") -> pd.DataFrame:
        """添加布林带指标

        Args:
            df: 数据源DataFrame
            length: 布林带周期
            std: 标准差倍数
            column: 计算列名，默认为close

        Returns:
            pd.DataFrame: 添加了布林带指标的DataFrame
        """
        try:
            # 计算布林带
            boll = ta.bbands(df[column], length=length, std=std)

            # 重命名列
            col_names = {
                f"BBL_{length}_{float(std):.1f}": "BOLL_LOWER",
                f"BBM_{length}_{float(std):.1f}": "BOLL_MIDDLE",
                f"BBU_{length}_{float(std):.1f}": "BOLL_UPPER",
                f"BBB_{length}_{float(std):.1f}": "BOLL_BANDWIDTH",
                f"BBP_{length}_{float(std):.1f}": "BOLL_PERCENT"
            }
            boll = boll.rename(columns=col_names)

            # 合并到原始数据
            result_df = pd.concat([df, boll], axis=1)

            # 判断价格与布林带关系
            result_df["PRICE_ABOVE_BOLL_UPPER"] = df[column] > result_df["BOLL_UPPER"]
            result_df["PRICE_BELOW_BOLL_LOWER"] = df[column] < result_df["BOLL_LOWER"]
            result_df["PRICE_IN_BOLL"] = (
                (df[column] <= result_df["BOLL_UPPER"]) &
                (df[column] >= result_df["BOLL_LOWER"])
            )

            # 判断布林带收缩/扩张
            result_df["BOLL_SQUEEZE"] = result_df["BOLL_BANDWIDTH"] < result_df["BOLL_BANDWIDTH"].shift(
                1)

            return result_df

        except Exception as e:
            logger.error(f"计算布林带指标出错: {str(e)}")
            return df

    @staticmethod
    def add_kdj(df: pd.DataFrame, k: int = 9, d: int = 3, j: int = 3) -> pd.DataFrame:
        """添加KDJ指标

        Args:
            df: 数据源DataFrame
            k: K值周期
            d: D值周期
            j: J值周期

        Returns:
            pd.DataFrame: 添加了KDJ指标的DataFrame
        """
        try:
            if not all(col in df.columns for col in ["high", "low", "close"]):
                logger.warning("数据中缺少high/low/close数据，无法计算KDJ指标")
                return df

            # 计算KDJ
            kdj = ta.kdj(df["high"], df["low"], df["close"], k, d, j)

            # 合并到原始数据
            result_df = pd.concat([df, kdj], axis=1)

            # 计算KDJ金叉和死叉
            result_df["KDJ_J_CROSS_D"] = (
                (result_df["J"].shift(1) < result_df["D"].shift(1)) &
                (result_df["J"] > result_df["D"])
            )
            result_df["KDJ_J_DEAD_CROSS_D"] = (
                (result_df["J"].shift(1) > result_df["D"].shift(1)) &
                (result_df["J"] < result_df["D"])
            )

            # 判断超买超卖
            result_df["KDJ_OVERBOUGHT"] = result_df["J"] > 80
            result_df["KDJ_OVERSOLD"] = result_df["J"] < 20

            return result_df

        except Exception as e:
            logger.error(f"计算KDJ指标出错: {str(e)}")
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
