#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块 - 负责从不同数据源获取股票数据
"""

import time
import datetime
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from retrying import retry
from loguru import logger

try:
    import akshare as ak
except ImportError:
    logger.warning("未安装akshare，部分功能可能受限")
    ak = None

try:
    import baostock as bs
except ImportError:
    logger.warning("未安装baostock，部分功能可能受限")
    bs = None

from config.settings import (
    DATA_SOURCE, EXCLUDE_ST, EXCLUDE_NEW, STOCK_MARKETS,
    HISTORY_DAYS, ENABLE_PROXY, PROXY_CONFIG
)


class DataFetcher:
    """数据获取器基类"""

    def __init__(self, data_source: str = DATA_SOURCE):
        """初始化数据获取器

        Args:
            data_source: 数据源名称，支持 akshare, baostock
        """
        self.data_source = data_source
        self.initialized = False
        self._session = None
        self.last_update_time = None

        # 初始化数据源
        self._init_data_source()

    def _init_data_source(self):
        """初始化数据源连接"""
        if self.data_source == "akshare" and ak:
            # akshare无需显式初始化
            self.initialized = True
            logger.info("成功初始化akshare数据源")
        elif self.data_source == "baostock" and bs:
            # 初始化baostock
            self._session = bs.login()
            if self._session.error_code == '0':
                self.initialized = True
                logger.info("成功登录baostock数据源")
            else:
                logger.error(f"登录baostock失败: {self._session.error_msg}")
        else:
            logger.error(f"不支持的数据源: {self.data_source}")

    def __del__(self):
        """析构函数，释放资源"""
        if self.data_source == "baostock" and self._session:
            bs.logout()
            logger.debug("已登出baostock")

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_stock_list(self, markets: List[str] = None) -> pd.DataFrame:
        """获取股票列表

        Args:
            markets: 市场列表，默认为配置中的STOCK_MARKETS

        Returns:
            pd.DataFrame: 股票列表，包含股票代码、名称等信息
        """
        if not self.initialized:
            logger.error("数据源未初始化")
            return pd.DataFrame()

        markets = markets or STOCK_MARKETS

        if self.data_source == "akshare":
            return self._get_stock_list_akshare(markets)
        elif self.data_source == "baostock":
            return self._get_stock_list_baostock(markets)
        else:
            logger.error(f"不支持的数据源: {self.data_source}")
            return pd.DataFrame()

    def _get_stock_list_akshare(self, markets: List[str]) -> pd.DataFrame:
        """从akshare获取股票列表"""
        all_stocks = pd.DataFrame()

        try:
            # 获取A股股票代码和名称
            if "上证" in markets:
                sh_stocks = ak.stock_info_sh_name_code()
                sh_stocks["market"] = "上证"
                all_stocks = pd.concat([all_stocks, sh_stocks])

            if "深证" in markets:
                sz_stocks = ak.stock_info_sz_name_code()
                sz_stocks["market"] = "深证"
                all_stocks = pd.concat([all_stocks, sz_stocks])

            # 重命名列
            all_stocks = all_stocks.rename(columns={
                "code": "symbol",
                "name": "name"
            })

            # 过滤ST股票
            if EXCLUDE_ST:
                all_stocks = all_stocks[~all_stocks["name"].str.contains("ST")]

            # 过滤新股
            if EXCLUDE_NEW:
                today = datetime.datetime.now()
                ipo_dates = {}
                for _, row in all_stocks.iterrows():
                    sym = row["symbol"]
                    # 获取上市日期可能需单独调用，此处简化处理
                    # 实际应用中可能需要调用ak.stock_individual_info_em(symbol=sym)

                # 过滤上市不足100天的股票
                # 实际应用中需要合理实现此过滤逻辑

            logger.info(f"成功获取{len(all_stocks)}只股票信息")
            return all_stocks

        except Exception as e:
            logger.error(f"从akshare获取股票列表失败: {str(e)}")
            return pd.DataFrame()

    def _get_stock_list_baostock(self, markets: List[str]) -> pd.DataFrame:
        """从baostock获取股票列表"""
        try:
            # baostock获取全部股票列表
            stock_rs = bs.query_all_stock(
                day=datetime.datetime.now().strftime("%Y-%m-%d"))
            stock_df = stock_rs.get_data()

            # 过滤市场
            market_map = {
                "上证": "sh",
                "深证": "sz"
            }
            filter_markets = [market_map.get(m, m) for m in markets]
            stock_df = stock_df[stock_df["type"].isin(filter_markets)]

            # 重命名列
            stock_df = stock_df.rename(columns={
                "code": "symbol",
                "code_name": "name"
            })

            # 添加市场信息
            stock_df["market"] = stock_df["symbol"].apply(
                lambda x: "上证" if x.startswith("sh") else "深证"
            )

            # 过滤ST股票
            if EXCLUDE_ST:
                stock_df = stock_df[~stock_df["name"].str.contains("ST")]

            logger.info(f"成功获取{len(stock_df)}只股票信息")
            return stock_df

        except Exception as e:
            logger.error(f"从baostock获取股票列表失败: {str(e)}")
            return pd.DataFrame()

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_k_data(self, symbol: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   period: str = "daily",
                   adjust: str = "qfq") -> pd.DataFrame:
        """获取K线数据

        Args:
            symbol: 股票代码
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            period: K线周期，daily-日线, weekly-周线, monthly-月线
            adjust: 复权方式，qfq-前复权, hfq-后复权, None-不复权

        Returns:
            pd.DataFrame: K线数据
        """
        if not self.initialized:
            logger.error("数据源未初始化")
            return pd.DataFrame()

        # 默认获取历史X天数据
        if not start_date:
            end_date = end_date or datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") -
                          datetime.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

        if self.data_source == "akshare":
            return self._get_k_data_akshare(symbol, start_date, end_date, period, adjust)
        elif self.data_source == "baostock":
            return self._get_k_data_baostock(symbol, start_date, end_date, period, adjust)
        else:
            logger.error(f"不支持的数据源: {self.data_source}")
            return pd.DataFrame()

    def _get_k_data_akshare(self, symbol: str, start_date: str, end_date: str,
                            period: str, adjust: str) -> pd.DataFrame:
        """从akshare获取K线数据"""
        try:
            # 转换代码格式
            if symbol.startswith("6"):
                formatted_symbol = f"sh{symbol}"
            else:
                formatted_symbol = f"sz{symbol}"

            # 日K线
            if period == "daily":
                if adjust == "qfq":
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                            start_date=start_date, end_date=end_date,
                                            adjust="qfq")
                elif adjust == "hfq":
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                            start_date=start_date, end_date=end_date,
                                            adjust="hfq")
                else:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                            start_date=start_date, end_date=end_date,
                                            adjust="")
            # 周K线
            elif period == "weekly":
                df = ak.stock_zh_a_hist(symbol=symbol, period="weekly",
                                        start_date=start_date, end_date=end_date)
            # 月K线
            elif period == "monthly":
                df = ak.stock_zh_a_hist(symbol=symbol, period="monthly",
                                        start_date=start_date, end_date=end_date)
            else:
                logger.error(f"不支持的周期: {period}")
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_chg",
                "涨跌额": "change",
                "换手率": "turnover"
            })

            # 确保日期为datetime类型
            df["date"] = pd.to_datetime(df["date"])
            # 按日期排序
            df = df.sort_values("date")
            # 设置日期为索引
            df.set_index("date", inplace=True)

            return df

        except Exception as e:
            logger.error(f"从akshare获取{symbol}K线数据失败: {str(e)}")
            return pd.DataFrame()

    def _get_k_data_baostock(self, symbol: str, start_date: str, end_date: str,
                             period: str, adjust: str) -> pd.DataFrame:
        """从baostock获取K线数据"""
        try:
            # 转换代码格式
            if symbol.startswith("6"):
                formatted_symbol = f"sh.{symbol}"
            else:
                formatted_symbol = f"sz.{symbol}"

            # 转换周期
            frequency_map = {
                "daily": "d",
                "weekly": "w",
                "monthly": "m"
            }
            bs_frequency = frequency_map.get(period, "d")

            # 转换复权方式
            adjust_map = {
                "qfq": "2",
                "hfq": "1",
                None: "3"
            }
            bs_adjust = adjust_map.get(adjust, "3")

            # 获取K线数据
            rs = bs.query_history_k_data_plus(
                code=formatted_symbol,
                fields="date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency=bs_frequency,
                adjustflag=bs_adjust
            )

            # 获取结果
            df = rs.get_data()

            # 数据类型转换
            numeric_cols = ["open", "high", "low", "close",
                            "volume", "amount", "turn", "pctChg"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 标准化列名
            df = df.rename(columns={
                "turn": "turnover",
                "pctChg": "pct_chg"
            })

            # 确保日期为datetime类型
            df["date"] = pd.to_datetime(df["date"])
            # 按日期排序
            df = df.sort_values("date")
            # 设置日期为索引
            df.set_index("date", inplace=True)

            return df

        except Exception as e:
            logger.error(f"从baostock获取{symbol}K线数据失败: {str(e)}")
            return pd.DataFrame()

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_real_time_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情数据

        Args:
            symbols: 股票代码列表

        Returns:
            pd.DataFrame: 实时行情数据
        """
        if not self.initialized:
            logger.error("数据源未初始化")
            return pd.DataFrame()

        if self.data_source == "akshare":
            return self._get_real_time_quotes_akshare(symbols)
        elif self.data_source == "baostock":
            return self._get_real_time_quotes_baostock(symbols)
        else:
            logger.error(f"不支持的数据源: {self.data_source}")
            return pd.DataFrame()

    def _get_real_time_quotes_akshare(self, symbols: List[str]) -> pd.DataFrame:
        """从akshare获取实时行情数据"""
        try:
            # akshare需要批量获取所有A股实时行情，然后筛选
            df = ak.stock_zh_a_spot_em()

            # 筛选指定的股票
            df = df[df["代码"].isin(symbols)]

            # 标准化列名
            df = df.rename(columns={
                "代码": "symbol",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "change_percent",
                "涨跌额": "change",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "pre_close",
                "换手率": "turnover"
            })

            # 格式化数据类型
            float_cols = ["price", "change_percent", "change", "volume", "amount",
                          "amplitude", "high", "low", "open", "pre_close", "turnover"]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            logger.error(f"从akshare获取实时行情失败: {str(e)}")
            return pd.DataFrame()

    def _get_real_time_quotes_baostock(self, symbols: List[str]) -> pd.DataFrame:
        """从baostock获取实时行情数据"""
        try:
            # 转换代码格式
            formatted_symbols = []
            for symbol in symbols:
                if symbol.startswith("6"):
                    formatted_symbols.append(f"sh.{symbol}")
                else:
                    formatted_symbols.append(f"sz.{symbol}")

            # 获取实时行情
            rs = bs.query_qt_data(formatted_symbols)
            df = rs.get_data()

            # 标准化列名
            df = df.rename(columns={
                "code": "symbol",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "price",
                "volume": "volume",
                "amount": "amount",
                "date": "date",
                "time": "time",
                "preclose": "pre_close"
            })

            # 计算涨跌幅
            df["change"] = df["price"] - df["pre_close"]
            df["change_percent"] = (df["change"] / df["pre_close"]) * 100

            # 格式化数据类型
            float_cols = ["open", "high", "low", "price", "volume",
                          "amount", "pre_close", "change", "change_percent"]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 提取股票代码
            df["symbol"] = df["symbol"].apply(lambda x: x.split(".")[1])

            return df

        except Exception as e:
            logger.error(f"从baostock获取实时行情失败: {str(e)}")
            return pd.DataFrame()

    def get_index_data(self, index_code: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取指数数据

        Args:
            index_code: 指数代码，如sh000001(上证指数)、sz399001(深证成指)
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD

        Returns:
            pd.DataFrame: 指数数据
        """
        if not self.initialized:
            logger.error("数据源未初始化")
            return pd.DataFrame()

        # 默认获取历史X天数据
        if not start_date:
            end_date = end_date or datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") -
                          datetime.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

        if self.data_source == "akshare":
            # 上证指数
            if index_code == "sh000001":
                try:
                    df = ak.stock_zh_index_daily_em(symbol="sh000001")
                    # 截取时间范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    return df
                except Exception as e:
                    logger.error(f"获取上证指数数据失败: {str(e)}")
                    return pd.DataFrame()
            # 深证成指
            elif index_code == "sz399001":
                try:
                    df = ak.stock_zh_index_daily_em(symbol="sz399001")
                    # 截取时间范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    return df
                except Exception as e:
                    logger.error(f"获取深证成指数据失败: {str(e)}")
                    return pd.DataFrame()
            else:
                logger.error(f"不支持的指数代码: {index_code}")
                return pd.DataFrame()
        else:
            logger.error(f"当前数据源{self.data_source}不支持获取指数数据")
            return pd.DataFrame()


# 创建全局实例
data_fetcher = DataFetcher(data_source=DATA_SOURCE)


if __name__ == "__main__":
    # 测试代码
    print("获取股票列表...")
    stock_list = data_fetcher.get_stock_list()
    print(f"共获取到{len(stock_list)}只股票")

    if not stock_list.empty:
        # 打印前5只股票
        print("\n前5只股票:")
        print(stock_list.head())

        # 获取第一只股票的K线数据
        first_stock = stock_list.iloc[0]["symbol"]
        print(f"\n获取{first_stock}的K线数据...")
        k_data = data_fetcher.get_k_data(first_stock, period="daily")
        print(f"共获取到{len(k_data)}条K线数据")

        if not k_data.empty:
            # 打印最近5天数据
            print("\n最近5天数据:")
            print(k_data.tail())

            # 获取实时行情
            print(f"\n获取{first_stock}的实时行情...")
            quotes = data_fetcher.get_real_time_quotes([first_stock])
            print(quotes)

    # 获取指数数据
    print("\n获取上证指数数据...")
    index_data = data_fetcher.get_index_data("sh000001")
    print(f"共获取到{len(index_data)}条上证指数数据")
    if not index_data.empty:
        print("\n最近5天上证指数数据:")
        print(index_data.tail())
