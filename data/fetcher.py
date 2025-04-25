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
import traceback

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
        self.last_request_time = time.time()
        self.request_interval = 0.2  # 请求间隔(秒)

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

        result_df = pd.DataFrame()
        if self.data_source == "akshare":
            result_df = self._get_stock_list_akshare(markets)
        elif self.data_source == "baostock":
            result_df = self._get_stock_list_baostock(markets)
        else:
            logger.error(f"不支持的数据源: {self.data_source}")
            return pd.DataFrame()

        # 进一步清理数据
        if not result_df.empty:
            # 过滤掉NaN值
            result_df = result_df.dropna(subset=["symbol", "name"])
            # 确保symbol是字符串类型
            result_df["symbol"] = result_df["symbol"].astype(str)
            # 移除symbol列中的空字符串
            result_df = result_df[result_df["symbol"].str.strip() != ""]

        return result_df

    def _get_stock_list_akshare(self, markets: List[str]) -> pd.DataFrame:
        """从akshare获取股票列表"""
        all_stocks = pd.DataFrame()

        try:
            # 获取A股股票代码和名称
            if "上证" in markets:
                self._rate_limit()
                sh_stocks = ak.stock_info_sh_name_code()
                # 确保存在所需列并进行标准化
                if "code" not in sh_stocks.columns and "公司代码" in sh_stocks.columns:
                    sh_stocks = sh_stocks.rename(
                        columns={"公司代码": "code", "公司简称": "name"})
                sh_stocks["market"] = "上证"
                all_stocks = pd.concat([all_stocks, sh_stocks])

            if "深证" in markets:
                self._rate_limit()
                sz_stocks = ak.stock_info_sz_name_code()
                # 确保存在所需列并进行标准化
                if "code" not in sz_stocks.columns and "A股代码" in sz_stocks.columns:
                    sz_stocks = sz_stocks.rename(
                        columns={"A股代码": "code", "A股简称": "name"})
                sz_stocks["market"] = "深证"
                all_stocks = pd.concat([all_stocks, sz_stocks])

            # 确保列名统一
            if not all_stocks.empty:
                # 检查必要的列是否存在
                if "code" not in all_stocks.columns or "name" not in all_stocks.columns:
                    logger.error(
                        f"获取的股票数据缺少必要列，实际列: {all_stocks.columns.tolist()}")
                    return pd.DataFrame()

                # 重命名列
                all_stocks = all_stocks.rename(columns={
                    "code": "symbol",
                    "name": "name"
                })

            # 过滤ST股票
            if EXCLUDE_ST and "name" in all_stocks.columns:
                all_stocks = all_stocks[~all_stocks["name"].fillna(
                    "").astype(str).str.contains("ST", na=False)]

            # 过滤新股简化处理
            if EXCLUDE_NEW:
                # 这里简化处理，可以根据实际需求增加逻辑
                pass

            logger.info(f"成功获取{len(all_stocks)}只股票信息")
            return all_stocks

        except Exception as e:
            logger.error(f"从akshare获取股票列表失败: {str(e)}")
            logger.debug(traceback.format_exc())
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
                stock_df = stock_df[~stock_df["name"].fillna(
                    "").str.contains("ST", na=False)]

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
            # 检查股票代码是否有效
            if not isinstance(symbol, str) or pd.isna(symbol) or symbol.strip() == "":
                logger.error(f"无效的股票代码: {symbol}")
                return pd.DataFrame()

            # 转换代码格式
            if symbol.startswith("6"):
                formatted_symbol = f"sh{symbol}"
            else:
                formatted_symbol = f"sz{symbol}"

            # 添加重试逻辑
            max_retries = 3
            for retry_count in range(max_retries):
                try:
                    # 应用频率限制
                    self._rate_limit()

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

                    # 检查API返回的数据是否有效
                    if df is None or df.empty:
                        if retry_count < max_retries - 1:
                            logger.warning(
                                f"获取{symbol}的K线数据为空，尝试第{retry_count+2}次重试...")
                            time.sleep(1)  # 等待1秒后重试
                            continue
                        else:
                            # 尝试从东方财富获取数据
                            logger.warning(
                                f"通过stock_zh_a_hist获取{symbol}数据失败，尝试使用stock_zh_a_daily...")
                            try:
                                df = self._get_k_data_alternative(
                                    symbol, start_date, end_date)
                                if df is None or df.empty:
                                    logger.error(
                                        f"获取{symbol}的K线数据失败，无法通过替代API获取")
                                    return pd.DataFrame()
                            except Exception as e:
                                logger.error(
                                    f"使用替代API获取{symbol}数据失败: {str(e)}")
                                return pd.DataFrame()

                    # 成功获取数据，跳出重试循环
                    break

                except Exception as inner_e:
                    if retry_count < max_retries - 1:
                        logger.warning(
                            f"获取{symbol}的K线数据出错，尝试第{retry_count+2}次重试: {str(inner_e)}")
                        time.sleep(1)  # 等待1秒后重试
                    else:
                        # 最后一次重试也失败，记录错误并返回空DataFrame
                        logger.error(
                            f"获取{symbol}的K线数据失败，重试{max_retries}次后仍未成功: {str(inner_e)}")
                        return pd.DataFrame()

            # 标准化列名
            column_mapping = {
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
                "换手率": "turnover",
                # 兼容英文列名
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "amount": "amount"
            }

            # 应用列映射
            renamed_cols = {}
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    renamed_cols[old_col] = new_col

            if renamed_cols:
                df = df.rename(columns=renamed_cols)

            # 确保必要的列存在
            required_cols = ["date", "open", "close", "high", "low", "volume"]
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(
                    f"获取的K线数据缺少必要列: {missing_cols}, 实际列: {df.columns.tolist()}")
                return pd.DataFrame()

            # 确保日期为datetime类型
            df["date"] = pd.to_datetime(df["date"])
            # 按日期排序
            df = df.sort_values("date")
            # 设置日期为索引
            df.set_index("date", inplace=True)

            return df

        except Exception as e:
            logger.error(f"从akshare获取{symbol}K线数据失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def _get_k_data_alternative(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用东方财富数据源获取K线数据（备用方法）"""
        try:
            # 尝试使用另一个akshare API
            if symbol.startswith("6"):
                exchange = "1"  # 上交所
            else:
                exchange = "0"  # 深交所

            # 东方财富数据接口
            try:
                # 应用频率限制
                self._rate_limit()

                # 使用东方财富数据
                df = ak.stock_zh_a_daily(
                    symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

                # 确保数据不为空
                if df is None or df.empty:
                    return pd.DataFrame()

                # 标准化列名
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })

                # 格式化日期列
                df["date"] = pd.to_datetime(df["date"])

                return df

            except Exception as e:
                logger.warning(f"使用stock_zh_a_daily获取{symbol}数据失败: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取{symbol}的替代K线数据失败: {str(e)}")
            return pd.DataFrame()

    def _get_k_data_baostock(self, symbol: str, start_date: str, end_date: str,
                             period: str, adjust: str) -> pd.DataFrame:
        """从baostock获取K线数据"""
        try:
            # 检查股票代码是否有效
            if not isinstance(symbol, str) or pd.isna(symbol) or symbol.strip() == "":
                logger.error(f"无效的股票代码: {symbol}")
                return pd.DataFrame()

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

        try:
            if self.data_source == "akshare":
                # 上证指数
                if index_code == "sh000001":
                    self._rate_limit()
                    df = ak.stock_zh_index_daily(symbol="sh000001")
                    # 确保日期是字符串格式以正确比较
                    df = df.reset_index()
                    df["date"] = df["date"].astype(str)
                    # 截取时间范围
                    df = df[(df["date"] >= start_date)
                            & (df["date"] <= end_date)]
                    # 设置日期为索引
                    df = df.set_index("date")
                    return df
                # 深证成指
                elif index_code == "sz399001":
                    self._rate_limit()
                    df = ak.stock_zh_index_daily(symbol="sz399001")
                    # 确保日期是字符串格式以正确比较
                    df = df.reset_index()
                    df["date"] = df["date"].astype(str)
                    # 截取时间范围
                    df = df[(df["date"] >= start_date)
                            & (df["date"] <= end_date)]
                    # 设置日期为索引
                    df = df.set_index("date")
                    return df
                else:
                    logger.error(f"不支持的指数代码: {index_code}")
                    return pd.DataFrame()
            else:
                logger.error(f"当前数据源{self.data_source}不支持获取指数数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取指数数据失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def _rate_limit(self):
        """API请求频率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()


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
