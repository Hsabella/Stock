#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块 - 负责从不同数据源获取股票数据
"""

import time
import datetime
import pandas as pd
import os
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
                all_stocks = all_stocks[~all_stocks["name"].astype(
                    str).str.contains("ST", na=False)]

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

        # 首先尝试从缓存获取数据
        cached_data = self._get_data_from_cache(
            symbol, start_date, end_date, period, adjust)
        if not cached_data.empty:
            return cached_data

        # 如果缓存不存在或过期，则从数据源获取
        result_df = pd.DataFrame()

        # 尝试主数据源
        if self.data_source == "akshare":
            result_df = self._get_k_data_akshare(
                symbol, start_date, end_date, period, adjust)
        elif self.data_source == "baostock":
            result_df = self._get_k_data_baostock(
                symbol, start_date, end_date, period, adjust)
        else:
            logger.error(f"不支持的数据源: {self.data_source}")
            return pd.DataFrame()

        # 如果主数据源获取失败，尝试备用数据源
        if result_df.empty and self.data_source == "akshare":
            logger.warning(f"从akshare获取{symbol}数据失败，尝试使用备用方法")
            result_df = self._get_k_data_alternative(
                symbol, start_date, end_date)

        # 如果akshare的备用方法获取失败，尝试baostock
        if result_df.empty and self.data_source == "akshare" and bs:
            logger.warning(f"从akshare备用方法获取{symbol}数据失败，尝试使用baostock")
            result_df = self._get_k_data_baostock(
                symbol, start_date, end_date, period, adjust)

        # 如果baostock获取失败，尝试akshare
        if result_df.empty and self.data_source == "baostock" and ak:
            logger.warning(f"从baostock获取{symbol}数据失败，尝试使用akshare")
            result_df = self._get_k_data_akshare(
                symbol, start_date, end_date, period, adjust)

            # 如果还是失败，尝试akshare的备用方法
            if result_df.empty:
                logger.warning(f"从akshare获取{symbol}数据失败，尝试使用akshare备用方法")
                result_df = self._get_k_data_alternative(
                    symbol, start_date, end_date)

        # 如果获取到数据，保存到缓存
        if not result_df.empty:
            self._save_data_to_cache(result_df, symbol, period, adjust)

        return result_df

    def _get_data_from_cache(self, symbol: str, start_date: str, end_date: str,
                             period: str, adjust: str) -> pd.DataFrame:
        """从本地缓存获取数据"""
        try:
            # 创建缓存目录
            cache_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            cache_file = os.path.join(
                cache_dir, f"{symbol}_{period}_{adjust}.csv")

            # 如果缓存文件存在
            if os.path.exists(cache_file):
                # 检查文件修改时间，如果是今天，直接使用缓存
                file_mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(cache_file))
                today = datetime.datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0)

                # 读取缓存数据
                cached_data = pd.read_csv(cache_file)

                # 如果缓存是今天的数据且不是交易时间，直接返回
                if file_mtime >= today and datetime.datetime.now().hour < 15:
                    if "date" in cached_data.columns:
                        cached_data["date"] = pd.to_datetime(
                            cached_data["date"])
                        cached_data = cached_data.set_index("date")
                    logger.info(f"使用缓存数据: {symbol}")
                    return cached_data

                # 否则检查是否需要更新
                if "date" in cached_data.columns:
                    cached_data["date"] = pd.to_datetime(cached_data["date"])
                    cached_data = cached_data.set_index("date")

                    # 检查缓存数据的日期范围
                    cached_start = cached_data.index.min().strftime("%Y-%m-%d")
                    cached_end = cached_data.index.max().strftime("%Y-%m-%d")

                    # 如果缓存的数据已经覆盖了请求的日期范围，直接返回
                    if cached_start <= start_date and cached_end >= end_date:
                        logger.info(f"使用缓存数据（日期范围已覆盖）: {symbol}")
                        # 过滤出请求的日期范围
                        mask = (cached_data.index >= pd.Timestamp(start_date)) & (
                            cached_data.index <= pd.Timestamp(end_date))
                        return cached_data.loc[mask]

                    # 如果有部分数据，只获取缺失的部分
                    if cached_end >= start_date:
                        # 更新起始日期为缓存结束日期后一天
                        new_start_date = (pd.Timestamp(
                            cached_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                        # 获取新数据
                        if new_start_date <= end_date:
                            logger.info(
                                f"更新缓存数据: {symbol}, 从 {new_start_date} 到 {end_date}")
                            if self.data_source == "akshare":
                                new_data = self._get_k_data_akshare(
                                    symbol, new_start_date, end_date, period, adjust)
                            else:
                                new_data = self._get_k_data_baostock(
                                    symbol, new_start_date, end_date, period, adjust)

                            if not new_data.empty:
                                # 合并新旧数据
                                combined_data = pd.concat(
                                    [cached_data, new_data])
                                # 去重
                                combined_data = combined_data[~combined_data.index.duplicated(
                                    keep='last')]
                                # 排序
                                combined_data = combined_data.sort_index()
                                # 保存到缓存
                                combined_data.to_csv(cache_file)

                                # 过滤出请求的日期范围
                                mask = (combined_data.index >= pd.Timestamp(start_date)) & (
                                    combined_data.index <= pd.Timestamp(end_date))
                                return combined_data.loc[mask]

                        # 如果不需要更新，使用过滤后的缓存数据
                        mask = (cached_data.index >= pd.Timestamp(start_date)) & (
                            cached_data.index <= pd.Timestamp(end_date))
                        filtered_data = cached_data.loc[mask]
                        if not filtered_data.empty:
                            return filtered_data

            # 缓存不存在或者无法使用
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"从缓存获取数据出错: {str(e)}")
            return pd.DataFrame()

    def _save_data_to_cache(self, data: pd.DataFrame, symbol: str, period: str, adjust: str):
        """保存数据到本地缓存"""
        try:
            # 创建缓存目录
            cache_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            cache_file = os.path.join(
                cache_dir, f"{symbol}_{period}_{adjust}.csv")

            # 保存数据
            data.to_csv(cache_file)
            logger.debug(f"数据已保存到缓存: {cache_file}")

        except Exception as e:
            logger.error(f"保存数据到缓存出错: {str(e)}")

    def _get_k_data_akshare(self, symbol: str, start_date: str, end_date: str,
                            period: str, adjust: str) -> pd.DataFrame:
        """从akshare获取K线数据"""
        try:
            # 检查股票代码是否有效
            if not isinstance(symbol, str) or pd.isna(symbol) or symbol.strip() == "":
                logger.error(f"无效的股票代码: {symbol}")
                return pd.DataFrame()

            # 格式化股票代码 - akshare可能需要不同格式
            original_symbol = symbol
            # 移除可能的前缀
            if symbol.startswith(('sh', 'sz')):
                symbol = symbol[2:]

            # 确保日期格式正确
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                # 部分akshare接口需要特定格式的日期字符串
                start_date_fmt = start_dt.strftime("%Y%m%d")
                end_date_fmt = end_dt.strftime("%Y%m%d")
                # 另一种格式
                start_date_dash = start_dt.strftime("%Y-%m-%d")
                end_date_dash = end_dt.strftime("%Y-%m-%d")
            except Exception as e:
                logger.error(f"日期格式转换错误: {str(e)}")
                return pd.DataFrame()

            # 添加重试逻辑
            max_retries = 3
            for retry_count in range(max_retries):
                try:
                    # 应用频率限制
                    self._rate_limit()

                    # 记录详细的调试信息
                    logger.debug(
                        f"尝试获取股票数据: 代码={symbol}, 开始日期={start_date_dash}, 结束日期={end_date_dash}")

                    # 尝试不同格式的日期参数
                    try:
                        # 先检查akshare的版本，获取支持的参数
                        try:
                            logger.debug(f"当前akshare版本: {ak.__version__}")
                        except:
                            logger.debug("无法获取akshare版本")

                        # 尝试最新的几个稳定接口
                        df = None

                        # 尝试1: 优先使用新版东方财富接口 stock_zh_a_hist
                        if df is None or df.empty:
                            try:
                                logger.debug(
                                    f"尝试使用stock_zh_a_hist方法: symbol={symbol}")
                                df = ak.stock_zh_a_hist(
                                    symbol=symbol,
                                    period="daily" if period == "daily" else period,
                                    start_date=start_date_dash,
                                    end_date=end_date_dash,
                                    adjust=adjust if adjust else ""
                                )
                                if df is not None and not df.empty:
                                    logger.debug(
                                        f"stock_zh_a_hist成功: {len(df)}行")
                            except Exception as e1:
                                logger.debug(f"stock_zh_a_hist失败: {str(e1)}")

                        # 尝试2: 使用腾讯的历史数据接口
                        if df is None or df.empty:
                            try:
                                logger.debug(
                                    f"尝试使用stock_zh_a_hist_tx方法: symbol={symbol}")
                                # 腾讯接口需要带市场前缀
                                tx_symbol = symbol
                                if symbol.startswith('6'):
                                    tx_symbol = f"sh{symbol}"
                                elif symbol.startswith('0') or symbol.startswith('3'):
                                    tx_symbol = f"sz{symbol}"

                                df = ak.stock_zh_a_hist_tx(
                                    symbol=tx_symbol,
                                    start_date=start_date_dash,
                                    end_date=end_date_dash
                                )
                                if df is not None and not df.empty:
                                    logger.debug(
                                        f"stock_zh_a_hist_tx成功: {len(df)}行")
                            except Exception as e2:
                                logger.debug(
                                    f"stock_zh_a_hist_tx失败: {str(e2)}")

                        # 尝试3: 使用东方财富分钟线接口，然后聚合
                        if df is None or df.empty and period == "daily":
                            try:
                                logger.debug(
                                    f"尝试使用stock_zh_a_hist_min_em方法: symbol={symbol}")
                                # 获取日内分钟线数据，然后聚合为日线
                                min_df = ak.stock_zh_a_hist_min_em(
                                    symbol=symbol,
                                    start_date=start_date_dash.replace(
                                        '-', ''),
                                    end_date=end_date_dash.replace('-', ''),
                                    period='60'  # 60分钟线，减少数据量
                                )
                                if min_df is not None and not min_df.empty:
                                    logger.debug(
                                        f"stock_zh_a_hist_min_em成功: {len(min_df)}行")
                                    # 聚合为日线
                                    min_df['日期'] = pd.to_datetime(
                                        min_df['时间']).dt.date
                                    df = min_df.groupby('日期').agg({
                                        '开盘': 'first',
                                        '收盘': 'last',
                                        '最高': 'max',
                                        '最低': 'min',
                                        '成交量': 'sum',
                                        '成交额': 'sum'
                                    }).reset_index()
                                    # 格式化日期
                                    df['日期'] = pd.to_datetime(df['日期'])
                            except Exception as e3:
                                logger.debug(
                                    f"stock_zh_a_hist_min_em失败: {str(e3)}")

                        # 尝试4: 使用股票最新日线行情来检查股票代码是否正确
                        if df is None or df.empty:
                            try:
                                logger.debug(
                                    f"尝试获取最新行情检查股票代码: symbol={symbol}")
                                spot_df = ak.stock_zh_a_spot_em()
                                if spot_df is not None and not spot_df.empty:
                                    # 提取代码列
                                    symbols = spot_df['代码'].tolist()
                                    if symbol in symbols:
                                        logger.debug(f"股票代码 {symbol} 存在于最新行情中")
                                    else:
                                        similar = [
                                            s for s in symbols if s.endswith(symbol[-4:])]
                                        if similar:
                                            logger.debug(
                                                f"找到可能的相似股票代码: {similar[:5]}")
                            except Exception as e4:
                                logger.debug(f"获取最新行情失败: {str(e4)}")

                        # 尝试5: 备用老方法
                        if df is None or df.empty:
                            try:
                                logger.debug(
                                    f"尝试使用stock_zh_a_daily方法: symbol={symbol}")
                                df = ak.stock_zh_a_daily(
                                    symbol=symbol,
                                    start_date=start_date_dash,
                                    end_date=end_date_dash,
                                    adjust=adjust
                                )
                                if df is not None and not df.empty:
                                    logger.debug(
                                        f"stock_zh_a_daily成功: {len(df)}行")
                            except Exception as e5:
                                logger.debug(f"stock_zh_a_daily失败: {str(e5)}")

                        # 检查API返回的数据是否有效
                        if df is None or df.empty:
                            if retry_count < max_retries - 1:
                                logger.warning(
                                    f"获取{symbol}的K线数据为空，尝试第{retry_count+2}次重试...")
                                time.sleep(2)  # 增加等待时间
                                continue
                            else:
                                # 最后一次重试失败，记录错误
                                logger.error(f"获取{symbol}的K线数据失败，数据为空")
                                return pd.DataFrame()

                        # 成功获取数据，跳出重试循环
                        logger.debug(f"成功获取{symbol}数据，共{len(df)}行")
                        break
                    except Exception as api_e:
                        logger.error(f"所有API调用方法都失败: {str(api_e)}")
                        df = pd.DataFrame()
                        if retry_count < max_retries - 1:
                            continue
                        else:
                            return pd.DataFrame()

                except Exception as inner_e:
                    if retry_count < max_retries - 1:
                        logger.warning(
                            f"获取{symbol}的K线数据出错，尝试第{retry_count+2}次重试: {str(inner_e)}")
                        time.sleep(2)  # 增加等待时间
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
                "amount": "amount",
                # 股票日线数据
                "证券代码": "symbol",
                "股票代码": "symbol",
                "代码": "symbol",
                "名称": "name"
            }

            # 应用列映射
            renamed_cols = {}
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    renamed_cols[old_col] = new_col

            if renamed_cols:
                df = df.rename(columns=renamed_cols)

            # 确保必要的列存在
            required_cols = ["date", "open", "close", "high", "low"]
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
            # 应用频率限制
            self._rate_limit()

            # 处理不同格式的股票代码
            original_symbol = symbol
            # 移除可能的前缀
            if symbol.startswith(('sh', 'sz')):
                symbol = symbol[2:]

            logger.debug(
                f"使用备用方法获取股票数据: 代码={symbol}, 开始日期={start_date}, 结束日期={end_date}")

            # 最新版akshare的东方财富接口参数变更
            # 需要转换代码格式
            formatted_symbols = []

            # 尝试多种格式
            if symbol.startswith("6"):
                formatted_symbols = [
                    symbol,                # 原始代码
                    f"sh{symbol}",         # sh前缀
                    f"1.{symbol}",         # 东方财富格式（上交所）
                    f"sh.{symbol}"         # baostock格式
                ]
            elif symbol.startswith("0") or symbol.startswith("3"):
                formatted_symbols = [
                    symbol,                # 原始代码
                    f"sz{symbol}",         # sz前缀
                    f"0.{symbol}",         # 东方财富格式（深交所）
                    f"sz.{symbol}"         # baostock格式
                ]
            else:
                formatted_symbols = [
                    symbol,
                    f"sh{symbol}",
                    f"sz{symbol}",
                    f"1.{symbol}",
                    f"0.{symbol}"
                ]

            # 尝试不同的股票代码格式
            for fmt_symbol in formatted_symbols:
                logger.debug(f"尝试使用股票代码格式: {fmt_symbol}")

                try:
                    # 尝试使用stock_zh_a_hist作为备用方法
                    self._rate_limit()
                    try:
                        df = ak.stock_zh_a_hist(
                            symbol=fmt_symbol,
                            period="daily",
                            start_date=start_date,
                            end_date=end_date,
                            adjust="qfq"
                        )

                        # 检查返回数据
                        if df is not None and not df.empty:
                            logger.info(
                                f"使用stock_zh_a_hist备用方法成功获取{fmt_symbol}数据，共{len(df)}行")

                            # 检查列名并标准化
                            if "日期" in df.columns:  # 中文列名
                                df = df.rename(columns={
                                    "日期": "date",
                                    "开盘": "open",
                                    "收盘": "close",
                                    "最高": "high",
                                    "最低": "low",
                                    "成交量": "volume",
                                    "成交额": "amount"
                                })

                            # 确保date是datetime类型
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.sort_values("date")

                            return df
                    except Exception as e:
                        logger.debug(
                            f"使用stock_zh_a_hist备用方法获取{fmt_symbol}数据失败: {str(e)}")

                    # 尝试使用新版东方财富接口
                    try:
                        self._rate_limit()
                        df = ak.stock_zh_a_daily(
                            symbol=fmt_symbol, start_date=start_date, end_date=end_date, adjust="qfq")

                        if df is not None and not df.empty:
                            logger.info(
                                f"使用stock_zh_a_daily成功获取{fmt_symbol}数据，共{len(df)}行")

                            # 确保有必要的列
                            if "date" not in df.columns:
                                if df.index.name == "date":
                                    df = df.reset_index()
                                else:
                                    continue

                            # 确保数据类型正确
                            df["date"] = pd.to_datetime(df["date"])

                            return df
                    except Exception as e2:
                        logger.debug(
                            f"使用stock_zh_a_daily获取{fmt_symbol}数据失败: {str(e2)}")

                    # 尝试第三种方法 - 新浪财经数据
                    try:
                        self._rate_limit()

                        # 不同格式尝试
                        sina_symbols = []
                        if fmt_symbol.startswith("sh") or fmt_symbol.startswith("sz"):
                            sina_symbols.append(fmt_symbol)
                        elif fmt_symbol.startswith("6"):
                            sina_symbols.append(f"sh{fmt_symbol}")
                        else:
                            sina_symbols.append(f"sz{fmt_symbol}")

                        for sina_symbol in sina_symbols:
                            try:
                                logger.debug(f"尝试使用新浪接口获取数据: {sina_symbol}")
                                df = ak.stock_zh_index_daily_tx(
                                    symbol=sina_symbol)

                                if df is not None and not df.empty:
                                    logger.info(
                                        f"使用新浪数据源成功获取{sina_symbol}数据，共{len(df)}行")

                                    # 标准化列名
                                    standard_columns = {
                                        "date": "date",
                                        "open": "open",
                                        "close": "close",
                                        "high": "high",
                                        "low": "low",
                                        "volume": "volume"
                                    }

                                    df = df.rename(columns={col: standard_columns[col]
                                                            for col in df.columns
                                                            if col in standard_columns})

                                    # 确保date是datetime类型
                                    df["date"] = pd.to_datetime(df["date"])

                                    return df
                            except Exception as e_sina:
                                logger.debug(
                                    f"使用新浪接口获取{sina_symbol}数据失败: {str(e_sina)}")
                                continue
                    except Exception as e3:
                        logger.debug(f"所有新浪接口尝试均失败: {str(e3)}")

                except Exception as e_all:
                    logger.debug(f"对代码{fmt_symbol}的所有尝试均失败: {str(e_all)}")
                    continue

            # 如果所有尝试都失败
            logger.error(f"所有备用方法获取{symbol}数据均失败")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取{symbol}的替代K线数据失败: {str(e)}")
            logger.debug(traceback.format_exc())
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

    def test_stock_data_fetch(self, symbol: str) -> bool:
        """测试获取特定股票的数据，用于调试和确认功能正常

        Args:
            symbol: 股票代码

        Returns:
            bool: 是否成功获取数据
        """
        logger.info(f"测试获取股票数据: {symbol}")

        # 设置较短的时间段，提高测试速度
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() -
                      datetime.timedelta(days=30)).strftime("%Y-%m-%d")

        # 先尝试从主数据源获取
        logger.info(f"尝试从主数据源 {self.data_source} 获取数据...")
        if self.data_source == "akshare":
            df = self._get_k_data_akshare(
                symbol, start_date, end_date, "daily", "qfq")
        else:
            df = self._get_k_data_baostock(
                symbol, start_date, end_date, "daily", "qfq")

        if not df.empty:
            logger.info(f"成功从主数据源获取到数据，共 {len(df)} 行")
            return True

        # 尝试备用方法
        logger.info(f"从主数据源获取失败，尝试备用方法...")
        df = self._get_k_data_alternative(symbol, start_date, end_date)

        if not df.empty:
            logger.info(f"成功从备用方法获取到数据，共 {len(df)} 行")
            return True

        # 如果akshare备用方法获取失败，尝试baostock
        if self.data_source == "akshare" and bs:
            logger.info(f"从akshare备用方法获取失败，尝试使用baostock...")
            df = self._get_k_data_baostock(
                symbol, start_date, end_date, "daily", "qfq")

            if not df.empty:
                logger.info(f"成功从baostock获取到数据，共 {len(df)} 行")
                return True

        # 如果baostock获取失败，尝试akshare
        if self.data_source == "baostock" and ak:
            logger.info(f"从baostock获取失败，尝试使用akshare...")
            df = self._get_k_data_akshare(
                symbol, start_date, end_date, "daily", "qfq")

            if not df.empty:
                logger.info(f"成功从akshare获取到数据，共 {len(df)} 行")
                return True

            # 如果还是失败，尝试akshare的备用方法
            if df.empty:
                logger.info(f"从akshare获取失败，尝试使用akshare备用方法...")
                df = self._get_k_data_alternative(symbol, start_date, end_date)

                if not df.empty:
                    logger.info(f"成功从akshare备用方法获取到数据，共 {len(df)} 行")
                    return True

        logger.error(f"所有方法获取 {symbol} 数据均失败!")
        return False


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

        # 测试特定股票数据获取
        # 选择几只常见会出问题的股票进行测试
        problem_stocks = ["002797", "002772", "002539"]
        print("\n测试常见问题股票数据获取:")
        for stock in problem_stocks:
            success = data_fetcher.test_stock_data_fetch(stock)
            print(f"{stock}: {'成功' if success else '失败'}")

    # 获取指数数据
    print("\n获取上证指数数据...")
    index_data = data_fetcher.get_index_data("sh000001")
    print(f"共获取到{len(index_data)}条上证指数数据")
    if not index_data.empty:
        print("\n最近5天上证指数数据:")
        print(index_data.tail())
