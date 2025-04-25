#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化工具模块
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Union, Tuple
from loguru import logger

try:
    from pyecharts import options as opts
    from pyecharts.charts import Kline, Line, Bar, Grid, Scatter
    HAS_PYECHARTS = True
except ImportError:
    logger.warning("未安装pyecharts，部分可视化功能将不可用")
    HAS_PYECHARTS = False

# 设置中文支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception as e:
    logger.warning(f"设置matplotlib中文支持失败: {str(e)}")


class StockVisualizer:
    """股票数据可视化工具"""

    @staticmethod
    def plot_stock_signals(data: pd.DataFrame,
                           strategy_name: str = "",
                           save_path: Optional[str] = None,
                           show_plot: bool = True,
                           figsize: Tuple[int, int] = (14, 10)) -> None:
        """绘制带有买入信号的股票K线和指标图

        Args:
            data: 包含价格、指标和信号的DataFrame
            strategy_name: 策略名称
            save_path: 图表保存路径，None则不保存
            show_plot: 是否显示图表
            figsize: 图表大小
        """
        if data.empty:
            logger.warning("输入数据为空，无法绘制图表")
            return

        # 确保日期索引
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("数据索引不是日期类型，尝试转换")
            try:
                if "date" in data.columns:
                    data = data.set_index("date")
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"转换日期索引失败: {str(e)}")
                return

        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f"{data['symbol'].iloc[0] if 'symbol' in data.columns else ''} - {strategy_name}",
                     fontsize=16)

        # 调整子图间距
        plt.subplots_adjust(hspace=0.05)

        # 绘制K线图
        ax1 = axes[0]
        ax1.set_title("价格与信号")

        # 绘制收盘价
        ax1.plot(data.index, data["close"], 'k-',
                 linewidth=1.5, alpha=0.7, label="收盘价")

        # 如果有均线数据，绘制均线
        ma_columns = [col for col in data.columns if col.startswith("MA_")]
        for ma in ma_columns:
            ax1.plot(data.index, data[ma], linewidth=1, alpha=0.7, label=ma)

        # 如果有布林带数据，绘制布林带
        if all(col in data.columns for col in ["BOLL_UPPER", "BOLL_MIDDLE", "BOLL_LOWER"]):
            ax1.plot(data.index, data["BOLL_UPPER"],
                     'g--', linewidth=1, alpha=0.4, label="布林上轨")
            ax1.plot(data.index, data["BOLL_MIDDLE"],
                     'b--', linewidth=1, alpha=0.4, label="布林中轨")
            ax1.plot(data.index, data["BOLL_LOWER"],
                     'g--', linewidth=1, alpha=0.4, label="布林下轨")
            ax1.fill_between(
                data.index, data["BOLL_UPPER"], data["BOLL_LOWER"], alpha=0.1, color='gray')

        # 绘制买入信号
        if "signal" in data.columns:
            buy_signals = data[data["signal"] > 0]
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals["close"],
                            marker='^', s=100, color='red', alpha=0.8, label="买入信号")

                # 添加信号标签
                for idx, row in buy_signals.iterrows():
                    ax1.annotate(f"{idx.strftime('%m-%d')}",
                                 xy=(idx, row["close"]),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center', va='bottom',
                                 fontsize=8)

        # 设置y轴格式为保留2位小数
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # 添加网格和图例
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # 绘制MACD图
        ax2 = axes[1]
        if all(col in data.columns for col in ["MACD", "MACD_signal", "MACD_histogram"]):
            ax2.set_title("MACD")
            ax2.plot(data.index, data["MACD"], 'b-',
                     linewidth=1.5, label="MACD")
            ax2.plot(data.index, data["MACD_signal"],
                     'r-', linewidth=1.5, label="信号线")

            # 绘制柱状图
            colors = np.where(data["MACD_histogram"] >= 0, 'g', 'r')
            ax2.bar(data.index, data["MACD_histogram"],
                    color=colors, alpha=0.5, label="柱状图")

            # 添加零线
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

            # 添加MACD零线金叉和MACD金叉标记
            if "MACD_zero_golden_cross" in data.columns:
                zero_cross = data[data["MACD_zero_golden_cross"]]
                if not zero_cross.empty:
                    ax2.scatter(zero_cross.index, zero_cross["MACD"],
                                marker='o', s=50, color='purple', alpha=0.8, label="零线金叉")

            if "MACD_golden_cross" in data.columns:
                macd_cross = data[data["MACD_golden_cross"]]
                if not macd_cross.empty:
                    ax2.scatter(macd_cross.index, macd_cross["MACD"],
                                marker='*', s=80, color='blue', alpha=0.8, label="MACD金叉")

            ax2.legend(loc='upper left')
        else:
            ax2.set_visible(False)

        # 绘制RSI图
        ax3 = axes[2]
        rsi_cols = [col for col in data.columns if col.startswith("RSI_")]
        if rsi_cols:
            rsi_col = rsi_cols[0]  # 使用第一个RSI指标
            ax3.set_title(f"{rsi_col}")
            ax3.plot(data.index, data[rsi_col], 'g-', linewidth=1.5)

            # 添加超买超卖线
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)

            # 添加区域填充
            ax3.fill_between(data.index, data[rsi_col], 30,
                             where=(data[rsi_col] <= 30),
                             color='g', alpha=0.3)
            ax3.fill_between(data.index, data[rsi_col], 70,
                             where=(data[rsi_col] >= 70),
                             color='r', alpha=0.3)

            # 设置y轴范围
            ax3.set_ylim(0, 100)
        else:
            ax3.set_visible(False)

        # 设置日期格式
        ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        if save_path:
            try:
                plt.savefig(save_path, dpi=120, bbox_inches='tight')
                logger.info(f"图表已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存图表失败: {str(e)}")

        # 显示图表
        if show_plot:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def create_html_chart(data: pd.DataFrame,
                          title: str = "",
                          save_path: Optional[str] = None) -> str:
        """创建HTML交互式图表

        Args:
            data: 包含价格、指标和信号的DataFrame
            title: 图表标题
            save_path: 图表保存路径，None则不保存

        Returns:
            str: HTML图表代码
        """
        if not HAS_PYECHARTS:
            logger.error("创建HTML图表需要安装pyecharts库")
            return ""

        if data.empty:
            logger.warning("输入数据为空，无法创建图表")
            return ""

        # 确保日期索引
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("数据索引不是日期类型，尝试转换")
            try:
                if "date" in data.columns:
                    data = data.set_index("date")
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"转换日期索引失败: {str(e)}")
                return ""

        # 日期字符串列表
        date_list = data.index.strftime('%Y-%m-%d').tolist()

        # 创建K线图
        kline = (
            Kline()
            .add_xaxis(date_list)
            .add_yaxis(
                "K线",
                data[["open", "close", "low", "high"]].values.tolist(),
                markpoint_opts=opts.MarkPointOpts(
                    data=[
                        opts.MarkPointItem(type_="max", name="最高价"),
                        opts.MarkPointItem(type_="min", name="最低价"),
                    ]
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True,
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=True,
                        type_="slider",
                        range_start=50,
                        range_end=100,
                    ),
                ],
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis", axis_pointer_type="cross"),
            )
        )

        # 添加均线
        line = (
            Line()
            .add_xaxis(date_list)
        )

        # 添加MA5/MA20/MA60
        ma_colors = {
            "MA_5": "#39c0c6",
            "MA_20": "#fe4365",
            "MA_60": "#ffd400"
        }

        for ma, color in ma_colors.items():
            if ma in data.columns:
                line.add_yaxis(
                    ma,
                    data[ma].round(2).tolist(),
                    is_smooth=True,
                    linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False),
                    itemstyle_opts=opts.ItemStyleOpts(color=color),
                )

        # 组合K线图和均线图
        overlap_kline = kline.overlap(line)

        # 创建MACD图
        if all(col in data.columns for col in ["MACD", "MACD_signal", "MACD_histogram"]):
            macd_bar = (
                Bar()
                .add_xaxis(date_list)
                .add_yaxis(
                    "MACD柱状",
                    data["MACD_histogram"].round(4).tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    itemstyle_opts=opts.ItemStyleOpts(
                        color=opts.JsCode(
                            """
                            function(params) {
                                var colorList;
                                if (params.data >= 0) {
                                    colorList = '#ef232a';
                                } else {
                                    colorList = '#14b143';
                                }
                                return colorList;
                            }
                            """
                        )
                    ),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="MACD", pos_left="50%"),
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        grid_index=1,
                        axislabel_opts=opts.LabelOpts(is_show=False),
                    ),
                    yaxis_opts=opts.AxisOpts(
                        grid_index=1,
                        split_number=4,
                        axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        splitline_opts=opts.SplitLineOpts(is_show=False),
                        axislabel_opts=opts.LabelOpts(is_show=True),
                    ),
                )
            )

            macd_line = (
                Line()
                .add_xaxis(date_list)
                .add_yaxis(
                    "DIF",
                    data["MACD"].round(4).tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    itemstyle_opts=opts.ItemStyleOpts(color="#8c3ee6"),
                )
                .add_yaxis(
                    "DEA",
                    data["MACD_signal"].round(4).tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    itemstyle_opts=opts.ItemStyleOpts(color="#dda0dd"),
                )
                .set_global_opts(
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        grid_index=1,
                        axislabel_opts=opts.LabelOpts(is_show=False),
                    ),
                    yaxis_opts=opts.AxisOpts(
                        grid_index=1,
                        split_number=4,
                        axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        splitline_opts=opts.SplitLineOpts(is_show=False),
                        axislabel_opts=opts.LabelOpts(is_show=True),
                    ),
                )
            )

            # 组合MACD图表
            overlap_macd = macd_bar.overlap(macd_line)
        else:
            overlap_macd = None

        # 创建RSI图
        rsi_cols = [col for col in data.columns if col.startswith("RSI_")]
        if rsi_cols:
            rsi_col = rsi_cols[0]  # 使用第一个RSI指标

            rsi_line = (
                Line()
                .add_xaxis(date_list)
                .add_yaxis(
                    rsi_col,
                    data[rsi_col].round(2).tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    itemstyle_opts=opts.ItemStyleOpts(color="#4169e1"),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="RSI", pos_left="50%"),
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        grid_index=2,
                        axislabel_opts=opts.LabelOpts(is_show=True),
                    ),
                    yaxis_opts=opts.AxisOpts(
                        grid_index=2,
                        min_=0,
                        max_=100,
                        split_number=4,
                        axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        splitline_opts=opts.SplitLineOpts(is_show=False),
                        axislabel_opts=opts.LabelOpts(is_show=True),
                    ),
                )
                .set_series_opts(
                    markline_opts=opts.MarkLineOpts(
                        data=[
                            opts.MarkLineItem(y=30, name="超卖"),
                            opts.MarkLineItem(y=70, name="超买")
                        ],
                        label_opts=opts.LabelOpts(position="end"),
                    )
                )
            )
        else:
            rsi_line = None

        # 买入信号标记
        if "signal" in data.columns:
            buy_signals = data[data["signal"] > 0]
            if not buy_signals.empty:
                scatter = (
                    Scatter()
                    .add_xaxis(buy_signals.index.strftime('%Y-%m-%d').tolist())
                    .add_yaxis(
                        "买入信号",
                        buy_signals["close"].tolist(),
                        symbol_size=10,
                        label_opts=opts.LabelOpts(is_show=False),
                        itemstyle_opts=opts.ItemStyleOpts(color="red"),
                    )
                )

                overlap_kline = overlap_kline.overlap(scatter)

        # 创建网格布局
        grid_chart = Grid()

        # 设置布局
        if overlap_macd is not None and rsi_line is not None:
            grid_chart.add(
                overlap_kline,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", height="50%"),
            )
            grid_chart.add(
                overlap_macd,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", pos_top="63%", height="16%"),
            )
            grid_chart.add(
                rsi_line,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", pos_top="82%", height="16%"),
            )
        elif overlap_macd is not None:
            grid_chart.add(
                overlap_kline,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", height="58%"),
            )
            grid_chart.add(
                overlap_macd,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", pos_top="70%", height="28%"),
            )
        elif rsi_line is not None:
            grid_chart.add(
                overlap_kline,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", height="58%"),
            )
            grid_chart.add(
                rsi_line,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", pos_top="70%", height="28%"),
            )
        else:
            grid_chart.add(
                overlap_kline,
                grid_opts=opts.GridOpts(
                    pos_left="5%", pos_right="1%", height="90%"),
            )

        # 保存HTML图表
        if save_path:
            try:
                grid_chart.render(save_path)
                logger.info(f"HTML图表已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存HTML图表失败: {str(e)}")

        return grid_chart.render_embed()


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(".")  # 添加项目根目录到路径

    from data.fetcher import data_fetcher
    from indicators.basic import BasicIndicator
    from strategies.macd_rsi import MacdRsiStrategy

    # 获取一支股票的数据
    stock_list = data_fetcher.get_stock_list()
    if not stock_list.empty:
        # 获取第一只股票的数据
        symbol = stock_list.iloc[0]["symbol"]
        name = stock_list.iloc[0]["name"]
        print(f"使用{symbol}({name})测试可视化模块...")

        # 获取K线数据
        data = data_fetcher.get_k_data(symbol, period="daily")

        if not data.empty:
            # 计算指标
            data = BasicIndicator.calculate_all(data)

            # 添加策略信号
            strategy = MacdRsiStrategy()
            data = strategy.scan(data)

            # 添加股票代码字段
            data["symbol"] = symbol

            # 测试静态图表
            StockVisualizer.plot_stock_signals(
                data.tail(100),  # 仅使用最近100条数据
                strategy_name=f"{name} - MACD与RSI策略",
                save_path="./results/test_plot.png"
            )

            # 测试HTML图表
            if HAS_PYECHARTS:
                html = StockVisualizer.create_html_chart(
                    data.tail(100),  # 仅使用最近100条数据
                    title=f"{symbol} {name} - 技术指标分析",
                    save_path="./results/test_chart.html"
                )
                print(f"HTML图表已创建，代码长度: {len(html)}")
