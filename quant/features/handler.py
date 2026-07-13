"""qlib DataHandler：量价 expression + parquet 扩展特征拼装，供模型训练（M3）用。

设计：
- QlibDataLoader 负责 13 个量价因子 + LABEL（alpha_pv 定义）
- StaticDataLoader 负责 ext_features.parquet（换手/市值/估值）
- NestedDataLoader 把两者按 (datetime, instrument) 对齐拼接
- 推理链: RobustZScoreNorm(截面外稳健标准化, 用训练段拟合) + Fillna
- 学习链: DropnaLabel + CSRankNorm(标签截面排名化, 学"相对强弱"而非绝对收益)

若 NestedDataLoader 在数据对齐上出问题（计划中的已知风险），降级方案是
ext_features 预计算并入一张大表后纯 StaticDataLoader 供数。
"""
from __future__ import annotations

from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor, get_group_columns

from quant.data import warehouse
from quant.features.alpha_pv import EXPRESSIONS, EXT_FEATURES, LABEL_EXPRESSION, REGIME_FEATURES


class CSRankNormExempt(Processor):
    """CSRankNorm 变体：exempt 列保持原值。

    regime 特征全截面同值，截面 rank 会把它们退化成常数（信息清零）；
    LightGBM 对量纲不敏感，原值直接可用。其余列与 qlib CSRankNorm 同款处理。
    """

    def __init__(self, fields_group="feature", exempt=()):
        self.fields_group = fields_group
        self.exempt = set(exempt)

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        cols = [c for c in cols if (c[-1] if isinstance(c, tuple) else c) not in self.exempt]
        t = df[cols].groupby("datetime", group_keys=False).rank(pct=True)
        t -= 0.5
        t *= 3.46  # 与 qlib CSRankNorm 相同：均匀分布 rank 的近似 zscore 缩放
        df[cols] = t
        return df


# 截面标准化（按日 rank 后 zscore，regime 列豁免）：无需拟合期参数 → 滚动训练零泄漏
_DEFAULT_INFER = [
    {"class": "CSRankNormExempt", "module_path": "quant.features.handler",
     "kwargs": {"fields_group": "feature", "exempt": REGIME_FEATURES}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]
_DEFAULT_LEARN = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]


class AlphaV1Handler(DataHandlerLP):
    """v1 因子集 handler（周频标签，池子默认 csi_union）。"""

    def __init__(
        self,
        instruments="csi_union",
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        infer_processors=None,
        learn_processors=None,
        with_ext: bool = True,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors if infer_processors is not None else _DEFAULT_INFER,
            fit_start_time, fit_end_time,
        )
        learn_processors = check_transform_proc(
            learn_processors if learn_processors is not None else _DEFAULT_LEARN, None, None
        )

        from qlib.data.dataset.loader import NestedDataLoader, QlibDataLoader, StaticDataLoader

        data_loader = QlibDataLoader(
            config={
                "feature": (list(EXPRESSIONS.values()), list(EXPRESSIONS)),
                "label": ([LABEL_EXPRESSION], ["LABEL0"]),
            }
        )
        ext_path = warehouse.warehouse_dir() / "ext_features.parquet"
        if with_ext and ext_path.exists():
            import pandas as pd

            ext = pd.read_parquet(ext_path)
            # 只取 EXT_FEATURES 注册过的列：parquet 可以携带未注册/未验证的列
            # （如 IC 体检不过的因子、留待 S3 的 regime 列）而不悄悄进模型
            ext = ext[[c for c in EXT_FEATURES if c in ext.columns]]
            # StaticDataLoader 要求列带 (group, name) 两级
            ext.columns = pd.MultiIndex.from_product([["feature"], ext.columns])
            data_loader = NestedDataLoader(
                dataloader_l=[data_loader, StaticDataLoader(config=ext)], join="left"
            )

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **kwargs,
        )
