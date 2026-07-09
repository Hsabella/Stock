"""基线模型训练：LightGBM / 线性，固定切分或季度滚动 walk-forward。

产出预测分（(datetime, instrument) 索引的 Series）落盘 results/quant/pred_<tag>.parquet，
供 run_backtest 作为选股信号。滚动模式下所有预测均为严格样本外。

用法:
    python -m quant.model.baseline --model lgb --mode fixed      # 打通用
    python -m quant.model.baseline --model lgb --mode rolling    # 正式评估用
    python -m quant.model.baseline --model linear --mode rolling
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant.config import load_config

OUT_DIR = Path("results/quant")


def _init_qlib():
    import os

    import qlib

    # 新版 mlflow 文件后端默认拒写；实验日志统一放 ~/.qlib/mlruns（不进项目仓库）
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    qlib.init(
        provider_uri=load_config("data")["qlib"]["provider_uri"],
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file:" + str(Path.home() / ".qlib" / "mlruns"),
                "default_exp_name": "Experiment",
            },
        },
    )


def make_dataset(handler, segments: dict):
    from qlib.data.dataset import DatasetH

    return DatasetH(handler=handler, segments={k: tuple(v) for k, v in segments.items()})


def make_model(kind: str, cfg: dict):
    if kind == "lgb":
        from qlib.contrib.model.gbdt import LGBModel

        p = dict(cfg["model"]["lgb"])
        return LGBModel(
            loss=p.pop("loss"),
            num_boost_round=p.pop("num_boost_round"),
            early_stopping_rounds=p.pop("early_stopping_rounds"),
            **p,
        )
    from qlib.contrib.model.linear import LinearModel

    return LinearModel(estimator="ols")


def build_handler(start: str, end: str):
    from quant.features.handler import AlphaV1Handler

    return AlphaV1Handler(start_time=start, end_time=end)


def train_fixed(kind: str, cfg: dict) -> pd.Series:
    seg = cfg["segments"]
    handler = build_handler(seg["train"][0], seg["test"][1])
    dataset = make_dataset(handler, seg)
    model = make_model(kind, cfg)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    return pred.rename("score")


def train_rolling(kind: str, cfg: dict) -> pd.Series:
    """季度滚动：每步用截至上一交易日的 train_days 天训练，预测未来 step_days 天。"""
    from qlib.data import D

    roll = cfg["rolling"]
    cal = [pd.Timestamp(t) for t in D.calendar(freq="day")]
    out_start = pd.Timestamp(roll["out_start"])
    out_end = pd.Timestamp(roll["out_end"])
    idx_start = next(i for i, t in enumerate(cal) if t >= out_start)

    # 数据只加载一次（handler 较重），各滚动窗口通过 DatasetH segments 切片
    handler_start = cal[max(0, idx_start - roll["train_days"] - 130)]  # 因子 warmup 余量
    handler = build_handler(str(handler_start.date()), str(out_end.date()))

    preds: list[pd.Series] = []
    i = idx_start
    step = 0
    while i < len(cal) and cal[i] <= out_end:
        train_lo = cal[max(0, i - roll["train_days"])]
        train_hi = cal[i - 1]
        # 验证段：训练窗末尾 10% 供 LightGBM 早停，与训练段不重叠
        valid_idx = max(1, i - roll["train_days"] // 10)
        test_hi = cal[min(i + roll["step_days"] - 1, len(cal) - 1)]
        if test_hi > out_end:
            test_hi = out_end
        segments = {
            "train": (str(train_lo.date()), str(cal[valid_idx - 1].date())),
            "valid": (str(cal[valid_idx].date()), str(train_hi.date())),
            "test": (str(cal[i].date()), str(test_hi.date())),
        }
        dataset = make_dataset(handler, segments)
        model = make_model(kind, cfg)
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
        preds.append(pred)
        step += 1
        print(f"[baseline] roll#{step} train[{segments['train'][0]}→{segments['valid'][1]}] "
              f"→ test[{segments['test'][0]}→{segments['test'][1]}] 预测 {len(pred):,} 条", flush=True)
        i += roll["step_days"]

    return pd.concat(preds).rename("score").sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lgb", "linear"], default="lgb")
    parser.add_argument("--mode", choices=["fixed", "rolling"], default="fixed")
    args = parser.parse_args()

    cfg = load_config("backtest")
    _init_qlib()
    pred = train_fixed(args.model, cfg) if args.mode == "fixed" else train_rolling(args.model, cfg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.mode}"
    out = OUT_DIR / f"pred_{tag}.parquet"
    pred.to_frame().to_parquet(out)
    print(f"[baseline] 预测已写 {out}: {len(pred):,} 条, "
          f"{pred.index.get_level_values('datetime').min().date()} → "
          f"{pred.index.get_level_values('datetime').max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
