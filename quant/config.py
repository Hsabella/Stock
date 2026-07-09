"""quant 包配置加载：读 quant/configs/*.yaml，路径字段统一展开 ~。"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"

_PATH_KEYS = {"provider_uri", "archive_path", "path", "st_cache"}


def _expand_paths(node):
    if isinstance(node, dict):
        return {
            k: (str(Path(v).expanduser()) if k in _PATH_KEYS and isinstance(v, str) else _expand_paths(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_expand_paths(v) for v in node]
    return node


def load_config(name: str = "data") -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_paths(cfg)
