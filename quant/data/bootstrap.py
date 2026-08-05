"""qlib 数据包引导：下载/解压 chenditc qlib_bin 并做初始化冒烟测试。

数据源是社区维护的 chenditc/investment_data（qlib 官方 cn_data 已停更），
每日发布 qlib_bin.tar.gz（2005 至今日线，tushare+akshare 交叉校验）。

用法:
    python -m quant.data.bootstrap                # 缺包则下载，解压，冒烟
    python -m quant.data.bootstrap --force-download
    python -m quant.data.bootstrap --skip-extract  # 只跑冒烟测试
"""
from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from quant.config import load_config

_CHUNK = 1 << 20  # 1 MB


def _ssl_context() -> ssl.SSLContext:
    """显式用 certifi 的根证书建 SSL 上下文。

    python.org 版 macOS Python 装完不跑 Install Certificates.command 的话，
    /Library/Frameworks/.../etc/openssl/ 是空的、ssl 默认信任库为空，
    urllib 走默认上下文必然 CERTIFICATE_VERIFY_FAILED（2026-08 踩坑：
    bootstrap 从装机起就 100% 挂，而 akshare 走 requests+certifi 所以没暴露）。
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_archive(url: str, dest: Path, force: bool = False) -> Path:
    """下载 release 包；已存在且非 force 时直接复用。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[bootstrap] 复用已有归档: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    print(f"[bootstrap] 下载 {url}")
    tmp = dest.with_suffix(".part")
    # 不用 urlretrieve：它不接受 context 参数，没法指定 certifi 根证书
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_report = _CHUNK * 100
        with open(tmp, "wb") as fh:
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"  ... {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                    next_report += _CHUNK * 100
    if total and done < total:
        tmp.unlink(missing_ok=True)
        raise IOError(f"下载不完整: {done}/{total} 字节")
    tmp.rename(dest)
    print(f"[bootstrap] 下载完成: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _find_data_root(extract_dir: Path) -> Path:
    """在解压目录里定位包含 calendars/day.txt 的数据根目录（容忍不同打包层级）。"""
    for cand in [extract_dir, *extract_dir.rglob("*")]:
        if cand.is_dir() and (cand / "calendars" / "day.txt").exists():
            return cand
    raise FileNotFoundError(f"解压结果中找不到 calendars/day.txt: {extract_dir}")


def extract_archive(archive: Path, provider_uri: Path) -> None:
    """解压归档并原子替换 provider_uri（旧数据先备份为 .bak 再删除）。"""
    print(f"[bootstrap] 解压 {archive} ...")
    with tempfile.TemporaryDirectory(dir=provider_uri.parent if provider_uri.parent.exists() else None) as td:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(td)
        root = _find_data_root(Path(td))

        provider_uri.parent.mkdir(parents=True, exist_ok=True)
        bak = provider_uri.with_suffix(".bak")
        if provider_uri.exists():
            if bak.exists():
                shutil.rmtree(bak)
            provider_uri.rename(bak)
        shutil.move(str(root), str(provider_uri))
        if bak.exists():
            shutil.rmtree(bak)
    print(f"[bootstrap] 数据就位: {provider_uri}")


def smoke_test(provider_uri: Path) -> dict:
    """qlib.init + 基础可用性检查，返回摘要信息（供 checks/文档使用）。"""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region="cn")

    cal = D.calendar(freq="day")
    info = {
        "calendar_start": str(cal[0].date()),
        "calendar_end": str(cal[-1].date()),
        "n_trading_days": len(cal),
    }

    inst_dir = provider_uri / "instruments"
    info["instruments_files"] = sorted(p.stem for p in inst_dir.glob("*.txt"))

    # 任选一只老牌股票探测字段清单
    sample_dir = provider_uri / "features" / "sh600000"
    if sample_dir.exists():
        info["fields"] = sorted(p.name.split(".")[0] for p in sample_dir.glob("*.day.bin"))

    df = D.features(["SH600000"], ["$close", "$volume", "$factor"], start_time="2020-01-01", end_time="2020-01-31")
    if df.empty:
        raise RuntimeError("冒烟失败: SH600000 2020-01 无数据")
    info["sample_rows"] = len(df)

    print("[bootstrap] 冒烟通过:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true", help="数据已就位时只跑冒烟")
    args = parser.parse_args()

    cfg = load_config("data")
    provider_uri = Path(cfg["qlib"]["provider_uri"])

    if not args.skip_extract:
        archive = download_archive(
            cfg["download"]["release_url"], Path(cfg["download"]["archive_path"]), force=args.force_download
        )
        extract_archive(archive, provider_uri)

    smoke_test(provider_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
