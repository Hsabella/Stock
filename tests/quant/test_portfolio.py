"""portfolio 数据层：止损线手算对拍 / atr_k 解析 / 陈旧判定 / 引擎信号解析。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.live import portfolio  # noqa: E402
from quant.live.portfolio import (  # noqa: E402
    _atr, load_kline, load_portfolio_config, stop_lines,
)


def _kline(n=40, close_start=100.0, step=-1.0, spread=2.0):
    """合成日K：每日 close 递变 step，high/low 对称 spread/2。"""
    dates = pd.bdate_range("2026-05-01", periods=n)
    close = pd.Series([close_start + step * i for i in range(n)])
    return pd.DataFrame({
        "date": dates, "close": close,
        "high": close + spread / 2, "low": close - spread / 2,
        "open": close,
    })


def test_stop_lines_hand_calculated():
    """恒定 TR=spread 时 ATR14=spread；追踪线=20日最高收盘-k×spread。"""
    k = _kline(n=40, close_start=100.0, step=-1.0, spread=2.0)
    # 手算: TR = max(high-low, |high-pc|, |low-pc|) = max(2, |−1+1|+... ) = 2? 逐日:
    # high-low=2; high-pc = (c+1)-(c_prev)= c_prev-1+1-c_prev = 0? 实际 close 递减 1:
    # high_t = c_t+1, pc = c_t+1 → high-pc = 0; low-pc = c_t-1-(c_t+1) = -2 → abs=2 → TR=2
    atr = _atr(k["high"], k["low"], k["close"], n=14)
    assert abs(atr.iloc[-1] - 2.0) < 1e-9
    out = stop_lines(k, entry_price=90.0, hard_stop_pct=0.08, atr_k=2.0)
    # 近20日最高收盘 = 40根中最后20根的最高 = close[20] = 100-20 = 80
    assert abs(out["atr_stop"] - (80.0 - 2.0 * 2.0)) < 1e-9
    assert abs(out["hard_stop"] - 90.0 * 0.92) < 1e-9
    # stop_line 取两者较严（较高）者 = 82.8; 现价 61 < 82.8 → 已破
    assert abs(out["stop_line"] - 82.8) < 1e-9
    assert out["broken"]
    assert out["dist_pct"] < 0


def test_stop_lines_not_broken_when_price_above():
    k = _kline(n=40, close_start=100.0, step=+0.5, spread=1.0)  # 上涨趋势
    out = stop_lines(k, entry_price=100.0, hard_stop_pct=0.08, atr_k=2.0)
    assert not out["broken"]
    assert out["dist_pct"] > 0


def test_atr_k_parsed_from_strategy(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(
        "default_stop_strategy: atr_3\nhard_stop_pct: 0.05\n"
        "watchlist:\n"
        "  - { symbol: '600000', name: x, state: HELD, position: 100, entry_price: 10.0 }\n"
        "  - { symbol: '600001', name: y, state: WATCHING }\n")
    cfg = load_portfolio_config(p)
    assert cfg["atr_k"] == 3.0
    assert cfg["hard_stop_pct"] == 0.05
    assert [h["symbol"] for h in cfg["held"]] == ["600000"]


def test_load_kline_stale_detection(tmp_path, monkeypatch):
    """缓存末日落后 >3 个交易日 → stale=True（akshare 失败时仍返回旧数据）。"""
    kdir = tmp_path / "kline"
    kdir.mkdir()
    _kline(n=30).assign(date=pd.bdate_range("2026-06-01", periods=30)).to_csv(
        kdir / "600000_daily_qfq.csv", index=False)
    monkeypatch.setattr(portfolio, "KLINE_DIR", kdir)
    monkeypatch.setattr(portfolio, "STALE_TRADING_DAYS", 3)

    import akshare as ak
    monkeypatch.setattr(ak, "stock_zh_a_daily",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("网络挂了")),
                        raising=False)
    last = pd.Timestamp(pd.bdate_range("2026-06-01", periods=30)[-1])
    df, stale = load_kline("600000", today=last + pd.offsets.BDay(1))
    assert not stale and not df.empty
    df, stale = load_kline("600000", today=last + pd.offsets.BDay(10))
    assert stale and not df.empty  # 降级失败仍给旧数据, 由调用方标 ⚠️


def _cfg(cash=80000.0, etf=None):
    return {
        "held": [], "atr_k": 2.0, "hard_stop_pct": 0.08,
        "account": {"cash": cash, "etf": etf or []},
        "target_structure": {"core_etf_pct": [0.40, 0.50],
                             "satellite_max_count": 3, "satellite_max_pct": 0.10},
    }


def _row(name, value, decision="", broken=False):
    return {"symbol": "000000", "name": name, "value": value,
            "decision": decision, "broken": broken}


def test_structure_drift_over_allocated():
    """5 只个股+零 ETF：超只数、超单票、ETF 缺口三类动作齐出，且处置排序按严重度。"""
    health = {"rows": [
        _row("翔鹭钨业", 41000, broken=True),
        _row("中国卫通", 33000, decision="DROP"),
        _row("东材科技", 24000, broken=True),
        _row("雪人集团", 14000, decision="REDUCE"),
        _row("湖南白银", 7500, decision="HOLD"),
    ]}
    s = portfolio.structure_drift(cfg=_cfg(cash=80000), health=health)
    assert not s["compliant"]
    assert abs(s["total"] - 199500) < 1e-6
    acts = "\n".join(s["actions"])
    assert "5 只超上限 3 只" in acts
    assert acts.index("中国卫通(DROP)") < acts.index("雪人集团(REDUCE)")  # 严重度排序
    assert "翔鹭钨业 占总资产 21%" in acts
    assert "核心宽基ETF 建仓至 ≥40%" in acts


def test_structure_drift_compliant():
    health = {"rows": [_row("卫星A", 15000), _row("卫星B", 12000)]}
    s = portfolio.structure_drift(
        cfg=_cfg(cash=63000, etf=[{"position": 30000, "entry_price": 3.0}]),
        health=health)  # 总 18万: ETF 9万(50%) 个股 2.7万 现金 6.3万
    assert s["compliant"] and s["actions"] == []


def test_structure_drift_etf_building_in_progress():
    """ETF 已建一部分但未到 40%：只剩缺口动作。"""
    health = {"rows": [_row("卫星A", 15000)]}
    s = portfolio.structure_drift(
        cfg=_cfg(cash=105000, etf=[{"position": 20000, "entry_price": 3.0}]), health=health)
    # 总 18万: ETF 6万(33%) → 缺口 = 18万×40% - 6万 = 1.2万
    assert len(s["actions"]) == 1
    assert "缺口约 12,000 元" in s["actions"][0]


def test_load_config_exited_requires_exit_date(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(
        "cooldown_days: 7\n"
        "watchlist:\n"
        "  - { symbol: '600000', name: a, state: EXITED, exit_date: '2026-07-17', exit_price: 10.0 }\n"
        "  - { symbol: '600001', name: b, state: EXITED }\n"
        "  - { symbol: '600002', name: c, state: HELD, position: 100, entry_price: 5.0 }\n")
    cfg = load_portfolio_config(p)
    assert [x["symbol"] for x in cfg["exited"]] == ["600000"]  # 无 exit_date 不跟踪
    assert cfg["cooldown_days"] == 7


def _reentry_cfg(exit_date, symbol="600000", cooldown=10):
    return {
        "exited": [{"symbol": symbol, "name": "样本", "exit_date": exit_date, "exit_price": 70.0}],
        "atr_k": 2.0, "cooldown_days": cooldown,
    }


def _setup_reentry(tmp_path, monkeypatch, kline, symbol="600000", decision="HOLD"):
    kdir = tmp_path / "kline"
    kdir.mkdir(exist_ok=True)
    kline.to_csv(kdir / f"{symbol}_daily_qfq.csv", index=False)
    dec = tmp_path / "decisions"
    dec.mkdir(exist_ok=True)
    (dec / "partial_20260715.csv").write_text(f"symbol,decision,dec_risks\n{symbol},{decision},[]\n")
    monkeypatch.setattr(portfolio, "KLINE_DIR", kdir)
    monkeypatch.setattr(portfolio, "DEC_DIR", dec)
    return pd.Timestamp(kline["date"].iloc[-1])


LIGHTS_GREEN = [{"symbol": "sh000852", "name": "中证1000", "state": "risk_on"},
                {"symbol": "sh000688", "name": "科创50", "state": "risk_off"}]


def test_reentry_watch_all_conditions_met(tmp_path, monkeypatch):
    """上涨K线（现价在追踪线上方）+ 绿灯 + 引擎 HOLD + 冷静期已满 → ready。"""
    k = _kline(n=40, close_start=100.0, step=+0.5, spread=1.0)
    today = _setup_reentry(tmp_path, monkeypatch, k)
    exit_date = str(pd.bdate_range(end=today, periods=15)[0].date())  # 14 交易日前
    w = portfolio.reentry_watch(cfg=_reentry_cfg(exit_date), today=today, lights=LIGHTS_GREEN)
    r = w["rows"][0]
    assert r["light_ok"] and r["trend_ok"] and r["engine_ok"]
    assert r["light_name"] == "中证1000"           # 非 688 → 中证1000 灯
    assert r["cooldown_done"] and r["met"] == 3 and r["ready"]


def test_reentry_watch_blocked_by_trend_and_cooldown(tmp_path, monkeypatch):
    """下跌K线未站回趋势线 + 引擎 DROP + 冷静期未满 → 全部拦住。"""
    k = _kline(n=40, close_start=100.0, step=-1.0, spread=2.0)
    today = _setup_reentry(tmp_path, monkeypatch, k, decision="DROP")
    exit_date = str(pd.bdate_range(end=today, periods=4)[0].date())  # 3 交易日前
    w = portfolio.reentry_watch(cfg=_reentry_cfg(exit_date), today=today, lights=LIGHTS_GREEN)
    r = w["rows"][0]
    assert not r["trend_ok"] and r["trail_dist"] < 0
    assert r["engine_ok"] is False and r["decision"] == "DROP"
    assert r["cooldown_elapsed"] == 3 and not r["cooldown_done"]
    assert r["met"] == 1 and not r["ready"]  # 只有灯是绿的


def test_reentry_watch_sector_weak_blocks_engine_condition(tmp_path, monkeypatch):
    """引擎非负面信号但风险项带"板块…弱势" → engine_ok=False（条件③含板块）。"""
    k = _kline(n=40, close_start=100.0, step=+0.5, spread=1.0)
    today = _setup_reentry(tmp_path, monkeypatch, k)
    (tmp_path / "decisions" / "partial_20260715.csv").write_text(
        'symbol,decision,dec_risks\n600000,CONTINUE_WATCH,"[\'板块 [小金属] 弱势（全市场后 11%）\']"\n')
    exit_date = str(pd.bdate_range(end=today, periods=15)[0].date())
    r = portfolio.reentry_watch(cfg=_reentry_cfg(exit_date), today=today,
                                lights=LIGHTS_GREEN)["rows"][0]
    assert r["sector_weak"] and r["engine_ok"] is False and not r["ready"]


def test_reentry_watch_unknown_light_blocks_ready(tmp_path, monkeypatch):
    """688 票映射科创50 灯；lights 缺失 → light_ok=None，宁可不 ready。"""
    k = _kline(n=40, close_start=100.0, step=+0.5, spread=1.0)
    today = _setup_reentry(tmp_path, monkeypatch, k, symbol="688001")
    exit_date = str(pd.bdate_range(end=today, periods=15)[0].date())
    cfg = _reentry_cfg(exit_date, symbol="688001")
    r = portfolio.reentry_watch(cfg=cfg, today=today, lights=LIGHTS_GREEN)["rows"][0]
    assert r["light_name"] == "科创50" and r["light_ok"] is False  # 科创灯是红的
    r = portfolio.reentry_watch(cfg=cfg, today=today, lights=None)["rows"][0]
    assert r["light_ok"] is None and r["met"] == 2 and not r["ready"]


def test_latest_engine_signals_parses_risks(tmp_path, monkeypatch):
    dec = tmp_path / "decisions"
    dec.mkdir()
    (dec / "partial_20260714.csv").write_text("symbol,decision,dec_risks\n600000,HOLD,[]\n")
    (dec / "partial_20260715.csv").write_text(
        'symbol,decision,dec_risks\n600000,DROP,"[\'PE_TTM=296（估值偏高）\', \'板块弱势\', \'第三条\']"\n')
    monkeypatch.setattr(portfolio, "DEC_DIR", dec)
    signals, date = portfolio.latest_engine_signals()
    assert date == "20260715"  # 取最新一期
    assert signals["600000"]["decision"] == "DROP"
    assert len(signals["600000"]["risks"]) == 2  # 只取前两条
