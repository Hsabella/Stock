"""Read-only local view of the Stock project's portfolio, lamps and news."""

import argparse
import csv
import json
import math
import re
import webbrowser
from bisect import bisect, bisect_left
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from html import unescape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from quant.live.portfolio import is_etf, load_portfolio_config, stop_lines


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("terminal")
BRIEF = ROOT / "results/brief"
MARKET_TZ = ZoneInfo("Asia/Shanghai")
INDEXES = {"sh000300": "沪深300", "sh000852": "中证1000", "sh000688": "科创50"}
BAR_FIELDS = ["open", "high", "low", "close", "volume"]
THEMES = {
    "市场政策": ("资本市场稳市", "平准基金", "资本市场投融资", "中长期资金入市"),
    "科技和半导体": ("半导体", "算力", "光模块", "覆铜板", "光通信", "芯片", "PCB"),
    "电力和电网": ("电网", "电力", "风电", "电价", "储能"),
    "资源和材料": ("铜价", "沪铜", "钴", "钼", "钾肥", "碳酸锂", "锂矿"),
    "红利和传统行业": ("高股息", "红利", "煤炭", "国有大行", "银行股", "石油"),
}
X_POST = {
    "url": "https://x.com/daweifs/status/2121436374813569321",
    "author": "南亥 @daweifs", "date": "2026-11-14",
    "text": "帖子讨论美股看盘网页，把行情、新闻和盘前盘后的入口集中起来。",
}
CHECKED_NEWS = (
    ("2026-11-27", "市场政策", "国常会研究资本市场投融资改革", "午间报道提出强化稳市机制、优化中央汇金稳定市场功能，并推动分红回购和中长期资金入市。", "https://finance.eastmoney.com/a/202611273941564045.html"),
    ("2026-11-27", "东材科技", "国风新材购买金张科技股权获准注册", "东材科技是交易对手方，批复同意国风新材向其发行约 854 万股。观察交易实施公告。", "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12725240&stockid=000859"),
    ("2026-11-26", "盐津铺子", "投资者交流纪要", "跟踪魔芋增速和进口鳕鱼原料成本。", "https://static.cninfo.com.cn/finalpage/2026-11-26/1225689327.PDF"),
    ("2026-11-26", "天齐锂业", "董事会审议股份回购方案", "回购金额拟为 4 至 8 亿元，观察锂板块相对强弱。", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12726774&stockid=002466"),
)


def records(path):
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)


def trade_days():
    return sorted({date.fromisoformat(row["trade_date"]) for row in records(ROOT / "cache/trade_calendar.csv")})


def session(now, calendar):
    today = now.date()
    clock = now.time()
    if not calendar or today > calendar[-1]:
        phase = "日历待更新"
    elif today not in calendar:
        phase = "休市"
    elif clock < time(9, 30):
        phase = "开盘前"
    elif clock < time(11, 30):
        phase = "上午交易"
    elif clock < time(13):
        phase = "午间休市"
    elif clock < time(15):
        phase = "下午交易"
    else:
        phase = "已收盘"
    previous = bisect_left(calendar, today)
    completed = bisect(calendar, today) if phase == "已收盘" else previous
    report_day = bisect(calendar, today) if today in calendar and clock >= time(9) else previous
    return {
        "date": today.isoformat(), "phase": phase,
        "previous": calendar[previous - 1].isoformat() if previous else None,
        "closed": calendar[completed - 1].isoformat() if completed else None,
        "reportDay": calendar[report_day - 1].isoformat() if report_day else None,
        "newsSince": f"{calendar[previous - 1]} 15:00:00" if previous else now.strftime("%Y-%m-%d 00:00:00"),
    }


@lru_cache(maxsize=160)
def closed_bars(path, file_version, last_session):
    fields = ["date", *BAR_FIELDS]
    frame = pd.read_csv(path, usecols=fields, parse_dates=["date"])
    frame[BAR_FIELDS] = frame[BAR_FIELDS].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=fields)
    frame = frame[(frame["date"] <= pd.Timestamp(last_session)) & (frame["close"] > 0)]
    return frame.sort_values("date").drop_duplicates("date", keep="last")


def candle_structure(frame):
    """Conservative recent swing review. The last two bars may still be forming a retest."""
    if len(frame) < 25:
        return {"label": "K 线不足", "points": []}
    recent = frame.tail(25).reset_index(drop=True)
    close = recent["close"].iloc[-1]
    fresh_low = recent["low"].iloc[-2:].min() < recent["low"].iloc[3:-2].min()
    weekly = frame.set_index("date")["close"].resample("W-FRI").last().dropna()
    weekly = weekly[weekly.index <= frame["date"].iloc[-1]]
    weekly_change = float(weekly.iloc[-1] / weekly.iloc[-5] - 1) if len(weekly) >= 5 else None
    result = {"label": "再创新低" if fresh_low else "观察平台", "points": [], "weeklyChange": weekly_change}

    a = int(recent["low"].iloc[5:-2].idxmin())
    if a >= len(recent) - 3:
        return result
    b = int(recent["high"].iloc[a + 1:-1].idxmax())
    c = int(recent["low"].iloc[b + 1:].idxmin())
    a_low, b_high, c_low = recent.loc[a, "low"], recent.loc[b, "high"], recent.loc[c, "low"]
    if b_high <= a_low or c_low >= b_high:
        return result
    bounce_volume = recent.loc[a + 1:b, "volume"].median()
    retest_volume = recent.loc[b + 1:c, "volume"].median()
    result.update({
        "points": [{"key": key, "date": str(recent.loc[i, "date"].date()), "price": float(value)}
                   for key, i, value in [("A", a, a_low), ("B", b, b_high), ("C", c, c_low)]],
        "volumeRatio": float(retest_volume / bounce_volume) if bounce_volume else None,
    })
    if fresh_low or c_low < a_low:
        result["label"] = "再创新低"
    elif close > b_high:
        last_volume = recent["volume"].iloc[-1]
        normal_volume = recent["volume"].iloc[-21:-1].median()
        result["label"] = "放量站上反弹高点" if normal_volume and last_volume > normal_volume else "站上反弹高点"
    elif c == len(recent) - 1:
        result["label"] = "缩量回踩待确认" if bounce_volume and retest_volume < bounce_volume else "回踩进行中"
    elif bounce_volume and retest_volume < bounce_volume:
        result["label"] = "缩量守住前低"
    else:
        result["label"] = "守住前低，量能待确认"
    return result


def candles(symbol, closed):
    path = ROOT / "cache/kline" / f"{symbol}_daily_qfq.csv"
    if not closed or not path.is_file():
        return pd.DataFrame()
    try:
        return closed_bars(path, path.stat().st_mtime_ns, closed)
    except (OSError, ValueError):
        return pd.DataFrame()


def stock_data(symbol, closed):
    frame = candles(symbol, closed)
    if frame.empty:
        return frame, {"close": None, "closeDate": None, "bars": [], "structure": candle_structure(frame)}
    last = frame.iloc[-1]
    chart = frame.tail(120).copy()
    chart["date"] = chart["date"].dt.strftime("%Y-%m-%d")
    return frame, {
        "close": float(last["close"]), "closeDate": str(last["date"].date()),
        "bars": chart[["date", *BAR_FIELDS]].to_numpy().tolist(),
        "structure": candle_structure(frame),
    }


def parse_lamps(report, report_date, market):
    if not report_date:
        return []
    rows = []
    for name, symbol, signals, close, drawdown, rsi, notes in re.findall(
        r"^  🟢 (.+?) (\d{6}) (.+?) 收盘([\d.]+) 距高(-?\d+)% RSI(\d+)(.*)$", report, re.M
    ):
        setups = []
        for kind, day, timing in re.findall(r"(L[16][^+(]+)\(信号(\d\d-\d\d)·([^)]*)\)", signals):
            signal = date.fromisoformat(f"{report_date.year}-{day}")
            if signal > report_date:
                signal = date.fromisoformat(f"{report_date.year - 1}-{day}")
            scheduled = "今日开盘" in timing
            setups.append({"type": kind, "date": signal.isoformat(), "timing": timing, "scheduled": scheduled})
        if setups:
            rows.append({
                "symbol": symbol, "name": name, "setups": setups, "signalClose": float(close),
                "drawdown": int(drawdown), "rsi": int(rsi), "notes": notes.strip(),
                "today": report_date.isoformat() == market["date"] and any(item["scheduled"] for item in setups),
                "scheduled": any(item["scheduled"] for item in setups),
            })
    return rows


def read_news(market, holdings, lamps, now):
    latest = None
    items = {theme: [] for theme in ("持仓", "抄底灯", *THEMES)}
    seen = set()
    until = now.strftime("%Y-%m-%d %H:%M:%S")
    held_names = [item["name"] for item in holdings if not item["core"]]
    lamp_names = [item["name"] for item in lamps if item["name"] not in held_names]
    for row in records(ROOT / "cache/news/history.csv"):
        stamp = row.get("ts", "")
        if not market["newsSince"] <= stamp <= until:
            continue
        latest = max(latest or stamp, stamp)
        body = " ".join((row.get("content") or "").split())
        title = " ".join((row.get("title") or "").split()) or body[:88]
        if len(title) < 4:
            continue
        signature = re.sub(r"[\W_]", "", title)[:46]
        if signature in seen:
            continue
        seen.add(signature)
        content = f"{title} {body}"
        for theme, words in [("持仓", held_names), ("抄底灯", lamp_names), *THEMES.items()]:
            matches = [word for word in words if word.lower() in content.lower()]
            if matches:
                items[theme].append({"time": stamp, "title": title[:140], "body": body[:180],
                                     "source": row.get("source"), "matches": matches[:3]})
                break
    for theme in items:
        items[theme] = sorted(items[theme], key=lambda item: item["time"], reverse=True)[:8]
    relevant = {"市场政策", *held_names, *lamp_names}
    checked = [{"date": day, "name": name, "title": title, "body": body, "url": url}
               for day, name, title, body, url in CHECKED_NEWS
               if name in relevant and 0 <= (now.date() - date.fromisoformat(day)).days <= 3]
    return {"latest": latest, "since": market["newsSince"], "topics": items, "checked": checked, "x": X_POST}


def dashboard(now):
    calendar = trade_days()
    market = session(now, calendar)
    cfg = load_portfolio_config()
    account, target = cfg["account"], cfg["target_structure"]
    report_path = BRIEF / "latest.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    report_match = re.match(r"\[(\d{4}-\d\d-\d\d)\]", report)
    report_date = date.fromisoformat(report_match[1]) if report_match else None
    report_day = report_match[1] if report_match else None
    signals = next((row for row in records(BRIEF / "signal_log.csv") if row["date"] == report_day), {})
    lights = {row["light"]: row["state"] for row in records(BRIEF / "lights_log.csv") if row["date"] == report_day}
    items, charts = [], {}
    for core, instruments in [(False, cfg["held"]), (True, account.get("etf", []))]:
        for holding in instruments:
            symbol = str(holding["symbol"]).zfill(6)
            frame, chart = stock_data(symbol, market["closed"])
            charts[symbol] = chart
            entry = float(holding["entry_price"])
            stop = None
            if not core and len(frame) >= 15 and entry > 0:
                stop = stop_lines(frame, entry, cfg["hard_stop_pct"], cfg["atr_k"], holding.get("entry_date"), holding.get("entry_arm", "right"))
            entry_date = str(holding.get("entry_date") or "")
            morning_hit = re.search(rf"^  [^\n]* {symbol} [^\n]*止损线 ([\d.]+) 已破", report, re.M)
            if report_day and entry_date >= report_day:
                morning_hit = None
            items.append({
                "symbol": symbol, "name": holding.get("name", symbol), "core": core, "etf": is_etf(symbol),
                "shares": float(holding["position"]), "entry": entry,
                "line": float(stop["stop_line"]) if stop else None,
                "degraded": stop["degrade_reason"] if stop else None,
                "broken": bool(stop and stop["broken"]), "morningHit": float(morning_hit[1]) if morning_hit else None,
            })
    held_symbols = {item["symbol"] for item in items}
    previous_sales = {str(item["symbol"]).zfill(6) for item in cfg["exited"]}
    lamps = parse_lamps(report, report_date, market)
    for lamp in lamps:
        symbol = lamp["symbol"]
        if symbol not in charts:
            _, charts[symbol] = stock_data(symbol, market["closed"])
        lamp["held"] = symbol in held_symbols
        lamp["exited"] = symbol in previous_sales or "EXITED" in lamp["notes"]
        lamp["overBudget"] = "超预算" in lamp["notes"]
    limit = signals.get("exposure")
    return {
        "market": market, "reportDate": report_day, "reportFresh": bool(report_day and report_day == market["reportDay"]),
        "mainState": signals.get("state"), "limit": float(limit) if limit else None, "lights": lights,
        "indices": INDEXES, "holdings": items, "lamps": lamps, "charts": charts,
        "cash": account.get("cash"), "cashDate": account.get("as_of"),
        "satelliteLimit": int(target.get("satellite_max_count", 3)),
        "news": read_news(market, items, lamps, now),
    }


def provider_symbol(symbol):
    return f"{'sh' if symbol.startswith(('5', '6')) else 'sz'}{symbol}" if symbol.isdecimal() else symbol


def quote_number(raw):
    try:
        number = float(raw)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def quote_batch(symbols):
    codes = {provider_symbol(symbol): symbol for symbol in symbols}
    response = requests.get(f"https://qt.gtimg.cn/q={','.join(codes)}", timeout=(3, 4))
    response.raise_for_status()
    quotes = {}
    for code, payload in re.findall(r'v_(\w+)="([^"\r\n]+)"', response.content.decode("gb18030", errors="replace")):
        fields = payload.split("~")
        if code not in codes or len(fields) <= 36 or not re.fullmatch(r"\d{14}", fields[30]):
            continue
        price = quote_number(fields[3])
        if not price or price < 0:
            continue
        timestamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S")
        quotes[codes[code]] = {
            "price": price, "prev": quote_number(fields[4]), "open": quote_number(fields[5]),
            "change": quote_number(fields[32]), "high": quote_number(fields[33]), "low": quote_number(fields[34]),
            "timestamp": timestamp.isoformat(),
        }
    return quotes


@lru_cache(maxsize=8)
def recent_quotes(symbols, interval):
    try:
        return {"quotes": quote_batch(symbols), "source": "腾讯行情"}
    except (requests.RequestException, ValueError):
        return {"quotes": {}, "source": "腾讯行情", "error": "行情暂未连通，显示本地已收盘 K 线"}


def x_lookup(url):
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    post = re.fullmatch(r"/([a-zA-Z0-9_]{1,15})/status/(\d{8,22})/?", parts.path)
    if parts.scheme != "https" or parts.netloc.lower() not in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"} or not post:
        return None
    canonical = f"https://x.com/{post[1]}/status/{post[2]}"
    try:
        response = requests.get("https://publish.twitter.com/oembed", params={"url": canonical, "omit_script": "true"}, timeout=(3, 5))
        response.raise_for_status()
        result = response.json()
        paragraphs = re.findall(r"<p\b[^>]*>(.*?)</p>", result.get("html", ""), re.S)
        body = " ".join(unescape(re.sub(r"<[^>]+>", " ", paragraph)).strip() for paragraph in paragraphs)
        return {"url": canonical, "author": result.get("author_name"), "text": body[:450]}
    except (requests.RequestException, ValueError):
        return {"url": canonical, "error": "X 暂未返回公开摘要，请打开原帖核对。"}


class TerminalHandler(BaseHTTPRequestHandler):
    def reply(self, status, body, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False).encode()
        self.reply(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            self.send_error(403)
            return
        parts = urlsplit(self.path)
        params = parse_qs(parts.query)
        files = {"/": ("index.html", "text/html"), "/styles.css": ("styles.css", "text/css"), "/app.js": ("app.js", "text/javascript")}
        if parts.path in files:
            filename, kind = files[parts.path]
            self.reply(200, (ASSETS / filename).read_bytes(), f"{kind}; charset=utf-8")
        elif parts.path == "/api/dashboard":
            try:
                self.json(dashboard(datetime.now(MARKET_TZ)))
            except (OSError, ValueError, KeyError) as error:
                self.log_error("读取本地看盘数据失败: %s", error)
                self.json({"error": "读取本地看盘数据失败，请检查晨报、行情缓存和持仓配置。"}, 500)
        elif parts.path == "/api/quotes":
            symbols = sorted(set(params.get("symbols", [""])[0].split(",")))
            if len(symbols) > 70 or any(not re.fullmatch(r"\d{6}", symbol) and symbol not in INDEXES for symbol in symbols):
                self.json({"error": "行情代码格式无效"}, 400)
            else:
                self.json(recent_quotes(tuple(symbols), int(monotonic() / 12)))
        elif parts.path == "/api/x":
            post = x_lookup(params.get("url", [""])[0])
            self.json(post or {"error": "请输入公开的 X 或 Twitter 帖子链接"}, 200 if post else 400)
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open", action="store_true", help="open the terminal in the default browser")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), TerminalHandler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"看盘台 {url}", flush=True)
        if args.open:
            webbrowser.open_new_tab(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
