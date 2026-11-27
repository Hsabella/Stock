"use strict";

const $ = (id) => document.getElementById(id);
const app = {data: null, quotes: {}, quoteError: "", selected: null, range: 60, tab: "holdings", filter: "today", topic: null, crosshair: null};
const activePhases = new Set(["上午交易", "下午交易"]);
const stableStructures = new Set(["缩量守住前低", "站上反弹高点", "放量站上反弹高点"]);
const fmt = (n, digits = 2) => Number.isFinite(n) ? n.toLocaleString("zh-CN", {minimumFractionDigits: digits, maximumFractionDigits: digits}) : "—";
const pct = (n, digits = 2) => Number.isFinite(n) ? `${n > 0 ? "+" : ""}${n.toFixed(digits)}%` : "—";
const moveColor = (value) => value > 0 ? "up" : value < 0 ? "down" : "muted";
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
const pill = (label, color = "") => `<span class="pill ${color}">${esc(label)}</span>`;
const price = (n, symbol) => fmt(n, /^(15|5)/.test(symbol) ? 3 : 2);
const clock = (value) => value ? value.slice(11, 19) : "—";
const titleCaseState = (value) => value === "risk_off" ? "防守" : value === "risk_on" ? "进攻" : "待更新";

async function get(url) {
  const response = await fetch(url, {cache: "no-store"});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "读取失败");
  return result;
}

function currentQuote(symbol) {
  const q = app.quotes[symbol];
  if (!q) return null;
  const {date, phase} = app.data.market;
  if (!q.timestamp.startsWith(date) || phase === "开盘前" || phase === "休市" || phase === "日历待更新") return null;
  const elapsed = (Date.now() - Date.parse(`${q.timestamp}+08:00`)) / 1000;
  if (elapsed < -20) return null;
  if (activePhases.has(phase) && elapsed > 120) return null;
  if (phase === "午间休市" && clock(q.timestamp) < "11:29:00") return null;
  if (phase === "已收盘" && clock(q.timestamp) < "15:00:00") return null;
  return q;
}

function instrument(symbol) {
  return app.data.holdings.find((item) => item.symbol === symbol) || app.data.lamps.find((item) => item.symbol === symbol);
}

function valuation(item) {
  const q = currentQuote(item.symbol);
  const chart = app.data.charts[item.symbol];
  if (q) return {value: q.price, date: q.timestamp.slice(0, 10), live: true};
  // A current-session quote can cover a stale local cache with its previous close.
  const raw = app.quotes[item.symbol];
  const last = app.data.market.closed;
  if (raw?.timestamp.startsWith(app.data.market.date) && raw.prev > 0 && last && last < app.data.market.date && chart.closeDate !== last) {
    return {value: raw.prev, date: last, live: false};
  }
  return {value: chart.close ?? item.signalClose, date: chart.closeDate, live: false};
}

function stopStatus(item) {
  const chart = app.data.charts[item.symbol];
  const q = currentQuote(item.symbol);
  const {closed, previous, phase} = app.data.market;
  const sameDay = Boolean(chart.closeDate && chart.closeDate === closed);
  // The engine calculates at full precision. Intraday comparisons use the displayed quote tick.
  const line = item.line === null ? null : Number(price(item.line, item.symbol).replaceAll(",", ""));
  const displayLine = price(line, item.symbol);
  const closeStatus = {tag: "收盘破线", color: "red", rank: 0};
  if (item.core) return {tag: "核心仓结构跟踪", color: "", rank: 6, detail: "宽基 ETF 在账户结构中跟踪。"};
  if (item.broken && sameDay) {
    return {...closeStatus, detail: `${chart.closeDate.slice(5)} 收 ${price(chart.close, item.symbol)}，低于止损 ${displayLine}。按纪律优先复核减仓或退出。`};
  }
  if (item.morningHit !== null) {
    return {...closeStatus, detail: `晨报记录已跌破 ${price(item.morningHit, item.symbol)}，持仓仍显示在册。先核对券商成交和仓位。`};
  }
  if (phase === "已收盘" && !sameDay && line !== null && chart.closeDate === previous && q?.price < item.line) {
    return {...closeStatus, detail: `今日收盘报价 ${price(q.price, item.symbol)}，低于上一交易日止损线 ${displayLine}。日 K 待更新，先核对券商成交和仓位。`};
  }
  if (line === null || !sameDay) return {tag: "日线待更新", color: "yellow", rank: 2, detail: "缺少最近已收盘 K 线或止损锚点，先核对券商行情。"};
  if (q?.price < line) {
    return {tag: "现价低于止损", color: "yellow", rank: 1, detail: `最新 ${price(q.price, item.symbol)}，低于止损 ${displayLine}。盘中告警，收盘复核。`};
  }
  if (q?.low > 0 && q.low <= line) {
    return {tag: "盘中触线后收回", color: "yellow", rank: 2, detail: `今日低点 ${price(q.low, item.symbol)}，止损 ${displayLine}，最新 ${price(q.price, item.symbol)}。收盘重点复核。`};
  }
  const reference = q?.price ?? chart.close;
  if (reference / line - 1 <= .02) {
    return {tag: "距离止损 ≤2%", color: "yellow", rank: 3, detail: `${q ? "最新价" : "最近收盘"}距止损 ${displayLine} 仅 ${pct((reference / line - 1) * 100)}。留意走弱。`};
  }
  return {tag: "止损未触发", color: "", rank: 5, detail: `收盘纪律线 ${displayLine}，继续跟踪。`};
}

function lampStatus(item) {
  const {reportFresh, market, lights, satelliteLimit, holdings} = app.data;
  if (item.held) return ["持仓中，按止损纪律", ""];
  if (item.exited) return ["已退出，查回补条件", ""];
  if (!reportFresh) return ["晨报待更新", "yellow"];
  if (!item.today) return [item.scheduled ? "历史入场窗" : "原入场日已过", ""];
  if (item.overBudget) return ["最小手数超预算", "yellow"];
  if (market.phase !== "开盘前") return ["开盘窗已过", ""];
  if (holdings.filter((holding) => !holding.core).length >= satelliteLimit) return ["卫星名额已满", "yellow"];
  if (lights.sh000852 !== "risk_off") return ["结构灯待核对", "yellow"];
  return ["开盘前核对执行条件", "green"];
}

function renderMetrics() {
  const {holdings, lamps, reportDate, reportFresh, limit, mainState, cash, cashDate, satelliteLimit, market} = app.data;
  const satellite = holdings.filter((item) => !item.core).length;
  const valued = holdings.map((item) => ({...valuation(item), core: item.core, shares: item.shares}));
  const complete = Number.isFinite(cash) && valued.every((item) => Number.isFinite(item.value));
  const securityValue = valued.reduce((sum, item) => sum + (item.value || 0) * item.shares, 0);
  const total = complete ? cash + securityValue : null;
  const exposure = total > 0 ? `${fmt(securityValue / total * 100, 1)}%` : "—";
  const cashShare = total > 0 ? `${fmt(cash / total * 100, 1)}%` : "—";
  const staleValues = valued.some((item) => !item.date || item.date < market.closed);
  const freshLamps = lamps.filter((item) => item.today).length;
  const cards = [
    ["大盘主灯", reportFresh ? titleCaseState(mainState) : "待更新", reportFresh && mainState === "risk_off" ? "rose" : "muted", `仓位上限 ${limit === null ? "—" : `${limit * 100}%`} · 晨报 ${reportDate || "—"}`],
    ["账户证券仓位", exposure, "", `现金 ${cashShare} · ${staleValues ? "含旧收盘估值" : "估算"}${cashDate ? ` · 现金 ${cashDate.slice(5)}` : ""}`],
    ["卫星持仓 / 上限", `${satellite} / ${satelliteLimit}`, satellite > satelliteLimit ? "amber" : "", satellite > satelliteLimit ? `超出 ${satellite - satelliteLimit} 只` : satellite === satelliteLimit ? "名额用满" : `还有 ${satelliteLimit - satellite} 个名额`],
    ["晨报亮灯", lamps.length, "mint", `${freshLamps} 只${market.phase === "开盘前" ? "等今日开盘" : "对应今日开盘"}`],
  ];
  $("metrics").innerHTML = cards.map(([label, value, color, note]) => `<div class="metric"><p>${label}</p><strong class="${color}">${value}</strong><small>${esc(note)}</small></div>`).join("");
}

function renderFeed() {
  const {market, indices, lights, mainState, reportFresh, holdings} = app.data;
  $("market-time").textContent = `${market.date} · ${market.phase}`;
  const quotes = [...Object.keys(indices), ...holdings.map((item) => item.symbol)].map(currentQuote).filter(Boolean);
  const latest = quotes.map((q) => q.timestamp).sort().at(-1);
  const feed = $("feed");
  feed.className = latest ? "feed live" : "feed warn";
  feed.textContent = latest ? `腾讯行情 ${clock(latest)} · ${market.phase === "已收盘" ? "收盘价" : "报价快照"}` : app.quoteError || "使用本地已收盘 K 线";
  $("indices").innerHTML = Object.entries(indices).map(([symbol, name]) => {
    const q = currentQuote(symbol);
    const light = reportFresh ? (symbol === "sh000300" ? mainState : lights[symbol]) : null;
    const state = light === "risk_off" ? ["红灯", "rose"] : light === "risk_on" ? ["绿灯", "mint"] : ["灯待更新", "muted"];
    return `<div class="index"><b>${name}</b><span>${fmt(q?.price)}</span><span class="${moveColor(q?.change)}">${pct(q?.change)}</span><span class="dot ${state[1]}">${state[0]}</span></div>`;
  }).join("");
}

function renderActions() {
  const {holdings, satelliteLimit, reportFresh, mainState, lamps, market} = app.data;
  const satellites = holdings.filter((item) => !item.core);
  const risks = satellites.map((item) => ({...item, ...stopStatus(item)})).sort((a, b) => a.rank - b.rank);
  const flagged = risks.filter((item) => item.rank < 5);
  const overdue = flagged.filter((item) => item.rank === 0).length;
  const messages = (flagged.length ? flagged : risks.slice(0, 1)).map((item) => `
    <button class="action ${item.color}" type="button" data-symbol="${item.symbol}">
      <span class="action-title"><strong>${esc(item.name)}</strong>${pill(item.tag, item.color)}</span><p>${esc(item.detail)}</p>
    </button>`);
  if (satellites.length >= satelliteLimit) {
    const detail = `卫星持仓 ${satellites.length} / ${satelliteLimit}。有开盘信号也先处理存量风险。`;
    messages.push(`<div class="action"><span class="action-title"><strong>新仓等待</strong>${pill("名额已满")}</span><p>${detail}</p></div>`);
  } else if (!reportFresh) {
    messages.push(`<div class="action"><span class="action-title"><strong>更新晨报</strong>${pill("数据待核对", "yellow")}</span><p>核对当日主灯、结构灯和持仓后，再决定是否处理新信号。</p></div>`);
  } else {
    const opening = lamps.filter((lamp) => lamp.today && lampStatus(lamp)[1] === "green").length;
    const detail = market.phase === "开盘前" && opening ? `${opening} 只待核对。开盘确认涨跌停、最小手数和风险预算。` : "查看筛选后的候选和 K 线结构，保留既定入场时间。";
    messages.push(`<div class="action"><span class="action-title"><strong>复核抄底灯</strong>${pill("观察")}</span><p>${detail}</p></div>`);
  }
  $("risk-count").textContent = overdue ? `${overdue} 只收盘破线` : flagged.length ? `${flagged.length} 只待复核` : (reportFresh && mainState === "risk_off") ? "防守" : "跟踪中";
  $("risk-count").className = `pill ${overdue ? "red" : flagged.length ? "yellow" : ""}`;
  $("actions").innerHTML = messages.join("");
}

function codeCell(item, sublabel = "") {
  return `<button type="button" class="symbol" data-symbol="${item.symbol}" aria-pressed="${app.selected === item.symbol}">${esc(item.name)}<small>${item.symbol}${sublabel ? ` · ${sublabel}` : ""}</small></button>`;
}

function holdingRows() {
  const headings = ["股票", "最新价 / 日期", "持有", "摊薄成本", "浮动盈亏", "棘轮止损", "复核状态"];
  const holdings = app.data.holdings.map((item) => ({...item, status: stopStatus(item)})).sort((a, b) => a.status.rank - b.status.rank);
  const rows = holdings.map((item) => {
    const {value, date, live} = valuation(item);
    const pnl = Number.isFinite(value) && item.entry > 0 ? (value / item.entry - 1) * 100 : null;
    const line = item.morningHit !== null && !item.line ? item.morningHit : item.line;
    return `<tr class="${app.selected === item.symbol ? "selected" : ""}"><td>${codeCell(item, item.core ? "核心 ETF" : "卫星")}</td><td>${price(value, item.symbol)}<small>${date?.slice(5) || "—"} ${live && app.data.market.phase !== "已收盘" ? "盘中" : "收盘"}</small></td><td>${fmt(item.shares, 0)} 股</td><td>${price(item.entry, item.symbol)}</td><td class="${moveColor(pnl)}">${pct(pnl)}</td><td>${price(line, item.symbol)}${item.degraded ? `<small>仅硬止损 · ${esc(item.degraded)}</small>` : ""}</td><td>${pill(item.status.tag, item.status.color)}</td></tr>`;
  });
  return {headings, rows, caption: "当前配置持仓 · 点股票名看日 K 和止损线。"};
}

function isStable(item) {
  const chart = app.data.charts[item.symbol];
  return chart.closeDate === app.data.market.closed && stableStructures.has(chart.structure.label);
}

function retestCandidate(item) {
  if (item.held || item.exited) return false;
  const chart = app.data.charts[item.symbol];
  if (chart.closeDate !== app.data.market.closed) return false;
  const low = currentQuote(item.symbol)?.low;
  const support = chart.structure.points.find((point) => point.key === "A");
  if (support && low > 0 && low < support.price) return false;
  return isStable(item) || chart.structure.label === "缩量回踩待确认";
}

function lampRows() {
  const {lamps, charts, reportDate} = app.data;
  const search = $("lamp-search").value.trim().toLowerCase();
  const filtered = lamps.filter((item) => {
    if (search) return `${item.name} ${item.symbol}`.toLowerCase().includes(search);
    if (app.filter === "today") return item.today;
    if (app.filter === "structure") return retestCandidate(item);
    return true;
  });
  const headings = ["股票", "亮灯 / 日期", "信号表收盘", "距高 / RSI", "日 K 结构", "现阶段"];
  const rows = filtered.map((item) => {
    const [status, color] = lampStatus(item);
    const chart = charts[item.symbol];
    const stale = chart.closeDate && chart.closeDate !== app.data.market.closed;
    const setups = item.setups.map((setup) => `${esc(setup.type)} · ${setup.date.slice(5)}`).join("<br>");
    const window = item.setups.map((setup) => setup.scheduled ? `入场窗 ${reportDate.slice(5)} 开盘` : esc(setup.timing)).at(-1);
    return `<tr class="${app.selected === item.symbol ? "selected" : ""}"><td>${codeCell(item)}</td><td>${setups}<small>${window}</small></td><td>${price(item.signalClose, item.symbol)}</td><td>${item.drawdown}% / ${item.rsi}</td><td>${pill(stale ? "日 K 待更新" : chart.structure.label, !stale && isStable(item) ? "green" : "")}</td><td>${pill(status, color)}${item.overBudget ? `<small>${esc(item.notes.replace(/^⚠️\s*/, ""))}</small>` : ""}</td></tr>`;
  });
  return {headings, rows, caption: `${filtered.length} / ${lamps.length} 只 · 晨报 ${reportDate || "—"} · 结构筛选基于最近已收盘 K 线。`};
}

function renderTable() {
  $("lamp-controls").hidden = app.tab !== "lamps";
  const {headings, rows, caption} = app.tab === "holdings" ? holdingRows() : lampRows();
  $("table-caption").textContent = caption;
  $("table-head").innerHTML = `<tr>${headings.map((heading) => `<th scope="col">${heading}</th>`).join("")}</tr>`;
  $("table-body").innerHTML = rows.join("") || `<tr><td class="empty" colspan="${headings.length}">当前筛选没有股票。</td></tr>`;
}

function renderChart() {
  const item = instrument(app.selected);
  if (!item) {
    $("chart-title").textContent = "暂无可查看股票";
    $("chart-quote").textContent = "";
    $("chart-bottom").textContent = "";
    drawChart();
    return;
  }
  const chart = app.data.charts[item.symbol];
  const q = currentQuote(item.symbol);
  const hold = app.data.holdings.find((holding) => holding.symbol === item.symbol);
  const {value, date, live} = valuation(item);
  const stale = chart.closeDate && chart.closeDate !== app.data.market.closed;
  $("chart-type").textContent = hold ? (hold.core ? "核心 ETF 日线" : "持仓日线") : "抄底灯日线";
  $("chart-title").innerHTML = `${esc(item.name)}<span>${item.symbol}</span>`;
  $("chart-quote").innerHTML = `<strong>${price(value, item.symbol)}</strong>${q ? `<span class="${moveColor(q.change)}">${pct(q.change)}</span>` : ""}<span>${live ? `腾讯 ${clock(q.timestamp)}` : date ? `${date} 收盘` : "参考报价"}</span><span>日 K 至 ${chart.closeDate || "—"}${stale ? " · 待更新" : ""}</span>`;
  const structure = chart.structure;
  const labels = {A: "前低", B: "反弹高点", C: "回踩"};
  const values = structure.points.map(({key, price: point}) => `<span>${labels[key]} <b>${price(point, item.symbol)}</b></span>`);
  const support = structure.points.find((point) => point.key === "A");
  if (support && q?.low > 0 && q.low < support.price) values.push(pill("今日低点已破前低", "yellow"));
  if (Number.isFinite(structure.volumeRatio)) values.push(`<span>回踩量 / 反弹量 <b>${fmt(structure.volumeRatio, 2)}×</b></span>`);
  if (Number.isFinite(structure.weeklyChange)) values.push(`<span>近四周 <b>${pct(structure.weeklyChange * 100, 1)}</b></span>`);
  $("chart-bottom").innerHTML = `${pill(stale ? "K 线待更新" : structure.label, !stale && stableStructures.has(structure.label) ? "green" : "")}${values.join("")}`;
  drawChart();
}

function drawChart() {
  const canvas = $("candles");
  const wrap = $("chart-wrap");
  const chart = app.data?.charts[app.selected];
  const selection = chart?.bars.slice(-app.range) ?? [];
  const scale = window.devicePixelRatio || 1;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  if (w < 100 || h < 100) return;
  if (canvas.width !== Math.round(w * scale) || canvas.height !== Math.round(h * scale)) {
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, w, h);
  $("chart-empty").hidden = selection.length > 0;
  $("chart-tooltip").hidden = true;
  if (!selection.length) return;
  const holding = app.data.holdings.find((item) => item.symbol === app.selected);
  const quote = currentQuote(app.selected);
  const ratio = w > 500 ? 62 : 55;
  const left = 13, right = w - ratio, top = 15, floor = h * .72, volumeBase = h - 25;
  const b = chart.structure.points.find((point) => point.key === "B");
  const reference = [
    {level: holding?.line ?? holding?.morningHit, color: "#ff818d", label: "止损"},
    {level: b?.price, color: "#9ab6ed", label: "B"},
    {level: quote?.price, color: "#e7bb76", label: "最新"},
  ].filter((line) => Number.isFinite(line.level));
  const extremes = [...selection.flatMap((row) => [row[2], row[3]]), ...reference.map((line) => line.level)];
  const hi = Math.max(...extremes), lo = Math.min(...extremes);
  const padding = (hi - lo || hi * .1) * .12;
  const high = hi + padding, low = lo - padding;
  const maxVolume = Math.max(...selection.map((row) => row[5]), 1);
  const step = (right - left) / selection.length;
  const x = (i) => left + (i + .5) * step;
  const y = (value) => top + (high - value) * (floor - top) / (high - low);
  const line = (x1, y1, x2, y2, color, dashes = []) => {
    ctx.strokeStyle = color;
    ctx.setLineDash(dashes);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const value = high - (high - low) * i / 4;
    line(left, y(value), right, y(value), "#303a47");
    ctx.fillStyle = "#8b9aaf";
    ctx.fillText(price(value, app.selected), right + 6, y(value));
  }
  const monthMarks = new Set([0, Math.floor(selection.length / 3), Math.floor(selection.length * 2 / 3), selection.length - 1]);
  for (const i of monthMarks) {
    line(x(i), top, x(i), volumeBase, "#252e38");
    ctx.fillStyle = "#8b9aaf";
    ctx.fillText(selection[i][0].slice(5), Math.min(right - 24, Math.max(left, x(i) - 12)), h - 10);
  }
  line(left, volumeBase, right, volumeBase, "#374352");
  for (const [i, [, open, highPrice, lowPrice, close, volume]] of selection.entries()) {
    const up = close >= open;
    const color = up ? "#f27e88" : "#69d5b3";
    const width = Math.max(1, Math.min(step * .64, 14));
    line(x(i), y(highPrice), x(i), y(lowPrice), color);
    ctx.fillStyle = color;
    ctx.fillRect(x(i) - width / 2, y(Math.max(open, close)), width, Math.max(1, Math.abs(y(close) - y(open))));
    const volumeHeight = volume / maxVolume * (volumeBase - floor - 9);
    ctx.fillStyle = up ? "#f27e8866" : "#69d5b366";
    ctx.fillRect(x(i) - width / 2, volumeBase - volumeHeight, width, volumeHeight);
  }
  let lastLabel = top - 16;
  for (const mark of reference.sort((a, b) => b.level - a.level)) {
    const axisY = y(mark.level);
    const labelY = Math.max(axisY, lastLabel + 15);
    lastLabel = labelY;
    line(left, axisY, right, axisY, mark.color, [5, 4]);
    line(right, axisY, right + 4, labelY, mark.color);
    ctx.fillStyle = "#181d24";
    ctx.fillRect(right + 4, labelY - 7, ratio - 4, 14);
    ctx.fillStyle = mark.color;
    ctx.fillText(`${mark.label} ${price(mark.level, app.selected)}`, right + 5, labelY, ratio - 7);
  }
  for (const point of chart.structure.points) {
    const i = selection.findIndex((row) => row[0] === point.date);
    if (i < 0) continue;
    const py = y(point.price) + (point.key === "B" ? -10 : 11);
    ctx.fillStyle = "#18202a";
    ctx.fillRect(x(i) - 7, py - 7, 14, 14);
    ctx.fillStyle = "#bdd0f7";
    ctx.fillText(point.key, x(i) - 3, py);
  }
  if (app.crosshair === null) return;
  const i = Math.max(0, Math.min(selection.length - 1, app.crosshair));
  const row = selection[i];
  line(x(i), top, x(i), volumeBase, "#8291a6", [3, 3]);
  const tip = $("chart-tooltip");
  tip.innerHTML = `<b>${row[0]}</b><br>开 ${price(row[1], app.selected)}　收 ${price(row[4], app.selected)}<br>高 ${price(row[2], app.selected)}　低 ${price(row[3], app.selected)}<br>成交量 ${fmt(row[5] / 10000, 1)} 万`;
  tip.hidden = false;
  tip.style.left = `${Math.max(4, x(i) < w / 2 ? Math.min(w - tip.offsetWidth - 4, x(i) + 9) : x(i) - tip.offsetWidth - 10)}px`;
}

function renderNews() {
  const news = app.data.news;
  const topics = Object.entries(news.topics);
  app.topic ??= topics.find(([, items]) => items.length)?.[0] ?? topics[0][0];
  $("news-stamp").textContent = `自 ${news.since.slice(5, 16)} · 本地消息流最新 ${news.latest?.slice(5, 16) || "暂无"}`;
  $("news-topics").innerHTML = topics.map(([topic, items]) => `<button type="button" data-topic="${esc(topic)}" aria-pressed="${topic === app.topic}">${esc(topic)} ${items.length}</button>`).join("");
  const names = {em: "东财抓取流", ths: "同花顺", futu: "富途", sina: "新浪", cls: "财联社"};
  $("news-list").innerHTML = news.topics[app.topic].map((item) => `<article class="news-item"><div class="news-meta">${esc(item.time.slice(5, 16))} · ${esc(names[item.source] || item.source)} · ${esc(item.matches.join(" / "))}</div><h3>${esc(item.title)}</h3>${item.body !== item.title ? `<p>${esc(item.body)}</p>` : ""}</article>`).join("") || '<p class="empty">这个时间窗内暂无匹配的本地消息。</p>';
  $("checked-news").innerHTML = news.checked.map((item) => `<article class="news-item"><div class="news-meta">${item.date} · ${esc(item.name)}</div><h3><a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">${esc(item.title)} ↗</a></h3><p>${esc(item.body)}</p></article>`).join("") || '<p class="empty">近期暂无已核对的相关公告。</p>';
  $("x-reference").innerHTML = `<article class="news-item"><div class="news-meta">${news.x.date} · 看盘工具</div><h3><a href="${esc(news.x.url)}" target="_blank" rel="noopener noreferrer">${esc(news.x.author)} ↗</a></h3><p>${esc(news.x.text)}</p></article>`;
}

function marketView() {
  renderMetrics();
  renderFeed();
  renderActions();
  renderChart();
  renderTable();
}

function selectSymbol(symbol) {
  if (!instrument(symbol)) return;
  app.selected = symbol;
  app.crosshair = null;
  renderChart();
  renderTable();
  $("chart-title").scrollIntoView({block: "nearest"});
}

let quotePending;
async function refreshQuotes() {
  if (!app.data || quotePending) return quotePending;
  const symbols = [...Object.keys(app.data.indices), ...Object.keys(app.data.charts)].join(",");
  quotePending = (async () => {
    try {
      const result = await get(`/api/quotes?symbols=${encodeURIComponent(symbols)}`);
      app.quotes = result.quotes;
      app.quoteError = result.error;
    } catch (error) {
      app.quotes = {};
      app.quoteError = "行情暂未连通，显示本地已收盘 K 线";
    }
    marketView();
  })();
  try {
    await quotePending;
  } finally {
    quotePending = null;
  }
}

let dataPending;
async function refreshData(withQuotes = true) {
  if (dataPending) return dataPending;
  const buttons = [$("refresh"), $("refresh-news")];
  buttons.forEach((button) => { button.disabled = true; });
  dataPending = (async () => {
    try {
      app.data = await get("/api/dashboard");
      if (!instrument(app.selected)) {
        const ranked = app.data.holdings.map((item) => ({...item, rank: stopStatus(item).rank})).sort((a, b) => a.rank - b.rank);
        app.selected = ranked[0]?.symbol ?? app.data.lamps[0]?.symbol ?? null;
      }
      $("error").hidden = app.data.reportFresh;
      $("error").textContent = app.data.reportFresh ? "" : "本地晨报待更新。复核日期后再使用开盘窗口与结构灯。";
      $("terminal-content").hidden = false;
      marketView();
      renderNews();
      $("sources").textContent = "数据来自本地晨报、前复权 K 线、持仓配置与消息缓存。盘中报价用于盯盘，账户以券商为准。";
      if (withQuotes) await refreshQuotes();
    } catch (error) {
      $("error").hidden = false;
      $("error").textContent = `更新失败。${error.message}`;
    }
  })();
  try {
    await dataPending;
  } finally {
    dataPending = null;
    buttons.forEach((button) => { button.disabled = false; });
  }
}

function toggleGroup(container, field, select) {
  $(container).addEventListener("click", (event) => {
    const button = event.target.closest(`button[data-${field}]`);
    if (!button) return;
    $(container).querySelectorAll("button").forEach((entry) => entry.setAttribute("aria-pressed", String(entry === button)));
    select(button.dataset[field]);
  });
}

toggleGroup("chart-range", "range", (range) => { app.range = Number(range); app.crosshair = null; drawChart(); });
toggleGroup("table-tabs", "tab", (tab) => { app.tab = tab; renderTable(); });
toggleGroup("lamp-filters", "filter", (filter) => { app.filter = filter; $("lamp-search").value = ""; renderTable(); });
toggleGroup("news-topics", "topic", (topic) => { app.topic = topic; renderNews(); });
$("lamp-search").addEventListener("input", renderTable);
$("terminal-content").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-symbol]");
  if (button) selectSymbol(button.dataset.symbol);
});
$("refresh").addEventListener("click", () => refreshData());
$("refresh-news").addEventListener("click", () => refreshData(false));
$("x-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = $("x-result");
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  result.textContent = "正在读取公开摘要…";
  try {
    const post = await get(`/api/x?url=${encodeURIComponent($("x-link").value.trim())}`);
    result.innerHTML = `<article class="news-item"><h3><a href="${esc(post.url)}" target="_blank" rel="noopener noreferrer">${esc(post.author || "打开原帖")} ↗</a></h3><p>${esc(post.error || post.text || "原帖没有公开文字摘要。")}</p></article>`;
  } catch (error) {
    result.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

const canvas = $("candles");
canvas.addEventListener("pointermove", (event) => {
  const count = Math.min(app.range, app.data?.charts[app.selected]?.bars.length || 0);
  const width = canvas.clientWidth;
  if (!count) return;
  app.crosshair = Math.floor((event.offsetX - 13) * count / (width - (width > 500 ? 62 : 55) - 13));
  drawChart();
});
canvas.addEventListener("pointerleave", () => { app.crosshair = null; drawChart(); });
canvas.addEventListener("blur", () => { app.crosshair = null; drawChart(); });
canvas.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const tip = $("chart-tooltip");
  const count = Math.min(app.range, app.data?.charts[app.selected]?.bars.length || 0);
  const delta = event.key === "ArrowLeft" ? -1 : 1;
  app.crosshair = Math.max(0, Math.min(count - 1, (app.crosshair ?? count) + delta));
  drawChart();
  if (!tip.hidden) canvas.setAttribute("aria-label", tip.textContent);
});
new ResizeObserver(drawChart).observe($("chart-wrap"));
refreshData();
setInterval(() => { if (!document.hidden) refreshQuotes(); }, 15000);
setInterval(() => { if (!document.hidden) refreshData(false); }, 60000);
