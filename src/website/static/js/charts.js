/**
 * charts.js — Interactive graph controls for the Strategy Backtesting Engine.
 *
 * Responsibilities:
 *   - Layout mode switching: single / dual / quad panels.
 *   - Per-panel chart loading via GET /api/chart.
 *   - Equity curve and signals chart rendering with Plotly.
 *   - Chart type tabs: Equity / Signals / Sleeves (hybrid only).
 *   - Stats sidebar population (10 metrics: avg win/loss, win rate, total/ann
 *     return, max/avg win/loss, max/avg drawdown, plus B&H comparison).
 *   - Panel label: shows "Strategy · Asset · Xy" after a chart loads.
 *   - Auto-load: panel 1 loads its default chart on DOMContentLoaded.
 *   - Auto-refresh: changing any dropdown immediately reloads the panel.
 *   - Full error recovery: stats and label reset when a load fails.
 */

"use strict";

/* =========================================================
   Per-panel data store
   ========================================================= */

/** @type {Object.<string, {equity:object, signals:object, sleeves:object, metrics:object, metadata:object}>} */
const _panelData = {};

/* Ordered chart types — tabs are shown/hidden based on data availability */
const _ALL_CHART_TYPES = ["equity", "signals", "sleeves"];


/* =========================================================
   Layout mode
   ========================================================= */

const LAYOUT_PANEL_COUNTS = { single: 1, dual: 2, quad: 4 };

function setLayout(mode) {
  const container = document.getElementById("panels-container");
  if (!container) return;

  container.className = `panels-${mode}`;

  const visibleCount = LAYOUT_PANEL_COUNTS[mode] || 1;
  for (let i = 1; i <= 4; i++) {
    const panel = document.getElementById(`graph-panel-${i}`);
    if (!panel) continue;
    panel.style.display = i <= visibleCount ? "" : "none";
  }

  document.querySelectorAll(".btn-layout").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.layout === mode);
  });

  // Resize any rendered Plotly charts
  for (let i = 1; i <= visibleCount; i++) {
    for (const type of _ALL_CHART_TYPES) {
      const plotDiv = document.getElementById(`plotly-${type}-${i}`);
      if (plotDiv && plotDiv.data) Plotly.relayout(plotDiv, {});
    }
  }
}


/* =========================================================
   Chart loading
   ========================================================= */

async function loadChart(panelId) {
  const strategy  = document.getElementById(`select-strategy-${panelId}`).value;
  const asset     = document.getElementById(`select-asset-${panelId}`).value;
  const timeframe = document.getElementById(`select-timeframe-${panelId}`).value;

  const btn = document.getElementById(`btn-load-${panelId}`);
  btn.textContent = "Loading…";
  btn.disabled = true;

  try {
    const url = `/api/chart?strategy=${encodeURIComponent(strategy)}&asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`;
    const resp = await fetch(url);
    const data = await resp.json();

    if (!resp.ok) {
      showError(panelId, data.error || "Chart not available");
      return;
    }

    _panelData[panelId] = data;

    // Show/hide optional tabs based on what data the server returned
    _setTabVisibility(panelId, "signals", !!data.signals);
    _setTabVisibility(panelId, "sleeves", !!data.sleeves);

    _showChartType(panelId, "equity");
    renderStats(panelId, data.metrics);
    updatePanelLabel(panelId, strategy, asset, timeframe);
    showChartTabs(panelId);

  } catch (err) {
    showError(panelId, "Failed to load chart. Is the server running?");
  } finally {
    btn.textContent = "Load Chart";
    btn.disabled = false;
  }
}

/** Show or hide a named tab button. */
function _setTabVisibility(panelId, type, visible) {
  const tab = document.getElementById(`tab-${type}-${panelId}`);
  if (tab) tab.style.display = visible ? "" : "none";
}


/* =========================================================
   Chart type switching
   ========================================================= */

function switchChartType(panelId, type) {
  if (!_panelData[panelId]) return;
  _showChartType(panelId, type);
}

function _showChartType(panelId, type) {
  const data = _panelData[panelId];
  if (!data) return;

  const figureDict = data[type];
  if (!figureDict) return;

  const placeholder = document.getElementById(`chart-placeholder-${panelId}`);
  if (placeholder) placeholder.style.display = "none";

  for (const t of _ALL_CHART_TYPES) {
    const div = document.getElementById(`plotly-${t}-${panelId}`);
    if (!div) continue;
    div.style.display = t === type ? "block" : "none";
  }

  const plotDiv = document.getElementById(`plotly-${type}-${panelId}`);
  const fixedHeight = type === "equity";   // equity is clamped to 350px; others use figure height
  const layoutOverrides = {
    autosize: true,
    margin: fixedHeight
      ? { l: 55, r: 20, t: 50, b: 45 }
      : { l: 60, r: 30, t: 80, b: 50 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(255,255,255,0.025)",
    font: Object.assign(
      {},
      (figureDict.layout && figureDict.layout.font) || {},
      { color: "#94A3B8", family: "'Inter', system-ui, sans-serif" }
    ),
  };
  if (fixedHeight) layoutOverrides.height = 350;

  const layout = Object.assign({}, figureDict.layout, layoutOverrides);

  // Deep-patch axis colors without clobbering existing axis config (title, tickformat, range, etc.)
  const _axisStyle = {
    gridcolor:     "rgba(255,255,255,0.07)",
    zerolinecolor: "rgba(255,255,255,0.15)",
    linecolor:     "rgba(255,255,255,0.07)",
    tickfont:      { color: "#64748B", size: 11 },
    titlefont:     { color: "#94A3B8" },
  };
  for (const ax of ["xaxis", "xaxis2", "yaxis", "yaxis2", "yaxis3"]) {
    if (layout[ax]) layout[ax] = Object.assign({}, layout[ax], _axisStyle);
  }

  Plotly.newPlot(plotDiv, figureDict.data, layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    displaylogo: false,
  });

  for (const t of _ALL_CHART_TYPES) {
    const tab = document.getElementById(`tab-${t}-${panelId}`);
    if (tab) tab.classList.toggle("active", t === type);
  }
}


/* =========================================================
   Panel label
   ========================================================= */

function _strategyLabel(panelId, strategyKey) {
  const select = document.getElementById(`select-strategy-${panelId}`);
  if (!select) return strategyKey;
  const option = select.querySelector(`option[value="${strategyKey}"]`);
  return option ? option.textContent.trim() : strategyKey;
}

function updatePanelLabel(panelId, strategyKey, asset, timeframe) {
  const el = document.getElementById(`panel-label-${panelId}`);
  if (!el) return;
  el.textContent = `${_strategyLabel(panelId, strategyKey)} · ${asset} · ${timeframe}y`;
}

function clearPanelLabel(panelId) {
  const el = document.getElementById(`panel-label-${panelId}`);
  if (el) el.textContent = "";
}

function showChartTabs(panelId) {
  const tabs = document.getElementById(`chart-tabs-${panelId}`);
  if (tabs) tabs.style.display = "";
}


/* =========================================================
   Stats sidebar — 10-metric set
   ========================================================= */

/**
 * Populate the stats sidebar with the 10 primary metrics + B&H comparison.
 *
 * Metric set:
 *   Total Return, Ann. Return, Win Rate,
 *   Avg Win, Avg Loss, Max Win, Max Loss,
 *   Max Drawdown, Avg Drawdown,
 *   Buy & Hold (benchmark), vs Buy & Hold (outperformance)
 */
function renderStats(panelId, metrics) {
  const pct    = v => (v == null || isNaN(v)) ? "—" : `${(v * 100).toFixed(2)}%`;
  const signed = v => (v == null || isNaN(v)) ? "—"
                    : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;

  const updates = {
    [`stat-total-return-${panelId}`]:  pct(metrics.total_return),
    [`stat-ann-return-${panelId}`]:    pct(metrics.annualized_return),
    [`stat-winrate-${panelId}`]:       pct(metrics.win_rate),
    [`stat-avg-win-${panelId}`]:       signed(metrics.avg_win),
    [`stat-avg-loss-${panelId}`]:      signed(metrics.avg_loss),
    [`stat-max-win-${panelId}`]:       signed(metrics.max_win),
    [`stat-max-loss-${panelId}`]:      signed(metrics.max_loss),
    [`stat-max-drawdown-${panelId}`]:  pct(metrics.max_drawdown),
    [`stat-avg-drawdown-${panelId}`]:  pct(metrics.avg_drawdown),
    [`stat-benchmark-${panelId}`]:     pct(metrics.benchmark_return),
    [`stat-outperformance-${panelId}`]: signed(metrics.outperformance),
  };

  for (const [id, value] of Object.entries(updates)) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.textContent = value;
    el.classList.remove("placeholder");
    el.style.color = "";
    el.style.fontWeight = "";

    // Colour-code signed metrics
    if (value !== "—") {
      const isPositiveGood = id.includes("outperformance") || id.includes("avg-win")
                          || id.includes("max-win") || id.includes("total-return")
                          || id.includes("ann-return");
      const isNegativeGood = id.includes("avg-loss") || id.includes("max-loss")
                          || id.includes("drawdown");

      if (isPositiveGood || id.includes("outperformance")) {
        el.style.color = value.startsWith("+") ? "#16A34A" : "#DC2626";
        if (id.includes("outperformance")) el.style.fontWeight = "700";
      }
    }
  }
}

function resetStats(panelId) {
  const statIds = [
    "total-return", "ann-return", "winrate",
    "avg-win", "avg-loss", "max-win", "max-loss",
    "max-drawdown", "avg-drawdown",
    "benchmark", "outperformance",
  ];
  for (const name of statIds) {
    const el = document.getElementById(`stat-${name}-${panelId}`);
    if (!el) continue;
    el.textContent = "—";
    el.classList.add("placeholder");
    el.style.color = "";
    el.style.fontWeight = "";
  }
}


/* =========================================================
   Error display
   ========================================================= */

function showError(panelId, message) {
  for (const type of _ALL_CHART_TYPES) {
    const div = document.getElementById(`plotly-${type}-${panelId}`);
    if (div) div.style.display = "none";
  }
  const placeholder = document.getElementById(`chart-placeholder-${panelId}`);
  if (placeholder) {
    placeholder.style.display = "";
    placeholder.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="1.5"
           style="width:40px;height:40px;margin-bottom:12px;" aria-hidden="true">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <p style="color:#EF4444;">${message}</p>`;
  }

  delete _panelData[panelId];
  resetStats(panelId);
  clearPanelLabel(panelId);

  const tabs = document.getElementById(`chart-tabs-${panelId}`);
  if (tabs) tabs.style.display = "none";
}


/* =========================================================
   Initialisation
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".btn-layout").forEach(btn => {
    btn.addEventListener("click", () => setLayout(btn.dataset.layout));
  });

  document.querySelectorAll(".btn-load").forEach(btn => {
    btn.addEventListener("click", () => loadChart(btn.dataset.panel));
  });

  // Auto-refresh when any dropdown changes
  for (let i = 1; i <= 4; i++) {
    for (const field of ["strategy", "asset", "timeframe"]) {
      const select = document.getElementById(`select-${field}-${i}`);
      if (select) select.addEventListener("change", () => loadChart(i));
    }
  }

  document.querySelectorAll(".btn-chart-type").forEach(btn => {
    btn.addEventListener("click", () => {
      switchChartType(btn.dataset.panel, btn.dataset.type);
    });
  });

  setLayout("single");
  loadChart(1);
});
