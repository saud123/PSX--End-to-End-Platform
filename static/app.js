/* ============================================================
   PSX INTELLIGENCE · app.js · Design System 1.0
   Live-only · interactive charts · tech indicators · sticky head
   ============================================================ */
'use strict';

/* ── Defensive helpers ─────────────────────────────────────── */
function safeRun(fn, fallback = null) {
  try {
    return fn();
  } catch (err) {
    console.warn('[safeRun]', err);
    return fallback;
  }
}

async function safeAsync(fn, fallback = null) {
  try {
    return await fn();
  } catch (err) {
    console.warn('[safeAsync]', err);
    return fallback;
  }
}

function withTimeout(promise, ms = 45000) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout after ' + ms + 'ms')), ms)),
  ]);
}

function showErrorBoundary(err) {
  const el = document.getElementById('error-boundary');
  const msg = document.getElementById('error-boundary-msg');
  if (!el || !msg) return;
  msg.textContent = (err && (err.message || String(err))) || 'Unknown error';
  el.hidden = false;
}

function installErrorBoundary() {
  window.addEventListener('error', (e) => {
    console.error('[GlobalError]', e.error);
    showErrorBoundary(e.error);
  });
  window.addEventListener('unhandledrejection', (e) => {
    console.error('[UnhandledRejection]', e.reason);
    showErrorBoundary(e.reason);
  });
  const rb = document.getElementById('error-boundary-retry');
  const db = document.getElementById('error-boundary-dismiss');
  if (rb) rb.addEventListener('click', () => location.reload());
  if (db) db.addEventListener('click', () => {
    const el = document.getElementById('error-boundary');
    if (el) el.hidden = true;
  });
}

/* ── Utilities ─────────────────────────────────────────────── */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}[c]));
const fmtNum = (v, d = 2) => (v == null || isNaN(v)) ? '—' : Number(v).toLocaleString('en-US', {
  minimumFractionDigits: d,
  maximumFractionDigits: d,
});
const fmtPct = (v, d = 2) => {
  if (v == null || isNaN(v)) return '—';
  const n = Number(v);
  return (n >= 0 ? '+' : '') + n.toFixed(d) + '%';
};
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ── NEW: history shape normalizer ──────────────────────────
   The backend returns price series in two different shapes:
     A) [{close: 100}, {close: 101}, …]        ← /api/analyze price_history
     B) {dates:[…], close:[100, 101, …], …}    ← /api/compare history[sym]
   This helper flattens either into a plain array of finite close prices.
   Returns [] when neither shape matches.
   --------------------------------------------------------- */
function _historyToCloseArray(h) {
  if (!h) return [];
  if (Array.isArray(h)) {
    const out = [];
    for (const r of h) {
      if (r && r.close != null && !isNaN(+r.close)) out.push(+r.close);
    }
    return out;
  }
  if (Array.isArray(h.close)) {
    const out = [];
    for (const v of h.close) {
      if (v != null && !isNaN(+v)) out.push(+v);
    }
    return out;
  }
  return [];
}

function toast(msg) {
  const el = $('#toast');
  if (!el) return;
  el.textContent = msg;
  el.dataset.open = 'true';
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.dataset.open = 'false';
    setTimeout(() => {
      el.hidden = true;
    }, 220);
  }, 2400);
}

function countUp(el, target, { decimals = 0, prefix = '', suffix = '', duration = 700 } = {}) {
  if (!el) return;
  if (target == null || isNaN(target)) {
    el.textContent = '—';
    return;
  }
  if (reduced()) {
    el.textContent = prefix + Number(target).toFixed(decimals) + suffix;
    return;
  }
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / duration);
    const current = target * (1 - Math.pow(1 - p, 3));
    el.textContent = prefix + current.toFixed(decimals) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ── Scroll progress + reveal ──────────────────────────────── */
function updateScrollProgress() {
  const el = document.getElementById('scroll-progress');
  if (!el) return;
  const h = document.documentElement.scrollHeight - window.innerHeight;
  el.style.transform = `scaleX(${h > 0 ? Math.min(1, Math.max(0, window.scrollY / h)) : 0})`;
}

let revealObserver = null;

function initScrollReveal() {
  if (reduced() || !('IntersectionObserver' in window)) {
    document.documentElement.classList.add('reveal-no-motion');
    return;
  }
  revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
}

function observeReveals() {
  if (!revealObserver) return;
  document.querySelectorAll('.reveal:not(.is-visible)').forEach((el) => revealObserver.observe(el));
}

/* ── Micro-interactions ───────────────────────────────────── */
function attachMicroInteractions(root = document) {
  root.querySelectorAll('[data-copy]').forEach((el) => {
    if (el._copyWired) return;
    el._copyWired = true;
    el.addEventListener('click', () => {
      const text = el.dataset.copy;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => toast(`Copied "${text}"`)).catch(() => {});
      }
    });
  });
}

/* ── State ─────────────────────────────────────────────────── */
const state = {
  route: 'analyze',
  connection: 'unknown',
  analyze: { phase: 'idle', symbol: null, query: '', data: null, error: null },
  compare: { selected: [], data: null, error: null, loading: false },
  screen: { profile: 'balanced', data: null, sortKey: 'rank', sortDir: 'asc', expanded: new Set(), loading: false, error: null },
  chat: {
    threads: [{ id: 't1', title: 'New conversation', when: 'now' }],
    currentId: 't1',
    messages: [],
    streaming: false,
  },
  sources: {},
};

/* ── Preferences ───────────────────────────────────────────── */
function loadPreferences() {
  try {
    const raw = localStorage.getItem('psx-prefs');
    if (!raw) return;
    const p = JSON.parse(raw);
    if (p.theme === 'light' || p.theme === 'dark') document.documentElement.dataset.theme = p.theme;
    if (p.screen?.profile && ['Quantitative Finance'].includes(p.screen.profile)) state.screen.profile = p.screen.profile;
  } catch (err) {
    console.warn('[prefs] load failed', err);
  }
}

function savePreferences() {
  try {
    localStorage.setItem('psx-prefs', JSON.stringify({
      theme: document.documentElement.dataset.theme || 'dark',
      screen: { profile: state.screen.profile },
    }));
  } catch (err) {
    console.warn('[prefs] save failed', err);
  }
}

/* ── Collapsible sidebar ──────────────────────────────────── */
function initSidebar() {
  const shell = document.querySelector('.shell');
  const toggle = document.getElementById('sidebar-toggle');
  if (!shell || !toggle) return;

  let persisted = 'open';
  try {
    persisted = localStorage.getItem('psx-sidebar') || 'open';
  } catch {}
  if (!['open', 'closed'].includes(persisted)) persisted = 'open';
  shell.dataset.sidebar = persisted;
  toggle.setAttribute('aria-expanded', String(persisted === 'open'));

  const setSidebarState = (next) => {
    shell.dataset.sidebar = next;
    toggle.setAttribute('aria-expanded', String(next === 'open'));
    try {
      localStorage.setItem('psx-sidebar', next);
    } catch {}
  };

  toggle.addEventListener('click', () => {
    const current = shell.dataset.sidebar === 'closed' ? 'closed' : 'open';
    setSidebarState(current === 'open' ? 'closed' : 'open');
  });

  document.addEventListener('keydown', (e) => {
    const sheet = document.getElementById('sheet');
    const sheetOpen = sheet && !sheet.hidden;
    if (e.key === 'Escape' && shell.dataset.sidebar === 'open' && window.innerWidth > 900 && !sheetOpen) {
      setSidebarState('closed');
    }
  });
}

/* ── Theme ─────────────────────────────────────────────────── */
function initTheme() {
  loadPreferences();
  const saved = localStorage.getItem('psx-theme');
  if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
  $('#theme-btn').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('psx-theme', next);
    } catch {}
    savePreferences();
    redrawCharts();
  });
}

/* ── Connection badge ─────────────────────────────────────── */
function setConnectionStatus(online) {
  state.connection = online ? 'online' : 'offline';
  const el = $('#mode-badge');
  if (!el) return;
  const today = new Date().toLocaleDateString('en-US', { day: '2-digit', month: 'short' });
  if (online) {
    el.dataset.mode = 'live';
    el.querySelector('.txt').textContent = `EOD · ${today}`;
  } else {
    el.dataset.mode = 'offline';
    el.querySelector('.txt').textContent = 'OFFLINE';
  }
}

/* ── Sources drawer ───────────────────────────────────────── */
function registerSource(key, source) {
  state.sources[key] = source;
  updateSourcesCount();
}

function updateSourcesCount() {
  const n = Object.keys(state.sources).length;
  const el = $('#src-count');
  if (!el) return;
  el.textContent = n;
  el.style.display = n > 0 ? 'grid' : 'none';
}

function openSources(focusKey) {
  const sheet = $('#sheet'), backdrop = $('#sheet-backdrop'), body = $('#sheet-body');
  if (!sheet || !backdrop || !body) return;
  const entries = Object.entries(state.sources);
  if (!entries.length) {
    body.innerHTML = '<div class="sheet-empty">No sources yet. Run an analysis to see cited data.</div>';
  } else {
    body.innerHTML = entries.map(([key, s]) => `
      <div class="source-item" id="src-${esc(key)}">
        <div class="source-head"><span class="source-name">${esc(s.name)}</span><span class="source-kind">${esc(s.kind || 'source')}</span></div>
        ${s.detail ? `<div class="source-detail">${esc(s.detail)}</div>` : ''}
        ${s.url ? `<a class="source-url" href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.url)} ↗</a>` : ''}
        ${s.note ? `<div class="source-detail" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--color-border);">${esc(s.note)}</div>` : ''}
      </div>`).join('');
  }
  sheet.hidden = false;
  backdrop.hidden = false;
  requestAnimationFrame(() => {
    sheet.dataset.open = 'true';
    backdrop.dataset.open = 'true';
  });
  if (focusKey) {
    setTimeout(() => {
      const n = $(`#src-${focusKey}`);
      if (n) n.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'center' });
    }, 260);
  }
}

function closeSources() {
  const sheet = $('#sheet'), backdrop = $('#sheet-backdrop');
  if (!sheet || !backdrop) return;
  sheet.dataset.open = 'false';
  backdrop.dataset.open = 'false';
  setTimeout(() => {
    sheet.hidden = true;
    backdrop.hidden = true;
  }, 220);
}

/* ── API client (LIVE ONLY) ───────────────────────────────── */
async function parseSSEStream(res, onEvent) {
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let sawTerminal = false;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === '{}') continue;
        try {
          const evt = JSON.parse(payload);
          if (evt.type === 'complete' || evt.type === 'done') sawTerminal = true;
          onEvent(evt);
        } catch (err) {
          console.warn('[sse] parse failed', err);
        }
      }
    }
  }
  return sawTerminal;
}

const api = {
  async analyzeStream(symbol, query, onEvent) {
    return safeAsync(async () => {
      const res = await withTimeout(fetch('/api/analyze/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, symbol, days: 180 }),
      }), 60000);
      return await parseSSEStream(res, onEvent);
    }, false);
  },
  async compare(symbols) {
    return safeAsync(async () => {
      const res = await withTimeout(fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols }),
      }), 90000);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    }, null);
  },
  async chatStream(threadId, message, onEvent) {
    return safeAsync(async () => {
      const res = await withTimeout(fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, message }),
      }), 60000);
      return await parseSSEStream(res, onEvent);
    }, false);
  },
  async quant(topN, profile, refresh = false) {
    return safeAsync(async () => {
      const res = await withTimeout(fetch(`/api/quant/top-picks?n=${topN}&profile=${profile}&refresh=${refresh}`), 90000);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    }, null);
  },
};

/* ── Canvas helpers ───────────────────────────────────────── */
function setupCanvas(canvas, cssHeight = 240) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width || 320, 120);
  const h = cssHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}

/* ── INTERACTIVE PRICE CHART ──────────────────────────────── */
function drawPriceChart(canvas, history, forecast, opts = {}) {
  if (!canvas) return;
  const animate = opts.animate !== false && !reduced();

  const historyRows = Array.isArray(history) ? history.filter(r => r && r.close != null && !isNaN(+r.close)) : [];
  const forecastRows = Array.isArray(forecast) ? forecast.filter(r => r && r.predicted_close != null && !isNaN(+r.predicted_close)) : [];

  if (!historyRows.length) {
    const { ctx, w, h } = setupCanvas(canvas, 240);
    ctx.fillStyle = cssVar('--color-text-muted') || '#8fa196';
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No price history available', w / 2, h / 2);
    return;
  }

  const hValues = historyRows.map(d => +d.close);
  const fValues = forecastRows.map(d => +d.predicted_close);
  const hasForecast = fValues.length > 0;
  const hasBand = hasForecast && forecastRows[0]?.lower_close != null && forecastRows[0]?.upper_close != null;
  const lowerValues = hasBand ? forecastRows.map(f => +f.lower_close) : [];
  const upperValues = hasBand ? forecastRows.map(f => +f.upper_close) : [];
  const all = [...hValues, ...fValues, ...lowerValues, ...upperValues];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = (max - min) || 1;
  const total = hValues.length + fValues.length;

  let currentProgress = animate ? 0 : 1;
  let currentHover = -1;
  let layout = null;

  function frame() {
    const { ctx, w, h } = setupCanvas(canvas, 240);
    const pad = { l: 46, r: 16, t: 16, b: 28 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;

    const x = (i) => pad.l + (iw * i) / Math.max(1, total - 1);
    const y = (v) => pad.t + ih - ((v - min) / range) * ih;

    layout = { w, h, pad, iw, ih, total, dpr: window.devicePixelRatio || 1 };

    const borderColor = cssVar('--color-border') || '#1f2a22';
    const borderHi = cssVar('--color-border-hi') || '#2e3f32';
    const textColor = cssVar('--color-text') || '#eef3ef';
    const textMuted = cssVar('--color-text-muted') || '#8fa196';
    const accent = cssVar('--color-signal-buy') || '#b8ff3d';
    const surface2 = cssVar('--color-surface-2') || '#1a231c';

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const gy = pad.t + (ih * i) / 3;
      ctx.beginPath();
      ctx.moveTo(pad.l, gy);
      ctx.lineTo(w - pad.r, gy);
      ctx.stroke();
    }
    ctx.fillStyle = textMuted;
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 3; i++) {
      const val = max - (range * i) / 3;
      ctx.fillText(val.toFixed(0), pad.l - 8, pad.t + (ih * i) / 3);
    }

    if (hasBand && currentProgress >= 1) {
      ctx.fillStyle = 'rgba(184, 255, 61, 0.10)';
      ctx.beginPath();
      ctx.moveTo(x(hValues.length - 1), y(hValues[hValues.length - 1]));
      forecastRows.forEach((f, i) => ctx.lineTo(x(hValues.length + i), y(+f.upper_close)));
      for (let i = forecastRows.length - 1; i >= 0; i--) {
        ctx.lineTo(x(hValues.length + i), y(+forecastRows[i].lower_close));
      }
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = 'rgba(184, 255, 61, 0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      forecastRows.forEach((f, i) => {
        const px = x(hValues.length + i);
        const pyU = y(+f.upper_close);
        i === 0 ? ctx.moveTo(px, pyU) : ctx.lineTo(px, pyU);
      });
      ctx.stroke();
      ctx.beginPath();
      forecastRows.forEach((f, i) => {
        const px = x(hValues.length + i);
        const pyL = y(+f.lower_close);
        i === 0 ? ctx.moveTo(px, pyL) : ctx.lineTo(px, pyL);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const limit = Math.floor(currentProgress * hValues.length);
    ctx.strokeStyle = textColor;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i <= limit && i < hValues.length; i++) {
      i === 0 ? ctx.moveTo(x(i), y(hValues[i])) : ctx.lineTo(x(i), y(hValues[i]));
    }
    ctx.stroke();

    if (currentProgress >= 1 && hasForecast) {
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x(hValues.length - 1), y(hValues[hValues.length - 1]));
      fValues.forEach((v, i) => ctx.lineTo(x(hValues.length + i), y(v)));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = accent;
      ctx.beginPath();
      ctx.arc(x(total - 1), y(fValues[fValues.length - 1]), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    if (currentProgress >= 1) {
      ctx.fillStyle = textMuted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      [0, Math.floor(hValues.length / 2), hValues.length - 1].forEach((i) => {
        const lbl = historyRows[i]?.trade_date?.slice(5) || '';
        if (lbl) ctx.fillText(lbl, x(i), h - pad.b + 8);
      });
    }

    if (currentHover >= 0 && currentHover < total) {
      const isFcst = currentHover >= hValues.length;
      const value = isFcst ? fValues[currentHover - hValues.length] : hValues[currentHover];
      const dateStr = isFcst
        ? forecastRows[currentHover - hValues.length]?.trade_date
        : historyRows[currentHover]?.trade_date;
      const hx = x(currentHover);
      const hy = y(value);

      ctx.strokeStyle = textMuted;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, pad.t);
      ctx.lineTo(hx, pad.t + ih);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = isFcst ? accent : textColor;
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = surface2;
      ctx.beginPath();
      ctx.arc(hx, hy, 2.5, 0, Math.PI * 2);
      ctx.fill();

      const label = `${(dateStr || '').slice(5)}  PKR ${value.toFixed(2)}${isFcst ? ' (fcst)' : ''}`;
      ctx.font = "12px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(label).width;
      const bw = tw + 20;
      const bh = 30;
      let bx = hx > w / 2 ? hx - bw - 10 : hx + 10;
      if (bx < pad.l) bx = pad.l;
      if (bx + bw > w - pad.r) bx = w - pad.r - bw;
      const by = pad.t + 4;

      ctx.fillStyle = surface2;
      ctx.strokeStyle = borderHi;
      ctx.lineWidth = 1;
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = isFcst ? accent : textColor;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, bx + 10, by + bh / 2);
    }
  }

  canvas._drawFrame = frame;
  canvas._getLayout = () => layout;
  canvas._getHover = () => currentHover;
  canvas._setHover = (i) => {
    currentHover = i;
    frame();
  };

  if (animate) {
    const start = performance.now();
    function tick(now) {
      currentProgress = Math.min(1, (now - start) / 800);
      frame();
      if (currentProgress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  } else {
    currentProgress = 1;
    frame();
  }

  if (!canvas._hoverWired) {
    canvas._hoverWired = true;
    canvas.addEventListener('mousemove', (e) => {
      const l = canvas._getLayout();
      if (!l) return;
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (l.w / rect.width);
      if (mx < l.pad.l || mx > l.pad.l + l.iw) {
        if (canvas._getHover() !== -1) canvas._setHover(-1);
        return;
      }
      const idx = Math.round(((mx - l.pad.l) / l.iw) * (l.total - 1));
      if (idx >= 0 && idx < l.total && idx !== canvas._getHover()) canvas._setHover(idx);
    });
    canvas.addEventListener('mouseleave', () => {
      if (canvas._getHover() !== -1) canvas._setHover(-1);
    });
  }
}

/* ── NORMALIZED COMPARE CHART (multi-line, animated) ────── */
function drawNormalizedChart(canvas, symbols, historyMap, opts = {}) {
  if (!canvas || !symbols || !symbols.length) return;

  // If layout hasn't happened yet (canvas width ≈ 0), retry next frame.
  if (canvas.getBoundingClientRect().width < 60) {
    setTimeout(() => drawNormalizedChart(canvas, symbols, historyMap, opts), 120);
    return;
  }

  const animate = opts.animate !== false && !reduced();

  const { ctx, w, h } = setupCanvas(canvas, 260);
  const pad = { l: 46, r: 16, t: 16, b: 28 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const colors = ['--color-signal-buy', '--color-text', '--color-signal-sell', '--color-signal-hold', '--color-text-muted'];
  const series = {};
  let minV = Infinity, maxV = -Infinity;

  symbols.forEach((sym) => {
    const closes = _historyToCloseArray(historyMap[sym]);
    if (closes.length < 2) return;
    const base = closes[0];
    if (!base) return;
    const vals = closes.map(c => (c / base) * 100);
    series[sym] = vals;
    minV = Math.min(minV, ...vals);
    maxV = Math.max(maxV, ...vals);
  });

  if (!Object.keys(series).length) {
    ctx.fillStyle = cssVar('--color-text-muted') || '#8fa196';
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No history available', w / 2, h / 2);
    return;
  }

  const range = (maxV - minV) || 1;
  let progress = animate ? 0 : 1;

  function frame() {
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = cssVar('--color-border') || '#1f2a22';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const gy = pad.t + (ih * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, gy);
      ctx.lineTo(w - pad.r, gy);
      ctx.stroke();
    }
    ctx.fillStyle = cssVar('--color-text-muted') || '#8fa196';
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const val = maxV - (range * i) / 4;
      ctx.fillText(val.toFixed(0), pad.l - 8, pad.t + (ih * i) / 4);
    }
    ctx.strokeStyle = cssVar('--color-border-hi') || '#2e3f32';
    ctx.setLineDash([3, 4]);
    const y100 = pad.t + ih - ((100 - minV) / range) * ih;
    ctx.beginPath();
    ctx.moveTo(pad.l, y100);
    ctx.lineTo(w - pad.r, y100);
    ctx.stroke();
    ctx.setLineDash([]);

    symbols.forEach((sym, idx) => {
      const vals = series[sym];
      if (!vals) return;
      const color = cssVar(colors[idx % colors.length]) || cssVar('--color-text');
      ctx.strokeStyle = color;
      ctx.lineWidth = idx === 0 ? 2.4 : 1.8;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      const limit = Math.floor(progress * vals.length);
      for (let i = 0; i <= limit && i < vals.length; i++) {
        const px = pad.l + (iw * i) / Math.max(1, vals.length - 1);
        const py = pad.t + ih - ((vals[i] - minV) / range) * ih;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
      if (progress >= 1) {
        const lastV = vals[vals.length - 1];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(pad.l + iw, pad.t + ih - ((lastV - minV) / range) * ih, idx === 0 ? 5 : 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  if (animate) {
    const start = performance.now();
    function tick(now) {
      progress = Math.min(1, (now - start) / 800);
      frame();
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  } else {
    frame();
  }
}

/* ── Redraw on theme / resize ─────────────────────────────── */
function redrawCharts() {
  const priceCanvas = $('#price-canvas');
  if (priceCanvas && state.analyze.phase === 'done' && state.analyze.data) {
    drawPriceChart(priceCanvas, state.analyze.data.price_history || [], state.analyze.data.prediction || [], { animate: false });
  }
  const normCanvas = $('#normalized-canvas');
  if (normCanvas && state.compare.data) {
    drawNormalizedChart(normCanvas, state.compare.data.symbols, state.compare.data.history || {}, { animate: false });
  }
}

let _resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(redrawCharts, 200);
}, { passive: true });

/* ── Report parsing ───────────────────────────────────────── */
function parseReport(md) {
  const vm = md.match(/VERDICT\s*[:]\s*(BUY|SELL|HOLD)/i);
  const cm = md.match(/CONFIDENCE\s*[:]\s*([\d.]+)/i);
  const hm = md.match(/HOLDING[_\s]*PERIOD\s*[:]\s*(\d+)/i);
  const signal = {
    verdict: vm ? vm[1].toUpperCase() : 'HOLD',
    confidence: cm ? Math.min(1, Math.max(0, parseFloat(cm[1]))) : 0.5,
    holding: hm ? parseInt(hm[1], 10) : 15,
  };
  const body = md
    .replace(/VERDICT\s*:.*$/gim, '')
    .replace(/CONFIDENCE\s*:.*$/gim, '')
    .replace(/HOLDING[_\s]*PERIOD\s*:.*$/gim, '')
    .replace(/^#\s+.*$/gm, '')
    .trim();
  return { body, signal };
}

function mdInline(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>');
}

function renderReportBody(body) {
  const lines = body.split('\n');
  const out = [];
  let inList = false;
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      if (inList) {
        out.push('</ul>');
        inList = false;
      }
      continue;
    }
    const h2 = line.match(/^##\s+(\d+)\.\s*(.+)/);
    const li = line.match(/^[-*]\s+(.+)/);
    if (h2) {
      if (inList) {
        out.push('</ul>');
        inList = false;
      }
      out.push(`<h2 class="report-h2"><span class="num">${h2[1].padStart(2, '0')}</span><span>${esc(h2[2])}</span></h2>`);
      continue;
    }
    if (li) {
      if (!inList) {
        out.push('<ul>');
        inList = true;
      }
      out.push(`<li>${mdInline(li[1])}</li>`);
      continue;
    }
    if (inList) {
      out.push('</ul>');
      inList = false;
    }
    out.push(`<p>${mdInline(line)}</p>`);
  }
  if (inList) out.push('</ul>');
  return out.join('\n');
}

function extractSection(md, n) {
  const re = new RegExp(`##\\s*${n}\\.\\s*[^\\n]+\\n+([\\s\\S]*?)(?=\\n##\\s*\\d+\\.|$)`);
  const m = md.match(re);
  return m ? m[1].trim() : '';
}

function firstSentence(text) {
  if (!text) return '';
  const clean = text.replace(/[-*]\s*/g, '').trim();
  const m = clean.match(/^[^.!?]+[.!?]/);
  return m ? m[0].trim() : clean.slice(0, 200);
}

function extractInsights(md) {
  const insights = [];
  const integrated = extractSection(md, 7);
  if (integrated) {
    insights.push({ featured: true, label: 'Why this call', glyph: '◈', tone: 'accent', body: firstSentence(integrated) });
  }
  const risk = extractSection(md, 6);
  if (risk) {
    const firstRisk = risk.split('\n').find(l => /^[-*]/.test(l.trim()));
    insights.push({
      label: 'Primary risk',
      glyph: '⚠',
      tone: 'amber',
      body: firstRisk ? firstRisk.replace(/^[-*]\s*/, '') : firstSentence(risk),
    });
  }
  const market = extractSection(md, 2);
  if (market) insights.push({ label: 'Market state', glyph: '↗', tone: 'default', body: firstSentence(market) });
  const quarterly = extractSection(md, 5);
  if (quarterly) insights.push({ label: 'Financials', glyph: '◎', tone: 'default', body: firstSentence(quarterly) });
  return insights;
}

/* ── State renderers ──────────────────────────────────────── */
function renderLoading(msg = 'Loading…', detail = '') {
  return `<div class="state-loading"><div class="state-icon"><span class="spinner"></span></div><div class="state-title">${esc(msg)}</div>${detail ? `<div class="state-desc">${esc(detail)}</div>` : ''}</div>`;
}

function renderEmpty(title, desc, actions = []) {
  return `<div class="state-empty"><div class="state-icon">○</div><div class="state-title">${esc(title)}</div><div class="state-desc">${esc(desc)}</div>${actions.length ? `<div class="state-actions">${actions.map(a => `<button class="btn-primary press" data-action="${esc(a.id)}">${esc(a.label)}</button>`).join('')}</div>` : ''}</div>`;
}

function renderError(title, desc, detail = '', actions = []) {
  return `<div class="state-error"><div class="state-icon">!</div><div class="state-title">${esc(title)}</div><div class="state-desc">${esc(desc)}</div>${actions.length ? `<div class="state-actions">${actions.map(a => `<button class="btn-primary press" data-action="${esc(a.id)}">${esc(a.label)}</button>`).join('')}</div>` : ''}${detail ? `<div class="state-detail">${esc(detail)}</div>` : ''}</div>`;
}

/* ── Breadcrumb ───────────────────────────────────────────── */
function updateBreadcrumb() {
  const route = currentRoute();
  const labels = { analyze: 'Analyze', compare: 'Compare', screen: 'Screen', chat: 'Chat' };
  const el = document.querySelector('[data-crumb-current]');
  if (el) el.textContent = labels[route] || 'Analyze';
  const backBtn = document.querySelector('[data-crumb-back]');
  if (backBtn) backBtn.disabled = history.length <= 1;
}

/* ── Tech indicators population ───────────────────────────── */
function populateTechIndicators(data) {
  if (!data) return;
  const history = data.price_history || [];
  const last = history.length ? history[history.length - 1] : {};
  const close = data.numerical_data?.close ?? (last.close != null ? +last.close : null);

  const rsi = last.rsi != null && !isNaN(+last.rsi) ? +last.rsi : null;
  const rsiEl = $('#tech-rsi');
  if (rsiEl) countUp(rsiEl, rsi, { decimals: 1 });
  const rsiStatus = $('#rsi-status');
  if (rsiStatus) {
    let stateTag = 'healthy', label = '—';
    if (rsi == null) {
      label = 'No data';
      stateTag = 'healthy';
    } else if (rsi >= 70) {
      stateTag = 'overbought';
      label = 'Overbought';
    } else if (rsi <= 30) {
      stateTag = 'oversold';
      label = 'Oversold';
    } else {
      label = 'Healthy range';
    }
    rsiStatus.dataset.state = stateTag;
    rsiStatus.textContent = label;
  }

  const macd = last.macd != null && !isNaN(+last.macd) ? +last.macd : null;
  const macdSig = last.macd_signal != null && !isNaN(+last.macd_signal) ? +last.macd_signal : null;
  const macdHist = last.macd_hist != null && !isNaN(+last.macd_hist) ? +last.macd_hist : null;

  const macdEl = $('#tech-macd');
  if (macdEl) {
    countUp(macdEl, macd, { decimals: 3, prefix: macd != null && macd >= 0 ? '+' : '' });
    macdEl.classList.remove('pos', 'neg');
    if (macd != null) macdEl.classList.add(macd >= 0 ? 'pos' : 'neg');
  }
  const sigEl = $('#macd-signal');
  if (sigEl) {
    sigEl.textContent = macdSig != null ? (macdSig >= 0 ? '+' : '') + macdSig.toFixed(3) : '—';
    sigEl.classList.remove('pos', 'neg');
    if (macdSig != null) sigEl.classList.add(macdSig >= 0 ? 'pos' : 'neg');
  }
  const histEl = $('#macd-hist');
  if (histEl) {
    histEl.textContent = macdHist != null ? (macdHist >= 0 ? '+' : '') + macdHist.toFixed(3) : '—';
    histEl.classList.remove('pos', 'neg');
    if (macdHist != null) histEl.classList.add(macdHist >= 0 ? 'pos' : 'neg');
  }

  const bbU = last.bb_upper != null && !isNaN(+last.bb_upper) ? +last.bb_upper : null;
  const bbM = last.bb_mid != null && !isNaN(+last.bb_mid) ? +last.bb_mid : null;
  const bbL = last.bb_lower != null && !isNaN(+last.bb_lower) ? +last.bb_lower : null;

  const bbEl = $('#tech-bb');
  if (bbEl) countUp(bbEl, close, { decimals: 2, prefix: 'PKR ' });
  const bbUEl = $('#bb-upper');
  if (bbUEl) bbUEl.textContent = bbU != null ? 'PKR ' + bbU.toFixed(2) : '—';
  const bbMEl = $('#bb-mid');
  if (bbMEl) bbMEl.textContent = bbM != null ? 'PKR ' + bbM.toFixed(2) : '—';
  const bbLEl = $('#bb-lower');
  if (bbLEl) bbLEl.textContent = bbL != null ? 'PKR ' + bbL.toFixed(2) : '—';

  const fill = $('#bb-fill'), marker = $('#bb-marker');
  if (fill && marker) {
    if (close != null && bbU != null && bbL != null && bbU > bbL) {
      const pos = Math.max(0, Math.min(1, (close - bbL) / (bbU - bbL)));
      setTimeout(() => {
        fill.style.width = (pos * 100) + '%';
        marker.style.left = (pos * 100) + '%';
      }, 100);
    } else {
      fill.style.width = '0%';
      marker.style.left = '0%';
    }
  }
}

/* ── Pages ─────────────────────────────────────────────────── */
const pages = {};

/* ANALYZE */
pages.analyze = () => {
  const a = state.analyze;

  if (a.phase === 'idle') {
    return `
      <div class="ask-wrap">
        <h1>Understand <span class="accent">one ticker.</span></h1>
        <p class="lede">Ask a PSX-listed company a question. Five specialist agents run against EOD prices, news, and indexed quarterly filings — then write a sourced intelligence brief.</p>
        <form class="ask-form" id="ask-form">
          <div class="ticker-field"><label for="ticker-input">Ticker</label><input id="ticker-input" name="symbol" placeholder="LUCK" maxlength="8" autocomplete="off" spellcheck="false"></div>
          <div class="query-field"><input id="query-input" name="query" placeholder="How is LUCK positioned for the next quarter?" autocomplete="off"></div>
          <button type="submit" class="submit-btn press">Analyze →</button>
        </form>
        <div class="example-row">
          <span class="lbl">Try</span>
          <button type="button" class="press" data-ex="LUCK|How is LUCK positioned for the next quarter?">LUCK · quarterly outlook</button>
          <button type="button" class="press" data-ex="OGDC|Summarize OGDC's recent price action and risks.">OGDC · risks</button>
          <button type="button" class="press" data-ex="MEBL|Is MEBL fundamentally strong right now?">MEBL · fundamentals</button>
        </div>
      </div>`;
  }

  const head = `
    <div class="analyze-head">
      <form class="analyze-head-form" id="analyze-head-form">
        <div class="head-field">
          <label for="head-ticker-input">Ticker</label>
          <input id="head-ticker-input" name="symbol" value="${esc(a.symbol || '')}" maxlength="8" autocomplete="off" spellcheck="false">
        </div>
        <div class="head-field head-query-field">
          <label for="head-query-input">Question</label>
          <input id="head-query-input" name="query" value="${esc(a.query || '')}" autocomplete="off">
        </div>
        <button type="submit" class="head-submit press" ${a.phase === 'running' ? 'disabled' : ''}>${a.phase === 'running' ? 'Analyzing…' : 'Re-analyze →'}</button>
        <button type="button" class="head-reset press" id="head-reset" aria-label="Start over">×</button>
      </form>
    </div>`;

  if (a.phase === 'running') {
    return head + `
      <div class="progress-panel">
        <div class="progress-bar-wrap"><div class="progress-bar" id="progress-bar"></div></div>
        <div class="progress-meta">
          <span><span class="count" id="progress-count">0</span> of 5 complete</span>
          <span class="clock" id="clock">00:00</span>
        </div>
        <div class="agent-chips" id="agent-chips">
          ${[['market_data_agent', 'Market data'], ['sentiment_agent', 'Sentiment'], ['risk_agent', 'Risk'], ['macro_agent', 'Macro'], ['rag_agent', 'Quarterly RAG']].map(([id, label]) => `
            <div class="agent-chip" data-id="${id}" data-status="queued"><span class="dot"></span><span>${label}</span><span class="time">—</span></div>
          `).join('')}
        </div>
      </div>`;
  }

  if (a.phase === 'error') {
    return head + renderError(
      'Analysis did not complete',
      'The backend did not return a final report. Check the server and try again.',
      a.error || '',
      [{ id: 'retry-analyze', label: 'Retry' }]
    );
  }

  const data = a.data || {};
  const { body, signal } = parseReport(data.final_report || '');
  const nd = data.numerical_data || {};
  const close = nd.close != null ? Number(nd.close) : null;
  const tradeDate = nd.trade_date || '—';
  const confPct = Math.round((signal.confidence || 0) * 100);
  const glyph = signal.verdict === 'BUY' ? '▲' : signal.verdict === 'SELL' ? '▼' : '●';
  const vLower = signal.verdict.toLowerCase();
  const insights = extractInsights(data.final_report || '');
  const execSummary = extractSection(data.final_report || '', 1);
  const sources = collectSources(data);
  registerSourcesForDrawer(data);

  const hist = data.price_history || [];
  const hasForecast = Array.isArray(data.prediction) && data.prediction.length > 0;
  const hasBand = hasForecast && data.prediction[0]?.lower_close != null && data.prediction[0]?.upper_close != null;

  return head + `
    <div class="result-wrap">
      <div class="verdict-hero animate-scale-in" data-verdict="${vLower}">
        <div class="meta-row">
          <span class="ticker">${esc(a.symbol)}</span>
          <span class="fresh"><span class="fresh-dot"></span>EOD · ${esc(tradeDate)}</span>
          <span>PKR ${fmtNum(close)}</span>
        </div>
        <div class="verdict-word"><span class="glyph">${glyph}</span><span>${signal.verdict}</span></div>
        <p class="reason-line">${execSummary ? esc(firstSentence(execSummary)) : 'Analysis complete.'}</p>
        <div class="verdict-grid">
          <div class="cell"><div class="k">Confidence</div><div class="v" id="conf-value">0%</div><div class="conf-bar"><div class="conf-bar-fill" id="conf-bar"></div></div></div>
          <div class="cell"><div class="k">Horizon</div><div class="v">${signal.holding} days</div></div>
        </div>
        <div class="foot">Research, not advice · EOD · ${sources.length} sources cited below</div>
      </div>

      <div class="chart-panel reveal">
        <div class="chart-panel-head">
          <span>Price · ${hist.length}d history${hasForecast ? ' + 7d forecast' : ''}</span>
          <span class="legend">
            <span><span class="line-solid"></span>Historical</span>
            ${hasForecast ? `<span><span class="line-dashed"></span>Forecast</span>` : ''}
            ${hasBand ? `<span><span class="band-swatch"></span>95% band</span>` : ''}
          </span>
        </div>
        <div class="chart-panel-body"><canvas id="price-canvas" aria-label="Interactive price chart. Hover to see values."></canvas></div>
      </div>

      <div class="tech-indicators reveal">
        <div class="tech-indicators-head">
          <span>Technical indicators</span>
          <span class="meta">From EOD · last session</span>
        </div>
        <div class="tech-grid">
          <div class="tech-card">
            <div class="tech-label"><span>RSI (14)</span><span class="sub">Wilder</span></div>
            <div class="tech-value" id="tech-rsi">—</div>
            <div class="rsi-status" id="rsi-status" data-state="healthy">—</div>
          </div>
          <div class="tech-card">
            <div class="tech-label"><span>MACD</span><span class="sub">12 · 26 · 9</span></div>
            <div class="tech-value" id="tech-macd">—</div>
            <div class="macd-rows">
              <div class="macd-row"><span class="k">Signal</span><span class="v" id="macd-signal">—</span></div>
              <div class="macd-row"><span class="k">Histogram</span><span class="v" id="macd-hist">—</span></div>
            </div>
          </div>
          <div class="tech-card">
            <div class="tech-label"><span>Bollinger</span><span class="sub">20 · 2σ</span></div>
            <div class="tech-value" id="tech-bb">—</div>
            <div class="bb-rows">
              <div class="bb-row"><span class="k">Upper</span><span class="v" id="bb-upper">—</span></div>
              <div class="bb-row hi"><span class="k">Mid</span><span class="v" id="bb-mid">—</span></div>
              <div class="bb-row"><span class="k">Lower</span><span class="v" id="bb-lower">—</span></div>
            </div>
            <div class="bb-position">
              <div class="bb-position-label"><span>Lower</span><span>Upper</span></div>
              <div class="bb-meter">
                <div class="bb-meter-fill" id="bb-fill" style="width:0%"></div>
                <div class="bb-meter-marker" id="bb-marker" style="left:0%"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="insights-head reveal"><span>Key insights</span><span>${insights.length} angles</span></div>
      <div class="insights-grid">
        ${insights.map((i) => `
          <div class="insight-card reveal" data-tone="${i.tone}" ${i.featured ? 'data-featured' : ''}>
            <div class="insight-label"><span class="glyph">${i.glyph}</span>${esc(i.label)}</div>
            <div class="insight-body">${esc(i.body)}</div>
          </div>`).join('')}
      </div>

      <details class="full-report reveal">
        <summary><span class="summary-left"><span>Read full analysis</span><span style="color: var(--color-text-muted); font-weight: 400;">· 9 sections</span></span><span class="chev">▸</span></summary>
        <div class="report-inner"><div class="report-body">${renderReportBody(body)}</div></div>
      </details>

      <div class="sources-row reveal">
        <span class="lbl">Sources</span>
        ${sources.map((s, i) => `<button class="source-chip press" data-source-idx="${i}">[${i + 1}] ${esc(s.label.split('·')[0].trim())}</button>`).join('')}
        <span class="disclaimer">Research, not advice · EOD</span>
      </div>

      <div class="next-actions reveal">
        <a class="next-action primary press" href="#/compare" data-goto-compare="${esc(a.symbol)}">Compare ${esc(a.symbol)} with peers →</a>
        <a class="next-action press" href="#/chat" data-to-chat="Explain the key risk for ${esc(a.symbol)} in more detail.">Ask about this →</a>
      </div>
    </div>`;
};

function collectSources(data) {
  const out = [];
  Object.values(data.data_sources || {}).forEach((v) => {
    (v.source_links || []).forEach((l) => {
      if (l.url && !out.some(o => o.url === l.url)) out.push({ label: l.label || l.url, url: l.url });
    });
  });
  if (!out.length) out.push({
    label: `PSX company page · ${data.symbol || '—'}`,
    url: `https://dps.psx.com.pk/company/${data.symbol || ''}`,
  });
  return out;
}

function registerSourcesForDrawer(data) {
  state.sources = {};
  const ds = data.data_sources || {};
  Object.entries(ds).forEach(([key, v]) => {
    let detail = '';
    if (v.latest_row) detail = `Latest: ${v.latest_row.trade_date} · close ${v.latest_row.close}`;
    else if (v.news_count) detail = `${v.news_count} articles`;
    else if (v.chunks_count) detail = `${v.chunks_count} PDF chunks`;
    else if (v.alerts) detail = `${v.alerts.length} rule alerts`;
    const link = (v.source_links || [])[0];
    if (link) {
      registerSource(key, {
        name: key.replace(/_/g, ' '),
        kind: 'agent',
        detail,
        url: link.url,
        note: v.articles ? v.articles.map(a => `[${a.label}] ${a.headline}`).join(' · ') : '',
      });
    }
  });
}

/* ANALYZE handlers */
function attachAnalyzeHandlers() {
  const form = $('#ask-form');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const symbol = ($('#ticker-input').value || '').trim().toUpperCase() || 'LUCK';
      const query = ($('#query-input').value || '').trim() || `Analyze ${symbol}`;
      runAnalyze(symbol, query);
    });
  }
  $$('[data-ex]').forEach((btn) => btn.addEventListener('click', () => {
    const [t, q] = btn.dataset.ex.split('|');
    runAnalyze(t, q);
  }));

  const headForm = $('#analyze-head-form');
  if (headForm) {
    headForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const symbol = ($('#head-ticker-input').value || '').trim().toUpperCase() || state.analyze.symbol;
      const query = ($('#head-query-input').value || '').trim() || `Analyze ${symbol}`;
      runAnalyze(symbol, query);
    });
  }
  const resetBtn = $('#head-reset');
  if (resetBtn) resetBtn.addEventListener('click', resetAnalyze);

  if (state.analyze.phase === 'done' && state.analyze.data) {
    const data = state.analyze.data;
    const confPct = Math.round((parseReport(data.final_report || '').signal.confidence || 0) * 100);

    const bar = $('#conf-bar');
    if (bar) setTimeout(() => {
      bar.style.width = confPct + '%';
    }, 100);
    const confVal = $('#conf-value');
    if (confVal) setTimeout(() => countUp(confVal, confPct, { suffix: '%', duration: 800 }), 100);

    populateTechIndicators(data);

    const canvas = $('#price-canvas');
    if (canvas) {
      requestAnimationFrame(() => {
        const width = canvas.getBoundingClientRect().width;
        if (width < 60) {
          setTimeout(() => drawPriceChart(canvas, data.price_history || [], data.prediction || [], { animate: true }), 120);
        } else {
          drawPriceChart(canvas, data.price_history || [], data.prediction || [], { animate: true });
        }
      });
    }
  }

  $$('.source-chip').forEach((chip) => chip.addEventListener('click', () => {
    const sources = collectSources(state.analyze.data || {});
    const idx = parseInt(chip.dataset.sourceIdx, 10);
    const source = sources[idx];
    if (source) openSources(Object.keys(state.sources).find(k => state.sources[k].url === source.url));
  }));

  $$('[data-goto-compare]').forEach((el) => el.addEventListener('click', (e) => {
    e.preventDefault();
    state.compare.selected = [el.dataset.gotoCompare];
    location.hash = '#/compare';
  }));
  $$('[data-to-chat]').forEach((el) => el.addEventListener('click', (e) => {
    e.preventDefault();
    state.chat.messages.push({ role: 'user', content: el.dataset.toChat });
    location.hash = '#/chat';
  }));
  $('[data-action="retry-analyze"]')?.addEventListener('click', () => {
    if (state.analyze.symbol) runAnalyze(state.analyze.symbol, state.analyze.query);
  });

  observeReveals();
}

function setAgentChip(id, status, time) {
  const chip = $(`.agent-chip[data-id="${id}"]`);
  if (!chip) return;
  chip.dataset.status = status;
  const t = chip.querySelector('.time');
  if (!t) return;
  if (time) t.textContent = time;
  else if (status === 'running') t.textContent = '…';
  else if (status === 'queued') t.textContent = '—';
  if (status === 'done' && !time) t.textContent = '✓';
}

function updateProgressBar() {
  const chips = $$('.agent-chip');
  if (!chips.length) return;
  const done = chips.filter(c => c.dataset.status === 'done').length;
  const bar = $('#progress-bar'), cnt = $('#progress-count');
  if (bar) bar.style.width = ((done / chips.length) * 100) + '%';
  if (cnt) cnt.textContent = done;
}

function startClock() {
  const start = Date.now();
  clearInterval(startClock._t);
  startClock._t = setInterval(() => {
    const el = $('#clock');
    if (!el) return clearInterval(startClock._t);
    const s = Math.floor((Date.now() - start) / 1000);
    el.textContent = String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
  }, 250);
}

async function runAnalyze(symbol, query) {
  if (state.analyze.phase === 'running') {
    toast('Analysis in progress…');
    return;
  }
  state.analyze = { phase: 'running', symbol, query, data: null, error: null };
  render();
  startClock();

  const onEvent = (evt) => {
    if (evt.type === 'node_update' || evt.node) {
      const id = evt.node || evt.id;
      if (!id) return;
      if (evt.status === 'completed' || evt.status === 'done') {
        setAgentChip(id, 'done', evt.ms || '✓');
        updateProgressBar();
      } else if (evt.status === 'running') {
        setAgentChip(id, 'running');
      } else if (evt.status === 'error') {
        setAgentChip(id, 'error', '✗');
      }
      return;
    }
    if (evt.type === 'complete') {
      state.analyze.data = evt;
      state.analyze.phase = 'done';
      state.analyze.error = null;
      clearInterval(startClock._t);
      render();
    }
  };

  const ok = await api.analyzeStream(symbol, query, onEvent);
  clearInterval(startClock._t);

  if (!ok || state.analyze.phase !== 'done') {
    state.analyze = {
      phase: 'error',
      symbol,
      query,
      data: null,
      error: 'The backend did not return a complete payload within 60 seconds. Verify the server is running at /api/analyze/stream and try again.',
    };
    render();
  }
}

function resetAnalyze() {
  state.analyze = { phase: 'idle', symbol: null, query: '', data: null, error: null };
  render();
}

/* COMPARE */
pages.compare = () => {
  const c = state.compare;
  const banner = `
    <div class="scope-banner reveal">
      <span style="font-family: var(--font-mono); font-weight: 700; color: var(--color-signal-hold);">[T]</span>
      <div><strong>Technical comparison only.</strong> Verdicts reflect price and volume signals. Fundamentals live on each ticker's Analyze page.</div>
    </div>`;
  const head = `
    <div class="page-head">
      <h1>Compare tickers</h1>
      <p class="lede">Rank 2–5 PSX equities side-by-side. Every column is sourced. Descriptive, not prescriptive.</p>
    </div>`;
  const input = `
    <div class="ticker-input-row">
      <span class="lbl">Tickers</span>
      <div class="ticker-chips">
        ${c.selected.map(s => `<span class="ticker-chip" data-sym="${esc(s)}" data-copy="${esc(s)}">${esc(s)}<span class="x" role="button" tabindex="0" aria-label="Remove ${esc(s)}">×</span></span>`).join('')}
      </div>
      <form class="ticker-add" id="ticker-add-form">
        <input id="ticker-add-input" placeholder="Add" maxlength="8" autocomplete="off" spellcheck="false">
        <button type="submit" aria-label="Add ticker">+</button>
      </form>
      <button class="btn-primary press" id="compare-go" type="button" ${c.selected.length < 2 || c.loading ? 'disabled' : ''}>${c.loading ? 'Comparing…' : 'Compare →'}</button>
    </div>`;

  if (c.error) return head + banner + input + renderError('Comparison failed', 'The backend did not return a comparison result.', c.error, [{ id: 'retry-compare', label: 'Retry' }]);
  if (!c.data) return head + banner + input + renderEmpty('Add at least 2 tickers', 'Up to 5 PSX tickers. Technical only. Side-by-side observations, not advice.', []);
  return head + banner + input + renderCompareResults(c.data);
};

/* ✅ FIXED — reads verdicts from data.signals, guards correlation */
function renderCompareResults(data) {
  const sigBySym = {};
  (data.signals || []).forEach(s => {
    sigBySym[s.symbol] = s;
  });

  const cards = data.symbols.map((sym) => {
    const m = data.per_symbol[sym] || {};
    const sig = sigBySym[sym] || {};
    const verdict = sig.verdict || 'HOLD';
    const confidence = sig.confidence ?? 0;
    const vLower = verdict.toLowerCase();
    const glyph = verdict === 'BUY' ? '▲' : verdict === 'SELL' ? '▼' : '●';
    const confPct = Math.round(confidence * 100);
    const chg = m.change_pct ?? 0;
    return `
      <div class="ticker-card reveal" data-verdict="${vLower}" data-copy="${esc(sym)}">
        <div class="tc-head"><span class="tc-symbol">${esc(sym)}</span><span class="tc-fresh recent">● EOD</span></div>
        <div class="tc-verdict"><span>${glyph}</span><span>${esc(verdict)}</span></div>
        <div class="tc-conf">Confidence ${confPct}%</div>
        <div class="tc-price-row"><span class="tc-price">PKR ${fmtNum(m.close)}</span><span class="tc-change ${chg >= 0 ? 'pos' : 'neg'}">${fmtPct(chg)}</span></div>
        <div class="tc-facts">
          <div class="tc-fact"><span>RSI</span><span class="v">${m.rsi != null ? Number(m.rsi).toFixed(1) : '—'}</span></div>
          <div class="tc-fact"><span>R:R</span><span class="v">${m.rr_ratio != null ? Number(m.rr_ratio).toFixed(2) : '—'}</span></div>
        </div>
        <a class="tc-analyze-btn press" href="#" data-to-analyze="${esc(sym)}">Full analysis →</a>
      </div>`;
  }).join('');

  const normalized = `
    <div class="chart-card reveal">
      <div class="chart-card-head"><span>Normalized · indexed to 100</span><span>${data.history_days || 90}d</span></div>
      <div class="chart-card-body"><canvas id="normalized-canvas"></canvas><div class="chart-legend" id="norm-legend"></div></div>
    </div>`;

  const corrData = data.correlation || null;
  const corr = corrData
    ? `<div class="chart-card reveal">
         <div class="chart-card-head"><span>Return correlation · 90d</span><span>Lag-0</span></div>
         <div class="chart-card-body">
           <table class="corr-table">
             <thead><tr><th></th>${data.symbols.map(s => `<th>${esc(s)}</th>`).join('')}</tr></thead>
             <tbody>
               ${data.symbols.map(a => `<tr><th>${esc(a)}</th>${data.symbols.map(b => {
                 if (a === b) return `<td class="diag">—</td>`;
                 const v = corrData[a]?.[b] ?? 0;
                 const cls = Math.abs(v) >= 0.7 ? 'hi' : Math.abs(v) >= 0.4 ? 'mid' : 'lo';
                 return `<td class="${cls}">${Number(v).toFixed(2)}</td>`;
               }).join('')}</tr>`).join('')}
             </tbody>
           </table>
         </div>
       </div>`
    : `<div class="chart-card reveal">
         <div class="chart-card-head"><span>Return correlation · 90d</span><span>Lead-lag</span></div>
         <div class="chart-card-body" style="padding: 24px; color: var(--color-text-muted); font-family: var(--font-mono); font-size: 12px; line-height: 1.7;">
           ${esc(data.lead_lag_summary || 'Correlation matrix not returned by the backend for this comparison.')}
         </div>
       </div>`;

  const summary = `
    <div class="summary-block reveal">
      <h3>Comparative observation</h3>
      ${String(data.comparative_summary || '').split('\n').filter(Boolean).map(p => `<p>${esc(p)}</p>`).join('')}
      <div class="summary-foot">Descriptive only · not advice · canonical verdicts</div>
    </div>
    <div class="next-actions reveal">
      <a class="next-action primary press" href="#/analyze" data-to-analyze-top>Analyze top pick →</a>
      <a class="next-action press" href="#/screen">See today's shortlist →</a>
    </div>`;

  return `<div class="compare-cards">${cards}</div><div class="compare-charts">${normalized}${corr}</div>${summary}`;
}

function attachCompareHandlers() {
  $('#ticker-add-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    const raw = ($('#ticker-add-input').value || '').trim().toUpperCase();
    if (!raw || state.compare.selected.length >= 5 || state.compare.selected.includes(raw)) return;
    state.compare.selected.push(raw);
    state.compare.data = null;
    state.compare.error = null;
    render();
  });
  $$('.ticker-chip .x').forEach((x) => {
    const handler = () => {
      const sym = x.closest('.ticker-chip').dataset.sym;
      state.compare.selected = state.compare.selected.filter(s => s !== sym);
      state.compare.data = null;
      state.compare.error = null;
      render();
    };
    x.addEventListener('click', handler);
    x.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') handler();
    });
  });
  $$('[data-add]').forEach((btn) => btn.addEventListener('click', () => {
    const sym = btn.dataset.add;
    if (!state.compare.selected.includes(sym) && state.compare.selected.length < 5) {
      state.compare.selected.push(sym);
      state.compare.data = null;
      state.compare.error = null;
      render();
    }
  }));
  $('#compare-go')?.addEventListener('click', runCompare);
  $$('[data-to-analyze]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault();
    const sym = a.dataset.toAnalyze;
    runAnalyze(sym, `Analyze ${sym}`);
    location.hash = '#/analyze';
  }));
  $('[data-to-analyze-top]')?.addEventListener('click', (e) => {
    e.preventDefault();
    const top = state.compare.data?.symbols?.[0];
    if (top) {
      runAnalyze(top, `Analyze ${top}`);
      location.hash = '#/analyze';
    }
  });
  $('[data-action="retry-compare"]')?.addEventListener('click', runCompare);

  if (state.compare.data) {
    const norm = $('#normalized-canvas');
    if (norm) requestAnimationFrame(() => drawNormalizedChart(norm, state.compare.data.symbols, state.compare.data.history || {}, { animate: true }));
    const legend = $('#norm-legend');
    if (legend) {
      const colors = ['--color-signal-buy', '--color-text', '--color-signal-sell', '--color-signal-hold', '--color-text-muted'];
      legend.innerHTML = state.compare.data.symbols.map((sym, i) => {
        const closes = _historyToCloseArray((state.compare.data.history || {})[sym]);
        if (closes.length < 2) return '';
        const base = closes[0];
        const last = closes[closes.length - 1];
        if (!base) return '';
        const pct = ((last - base) / base) * 100;
        const color = cssVar(colors[i % colors.length]);
        return `<div class="item"><span class="swatch" style="background:${color}"></span><span>${esc(sym)}</span><span class="val" style="color:${pct >= 0 ? 'var(--color-positive)' : 'var(--color-negative)'}">${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</span></div>`;
      }).join('');
    }
  }
  observeReveals();
}

/* ✅ FIXED — defensive try/catch so render always fires */
async function runCompare() {
  if (state.compare.selected.length < 2 || state.compare.loading) return;
  state.compare.loading = true;
  state.compare.error = null;
  render();

  let data = null;
  try {
    data = await api.compare(state.compare.selected);
  } catch (err) {
    console.error('[runCompare] failed:', err);
  }

  state.compare.loading = false;
  if (data) {
    state.compare.data = data;
    state.compare.error = null;
  } else {
    state.compare.error = 'Backend did not respond within 90 seconds.';
    state.compare.data = null;
  }
  render();
}

/* SCREEN */
pages.screen = () => {
  const s = state.screen;
  const head = `
    <div class="page-head">
      <h1>Today's shortlist</h1>
      <p class="lede">Pure-math screen over the PSX universe. No LLM, no narrative. Ranked by composite technical score, filtered for liquidity and risk-reward.</p>
    </div>`;
  const controls = `
    <div class="screen-controls">
      <div class="left">
        <div class="profile-switcher" role="group">
          ${['Quantitative Finance', ''].map(p => `<button type="button" class="press" data-profile="${p}" aria-pressed="${s.profile === p}">${p}</button>`).join('')}
        </div>
        ${s.data ? `
          <div class="pass-rate">
            ${s.data.screened_universe_size?.toLocaleString() || '—'} screened · ${s.data.candidates || 0} passed
            <div class="bar"><div class="bar-fill" id="pass-fill"></div></div>
          </div>` : ''}
      </div>
      <div class="right">
        <button class="btn-primary press" id="screen-refresh" type="button" ${s.loading ? 'disabled' : ''}>${s.loading ? 'Running…' : (s.data ? 'Refresh' : 'Run screen')}</button>
      </div>
    </div>`;

  if (s.error) return head + controls + renderError('Screen failed', 'The backend did not return a screener result.', s.error, [{ id: 'retry-screen', label: 'Retry' }]);
  if (s.loading && !s.data) return head + controls + renderLoading('Running the screen…', 'First run computes ~1,058 symbols. This takes 30–60 seconds.');
  if (!s.data) return head + controls + renderEmpty("Run the screen to see today's shortlist", 'First run takes 30–60 seconds to compute the universe.', [{ id: 'run-screen', label: 'Run screen' }]);

  const rows = [...(s.data.top_picks || [])].sort((a, b) => {
    const dir = s.sortDir === 'asc' ? 1 : -1;
    const va = a[s.sortKey] ?? 0, vb = b[s.sortKey] ?? 0;
    return va > vb ? dir : va < vb ? -dir : 0;
  });

  const cols = [['rank', 'Rank'], ['symbol', 'Symbol'], ['grade', 'Grade'], ['composite_score', 'Score'], ['entry_price', 'Entry'], ['rr_ratio', 'R:R'], ['return_3m_pct', 'Δ3M'], ['verdict', 'Verdict']];

  const table = `
    <div class="data-table-wrap reveal">
      <table class="data-table">
        <thead><tr>${cols.map(([k, l]) => `<th data-sort="${k}">${l}</th>`).join('')}<th></th></tr></thead>
        <tbody>
          ${rows.map((p) => {
            // Screener = "buy-today" shortlist; default BUY when verdict missing.
            const verdict = p.verdict || (p.grade === 'C' ? 'HOLD' : 'BUY');
            const glyph = verdict === 'BUY' ? '▲' : verdict === 'SELL' ? '▼' : '●';
            const vColor = verdict === 'BUY' ? 'var(--color-signal-buy)' : verdict === 'SELL' ? 'var(--color-signal-sell)' : 'var(--color-signal-hold)';
            return `
              <tr data-pick="${esc(p.symbol)}">
                <td>#${p.rank}</td><td>${esc(p.symbol)}</td><td>${esc(p.grade || '—')}</td>
                <td class="pos">${p.composite_score ?? '—'}<span class="score-bar"><span class="score-bar-fill" data-score="${p.composite_score || 0}"></span></span></td>
                <td>${fmtNum(p.entry_price)}</td>
                <td>${p.rr_ratio != null ? Number(p.rr_ratio).toFixed(2) + ':1' : '—'}</td>
                <td class="${(p.return_3m_pct ?? 0) >= 0 ? 'pos' : 'neg'}">${fmtPct(p.return_3m_pct, 1)}</td>
                <td style="color: ${vColor}; font-weight: 700;">${glyph} ${esc(verdict)}</td>
                <td style="text-align:right;color:var(--color-text-muted)">${s.expanded.has(p.symbol) ? '▾' : '▸'}</td>
              </tr>
              ${s.expanded.has(p.symbol) ? `
                <tr class="pick-expand"><td colspan="${cols.length + 1}">
                  <div class="expand-inner">
                    <div class="chart-mini"><canvas data-expand-chart="${esc(p.symbol)}"></canvas></div>
                    <div>
                      <div class="rationale">${esc(p.screen_observation || '—')}</div>
                      <div class="tags">${(p.tags || []).map((t, i) => `<span class="tag${i === 0 ? ' accent' : ''}">${esc(t)}</span>`).join('')}</div>
                    </div>
                  </div>
                </td></tr>` : ''}`;
          }).join('')}
        </tbody>
      </table>
    </div>`;

  const breakdown = s.data.skipped_breakdown ? `
    <div class="rejection-breakdown reveal">
      <strong>Why the list is short</strong><br>
      ${Object.entries(s.data.skipped_breakdown).filter(([, v]) => v > 0).map(([k, v]) => `${esc(k.replace(/_/g, ' '))}: ${v}`).join(' · ')}
    </div>` : '';

  return head + controls + table + breakdown;
};

function attachScreenHandlers() {
  $$('[data-profile]').forEach((btn) => btn.addEventListener('click', () => {
    state.screen.profile = btn.dataset.profile;
    state.screen.data = null;
    state.screen.error = null;
    savePreferences();
    render();
  }));
  $('#screen-refresh')?.addEventListener('click', () => runScreen(!!state.screen.data));
  $('#screen-run')?.addEventListener('click', () => runScreen(false));
  $('[data-action="run-screen"]')?.addEventListener('click', () => runScreen(false));
  $('[data-action="retry-screen"]')?.addEventListener('click', () => runScreen(false));
  $$('.data-table thead th[data-sort]').forEach((th) => th.addEventListener('click', () => {
    const k = th.dataset.sort;
    if (state.screen.sortKey === k) state.screen.sortDir = state.screen.sortDir === 'asc' ? 'desc' : 'asc';
    else {
      state.screen.sortKey = k;
      state.screen.sortDir = 'asc';
    }
    render();
  }));
  $$('[data-pick]').forEach((tr) => tr.addEventListener('click', () => {
    const sym = tr.dataset.pick;
    if (state.screen.expanded.has(sym)) state.screen.expanded.delete(sym);
    else state.screen.expanded.add(sym);
    render();
  }));
  $$('[data-expand-chart]').forEach((canvas) => {
    const sym = canvas.dataset.expandChart;
    const hist = (state.screen.data?.history?.[sym]) || null;
    requestAnimationFrame(() => {
      if (!hist || !hist.length) {
        const { ctx, w, h } = setupCanvas(canvas, 120);
        ctx.fillStyle = cssVar('--color-text-muted');
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No history', w / 2, h / 2);
        return;
      }
      const { ctx, w, h } = setupCanvas(canvas, 120);
      const pad = 8, iw = w - pad * 2, ih = h - pad * 2;
      const vals = hist.map(d => d.close);
      const min = Math.min(...vals), max = Math.max(...vals);
      const range = (max - min) || 1;
      ctx.strokeStyle = cssVar('--color-signal-buy');
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = pad + (iw * i) / Math.max(1, vals.length - 1);
        const y = pad + ih - ((v - min) / range) * ih;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  });

  $$('[data-score]').forEach((el) => {
    const target = parseInt(el.dataset.score, 10);
    requestAnimationFrame(() => {
      el.style.width = Math.min(100, Math.max(0, target)) + '%';
    });
  });

  if (state.screen.data) {
    const fill = $('#pass-fill');
    const screened = state.screen.data.screened_universe_size || 0;
    const passed = state.screen.data.candidates || 0;
    if (fill && screened > 0) {
      const pct = Math.max(2, (passed / screened) * 100);
      requestAnimationFrame(() => {
        fill.style.width = pct + '%';
      });
    }
  }
  observeReveals();
}

async function runScreen(forceRefresh = false) {
  if (state.screen.loading) return;
  state.screen.loading = true;
  state.screen.error = null;
  render();
  const data = await api.quant(10, state.screen.profile, forceRefresh);
  state.screen.loading = false;
  if (data) {
    state.screen.data = data;
    state.screen.error = null;
  } else {
    state.screen.error = 'Backend did not respond within 90 seconds.';
  }
  render();
}

/* CHAT */
pages.chat = () => {
  const c = state.chat;
  return `
    <div class="page-head">
      <h1>Chat</h1>
      <p class="lede">Ask about any PSX ticker or concept. Answers cite sources and stay within PSX equity scope.</p>
    </div>
    <div class="chat-layout">
      <aside class="thread-list">
        <button class="new-thread-btn press" id="new-thread">+ New thread</button>
        ${c.threads.map(t => `<div class="thread" data-thread="${esc(t.id)}" ${t.id === c.currentId ? 'aria-current="true"' : ''}>${esc(t.title)}<div class="meta">${esc(t.when)}</div></div>`).join('')}
      </aside>
      <div class="chat-panel">
        <div class="chat-messages" id="chat-messages">${renderChatMessages()}</div>
        <form class="chat-input-row" id="chat-form">
          <input id="chat-input" placeholder="Ask about a PSX ticker or concept…" autocomplete="off">
          <button type="submit" class="press" id="chat-send" ${c.streaming ? 'disabled' : ''}>${c.streaming ? 'Thinking…' : 'Send →'}</button>
        </form>
      </div>
    </div>`;
};

function renderChatMessages() {
  const c = state.chat;
  if (!c.messages.length) {
    return `
      <div class="chat-empty">
        <h3>Ask about a PSX ticker or concept</h3>
        <p>Prices, news, financials, risks — every answer cites its source. Research, not advice.</p>
        <div class="suggestions">
          <button type="button" class="press" data-chat-suggest="What's LUCK trading at and any recent news?">What's LUCK trading at?</button>
          <button type="button" class="press" data-chat-suggest="Compare OGDC and PPL on risk and momentum.">Compare OGDC and PPL</button>
          <button type="button" class="press" data-chat-suggest="What is RSI and how do I read it?">What is RSI?</button>
        </div>
      </div>`;
  }
  return c.messages.map((m, i) => renderMessage(m, i)).join('');
}

function renderMessage(m, idx) {
  if (m.role === 'user') return `<div class="msg-user">${esc(m.content)}</div>`;
  if (m.role === 'error') return `<div class="msg-assistant"><div class="msg-error">${esc(m.content)}</div></div>`;
  const tools = m.tools || [];
  const totalTime = tools.reduce((acc, t) => acc + parseFloat(t.time || '0'), 0).toFixed(1);
  const toolSummary = tools.length > 0 ? `
    <button class="tool-summary press" data-tool-toggle data-msg-idx="${idx}" aria-expanded="${m._toolsOpen ? 'true' : 'false'}">
      <span class="caret">▸</span><span>${tools.length} tool${tools.length === 1 ? '' : 's'} ran · ${totalTime}s</span>
    </button>` : '';
  const toolDetail = tools.length > 0 && m._toolsOpen ? `
    <div class="tool-detail">${tools.map(t => `<div class="tool-row" data-status="${esc(t.status)}"><span class="dot"></span><span class="name">${esc(t.name)}</span><span class="time">${esc(t.time || '')}</span></div>`).join('')}</div>` : '';
  const chart = m.chart ? `<div class="chat-mini-chart"><canvas data-chat-chart-idx="${idx}"></canvas><div class="cap">${esc(m.chart.symbol)} · 90d EOD</div></div>` : '';
  const signal = m.signal && m.signal.applicable ? `<div class="signal-chip" data-v="${esc(m.signal.verdict)}">${m.signal.verdict === 'BUY' ? '▲' : m.signal.verdict === 'SELL' ? '▼' : '●'} ${esc(m.signal.verdict)} <span class="conf">conf ${m.signal.confidence.toFixed(2)}</span></div>` : '';
  const typing = m._streaming && !m.content ? `<span class="typing-dots"><span></span><span></span><span></span></span>` : '';
  return `<div class="msg-assistant">${toolSummary}${toolDetail}${chart}<div>${mdInline(m.content || '')}${typing}</div>${signal}</div>`;
}

function attachChatHandlers() {
  const form = document.getElementById('chat-form');
  if (form && !form._chatWired) {
    form._chatWired = true;
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (state.chat.streaming) return;
      const text = (document.getElementById('chat-input').value || '').trim();
      if (!text) return;
      document.getElementById('chat-input').value = '';
      await runChat(text);
    });
  }

  const newThread = document.getElementById('new-thread');
  if (newThread && !newThread._chatWired) {
    newThread._chatWired = true;
    newThread.addEventListener('click', () => {
      const id = 't' + (state.chat.threads.length + 1);
      state.chat.threads.unshift({ id, title: 'New conversation', when: 'now' });
      state.chat.currentId = id;
      state.chat.messages = [];
      render();
    });
  }

  document.querySelectorAll('.thread').forEach((t) => {
    if (t._chatWired) return;
    t._chatWired = true;
    t.addEventListener('click', () => {
      state.chat.currentId = t.dataset.thread;
      render();
    });
  });

  document.querySelectorAll('[data-chat-suggest]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.getElementById('chat-input').value = btn.dataset.chatSuggest;
      document.getElementById('chat-form').dispatchEvent(new Event('submit', { cancelable: true }));
    });
  });

  attachChatDynamicHandlers();
}

function attachChatDynamicHandlers() {
  document.querySelectorAll('[data-tool-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.msgIdx, 10);
      const msg = state.chat.messages[idx];
      if (msg && msg.role === 'assistant') {
        msg._toolsOpen = !msg._toolsOpen;
        renderChatOnly();
      }
    });
  });

  document.querySelectorAll('[data-chat-chart-idx]').forEach((canvas) => {
    const idx = parseInt(canvas.dataset.chatChartIdx, 10);
    const msg = state.chat.messages[idx];
    const chart = msg?.chart;
    requestAnimationFrame(() => {
      if (!chart || !chart.values || !chart.values.length) {
        const { ctx, w, h } = setupCanvas(canvas, 100);
        ctx.strokeStyle = cssVar('--color-text-muted');
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(12, h / 2);
        ctx.lineTo(w - 12, h / 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = cssVar('--color-text-muted');
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Chart data not available', w / 2, h / 2 - 14);
        return;
      }
      const { ctx, w, h } = setupCanvas(canvas, 100);
      const pad = { l: 8, r: 8, t: 8, b: 8 };
      const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
      const vals = chart.values;
      const min = Math.min(...vals), max = Math.max(...vals);
      const range = (max - min) || 1;
      ctx.strokeStyle = cssVar('--color-text');
      ctx.lineWidth = 1.4;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = pad.l + (iw * i) / Math.max(1, vals.length - 1);
        const y = pad.t + ih - ((v - min) / range) * ih;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  });
}

async function runChat(text) {
  state.chat.streaming = true;
  state.chat.messages.push({ role: 'user', content: text });
  const assistantMsg = { role: 'assistant', content: '', tools: [], chart: null, signal: null, _streaming: true };
  state.chat.messages.push(assistantMsg);
  render();

  const onEvent = (evt) => {
    if (evt.type === 'tool') {
      const existing = assistantMsg.tools.find(t => t.name === evt.tool);
      if (existing) existing.status = evt.status || 'running';
      else assistantMsg.tools.push({ name: evt.tool, status: evt.status || 'running', time: '' });
      renderChatOnly();
      return;
    }
    if (evt.type === 'token') {
      assistantMsg.content += evt.text || '';
      renderChatOnly();
      return;
    }
    if (evt.type === 'chart' && evt.chart) {
      assistantMsg.chart = {
        symbol: evt.chart.ticker || evt.chart.symbol,
        dates: evt.chart.dates || [],
        values: evt.chart.close || [],
      };
      renderChatOnly();
      return;
    }
    if (evt.type === 'done') {
      if (evt.signal_applicable && evt.signal) {
        assistantMsg.signal = { applicable: true, verdict: evt.signal, confidence: evt.confidence || 0 };
      }
      assistantMsg._streaming = false;
    }
  };

  const ok = await api.chatStream(state.chat.currentId, text, onEvent);
  state.chat.streaming = false;

  if (!ok && !assistantMsg.content) {
    assistantMsg._streaming = false;
    state.chat.messages.pop();
    state.chat.messages.push({ role: 'error', content: 'Backend did not respond. Check that the server is running at /api/chat/stream and try again.' });
  }
  render();
}

function renderChatOnly() {
  const container = document.getElementById('chat-messages');
  if (!container) return;
  container.innerHTML = renderChatMessages();
  attachChatDynamicHandlers();
  container.scrollTop = container.scrollHeight;
}

/* ── Router ───────────────────────────────────────────────── */
function currentRoute() {
  const hash = location.hash.replace(/^#\/?/, '');
  const [name] = hash.split('/');
  return ['analyze', 'compare', 'screen', 'chat'].includes(name) ? name : 'analyze';
}

/* ✅ FIXED — inner try/catch so a broken page can't strand the loading state */
function render() {
  return safeRun(() => {
    const route = currentRoute();
    state.route = route;
    $$('[data-route]').forEach((a) => a.setAttribute('aria-current', a.dataset.route === route ? 'page' : 'false'));
    const page = $('#page');
    if (!page) return;

    let html;
    try {
      html = pages[route]();
    } catch (err) {
      console.error('[render] page builder threw:', err);
      showErrorBoundary(err);
      html = renderError(
        'Could not render this page',
        'The page data loaded but rendering failed. See the console for the stack trace.',
        err?.message || ''
      );
    }
    page.innerHTML = html;

    if (route === 'analyze') attachAnalyzeHandlers();
    if (route === 'compare') attachCompareHandlers();
    if (route === 'screen') attachScreenHandlers();
    if (route === 'chat') attachChatHandlers();
    attachMicroInteractions(page);
    updateSourcesCount();
    updateBreadcrumb();
    updateScrollProgress();
  });
}

/* ── Boot ──────────────────────────────────────────────────── */
installErrorBoundary();

window.addEventListener('hashchange', () => {
  render();
  window.scrollTo(0, 0);
});
window.addEventListener('scroll', updateScrollProgress, { passive: true });
window.addEventListener('resize', updateScrollProgress, { passive: true });

$('#sources-btn').addEventListener('click', () => openSources());
$('#sheet-close').addEventListener('click', closeSources);
$('#sheet-backdrop').addEventListener('click', closeSources);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#sheet').hidden) closeSources();
});
document.querySelector('[data-crumb-back]')?.addEventListener('click', () => {
  if (window.history.length > 1) window.history.back();
});

initTheme();
initSidebar();
initScrollReveal();
observeReveals();

(async () => {
  try {
    const res = await withTimeout(fetch('/health'), 5000);
    if (res.ok) {
      $('#api-status').textContent = 'API online';
      setConnectionStatus(true);
    } else {
      throw new Error();
    }
  } catch {
    $('#api-status').textContent = 'Offline';
    setConnectionStatus(false);
  }
})();

render();