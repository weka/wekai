// Request timeline renderer: one thin bar per request on a shared time axis.
// Used verbatim by the benchmark report's Timeline view, the standalone
// `wekai benchmark timeline` page and the live view served during a run.
//
// Only the visible rows are painted (virtual vertical scrolling over a single
// canvas), so tens of thousands of requests cost the same as a screenful.
//
// Data (produced by benchmark/timeline.go):
//   arm  = { name, t0 (epoch ms), span (ms), reqs: [req], guids?: [instance ids] }
//   Layout: one row GROUP per series (req.r = the series' group index, in order of
//   first request); requests of a series that overlap in time (sub-agent fan-out)
//   are packed into sub-rows (req.u) inside the group. Groups are separated by a
//   thin gap.
//   req  = { r group, u sub-row, s series, n turn, gi/gu instance, t start ms, e end-to-end ms, f ttft ms, i/c/o tokens,
//            x flag bits, src [gpu,host,ext,computed], rc [recompute], tm [timings ms],
//            g [kind, ms, kind, ms ...] }
//   kind = { id, key, label, color }
var WekaiTimeline = (function () {
  "use strict";

  var FLAG_ERR = 1, FLAG_COLD = 2, FLAG_EMPTY = 4, FLAG_TIMINGS = 8, FLAG_SRC = 16, FLAG_INFL = 32, FLAG_TRUNC = 64, FLAG_SESSION = 128, FLAG_HOT = 256, FLAG_WAITING = 512;
  var INFLIGHT_KIND = 10, WAIT_CLIENT = 11, WAITED = 13;
  var WAIT_KINDS = { 0: 1, 1: 1, 11: 1, 12: 1, 14: 1, 15: 1 };
  // One word for the UI: a series is a simulated user lane (rows on the timeline); it runs one replayed session after
  // another (purple ticks mark session boundaries). Admission adds series, --ttft-retire retires them.
  var SERIES_TIP = "Active sessions: replayed user sessions in progress. Each timeline row is one simulated user replaying one session after another (purple ticks mark session boundaries). Admission adds rows; --ttft-retire retires them.";   // kinds the "merge waits" toggle collapses into WAITED
  var AXIS_H = 34, RIGHT = 14, GAP = 2;
  var ROW_HEIGHTS = [1, 2, 3, 4, 6, 10, 14, 20];

  function fmtDur(ms) {
    if (ms < 1000) return Math.round(ms) + "ms";
    var s = ms / 1000;
    if (s < 60) return s.toFixed(s < 10 ? 2 : 1) + "s";
    var m = Math.floor(s / 60), r = Math.round(s - m * 60);
    if (m < 60) return m + "m" + (r ? r + "s" : "");
    var h = Math.floor(m / 60);
    return h + "h" + (m % 60 ? (m % 60) + "m" : "");
  }
  function fmtElapsed(ms, step) {
    if (step < 1000) return (ms / 1000).toFixed(step < 100 ? 2 : 1) + "s";
    var s = Math.round(ms / 1000);
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m" + (s % 60 ? (s % 60) + "s" : "");
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h + "h" + (m ? m + "m" : "");
  }
  function pad2(n) { return n < 10 ? "0" + n : "" + n; }
  function fmtClock(epoch, withSec) {
    var d = new Date(epoch);
    return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + (withSec ? ":" + pad2(d.getSeconds()) : "");
  }
  function fmtTok(n) { return n >= 10000 ? (n / 1000).toFixed(1) + "k" : String(n); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  var STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600].map(function (s) { return s * 1000; });
  function pickStep(spanMs, plotW) {
    var want = Math.max(2, plotW / 120);
    for (var i = 0; i < STEPS.length; i++) if (spanMs / STEPS[i] <= want) return STEPS[i];
    return STEPS[STEPS.length - 1];
  }

  function create(host, opts) {
    opts = opts || {};
    var kinds = opts.kinds || [];
    var colors = [], labels = [];
    kinds.forEach(function (k) { colors[k.id] = k.color; labels[k.id] = k.label; });
    var maxReqs = opts.maxRequests || 250000;

    host.classList.add("wtl");
    host.innerHTML =
      '<div class="wtl-bar">' +
      '<button class="wtl-expand" title="Fill the window with the timeline (Esc or Back to return)">&#x2922; Expand</button>' +
      '<label class="wtl-paneswrap" hidden title="Stack several timelines, one arm each; they share zoom and replay">Panes <select class="wtl-panes"></select></label>' +
      '<span class="wtl-play" title="Replay the run: requests appear as they started and fill in stage by stage">' +
      '<button class="wtl-restart" title="Replay from the start">&#x23EE;</button>' +
      '<button class="wtl-playbtn" title="Play / pause">&#x25B6;</button>' +
      '<button class="wtl-stop" title="Stop the replay and show the whole run" hidden>&#x25A0;</button>' +
      '<input type="range" class="wtl-speed" min="0" max="10" step="1" value="5" title="Replay speed">' +
      '<span class="wtl-speedv"></span><span class="wtl-playt"></span></span>' +
      '<label class="wtl-armwrap" hidden>Arm <select class="wtl-arm"></select></label>' +
      '<label>Row height <select class="wtl-rowh"></select></label>' +
      '<label title="Collapse client wait, engine queue and unattributed wait into one colour"><input type="checkbox" class="wtl-merge"> Merge waits</label>' +
      '<button class="wtl-reset" hidden>Reset zoom</button>' +
      '<label class="wtl-followwrap" hidden><input type="checkbox" class="wtl-follow"> Follow latest</label>' +
      '<span class="wtl-panels">Show' +
      '<button data-p="state" title="Run state: sessions, in flight, gate">State</button>' +
      '<button data-p="legend" title="Colour legend">Legend</button>' +
      '<button data-p="act" title="Active sessions / in flight over time">Activity</button>' +
      '<button data-p="rate" title="Input and output tokens/s over time">Tokens/s</button>' +
      '<button data-p="ttft" title="TTFT over time">TTFT</button></span>' +
      '<span class="wtl-at"></span>' +
      '<span class="wtl-info"></span>' +
      '<span class="wtl-hint">Drag to zoom &middot; double-click or Esc to reset &middot; Ctrl+wheel zooms, Shift+wheel pans</span>' +
      '</div><div class="wtl-state"></div><div class="wtl-legend"></div>' +
      '<canvas class="wtl-strip wtl-strip-act" hidden></canvas><canvas class="wtl-strip wtl-strip-rate" hidden></canvas><canvas class="wtl-strip wtl-strip-ttft" hidden></canvas>' +
      '<div class="wtl-scroll"><div class="wtl-content"><canvas class="wtl-canvas"></canvas><canvas class="wtl-overlay"></canvas></div><div class="wtl-empty" hidden>No requests yet</div></div>' +
      '<div class="wtl-tip"></div>';
    var $ = function (c) { return host.querySelector(c); };
    if (opts.paneTitle) {
      // A split pane: the split's own bar holds expand, replay and the arm choice.
      host.classList.add("wtl-in-split");
      var pt = document.createElement("span"); pt.className = "wtl-panetitle"; pt.textContent = opts.paneTitle; pt.title = opts.paneTitle;
      $(".wtl-bar").insertBefore(pt, $(".wtl-bar").firstChild);
    }
    var elArm = $(".wtl-arm"), elRowH = $(".wtl-rowh"), elReset = $(".wtl-reset"), elFollow = $(".wtl-follow");
    var elInfo = $(".wtl-info"), elLegend = $(".wtl-legend"), elScroll = $(".wtl-scroll");
    var elContent = $(".wtl-content"), canvas = $(".wtl-canvas"), tip = $(".wtl-tip"), elEmpty = $(".wtl-empty");
    var ctx = canvas.getContext("2d");
    var elState = $(".wtl-state");
    var stripRate = $(".wtl-strip-rate"), stripTTFT = $(".wtl-strip-ttft"), stripAct = $(".wtl-strip-act");
    var elExpand = $(".wtl-expand"), expanded = false, elAt = $(".wtl-at");
    // Panes of one split view share zoom, expand and replay through opts.sync
    // (createSplit); a lone timeline has no sync and expands itself.
    var syncKey = {}, expandHost = opts.expandHost || host;
    function emit(kind, v) { if (opts.sync) opts.sync.emit(syncKey, kind, v); }
    // Expand: the timeline (with its own bar: arm, row height, zoom) takes the whole window; Back / Esc returns.
    function setExpanded(on, fromSync) {
      expanded = on;
      expandHost.classList.toggle("wtl-expanded-host", on);
      host.classList.toggle("wtl-x", on);
      document.documentElement.classList.toggle("wtl-expanded-page", on);
      if (!fromSync) emit("expand", on);
      elExpand.innerHTML = on ? "&#x2715; Back" : "&#x2922; Expand";
      elExpand.title = on ? "Back to the full report (Esc)" : "Fill the window with the timeline (Esc or Back to return)";
      applyPanels();
      sizeKey = ""; resize();
    }
    // Panels around the rows. The report view shows them all; Expand starts with
    // none, so the rows get the window and everything else is the one-line
    // readout in the bar. Each mode remembers its own toggles.
    var PANELS = ["state", "legend", "act", "rate", "ttft"];
    var panelsOn = { normal: { state: 1, legend: 1, act: 1, rate: 1, ttft: 1 }, expanded: {} };
    var panelBtns = host.querySelectorAll(".wtl-panels button");
    function applyPanels() {
      var on = panelsOn[expanded ? "expanded" : "normal"];
      PANELS.forEach(function (p) { host.classList.toggle("wtl-hide-" + p, !on[p]); });
      for (var i = 0; i < panelBtns.length; i++) panelBtns[i].classList.toggle("on", !!on[panelBtns[i].getAttribute("data-p")]);
      renderAt();
    }
    for (var pb = 0; pb < panelBtns.length; pb++) panelBtns[pb].addEventListener("click", function (e) {
      var on = panelsOn[expanded ? "expanded" : "normal"], p = e.currentTarget.getAttribute("data-p");
      on[p] = !on[p]; applyPanels(); sizeKey = ""; resize();
    });
    elExpand.addEventListener("click", function () { setExpanded(!expanded); });
    window.addEventListener("keydown", function (e) { if (e.key === "Escape" && expanded && !isZoomed()) setExpanded(false); });
    var hoverT = null;   // arm time under the cursor (offline state line)
    var STRIP_H = 56;
    var elMerge = $(".wtl-merge"), overlay = $(".wtl-overlay"), octx = overlay.getContext("2d");
    var merge = /[?&]merge=1/.test(location.search);
    elMerge.checked = merge;

    function buildLegend() {
      elLegend.innerHTML = kinds.filter(function (k) {
        if (k.id === INFLIGHT_KIND) return !!opts.live;
        if (k.id === WAITED) return merge;
        if (WAIT_KINDS[k.id]) return !merge;
        return true;
      }).map(function (k) {
        return '<span><i style="background:' + k.color + '"></i>' + esc(k.label) + "</span>";
      }).join("") + '<span><i style="background:rgba(124,150,190,0.35)"></i>think pause (idle by design)</span>';
    }
    buildLegend();
    ROW_HEIGHTS.forEach(function (h) { var o = document.createElement("option"); o.value = h; o.textContent = h + "px"; elRowH.appendChild(o); });

    var arms = [], cur = 0;
    var rowH = 3, userRowH = false;
    var vMin = null, vMax = null;           // null = full range
    var nowMs = 0, nowAt = 0;               // live clock: arm-time at wall-clock nowAt
    var W = 0, H = 0, plotL = 8, plotW = 100;
    var hover = null, drag = null, hoverGroup = -1;

    function arm() { return arms[cur]; }
    function curNow() { return nowMs + (performance.now() - nowAt); }
    function fullRange() {
      var a = arm(); if (!a) return [0, 1000];
      var hi = Math.max(a.span || 0, opts.live ? curNow() : 0, 1000);
      return [0, hi * 1.005];
    }
    function range() {
      var f = fullRange();
      if (vMin === null) return f;
      if (opts.live && elFollow.checked) { var w = vMax - vMin, hi = f[1]; return [hi - w, hi]; }
      return [vMin, vMax];
    }
    function isZoomed() { return vMin !== null; }
    function mapX(t, r) { return plotL + (t - r[0]) / (r[1] - r[0]) * plotW; }
    function unmapX(px, r) { return r[0] + (px - plotL) / plotW * (r[1] - r[0]); }

    function addToGroup(a, q) {
      var g = a.groups[q.r];
      if (!g) { for (var k = a.groups.length; k <= q.r; k++) a.groups[k] = newGroup(); g = a.groups[q.r]; }
      var n = g.reqs.length;
      if (n && q.t < g.reqs[n - 1].t) g.unsorted = true;
      g.reqs.push(q);
      if (q.e + (q.tp || 0) > g.maxE) g.maxE = q.e + (q.tp || 0);
      if (q.x & FLAG_SESSION) { g.bounds.push(q.t); if (g.bounds.length > 1 && q.t < g.bounds[g.bounds.length - 2]) g.unsorted = true; }
      if ((q.u || 0) + 1 > g.subs) g.subs = (q.u || 0) + 1;
      a.dirty = true; a.ver = (a.ver || 0) + 1;
    }
    function newGroup() { return { reqs: [], infl: [], subs: 1, maxE: 0, bounds: [], unsorted: false }; }
    // ensureSorted keeps a group's requests ordered by start so a time window can
    // be found by binary search instead of scanning every request.
    function ensureSorted(g) {
      if (!g.unsorted) return;
      g.reqs.sort(function (x, y) { return x.t - y.t; });
      g.bounds.sort(function (x, y) { return x - y; });
      g.unsorted = false;
    }
    // firstAtOrAfter: first index in g.reqs (sorted by start) with start >= t.
    function firstAtOrAfter(list, t) {
      var lo = 0, hi = list.length;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (list[mid].t < t) lo = mid + 1; else hi = mid; }
      return lo;
    }
    function indexArm(a) {
      a.groups = [];
      a.reqs.forEach(function (q) { addToGroup(a, q); });
      a.inflight = a.inflight || [];
      indexInflight(a);
      a.dirty = true; a.ver = (a.ver || 0) + 1;
    }
    function indexInflight(a) {
      a.groups.forEach(function (g) { g.infl = []; });
      (a.inflight || []).forEach(function (q) {
        var g = a.groups[q.r];
        if (!g) { for (var k = a.groups.length; k <= q.r; k++) a.groups[k] = newGroup(); g = a.groups[q.r]; }
        g.infl.push(q);
        if ((q.u || 0) + 1 > g.subs) g.subs = (q.u || 0) + 1;
      });
      a.dirty = true;
    }
    // layout computes each group's first sub-row (offRows) and the total row count.
    function layout(a) {
      if (!a || !a.dirty) return;
      var off = 0;
      a.offRows = new Array(a.groups.length);
      for (var g = 0; g < a.groups.length; g++) { a.offRows[g] = off; off += a.groups[g].subs; }
      a.totalRows = off;
      a.dirty = false;
    }
    function groupTop(a, g) { return g * GAP + a.offRows[g] * rowH; }  // px from the top of the rows area
    function defaultRowH(a) {
      var n = a.groups.length;
      return n <= 60 ? 14 : n <= 300 ? 6 : n <= 1000 ? 4 : 3;
    }
    function contentHeight(a) {
      layout(a);
      return AXIS_H + (a ? a.groups.length * GAP + (a.totalRows || 0) * rowH : 0) + 8;
    }
    // groupAt finds the group whose rows contain content-area y (px from rows top).
    function groupAt(a, c) {
      layout(a);
      var lo = 0, hi = a.groups.length - 1, ans = -1;
      while (lo <= hi) {
        var mid = (lo + hi) >> 1;
        if (groupTop(a, mid) <= c) { ans = mid; lo = mid + 1; } else hi = mid - 1;
      }
      return ans;
    }

    function applyArm() {
      var a = arm();
      if (!a) return;
      if (!userRowH) rowH = defaultRowH(a);
      elRowH.value = rowH;
      resize();
      updateInfo();
    }

    function updateInfo() {
      var a = arm();
      if (!a) { elInfo.textContent = ""; return; }
      var s = a.reqs.length + " requests";
      if (opts.live) s += " · " + (a.inflight ? a.inflight.length : 0) + " in flight";
      if (a.dropped) s += " · " + a.dropped + " older dropped";
      elInfo.textContent = s;
      elEmpty.hidden = a.reqs.length + (a.inflight ? a.inflight.length : 0) > 0;
      elReset.hidden = !isZoomed();
    }

    // Redraws are coalesced to one per animation frame so scroll, wheel, zoom and
    // resize storms cost one paint each frame.
    var rafDraw = 0, rafOverlay = 0;
    function scheduleDraw() {
      if (rafDraw) return;
      rafDraw = requestAnimationFrame(function () { rafDraw = 0; draw(); });
    }
    function scheduleOverlay() {
      if (rafOverlay) return;
      rafOverlay = requestAnimationFrame(function () { rafOverlay = 0; drawOverlay(); });
    }
    var sizeKey = "";
    function resize() {
      var a = arm();
      var w = elScroll.clientWidth, h = elScroll.clientHeight;
      if (w < 50 || h < 50) return;
      var dpr = window.devicePixelRatio || 1;
      plotL = 44;
      var key = w + "x" + h + "@" + dpr;
      if (key === sizeKey) {   // same canvas: only the scroll extent may have changed
        elContent.style.height = Math.max(h, contentHeight(a)) + "px";
        draw();
        return;
      }
      sizeKey = key;
      W = w; H = h;
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      canvas.style.marginBottom = (-h) + "px";
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlay.style.width = w + "px"; overlay.style.height = h + "px";
      overlay.style.marginBottom = (-h) + "px";
      overlay.width = canvas.width; overlay.height = canvas.height;
      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      [stripAct, stripRate, stripTTFT].forEach(function (c) {
        c.style.width = w + "px"; c.style.height = STRIP_H + "px";
        c.width = Math.round(w * dpr); c.height = Math.round(STRIP_H * dpr);
        c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
      });
      plotW = Math.max(50, W - plotL - RIGHT);
      elContent.style.height = Math.max(h, contentHeight(a)) + "px";
      draw();
    }

    // ---- strips: token rates and TTFT over time, on the timeline's x axis ----
    // tokens per second of series s over [t0, t1] ms (pro rata over buckets).
    function rateAt(rs, key, t0, t1) {
      var v = rs[key], bw = rs.bw, sum = 0;
      for (var i = Math.max(0, Math.floor(t0 / bw)); i < v.length && i * bw < t1; i++) {
        var lo = Math.max(t0, i * bw), hi = Math.min(t1, (i + 1) * bw);
        if (hi > lo) sum += v[i] * (hi - lo) / bw;
      }
      return sum / ((t1 - t0) / 1000);
    }
    function fmtRate(v) { return v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : v >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 1 : 2) + "k" : Math.round(v) + ""; }
    function stripFrame(c, title, ymax, yfmt) {
      var x = c.getContext("2d");
      x.clearRect(0, 0, W, STRIP_H);
      x.fillStyle = "#0D1013"; x.fillRect(0, 0, W, STRIP_H);
      x.strokeStyle = "#1c2226"; x.lineWidth = 1;
      x.beginPath(); x.moveTo(plotL, STRIP_H - 14.5); x.lineTo(plotL + plotW, STRIP_H - 14.5); x.stroke();
      x.font = "10px monospace"; x.fillStyle = "#8a9096"; x.textAlign = "right";
      x.fillText(yfmt(ymax), plotL - 4, 10); x.fillText("0", plotL - 4, STRIP_H - 16);
      // title: a string, or [name, then {t, c, line?} keys]: each key is a colour
      // swatch (a bar, or a line for line series) followed by its label.
      x.textAlign = "left";
      if (typeof title === "string") { x.fillStyle = "#6f767d"; x.fillText(title, plotL + 4, 10); return x; }
      var px = plotL + 4;
      title.forEach(function (it, i) {
        if (typeof it === "string") { x.fillStyle = "#8a9096"; x.fillText(it, px, 10); px += x.measureText(it).width + 10; return; }
        if (it.line) {
          x.strokeStyle = it.c; x.lineWidth = it.w || 1.6; if (it.dash) x.setLineDash([4, 3]);
          x.beginPath(); x.moveTo(px, 6.5); x.lineTo(px + 12, 6.5); x.stroke(); x.setLineDash([]);
        } else { x.fillStyle = it.c; x.globalAlpha = 0.85; x.fillRect(px, 3, 10, 7); x.globalAlpha = 1; }
        px += 15; x.fillStyle = "#b8bec4"; x.fillText(it.t, px, 10); px += x.measureText(it.t).width + 12;
      });
      return x;
    }
    function drawRateStrip(a, r) {
      var c = stripRate;
      if (!a || !a.rates || !a.rates.in || !a.rates.in.length) { c.hidden = true; return; }
      c.hidden = false;
      var rs = a.rates, n = Math.max(1, Math.floor(plotW / 2)), step = (r[1] - r[0]) / n, cols = [], ymax = 1;
      for (var i = 0; i < n; i++) {
        var t0 = r[0] + i * step, t1 = t0 + step;
        var tot = rateAt(rs, "in", t0, t1), comp = rateAt(rs, "ic", t0, t1), ext = rateAt(rs, "ie", t0, t1), gpu = rateAt(rs, "ig", t0, t1), out = rateAt(rs, "out", t0, t1);
        cols.push([tot, comp, ext, gpu, out]);
        if (tot > ymax) ymax = tot; if (out > ymax) ymax = out;
      }
      var x = stripFrame(c, ["tokens/s", { t: "input computed", c: "#d9b73a" }, { t: "input external (offload)", c: "#a678e8" },
        { t: "input GPU cache", c: "#8094b5" }, { t: "input other", c: "#9aa5b1" }, { t: "output", c: "#6BE0A0", line: true, w: 1.5 }], ymax, fmtRate);
      var base = STRIP_H - 15, hgt = STRIP_H - 26, w = plotW / n;
      for (i = 0; i < n; i++) {
        var col = cols[i], px = plotL + i * w, y = base;
        var parts = [[col[1], "#d9b73a"], [col[2], "#a678e8"], [col[3], "#8094b5"], [Math.max(0, col[0] - col[1] - col[2] - col[3]), "#9aa5b1"]];
        for (var k = 0; k < 4; k++) {
          var hh = parts[k][0] / ymax * hgt; if (hh < 0.01) continue;
          x.fillStyle = parts[k][1]; x.globalAlpha = 0.75; x.fillRect(px, y - hh, w + 0.5, hh); y -= hh;
        }
      }
      x.globalAlpha = 1; x.strokeStyle = "#6BE0A0"; x.lineWidth = 1.5; x.beginPath();
      for (i = 0; i < n; i++) { var yy = base - cols[i][4] / ymax * hgt, xx = plotL + (i + 0.5) * w; if (i) x.lineTo(xx, yy); else x.moveTo(xx, yy); }
      x.stroke();
    }
    function drawTTFTStrip(a, r) {
      var c = stripTTFT, t = a && a.ttft;
      if (!t || !t.t || !t.t.length) { c.hidden = true; return; }
      c.hidden = false;
      var lim = t.limit_ms || 0, ymax = lim * 1.15 || 1;
      for (var i = 0; i < t.t.length; i++) if (t.t[i] >= r[0] && t.t[i] <= r[1]) { if (t.p95[i] * 1.1 > ymax) ymax = t.p95[i] * 1.1; if (t.mean[i] > ymax) ymax = t.mean[i]; }
      var key = ["TTFT, " + fmtSec((t.window_s || 30)) + " window", { t: "mean", c: "#C79FF1", line: true }, { t: "p95", c: "#f08a3c", line: true, w: 1 }];
      if (lim) key.push({ t: "limit", c: "#ff3b3b", line: true, w: 1, dash: true });
      var x = stripFrame(c, key, ymax, function (v) { return fmtSec(v / 1000); });
      var base = STRIP_H - 15, hgt = STRIP_H - 26;
      function line(arr, color, width) {
        x.strokeStyle = color; x.lineWidth = width; x.beginPath(); var first = true;
        for (var j = 0; j < t.t.length; j++) {
          if (t.t[j] < r[0] - (r[1] - r[0]) * 0.02 || t.t[j] > r[1] + (r[1] - r[0]) * 0.02) continue;
          var xx = mapX(t.t[j], r), yy = base - Math.min(arr[j], ymax) / ymax * hgt;
          if (first) { x.moveTo(xx, yy); first = false; } else x.lineTo(xx, yy);
        }
        x.stroke();
      }
      line(t.p95, "#f08a3c", 1); line(t.mean, "#C79FF1", 1.6);
      if (lim) {
        var ly = base - lim / ymax * hgt;
        x.strokeStyle = "#ff3b3b"; x.lineWidth = 1; x.setLineDash([4, 3]); x.beginPath(); x.moveTo(plotL, ly); x.lineTo(plotL + plotW, ly); x.stroke(); x.setLineDash([]);
      }
    }
    // ---- activity: active series and requests in flight over time ----
    // A series is one simulated user lane: it is active from its first request becoming due until its last request
    // ends (a live one until now), running one session after another. In flight = sent and not yet finished.
    function stepFrom(ev) {
      ev.sort(function (x, y) { return x[0] - y[0] || x[1] - y[1]; });
      var T = [], V = [], v = 0;
      for (var i = 0; i < ev.length; i++) { v += ev[i][1]; if (T.length && T[T.length - 1] === ev[i][0]) V[V.length - 1] = v; else { T.push(ev[i][0]); V.push(v); } }
      return { t: T, v: V };
    }
    function activity(a, now) {
      var key = (a.ver || 0) + "|" + (opts.live ? Math.floor(now / 1000) : 0);
      if (a.actKey === key) return a.act;
      var sev = [], fev = [];
      for (var g = 0; g < a.groups.length; g++) {
        var grp = a.groups[g], lo = Infinity, hi = -Infinity;
        for (var k = 0; k < grp.reqs.length; k++) {
          var q = grp.reqs[k], end = q.t + q.e;
          if (q.t < lo) lo = q.t; if (end > hi) hi = end;
          fev.push([q.t + (q.cw || 0), 1], [end, -1]);
        }
        for (var j = 0; j < grp.infl.length; j++) {
          var qi = grp.infl[j];
          if (qi.t < lo) lo = qi.t; hi = Math.max(hi, now);
          if (!(qi.x & FLAG_WAITING)) fev.push([qi.t + (qi.cw || 0), 1], [now, -1]);
        }
        if (lo < hi) sev.push([lo, 1], [hi, -1]);
      }
      a.act = { series: stepFrom(sev), inflight: stepFrom(fev) }; a.actKey = key;
      return a.act;
    }
    function stepAt(st, t) {
      var lo = 0, hi = st.t.length - 1, ans = -1;
      while (lo <= hi) { var m = (lo + hi) >> 1; if (st.t[m] <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
      return ans < 0 ? 0 : st.v[ans];
    }
    function drawActStrip(a, r, now) {
      var c = stripAct;
      if (!a || !a.groups.length) { c.hidden = true; return; }
      c.hidden = false;
      var act = activity(a, now), ymax = 1;
      [act.series, act.inflight].forEach(function (st) { for (var i = 0; i < st.t.length; i++) if (st.t[i] <= r[1] && st.v[i] > ymax) ymax = st.v[i]; });
      ymax *= 1.1;
      var x = stripFrame(c, [{ t: "active sessions", c: "#C79FF1", line: true }, { t: "requests in flight", c: "#5f987d", line: true, w: 1 }], ymax, function (v) { return Math.round(v) + ""; });
      var base = STRIP_H - 15, hgt = STRIP_H - 26;
      function step(st, color, width) {
        x.strokeStyle = color; x.lineWidth = width; x.beginPath();
        var y = base - stepAt(st, r[0]) / ymax * hgt; x.moveTo(plotL, y);
        for (var i = 0; i < st.t.length; i++) {
          if (st.t[i] <= r[0]) continue; if (st.t[i] > r[1]) break;
          var xx = mapX(st.t[i], r); x.lineTo(xx, y); y = base - st.v[i] / ymax * hgt; x.lineTo(xx, y);
        }
        x.lineTo(plotL + plotW, y); x.stroke();
      }
      step(act.inflight, "#5f987d", 1); step(act.series, "#C79FF1", 1.6);
    }
    function drawStrips(a, r) {
      drawActStrip(a, r, opts.live ? curNow() : 0); drawRateStrip(a, r); drawTTFTStrip(a, r);
      if (playT === null) return;
      // replay: the strips show the whole run, dimmed past the clock
      var px = Math.max(plotL, mapX(playT, r));
      [stripAct, stripRate, stripTTFT].forEach(function (c) {
        if (c.hidden || px >= plotL + plotW) return;
        var x = c.getContext("2d");
        x.fillStyle = "rgba(13,16,19,0.72)"; x.fillRect(px, 0, plotL + plotW - px, STRIP_H);
        x.strokeStyle = "#C79FF1"; x.beginPath(); x.moveTo(px + 0.5, 0); x.lineTo(px + 0.5, STRIP_H); x.stroke();
      });
    }

    // ---- slot state header ----
    function setStateText(html) {
      if (elState.innerHTML === html) return;
      elState.innerHTML = html;
      elAt.title = elState.textContent;   // the full state, on hover over the one-line readout
    }
    // One-line readout at the cursor (or the view end / now): what the State,
    // Activity, Tokens/s and TTFT panels say at that instant, so a view without
    // them still answers "what was happening here".
    function renderAt() {
      var a = arm();
      if (!a || !host.classList.contains("wtl-hide-state")) { if (elAt.innerHTML) elAt.innerHTML = ""; return; }
      var rr = range(), now = opts.live ? curNow() : 0;
      var cursor = hoverT !== null && hoverT >= rr[0] && hoverT <= rr[1];
      var T = cursor ? hoverT : playT !== null ? playT : opts.live ? Math.min(rr[1], now) : rr[1];
      var act = activity(a, now);
      var h = "<span class='wtl-dim'>" + (cursor ? "at" : playT !== null ? "replay" : opts.live ? "now" : "view end") + "</span> <b>" + fmtElapsed(T, 1000) + "</b>" +
        " &middot; sessions <b>" + stepAt(act.series, T) + "</b> &middot; in flight <b>" + stepAt(act.inflight, T) + "</b>";
      if (a.rates && a.rates.in && a.rates.in.length) {
        // A short window around the point: a few pixels of the view, never under two buckets or 10 s.
        var w = Math.max(2 * a.rates.bw, 6 * (rr[1] - rr[0]) / plotW, 10000);
        var t0 = Math.max(0, cursor ? T - w / 2 : T - w), t1 = t0 + w;
        h += " &middot; in <b>" + fmtRate(rateAt(a.rates, "in", t0, t1)) + "</b> &middot; out <b>" + fmtRate(rateAt(a.rates, "out", t0, t1)) + "</b> tok/s";
      }
      var tt = a.ttft;
      if (tt && tt.t && tt.t.length) {
        var lo = 0, hi = tt.t.length - 1, k = -1;
        while (lo <= hi) { var m = (lo + hi) >> 1; if (tt.t[m] <= T) { k = m; lo = m + 1; } else hi = m - 1; }
        if (k >= 0) h += " &middot; TTFT mean <b>" + fmtSec(tt.mean[k] / 1000) + "</b> p95 <b>" + fmtSec(tt.p95[k] / 1000) + "</b>";
      }
      if (elAt.innerHTML !== h) elAt.innerHTML = h;
    }
    function fmtSec(s) { return s >= 100 ? Math.round(s) + "s" : s.toFixed(s >= 10 ? 1 : 2) + "s"; }
    // Live: the server's counts and admission gate.
    function renderState(st) {
      if (!st) return;
      var h = "<span title='" + SERIES_TIP + "'>active sessions</span> <b>" + st.slots + "</b>" + (st.max ? " / <b>" + st.max + "</b> max" : "") +
        " &middot; <span title='request outstanding at the engine'>in flight <b>" + st.in_flight + "</b></span>" +
        " &middot; <span title='next turn due, waiting under the client concurrency limit'>waiting at gate <b>" + st.waiting + "</b></span>" +
        " &middot; <span title='pausing until the next turn is due'>think pause <b>" + st.think + "</b></span>" +
        " &middot; <span title='no turn pending: between sessions'>idle <b>" + st.idle + "</b></span>" +
        " <span class='wtl-dim'>(" + st.requests + " requests outstanding)</span>";
      if (st.rates) {
        var rt = st.rates;
        h += "<br>tokens/s: input <b>" + fmtRate(rt.in10) + "</b> (10s) &middot; <b>" + fmtRate(rt.in60) + "</b> (60s) &nbsp;|&nbsp; output <b>" + fmtRate(rt.out10) + "</b> (10s) &middot; <b>" + fmtRate(rt.out60) + "</b> (60s)";
      }
      var g = st.gate;
      if (g && g.n > 0) {
        var lim = g.limit_ms / 1000, near = g.limit_ms > 0 && g.value_ms >= 0.8 * g.limit_ms, bad = g.limit_ms > 0 && g.value_ms >= g.limit_ms;
        function stat(name, ms) {
          var gating = g.stat === name, cls = gating && g.enabled ? (bad ? "wtl-bad" : near ? "wtl-warn" : "wtl-gating") : "";
          return name + " " + (cls ? "<b class='" + cls + "'>" : "<b>") + fmtSec(ms / 1000) + "</b>";
        }
        h += "<br>TTFT " + fmtSec(g.window_s) + " window (n=" + g.n + "): " + stat("mean", g.mean_ms) + " &middot; " + stat("p50", g.p50_ms) + " &middot; " + stat("p95", g.p95_ms) + " &middot; " + stat("p99", g.p99_ms);
      }
      if (g) {
        if (!g.enabled) h += "<br>admission: ungated" + (g.admit_count ? ", +" + g.admit_count + " every " + fmtSec(g.every_s) : "");
        else {
          h += " &mdash; gate: " + g.stat + " &lt; " + fmtSec(g.limit_ms / 1000) + " &rarr; <b class='" + (g.open ? "wtl-gating" : "wtl-bad") + "'>" + (g.open ? "OPEN" : "CLOSED") + "</b>";
          h += g.capped ? " (at the session cap, no more admissions)" : g.every_s ? (g.open ? " (next +" + g.admit_count + " in " + fmtSec(g.next_in_s) + ")" : " (admissions held; next check in " + fmtSec(g.next_in_s) + ")") : "";
          if (g.decode_min) {
            var slow = (g.decode_rate || 0) < g.decode_min;
            h += " &middot; <span title='--decode-rate-min: due requests are held while the per-request decode rate over the window is below the minimum'>decode <b class='" + (slow ? "wtl-bad" : "wtl-gating") + "'>" + (g.decode_rate || 0).toFixed(1) + "</b> / " + g.decode_min + " tok/s per request, held <b>" + (g.decode_holding || 0) + "</b></span>";
          }
          if (g.retire_ms) {
            var shedding = g.value_ms > g.retire_ms;
            h += " &middot; <span title='--ttft-retire: a row whose session ends while the gate statistic is above this does not take another session'>retire above " + fmtSec(g.retire_ms / 1000) + " &rarr; <b class='" + (shedding ? "wtl-bad" : "wtl-gating") + "'>" + (shedding ? "SHRINKING" : "holding") + "</b>, retired <b>" + (g.retired || 0) + "</b></span>";
          }
        }
      }
      setStateText(h);
      renderAt();
    }
    // Offline: reconstruct from the requests at the right edge of the view.
    function renderOfflineState() {
      var a = arm(); if (!a || opts.live) return;
      var rr0 = range(), atCursor = hoverT !== null && hoverT >= rr0[0] && hoverT <= rr0[1];
      var T = atCursor ? hoverT : playT !== null ? playT : rr0[1], c = { f: 0, w: 0, t: 0, i: 0 };
      var act = activity(a, 0), activeAtT = stepAt(act.series, T);
      for (var g = 0; g < a.groups.length; g++) {
        var grp = a.groups[g]; ensureSorted(grp);
        var st = 3;   // 0 in flight, 1 waiting, 2 think, 3 idle
        for (var k = firstAtOrAfter(grp.reqs, T - grp.maxE); k < grp.reqs.length && grp.reqs[k].t - (grp.reqs[k].tp || 0) <= T; k++) {
          var q = grp.reqs[k], cw = q.cw || 0, s2;
          if (T >= q.t + cw && T <= q.t + q.e) s2 = 0;
          else if (T >= q.t && T < q.t + cw) s2 = 1;
          else if (q.tp && T >= q.t - q.tp && T < q.t) s2 = 2;
          else continue;
          if (s2 < st) st = s2;
        }
        if (st === 0) c.f++; else if (st === 1) c.w++; else if (st === 2) c.t++;
      }
      c.i = Math.max(0, activeAtT - c.f - c.w - c.t);
      var rate = "";
      if (a.rates && a.rates.in && a.rates.in.length) {
        var rr = range();
        rate = "<br>view average tokens/s: input <b>" + fmtRate(rateAt(a.rates, "in", rr[0], rr[1])) + "</b> &middot; output <b>" + fmtRate(rateAt(a.rates, "out", rr[0], rr[1])) + "</b>";
      }
      renderAt();
      setStateText("at <b>" + fmtElapsed(T, 1000) + "</b> (" + (atCursor ? "cursor" : playT !== null ? "replay" : "view end") + "): <b><span title='" + SERIES_TIP + "'>" + activeAtT + "</span></b> active sessions <span class='wtl-dim'>(" + a.groups.length + " rows over the run)</span> &middot; in flight <b>" + c.f +
        "</b> &middot; waiting at gate <b>" + c.w + "</b> &middot; think pause <b>" + c.t + "</b> &middot; idle <b>" + c.i + "</b>" + rate);
    }

    // ---- replay ----
    // playT (ms of arm time) is the replay clock, null when not replaying: only
    // requests started by then are drawn, each cut at playT, so stages fill in as
    // the clock passes them. One pane drives the clock (requestAnimationFrame)
    // and the others follow its ticks, so split panes stay in lockstep.
    var SPEEDS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3000];
    var playT = null, playing = false, playRaf = 0, playLast = 0;
    var elPlay = $(".wtl-playbtn"), elRestart = $(".wtl-restart"), elStop = $(".wtl-stop"), elSpeed = $(".wtl-speed");
    var elSpeedV = $(".wtl-speedv"), elPlayT = $(".wtl-playt");
    if (opts.live) $(".wtl-play").hidden = true;
    function speed() { return SPEEDS[+elSpeed.value]; }
    function playEnd() { return opts.sync ? opts.sync.maxSpan() : (arm() ? arm().span : 0); }
    function updPlayUI() {
      elPlay.innerHTML = playing ? "&#x23F8;" : "&#x25B6;";
      elStop.hidden = playT === null;
      elSpeedV.textContent = speed() + "\u00d7";
      elPlayT.innerHTML = playT === null ? "" : "<b>" + fmtElapsed(playT, 1000) + "</b> / " + fmtElapsed(playEnd(), 1000);
    }
    function playFrame(now) {
      if (!playing) return;
      playT = Math.min(playEnd(), playT + (now - playLast) * speed());
      playLast = now;
      if (playT >= playEnd()) playing = false;
      emit("tick", { t: playT, playing: playing });
      updPlayUI(); draw();
      if (playing) playRaf = requestAnimationFrame(playFrame);
    }
    function play(fromStart) {
      if (playT === null || fromStart || playT >= playEnd()) playT = 0;
      playing = true; playLast = performance.now();
      cancelAnimationFrame(playRaf); playRaf = requestAnimationFrame(playFrame);
      emit("play", { t: playT, playing: true });
      updPlayUI(); draw();
    }
    function pause() {
      playing = false; cancelAnimationFrame(playRaf);
      emit("play", { t: playT, playing: false });
      updPlayUI();
    }
    function stopPlay() {
      playing = false; cancelAnimationFrame(playRaf); playT = null;
      emit("stop");
      updPlayUI(); draw();
    }
    elPlay.addEventListener("click", function () { if (playing) pause(); else play(false); });
    elRestart.addEventListener("click", function () { play(true); });
    elStop.addEventListener("click", stopPlay);
    elSpeed.addEventListener("input", function () { updPlayUI(); emit("speed", +elSpeed.value); });
    updPlayUI();
    // apply is what the other panes' events do here (no re-emit).
    function applySync(kind, v) {
      if (kind === "view") setView(v[0], v[1], true);
      else if (kind === "expand") setExpanded(v, true);
      else if (kind === "speed") { elSpeed.value = v; updPlayUI(); }
      else if (kind === "stop") { playing = false; cancelAnimationFrame(playRaf); playT = null; updPlayUI(); draw(); }
      else if (kind === "play" || kind === "tick") {
        // A follower never runs its own clock: it shows the driver's.
        cancelAnimationFrame(playRaf);
        playT = v.t; playing = v.playing; updPlayUI(); draw();
      }
    }

    // ---- tile cache ----
    var TILE_H = 512, MAX_TILES = 10;
    var tiles = { key: "", map: {}, order: [], r: null };
    var preview = false, settleTimer = 0;
    function startPreview() {
      preview = true;
      clearTimeout(settleTimer);
      settleTimer = setTimeout(function () { preview = false; draw(); }, 150);
    }
    var pc = ctx;   // the context drawReq / flushLod paint on (main canvas or a tile)
    function getTile(a, r, ty, rh) {
      var t = tiles.map[ty];
      if (t) return t;
      var dpr = window.devicePixelRatio || 1;
      t = document.createElement("canvas");
      t.width = Math.round(W * dpr); t.height = Math.round(TILE_H * dpr);
      var c = t.getContext("2d");
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      paintRows(c, a, r, ty * TILE_H, ty * TILE_H + TILE_H, rh);
      tiles.map[ty] = t; tiles.order.push(ty);
      if (tiles.order.length > MAX_TILES) delete tiles.map[tiles.order.shift()];
      return t;
    }
    // paintRows paints the completed requests of every row group intersecting
    // [top, bottom) (px from the top of the rows area) on c, offset by top.
    function paintRows(c, a, r, top, bottom, rh) {
      pc = c;
      var g0 = Math.max(0, groupAt(a, top));
      for (var g = g0; g < a.groups.length; g++) {
        var gy = groupTop(a, g) - top;
        if (gy > bottom - top) break;
        var grp = a.groups[g];
        ensureSorted(grp);
        if (g > 0 && rowH >= 6) { c.fillStyle = "#13181b"; c.fillRect(0, gy - 1, W, 1); }
        // Session boundaries inside the window: the slot moved on to a new session.
        c.fillStyle = "rgba(199,159,241,0.45)";
        var gh = grp.subs * rowH;
        var bEnd = playT === null ? r[1] : Math.min(r[1], playT);
        for (var n = firstAtOrAfter(grp.bounds, r[0]); n < grp.bounds.length && grp.bounds[n] <= bEnd; n++) {
          c.fillRect(Math.round(mapX(grp.bounds[n], r)) - 1, gy - 1, 1, gh + 1);
        }
        // Only requests that can intersect the window: starts in [r0 - longest, r1].
        var i0 = firstAtOrAfter(grp.reqs, r[0] - grp.maxE);
        var kEnd = playT === null ? r[1] : Math.min(r[1], playT);
        for (var k = i0; k < grp.reqs.length && grp.reqs[k].t <= kEnd; k++) drawReq(grp.reqs[k], gy + (grp.reqs[k].u || 0) * rowH, rh, r, 0);
        for (var u = 0; u < grp.subs; u++) flushLod(u, gy + u * rowH, rh);
      }
    }

    function draw() {
      if (!W) return;
      var a = arm(), r = range(), now = opts.live ? curNow() : 0;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0D1013"; ctx.fillRect(0, 0, W, H);
      var span = r[1] - r[0];
      // vertical grid + axis
      var step = pickStep(span, plotW);
      ctx.font = "11px monospace"; ctx.textBaseline = "alphabetic";
      var first = Math.ceil(r[0] / step) * step;
      var withSec = step < 60000;
      for (var t = first; t <= r[1]; t += step) {
        var x = Math.round(mapX(t, r)) + 0.5;
        ctx.strokeStyle = "#1c2226"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, AXIS_H - 4); ctx.lineTo(x, H); ctx.stroke();
        ctx.fillStyle = "#C9C9C9"; ctx.textAlign = "center";
        ctx.fillText(fmtElapsed(t, step), x, 13);
        if (a) { ctx.fillStyle = "#6f767d"; ctx.fillText(fmtClock(a.t0 + t, withSec), x, 27); }
      }
      if (!a) return;
      var rh = rowH >= 3 ? rowH - 1 : rowH;
      var scrollTop = elScroll.scrollTop;
      layout(a);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, AXIS_H, W, H - AXIS_H); ctx.clip();
      // Completed requests come from cached tiles (a tile is a band of rows
      // painted once per view, so scrolling is a few blits); anything that
      // changes with the clock -- in-flight bars, labels -- is painted per frame.
      var tilesKey = r[0] + "|" + r[1] + "|" + rowH + "|" + merge + "|" + W + "|" + a.ver + "|" + (window.devicePixelRatio || 1) +
        "|" + (playT === null ? "" : Math.floor(mapX(playT, r)));   // a replay repaints once per pixel of clock
      if (preview && tiles.r) {
        // Mid wheel-zoom: show the tiles painted for the previous window, scaled
        // to the new one, and repaint properly once the zoom settles.
        var k = (tiles.r[1] - tiles.r[0]) / (r[1] - r[0]);
        var ox = plotL + (tiles.r[0] - r[0]) / (r[1] - r[0]) * plotW - plotL * k;
        ctx.save();
        ctx.beginPath(); ctx.rect(plotL, AXIS_H, plotW, H - AXIS_H); ctx.clip();
        for (var key in tiles.map) {
          ctx.drawImage(tiles.map[key], ox, AXIS_H + (+key) * TILE_H - scrollTop, W * k, TILE_H);
        }
        ctx.restore();
      } else {
        if (tilesKey !== tiles.key) { tiles = { key: tilesKey, map: {}, order: [], r: r }; }
        for (var ty = Math.floor(scrollTop / TILE_H); ty * TILE_H < scrollTop + H - AXIS_H; ty++) {
          var tile = getTile(a, r, ty, rh);
          ctx.drawImage(tile, 0, AXIS_H + ty * TILE_H - scrollTop, W, TILE_H);
        }
      }
      var g0 = Math.max(0, groupAt(a, scrollTop));
      for (var g = g0; g < a.groups.length; g++) {
        var gy = AXIS_H + groupTop(a, g) - scrollTop;
        if (gy > H) break;
        var grp = a.groups[g];
        pc = ctx;
        for (var k = 0; k < grp.infl.length; k++) drawReq(grp.infl[k], gy + (grp.infl[k].u || 0) * rowH, rh, r, now);
        for (var u = 0; u < grp.subs; u++) flushLod(u, gy + u * rowH, rh);
        if (rowH >= 8) {
          var any = grp.reqs[0] || grp.infl[0];
          if (any) { ctx.fillStyle = "#6f767d"; ctx.textAlign = "right"; ctx.font = "10px monospace"; ctx.fillText(any.sl ? ((any.x & FLAG_HOT) ? "h" : "s") + any.sl : "s" + any.s, plotL - 5, gy + rh - 3); }
        }
      }
      ctx.restore();
      if (playT !== null) {
        var px = mapX(playT, r);
        if (px >= plotL && px <= plotL + plotW) { ctx.strokeStyle = "#C79FF1"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px, AXIS_H - 4); ctx.lineTo(px, H); ctx.stroke(); ctx.lineWidth = 1; }
      }
      if (opts.live) {
        var nx = mapX(now, r);
        if (nx >= plotL && nx <= plotL + plotW) { ctx.strokeStyle = "#7C03EC"; ctx.beginPath(); ctx.moveTo(nx + 0.5, AXIS_H); ctx.lineTo(nx + 0.5, H); ctx.stroke(); }
      }
      drawOverlay();
      drawStrips(a, r);
      if (!opts.live) renderOfflineState(); else renderAt();
    }

    // The overlay is a second canvas holding what changes on every mouse move
    // (hovered row band, hovered bar, zoom drag), so hovering never repaints the
    // data canvas.
    function drawOverlay() {
      if (!W) return;
      octx.clearRect(0, 0, W, H);
      var a = arm(); if (!a) return;
      var scrollTop = elScroll.scrollTop, rh = rowH >= 3 ? rowH - 1 : rowH;
      octx.save();
      octx.beginPath(); octx.rect(0, AXIS_H, W, H - AXIS_H); octx.clip();
      if (hoverGroup >= 0 && hoverGroup < a.groups.length) {
        layout(a);
        var gy = AXIS_H + groupTop(a, hoverGroup) - scrollTop, gh = a.groups[hoverGroup].subs * rowH;
        octx.fillStyle = "rgba(199,159,241,0.10)"; octx.fillRect(0, gy, W, gh);
        octx.strokeStyle = "#C79FF1"; octx.lineWidth = 1;
        octx.strokeRect(0.5, gy - 0.5, W - 1, gh + 1);
      }
      if (hover) {
        var r = range(), now = opts.live ? curNow() : 0, q = hover.q;
        var hy = AXIS_H + groupTop(a, q.r) + (q.u || 0) * rowH - scrollTop;
        var he = (q.x & FLAG_INFL) ? Math.max(0, now - q.t) : q.e;
        var hx0 = mapX(q.t, r), hx1 = mapX(q.t + he, r);
        octx.strokeStyle = "#F2F2EB"; octx.lineWidth = 1;
        octx.strokeRect(Math.round(hx0) - 0.5, hy - 0.5, Math.max(2, hx1 - hx0) + 1, rh + 1);
      }
      octx.restore();
      if (drag) {
        var dx0 = Math.min(drag.x0, drag.x1), dx1 = Math.max(drag.x0, drag.x1);
        octx.fillStyle = "rgba(124,3,236,0.18)"; octx.fillRect(dx0, AXIS_H, dx1 - dx0, H - AXIS_H);
        octx.strokeStyle = "#7C03EC"; octx.strokeRect(dx0 + 0.5, AXIS_H + 0.5, dx1 - dx0, H - AXIS_H - 1);
      }
    }

    // segsFor returns a request's flattened (kind, ms) segments: the stored ones,
    // or for a live request the client wait so far / since it was sent followed by
    // the still-running part. With merge on, consecutive wait kinds become one.
    function segsFor(q, now) {
      var segs = q.g;
      if (q.x & FLAG_INFL) {
        var el = Math.max(0, now - q.t);
        if (q.x & FLAG_WAITING) segs = [WAIT_CLIENT, el];
        else { var cw = Math.min(q.cw || 0, el); segs = cw > 0 ? [WAIT_CLIENT, cw, INFLIGHT_KIND, el - cw] : [INFLIGHT_KIND, el]; }
      }
      if (!merge) return segs;
      var out = [], accum = 0;
      for (var i = 0; i < segs.length; i += 2) {
        if (WAIT_KINDS[segs[i]]) { accum += segs[i + 1]; continue; }
        if (accum) { out.push(WAITED, accum); accum = 0; }
        out.push(segs[i], segs[i + 1]);
      }
      if (accum) out.push(WAITED, accum);
      return out;
    }

    var lodK = [], lodX0 = [], lodX1 = [];
    function flushLod(u, y, rh) {
      if (lodK[u] === undefined) return;
      pc.fillStyle = colors[lodK[u]] || "#888";
      pc.fillRect(lodX0[u], y, lodX1[u] - lodX0[u], rh);
      lodK[u] = undefined;
    }
    function drawReq(q, y, rh, r, now) {
      var infl = (q.x & FLAG_INFL) !== 0;
      var end = infl ? Math.max(0, now - q.t) : q.e;
      if (playT !== null) { if (q.t > playT) return; end = Math.min(end, playT - q.t); }   // replay: cut at the clock
      var xs = mapX(q.t, r), xe = mapX(q.t + end, r);
      if (q.tp) {
        // The deliberate pause before this turn: idle by design, not for want of a session.
        var tx0 = Math.max(mapX(q.t - q.tp, r), plotL), tx1 = Math.min(xs, plotL + plotW);
        if (tx1 > tx0) { pc.fillStyle = "rgba(124,150,190,0.16)"; pc.fillRect(tx0, y, tx1 - tx0, rh); }
      }
      if (xe < plotL || xs > plotL + plotW) return;
      var segs = segsFor(q, now);
      if (xe - xs < 6) {
        // Too narrow to tell segments apart: one rect in the dominant kind.
        var best = 0, bd = -1;
        for (var j = 0; j < segs.length; j += 2) if (segs[j + 1] > bd) { bd = segs[j + 1]; best = segs[j]; }
        var x0 = Math.max(xs, plotL), x1 = Math.max(Math.min(xe, plotL + plotW), x0 + 0.6), u = q.u || 0;
        // Neighbouring narrow bars of the same kind in a sub-row paint as one run.
        if (lodK[u] === best && x0 <= lodX1[u] + 0.75) { if (x1 > lodX1[u]) lodX1[u] = x1; return; }
        flushLod(u, y, rh);
        lodK[u] = best; lodX0[u] = x0; lodX1[u] = x1;
        return;
      }
      var t = q.t, rem = end;
      for (var i = 0; i < segs.length && rem > 0; i += 2) {
        var d = Math.min(segs[i + 1], rem);
        rem -= d;
        var sx0 = mapX(t, r), sx1 = mapX(t + d, r);
        t += d;
        if (sx1 < plotL || sx0 > plotL + plotW) continue;
        if (sx0 < plotL) sx0 = plotL;
        if (sx1 > plotL + plotW) sx1 = plotL + plotW;
        var sw = sx1 - sx0; if (sw < 0.6) sw = 0.6;
        pc.fillStyle = colors[segs[i]] || "#888";
        pc.fillRect(sx0, y, sw, rh);
      }
    }

    // ---- interaction ----
    function pos(e) { var b = canvas.getBoundingClientRect(); return { x: e.clientX - b.left, y: e.clientY - b.top }; }
    function pick(p) {
      var a = arm(); if (!a || p.y < AXIS_H || p.x < plotL) return null;
      var c = p.y - AXIS_H + elScroll.scrollTop;
      var g = groupAt(a, c);
      if (g < 0) return null;
      var sub = Math.floor((c - groupTop(a, g)) / rowH);
      var grp = a.groups[g];
      if (sub < 0 || sub >= grp.subs) return null;
      var r = range(), now = opts.live ? curNow() : 0, t = unmapX(p.x, r), tol = 3 * (r[1] - r[0]) / plotW;
      var best = null, bd = Infinity;
      ensureSorted(grp);
      var lists = [grp.reqs, grp.infl];
      for (var l = 0; l < 2; l++) for (var i = l ? 0 : firstAtOrAfter(grp.reqs, t - tol - grp.maxE); i < lists[l].length; i++) {
        var q = lists[l][i];
        if (l === 0 && q.t > t + tol) break;
        if ((q.u || 0) !== sub) continue;
        if (playT !== null && q.t > playT) continue;
        var e = (q.x & FLAG_INFL) ? Math.max(0, now - q.t) : q.e;
        var d = t < q.t ? q.t - t : (t > q.t + e ? t - q.t - e : 0);
        if (d <= tol && d < bd) { bd = d; best = q; }
      }
      return best ? { q: best } : null;
    }
    // rowAt is the row group under the pointer, counting the gaps between a row's
    // requests and the label gutter; -1 over the axis, the separators or below the data.
    function rowAt(p) {
      var a = arm(); if (!a || p.y < AXIS_H) return -1;
      var c = p.y - AXIS_H + elScroll.scrollTop, g = groupAt(a, c);
      if (g < 0 || c >= groupTop(a, g) + a.groups[g].subs * rowH) return -1;
      return g;
    }
    function showTip(h, e) {
      if (!h) { tip.style.display = "none"; return; }
      var q = h.q, a = arm(), now = opts.live ? curNow() : 0;
      var infl = (q.x & FLAG_INFL) !== 0, e2e = infl ? Math.max(0, now - q.t) : q.e;
      var inst = q.gu || (a.guids && q.gi ? a.guids[q.gi - 1] : "");
      var ic = inst.indexOf(":"); if (ic >= 0) inst = inst.slice(ic + 1);
      var head = q.sl ? "<b>" + ((q.x & FLAG_HOT) ? "hot row " : "row ") + q.sl + "</b> &middot; session " + q.s : "<b>session " + q.s + "</b>";
      var html = head + " turn " + q.n + ((q.x & FLAG_SESSION) ? " (first of session)" : "") + (inst ? " &middot; instance " + esc(inst.length > 24 ? inst.slice(0, 24) + "\u2026" : inst) : "") + (infl ? " <b>(in flight)</b>" : "") +
        ((q.x & FLAG_ERR) ? " <b style='color:#FF6B6B'>error</b>" : "") + ((q.x & FLAG_COLD) ? " cold start" : "") + ((q.x & FLAG_EMPTY) ? " empty" : "") + "<br>" +
        "start " + fmtClock(a.t0 + q.t, true) + " (+" + fmtDur(q.t) + ")<br>";
      var cw = q.cw || 0;
      if (q.tp > 0) html += "think pause before this turn <b>" + fmtDur(q.tp) + "</b><br>";
      if (!infl) e2e -= cw;   // e includes the client wait; e2e below is sent -> done
      html += "e2e <b>" + fmtDur(e2e) + "</b>" + (q.f ? " &middot; ttft <b>" + fmtDur(q.f) + "</b>" : "") + "<br>";
      var total = (q.i || 0) + (q.c || 0);
      html += "tokens: " + fmtTok(total) + " prompt (" + fmtTok(q.c || 0) + " cached) &rarr; " + fmtTok(q.o || 0) + " out<br>";
      if (q.src) {
        var s = q.src, tot = s[0] + s[1] + s[2] + s[3] || 1, names = ["GPU cache", "host DRAM", "external", "computed"];
        var ids = [5, 4, 3, 2];
        html += "<table>";
        for (var i = 0; i < 4; i++) if (s[i] > 0) html += "<tr><td><i style='background:" + colors[ids[i]] + "'></i>" + names[i] + "</td><td>" + fmtTok(s[i]) + " tok (" + Math.round(100 * s[i] / tot) + "%)</td></tr>";
        html += "</table>";
      }
      // Every wait, by where it happened.
      var unattr = 0;
      if (q.tm && !(q.x & FLAG_TRUNC)) unattr = Math.max(0, e2e - (q.rw || 0) - (q.tm[0] + q.tm[1] + q.tm[2] + q.tm[3] + q.tm[4] + q.tm[5]));
      if (cw > 0 || q.rw > 0 || (q.x & FLAG_WAITING) || (q.tm && (q.tm[0] > 0 || q.tm[1] > 0 || unattr > 1))) {
        html += "<table>";
        if (q.rw > 0) html += "<tr><td><i style='background:" + colors[14] + "'></i>429 shed / backoff (" + (q.r4 || 0) + " retries)</td><td>" + fmtDur(q.rw) + "</td></tr>";
        if ((q.x & FLAG_WAITING)) html += "<tr><td><i style='background:" + colors[WAIT_CLIENT] + "'></i>client wait (still waiting)</td><td>" + fmtDur(Math.max(0, now - q.t)) + "</td></tr>";
        else if (cw > 0) html += "<tr><td><i style='background:" + colors[WAIT_CLIENT] + "'></i>client wait (concurrency limit)</td><td>" + fmtDur(cw) + "</td></tr>";
        if (q.tm && q.tm[1] > 0) html += "<tr><td><i style='background:" + colors[0] + "'></i>engine queue: other</td><td>" + fmtDur(q.tm[1]) + "</td></tr>";
        if (q.tm && q.tm[0] > 0) html += "<tr><td><i style='background:" + colors[1] + "'></i>engine queue: no KV blocks</td><td>" + fmtDur(q.tm[0]) + "</td></tr>";
        if (unattr > 1) html += "<tr><td><i style='background:" + colors[12] + "'></i>unattributed (network / frontend)</td><td>" + fmtDur(unattr) + "</td></tr>";
        html += "</table>";
      }
      if (q.tm) {
        var tn = ["queue wait (KV)", "queue wait (other)", "kv fetch", "prefill", "decode", "preempted"], tid = [1, 0, 3, 6, 8, 7];
        html += "<table>";
        for (var j = 2; j < 6; j++) if (q.tm[j] > 0) html += "<tr><td><i style='background:" + colors[tid[j]] + "'></i>" + tn[j] + "</td><td>" + fmtDur(q.tm[j]) + "</td></tr>";
        if (q.tm[6] > 0) html += "<tr><td>preemptions</td><td>" + q.tm[6] + "</td></tr>";
        html += "</table>";
      } else if (!infl) {
        html += "<span style='color:#8a9096'>client-side timing only (no engine details)</span><br>";
      }
      if (q.rc && (q.rc[0] || q.rc[2])) html += "<span style='color:#8a9096'>recompute: " + q.rc[1] + " discarded tok, " + q.rc[2] + " load failures (" + q.rc[3] + " tok)</span><br>";
      if (q.x & FLAG_TRUNC) html += "<span style='color:#8a9096'>engine phases scaled to fit client e2e</span>";
      tip.innerHTML = html;
      tip.style.display = "block";
      var tw = tip.offsetWidth, th = tip.offsetHeight;
      var left = e.clientX + 14, top = e.clientY + 14;
      if (left + tw > window.innerWidth - 8) left = e.clientX - tw - 14;
      if (top + th > window.innerHeight - 8) top = Math.max(8, e.clientY - th - 14);
      tip.style.left = left + "px"; tip.style.top = top + "px";
    }

    function setView(a, b, silent) {
      if (a === null || a === undefined) { vMin = vMax = null; }
      else { vMin = a; vMax = Math.max(b, a + 1); }
      elReset.hidden = !isZoomed();
      scheduleDraw();
      if (!silent) emit("view", [vMin, vMax]);
    }
    function userZoom(a, b) {
      setView(a, b);
      if (opts.onZoom) opts.onZoom(a, b);
    }

    canvas.addEventListener("mousedown", function (e) {
      var p = pos(e);
      if (p.x >= plotL && p.x <= plotL + plotW) { drag = { x0: p.x, x1: p.x }; tip.style.display = "none"; }
    });
    canvas.addEventListener("mousemove", function (e) {
      var p = pos(e);
      var nt = p.x >= plotL && p.x <= plotL + plotW ? unmapX(p.x, range()) : null;
      if (nt !== hoverT) { hoverT = nt; if (opts.live) renderAt(); else renderOfflineState(); }
      if (drag) { drag.x1 = Math.max(plotL, Math.min(plotL + plotW, p.x)); scheduleOverlay(); return; }
      var h = pick(p);
      var g = rowAt(p);
      if (g !== hoverGroup || (h ? h.q : null) !== (hover ? hover.q : null)) { hover = h; hoverGroup = g; scheduleOverlay(); }
      showTip(h, e);
    });
    function endDrag(e) {
      if (!drag) return;
      var d = drag; drag = null;
      var lo = Math.min(d.x0, d.x1), hi = Math.max(d.x0, d.x1);
      if (hi - lo < 5) { drawOverlay(); return; }
      var r = range();
      userZoom(unmapX(lo, r), unmapX(hi, r));
    }
    canvas.addEventListener("mouseup", endDrag);
    window.addEventListener("mouseup", endDrag);
    canvas.addEventListener("mouseleave", function () { hover = null; hoverGroup = -1; tip.style.display = "none"; if (!drag) drawOverlay(); if (hoverT !== null) { hoverT = null; if (opts.live) renderAt(); else renderOfflineState(); } });
    canvas.addEventListener("dblclick", function () { userZoom(null, null); });
    elReset.addEventListener("click", function () { userZoom(null, null); });
    if (opts.handleEsc) window.addEventListener("keydown", function (e) { if (e.key === "Escape" && isZoomed()) userZoom(null, null); });
    canvas.addEventListener("wheel", function (e) {
      if (!(e.ctrlKey || e.metaKey || e.shiftKey)) return;   // plain wheel scrolls rows
      e.preventDefault();
      var r = range(), p = pos(e), span = r[1] - r[0], f = fullRange();
      startPreview();
      if (e.shiftKey && !e.ctrlKey && !e.metaKey) {
        var dt = (e.deltaY || e.deltaX) / plotW * span;
        var lo = Math.max(f[0], Math.min(f[1] - span, r[0] + dt));
        userZoom(lo, lo + span); return;
      }
      var cx = unmapX(p.x, r), k = Math.exp(e.deltaY * 0.002);
      var nspan = Math.max(200, Math.min(f[1] - f[0], span * k));
      var lo2 = cx - (cx - r[0]) * nspan / span;
      lo2 = Math.max(f[0], Math.min(f[1] - nspan, lo2));
      if (nspan >= f[1] - f[0]) userZoom(null, null); else userZoom(lo2, lo2 + nspan);
    }, { passive: false });
    elScroll.addEventListener("scroll", function () { hover = null; hoverGroup = -1; tip.style.display = "none"; scheduleDraw(); });
    elMerge.addEventListener("change", function () { merge = elMerge.checked; buildLegend(); draw(); });
    elRowH.addEventListener("change", function () { rowH = parseInt(elRowH.value, 10); userRowH = true; resize(); });
    elArm.addEventListener("change", function () { selectArm(parseInt(elArm.value, 10)); if (opts.onArm) opts.onArm(cur); });
    elFollow.addEventListener("change", function () { draw(); });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(function () { resize(); }).observe(elScroll);
    else window.addEventListener("resize", resize);
    applyPanels();

    function selectArm(i) {
      cur = Math.max(0, Math.min(arms.length - 1, i));
      elArm.value = cur;
      elArm.title = arms[cur] ? arms[cur].name : "";
      hover = null; hoverGroup = -1; elScroll.scrollTop = 0;
      applyArm();
    }

    if (opts.onPanes) {
      var elPanes = $(".wtl-panes");
      $(".wtl-paneswrap").hidden = !(opts.maxPanes > 1);
      for (var pn = 1; pn <= (opts.maxPanes || 1); pn++) { var po = document.createElement("option"); po.value = pn; po.textContent = pn; elPanes.appendChild(po); }
      elPanes.value = opts.paneCount || 1;
      elPanes.addEventListener("change", function () { opts.onPanes(+elPanes.value); });
    }
    if (opts.sync) opts.sync.add({ key: syncKey, apply: applySync, span: function () { return arm() ? arm().span : 0; } });

    var api = {
      // setArms replaces all data. Each arm: {name, t0, span, layout, rows, reqs}.
      setArms: function (list, select) {
        arms = list;
        arms.forEach(indexArm);
        var hm = /[?&]highlight_row=(\d+)/.exec(location.search);
        var hs = /[?&]scroll_row=(\d+)/.exec(location.search);
        elArm.innerHTML = arms.map(function (a, i) { return '<option value="' + i + '">' + esc(a.name) + "</option>"; }).join("");
        host.querySelector(".wtl-armwrap").hidden = arms.length < 2;
        selectArm(select || 0);
        // ?highlight_row=N / ?scroll_row=N pre-select a row (screenshots, bug reports).
        if (hs) { layout(arm()); elScroll.scrollTop = Math.max(0, groupTop(arm(), +hs[1])); }
        if (hm) { hoverGroup = +hm[1]; drawOverlay(); }
        // ?mouse=X,Y dispatches a real mousemove at viewport (X,Y) once drawn.
        var mm = /[?&]mouse=(\d+),(\d+)/.exec(location.search);
        if (mm) setTimeout(function () {
          canvas.dispatchEvent(new MouseEvent("mousemove", { clientX: +mm[1], clientY: +mm[2], bubbles: true }));
        }, 50);
      },
      selectArm: selectArm,
      setExpanded: function (on, silent) { setExpanded(on, silent); },
      // setReplay drives this pane's replay clock from outside (the split's bar):
      // t in arm ms, or null to show the whole run.
      setReplay: function (t, isPlaying) { cancelAnimationFrame(playRaf); playT = t; playing = !!isPlaying; updPlayUI(); draw(); },
      span: function () { return arm() ? arm().span : 0; },
      currentArm: function () { return cur; },
      setView: function (a, b) { setView(a, b); },
      getView: function () { return range(); },
      isZoomed: isZoomed,
      resize: resize,
      redraw: draw,
      // setClock anchors the live clock: nowMsInArmTime is the server's
      // elapsed-ms for the arm at the moment the response was produced.
      setClock: function (nowMsInArmTime) { nowMs = nowMsInArmTime; nowAt = performance.now(); },
      // append adds newly completed requests and replaces the in-flight set.
      append: function (armIdx, reqs, inflight, meta) {
        var a = arms[armIdx]; if (!a) return;
        meta = meta || {};
        if (meta.reset) { a.reqs = []; a.dropped = 0; }
        for (var i = 0; i < reqs.length; i++) a.reqs.push(reqs[i]);
        if (meta.reset) {
          indexArm(a);
        } else if (a.reqs.length > maxReqs) {
          var drop = Math.floor(maxReqs / 10);
          a.reqs.splice(0, drop); a.dropped = (a.dropped || 0) + drop;
          indexArm(a);
        } else {
          for (var j = 0; j < reqs.length; j++) addToGroup(a, reqs[j]);
        }
        a.inflight = inflight || [];
        indexInflight(a);
        if (meta.t0 !== undefined) a.t0 = meta.t0;
        if (meta.span !== undefined) a.span = meta.span;
        if (armIdx === cur) { if (!userRowH) { rowH = defaultRowH(a); elRowH.value = rowH; } updateInfo(); resize(); }
      },
      arms: function () { return arms; },
      setState: renderState,
      // setSeries installs the run's token-rate and TTFT history (live: /api/series).
      setSeries: function (armIdx, s) {
        var a = arms[armIdx]; if (!a || !s) return;
        a.rates = s.rates; a.ttft = s.ttft;
        if (armIdx === cur) scheduleDraw();
      },
      // perf runs a fixed interaction workload and returns per-operation timings
      // in ms (used by ?perf=1 on the standalone page to track regressions).
      perf: function () {
        function time(n, fn) { var t = performance.now(); for (var i = 0; i < n; i++) fn(i); return +((performance.now() - t) / n).toFixed(3); }
        var a = arm(), out = { requests: a.reqs.length, groups: a.groups.length, rowH: rowH };
        out.drawFull = time(20, function () { draw(); });
        var maxScroll = Math.max(0, elScroll.scrollHeight - elScroll.clientHeight);
        out.scrollCold = time(100, function (i) { elScroll.scrollTop = maxScroll * i / 100; draw(); });
        out.scrollWarm = time(100, function (i) { elScroll.scrollTop = maxScroll * (i % 3) / 100; draw(); });
        elScroll.scrollTop = 0;
        out.hover = time(200, function (i) {
          var p = { x: plotL + (i * 37) % plotW, y: AXIS_H + (i * 11) % Math.max(1, H - AXIS_H) };
          hover = pick(p); hoverGroup = rowAt(p); drawOverlay();
        });
        var f = fullRange(), span = f[1] - f[0];
        out.zoomStep = time(60, function (i) { var w = span * (1 - i / 80); setView(f[0] + span * 0.3, f[0] + span * 0.3 + w); draw(); });
        preview = true;
        out.zoomPreview = time(60, function (i) { var w = span * (1 - i / 80); setView(f[0] + span * 0.3, f[0] + span * 0.3 + w); draw(); });
        preview = false;
        setView(null, null);
        return out;
      },
    };
    return api;
  }

  // syncGroup relays events between the panes of one split view.
  function syncGroup() {
    var members = [];
    return {
      add: function (m) { members.push(m); },
      emit: function (from, kind, v) { members.forEach(function (m) { if (m.key !== from) m.apply(kind, v); }); },
      maxSpan: function () { return members.reduce(function (mx, m) { return Math.max(mx, m.span()); }, 0); },
    };
  }

  // createSplit lays out one timeline per chosen arm side by side under one
  // bar that owns expand, the replay controls and the arm choice (all arms by
  // default). Panes share zoom through a sync group; the replay clock runs
  // here and drives every pane, so the arms replay in lockstep, each on its
  // own elapsed clock. It exposes the subset of a timeline's API pages use.
  function createSplit(host, opts) {
    var SPEEDS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3000];
    host.classList.add("wtl-split");
    host.innerHTML =
      '<div class="wtl-bar wtl-splitbar">' +
      '<button class="wtl-expand" title="Fill the window with the timelines (Esc or Back to return)">&#x2922; Expand</button>' +
      '<span class="wtl-armsel"></span>' +
      '<span class="wtl-play" title="Replay the run: requests appear as they started and fill in stage by stage">' +
      '<button class="wtl-restart" title="Replay from the start">&#x23EE;</button>' +
      '<button class="wtl-playbtn" title="Play / pause">&#x25B6;</button>' +
      '<button class="wtl-stop" title="Stop the replay and show the whole run" hidden>&#x25A0;</button>' +
      '<input type="range" class="wtl-speed" min="0" max="10" step="1" value="5" title="Replay speed">' +
      '<span class="wtl-speedv"></span><span class="wtl-playt"></span></span>' +
      '</div><div class="wtl-panesrow"></div>';
    var $ = function (c) { return host.querySelector(c); };
    var row = $(".wtl-panesrow"), elArmSel = $(".wtl-armsel"), elExpand = $(".wtl-expand");
    var elPlay = $(".wtl-playbtn"), elRestart = $(".wtl-restart"), elStop = $(".wtl-stop"), elSpeed = $(".wtl-speed");
    var elSpeedV = $(".wtl-speedv"), elPlayT = $(".wtl-playt");
    var arms = [], chosen = null, panes = [], view = [null, null], expanded = false;
    var playT = null, playing = false, raf = 0, last = 0;
    var me = { key: {}, apply: function (kind, v) { if (kind === "expand") markExpanded(v); else if (kind === "view") view = v; }, span: function () { return 0; } };

    function markExpanded(on) {
      expanded = on;
      host.classList.toggle("wtl-expanded-host", on);
      document.documentElement.classList.toggle("wtl-expanded-page", on);
      elExpand.innerHTML = on ? "&#x2715; Back" : "&#x2922; Expand";
    }
    function setExpanded(on) {
      markExpanded(on);
      panes.forEach(function (t) { t.setExpanded(on, true); });
    }
    elExpand.addEventListener("click", function () { setExpanded(!expanded); });
    window.addEventListener("keydown", function (e) { if (e.key === "Escape" && expanded) setExpanded(false); });

    function end() { return panes.reduce(function (m, t) { return Math.max(m, t.span()); }, 0); }
    function ui() {
      elPlay.innerHTML = playing ? "&#x23F8;" : "&#x25B6;";
      elStop.hidden = playT === null;
      elSpeedV.textContent = SPEEDS[+elSpeed.value] + "×";
      elPlayT.innerHTML = playT === null ? "" : "<b>" + fmtElapsed(playT, 1000) + "</b> / " + fmtElapsed(end(), 1000);
    }
    function push() { panes.forEach(function (t) { t.setReplay(playT, playing); }); ui(); }
    function frame(now) {
      if (!playing) return;
      playT = Math.min(end(), playT + (now - last) * SPEEDS[+elSpeed.value]);
      last = now;
      if (playT >= end()) playing = false;
      push();
      if (playing) raf = requestAnimationFrame(frame);
    }
    function play(fromStart) {
      if (playT === null || fromStart || playT >= end()) playT = 0;
      playing = true; last = performance.now();
      cancelAnimationFrame(raf); raf = requestAnimationFrame(frame);
      push();
    }
    elPlay.addEventListener("click", function () { if (playing) { playing = false; cancelAnimationFrame(raf); push(); } else play(false); });
    elRestart.addEventListener("click", function () { play(true); });
    elStop.addEventListener("click", function () { playing = false; cancelAnimationFrame(raf); playT = null; push(); });
    elSpeed.addEventListener("input", ui);
    ui();

    function buildArmSel() {
      elArmSel.hidden = arms.length < 2;
      elArmSel.innerHTML = "Arms " + arms.map(function (a, i) {
        return '<label title="' + esc(a.name) + '"><input type="checkbox" data-i="' + i + '"' + (chosen[i] ? " checked" : "") + '> ' + esc(a.name) + "</label>";
      }).join("");
      [].forEach.call(elArmSel.querySelectorAll("input"), function (cb) {
        cb.addEventListener("change", function () {
          var i = +cb.getAttribute("data-i");
          chosen[i] = cb.checked;
          if (!chosen.some(Boolean)) { chosen[i] = true; cb.checked = true; return; }   // keep one
          build();
        });
      });
    }
    function build() {
      row.innerHTML = "";
      var hub = syncGroup();
      hub.add(me);
      panes = [];
      arms.forEach(function (a, i) {
        if (!chosen[i]) return;
        var d = document.createElement("div");
        d.className = "wtl-pane";
        row.appendChild(d);
        var o = {};
        for (var k in opts) o[k] = opts[k];
        o.sync = hub; o.expandHost = host; o.paneTitle = arms.length > 1 ? a.name : "";
        if (panes.length > 0) o.onZoom = null;   // the page hears zoom once
        var t = create(d, o);
        t.setArms(arms, i);
        if (view[0] !== null) t.setView(view[0], view[1]);
        panes.push(t);
      });
      panes.forEach(function (t) { t.resize(); if (expanded) t.setExpanded(true, true); t.setReplay(playT, playing); });
    }
    return {
      setArms: function (list) {
        arms = list;
        chosen = arms.map(function () { return true; });   // every arm by default
        buildArmSel(); build();
      },
      setView: function (a, b) { view = [a === undefined ? null : a, b]; if (panes[0]) panes[0].setView(a, b); },
      getView: function () { return panes[0] ? panes[0].getView() : [0, 1]; },
      resize: function () { panes.forEach(function (t) { t.resize(); }); },
      redraw: function () { panes.forEach(function (t) { t.redraw(); }); },
      panes: function () { return panes; },
      perf: function () { return panes[0] ? panes[0].perf() : {}; },
    };
  }

  return { create: create, createSplit: createSplit, fmtDur: fmtDur };
})();
