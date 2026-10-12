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
  var WAIT_KINDS = { 0: 1, 1: 1, 11: 1, 12: 1 };   // kinds the "merge waits" toggle collapses into WAITED
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
      '<label class="wtl-armwrap" hidden>Arm <select class="wtl-arm"></select></label>' +
      '<label>Row height <select class="wtl-rowh"></select></label>' +
      '<label title="Collapse client wait, engine queue and unattributed wait into one colour"><input type="checkbox" class="wtl-merge"> Merge waits</label>' +
      '<button class="wtl-reset" hidden>Reset zoom</button>' +
      '<label class="wtl-followwrap" hidden><input type="checkbox" class="wtl-follow"> Follow latest</label>' +
      '<span class="wtl-info"></span>' +
      '<span class="wtl-hint">Drag to zoom &middot; double-click or Esc to reset &middot; Ctrl+wheel zooms, Shift+wheel pans</span>' +
      '</div><div class="wtl-legend"></div>' +
      '<div class="wtl-scroll"><div class="wtl-content"><canvas class="wtl-canvas"></canvas><canvas class="wtl-overlay"></canvas></div><div class="wtl-empty" hidden>No requests yet</div></div>' +
      '<div class="wtl-tip"></div>';
    var $ = function (c) { return host.querySelector(c); };
    var elArm = $(".wtl-arm"), elRowH = $(".wtl-rowh"), elReset = $(".wtl-reset"), elFollow = $(".wtl-follow");
    var elInfo = $(".wtl-info"), elLegend = $(".wtl-legend"), elScroll = $(".wtl-scroll");
    var elContent = $(".wtl-content"), canvas = $(".wtl-canvas"), tip = $(".wtl-tip"), elEmpty = $(".wtl-empty");
    var ctx = canvas.getContext("2d");
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
      }).join("");
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
      if (!g) { for (var k = a.groups.length; k <= q.r; k++) a.groups[k] = { reqs: [], infl: [], subs: 1 }; g = a.groups[q.r]; }
      g.reqs.push(q);
      if ((q.u || 0) + 1 > g.subs) g.subs = (q.u || 0) + 1;
      a.dirty = true;
    }
    function indexArm(a) {
      a.groups = [];
      a.reqs.forEach(function (q) { addToGroup(a, q); });
      a.inflight = a.inflight || [];
      indexInflight(a);
      a.dirty = true;
    }
    function indexInflight(a) {
      a.groups.forEach(function (g) { g.infl = []; });
      (a.inflight || []).forEach(function (q) {
        var g = a.groups[q.r];
        if (!g) { for (var k = a.groups.length; k <= q.r; k++) a.groups[k] = { reqs: [], infl: [], subs: 1 }; g = a.groups[q.r]; }
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

    function resize() {
      var a = arm();
      var w = elScroll.clientWidth, h = elScroll.clientHeight;
      if (w < 50 || h < 50) return;
      var dpr = window.devicePixelRatio || 1;
      W = w; H = h;
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      canvas.style.marginBottom = (-h) + "px";
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlay.style.width = w + "px"; overlay.style.height = h + "px";
      overlay.style.marginBottom = (-h) + "px";
      overlay.width = canvas.width; overlay.height = canvas.height;
      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      plotL = 44;
      plotW = Math.max(50, W - plotL - RIGHT);
      elContent.style.height = Math.max(h, contentHeight(a)) + "px";
      draw();
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
      var g0 = Math.max(0, groupAt(a, scrollTop));
      for (var g = g0; g < a.groups.length; g++) {
        var gy = AXIS_H + groupTop(a, g) - scrollTop;
        if (gy > H) break;
        var grp = a.groups[g];
        if (g > 0) { ctx.fillStyle = "#1c2226"; ctx.fillRect(0, gy - GAP, W, 1); }
        // Session boundary: the slot moved on to a new session here.
        ctx.fillStyle = "rgba(199,159,241,0.55)";
        var gh = grp.subs * rowH;
        for (var m = 0; m < 2; m++) {
          var bl = m ? grp.infl : grp.reqs;
          for (var n = 0; n < bl.length; n++) if (bl[n].x & FLAG_SESSION) {
            var bx = mapX(bl[n].t, r);
            if (bx >= plotL && bx <= plotL + plotW) ctx.fillRect(Math.round(bx) - 1, gy - 1, 1, gh + 1);
          }
        }
        for (var k = 0; k < grp.reqs.length; k++) drawReq(grp.reqs[k], gy + (grp.reqs[k].u || 0) * rowH, rh, r, now);
        for (k = 0; k < grp.infl.length; k++) drawReq(grp.infl[k], gy + (grp.infl[k].u || 0) * rowH, rh, r, now);
        if (rowH >= 8) {
          var any = grp.reqs[0] || grp.infl[0];
          if (any) { ctx.fillStyle = "#6f767d"; ctx.textAlign = "right"; ctx.font = "10px monospace"; ctx.fillText(any.sl ? ((any.x & FLAG_HOT) ? "h" : "s") + any.sl : "s" + any.s, plotL - 5, gy + rh - 3); }
        }
      }
      ctx.restore();
      if (opts.live) {
        var nx = mapX(now, r);
        if (nx >= plotL && nx <= plotL + plotW) { ctx.strokeStyle = "#7C03EC"; ctx.beginPath(); ctx.moveTo(nx + 0.5, AXIS_H); ctx.lineTo(nx + 0.5, H); ctx.stroke(); }
      }
      drawOverlay();
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

    function drawReq(q, y, rh, r, now) {
      var t = q.t;
      var infl = (q.x & FLAG_INFL) !== 0;
      var end = infl ? Math.max(0, now - q.t) : q.e;
      if (mapX(q.t + end, r) < plotL || mapX(q.t, r) > plotL + plotW) return;
      var segs = segsFor(q, now);
      for (var i = 0; i < segs.length; i += 2) {
        var d = segs[i + 1];
        var x0 = mapX(t, r), x1 = mapX(t + d, r);
        t += d;
        if (x1 < plotL || x0 > plotL + plotW) continue;
        if (x0 < plotL) x0 = plotL;
        if (x1 > plotL + plotW) x1 = plotL + plotW;
        var w = x1 - x0; if (w < 0.6) w = 0.6;
        ctx.fillStyle = colors[segs[i]] || "#888";
        ctx.fillRect(x0, y, w, rh);
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
      var lists = [grp.reqs, grp.infl];
      for (var l = 0; l < 2; l++) for (var i = 0; i < lists[l].length; i++) {
        var q = lists[l][i];
        if ((q.u || 0) !== sub) continue;
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
      var head = q.sl ? "<b>" + ((q.x & FLAG_HOT) ? "hot slot " : "slot ") + q.sl + "</b> &middot; session " + q.s : "<b>series " + q.s + "</b>";
      var html = head + " turn " + q.n + ((q.x & FLAG_SESSION) ? " (first of session)" : "") + (inst ? " &middot; instance " + esc(inst.length > 24 ? inst.slice(0, 24) + "\u2026" : inst) : "") + (infl ? " <b>(in flight)</b>" : "") +
        ((q.x & FLAG_ERR) ? " <b style='color:#FF6B6B'>error</b>" : "") + ((q.x & FLAG_COLD) ? " cold start" : "") + ((q.x & FLAG_EMPTY) ? " empty" : "") + "<br>" +
        "start " + fmtClock(a.t0 + q.t, true) + " (+" + fmtDur(q.t) + ")<br>";
      var cw = q.cw || 0;
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
      if (q.tm && !(q.x & FLAG_TRUNC)) unattr = Math.max(0, e2e - (q.tm[0] + q.tm[1] + q.tm[2] + q.tm[3] + q.tm[4] + q.tm[5]));
      if (cw > 0 || (q.x & FLAG_WAITING) || (q.tm && (q.tm[0] > 0 || q.tm[1] > 0 || unattr > 1))) {
        html += "<table>";
        if ((q.x & FLAG_WAITING)) html += "<tr><td><i style='background:" + colors[WAIT_CLIENT] + "'></i>client wait (still waiting)</td><td>" + fmtDur(Math.max(0, now - q.t)) + "</td></tr>";
        else if (cw > 0) html += "<tr><td><i style='background:" + colors[WAIT_CLIENT] + "'></i>client wait (no free slot)</td><td>" + fmtDur(cw) + "</td></tr>";
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
      draw();
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
      if (drag) { drag.x1 = Math.max(plotL, Math.min(plotL + plotW, p.x)); drawOverlay(); return; }
      var h = pick(p);
      var g = rowAt(p);
      if (g !== hoverGroup || (h ? h.q : null) !== (hover ? hover.q : null)) { hover = h; hoverGroup = g; drawOverlay(); }
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
    canvas.addEventListener("mouseleave", function () { hover = null; hoverGroup = -1; tip.style.display = "none"; if (!drag) drawOverlay(); });
    canvas.addEventListener("dblclick", function () { userZoom(null, null); });
    elReset.addEventListener("click", function () { userZoom(null, null); });
    if (opts.handleEsc) window.addEventListener("keydown", function (e) { if (e.key === "Escape" && isZoomed()) userZoom(null, null); });
    canvas.addEventListener("wheel", function (e) {
      if (!(e.ctrlKey || e.metaKey || e.shiftKey)) return;   // plain wheel scrolls rows
      e.preventDefault();
      var r = range(), p = pos(e), span = r[1] - r[0], f = fullRange();
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
    elScroll.addEventListener("scroll", function () { hover = null; hoverGroup = -1; tip.style.display = "none"; draw(); });
    elMerge.addEventListener("change", function () { merge = elMerge.checked; buildLegend(); draw(); });
    elRowH.addEventListener("change", function () { rowH = parseInt(elRowH.value, 10); userRowH = true; resize(); });
    elArm.addEventListener("change", function () { selectArm(parseInt(elArm.value, 10)); if (opts.onArm) opts.onArm(cur); });
    elFollow.addEventListener("change", function () { draw(); });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(function () { resize(); }).observe(elScroll);
    else window.addEventListener("resize", resize);

    function selectArm(i) {
      cur = Math.max(0, Math.min(arms.length - 1, i));
      elArm.value = cur;
      hover = null; hoverGroup = -1; elScroll.scrollTop = 0;
      applyArm();
    }

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
      },
      selectArm: selectArm,
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
    };
    return api;
  }

  return { create: create, fmtDur: fmtDur };
})();
