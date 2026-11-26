/* Recycled-paper back-test UI: small progressive enhancements only.
   Core operation works without JS via normal form POST + redirect.
   This file adds file-name labels and an SVG feature scatter. */
(function () {
  "use strict";

  // ---- show chosen file names ----
  document.querySelectorAll('input[type="file"]').forEach(function (input) {
    input.addEventListener("change", function () {
      var label = document.querySelector('.file-name[data-for="' + input.id + '"]');
      if (label) {
        label.textContent = input.files && input.files.length
          ? input.files[0].name
          : "未选择文件";
      }
    });
  });

  var STATE = window.__STATE__ || null;
  var svg = document.getElementById("scatter");
  if (svg && STATE && STATE.loaded) {
    try {
      drawScatter(svg, STATE);
    } catch (err) {
      /* chart must never break the page */
      if (window && window.console) console.error("scatter render failed:", err);
    }
  }

  function el(name, attrs, text) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs || {}).forEach(function (k) {
      node.setAttribute(k, attrs[k]);
    });
    if (text != null) node.textContent = text;
    return node;
  }

  function drawScatter(svg, state) {
    var W = 520, H = 420;
    var M = { l: 52, r: 20, t: 18, b: 44 };
    var pw = W - M.l - M.r, ph = H - M.t - M.b;

    var pts = [];
    (state.seed_samples || []).forEach(function (s) {
      pts.push({ x: s.moisture, y: s.whiteness });
    });
    (state.batches || []).forEach(function (b) {
      pts.push({ x: b.moisture, y: b.whiteness });
    });
    Object.keys(state.models).forEach(function (k) {
      var c = state.models[k].centroids;
      if (c.qualified) pts.push({ x: c.qualified[0], y: c.qualified[1] });
      if (c.unqualified) pts.push({ x: c.unqualified[0], y: c.unqualified[1] });
    });

    var xs = pts.map(function (p) { return p.x; });
    var ys = pts.map(function (p) { return p.y; });
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    var xpad = Math.max((xmax - xmin) * 0.12, 1.5);
    var ypad = Math.max((ymax - ymin) * 0.12, 2.0);
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;

    function X(v) { return M.l + ((v - xmin) / (xmax - xmin)) * pw; }
    function Y(v) { return M.t + ph - ((v - ymin) / (ymax - ymin)) * ph; }

    // background + grid
    svg.appendChild(el("rect", { x: M.l, y: M.t, width: pw, height: ph, fill: "#ffffff", stroke: "#e5e7eb" }));
    for (var i = 0; i <= 5; i++) {
      var gx = xmin + (i / 5) * (xmax - xmin);
      var gy = ymin + (i / 5) * (ymax - ymin);
      svg.appendChild(el("line", { x1: X(gx), x2: X(gx), y1: M.t, y2: M.t + ph, stroke: "#f1f5f9" }));
      svg.appendChild(el("line", { x1: M.l, x2: M.l + pw, y1: Y(gy), y2: Y(gy), stroke: "#f1f5f9" }));
      var tx = el("text", { x: X(gx), y: H - 22, "text-anchor": "middle", "font-size": 10, fill: "#64748b" }, gx.toFixed(0));
      svg.appendChild(tx);
      svg.appendChild(el("text", { x: M.l - 7, y: Y(gy) + 3, "text-anchor": "end", "font-size": 10, fill: "#64748b" }, gy.toFixed(0)));
    }
    svg.appendChild(el("text", { x: M.l + pw / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: "#334155" }, "含水率 (%)"));
    var ylab = el("text", {
      x: 14, y: M.t + ph / 2, "text-anchor": "middle", "font-size": 12, fill: "#334155",
      transform: "rotate(-90 14 " + (M.t + ph / 2) + ")"
    }, "白度");
    svg.appendChild(ylab);

    var clip = el("clipPath", { id: "plot-clip" });
    clip.appendChild(el("rect", { x: M.l, y: M.t, width: pw, height: ph }));
    svg.appendChild(clip);

    var layer = el("g", { "clip-path": "url(#plot-clip)" });
    svg.appendChild(layer);

    // decision boundaries
    addBoundary(layer, state.models.fixed.centroids, "#94a3b8", "6 4", X, Y, xmin, xmax, ymin, ymax);
    addBoundary(layer, state.models.adaptive.centroids, "#7c3aed", null, X, Y, xmin, xmax, ymin, ymax);

    // seed samples
    (state.seed_samples || []).forEach(function (s) {
      layer.appendChild(el("circle", {
        cx: X(s.moisture), cy: Y(s.whiteness), r: 4,
        fill: s.qualified === 1 ? "#0f766e" : "#9f1239", opacity: 0.75
      }));
    });

    // batches
    (state.batches || []).forEach(function (b) {
      var truthKnown = !!b.receipt;
      var fill = !truthKnown ? "#2563eb" : (b.receipt.qualified === 1 ? "#14b8a6" : "#f43f5e");
      var wrong = b.error_fixed || b.error_adaptive;
      layer.appendChild(el("circle", {
        cx: X(b.moisture), cy: Y(b.whiteness), r: 6.5, fill: fill,
        stroke: wrong ? "#b91c1c" : "#ffffff", "stroke-width": wrong ? 2.5 : 1.5
      }));
      var t = el("text", {
        x: X(b.moisture) + 8, y: Y(b.whiteness) - 7,
        "font-size": 10.5, "font-weight": 700, fill: "#1e293b"
      }, b.batch);
      layer.appendChild(t);
    });

    // centroids as X markers
    addCentroid(svg, state.models.fixed.centroids, X, Y, "#334155", 7, false);
    addCentroid(svg, state.models.adaptive.centroids, X, Y, "#7c3aed", 9, true);
  }

  function addBoundary(g, centroids, color, dash, X, Y, xmin, xmax, ymin, ymax) {
    var cq = centroids.qualified, cb = centroids.unqualified;
    if (!cq || !cb) return;
    var dx = cq[0] - cb[0], dy = cq[1] - cb[1];
    var len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) return;
    // x·(dx,dy) = (|cq|^2 - |cb|^2)/2
    var k = (cq[0] * cq[0] + cq[1] * cq[1] - cb[0] * cb[0] - cb[1] * cb[1]) / 2;
    var bx = (k * dx) / len2, by = (k * dy) / len2;
    var vx = -dy, vy = dx;
    var scale = Math.max(xmax - xmin, ymax - ymin) * 3;
    var attrs = {
      x1: X(bx - vx * scale), y1: Y(by - vy * scale),
      x2: X(bx + vx * scale), y2: Y(by + vy * scale),
      stroke: color, "stroke-width": 1.8
    };
    if (dash) attrs["stroke-dasharray"] = dash;
    g.appendChild(el("line", attrs));
  }

  function addCentroid(svg, centroids, X, Y, color, r, labels) {
    [["qualified", "Q"], ["unqualified", "B"]].forEach(function (pair) {
      var c = centroids[pair[0]];
      if (!c) return;
      var cx = X(c[0]), cy = Y(c[1]);
      svg.appendChild(el("line", { x1: cx - r, y1: cy - r, x2: cx + r, y2: cy + r, stroke: "#fff", "stroke-width": 4.5, "stroke-linecap": "round" }));
      svg.appendChild(el("line", { x1: cx - r, y1: cy + r, x2: cx + r, y2: cy - r, stroke: "#fff", "stroke-width": 4.5, "stroke-linecap": "round" }));
      svg.appendChild(el("line", { x1: cx - r, y1: cy - r, x2: cx + r, y2: cy + r, stroke: color, "stroke-width": 2.4, "stroke-linecap": "round" }));
      svg.appendChild(el("line", { x1: cx - r, y1: cy + r, x2: cx + r, y2: cy - r, stroke: color, "stroke-width": 2.4, "stroke-linecap": "round" }));
      if (labels) {
        svg.appendChild(el("text", { x: cx + r + 3, y: cy + 4, "font-size": 11, "font-weight": 800, fill: color }, pair[1]));
      }
    });
  }
})();
