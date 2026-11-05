/* Gap Explorer. The page computes nothing: every number is looked up in
   docs/data/*.json (written by `python -m ordergaps.viz build`) and only
   formatted here. Colors come from tokens.css at render time, so light and
   dark mode need no second palette; charts re-render when the scheme flips. */
(function () {
  "use strict";

  var state = { summary: null, grid: null, week: null, shareIndex: 0, gapIndex: 0, charts: {} };

  function byId(id) { return document.getElementById(id); }

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function pct(value, digits) {
    if (value === null || value === undefined) { return "–"; }
    return (value * 100).toFixed(digits) + "%";
  }

  function num(value, digits) {
    if (value === null || value === undefined) { return "–"; }
    return value.toFixed(digits);
  }

  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) { node.textContent = text; }
    if (className) { node.className = className; }
    return node;
  }

  function fillTable(table, headers, rows, numericColumns) {
    table.textContent = "";
    var thead = el("thead");
    var headRow = el("tr");
    headers.forEach(function (header, index) {
      var th = el("th", header);
      if (numericColumns && numericColumns.indexOf(index) !== -1) { th.className = "num"; }
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function (row) {
      var tr = el("tr");
      if (row.className) { tr.className = row.className; }
      row.cells.forEach(function (cell, index) {
        var td = el("td", cell);
        if (numericColumns && numericColumns.indexOf(index) !== -1) { td.className = "num"; }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  function shareRow() { return state.summary.by_share[state.shareIndex]; }
  function weekEntry() { return state.week.by_share[state.shareIndex]; }

  /* ---------- static text ---------- */

  function renderStatic() {
    var summary = state.summary;
    document.querySelectorAll("[data-prov]").forEach(function (node) {
      node.textContent = summary.provenance[node.getAttribute("data-prov")] || "";
    });
    document.querySelectorAll("[data-why]").forEach(function (node) {
      var why = summary.why[node.getAttribute("data-why")];
      node.textContent = "";
      node.appendChild(el("b", "Why this metric"));
      node.appendChild(el("p", "Owner's question: " + why.question));
      node.appendChild(el("p", "Kind of number: " + why.kind));
      node.appendChild(el("p", "What would make it lie: " + why.lie));
    });
    byId("footer-text").textContent = summary.footer;
    byId("tile-caveat").textContent = summary.caveat;
    var acceptance = summary.acceptance;
    byId("acceptance-note").textContent =
      "Acceptance band (frozen before the seeds): Spearman rho = " + num(acceptance.spearman, 3) +
      " (required <= -0.8), inversions = " + acceptance.inversions + " (required <= 1): " +
      (acceptance.passed ? "PASS" : "FAIL") + ".";

    var ticks = byId("ticks");
    ticks.textContent = "";
    var last = summary.by_share.length - 1;
    summary.by_share.forEach(function (row, index) {
      var stop = el("span", null, "stop");
      if (index === 0) { stop.classList.add("first"); }
      if (index === last) { stop.classList.add("last"); }
      stop.style.left = (last === 0 ? 0 : (index / last) * 100) + "%";
      stop.appendChild(el("span", null, "mark"));
      stop.appendChild(document.createTextNode(row.share_label));
      ticks.appendChild(stop);
    });
    var slider = byId("share");
    slider.max = String(last);
    slider.value = String(state.shareIndex);
  }

  /* ---------- per-share views ---------- */

  function renderTiles() {
    var row = shareRow();
    byId("share-now").textContent = row.share_label;
    document.querySelectorAll("[data-share-now]").forEach(function (node) {
      node.textContent = row.share_label;
    });
    byId("tile-windows").textContent = row.text.windows;
    byId("tile-weekly").textContent = row.text.weekly_err;
    byId("tile-weekly-note").textContent =
      "uniform-in-time baseline " + row.text.uniform_err + " (" + row.text.versus_uniform + ")";
    byId("tile-yield").textContent = row.text.yield_text;
    var stops = byId("ticks").children;
    for (var index = 0; index < stops.length; index += 1) {
      stops[index].classList.toggle("on", index === state.shareIndex);
    }
    byId("share").setAttribute("aria-valuetext", row.share_label);
  }

  function renderHeroTable() {
    var rows = state.summary.by_share.map(function (row) {
      return {
        className: row.share === shareRow().share ? "chosen" : "",
        cells: [row.share_label, num(row.mean_fraction_observed, 3), row.text.weekly_err, row.text.uniform_err]
      };
    });
    fillTable(byId("hero-table"),
      ["vendor share", "fraction observed", "median weekly error", "uniform baseline"], rows, [1, 2, 3]);
  }

  function renderWeekText() {
    var entry = weekEntry();
    byId("week-note").textContent =
      "Week total: estimated " + num(entry.week_est_total, 1) + " orders vs " + entry.week_true_total +
      " true (" + pct(entry.week_abs_rel_err, 2) + " off). You saw " + entry.n_observed +
      " of the store's orders over all simulated weeks; " + entry.n_gaps_touching_week +
      " gaps touch this week.";
    var chips = byId("day-chips");
    chips.textContent = "";
    entry.days.forEach(function (day) {
      var chip = el("div", null, "chip " + day.label);
      chip.appendChild(el("b", day.weekday));
      chip.appendChild(document.createTextNode(day.label));
      chips.appendChild(chip);
    });
    var rows = entry.days.map(function (day) {
      return {
        cells: [day.day_text, day.label, String(day.seen), num(day.unseen_in_store, 1),
          num(day.unseen_online, 1), num(day.est_total, 1), String(day.true_total)]
      };
    });
    fillTable(byId("week-table"),
      ["day", "label", "seen", "allocated in-store", "allocated online", "estimated", "true"],
      rows, [2, 3, 4, 5, 6]);
  }

  function renderPaceText() {
    var row = shareRow();
    byId("pace-note").textContent =
      "Mean bias of the learned pace at this share, all seeds: " + row.text.mean_bias +
      ". The vendor's own sale times oversample busy hours, so the pace runs high.";
    var rows = weekEntry().pace.map(function (day) {
      return {
        cells: [day.weekday, day.open ? num(day.learned_per_hour, 2) : "closed",
          day.open ? num(day.true_per_hour, 2) : "closed", day.source]
      };
    });
    fillTable(byId("pace-table"), ["weekday", "learned per hour", "true per hour", "source"], rows, [1, 2]);
  }

  function renderGaps() {
    var gaps = weekEntry().gaps;
    if (state.gapIndex >= gaps.length) { state.gapIndex = 0; }
    var buttons = byId("gap-buttons");
    buttons.textContent = "";
    gaps.forEach(function (gap, index) {
      var button = el("button", gap.t1.slice(5, 10) + " · " + gap.unseen + " unseen");
      button.type = "button";
      button.setAttribute("aria-pressed", index === state.gapIndex ? "true" : "false");
      button.addEventListener("click", function () {
        state.gapIndex = index;
        renderGaps();
      });
      buttons.appendChild(button);
    });
    var gap = gaps[state.gapIndex];
    var story = byId("gap-story");
    story.textContent = "";
    if (!gap) { return; }
    story.appendChild(document.createTextNode(
      "Saw #" + gap.n1 + " at " + gap.t1 + " and #" + gap.n2 + " at " + gap.t2 + ", so exactly "));
    story.appendChild(el("span", gap.n2 + " − " + gap.n1 + " − 1 = " + gap.unseen, "big"));
    story.appendChild(document.createTextNode(
      " orders happened in between (counted). The simulator created " + gap.truth_between + " "));
    var match = gap.truth_between === gap.unseen;
    story.appendChild(el("span", match ? "equal" : "different", "badge " + (match ? "good" : "bad")));
    story.appendChild(document.createTextNode(
      ". Which day and channel they went to is modelled (" + gap.kind + " gap):"));
    var rows = gap.split.map(function (part) {
      return { className: "modelled", cells: [part.day, num(part.in_store, 2), num(part.online, 2)] };
    });
    fillTable(byId("gap-table"), ["day", "in-store (modelled)", "online / after hours (modelled)"], rows, [1, 2]);
  }

  function renderCountedTable() {
    var row = shareRow();
    var rows = row.counted_rows.map(function (item) {
      var counted = item.kind.indexOf("counted") === 0;
      return { className: counted ? "counted" : "modelled", cells: [item.quantity, item.kind, item.error] };
    });
    rows.push({
      className: "modelled",
      cells: ["weekly total volume vs uniform-in-time baseline", "model vs baseline",
        row.text.weekly_err + " vs " + row.text.uniform_err + " (" + row.text.versus_uniform + ")"]
    });
    fillTable(byId("counted-table"), ["quantity", "kind", "error"], rows);
  }

  /* ---------- charts ---------- */

  function chartsAvailable() { return typeof window.Chart === "function"; }

  function showFallbacks() {
    document.querySelectorAll("[data-fallback]").forEach(function (node) { node.hidden = false; });
    document.querySelectorAll("[data-fallback-details]").forEach(function (node) { node.open = true; });
    ["hero-box", "week-box", "pace-box"].forEach(function (id) { byId(id).hidden = true; });
  }

  function baseOptions() {
    var text = token("--text");
    var muted = token("--text-muted");
    var border = token("--border");
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      color: text,
      plugins: { legend: { labels: { color: text, boxWidth: 12 } }, tooltip: { enabled: true } },
      scales: {
        x: { ticks: { color: muted }, grid: { color: border }, title: { display: true, color: muted } },
        y: { ticks: { color: muted }, grid: { color: border }, title: { display: true, color: muted } }
      }
    };
  }

  /* Log-axis tick text: label only the 1, 2 and 5 steps of each decade. */
  function logTick(value, digits) {
    var decade = Math.pow(10, Math.floor(Math.log10(value)));
    var lead = Math.round(value / decade);
    if (lead !== 1 && lead !== 2 && lead !== 5) { return ""; }
    return pct(value, digits);
  }

  function destroyCharts() {
    Object.keys(state.charts).forEach(function (key) { state.charts[key].destroy(); });
    state.charts = {};
  }

  function heroData() {
    var seedPoints = state.grid.cells.map(function (cell) {
      return { x: cell.fraction_observed, y: cell.weekly_err_median };
    });
    var median = state.summary.by_share.map(function (row) {
      return { x: row.mean_fraction_observed, y: row.median_weekly_err };
    });
    var chosen = shareRow();
    return { seeds: seedPoints, median: median, chosen: [{ x: chosen.mean_fraction_observed, y: chosen.median_weekly_err }] };
  }

  function buildHero() {
    var data = heroData();
    var options = baseOptions();
    options.scales.x.type = "logarithmic";
    options.scales.y.type = "logarithmic";
    options.scales.x.title.text = "fraction of store orders observed";
    options.scales.y.title.text = "median weekly error";
    options.scales.x.ticks.callback = function (value) { return logTick(value, 0); };
    options.scales.y.ticks.callback = function (value) { return logTick(value, 3); };
    options.plugins.tooltip.callbacks = {
      label: function (context) {
        return context.dataset.label + ": " + pct(context.parsed.x, 1) + " seen, error " + pct(context.parsed.y, 4);
      }
    };
    state.charts.hero = new window.Chart(byId("hero-chart"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "this share", data: data.chosen, pointRadius: 9, pointHoverRadius: 10, borderWidth: 3,
            borderColor: token("--warn"), backgroundColor: "transparent", order: 0 },
          { label: "median across seeds", data: data.median, showLine: true, borderColor: token("--accent"),
            backgroundColor: token("--accent"), pointRadius: 4, borderWidth: 2, order: 1 },
          { label: "one seed", data: data.seeds, pointRadius: 3, borderColor: token("--baseline"),
            backgroundColor: token("--baseline"), order: 2 }
        ]
      },
      options: options
    });
  }

  function updateHero() {
    var chart = state.charts.hero;
    chart.data.datasets[0].data = heroData().chosen;
    chart.update();
  }

  function weekDatasets() {
    var days = weekEntry().days;
    return {
      labels: days.map(function (day) { return [day.day_text, day.label]; }),
      seen: days.map(function (day) { return day.seen; }),
      inStore: days.map(function (day) { return day.unseen_in_store; }),
      online: days.map(function (day) { return day.unseen_online; }),
      truth: days.map(function (day) { return day.true_total; })
    };
  }

  function buildWeek() {
    var data = weekDatasets();
    var options = baseOptions();
    options.scales.x.stacked = true;
    options.scales.y.stacked = true;
    options.scales.x.title.display = false;
    options.scales.y.title.text = "orders that day";
    options.scales.y.beginAtZero = true;
    options.plugins.tooltip.callbacks = {
      label: function (context) { return context.dataset.label + ": " + num(context.parsed.y, 1); }
    };
    state.charts.week = new window.Chart(byId("week-chart"), {
      data: {
        labels: data.labels,
        datasets: [
          { type: "line", label: "true total", data: data.truth, showLine: false, pointStyle: "line",
            pointRadius: 14, pointHoverRadius: 14, borderWidth: 3, borderColor: token("--text"),
            backgroundColor: token("--text"), stack: "truth", order: 0 },
          { type: "bar", label: "seen (counted)", data: data.seen, backgroundColor: token("--series-1"), stack: "est", order: 1 },
          { type: "bar", label: "allocated in-store (modelled)", data: data.inStore, backgroundColor: token("--series-3"), stack: "est", order: 1 },
          { type: "bar", label: "allocated online (modelled)", data: data.online, backgroundColor: token("--series-2"), stack: "est", order: 1 }
        ]
      },
      options: options
    });
  }

  function updateWeek() {
    var chart = state.charts.week;
    var data = weekDatasets();
    chart.data.labels = data.labels;
    chart.data.datasets[0].data = data.truth;
    chart.data.datasets[1].data = data.seen;
    chart.data.datasets[2].data = data.inStore;
    chart.data.datasets[3].data = data.online;
    chart.update();
  }

  function paceDatasets() {
    var pace = weekEntry().pace;
    return {
      labels: pace.map(function (day) { return day.weekday; }),
      learned: pace.map(function (day) { return day.learned_per_hour; }),
      truth: pace.map(function (day) { return day.true_per_hour; })
    };
  }

  function buildPace() {
    var data = paceDatasets();
    var options = baseOptions();
    options.scales.x.title.display = false;
    options.scales.y.title.text = "orders per open hour";
    options.scales.y.beginAtZero = true;
    options.plugins.tooltip.callbacks = {
      label: function (context) {
        return context.dataset.label + ": " + (context.parsed.y === null ? "closed" : num(context.parsed.y, 2));
      }
    };
    state.charts.pace = new window.Chart(byId("pace-chart"), {
      type: "bar",
      data: {
        labels: data.labels,
        datasets: [
          { label: "learned by the estimator", data: data.learned, backgroundColor: token("--accent") },
          { label: "true (simulator)", data: data.truth, backgroundColor: token("--baseline") }
        ]
      },
      options: options
    });
  }

  function updatePace() {
    var chart = state.charts.pace;
    var data = paceDatasets();
    chart.data.datasets[0].data = data.learned;
    chart.data.datasets[1].data = data.truth;
    chart.update();
  }

  function buildCharts() {
    if (!chartsAvailable()) { showFallbacks(); return; }
    destroyCharts();
    buildHero();
    buildWeek();
    buildPace();
  }

  function updateCharts() {
    if (!chartsAvailable() || !state.charts.hero) { return; }
    updateHero();
    updateWeek();
    updatePace();
  }

  /* ---------- wiring ---------- */

  function renderShare() {
    renderTiles();
    renderHeroTable();
    renderWeekText();
    renderPaceText();
    renderGaps();
    renderCountedTable();
    updateCharts();
  }

  function loadJson(path) {
    return fetch(path).then(function (response) {
      if (!response.ok) { throw new Error(path + " " + response.status); }
      return response.json();
    });
  }

  function start() {
    Promise.all([loadJson("data/summary.json"), loadJson("data/share_grid.json"), loadJson("data/week.json")])
      .then(function (loaded) {
        state.summary = loaded[0];
        state.grid = loaded[1];
        state.week = loaded[2];
        var shares = state.summary.shares;
        state.shareIndex = Math.max(0, shares.indexOf(state.summary.default_share));
        renderStatic();
        buildCharts();
        renderShare();
        byId("share").addEventListener("input", function (event) {
          state.shareIndex = Number(event.target.value);
          renderShare();
        });
        if (window.matchMedia) {
          var scheme = window.matchMedia("(prefers-color-scheme: dark)");
          var rebuild = function () { buildCharts(); updateCharts(); };
          if (scheme.addEventListener) { scheme.addEventListener("change", rebuild); }
          else if (scheme.addListener) { scheme.addListener(rebuild); }
        }
      })
      .catch(function () { byId("load-error").hidden = false; });
  }

  start();
})();
