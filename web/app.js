const state = {
  config: null,
  selected: new Set(),
  algorithm: "ppo",
  jobId: null,
  poller: null,
  chart: null,
  snapshots: [],
};

const $ = (id) => document.getElementById(id);

function formatPercent(value) {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

function formatMoney(value) {
  return new Intl.NumberFormat("zh-TW", {
    style: "currency",
    currency: "TWD",
    maximumFractionDigits: 0,
  }).format(value);
}

function setClock() {
  $("clock").textContent = new Intl.DateTimeFormat("zh-TW", {
    dateStyle: "medium",
    timeStyle: "medium",
    hour12: false,
  }).format(new Date());
}

function renderStocks(filter = "") {
  const query = filter.trim().toLowerCase();
  const stocks = state.config.stocks.filter((stock) =>
    `${stock.code}${stock.name}${stock.sector}`.toLowerCase().includes(query)
  );
  $("stockList").innerHTML = stocks.map((stock) => `
    <button type="button" class="stock-item ${state.selected.has(stock.symbol) ? "selected" : ""}" data-symbol="${stock.symbol}">
      <span class="stock-check"></span><b>${stock.code}</b><span>${stock.name}</span><small>${stock.sector}</small>
    </button>`).join("");
  $("selectedCount").textContent = state.selected.size;
  document.querySelectorAll(".stock-item").forEach((button) => {
    button.addEventListener("click", () => {
      const symbol = button.dataset.symbol;
      state.selected.has(symbol) ? state.selected.delete(symbol) : state.selected.add(symbol);
      renderStocks($("stockSearch").value);
    });
  });
}

function selectAlgorithm(key) {
  state.algorithm = key;
  document.querySelectorAll(".algo-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.key === key);
  });
  const algo = state.config.algorithms[key];
  $("activeAlgoShort").textContent = algo.label.replace(" + ", "+");
  $("algoDescription").textContent = algo.description;
  const defaults = algo.defaults;
  if (defaults.learning_rate !== undefined) $("learningRate").value = defaults.learning_rate;
  if (defaults.n_steps !== undefined) $("nSteps").value = defaults.n_steps;
  if (defaults.batch_size !== undefined) {
    const batchSelect = $("batchSize");
    if (![...batchSelect.options].some((option) => Number(option.value) === defaults.batch_size)) {
      batchSelect.add(new Option(defaults.batch_size, defaults.batch_size));
    }
    batchSelect.value = defaults.batch_size;
  }
}

function renderAlgorithms() {
  $("algorithmGrid").innerHTML = Object.entries(state.config.algorithms).map(([key, algo]) => `
    <button type="button" class="algo-card ${key === state.algorithm ? "active" : ""}" data-key="${key}">
      <b>${algo.label}</b><small>${algo.family}</small>
    </button>`).join("");
  document.querySelectorAll(".algo-card").forEach((card) => {
    card.addEventListener("click", () => selectAlgorithm(card.dataset.key));
  });
  selectAlgorithm(state.algorithm);
}

function requestPayload() {
  return {
    algorithm: state.algorithm,
    tickers: [...state.selected],
    train_start: $("trainStart").value,
    trade_start: $("tradeStart").value,
    trade_end: $("tradeEnd").value,
    total_timesteps: Number($("timesteps").value),
    learning_rate: Number($("learningRate").value),
    n_steps: Number($("nSteps").value),
    batch_size: Number($("batchSize").value),
    initial_amount: 1000000,
    hmax: 100,
    transaction_cost: 0.001,
    sequence_length: 16,
    seed: 42,
    device: "auto",
  };
}

async function startJob() {
  if (!state.selected.size) {
    alert("請至少選擇一檔股票。");
    return;
  }
  $("trainButton").disabled = true;
  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload()),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
      throw new Error(detail || "無法建立訓練工作");
    }
    state.jobId = data.id;
    updateJob(data);
    clearInterval(state.poller);
    state.poller = setInterval(pollJob, 1500);
  } catch (error) {
    alert(error.message);
    $("trainButton").disabled = false;
  }
}

async function pollJob() {
  const response = await fetch(`/api/jobs/${state.jobId}`);
  if (!response.ok) return;
  const job = await response.json();
  updateJob(job);
  if (["completed", "failed"].includes(job.status)) {
    clearInterval(state.poller);
    $("trainButton").disabled = false;
  }
}

function updateJob(job) {
  $("jobStatus").textContent = job.status.toUpperCase();
  $("jobStatus").className = `job-status ${job.status}`;
  $("jobStage").textContent = job.stage;
  $("jobDetail").textContent = job.error || `${state.config.algorithms[job.config.algorithm].label} · ${job.config.tickers.length} 檔股票 · ${job.config.total_timesteps.toLocaleString()} timesteps`;
  const progress = job.progress || 0;
  $("progressValue").textContent = `${progress}%`;
  $("progressRing").style.setProperty("--progress", `${progress * 3.6}deg`);
  $("logList").innerHTML = job.logs.length ? job.logs.map((line) =>
    `<p><time>${line.time}</time><span class="${line.level}">${line.message}</span></p>`
  ).join("") : "<p><time>--:--:--</time><span>工作已排入背景執行</span></p>";
  $("logList").scrollTop = $("logList").scrollHeight;
  if (job.result) renderResult(job.result);
}

function renderResult(result) {
  $("metricReturn").textContent = formatPercent(result.metrics.total_return);
  $("metricSharpe").textContent = result.metrics.sharpe.toFixed(2);
  $("metricDrawdown").textContent = formatPercent(result.metrics.max_drawdown);
  $("metricWinRate").textContent = formatPercent(result.metrics.win_rate);
  renderChart(result.curve);
  renderSnapshots(result.daily_snapshots);
}

function renderSnapshots(snapshots) {
  state.snapshots = snapshots || [];
  const picker = $("snapshotDate");
  picker.disabled = !state.snapshots.length;
  picker.innerHTML = state.snapshots.map((snapshot) =>
    `<option value="${snapshot.date}">${snapshot.date.replaceAll("-", "/")}</option>`
  ).join("");
  if (state.snapshots.length) {
    picker.value = state.snapshots[state.snapshots.length - 1].date;
    renderSnapshot(picker.value);
  }
}

function renderSnapshot(date) {
  const snapshot = state.snapshots.find((item) => item.date === date);
  if (!snapshot) return;
  $("snapshotCash").textContent = formatMoney(snapshot.cash);
  $("snapshotCashWeight").textContent = `現金比例 ${(snapshot.cash_weight * 100).toFixed(2)}%`;
  $("snapshotStockValue").textContent = formatMoney(snapshot.stock_value);
  $("snapshotTotalValue").textContent = formatMoney(snapshot.total_value);
  $("snapshotRows").innerHTML = snapshot.assets.map((asset) => `
    <tr>
      <td><strong>${asset.ticker.replace(".TW", "")}</strong><small>${stockName(asset.ticker)}</small></td>
      <td><span class="decision ${asset.action > 0 ? "buy" : asset.action < 0 ? "sell" : "hold"}">${asset.decision}</span></td>
      <td class="${asset.action > 0 ? "positive" : asset.action < 0 ? "negative" : ""}">${asset.action > 0 ? "+" : ""}${asset.action}</td>
      <td>${asset.shares.toLocaleString()}</td>
      <td>${formatMoney(asset.price)}</td>
      <td>${formatMoney(asset.market_value)}</td>
      <td>${(asset.weight * 100).toFixed(2)}%</td>
    </tr>
  `).join("");
}

function stockName(ticker) {
  return state.config.stocks.find((stock) => stock.symbol === ticker)?.name || ticker;
}

function renderChart(curve) {
  $("chartEmpty").style.display = "none";
  const base = curve.agent[0];
  const data = {
    labels: curve.dates,
    datasets: [
      {
        label: "RL Agent",
        data: curve.agent.map((value) => (value / base - 1) * 100),
        borderColor: "#50e3c2",
        backgroundColor: "rgba(80,227,194,.08)",
        fill: true,
        tension: .25,
        pointRadius: 0,
      },
      {
        label: curve.baseline_name || "等權買入持有",
        data: curve.baseline.map((value) => (value / base - 1) * 100),
        borderColor: "#526074",
        borderDash: [5, 5],
        tension: .25,
        pointRadius: 0,
      },
    ],
  };
  if (state.chart) state.chart.destroy();
  state.chart = new Chart($("returnChart"), {
    type: "line",
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "rgba(82,96,116,.12)" }, ticks: { color: "#526074", maxTicksLimit: 8 } },
        y: { grid: { color: "rgba(82,96,116,.12)" }, ticks: { color: "#526074", callback: (value) => `${value}%` } },
      },
    },
  });
}

async function init() {
  setClock();
  setInterval(setClock, 1000);
  const response = await fetch("/api/config");
  state.config = await response.json();
  state.selected = new Set(state.config.default_tickers);
  renderStocks();
  renderAlgorithms();
  const missing = Object.entries(state.config.dependencies)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  $("dependencyBadge").textContent = missing.length ? `缺少 ${missing.join(", ")}` : "訓練環境就緒";
  $("dependencyBadge").className = `pill ${missing.length ? "warn" : "good"}`;

  const jobsResponse = await fetch("/api/jobs");
  const jobs = await jobsResponse.json();
  if (jobs.length) {
    const latestResponse = await fetch(`/api/jobs/${jobs[0].id}`);
    const latestJob = await latestResponse.json();
    state.jobId = latestJob.id;
    updateJob(latestJob);
    if (["queued", "running"].includes(latestJob.status)) {
      $("trainButton").disabled = true;
      state.poller = setInterval(pollJob, 1500);
    }
  }
}

$("stockSearch").addEventListener("input", (event) => renderStocks(event.target.value));
$("selectTop10").addEventListener("click", () => {
  state.selected = new Set(state.config.default_tickers);
  renderStocks($("stockSearch").value);
});
$("trainButton").addEventListener("click", startJob);
$("snapshotDate").addEventListener("change", (event) => renderSnapshot(event.target.value));
init();
