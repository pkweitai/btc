let charts = {};
let META = null;

// ---------- helpers ----------
async function fetchJSON(url){
  const r = await fetch(url);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
function clampPoints(points, maxN=800){
  if(points.length<=maxN) return points;
  const step = Math.ceil(points.length/maxN);
  const out=[]; for(let i=0;i<points.length;i+=step) out.push(points[i]);
  return out;
}
function makeChart(ctxId, label, points, yKey="value"){
  const el = document.getElementById(ctxId);
  if(!el) return;
  const ctx = el.getContext("2d");
  if(charts[ctxId]) charts[ctxId].destroy();

  points = points.filter(p => typeof p[yKey] === "number" && isFinite(p[yKey]));
  points = clampPoints(points, 800);

  const data = {
    labels: points.map(p => p.t),
    datasets: [{
      label,
      data: points.map(p => p[yKey]),
      fill: false,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 2
    }]
  };

  charts[ctxId] = new Chart(ctx, {
    type: "line",
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { boxWidth: 10 } },
        decimation: { enabled: true, algorithm: "lttb", samples: 500 },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.formattedValue}` } }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: { beginAtZero: false }
      }
    }
  });
}
function setOptions(selectEl, items, {placeholder="(select)", keepFirst=false}={}){
  const el = (typeof selectEl === "string") ? document.getElementById(selectEl) : selectEl;
  if(!el) return;
  el.innerHTML = "";
  if(placeholder && !keepFirst){
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = placeholder;
    el.appendChild(opt);
  }
  items.forEach(v=>{
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    el.appendChild(opt);
  });
}
function uniq(a){ return Array.from(new Set(a)); }

// ---------- populate UI from /api/summary ----------
async function loadMeta(){
  META = await fetchJSON("/api/summary");
  // raw JSON
  const pre = document.getElementById("meta");
  if(pre) pre.textContent = JSON.stringify(META, null, 2);

  // summary table
  const tb = document.querySelector("#summaryTable tbody");
  if(tb){
    tb.innerHTML = "";
    const order = ["ohlcv","funding","open_interest","blockchain","blockchair","mempool","bitnodes","fng","orderbook"];
    const keys = order.filter(k => META[k] !== undefined).concat(Object.keys(META).filter(k => !order.includes(k)));
    keys.forEach(k=>{
      const m = META[k] || {};
      const tr = document.createElement("tr");
      const groups = (m.groups || []).slice(0, 6).map(g => Array.isArray(g)? g.join(" · "): String(g)).join(" | ");
      tr.innerHTML = `
        <td><strong>${k}</strong></td>
        <td>${m.present ? '<span class="badge-ok">yes</span>' : '<span class="badge-miss">no</span>'}</td>
        <td>${m.rows ?? ""}</td>
        <td>${m.last_ts ?? ""}</td>
        <td>${groups}</td>
      `;
      tb.appendChild(tr);
    });
  }

  // Dropdown data
  // OHLCV symbols
  const ohlcvSyms = META.ohlcv?.groups ? META.ohlcv.groups.map(g => Array.isArray(g)? g[0]: g).filter(Boolean) : [];
  setOptions("ohlcvSymbol", uniq(ohlcvSyms), {placeholder: "(symbol)"});

  // Orderbook dropdowns (if groups include exchange/symbol)
  const obGroups = META.orderbook?.groups || [];
  const obEx = uniq(obGroups.map(g => Array.isArray(g)? g[0] : "").filter(Boolean));
  const obSym = uniq(obGroups.map(g => Array.isArray(g)? g[1] : "").filter(Boolean));
  setOptions("obEx", obEx, {placeholder: "(exchange)"});
  setOptions("obSym", obSym, {placeholder: "(symbol)"});

  // Funding/Open interest: groups may be [exchange, instrument, symbol]
  const fundG = META.funding?.groups || [];
  const oiG   = META.open_interest?.groups || [];
  const allG  = fundG.concat(oiG);

  const allEx = uniq(allG.map(g => Array.isArray(g)? g[0] : "").filter(Boolean));
  setOptions("fundEx", allEx, {placeholder: "(exchange)"});
  setOptions("oiEx",   allEx, {placeholder: "(exchange)"});

  // Build instrument lists keyed by exchange
  const mapInstByEx = {};
  allG.forEach(g=>{
    if(Array.isArray(g) && g.length>=2){
      const ex = g[0], inst = g[1];
      if(!mapInstByEx[ex]) mapInstByEx[ex] = new Set();
      if(inst) mapInstByEx[ex].add(inst);
    }
  });

  function updateInst(exSelId, instSelId){
    const ex = document.getElementById(exSelId).value;
    const items = ex && mapInstByEx[ex] ? Array.from(mapInstByEx[ex]) : [];
    setOptions(instSelId, items, {placeholder: "(instrument)"});
  }

  // hook exchange change to update instrument list
  document.getElementById("fundEx").addEventListener("change", ()=>updateInst("fundEx","fundInst"));
  document.getElementById("oiEx").addEventListener("change",   ()=>updateInst("oiEx","oiInst"));
  // initial fill (if there is only one exchange, auto-populate)
  updateInst("fundEx","fundInst");
  updateInst("oiEx","oiInst");

  // On-chain metrics per source
  const metricsBySource = {
    "onchain_blockchain_dot_com.parquet": META.blockchain?.groups?.map(g=>Array.isArray(g)? g[0]: g) || [],
    "onchain_blockchair.parquet":         META.blockchair?.groups?.map(g=>Array.isArray(g)? g[0]: g) || [],
    "onchain_mempool_space.parquet":      META.mempool?.groups?.map(g=>Array.isArray(g)? g[0]: g) || []
  };
  function refreshOnchainMetric(){
    const src = document.getElementById("onchainSource").value;
    const metrics = uniq(metricsBySource[src] || []);
    setOptions("onchainMetric", metrics, {placeholder: "(metric)"});
  }
  document.getElementById("onchainSource").addEventListener("change", refreshOnchainMetric);
  refreshOnchainMetric();

  // reasonable defaults (pick first options if present)
  const sel = id => document.getElementById(id);
  if(sel("ohlcvSymbol")?.options.length>1) sel("ohlcvSymbol").selectedIndex = 1;
  if(sel("fundEx")?.options.length>1){ sel("fundEx").selectedIndex = 1; updateInst("fundEx","fundInst"); }
  if(sel("oiEx")?.options.length>1){ sel("oiEx").selectedIndex = 1; updateInst("oiEx","oiInst"); }
  if(sel("onchainMetric")?.options.length>1) sel("onchainMetric").selectedIndex = 1;
}

// ---------- data loaders ----------
async function loadOhlcv(){
  const sym = document.getElementById("ohlcvSymbol").value.trim();
  const n = document.getElementById("ohlcvN").value.trim();
  let url = "/api/ohlcv?";
  if(sym) url += `symbol=${encodeURIComponent(sym)}&`;
  if(n) url += `last_n=${encodeURIComponent(n)}&`;
  const data = await fetchJSON(url);
  const pts = data.map(d => ({ t: d.t, value: d.close }));
  makeChart("ohlcvChart", sym ? `${sym} Close` : "Close", pts);
}
async function loadFunding(){
  const ex = document.getElementById("fundEx").value.trim();
  const inst = document.getElementById("fundInst").value.trim();
  let url = "/api/funding?";
  if(ex) url += `exchange=${encodeURIComponent(ex)}&`;
  if(inst) url += `instrument=${encodeURIComponent(inst)}&`;
  const data = await fetchJSON(url);
  const pts = data.map(d => ({ t: d.t, value: d.funding_rate }));
  makeChart("fundingChart", "Funding Rate", pts);
}
async function loadOI(){
  const ex = document.getElementById("oiEx").value.trim();
  const inst = document.getElementById("oiInst").value.trim();
  let url = "/api/open_interest?";
  if(ex) url += `exchange=${encodeURIComponent(ex)}&`;
  if(inst) url += `instrument=${encodeURIComponent(inst)}&`;
  const data = await fetchJSON(url);
  const pts = data.map(d => ({ t: d.t, value: d.open_interest }));
  makeChart("oiChart", "Open Interest", pts);
}
async function loadOnchain(){
  const src = document.getElementById("onchainSource").value;
  const metric = document.getElementById("onchainMetric").value.trim();
  let url = `/api/onchain?source=${encodeURIComponent(src)}`;
  if(metric) url += `&metric=${encodeURIComponent(metric)}`;
  const data = await fetchJSON(url);
  const pts = data.map(d => ({ t: d.t, value: d.value }));
  makeChart("onchainChart", metric || "On-chain", pts);
}
async function loadFNG(){
  const data = await fetchJSON("/api/fng");
  const pts = data.map(d => ({ t: d.t, value: d.value }));
  makeChart("fngChart", "Fear & Greed", pts);
}
async function loadOrderbook(){
  // Placeholder depth plot (until real endpoint is added)
  const ctxId = "orderbookChart";
  const ptsBid = Array.from({length: 10}, (_,i)=>({ t:`L${i+1}`, value: 10+i*4 }));
  const ptsAsk = Array.from({length: 10}, (_,i)=>({ t:`L${i+1}`, value: 9+i*3.2 }));
  const el = document.getElementById(ctxId).getContext("2d");
  if(charts[ctxId]) charts[ctxId].destroy();
  charts[ctxId] = new Chart(el, {
    type: "line",
    data: {
      labels: ptsBid.map(p=>p.t),
      datasets: [
        { label: "Bids (cum)", data: ptsBid.map(p=>p.value), borderWidth: 2, pointRadius: 0 },
        { label: "Asks (cum)", data: ptsAsk.map(p=>p.value), borderWidth: 2, pointRadius: 0 }
      ]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ labels:{ boxWidth:10 } } } }
  });
}

// ---------- wiring ----------
window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("btnOhlcv").addEventListener("click", loadOhlcv);
  document.getElementById("btnFunding").addEventListener("click", loadFunding);
  document.getElementById("btnOI").addEventListener("click", loadOI);
  document.getElementById("btnOnchain").addEventListener("click", loadOnchain);
  document.getElementById("btnFNG").addEventListener("click", loadFNG);
  document.getElementById("btnOB").addEventListener("click", loadOrderbook);

  try{
    await loadMeta();          // fills dropdowns + table
    await loadOhlcv();         // draw initial charts
    await loadFNG();
  }catch(e){
    console.error(e);
  }
});
