// app.js
// Titanic EDA — Browser-only (GitHub Pages Ready)
// Uses PapaParse for CSV parsing and Chart.js (plus matrix plugin) for charts.
// Reuse: To adapt for a different train/test split dataset, change the SCHEMA section below
// and adjust any charts/stats you want to visualize.

// ---------- SCHEMA (swap here for other datasets) ----------
const IDENTIFIER = "PassengerId"; // excluded from modeling
const TARGET = "Survived";        // present only in train
const FEATURES_NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"];
const FEATURES_CATEGORICAL = ["Sex", "Embarked"];
const CORE_COLUMNS = [IDENTIFIER, TARGET].concat(FEATURES_NUMERIC, FEATURES_CATEGORICAL);

// ---------- STATE ----------
let trainRows = [];
let testRows = [];
let mergedRows = [];
let lastSummary = null;

let charts = {
  missing: null,
  sexBar: null,
  pclassBar: null,
  embarkedBar: null,
  ageHist: null,
  fareHist: null,
  corrHeatmap: null
};

// ---------- DOM HELPERS ----------
const $ = (id) => document.getElementById(id);

function setStatus(msg, ok = false) {
  const el = $("statusBadge");
  el.textContent = msg;
  el.style.borderColor = ok ? "#22c55e" : "#1f2937";
  el.style.color = ok ? "#bbf7d0" : "#cbd5e1";
}

// ---------- UTILITIES ----------
function safeNumber(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}
function mean(arr) {
  const v = arr.filter((x) => x !== null && x !== undefined);
  if (!v.length) return null;
  return v.reduce((a, b) => a + b, 0) / v.length;
}
function median(arr) {
  const v = arr.filter((x) => x !== null && x !== undefined).slice().sort((a, b) => a - b);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
function std(arr) {
  const v = arr.filter((x) => x !== null && x !== undefined);
  if (!v.length) return null;
  const m = mean(v);
  const variance = mean(v.map((x) => Math.pow(x - m, 2)));
  return variance === null ? null : Math.sqrt(variance);
}
function unique(arr) {
  return Array.from(new Set(arr));
}
function toCSV(rows) {
  try {
    return Papa.unparse(rows);
  } catch (e) {
    alert("Failed to convert to CSV: " + e.message);
    return null;
  }
}
function download(filename, text) {
  if (!text) return;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function clearCharts() {
  Object.keys(charts).forEach((k) => {
    if (charts[k]) { charts[k].destroy();
}
