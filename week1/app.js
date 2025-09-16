// app.js
/*
  Titanic Interactive EDA (Browser-only, GitHub Pages-ready)

  - Uses PapaParse to load train.csv and test.csv from file inputs (client-side).
  - Merges with a source column (train/test).
  - Overview: shape + preview.
  - Missing values: percent per column (Chart.js bar).
  - Stats: numeric (mean, median, std), categorical counts; grouped by Survived where available (train).
  - Visualizations: bar (Sex, Pclass, Embarked), histograms (Age, Fare), correlation heatmap (train numeric) using Chart.js matrix.
  - Export: merged CSV and JSON summary.

  Reuse note: To adapt for other datasets with train/test splits, edit SCHEMA + FEATURE lists and any chart sections below.
*/

// ----- Configurable schema (swap here for other datasets) -----
const IDENTIFIER = "PassengerId"; // excluded from modeling
const TARGET = "Survived";        // present only in train
const FEATURES_NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]; // age/fare continuous; others treated numeric
const FEATURES_CATEGORICAL = ["Sex", "Embarked"];
const CORE_COLUMNS = [IDENTIFIER, TARGET].concat(FEATURES_NUMERIC, FEATURES_CATEGORICAL);

// ----- State -----
let trainRows = [];
let testRows = [];
let mergedRows = [];
let lastSummary = null;

// Chart instances to clean up on re-run
let charts = {
  missing: null,
  sexBar: null,
  pclassBar: null,
  embarkedBar: null,
  ageHist: null,
  fareHist: null,
  corrHeatmap: null
};

// ----- DOM helpers -----
function $(id) { return document.getElementById(id);
