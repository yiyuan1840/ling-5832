// Stratified vocabulary-size estimation, ported from experiment/analyze.py.
// Works in the browser (globals) and in Node (module.exports) for testing.
(function (root) {
  "use strict";

  const STRATA_N = { 1: 11424, 2: 45086, 3: 6690, 4: 6497, 5: 717 };
  const POPULATION = 70414;
  const Z = 1.959963985;

  function wilson(k, n) {
    if (n === 0) return [0, 0];
    const p = k / n;
    const d = 1 + (Z * Z) / n;
    const c = p + (Z * Z) / (2 * n);
    const h = Z * Math.sqrt((p * (1 - p)) / n + (Z * Z) / (4 * n * n));
    return [(c - h) / d, (c + h) / d];
  }

  // rows: [{len, rating, verify, ms1}]; pred: row -> bool ("knows this entry")
  function estimate(rows, pred) {
    let total = 0, variance = 0, loW = 0, hiW = 0;
    const per = {};
    for (const h of [1, 2, 3, 4, 5]) {
      const sub = rows.filter(r => Math.min(r.len, 5) === h);
      const n = sub.length;
      const k = sub.filter(pred).length;
      const p = k / n;
      const N = STRATA_N[h];
      const fpc = 1 - n / N;
      total += N * p;
      variance += N * N * fpc * (p * (1 - p)) / (n - 1);
      const [wl, wh] = wilson(k, n);
      loW += N * wl;
      hiW += N * wh;
      per[h] = { N: N, n: n, k: k, p: p, est: N * p };
    }
    const se = Math.sqrt(variance);
    return { total, lo: total - Z * se, hi: total + Z * se, loW, hiW, per };
  }

  function median(xs) {
    if (!xs.length) return null;
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  const DEFS = {
    selfReport: { label: "仅自评（自称“认识”及以上）", pred: r => r.rating >= 2 },
    lenient:    { label: "理解（宽）：核对为 对/部分对", pred: r => r.rating >= 2 && (r.verify === 1 || r.verify === 2) },
    strict:     { label: "理解（严）：核对为 对", pred: r => r.rating >= 2 && r.verify === 2 },
    productive: { label: "会用：自评“掌握”且核对为 对", pred: r => r.rating === 3 && r.verify === 2 },
  };

  function analyzeAll(rows) {
    const ratingCounts = [0, 0, 0, 0];
    let claimed = 0, wrong = 0, partial = 0;
    for (const r of rows) {
      ratingCounts[r.rating] += 1;
      if (r.rating >= 2) {
        claimed += 1;
        if (r.verify === 0) wrong += 1;
        if (r.verify === 1) partial += 1;
      }
    }
    const defs = {};
    for (const key of Object.keys(DEFS)) {
      defs[key] = Object.assign({ label: DEFS[key].label }, estimate(rows, DEFS[key].pred));
    }
    const rtMedians = [0, 1, 2, 3].map(i =>
      median(rows.filter(r => r.rating === i && r.ms1 > 0).map(r => r.ms1)));
    return { population: POPULATION, ratingCounts, claimed, wrong, partial, defs, rtMedians };
  }

  const api = { STRATA_N, POPULATION, wilson, estimate, median, analyzeAll, DEFS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VocabStats = api;
})(typeof self !== "undefined" ? self : this);
