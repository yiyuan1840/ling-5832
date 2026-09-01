// Validate stats.js against the Python analyze.py reference output, using the
// real vocab_results.json as the test vector.
const fs = require("fs");
const path = require("path");
const { analyzeAll } = require("./stats.js");

const rows = JSON.parse(fs.readFileSync(
  path.join(__dirname, "..", "experiment", "vocab_results.json"), "utf-8")).results;
const a = analyzeAll(rows);

// Reference values from experiment/analyze.py on the same data.
const expected = {
  selfReport: { total: 65604, lo: 63924, hi: 67284, loW: 61587, hiW: 67426 },
  lenient:    { total: 63140, lo: 61032, hi: 65248, loW: 58316, hiW: 65991 },
  strict:     { total: 60844, lo: 58441, hi: 63247, loW: 55599, hiW: 64299 },
  productive: { total: 59581, lo: 57085, hi: 62078, loW: 54270, hiW: 63215 },
};

let fail = 0;
for (const [key, exp] of Object.entries(expected)) {
  for (const [f, v] of Object.entries(exp)) {
    const got = Math.round(a.defs[key][f]);
    if (Math.abs(got - v) > 1) {
      console.log(`MISMATCH ${key}.${f}: got ${got}, expected ${v}`);
      fail++;
    }
  }
}
const checks = [
  [a.claimed, 336, "claimed"], [a.wrong, 13, "wrong"], [a.partial, 12, "partial"],
  [a.ratingCounts.join(","), "15,9,23,313", "ratingCounts"],
  [Math.round(a.rtMedians[3]), 674, "rt median 掌握"],
];
for (const [got, exp, name] of checks) {
  if (String(got) !== String(exp)) { console.log(`MISMATCH ${name}: got ${got}, expected ${exp}`); fail++; }
}
console.log(fail === 0 ? "ALL CHECKS PASS" : `${fail} FAILURES`);
process.exit(fail === 0 ? 0 : 1);
