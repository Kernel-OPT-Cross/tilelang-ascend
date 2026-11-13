import fs from "node:fs";
import path from "node:path";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell || row.length) {
    row.push(cell.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows.shift().map((h) => h.replace(/^\uFEFF/, ""));
  return rows.filter((r) => r.length === headers.length).map((r) =>
    Object.fromEntries(headers.map((h, i) => [h, r[i]])),
  );
}

const repo = path.resolve(import.meta.dirname, "..");
const sourceDir = path.join(repo, "benchmark", "results", "full_npu_perf_20261109");
const outputDir = path.join(sourceDir, "reconstructed");
const perFileDir = path.join(outputDir, "per_file");
fs.mkdirSync(perFileDir, { recursive: true });

const classifications = parseCsv(
  fs.readFileSync(path.join(sourceDir, "test_classification.csv"), "utf8"),
);
const performance = parseCsv(
  fs.readFileSync(path.join(sourceDir, "kernel_performance.csv"), "utf8"),
);
const perfByNode = new Map(performance.map((r) => [r.nodeid, r]));
const groups = new Map();

for (const item of classifications) {
  const originalLog = item.log || "unknown.log";
  const key = path.basename(originalLog, ".log");
  if (!groups.has(key)) groups.set(key, { originalLog, tests: [] });
  const perf = perfByNode.get(item.nodeid);
  groups.get(key).tests.push({
    nodeid: item.nodeid,
    category: item.category,
    call_duration_s: item.call_duration_s ? Number(item.call_duration_s) : null,
    failure_phase: item.failure_phase || null,
    failure_summary: item.failure_summary || null,
    performance: perf
      ? {
          calls: Number(perf.calls),
          median_ms: Number(perf.median_ms),
          min_ms: Number(perf.min_ms),
          max_ms: Number(perf.max_ms),
          samples_ms: JSON.parse(perf.samples_ms),
        }
      : null,
  });
}

const manifest = {
  recovery_status: "reconstructed_from_surviving_csv_and_report",
  is_original: false,
  warning:
    "The original server was released. These files reconstruct summaries only; omitted stdout/stderr and stack frames cannot be recovered.",
  source_files: ["test_classification.csv", "kernel_performance.csv"],
  environment: {
    npu: "Ascend 910B2C (single device)",
    driver: "26.0.rc1",
    os: "openEuler 24.03 user environment; Linux 5.15.0-113-generic; x86_64",
    cpu: "128 vCPU; 2 sockets x 32 cores x 2 threads (virtualized display)",
    cann: "9.0.0",
    python: "3.11.15",
    pytorch: "2.10.0+cpu",
    torch_npu: "2.10.0",
    tilelang_ascend: "0.1.4; base a6de467c54e57c3328aa49f9bc7c8d1be11261e7",
  },
  totals: {},
  files: [],
};

for (const item of classifications) {
  manifest.totals[item.category] = (manifest.totals[item.category] || 0) + 1;
}

for (const [key, group] of [...groups.entries()].sort()) {
  const counts = {};
  for (const test of group.tests) counts[test.category] = (counts[test.category] || 0) + 1;
  const jsonName = `${key}.reconstructed.json`;
  const logName = `${key}.reconstructed.log`;
  const payload = {
    recovery_status: "reconstructed_summary_not_original_output",
    original_log_path: group.originalLog,
    counts,
    tests: group.tests,
  };
  fs.writeFileSync(path.join(perFileDir, jsonName), `${JSON.stringify(payload, null, 2)}\n`);

  const lines = [
    "RECOVERED SUMMARY — NOT THE ORIGINAL PYTEST LOG",
    `Original path: ${group.originalLog}`,
    "The original stdout/stderr and complete stack traces were lost when the server was released.",
    "The following records were reconstructed from test_classification.csv and kernel_performance.csv.",
    "",
  ];
  for (const test of group.tests) {
    lines.push(`[${test.category}] ${test.nodeid}`);
    lines.push(`  call_duration_s: ${test.call_duration_s ?? "unavailable"}`);
    if (test.failure_phase) lines.push(`  failure_phase: ${test.failure_phase}`);
    if (test.failure_summary) lines.push(`  failure_summary: ${test.failure_summary}`);
    if (test.performance) {
      lines.push(
        `  npu_event_ms: calls=${test.performance.calls}, median=${test.performance.median_ms}, min=${test.performance.min_ms}, max=${test.performance.max_ms}`,
      );
      lines.push(`  samples_ms: ${JSON.stringify(test.performance.samples_ms)}`);
    }
    lines.push("");
  }
  fs.writeFileSync(path.join(perFileDir, logName), `${lines.join("\n")}\n`);
  manifest.files.push({
    original_log_path: group.originalLog,
    reconstructed_json: `per_file/${jsonName}`,
    reconstructed_log: `per_file/${logName}`,
    counts,
  });
}

fs.writeFileSync(
  path.join(outputDir, "reconstructed_manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);

console.log(
  JSON.stringify({ testCases: classifications.length, performanceRows: performance.length, files: groups.size }),
);
