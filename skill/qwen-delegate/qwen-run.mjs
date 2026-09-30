#!/usr/bin/env node
// qwen-run.mjs — run one coding job on a local model through the omp CLI.
// Node 24, ES module, no npm dependencies.

import { spawn, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, appendFileSync, mkdirSync, rmSync, mkdtempSync, existsSync, statSync, readdirSync, copyFileSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

const HOUSE_RULES = `House rules for this job:
- Your context is limited and older tool output can be dropped when it fills up. Work incrementally: finish and save one piece of work before reading much more.
- Read at most 3 files per turn. Prefer reading the part of a file you need over the whole file when the file is long.
- Use \`write\` only to create a file that does not exist yet. To change an existing file, including one you created earlier in this job, use \`edit\`. Never rewrite a whole existing file.
- Only create or change the files the task names. Do not touch anything else.
- When you are done, reply with a short summary of what you changed.
`;

// ---------- arg parsing ----------

function usage() {
  console.error(`usage: node qwen-run.mjs --cwd <dir> --brief <file> --task <label> [options]
  --cwd <dir>        project folder the model works in (required)
  --brief <file>     file holding the full task prompt (required)
  --task <label>     short label for the ledger and run files (required)
  --files <list>     comma-separated paths (relative to --cwd) the job may touch
  --tools <list>     tools passed to omp (default read,grep,glob,edit,write)
  --thinking <lvl>   low|medium|high (default low)
  --max-time <dur>   Ns|Nm|Nh (default 30m)
  --model <id>       omp model id (default: auto-detect)
  --ledger <file>    CSV to append to (default <cwd>/qwen/ledger.csv)
  --no-ledger        never append to the ledger
  --reruns <n>       reruns column value (default 0)
  --dry-run          print the omp command line and snapshot count, then exit`);
}

function parseArgs(argv) {
  const opts = {
    cwd: null, brief: null, task: null, files: null,
    tools: "read,grep,glob,edit,write", thinking: "low", maxTime: "30m",
    model: null, ledger: null, noLedger: false, reruns: "0", dryRun: false,
  };
  const boolean = new Set(["--no-ledger", "--dry-run"]);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) { console.error(`unknown argument: ${a}`); usage(); process.exit(2); }
    if (boolean.has(a)) {
      opts[a === "--no-ledger" ? "noLedger" : "dryRun"] = true;
      continue;
    }
    const name = a.slice(2);
    const value = argv[++i];
    if (value === undefined || value.startsWith("--")) { console.error(`missing value for ${a}`); usage(); process.exit(2); }
    switch (name) {
      case "cwd": opts.cwd = value; break;
      case "brief": opts.brief = value; break;
      case "task": opts.task = value; break;
      case "files": opts.files = value; break;
      case "tools": opts.tools = value; break;
      case "thinking": opts.thinking = value; break;
      case "max-time": opts.maxTime = value; break;
      case "model": opts.model = value; break;
      case "ledger": opts.ledger = value; break;
      case "reruns": opts.reruns = value; break;
      default: console.error(`unknown option: --${name}`); usage(); process.exit(2);
    }
  }
  for (const req of ["cwd", "brief", "task"]) {
    if (!opts[req]) { console.error(`missing required option: --${req}`); usage(); process.exit(2); }
  }
  return opts;
}

// ---------- small helpers ----------

function parseDuration(s) {
  const m = /^(\d+)([smh])$/.exec(s);
  if (!m) { console.error(`bad --max-time: ${s} (use Ns, Nm, or Nh)`); process.exit(2); }
  const n = Number(m[1]);
  return m[2] === "s" ? n : m[2] === "m" ? n * 60 : n * 3600;
}

function fmtDur(totalSec) {
  const sec = Math.max(0, Math.round(totalSec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h > 0) return `${h}h${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

const fmtNum = (n) => n.toLocaleString("en-US");

function slug(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function csvField(v) {
  const s = String(v);
  return /[",]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function stamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

// ---------- step 1: model detection ----------

async function detectModel() {
  let res;
  try {
    res = await fetch("http://127.0.0.1:8080/v1/models", { signal: AbortSignal.timeout(5000) });
  } catch {
    console.error("no model server on :8080");
    process.exit(3);
  }
  let body;
  try { body = await res.json(); } catch { body = null; }
  const data = Array.isArray(body?.data) ? body.data : [];
  if (data.length === 0) { console.error("no model server on :8080"); process.exit(3); }
  return data.some((e) => e.owned_by === "strata" || /iq2|flash/i.test(String(e.id))) ? "llamacpp/qwen3.8-flash-next" : "llamacpp/qwen3.8-27b";
}

// ---------- step 2: snapshot ----------

const SKIP_DIRS = new Set(["node_modules", ".git", "qwen"]);

function snapshot(cwd) {
  const map = new Map();
  let count = 0, warned = false;
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (count >= 5000) {
        if (!warned) { console.error("warning: snapshot stopped at 5000 files"); warned = true; }
        return;
      }
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        walk(full);
      } else if (entry.isFile()) {
        const st = statSync(full);
        const rel = path.relative(cwd, full).split(path.sep).join("/");
        map.set(rel, {
          size: st.size,
          mtimeMs: st.mtimeMs,
          sha1: crypto.createHash("sha1").update(readFileSync(full)).digest("hex"),
        });
        count++;
      }
    }
  };
  walk(cwd);
  return map;
}

// ---------- step 4: run omp ----------

function runOmp(args, cwd) {
  return new Promise((resolve, reject) => {
    // stdin must be closed: with an open pipe, omp -p waits for piped input forever
    const child = spawn("omp", args, { cwd, shell: false, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    const started = performance.now();
    const timer = setInterval(() => {
      const elapsed = (performance.now() - started) / 1000;
      console.error(`  ... running ${fmtDur(elapsed)}, stdout ${fmtNum(stdout.length)} chars`);
    }, 30000);
    child.stdout.on("data", (c) => { stdout += c; });
    child.stderr.on("data", (c) => { stderr += c; });
    child.on("error", (err) => {
      clearInterval(timer);
      reject(err);
    });
    child.on("close", (code) => {
      clearInterval(timer);
      resolve({ code: code ?? -1, stdout, stderr, elapsedSec: (performance.now() - started) / 1000 });
    });
  });
}

// ---------- main ----------

const opts = parseArgs(process.argv.slice(2));
const cwd = path.resolve(opts.cwd);
const briefPath = path.resolve(opts.brief);
const maxTimeSec = parseDuration(opts.maxTime);
const files = opts.files ? opts.files.split(",").map((s) => s.trim()).filter(Boolean) : [];
const ledgerPath = path.resolve(opts.ledger ?? path.join(cwd, "qwen", "ledger.csv"));

if (!existsSync(cwd)) { console.error(`cwd not found: ${cwd}`); process.exit(2); }
if (!existsSync(briefPath)) { console.error(`brief not found: ${briefPath}`); process.exit(2); }

const model = opts.model ?? await detectModel();
const briefText = readFileSync(briefPath, "utf8");

// snapshot + temp folder with copies of the expected files
const tmp = mkdtempSync(path.join(os.tmpdir(), "qwen-run-"));
const snapDir = path.join(tmp, "snap");
const before = snapshot(cwd);
for (const rel of files) {
  const src = path.join(cwd, rel);
  if (!existsSync(src)) continue;
  const dest = path.join(snapDir, rel);
  mkdirSync(path.dirname(dest), { recursive: true });
  copyFileSync(src, dest);
}
const emptyFile = path.join(tmp, ".empty");
writeFileSync(emptyFile, "");

// step 3: prompt file
let prompt = HOUSE_RULES;
if (files.length > 0) prompt += `- Files you may create or change: ${files.join(", ")}.\n`;
prompt += "\n" + briefText;
const promptPath = path.join(tmp, "prompt.md");
writeFileSync(promptPath, prompt);

// step 4: omp command line
const ompArgs = [
  "-p", "--no-session",
  "--model", model,
  "--thinking", opts.thinking,
  "--tools", opts.tools,
  "--cwd", cwd,
  "--max-time", opts.maxTime,
  `@${promptPath}`,
];

if (opts.dryRun) {
  console.log(`omp ${ompArgs.join(" ")}`);
  console.log(`snapshot: ${fmtNum(before.size)} files`);
  rmSync(tmp, { recursive: true, force: true });
  process.exit(0);
}

let run;
try {
  run = await runOmp(ompArgs, cwd);
} catch (err) {
  console.error(`failed to start omp: ${err.message}`);
  rmSync(tmp, { recursive: true, force: true });
  process.exit(4);
}
const { code, stdout, stderr, elapsedSec } = run;
// omp's stderr is mostly its "Working..." spinner; only show it when something went wrong
if (code !== 0 && stderr.trim()) console.error(stderr.trim());

// step 5: second snapshot + compare
const after = snapshot(cwd);
const changed = [], added = [], removed = [];
for (const [rel, info] of after) {
  const prev = before.get(rel);
  if (!prev) added.push(rel);
  else if (prev.sha1 !== info.sha1) changed.push(rel);
}
for (const rel of before.keys()) if (!after.has(rel)) removed.push(rel);
const outOfScope = files.length > 0
  ? [...changed, ...added, ...removed].filter((rel) => !files.includes(rel))
  : [];

// step 6: diff
const runsDir = path.join(os.tmpdir(), "qwen-runs");
mkdirSync(runsDir, { recursive: true });
const diffName = `${stamp()}-${slug(opts.task)}.diff`;
const diffPath = path.join(runsDir, diffName);
const outPath = path.join(runsDir, diffName.replace(/\.diff$/, ".out.txt"));

let diffText = "";
const addedChars = new Map();
for (const rel of [...changed, ...added, ...removed]) {
  const oldSide = existsSync(path.join(snapDir, rel)) ? path.join(snapDir, rel) : emptyFile;
  const newSide = existsSync(path.join(cwd, rel)) ? path.join(cwd, rel) : emptyFile;
  const r = spawnSync("git", ["diff", "--no-index", "--no-color", "--", oldSide, newSide], { encoding: "utf8" });
  const text = r.stdout ?? "";
  diffText += text;
  let n = 0;
  for (const line of text.split("\n")) {
    if (line.startsWith("+") && !line.startsWith("+++")) n += line.length;
  }
  addedChars.set(rel, n);
}
writeFileSync(diffPath, diffText);
writeFileSync(outPath, stdout);
let diffChars = 0;
for (const line of diffText.split("\n")) {
  if (line.startsWith("+") && !line.startsWith("+++")) diffChars += line.length;
}

// step 7: flags
const flags = [];
if (stdout.trim() === "") flags.push("EMPTY-ANSWER");
if (elapsedSec >= maxTimeSec - 5) flags.push("TIMEOUT");
if (code !== 0) flags.push(`EXIT-${code}`);
const toolList = opts.tools.split(",").map((s) => s.trim());
if ((toolList.includes("edit") || toolList.includes("write")) && changed.length + added.length + removed.length === 0) {
  flags.push("NO-CHANGES");
}
if (outOfScope.length > 0) {
  const shown = outOfScope.slice(0, 5).join(";");
  flags.push(`OUT-OF-SCOPE:${shown}${outOfScope.length > 5 ? ";..." : ""}`);
}
const missing = files.filter((rel) => !changed.includes(rel) && !added.includes(rel));
if (missing.length > 0) flags.push(`MISSING:${missing.slice(0, 5).join(";")}${missing.length > 5 ? ";..." : ""}`);

// step 8: ledger
let ledgerLine;
if (opts.noLedger) {
  ledgerLine = "disabled";
} else if (!existsSync(ledgerPath)) {
  ledgerLine = `skipped, no ledger file at ${ledgerPath}`;
} else {
  const notes = `${model.split("/").pop()} ${opts.thinking} ${fmtDur(elapsedSec)} ${fmtNum(changed.length + added.length + removed.length)} files` +
    (flags.length > 0 ? ` FLAGS ${flags.join(";")}` : "");
  const row = [
    stamp().slice(0, 8).replace(/(\d{4})(\d{2})(\d{2})/, "$1-$2-$3"),
    opts.task,
    String(briefText.length),
    String(diffChars),
    opts.reruns,
    notes,
  ].map(csvField).join(",");
  const prev = readFileSync(ledgerPath, "utf8");
  appendFileSync(ledgerPath, (prev && !prev.endsWith("\n") ? "\n" : "") + row + "\n");
  ledgerLine = `appended to ${ledgerPath}`;
}

// step 9: summary
const listLine = (arr) => arr.length === 0 ? "-" : arr.map((rel) => `${rel} (+${fmtNum(addedChars.get(rel) ?? 0)})`).join(", ");
console.log(`qwen-run: ${opts.task}`);
console.log(`  model     ${model}, thinking ${opts.thinking}`);
console.log(`  time      ${fmtDur(elapsedSec)}, exit ${code}`);
console.log(`  answer    ${fmtNum(stdout.length)} chars`);
console.log(`  changed   ${listLine(changed)}`);
console.log(`  added     ${listLine(added)}`);
console.log(`  removed   ${listLine(removed)}`);
console.log(`  diff      ${fmtNum(diffChars)} added chars -> ${diffPath}`);
console.log(`  ledger    ${ledgerLine}`);
console.log(`  FLAGS     ${flags.length === 0 ? "none" : flags.join("; ")}`);
console.log("--- answer ---");
console.log(stdout);

// step 10: cleanup + exit code
rmSync(tmp, { recursive: true, force: true });
process.exit(flags.length > 0 ? 1 : 0);
