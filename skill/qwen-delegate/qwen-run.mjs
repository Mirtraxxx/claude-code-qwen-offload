#!/usr/bin/env node
// qwen-run.mjs — run one coding job on a local model through the omp CLI.
// Node 24, ES module, no npm dependencies.

import { spawn, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, appendFileSync, mkdirSync, rmSync, mkdtempSync, existsSync, statSync, readdirSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

const HOUSE_RULES = `House rules for this job:
- Your context is limited and older tool output can be dropped when it fills up. Work incrementally: finish and save one piece of work before reading much more.
- Read at most 3 files per turn. Prefer reading the part of a file you need over the whole file when the file is long.
- Use \`write\` only to create a file that does not exist yet. To change an existing file, including one you created earlier in this job, use \`edit\`. Never rewrite a whole existing file.
- Only create or change the files the task names. Do not touch anything else.
- Never delete existing code the task does not ask you to change. That includes imports, fields, fallback returns and lines in HTML templates such as <style> or <script>. If you think something must go, leave it and say so in your summary.
- When you replace a function or block with a new version, remove the old version, so only one copy exists.
- When you are done, reply with a short summary of what you changed, and list anything you deleted.
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

const SKIP_DIRS = new Set(["node_modules", ".git", "qwen", "__pycache__", "venv", "site-packages"]);
// Qwen only writes text, so images, audio, 3D models, archives and weights are never read or compared
// (a project with screenshots and art folders blew the old 5000-file cap and gave false flags)
const SKIP_EXT = new Set(("png jpg jpeg gif webp bmp tga tif tiff psd ico exr hdr dds ktx ktx2 basis " +
  "wav mp3 ogg flac m4a aac opus mp4 webm mov avi mkv " +
  "glb gltf bin fbx obj mtl dae blend vox ply stl usdz " +
  "tflite pb binarypb onnx hyb " +
  "zip 7z rar gz tgz bz2 xz tar " +
  "ttf otf woff woff2 pdf exe dll so dylib pyd pyc " +
  "safetensors gguf pt pth ckpt onnx npy npz pkl h5 db sqlite sqlite3 ldb log").split(" "));
const MAX_FILES = 20000;
// files up to this size keep their old contents, so every change diffs against the real "before"
const KEEP_BYTES = 512 * 1024;

// a browser profile (e.g. from a perf test) holds thousands of cache files that change on their own
const isBrowserProfile = (dir) => existsSync(path.join(dir, "Local State")) || existsSync(path.join(dir, "prefs.js"));

// returns Map(rel -> info); map.capped is true when MAX_FILES was hit. The job's --files are
// always included, even past the cap, so their diff and line checks still work.
function snapshot(cwd, keep = false, mustHave = []) {
  const map = new Map();
  map.capped = false;
  const add = (full, rel) => {
    const st = statSync(full);
    const buf = readFileSync(full);
    map.set(rel, {
      size: st.size,
      mtimeMs: st.mtimeMs,
      sha1: crypto.createHash("sha1").update(buf).digest("hex"),
      buf: keep && st.size <= KEEP_BYTES ? buf : null,
    });
  };
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (map.size >= MAX_FILES) {
        if (!map.capped) console.error(`warning: snapshot stopped at ${MAX_FILES} files; only the --files are compared`);
        map.capped = true;
        return;
      }
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".") || isBrowserProfile(full)) continue;
        walk(full);
      } else if (entry.isFile()) {
        if (SKIP_EXT.has(path.extname(entry.name).slice(1).toLowerCase())) continue;
        add(full, path.relative(cwd, full).split(path.sep).join("/"));
      }
    }
  };
  walk(cwd);
  for (const rel of mustHave) {
    const full = path.join(cwd, rel);
    if (!map.has(rel) && existsSync(full) && statSync(full).isFile()) add(full, rel);
  }
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

// snapshot (keeping old contents in memory) + temp folder for the diff's old side
const tmp = mkdtempSync(path.join(os.tmpdir(), "qwen-run-"));
const snapDir = path.join(tmp, "snap");
const before = snapshot(cwd, true, files);
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
const after = snapshot(cwd, false, files);
// past the cap the two file lists can differ for no reason, so only the job's own files are compared
const capped = before.capped || after.capped;
const compared = (rel) => !capped || files.includes(rel);
const changed = [], added = [], removed = [];
for (const [rel, info] of after) {
  if (!compared(rel)) continue;
  const prev = before.get(rel);
  if (!prev) added.push(rel);
  else if (prev.sha1 !== info.sha1) changed.push(rel);
}
for (const rel of before.keys()) if (compared(rel) && !after.has(rel)) removed.push(rel);
const outOfScope = files.length > 0
  ? [...changed, ...added, ...removed].filter((rel) => !files.includes(rel))
  : [];

// step 6: diff
const runsDir = path.join(os.tmpdir(), "qwen-runs");
mkdirSync(runsDir, { recursive: true });
const diffName = `${stamp()}-${slug(opts.task)}.diff`;
const diffPath = path.join(runsDir, diffName);
const outPath = path.join(runsDir, diffName.replace(/\.diff$/, ".out.txt"));

// only the job's own files count toward the ledger; anything else changed in the folder
// meanwhile (a parallel job, Claude's own edits) is listed but not credited to this job
const inScope = (rel) => files.length === 0 || files.includes(rel);

// lines whose deletion has broken things before: imports, definitions, template tags
const KEY_LINE = /^\s*(import\s|from\s+\S+\s+import\s|(export\s+)?(default\s+)?(async\s+)?(def|class|function)\s+\w|(export\s+)?(const|let|var)\s+\w+\s*=\s*(require\(|await import\()|<(style|script|link)\b)/;
// definitions that should exist once per file: top-level Python defs/classes, JS functions anywhere
function defNames(text, rel) {
  const re = /\.py$/.test(rel)
    ? /^(?:async\s+)?(?:def|class)\s+(\w+)/gm
    : /\.(m?js|cjs|jsx?|tsx?|html?)$/.test(rel)
      ? /^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\(/gm
      : null;
  const counts = new Map();
  if (!re) return counts;
  for (const m of text.matchAll(re)) counts.set(m[1], (counts.get(m[1]) ?? 0) + 1);
  return counts;
}

let diffText = "";
const addedChars = new Map();
const removedKey = [];   // "file: line" for key lines removed and not re-added anywhere in the file
const duplicates = [];   // "file: name x2" for definitions that became duplicated
for (const rel of [...changed, ...added, ...removed]) {
  const prev = before.get(rel);
  let oldSide = emptyFile;
  if (prev?.buf) {
    oldSide = path.join(snapDir, rel);
    mkdirSync(path.dirname(oldSide), { recursive: true });
    writeFileSync(oldSide, prev.buf);
  }
  const newSide = existsSync(path.join(cwd, rel)) ? path.join(cwd, rel) : emptyFile;
  const r = spawnSync("git", ["diff", "--no-index", "--no-color", "--", oldSide, newSide], { encoding: "utf8" });
  const text = r.stdout ?? "";
  diffText += text;
  let n = 0;
  const plus = new Set(), minus = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("+") && !line.startsWith("+++")) { n += line.length; plus.add(line.slice(1).trim()); }
    else if (line.startsWith("-") && !line.startsWith("---")) minus.push(line.slice(1));
  }
  addedChars.set(rel, n);
  if (!inScope(rel) || !prev?.buf) continue;
  for (const line of minus) {
    if (KEY_LINE.test(line) && !plus.has(line.trim())) removedKey.push(`${rel}: ${line.trim().slice(0, 100)}`);
  }
  if (existsSync(newSide) && newSide !== emptyFile) {
    const oldDefs = defNames(prev.buf.toString("utf8"), rel);
    for (const [name, count] of defNames(readFileSync(newSide, "utf8"), rel)) {
      if (count > 1 && count > (oldDefs.get(name) ?? 0)) duplicates.push(`${rel}: ${name} x${count}`);
    }
  }
}
// brand-new files have no old side; still check them for duplicated definitions
for (const rel of added.filter(inScope)) {
  for (const [name, count] of defNames(readFileSync(path.join(cwd, rel), "utf8"), rel)) {
    if (count > 1) duplicates.push(`${rel}: ${name} x${count}`);
  }
}
writeFileSync(diffPath, diffText);
writeFileSync(outPath, stdout);
let diffChars = 0, otherChars = 0;
for (const [rel, n] of addedChars) {
  if (inScope(rel)) diffChars += n; else otherChars += n;
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
if (removedKey.length > 0) flags.push(`REMOVED-CODE:${removedKey.length}`);
if (capped) flags.push(`SNAPSHOT-CAPPED:${MAX_FILES}`);
if (duplicates.length > 0) flags.push(`DUPLICATE-DEF:${duplicates.map((d) => d.replace(/ x\d+$/, "")).slice(0, 3).join(";")}`);

// step 8: ledger
let ledgerLine;
if (opts.noLedger) {
  ledgerLine = "disabled";
} else if (!existsSync(ledgerPath)) {
  ledgerLine = `skipped, no ledger file at ${ledgerPath}`;
} else {
  const notes = `${model.split("/").pop()} ${opts.thinking} ${fmtDur(elapsedSec)} ${fmtNum(changed.length + added.length + removed.length)} files` +
    (files.length === 0 ? " NO-FILES-LIST (diff_chars counts every changed file)" : "") +
    (otherChars > 0 ? ` (+${otherChars} chars in other files not counted)` : "") +
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
console.log(`  diff      ${fmtNum(diffChars)} added chars in the job's files` +
  (otherChars > 0 ? ` (+${fmtNum(otherChars)} in other files, not counted)` : "") + ` -> ${diffPath}`);
console.log(`  ledger    ${ledgerLine}`);
console.log(`  FLAGS     ${flags.length === 0 ? "none" : flags.join("; ")}`);
if (removedKey.length > 0) {
  console.log("--- removed imports/definitions (check each one was meant to go) ---");
  for (const r of removedKey) console.log(`  ${r}`);
}
if (duplicates.length > 0) {
  console.log("--- definitions that now appear more than once (old copy left behind?) ---");
  for (const d of duplicates) console.log(`  ${d}`);
}
console.log("--- answer ---");
console.log(stdout);

// step 10: cleanup + exit code
rmSync(tmp, { recursive: true, force: true });
process.exit(flags.length > 0 ? 1 : 0);
