/**
 * Run every Dawn-gated test file (one that reads WEBGPU_NODE_MODULE) serially,
 * one isolated `node --test` process per file.
 *
 * Many of these files take the repository-wide WebGPU lease themselves, so this
 * runner never holds it: it waits for the lease to be free before each file and
 * re-runs a file whose only failure was losing the lease race to another run.
 *
 *   npm run test:dawn                      # every Dawn file
 *   npm run test:dawn -- uniform svo       # files whose path contains a filter
 *   npm run test:dawn -- --list
 */
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { WEBGPU_EXCLUSIVE_LOCK } from "../lib/harness/webgpu-smoke-isolation";

const FILE_TIMEOUT_MS = Number(process.env.FLUID_DAWN_FILE_TIMEOUT_MS ?? 900_000);
const LOCK_CONFLICT = "Refusing concurrent GPU execution";

async function discover(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const groups = await Promise.all(entries.map(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return discover(path);
    return Promise.resolve(entry.isFile() && /\.test\.tsx?$/.test(entry.name) ? [path] : []);
  }));
  return groups.flat();
}

async function lockHeld(): Promise<boolean> {
  try { await stat(WEBGPU_EXCLUSIVE_LOCK); return true; } catch { return false; }
}

async function waitForLease(): Promise<void> {
  let announced = false;
  while (await lockHeld()) {
    if (!announced) {
      let owner = "unknown owner";
      try { owner = await readFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* diagnostic only */ }
      console.log(`waiting for WebGPU lease (${owner})`);
      announced = true;
    }
    // Notice the gap between consecutive profiling jobs before another run
    // takes the lease. Acquisition in the fixture still arbitrates races.
    await new Promise(resolve => setTimeout(resolve, 250));
  }
}

function runFile(file: string): Promise<{ code: number; output: string; ms: number }> {
  const started = performance.now();
  return new Promise(resolve => {
    const child = spawn(process.execPath, ["--import", "tsx", "--test", "--test-concurrency=1", file], {
      env: { ...process.env, WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? join(process.cwd(), "node_modules/webgpu/index.js") },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.on("data", chunk => { output += chunk; });
    child.stderr.on("data", chunk => { output += chunk; });
    const timer = setTimeout(() => { output += `\nTIMEOUT after ${FILE_TIMEOUT_MS} ms\n`; child.kill("SIGTERM"); }, FILE_TIMEOUT_MS);
    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      resolve({ code: code ?? (signal ? 1 : 0), output, ms: performance.now() - started });
    });
  });
}

const args = process.argv.slice(2);
const filters = args.filter(arg => !arg.startsWith("--"));
const all = (await Promise.all(["tests", "lib", "advance-lab"].map(discover))).flat().sort();
const files: string[] = [];
for (const file of all) {
  if (filters.length && !filters.some(filter => file.includes(filter))) continue;
  if ((await readFile(file, "utf8")).includes("WEBGPU_NODE_MODULE")) files.push(file);
}
if (args.includes("--list")) {
  process.stdout.write(files.join("\n") + "\n");
  process.exit(0);
}

const failed: string[] = [];
for (const file of files) {
  let result;
  for (let attempt = 0; ; attempt++) {
    await waitForLease();
    result = await runFile(file);
    if (result.code === 0 || !result.output.includes(LOCK_CONFLICT) || attempt >= 20) break;
  }
  const count = (name: string) => Number(result.output.match(new RegExp(`^# ${name} (\\d+)`, "m"))?.[1] ?? 0);
  const status = result.code !== 0 ? "FAIL" : count("pass") === 0 ? "SKIP" : "pass";
  console.log(`${status} ${(result.ms / 1000).toFixed(1).padStart(7)} s  ${file}  (${count("pass")} pass, ${count("skipped")} skipped)`);
  if (result.code !== 0) {
    failed.push(file);
    console.log(result.output.split("\n").map(line => `    ${line}`).join("\n"));
  }
}
console.log(`\n${files.length - failed.length}/${files.length} Dawn files passed`);
if (failed.length) {
  console.log(`failed:\n${failed.map(file => `  ${file}`).join("\n")}`);
  process.exitCode = 1;
}
