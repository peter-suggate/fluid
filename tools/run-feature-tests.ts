/**
 * Discover colocated feature tests as well as cross-feature integration tests.
 *
 * `advance-lab/` includes the independent 2-D lab's drawing, controls,
 * declarations and playback checks.
 */
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { spawn } from "node:child_process";

async function discover(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const groups = await Promise.all(entries.map(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return discover(path);
    return Promise.resolve(entry.isFile() && /\.test\.tsx?$/.test(entry.name) ? [path] : []);
  }));
  return groups.flat();
}
const files = (await Promise.all(["tests", "lib", "advance-lab"].map(discover))).flat().sort();
if (process.argv.includes("--list")) {
  process.stdout.write(files.join("\n") + "\n");
} else {
  // GPU runs take one test file at a time: node --test defaults to cores - 1 files at once (9 here), a GPU file alone
  // peaks near 7.8 GB, and 9 of them on a 32 GB Mac hung the GPU. The plain (CPU) gate stays parallel.
  const serial = process.env.WEBGPU_NODE_MODULE && !process.argv.some(a => a.startsWith("--test-concurrency")) ? ["--test-concurrency=1"] : [];
  const child = spawn(process.execPath, ["--import", "tsx", "--test", ...serial, ...process.argv.slice(2), ...files], {
    stdio: "inherit", env: process.env,
  });
  child.once("error", error => { console.error(error); process.exitCode = 1; });
  child.once("exit", (code, signal) => { process.exitCode = code ?? (signal ? 1 : 0); });
}
