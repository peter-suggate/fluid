/** Discover colocated feature tests as well as cross-feature integration tests. */
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
const files = (await Promise.all(["tests", "lib"].map(discover))).flat().sort();
if (process.argv.includes("--list")) {
  process.stdout.write(files.join("\n") + "\n");
} else {
  const child = spawn(process.execPath, ["--import", "tsx", "--test", ...process.argv.slice(2), ...files], {
    stdio: "inherit", env: process.env,
  });
  child.once("error", error => { console.error(error); process.exitCode = 1; });
  child.once("exit", (code, signal) => { process.exitCode = code ?? (signal ? 1 : 0); });
}
