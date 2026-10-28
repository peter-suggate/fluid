/** Rebuild all three actual-fluid Cycles scenes in parallel, with per-scene logs. */
import { spawn } from "node:child_process";
import { createWriteStream, existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const jobs = [
  { id: "tokyo-rain", script: "tokyo_rain.py", time: .8 },
  { id: "alpine-lake-sunrise", script: "alpine_lake.py", time: .5 },
  { id: "atlantic-lighthouse", script: "atlantic_lighthouse.py", time: 1.4 },
];
const options = new Map(process.argv.slice(2).map(argument => {
  const match = /^--(scene|blender|width|height|samples|out)=(.+)$/.exec(argument);
  if (!match) throw new Error(`Unknown argument ${argument}`);
  return [match[1], match[2]];
}));
const blender = options.get("blender") ?? process.env.BLENDER ?? "/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender";
if (!existsSync(blender)) throw new Error("Supply --blender=/absolute/path/to/blender or BLENDER");
const selected = options.has("scene") ? jobs.filter(job => job.id === options.get("scene")) : jobs;
if (!selected.length) throw new Error(`Unknown scene. Choose ${jobs.map(job => job.id).join(", ")}`);
const numeric = (name, fallback) => {
  const value = Number(options.get(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`--${name} must be a positive integer`);
  return String(value);
};
const width = numeric("width", 1800), height = numeric("height", 1200), samples = numeric("samples", 192);
const destination = resolve(root, options.get("out") ?? "artifacts/water-scenes");
function run(command, args, logPath) {
  return new Promise((accept, reject) => {
    const log = createWriteStream(logPath);
    const child = spawn(command, args, { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
    child.stdout.pipe(log); child.stderr.pipe(log);
    child.once("error", error => { log.end(); reject(error); });
    child.once("close", code => { log.end(); code === 0 ? accept() : reject(new Error(`Exit ${code}: see ${logPath}`)); });
  });
}
const results = await Promise.allSettled(selected.map(async job => {
  const directory = resolve(destination, job.id);
  await mkdir(directory, { recursive: true });
  const water = resolve(directory, "water.obj"), render = resolve(directory, "render.png");
  console.log(`${job.id}: simulating to ${job.time}s`);
  await run(process.execPath, ["--import", "tsx", "tools/blender/export-nbflip-water.ts",
    `--scene=${job.id}`, `--time=${job.time}`, `--out=${water}`], resolve(directory, "simulation.log"));
  console.log(`${job.id}: rendering Cycles ${width}×${height}, ${samples} samples`);
  await run(blender, ["--background", "--python-exit-code", "1", "--python", `tools/blender/${job.script}`, "--",
    "--water-mesh", water, "--output", render, "--samples", samples, "--width", width, "--height", height], resolve(directory, "render.log"));
  console.log(`${job.id}: ${render}`);
}));
for (const result of results) if (result.status === "rejected") console.error(result.reason);
if (results.some(result => result.status === "rejected")) process.exitCode = 1;
