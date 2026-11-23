// GPU-shape M0c (lane B): hardware counters per kernel of a brick-engine scene, with lane A's chase as calibration.
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/footprint-ctr.mts <scene> [outdir] [seconds]
// Spawns footprint.mts MODE=counters (split-pass steps, a dispatch an encoder, plus four calibration chases after each
// step), attaches Instruments' Metal GPU Counters for `seconds` once it prints READY, exports the trace tables, and
// reduces them with tools/xctrace-hardware-limiters.ts (counters only where one of our encoders runs alone). Adapted
// from ~/code/me/fluid-scratch/brick-tune/counters.mts. Prints a table: time share, occupancy, ALU, buffer load,
// LLC, MMU, TLB miss, read bandwidth, per kernel entry and per calibration chase.
import { execFileSync, spawn } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const { parseTraceTable } = await import(`${root}/tools/xctrace-trace-tables.ts`);
const [scene = "dam", outArg, secs = "3"] = process.argv.slice(2);
const out = resolve(outArg ?? resolve(homedir(), "code/me/fluid-scratch/gpu-shape/m0c-ctr", scene)); mkdirSync(out, { recursive: true });
const trace = `${out}/run.trace`; if (existsSync(trace)) rmSync(trace, { recursive: true });
const env = { ...process.env, MODE: "counters", SCENE: scene, OUT: resolve(out, ".."), FLUID_WEBGPU_DAWN_FEATURES: "use_user_defined_labels_in_backend" };
const child = spawn(process.execPath, ["--import", "tsx", resolve(here, "footprint.mts")], { cwd: root, env, stdio: ["ignore", "pipe", "inherit"] });
let attached = false;
const recorded = new Promise<void>((done, fail) => {
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (c: string) => {
    process.stdout.write(c);
    if (attached || !c.includes("READY")) return;
    attached = true;
    setTimeout(() => {
      console.log(`attaching to ${child.pid}`);
      const blank = resolve(execFileSync("xcode-select", ["-p"]).toString().trim(), "../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate");
      const x = spawn("xcrun", ["xctrace", "record", "--template", blank, "--instrument", "Metal Application", "--instrument", "GPU", "--instrument", "Metal GPU Counters",
        "--output", trace, "--no-prompt", "--time-limit", `${secs}s`, "--attach", String(child.pid)], { stdio: ["ignore", "pipe", "pipe"] });
      x.stdout.on("data", (d: Buffer) => process.stdout.write(`[xctrace] ${d}`));
      x.stderr.on("data", (d: Buffer) => process.stdout.write(`[xctrace] ${d}`));
      x.once("exit", code => code === 0 ? done() : fail(new Error(`xctrace ${code}`)));
    }, 1000);
  });
});
const finished = new Promise<number>(r => child.once("exit", c => r(c ?? 1)));
await recorded; const rc = await finished; console.log("recorded; child rc", rc);
const tables = [
  ["metal-gpu-intervals", "gpu-intervals", ["start", "duration", "channel-name", "event-label", "process", "encoder-id"]],
  ["metal-application-encoders-list", "encoders", ["encoder-id", "encoder-label", "process"]],
  ["gpu-counter-info", "counter-info", ["counter-id", "name", "type", "sample-interval"]],
  ["gpu-counter-value", "counter-value", ["timestamp", "counter-id", "value", "ring-buffer-index"]],
] as const;
for (const [schema, file, columns] of tables) {
  const p = spawn("xcrun", ["xctrace", "export", "--input", trace, "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${schema}"]`], { stdio: ["ignore", "pipe", "inherit"] });
  p.stdout.setEncoding("utf8");
  const sink = createWriteStream(`${out}/${file}.ndjson`); let n = 0;
  for await (const row of parseTraceTable(p.stdout as AsyncIterable<string>, { columns })) { sink.write(JSON.stringify(row) + "\n"); n++; }
  await new Promise(r => sink.end(r)); console.log(file, n);
}
execFileSync(process.execPath, ["--import", "tsx", `${root}/tools/xctrace-hardware-limiters.ts`, `--artifact=${out}`, `--pid=${child.pid}`], { cwd: root, stdio: "inherit" });
// The table (ctrtab.py's columns), time-weighted per label.
const C = ["Compute Occupancy", "ALU Utilization", "ALU Limiter", "Buffer Load Utilization", "Buffer Read Limiter", "GPU Last Level Cache Utilization", "GPU Last Level Cache Limiter", "MMU Utilization", "MMU Limiter", "MMU TLB Miss Rate", "GPU Read Bandwidth"];
const short = ["occ", "aluU", "aluL", "ldU", "bufRL", "llcU", "llcL", "mmuU", "mmuL", "tlbMiss", "rdGBs"];
const d = JSON.parse(readFileSync(`${out}/hardware-limiters.json`, "utf8"));
const agg = new Map<string, { us: number; c: Map<string, [number, number]> }>();
for (const st of d.stages) {
  const lab = String(st.label).replace(/step\d+\./, "").split("|").pop()!.split(":")[0]!;
  const a = agg.get(lab) ?? { us: 0, c: new Map() }; agg.set(lab, a); a.us += st.exclusiveUs;
  for (const k of C) if (st.counters[k]) { const v = a.c.get(k) ?? [0, 0]; v[0] += st.counters[k].meanAcrossRings * st.exclusiveUs; v[1] += st.exclusiveUs; a.c.set(k, v); }
}
const tot = [...agg].filter(([l]) => !l.startsWith("cal.")).reduce((s, [, a]) => s + a.us, 0);
const lines = [["label", "share%", ...short].join("\t")];
for (const [lab, a] of [...agg].sort((x, y) => y[1].us - x[1].us))
  lines.push([lab, lab.startsWith("cal.") ? "cal" : (100 * a.us / tot).toFixed(1), ...C.map(k => { const v = a.c.get(k); return v && v[1] ? (v[0] / v[1]).toFixed(1) : "nan"; })].join("\t"));
writeFileSync(`${out}/counters.tsv`, lines.join("\n") + "\n");
console.log(lines.slice(0, 40).join("\n"));
