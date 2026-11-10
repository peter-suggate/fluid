/** The paddleboats collar gate (docs/plans/paddleboats-60fps-2026-11-10/progress-lane-d.md):
 * interleaved A/B pairs, each arm against a base run just before it, on three scenes, one
 * GPU check run a scene and arm, every one through tools/gpu/fluid-gpu-run.sh check:
 *   paddleboats   300 steps, bodies driven (tools/probe-census-reasons.ts --summary)
 *   letters       120 steps at 1/60 s (tools/probe-reservoir-bulk.ts, nbflip-figure-8-letters)
 *   channel dam    60 steps at 1/60 s (the dam of tests/uniform-channel-dam-volume.test.ts; this file's --worker=dam)
 * It prints one table: h tiles, particles, pressure band tiles and volume drift a run, and each
 * arm's change against its paired base. A run whose runner exits 76/77/78 (no GPU work, a Metal
 * abort, the guard) is invalid: it is marked and the gate stops (no reruns).
 *   node --import tsx tools/gate-paddleboats-collar.ts [--arms=liquid,deep,both,band25,shape1] [--scenes=paddleboats,letters,dam] [--dry-run]
 * --dry-run prints the commands only. */
import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import type { MethodParamValues } from "../lib/core/method-contract";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);

if (option("worker") !== undefined) await worker(option("worker")!);
else gate();

function gate(): void {
  const ARMS: Record<string, MethodParamValues> = {
    base: {},
    liquid: { collarSpread: "liquid" },
    deep: { collarDensity: "deep" },
    both: { collarSpread: "liquid", collarDensity: "deep" },
    band25: { particleBandDepth: 2.5 },
    shape1: { detailShapeTolerance: 1 },
  };
  const arms = (option("arms") ?? "liquid,deep,both,band25,shape1").split(",");
  const scenes = (option("scenes") ?? "paddleboats,letters,dam").split(",");
  for (const a of arms) if (!ARMS[a] || a === "base") throw new Error(`Unknown arm ${a}: ${Object.keys(ARMS).filter(k => k !== "base").join(", ")}`);
  for (const s of scenes) if (!["paddleboats", "letters", "dam"].includes(s)) throw new Error(`Unknown scene ${s}: paddleboats, letters, dam`);
  const runner = resolve("tools/gpu/fluid-gpu-run.sh"), out = join(tmpdir(), "fluid-collar-gate");
  mkdirSync(out, { recursive: true });
  const command = (scene: string, arm: string): string[] => {
    const values = JSON.stringify(ARMS[arm]);
    if (scene === "paddleboats") return ["node", "--import", "tsx", "tools/probe-census-reasons.ts", "--scene=paddleboats", "--steps=300", "--summary", `--label=${arm}`, `--values=${values}`];
    if (scene === "letters") return ["node", "--import", "tsx", "tools/probe-reservoir-bulk.ts", "--scene=nbflip-figure-8-letters", "--steps=120", "--every=120", "--dt=0.016666666666666666", "--stats-only", `--values=${values}`, `--out=${join(out, `letters-${arm}-${Date.now()}.json`)}`];
    return ["node", "--import", "tsx", "tools/gate-paddleboats-collar.ts", "--worker=dam", `--label=${arm}`, `--values=${values}`];
  };
  interface Row { scene: string; arm: string; pair: number; fine?: number; particles?: number; bandTiles?: number; drift?: number; rc: number; note?: string }
  const rows: Row[] = [];
  let pair = 0, invalid = false;
  // Interleaved: base, arm; base, arm; ... a scene at a time.
  outer: for (const scene of scenes) for (const arm of arms) {
    pair++;
    for (const which of ["base", arm]) {
      const args = command(scene, which);
      console.error(`[${scene} ${which} pair ${pair}] ${runner} check ${args.join(" ")}`);
      if (process.argv.includes("--dry-run")) { rows.push({ scene, arm: which, pair, rc: 0, note: "dry run" }); continue; }
      const run = spawnSync(runner, ["check", ...args], { encoding: "utf8", maxBuffer: 1 << 28 });
      const rc = run.status ?? -1, text = `${run.stdout ?? ""}\n${run.stderr ?? ""}`, row: Row = { scene, arm: which, pair, rc, ...parse(scene, text) };
      if (rc === 76 || rc === 77 || rc === 78) { row.note = `invalid (rc ${rc}): reset the GPU, do not rerun in a loop`; invalid = true; }
      else if (rc !== 0) row.note = `failed (rc ${rc}): ${text.trim().split("\n").slice(-3).join(" | ").slice(0, 200)}`;
      rows.push(row);
      if (invalid) break outer;
    }
  }
  const fmt = (v: number | undefined, digits = 0) => v === undefined || !Number.isFinite(v) ? "-" : v.toFixed(digits);
  const pct = (v: number | undefined) => v === undefined || !Number.isFinite(v) ? "-" : `${(100 * v).toFixed(3)}%`;
  const change = (a: number | undefined, b: number | undefined) => a === undefined || b === undefined || !b ? "-" : `${a >= b ? "+" : ""}${(100 * (a - b) / b).toFixed(1)}%`;
  const header = ["scene", "pair", "arm", "h tiles", "particles", "band tiles", "drift", "d tiles", "d particles", "d band", "d drift (pts)", "note"];
  const table = [header];
  for (const row of rows) {
    const base = row.arm === "base" ? undefined : rows.find(r => r.pair === row.pair && r.arm === "base");
    table.push([row.scene, String(row.pair), row.arm, fmt(row.fine), fmt(row.particles), fmt(row.bandTiles), pct(row.drift),
      base ? change(row.fine, base.fine) : "", base ? change(row.particles, base.particles) : "", base ? change(row.bandTiles, base.bandTiles) : "",
      base && row.drift !== undefined && base.drift !== undefined ? `${row.drift >= base.drift ? "+" : ""}${(100 * (row.drift - base.drift)).toFixed(3)}` : "", row.note ?? ""]);
  }
  const widths = header.map((_, i) => Math.max(...table.map(r => r[i]!.length)));
  for (const r of table) console.log(r.map((c, i) => c.padEnd(widths[i]!)).join("  ").trimEnd());
  if (invalid) process.exitCode = 76;
}

// Each scene's last report: paddleboats and the dam print a SUMMARY line; letters a row JSON a report step.
function parse(scene: string, text: string): { fine?: number; particles?: number; bandTiles?: number; drift?: number } {
  const lines = text.split("\n");
  if (scene !== "letters") {
    const line = lines.filter(l => l.startsWith("SUMMARY ")).pop(); if (!line) return {};
    const s = JSON.parse(line.slice(8)) as { fine?: number; particles?: number; bandTiles?: number; drift?: number };
    return { fine: s.fine, particles: s.particles, bandTiles: s.bandTiles, drift: s.drift };
  }
  const row = lines.filter(l => l.startsWith("{\"step\":")).map(l => JSON.parse(l) as { fine?: number; drift?: number; pressure?: { bandTiles?: number }; particles?: { live?: number } }).pop();
  return row ? { fine: row.fine, particles: row.particles?.live, bandTiles: row.pressure?.bandTiles, drift: row.drift } : {};
}

// --worker=dam: the channel dam of tests/uniform-channel-dam-volume.test.ts, 60 steps, one SUMMARY line.
async function worker(kind: string): Promise<void> {
  if (kind !== "dam") throw new Error(`Unknown worker ${kind}: dam`);
  process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
  const [{ createUniformTroughScene }, helpers, { paceDawnEventPolling }] = await Promise.all([
    import("../lib/core/uniform-trough-scenes"), import("../tests/helpers/uniform-geometric"), import("../lib/harness/node-dawn-provider"), import("../lib/methods")]);
  paceDawnEventPolling();
  const scene = createUniformTroughScene("dam-break");
  scene.container = { ...scene.container, width_m: 2.5, depth_m: 0.55 };
  scene.solidVoxels = []; scene.solidSurface = undefined;
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -1.25, y: 0, z: -0.275 }, max_m: { x: -0.65, y: 0.8, z: 0.275 } }];
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
  scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: 64 };
  const values: MethodParamValues = { volumeControlSeconds: 0, ...JSON.parse(option("values") ?? "{}") }, steps = 60;
  await helpers.withUniformDevice("collar gate channel dam", async device => {
    const solver = await helpers.createUniformSolver(device, scene, values);
    try {
      const frame = (solver as unknown as { mixedFrame: { requestNarrowBandVolumeProbe(): void; narrowBandFlip?: { count: number } } }).mixedFrame;
      for (let step = 1; step <= steps; step++) {
        if (step === steps) frame.requestNarrowBandVolumeProbe();
        if (!solver.advanceTo(step / 60, [])) throw new Error(`advance refused at step ${step}`);
        await solver.awaitFrameCompletion();
        if (solver.info.simulationPipelineError !== undefined) throw new Error(`pipeline error: ${solver.info.simulationPipelineError}`);
      }
      const stats = await solver.readStats();
      console.log(`SUMMARY ${JSON.stringify({ label: option("label") ?? "", scene: "channel-dam", step: steps, fine: solver.info.uniformMixedFineTiles, particles: frame.narrowBandFlip?.count, bandTiles: stats.uniformPressureBandTiles, drift: stats.volumeDrift })}`);
    } finally { solver.destroy(); }
  });
}
