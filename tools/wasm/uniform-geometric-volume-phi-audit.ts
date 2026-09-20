/** Native 2D causal audit; no GPU lease or browser required.
 * node --import tsx tools/wasm/uniform-geometric-volume-phi-audit.ts
 * Optional: --seconds=4 --arm=dt30-eight --out=/absolute/directory
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { createSparseCM12ComplexityScene } from "../../lib/core/scenes";
import {
  uniformLabSeed,
  UNIFORM_LAB_VALUES,
} from "../../lib/physics-wasm/uniform-controller";

const args = new Map(
  process.argv.slice(2).map((a): [string, string] => {
    const [key, ...value] = a.replace(/^--/, "").split("=");
    return [key!, value.join("=")];
  }),
);
const seconds = Number(args.get("seconds") ?? 4);
assert.ok(Number.isFinite(seconds) && seconds > 0);
const out =
  args.get("out") ?? "docs/research/uniform-geometric-volume-phi-2026-09-20";
mkdirSync(out, { recursive: true });
const scene = createSparseCM12ComplexityScene("long-dam");
const seed = uniformLabSeed(scene);
const arms = [
  { id: "dt30-eight", dt: 1 / 30, rounds: 8, options: {} },
  { id: "dt120-eight", dt: 1 / 120, rounds: 8, options: {} },
  { id: "dt30-thirtytwo", dt: 1 / 30, rounds: 32, options: {} },
  {
    id: "dt30-no-redistance",
    dt: 1 / 30,
    rounds: 8,
    options: { redistance: "off" },
  },
  { id: "dt30-no-sharpen", dt: 1 / 30, rounds: 0, options: {} },
  {
    id: "dt30-fine-extension",
    dt: 1 / 30,
    rounds: 8,
    options: { twoLevelVelocity: "off", extensionFrontSweeps: 16 },
  },
  {
    id: "dt30-front16",
    dt: 1 / 30,
    rounds: 8,
    options: { extensionFrontSweeps: 16 },
  },
  {
    id: "dt30-tight-pressure",
    dt: 1 / 30,
    rounds: 8,
    options: {
      pressureResidualTolerance: 0.01,
      pressureCycleBudget: "fixed",
      pressureFullCycles: 3,
      pressureVCycles: 8,
    },
  },
];
const selected = args.has("arm")
  ? arms.filter((a) => a.id === args.get("arm"))
  : arms.slice(0, 3);
assert.ok(selected.length, "unknown arm");
const build = spawnSync(
  "cargo",
  [
    "build",
    "--release",
    "--manifest-path",
    "rust/Cargo.toml",
    "-p",
    "fluid-core",
    "--example",
    "uniform_geometric_scene",
  ],
  { stdio: "inherit" },
);
assert.equal(build.status, 0);
for (const arm of selected) {
  const request = {
    ...seed,
    options: { ...UNIFORM_LAB_VALUES, totalSurfaceVolume: "off", ...arm.options },
    dt: arm.dt,
    frames: Math.round(seconds / arm.dt),
    auditStages: true,
    sharpeningRounds: arm.rounds,
    auditReplayFrames: [0.5, 1, 2, 4]
      .filter((t) => t <= seconds)
      .map((t) => Math.round(t / arm.dt)),
    auditTraceFrames: args.has("traces")
      ? [0.2, 0.5, 1]
          .filter((t) => t <= seconds)
          .map((t) => Math.round(t / arm.dt))
      : [],
  };
  writeFileSync(
    `${out}/${arm.id}-input.json.gz`,
    gzipSync(JSON.stringify(request)),
  );
  console.log(
    `Running ${arm.id}: ${request.frames} frames on ${seed.dimensions.join("x")}`,
  );
  const run = spawnSync(
    "rust/target/release/examples/uniform_geometric_scene",
    [],
    {
      input: JSON.stringify(request),
      encoding: "utf8",
      maxBuffer: 128 * 1024 * 1024,
    },
  );
  assert.equal(run.status, 0, run.stderr);
  const result = JSON.parse(run.stdout);
  writeFileSync(`${out}/${arm.id}.json.gz`, gzipSync(JSON.stringify(result)));
  const stages = result.stages as {
    frame: number;
    stage: string;
    metrics: Record<string, number>;
  }[];
  let advection = 0,
    redistance = 0;
  for (let i = 0; i < stages.length; i += 6) {
    advection +=
      stages[i + 1]!.metrics.contourArea! - stages[i]!.metrics.contourArea!;
    redistance +=
      stages[i + 2]!.metrics.contourArea! - stages[i + 1]!.metrics.contourArea!;
  }
  const initial = stages[0]!.metrics;
  const final = result.finalHighResolutionMetrics;
  const dust = result.receipts.reduce(
    (
      s: number,
      r: { sharpeningDust: number; transport: { dustVolume: number } },
    ) => s + r.sharpeningDust + r.transport.dustVolume,
    0,
  );
  const summary = {
    arm,
    seconds,
    dimensions: seed.dimensions,
    cellSize: seed.cellSize,
    initial,
    final,
    advectionAreaChange: advection,
    redistanceAreaChange: redistance,
    dust,
    conservationError: final.volume + dust - initial.volume!,
    quadratureDifference:
      final.contourArea - stages.at(-1)!.metrics.contourArea!,
    maxCourant: Math.max(...stages.map((s) => s.metrics.maxCourant!)),
    maxPressureResidual: Math.max(
      ...result.receipts.map(
        (r: { pressure: { residual: number } }) => r.pressure.residual,
      ),
    ),
    elapsedMs: result.elapsedMs,
    sharpeningReplays: result.sharpeningReplays,
  };
  writeFileSync(
    `${out}/${arm.id}-summary.json`,
    JSON.stringify(summary, null, 2) + "\n",
  );
  console.log(JSON.stringify({ ...summary, sharpeningReplays: undefined }));
}
