/** Native 2D energy attribution using the exact advance-lab seed and defaults.
 * node --import tsx tools/wasm/uniform-geometric-energy-audit.ts --seconds=120
 * --controls also runs option controls and same-state pressure replays.
 * The overfill ablation is diagnostic only, not a proposed production fix.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { uniformLabSeed, UNIFORM_LAB_VALUES } from "../../lib/physics-wasm/uniform-controller";

const args = new Map(process.argv.slice(2).map((arg) => {
  const [key, ...value] = arg.replace(/^--/, "").split("=");
  return [key!, value.join("=")];
}));
const out = args.get("out") ?? "docs/research/uniform-geometric-late-energy-2026-09-20";
mkdirSync(out, { recursive: true });
const scene = args.get("scene") ?? "coarse-first-pool-impact-half";
const definition = findSceneDefinition(scene);
assert.ok(definition, scene);
const seed = uniformLabSeed(sceneDocument(definition));
const request = {
  ...seed, options: UNIFORM_LAB_VALUES, dt: 1 / 30,
  frames: Math.round(Number(args.get("seconds") ?? 120) * 30), auditEnergy: true,
};
const build = spawnSync("cargo", ["build", "--release", "--manifest-path", "rust/Cargo.toml", "-p", "fluid-core", "--example", "uniform_geometric_scene"], { stdio: "inherit" });
assert.equal(build.status, 0);
function save(name: string, value: unknown) {
  writeFileSync(`${out}/${name}.json.gz`, gzipSync(JSON.stringify(value)));
}
function run(name: string, input: object) {
  const child = spawnSync("rust/target/release/examples/uniform_geometric_scene", [], {
    input: JSON.stringify(input), encoding: "utf8", maxBuffer: 256 * 1024 * 1024,
  });
  assert.equal(child.status, 0, child.stderr);
  const result = JSON.parse(child.stdout);
  save(`${name}-input`, input);
  save(name, result);
  console.log(`${name}: ${result.receipts.length} frames, ${Math.round(result.elapsedMs)} ms`);
  return result;
}
const baseline = run("baseline", { ...request, energySnapshotFrames: [150, 158, 194, 209, 298, 308, 450, 900] });
run("no-overfill", { ...request, disableOverfillCorrection: true });

// A pressure source must not be mistaken for a poorly converged solve:
// stationary, zero gravity, planar surface, uniform 5% overfill, one step.
const flat = {
  ...request, dimensions: [32, 24], cellSize: [0.1, 0.1], gravity: [0, 0], frames: 1,
  options: { ...request.options, totalSurfaceVolume: "off" },
  volume: Array.from({ length: 32 * 24 }, (_, i) => i < 32 * 12 ? 1.05 : 0),
  capacity: Array(32 * 24).fill(1),
  phi: Array.from({ length: 33 * 25 }, (_, i) => Math.floor(i / 33) * 0.1 - 1.2),
};
run("overfilled", flat);
run("overfilled-no-overfill", { ...flat, disableOverfillCorrection: true });

if (args.has("controls")) {
  for (const [name, options] of Object.entries({
    "no-area": { totalSurfaceVolume: "off" },
    "tight-pressure": { pressureResidualTolerance: 0.001, pressureCycleBudget: "fixed" },
    "all-fine": { twoLevelVelocity: "off" },
    "liquid-only": { liquidOnlyVelocityAdvection: "on" },
    "no-sharpen": { densitySharpening: "off" },
  })) run(name, { ...request, frames: name === "no-area" ? request.frames : 900, options: { ...request.options, ...options } });
  const hydrostatic = { ...flat, frames: 900, gravity: seed.gravity,
    volume: flat.volume.map((v) => v > 0 ? 1 : 0), options: request.options };
  run("flat", hydrostatic);
  for (const frame of [150, 158, 194, 209, 298, 308, 450, 900]) {
    const state = baseline.energySnapshots?.find((s: { frame: number; stage: string }) => s.frame === frame && s.stage === "start");
    if (!state) continue;
    const { volume, phi, velocity, lowX, lowY, released } = state;
    const replay = { ...request, volume, phi, velocity, lowX, lowY, released, frames: 1 };
    run(`replay-${frame}`, replay);
    run(`replay-${frame}-no-overfill`, { ...replay, disableOverfillCorrection: true });
  }
}
save("manifest", { scene, sourceCommit: spawnSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).stdout.trim(),
  dimensions: seed.dimensions, cellSize: seed.cellSize, options: request.options, dt: request.dt,
  description: "Native Rust 2D, default physics plus read-only energy observers and opt-in pressure-source ablation. Energies in J/m of depth; rho times cell area. MAC face-square and contour-weighted quadratures are reported separately." });
