/** Diagnostic global compensation study; all variants retain positive overfill RHS.
 * node --import tsx tools/wasm/uniform-geometric-compensation.ts --seconds=120 --controls
 * --modes=balance-deficit,source-work selects a subset; --seconds=300 for a longer soak.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { uniformLabSeed, UNIFORM_LAB_VALUES } from "../../lib/physics-wasm/uniform-controller";
const args = new Map(process.argv.slice(2).map((arg) => {
  const [key, ...rest] = arg.replace(/^--/, "").split("="); return [key!, rest.join("=")];
}));
const out = args.get("out") ?? "docs/research/uniform-geometric-compensation-2026-09-20";
mkdirSync(out, { recursive: true });
const seed = uniformLabSeed(sceneDocument(findSceneDefinition("coarse-first-pool-impact-half")!));
const base = { ...seed, options: UNIFORM_LAB_VALUES, frames: Math.round(Number(args.get("seconds") ?? 120)*30), dt: 1/30, auditEnergy: true };
const variants: Record<string, object> = {
  baseline: { mode: "off" },
  "balance-uniform": { mode: "balance-uniform" },
  "balance-deficit": { mode: "balance-deficit" },
  "balance-surface-deficit": { mode: "balance-surface-deficit" },
  "source-work": { mode: "source-work" },
  "source-work-debt": { mode: "source-work", carryDebt: true },
  "source-rate-025": { mode: "source-rate", gain: 0.25 },
  "source-rate-1": { mode: "source-rate", gain: 1 },
  "source-rate-4": { mode: "source-rate", gain: 4 },
  "energy-cap": { mode: "energy-cap" },
};
const selected = (args.get("modes") ?? Object.keys(variants).join(",")).split(",");
for (const name of selected) assert.ok(variants[name], `Unknown mode ${name}`);
assert.equal(spawnSync("cargo", ["build", "--release", "--manifest-path", "rust/Cargo.toml", "-p", "fluid-core", "--example", "uniform_geometric_scene"], { stdio: "inherit" }).status, 0);
function save(name: string, data: unknown) { writeFileSync(`${out}/${name}.json.gz`, gzipSync(JSON.stringify(data))); }
function run(name: string, input: object) {
  const child = spawnSync("rust/target/release/examples/uniform_geometric_scene", [], { input: JSON.stringify(input), encoding: "utf8", maxBuffer: 256*1024*1024 });
  assert.equal(child.status, 0, child.stderr);
  const result = JSON.parse(child.stdout);
  save(`${name}-input`, input); save(name, result);
  const end = result.energy.at(-1).metrics;
  console.log(`${name}: ${result.receipts.length/30}s; final KE ${end.phiKinetic.toFixed(3)} J/m; ${(result.elapsedMs/1000).toFixed(2)} wall s`);
  return result;
}
for (const name of selected) run(name, { ...base, energyExperiment: variants[name] });
if (args.has("controls")) {
  const flat = { ...base, dimensions: [32,24], cellSize: [0.1,0.1],
    phi: Array.from({length:33*25},(_,i)=>Math.floor(i/33)*0.1-1.2),
    volume: Array.from({length:32*24},(_,i)=>i<32*12?1:0), capacity: Array(32*24).fill(1), frames: 900 };
  const drop = { ...base, dimensions: [32,48], cellSize: [0.1,0.1], frames: 12,
    phi: Array.from({length:33*49},(_,i)=>Math.hypot((i%33)*0.1-1.6,Math.floor(i/33)*0.1-3.4)-0.5),
    volume: Array.from({length:32*48},(_,i)=>{
      let fill=0; for(let y=0;y<8;y++) for(let x=0;x<8;x++)
        if(Math.hypot((i%32+(x+0.5)/8)*0.1-1.6,(Math.floor(i/32)+(y+0.5)/8)*0.1-3.4)<0.5) fill++;
      return fill/64;
    }), capacity: Array(32*48).fill(1) };
  for (const name of selected) {
    run(`flat-${name}`, { ...flat, energyExperiment: variants[name] });
    run(`falling-drop-${name}`, { ...drop, energyExperiment: variants[name] });
    const mandatory = run(`overfilled-${name}`, { ...flat, frames: 1, gravity: [0,0],
      volume: flat.volume.map(v=>1.05*v), options: {...base.options,totalSurfaceVolume:"off"}, energyExperiment: variants[name] });
    assert.ok(mandatory.energy.at(-1).metrics.phiKinetic>500, "Mandatory expansion must not be damped away");
    const maxDiv = mandatory.energyCompensation?.[0].maxLiquidDivergenceError;
    if (maxDiv !== undefined) assert.ok(maxDiv<0.005, `Expansion divergence lost: ${maxDiv}`);
  }
}
save("manifest", { sourceCommit: spawnSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).stdout.trim(),
  selected, variants, dt: base.dt, frames: base.frames, options: base.options,
  note: "Diagnostic-only native 2D experiments. Final pressure solve retains positive overfill RHS. No shared method defaults changed." });
