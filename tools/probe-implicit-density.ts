import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { applyStencil, compileStencil, packetCost, type Donor } from "./implicit-density/compiled-stencil";
import { evaluate, frameForBox, mean, splitBox, type Vec3 } from "./implicit-density/field";
import { deferredExpectations, fixtures } from "./implicit-density/ladder-fixtures";
import { LADDER_BUDGETS, runLadder, verifyUnrepresentableMerge } from "./implicit-density/ladder";

function support(mixed: boolean): Donor[] {
  const donors: Donor[] = [];
  for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
    const cell = { lower: [x, y, z].map(i => -1 + i * 2 / 3) as unknown as Vec3,
      upper: [x, y, z].map(i => -1 + (i + 1) * 2 / 3) as unknown as Vec3 };
    const cells = mixed && x === 2 ? splitBox(cell) : [cell];
    for (const box of cells) donors.push({ id: donors.length * 7 + 13, box });
  }
  return donors;
}
function reconstructionReceipt(mixed: boolean) {
  const donors = support(mixed);
  const home = donors.find(d => d.box.lower.every((x, a) => x < 0 && d.box.upper[a] > 0))!;
  const frame = frameForBox(home.box);
  const start = performance.now();
  const packet = compileStencil(donors, home.id, frame, 1);
  const compile_ms = performance.now() - start;
  const rows = fixtures.map(f => {
    const densities = new Map(donors.map(d => [d.id, f.exactMean(d.box)]));
    const fitted = applyStencil(packet, 1, id => densities.get(id)!);
    const f32 = applyStencil(packet, 1, id => densities.get(id)!, true);
    const surfaceError = Math.max(...f.surfacePoints.map(p => Math.abs(evaluate(fitted, p) - 0.5)));
    const f32SurfaceError = Math.max(...f.surfacePoints.map(p => Math.abs(evaluate(f32, p) - 0.5)));
    const donorResidual = Math.max(...donors.map(d => Math.abs(mean(fitted, d.box) - densities.get(d.id)!)));
    return { fixture: f.id, exactQuadraticFamily: f.field.kind === "polynomial", surfaceDensityResidual: surfaceError,
      f32SurfaceDensityResidual: f32SurfaceError, donorMeanResidual: donorResidual,
      homeMeanResidual: Math.abs(mean(fitted, home.box) - densities.get(home.id)!) };
  });
  const values = new Map(donors.map(d => [d.id, fixtures[3].exactMean(d.box)]));
  for (let warmup = 0; warmup < 1000; warmup++) applyStencil(packet, 1, id => values.get(id)!);
  const applies = 10000, began = performance.now(); let checksum = 0;
  for (let i = 0; i < applies; i++) checksum += applyStencil(packet, 1, id => values.get(id)!).coefficients[0];
  return { mixed, compile_ms, apply_microseconds: 1000 * (performance.now() - began) / applies,
    timingScope: "CPU JS Map reads and allocations; not GPU or production frame cost", checksum,
    cost: packetCost(packet), rows };
}

const output = resolve(process.argv.find(a => a.startsWith("--out="))?.slice(6)
  ?? "artifacts/implicit-density/ladder.json");
const sources = ["tools/implicit-density/field.ts", "tools/implicit-density/compiled-stencil.ts",
  "tools/implicit-density/ladder-fixtures.ts", "tools/implicit-density/ladder.ts", "tools/probe-implicit-density.ts"];
const hashes = Object.fromEntries(await Promise.all(sources.map(async path => [path,
  createHash("sha256").update(await readFile(path)).digest("hex")])));
const began = performance.now();
const rows = runLadder(true), reconstruction = [reconstructionReceipt(false), reconstructionReceipt(true)];
const rejectNewDetail = verifyUnrepresentableMerge();
const passed = rows.every(r => r.passed) && rejectNewDetail && reconstruction.every(r => r.rows.every(f =>
  !f.exactQuadraticFamily || f.surfaceDensityResidual < LADDER_BUDGETS.surfaceResidual));
const report = { schema: 1, createdAt: new Date().toISOString(), runtime: process.version, sources: hashes,
  scope: "Standalone local density algebra and mean-based reconstruction. No global patch assembly, feature inference, CM12 transport, pressure, production topology adapter or GPU validation.",
  budgets: LADDER_BUDGETS, passed, rejectNewDetail, elapsed_ms: performance.now() - began,
  rows, reconstruction, deferred: deferredExpectations };
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ output, passed, fixtures: rows.length,
  maxSurfaceResidual: Math.max(...rows.map(r => r.maximumSurfaceResidual)),
  maxMassError: Math.max(...rows.map(r => r.maximumMassError)),
  reconstruction: reconstruction.map(r => ({ mixed: r.mixed, cost: r.cost,
    compile_ms: r.compile_ms, apply_microseconds: r.apply_microseconds,
    exactFamilyMaxSurfaceResidual: Math.max(...r.rows.filter(f => f.exactQuadraticFamily).map(f => f.surfaceDensityResidual)),
    quadraticFitEdgeResidual: Math.max(...r.rows.filter(f => !f.exactQuadraticFamily).map(f => f.surfaceDensityResidual)) })) }, null, 2));
if (!passed) process.exitCode = 1;
