/** Where the long slide's work lies at one frame: particle and tile census.
 * Runs the UI-default scene to --frame, then reads back the live samples and
 * the ownership's tile words (diagnostic readback after the timed work).
 * node --import tsx tools/probe-nbflip-occupancy.ts [--scene=nbflip-water-slide-long] [--frame=600]
 * Prints: particle y profile, h/4h tiles by distance (in tiles, Chebyshev)
 * to the nearest tile holding a sample, and a top-down sample map. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { UNIFORM_PAGE_TABLE_WORDS } from "../lib/methods/uniform/uniform-lattice-address";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const target = Number(option("frame") ?? 600);
paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
const raw = await adapter.requestDevice({ requiredFeatures: [...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const device = managedGPUDevice(raw, { requireWorkerRealm: false });
const definition = getSceneDefinition(option("scene") ?? "nbflip-water-slide-long"), quality = definition.methodProfile!.quality;
const scene = structuredClone(sceneDocument(definition));
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
const internal = solver as unknown as { mixedFrame: { setLayoutViews(on: boolean): void }; mixedDynamic: { reasons: { buffer: GPUBuffer; offset: number } } };
for (let frame = 1; frame <= target; frame++) { if (frame === target) internal.mixedFrame.setLayoutViews(true); assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion(); }
const read = async (buffer: GPUBuffer, offset: number, size: number) => {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, offset, staging, 0, size); device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ); const out = staging.getMappedRange().slice(0); staging.destroy(); return out;
};
const source = solver.particleSource!;
const live = new Uint32Array(await read(source.liveCount!.buffer, source.liveCount!.byteOffset ?? 0, 4))[0]!;
const samples = new Float32Array(await read(source.buffer, 0, live * source.strideFloats * 4));
const frame = (solver as unknown as { mixedFrame: { ownership: { presentation: GPUBufferBinding; speeds: GPUBuffer; capacity: { tiles: number } } } }).mixedFrame;
const n = frame.ownership.capacity.tiles, blocks = n / 512;
const words = new Uint32Array(await read(frame.ownership.presentation.buffer, frame.ownership.presentation.offset ?? 0, 4 * (5 * n + UNIFORM_PAGE_TABLE_WORDS + blocks)));
const recordBase = 5 * n + UNIFORM_PAGE_TABLE_WORDS;
const speeds = new Uint32Array(await read(frame.ownership.speeds, 0, 8 * n));
const reasons = new Uint32Array(await read(internal.mixedDynamic.reasons.buffer, internal.mixedDynamic.reasons.offset, 4 * n));
const reasonNames = ["skipped", "traced", "crossing", "closure", "travel", "join", "source", "solid", "bulk"];
const byReason = new Map<string, number[]>();
const reachHist = new Map<string, number>();
const still = { h: new Array(8).fill(0), c: new Array(8).fill(0) };
const framePage = [0, 1, 2].map(k => words[5 * n + 64 ** 3 + k]! | 0);
// Tiles holding a sample, by tile coordinate in the window.
const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
const wet = new Map<string, number>();
const yHist = new Map<number, number>();
let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
for (let i = 0; i < live; i++) {
  const p = [0, 1, 2].map(k => samples[i * source.strideFloats + k]!);
  for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k]!, p[k]!); hi[k] = Math.max(hi[k]!, p[k]!); }
  const t = p.map(v => Math.floor(v / 4));
  wet.set(key(t[0]!, t[1]!, t[2]!), (wet.get(key(t[0]!, t[1]!, t[2]!)) ?? 0) + 1);
  yHist.set(t[1]!, (yHist.get(t[1]!) ?? 0) + 1);
}
console.log(JSON.stringify({ frame: target, live, framePage, sampleBounds: [lo, hi], wetTiles: wet.size }));
console.log("samples by tile row y (cells 4y..4y+3):", [...yHist].sort((a, b) => a[0] - b[0]).map(([y, c]) => `${y}:${c}`).join(" "));
// Every stored tile with its width; distance to the nearest wet tile (Chebyshev, capped).
const wetCoords = [...wet.keys()].map(s => s.split(",").map(Number));
const near = new Map<string, number>();
const CAP = 6;
for (const [x, y, z] of wetCoords) for (let dz = -CAP; dz <= CAP; dz++) for (let dy = -CAP; dy <= CAP; dy++) for (let dx = -CAP; dx <= CAP; dx++) {
  const d = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)), k = key(x! + dx, y! + dy, z! + dz);
  if ((near.get(k) ?? 99) > d) near.set(k, d);
}
const census = { h: new Array(CAP + 2).fill(0), c: new Array(CAP + 2).fill(0) };
let stored = 0;
const columns = new Map<string, { h: number; c: number; wet: number }>();
for (let b = 0; b < blocks; b++) {
  const r = words[recordBase + b]!;
  if (r === 0xffffffff || b === 0 || b === blocks - 1) continue;
  const page = [r & 63, (r >>> 6) & 63, (r >>> 12) & 63];
  stored++;
  for (let l = 0; l < 512; l++) {
    const t = [page[0]! * 8 + (l & 7), page[1]! * 8 + ((l >> 3) & 7), page[2]! * 8 + (l >> 6)];
    const fine = (words[512 * b + l]! & 0x80000000) !== 0, d = Math.min(near.get(key(t[0]!, t[1]!, t[2]!)) ?? CAP + 1, CAP + 1);
    census[fine ? "h" : "c"][d]++;
    if (fine) { const name = reasonNames[reasons[512 * b + l]!] ?? `r${reasons[512 * b + l]}`; const row = byReason.get(name) ?? new Array(CAP + 2).fill(0); row[d]++; byReason.set(name, row); }
    const r = speeds[512 * b + l]!; if (r === (2 | 2 << 5 | 2 << 10 | 2 << 15 | 2 << 20 | 2 << 25)) still[fine ? "h" : "c"][d]++;
    if (d >= 5) { const w = speeds[n + 512 * b + l]! & 0x3fffffff; const m = Math.max(...[0, 5, 10, 15, 20, 25].map(k => (w >>> k) & 31)), lm = Math.max(...[0, 5, 10, 15, 20, 25].map(k => (r >>> k) & 31)); const key = `${fine ? "h" : "c"}${d} spread${m} local${lm}`; reachHist.set(key, (reachHist.get(key) ?? 0) + 1); }
    const col = key(t[0]! >> 2, 0, t[2]! >> 2), e = columns.get(col) ?? { h: 0, c: 0, wet: 0 };
    if (fine) e.h++; if (d === 0) e.wet++; columns.set(col, e);
  }
}
console.log(JSON.stringify({ storedBlocks: stored, hTilesByDistanceToWet: census.h, coarseTilesByDistanceToWet: census.c, zeroSpeedH: still.h, zeroSpeedCoarse: still.c }));
console.log("h tiles by census reason x distance to wet:", JSON.stringify(Object.fromEntries(byReason)));
console.log([...reachHist].sort().map(([k, v]) => `${k}:${v}`).join("  "));
// Top-down: 16-cell columns, h tiles in each ('.' none, digits log2, '#' 64+).
const sx = Math.max(...[...columns.keys()].map(s => Number(s.split(",")[0]))) + 1, sz = Math.max(...[...columns.keys()].map(s => Number(s.split(",")[2]))) + 1;
for (const field of ["h", "wet"] as const) {
  console.log(`${field} tiles per 16x16-cell column (x across, z down)`);
  for (let z = 0; z < sz; z++) {
    let row = "";
    for (let x = 0; x < sx; x++) { const v = columns.get(key(x, 0, z))?.[field] ?? 0; row += v === 0 ? "." : v >= 64 ? "#" : String(Math.min(9, Math.floor(Math.log2(v)) + 1)); }
    console.log(row);
  }
}
solver.destroy(); await device.queue.onSubmittedWorkDone(); device.destroy();
