/**
 * A sheet of water at rest on the slope clearance's 1:16 incline (x 4..88 cells, the full width), on NB-FLIP's brick
 * engine: on a frictionless free-slip bed its mean downhill speed grows at g sin(theta) (0.612 m/s^2) until it reaches
 * the pit. Every --every steps: the sheet's particles' mean speed along the bed's downhill tangent, coupled and
 * ballistic apart, and the grid velocity the coupled ones sampled (pic), against g sin(theta) t.
 *
 *   tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-incline-film.ts [--thickness=0.6] [--steps=60]
 *     [--every=10] [--boundary=terrain|procedural]
 */
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { getMethod } from "../lib/core/method-registry";
import "../lib/methods/index";
import { resolveMethodValues } from "../lib/core/method-contract";
import { NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_GRID } from "../lib/core/nbflip-ramp-scene";
import { NBFLIP_METHOD_PROFILE, cellPoint } from "../lib/core/nbflip-paper-scenes";
import { createNbflipSlopeClearance, nbflipSlopeClearanceBed } from "../lib/core/nbflip-slope-clearance-scene";
import { bakeProceduralSolid } from "../lib/core/procedural-solid";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";

const args = process.argv.slice(2), opt = (k: string) => args.find(a => a.startsWith(`--${k}=`))?.slice(k.length + 3);
const thickness = Number(opt("thickness") ?? 0.6), steps = Number(opt("steps") ?? 60), every = Number(opt("every") ?? 10);
const h = NBFLIP_RAMP_CELL_SIZE_M, grid = NBFLIP_RAMP_GRID, [, , nz] = grid, X0 = 4, X1 = 88;
const scene = createNbflipSlopeClearance(60);
scene.fluid.initialLiquidVolumes = Array.from({ length: X1 - X0 }, (_, i) => {
  const x = X0 + i, bed = Math.max(nbflipSlopeClearanceBed(x), nbflipSlopeClearanceBed(x + 1));
  return { shape: "box" as const, min_m: cellPoint(grid, h, [x, bed + 0.02, 1]), max_m: cellPoint(grid, h, [x + 1, bed + 0.02 + thickness, nz - 1]) };
});
scene.fluid.brickPools = { f: 425, c: 20, b: 16, k: Number(opt("k") ?? 0), particles: 200000 };
// --u0=U: the sheet starts moving at U m/s along the bed's downhill tangent; --nog: no gravity (an inviscid sheet on a
// frictionless bed then keeps U; anything else is the solver's).
const u0 = Number(opt("u0") ?? 0), tan = [16 / Math.sqrt(257), -1 / Math.sqrt(257)];
if (u0) scene.fluid.initialVelocity_m_s = { x: u0 * tan[0]!, y: u0 * tan[1]!, z: 0 };
if (args.includes("--nog")) scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
if (opt("boundary") === "procedural") {
  delete scene.terrain;
  Object.assign(scene, bakeProceduralSolid(grid, [h, h, h], ([x, y]) => y - nbflipSlopeClearanceBed(x)));
}
const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), profile = NBFLIP_METHOD_PROFILE;
const engine = opt("engine") ?? "brick";
const values = resolveMethodValues(method, profile.quality, { ...method.appDefaults, ...profile.overrides, engine, ...JSON.parse(opt("values") ?? "{}") });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = managedGPUDevice(await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
let gpuError = "";
device.addEventListener("uncapturederror", (e: GPUUncapturedErrorEvent) => { e.preventDefault(); gpuError ||= e.error.message.slice(0, 300); });
async function readBuffer(buffer: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); if (bytes) enc.copyBufferToBuffer(buffer, offset, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ); const out = read.getMappedRange().slice(0); read.unmap(); read.destroy(); return out;
}
const median = (a: number[]) => { if (!a.length) return NaN; const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]!; };
const f = (x: number) => +x.toFixed(3);
// --dump=k (brick): at step k, an x-y slice at z = 16 over x --dumpX0..+20 and the rows about the bed: per cell the
// FLAGS (O open, L liquid), PHI (the cell's low vertex, cells), U0 (its low x face, m/s) and its particles' mean u.
const dumpStep = Number(opt("dump") ?? -1), dumpX0 = Number(opt("dumpX0") ?? 30);
async function dumpSlice(solver: any, p: Float32Array, count: number, stride: number, scale: number[]) {
  const e = solver.brickEngine, L = e.layout, dir = await e.read("dir"), vel = new Float32Array((await e.read("vel")).buffer), phiW = await e.read("phi"), phi = new Float32Array(phiW.buffer), pres = new Float32Array((await e.read("pres")).buffer);
  const slot = (c: number[]) => {
    const cs = dir[L.dir.page + L.pageIndex(c[0]! >> 5, c[1]! >> 5, c[2]! >> 5)]!; if (cs >= 0x80000000) return -1;
    const b = c.map(x => x >> 3), s = dir[L.dir.child + 64 * cs + ((b[0]! & 3) | ((b[1]! & 3) << 2) | ((b[2]! & 3) << 4))]!; return s >= 0x80000000 ? -1 : s;
  };
  const local = (c: number[]) => (c[0]! & 7) | ((c[1]! & 7) << 3) | ((c[2]! & 7) << 6);
  const xB = L.base("pres", "F", "X"), dB = L.base("pres", "F", "DIAG"), uB = L.base("vel", "F", "U0"), vB = L.base("vel", "F", "U1"), fB = L.base("phi", "F", "FLAGS"), pB = L.base("phi", "F", "PHI");
  const z = 16, mean = new Map<string, number[]>();
  for (let i = 0; i < count; i++) { const o = stride * i; if (!(p[o + 3]! > 0)) continue; const c = [Math.floor(p[o]! * scale[0]!), Math.floor(p[o + 1]! * scale[1]!), Math.floor(p[o + 2]! * scale[2]!)]; if (c[2] !== z) continue;
    const key = `${c[0]},${c[1]}`, m = mean.get(key) ?? [0, 0, 0]; m[0]++; m[1] += p[o + 4]!; m[2] += p[o + 5]!; mean.set(key, m); }
  for (let x = dumpX0; x < dumpX0 + 20; x++) {
    const bed = nbflipSlopeClearanceBed(x + 0.5), rows: string[] = [];
    for (let y = Math.floor(bed) - 1; y <= Math.floor(bed) + 2; y++) {
      const c = [x, y, z], s = slot(c); if (s < 0) { rows.push(`${y}:-`); continue; }
      const l = local(c), fl = phiW[fB + 520 * s + l]!, m = mean.get(`${x},${y}`);
      rows.push(`${y}:${fl & 1 ? "O" : "."}${fl & 2 ? "L" : "."} phi${phi[pB + 520 * s + l]!.toFixed(2)} v1${phi[L.base("phi", "F", "PHI1") + 520 * s + l]!.toFixed(2)} X${pres[xB + 520 * s + l]!.toPrecision(3)} D${pres[dB + 520 * s + l]!.toPrecision(3)} u${vel[uB + 520 * s + l]!.toFixed(3)} v${vel[vB + 520 * s + l]!.toFixed(3)}${m ? ` n${m[0]} pu${(m[1]! / m[0]!).toFixed(3)} pv${(m[2]! / m[0]!).toFixed(3)}` : ""}`);
    }
    console.log(`x${x} bed${bed.toFixed(2)} | ${rows.join(" | ")}`);
  }
}
let code = 0;
try {
  const solver = await method.createSolverAsync!(device, scene, profile.quality, values, undefined, () => {});
  await solver.waitForSimulationReady?.();
  const dt = scene.numerics.fixedDt_s!, gs = 9.81 / Math.sqrt(257), tx = 16 / Math.sqrt(257), ty = -1 / Math.sqrt(257);
  console.log(JSON.stringify({ engine, thickness, boundary: opt("boundary") ?? "terrain", gSinTheta: f(gs) }));
  for (let k = 1; k <= steps; k++) {
    if (!solver.advanceTo(k * dt + 1e-12, []) && (await solver.awaitFrameCompletion?.(), !solver.advanceTo(k * dt + 1e-12, []))) { code = 3; break; }
    if (k % every !== 0 && k !== steps) continue;
    await solver.assertSimulationHealthy?.(); await solver.awaitFrameCompletion?.();
    const src = solver.particleSource!, first = src.firstRecord ?? 0, stride = src.strideFloats;
    const count = src.liveCount ? new Uint32Array(await readBuffer(src.liveCount.buffer, src.liveCount.byteOffset, 4))[0]! : src.capacity;
    const scale = src.positionScale_m.map(s => s / h);
    const buf = await readBuffer(src.buffer, 4 * first * stride, 4 * count * stride), p = new Float32Array(buf), w = new Uint32Array(buf);
    // |velocity - pic| per record (pDelta's motion energy, the grid sample pMove left in before.xyz), where published.
    const energy = src.motionEnergy ? new Float32Array(await readBuffer(src.motionEnergy, 0, 4 * count)) : undefined, slip: number[] = [];
    if (k === dumpStep) await dumpSlice(solver, p, count, stride, scale);
    const along: number[][] = [[], []], pic: number[] = [], above: number[][] = [[], []];
    for (let i = 0; i < count; i++) {
      const o = stride * i, x = p[o]! * scale[0]!, y = p[o + 1]! * scale[1]!; if (!(p[o + 3]! > 0) || x >= 96) continue;
      const mode = w[o + 11]! >>> 31, t = p[o + 4]! * tx + p[o + 5]! * ty;
      along[mode]!.push(t); above[mode]!.push(y - nbflipSlopeClearanceBed(x));
      if (mode === 0) { pic.push(p[o + 8]! * tx + p[o + 9]! * ty); if (energy) slip.push(Math.sqrt(energy[i]!)); }
    }
    console.log(JSON.stringify({ k, t: f(k * dt), ideal: f(gs * k * dt),
      coupled: { n: along[0]!.length, along: f(median(along[0]!)), delta: f(median(pic)), slip: f(median(slip)), above: f(median(above[0]!)) },
      ballistic: { n: along[1]!.length, along: f(median(along[1]!)), above: f(median(above[1]!)) } }));
  }
  solver.destroy?.();
} catch (err) { console.log(JSON.stringify({ error: (err as Error).message.slice(0, 400) })); code = 3; }
if (gpuError) { console.log(JSON.stringify({ gpuError })); code = 3; }
process.exit(code);
