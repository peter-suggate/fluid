/**
 * nbflip-slope-clearance on NB-FLIP's brick engine (the app's createBrickSolver): the liquid and the particles left on
 * the slope (x < NBFLIP_RAMP_LENGTH cells) over the run, the pit's, the volume against t = 0, and the pools' peaks.
 * The volume is the presentation volume texture's sum (cells: each cell's six-tetrahedron fill); the particles are the
 * live bank's records (positions in cells from the lattice's corner).
 *
 *   tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-slope-clearance-brick.ts [--hz=60] [--steps=360]
 *     [--every=30] [--pools='{"f":..,"c":..,"b":16,"k":..,"particles":..}']
 *
 * One JSON line a report; exit 1 when the slope is not clear (volume or particles) at the end, 3 on a failure.
 */
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { getMethod } from "../lib/core/method-registry";
import "../lib/methods/index";
import { resolveMethodValues } from "../lib/core/method-contract";
import { NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_GRID, NBFLIP_RAMP_LENGTH } from "../lib/core/nbflip-ramp-scene";
import { bakeProceduralSolid } from "../lib/core/procedural-solid";
import { NBFLIP_METHOD_PROFILE } from "../lib/core/nbflip-paper-scenes";
import { createNbflipSlopeClearance, nbflipSlopeClearanceBed } from "../lib/core/nbflip-slope-clearance-scene";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { H } from "../lib/methods/narrow-band/brick-abi";

const args = process.argv.slice(2), opt = (k: string) => args.find(a => a.startsWith(`--${k}=`))?.slice(k.length + 3);
const hz = Number(opt("hz") ?? 60), every = Number(opt("every") ?? hz / 2);
const scene = createNbflipSlopeClearance(hz), steps = Number(opt("steps") ?? Math.round(scene.duration_s * hz));
const pools = opt("pools"); if (pools) scene.fluid.brickPools = JSON.parse(pools);
// --boundary=procedural: the same bed as the sampled procedural boundary (bakeProceduralSolid), not the heightfield.
if (opt("boundary") === "procedural") {
  delete scene.terrain;
  Object.assign(scene, bakeProceduralSolid(NBFLIP_RAMP_GRID, [NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_CELL_SIZE_M], ([x, y]) => y - nbflipSlopeClearanceBed(x)));
}
const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), profile = NBFLIP_METHOD_PROFILE;
const values = resolveMethodValues(method, profile.quality, { ...method.appDefaults, ...profile.overrides, engine: "brick",
  ...JSON.parse(opt("values") ?? "{}") });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
let gpuError = "";
device.addEventListener("uncapturederror", (e: GPUUncapturedErrorEvent) => { e.preventDefault(); gpuError ||= e.error.message.slice(0, 300); });

async function readBuffer(buffer: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); if (bytes) enc.copyBufferToBuffer(buffer, offset, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ); const out = read.getMappedRange().slice(0); read.unmap(); read.destroy(); return out;
}

let code = 0;
try {
  const solver = await createBrickSolver(device, scene, values as Record<string, unknown>, profile.quality);
  await solver.waitForSimulationReady();
  const e = solver.brickEngine, dt = e.options.dt_s;
  console.log(JSON.stringify({ scene: scene.sceneId, hz, steps, lattice: e.options.lattice, pools: e.options.capacities }));
  const measure = async (k: number) => {
    await e.fence();
    const tex = solver.brickPresentation.volume, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
    const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ); const f = new Float32Array(buf.getMappedRange());
    let ramp = 0, pit = 0, front = 0, back = Infinity;
    for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) { const o = (z * hgt + y) * bpr / 4; for (let x = 0; x < w; x++) {
      const v = f[o + x]!; if (x < NBFLIP_RAMP_LENGTH) ramp += v; else pit += v;
      if (v > 1e-3) { front = Math.max(front, x); back = Math.min(back, x); }
    } }
    buf.unmap(); buf.destroy();
    const h = await e.read("set", 0, 256), src = solver.particleSource, first = src.firstRecord ?? 0, bank = first / src.capacity;
    const count = h[H.PCOUNT0 + bank]!, stride = src.strideFloats;
    const parts = new Float32Array(await readBuffer(src.buffer, 4 * first * stride, 4 * count * stride));
    let rampP = 0, maxU = 0, upEdge = 0; const stray: number[][] = [];
    for (let i = 0; i < count; i++) {
      const x = parts[stride * i]!; if (!Number.isFinite(x)) continue;
      if (x < NBFLIP_RAMP_LENGTH) { rampP++; if (stray.length < 8) stray.push(Array.from(parts.subarray(stride * i, stride * (i + 1)), v => +v.toFixed(3))); } if (x < 8) upEdge++;
      maxU = Math.max(maxU, Math.hypot(parts[stride * i + 4]!, parts[stride * i + 5]!, parts[stride * i + 6]!));
    }
    return { k, t: +(k * dt).toFixed(3), ramp: +ramp.toFixed(2), pit: +pit.toFixed(2), total: +(ramp + pit).toFixed(2), back, front,
      particles: count, rampParticles: rampP, ...(rampP > 0 && rampP <= 8 ? { rampRecords: stray } : {}), upEdgeParticles: upEdge, maxParticleSpeed: +maxU.toFixed(3),
      liveF: h[H.LIVE_F], liveC: h[H.LIVE_C], peakF: h[H.PEAK_F], peakC: h[H.PEAK_C], peakP: h[H.PEAK_PARTICLES], fail: h[H.FAIL] };
  };
  // --columns=k:x,z;x,z..: at step k, φ (cells) at the column's vertices about the bed, and its particles' heights.
  const columnsAt = opt("columns"), columnStep = columnsAt ? Number(columnsAt.split(":")[0]) : -1;
  const columns = async (k: number) => {
    await e.fence();
    const tex = solver.brickPresentation.vertexPhi, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
    const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ); const phi = new Float32Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
    const h = await e.read("set", 0, 256), src = solver.particleSource, first = src.firstRecord ?? 0, bank = first / src.capacity, count = h[H.PCOUNT0 + bank]!, stride = src.strideFloats;
    const parts = new Float32Array(await readBuffer(src.buffer, 4 * first * stride, 4 * count * stride));
    for (const xz of columnsAt!.split(":")[1]!.split(";")) {
      const [x, z] = xz.split(",").map(Number) as [number, number], bed = nbflipSlopeClearanceBed(x);
      const ys: number[] = [];
      for (let i = 0; i < count; i++) { const px = parts[stride * i]!, pz = parts[stride * i + 2]!; if (px >= x && px < x + 1 && pz >= z && pz < z + 1) ys.push(+parts[stride * i + 1]!.toFixed(2)); }
      const y0 = Math.max(0, Math.floor(bed) - 2), col: Record<number, number> = {};
      for (let y = y0; y < Math.min(hgt, y0 + 7); y++) col[y] = +(phi[(z * hgt + y) * bpr / 4 + x]! / e.options.h_m).toFixed(3);
      console.log(JSON.stringify({ k, x, z, bed: +bed.toFixed(3), phiAtVertices: col, particleY: ys.sort((a, b) => a - b) }));
    }
  };
  const m0 = await measure(0), v0 = m0.total; console.log(JSON.stringify(m0));
  let last = m0, refused: number | undefined, error: string | undefined;
  for (let k = 1; k <= steps; k++) {
    if (!solver.advanceTo(k * dt + 1e-12, [])) { refused = k; break; }
    try { await solver.assertSimulationHealthy(); } catch (err) { error = (err as Error).message.slice(0, 400); last = await measure(k); break; }
    if (k === columnStep) await columns(k);
    if (k % every === 0 || k === steps) { last = await measure(k); console.log(JSON.stringify({ ...last, drift: +((last.total - v0) / v0).toFixed(4) })); }
  }
  const clear = last.ramp < 1e-3 && last.rampParticles === 0;
  console.log(JSON.stringify({ result: { clear, refused, error, gpuError: gpuError || undefined, end: last, drift: +((last.total - v0) / v0).toFixed(4) } }));
  code = error || refused !== undefined || gpuError ? 3 : clear ? 0 : 1;
  solver.destroy();
} catch (err) { console.log(JSON.stringify({ error: (err as Error).message.slice(0, 400) })); code = 3; }
process.exit(code);
