/**
 * nbflip-water-slide's drain after the jet: either engine (--engine=brick | uniform) through the method's
 * createSolverAsync, and every --every steps the particles on the slide (y at or above the hood's floor) and in the pit,
 * their position along the slide (the turn angle a, 0 at the source, NBFLIP_SLIDE_END_RAD at the stop wall), and their
 * velocity split on the centreline's downhill tangent: along (+ downhill), across (radial) and vertical. Water held on
 * the slide shows as particles whose along speed stays near zero while the floor under them falls.
 *
 *   tools/gpu/fluid-gpu-run.sh check node --import tsx tools/probe-slide-drain.ts [--engine=brick] [--steps=900]
 *     [--every=60] [--from=420] [--bins=8]
 *
 * One JSON line a report.
 */
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { getMethod } from "../lib/core/method-registry";
import "../lib/methods/index";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { NBFLIP_SLIDE_CELL_SIZE_M, NBFLIP_SLIDE_END_RAD, nbflipSlideAngleAt, nbflipSlideFloorAt } from "../lib/core/nbflip-water-slide-scene";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const args = process.argv.slice(2), opt = (k: string) => args.find(a => a.startsWith(`--${k}=`))?.slice(k.length + 3);
const engine = opt("engine") ?? "brick", id = opt("scene") ?? "nbflip-water-slide";
const steps = Number(opt("steps") ?? 900), every = Number(opt("every") ?? 60), from = Number(opt("from") ?? 420), bins = Number(opt("bins") ?? 8);
const HOOD_FLOOR = 32, SLIDE_AXIS = [76, 84.5] as const, sectionFrom = Number(opt("sectionFrom") ?? 900);
const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = getSceneDefinition(id), scene = sceneDocument(definition);
const pools = opt("pools"); if (pools) scene.fluid.brickPools = JSON.parse(pools);
const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine, ...JSON.parse(opt("values") ?? "{}") });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const raw = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const device = managedGPUDevice(raw, { requireWorkerRealm: false });
let gpuError = "";
device.addEventListener("uncapturederror", (e: GPUUncapturedErrorEvent) => { e.preventDefault(); gpuError ||= e.error.message.slice(0, 300); });

async function readBuffer(buffer: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); if (bytes) enc.copyBufferToBuffer(buffer, offset, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ); const out = read.getMappedRange().slice(0); read.unmap(); read.destroy(); return out;
}

let code = 0;
try {
  const solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
  await solver.waitForSimulationReady?.();
  const dt = scene.numerics.fixedDt_s!, h = NBFLIP_SLIDE_CELL_SIZE_M;
  console.log(JSON.stringify({ scene: id, engine, info: (solver.info as unknown as Record<string, unknown>).engine, steps, dt }));
  const report = async (k: number) => {
    await solver.awaitFrameCompletion?.();
    const src = solver.particleSource;
    if (!src) throw new Error("the solver publishes no particle source");
    const first = src.firstRecord ?? 0, stride = src.strideFloats;
    const count = src.liveCount ? new Uint32Array(await readBuffer(src.liveCount.buffer, src.liveCount.byteOffset, 4))[0]! : src.capacity;
    const parts = new Float32Array(await readBuffer(src.buffer, 4 * first * stride, 4 * count * stride));
    const scale = src.positionScale_m.map(s => s / h);
    const bin = Array.from({ length: bins }, () => ({ n: 0, along: 0, across2: 0, vert2: 0, slow: 0 }));
    const film = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
    const section = Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => [0, 0, 0]));
    let slide = 0, pit = 0, along = 0, across2 = 0, vert2 = 0, slow = 0, uphill = 0, maxSpeed = 0;
    for (let i = 0; i < count; i++) {
      const o = stride * i; if (!(parts[o + 3]! > 0) || !Number.isFinite(parts[o]!)) continue;
      const x = parts[o]! * scale[0]!, y = parts[o + 1]! * scale[1]!, z = parts[o + 2]! * scale[2]!;
      const u = parts[o + 4]!, v = parts[o + 5]!, w = parts[o + 6]!;
      maxSpeed = Math.max(maxSpeed, Math.hypot(u, v, w));
      if (y < HOOD_FLOOR) { pit++; continue; }
      const a = nbflipSlideAngleAt(x, z);
      // The centreline's tangent at bearing a (x = A + r sin a, z = B - r cos a): (cos a, sin a); its radial: (sin a, -cos a).
      const t = u * Math.cos(a) + w * Math.sin(a), r = u * Math.sin(a) - w * Math.cos(a);
      // Where on the section: height over the continuous floor, and offset from the centreline (+ outward; the
      // channel is 14 wide, its walls 7..11 out).
      const [fl, rad] = nbflipSlideFloorAt(a), above = y - fl, off = Math.hypot(x - SLIDE_AXIS[0], z - SLIDE_AXIS[1]) - rad;
      const hc = above < 0 ? 0 : above < 1 ? 1 : above < 2 ? 2 : above < 4 ? 3 : above < 8 ? 4 : 5;
      const oc = off < -7 ? 0 : off < -4 ? 1 : off <= 4 ? 2 : off <= 7 ? 3 : 4;
      if (k >= sectionFrom) { const cl = section[hc]![oc]!; cl[0]++; cl[1] += t; cl[2] += Math.hypot(r, v); }
      if (k >= sectionFrom && above < 1) {
        // The film: coupled (mode 0) or ballistic (bit 31 of word 11), the grid velocity it sampled (pic, words 8..10;
        // a ballistic record keeps 0) along the tangent, and how many sampled exactly zero.
        const ballisticRec = (new Uint32Array(parts.buffer, 4 * (o + 11), 1)[0]! >>> 31) === 1;
        const pu = parts[o + 8]!, pw = parts[o + 10]!, pt = pu * Math.cos(a) + pw * Math.sin(a), zero = pu === 0 && parts[o + 9] === 0 && pw === 0;
        const fm = film[ballisticRec ? 1 : 0]!; fm[0]++; fm[1] += t; fm[2] += pt; fm[3] += +zero; fm[4] += parts[o + 7]!;
      }
      slide++; along += t; across2 += r * r; vert2 += v * v; if (Math.abs(t) < 0.1) slow++; if (t < -0.05) uphill++;
      const b = bin[Math.min(bins - 1, Math.max(0, Math.floor(bins * a / NBFLIP_SLIDE_END_RAD)))]!;
      b.n++; b.along += t; b.across2 += r * r; b.vert2 += v * v; if (Math.abs(t) < 0.1) b.slow++;
    }
    const f = (x: number) => +x.toFixed(3);
    // The brick engine's liquid volume (litres): its presentation volume texture's sum (each cell's fill), slide and pit.
    let volume: { slideL: number; pitL: number } | undefined;
    const tex = (solver as unknown as { brickPresentation?: { volume: GPUTexture } }).brickPresentation?.volume;
    if (tex) {
      const [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
      const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ); const v = new Float32Array(buf.getMappedRange()); let up = 0, down = 0;
      for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) { const o = (z * hgt + y) * bpr / 4; for (let x = 0; x < w; x++) { if (y < HOOD_FLOOR) down += v[o + x]!; else up += v[o + x]!; } }
      buf.unmap(); buf.destroy(); const L = 1000 * h ** 3; volume = { slideL: f(up * L), pitL: f(down * L) };
    }
    console.log(JSON.stringify({ k, t: f(k * dt), particles: count, slide, pit, ...volume,
      slideAlong: f(along / Math.max(1, slide)), slideAcrossRms: f(Math.sqrt(across2 / Math.max(1, slide))), slideVertRms: f(Math.sqrt(vert2 / Math.max(1, slide))),
      slow, uphill, maxSpeed: f(maxSpeed),
      ...(k >= sectionFrom ? { sectionRows: "above floor <0,<1,<2,<4,<8,>=8; columns offset <-7,<-4,|o|<=4,<=7,>7; [n, along, |across,vert|]",
        film: film.map(([n, a1, p1, z0, d]) => ({ n, along: f(a1! / Math.max(1, n!)), picAlong: f(p1! / Math.max(1, n!)), picZero: z0, depth: f(d! / Math.max(1, n!)) })),
        section: section.map(row => row.map(([n, a2, c2]) => n ? [n, f(a2! / n), f(c2! / n)] : [0])) } : {}),
      bins: bin.map(b => b.n ? [b.n, f(b.along / b.n), f(Math.sqrt(b.across2 / b.n)), f(Math.sqrt(b.vert2 / b.n)), b.slow] : [0]) }));
  };
  for (let k = 1; k <= steps; k++) {
    // The old engine splits a frame over submissions: it refuses an advance until the last frame completes.
    if (!solver.advanceTo(k * dt + 1e-12, []) && (await solver.awaitFrameCompletion?.(), !solver.advanceTo(k * dt + 1e-12, []))) { console.log(JSON.stringify({ refused: k })); code = 3; break; }
    if (k % every === 0 && k >= from || k === steps) { await solver.assertSimulationHealthy?.(); await report(k); }
  }
  solver.destroy?.();
} catch (err) { console.log(JSON.stringify({ error: (err as Error).message.slice(0, 500) })); code = 3; }
if (gpuError) { console.log(JSON.stringify({ gpuError })); code = 3; }
process.exit(code);
