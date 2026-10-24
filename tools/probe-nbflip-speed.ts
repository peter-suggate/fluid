/** Speed-bound probe for Narrow-band FLIP: advances a scene headless and, every
 * step, reads the dynamic census's own words back after the frame: the largest
 * tile travel the residency certificate is built from (census word 18, cells a
 * step), the closure it gives (ceil(travel/4)+5 tiles, capped by the page
 * pool), and the tiles that hold the largest face-velocity bounds, each with
 * its coordinate, its width as the census saw it, its wet and near bits and
 * its signed bounds per axis in cells a step. Simulation only. Browser GPU off.
 *
 * WEBGPU_NODE_MODULE=<repo>/node_modules/webgpu/index.js node --import tsx tools/probe-nbflip-speed.ts <scene> <steps> [--dt=0.016666666666666666] [--top=6] [--variant=<json>] [--stages=<step,step,...>] [--capture=<stage,...>] [--over=<cells a step>] [--cells=8] [--trace=<stage>] [--sinks] [--slab=x0,x1,y0,y1,z] [--slab-stages=<stage,...>]
 *
 * The census of frame N classifies the extension frame N-1's tail left, on
 * the layout frame N starts from: a row `step N` describes that state. The
 * row also carries the root's and the h band's cycle counts and residuals and
 * the liquid volume in cells. --variant gives method values over the app's
 * defaults.
 * --stages names steps whose velocity fields are also captured as each
 * velocity stage of the frame leaves them: head.* (the extension the head
 * reads), remap.*, move.extension, momentum.*, p2g.* (the particle transfer),
 * forces.* (forces.scratch is the velocity the pressure solves start from),
 * toSimulation.velocity (the root's projection in simulation ownership),
 * band.velocity (the h band), extended.* (the tail extension), g2p.* and
 * tail.*; .velocity and .scratch are the solver's two velocity fields.
 * --capture keeps the named stages alone. Per stage: the largest |component|
 * in cells a step and its cell, and, with --over, the count of texels over
 * that speed and the first --cells of them (cell, component, value, tile
 * width, cell volume, the eight vertex phi about the cell in cells).
 * --trace follows the named stage's largest face through every captured
 * stage: the two cells it joins (centre phi, volume, their neighbours') and,
 * per stage, the face and both cells' six faces with their divergence. It
 * holds every captured field of the step at once: use it with --capture.
 * --sinks adds, per captured stage, the count and the summed inflow of the
 * cells of h tiles whose six faces take in more than one cell volume a step
 * (divergence under -1), the part of it into air cells and into enclosed
 * one-cell voids (an air cell with liquid or a wall on all six sides, by the
 * end of the step's vertex phi), and the largest of them.
 * --slab prints a window of one z plane: the end of the step's vertex phi and
 * cell volume, and for --slab-stages (default forces.scratch,
 * toSimulation.velocity, band.velocity) the three face velocities and the
 * six-face divergence of every cell. A texel at cell c, component a, is the
 * velocity on the cell's upper a face. --slab-z=z,z,... prints several z
 * planes of the same window.
 * --lock=<file> builds the solver outside the GPU lock and takes it for the
 * steps (tools/gpu-hold.ts). */
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { createFastJetScene, FAST_JET_METHOD_OVERRIDES } from "./fast-jet-scene";
import { takeGpuHold } from "./gpu-hold";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformTileCoord, uniformTileKey, type UniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readMixedTexture, readMixedTileWords } from "../tests/helpers/uniform-mixed-native-fields";

const method = uniformNarrowBandMethod;
const [sceneId, stepsText] = process.argv.slice(2).filter(a => !a.startsWith("--"));
const steps = Number(stepsText);
assert.ok(sceneId && Number.isInteger(steps) && steps >= 1, "usage: probe-nbflip-speed.ts <scene> <steps> [--dt=] [--top=] [--variant=] [--stages=] [--capture=] [--over=] [--cells=] [--trace=] [--sinks] [--slab=] [--slab-stages=]");
const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.split("=")[1];
const dt = Number(option("dt") ?? 1 / 60), top = Number(option("top") ?? 6), over = Number(option("over") ?? 0), cells = Number(option("cells") ?? 8);
// With --stages, a window of raw face texels on one z plane as chosen stages leave it: --slab=x0,x1,y0,y1,z [--slab-stages=forces.scratch,toSimulation.velocity,band.velocity].
const slab = option("slab")?.split(",").map(Number), slabZ = option("slab-z")?.split(",").map(Number), slabStages = new Set((option("slab-stages") ?? "forces.scratch,toSimulation.velocity,band.velocity").split(","));
const stageSteps = new Set((option("stages") ?? "").split(",").filter(Boolean).map(Number));
// --capture=<stage,...> keeps the named stages alone (every stage when absent).
const captured = option("capture") ? new Set(option("capture")!.split(",")) : undefined;
// --trace=<stage> follows that stage's largest texel through every captured stage of the step.
const trace = option("trace");
// --sinks adds, per captured stage, the h cells whose six faces take in more than one cell volume a step.
const sinks = process.argv.includes("--sinks");
const variant = JSON.parse(option("variant") ?? "{}") as Record<string, string | number>;
// fast-jet:<cell mm> is the fast-front fixture (tools/fast-jet-scene.ts), with reservoir-seep's profile under --variant.
const fastJet = sceneId.startsWith("fast-jet:") ? createFastJetScene({ cellSize_m: Number(sceneId.slice(9)) / 1000 }) : undefined;
if (fastJet) Object.assign(variant, { ...FAST_JET_METHOD_OVERRIDES, ...variant });
const scene = fastJet ? fastJet.scene : structuredClone(sceneDocument(getSceneDefinition(sceneId)));
scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
/** Census header words and the margin of the residency radius (uniform-mixed-dynamic.ts: HEADER, RESIDENCY_MARGIN_TILES). */
const HEADER = 20, MARGIN = 5;
const orderValue = (() => { const u = new Uint32Array(1), f = new Float32Array(u.buffer); return (k: number) => { u[0] = (k & 0x80000000) ? k & 0x7fffffff : ~k; return f[0]!; }; })();

let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
  device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  const started = performance.now();
  solver = await method.createSolverAsync!(device, scene, "balanced", { ...method.appDefaults, timeStep: "scene", ...variant }, undefined, () => {}) as WebGPUUniformReferenceSolver;
  // The lattice is the container rounded up to whole pages: h is the voxel
  // domain's cell, not the container over nx.
  const { nx, ny, nz } = solver.info, H = scene.voxelDomain?.finestCellSize_m ?? scene.container.width_m / nx, cell = dt / H;
  const release = option("lock") ? await takeGpuHold(option("lock")!) : () => {};
  const address = (solver as unknown as { address: UniformLatticeAddress }).address;
  const census = (solver as unknown as { mixedDynamic: { work: GPUBuffer; words: number; wetOffset: number; nearOffset: number } }).mixedDynamic;
  const frame = (solver as unknown as { mixedFrame: { pagePool: UniformPagePool; ownership: { capacity: { tiles: number } } } }).mixedFrame;
  const tiles = frame.ownership.capacity.tiles, pool = frame.pagePool, l = pool.layout;
  console.log(`${sceneId}: ${nx}x${ny}x${nz}, h ${H}, dt ${dt}, ${cell.toFixed(4)} cells a step per m/s, created in ${((performance.now() - started) / 1000).toFixed(1)} s, pool of ${address.blocks} blocks, ${tiles} tile keys, closure cap ${l.closureCap}`);
  const readBuffer = async (source: GPUBuffer, bytes: number) => {
    const read = device!.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try { const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, read, 0, bytes); device!.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange()).slice(); }
    finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
  };
  const status = new Uint32Array(S.words), statusRead = device.createBuffer({ size: 4 * S.words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const coord: [number, number, number] = [0, 0, 0];

  // Velocity stages of a step: each capture is the logical field as the stage leaves it.
  type Capture = { name: string; done: () => Promise<Float32Array> };
  let captures: Capture[] | undefined;
  {
    const g = (frame as unknown as { fields: { velocity: GPUTexture; velocityScratch: GPUTexture }; momentum: Record<string, (...a: unknown[]) => unknown>; forces: Record<string, (...a: unknown[]) => unknown>;
      split: { transfer: Record<string, (...a: unknown[]) => unknown> }; band: Record<string, (...a: unknown[]) => unknown>; narrowBandFlip: Record<string, (...a: unknown[]) => unknown>;
      relayout?: Record<string, (...a: unknown[]) => unknown>; remap: Record<string, (...a: unknown[]) => unknown>; plan: Record<string, (...a: unknown[]) => unknown> });
    const take = (encoder: GPUCommandEncoder, name: string, texture: GPUTexture) => {
      if (!captures || (captured && !captured.has(name))) return;
      const read = uniformDetailField(texture)!.storage.capture(encoder, texture); let field: Promise<Float32Array> | undefined;
      const entry: Capture = { name, done: () => field ??= read() };
      captures.push(entry);
    };
    const hook = (object: Record<string, (...a: unknown[]) => unknown> | undefined, key: string, name: string, texture: () => GPUTexture, before = false) => {
      if (!object?.[key]) { console.log(`no stage ${name}: ${key} is absent`); return; }
      const original = object[key]!.bind(object);
      object[key] = (...a: unknown[]) => { if (before) take(a[0] as GPUCommandEncoder, name, texture()); const result = original(...a); if (!before) take(a[0] as GPUCommandEncoder, name, texture()); return result; };
    };
    const velocity = () => g.fields.velocity, scratch = () => g.fields.velocityScratch;
    hook(g.relayout, "encode", "head.extension", scratch, true); hook(g.relayout, "encode", "head.velocity", velocity, true);
    hook(g.remap, "applyGpu", "remap.extension", scratch); hook(g.remap, "applyGpu", "remap.velocity", velocity);
    hook(g.narrowBandFlip, "move", "move.extension", scratch, true);
    hook(g.momentum, "encode", "momentum.velocity", velocity); hook(g.momentum, "encode", "momentum.scratch", scratch);
    hook(g.narrowBandFlip, "transfer", "p2g.velocity", velocity); hook(g.narrowBandFlip, "transfer", "p2g.scratch", scratch);
    hook(g.forces, "encode", "forces.velocity", velocity); hook(g.forces, "encode", "forces.scratch", scratch);
    hook(g.split.transfer, "encodeToSimulation", "toSimulation.velocity", velocity);
    hook(g.band, "encodeSolve", "band.velocity", velocity);
    hook(g.narrowBandFlip, "update", "extended.velocity", velocity, true); hook(g.narrowBandFlip, "update", "extended.scratch", scratch, true);
    hook(g.narrowBandFlip, "update", "g2p.velocity", velocity); hook(g.narrowBandFlip, "update", "g2p.scratch", scratch);
    hook(g.plan, "encodeTail", "tail.velocity", velocity); hook(g.plan, "encodeTail", "tail.scratch", scratch);
  }

  let before = await readMixedTileWords(device, solver);
  for (let step = 1; step <= steps; step++) {
    let thrown: string | undefined;
    captures = stageSteps.has(step) ? [] : undefined;
    // The census records each tile's band reason only while layout views are on.
    if (captures) (solver as unknown as { mixedFrame: { setLayoutViews(on: boolean): void } }).mixedFrame.setLayoutViews(true);
    try { solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion(); } catch (e) { const cause = (e as Error).cause; thrown = `${(e as Error).message}${cause instanceof Error ? `: ${cause.message}` : ""}`; }
    const taken = captures ?? []; captures = undefined;
    if (stageSteps.has(step)) (solver as unknown as { mixedFrame: { setLayoutViews(on: boolean): void } }).mixedFrame.setLayoutViews(false);
    { const encoder = device.createCommandEncoder(); pool.encodeReceipt(encoder, statusRead, 0); device.queue.submit([encoder.finish()]); await statusRead.mapAsync(GPUMapMode.READ); status.set(new Uint32Array(statusRead.getMappedRange())); statusRead.unmap(); }
    const fatal = uniformPagePoolFatalMessage(l, status), failure = thrown ?? solver.info.simulationPipelineError;
    const words = await readBuffer(census.work, census.work.size), floats = new Float32Array(words.buffer);
    const travel = floats[18]!, bound = (t: number, k: number) => orderValue(words[HEADER + census.words + 6 * t + k]!);
    const bit = (offset: number, t: number) => (words[offset + (t >>> 5)]! >>> (t & 31)) & 1;
    // A tile's largest bound over its six keys, in cells a step; a key never written (low all ones, high zero) bounds nothing.
    const speed = (t: number) => { let s = 0; for (let k = 0; k < 6; k++) { const key = words[HEADER + census.words + 6 * t + k]!; if (key === (k < 3 ? 0xffffffff : 0)) continue; s = Math.max(s, Math.abs(bound(t, k)) * cell); } return s; };
    const live: number[] = []; for (const b of address.seed.resident) for (let i = 0; i < 512; i++) live.push(512 * b + i);
    const ranked = live.map(t => ({ t, s: speed(t) })).sort((a, b) => b.s - a.s);
    let wetMax = 0, airMax = 0; for (const { t, s } of ranked) { if (bit(census.wetOffset, t)) wetMax = Math.max(wetMax, s); else airMax = Math.max(airMax, s); }
    const histogram = [4, 8, 12, 16, 24, 32, 44].map(limit => `>${limit}:${ranked.filter(r => r.s > limit).length}`).join(" ");
    const stats = failure ? undefined : await solver.readStats();
    console.log(`step ${step} travel ${travel.toFixed(3)} cells a step, closure ${Math.ceil(travel / 4) + MARGIN} tiles; tiles' largest bound wet ${wetMax.toFixed(3)} air ${airMax.toFixed(3)}; tiles over ${histogram}; h tiles ${solver.info.uniformMixedFineTiles ?? 0} particles ${solver.narrowBandFlipInfo?.particles ?? 0} root ${solver.info.uniformPressureCyclesExecuted ?? "?"} cycles residual ${(solver.info.uniformPressureAcceptedResidual ?? NaN).toExponential(2)} band ${solver.info.uniformPressureBandTiles ?? "?"} tiles ${solver.info.uniformPressureBandCycles ?? "?"} cycles residual ${(solver.info.uniformPressureBandResidual ?? NaN).toExponential(2)} liquid ${Math.round(stats?.volumeCellSum ?? NaN)} drift ${(stats?.volumeDrift ?? NaN).toFixed(4)} fatal ${fatal ?? "none"}`);
    for (const { t, s } of ranked.slice(0, top)) {
      uniformTileCoord(address, t, coord);
      const width = (before[t]! & 0x80000000) ? "h" : "4h";
      console.log(`  tile (${coord.join(",")}) cells ${coord.map(c => 4 * c).join(",")} page (${coord.map(c => c >> 3).join(",")}) width ${width} wet ${bit(census.wetOffset, t)} near ${bit(census.nearOffset, t)} speed ${s.toFixed(3)} low ${[0, 1, 2].map(k => (bound(t, k) * cell).toFixed(2)).join(" ")} high ${[3, 4, 5].map(k => (bound(t, k) * cell).toFixed(2)).join(" ")}`);
    }
    const after = failure ? before : await readMixedTileWords(device, solver);
    if (taken.length) {
      const [phi, volume] = failure ? [undefined, undefined] : await Promise.all([readMixedTexture(device, solver.vertexPhiTexture!), readMixedTexture(device, solver.volumeTexture)]);
      if (slab && phi && volume) {
        const [x0, x1, y0, y1] = slab as [number, number, number, number, number];
        for (const Z of slabZ ?? [slab[4]!]) {
        console.log(`   slab step ${step} (end of the step): vertex phi in cells, z ${Z}, vertices x = ${x0}..${x1 + 1}`);
        for (let y = y1 + 1; y >= y0; y--) console.log(`    y ${String(y).padStart(3)} ${Array.from({ length: x1 - x0 + 2 }, (_, k) => (phi[x0 + k + (nx + 1) * (y + (ny + 1) * Z)]! / H).toFixed(2).padStart(7)).join(" ")}`);
        console.log(`   slab step ${step} (end of the step): cell volume, the tile's width before and after the step (h or C) and its census band reason (UNIFORM_STAGE_REASON), z ${Z}`);
        const width = (w: Uint32Array, x: number, y: number) => (w[uniformTileKey(address, x >> 2, y >> 2, Z >> 2)]! & 0x80000000) ? "h" : "C";
        for (let y = y1; y >= y0; y--) console.log(`    y ${String(y).padStart(3)} ${Array.from({ length: x1 - x0 + 1 }, (_, k) => `${volume[x0 + k + nx * (y + ny * Z)]!.toFixed(2)}${width(before, x0 + k, y)}${width(after, x0 + k, y)}${words[census.wetOffset + 2 * census.words + uniformTileKey(address, (x0 + k) >> 2, y >> 2, Z >> 2)]}`.padStart(9)).join(" ")}`);
        }
      }
      if (trace && taken.some(c => c.name === trace)) {
        // The traced stage's largest face, then that face and the six faces of the two cells it joins in every captured stage.
        const fields: Float32Array[] = []; for (const capture of taken) fields.push(await capture.done());
        const traced = fields[taken.findIndex(c => c.name === trace)]!;
        let best = 0, at = 0; for (let i = 0; i < nx * ny * nz; i++) for (let a = 0; a < 3; a++) { const m = Math.abs(traced[4 * i + a]!); if (!(m <= best)) { best = m; at = 4 * i + a; } }
        const i = at >> 2, a = at & 3, low = [i % nx, Math.floor(i / nx) % ny, Math.floor(i / (nx * ny))], high = low.map((v, k) => v + (k === a ? 1 : 0));
        const inside = (c: number[]) => c[0]! < nx && c[1]! < ny && c[2]! < nz;
        const centre = (c: number[]) => { let sum = 0; for (let k = 0; k < 8; k++) sum += phi![c[0]! + (k & 1) + (nx + 1) * (c[1]! + ((k >> 1) & 1) + (ny + 1) * (c[2]! + (k >> 2)))]!; return sum / (8 * H); };
        const describe = (c: number[]) => !inside(c) ? `(${c.join(",")}) is past the box` : `(${c.join(",")}) ${(after[uniformTileKey(address, c[0]! >> 2, c[1]! >> 2, c[2]! >> 2)]! & 0x80000000) ? "h" : "4h"}${phi && volume ? ` centre phi ${centre(c).toFixed(3)} cells, V ${volume[c[0]! + nx * (c[1]! + ny * c[2]!)]!.toFixed(2)}` : ""}`;
        console.log(`   trace ${trace} step ${step}: face (${low.join(",")}).${"xyz"[a]} joins cell ${describe(low)} and cell ${describe(high)}; phi and V are the end of the step's`);
        for (const [name, c] of [["low", low], ["high", high]] as const) {
          if (!inside(c) || !phi || !volume) continue;
          console.log(`    ${name} cell's neighbours -x +x -y +y -z +z: ${[0, 1, 2].flatMap(k => [-1, 1].map(d => { const q = c.map((v, j) => v + (j === k ? d : 0)); return q[k]! < 0 || !inside(q) ? "wall" : `${centre(q).toFixed(2)}/${volume[q[0]! + nx * (q[1]! + ny * q[2]!)]!.toFixed(2)}`; })).join(" ")} (centre phi/V)`);
        }
        taken.forEach((capture, k) => {
          const field = fields[k]!, value = (x: number, y: number, z: number, c: number) => x < 0 || y < 0 || z < 0 ? 0 : field[4 * (x + nx * (y + ny * z)) + c]! * cell;
          const faces = (c: number[]) => { if (!inside(c)) return "past the box"; const f = [0, 1, 2].flatMap(k => [value(c[0]! - (k === 0 ? 1 : 0), c[1]! - (k === 1 ? 1 : 0), c[2]! - (k === 2 ? 1 : 0), k), value(c[0]!, c[1]!, c[2]!, k)]); return `${f.map(v => v.toFixed(2)).join(" ")} divergence ${(f[1]! - f[0]! + f[3]! - f[2]! + f[5]! - f[4]!).toFixed(2)}`; };
          console.log(`    ${capture.name.padEnd(22)} face ${(field[at]! * cell).toFixed(3)} | low cell faces -x +x -y +y -z +z ${faces(low)} | high cell ${faces(high)}`);
        });
      }
      for (const capture of taken) {
        const field = await capture.done();
        // Stages up to the remap are on the layout the frame starts from.
        const layout = capture.name.startsWith("head.") ? before : after;
        let best = 0, at = 0, count = 0; const listed: string[] = [];
        for (let i = 0; i < nx * ny * nz; i++) for (let a = 0; a < 3; a++) {
          const v = field[4 * i + a]! * cell, m = Math.abs(v);
          if (!(m <= best)) { best = m; at = 4 * i + a; }
          if (over > 0 && !(m <= over)) {
            count++;
            if (listed.length < cells) {
              const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny)), key = uniformTileKey(address, x >> 2, y >> 2, z >> 2);
              const corners = phi ? Array.from({ length: 8 }, (_, k) => (phi[x + (k & 1) + (nx + 1) * (y + ((k >> 1) & 1) + (ny + 1) * (z + (k >> 2)))]! / H).toFixed(1)).join(" ") : "";
              listed.push(`(${x},${y},${z}).${"xyz"[a]} ${v.toFixed(2)} ${(layout[key]! & 0x80000000) ? "h" : "4h"} V ${volume ? volume[i]!.toFixed(2) : "?"} phi ${corners}`);
            }
          }
        }
        const i = at >> 2, x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny)), key = uniformTileKey(address, x >> 2, y >> 2, z >> 2);
        console.log(`  stage ${capture.name.padEnd(22)} max ${(field[at]! * cell).toFixed(3)} at (${x},${y},${z}).${"xyz"[at & 3]} tile (${x >> 2},${y >> 2},${z >> 2}) ${(layout[key]! & 0x80000000) ? "h" : "4h"}${over > 0 ? ` over ${over}: ${count}${listed.length ? ` | ${listed.join(" | ")}` : ""}` : ""}`);
        if (sinks) {
          // A cell cannot take in more than its own volume: what converges on it beyond that leaves the liquid.
          // An air cell (vertex phi's mean at the centre not under zero) with liquid or a wall on all six sides is an enclosed one-cell void.
          const mean = (x: number, y: number, z: number) => { let sum = 0; for (let k = 0; k < 8; k++) sum += phi![x + (k & 1) + (nx + 1) * (y + ((k >> 1) & 1) + (ny + 1) * (z + (k >> 2)))]!; return sum / 8; };
          const wet = (x: number, y: number, z: number) => x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz || mean(x, y, z) < 0;
          let n = 0, total = 0, worst = 0, where = 0, air = 0, airTotal = 0, voids = 0, voidTotal = 0;
          for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
            if (!(layout[uniformTileKey(address, x >> 2, y >> 2, z >> 2)]! & 0x80000000)) continue;
            const i = x + nx * (y + ny * z), d = (field[4 * i]! - (x ? field[4 * (i - 1)]! : 0) + field[4 * i + 1]! - (y ? field[4 * (i - nx) + 1]! : 0) + field[4 * i + 2]! - (z ? field[4 * (i - nx * ny) + 2]! : 0)) * cell;
            if (!(d < -1)) continue;
            n++; total -= d; if (d < worst) { worst = d; where = i; }
            if (!phi || mean(x, y, z) < 0) continue;
            air++; airTotal -= d;
            if (wet(x - 1, y, z) && wet(x + 1, y, z) && wet(x, y - 1, z) && wet(x, y + 1, z) && wet(x, y, z - 1) && wet(x, y, z + 1)) { voids++; voidTotal -= d; }
          }
          console.log(`   sinks ${capture.name}: ${n} h cells with divergence under -1, ${total.toFixed(1)} cell volumes a step in all${phi ? `; ${air} of them air cells taking ${airTotal.toFixed(1)}, ${voids} of those enclosed one-cell voids taking ${voidTotal.toFixed(1)}` : ""}${n ? `, the largest ${worst.toFixed(2)} at (${where % nx},${Math.floor(where / nx) % ny},${Math.floor(where / (nx * ny))}) V ${volume ? volume[where]!.toFixed(2) : "?"}` : ""}`);
        }
        if (slab && slabStages.has(capture.name)) {
          const [x0, x1, y0, y1] = slab as [number, number, number, number, number];
          for (const Z of slabZ ?? [slab[4]!]) {
          const at = (x: number, y: number, z: number, c: number) => x < 0 || y < 0 || z < 0 ? 0 : field[4 * (x + nx * (y + ny * z)) + c]! * cell;
          for (const c of [0, 1, 2]) {
            console.log(`   slab ${capture.name} step ${step}: velocity ${"xyz"[c]} on each cell's upper face (cells a step), z ${Z}, columns x = ${x0}..${x1}`);
            for (let y = y1; y >= y0; y--) console.log(`    y ${String(y).padStart(3)} ${Array.from({ length: x1 - x0 + 1 }, (_, k) => at(x0 + k, y, Z, c).toFixed(2).padStart(7)).join(" ")}`);
          }
          console.log(`   slab ${capture.name} step ${step}: divergence of the cell's six faces (cells a step), z ${Z}`);
          for (let y = y1; y >= y0; y--) console.log(`    y ${String(y).padStart(3)} ${Array.from({ length: x1 - x0 + 1 }, (_, k) => { const x = x0 + k; return (at(x, y, Z, 0) - at(x - 1, y, Z, 0) + at(x, y, Z, 1) - at(x, y - 1, Z, 1) + at(x, y, Z, 2) - at(x, y, Z - 1, 2)).toFixed(2).padStart(7); }).join(" ")}`);
          }
        }
      }
    }
    before = after;
    if (failure || fatal) { if (failure) console.log(`solver: ${failure}`); process.exitCode = 1; break; }
  }
  statusRead.destroy();
  console.log(`errors ${JSON.stringify(errors)}`);
  release();
} finally { solver?.destroy(); await device?.queue.onSubmittedWorkDone(); device?.destroy(); }
