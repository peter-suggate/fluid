/** Particles on a calm surface: how many NB-FLIP samples a gently moving
 * surface holds, and which census criterion heats the tiles that seed them.
 * A gently undulating surface should hold none: the level set carries it.
 * Per reported step: live samples (state[1]), hot tiles (target heat > 0),
 * and the tiles each criterion (UNIFORM_DETAIL_CRITERIA) triggers, with the
 * hot tiles each one triggers alone.
 * node --import tsx tools/probe-nbflip-calm-seeding.ts [--scene=coarse-surface-standing-wave-live]
 *   [--steps=60] [--every=10] [--values={...}] [--pool=cell,length,width,wavelength,amplitude] [--flat]
 *   [--first [--after=STEP]]: dump the first shape-triggered tiles (at or after STEP), then stop
 * Run through tools/gpu/fluid-gpu-run.sh check. */
import { resolve } from "node:path";
import "../lib/methods";
import { createUniformSolver, readUniformFields, readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { uniformTileKey } from "../lib/methods/uniform/uniform-lattice-address";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import type { MethodParamValues } from "../lib/core/method-contract";
import { UNIFORM_DETAIL_CRITERIA, UNIFORM_STAGE_IMPORTANCE as I } from "../lib/methods/uniform/uniform-stage-grids";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const sceneId = option("scene") ?? "coarse-surface-standing-wave-live";
const steps = Number(option("steps") ?? 60), every = Number(option("every") ?? 10);
const values: MethodParamValues = JSON.parse(option("values") ?? "{}");
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
// --pool=cell,length,width,wavelength,amplitude (m): the scene's tank resized
// and refilled with a cosine swell along x, its refinement regions removed.
// The scene is rebuilt from its factory, so the container walls the document
// bakes into solid voxels follow the new tank and cell.
const pool = option("pool")?.split(",").map(Number);
const authored = getSceneDefinition(sceneId);
// --flat: the scene's initial liquid volumes (a drop, a mound) removed, so
// only its resting pool and its walls remain.
const flat = process.argv.includes("--flat");
const base = !flat ? authored : { ...authored, buildAt: undefined, build: () => ({ ...authored.build(), fluid: { ...authored.build().fluid, initialLiquidVolumes: [] } }) };
const definition = !pool ? base : { ...base, buildAt: undefined, build: () => {
  const scene = base.build(), [cell, length, width, wavelength, amplitude] = pool as [number, number, number, number, number];
  const depth = scene.fluid.initialHeightField?.baseHeight_m ?? scene.container.height_m * scene.container.fillFraction;
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: cell };
  scene.container = { ...scene.container, width_m: length, depth_m: width };
  scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: depth, amplitude_m: amplitude, wavelength_m: wavelength, originX_m: -length / 2 };
  scene.fluid.refinementRegions = [];
  return scene;
} };
const scene = structuredClone(sceneDocument(definition));
const dt = scene.numerics.fixedDt_s;
scene.duration_s = Math.max(scene.duration_s, (steps + 1) * dt);

await withUniformDevice("calm seeding probe", async device => {
  const solver = await createUniformSolver(device, scene, { ...definition.methodProfile?.overrides, ...values });
  // The census writes its importance words only while a layout view reads them.
  solver.setLayoutViewsEnabled(true);
  const internal = solver as unknown as {
    mixedFrame: { ownership: { capacity: { tiles: number; lattice: { dimensions: readonly number[] } } }; narrowBandFlip: { state: GPUBuffer; bins: GPUBuffer; activityHeatWord: number } };
    mixedDynamic: { importance: { buffer: GPUBuffer; offset: number }; reasons: { buffer: GPUBuffer; offset: number } };
  };
  const dims = internal.mixedFrame.ownership.capacity.lattice.dimensions, tiles = internal.mixedFrame.ownership.capacity.tiles;
  const flip = internal.mixedFrame.narrowBandFlip, importance = internal.mixedDynamic.importance;
  const read = async (buffer: GPUBuffer, offset: number, size: number) => {
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, offset, staging, 0, size); device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const out = staging.getMappedRange().slice(0); staging.unmap(); staging.destroy(); return out;
  };
  console.log(JSON.stringify({ scene: sceneId, dt, grid: dims, tiles }));
  // --dump=0 reads the constructed field before any step.
  for (let step = (option("dump") ?? "").split(",").includes("0") ? 0 : 1; step <= steps; step++) {
    if (step > 0) { solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion(); }
    if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
    const first = process.argv.includes("--first"), dumping = option("dump") !== undefined;
    if (!first && !dumping && step % every !== 0 && step !== 1 && step !== steps) continue;
    const state = new Uint32Array(await read(flip.state, 0, 16));
    const words = new Uint32Array(await read(importance.buffer, importance.offset, 8 * tiles));
    // Heat (own) then target (spread) per tile.
    const heat = new Float32Array(await read(flip.bins, 4 * flip.activityHeatWord, 8 * tiles));
    const triggered: Record<string, number> = {}, alone: Record<string, number> = {};
    let hot = 0, ownHot = 0, wet = 0, crossing = 0, dropped = 0;
    for (let t = 0; t < tiles; t++) {
      const w1 = words[2 * t + 1]!, bits = (w1 >>> I.triggeredShift) & 63;
      if (w1 & I.wet) wet++; if (w1 & I.crossing) crossing++; if (w1 & I.dropped) dropped++;
      if (heat[t]! > 0) ownHot++; if (heat[tiles + t]! > 0) hot++;
      UNIFORM_DETAIL_CRITERIA.forEach((name, k) => {
        if (!(bits >> k & 1)) return;
        triggered[name] = (triggered[name] ?? 0) + 1;
        if (bits === 1 << k) alone[name] = (alone[name] ?? 0) + 1;
      });
    }
    // The shape-triggered tiles: score (byte 0 of word 0, scoreOne 64) and tile coordinate.
    const shaped: number[][] = [];
    if (process.argv.includes("--tiles")) {
      const address = await readUniformLiveAddress(device, solver);
      for (let z = 0; z < dims[2]! / 4; z++) for (let y = 0; y < dims[1]! / 4; y++) for (let x = 0; x < dims[0]! / 4; x++) {
        const t = uniformTileKey(address, x, y, z);
        if (t >= 0 && t < tiles && ((words[2 * t + 1]! >>> I.triggeredShift) & 1)) shaped.push([x, y, z, +((words[2 * t]! & 255) / I.scoreOne).toFixed(2)]);
      }
    }
    // --first: at the first step any tile triggers shape, each such tile's
    // width and its h surface (the zero crossing up each vertex column, in
    // cells) beside the surface its 4h corners carry, then stop.
    if (process.argv.includes("--tiles")) console.log(JSON.stringify({ step, shaped: shaped.slice(0, 80) }));
    const dumpAt = new Set((option("dump") ?? "").split(",").filter(Boolean).map(Number));
    if ((first && step >= Number(option("after") ?? 0) && (triggered.shape ?? 0) > 0) || dumpAt.has(step)) {
      const fields = await readUniformFields(device, solver), address = await readUniformLiveAddress(device, solver);
      const [nx, ny] = [dims[0]!, dims[1]!], raw = (x: number, y: number, z: number) => fields.phi[x + (nx + 1) * (y + (ny + 1) * z)]!;
      const crossing = (value: (y: number) => number, y0: number, y1: number) => { for (let y = y0; y < y1; y++) { const a = value(y), b = value(y + 1); if ((a < 0) !== (b < 0)) return +(y + a / (a - b)).toFixed(3); } return NaN; };
      // A profile across z at the first triggered tile's column x, and across x at z = 48 (mid-pool):
      // the surface (from the h or 4h phi as each tile holds it) and each column's tile width.
      const profile = (X: number, Z: number) => { const v = (Y: number) => fields.vertex(X, Y, Z); return [crossing(v, 4, ny - 4), fields.widthAt(Math.min(X, nx - 1), 24, Math.min(Z, dims[2]! - 1))]; };
      const firstX = option("x") ? Number(option("x")) : (() => { for (let z = 0; z < dims[2]! / 4; z++) for (let y = 0; y < ny / 4; y++) for (let x = 0; x < nx / 4; x++) { const t = uniformTileKey(address, x, y, z); if (t >= 0 && t < tiles && ((words[2 * t + 1]! >>> I.triggeredShift) & 1)) return 4 * x + 2; } return 0; })();
      console.log(JSON.stringify({ step, alongZ: { x: firstX, rows: Array.from({ length: Math.min(49, dims[2]! + 1) }, (_, z) => [z, ...profile(firstX, z)]) } }));
      const midZ = Math.floor(dims[2]! / 2);
      // --column=x,z[;x,z]: phi up each vertex column (in h), raw and as the owner holds it, with each cell's tile width.
      for (const spec of (option("column") ?? "").split(";").filter(Boolean)) {
        const [X, Z] = spec.split(",").map(Number) as [number, number], hm = scene.voxelDomain.finestCellSize_m;
        console.log(JSON.stringify({ step, column: [X, Z], rows: Array.from({ length: ny + 1 }, (_, Y) => [Y, +(raw(X, Y, Z) / hm).toFixed(3), +(fields.vertex(X, Y, Z) / hm).toFixed(3), fields.widthAt(Math.min(X, nx - 1), Math.min(Y, ny - 1), Math.min(Z, dims[2]! - 1))]).slice(0, Number(option("rows") ?? 20)) }));
      }
      console.log(JSON.stringify({ step, alongX: { z: midZ, rows: Array.from({ length: nx + 1 }, (_, i) => [i, ...profile(i, midZ)]) } }));
      // The h tiles of the surface's tile rows: a map over (x, z) tiles, each the census reason
      // (UNIFORM_STAGE_REASON) of the highest h tile in the column, '.' where the column is 4h.
      const reasons = new Uint32Array(await read(internal.mixedDynamic.reasons.buffer, internal.mixedDynamic.reasons.offset, 4 * tiles));
      const map: string[] = [];
      for (let z = 0; z < dims[2]! / 4; z++) { let row = ""; for (let x = 0; x < nx / 4; x++) { let c = "."; for (let y = 5; y <= 8; y++) if (fields.widthAt(4 * x, 4 * y, 4 * z) === 1) c = String(reasons[uniformTileKey(address, x, y, z)] ?? "?"); row += c; } map.push(row); }
      console.log(`step ${step} h-tile reasons (rows z, columns x):\n${map.join("\n")}`);
      let shown = 0;
      for (let z = 0, most = first ? 4 : 8; z < dims[2]! / 4 && shown < most; z++) for (let y = 0; y < ny / 4 && shown < most; y++) for (let x = 0; x < nx / 4 && shown < most; x++) {
        const t = uniformTileKey(address, x, y, z); if (t < 0 || t >= tiles || !((words[2 * t + 1]! >>> I.triggeredShift) & 1)) continue; shown++;
        const h: number[][] = [], coarse: number[][] = [];
        for (let c = 0; c <= 4; c++) { h.push([]); coarse.push([]); for (let a = 0; a <= 4; a++) {
          const X = 4 * x + a, Z = 4 * z + c;
          h[c]!.push(crossing(Y => raw(X, Y, Z), 4 * y - 4, 4 * y + 8));
          // The 4h lattice's surface: trilinear in x, z of its corner columns, linear in y between 4h vertices.
          const corner = (Y: number) => { let v = 0; for (let k = 0; k < 4; k++) { const cx = k & 1, cz = k >> 1; v += (cx ? a / 4 : 1 - a / 4) * (cz ? c / 4 : 1 - c / 4) * raw(4 * x + 4 * cx, Y, 4 * z + 4 * cz); } return v; };
          coarse[c]!.push(crossing(Y => { const lo = 4 * Math.floor(Y / 4), f = (Y - lo) / 4; return (1 - f) * corner(lo) + f * corner(lo + 4); }, 4 * y - 4, 4 * y + 8));
        } }
        // The mean height the 4h lattice moves the surface by, in h (positive: the 4h surface higher).
        const moved = h.flat().map((v, i) => coarse.flat()[i]! - v).filter(Number.isFinite);
        const mean = moved.length ? +(moved.reduce((a, b) => a + b, 0) / moved.length).toFixed(3) : null;
        console.log(JSON.stringify({ step, tile: [x, y, z], width: fields.widthAt(4 * x, 4 * y, 4 * z), score: +((words[2 * t]! & 255) / I.scoreOne).toFixed(2), mean, hSurface: h, coarseSurface: coarse }));
      }
      if (first) break;
    }
    console.log(JSON.stringify({ step, time_s: +(step * dt).toFixed(4), particles: state[1], wet, crossing, ownHot, hot, dropped, triggered, alone }));
  }
  solver.destroy();
});
