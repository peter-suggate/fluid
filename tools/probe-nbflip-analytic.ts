/** Run the analytic NB-FLIP cases (lib/core/nbflip-analytic-scenes.ts) on the
 * production solver at 1/60 s and score each against its answer. Every frame
 * also reads the 16h region work tiers, so the report carries the COLD/WARM/HOT
 * populations and how often regions change tier (flicker).
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-nbflip-analytic.ts
 *   [--case=rest,free-fall,...] [--frames=N] [--out=artifacts/nbflip-analytic] [--set=key:value,...]
 * --set overrides NB-FLIP method values (numbers or strings), for A/Bs.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import {
  createNbflipAnalyticScene, martinMoyceFront, nbflipAnalyticWaveOmega, NBFLIP_ANALYTIC_CASES, NBFLIP_ANALYTIC_CELL_M,
  NBFLIP_ANALYTIC_DAM_CELLS, NBFLIP_ANALYTIC_DT_S, NBFLIP_ANALYTIC_FALL_BLOCK, NBFLIP_ANALYTIC_FALL_GRAVITY, NBFLIP_ANALYTIC_FRAMES, NBFLIP_ANALYTIC_GRAVITY_M_S2,
  NBFLIP_ANALYTIC_GRID, NBFLIP_ANALYTIC_POOL_CELLS, NBFLIP_ANALYTIC_WAVE_AMPLITUDE_M, type NbflipAnalyticCase,
} from "../lib/core/nbflip-analytic-scenes";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { MethodParamValues } from "../lib/core/method-contract";
import { UNIFORM_REGION_TIER } from "../lib/methods/uniform/uniform-region-heat";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { advanceUniform, createUniformSolver, readUniformFields, readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
for (const argument of process.argv.slice(2)) assert.match(argument, /^--(?:case|frames|out|set)=.+$/);
const cases = (option("case")?.split(",") ?? [...NBFLIP_ANALYTIC_CASES]) as NbflipAnalyticCase[];
for (const name of cases) assert.ok(NBFLIP_ANALYTIC_CASES.includes(name), `unknown case ${name}`);
const outDir = option("out") ?? "artifacts/nbflip-analytic";
const overrides: MethodParamValues = Object.fromEntries((option("set")?.split(",") ?? []).map(pair => {
  const [key, value] = pair.split(":") as [string, string];
  return [key, Number.isFinite(Number(value)) ? Number(value) : value];
}));
const [NX, NY, NZ] = NBFLIP_ANALYTIC_GRID, h = NBFLIP_ANALYTIC_CELL_M, rho = 1000;
mkdirSync(outDir, { recursive: true });
paceDawnEventPolling();

interface Row {
  frame: number; t: number; particles: number; volume_m3: number; phiVolume_m3: number; centroid: number[]; meanVelocity: number[];
  maxSpeed: number; ke_J: number; pe_J: number; surface: { mean: number; min: number; max: number; left: number; right: number; middle: number };
  front_cells: number; bboxCells: number[]; extentCells: number[]; tiers: { cold: number; warm: number; hot: number; live: number; changed: number; reversed: number };
}

async function readTiers(device: GPUDevice, frame: UniformMixedFrame): Promise<Uint32Array> {
  const words = frame.heat.regions, read = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(frame.heat.tiers, 0, read, 0, 4 * words); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

async function readParticles(device: GPUDevice, particles: UniformNarrowBandFlip): Promise<number> {
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(particles.state, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange()); assert.equal(words[3], 0, "particle fatal"); return words[1]!;
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

async function sample(device: GPUDevice, solver: WebGPUUniformReferenceSolver, frameIndex: number, gravity: { x: number; y: number; z: number }, piers: readonly (readonly [number, number])[], solids: readonly (readonly [number, number, number])[], extended?: GPUTexture) {
  // The detail storage's own logical read: a live address passed here remaps
  // it through the wrong layout and reads zeros.
  // Liquid outside the pressure rows (a partly filled owner whose centre is
  // air) has no projected velocity: the projection zeroes every face between
  // two air owners, and a cell average over one read half or none of the
  // liquid's speed (free fall's 4h end rows: up to 16% of the mass, the mean
  // 7% slow at frame 15 while every face was exact). The velocity the solver
  // carries that liquid with, and the next step transports, is the
  // extension of the projected field (extended; the lattice where the
  // advance extended nothing).
  const fields = await readUniformFields(device, solver), velocity = await readMixedTexture(device, extended ?? solver.velocityTexture);
  const { nx, ny, nz } = solver.info; assert.deepEqual([nx, ny, nz], [NX, NY, NZ]);
  let mass = 0, ke = 0, pe = 0, maxSpeed = 0; const c = [0, 0, 0], v = [0, 0, 0], lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  let front = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = x + nx * (y + ny * z), d = fields.density[i]!;
    assert.ok(Number.isFinite(d), `frame ${frameIndex}: finite volume`);
    if (!(d > 0)) continue;
    const p = [(x + .5) * h, (y + .5) * h, (z + .5) * h];
    // MAC faces live at each owner's +face anchor (origin + w - 1 on the
    // axis); the -face is the neighbour's +anchor, a closed wall outside.
    const w = fields.widthAt(x, y, z), o = [x - x % w, y - y % w, z - z % w], n = [nx, ny, nz];
    const face = (q: number[], a: number) => q[a]! < 0 || q[a]! >= n[a]! ? 0 : velocity[4 * (q[0]! + nx * (q[1]! + ny * q[2]!)) + a]!;
    const u = [0, 1, 2].map(a => .5 * (face(o.map((value, b) => b === a ? value + w - 1 : value), a) + face(o.map((value, b) => b === a ? value - 1 : value), a)));
    const s2 = u[0]! ** 2 + u[1]! ** 2 + u[2]! ** 2;
    if (d > .5) maxSpeed = Math.max(maxSpeed, Math.sqrt(s2));
    mass += d; ke += .5 * d * s2; pe -= d * (gravity.x * p[0]! + gravity.y * p[1]! + gravity.z * p[2]!);
    for (let a = 0; a < 3; a++) { c[a]! += d * p[a]!; v[a]! += d * u[a]!; }
    if (d > .5) { lo[0] = Math.min(lo[0]!, x); hi[0] = Math.max(hi[0]!, x); lo[1] = Math.min(lo[1]!, y); hi[1] = Math.max(hi[1]!, y); }
    if (y < 2 && d > .5) front = Math.max(front, x + 1);
  }
  const cell3 = h ** 3;
  // Surface height per column on the mid-depth plane: phi's first upward zero crossing.
  const heights: number[] = [];
  for (let x = 0; x <= nx; x++) {
    let height = NaN;
    // Columns touching a solid that pierces the surface have no free surface.
    if (piers.some(([x0, x1]) => x >= x0 && x <= x1)) { heights.push(height); continue; }
    // Start above any solid under the column: buried padding phi is no surface.
    const floor = Math.max(0, ...solids.filter(([x0, x1]) => x >= x0 && x <= x1).map(([, , top]) => top));
    for (let y = floor; y < ny; y++) {
      const a = fields.vertex(x, y, nz / 2), b = fields.vertex(x, y + 1, nz / 2);
      if (a <= 0 && b > 0) { height = (y + a / (a - b)) * h; break; }
    }
    heights.push(height);
  }
  const finite = heights.filter(Number.isFinite), at = (x: number) => heights[x] ?? NaN;
  return {
    volume_m3: mass * cell3, centroid: c.map(value => value / mass), meanVelocity: v.map(value => value / mass), maxSpeed,
    ke_J: rho * ke * cell3, pe_J: rho * pe * cell3,
    surface: { mean: finite.reduce((s, x) => s + x, 0) / Math.max(1, finite.length), min: Math.min(...finite), max: Math.max(...finite),
      left: .5 * (at(1) + at(2)), right: .5 * (at(nx - 1) + at(nx - 2)), middle: .5 * (at(nx / 2) + at(nx / 2 + 1)) },
    front_cells: front, bboxCells: [hi[0]! - lo[0]! + 1, hi[1]! - lo[1]! + 1, lo[0]!, lo[1]!],
    // A uniform box's extent from its second moment: sqrt(12 var), in cells.
    extentCells: phiExtent(fields, nx, ny, nz),
  };
}

/** The liquid's x and y extent from its second moment, sqrt(12 var) in
 * cells, on the mid-depth plane of the resolved phi: each h cell's liquid
 * area from its bilinear corners. The fill field broadcasts a 4h owner's V
 * over its cells, which spreads a partly filled owner's liquid across it: a
 * 16-cell block whose ends sit mid-owner read up to 1.4 cells long. */
function phiExtent(fields: Awaited<ReturnType<typeof readUniformFields>>, nx: number, ny: number, nz: number): number[] {
  const z = nz / 2, n = 8; let mass = 0; const m1 = [0, 0], m2 = [0, 0];
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const a = fields.vertex(x, y, z), b = fields.vertex(x + 1, y, z), c = fields.vertex(x, y + 1, z), d = fields.vertex(x + 1, y + 1, z);
    let area = 0;
    if (a < 0 && b < 0 && c < 0 && d < 0) area = 1;
    else if (a < 0 || b < 0 || c < 0 || d < 0) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const u = (i + .5) / n, v = (j + .5) / n;
        if ((a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v < 0) area += 1 / (n * n);
      }
    }
    if (!area) continue;
    mass += area; m1[0]! += area * (x + .5); m1[1]! += area * (y + .5); m2[0]! += area * (x + .5) ** 2; m2[1]! += area * (y + .5) ** 2;
  }
  return [0, 1].map(k => Math.sqrt(12 * Math.max(0, m2[k]! / mass - (m1[k]! / mass) ** 2)));
}

type Check = { name: string; value: number; limit: number; pass: boolean };
const check = (name: string, value: number, limit: number): Check => ({ name, value, limit, pass: Number.isFinite(value) && value <= limit });

function score(name: NbflipAnalyticCase, rows: Row[]): Check[] {
  const first = rows[0]!, last = rows[rows.length - 1]!, checks: Check[] = [];
  // The phi gauge can read 0 before the first advance resolves phi (mound).
  const phi0 = rows.find(r => r.phiVolume_m3 > 0)?.phiVolume_m3 ?? NaN;
  checks.push(check("phi volume drift (fraction)", Math.max(...rows.filter(r => r.phiVolume_m3 > 0).map(r => Math.abs(r.phiVolume_m3 / phi0 - 1))), .01));
  checks.push(check("fill-field volume drift (fraction)", Math.max(...rows.map(r => Math.abs(r.volume_m3 / first.volume_m3 - 1))), .01));
  const late = rows.filter(r => r.frame > rows.length / 2);
  const liveFrames = late.reduce((s, r) => s + r.tiers.live, 0) || 1;
  checks.push(check("late tier changes per live region-frame", late.reduce((s, r) => s + r.tiers.changed, 0) / liveFrames, name === "rest" || name === "rest-step" ? 0 : .05));
  checks.push(check("tier reversals per live region-frame", rows.reduce((s, r) => s + r.tiers.reversed, 0) / (rows.reduce((s, r) => s + r.tiers.live, 0) || 1), .01));
  if (name === "rest" || name === "rest-step") {
    const H = NBFLIP_ANALYTIC_POOL_CELLS * h;
    checks.push(check("max speed after frame 10 (m/s)", Math.max(...rows.filter(r => r.frame > 10).map(r => r.maxSpeed)), 1e-3));
    checks.push(check("surface deviation from H (h)", Math.max(...rows.map(r => Math.max(Math.abs(r.surface.max - H), Math.abs(r.surface.min - H)))) / h, .1));
    checks.push(check("non-COLD live regions, last frame", last.tiers.warm + last.tiers.hot, 0));
  }
  if (name === "free-fall") {
    const g = NBFLIP_ANALYTIC_FALL_GRAVITY, c0 = first.centroid, v0 = first.meanVelocity, t0 = first.t;
    let centroidError = 0, velocityError = 0, shape = 0;
    // Ballistic while the exact block's bottom is two cells above the floor.
    const bottom0 = NBFLIP_ANALYTIC_FALL_BLOCK[0][1] * h;
    const flight = rows.filter(r => bottom0 + .5 * g.y * (r.t - t0) ** 2 >= 2 * h);
    for (const r of flight) {
      const dt = r.t - t0, ex = c0[0]! + v0[0]! * dt + .5 * g.x * dt * dt, ey = c0[1]! + v0[1]! * dt + .5 * g.y * dt * dt;
      centroidError = Math.max(centroidError, Math.hypot(r.centroid[0]! - ex, r.centroid[1]! - ey) / h);
      // The grid velocity after a step is the leapfrog half-step velocity
      // v0 + g (t + dt/2), the one the positions integrate exactly.
      const lead = r.frame > 0 ? dt + NBFLIP_ANALYTIC_DT_S / 2 : 0;
      velocityError = Math.max(velocityError, Math.hypot(r.meanVelocity[0]! - v0[0]! - g.x * lead, r.meanVelocity[1]! - v0[1]! - g.y * lead));
      shape = Math.max(shape, Math.abs(r.extentCells[0]! - first.extentCells[0]!), Math.abs(r.extentCells[1]! - first.extentCells[1]!));
    }
    checks.push(check("centroid error vs x0+v0t+gt^2/2 (h)", centroidError, .5));
    checks.push(check("mean velocity error vs v0+g(t+dt/2) (m/s)", velocityError, .02 * Math.hypot(g.x, g.y) * (last.t - t0) + .01));
    checks.push(check("moment extent change (cells)", shape, .5));
    const internal = Math.max(...flight.map(r => r.ke_J - .5 * rho * r.volume_m3 * (r.meanVelocity[0]! ** 2 + r.meanVelocity[1]! ** 2 + r.meanVelocity[2]! ** 2)));
    checks.push(check("internal (non-translational) KE (J)", internal, 1e-3 * Math.max(...rows.map(r => r.ke_J)) + 1e-6));
  }
  if (name === "standing-wave") {
    const omega = nbflipAnalyticWaveOmega(), H = NBFLIP_ANALYTIC_POOL_CELLS * h, A = NBFLIP_ANALYTIC_WAVE_AMPLITUDE_M;
    let se = 0, middle = 0;
    for (const r of rows) {
      const expected = A * Math.cos(omega * r.t), left = r.surface.left - H, right = r.surface.right - H;
      se += (left - expected) ** 2 + (right + expected) ** 2; middle = Math.max(middle, Math.abs(r.surface.middle - H));
    }
    checks.push(check("end-height RMS error vs A cos(wt) (fraction of A)", Math.sqrt(se / (2 * rows.length)) / A, .25));
    checks.push(check("middle node excursion (fraction of A)", middle / A, .3));
    // Half period: the left end's first upward zero crossing relative to H, from a trough.
    let crossing = NaN;
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1]!.surface.left - H, b = rows[i]!.surface.left - H;
      if (a > 0 && b <= 0) { crossing = rows[i - 1]!.t + (rows[i]!.t - rows[i - 1]!.t) * a / (a - b); break; }
    }
    checks.push(check("first zero crossing error vs pi/(2w) (fraction)", Math.abs(crossing * 2 * omega / Math.PI - 1), .05));
    const lastPeak = Math.max(...rows.filter(r => r.t > 1.5 * Math.PI / omega).map(r => r.surface.left - H));
    checks.push(check("amplitude loss after one period (fraction of A)", 1 - lastPeak / A, .2));
  }
  if (name === "dam-break") {
    const a = NBFLIP_ANALYTIC_DAM_CELLS, scale = Math.sqrt(2 * NBFLIP_ANALYTIC_GRAVITY_M_S2 / (a * h));
    let worst = 0;
    for (const r of rows) {
      const reference = martinMoyceFront(r.t * scale);
      if (reference === undefined || reference * a > NX - 2) continue;
      worst = Math.max(worst, Math.abs(r.front_cells / a / reference - 1));
    }
    checks.push(check("front Z error vs Martin & Moyce (fraction)", worst, .12));
  }
  if (name === "mound" || name === "dam-break") {
    const energy = rows.map(r => r.ke_J + r.pe_J), e0 = energy[0]!, floor = Math.min(...energy);
    let rise = 0; for (let i = 1; i < energy.length; i++) rise = Math.max(rise, energy[i]! - Math.min(...energy.slice(0, i)));
    checks.push(check("largest energy rise (fraction of released energy)", rise / Math.max(1e-12, e0 - floor), .02));
  }
  if (name === "mound") {
    const peak = Math.max(...rows.map(r => r.ke_J));
    checks.push(check("final KE (fraction of peak)", last.ke_J / peak, .1));
  }
  return checks;
}

const summary: Record<string, unknown>[] = [];
await withUniformDevice("NB-FLIP analytic cases", async device => {
  for (const name of cases) {
    const scene = createNbflipAnalyticScene(name), frames = Number(option("frames") ?? NBFLIP_ANALYTIC_FRAMES[name]);
    const gravity = scene.fluid.gravity_m_s2, began = performance.now();
    const surfaceTop = NBFLIP_ANALYTIC_POOL_CELLS - 1;
    const piers = (scene.solidVoxels ?? []).filter(v => v.maximumExclusive[1] > surfaceTop).map(v => [v.minimum[0], v.maximumExclusive[0]] as const);
    const solids = (scene.solidVoxels ?? []).map(v => [v.minimum[0], v.maximumExclusive[0], v.maximumExclusive[1]] as const);
    const solver = await createUniformSolver(device, scene, overrides);
    const built_ms = performance.now() - began;
    try {
      const mixed = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
      const volumeProbe = await createNbflipVolumeProbe(device, solver);
      const rows: Row[] = [], history: Map<number, number>[] = [];
      let previous: Map<number, number> | undefined;
      const stepStart = performance.now();
      for (let frame = 0; frame <= frames; frame++) {
        if (frame > 0) await advanceUniform(solver, frame * NBFLIP_ANALYTIC_DT_S);
        const tiers = await readTiers(device, mixed), state = await sample(device, solver, frame, gravity, piers, solids,
          frame > 0 && !(mixed as unknown as { extensionIdle: boolean }).extensionIdle ? (mixed as unknown as { fields: { velocityScratch: GPUTexture } }).fields.velocityScratch : undefined);
        // Region r is the 16h octant r&7 of pool block r>>3; the block is live
        // when its word holds a world page residue (0xffffffff: free). Key
        // regions by world page and octant so history follows space, not slots.
        const address = await readUniformLiveAddress(device, solver), blocks = address.seed.blocks;
        const keyed = new Map<number, number>();
        for (let r = 0; r < tiers.length; r++) {
          const word = blocks[r >> 3];
          if (word === undefined || word === 0xffffffff) continue;
          keyed.set(word * 8 + (r & 7), tiers[r]!);
        }
        let cold = 0, warm = 0, hot = 0, changed = 0, reversed = 0;
        for (const [key, tier] of keyed) {
          if (tier === UNIFORM_REGION_TIER.cold) cold++; else if (tier === UNIFORM_REGION_TIER.warm) warm++; else hot++;
          const before = previous?.get(key);
          if (before !== undefined && before !== tier) {
            changed++;
            // A reversal: the tier returns to one it held in the three frames before.
            if (history.slice(-3, -1).some(frame => frame.get(key) === tier)) reversed++;
          }
        }
        history.push(keyed); if (history.length > 4) history.shift(); previous = keyed;
        const liveCount = keyed.size;
        const phiVolume_m3 = (await volumeProbe.read()).reduce((sum, value) => sum + value, 0) * h ** 3;
        const row: Row = { frame, t: frame * NBFLIP_ANALYTIC_DT_S, particles: await readParticles(device, mixed.narrowBandFlip), phiVolume_m3, ...state,
          tiers: { cold, warm, hot, live: liveCount, changed, reversed } };
        rows.push(row);
      }
      const step_ms = (performance.now() - stepStart) / frames;
      const checks = score(name, rows);
      const result = { case: name, frames, dt: NBFLIP_ANALYTIC_DT_S, overrides, built_ms, step_ms_including_readback: step_ms, checks, rows };
      writeFileSync(`${outDir}/${name}.json`, JSON.stringify(result, null, 1) + "\n");
      const line = { case: name, built_s: +(built_ms / 1000).toFixed(1), pass: checks.every(c => c.pass),
        failed: checks.filter(c => !c.pass).map(c => `${c.name}: ${c.value.toPrecision(3)} > ${c.limit}`),
        tiersLast: rows[rows.length - 1]!.tiers, particlesLast: rows[rows.length - 1]!.particles };
      summary.push(line); console.log(JSON.stringify(line));
    } finally { solver.destroy(); }
  }
}, ["subgroups"]);
writeFileSync(`${outDir}/summary.json`, JSON.stringify(summary, null, 1) + "\n");
