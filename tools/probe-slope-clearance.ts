import { bakeProceduralSolid } from "../lib/core/procedural-solid";
/**
 * Finite-release baseline, with the production bed-gradient bake enabled.
 * Run with the browser simulation stopped:
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-slope-clearance.ts
 * Optional positional step rate: 30 or 60 (default). Reports every 0.5 s.
 * Pins the four pages for valid diagnostic field readback, as probe-ramp-residue does.
 * Exit 1 means residue, >1% volume drift, or a solver failure; it is a target,
 * not an assertion that the current solver already clears this scene.
 */
import "../lib/methods";
import {
  NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_GRID, NBFLIP_RAMP_LENGTH,
} from "../lib/core/nbflip-ramp-scene";
import {
  NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS, createNbflipSlopeClearance, nbflipSlopeClearanceTransitSeconds, nbflipSlopeClearanceBed,
} from "../lib/core/nbflip-slope-clearance-scene";
import { advanceUniform, createUniformSolver, readUniformFields, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";

const hz = Number(process.argv[2] ?? 60);
if (hz !== 30 && hz !== 60) throw new Error("Choose a step rate of 30 or 60 Hz");
const [nx, ny, nz] = NBFLIP_RAMP_GRID, h = NBFLIP_RAMP_CELL_SIZE_M;
const scene = createNbflipSlopeClearance(hz);
if (process.env.FLUID_CLEARANCE_BOUNDARY === "procedural") {
  delete scene.terrain;
  Object.assign(scene,bakeProceduralSolid(NBFLIP_RAMP_GRID,[h,h,h],([x,y])=>y-nbflipSlopeClearanceBed(x)));
}
scene.fluid.pagePool = { blocks: nx / 32 + 2, air: [] };

async function readPrefix(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<Float32Array> {
  if (bytes === 0) return new Float32Array();
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, read, 0, bytes); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    return new Float32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

await withUniformDevice("slope clearance", async device => {
  console.log(JSON.stringify({ scene: scene.sceneId, hz, target_s: scene.duration_s,
    reference_transit_s: nbflipSlopeClearanceTransitSeconds(), initial_cells: NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS }));
  const began = performance.now();
  const progress = setInterval(() => console.log(JSON.stringify({ initializing_s: (performance.now() - began) / 1000,
    compilation: gpuCompilationManagerFor(device).snapshot() })), 15000);
  const solver = await createUniformSolver(device, scene, JSON.parse(process.env.FLUID_CLEARANCE_VALUES ?? "{}"))
    .finally(() => clearInterval(progress));
  try {
    const frame = (solver as unknown as { mixedFrame: {
      pagePool: { pin(pages: { x: number; y: number; z: number }[]): void };
      narrowBandFlip: { state: GPUBuffer; activeParticles: GPUBuffer };
    } }).mixedFrame;
    frame.pagePool.pin(Array.from({ length: nx / 32 }, (_, x) => ({ x, y: 0, z: 0 })));
    // Approximate per-stage ledger, for locating growth, not for acceptance.
    // Captures follow phi resolve so hanging vertices have their physical value.
    const stages: { name: string; done: () => Promise<Float32Array> }[] = [];
    let stageName = "head";
    if (process.env.FLUID_CLEARANCE_LEDGER) {
      const f = frame as unknown as {
        fields: { phi: GPUTexture; phiScratch: GPUTexture };
        phiResolve: { encode(e: GPUCommandEncoder, g: unknown): void };
        phiResolveGroups: { scratch: unknown };
        narrowBandFlip: Record<"reconstruct" | "retire" | "redistance", (e: GPUCommandEncoder) => unknown>;
      };
      const resolve = f.phiResolve.encode.bind(f.phiResolve);
      f.phiResolve.encode = (e, g) => {
        resolve(e, g);
        const scratch = g === f.phiResolveGroups.scratch;
        const texture = scratch ? f.fields.phiScratch : f.fields.phi;
        stages.push({ name: scratch ? "advect" : stageName, done: uniformDetailField(texture)!.storage.capture(e, texture) });
      };
      for (const name of ["reconstruct", "retire", "redistance"] as const) {
        const original = f.narrowBandFlip[name].bind(f.narrowBandFlip);
        f.narrowBandFlip[name] = e => { stageName = name; return original(e); };
      }
    }
    let initial = 0, clearedSince: number | null = null, passed = false;
    const steps = Number(process.env.FLUID_CLEARANCE_STEPS ?? Math.round(scene.duration_s * hz));
    const every = Number(process.env.FLUID_CLEARANCE_EVERY ?? hz / 2);
    for (let step = 0; step <= steps; step++) {
      stageName = "head";
      if (step > 0) await advanceUniform(solver, step / hz);
      for (const stage of stages.splice(0)) {
        const phi = await stage.done(); let approximate = 0;
        const at = (x: number, y: number, z: number) => phi[x + (nx + 1) * (y + (ny + 1) * z)]!;
        for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
          const open = Math.max(0, Math.min(1, y + 1 - nbflipSlopeClearanceBed(x + 0.5)));
          if (open === 0) continue;
          let mean = 0;
          for (let k = 0; k < 8; k++) mean += at(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1)) / (8 * h);
          approximate += open * Math.max(0, Math.min(1, 0.5 - mean));
        }
        console.log(JSON.stringify({ step, stage: stage.name, approximate_cells: approximate }));
      }
      if (step % every !== 0 && step !== steps) continue;
      const fields = await readUniformFields(device, solver);
      // Expanded owner averages already include solid capacity. Applying the
      // binary voxel mask again discards liquid in continuous cut cells.
      let ramp = 0, pit = 0;
      for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
          const density = fields.density[x + nx * (y + ny * z)]!;
          if (!Number.isFinite(density) || density < 0) throw new Error(`Invalid density at ${x},${y},${z}: ${density}`);
          if (x < NBFLIP_RAMP_LENGTH) ramp += density; else pit += density;
        }
      }
      const state = await readPrefix(device, frame.narrowBandFlip.state, 4);
      const count = new Uint32Array(state.buffer)[0]!;
      const particles = await readPrefix(device, frame.narrowBandFlip.activeParticles, 48 * count);
      let rampSamples = 0;
      let positionX = 0, velocityX = 0, ballistic = 0;
      for (let i = 0; i < count; i++) {
        const x = particles[12 * i]!;
        positionX += x; velocityX += particles[12 * i + 4]!;
        ballistic += +(particles[12 * i + 11] === 1);
        if (x >= 0 && x < NBFLIP_RAMP_LENGTH) rampSamples++;
      }
      const total = ramp + pit;
      const stats = await solver.readStats();
      if (Math.abs(total-stats.volumeCellSum!) > 2e-5*Math.max(1,total))
        throw new Error(`Volume readback disagrees with GPU accounting: ${total} vs ${stats.volumeCellSum}`);
      if (step === 0) {
        initial = total;
        if (!(initial > 0)) throw new Error("The release contains no liquid");
      }
      const drift = (total - initial) / initial;
      const seedError = (initial - NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS) / NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS;
      const clear = ramp === 0 && rampSamples === 0;
      clearedSince = clear ? clearedSince ?? step / hz : null;
      passed = clear && Math.abs(drift) <= 0.01 && Math.abs(seedError) <= 0.01;
      console.log(JSON.stringify({ t: step / hz, ramp_cells: ramp, ramp_litres: ramp * h ** 3 * 1000,
        pit_cells: pit, total_cells: total, volume_drift: drift, seed_error: seedError,
        stats_cells: stats.volumeCellSum, stats_drift: stats.volumeDrift, speed_m_s: stats.maxSpeed_m_s,
        budget_cells: stats.narrowBandTargetVolume_cells, control_per_s: stats.narrowBandVolumeControlRate_per_s,
        samples: count, mean_x: positionX / Math.max(1, count), mean_u: velocityX / Math.max(1, count), ballistic,
        ramp_samples: rampSamples, clear, cleared_since_s: clearedSince }));
    }
    passed &&= steps >= scene.duration_s * hz;
    console.log(JSON.stringify({ passed, target_s: scene.duration_s, cleared_since_s: clearedSince,
      criterion: "Zero slope volume and samples, with seed error and total volume drift each at most 1%." }));
    if (!passed) process.exitCode = 1;
  } finally { solver.destroy(); }
});
