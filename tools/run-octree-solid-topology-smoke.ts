/**
 * Adaptive-topology gate for solids moving through the domain.
 *
 * Aanjaneya et al. 2017 refine the octree from proximity to solid geometry and
 * re-evaluate it every frame for moving solids: "The octree topology is fine
 * within a bounding box of the object and the sources, and coarser elsewhere
 * ... The box enclosing the object is updated every frame, so that it can
 * track complex solid-fluid interactions." (Section 6, Figure 3.)
 *
 * This lane drives a KINEMATIC body, so nothing here depends on fluid/rigid
 * coupling: it isolates the topology transaction alone. The body starts
 * straddling the free surface and sweeps down until its refined halo separates
 * from the interface-refined band, then sweeps back up so the two regions
 * merge again. Both transitions must keep publishing a non-empty candidate
 * topology.
 *
 * The measured failure this gate exists for: on the separation step the
 * candidate retires every row in a single transaction (carried 0, added 0,
 * retired all) and the frontier tail bank emits zero leaves, after which the
 * dirty-tile authority latches its failure magic and never re-arms. A body
 * that starts already submerged, or that rises into the band, never trips it.
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import { createTinyHydrostaticScene } from "../lib/scenes";
import { initializeRigidBodies } from "../lib/rigid-body";
import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU; globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("WebGPU did not expose an adapter");
const device = await adapter.requestDevice({
  requiredFeatures: fluidExecutionDeviceFeatures(adapter.features),
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

const side_m = Number(process.env.FLUID_SOLID_TOPOLOGY_SIDE_M ?? 0.2);
const sweep_m_s = Number(process.env.FLUID_SOLID_TOPOLOGY_SWEEP_M_S ?? 1.2);
// Half the run descends, half ascends, so the pose returns to its release
// height under its own velocity rather than against any container assumption.
const descentSteps = Number(process.env.FLUID_SOLID_TOPOLOGY_DESCENT_STEPS ?? 70);

const scene = createTinyHydrostaticScene();
scene.sceneId = "smoke-octree-solid-topology-separation";
const surface_m = scene.container.height_m * scene.container.fillFraction;
// Straddling the surface: the solid halo and the interface band start merged,
// so the descent is what separates them.
const startY_m = surface_m + 0.2 * side_m;
scene.rigidBodies = [{
  id: "sweeping-box", name: "Sweeping box", shape: "box",
  dimensions_m: { x: side_m, y: side_m, z: side_m }, density_kg_m3: 1000,
  position_m: { x: 0, y: startY_m, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0, friction: 0, motion: "static",
}];

const values = { ...octreeMethod.presetFor("balanced") };
if (process.env.FLUID_MAXIMUM_LEAF_SIZE) values.maximumLeafSize = process.env.FLUID_MAXIMUM_LEAF_SIZE;
if (process.env.FLUID_OCTREE_INTERFACE_BAND) {
  values.interfaceRefinementBandCells = Number(process.env.FLUID_OCTREE_INTERFACE_BAND);
}
const bodies = initializeRigidBodies(scene.rigidBodies);
const solver = await octreeMethod.createSolverAsync!(
  device, scene, "balanced", values, undefined, () => {},
);
assert.equal(solver.initialSparseAuthorityReady, true,
  "octree construction must fence the complete t=0 sparse authority");
const projection = (solver as unknown as {
  octreeProjection: { readPowerFrontierFailure(): Promise<Record<string, number[]>> };
}).octreeProjection;

const dt = scene.numerics.fixedDt_s;
const steps = 2 * descentSteps;
const trace: Array<Record<string, number>> = [];
let collapseStep = -1, collapseY_m = 0, minimumRows = Number.POSITIVE_INFINITY;
let previousGeneration = -1, frozenGenerationSteps = 0;
for (let step = 1; step <= steps; step += 1) {
  const direction = step <= descentSteps ? -1 : 1;
  const elapsed = step <= descentSteps ? step : 2 * descentSteps - step;
  bodies[0].position_m.y = startY_m - sweep_m_s * elapsed * dt;
  bodies[0].linearVelocity_m_s.y = direction * sweep_m_s;
  while (!solver.advanceTo(step * dt, bodies)) await new Promise((resolve) => setTimeout(resolve, 0));
  await device.queue.onSubmittedWorkDone();
  const failure = await projection.readPowerFrontierFailure();
  // Row-delta control header: current, previous, carried, added, retired,
  // dirty, affected, generation.
  const rowDelta = failure.rowDelta ?? [];
  const current = rowDelta[0] ?? -1, generation = rowDelta[7] ?? -1;
  minimumRows = Math.min(minimumRows, current);
  if (generation === previousGeneration) frozenGenerationSteps += 1;
  previousGeneration = generation;
  if (current === 0 && collapseStep < 0) {
    collapseStep = step;
    collapseY_m = bodies[0].position_m.y;
  }
  if (step % 5 === 0 || step === collapseStep) {
    trace.push({ step, y_m: Number(bodies[0].position_m.y.toFixed(4)), current, generation,
      coarseState: (failure.coarseDirectory ?? [])[0] ?? -1,
      coarseGeneration: (failure.coarseDirectory ?? [])[1] ?? -1,
      coarseRows: (failure.coarseDirectory ?? [])[2] ?? -1,
      frontierReject: (failure.frontier ?? [])[9] ?? -1 });
  }
}

const info = await solver.readStats();
const result = {
  scenario: scene.sceneId, grid: [solver.info.nx, solver.info.ny, solver.info.nz],
  surface_m, startY_m, side_m, sweep_m_s, descentSteps, steps, dt,
  collapseStep, collapseY_m, minimumRows, frozenGenerationSteps,
  nonFiniteVelocityCount: info.nonFiniteCount ?? 0, validationErrors, trace,
};
console.log(JSON.stringify(result));

assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
assert.equal(info.nonFiniteCount ?? 0, 0, "fluid velocity must remain finite");
// The defect signature: one transaction retires every row and publishes an
// empty candidate topology, from which the dirty-tile authority never re-arms.
assert.equal(collapseStep, -1,
  `the candidate topology published zero rows at step ${collapseStep}, with the body at `
  + `y=${collapseY_m.toFixed(4)} m (surface ${surface_m.toFixed(4)} m). A solid whose refined `
  + "halo separates from the interface band must keep publishing a non-empty topology.");
assert.ok(minimumRows > 0, `candidate row count fell to ${minimumRows}`);
// A latched failure shows up as a candidate generation that stops advancing
// even though every step submits a new transaction.
assert.ok(frozenGenerationSteps <= 1,
  `the candidate generation stalled on ${frozenGenerationSteps} steps, so the topology `
  + "transaction latched instead of re-arming");

solver.destroy(); device.destroy();
