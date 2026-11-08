import {test} from "node:test";
import assert from "node:assert/strict";
import "../lib/methods";
import {getSceneDefinition} from "../lib/core/scenes";
import {sceneDocument} from "../lib/core/scene-definition";
import {createUniformSolver, readUniformFields, withUniformDevice} from "./helpers/uniform-geometric";
import type {WebGPUUniformReferenceSolver as UniformSolver} from "../lib/methods/uniform/webgpu-uniform-reference";

// Figure 4's tank: 32^3, a solid wall layer one cell thick (wallLayer), its
// pool 0.3 of the grid deep, and a sphere mound that sets it oscillating.
const SCENE = "nbflip-figure-4-oscillating-surface";
const document = (flat: boolean) => {
 const authored = getSceneDefinition(SCENE);
 const definition = !flat ? authored : {...authored, buildAt: undefined, build: () => {const scene = authored.build(); return {...scene, fluid: {...scene.fluid, initialLiquidVolumes: []}};}};
 return {scene: structuredClone(sceneDocument(definition)), overrides: definition.methodProfile?.overrides ?? {}};
};
const liveSamples = async (device: GPUDevice, solver: UniformSolver) => {
 const state = (solver as unknown as {mixedFrame: {narrowBandFlip: {state: GPUBuffer}}}).mixedFrame.narrowBandFlip.state;
 const read = device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
 const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(state, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
 await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
 return words[1]!;
};
const advance = async (solver: UniformSolver, steps: number, dt: number) => {
 for (let step = 1; step <= steps; step++) {
  solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion();
  if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
 }
};

// A still pool beside the tank's walls has no shape for particles to keep.
// The census measured the waterline ring's tiles against their 4h corners
// buried in the wall layer (air markers, not samples): every one of the 28
// scored the saturated shape error, and the pool held 12-14k particles at
// rest. A coarsened owner reads the surface plane continued into the solid
// there, which is what the census now measures against.
test("NB-FLIP: a resting pool beside one-cell tank walls holds no particles", {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
 await withUniformDevice("NB-FLIP calm pool", async device => {
  const {scene, overrides} = document(true), dt = scene.numerics.fixedDt_s;
  scene.duration_s = Math.max(scene.duration_s, 31 * dt);
  const solver = await createUniformSolver(device, scene, overrides);
  try {
   await advance(solver, 30, dt);
   assert.equal(await liveSamples(device, solver), 0, "live particle samples on a resting pool");
  } finally { solver.destroy(); }
 }, ["subgroups"]);
});

// The liquid beneath Figure 4's mound stays a distance as its h tiles
// coarsen. The retiring 4h corner's metric bound (nbBoundCoarseMetric) read a
// sign change into the tank floor's buried far marker (+32 h) as a surface
// 0.6 h above the corner: -5.6 h became -0.6 h in one step and -0.003 h in
// five, a phantom surface on the y = 4 plane that kept shape and thin heat,
// and particles, for the whole run.
test("NB-FLIP: coarsening beside a tank floor keeps deep liquid deep", {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
 await withUniformDevice("NB-FLIP deep coarse corners", async device => {
  const {scene, overrides} = document(false), dt = scene.numerics.fixedDt_s, h = scene.voxelDomain.finestCellSize_m;
  scene.duration_s = Math.max(scene.duration_s, 6 * dt);
  const solver = await createUniformSolver(device, scene, overrides);
  try {
   await advance(solver, 5, dt);
   const fields = await readUniformFields(device, solver);
   // The pool's surface rests at 9.6 cells and the mound only raises it: y = 4
   // lies 5.6 h deep or more at every interior column.
   let shallowest = -Infinity, at = "";
   for (let z = 2; z <= 30; z++) for (let x = 2; x <= 30; x++) {
    const phi = fields.vertex(x, 4, z) / h;
    if (phi > shallowest) { shallowest = phi; at = `(${x}, 4, ${z})`; }
   }
   assert.ok(shallowest < -4, `the y = 4 plane's shallowest liquid phi ${shallowest.toFixed(3)} h at ${at}`);
  } finally { solver.destroy(); }
 }, ["subgroups"]);
});
