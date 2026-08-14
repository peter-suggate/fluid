/** Dawn-only t=0 / first-publication audit for the factor-one dam surface. */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

// Composition root for this entry point: importing the method catalog installs
// the simulation methods and the octree coarse-dynamics lanes, without which
// constructing a solver throws rather than silently running the wrong backend.
import "../lib/methods";
import { losassoMethod } from "../lib/methods/losasso/method";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { analyzeAdaptiveSurfaceFeatureGeometry, analyzeAdaptiveSurfacePublication } from
  "../lib/methods/octree-shared/octree-adaptive-surface-diagnostics";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { unpackAdaptivePhiReceipt } from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-phi";

const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU; globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ...(process.env.FLUID_WEBGPU_ADAPTER ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : [])]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter);
const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("subgroups")
  ? ["subgroups"] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", event => validationErrors.push(event.error.message));

const scene = getScenePreset("water-box-dam-break").create();
scene.numerics.fixedDt_s = 0.004; scene.numerics.maxDt_s = 0.004;
const solver = await losassoMethod.createSolverAsync!(device, scene, "balanced", {
  ...losassoMethod.presetFor("balanced"),
  losassoVelocityExtension: "causal-front",
  maximumLeafSize: process.env.FLUID_MAXIMUM_LEAF_SIZE ?? "16",
  interfaceRefinementBandCells: 4, globalFineLevelSetFactor: "1", secondaryParticles: "off",
}, undefined, () => {}) as GPUSolverInstance;
const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
// The analysis snapshot type declares its word arrays as `ArrayLike<number>`;
// the projection actually publishes concrete typed arrays (webgpu-octree.ts
// `readAdaptiveSurfacePublicationDiagnostics`), which this probe slices.
const projection = (solver as unknown as { octreeProjection?: {
  readAdaptiveSurfacePublicationDiagnostics(): Promise<Parameters<
    typeof analyzeAdaptiveSurfacePublication>[0] & {
      graphControl: Uint32Array;
      phiControl: Uint32Array;
      nodalPhi: Uint32Array;
      constraints: Uint32Array;
      phiReceipts: Uint32Array;
    } | undefined>;
} }).octreeProjection;
assert.ok(projection);
const samples: unknown[] = [];
const bits = new Uint32Array(1); const scalar = new Float32Array(bits.buffer);
const fromBits = (value: number) => (bits[0] = value >>> 0, scalar[0]!);
for (let step = 0; step <= 1; step += 1) {
  if (step) while (!solver.advanceTo(0.004, [])) await new Promise(resolve => setImmediate(resolve));
  await device.queue.onSubmittedWorkDone();
  const snapshot = await projection.readAdaptiveSurfacePublicationDiagnostics();
  assert.ok(snapshot);
  const publication = analyzeAdaptiveSurfacePublication(snapshot);
  const geometry = analyzeAdaptiveSurfaceFeatureGeometry(snapshot, 0, []);
  const receipt = unpackAdaptivePhiReceipt(snapshot.phiReceipts);
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  let maximumStoredConstraintNode = -1, maximumStoredConstraintError = -1;
  for (let node = 0; node < publication.nodeCount; node += 1) {
    const count = snapshot.constraints[12 * node + 1] ?? 0;
    if (count !== 2 && count !== 4) continue;
    const denominator = snapshot.constraints[12 * node + 2] ?? 0;
    let expected = 0;
    for (let term = 0; term < count; term += 1) expected +=
      (snapshot.constraints[12 * node + 8 + term] ?? 0)
      * fromBits(snapshot.nodalPhi[2 * (snapshot.constraints[12 * node + 4 + term] ?? 0) + bank] ?? 0);
    const error = Math.abs(fromBits(snapshot.nodalPhi[2 * node + bank] ?? 0) - expected / denominator);
    if (error > maximumStoredConstraintError) {
      maximumStoredConstraintError = error; maximumStoredConstraintNode = node;
    }
  }
  const maxConstraint = maximumStoredConstraintNode < 0 ? undefined : {
    node: maximumStoredConstraintNode,
    header: Array.from(snapshot.constraints.slice(12 * maximumStoredConstraintNode,
      12 * maximumStoredConstraintNode + 4)),
    masters: Array.from(snapshot.constraints.slice(12 * maximumStoredConstraintNode + 4,
      12 * maximumStoredConstraintNode + 8)),
    masterConstraintCounts: Array.from(snapshot.constraints.slice(12 * maximumStoredConstraintNode + 4,
      12 * maximumStoredConstraintNode + 8)).map(master => snapshot.constraints[12 * master + 1] ?? 0),
    stored: fromBits(snapshot.nodalPhi[2 * maximumStoredConstraintNode + bank] ?? 0),
    masterValues: Array.from(snapshot.constraints.slice(12 * maximumStoredConstraintNode + 4,
      12 * maximumStoredConstraintNode + 8)).map(master =>
        fromBits(snapshot.nodalPhi[2 * master + bank] ?? 0)),
  };
  samples.push({ step, generation: snapshot.graphControl[5], topologyEpoch: snapshot.graphControl[0],
    phiControl: Array.from(snapshot.phiControl.slice(0, 12)),
    graphErrors: snapshot.graphControl[4], leafCount: publication.leafCount,
    nodeCount: publication.nodeCount, interfaceLeafCountsBySize: publication.interfaceLeafCountsBySize,
    constrainedNodeCount: publication.constrainedNodeCount,
    maximumStoredConstraintError: publication.maximumStoredConstraintError,
    maximumStoredConstraint: maxConstraint,
    maximumRendererCornerError: publication.maximumRendererCornerError,
    maximumSharedNodeMismatch: geometry.maximumSharedNodeMismatch,
    activeCubeCount: geometry.activeCubeCount, zeroSetExtentsCells: geometry.zeroSetExtentsCells,
    analyticNodeErrorsByLeafSize: publication.analyticNodeErrorsByLeafSize,
    acceptedAdvanceValid: receipt.acceptedAdvanceValid,
    rendererValidRows: receipt.rendererValidRows,
    volumeTransaction: receipt.volumeTransaction,
  });
}
solver.destroy(); await device.queue.onSubmittedWorkDone(); device.destroy();
console.log(JSON.stringify({ phase: "dam-adaptive-surface", dimensions, validationErrors, samples }, null, 2));
assert.deepEqual(validationErrors, []);
