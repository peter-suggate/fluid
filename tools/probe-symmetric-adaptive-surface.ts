/**
 * Dawn authority probe for factor-one symmetric-expansion surface quality.
 *
 * This intentionally reads the accepted adaptive graph rather than a browser
 * frame.  At construction it compares the published nodal phi with the exact
 * authored brick SDF; at every requested step it audits interface-leaf spans,
 * hanging-node constraints, and the renderer's copied corner values.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-webgpu-exclusive.ts \
 *     --import tsx tools/probe-symmetric-adaptive-surface.ts
 *
 * Environment:
 *   FLUID_SURFACE_AUDIT_STEPS  accepted steps to inspect (default 4)
 *   FLUID_SURFACE_AUDIT_COMPACT  3 emits one physics-gate summary line
 */
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
import { getScenePreset, SYMMETRIC_EXPANSION_METHOD_PROFILE } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { initialOctreeNodalLevelSet } from "../lib/methods/octree-shared/webgpu-octree";
import { unpackAdaptiveMassReceipt } from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-mass";
import { unpackAdaptivePhiReceipt } from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-phi";

const steps = Number(process.env.FLUID_SURFACE_AUDIT_STEPS ?? 4);
const redistanceGeometryAudit = process.env.FLUID_SURFACE_AUDIT_REDISTANCE_GEOMETRY === "1";
assert.ok(Number.isSafeInteger(steps) && steps >= 0,
  "FLUID_SURFACE_AUDIT_STEPS must be a non-negative integer");

const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([
  `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ...(process.env.FLUID_WEBGPU_ADAPTER
    ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
]);
Object.defineProperty(globalThis, "navigator", {
  configurable: true, value: { gpu },
});
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU did not expose an adapter");
const requiredFeatures: GPUFeatureName[] = ["subgroups"];
if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
const device = await adapter.requestDevice({
  requiredFeatures,
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => {
  validationErrors.push(event.error.message);
});

const scene = getScenePreset("symmetric-expansion").create();
const dt = 0.004;
scene.numerics.fixedDt_s = dt;
scene.numerics.maxDt_s = dt;
const solver = await losassoMethod.createSolverAsync!(device, scene, "balanced", {
  ...losassoMethod.presetFor("balanced"),
  ...SYMMETRIC_EXPANSION_METHOD_PROFILE.overrides,
  secondaryParticles: "off",
}, undefined, () => {}) as GPUSolverInstance;
const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
const exactNodalPhi = initialOctreeNodalLevelSet(scene, {
  nx: dimensions[0], ny: dimensions[1], nz: dimensions[2],
});
assert.ok(exactNodalPhi, "symmetric expansion must expose an exact nodal seed");
const nodeWidth = dimensions[0] + 1;
const nodeHeight = dimensions[1] + 1;
const analyticPhi = (x: number, y: number, z: number) =>
  exactNodalPhi[x + nodeWidth * (y + nodeHeight * z)]!;
const projection = (solver as unknown as {
  octreeProjection?: {
    // The analysis snapshot type declares its word arrays as `ArrayLike<number>`.
    // The projection publishes concrete typed arrays (webgpu-octree.ts
    // `readAdaptiveSurfacePublicationDiagnostics`), which the D4 audits below
    // slice and hand to typed-array helpers, plus the candidate-bank and
    // face-velocity rows the analysis snapshot does not carry at all.
    readAdaptiveSurfacePublicationDiagnostics(): Promise<Parameters<
      typeof analyzeAdaptiveSurfacePublication>[0] & {
        graphControl: Uint32Array;
        phiControl: Uint32Array;
        nodes: Uint32Array;
        constraints: Uint32Array;
        adjacency: Uint32Array;
        phiReceipts: Uint32Array;
        nodalVelocity: Uint32Array;
        authorityControl: Uint32Array;
        faceGeometry: Uint32Array;
        advectedFaceVelocity: Uint32Array;
        predictedFaceVelocity: Uint32Array;
        projectedFaceVelocity: Uint32Array;
        extendedFaceVelocity: Uint32Array;
        candidateGraphControl: Uint32Array;
        candidateNodes: Uint32Array;
        candidateNodalPhi: Uint32Array;
        candidateNodalVelocity: Uint32Array;
        candidateAuthorityControl: Uint32Array;
        candidateFaceGeometry: Uint32Array;
        candidateExtendedFaceVelocity: Uint32Array;
        redistanceDistanceA: Float32Array;
        acceptedMass: Float32Array;
        leafRhoPhi: Float32Array;
      } | undefined>;
    readLosassoAuthorityDiagnostics(): Promise<Readonly<{
      adaptiveGraph: readonly number[];
      candidateAdaptiveGraph: readonly number[];
      ownerCandidate: readonly number[];
      frontierControl: readonly number[];
      adaptivePhi: readonly number[];
      adaptiveMassReceipts: readonly number[];
    }> | undefined>;
    readPowerFrontierFailure(): Promise<Record<string, number[]> | undefined>;
  };
}).octreeProjection;
assert.ok(projection, "octree solver did not expose its diagnostic projection");

const auditSurfaceDensity = (snapshot: Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>) => {
  assert.ok(snapshot, "adaptive density audit requires a publication");
  const cellSize_m = solver.info.cellSize_m;
  const bins = { empty: 0, airSide: 0, liquidSide: 0, full: 0, overfull: 0 };
  let airSideMass_m3 = 0;
  let liquidSideMass_m3 = 0;
  let thresholdLeafVolume_m3 = 0;
  let partialLeafVolume_m3 = 0;
  let minimumRho = Number.POSITIVE_INFINITY;
  let maximumRho = Number.NEGATIVE_INFINITY;
  let floorWetCells = 0;
  let floorRhoCellSum = 0;
  let floorSubgridCells = 0;
  const wallMaximumWetHeight_cells = {
    negativeX: 0, positiveX: 0, negativeZ: 0, positiveZ: 0,
  };
  const wallMaximumRho = {
    negativeX: 0, positiveX: 0, negativeZ: 0, positiveZ: 0,
  };
  for (let leaf = 0; leaf < snapshot.graphControl[1]!; leaf += 1) {
    const mass_m3 = snapshot.acceptedMass[leaf]!;
    const span = snapshot.leaves[16 * leaf + 3]!;
    const volume_m3 = Math.pow(span * cellSize_m, 3);
    // leafRhoPhi is a shared derived-output scratch and may contain the most
    // recently materialized candidate bank. Reconstruct accepted rho from the
    // accepted graph-owned integral mass and its exact dyadic leaf volume.
    const rho = mass_m3 / volume_m3;
    minimumRho = Math.min(minimumRho, rho);
    maximumRho = Math.max(maximumRho, rho);
    if (rho <= 1e-5) bins.empty += 1;
    else if (rho < 0.5) bins.airSide += 1;
    else if (rho < 1 - 1e-5) bins.liquidSide += 1;
    else if (rho <= 1 + 1e-5) bins.full += 1;
    else bins.overfull += 1;
    if (rho < 0.5) airSideMass_m3 += mass_m3;
    else {
      liquidSideMass_m3 += mass_m3;
      thresholdLeafVolume_m3 += volume_m3;
    }
    if (rho > 1e-5 && rho < 1 - 1e-5) partialLeafVolume_m3 += volume_m3;
    const originX = snapshot.leaves[16 * leaf]!, originY = snapshot.leaves[16 * leaf + 1]!,
      originZ = snapshot.leaves[16 * leaf + 2]!;
    if (originY === 0) {
      const floorCells = span * span;
      floorRhoCellSum += rho * floorCells;
      if (rho >= 0.5) floorWetCells += floorCells;
      else if (rho > 1e-5) floorSubgridCells += floorCells;
    }
    if (rho >= 0.5) {
      const top = originY + span;
      if (originX === 0) wallMaximumWetHeight_cells.negativeX = Math.max(
        wallMaximumWetHeight_cells.negativeX, top);
      if (originX + span === dimensions[0]) wallMaximumWetHeight_cells.positiveX = Math.max(
        wallMaximumWetHeight_cells.positiveX, top);
      if (originZ === 0) wallMaximumWetHeight_cells.negativeZ = Math.max(
        wallMaximumWetHeight_cells.negativeZ, top);
      if (originZ + span === dimensions[2]) wallMaximumWetHeight_cells.positiveZ = Math.max(
        wallMaximumWetHeight_cells.positiveZ, top);
    }
    if (originX === 0) wallMaximumRho.negativeX = Math.max(wallMaximumRho.negativeX, rho);
    if (originX + span === dimensions[0]) wallMaximumRho.positiveX = Math.max(
      wallMaximumRho.positiveX, rho);
    if (originZ === 0) wallMaximumRho.negativeZ = Math.max(wallMaximumRho.negativeZ, rho);
    if (originZ + span === dimensions[2]) wallMaximumRho.positiveZ = Math.max(
      wallMaximumRho.positiveZ, rho);
  }
  const floorCellCount = dimensions[0] * dimensions[2];
  return Object.freeze({ bins, minimumRho, maximumRho, airSideMass_m3,
    liquidSideMass_m3, thresholdLeafVolume_m3, partialLeafVolume_m3,
    floorWetCells, floorWetFraction: floorWetCells / floorCellCount,
    floorSubgridCells, floorMeanRho: floorRhoCellSum / floorCellCount,
    wallMaximumWetHeight_cells, wallMaximumRho });
};

const readOwnerTopologyBanks = async () => {
  const arena = (projection as unknown as { powerOwnerArena?: GPUBuffer }).powerOwnerArena;
  if (!arena) return undefined;
  const staging = device.createBuffer({ label: "Probe owner topology banks", size: arena.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Copy probe owner topology banks" });
  encoder.copyBufferToBuffer(arena, 0, staging, 0, arena.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();
  if (words.length < 16 || words[15] !== 0x4f57_4e52) return undefined;
  const capacity = words[3]!, logicalCount = words[4]!, payloadA = words[6]!;
  const directoryA = 16 + 4 * capacity;
  const activeTable = words[10]! >>> 31;
  const summarize = (table: number) => {
    const directory = directoryA + table * logicalCount;
    const payload = payloadA + table * capacity * 512;
    const seen = new Set<string>();
    const identities: Array<readonly [number, number, number, number]> = [];
    const counts: Record<string, number> = {};
    let invalidCells = 0;
    for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) {
      for (let x = 0; x < dimensions[0]; x += 1) {
        const logical = (x >> 3) + Math.ceil(dimensions[0] / 8)
          * ((y >> 3) + Math.ceil(dimensions[1] / 8) * (z >> 3));
        const page = words[directory + logical]!;
        const local = (x & 7) + 8 * (y & 7) + 64 * (z & 7);
        const packed = page > 0 && page <= capacity ? words[payload + (page - 1) * 512 + local]! : 0;
        if ((packed & 0x8000_0000) === 0) { invalidCells += 1; continue; }
        const size = 1 << ((packed >>> 18) & 7);
        const brick = [x & ~7, y & ~7, z & ~7];
        const origin = [brick[0]! + (packed & 63) - 32,
          brick[1]! + ((packed >>> 6) & 63) - 32,
          brick[2]! + ((packed >>> 12) & 63) - 32];
        const key = `${origin.join(",")}:${size}`;
        if (!seen.has(key)) {
          seen.add(key);
          identities.push([origin[0]!, origin[1]!, origin[2]!, size]);
          counts[String(size)] = (counts[String(size)] ?? 0) + 1;
        }
      }
    }
    const transforms = [
      ([x, y, z, size]: readonly number[]) => [dimensions[0] - x - size, y, z, size],
      ([x, y, z, size]: readonly number[]) => [x, y, dimensions[2] - z - size, size],
      ([x, y, z, size]: readonly number[]) => [z, y, x, size],
    ] as const;
    const identityKey = ([x, y, z, size]: readonly number[]) => `${x},${y},${z}:${size}`;
    let firstMismatch: Readonly<{ transformIndex: number;
      source: readonly number[]; target: readonly number[] }> | undefined;
    const mismatchCounts = transforms.map((transform, transformIndex) =>
      identities.reduce((count, identity) => {
        const target = transform(identity);
        if (seen.has(identityKey(target))) return count;
        firstMismatch ??= Object.freeze({ transformIndex, source: identity, target });
        return count + 1;
      }, 0));
    return { leafCountsBySize: counts, invalidCells,
      d4: { mismatches: mismatchCounts.reduce((sum, count) => sum + count, 0),
        reflectXMismatches: mismatchCounts[0], reflectZMismatches: mismatchCounts[1],
        swapXZMismatches: mismatchCounts[2],
        ...(firstMismatch ? { firstMismatch } : {}) } };
  };
  return { activeTable, accepted: summarize(activeTable), candidate: summarize(1 - activeTable) };
};

// Probe-only stage capture. The normal cadence must remain enabled because a
// candidate publication advances the graph/phi tuple clocks. Copy accepted
// redistance state immediately before that candidate can reuse the scratch.
const stageCapture = (() => {
  if (!redistanceGeometryAudit) return undefined;
  const internal = projection as unknown as {
    losassoBackend?: {
      adaptivePhiSource?: { control: GPUBuffer; redistanceDistanceA: GPUBuffer };
      adaptiveSurfaceGraphSources?: { accepted: {
        control: GPUBuffer; leaves: GPUBuffer; nodes: GPUBuffer;
        constraints: GPUBuffer; phi: GPUBuffer;
      } };
    };
    encodeInactiveTopologyCandidateIfDue(encoder: GPUCommandEncoder): boolean;
  };
  const source = internal.losassoBackend?.adaptivePhiSource;
  const graph = internal.losassoBackend?.adaptiveSurfaceGraphSources?.accepted;
  assert.ok(source && graph, "adaptive redistance probe found no accepted source");
  const originals = { graphControl: graph.control, phiControl: source.control,
    leaves: graph.leaves, nodes: graph.nodes, constraints: graph.constraints,
    nodalPhi: graph.phi, redistanceDistanceA: source.redistanceDistanceA };
  const captured = Object.fromEntries(Object.entries(originals).map(([name, buffer]) =>
    [name, device.createBuffer({ label: `Probe accepted redistance ${name}`,
      size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })])) as
    Record<keyof typeof originals, GPUBuffer>;
  const originalCandidate = internal.encodeInactiveTopologyCandidateIfDue.bind(internal);
  internal.encodeInactiveTopologyCandidateIfDue = (encoder: GPUCommandEncoder) => {
    for (const name of Object.keys(originals) as Array<keyof typeof originals>) {
      encoder.copyBufferToBuffer(originals[name], 0, captured[name], 0, originals[name].size);
    }
    return originalCandidate(encoder);
  };
  const read = async () => {
    const names = Object.keys(captured) as Array<keyof typeof captured>;
    const offsets = new Map<keyof typeof captured, number>();
    let bytes = 0;
    for (const name of names) { offsets.set(name, bytes); bytes += captured[name].size; }
    const staging = device.createBuffer({ label: "Probe accepted redistance readback",
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: "Probe accepted redistance copy" });
    for (const name of names) encoder.copyBufferToBuffer(captured[name], 0, staging,
      offsets.get(name)!, captured[name].size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const result = {
      graphControl: Uint32Array.from(new Uint32Array(mapped, offsets.get("graphControl")!,
        originals.graphControl.size / 4)),
      phiControl: Uint32Array.from(new Uint32Array(mapped, offsets.get("phiControl")!,
        originals.phiControl.size / 4)),
      leaves: Uint32Array.from(new Uint32Array(mapped, offsets.get("leaves")!,
        originals.leaves.size / 4)),
      nodes: Uint32Array.from(new Uint32Array(mapped, offsets.get("nodes")!,
        originals.nodes.size / 4)),
      constraints: Uint32Array.from(new Uint32Array(mapped, offsets.get("constraints")!,
        originals.constraints.size / 4)),
      nodalPhi: Uint32Array.from(new Uint32Array(mapped, offsets.get("nodalPhi")!,
        originals.nodalPhi.size / 4)),
      redistanceDistanceA: Float32Array.from(new Float32Array(mapped,
        offsets.get("redistanceDistanceA")!, originals.redistanceDistanceA.size / 4)),
    };
    staging.unmap(); staging.destroy();
    return result;
  };
  const destroy = () => { for (const buffer of Object.values(captured)) buffer.destroy(); };
  return Object.freeze({ read, destroy });
})();

const samples: Array<Record<string, unknown>> = [];
const topFeatureGroups = [
  { name: "face-center", points: [[16, 16]] },
  { name: "edge-midpoints", points: [[8, 16], [24, 16], [16, 8], [16, 24]] },
  { name: "corners", points: [[8, 8], [8, 24], [24, 8], [24, 24]] },
  // Exact edge/corner columns also lie on the body's vertical zero-set wall,
  // so their highest-zero scan becomes discontinuous as soon as the top node
  // moves positive. One node inward measures the top sheet without that
  // geometrically ambiguous vertical branch.
  { name: "edge-midpoints-inset-one", points: [[9, 16], [23, 16], [16, 9], [16, 23]] },
  { name: "corners-inset-one", points: [[9, 9], [9, 23], [23, 9], [23, 23]] },
] as const;
const bitFloat = (bits: number) => new Float32Array(Uint32Array.of(bits).buffer)[0]!;
const floatBits = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0]!;
const auditRedistanceZeroSet = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0,
    Math.floor(snapshot.leaves.length / 16));
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodalPhi.length / 2), snapshot.redistanceDistanceA.length);
  const currentBank = (snapshot.phiControl[6] ?? 0) & 1;
  // A probe-only cadence hold can make the transaction reject after the
  // attempted target was fully computed. distanceA belongs to that target,
  // so compare like with like instead of silently reading the retained bank.
  const bank = snapshot.phiReceipts[22] === 1 ? currentBank : 1 - currentBank;
  const edgeCorners = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3],
    [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]] as const;
  const crossing = (a: number, b: number): number | undefined => {
    if (a === 0 && b === 0) return undefined;
    if (a === 0) return 0;
    if (b === 0) return 1;
    if ((a < 0) === (b < 0)) return undefined;
    return -a / (b - a);
  };
  let mixedLeaves = 0, mixedLeavesWithUnfrozenCorners = 0, comparedCorners = 0;
  let changedCornerBits = 0, maximumCornerDelta_m = 0;
  let comparedEdgeCrossings = 0, appearedOrDisappearedEdgeCrossings = 0;
  let maximumEdgeCrossingDisplacement_m = 0;
  let preLiquidCells = 0, postLiquidCells = 0;
  const quadrature = 8;
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const span = snapshot.leaves[16 * leaf + 3] ?? 0;
    if (span === 0) continue;
    const pre = new Array<number>(8), post = new Array<number>(8);
    let postMinimum = Number.POSITIVE_INFINITY, postMaximum = Number.NEGATIVE_INFINITY;
    let frozenCorners = 0;
    for (let corner = 0; corner < 8; corner += 1) {
      const slot = snapshot.leaves[16 * leaf + 8 + corner] ?? 0xffff_ffff;
      const value = slot < nodeCount ? bitFloat(snapshot.nodalPhi[2 * slot + bank] ?? 0)
        : Number.NaN;
      post[corner] = value;
      postMinimum = Math.min(postMinimum, value); postMaximum = Math.max(postMaximum, value);
      const encoded = slot < nodeCount ? snapshot.redistanceDistanceA[slot]! : Number.NaN;
      const frozen = Number.isFinite(encoded) && (floatBits(encoded) & 0x8000_0000) !== 0;
      frozenCorners += Number(frozen);
      const magnitude = Math.abs(encoded);
      pre[corner] = frozen ? (value === 0 ? 0 : Math.sign(value) * magnitude) : value;
    }
    const postMixed = postMinimum < 0 && postMaximum > 0;
    const preMinimum = Math.min(...pre), preMaximum = Math.max(...pre);
    const preMixed = preMinimum < 0 && preMaximum > 0;
    const postTouches = postMinimum <= 0 && postMaximum >= 0;
    const preTouches = preMinimum <= 0 && preMaximum >= 0;
    if (postMixed || preMixed) {
      mixedLeaves += 1;
      if (frozenCorners !== 8) mixedLeavesWithUnfrozenCorners += 1;
      if (frozenCorners === 8) for (let corner = 0; corner < 8; corner += 1) {
        comparedCorners += 1;
        changedCornerBits += Number(floatBits(pre[corner]!) !== floatBits(post[corner]!));
        maximumCornerDelta_m = Math.max(maximumCornerDelta_m,
          Math.abs(pre[corner]! - post[corner]!));
      }
      for (const [a, b] of edgeCorners) {
        const before = crossing(pre[a]!, pre[b]!);
        const after = crossing(post[a]!, post[b]!);
        if (before === undefined || after === undefined) {
          appearedOrDisappearedEdgeCrossings += Number(before !== after);
          continue;
        }
        comparedEdgeCrossings += 1;
        maximumEdgeCrossingDisplacement_m = Math.max(maximumEdgeCrossingDisplacement_m,
          Math.abs(before - after) * span * scene.container.width_m / dimensions[0]);
      }
    }
    if (!preTouches && !postTouches) {
      const cells = span ** 3;
      preLiquidCells += Number(preMaximum < 0) * cells;
      postLiquidCells += Number(postMaximum < 0) * cells;
      continue;
    }
    for (let z = 0; z < quadrature; z += 1) for (let y = 0; y < quadrature; y += 1)
      for (let x = 0; x < quadrature; x += 1) {
        const t = [(x + .5) / quadrature, (y + .5) / quadrature,
          (z + .5) / quadrature] as const;
        let before = 0, after = 0;
        for (let corner = 0; corner < 8; corner += 1) {
          const weight = (corner & 1 ? t[0] : 1 - t[0])
            * (corner & 2 ? t[1] : 1 - t[1])
            * (corner & 4 ? t[2] : 1 - t[2]);
          before += weight * pre[corner]!; after += weight * post[corner]!;
        }
        const sampleVolume = span ** 3 / quadrature ** 3;
        preLiquidCells += Number(before <= 0) * sampleVolume;
        postLiquidCells += Number(after <= 0) * sampleVolume;
      }
  }
  const cellVolume_m3 = (scene.container.width_m / dimensions[0]) ** 3;
  return Object.freeze({ mixedLeaves, mixedLeavesWithUnfrozenCorners, comparedCorners,
    changedCornerBits, maximumCornerDelta_m, comparedEdgeCrossings,
    appearedOrDisappearedEdgeCrossings, maximumEdgeCrossingDisplacement_m,
    quadraturePerAxis: quadrature, preGeometricVolume_m3: preLiquidCells * cellVolume_m3,
    postGeometricVolume_m3: postLiquidCells * cellVolume_m3,
    signedGeometricDrift_m3: (preLiquidCells - postLiquidCells) * cellVolume_m3 });
};
const auditVelocityMagnitude = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodalVelocity.length / 8));
  let acceptedMaximum_m_s = 0, predictorMaximum_m_s = 0;
  for (let slot = 0; slot < nodeCount; slot += 1) {
    for (let field = 0; field < 2; field += 1) {
      const base = 8 * slot + 4 * field;
      if (((snapshot.nodalVelocity[base + 3] ?? 0) & 7) !== 7) continue;
      const magnitude = Math.hypot(bitFloat(snapshot.nodalVelocity[base] ?? 0),
        bitFloat(snapshot.nodalVelocity[base + 1] ?? 0),
        bitFloat(snapshot.nodalVelocity[base + 2] ?? 0));
      if (field === 0) acceptedMaximum_m_s = Math.max(acceptedMaximum_m_s, magnitude);
      else predictorMaximum_m_s = Math.max(predictorMaximum_m_s, magnitude);
    }
  }
  return Object.freeze({ acceptedMaximum_m_s, predictorMaximum_m_s });
};
const auditNodalD4 = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const [nx, ny, nz] = snapshot.dimensions;
  const dx = nx + 1, dy = ny + 1;
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodes.length / 4), Math.floor(snapshot.nodalPhi.length / 2));
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  const byItem = new Map<number, number>();
  for (let slot = 0; slot < nodeCount; slot += 1) {
    byItem.set(snapshot.nodes[4 * slot] ?? 0xffff_ffff, slot);
  }
  const transforms = [
    (x: number, y: number, z: number) => [nx - x, y, z] as const,
    (x: number, y: number, z: number) => [x, y, nz - z] as const,
    (x: number, y: number, z: number) => [z, y, x] as const,
  ];
  const mismatchCounts = [0, 0, 0];
  let firstMismatch: Readonly<Record<string, unknown>> | undefined;
  let maximumMismatch: Readonly<Record<string, unknown>> | undefined;
  let mismatches = 0, maximumAbsoluteError_m = 0;
  for (let slot = 0; slot < nodeCount; slot += 1) {
    const item = snapshot.nodes[4 * slot] ?? 0;
    const z = Math.floor(item / (dx * dy));
    const remainder = item - z * dx * dy;
    const y = Math.floor(remainder / dx), x = remainder - y * dx;
    const sourceBits = snapshot.nodalPhi[2 * slot + bank] ?? 0;
    for (let transformIndex = 0; transformIndex < transforms.length; transformIndex += 1) {
      const transform = transforms[transformIndex]!;
      const [tx, ty, tz] = transform(x, y, z);
      const target = byItem.get(tx + dx * (ty + dy * tz));
      if (target === undefined) {
        mismatches += 1; mismatchCounts[transformIndex]! += 1; continue;
      }
      const targetBits = snapshot.nodalPhi[2 * target + bank] ?? 0;
      if (targetBits === sourceBits) continue;
      mismatches += 1;
      mismatchCounts[transformIndex]! += 1;
      firstMismatch ??= Object.freeze({ transformIndex, source: [x, y, z],
        target: [tx, ty, tz], sourceValue_m: bitFloat(sourceBits),
        targetValue_m: bitFloat(targetBits) });
      const absoluteError_m = Math.abs(bitFloat(sourceBits) - bitFloat(targetBits));
      if (absoluteError_m > maximumAbsoluteError_m) {
        maximumAbsoluteError_m = absoluteError_m;
        maximumMismatch = Object.freeze({ transformIndex, source: [x, y, z],
          target: [tx, ty, tz], sourceValue_m: bitFloat(sourceBits),
          targetValue_m: bitFloat(targetBits) });
      }
    }
  }
  return Object.freeze({ mismatches, maximumAbsoluteError_m,
    reflectXMismatches: mismatchCounts[0], reflectZMismatches: mismatchCounts[1],
    swapXZMismatches: mismatchCounts[2], ...(firstMismatch ? { firstMismatch } : {}),
    ...(maximumMismatch ? { maximumMismatch } : {}) });
};
const auditNodalPhiRange = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodalPhi.length / 2));
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  let minimum_m = Number.POSITIVE_INFINITY, maximum_m = Number.NEGATIVE_INFINITY;
  let exactZeroNodes = 0, withinTwoCellsNodes = 0;
  for (let slot = 0; slot < nodeCount; slot += 1) {
    const value = bitFloat(snapshot.nodalPhi[2 * slot + bank] ?? 0);
    minimum_m = Math.min(minimum_m, value);
    maximum_m = Math.max(maximum_m, value);
    if (value === 0) exactZeroNodes += 1;
    if (Math.abs(value) < 0.1) withinTwoCellsNodes += 1;
  }
  return Object.freeze({ minimum_m, maximum_m, exactZeroNodes, withinTwoCellsNodes });
};
const auditAdjacencyD4 = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const [nx, ny, nz] = snapshot.dimensions;
  const dx = nx + 1, dy = ny + 1;
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodes.length / 4), Math.floor(snapshot.adjacency.length / 12));
  const byItem = new Map<number, number>();
  const position = (item: number) => {
    const z = Math.floor(item / (dx * dy));
    const remainder = item - z * dx * dy;
    const y = Math.floor(remainder / dx);
    return [remainder - y * dx, y, z] as const;
  };
  const item = (x: number, y: number, z: number) => x + dx * (y + dy * z);
  for (let slot = 0; slot < nodeCount; slot += 1) {
    byItem.set(snapshot.nodes[4 * slot] ?? 0xffff_ffff, slot);
  }
  const transforms = [
    (x: number, y: number, z: number) => [nx - x, y, z] as const,
    (x: number, y: number, z: number) => [x, y, nz - z] as const,
    (x: number, y: number, z: number) => [z, y, x] as const,
  ];
  const directionMaps = [
    [3, 1, 2, 0, 4, 5],
    [0, 1, 5, 3, 4, 2],
    [2, 1, 0, 5, 4, 3],
  ] as const;
  const mismatchCounts = [0, 0, 0];
  let mismatches = 0;
  let firstMismatch: Readonly<Record<string, unknown>> | undefined;
  for (let slot = 0; slot < nodeCount; slot += 1) {
    const sourceItem = snapshot.nodes[4 * slot] ?? 0;
    const [x, y, z] = position(sourceItem);
    for (let transformIndex = 0; transformIndex < transforms.length; transformIndex += 1) {
      const [tx, ty, tz] = transforms[transformIndex]!(x, y, z);
      const target = byItem.get(item(tx, ty, tz));
      if (target === undefined) continue;
      for (let direction = 0; direction < 6; direction += 1) {
        const targetDirection = directionMaps[transformIndex]![direction]!;
        const sourceNeighbor = snapshot.adjacency[12 * slot + direction] ?? 0xffff_ffff;
        const targetNeighbor = snapshot.adjacency[12 * target + targetDirection]
          ?? 0xffff_ffff;
        let expectedNeighbor = 0xffff_ffff;
        if (sourceNeighbor !== 0xffff_ffff && sourceNeighbor < nodeCount) {
          const [sx, sy, sz] = position(snapshot.nodes[4 * sourceNeighbor] ?? 0);
          const [ex, ey, ez] = transforms[transformIndex]!(sx, sy, sz);
          expectedNeighbor = byItem.get(item(ex, ey, ez)) ?? 0xffff_ffff;
        }
        const sourceSpan = snapshot.adjacency[12 * slot + 6 + direction]
          ?? 0xffff_ffff;
        const targetSpan = snapshot.adjacency[12 * target + 6 + targetDirection]
          ?? 0xffff_ffff;
        if (expectedNeighbor === targetNeighbor && sourceSpan === targetSpan) continue;
        mismatches += 1;
        mismatchCounts[transformIndex]! += 1;
        firstMismatch ??= Object.freeze({ transformIndex, source: [x, y, z],
          target: [tx, ty, tz], direction, targetDirection, sourceNeighbor,
          expectedNeighbor, targetNeighbor, sourceSpan, targetSpan });
      }
    }
  }
  return Object.freeze({ mismatches, reflectXMismatches: mismatchCounts[0],
    reflectZMismatches: mismatchCounts[1], swapXZMismatches: mismatchCounts[2],
    ...(firstMismatch ? { firstMismatch } : {}) });
};
const auditConstraintsD4 = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const [nx, ny, nz] = snapshot.dimensions;
  const dx = nx + 1, dy = ny + 1;
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0,
    Math.floor(snapshot.nodes.length / 4), Math.floor(snapshot.constraints.length / 12));
  const byItem = new Map<number, number>();
  const position = (nodeItem: number) => {
    const z = Math.floor(nodeItem / (dx * dy));
    const remainder = nodeItem - z * dx * dy;
    const y = Math.floor(remainder / dx);
    return [remainder - y * dx, y, z] as const;
  };
  const item = (x: number, y: number, z: number) => x + dx * (y + dy * z);
  for (let slot = 0; slot < nodeCount; slot += 1) {
    byItem.set(snapshot.nodes[4 * slot] ?? 0xffff_ffff, slot);
  }
  const transforms = [
    (x: number, y: number, z: number) => [nx - x, y, z] as const,
    (x: number, y: number, z: number) => [x, y, nz - z] as const,
    (x: number, y: number, z: number) => [z, y, x] as const,
  ];
  const mismatchCounts = [0, 0, 0];
  let mismatches = 0;
  let firstMismatch: Readonly<Record<string, unknown>> | undefined;
  for (let slot = 0; slot < nodeCount; slot += 1) {
    const sourceItem = snapshot.nodes[4 * slot] ?? 0;
    const [x, y, z] = position(sourceItem);
    for (let transformIndex = 0; transformIndex < transforms.length; transformIndex += 1) {
      const transform = transforms[transformIndex]!;
      const [tx, ty, tz] = transform(x, y, z);
      const target = byItem.get(item(tx, ty, tz));
      if (target === undefined) continue;
      const sourceHeader = Array.from(snapshot.constraints.slice(12 * slot, 12 * slot + 4));
      const targetHeader = Array.from(snapshot.constraints.slice(12 * target, 12 * target + 4));
      const sourceCount = Math.min(sourceHeader[1] ?? 0, 4);
      const targetCount = Math.min(targetHeader[1] ?? 0, 4);
      const sourceTerms: string[] = [];
      for (let term = 0; term < sourceCount; term += 1) {
        const master = snapshot.constraints[12 * slot + 4 + term] ?? 0xffff_ffff;
        const weight = snapshot.constraints[12 * slot + 8 + term] ?? 0;
        if (master >= nodeCount) {
          sourceTerms.push(String(master) + ":" + String(weight));
          continue;
        }
        const [mx, my, mz] = position(snapshot.nodes[4 * master] ?? 0);
        const [emx, emy, emz] = transform(mx, my, mz);
        const expected = byItem.get(item(emx, emy, emz)) ?? 0xffff_ffff;
        sourceTerms.push(String(expected) + ":" + String(weight));
      }
      const targetTerms: string[] = [];
      for (let term = 0; term < targetCount; term += 1) {
        const master = snapshot.constraints[12 * target + 4 + term] ?? 0xffff_ffff;
        const weight = snapshot.constraints[12 * target + 8 + term] ?? 0;
        targetTerms.push(String(master) + ":" + String(weight));
      }
      sourceTerms.sort(); targetTerms.sort();
      const matches = sourceHeader.join(",") === targetHeader.join(",")
        && sourceTerms.join(",") === targetTerms.join(",");
      if (matches) continue;
      mismatches += 1;
      mismatchCounts[transformIndex]! += 1;
      firstMismatch ??= Object.freeze({ transformIndex, source: [x, y, z],
        target: [tx, ty, tz], sourceHeader, targetHeader, sourceTerms, targetTerms });
    }
  }
  return Object.freeze({ mismatches, reflectXMismatches: mismatchCounts[0],
    reflectZMismatches: mismatchCounts[1], swapXZMismatches: mismatchCounts[2],
    ...(firstMismatch ? { firstMismatch } : {}) });
};
const auditNodalVelocityD4 = (graphControl: Uint32Array, nodes: Uint32Array,
  nodalVelocity: Uint32Array) => {
  const [nx, ny, nz] = dimensions;
  const dx = nx + 1, dy = ny + 1;
  const nodeCount = Math.min(graphControl[2] ?? 0, Math.floor(nodes.length / 4),
    Math.floor(nodalVelocity.length / 8));
  const position = (node: number) => {
    const item = nodes[4 * node] ?? 0;
    return [item % dx, Math.floor(item / dx) % dy,
      Math.floor(item / (dx * dy))] as const;
  };
  const byPosition = new Map<string, number>();
  for (let node = 0; node < nodeCount; node += 1) {
    byPosition.set(position(node).join(":"), node);
  }
  const transforms = [
    { position: ([x, y, z]: readonly number[]) => [nx - x, y, z] as const,
      value: ([x, y, z]: readonly number[]) => [-x, y, z] as const },
    { position: ([x, y, z]: readonly number[]) => [x, y, nz - z] as const,
      value: ([x, y, z]: readonly number[]) => [x, y, -z] as const },
    { position: ([x, y, z]: readonly number[]) => [z, y, x] as const,
      value: ([x, y, z]: readonly number[]) => [z, y, x] as const },
  ] as const;
  let missingNodes = 0, maskMismatches = 0, valueMismatches = 0;
  let maximumAbsoluteError_m_s = 0;
  let firstMissing: Readonly<{ bank: number; source: readonly number[];
    target: readonly number[] }> | undefined;
  for (let bank = 0; bank < 2; bank += 1) for (const transform of transforms) {
    for (let node = 0; node < nodeCount; node += 1) {
      const sourcePosition = position(node);
      const targetPosition = transform.position(sourcePosition);
      const target = byPosition.get(targetPosition.join(":"));
      if (target === undefined) {
        missingNodes += 1;
        firstMissing ??= Object.freeze({ bank, source: sourcePosition, target: targetPosition });
        continue;
      }
      const sourceAt = 8 * node + 4 * bank, targetAt = 8 * target + 4 * bank;
      maskMismatches += Number((nodalVelocity[sourceAt + 3]! & 7)
        !== (nodalVelocity[targetAt + 3]! & 7));
      const expected = transform.value([bitFloat(nodalVelocity[sourceAt]!),
        bitFloat(nodalVelocity[sourceAt + 1]!), bitFloat(nodalVelocity[sourceAt + 2]!)]);
      for (let component = 0; component < 3; component += 1) {
        const actual = bitFloat(nodalVelocity[targetAt + component]!);
        if (Object.is(expected[component], actual)) continue;
        valueMismatches += 1;
        maximumAbsoluteError_m_s = Math.max(maximumAbsoluteError_m_s,
          Math.abs(expected[component]! - actual));
      }
    }
  }
  return Object.freeze({ missingNodes, maskMismatches, valueMismatches,
    maximumAbsoluteError_m_s, ...(firstMissing ? { firstMissing } : {}) });
};
const auditFaceVelocityD4 = (control: Uint32Array, faceGeometry: Uint32Array,
  values: Uint32Array) => {
  const count = Math.min(control[2] ?? 0, Math.floor(faceGeometry.length / 4), values.length);
  const geometry = (face: number) => {
    const packed = faceGeometry[4 * face] ?? 0;
    return { axis: packed & 3, span: 1 << (packed >>> 2),
      q: [faceGeometry[4 * face + 1] ?? 0, faceGeometry[4 * face + 2] ?? 0,
        faceGeometry[4 * face + 3] ?? 0] as const };
  };
  const key = (g: ReturnType<typeof geometry>) => `${g.axis}:${g.span}:${g.q.join(":")}`;
  const byGeometry = new Map<string, number>();
  for (let face = 0; face < count; face += 1) byGeometry.set(key(geometry(face)), face);
  const transforms = [
    { geometry: (g: ReturnType<typeof geometry>) => ({ ...g,
      q: [dimensions[0] - g.q[0] - (g.axis === 0 ? 0 : g.span), g.q[1], g.q[2]] as const }),
      sign: (axis: number) => axis === 0 ? -1 : 1 },
    { geometry: (g: ReturnType<typeof geometry>) => ({ ...g,
      q: [g.q[0], g.q[1], dimensions[2] - g.q[2] - (g.axis === 2 ? 0 : g.span)] as const }),
      sign: (axis: number) => axis === 2 ? -1 : 1 },
    { geometry: (g: ReturnType<typeof geometry>) => ({ axis: g.axis === 0 ? 2
      : g.axis === 2 ? 0 : 1, span: g.span, q: [g.q[2], g.q[1], g.q[0]] as const }),
      sign: () => 1 },
  ] as const;
  let missingFaces = 0, mismatches = 0, maximumAbsoluteError_m_s = 0;
  for (let face = 0; face < count; face += 1) for (const transform of transforms) {
    const sourceGeometry = geometry(face);
    const target = byGeometry.get(key(transform.geometry(sourceGeometry)));
    if (target === undefined) { missingFaces += 1; continue; }
    const expected = transform.sign(sourceGeometry.axis) * bitFloat(values[face]!);
    const actual = bitFloat(values[target]!);
    if (Object.is(expected, actual)) continue;
    mismatches += 1;
    maximumAbsoluteError_m_s = Math.max(maximumAbsoluteError_m_s, Math.abs(expected - actual));
  }
  return Object.freeze({ count, missingFaces, mismatches, maximumAbsoluteError_m_s });
};
const auditRefinementPair = (snapshot: NonNullable<Awaited<ReturnType<
  NonNullable<typeof projection>["readAdaptiveSurfacePublicationDiagnostics"]>>>) => {
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  const count = Math.min(snapshot.graphControl[1] ?? 0, Math.floor(snapshot.leaves.length / 16));
  const leaf = (row: number) => ({
    origin: [snapshot.leaves[16 * row]!, snapshot.leaves[16 * row + 1]!,
      snapshot.leaves[16 * row + 2]!] as const,
    span: snapshot.leaves[16 * row + 3]!,
    corners: Array.from({ length: 8 }, (_, corner) =>
      bitFloat(snapshot.nodalPhi[2 * snapshot.leaves[16 * row + 8 + corner]! + bank]!)),
  });
  const containing = (q: readonly number[]) => {
    for (let row = 0; row < count; row += 1) {
      const record = leaf(row);
      if (q.every((value, axis) => value >= record.origin[axis]!
        && value < record.origin[axis]! + record.span)) return { row, ...record };
    }
    return undefined;
  };
  return [[12, 4, 2], [18, 4, 2]].map((origin) => {
    const record = containing(origin.map((value) => value + 1));
    if (!record) return { origin, missing: true };
    return { origin, ownerOrigin: record.origin, ownerSpan: record.span,
      minimumPhi_m: Math.min(...record.corners), maximumPhi_m: Math.max(...record.corners),
      corners_m: record.corners };
  });
};
const capture = async (step: number) => {
  await device.queue.onSubmittedWorkDone();
  const snapshot = await projection.readAdaptiveSurfacePublicationDiagnostics();
  assert.ok(snapshot, "factor-one solver published no adaptive surface snapshot");
  const analysis = analyzeAdaptiveSurfacePublication(snapshot,
    step === 0 ? analyticPhi : undefined);
  const featureGeometry = analyzeAdaptiveSurfaceFeatureGeometry(
    snapshot, 8, topFeatureGroups);
  const phiReceipt = unpackAdaptivePhiReceipt(snapshot.phiReceipts);
  const nodalD4 = auditNodalD4(snapshot);
  const nodalPhiRange = auditNodalPhiRange(snapshot);
  const adjacencyD4 = auditAdjacencyD4(snapshot);
  const constraintsD4 = auditConstraintsD4(snapshot);
  const candidateNodalD4 = auditNodalD4({ ...snapshot,
    graphControl: snapshot.candidateGraphControl,
    nodes: snapshot.candidateNodes,
    nodalPhi: snapshot.candidateNodalPhi,
    phiControl: new Uint32Array(20),
  });
  const nodalSpeed = auditVelocityMagnitude(snapshot);
  const acceptedVelocityD4 = auditNodalVelocityD4(snapshot.graphControl,
    snapshot.nodes, snapshot.nodalVelocity);
  const candidateVelocityD4 = auditNodalVelocityD4(snapshot.candidateGraphControl,
    snapshot.candidateNodes, snapshot.candidateNodalVelocity);
  const faceVelocityD4 = Object.freeze({
    advected: auditFaceVelocityD4(snapshot.authorityControl, snapshot.faceGeometry,
      snapshot.advectedFaceVelocity),
    predicted: auditFaceVelocityD4(snapshot.authorityControl, snapshot.faceGeometry,
      snapshot.predictedFaceVelocity),
    projected: auditFaceVelocityD4(snapshot.authorityControl, snapshot.faceGeometry,
      snapshot.projectedFaceVelocity),
    extended: auditFaceVelocityD4(snapshot.authorityControl, snapshot.faceGeometry,
      snapshot.extendedFaceVelocity),
    candidateExtended: auditFaceVelocityD4(snapshot.candidateAuthorityControl,
      snapshot.candidateFaceGeometry, snapshot.candidateExtendedFaceVelocity),
  });
  const acceptedStage = stageCapture && step > 0 ? await stageCapture.read() : undefined;
  const redistanceZeroSet = redistanceGeometryAudit && acceptedStage
    ? auditRedistanceZeroSet({ ...snapshot, ...acceptedStage }) : undefined;
  const [stats, authority] = await Promise.all([
    solver.readStats() as unknown as Promise<Record<string, unknown>>,
    projection.readLosassoAuthorityDiagnostics(),
  ]);
  assert.ok(authority, "factor-one solver published no Losasso authority receipt");
  const massReceipt = unpackAdaptiveMassReceipt(authority.adaptiveMassReceipts);
  const surfaceDensity = auditSurfaceDensity(snapshot);
  samples.push({ step, time_s: step * dt, ...analysis, featureGeometry, phiReceipt,
    redistanceZeroSet,
    graphControl: snapshot.graphControl, phiControl: snapshot.phiControl,
    nodalD4, nodalPhiRange, adjacencyD4, constraintsD4, candidateNodalD4, nodalSpeed,
    acceptedVelocityD4,
    candidateVelocityD4, surfaceDensity,
    ...(step >= 82 ? { refinementPair: auditRefinementPair(snapshot) } : {}),
    faceVelocityD4,
    pressureRows: stats.pressureRequiredRows,
    acceptedMass_m3: massReceipt.acceptedMass_m3,
    transportedMass_m3: massReceipt.transportedMass_m3,
    signedTransportDrift_m3: massReceipt.signedTransportDrift_m3,
    massErrors: massReceipt.errors });
};

await capture(0);
for (let step = 1; step <= steps; step += 1) {
  while (!solver.advanceTo(step * dt, [])) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  await capture(step);
}
const finalAuthority = await projection.readLosassoAuthorityDiagnostics();
const finalFrontierFailure = await projection.readPowerFrontierFailure();
const finalOwnerTopologies = await readOwnerTopologyBanks();

solver.destroy();
stageCapture?.destroy();
await device.queue.onSubmittedWorkDone();
device.destroy();
const compact = process.env.FLUID_SURFACE_AUDIT_COMPACT;
const reportedSamples = compact === "1" || compact === "2"
  ? samples.map((sample) => {
    const receipt = sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>;
    const geometry = sample.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>;
    const graphControl = sample.graphControl as Uint32Array;
    const phiControl = sample.phiControl as Uint32Array;
    return { step: sample.step, time_s: sample.time_s,
      acceptedTopologyEpoch: graphControl[0],
      acceptedSurfaceGeneration: graphControl[5],
      candidateTopologyEpoch: phiControl[8],
      candidateSurfaceGeneration: phiControl[9],
      leafCount: sample.leafCount, nodeCount: sample.nodeCount,
      interfaceLeafCountsBySize: sample.interfaceLeafCountsBySize,
      coarseInterfaceLeafCount: sample.coarseInterfaceLeafCount,
      coarseStrictInterfaceLeafCount: sample.coarseStrictInterfaceLeafCount,
      coarseTouchingInterfaceLeafCount: sample.coarseTouchingInterfaceLeafCount,
      activeCubeCount: geometry.activeCubeCount,
      zeroSetExtentsCells: geometry.zeroSetExtentsCells,
      maximumSharedNodeMismatch: geometry.maximumSharedNodeMismatch,
      nodalD4: sample.nodalD4,
      nodalPhiRange: sample.nodalPhiRange,
      adjacencyD4: sample.adjacencyD4,
      constraintsD4: sample.constraintsD4,
      candidateNodalD4: sample.candidateNodalD4,
      acceptedVelocityD4: sample.acceptedVelocityD4,
      candidateVelocityD4: sample.candidateVelocityD4,
      ...(sample.refinementPair ? { refinementPair: sample.refinementPair } : {}),
      faceVelocityD4: sample.faceVelocityD4,
      nodalSpeed: sample.nodalSpeed,
      surfaceDensity: sample.surfaceDensity,
      ...(sample.redistanceZeroSet ? { redistanceZeroSet: sample.redistanceZeroSet } : {}),
      ...(compact === "1" ? { topFeatures: geometry.topFeatures } : {}),
      dt_s: receipt.dt_s,
      maximumTransportDelta_m: receipt.maximumTransportDelta_m,
      transportedNodes: receipt.transportedNodes,
      departurePhiFailures: receipt.departurePhiFailures,
      transportBand: receipt.transportBand,
      preRedistanceVolume_m3: receipt.measuredVolume_m3
        + receipt.signedRedistanceVolumeDrift_m3,
      measuredVolume_m3: receipt.measuredVolume_m3,
      targetVolume_m3: receipt.targetVolume_m3,
      signedRedistanceVolumeDrift_m3: receipt.signedRedistanceVolumeDrift_m3,
      redistanceResidual_m: receipt.redistanceResidual_m,
      redistanceReachedNodes: receipt.redistanceReachedNodes,
      redistanceSeedNodes: receipt.redistanceSeedNodes,
      candidateRedistanceConverged: receipt.candidate.converged,
      volumeTransaction: receipt.volumeTransaction,
      acceptedAdvanceValid: receipt.acceptedAdvanceValid,
      pressureRows: sample.pressureRows,
      acceptedMass_m3: sample.acceptedMass_m3,
      transportedMass_m3: sample.transportedMass_m3,
      signedTransportDrift_m3: sample.signedTransportDrift_m3,
      massErrors: sample.massErrors };
  }) : samples;
const gateSummary = compact === "3" ? (() => {
  const first = samples[0]!;
  const last = samples[samples.length - 1]!;
  // By 0.26 s the symmetric-expansion front reaches the finite tank's side
  // walls, so it is no longer a closed radial contour. Audit circularity at
  // the last pre-contact checkpoint and retain the final sample for topology,
  // mass, velocity, and extent invariants.
  const circularityStep = Math.min(45, steps);
  const circularitySample = samples.find((sample) => sample.step === circularityStep)!;
  const visibleVolume = (sample: Record<string, unknown>) =>
    (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).measuredVolume_m3;
  const volumeJumps = samples.slice(1).map((sample, index) => {
    const previous = samples[index]!;
    return { step: sample.step as number, time_s: sample.time_s as number,
      delta_m3: visibleVolume(sample) - visibleVolume(previous) };
  });
  const maximumVisibleVolumeIncrease = volumeJumps.reduce((maximum, jump) =>
    jump.delta_m3 > maximum.delta_m3 ? jump : maximum,
  { step: 0, time_s: 0, delta_m3: 0 });
  const firstNearWall = samples.find((sample) => {
    const extents = (sample.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).zeroSetExtentsCells;
    return extents && (extents.minimum[0] <= 1 || extents.minimum[2] <= 1
      || extents.maximum[0] >= dimensions[0] - 1
      || extents.maximum[2] >= dimensions[2] - 1);
  });
  const firstVolume = (first.phiReceipt as ReturnType<
    typeof unpackAdaptivePhiReceipt>).measuredVolume_m3;
  const finalVolume = (last.phiReceipt as ReturnType<
    typeof unpackAdaptivePhiReceipt>).measuredVolume_m3;
  return {
    steps,
    time_s: last.time_s,
    initialVolume_m3: firstVolume,
    finalVolume_m3: finalVolume,
    relativeVolumeDrift: (finalVolume - firstVolume) / firstVolume,
    maximumVisibleVolumeIncrease: {
      ...maximumVisibleVolumeIncrease,
      relativeToInitial: maximumVisibleVolumeIncrease.delta_m3 / firstVolume,
    },
    firstNearWall: firstNearWall ? {
      step: firstNearWall.step,
      time_s: firstNearWall.time_s,
      visibleVolume_m3: visibleVolume(firstNearWall),
      zeroSetExtentsCells: (firstNearWall.featureGeometry as ReturnType<
        typeof analyzeAdaptiveSurfaceFeatureGeometry>).zeroSetExtentsCells,
    } : undefined,
    initialConservedVolume_m3: first.acceptedMass_m3 as number,
    finalConservedVolume_m3: last.acceptedMass_m3 as number,
    finalSurfaceDensity: last.surfaceDensity,
    maximumAbsoluteTransportDrift_m3: Math.max(...samples.map((sample) =>
      Math.abs(sample.signedTransportDrift_m3 as number))),
    maximumMassErrors: Math.max(...samples.map((sample) => sample.massErrors as number)),
    initialActiveCubeCount: (first.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).activeCubeCount,
    finalActiveCubeCount: (last.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).activeCubeCount,
    finalZeroSetExtentsCells: (last.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).zeroSetExtentsCells,
    circularityCheckpointStep: circularitySample.step as number,
    circularityCheckpointTime_s: circularitySample.time_s as number,
    checkpointBottomFrontCircularity: (circularitySample.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).bottomFrontCircularity,
    maximumAcceptedSpeed_m_s: Math.max(...samples.map((sample) =>
      (sample.nodalSpeed as ReturnType<typeof auditVelocityMagnitude>).acceptedMaximum_m_s)),
    maximumD4Mismatches: Math.max(...samples.map((sample) =>
      (sample.nodalD4 as ReturnType<typeof auditNodalD4>).mismatches)),
    maximumD4AbsoluteError_m: Math.max(...samples.map((sample) =>
      (sample.nodalD4 as ReturnType<typeof auditNodalD4>).maximumAbsoluteError_m)),
    finalNodalD4: last.nodalD4,
    finalNodalPhiRange: last.nodalPhiRange,
    finalAdjacencyD4: last.adjacencyD4,
    finalConstraintsD4: last.constraintsD4,
    maximumCandidateD4Mismatches: Math.max(...samples.map((sample) =>
      (sample.candidateNodalD4 as ReturnType<typeof auditNodalD4>).mismatches)),
    maximumCandidateD4AbsoluteError_m: Math.max(...samples.map((sample) =>
      (sample.candidateNodalD4 as ReturnType<typeof auditNodalD4>).maximumAbsoluteError_m)),
    maximumAcceptedVelocityD4AbsoluteError_m_s: Math.max(...samples.map((sample) =>
      (sample.acceptedVelocityD4 as ReturnType<typeof auditNodalVelocityD4>)
        .maximumAbsoluteError_m_s)),
    maximumCandidateVelocityD4AbsoluteError_m_s: Math.max(...samples.map((sample) =>
      (sample.candidateVelocityD4 as ReturnType<typeof auditNodalVelocityD4>)
        .maximumAbsoluteError_m_s)),
    finalFaceVelocityD4: last.faceVelocityD4,
    maximumSharedNodeMismatch: Math.max(...samples.map((sample) =>
      (sample.featureGeometry as ReturnType<
        typeof analyzeAdaptiveSurfaceFeatureGeometry>).maximumSharedNodeMismatch)),
    maximumCoarseInterfaceLeaves: Math.max(...samples.map((sample) =>
      sample.coarseInterfaceLeafCount as number)),
    maximumCoarseStrictInterfaceLeaves: Math.max(...samples.map((sample) =>
      sample.coarseStrictInterfaceLeafCount as number)),
    maximumCoarseTouchingInterfaceLeaves: Math.max(...samples.map((sample) =>
      sample.coarseTouchingInterfaceLeafCount as number)),
    totalDeparturePhiFailures: samples.reduce((total, sample) => total
      + (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).departurePhiFailures, 0),
    allAcceptedAdvancesValid: samples.every((sample) =>
      (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).acceptedAdvanceValid),
    allCandidatesConverged: samples.every((sample) =>
      (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).candidate.converged),
    finalAuthority, finalFrontierFailure, finalOwnerTopologies,
  };
})() : undefined;
if (process.env.FLUID_SURFACE_AUDIT_ASSERT_PHYSICS === "1") {
  assert.ok(gateSummary, "the physical gate requires FLUID_SURFACE_AUDIT_COMPACT=3");
  assert.equal(gateSummary.maximumCoarseInterfaceLeaves, 0,
    "every accepted leaf cut by the moving factor-one interface must remain size one");
  const conservedDrift = (gateSummary.finalConservedVolume_m3
    - gateSummary.initialConservedVolume_m3) / gateSummary.initialConservedVolume_m3;
  assert.ok(Math.abs(conservedDrift) <= 1e-4,
    `conserved liquid volume drifted by ${(100 * conservedDrift).toFixed(4)}%`);
  assert.equal(gateSummary.maximumMassErrors, 0,
    "adaptive conservative mass transport reported an invalid transaction");
  assert.ok(gateSummary.maximumVisibleVolumeIncrease.relativeToInitial <= 0.01,
    `visible volume jumped by ${
      (100 * gateSummary.maximumVisibleVolumeIncrease.relativeToInitial).toFixed(3)}% in one step`);
  const visibleMassFraction = gateSummary.finalVolume_m3
    / gateSummary.finalConservedVolume_m3;
  assert.ok(visibleMassFraction >= 0.75 && visibleMassFraction <= 1.05,
    `the rho=.5 surface represents only ${(100 * visibleMassFraction).toFixed(2)}% of conserved mass`);
  assert.ok(gateSummary.maximumAcceptedSpeed_m_s >= 0.25,
    "gravity did not produce a physically moving dam front");
  assert.ok(gateSummary.maximumAcceptedSpeed_m_s <= 3.5,
    `accepted nodal speed ${gateSummary.maximumAcceptedSpeed_m_s} m/s exceeds the bounded dam scale`);
  assert.ok(gateSummary.maximumSharedNodeMismatch <= 1e-6,
    "the adaptive scalar field is discontinuous across shared leaf nodes");
  assert.ok(gateSummary.maximumD4AbsoluteError_m <= 1e-6,
    `the symmetric surface differs by ${gateSummary.maximumD4AbsoluteError_m} m under D4 transforms`);
  assert.equal(gateSummary.totalDeparturePhiFailures, 0,
    "surface transport sampled outside its valid characteristic support");
  assert.equal(gateSummary.allAcceptedAdvancesValid, true,
    "at least one transported scalar publication was rejected");
  const extents = gateSummary.finalZeroSetExtentsCells;
  assert.ok(extents && extents.minimum[0] <= 4 && extents.minimum[2] <= 4
    && extents.maximum[0] >= 28 && extents.maximum[2] >= 28,
  "the bottom dam front did not expand by at least four cells in every horizontal direction");
  const circularity = gateSummary.checkpointBottomFrontCircularity;
  assert.ok(circularity, "the bottom liquid slice has no closed front contour");
  assert.ok(Math.abs(circularity.axisLead_cells) <= 1.25
    && circularity.radialRmsDeviation_cells <= 0.5
    && circularity.radialMaximumDeviation_cells <= 1,
  `the bottom front is not circular: ${JSON.stringify(circularity)}`);
}
console.log(JSON.stringify({ phase: "symmetric-adaptive-surface", dimensions,
  dt, validationErrors, ...(gateSummary ? { gate: gateSummary } : { samples: reportedSamples })
}, null, compact === "2" || compact === "3" ? undefined : 1));
if (validationErrors.length > 0) {
  throw new Error(`Dawn reported ${validationErrors.length} validation error(s)`);
}
