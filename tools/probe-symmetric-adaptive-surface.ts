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

import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { analyzeAdaptiveSurfaceFeatureGeometry, analyzeAdaptiveSurfacePublication } from
  "../lib/octree-adaptive-surface-diagnostics";
import { getScenePreset, SYMMETRIC_EXPANSION_METHOD_PROFILE } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { initialOctreeNodalLevelSet } from "../lib/webgpu-octree";
import { unpackAdaptivePhiReceipt } from "../lib/webgpu-octree-losasso-adaptive-phi";

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
const solver = await octreeMethod.createSolverAsync!(device, scene, "balanced", {
  ...octreeMethod.presetFor("balanced"),
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
    readAdaptiveSurfacePublicationDiagnostics(): Promise<Parameters<
      typeof analyzeAdaptiveSurfacePublication>[0] & {
        phiReceipts: Uint32Array;
        nodalVelocity: Uint32Array;
        redistanceDistanceA: Float32Array;
      } | undefined>;
  };
}).octreeProjection;
assert.ok(projection, "octree solver did not expose its diagnostic projection");

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
      maximumAbsoluteError_m = Math.max(maximumAbsoluteError_m,
        Math.abs(bitFloat(sourceBits) - bitFloat(targetBits)));
    }
  }
  return Object.freeze({ mismatches, maximumAbsoluteError_m,
    reflectXMismatches: mismatchCounts[0], reflectZMismatches: mismatchCounts[1],
    swapXZMismatches: mismatchCounts[2], ...(firstMismatch ? { firstMismatch } : {}) });
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
  const nodalSpeed = auditVelocityMagnitude(snapshot);
  const acceptedStage = stageCapture && step > 0 ? await stageCapture.read() : undefined;
  const redistanceZeroSet = redistanceGeometryAudit && acceptedStage
    ? auditRedistanceZeroSet({ ...snapshot, ...acceptedStage }) : undefined;
  const stats = await solver.readStats() as unknown as Record<string, unknown>;
  samples.push({ step, time_s: step * dt, ...analysis, featureGeometry, phiReceipt,
    redistanceZeroSet,
    graphControl: snapshot.graphControl, phiControl: snapshot.phiControl, nodalD4, nodalSpeed,
    pressureRows: stats.pressureRequiredRows,
    currentVolume: stats.currentVolume,
    referenceVolume: stats.referenceVolume });
};

await capture(0);
for (let step = 1; step <= steps; step += 1) {
  while (!solver.advanceTo(step * dt, [])) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  await capture(step);
}

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
      activeCubeCount: geometry.activeCubeCount,
      zeroSetExtentsCells: geometry.zeroSetExtentsCells,
      maximumSharedNodeMismatch: geometry.maximumSharedNodeMismatch,
      nodalD4: sample.nodalD4,
      nodalSpeed: sample.nodalSpeed,
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
      currentVolume: sample.currentVolume,
      referenceVolume: sample.referenceVolume };
  }) : samples;
const gateSummary = compact === "3" ? (() => {
  const first = samples[0]!;
  const last = samples[samples.length - 1]!;
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
    initialActiveCubeCount: (first.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).activeCubeCount,
    finalActiveCubeCount: (last.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).activeCubeCount,
    finalZeroSetExtentsCells: (last.featureGeometry as ReturnType<
      typeof analyzeAdaptiveSurfaceFeatureGeometry>).zeroSetExtentsCells,
    maximumAcceptedSpeed_m_s: Math.max(...samples.map((sample) =>
      (sample.nodalSpeed as ReturnType<typeof auditVelocityMagnitude>).acceptedMaximum_m_s)),
    maximumD4Mismatches: Math.max(...samples.map((sample) =>
      (sample.nodalD4 as ReturnType<typeof auditNodalD4>).mismatches)),
    maximumSharedNodeMismatch: Math.max(...samples.map((sample) =>
      (sample.featureGeometry as ReturnType<
        typeof analyzeAdaptiveSurfaceFeatureGeometry>).maximumSharedNodeMismatch)),
    maximumCoarseInterfaceLeaves: Math.max(...samples.map((sample) =>
      sample.coarseInterfaceLeafCount as number)),
    totalDeparturePhiFailures: samples.reduce((total, sample) => total
      + (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).departurePhiFailures, 0),
    allAcceptedAdvancesValid: samples.every((sample) =>
      (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).acceptedAdvanceValid),
    allCandidatesConverged: samples.every((sample) =>
      (sample.phiReceipt as ReturnType<typeof unpackAdaptivePhiReceipt>).candidate.converged),
  };
})() : undefined;
console.log(JSON.stringify({ phase: "symmetric-adaptive-surface", dimensions,
  dt, validationErrors, ...(gateSummary ? { gate: gateSummary } : { samples: reportedSamples })
}, null, compact === "2" || compact === "3" ? undefined : 1));
if (validationErrors.length > 0) {
  throw new Error(`Dawn reported ${validationErrors.length} validation error(s)`);
}
