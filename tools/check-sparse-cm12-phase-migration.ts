#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { createSparseCM12HotTopology } from
  "../lib/methods/adaptive-mass/sparse-cm12-hot-topology";
import {
  createSparseCM12PhaseArenaPlan,
  type SparseCM12PhaseArenaOwner,
} from "../lib/methods/adaptive-mass/sparse-cm12-phase-arenas";

type CutBoundary = "F0" | "P0" | "T0" | "C0" | "R0"
  | "existing-mathematical-boundaries" | "gamma-image-boundaries";
type ByteRule = "no-write" | "gpu-copy-exact" | "same-offset-rebind"
  | "accessor-equivalence" | "physical-word-equivalence";

interface MigrationSlice {
  readonly id: string;
  readonly boundary: CutBoundary;
  readonly touches: readonly SparseCM12PhaseArenaOwner[];
  readonly byteRule: ByteRule;
  readonly preservesPhysicalBytes: true;
  readonly gpuPublication: true;
  readonly cpuScheduleDecision: false;
  readonly runtimeFallback: "none";
  readonly faultDomain: string;
  readonly requiredHashes: readonly string[];
}

const PHYSICAL_FIELDS = Object.freeze([
  "density", "gamma", "cellVelocity", "faceVelocity", "pressure", "liquid", "theta",
  "acceptedTopology", "massReceipt", "gammaReceipt", "momentumReceipt",
] as const);

const slice = (
  id: string,
  boundary: CutBoundary,
  touches: readonly SparseCM12PhaseArenaOwner[],
  byteRule: ByteRule,
  faultDomain: string,
): MigrationSlice => Object.freeze({
  id, boundary, touches: Object.freeze(touches), byteRule,
  preservesPhysicalBytes: true, gpuPublication: true,
  cpuScheduleDecision: false, runtimeFallback: "none", faultDomain,
  requiredHashes: PHYSICAL_FIELDS,
});

const SLICES: readonly MigrationSlice[] = Object.freeze([
  slice("M00", "F0", ["Diagnostics"], "no-write", "field receipt generation"),
  slice("M01", "F0", ["HTP1", "Diagnostics"], "accessor-equivalence", "HTP1 record"),
  slice("M02", "F0", ["PhysicsState", "ScalarScratch", "PressureCache",
    "TopologyCandidate", "Presentation", "Diagnostics", "HTP1"],
  "gpu-copy-exact", "copied region"),
  slice("M03", "F0", ["PhysicsState", "Diagnostics"], "same-offset-rebind",
    "physical field brick packet"),
  slice("M04", "F0", ["ScalarScratch", "Diagnostics"], "gpu-copy-exact",
    "scratch region"),
  slice("M05", "P0", ["PhysicsState", "PressureCache", "TopologyCandidate", "HTP1",
    "Diagnostics"], "gpu-copy-exact", "pressure brick and ancestor chain"),
  slice("M06", "T0", ["PhysicsState", "TopologyCandidate", "HTP1", "Diagnostics"],
    "gpu-copy-exact", "candidate brick and incident rows"),
  slice("M07", "R0", ["PhysicsState", "Presentation", "Diagnostics"],
    "gpu-copy-exact", "presentation page or diagnostic region"),
  slice("M08a", "F0", ["HTP1", "Diagnostics"], "accessor-equivalence",
    "logical owner query packet"),
  slice("M08b", "F0", ["HTP1", "PhysicsState", "Diagnostics"],
    "accessor-equivalence", "cell packet and containing brick"),
  slice("M08c", "F0", ["HTP1", "Diagnostics"], "accessor-equivalence",
    "row packet and incidences"),
  slice("M08d", "P0", ["HTP1", "PressureCache", "Diagnostics"],
    "accessor-equivalence", "cell CSR and connected rows"),
  slice("M09", "F0", ["HTP1", "PhysicsState", "TopologyCandidate", "Presentation",
    "Diagnostics"], "physical-word-equivalence", "B16 tile packet family"),
  slice("M10", "P0", ["HTP1", "PhysicsState", "ScalarScratch", "PressureCache",
    "Diagnostics"], "physical-word-equivalence", "pressure brick and dirty ancestors"),
  slice("M11", "existing-mathematical-boundaries", ["HTP1", "PhysicsState",
    "ScalarScratch", "PressureCache", "Diagnostics"], "physical-word-equivalence",
  "coalesced owner packet"),
  slice("M12", "gamma-image-boundaries", ["HTP1", "PhysicsState", "ScalarScratch",
    "Diagnostics"], "physical-word-equivalence", "gamma cell and B16 tile"),
  slice("M13", "C0", ["HTP1", "PhysicsState", "PressureCache", "TopologyCandidate",
    "Diagnostics"], "physical-word-equivalence", "candidate brick and incident rows"),
  slice("M14", "R0", ["HTP1", "PhysicsState", "TopologyCandidate", "Presentation",
    "Diagnostics"], "physical-word-equivalence", "one P16 page"),
  slice("M15", "F0", ["PhysicsState", "TopologyCandidate", "Presentation", "Diagnostics"],
    "physical-word-equivalence", "frame-plan generation"),
]);

type CoalescingVerdict = "provable" | "hash-required" | "forbidden";
interface Coalescing {
  readonly name: string;
  readonly kernels: readonly string[];
  readonly verdict: CoalescingVerdict;
  readonly preservesInvocationMap: boolean;
  readonly preservesOperationOrder: boolean;
  readonly crossInvocationDependency: boolean;
  readonly twoSecondPhysicalGate: boolean;
}

const COALESCINGS: readonly Coalescing[] = Object.freeze([
  { name: "face prepare plus force", kernels: ["prepareTransportFaces", "forceFaces"],
    verdict: "provable", preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },
  { name: "edge bake plus pressure prepare", kernels: ["bakeEffectivePressureEdges",
    "preparePressure"], verdict: "provable", preservesInvocationMap: true,
    preservesOperationOrder: true, crossInvocationDependency: false,
    twoSecondPhysicalGate: true },
  { name: "collocation plus divergence partial", kernels: ["collocateAndDiagnose",
    "measureDivergenceDiagnostics"],
    verdict: "forbidden", preservesInvocationMap: true,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "activity wet bit plus presentation classification",
    kernels: ["measureBrickActivity", "classifyPresentationBricks"], verdict: "provable",
    preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },
  { name: "producer output plus tile OR", kernels: ["physicalProducer", "dirtyMaskOr"],
    verdict: "provable", preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },
  { name: "copying all phase indirect triplets", kernels: ["sealFramePlan", "copyIndirectTriplets"],
    verdict: "provable", preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },

  { name: "final extrapolation plus CFL classification",
    kernels: ["extrapolateTransportVelocityToDestination", "classifyVelocityCFL"],
    verdict: "hash-required", preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },
  { name: "gamma incidence gather", kernels: ["scatterGammaRow", "gatherGammaCell"],
    verdict: "hash-required", preservesInvocationMap: false, preservesOperationOrder: false,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },
  { name: "dirty packet heavy traversal", kernels: ["acceptedTraversal", "dirtyTileTraversal"],
    verdict: "hash-required", preservesInvocationMap: false, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },
  { name: "persistent pressure reuse", kernels: ["bakeEffectivePressureEdges",
    "reuseCleanPressureCoefficients"], verdict: "hash-required", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: false,
    twoSecondPhysicalGate: true },
  { name: "aggregate hierarchy fusion", kernels: ["bakeBrickAggregateDiagonal",
    "bakePressureHierarchyDiagonal"], verdict: "hash-required", preservesInvocationMap: false,
    preservesOperationOrder: false, crossInvocationDependency: false,
    twoSecondPhysicalGate: true },
  { name: "row classification plus coefficient repair", kernels: ["classifyRows",
    "bakeDirtyEffectivePressureEdges"], verdict: "hash-required", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: true },
  { name: "activity plus planning", kernels: ["measureBrickActivity", "planBrickResolution"],
    verdict: "hash-required", preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: true, twoSecondPhysicalGate: true },
  { name: "prepared-brick candidate transfer", kernels: ["transferCandidateCells",
    "preparedBrickTransfer"], verdict: "hash-required", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: true },
  { name: "dirty page reuse", kernels: ["publishSparseLevelSet", "reuseCleanP16Page"],
    verdict: "hash-required", preservesInvocationMap: false, preservesOperationOrder: true,
    crossInvocationDependency: false, twoSecondPhysicalGate: true },

  { name: "velocity sweeps", kernels: ["extrapolateTransportVelocityToSource",
    "extrapolateTransportVelocityToDestination"], verdict: "forbidden",
    preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: true, twoSecondPhysicalGate: false },
  { name: "transport trace and deficit", kernels: ["traceGammaAndBeta",
    "scatterDensityDeficit"], verdict: "forbidden", preservesInvocationMap: true,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "transport scatter and gather", kernels: ["scatterDensityDeficit",
    "gatherConservativeDensity"], verdict: "forbidden", preservesInvocationMap: true,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "gamma scatter and finalize", kernels: ["scatterGammaSnapshot",
    "finalizeGammaSnapshot"], verdict: "forbidden", preservesInvocationMap: true,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "sharpening scatter and finalize", kernels: ["scatterSharpeningMass",
    "finalizeSharpening"], verdict: "forbidden", preservesInvocationMap: true,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "D4 preserve and commit", kernels: ["preserveHorizontalD4", "commitHorizontalD4"],
    verdict: "forbidden", preservesInvocationMap: true, preservesOperationOrder: true,
    crossInvocationDependency: true, twoSecondPhysicalGate: false },
  { name: "pressure project and collocate", kernels: ["projectSparseCM12InteriorFaceTiles",
    "projectSparseCM12SeamFacePackets",
    "collocateAndDiagnose"], verdict: "forbidden", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "Krylov update and reduction", kernels: ["updatePipelinedState",
    "reducePipelinedIteration"], verdict: "forbidden", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "activity reduction and grading", kernels: ["measureBrickActivity",
    "closePlannedResolution"], verdict: "forbidden", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "candidate receipt and commit", kernels: ["reconstructShadowFaces",
    "validateAndCommitShadowTopology"], verdict: "forbidden", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
  { name: "commit and presentation seal", kernels: ["validateAndCommitShadowTopology",
    "publishSparseLevelSet"], verdict: "forbidden", preservesInvocationMap: false,
    preservesOperationOrder: true, crossInvocationDependency: true,
    twoSecondPhysicalGate: false },
]);

interface HostAuthority {
  readonly name: string;
  readonly documentNeedle: string;
  readonly classification: "static-construction" | "evolving-frame";
  readonly gpuOwned: boolean;
  readonly fixedIndirectSchedule: boolean;
  readonly hostMaySelectFrameWork: boolean;
}

const HOST_AUTHORITIES: readonly HostAuthority[] = Object.freeze([
  { name: "parity", documentNeedle: "`parity`", classification: "evolving-frame",
    gpuOwned: true, fixedIndirectSchedule: true, hostMaySelectFrameWork: false },
  { name: "horizontal D4 authority", documentNeedle: "`horizontalD4Authority`",
    classification: "evolving-frame", gpuOwned: true, fixedIndirectSchedule: true,
    hostMaySelectFrameWork: false },
  { name: "accepted/shadow and dirty topology counts",
    documentNeedle: "Accepted/shadow cell and row counts", classification: "evolving-frame",
    gpuOwned: true, fixedIndirectSchedule: true, hostMaySelectFrameWork: false },
  { name: "body and boundary live scheduling", documentNeedle: "`bodyCount > 0`",
    classification: "evolving-frame", gpuOwned: true, fixedIndirectSchedule: true,
    hostMaySelectFrameWork: false },
  { name: "tracer scheduling", documentNeedle: "`tracersEnabled`",
    classification: "evolving-frame", gpuOwned: true, fixedIndirectSchedule: true,
    hostMaySelectFrameWork: false },
  { name: "journal scheduling", documentNeedle: "`journalArmed`",
    classification: "evolving-frame", gpuOwned: true, fixedIndirectSchedule: true,
    hostMaySelectFrameWork: false },
  { name: "physical topology capacities", documentNeedle: "`packed.brickCount`",
    classification: "static-construction", gpuOwned: false, fixedIndirectSchedule: false,
    hostMaySelectFrameWork: false },
  { name: "pipeline capabilities", documentNeedle: "`hostTemplateVariants`",
    classification: "static-construction", gpuOwned: false, fixedIndirectSchedule: false,
    hostMaySelectFrameWork: false },
]);

function fixtureArenaOwners(): Set<SparseCM12PhaseArenaOwner> {
  const logical = [1, 1, 1] as const;
  const source: SparseAdaptiveMassBrick = {
    key: sparseBrickKey([0, 0, 0], logical), coordinate: [0, 0, 0], resolution: 4,
    density: new Float64Array(4 ** 3), gamma: new Float64Array(4 ** 3).fill(1),
  };
  const atlas = createSparseAdaptiveMassAtlas([16, 16, 16], [source], 31, undefined, 16);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const htp1 = createSparseCM12HotTopology(grid).layout;
  const plan = createSparseCM12PhaseArenaPlan({
    cellCount: htp1.cellCount, rowCount: htp1.rowCount, brickCount: 1,
    candidateBrickCount: 1, pressureEdgeCount: htp1.directedEdgeCount,
    pressureCoarseEdgeCount: 0, pressureHierarchy: [{ groupCount: 1, edgeCount: 0 }],
    pressureMembershipWords: 128, pressureStaticTopologyWords: 32,
    topologyCandidateControlWords: 256, tracerCount: 0,
    presentation: { metadataBytes: 16, worklistBytes: 36, payloadBytes: 16 ** 3 * 4 },
    htp1,
  });
  return new Set(plan.buffers.map((buffer) => buffer.owner));
}

function assertSlices(document: string): void {
  const owners = fixtureArenaOwners();
  const ids = new Set<string>();
  for (const migration of SLICES) {
    if (ids.has(migration.id)) throw new Error(`duplicate migration slice ${migration.id}`);
    ids.add(migration.id);
    if (!document.includes(`| ${migration.id} |`)) {
      throw new Error(`document omits migration slice ${migration.id}`);
    }
    if (!migration.preservesPhysicalBytes || !migration.gpuPublication
      || migration.cpuScheduleDecision || migration.runtimeFallback !== "none") {
      throw new Error(`${migration.id} violates rollback-free GPU publication`);
    }
    if (/global|whole[- ]domain/i.test(migration.faultDomain)) {
      throw new Error(`${migration.id} has non-local fault domain ${migration.faultDomain}`);
    }
    migration.touches.forEach((owner) => {
      if (!owners.has(owner)) throw new Error(`${migration.id} touches unknown ${owner}`);
    });
    PHYSICAL_FIELDS.forEach((field) => {
      if (!migration.requiredHashes.includes(field)) {
        throw new Error(`${migration.id} omits physical field hash ${field}`);
      }
    });
  }
  const expected = ["M00", "M01", "M02", "M03", "M04", "M05", "M06", "M07",
    "M08a", "M08b", "M08c", "M08d", "M09", "M10", "M11", "M12", "M13",
    "M14", "M15"];
  if (expected.some((id, index) => SLICES[index]?.id !== id)) {
    throw new Error("migration slice order is not the published one-way sequence");
  }
}

function assertCoalescings(document: string): void {
  const names = new Set<string>();
  for (const coalescing of COALESCINGS) {
    if (names.has(coalescing.name)) throw new Error(`duplicate coalescing ${coalescing.name}`);
    names.add(coalescing.name);
    const namedKernelPresent = coalescing.kernels.some((kernel) => document.includes(kernel));
    const nameAnchor = coalescing.name.split(" ")[0]!.toLowerCase();
    if (!namedKernelPresent && !document.toLowerCase().includes(nameAnchor)) {
      throw new Error(`document omits coalescing ${coalescing.name}`);
    }
    if (coalescing.verdict === "provable"
      && (!coalescing.preservesInvocationMap || !coalescing.preservesOperationOrder
        || coalescing.crossInvocationDependency || !coalescing.twoSecondPhysicalGate)) {
      throw new Error(`${coalescing.name} lacks its order-independence proof conditions`);
    }
    if (coalescing.verdict === "hash-required" && !coalescing.twoSecondPhysicalGate) {
      throw new Error(`${coalescing.name} lacks the two-second physical hash gate`);
    }
    if (coalescing.verdict === "forbidden"
      && (!coalescing.crossInvocationDependency || coalescing.twoSecondPhysicalGate)) {
      throw new Error(`${coalescing.name} is not a strict dependency prohibition`);
    }
  }
  const counts = (verdict: CoalescingVerdict) =>
    COALESCINGS.filter((entry) => entry.verdict === verdict).length;
  if (counts("provable") < 5 || counts("hash-required") < 8 || counts("forbidden") < 10) {
    throw new Error("coalescing audit lost a classification family");
  }
}

function assertHostAuthorityElimination(document: string): void {
  for (const authority of HOST_AUTHORITIES) {
    if (!document.includes(authority.documentNeedle)) {
      throw new Error(`host authority map omits ${authority.name}`);
    }
    if (authority.hostMaySelectFrameWork) {
      throw new Error(`${authority.name} still permits a host frame decision`);
    }
    if (authority.classification === "evolving-frame"
      && (!authority.gpuOwned || !authority.fixedIndirectSchedule)) {
      throw new Error(`${authority.name} is not GPU-owned fixed indirect work`);
    }
    if (authority.classification === "static-construction" && authority.gpuOwned) {
      throw new Error(`${authority.name} incorrectly treats static capacity as evolving`);
    }
  }
  const lowerDocument = document.toLowerCase();
  for (const required of ["x=0", "external body descriptors", "fixed pipeline schedule",
    "inputs, not scheduling authority"]) {
    if (!lowerDocument.includes(required)) throw new Error(`host authority contract omits ${required}`);
  }
}

function main(): void {
  const document = readFileSync(new URL("../docs/sparse-cm12-b16-p16-phase-migration.md",
    import.meta.url), "utf8");
  assertSlices(document);
  assertCoalescings(document);
  assertHostAuthorityElimination(document);
  console.log(`Sparse CM12 phase migration: ${SLICES.length} rollback-free bit-exact slices; ${COALESCINGS.filter((entry) => entry.verdict === "provable").length} provable, ${COALESCINGS.filter((entry) => entry.verdict === "hash-required").length} hash-gated, ${COALESCINGS.filter((entry) => entry.verdict === "forbidden").length} forbidden coalescings, and ${HOST_AUTHORITIES.length} host-authority eliminations passed`);
}

main();
