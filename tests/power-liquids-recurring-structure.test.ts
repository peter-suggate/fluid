import assert from "node:assert/strict";
import test from "node:test";
import {
  WebGPUFineLevelSetSummaries,
  fineLevelSetSummaryWGSL,
} from "../lib/webgpu-octree-fine-levelset-summary";
import {
  WebGPUFineLevelSetTransport,
  fineLevelSetFusedTransportPublicationWGSL,
} from "../lib/webgpu-octree-fine-levelset-transport";
import {
  WebGPUFineLevelSetVolumeCorrection,
  fineLevelSetVolumeCorrectionWGSL,
} from "../lib/webgpu-octree-fine-levelset-volume";
import {
  WebGPUFineToCoarseLevelSet,
  fineToCoarseLevelSetWGSL,
} from "../lib/webgpu-octree-fine-to-coarse-levelset";
import {
  WebGPUOctreeSimulationOwnerPages,
  octreeDeterministicOwnerPageLifecycleShader,
} from "../lib/webgpu-octree-owner-pages";
import {
  fineSeedCandidateCommitShader,
  fineSeedCandidateResidencyShader,
  sparseFineSeedCandidateResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import {
  WebGPUOctreePowerCoarseLevelSet,
  octreePowerCoarseLevelSetShader,
} from "../lib/webgpu-octree-power-coarse-levelset";
import {
  WebGPUOctreePowerDescriptor,
  octreePowerDescriptorShader,
} from "../lib/webgpu-octree-power-descriptor";
import {
  WebGPUOctreePowerFaces,
  octreePowerBoundaryCoarsePhiShader,
  octreePowerBoundaryPhiShader,
  octreePowerFaceShader,
} from "../lib/webgpu-octree-power-faces";
import {
  makePowerVelocityPrepassBuilderWGSL,
  powerVelocityFusedAuthorityWGSL,
} from "../lib/webgpu-octree-power-velocity-prepass";
import {
  octreePowerVelocityPrepareFromFaceControlShader,
  octreePowerVelocityPublishShader,
  octreePowerVelocityShader,
} from "../lib/webgpu-octree-power-velocity";
import {
  WebGPUOctreeSPGridVCycle,
  octreeSPGridPersistentMGPCGShader,
  octreeSPGridVCycleShader,
} from "../lib/webgpu-octree-spgrid-vcycle";
import {
  WebGPUOctreePowerTopology,
  octreePowerTopologyShader,
} from "../lib/webgpu-octree-power-topology";
import {
  WebGPUOctreePowerOperator,
  octreePowerOperatorShader,
} from "../lib/webgpu-octree-power-operator";
import {
  octreePowerSolidExchangeShader,
  octreePowerSolidFaceShader,
  octreePowerSolidImpulseShader,
} from "../lib/webgpu-octree-power-solid-faces";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

interface WgslGraph {
  readonly computeEntries: ReadonlySet<string>;
  readonly functions: ReadonlyMap<string, string>;
}

function stripWgslComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n\r]*/g, "");
}

/**
 * Parse the same balanced WGSL function graph used by the binding-reachability
 * tests. Atomics in an unrelated setup or diagnostic entry point must not make
 * a recurring publication entry fail, and unreachable helpers must not hide a
 * violation.
 */
function parseWgslGraph(source: string): WgslGraph {
  const shader = stripWgslComments(source);
  const functions = new Map<string, string>();
  for (const match of shader.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = shader.indexOf("{", match.index);
    let depth = 0;
    let close = -1;
    for (let at = open; at < shader.length; at += 1) {
      if (shader[at] === "{") depth += 1;
      if (shader[at] === "}" && --depth === 0) {
        close = at;
        break;
      }
    }
    assert.ok(open >= 0 && close > open, `WGSL function ${match[1]} must be complete`);
    functions.set(match[1], shader.slice(open + 1, close));
  }
  const computeEntries = new Set(
    [...shader.matchAll(
      /@compute\b(?:\s*@[A-Za-z_]\w*(?:\([^)]*\))?)*\s*fn\s+([A-Za-z_]\w*)\s*\(/g,
    )].map((match) => match[1]),
  );
  assert.ok(computeEntries.size > 0, "audited WGSL must expose at least one compute entry point");
  return { computeEntries, functions };
}

function reachableWgslFunctions(
  source: string,
  entryPoints: readonly string[],
): ReadonlyMap<string, string> {
  const graph = parseWgslGraph(source);
  const pending = [...entryPoints];
  const reached = new Map<string, string>();
  for (const entryPoint of entryPoints) {
    assert.ok(graph.computeEntries.has(entryPoint), `${entryPoint} must be a WGSL compute entry point`);
  }
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reached.has(name)) continue;
    const body = graph.functions.get(name);
    assert.notEqual(body, undefined, `reachable WGSL function ${name} must exist`);
    reached.set(name, body!);
    for (const callee of graph.functions.keys()) {
      if (!reached.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) {
        pending.push(callee);
      }
    }
  }
  return reached;
}

const synchronizationAtomic =
  /\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)\s*\(|\batomicAddF\s*\(/;

function reachableAtomicFunctions(source: string, entryPoints: readonly string[]): string[] {
  return [...reachableWgslFunctions(source, entryPoints)]
    .filter(([, body]) => synchronizationAtomic.test(body))
    .map(([name]) => name)
    .sort();
}

function allComputeEntries(source: string): string[] {
  return [...parseWgslGraph(source).computeEntries].sort();
}

test("recurring Power Liquids publication entry graphs contain no synchronization atomics", () => {
  const allEntries = (name: string, source: string) => ({ name, source, entries: allComputeEntries(source) });
  const recurringGraphs = [
    allEntries("deterministic owner-page lifecycle", octreeDeterministicOwnerPageLifecycleShader),
    allEntries("power descriptors", octreePowerDescriptorShader),
    allEntries("power topology", octreePowerTopologyShader),
    allEntries("power operator", octreePowerOperatorShader),
    allEntries("power faces", octreePowerFaceShader),
    allEntries("power solid face classification", octreePowerSolidFaceShader),
    allEntries("power solid pressure reaction", octreePowerSolidImpulseShader),
    allEntries("power solid rigid exchange", octreePowerSolidExchangeShader),
    allEntries("fine boundary phi", octreePowerBoundaryPhiShader),
    allEntries("coarse boundary phi", octreePowerBoundaryCoarsePhiShader),
    allEntries("coarse level set", octreePowerCoarseLevelSetShader),
    allEntries("power velocity", octreePowerVelocityShader),
    allEntries("power velocity face-control preparation", octreePowerVelocityPrepareFromFaceControlShader),
    allEntries("power velocity publication", octreePowerVelocityPublishShader),
    allEntries("fused power velocity authority", powerVelocityFusedAuthorityWGSL),
    allEntries("direct power velocity sampler", makePowerVelocityPrepassBuilderWGSL()),
    allEntries("fused fine transport publication", fineLevelSetFusedTransportPublicationWGSL),
    allEntries("fine volume correction", fineLevelSetVolumeCorrectionWGSL),
    allEntries("fine level-set summaries", fineLevelSetSummaryWGSL),
    allEntries("fine-to-coarse level-set restriction", fineToCoarseLevelSetWGSL),
    allEntries("fine-seed dense residency publication", fineSeedCandidateResidencyShader),
    allEntries("fine-seed sparse residency publication", sparseFineSeedCandidateResidencyShader),
    allEntries("fine-seed residency commit", fineSeedCandidateCommitShader),
    allEntries("persistent small-domain MGPCG", octreeSPGridPersistentMGPCGShader),
    {
      name: "SPGrid recurring structural publication",
      source: octreeSPGridVCycleShader,
      entries: [
        "beginL1CapturePlan",
        "planL1CaptureDelta",
        "commitChangedL1",
        "finalizeL1CapturePublication",
        "buildCandidateLevelDeltas",
        "commitCandidateLevels",
      ],
    },
  ] as const;

  for (const graph of recurringGraphs) {
    const expected = graph.name === "fine-seed dense residency publication"
      ? ["commitCandidateGeneration", "markFineSeedCandidateResidency", "markTileRing", "prepareFineSeedCandidateResidency",
        "publishFineSeedCandidateResidency", "recordDenseError", "resolveFineSeedCandidateResidency"]
      : [];
    assert.deepEqual(
      reachableAtomicFunctions(graph.source, graph.entries),
      expected,
      `${graph.name} must contain only its explicitly reviewed synchronization atomics`,
    );
  }
});

test("the atomic gate distinguishes recurring SPGrid publication from setup diagnostics", () => {
  const setupAndDiagnosticEntries = [
    "validateCandidateHierarchy",
    "finalizeLifecycle",
  ] as const;
  for (const entryPoint of setupAndDiagnosticEntries) {
    assert.ok(
      reachableAtomicFunctions(octreeSPGridVCycleShader, [entryPoint]).length > 0,
      `${entryPoint} must demonstrate that the gate follows entry-point reachability`,
    );
  }
});

test("fine-summary builder publishes an exact indirect extent for parallel sorting", () => {
  const prepare = parseWgslGraph(fineLevelSetSummaryWGSL).functions.get("prepareFineSummaryDelta");
  const tiles = parseWgslGraph(fineLevelSetSummaryWGSL).functions.get("sortFineSummaryTiles");
  assert.notEqual(prepare, undefined); assert.notEqual(tiles, undefined);
  assert.match(prepare!,
    /workState\[3\]=count;workState\[4\]=padded[\s\S]*publishSortDispatch\(padded\)/,
    "the serial transaction decision publishes both the exact record and padded sort extents");
  assert.match(tiles!,
    /sortTile\[lid\][\s\S]*for\(var width=2u;width<=256u;width<<=1u\)[\s\S]*workgroupBarrier/,
    "each indirectly dispatched tile has a fixed, workgroup-uniform shared-memory sort network");
  assert.match(tiles!, /for\(var width=2u;width<=256u/,
    "tile barrier reachability is controlled only by a compile-time workgroup width");
});

test("coarse-phi delta commit broadcasts authority and bounds before every barrier", () => {
  const commit = parseWgslGraph(octreePowerCoarseLevelSetShader)
    .functions.get("publishPowerCoarsePhiDeltaAndCommit");
  assert.notEqual(commit, undefined);
  assert.match(commit!,
    /if\(lid==0u\)\{let rejected=rejectedFine\(\);let enabled=!rejected&&control\.valid==VALID&&candidateDirectory\.state==VALID;if\(!rejected&&!enabled\)\{sampleDirectory\.state=0u;\}[\s\S]*coarseDeltaCommitState\[1\]=select\(0u,candidateDirectory\.rowCount,enabled\)[\s\S]*coarseDeltaCommitState\[2\]=select\(0u,previous,enabled\)/,
    "lane zero must snapshot the complete immutable-candidate authority transaction");
  for (const word of [0, 1, 2]) {
    assert.match(commit!, new RegExp(`workgroupUniformLoad\\(&coarseDeltaCommitState\\[${word}\\]\\)`),
      `coarse delta commit state word ${word} must be formally workgroup-uniform`);
  }
  assert.doesNotMatch(commit!,
    /if\([^)]*(?:rejectedFine|control\.valid|candidateDirectory\.state)[^)]*\)\{return;\}[\s\S]*workgroupBarrier/,
    "no storage-derived early return may bypass the reduction or immutable-commit barriers");
  assert.match(commit!,
    /if\(lid==0u&&enabled!=0u\)\{sampleDirectory\.state=0u;\}[\s\S]*workgroupBarrier\(\)[\s\S]*if\(enabled!=0u\)\{for\(var slot=lid;slot<currentCount[\s\S]*workgroupBarrier\(\)[\s\S]*if\(lid==0u&&enabled!=0u\)/,
    "rejected fine candidates must cross the full synchronization skeleton without mutating the prior directory");
});

test("coarse-phi finalize broadcasts rejection before its reduction barriers", () => {
  const finalize = parseWgslGraph(octreePowerCoarseLevelSetShader)
    .functions.get("finalizePowerCoarsePhi");
  assert.notEqual(finalize, undefined);
  assert.match(finalize!,
    /if\(lid==0u\)\{coarseFinalizeEnabled=select\(0u,1u,!rejectedFine\(\)\);\}[\s\S]*let enabled=workgroupUniformLoad\(&coarseFinalizeEnabled\)/,
    "lane zero must convert storage-backed fine authority into a uniform reduction gate");
  assert.doesNotMatch(finalize!,
    /if\(rejectedFine\(\)\)\{return;\}[\s\S]*workgroupBarrier/,
    "rejected fine authority must execute the complete reduction synchronization skeleton");
  assert.match(finalize!,
    /let count=select\(0u,requested\(\),enabled!=0u\)[\s\S]*workgroupBarrier\(\)[\s\S]*if\(lid==0u&&enabled!=0u\)/,
    "the disabled transaction must reduce zero rows and preserve prior global publication bytes");
});

type ConstructorWithPrototype = { readonly prototype: object };

function prototypeMethod(constructor: ConstructorWithPrototype, name: string): Function {
  const candidate = Reflect.get(constructor.prototype, name);
  assert.equal(typeof candidate, "function", `${constructor.constructor.name}.${name} must exist`);
  return candidate as Function;
}

test("recurring host encoders add no CPU readback, queue submission, or host fence", () => {
  const methods: readonly [string, Function][] = [
    ["octree projection encode", prototypeMethod(WebGPUOctreeProjection, "encode")],
    ["octree projection surface encode", prototypeMethod(WebGPUOctreeProjection, "encodeSurface")],
    ["octree projection inline rebuild", prototypeMethod(WebGPUOctreeProjection, "encodeInlineRebuild")],
    ["octree projection inline rebuild completion", prototypeMethod(WebGPUOctreeProjection, "finishInlineRebuild")],
    ["global fine face-band encode", prototypeMethod(WebGPUOctreeProjection, "encodeGlobalFineFaceBand")],
    ["global fine face-band phase", prototypeMethod(WebGPUOctreeProjection, "encodeGlobalFineFaceBandPhase")],
    ["native power assembly", prototypeMethod(WebGPUOctreeProjection, "encodeNativePowerAssembly")],
    ["native power projection", prototypeMethod(WebGPUOctreeProjection, "encodeNativePowerProjection")],
    ["power velocity publication", prototypeMethod(WebGPUOctreeProjection, "encodePowerVelocityPublication")],
    ["frontier-row publication", prototypeMethod(WebGPUOctreeProjection, "encodeFrontierRows")],
    ["owner-page publication", prototypeMethod(WebGPUOctreeSimulationOwnerPages, "encode")],
    ["power descriptor publication", prototypeMethod(WebGPUOctreePowerDescriptor, "encode")],
    ["power topology publication", prototypeMethod(WebGPUOctreePowerTopology, "encode")],
    ["power operator assembly", prototypeMethod(WebGPUOctreePowerOperator, "encodeAssemblyFromControl")],
    ["power operator projection", prototypeMethod(WebGPUOctreePowerOperator, "encodeProjectionFromControl")],
    ["power face publication", prototypeMethod(WebGPUOctreePowerFaces, "encode")],
    ["coarse level-set publication", prototypeMethod(WebGPUOctreePowerCoarseLevelSet, "encode")],
    ["SPGrid capture", prototypeMethod(WebGPUOctreeSPGridVCycle, "encodeCapture")],
    ["SPGrid setup", prototypeMethod(WebGPUOctreeSPGridVCycle, "encodeSetup")],
    ["SPGrid correction", prototypeMethod(WebGPUOctreeSPGridVCycle, "encodeCorrection")],
    ["fine level-set transport", prototypeMethod(WebGPUFineLevelSetTransport, "encode")],
    ["fine level-set summary", prototypeMethod(WebGPUFineLevelSetSummaries, "encode")],
    ["fine volume correction", prototypeMethod(WebGPUFineLevelSetVolumeCorrection, "encode")],
    ["fine-to-coarse restriction", prototypeMethod(WebGPUFineToCoarseLevelSet, "encode")],
  ];
  const forbiddenHostSynchronization =
    /\bmapAsync\b|\bgetMappedRange\b|\bonSubmittedWorkDone\b|\bGPUMapMode\s*\.\s*READ\b|\bMAP_READ\b|\bqueue\s*\.\s*submit\b|\bcreateCommandEncoder\b/;

  for (const [label, method] of methods) {
    assert.notEqual(method.constructor.name, "AsyncFunction", `${label} must remain synchronous`);
    assert.doesNotMatch(method.toString(), forbiddenHostSynchronization,
      `${label} must not introduce a host readback, wait, command submission, or fence`);
  }
});
