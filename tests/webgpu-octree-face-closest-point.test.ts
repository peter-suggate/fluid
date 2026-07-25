import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  OCTREE_FACE_BAND_FACE_BYTES,
  OCTREE_FACE_BAND_CONTROL_BYTES,
  OCTREE_FACE_BAND_ROW_BYTES,
  OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES,
  OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
  OCTREE_FACE_BAND_TRANSITION_DETAIL,
  OCTREE_FACE_BAND_TRANSITION_ERROR,
  OCTREE_FACE_BAND_OWNER_FAILURE_STAGE,
  OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES,
  OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES,
  OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES,
  OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES,
  OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS,
  OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES,
  OCTREE_FACE_BAND_POWER_PUBLICATION_ERROR,
  OCTREE_FACE_BAND_ENCODE_PHASES,
  WebGPUOctreeFaceClosestPointExtension,
  classifyOctreeFaceBandBoundaryCrossing,
  makeOctreeFaceBandAirSampleWGSL,
  octreeFaceBandCoarseGenerationPairIsValid,
  octreeFaceBandSupportScatterWGSL,
  octreeFaceBandWGSL,
  planOctreeFaceBandGPU,
  unpackOctreeFaceBandControl,
  unpackOctreeFaceBandPointFieldControl,
  unpackOctreeFaceBandPowerPublication,
  unpackOctreeFaceBandTransientPowerControl,
  unpackOctreeFaceBandTransitionControl,
} from "../lib/webgpu-octree-face-closest-point";
import {
} from "../lib/octree-power-descriptor";
import {
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  decodeGeneratedOctreePowerCatalog,
} from "../lib/generated/octree-power-catalog";
import {
  OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
  OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW,
} from "../lib/octree-face-band";
import { WebGPUFineLevelSetTransport } from "../lib/webgpu-octree-fine-levelset-transport";
import type { PassBroker } from "../lib/webgpu-pass-broker";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

const compact = (value: { toString(): string }): string => value.toString().replace(/\s+/g, "");

function wgslFunction(name: string, wgsl = octreeFaceBandWGSL): string {
  const source = compact(wgsl);
  const start = source.indexOf(`fn${name}(`);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const open = source.indexOf("{", start);
  assert.notEqual(open, -1, `missing WGSL body for ${name}`);
  let depth = 0;
  for (let cursor = open; cursor < source.length; cursor += 1) {
    if (source[cursor] === "{") depth += 1;
    else if (source[cursor] === "}" && --depth === 0) return source.slice(start, cursor + 1);
  }
  assert.fail(`unterminated WGSL function ${name}`);
}

function wgslReachableBindings(entryPoint: string, wgsl = octreeFaceBandWGSL): number[] {
  const source = wgsl.replace(/\/\/[^\n\r]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
  const globals = new Map<string, number>();
  for (const match of source.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s*([A-Za-z_]\w*)/g,
  )) globals.set(match[2], Number(match[1]));
  const bodies = new Map<string, string>();
  for (const match of source.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = source.indexOf("{", match.index); let depth = 0; let close = -1;
    for (let at = open; at < source.length; at += 1) {
      if (source[at] === "{") depth += 1;
      if (source[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.ok(open >= 0 && close > open, `WGSL function ${match[1]} must have a complete body`);
    bodies.set(match[1], source.slice(open + 1, close));
  }
  const pending = [entryPoint]; const reached = new Set<string>(); const bindings = new Set<number>();
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reached.has(name)) continue;
    reached.add(name);
    const body = bodies.get(name);
    assert.notEqual(body, undefined, `reachable WGSL function ${name} must exist`);
    const scopes: Set<string>[] = [new Set()];
    let declaresLocal = false;
    for (const token of body!.matchAll(/[A-Za-z_]\w*|[{}]/g)) {
      const value = token[0];
      if (value === "{") { scopes.push(new Set()); declaresLocal = false; continue; }
      if (value === "}") { scopes.pop(); declaresLocal = false; continue; }
      if (value === "let" || value === "var" || value === "const") { declaresLocal = true; continue; }
      if (declaresLocal) { scopes.at(-1)!.add(value); declaresLocal = false; continue; }
      const binding = globals.get(value);
      if (binding !== undefined && !scopes.some((scope) => scope.has(value))) bindings.add(binding);
    }
    for (const callee of bodies.keys()) {
      if (!reached.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
    }
  }
  return [...bindings].sort((a, b) => a - b);
}

function faceBandPipelineEntryPoints(): Map<string, {
  readonly entryPoint: string;
  readonly wgsl: string;
}> {
  const source = compact(WebGPUOctreeFaceClosestPointExtension);
  const start = source.indexOf("constpipelines={");
  const end = source.indexOf("};this.pipelines=", start);
  assert.ok(start >= 0 && end > start, "face-band pipeline table must remain statically inspectable");
  const result = new Map<string, { readonly entryPoint: string; readonly wgsl: string }>();
  for (const match of source.slice(start, end).matchAll(/([A-Za-z_]\w*):pipeline\("([A-Za-z_]\w*)"/g)) {
    result.set(match[1], { entryPoint: match[2], wgsl: octreeFaceBandWGSL });
  }
  for (const match of source.slice(start, end).matchAll(
    /([A-Za-z_]\w*):supportPipeline\("([A-Za-z_]\w*)"/g,
  )) {
    result.set(match[1], { entryPoint: match[2], wgsl: octreeFaceBandSupportScatterWGSL });
  }
  return result;
}

function faceBandEncodeBindings(): Map<string, number[]> {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const call = /run\("([A-Za-z_]\w*)",\[\[/g;
  const result = new Map<string, number[]>();
  for (const match of source.matchAll(call)) {
    const key = match[1];
    const entriesStart = match.index! + match[0].length - 2;
    const entriesEnd = source.indexOf("]]", entriesStart);
    assert.ok(entriesEnd > entriesStart, `${key} bind entries must remain statically inspectable`);
    const bindings = [...source.slice(entriesStart, entriesEnd + 2).matchAll(/\[(\d+),/g)]
      .map((binding) => Number(binding[1])).sort((a, b) => a - b);
    const prior = result.get(key);
    if (prior) assert.deepEqual(bindings, prior, `${key} must use one exact bind contract at every call site`);
    result.set(key, bindings);
  }
  return result;
}

test("every Section 5 encode pipeline binds exactly its transitively reachable WGSL resources", () => {
  const pipelines = faceBandPipelineEntryPoints();
  const encoded = faceBandEncodeBindings();
  const uniformBindings = new Set([0, 1, 20]);
  assert.deepEqual([...encoded.keys()].sort(), [...pipelines.keys()].sort(),
    "every encodePhase pipeline key must have one audited run call");
  for (const [key, pipeline] of pipelines) {
    const { entryPoint, wgsl } = pipeline;
    const reachable = wgslReachableBindings(entryPoint, wgsl);
    assert.deepEqual(encoded.get(key), reachable,
      `${key} (${entryPoint}) host bindings must exactly equal transitive WGSL reachability`);
    const storageBindings = reachable.filter(binding => !uniformBindings.has(binding));
    assert.ok(storageBindings.length <= 10,
      `${key} (${entryPoint}) reaches ${storageBindings.length} storage buffers: ${storageBindings.join(", ")}`);
  }
});

test("Section 5 catalog adjacency keeps exact delta, resolution, commit, and indirect consumers in broker order", () => {
  assert.deepEqual(OCTREE_FACE_BAND_ENCODE_PHASES, [
    "topology-build", "transition-adjacency", "closest-point-extension", "power-publication",
  ]);
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const topology = source.indexOf('case"topology-build"');
  const transitions = source.indexOf('case"transition-adjacency"', topology);
  const march = source.indexOf('case"closest-point-extension"', transitions);
  const publication = source.indexOf('case"power-publication"', march);
  assert.ok(topology >= 0 && transitions > topology && march > transitions && publication > march);
  const adjacency = source.slice(transitions, march);
  assert.equal(adjacency.match(/computePass\("/g)?.length, 4);
  assert.doesNotMatch(source, /newPassBroker|broker\.fence/,
    "the face-band encoder must preserve its caller's pass across routine stage labels");
  assert.match(adjacency,
    /run\("prepareTransition"[\s\S]*run\("classifyTransitionDelta"[\s\S]*run\("carryTransitionAdjacency"[\s\S]*run\("describeCatalogRows"[\s\S]*run\("resolveCatalogAdjacency"[\s\S]*run\("validateCatalogAdjacency"[\s\S]*run\("publishCatalogAdjacency"[\s\S]*run\("carryFaceTopology"[\s\S]*run\("emit"[\s\S]*run\("carryFaceIncidence"[\s\S]*run\("rebuildIncidence"[\s\S]*run\("publishFaceCounts"[\s\S]*run\("gateTransition"/);
  assert.doesNotMatch(adjacency,
    /scanTransitionDeltaBlocks|prefixTransitionDeltaBlocks|scatterTransitionDeltaRows/,
    "the dense affected-row mask deletes global scan, prefix, and scatter dispatches");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /scanCatalogTransitionDeltaBlocks|prefixCatalogTransitionDeltaBlocks|scatterCatalogTransitionDeltaRows/,
    "the deleted compaction kernels are not compiled at construction");
  assert.match(wgslFunction("prepareCatalogTransitionAdjacency"),
    /indirect\[51\]=\(transitionControl\.rowCount\+63u\)\/64u/,
    "the immutable dense work domain is produced without a host count or compact-list producer");
  assert.match(wgslFunction("emitBandFaces"),
    /band=faceDeltaRow\(item\);if\(!faceRowAffected\(band\)\)\{return;\}/);
  assert.match(wgslFunction("rebuildFaceBandIncidence"),
    /band=faceDeltaRow\(item\);if\(!incidenceRowAffected\(band\)\)\{return;\}/);
  assert.doesNotMatch(wgslFunction("publishCatalogTransitionAdjacency"),
    /transitionDeltaScan\[item\]=transitionDeltaRow\(item\)/,
    "publication preserves the dense mask instead of rewriting it as a compact list");
  assert.match(adjacency,
    /run\("publishCatalogAdjacency"[\s\S]*computePass\("PublishSection5regular-facetopology"/,
    "the downstream indirect producer must precede regular-face consumers in broker order");
  assert.doesNotMatch(adjacency,
    /synchronizeTransitionStorage|enumerateSupport|resolveSupportOwners|insertSupport|captureSupport/);
  const topologySchedule = source.slice(topology, transitions);
  assert.match(topologySchedule,
    /run\("prepare"[\s\S]*run\("clearSupportIdentityMarks"[\s\S]*run\("buildTopologyDelta"[\s\S]*run\("markSupport1"[\s\S]*run\("scatterSupport1"[\s\S]*run\("markSupport3"[\s\S]*run\("scatterSupport4"[\s\S]*run\("markPublishedSupportRows"[\s\S]*run\("scatterCanonicalRowDirectory"[\s\S]*run\("validateRowDirectory"[\s\S]*run\("classifyRowSigns"/);
  assert.equal(topologySchedule.match(/\brun\("/g)?.length, 33,
    "support publication initializes, closes four exact tiers, scans the identity plane, and validates");
  assert.doesNotMatch(topologySchedule, /sortUniqueCatalogSupport|sortUniqueEndpointSupport|sortValidateRowDirectory/,
    "the retired single-workgroup radix schedule must not survive");
});

test("transition adjacency has one exact row-delta path and no whole-row resolve launch", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const transition = source.slice(source.indexOf('case"transition-adjacency"'),
    source.indexOf('case"closest-point-extension"'));
  assert.match(transition,
    /run\("prepareTransition"[\s\S]*run\("classifyTransitionDelta"[\s\S]*run\("carryTransitionAdjacency"/);
  assert.doesNotMatch(transition, /run\("carryTransitionMetrics"/,
    "unchanged metrics and adjacency are carried in one row-parallel dispatch");
  assert.match(wgslFunction("carryCatalogTransitionAdjacency"),
    /carryCatalogTransitionMetricAt\(band\);carryCatalogTransitionAdjacencyAt\(band\)/,
    "the fused carry preserves metric publication before adjacency remapping for each row");
  assert.match(transition,
    /run\("describeCatalogRows",[\s\S]*0,pass,204[\s\S]*run\("resolveCatalogAdjacency",[\s\S]*0,pass,204/,
    "descriptor and adjacency work consume the GPU-authored dense row work domain");
  assert.doesNotMatch(transition,
    /run\("(?:commitRows|commitFaces|commitTransition|commitCatalogTransitionState|commitControls)"/,
    "topology, adjacency, and faces remain unpublished until closest-point validation");
  const closest = source.slice(source.indexOf('case"closest-point-extension"'),
    source.indexOf('case"power-publication"'));
  assert.match(closest,
    /run\("commitRows"[\s\S]*run\("commitFaces"[\s\S]*run\("commitTransition"[\s\S]*run\("commitControls"/,
    "one final validated candidate replaces every immutable Section 5 authority");
  assert.doesNotMatch(closest, /run\("publish"/,
    "the first commit kernel publishes the private candidate gate without a launch-only singleton");
  assert.doesNotMatch(closest, /commitCatalogTransitionState/,
    "the terminal control commit publishes the directory epoch without another singleton launch");
  const workCount = wgslFunction("transitionDeltaWorkCount");
  assert.match(workCount, /returntransitionControl\.rowCount/);
  assert.match(wgslFunction("transitionDeltaRow"),
    /select\(INVALID,item,item<transitionControl\.rowCount\)/);
  const classify = wgslFunction("classifyCatalogTransitionDelta");
  assert.match(classify,
    /affected=transitionIdentityAffected\(band\).*if\(!affected&&band>=transitionControl\.coreEnd&&band<transitionControl\.support3NodeEnd\)\{affected=transitionSupportNeighborhoodAffected\(band\);\}if\(!affected\)\{affected=transitionFaceNeighborhoodAffected\(band\);\}/s,
    "identity changes short-circuit generation-fixed owner probes while preserving the complete axial face 1-ring delta");
  assert.match(wgslFunction("transitionSupportNeighborhoodAffected"),
    /for\(varbit=0u;bit<18u;bit\+=1u\).*transitionIdentityAffected\(neighbor\)/s,
    "the paper's face/edge-neighbor descriptor is the exact support delta boundary");
  const faceDelta = wgslFunction("transitionFaceNeighborhoodAffected");
  assert.match(faceDelta,
    /sampleCount=select\(1u,4u,row\.size>1u\).*positive<2u.*sample<sampleCount.*priorNeighbor=committedContainingRow\(q\).*transitionPriorRow\(neighbor\)!=priorNeighbor.*transitionIdentityAffected\(neighbor\)/s,
    "every unchanged incident row probes all six faces and all four 2:1 quadrants when an owner changes");
  assert.doesNotMatch(classify, /band>=transitionControl\.support3NodeEnd&&transitionFaceNeighborhoodAffected/,
    "the face 1-ring closure applies to core and support rows, not only terminal endpoints");
  assert.match(wgslFunction("describeCatalogBandRows"),
    /band>=transitionControl\.support3NodeEnd.*return/s,
    "S3 face invalidation never expands metric reconstruction past the S2 node prefix");
  assert.match(wgslFunction("resolveCatalogTransitionAdjacency"),
    /band>=transitionControl\.support3NodeEnd.*return/s,
    "S3 face invalidation never expands Delaunay adjacency reconstruction past the S2 node prefix");
  assert.match(wgslFunction("transitionPriorRow"),
    /letold=committedRowOfIdentity\(rows\[band\]\.cell,rows\[band\]\.size\)/,
    "unchanged support ids map through the immutable prior direct row table");
  assert.doesNotMatch(octreeFaceBandWGSL, /scatterCatalogTransitionDeltaRows/,
    "affected rows remain in a dense mask instead of any compacting append or scatter");
  assert.doesNotMatch(wgslFunction("carryCatalogTransitionMetrics"),
    /band>=transitionControl\.coreEnd/,
    "unchanged support metrics participate in exact carry instead of recurring suffix rebuilds");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /transitionAffectedCount|transitionControl\.rowCount-transitionControl\.coreEnd|transitionDeltaWorklist/,
    "the retired whole-support-suffix reconstruction path must stay deleted");
  const admission = wgslFunction("transitionDeltaAccepted");
  assert.match(admission, /rowDelta\[base\+7u\]==p\.powerGeneration/);
  assert.match(admission, /rowDelta\[base\+8u\]==ROW_DELTA_VALID/);
  assert.match(admission, /requested==carried\+added&&requested==previous\+added-retired/);
});

test("co-spherical entry 7946 closes its axial-star octahedron in the immutable catalog", () => {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(
    bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
  ));
  let maximumSelectorCoordinate = 0;
  const selectorSizeRatios = new Set<number>();
  for (let offset = 0; offset < catalog.tetrahedronVertexData.length; offset += 4) {
    maximumSelectorCoordinate = Math.max(maximumSelectorCoordinate,
      Math.abs(catalog.tetrahedronVertexData[offset]),
      Math.abs(catalog.tetrahedronVertexData[offset + 1]),
      Math.abs(catalog.tetrahedronVertexData[offset + 2]));
    selectorSizeRatios.add(catalog.tetrahedronVertexData[offset + 3]);
  }
  assert.equal(maximumSelectorCoordinate, 1.5,
    "the bounded catalog-search radius must cover every generated selector");
  assert.deepEqual([...selectorSizeRatios].sort((a, b) => a - b), [0.5, 1, 2],
    "the catalog search enumerates every dyadic owner scale");
  const entry = 7946;
  const [first, count, flags] = catalog.tetrahedronHeaders.slice(entry * 3, entry * 3 + 3);
  assert.deepEqual([first, count, flags], [361244, 40, 0]);
  const point = [-0.375, -0.375, -0.375] as const;
  const cross = (a: readonly number[], b: readonly number[]) => [
    a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
  ] as const;
  const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const weights = (p: readonly number[], a: readonly number[], b: readonly number[], c: readonly number[]) => {
    const determinant = dot(a, cross(b, c));
    const wa = dot(p, cross(b, c)) / determinant;
    const wb = dot(a, cross(p, c)) / determinant;
    const wc = dot(a, cross(b, p)) / determinant;
    return [1 - wa - wb - wc, wa, wb, wc];
  };
  const contained = (value: readonly number[]) => value.every((component) => component >= -2e-6 && component <= 1.000002);
  const selectorPoint = (selector: number) => Array.from(
    catalog.tetrahedronVertexData.slice(selector * 4, selector * 4 + 3),
  );
  assert.equal(Array.from({ length: count }, (_, local) => {
    const packed = catalog.tetrahedronData[first + local];
    return weights(point, selectorPoint(packed & 255), selectorPoint((packed >>> 8) & 255),
      selectorPoint((packed >>> 16) & 255));
  }).some(contained), true,
  "the catalog must retain vertex-only co-spherical sites and cover the former central hole");

  const body = [-1, -1, -1], edgeA = [0, -1, -1], edgeB = [-1, 0, -1];
  assert.deepEqual(weights(point, body, edgeA, edgeB).map((value) => value === 0 ? 0 : value),
    [0.625, 0.375, 0, 0]);
  assert.match(wgslFunction("containingPublishedRow"), /rowOfIdentity\(cell\(origin\),size\)/,
    "spatial-directory hits query the exact owner origin and size");
  assert.doesNotMatch(octreeFaceBandWGSL, /surroundingOwnerDelaunay|SurroundingOwnerVectorMeasurement/,
    "the dynamic eight-owner Delaunay fallback is deleted; the immutable catalog is authoritative");
  const prepareCompletion = wgslFunction("preparePowerFaceRegularCompletion");
  assert.match(prepareCompletion,
    /oldAdvectionSeed\.valid=0u.*requested=oldAdvectionControl\.requestedFaces/s,
    "mandatory completion invalidates the downstream seed before checking its exact live count");
  assert.match(prepareCompletion,
    /requested!=powerFaceControl\[1\].*failPowerFaceRegularCompletion\(0u,1u,\(requested<<8u\)\|27u\)/s,
    "completion consumes the exact device-published generalized-face count without a copied uniform");
  assert.match(octreeFaceBandWGSL,
    /fn regularCompletionAuthorityFailures\(\)->u32\{var failures=0u;.*control\.generation!=sp\.fineGeneration.*pointControl\.generation!=sp\.fineGeneration.*powerFaceControl\[7\]!=sp\.p2/s,
    "rejection diagnostics preserve the exact retained-band and power-face authority predicates");
  assert.match(prepareCompletion,
    /authorityFailures=regularCompletionAuthorityFailures\(\).*failPowerFaceRegularCompletion\(0u,1u,\(authorityFailures<<8u\)\|21u\)/s,
    "a retained-authority rejection publishes its predicate bits without changing authority selection");
  assert.doesNotMatch(prepareCompletion,
    /priorFlags|priorFirstError|priorAdvected|priorValid|atomic(?:Load|Store|Add|Min|Or)\(&oldAdvection/,
    "completion neither resets a pseudo-attempt nor atomically synchronizes its single-owner controls");
  const complete = wgslFunction("completePowerFaceAdvectionFromRegularBand");
  assert.match(complete,
    /priorStatus==STATUS_VALID\|\|priorStatus==STATUS_REGULAR_COMPLETED/,
    "regular completion preserves every face already completed by transition interpolation");
  assert.match(complete,
    /powerFaceNormals\[i\]\.w=value;powerFaceCentroids\[i\]\.w=bitcast<f32>\(STATUS_REGULAR_COMPLETED\)/,
    "a completed scalar uses bounded per-face scratch until the sole publication reduction commits it");
  const continueMissingBand = wgslFunction("continueMissingBandVector");
  assert.match(continueMissingBand,
    /!velocityValid\(sampled\)&&sampled\.w==-20\.&&velocityValid\(previous\)/,
    "only the exact missing retained-band sentinel continues the already closest-point-extended vector");
  assert.match(complete,
    /vm=continueMissingBandVector\(vm,v0\).*va=continueMissingBandVector\(va,vm\)/s,
    "RK midpoint and departure may continue the last valid constant extension without masking other failures");
  assert.doesNotMatch(complete, /sp\.fineGeneration==2u/,
    "retained fine-band generation 2 remains a valid recurring interpolation epoch");
  const incident = wgslFunction("retainedBandIncidentVector");
  assert.match(incident,
    /for\(varaxis=0u;axis<3u;axis\+=1u\).*negativeBoundaryBit\(axis\).*grid\[axis\]<-f32\(sp\.maximumLeaf\).*positiveBoundaryBit\(axis\).*openCeiling=axis==1u&&!closed.*grid\[axis\]>extent\+f32\(sp\.maximumLeaf\).*reflectComponents\(boundary,flips\)/s,
    "a boundary-crossing RK query uses the bounded product ghost policy");
  assert.doesNotMatch(incident, /normal\.y/,
    "a tangential face may backtrace through the open ceiling");
  assert.match(incident, /grid\[axis\]=select\(extent-1e-5,2\.\*extent-grid\[axis\],closed\)/,
    "closed walls mirror while the authored open ceiling uses zero-gradient clamping");
  assert.equal(Math.min(16 - 1e-5, 16), 15.99999,
    "the 16-cell mini-dam ceiling normalizes to the final interior sample plane");
  const publishCompletion = wgslFunction("publishCompletedPowerFaceAdvection");
  assert.match(publishCompletion,
    /@builtin\(local_invocation_index\)lid:u32.*for\(vari=lid;i<requested;i\+=256u\)/s,
    "one cooperative workgroup validates the exact live prefix without a serial face loop");
  assert.match(publishCompletion, /status==STATUS_VALID/,
    "the final reduction accepts direct transition results");
  assert.match(publishCompletion,
    /status==STATUS_REGULAR_COMPLETED.*powerFaces\[i\]\.normalVelocity=value/s,
    "the final reduction commits regular-band scalars");
  assert.match(publishCompletion,
    /oldAdvectionControl\.valid=select\(0u,VALID,complete\).*oldAdvectionSeed\.valid=select\(0u,VALID,complete\)/s,
    "one complete reduction publishes both advection and downstream velocity-seed authority");
  assert.doesNotMatch(publishCompletion, /atomic(?:Load|Store|Add|Min|Or)/,
    "the sole completion publisher has no recurring synchronization atomics");
  const completionEncoder = compact(
    WebGPUOctreeFaceClosestPointExtension.prototype.encodeCompletePowerFaceAdvectionFromRegularBand,
  );
  assert.match(completionEncoder,
    /words\.set\(\[\.\.\.input\.dimensions,positive\(input\.maximumLeafSize,"Oldface-bandmaximumleaf"\),0,0,this\.plan\.rowDirectoryCapacity,this\.plan\.rowCapacity\]\)/,
    "completion parameters do not substitute a host capacity for the live generalized-face count");
  assert.match(completionEncoder,
    /complete\.dispatchWorkgroupsIndirect\(input\.faces\.liveFaceDispatch,0\)/,
    "regular completion launches only the device-published live face prefix");
  assert.match(completionEncoder,
    /bind\(this\.preparePowerAdvectionCompletionPipeline,\[5,20,36,37,38,39,48,58,59\]\)/,
    "the recurring preparation bind group exactly includes every reachable face payload");
  assert.doesNotMatch(completionEncoder,
    /copyBufferToBuffer|Math\.ceil\(this\.plan\.powerFaceCapacity\/64\)|coldPowerAdvection/,
    "the copied-count pass break, capacity launch, and duplicate cold producer stay deleted");
  assert.doesNotMatch(completionEncoder, /beginComputePass|newPassBroker|broker\.fence/,
    "prepare, completion, and publication remain in the caller-owned broker pass");
  assert.match(completionEncoder,
    /constprepare=broker\.compute[\s\S]*constcomplete=broker\.compute[\s\S]*constpublish=broker\.compute/,
    "the mandatory transaction executes source gate, bulk completion, and publication in order");
  assert.match(completionEncoder, /input\.powerGeneration<=1/,
    "cold start is owned exclusively by the primary advection seed transaction");
});

test("Section 5 admits only one clean same-generation fine/coarse publication", () => {
  const clean = new Uint32Array([0, 1, 1, 1, 1, 0, 1, 0]);
  const rolledBack = new Uint32Array([16, 1, 1, 1, 1, 1, 1, 8]);
  assert.equal(octreeFaceBandCoarseGenerationPairIsValid(7, 7, clean), true);
  assert.equal(octreeFaceBandCoarseGenerationPairIsValid(6, 7, clean), false);
  assert.equal(octreeFaceBandCoarseGenerationPairIsValid(7, 7, rolledBack), false,
    "retagged rollback scratch is not a Section 5 publication");
  assert.equal(octreeFaceBandCoarseGenerationPairIsValid(6, 7, rolledBack), false,
    "the former one-generation rollback exception must stay removed");

  const gate = wgslFunction("validCoarseGeneration");
  assert.match(gate,
    /fineTopologyControl\[0\]==0u&&fineTopologyControl\[4\]==1u&&fineTopologyControl\[5\]==0u&&fineTopologyControl\[7\]==0u/);
  assert.match(gate, /returnclean&&coarseGeneration==fineGeneration/);
  assert.doesNotMatch(gate, /rollback|coarseGeneration\+1u/,
    "GPU admission must not reinterpret rejected A/B scratch as current paper authority");
});

test("topology publishes three exact catalog-closed anchor tiers plus terminal endpoints", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const topology = source.slice(source.indexOf('case"topology-build"'),
    source.indexOf('case"transition-adjacency"'));
  assert.match(topology,
    /run\("prepare"[\s\S]*run\("clearSupportIdentityMarks"[\s\S]*run\("buildTopologyDelta"[\s\S]*run\("emitCatalogSupport1"[\s\S]*run\("markSupport1"[\s\S]*run\("finalizeSupport1"[\s\S]*run\("scatterSupport1"[\s\S]*run\("emitCatalogSupport2"[\s\S]*run\("markSupport2"[\s\S]*run\("scatterSupport2"[\s\S]*run\("emitCatalogSupport3"[\s\S]*run\("markSupport3"[\s\S]*run\("scatterSupport3"[\s\S]*run\("emitEndpointSupport4"[\s\S]*run\("markSupport4"[\s\S]*run\("scatterSupport4"[\s\S]*run\("markPublishedSupportRows"[\s\S]*run\("scatterCanonicalRowDirectory"[\s\S]*run\("validateRowDirectory"[\s\S]*run\("classifyRowSigns"/);
  assert.doesNotMatch(topology,
    /sortUniqueCatalogSupport|sortUniqueEndpointSupport|sortValidateRowDirectory/);
  assert.match(wgslFunction("findSiteInCount"), /while\(low<high\).*powerKeyLess/s,
    "power rows are resolved by the producer's canonical sorted directory");
  assert.doesNotMatch(wgslFunction("findSiteInCount"), /probe|siteHash/,
    "the deleted power-row open hash has no lookup fallback");
  const prepare = wgslFunction("prepareFaceBandDelta");
  assert.match(prepare, /!topologyDeltaAccepted\(\)/,
    "a missing or inconsistent row delta rejects the candidate rather than rebuilding globally");
  assert.match(prepare, /control\.reserved0=INVALID/,
    "the packed producer-failure diagnostic must begin at the invalid sentinel");
  assert.match(prepare,
    /firstFaceEmissionFailure,INVALID.*firstPhiFailure,INVALID.*firstClosestPointFailure,INVALID.*firstVectorFailure,INVALID.*firstCptNoOwnerFailure,INVALID.*firstCptSupportOwnerFailure,INVALID.*firstCptNoSimplexFailure,INVALID.*firstCptMissingVertexFailure,INVALID/s,
    "every stable stage/cause diagnostic starts from the invalid sentinel each generation");
  assert.match(prepare,
    /rowDirectory\[candidateState\]=core;rowDirectory\[candidateState\+1u\]=core;rowDirectory\[candidateState\+2u\]=core;rowDirectory\[candidateState\+3u\]=0u/,
    "the primary transaction initializer publishes the three exact empty support prefixes");
  const admission = wgslFunction("topologyDeltaAccepted");
  assert.match(admission, /rowDelta\[base\+7u\]==p\.powerGeneration/);
  assert.match(admission, /rowDelta\[base\+8u\]==ROW_DELTA_VALID/);
  assert.match(admission, /current==carried\+added&&current==previous\+added-retired/);
  assert.doesNotMatch(admission, /committedValid\(\)|committedGeneration\(\)|priorReady/,
    "candidate construction must recover without an exact committed predecessor");
  const transitionIdentity = wgslFunction("transitionIdentityAffected");
  assert.match(transitionIdentity, /!committedTransitionValid\(\).*returntrue/,
    "a missing predecessor converts the candidate transaction to a full affected-row rebuild");
  const build = wgslFunction("buildFaceBandTopologyDelta");
  assert.match(build, /@builtin\(global_invocation_id\)g.*letband=g\.x.*band>=core/s,
    "the immutable power prefix is copied row-parallel rather than by a singleton loop");
  const emit = wgslFunction("emitCatalogNeighborhood");
  assert.match(emit,
    /for\(varslot=0u;slot<MAX_GUARDS;slot\+=1u\).*emitRegularFaceEndpointSupport\(row,base\).*emitUniformCatalogSupport\(row,base\).*emitDelaunayCatalogSupport\(row,base,header,metric\.y\)/s,
    "each source emits one fixed 24-endpoint plus 36-catalog request record");
  assert.match(emit,
    /letsupport2End=select\(support1End,rowDirectory\[state\+2u\].*letsupport3End=select\(support2End,rowDirectory\[state\+3u\].*support2End,tier==4u.*support3End,tier==4u/s,
    "the terminal tier closes every S3 anchor over the same exact catalog request stream");
  assert.match(wgslFunction("emitFaceBandCatalogSupport1"), /emitCatalogNeighborhood\(g\.x,1u\)/);
  assert.match(wgslFunction("emitFaceBandCatalogSupport2"), /emitCatalogNeighborhood\(g\.x,2u\)/);
  assert.match(wgslFunction("emitFaceBandCatalogSupport3"), /emitCatalogNeighborhood\(g\.x,3u\)/);
  assert.match(wgslFunction("emitFaceBandEndpointSupport4"), /emitCatalogNeighborhood\(g\.x,4u\)/);
  assert.doesNotMatch(octreeFaceBandWGSL, /fn emitEndpointNeighborhood\(/,
    "the incomplete endpoint-only S3 closure and its narrower candidate stride stay deleted");
  const mark = wgslFunction("markSupportSource", octreeFaceBandSupportScatterWGSL);
  assert.match(mark,
    /source\*MAX_GUARDS.*slot<MAX_GUARDS.*identityRank\(cellKey,value\.size\).*alreadyPublished\(cellKey,value\.size\).*atomicStore\(&supportScratch\[rank\],1u\)/s,
    "every tier atomically marks the collision-free identity of its complete fixed-fanout request stream");
  const blockScan = wgslFunction("scanSupportIdentityBlocks", octreeFaceBandSupportScatterWGSL);
  assert.match(blockScan,
    /workgroup_size\(256\).*scanValues\[lane\]=value.*offset<256u.*prefixBase\(\)\+rank.*blockTotalBase\(\)\+wid\.x/s,
    "support identity marks are scanned in parallel 256-entry blocks");
  const prefix = wgslFunction("prefixSupportIdentityBlocks", octreeFaceBandSupportScatterWGSL);
  assert.match(prefix,
    /laneBlockRange\(lane\).*localTotal.*scanValues\[lane\].*blockTotalBase\(\)\+blockCount\(\)/s,
    "one bounded block-total scan publishes the exact global support count");
  const finalize = wgslFunction("finalizeTier", octreeFaceBandSupportScatterWGSL);
  assert.match(finalize,
    /tier==1u\).*state\+1u.*tier==2u\).*state\+2u.*tier==3u\).*state\+3u.*controlWords\[2u\]=end/s,
    "the candidate publishes distinct S1, S2, S3-node, and terminal endpoint prefixes");
  const scatter = wgslFunction("scatterTier", octreeFaceBandSupportScatterWGSL);
  assert.match(scatter,
    /rank\/6u.*rank%6u.*tierFlag\(tier\).*committedRowOfIdentity.*rowDirectory\[identity\]=band\+1u/s,
    "canonical identity rank deterministically scatters each unique row and retains prior-row linkage");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /addCurrentSupportRing|retirePriorSupportRing|topologyRingProbe|retireSupportIdentity/,
    "the serial geometric one-ring/carry implementation and its backing reference counts stay deleted");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /supportHash|insertCatalogSupport|buildFaceBandCatalogClosure|publishFaceBandCatalogClosure/,
    "the singleton support hash, probing, and publication path is deleted");
  const faceEndpoints = wgslFunction("emitRegularFaceEndpointSupport");
  assert.match(faceEndpoints,
    /sampleCount=select\(1u,4u,row\.size>1u\).*axis<3u.*side<2u.*sample<4u/s,
    "the paper's regular-face graph closes six unit endpoints or exact 2:1 face quadrants");
  assert.match(faceEndpoints, /writeSupportCandidate\(base,request,candidate\)/,
    "regular-face endpoints join the same exact fixed-fanout identity stream");
  const directory = wgslFunction("scatterCanonicalRowDirectory", octreeFaceBandSupportScatterWGSL);
  assert.match(directory,
    /position=atomicLoad.*cellKey=rank\/6u.*size=1u<<\(rank%6u\).*rowDirectory\[position\*2u\]=cellKey.*rowDirectory\[position\*2u\+1u\]=row/s,
    "the same cell-major, size-minor identity rank directly publishes the canonical directory");
  assert.match(wgslFunction("validateFaceBandRowDirectoryIndex"),
    /rowIdentityLess\(rowDirectory\[\(index-1u\)\*2u\],rows\[prior\]\.size,key,rows\[row\]\.size\)/,
    "duplicate and unsorted exact (cell,size) row identities fail closed");
  assert.match(wgslFunction("validateFaceBandRowDirectory"),
    /index<count.*validateFaceBandRowDirectoryIndex\(index,count\)/s,
    "the parallel directory publication retains fail-closed validation over every live row");
});

test("S3 anchor closure publishes the exact selector missing from the row-1216 failure", () => {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(
    bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
  ));
  const selector = Array.from(catalog.tetrahedronVertexData.slice(19 * 4, 19 * 4 + 4));
  assert.deepEqual(selector, [-0.75, -0.25, 0.75, 0.5]);

  // Exact failure fixture: S2 row cell 4718 is the size-two leaf at
  // [14,16,10]. Transform 3 reflects x/y, so selector 19 requires the
  // size-one owner at [16,17,12], cell 5608.
  const dims = [24, 18, 16] as const;
  const anchorCell = 4718;
  const anchorOrigin = [
    anchorCell % dims[0],
    Math.floor(anchorCell / dims[0]) % dims[1],
    Math.floor(anchorCell / (dims[0] * dims[1])),
  ];
  assert.deepEqual(anchorOrigin, [14, 16, 10]);
  const anchorSize = 2;
  const transformed = [-selector[0], -selector[1], selector[2]];
  const neighborSize = Math.round(anchorSize * selector[3]);
  const neighborOrigin = transformed.map((component, axis) =>
    Math.round(anchorOrigin[axis] + anchorSize * 0.5 + anchorSize * component
      - neighborSize * 0.5));
  const neighborCell = neighborOrigin[0]
    + dims[0] * (neighborOrigin[1] + dims[1] * neighborOrigin[2]);
  assert.deepEqual(neighborOrigin, [16, 17, 12]);
  assert.equal(neighborSize, 1);
  assert.equal(neighborCell, 5608,
    "the reported valid owner must be present before S2 Delaunay adjacency is resolved");

  assert.match(wgslFunction("emitFaceBandCatalogSupport3"),
    /emitCatalogNeighborhood\(g\.x,3u\)/);
  assert.match(wgslFunction("emitCatalogNeighborhood"),
    /emitDelaunayCatalogSupport\(row,base,header,metric\.y,&ownerCache\)/,
    "S3 publication includes selector 19 instead of stopping at regular face endpoints");
});

test("face-band owner sampling consumes the sorted brick-relative owner-page ABI", () => {
  assert.doesNotMatch(octreeFaceBandWGSL, /fn decodeOwner\(/,
    "the dense owner-word decoder must be deleted after the paged cutover");
  const decode = wgslFunction("decodePagedOwner");
  assert.match(decode, /letexponent=\(word>>18u\)&7u/);
  assert.match(decode,
    /letbrickOrigin=vec3i\(\(q\/vec3u\(8u\)\)\*vec3u\(8u\)\)/);
  assert.match(decode,
    /i32\(word&63u\)-32.*i32\(\(word>>6u\)&63u\)-32.*i32\(\(word>>12u\)&63u\)-32/s);
  const publication = wgslFunction("pagedOwnerPublicationValid");
  assert.match(publication, /arrayLength\(&owners\)<16u\|\|owners\[15\]!=0x4f574e52u/,
    "short or non-owner buffers fail closed instead of selecting a legacy format");
  assert.match(publication,
    /accepted!=0u&&owners\[10\]==1u&&owners\[11\]==accepted&&owners\[2\]==0u&&owners\[12\]==0u&&resident<=capacity&&owners\[0\]==capacity-resident/,
    "Section 5 admits only the owner arena's internally complete immutable publication");
  assert.match(publication,
    /recordPageOffset==16u\+capacity&&payloadOffset==recordPageOffset\+capacity/);
  assert.match(publication,
    /capacity<=\(arrayLength\(&owners\)-payloadOffset\)\/512u/,
    "Section 5 requires every declared physical page to fit in the owner arena");
  assert.doesNotMatch(publication,
    /p\.powerGeneration|p\.generation|owners\[7\]\s*[!=]=|owners\[7\]\s*[+-]\s*1u/,
    "topology-owner and Section 5 generations remain independent namespaces with no adjacent-generation fallback");
  const lookup = wgslFunction("ownerAt");
  assert.match(lookup, /if\(!pagedOwnerPublicationValid\(\)\)\{returninvalidOwner\(\);\}/,
    "every paged Section 5 owner query is fenced by the exact self-publication contract");
  assert.doesNotMatch(lookup, /denseIndex|decodeOwner|owners\[15\]/,
    "ownerAt has one paged ABI and no magic/length format switch");
  assert.doesNotMatch(lookup,
    /owners\[7\].*p\.(?:powerGeneration|generation)|p\.(?:powerGeneration|generation).*owners\[7\]/,
    "owner lookup cannot reject a current topology snapshot because the power or fine epoch advanced");
  assert.match(lookup,
    /letrecordKeyOffset=16u;letrecordPageOffset=owners\[5\];letpayloadOffset=owners\[6\]/);
  assert.match(lookup, /while\(low<high\).*owners\[recordKeyOffset\+middle\]/s);
  assert.match(lookup, /letencoded=owners\[recordPageOffset\+low\]/);
  assert.match(lookup,
    /if\(low>=resident\|\|owners\[recordKeyOffset\+low\]!=key\)\{returnresidentCanonicalOwner\(q\);\}/,
    "an absent logical page retains the paper's canonical coarse-air owner");
  assert.doesNotMatch(lookup, /hashCapacity|freeListOffset|0x9e3779b1u|for\(varprobe/,
    "the retired owner-page hash/free-list lookup must not survive in Section 5");
});

test("Section 5 row lookup never searches unpublished capacity tails", () => {
  assert.match(octreeFaceBandWGSL,
    /fn rowDirectoryCount\(\)->u32\{let state=p\.rowDirectoryCapacity\*2u;[\s\S]*rowDirectory\[state\+3u\]==COMMITTED_TRANSITION_VALID\)\{return rowDirectory\[state\+1u\];\}return rowDirectory\[state\];\}[\s\S]*fn rowOfIdentity\(cellKey:u32,size:u32\)->u32\{let count=min\(rowDirectoryCount\(\),p\.rowDirectoryCapacity\);let slot=rowIdentitySlot\(cellKey,size\)/,
    "lookup must bound the direct slot result by the exact candidate or committed live count");
  assert.match(wgslFunction("prepareFaceBandDelta"),
    /rowDirectory\[candidateState\+2u\]=core;rowDirectory\[candidateState\+3u\]=0u/,
    "each candidate transaction clears the committed-state discriminator before publishing its live count");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /\bfn (?:rowOf|committedRowOf|sampleBandRow)\s*\(/,
    "cell-only lookup helpers are deleted because overlapping octree levels can share an origin");
  assert.match(wgslFunction("sortUniqueFaceBandCatalogSupport"),
    /tier==2u\)\{rowDirectory\[candidateState\+2u\]=end[\s\S]*else\{rowDirectory\[candidateState\]=end/,
    "the topology transaction must publish the exact sorted prefix length beside its directory");
});

test("Section 5 row identity retains overlapping origins and resolves size-one [16,0,12]", () => {
  const dims = [24, 18, 16] as const;
  const cell = ([x, y, z]: readonly [number, number, number]) =>
    x + dims[0] * (y + dims[1] * z);
  const failureOrigin = [16, 0, 12] as const;
  const failureCell = cell(failureOrigin);
  assert.equal(failureCell, 5200);

  const identitySlot = (cellKey: number, size: number) =>
    Math.log2(size) * dims[0] * dims[1] * dims[2] + cellKey;
  const direct = new Map<number, number>([
    [identitySlot(failureCell - 1, 4), 40],
    [identitySlot(failureCell, 1), 351],
    [identitySlot(failureCell, 2), 352],
    [identitySlot(failureCell + 1, 1), 353],
  ]);
  const rowOfIdentity = (cellKey: number, size: number): number | undefined =>
    direct.get(identitySlot(cellKey, size));
  assert.equal(rowOfIdentity(failureCell, 1), 351,
    "the exact size-one support row must not be shadowed by the size-two row at the same origin");
  assert.equal(rowOfIdentity(failureCell, 2), 352);
  assert.equal(rowOfIdentity(failureCell, 4), undefined);

  assert.match(wgslFunction("rowIdentityLess"),
    /cellA<cellB\|\|\(cellA==cellB&&sizeA<sizeB\)/);
  assert.match(wgslFunction("rowOfIdentity"),
    /rowIdentitySlot\(cellKey,size\)[\s\S]*candidate\.cell==cellKey&&candidate\.size==size/,
    "candidate direct lookup validates both identity components");
  assert.match(wgslFunction("committedRowOfIdentity"),
    /rowIdentitySlot\(cellKey,size\)[\s\S]*candidate\.cell==cellKey&&candidate\.size==size/,
    "warm-delta direct lookup validates both identity components");
  assert.match(wgslFunction("sortUniqueFaceBandCatalogSupport"),
    /prior\.cellPlusOne==value\.cellPlusOne&&prior\.size==value\.size/,
    "support publication deduplicates only identical (cell,size) requests");
  assert.match(wgslFunction("publishedSupportRecord"),
    /rowOfIdentity\(cellPlusOne-1u,size\)!=INVALID/,
    "published support resolves exact level-sensitive identity in O(1)");
  assert.match(wgslFunction("sortUniqueFaceBandCatalogSupport"),
    /rowDirectory\[identitySlot\]=band\+1u[\s\S]*rowDirectory\[candidateState\]=end/,
    "each completed tier publishes its exact identity slots before exposing the prefix");
  assert.doesNotMatch(octreeFaceBandWGSL, /\bfn (?:rowOf|committedRowOf|sampleBandRow)\s*\(/);
});

test("closed-wall ghost policy has a dedicated parameter word and preserves face strides", () => {
  const constructor = compact(WebGPUOctreeFaceClosestPointExtension);
  const encode = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(constructor, /label:"octreeface-bandparameters",size:112/);
  assert.match(encode, /constwords=newUint32Array\(24\)/,
    "the 96-byte authored topology/delta payload leaves the WGSL P padding zeroed");
  assert.match(encode, /words\[16\]=47\|\(input\.closedTop\?16:0\)/);
  assert.match(encode,
    /words\.set\(\[input\.rowDelta\.controlOffsetWords,input\.rowDelta\.newToOldOffsetWords,input\.rowDelta\.affectedRowsOffsetWords,1,input\.rowDelta\.oldToNewOffsetWords,input\.rowDelta\.dirtyRowsOffsetWords\],17\)/,
    "both exact maps and the dirty/affected lists must fit in the uploaded parameter array");
  assert.match(octreeFaceBandWGSL,
    /axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,rowDeltaControl:u32/,
    "the boundary mask must not alias either regular-face stride word");
});

test("factor-4 GPU face band is compact, bounded, and has no fine velocity channel", () => {
  const plan = planOctreeFaceBandGPU(100, 20, 4, 4, [24, 18, 16]);
  assert.equal(plan.ownerCandidatesPerBrick, 1);
  assert.equal(plan.rowCapacity, 100,
    "the pressure capacity is one shared deterministic owner-row arena");
  assert.equal(OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW, 12);
  assert.equal(OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, 24);
  assert.equal(plan.faceCapacity, plan.rowCapacity * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW);
  assert.equal(OCTREE_FACE_BAND_ROW_BYTES, 32);
  assert.equal(OCTREE_FACE_BAND_FACE_BYTES, 64,
    "four u32 fields, two vec4f values, and four scalar fields define the face ABI");
  assert.equal(plan.rowBytes, plan.rowCapacity * OCTREE_FACE_BAND_ROW_BYTES,
    "the allocation must match the WGSL row ABI with representative/min/max phi");
  assert.equal(plan.identityToRowOffsetWords, plan.rowDirectoryCapacity * 2 + 4);
  assert.equal(plan.identityToRowCapacity, 24 * 18 * 16 * 6,
    "the six collision-free direct planes are constructor-sized from immutable dimensions");
  assert.equal(plan.rowDirectoryBytes,
    (plan.identityToRowOffsetWords + plan.identityToRowCapacity) * 4);
  assert.equal(plan.bandFaceBytes, plan.faceCapacity * OCTREE_FACE_BAND_FACE_BYTES);
  assert.equal(plan.indirectDispatchBytes, 264,
    "two exact support-tier records plus live row/face prefix records replace capacity-sized dispatches");
  assert.equal(plan.allocatedBytes,
    plan.bandFaceBytes + plan.incidenceBytes,
    "face storage has no legacy closest-point repair ping-pong snapshots");
  assert.match(compact(planOctreeFaceBandGPU),
    /gpuAllocatedBytes:rowBytes\+bandFaceBytes\+incidenceBytes\+rowDirectoryBytes\+rowDirectoryScratchBytes/,
    "complete GPU accounting contains no repair heap or full-face ping-pong snapshots");
  assert.match(compact(planOctreeFaceBandGPU),
    /\+rowBytes\+rowDirectoryBytes\+bandFaceBytes\+incidenceBytes\+transitionAdjacencyBytes\+transitionMetricBytes\+OCTREE_FACE_BAND_CONTROL_BYTES\+OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES\+rowCapacity\*4\+endpointIncomingCountBytes\+bandPhiFrontierBytes\+liveFaceWorklistBytes\+powerVelocityScratchBytes/,
    "GPU accounting includes the unpublished B-side topology and its directory-owned immutable epoch");
  assert.equal(plan.endpointIncomingCountBytes, plan.rowCapacity * 4,
    "terminal reverse adjacency uses one bounded append counter per immutable row identity");
  assert.equal(plan.bandPhiFrontierBytes, (plan.rowCapacity + 1) * 4,
    "mutable phi rows have a dedicated count-delimited frontier that cannot corrupt immutable endpoint state");
  assert.equal(plan.liveFaceWorklistBytes, (plan.faceCapacity + 1) * 4,
    "one canonical live-face work package is shared by every downstream face stage");
  const implementation = compact(WebGPUOctreeFaceClosestPointExtension);
  assert.doesNotMatch(implementation, /frontierA|nextFrontier|binding\(16\).*frontier|binding\(17\).*frontier/,
    "the retired serial-heap frontier scratch and shader bindings must not be allocated");
  assert.doesNotMatch(implementation, /closestPointRepair|cptRepair/,
    "the direct paper closest-point sample has no graph-propagation fallback allocation");
  assert.equal(plan.velocityBytes, plan.rowCapacity * 16,
    "only transient regular octree rows receive full vectors; fine samples do not");
  assert.equal(plan.transitionAdjacencyCapacity,
    plan.metricRowCapacity * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra,
    "S0, S1, and S2 metric owners retain complete catalog adjacency");
  assert.equal(plan.transitionAdjacencyBytes,
    plan.transitionAdjacencyCapacity * OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES);
  assert.equal(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES, 160,
    "the transition producer retains its ABI-stable gate/failure record and appended S4 prefix");
  assert.equal(OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES, 64);
  assert.equal(plan.powerVelocityScratchBytes, plan.powerFaceCapacity * 16,
    "split regular-to-power scratch carries scalar bits, target marker, and both mapped endpoint bands");
  assert.equal(plan.transientPowerFaceCapacity,
    plan.rowCapacity * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
    "every possible S0/S1 catalog face has one bounded transient physical-face slot");
  assert.equal(plan.transientPowerFaceBytes,
    plan.transientPowerFaceCapacity * OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES);
  assert.equal(plan.transientPowerIncidenceBytes,
    plan.rowCapacity * Math.max(
      OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
      OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW + Math.max(
        OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
        OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS,
      ),
    ) * OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES,
    "reused incidence scratch covers all 24 face endpoints plus the 36-request catalog bound");
  assert.equal(plan.catalogSupportCandidateCapacity,
    plan.rowCapacity * (OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW + Math.max(
      OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
      OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS,
    )));
  assert.equal(plan.catalogSupportCandidateBytes, plan.catalogSupportCandidateCapacity * 8);
  const identityCount = 24 * 18 * 16 * 6;
  assert.equal(plan.catalogSupportScratchBytes, Math.max(
    plan.catalogSupportCandidateBytes,
    (identityCount * 2 + Math.ceil(identityCount / 256) + 1) * 4,
  ), "support scratch covers parallel marks, local prefixes, block totals, and later face repair");
  assert.equal(plan.transientPowerRowBytes,
    (plan.rowCapacity + 1) * OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES);
  assert.equal(OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES, 64);
  assert.equal(plan.maximumDirectWorkgroups, Math.ceil(Math.max(plan.rowCapacity, plan.faceCapacity,
    plan.powerFaceCapacity, plan.transientPowerFaceCapacity, 24 * 18 * 16) / 64));
  assert.match(compact(WebGPUOctreeFaceClosestPointExtension),
    /this\.plan\.maximumDirectWorkgroups>device\.limits\.maxComputeWorkgroupsPerDimension/,
    "a buffer-admissible plan still fails closed when its one-dimensional dispatch would exceed the adapter limit");
});

test("factor-8 B4 face-band discovery deduplicates the eight fine bricks containing one finest cell", () => {
  const plan = planOctreeFaceBandGPU(100, 160, 4, 8, [16, 16, 16], 300);
  assert.equal(plan.ownerCandidatesPerBrick, 1);
  assert.equal(plan.powerFaceCapacity, 300);
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(source, /input\.fine\.plan\.fineFactor!==4&&input\.fine\.plan\.fineFactor!==8/);
  assert.doesNotMatch(source, /brickResolution!==input\.fine\.plan\.fineFactor/);
});

test("transition adjacency scales with every owner candidate in wider fine bricks", () => {
  const plan = planOctreeFaceBandGPU(100, 20, 8, 4, [24, 18, 16]);
  assert.equal(plan.ownerCandidatesPerBrick, 8);
  assert.equal(plan.transitionAdjacencyCapacity,
    plan.metricRowCapacity * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra,
    "B8/factor-4 support adjacency includes every S0/S1/S2 owner");
  assert.equal(plan.transitionAdjacencyBytes,
    plan.transitionAdjacencyCapacity * OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES);
});

test("catalog row planning reserves a compact owner directory without legacy support arenas", () => {
  const plan = planOctreeFaceBandGPU(100, 20, 8, 4, [24, 18, 16]);
  assert.equal(plan.rowCapacity, 100);
  assert.equal(plan.metricRowCapacity, plan.rowCapacity);
  assert.equal(plan.transitionAdjacencyCapacity,
    plan.rowCapacity * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra);
  assert.doesNotMatch(compact(planOctreeFaceBandGPU),
    /supportDirectory|support0RowCapacity|support1RowCapacity|support2RowCapacity|support3NodeRowCapacity|endpointRowCapacity|guardRowCapacity/);
  assert.doesNotMatch(compact(WebGPUOctreeFaceClosestPointExtension),
    /supportDirectory|guardCandidate|supportRingArena/);
});

test("face-band control diagnostics distinguish fail-closed causes", () => {
  assert.equal(OCTREE_FACE_BAND_CONTROL_BYTES, 128);
  const words = new Uint32Array(32);
  words.set([1 | 2 | 64 | 128, 91, 120, 315, 630, 7, 0x8000_0000, 11, 8, 300, 15, 0, 4, 27, 3, 44]);
  words.set([292, 4, 3, 0, 0, 0, 0, 0], 16);
  words.set([71, 82, 93, 104], 24);
  words[19] = 211;
  words[20] = 255;
  words.set([312, 413, 514], 21);
  words.set([615, 716, 817, 918], 28);
  assert.deepEqual(unpackOctreeFaceBandControl(words), {
    flags: 195, firstError: 91, rowCount: 120, faceCount: 315, incidenceCount: 630,
    generation: 7, valid: false, seedCount: 8, acceptedCount: 300,
    unresolvedCount: 15, sampleFailures: 4, coarsePhiSamples: 27, coarsePhiFailures: 3,
    bandPhiExtensions: 44,
    closestPointFaces: 292, closestPointFailures: 4, liquidInterpolationFailures: 3,
    cptNoOwnerFailures: 71, cptSupportOwnerFailures: 82,
    cptNoContainingSimplexFailures: 93, cptMissingLiquidVertexFailures: 104,
    stageFirstFailures: {
      faceEmission: 211, phi: 312, closestPoint: 413, vectorReconstruction: 514,
    },
    firstPhiFailureSlot: 255,
    firstClosestPointFailureSlotByCause: {
      noOwner: 615, supportOwner: 716, noContainingSimplex: 817, missingLiquidVertex: 918,
    },
    capacityFailure: true, invalidDirectory: true,
    invalidSource: false, invalidRow: false, invalidFace: false, invalidPhi: false,
    unresolved: true, incompleteVector: true, outsideFineBand: false,
  });
  words.set([0, 0xffff_ffff, 120, 315, 630, 7, 0x8000_0000, 11, 8, 315, 0, 0, 4]);
  assert.equal(unpackOctreeFaceBandControl(words).valid, true);
  assert.throws(() => unpackOctreeFaceBandControl(new Uint32Array(12)), /at least 13/);
});

test("air-band evaluation is atomic-free with one exact bounded catalog search", () => {
  const resolve = wgslFunction("resolveFinalPointVectorMeasured");
  assert.match(resolve, /candidateRowsTested\+=1u/,
    "each bounded local catalog candidate is counted");
  assert.match(resolve, /for\(vardz=-2i;dz<=2i[\s\S]*for\(vardy=-2i;dy<=2i[\s\S]*for\(vardx=-2i;dx<=2i/,
    "catalog candidates cover the proven five-cubed selector-radius box");
  assert.match(resolve, /if\(candidate>=bestRow\)\{continue;\}[\s\S]*bestRow=candidate;bestValue=value/,
    "spatial enumeration preserves the old ascending-row result");
  assert.doesNotMatch(resolve, /candidate<count|min\(p\.rowCapacity/,
    "the recurring sampler must not do work proportional to allocated row capacity");
  assert.doesNotMatch(resolve, /fallback|surroundingOwner|nearest|project/);
  assert.match(resolve,
    /interpolationGap=directReason==2u\|\|directReason==9u\|\|directReason==11u.*carrierEligible=interpolationGap&&initialAnchor<arrayLength\(&rows\)&&rows\[initialAnchor\]\.minimumPhi>=0\..*carrier=finalCellVector\(origin,size\).*delta=max\(max\(low-pointGrid,vec3f\(0\.\)\),pointGrid-high\).*distance<bestCarrierDistance\|\|\(distance==bestCarrierDistance&&candidate<bestCarrierRow\).*if\(bestCarrierRow!=INVALID\)\{returnResolvedPointVectorMeasurement\(bestCarrierValue,0u,1u,candidateRowsTested\);\}/s,
    "only air-side interpolation gaps may use the deterministic nearest completed carrier");
  assert.match(wgslFunction("containingPublishedRow"),
    /if\(any\(probe<vec3i\(0\)\)\|\|any\(probe>=vec3i\(p\.dims\)\)\)\{returnINVALID;\}.*ROW_SUPPORT3_ENDPOINT.*returnselected/s,
    "the bounded carrier path remains fail-closed outside the domain and on endpoint-only rows");
  assert.match(wgslFunction("resolveFinalPointVector"),
    /returnresolveFinalPointVectorMeasured\(initialAnchor,pointGrid\)\.value/,
    "power publication and retained repair share the sole bounded resolver");
  const hot = makeOctreeFaceBandAirSampleWGSL();
  assert.doesNotMatch(hot, /\batomic(?:Add|CompareExchangeWeak|Load|Max|Min|Or|Store)\b/,
    "the recurring sampler has no atomic operation");
  assert.match(hot, /@binding\(7\)var<storage,read>rowDirectory:array<u32>/,
    "the direct row table is immutable during recurring transport");
  assert.match(wgslFunction("rowOfIdentity", hot),
    /rowIdentitySlot\(cellKey,size\).*encoded=rowDirectory\[slot\].*candidate\.cell==cellKey&&candidate\.size==size/s,
    "row lookup is one exact bounded direct (cell,size) probe");
  assert.match(hot, /@binding\(23\)var<storage,read_write>sampleStatus:array<u32>/,
    "exclusive per-query statuses use ordinary stores");
  assert.match(wgslFunction("finalPointVector", hot), /finalSignedVector\(cornerOrigin,row\.size\)/,
    "ordinary same-size cube interpolation performs the exact immutable row-hash lookup");
  assert.match(wgslFunction("finalTetraPointVector", hot), /finalSelectorVector\(anchor,selectors\.[xyz]/,
    "catalog interpolation preserves the exact selector-to-owner lookup");
  const classify = wgslFunction("classifyAirBandVelocity", hot);
  assert.match(classify, /sampleStatus\[i\]=SAMPLE_EVALUATE/);
  assert.match(wgslFunction("sampledBandGenerationValid", hot),
    /paramsFine==fine&&\(band==fine\|\|\(power==fine&&band==predecessor\)\)/,
    "the fused sampler accepts only a current band or one exact predecessor on aligned clocks");
  assert.match(classify, /!sampledBandGenerationValid\(\)\|\|pointControl\.generation!=sp\.fineGeneration/,
    "air classification requires the current reconstructed point field even when its immutable band is the predecessor");
  const boundary = wgslFunction("airSampleGrid", hot);
  assert.match(boundary, /p\.closedBoundaryMask&negativeBoundaryBit\(axis\).*grid\[axis\]=0\./,
    "Stage-B velocity constantly extends through closed negative container walls");
  assert.match(boundary, /p\.closedBoundaryMask&positiveBoundaryBit\(axis\).*openCeiling=axis==1u&&!closed.*grid\[axis\]=f32\(sp\.dims\[axis\]\)-1e-5/,
    "Stage-B velocity extends through closed positive walls and the authored open ceiling");
  const evaluate = wgslFunction("evaluateAirBandVelocity", hot);
  assert.match(evaluate, /letsample=airSampleGrid\(positions\[i\]\.xyz\)/,
    "classification and evaluation consume one identical boundary-adjusted point");
  assert.match(evaluate, /resolveFinalPointVectorMeasured\(band,sample\.xyz\)\.value/);
  assert.doesNotMatch(evaluate, /\bcontrol\b/,
    "the 10-buffer evaluator must not make face-band control reachable");
  assert.equal("encodeAirSamples" in WebGPUOctreeFaceClosestPointExtension.prototype, false,
    "the split recurring air-sample API must be deleted after the fused cutover");
});

test("Dawn compiles recurring air sampling and production Section 5 entries at the ten-storage limit", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for recurring air-sampler checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const shaderModule = device.createShaderModule({ code: makeOctreeFaceBandAirSampleWGSL() });
  const errors = (await shaderModule.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of ["classifyAirBandVelocity", "evaluateAirBandVelocity", "finalizeAirBandVelocity"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  assert.equal(await device.popErrorScope(), null);
  const productionModule = device.createShaderModule({ code: octreeFaceBandWGSL });
  const productionErrors = (await productionModule.getCompilationInfo()).messages
    .filter(message => message.type === "error");
  assert.deepEqual(productionErrors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of ["emitTransientBandPowerGraph", "resolveCatalogTransitionAdjacency",
    "interpolatePowerFaceVector", "completePowerFaceAdvectionFromRegularBand"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: productionModule, entryPoint } });
  }
  assert.equal(await device.popErrorScope(), null,
    "every formerly over-limit Section 5 production entry must construct with exactly ten storage bindings");
  const storage = (words: Uint32Array<ArrayBuffer>, extraUsage = 0) => {
    const buffer = device.createBuffer({ size: Math.max(4, words.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage });
    device.queue.writeBuffer(buffer, 0, words);
    return buffer;
  };
  const controlWords = new Uint32Array(32); controlWords[5] = 7; controlWords[6] = 0x8000_0000;
  const pointWords = new Uint32Array(8); pointWords[3] = 7; pointWords[5] = 0x8000_0000;
  const sampleWords = new Uint32Array(12); sampleWords[5] = 4; sampleWords[9] = 7;
  const topologyWords = new Uint32Array(28); topologyWords[10] = 7;
  const statuses = storage(new Uint32Array([0x0200_0000, 0x0100_0123, 0x8000_0000, 0x0400_0000]),
    GPUBufferUsage.COPY_SRC);
  const control = storage(controlWords), point = storage(pointWords);
  const topologyParams = device.createBuffer({ size: 112,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(topologyParams, 0, topologyWords);
  device.queue.writeBuffer(params, 0, sampleWords);
  const finalize = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule,
    entryPoint: "finalizeAirBandVelocity" } });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(finalize); pass.setBindGroup(0, device.createBindGroup({
    layout: finalize.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: topologyParams } },
      { binding: 5, resource: { buffer: control } }, { binding: 20, resource: { buffer: params } },
      { binding: 23, resource: { buffer: statuses } }, { binding: 48, resource: { buffer: point } },
    ],
  })); pass.dispatchWorkgroups(1); pass.end();
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(statuses, 0, readback, 0, 16); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  assert.deepEqual(Array.from(new Uint32Array(readback.getMappedRange())),
    [0x9000_0000, 0x0800_0123, 0x8000_0000, 0x0800_0000],
    "ordinary status stores preserve the exact authority/failure publication ABI");
  readback.unmap();
  for (const buffer of [statuses, control, point, topologyParams, params, readback]) buffer.destroy();
  device.destroy();
});

test("every formerly over-limit Section 5 entry has at most ten reachable storage buffers", () => {
  const uniformBindings = new Set([0, 1, 20]);
  for (const entryPoint of ["emitTransientBandPowerGraph", "resolveCatalogTransitionAdjacency",
    "interpolatePowerFaceVector", "completePowerFaceAdvectionFromRegularBand"]) {
    const storageBindings = wgslReachableBindings(entryPoint)
      .filter(binding => !uniformBindings.has(binding));
    assert.ok(storageBindings.length <= 10,
      `${entryPoint} reaches ${storageBindings.length} storage buffers: ${storageBindings.join(", ")}`);
  }
});

test("face phi uses the paper's fine field then the redistanced dry-owner cube/Delaunay field", () => {
  const sample = wgslFunction("sampleBandFacePhi");
  const coarseSample = wgslFunction("sampleBandFaceCoarsePhi");
  assert.match(sample, /finePhiAtFaceCentroid\(face\.centroid\.xyz\)/,
    "valid fine phi has priority");
  assert.match(coarseSample,
    /\(face\.flags&\(LIVE\|PHI_VALID\)\)!=LIVE.*coarsePhiCptAtPoint\(anchor,face\.centroid\.xyz\).*face\.positiveRow<transitionControl\.support2End.*coarsePhiCptAtPoint\(anchor,face\.centroid\.xyz\)/,
    "only unresolved faces fall through, and only fully closed S0-S2 rows may anchor local interpolation");
  assert.match(coarseSample, /faces\[index\]\.flags\|=COARSE_PHI/);
  assert.match(coarseSample, /face\.flags\|=PHI_DIAGNOSTIC/,
    "missing fine and coarse authority remains publication-fatal in its fixed face record");
  assert.match(wgslFunction("finalizeFaceBandClosestPointDiagnostics"),
    /if\(\(face\.flags&PHI_DIAGNOSTIC\)!=0u\).*firstPhiSlot=min\(firstPhiSlot,slot\).*if\(!faceVelocityTarget\(face\)\)\{continue;\}/s,
    "scalar failures on topology-only terminal faces are attributed before velocity-target filtering");

  const exactCoarseCell = wgslFunction("exactCoarseCellScalar");
  assert.match(exactCoarseCell, /validCoarse\(\).*coarseSlot\(cell\(origin\),size\)/,
    "redistance seeds read only the current exact compact octree owner record");
  const coarseEntry = wgslFunction("coarseEntryRecord");
  assert.match(coarseEntry, /\(entry\.flags&9u\)!=9u.*entry\.minimumPhi>entry\.phi\|\|entry\.phi>entry\.maximumPhi/,
    "malformed or stale coarse signed distance is rejected");
  assert.doesNotMatch(exactCoarseCell, /ROW_COARSE_AIR|select\([^)]*,[^)]*,entry/,
    "exact seeds cannot synthesize a sign or capped distance");
  const coarseSeed = wgslFunction("coarseCellSeedRecordValid");
  assert.match(coarseSeed,
    /letexact=coarseEntryRecord\(coarseSlot\(cell\(origin\),size\)\).*letq=min\(origin\+vec3u\(size\/2u\),p\.dims-vec3u\(1u\)\).*coarseSlot\(cell\(priorOrigin\),scale\)/s,
    "a changed leaf samples the same-time spatial coarse publication at its centre");
  assert.match(coarseSeed, /if\(scale>=coarsePhi\.maximumLeafSize\)\{break;\}scale\*=2u/,
    "spatial migration is bounded by the published octree hierarchy");
  const coarseCell = wgslFunction("coarseCellScalar");
  assert.match(coarseCell, /rowOfIdentity\(cell\(origin\),size\).*row\.flags&ROW_PHI.*row\.representativePhi/s,
    "the complete transient owner field has priority at dry interpolation vertices");
  assert.match(coarseCell, /returnexactCoarseCellScalar\(origin,size\)/,
    "live compact rows remain exact seeds of the same field");

  const initialize = wgslFunction("initializeBandRowPhi");
  const extend = wgslFunction("extendBandRowPhi");
  assert.match(initialize, /finePhiAtFaceCentroid\(center\).*coarseCellSeedScalar/s,
    "the paper's current fine field has priority over the spatial coarse level-set seed");
  assert.match(extend, /localTetraEikonal\(rowIndex,sign\)/,
    "transition dry owners use their local Delaunay Eikonal update");
  assert.match(wgslFunction("localTetraEikonal"),
    /transitionAdjacency\[at\].*solveTranspose3\(a,b,c,known\).*candidate=\(bb\+sqrt/s,
    "the transition update solves |grad phi|=1 on the row-local catalog tetrahedra");
  assert.match(wgslFunction("nonobtuseIncidentSolidAngle"),
    /denominator=la\*lb\*lc\+dot\(a,b\)\*lc\+dot\(a,c\)\*lb\+dot\(b,c\)\*la.*denominator\+2e-5\*scale>=determinant/s,
    "the runtime consumer enforces the paper's nonobtuse incident-solid-angle contract");

  const interpolant = wgslFunction("coarsePhiAtPoint");
  assert.match(interpolant, /for\(varcorner=0u;corner<8u;corner\+=1u\)/,
    "uniform regions enumerate the regular cube vertices");
  assert.match(interpolant, /letweight=.*if\(weight==0\.\)\{continue;\}.*compactPublishedBandScalar/s,
    "uniform interpolation requires exactly the nonzero product-weight cube vertices");
  assert.match(interpolant, /tetraWeights\(point,tetraVertices\[selectors\.x\]\.v\.xyz,tetraVertices\[selectors\.y\]\.v\.xyz,tetraVertices\[selectors\.z\]\.v\.xyz\)/,
    "T-junction regions use the generated local Delaunay tetrahedron");
  assert.match(interpolant,
    /localTetraBandScalar\(anchor,local,0u,selectors\.x.*localTetraBandScalar\(anchor,local,1u,selectors\.y.*localTetraBandScalar\(anchor,local,2u,selectors\.z/s,
    "the selected tetrahedron consumes the exact adjacency rows validated by the closure transaction");
  const localScalar = wgslFunction("localTetraBandScalar");
  assert.match(localScalar,
    /transitionAdjacency\[at\].*adjacency\.band==anchor.*returnpublishedBandScalar\(neighbor\)/s,
    "in-domain Delaunay vertices must not repeat a fallible row-hash lookup after adjacency validation");
  assert.match(localScalar,
    /if\(all\(origin>=vec3i\(0\)\)&&all\(origin\+vec3i\(i32\(size\)\)<=vec3i\(p\.dims\)\)\)\{returnvec2f\(0\.\);\}/,
    "only a genuine boundary extension may bypass the published adjacency row");

  const transitionPhase = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const fineBindings = transitionPhase.match(/run\("sampleFacePhi",\[([\s\S]*?)\],0,pass,228/)?.[1];
  const coarseBindings = transitionPhase.match(/run\("sampleFaceCoarsePhi",\[([\s\S]*?)\],0,pass,228/)?.[1];
  assert.ok(fineBindings && coarseBindings);
  for (const binding of [0, 1, 2, 3, 5, 8, 12, 24]) {
    assert.match(fineBindings, new RegExp(`\\[${binding},`));
  }
  for (const binding of [0, 1, 5, 6, 7, 12, 27, 28, 29, 30, 31]) {
    assert.match(coarseBindings, new RegExp(`\\[${binding},`));
  }
  for (const binding of [25, 42]) assert.doesNotMatch(coarseBindings, new RegExp(`\\[${binding},`));
});

test("cube and Delaunay interpolants load only vertices that geometrically contribute", () => {
  for (const [functionName, selectorLoad] of [
    ["coarsePhiAtPoint", "localTetraBandScalar(anchor,local,0u,selectors.x"],
    ["liquidCentroidVector", "selectorVector(anchor,selectors.x"],
    ["marchedCentroidVector", "provisionalSelectorVector(anchor,selectors.x"],
    ["finalPointVector", "finalSelectorVector(anchor,selectors.x"],
  ] as const) {
    const source = wgslFunction(functionName);
    const weightsAt = source.indexOf("letweights=tetraWeights");
    const containedAt = source.indexOf("if(!contained(weights)){continue;}", weightsAt);
    const loadAt = source.indexOf(selectorLoad, containedAt);
    assert.ok(weightsAt >= 0 && containedAt > weightsAt && loadAt > containedAt,
      `${functionName} must select the paper's containing tetrahedron before loading its values`);
  }

  const diagnostic = wgslFunction("diagnoseCoarsePhiAtPoint");
  assert.match(diagnostic,
    /letweights=tetraWeights[^;]+;if\(!contained\(weights\)\)\{continue;\}varphiRecord=diagnoseLocalTetraBandScalar/,
    "failure telemetry must inspect only a selected tetrahedron dependency");

  for (const [functionName, cornerLoader] of [
    ["liquidCentroidVector", "compactSignedVector"],
    ["marchedCentroidVector", "provisionalSignedVector"],
    ["finalPointVector", "finalSignedVector"],
  ] as const) {
    assert.match(wgslFunction(functionName), new RegExp(
      `letweight=[^;]+;if\\(weight==0\\.\\)\\{continue;\\}letcornerOrigin=[^;]+;let(?:v|value)=${cornerLoader}`),
    `${functionName} must not require a zero-weight trilinear corner`);
  }
});

test("closest-point velocity interpolation reads only immutable seeded wet faces", () => {
  const seeded = wgslFunction("seededIncidentVector");
  assert.match(seeded, /incidence\[rowIndex\].*p\.axisStride/s,
    "the wet carrier gather is bounded by the row's fixed incidence capacity");
  assert.match(seeded, /incidence\[p\.rowCapacity\+rowIndex\*p\.axisStride\+local\].*faces\[faceIndex\]/s,
    "the gather follows the already-published regular-face incidence graph");
  assert.match(seeded, /(?:face|candidate)\.flags.*LIVE.*SEED.*velocityValid\((?:face|candidate)\.velocity\)/s,
    "only immutable wet seeds from the preceding dispatch may contribute");
  assert.doesNotMatch(seeded, /FACE_VELOCITY_VALID/,
    "a dry face written by another invocation in the same dispatch is never an authority");

  const wet = wgslFunction("wetFaceVectorAtPoint");
  assert.match(wet, /seededIncidentVector/,
    "cube and Delaunay vertices source vectors from seeded regular faces");
  assert.match(wet,
    /if\(validWeight<=1e-6\)\{returnLiquidInterpolation\(vec4f\(0\),CPT_MISSING_VERTEX\);\}result\/=validWeight/s,
    "the bounded one-sided gather fails only on an empty wet stencil and renormalizes retained weights");
  assert.doesNotMatch(wet,
    /\bcellVector\b|\bcompactSignedVector\b|\bselectorVector\b|powerRowVelocities|nearest|fallback|Bfs|heap/i,
    "dry closest-point evaluation cannot reach the pre-extension power-row field or a legacy repair path");

  const closest = wgslFunction("closestPointLiquidVector");
  assert.match(closest, /resolveWetFaceVectorAtPoint/,
    "the exact closest point is evaluated from immutable wet regular-face carriers");
  const resolvedWet = wgslFunction("resolveWetFaceVectorAtPoint");
  assert.match(resolvedWet,
    /direct=wetFaceVectorAtPoint\(initialAnchor,pointGrid\).*localFanGap=direct\.reason==CPT_NO_SIMPLEX\|\|direct\.reason==CPT_MISSING_VERTEX.*for\(vardz=-2i;dz<=2i.*for\(vardy=-2i;dy<=2i.*for\(vardx=-2i;dx<=2i.*candidate>=bestRow.*sampled=wetFaceVectorAtPoint\(candidate,pointGrid\).*bestRow=candidate;best=sampled/s,
    "a local fan or seeded-vertex gap retries only the complete fixed catalog-radius box and selects the lowest valid containing fan");
  const closestCarrier = wgslFunction("closestSeededFaceCarrier");
  assert.match(closestCarrier,
    /count=min\(incidence\[rowIndex\],p\.axisStride\).*face\.flags&\(LIVE\|SEED\|PHI_VALID\|FACE_VELOCITY_VALID\).*velocityValid\(face\.velocity\).*distanceSquared=dot\(delta,delta\).*faceIndex<best\.face/s,
    "an empty wet stencil can see only the closest immutable seed in each fixed incidence star");
  assert.match(resolvedWet,
    /carrierEligible=localFanGap.*carrierEligible&&bestRow==INVALID.*if\(bestRow!=INVALID\)\{returnbest;\}if\(bestCarrier\.face!=INVALID\)\{returnLiquidInterpolation\(bestCarrier\.value,0u\);\}returndirect/s,
    "closest-face constant extension is restricted to an exhausted local-fan gap and remains subordinate to every exact interpolant");
  assert.doesNotMatch(resolvedWet, /rowCapacity|for\([^)]*control\.rowCount|supportCellVector|finalCellVector/,
    "liquid-side recovery neither scans the row arena nor substitutes an air-side cell carrier");
  assert.doesNotMatch(closestCarrier, /rowCount|rowDirectory|powerRowVelocities|while\(|loop\{/,
    "the closest-face carrier adds neither a row scan nor a mutable dry-face dependency");
  assert.match(closest,
    /!velocityValid\(sampled\.value\).*sampled\.value\.w=-f32\(band\+1u\)/s,
    "a rejected gather preserves its exact containing row for failure-only readback");
  assert.doesNotMatch(closest,
    /liquidStencilInteriorDepth|liquidCentroidVector|row\.globalRow==INVALID|row\.globalRow>=p\.powerRowCapacity/,
    "the deleted inward-depth and dry power-row path cannot regain authority");
  assert.doesNotMatch(wet + resolvedWet + closestCarrier + closest, /Bfs|heap|fastMarch|rowCount/i,
    "one-sided interpolation and constant extension remain fixed local gathers without iterative marching");
});

test("regular-to-power publication diagnostics require an actually committed transaction", () => {
  const words = new Uint32Array(16);
  words.set([0, 0xffff_ffff, 90, 21, 21, 21, 7, 13, 0x8000_0000]);
  assert.deepEqual(unpackOctreeFaceBandPowerPublication(words), {
    flags: 0, firstError: 0xffff_ffff, faceCount: 90, targetCount: 21,
    interpolatedCount: 21, committedCount: 21, fineGeneration: 7,
    powerGeneration: 13, valid: true,
  });
  words[5] = 20;
  assert.equal(unpackOctreeFaceBandPowerPublication(words).valid, false);
  assert.throws(() => unpackOctreeFaceBandPowerPublication(new Uint32Array(8)), /at least nine/);
});

test("final Section 5 graph and point-field controls retain attributable transactions", () => {
  const point = new Uint32Array([0, 0xffff_ffff, 120, 7, 120, 0x8000_0000, 18, 35]);
  assert.deepEqual(unpackOctreeFaceBandPointFieldControl(point), {
    flags: 0, firstError: 0xffff_ffff, rowCount: 120, generation: 7,
    solvedCount: 120, valid: true, wallContributions: 18, coreRowCount: 35,
  });
  point[4] = 119;
  assert.equal(unpackOctreeFaceBandPointFieldControl(point).valid, false);
  assert.throws(() => unpackOctreeFaceBandPointFieldControl(new Uint32Array(7)), /eight/);

  const transient = new Uint32Array(16);
  transient.set([0, 0xffff_ffff, 120, 3_600, 901, 901, 120, 7, 0x8000_0000]);
  assert.deepEqual(unpackOctreeFaceBandTransientPowerControl(transient), {
    flags: 0, firstError: 0xffff_ffff, rowCount: 120, faceSlots: 3_600,
    emittedCount: 901, sampledCount: 901, validatedCount: 120,
    generation: 7, valid: true,
  });
  transient[5] = 900;
  assert.equal(unpackOctreeFaceBandTransientPowerControl(transient).valid, false);
  transient.set([8, 0x8000_0000 | 37955]);
  assert.deepEqual(unpackOctreeFaceBandTransientPowerControl(transient), {
    flags: 8, firstError: 37955, failureDomain: "face", rowCount: 120, faceSlots: 3_600,
    emittedCount: 901, sampledCount: 900, validatedCount: 120,
    generation: 7, valid: false,
  });
  assert.throws(() => unpackOctreeFaceBandTransientPowerControl(new Uint32Array(8)), /at least nine/);
});

test("transition diagnostics preserve the exact pre-emission catalog failure", () => {
  const words = new Uint32Array(16);
  words.set([
    OCTREE_FACE_BAND_TRANSITION_ERROR.invalidBandDescriptor,
    17, 3582, 211, 917, 0, 0,
    OCTREE_FACE_BAND_TRANSITION_DETAIL.aboveDomain | OCTREE_FACE_BAND_TRANSITION_DETAIL.missingBandRow,
  ]);
  words.set([3582, 3582, 3582, 3582, 3582], 8);
  assert.deepEqual(unpackOctreeFaceBandTransitionControl(words), {
    flags: 8, firstError: 17, rowCount: 3582, transitionRows: 211, adjacencyCount: 917,
    ready: false, transferReady: false, hierarchyReady: false,
    invalidSource: false, capacityFailure: false,
    unresolvedAdjacency: false, invalidBandDescriptor: true,
    detailFlags: 36, malformedGeometry: false, belowDomain: false, aboveDomain: true,
    misalignedGeometry: false, ownerMismatch: false, missingBandRow: true,
    rowOutOfRange: false, ownerSizeMismatch: false,
    coreRowCount: 3582, support1RowCount: 3582, support2RowCount: 3582,
    support3NodeRowCount: 3582, endpointRowCount: 3582, boundaryGhostRequests: 0,
    phiFailureCounts: { missingRow: 0, exactCoarseMiss: 0, invalidMetric: 0, invalidSelector: 0 },
  });
  words.set([0, 0xffff_ffff, 3582, 211, 917, 0x8000_0000, 0x8000_0000, 0]);
  words[14] = 0x8000_0000;
  assert.equal(unpackOctreeFaceBandTransitionControl(words).ready, true);
  assert.equal(unpackOctreeFaceBandTransitionControl(words).transferReady, true);
  assert.equal(unpackOctreeFaceBandTransitionControl(words).hierarchyReady, true);
  assert.throws(() => unpackOctreeFaceBandTransitionControl(new Uint32Array(7)), /eight/);
});

test("transition diagnostics decode one atomically claimed exact-owner mismatch", () => {
  const words = new Uint32Array(32);
  words[0] = OCTREE_FACE_BAND_TRANSITION_ERROR.unresolvedAdjacency;
  words[1] = 1747;
  words[7] = OCTREE_FACE_BAND_TRANSITION_DETAIL.ownerMismatch;
  words.set([
    1747, OCTREE_FACE_BAND_OWNER_FAILURE_STAGE.support1, 1234, 2,
    0x0003_ffff, 91, 0x8000_0005, 0x8000_000c,
    (-2) >>> 0, 6, 8, 2, 4567, 1, 8910, 1 | (1 << 16),
  ], 16);
  assert.deepEqual(unpackOctreeFaceBandTransitionControl(words).ownerFailure, {
    band: 1747, stage: OCTREE_FACE_BAND_OWNER_FAILURE_STAGE.support1,
    rowCell: 1234, rowSize: 2, descriptor: 0x0003_ffff,
    topology: 91, transformFlags: 0x8000_0005, selector: 0x8000_000c,
    rawOrigin: [-2, 6, 8], requestedSize: 2, resolvedOriginCell: 4567,
    boundaryFlips: 1, actualOwnerCell: 8910, actualOwnerSize: 1, actualOwnerValid: true,
  });
});

test("transition diagnostics decode one deterministic face-phi interpolation failure", () => {
  const words = new Uint32Array(32);
  words[15] = 2 | (3 << 8) | (4 << 16) | (5 << 24);
  const centroid = new Float32Array([12.5, 7, 3.25]);
  const centroidBits = new Uint32Array(centroid.buffer);
  words.set([
    0x8000_0000 | 19, 19, 4301, 71, 84, 71,
    centroidBits[0], centroidBits[1], centroidBits[2], 2,
    (-4) >>> 0, 8, 10, 2, 17, 3 | (0x205 << 8),
  ], 16);
  const decoded = unpackOctreeFaceBandTransitionControl(words);
  assert.deepEqual(decoded.phiFailureCounts,
    { missingRow: 2, exactCoarseMiss: 3, invalidMetric: 4, invalidSelector: 5 });
  assert.deepEqual(decoded.phiFailure, {
    cause: 3, faceIndex: 19, globalFace: 4301, negativeRow: 71, positiveRow: 84,
    anchorRow: 71, centroid: [12.5, 7, 3.25], interpolantPath: 2,
    missingOrigin: [-4, 8, 10], missingSize: 2, selectorOrCorner: 17, detail: 0x205,
  });
});

test("paper Section 5 orders LIVE regular faces by the current two-resolution phi at their actual centroids", () => {
  const emit = wgslFunction("emitBandFaces");
  assert.doesNotMatch(emit, /representativePhi|minimumPhi|maximumPhi|facePhi/,
    "face emission must never select an endpoint or coarse row distance");
  assert.match(octreeFaceBandWGSL, /const PHI_VALID:u32=4u/,
    "fine-distance authority must use an explicit flag separate from the scalar channel");
  assert.match(emit, /Face\([^;]+,0\.,[^;]+,LIVE\|select\(0u,VELOCITY_TARGET,velocityTarget\),0u\)/,
    "an emitted face starts with target topology but no sampled distance or velocity authority");
  assert.doesNotMatch(octreeFaceBandWGSL, /0x7fc00000u/,
    "face-band WGSL must not materialize a NaN constant rejected by Dawn/Tint");
  assert.match(emit, /band>=transitionControl\.endpointEnd/,
    "regular-face ownership is bounded by the complete S0/S1/S2 plus terminal S3 endpoint prefix");
  assert.match(emit,
    /band>=transitionControl\.support3NodeEnd[\s\S]*neighbor>=transitionControl\.support1End[\s\S]*neighbor<transitionControl\.support3NodeEnd/,
    "terminal S3 owns only positive faces whose opposite endpoint is an S2 node");
  assert.match(emit,
    /band>=transitionControl\.support3NodeEnd\)\{continue;\}[\s\S]*adjacencyFail/,
    "a missing endpoint is fatal through S2 and ignored only beyond terminal S3");

  const sampler = wgslFunction("finePhiAtFaceCentroid");
  assert.match(sampler,
    /letworld=fp\.domainOrigin\+pointGrid\*coarseWidth;letraw=\(world-fp\.domainOrigin\)\/fp\.fineWidth-vec3f\(\.5\)/,
    "the fine lattice query is formed from the actual regular-face centroid in world space");
  assert.match(sampler, /sample=loadFineScalarExtended.*result\+=wx\*wy\*wz\*sample\.x.*gradient\+=sample\.x/s,
    "the fine scalar and its analytic trilinear gradient require one authoritative eight-corner stencil");
  assert.match(sampler, /fp\.generation!=p\.generation/,
    "face phi never admits a stale or rollback-adjacent generation");
  const load = wgslFunction("loadFineScalarExtended");
  assert.match(load, /q\[axis\]!=-1.*q\[axis\]=0/s,
    "the lower world plane permits exactly one virtual fine-center layer and mirrors it evenly");
  assert.match(load, /q\[axis\]!=limit.*q\[axis\]=limit-1/s,
    "the upper world plane permits exactly one virtual fine-center layer and mirrors it evenly");
  const lookup = wgslFunction("finePage");
  assert.match(lookup,
    /directoryBase=5u\+fp\.worklistCapacity.*id=worklist\[directoryBase\+key\]/s,
    "fine pages use the immutable generation-local direct directory");
  assert.match(lookup,
    /metadata\[base\]==id&&metadata\[base\+1u\]==key&&metadata\[base\+2u\]==fp\.generation/,
    "directory hit, page identity, key, and generation must all agree");
  assert.doesNotMatch(lookup, /while|low|high|probe|Hash/,
    "the direct fine-page lookup has no binary-search or hash fallback");
  assert.match(load, /\(sampleFlags\[index\]&1u\)==0u/);

  const sampleKernel = wgslFunction("sampleBandFacePhi");
  const coarseSampleKernel = wgslFunction("sampleBandFaceCoarsePhi");
  assert.match(sampleKernel, /finePhiAtFaceCentroid\(face\.centroid\.xyz\)/,
    "fine phi is always attempted first");
  assert.match(coarseSampleKernel,
    /anchor=face\.negativeRow.*coarsePhiCptAtPoint\(anchor,face\.centroid\.xyz\).*face\.positiveRow<transitionControl\.support2End.*anchor=face\.positiveRow.*coarsePhiCptAtPoint\(anchor,face\.centroid\.xyz\)/,
    "fine phi is preferred and only the fully closed S0-S2 octree field supplies the outside-band interpolant");
  assert.match(coarseSampleKernel,
    /phiRecord\.cause==INVALID&&face\.positiveRow<transitionControl\.support2End/,
    "an endpoint-only row can neither evaluate nor diagnose a Delaunay phi anchor without a catalog one-ring");
  assert.match(coarseSampleKernel, /sampled\.valid==0u[\s\S]*face\.flags\|=PHI_DIAGNOSTIC/,
    "missing, stale, invalid, or non-finite support publishes a fixed rejection record");
  assert.match(coarseSampleKernel,
    /diagnoseCoarsePhiAtPoint\(face\.negativeRow,face\.centroid\.xyz\).*PHI_DIAGNOSTIC/s,
    "a failed face retains bounded first-failure evidence without weakening the publication gate");
  const phiReduction = wgslFunction("reduceBandPhiFailure");
  assert.match(phiReduction,
    /topologyPublishLaneRange\(liveCount,lane\).*counts\+=1u<<\(cause\*8u\).*first=min\(first,index\).*topologyPublishSums\[lane\]/s,
    "failed faces are parallel-counted by cause and select one deterministic minimum face index");
  assert.match(phiReduction,
    /phiFailureCounts=reduced\.x;letfirst=reduced\.y;if\(first==INVALID\)\{return;\}transitionControl\.failureBand=PHI_FAILURE_TAG\|first/,
    "the face reduction preserves an earlier exact band-row failure when no face diagnostic exists");
  assert.match(phiReduction,
    /failureStage=first.*failureRowCell=face\.globalFace.*failureOwnerSizeValid=cause\|\(detail<<8u\)/s,
    "the selected face publishes its exact interpolation record in the same terminal reduction");
  assert.match(wgslFunction("publishClosestPoint"), /result\.phi=sampled\.phi/,
    "the physical signed distance is retained unchanged for closest-face ordering");
  assert.match(wgslFunction("publishClosestPoint"), /result\.flags\|=PHI_VALID/,
    "distance authority is published only after an exact current centroid sample succeeds");
  const summary = wgslFunction("summarizeBandRowPhi");
  assert.match(summary, /row>=transitionControl\.support1End/,
    "only S0/S1 rows publish incident-face phi summaries; S2 retains its committed coarse phi");
  assert.doesNotMatch(summary, /row>=transitionControl\.support2End/,
    "face-band summarization must not overwrite the committed coarse phi carried by S2");
  assert.match(summary,
    /minimum=min\(minimum,face\.phi\);maximum=max\(maximum,face\.phi\)/,
    "row min/max are reductions of real sampled incident face positions");
  assert.match(summary, /rows\[row\]\.flags\|=ROW_PHI/,
    "row distance authority is published only after every incident face validates");
  assert.match(summary, /face\.flags&\(LIVE\|PHI_VALID\)/,
    "row summaries reject every face without explicit current-distance authority");
  const commitRowPhi = wgslFunction("commitBandRowPhi");
  assert.match(commitRowPhi,
    /row\.flags=\(row\.flags&~ROW_CENTER_WET\)\|ROW_PHI\|select\(0u,ROW_CENTER_WET,state\.x<0\.\).*rows\[rowIndex\]=row/s,
    "the exact current cell-centre sample publishes one immutable wet bit before face summaries");
  const wetRow = wgslFunction("rowCarriesLiquid");
  assert.match(wetRow,
    /rows\[rowIndex\]\.flags&\(ROW_PHI\|ROW_CENTER_WET\)\)==\(ROW_PHI\|ROW_CENTER_WET\)/,
    "uniform wet cells remain in the known velocity domain through their current committed centre sign");
  assert.match(wetRow,
    /flags&\(ROW_COARSE_LIQUID\|ROW_COARSE_MIXED\)\)!=0u/,
    "a current coarse mixed cell retains corner-cut liquid even when its centre and face centroids are positive");
  assert.doesNotMatch(wetRow, /representativePhi|minimumPhi|maximumPhi/,
    "later incident-face summaries cannot replace the immutable centre or coarse-volume classifications");
  const wetFace = wgslFunction("faceCarriesLiquid");
  assert.match(wetFace,
    /face\.phi<=0\.\|\|rowCarriesLiquid\(face\.negativeRow\)\|\|rowCarriesLiquid\(face\.positiveRow\)/,
    "a wet centroid or either wet endpoint publishes an immutable regular-face seed");
  assert.match(wgslFunction("seedFaceCentroids"),
    /f\.flags&\(LIVE\|PHI_VALID\).*faceCarriesLiquid\(f\).*exactRegularFaceCoreVector\(f\).*velocityValid\(exact\).*SEED\|FACE_VELOCITY_VALID/s,
    "the exact core-to-core regular edge remains inside the complete current wet-face seed gate");
  assert.doesNotMatch(wgslFunction("seedFaceCentroids"),
    /f\.flags\|=SEED;(?!\|FACE_VELOCITY_VALID)/,
    "seed authority is never premarked before its immutable velocity is available");
  assert.match(wgslFunction("seedFaceCentroids"),
    /if\(!velocityValid\(velocity\)\)\{f\.pad=reason;faces\[face\]=f;return;\}f\.velocity=velocity;f\.pad=0u;f\.flags\|=SEED\|FACE_VELOCITY_VALID/s,
    "an eligible support face without a direct vector remains an ordinary closest-point target");
  assert.match(sampleKernel,
    /publishClosestPoint\(face,sampled,fp\.fineWidth\*f32\(fp\.fineFactor\)\)/,
    "the fine signed-distance sample directly publishes the inward closest-point query");
  assert.match(coarseSampleKernel,
    /anchor=face\.negativeRow.*coarsePhiCptAtPoint\(anchor,face\.centroid\.xyz\).*publishClosestPoint/s,
    "the redistanced cube/Delaunay field supplies the same closest-point contract");
  const publishClosestPoint = wgslFunction("publishClosestPoint");
  assert.doesNotMatch(publishClosestPoint, /sampled\.phi<=0/,
    "inside orphan face samples still need the same closest-interface query as air support faces");
  assert.match(publishClosestPoint,
    /result\.flags\|=PHI_VALID;letmagnitude=length\(sampled\.gradient\)/,
    "signed-distance authority is independent of whether its gradient can publish a closest point");
  assert.match(publishClosestPoint,
    /rawClosest=face\.centroid\.xyz-\(sampled\.phi\/width\)\*normal.*closest=clamp\(rawClosest,vec3f\(0\.\),vec3f\(p\.dims\)\)/s,
    "the signed closest point moves either side of the interface to the same explicit physical boundary policy");
  assert.match(publishClosestPoint, /result\.velocity=vec4f\(closest,1\.\).*CLOSEST_POINT_VALID/s,
    "the boundary-composed closest point, not an ownership epsilon, is carried in the face");
  assert.match(wgslFunction("halfOpenDomainPoint"),
    /clamp\(point,vec3f\(0\.\),max\(vec3f\(0\.\),vec3f\(p\.dims\)-vec3f\(1e-4\)\)\)/,
    "owner queries use a deterministic half-open representative on physical planes");
  const coarseGradient = wgslFunction("coarsePhiCptAtPoint");
  assert.doesNotMatch(coarseGradient, /center\.x<=0/,
    "a negative face sample without a liquid pressure endpoint still requires a physical closest point");
  assert.match(coarseGradient,
    /attempt=0u;attempt<4u.*\.5\*rowScale.*\.0625\*rowScale\*f32\(1u<<attempt\)/s,
    "coarse closest-point gradients expand over a bounded owner-scale stencil when a local plateau is flat");
  assert.match(coarseGradient,
    /signedDistanceAxisDerivative\(center,plus,minus,h\)/,
    "coarse closest-point gradients use the canonical nonsmooth signed-distance derivative");
  const axisDerivative = wgslFunction("signedDistanceAxisDerivative");
  assert.match(axisDerivative,
    /letcentral=\(plus\.x-minus\.x\)\/\(2\.\*h\).*if\(abs\(central\)>1e-8\)/s,
    "smooth signed-distance samples retain their centered derivative");
  assert.match(axisDerivative,
    /letforward=\(plus\.x-center\.x\)\/h;letbackward=\(center\.x-minus\.x\)\/h;letoneSided=select\(backward,forward,abs\(forward\)>abs\(backward\)\)/,
    "a centered cancellation at a boundary or medial cusp selects one deterministic one-sided subgradient");
  assert.match(coarseGradient, /magnitude>1e-8.*returnPhiCPT\(center\.x,gradient,1u\)/s,
    "only a finite nonzero multiscale derivative may publish a closest point");
  const boundaryDiagnostic = wgslFunction("diagnoseCompactPublishedBandScalar");
  assert.match(boundaryDiagnostic,
    /letboundary=.*varcandidateSize=size;varrecord=.*loop\{letcandidateOrigin=.*record=diagnosePublishedBandRow.*if\(record\.cause==INVALID\|\|!boundary\|\|candidateSize>=p\.maximumLeaf\)\{break;\}candidateSize\*=2u;\}returnrecord/s,
    "boundary diagnostics follow the same exact reflected owner hierarchy as the runtime scalar lookup");
  const extendClosestPoints = wgslFunction("extendFaceClosestPoints");
  assert.match(extendClosestPoints,
    /face\.flags&CLOSEST_POINT_VALID.*closestPointLiquidVector\(face\).*face\.velocity=sampled\.value.*FACE_VELOCITY_VALID/s,
    "each air face independently samples the liquid-only full-vector interpolant at its closest point");
  const rejectClosestPoints = wgslFunction("rejectClosestPointFace");
  assert.match(rejectClosestPoints,
    /unresolvedCount.*closestPointFailures.*liquidInterpolationFailures/s,
    "an unresolved exact sample publishes attributable closest-point/interpolation failures");

  const schedule = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(schedule,
    /run\("sampleFacePhi",\[\[0,this\.params\],\[1,input\.fine\.params\],\[2,input\.fine\.metadata\],\[3,input\.fine\.worklist\],\[5,this\.candidateControl\],\[8,input\.fine\.flags\],\[12,this\.candidateFaces\],\[24,input\.fine\.phi\]\]/,
    "the fine sampler uses the canonical sorted worklist and metadata directory without a page hash");
  assert.match(schedule,
    /run\("sampleFaceCoarsePhi",\[\[0,this\.params\],\[1,input\.fine\.params\],\[5,this\.candidateControl\],\[6,this\.candidateRows\],\[7,this\.candidateRowDirectory\],\[12,this\.candidateFaces\],\[27,this\.candidateTransitionMetrics\],\[28,tetrahedronHeaders\],\[29,tetrahedra\],\[30,tetrahedronVertices\],\[31,this\.candidateTransitionAdjacency\],\[32,this\.candidateTransitionControl\]\]/,
    "the ordered coarse sampler consumes candidate row phi, validated adjacency, and the S0-S2 prefix gate");
  assert.match(schedule,
    /run\("commitBandPhi",\[\[5,this\.candidateControl\],\[6,this\.candidateRows\],\[19,this\.velocities\],\[32,this\.candidateTransitionControl\]\]/,
    "the row-field commit binds its exact scalar-publication prefix and output globals");
  assert.match(wgslFunction("commitBandRowPhi"), /rowIndex>=transitionControl\.endpointEnd/,
    "terminal regular-face endpoints publish phi only after their exact closure edges redistance");
  assert.match(wgslFunction("extendBandRowPhi"),
    /lowerSimplexCandidate\(rowIndex,sign\).*localTetraEikonal\(rowIndex,sign\)/s,
    "terminal support edges remain causal lower-dimensional simplices while transition rows use local tetrahedra");
  assert.match(wgslFunction("lowerSimplexCandidate"),
    /base=rowIndex\*MAX_CATALOG_GUARDS.*item=transientPowerIncidences\[base\+local\].*best=min\(best,abs\(source\.x\)\+length\(center-bandCenter\(parent\)\)\*unit\)/s,
    "support-only nodes traverse every exact closure-parent edge");
  const incoming = wgslFunction("buildBandPhiIncomingEdges");
  assert.match(incoming,
    /metric\.reserved==0u[\s\S]*for\(varz=-1;z<=1;z\+=1\)[\s\S]*rowOfIdentity\(cell\(owner\.origin\),owner\.size\)[\s\S]*appendBandPhiNeighbor/,
    "uniform cube closure publishes every exact incoming scalar-support edge");
  assert.match(incoming,
    /transitionAdjacency\[at\][\s\S]*appendBandPhiNeighbor\(parent,adjacency\.a[\s\S]*appendBandPhiNeighbor\(parent,adjacency\.b[\s\S]*appendBandPhiNeighbor\(parent,adjacency\.c/,
    "transition closure publishes reverse incidence from the validated local Delaunay mesh");
  assert.match(wgslFunction("appendBandPhiNeighbor"),
    /edge=parent\*MAX_CATALOG_GUARDS\+\*count[\s\S]*PowerIncidence\(neighbor,0\)/,
    "each source row owns a bounded fixed neighbor record");
  const endpointIncoming = wgslFunction("validateBandPhiEndpointIncomingEdges");
  assert.match(endpointIncoming,
    /atomicLoad\(&endpointIncomingCounts\[endpoint\]\)[\s\S]*prior\.face<=value\.face[\s\S]*transientPowerIncidences\[base\+cursor\]=value/,
    "each endpoint validates and canonically orders only its directly appended sparse reverse edges");
  assert.match(wgslFunction("appendBandPhiNeighbor"),
    /neighbor>=transitionControl\.support3NodeEnd[\s\S]*atomicAdd\(&endpointIncomingCounts\[neighbor\],1u\)[\s\S]*PowerIncidence\(parent,0\)/,
    "source construction scatters reciprocal endpoint edges in O(actual edges)");
  assert.doesNotMatch(endpointIncoming, /source<transitionControl\.support3NodeEnd/,
    "terminal endpoint publication must never scan the source-row cross product");
  assert.match(schedule,
    /run\("clearBandPhiEndpointEdges"[\s\S]*run\("buildBandPhiEdges"[\s\S]*run\("buildBandPhiEndpointEdges"/,
    "endpoint counters clear before sparse source scatter and canonical endpoint validation");
  const emitAt = schedule.indexOf('run("emit"');
  const clearEndpointAt = schedule.indexOf('run("clearBandPhiEndpointEdges"');
  const incomingAt = schedule.indexOf('run("buildBandPhiEdges"');
  const endpointIncomingAt = schedule.indexOf('run("buildBandPhiEndpointEdges"', incomingAt);
  const sampleAt = schedule.indexOf('run("sampleFacePhi"', emitAt);
  const initializePhiAt = schedule.indexOf('run("initializeBandPhi"', sampleAt);
  const seedPhiAt = schedule.indexOf('run("seedBandPhiFaces"', initializePhiAt);
  const extendPhiAt = schedule.indexOf('run("extendBandPhi"', seedPhiAt);
  const commitPhiAt = schedule.indexOf('run("commitBandPhi"', extendPhiAt);
  const coarseSampleAt = schedule.indexOf('run("sampleFaceCoarsePhi"', commitPhiAt);
  const summaryAt = schedule.indexOf('run("summarizeRowPhi"', coarseSampleAt);
  const gateAt = schedule.indexOf('run("gateTransition"', summaryAt);
  assert.ok(clearEndpointAt >= 0 && incomingAt > clearEndpointAt && endpointIncomingAt > incomingAt
    && emitAt > endpointIncomingAt && sampleAt > emitAt && initializePhiAt > sampleAt
    && seedPhiAt > initializePhiAt && extendPhiAt > seedPhiAt && commitPhiAt > extendPhiAt
    && coarseSampleAt > commitPhiAt
    && summaryAt > coarseSampleAt && gateAt > summaryAt,
    "face topology, current phi, row summaries, then transaction gate is the only publication order");
  assert.doesNotMatch(wgslFunction("commitBandRowPhi"), /atomicLoad\(&control\.flags\)/,
    "one unresolved row must not race valid row phi publication in the same dispatch");
});

test("top-side nonuniform transition uses explicit physical world faces", () => {
  assert.deepEqual(classifyOctreeFaceBandBoundaryCrossing([-2, 8, 2], 2, [8, 8, 8], 0b10_1111), {
    valid: true, closedComponents: 1, openPlanes: 16,
  });
  assert.deepEqual(classifyOctreeFaceBandBoundaryCrossing([-3, 8, 2], 2, [8, 8, 8], 0b10_1111), {
    valid: false, closedComponents: 0, openPlanes: 0,
  });
  const emit = wgslFunction("emitTransientBandPowerGraph");
  assert.match(emit,
    /letboundaryBit=transientWorldBoundaryBit\(geometry\);letworld=geometry\.neighborSize==0\.\|\|\(boundaryBit&declared\)!=0u/,
    "a declared out-of-domain ghost site and the catalog's zero-size sentinel both denote a world face");
  assert.match(emit,
    /if\(world\)\{plane=boundaryBit&declared;if\(plane==0u\)/,
    "the resolved boundary-aware metric is the authority for a world face's physical plane");
  assert.doesNotMatch(emit, /exact\.centroid\[axis\]|letoutward=/,
    "a clipped slanted ghost-site face is not rejected by re-deriving an axis plane from its centroid or normal");
  const sample = wgslFunction("sampleTransientBandPowerFaces");
  assert.doesNotMatch(octreeFaceBandWGSL, /homologousRegularFaceScalar/,
    "the terminal endpoint closure deletes the homologous-face sampling shortcut");
  assert.match(sample, /else\{letnegative=marchedCentroidVector/,
    "every interior generalized face uses the paper cube/Delaunay vector interpolant directly");
  assert.match(sample, /if\(\(p\.closedBoundaryMask&plane\)==0u\)/,
    "closed and open planes are applied independently per physical world face");
  assert.match(sample, /plane!=positiveBoundaryBit\(1u\)/,
    "padf/open extension is authored only for the dam-break +y world plane");
  assert.doesNotMatch(octreeFaceBandWGSL, /accumulateBandPowerLSBatch|reflectedBoundaryPoint/,
    "the final field has no catalog-local or reflected scalar fallback");
});

test("air-side sampler has no standalone runtime pipeline after fusion", () => {
  const implementation = compact(WebGPUOctreeFaceClosestPointExtension);
  assert.doesNotMatch(implementation,
    /sampleClassifyPipeline|sampleEvaluatePipeline|sampleFinalizePipeline|atomic-freeoctreeface-bandairsampler/);
});

test("band phi retains fail-closed graph extension for closure-only rows", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(source, /run\("initializeBandPhi"[\s\S]*run\("seedBandPhiFaces"[\s\S]*run\("extendBandPhi"[\s\S]*run\("commitBandPhi"/);
});

test("2:1 face emission publishes bounded incidence directly", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(source,
    /run\("emit",\[\[0,this\.params\],\[5,this\.candidateControl\],\[6,this\.candidateRows\],\[7,this\.candidateRowDirectory\],\[12,this\.candidateFaces\],\[26,input\.owners\],\[32,this\.candidateTransitionControl\],\[65,this\.transitionDeltaScan\]\],0,pass,204\)/);
  assert.match(source,
    /run\("rebuildIncidence",\[\[0,this\.params\],\[5,this\.candidateControl\],\[6,this\.candidateRows\],\[7,this\.candidateRowDirectory\],\[12,this\.candidateFaces\],\[14,this\.candidateIncidence\],\[26,input\.owners\],\[32,this\.candidateTransitionControl\],\[65,this\.transitionDeltaScan\]\],0,pass,204\)/);
  assert.doesNotMatch(source, /run\("incidence"/);
  assert.doesNotMatch(wgslFunction("emitBandFaces"), /atomicAdd|appendIncidence/,
    "affected face slots are fixed-address and never atomically appended");
  const carryTopology = wgslFunction("carryFaceBandTopology");
  assert.doesNotMatch(carryTopology,
    /band>=transitionControl\.support1End[\s\S]*faces\[base\+local\]\.flags=0u[\s\S]*return;/,
    "S2 slots cannot be retired wholesale because a restricted S2-to-S1 positive face may remain live");
  assert.match(carryTopology,
    /face\.flags&=~\(SEED\|PHI_VALID\|PHI_DIAGNOSTIC\|CLOSEST_POINT_VALID\|FACE_VELOCITY_VALID\|COARSE_PHI\|PRIMARY_EXTENSION\);face\.phi=0\.;face\.velocity=vec4f\(0\.\);face\.pad=0u;/,
    "carried topology clears every previous-generation phi, CPT, seed, and velocity publication");
  assert.match(wgslFunction("publishCatalogTransitionAdjacency"),
    /affectedRows\+=select\(0u,1u,band<arrayLength\(&transitionDeltaScan\)&&transitionDeltaScan\[band\]!=0u\).*topologyPublishSums\[lane\]=vec4u\(validatedRows,adjacencyCount,boundaryGhostRequests,affectedRows\).*transitionControl\.pad35=totals\.w/s,
    "the parallel adjacency reduction retains the complete dense S0/S1/S2 mask and reports its population");
  assert.doesNotMatch(source, /run\("preserveAffectedRows"/,
    "preserving an already compact list does not relaunch a parallel kernel");
  assert.match(wgslFunction("carryFaceBandIncidence"), /incidenceRowAffected\(band\)/,
    "incidence carry consults the preserved immutable S0/S1/S2 delta after validation clears row flags");
  assert.match(wgslFunction("incidenceRowAffected"), /returnfaceRowAffected\(band\)/,
    "the immutable dense mask, not incidence scratch or cleared row flags, owns incidence rebuild selection");
  const rebuildIncidence = wgslFunction("rebuildFaceBandIncidence");
  assert.match(rebuildIncidence,
    /item>=faceDeltaCount\(\).*band=faceDeltaRow\(item\)/s,
    "incidence rebuild consumes the preserved row list after ordinary incidence overwrites scan scratch");
  assert.doesNotMatch(rebuildIncidence, /transitionDeltaWorkCount|transitionDeltaRow|incidence\[0\]/,
    "post-emission rebuild cannot alias its work domain with the incidence arena it writes");
  assert.match(rebuildIncidence,
    /sampleCount=select\(1u,4u,row\.size>1u\).*sample<sampleCount/s,
    "negative incidence probes exactly one unit-face footprint or four 2:1 quadrants");
  assert.doesNotMatch(rebuildIncidence, /sample<4u/,
    "unit leaves never probe transverse cells outside their physical face");
  assert.match(rebuildIncidence,
    /neighbor==INVALID.*band>=transitionControl\.support3NodeEnd.*continue.*faceEmissionFail\(BAD_ROW,band,27u\).*neighbor>=transitionControl\.endpointEnd.*band>=transitionControl\.support3NodeEnd.*continue/s,
    "incidence requires a complete incoming star through S2 and terminates only at S3 endpoints");
  assert.match(rebuildIncidence,
    /varitems:array<u32,24>.*incidence\[band\]=count/s,
    "affected rows rebuild one deterministic bounded incidence record");
  const publishCounts = wgslFunction("publishFaceBandCounts");
  assert.match(publishCounts,
    /incidenceCount\+=.*topologyFaceCount\+=1u;velocityTargetCount\+=select\(0u,1u,faceVelocityTarget/s,
    "topology counts every LIVE closure face while velocity acceptance counts physical targets only");
  assert.match(publishCounts,
    /totals\.z!=2u\*totals\.x.*control\.faceCount=totals\.y/s,
    "incidence validates against complete topology without making terminal closure a velocity requirement");
  assert.match(publishCounts,
    /topologyPublishLaneRange\(count,lane\).*topologyPublishOffsets\[sourceLane\]=offset.*liveFaceWorklist\[output\+1u\]=faceIndex/s,
    "contiguous lane ranges and their exact prefix retain canonical row/slot ordering");
  assert.doesNotMatch(publishCounts,
    /if\(band<transitionControl\.support1End\)/,
    "S2 physical faces remain velocity targets unless they terminate at topology-only S3");

  const emit = wgslFunction("emitBandFaces");
  assert.match(emit,
    /velocityTarget=band<transitionControl\.support3NodeEnd&&neighbor<transitionControl\.support3NodeEnd.*LIVE\|select\(0u,VELOCITY_TARGET,velocityTarget\)/s,
    "face emission permanently distinguishes physical S0/S1/S2 faces from terminal S3 scalar closure");
  assert.match(wgslFunction("faceVelocityTarget"),
    /face\.flags&\(LIVE\|VELOCITY_TARGET\)\)==\(LIVE\|VELOCITY_TARGET\)/,
    "later velocity stages consume the emitted target authority without another support-tier binding");
  assert.match(wgslFunction("reconstructBandRowVelocity"),
    /letf=faces\[fi\];if\(!faceVelocityTarget\(f\)\)\{continue;\}if\(\(f\.flags&FACE_VELOCITY_VALID\)==0u\)\{vectorFail/,
    "cell-vector least squares ignores topology-only terminal faces and fails closed on every physical target");
});

test("terminal endpoint inversion is sparse direct scatter", () => {
  const append = wgslFunction("appendBandPhiNeighbor");
  const clear = wgslFunction("clearBandPhiEndpointIncomingEdges");
  const validate = wgslFunction("validateBandPhiEndpointIncomingEdges");
  assert.match(clear,
    /atomicStore\(&endpointIncomingCounts\[endpoint\],0u\)[\s\S]*PowerIncidence\(INVALID,0\)/,
    "each terminal identity exclusively clears its bounded reverse-edge range");
  assert.match(append,
    /neighbor>=transitionControl\.support3NodeEnd[\s\S]*atomicAdd\(&endpointIncomingCounts\[neighbor\],1u\)[\s\S]*PowerIncidence\(parent,0\)/,
    "each source edge directly appends its reciprocal terminal edge");
  assert.doesNotMatch(validate, /source<transitionControl\.support3NodeEnd/,
    "endpoint validation cannot regress to the endpoint/source cross product");
  assert.match(validate,
    /atomicLoad\(&endpointIncomingCounts\[endpoint\]\)[\s\S]*prior\.face<=value\.face[\s\S]*transientPowerIncidences\[base\+cursor\]=value/,
    "exclusive endpoint validation restores deterministic source-row ordering");
  const schedule = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(schedule,
    /run\("clearBandPhiEndpointEdges"[\s\S]*run\("buildBandPhiEdges"[\s\S]*run\("buildBandPhiEndpointEdges"/,
    "clear, sparse scatter, and canonical validation remain dependency ordered");
});

test("direct closest-point extension is the sole Section 5 velocity authority", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const implementation = compact(WebGPUOctreeFaceClosestPointExtension);
  assert.match(implementation, /extendClosestPoints:pipeline\("extendFaceClosestPoints"\)/,
    "the direct closest-point kernel must have a live production pipeline");
  assert.match(source,
    /run\("extendClosestPoints",\[\[0,this\.params\],\[6,this\.candidateRows\],\[7,this\.candidateRowDirectory\],\[12,this\.candidateFaces\],\[14,this\.candidateIncidence\],\[26,input\.owners\],\[27,this\.candidateTransitionMetrics\],\[28,tetrahedronHeaders\],\[29,tetrahedra\],\[30,tetrahedronVertices\]\],0,pass,228\)/,
    "one dispatch binds the unpublished exact graph used at each physical closest point");
  assert.doesNotMatch(source,
    /run\("extendClosestPoints",[^\n;]*input\.powerRowVelocities/,
    "dry extension no longer binds the cell-centre power-row field");
  assert.doesNotMatch(source, /Math\.ceil\(this\.plan\.faceCapacity\/64\)/,
    "no face stage may return to a fixed-capacity dispatch tail");
  assert.deepEqual(wgslReachableBindings("extendFaceClosestPoints"),
    [0, 6, 7, 12, 14, 26, 27, 28, 29, 30],
    "direct extension stays at the portable ten-storage-buffer limit");
  const closestPointLiquid = wgslFunction("closestPointLiquidVector");
  assert.match(closestPointLiquid,
    /point=halfOpenDomainPoint\(face\.velocity\.xyz\).*ownerPoint=select\(point,halfOpenDomainPoint\(point\+1e-4\*inward\/inwardLength\).*owner=ownerAt\(vec3u\(floor\(ownerPoint\)\)\).*resolveWetFaceVectorAtPoint\(band,point\)/s,
    "ownership moves inward while wet interpolation stays at the physical closest point");
  assert.match(wgslFunction("publishClosestPoint"),
    /rawClosest=face\.centroid\.xyz-\(sampled\.phi\/width\)\*normal.*closest=halfOpenDomainPoint\(rawClosest\)/s,
    "Track G publishes the physical closest point without shifting the velocity query");
  assert.doesNotMatch(compact(octreeFaceBandWGSL), /fnliquidStencilInteriorDepth\(/,
    "the obsolete inward-to-deepest-simplex workaround is deleted");
  assert.doesNotMatch(closestPointLiquid, /liquidCentroidVector|powerRowVelocities/,
    "the dry closest-point path cannot sample the pre-extension power-row field");
  assert.match(wgslFunction("rejectClosestPointFace"),
    /faces\[index\]=face/,
    "CPT diagnostics remain on the invocation-owned face record");
  assert.match(wgslFunction("finalizeFaceBandClosestPointDiagnostics"),
    /if\(reason==CPT_NO_OWNER\).*noOwner\+=1u.*firstNoOwner=min\(firstNoOwner,slot\).*CPT_SUPPORT_OWNER.*CPT_NO_SIMPLEX/s,
    "the terminal reduction deterministically reduces every CPT rejection cause");
  assert.doesNotMatch(closestPointLiquid, /face\.negativeRow|face\.positiveRow|transitionAdjacency/,
    "the distant air face and adjacency walk cannot become interpolation authorities");
  assert.doesNotMatch(source + implementation + compact(octreeFaceBandWGSL),
    /prepareFaceBfsFallback|propagateFaceBfsLayer|faceBfsLayer|faceBfsCausal|closestPointAnchorVector|cptParent|this\.state|@binding\(15\)/,
    "the former predecessor arena and in-place BFS implementation stay deleted");
});

test("terminal S3 rows are topology-only and exact selectors fail closed", () => {
  const reconstruct = wgslFunction("reconstructBandRowVelocity");
  assert.match(reconstruct,
    /provisionalVelocities\[row\]=vec4f\(0\.\);if\(row>=transitionControl\.support3NodeEnd\)\{rowVelocities\[row\]=vec4f\(0\.\);return;\}/,
    "terminal endpoints scrub both phi scratch and provisional velocity before leaving reconstruction");
  const selector = wgslFunction("supportSelectorCellVector");
  assert.match(selector, /\(row\.flags&ROW_SUPPORT3_ENDPOINT\)!=0u/,
    "terminal endpoints are never admitted as cube or Delaunay vectors");
  assert.doesNotMatch(selector, /generation|coldEndpoint/,
    "selector validity is identical at t=0 and every recurring generation");
  const signed = wgslFunction("finalSignedVector");
  assert.match(signed,
    /outside=any\(origin<vec3i\(0\)\)\|\|any\(origin\+vec3i\(i32\(size\)\)>vec3i\(p\.dims\)\).*select\(supportSelectorCellVector\(resolved,size\),reflectedBoundarySelectorVector\(resolved\),outside\)/s,
    "interior selectors stay exact while a wall ghost reads its exact reflected adaptive owner");
  assert.doesNotMatch(signed, /ContainingVector/,
    "an in-domain exact selector miss cannot substitute a different containing cell");
  assert.doesNotMatch(octreeFaceBandWGSL, /fn supportContainingVector|coldEndpoint|generation==2u/,
    "the generation-specific endpoint and containing-cell fallback code is deleted");
});

test("catalog support publishes the exact reflected selector used at container walls", () => {
  const selectorSupport = wgslFunction("selectorSupportCandidate");
  assert.match(selectorSupport,
    /extended=velocityExtendedOrigin\(origin,size\).*resolved=vec3u\(extended\.xyz\).*owner=ownerAt\(resolved\).*outside=any\(origin<vec3i\(0\)\).*if\(!outside&&\(owner\.size!=size\|\|any\(owner\.origin!=resolved\)\)\).*returncandidateFromOwner\(owner,row\)/s,
    "boundary Delaunay vertices request the exact reflected adaptive owner while interior selectors stay exact");
  assert.doesNotMatch(selectorSupport,
    /if\(any\(origin<vec3i\(0\)\)\|\|any\(origin\+vec3i\(i32\(size\)\)>vec3i\(p\.dims\)\)\)\{returninvalidSupportCandidate\(\);\}/,
    "virtual wall selectors cannot be silently dropped during support closure");
  const signed = wgslFunction("finalSignedVector");
  assert.match(signed,
    /reflected=velocityExtendedOrigin\(origin,size\).*reflectedBoundarySelectorVector\(resolved\)/s,
    "publication and interpolation share the same owner-directed boundary mapping");
  const reflected = wgslFunction("reflectedBoundarySelectorVector");
  assert.match(reflected,
    /size=1u.*origin=\(resolved\/vec3u\(size\)\)\*vec3u\(size\).*supportSelectorCellVector\(origin,size\).*if\(size>=p\.maximumLeaf\)\{break;\}size<<=1u/s,
    "a wall ghost resolves only through the bounded published adaptive hierarchy");
  assert.doesNotMatch(reflected, /ownerAt|nearest|distance/,
    "the portable completion path needs neither an extra owner binding nor a nearest-cell repair");
});

test("transient power reduction retains the first sampled-face rejection", () => {
  assert.match(wgslFunction("reduceBandPointField"),
    /TRANSIENT_FACE_ERROR.*atomicMin\(&publicationReductions\[4\],item\)/s,
    "the parallel reduction retains the lowest sampled-face rejection");
  assert.match(wgslFunction("finalizeBandPointField"),
    /diagnosticIndex=atomicLoad\(&publicationReductions\[4\]\).*p0=diagnosticIndex%POINT_MAX_FACES.*p6=failedFace\.pad>>8u/s,
    "the finalizer reports the exact retained face detail");
});

test("closest-point extension has one direct paper path and one bounded dry-face closure", () => {
  const schedule = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.doesNotMatch(schedule, /run\("cptReject"/,
    "the per-face closest-point candidate and rejection are one exact live-face dispatch");
  const extend = wgslFunction("extendFaceClosestPoints");
  assert.match(extend, /rejectClosestPointFace\(index,face\)/,
    "every rejected candidate is attributed in the producing invocation");
  const reject = wgslFunction("rejectClosestPointFace");
  assert.match(reject,
    /faces\[index\]=face/,
    "every rejected live face leaves a fixed attributable record");
  assert.match(wgslFunction("finalizeFaceBandClosestPointDiagnostics"),
    /unresolved\+=1u;firstClosest=min\(firstClosest,face\.globalFace\)/,
    "the terminal reduction counts every unresolved face in stable order");
  assert.doesNotMatch(reject, /FACE_VELOCITY_VALID\s*\|=/,
    "the fused rejection helper cannot manufacture a fallback velocity");
  const gather = wgslFunction("gatherDryFaceClosestPointRepairs");
  const commit = wgslFunction("commitDryFaceClosestPointRepairs");
  const carrier = wgslFunction("closestValidTargetCarrier");
  const carrierRank = wgslFunction("closerFaceCarrier");
  const twoRing = wgslFunction("closestCarrierThroughEndpoint");
  const failedAnchor = wgslFunction("failedClosestPointAnchor");
  const anchorRing = wgslFunction("closestCarrierThroughFailedAnchor");
  const anchorWetSeed = wgslFunction("closestWetSeedAcrossAnchor");
  const residualDry = wgslFunction("residualFaceIsDry");
  const residualEndpointDry = wgslFunction("residualEndpointIsDry");
  const localInterpolationRepair = wgslFunction("localInterpolationRepairEligible");
  assert.match(schedule,
    /run\("extendClosestPoints".*run\("gatherClosestPointRepairs".*run\("commitClosestPointRepairs".*run\("reconstruct"/s,
    "the immutable primary extension precedes the bounded gather/commit waves and row reconstruction");
  assert.match(schedule, /for\(letwave=0;wave<8;wave\+=1\)/,
    "the residual closure has exactly eight dependency waves rather than a convergence loop");
  assert.match(gather,
    /face\.pad!=CPT_SUPPORT_OWNER&&face\.pad!=CPT_NO_SIMPLEX&&face\.pad!=CPT_MISSING_VERTEX.*localInterpolationGap=localInterpolationRepairEligible\(face\).*if\(!residualFaceIsDry\(face\)&&!localInterpolationGap\).*closestCarrierThroughEndpoint\(face\.negativeRow.*closestCarrierThroughEndpoint\(face\.positiveRow/s,
    "only definitely-dry support-row or local-interpolation gaps inspect their two endpoint incidence stars");
  assert.doesNotMatch(gather + commit, /CPT_NO_OWNER/,
    "a closest point without a valid nonzero owner remains fail-closed");
  assert.match(residualDry,
    /finite\(face\.phi\)&&face\.phi>0\..*residualEndpointIsDry\(face\.negativeRow\).*face\.positiveRow!=INVALID&&residualEndpointIsDry\(face\.positiveRow\)/s,
    "residual repair requires a positive target and two present, independently dry endpoints");
  assert.match(localInterpolationRepair,
    /face\.pad==CPT_MISSING_VERTEX\|\|face\.pad==CPT_NO_SIMPLEX\|\|face\.pad==CPT_SUPPORT_OWNER.*finite\(face\.phi\).*face\.negativeRow!=INVALID&&face\.positiveRow!=INVALID/s,
    "only a finite two-ended interpolation-domain gap may bypass the outward march's dry-star proof");
  assert.match(residualEndpointDry,
    /rowIndex<transitionControl\.support1End.*row\.minimumPhi>row\.representativePhi.*row\.representativePhi>row\.maximumPhi.*row\.minimumPhi>residualPhiEpsilon\(scale\)/s,
    "S0/S1 repair dryness uses finite ordered current incident-face extrema");
  assert.match(residualEndpointDry,
    /count=min\(incidence\[rowIndex\],p\.axisStride\).*incident\.flags&\(LIVE\|PHI_VALID\).*minimum=min\(minimum,incident\.phi\).*minimum>residualPhiEpsilon\(scale\)/s,
    "marched S2 rows prove dryness by a bounded complete-star rescan");
  assert.doesNotMatch(gather + commit, /faceCarriesLiquid/,
    "coarse-mixed seed eligibility cannot veto a later current-phi residual repair");
  assert.match(carrier,
    /candidate\.flags&\(PHI_VALID\|FACE_VELOCITY_VALID\).*if\(allowUnorderedPositive\).*candidate\.phi<=epsilon.*elseif\(positivePath\).*candidate\.phi<=epsilon\|\|candidate\.phi\+epsilon>=targetPhi.*abs\(candidate\.phi\).*distanceSquared.*candidate\.globalFace/s,
    "all immutable valid endpoint carriers rank by interface distance, physical distance, then stable face identity");
  assert.doesNotMatch(carrier + commit,
    /\(candidate\.flags&\(SEED\|PRIMARY_EXTENSION\)\)==0u|\(carrier\.flags&\(SEED\|PRIMARY_EXTENSION\)\)==0u/,
    "the separate gather/commit dispatches may consume any already-valid endpoint carrier without chaining repairs");
  assert.match(carrierRank,
    /candidate\.absolutePhi<best\.absolutePhi.*candidate\.seedRank<best\.seedRank.*candidate\.distanceSquared<best\.distanceSquared.*candidate\.globalFace<best\.globalFace/s,
    "an exact-interface-distance tie prefers the immutable seed before physical and stable-identity ties");
  assert.match(twoRing,
    /immediate=closestValidTargetCarrier\(rowIndex,excluded,point,targetPhi,true,false,false\).*count=min\(incidence\[rowIndex\],p\.axisStride\).*bridgeSlot=incidence\[at\].*faceVelocityTarget\(bridge\).*bridge\.flags&PHI_VALID.*bridge\.phi<=bridgeEpsilon\|\|bridge\.phi\+bridgeEpsilon>=targetPhi.*opposite>=transitionControl\.support3NodeEnd.*closestValidTargetCarrier\(opposite,excluded,point,bridge\.phi,true,false,false\)/s,
    "one gather scans fixed first- and second-ring stars through a physical monotone-positive bridge");
  assert.match(carrier,
    /requirePrimary&&!primary.*if\(positivePath\).*candidate\.phi<=epsilon\|\|candidate\.phi\+epsilon>=targetPhi.*elseif\(primary/s,
    "second-ring immutable carriers are strictly between the bridge and a scale-aware positive epsilon");
  assert.match(failedAnchor,
    /face\.pad!=CPT_NO_SIMPLEX.*encoded=-face\.velocity\.w.*encoded<1\.\|\|encoded>16777216\.\|\|abs\(encoded-rounded\)>1e-5.*u32\(rounded\)-1u/s,
    "only exact integer containing-band provenance from interpolation failures opens the anchor path");
  assert.match(anchorRing,
    /anchor<transitionControl\.support3NodeEnd.*anchor>=transitionControl\.endpointEnd.*ROW_PHI\|ROW_SUPPORT3_ENDPOINT.*count=min\(incidence\[anchor\],p\.axisStride\).*bridge\.flags&\(LIVE\|PHI_VALID\).*finite\(bridge\.centroid\.x\).*minimum<=residualPhiEpsilon\(scale\).*bridge\.phi\+bridgeEpsilon>=face\.phi.*opposite<transitionControl\.support1End.*opposite>=transitionControl\.support3NodeEnd.*closestValidTargetCarrier\(opposite,excluded,point,bridge\.phi,true,true,false\).*closestWetSeedAcrossAnchor\(opposite,excluded,point,bridge\)/s,
    "the exact terminal support anchor crosses one complete positive closure star back to immutable nonterminal carriers");
  assert.match(anchorWetSeed,
    /rowIndex<transitionControl\.support1End.*rowIndex>=transitionControl\.support3NodeEnd.*coarseWidth=fp\.fineWidth\*f32\(fp\.fineFactor\).*required=LIVE\|VELOCITY_TARGET\|PHI_VALID\|SEED\|FACE_VELOCITY_VALID.*candidate\.flags&PRIMARY_EXTENSION.*candidate\.phi>epsilon.*phiDistance>edgeDistance\+tolerance.*abs\(candidate\.phi\).*candidate\.globalFace/s,
    "the anchor-only terminal seed is current immutable wet authority with a local signed-distance edge proof");
  assert.doesNotMatch(anchorRing, /while\(|loop\{/,
    "anchor provenance cannot become an arbitrary graph walk");
  assert.match(gather,
    /closestCarrierThroughEndpoint\(face\.positiveRow.*closestCarrierThroughFailedAnchor\(index,face,point,best\).*best\.slot==INVALID&&localInterpolationGap.*closestValidTargetCarrier\(face\.negativeRow,index,point,face\.phi,false,false,true\).*closestValidTargetCarrier\(face\.positiveRow,index,point,face\.phi,false,false,true\)/s,
    "endpoint and anchor-directed candidates share one deterministic global rank");
  assert.match(commit,
    /localInterpolationGap=localInterpolationRepairEligible\(face\).*carrier\.flags&\(PHI_VALID\|FACE_VELOCITY_VALID\).*wetSeed=\(carrier\.flags&SEED\)!=0u&&carrier\.phi<=epsilon.*outwardAir=carrier\.phi>epsilon&&carrier\.phi\+epsilon<face\.phi.*carrierDistance=length\(carrier\.centroid\.xyz-face\.centroid\.xyz\)\*fp\.fineWidth\*f32\(fp\.fineFactor\).*localInterpolationCarrier=localInterpolationGap&&carrier\.phi>epsilon&&abs\(face\.phi-carrier\.phi\)<=carrierDistance\+epsilon.*face\.flags=\(face\.flags&~PRIMARY_EXTENSION\)\|FACE_VELOCITY_VALID/s,
    "commit revalidates a strictly inward or physically local signed-distance carrier and keeps the repair non-primary");
  assert.deepEqual(wgslReachableBindings("gatherDryFaceClosestPointRepairs"), [0, 1, 6, 12, 14, 32, 67, 75]);
  assert.deepEqual(wgslReachableBindings("commitDryFaceClosestPointRepairs"), [0, 1, 6, 12, 14, 32, 67, 75]);
  assert.doesNotMatch(gather + commit, /rowCount|rowDirectory|while\(|loop\{/,
    "the closure cannot iterate or scan either row arena");
  assert.doesNotMatch(schedule + compact(octreeFaceBandWGSL),
    /rejectUnresolvedClosestPoints|repairClosestPointLayer|@binding\(56\)|@binding\(57\)/,
    "the deleted iterative graph-repair kernels, buffers, and bindings stay absent");
});

test("band-phi Jacobi remains bounded by the Section 5 narrow band", () => {
  const projection = compact(WebGPUOctreeProjection);
  assert.match(projection,
    /bandPhiRelaxationRounds=/);
  assert.doesNotMatch(projection, /dims\.nx\+this\.dims\.ny\+this\.dims\.nz/,
    "the 2017 fine level set must not require whole-domain propagation");
  const schedule = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.equal(schedule.match(/run\("extendBandPhi"/g)?.length, 1,
    "one persistent workgroup owns every bounded Jacobi round");
  assert.doesNotMatch(schedule, /round<this\.bandPhiRelaxationRounds|currentPhi|nextPhi/,
    "the host does not rebuild bindings or dispatch once per relaxation round");
  assert.match(compact(WebGPUOctreeFaceClosestPointExtension),
    /BAND_PHI_RELAXATION_ROUNDS:bandPhiRelaxationRounds/,
    "the immutable bounded round count is a construction-time pipeline override");
  assert.doesNotMatch(projection + schedule,
    /maximumCptGraphDepth|maximumNarrowBandGraphDepth|cptPlan/,
    "no graph-depth control survives the direct cutover");
});

test("face emission binds only globals statically used by its auto layout", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const emit = source.match(/run\("emit",\[([\s\S]*?)\],0,pass,204\)/)?.[1];
  assert.ok(emit);
  assert.doesNotMatch(emit, /\[4,input\.powerRowDirectory\]/,
    "power-row lookup is consumed while mapping rows, not while emitting the induced face graph");
  for (const binding of [0, 5, 6, 7, 12, 26, 32, 65]) assert.match(emit, new RegExp(`\\[${binding},`));
  for (const binding of [9, 10, 13, 14]) assert.doesNotMatch(emit, new RegExp(`\\[${binding},`));
});

test("catalog adjacency resolves at stable row ids within the portable storage limit", () => {
  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const topology = source.slice(source.indexOf('case"topology-build"'),
    source.indexOf('case"transition-adjacency"'));
  const encoded = faceBandEncodeBindings();
  assert.match(topology, /run\("buildTopologyDelta".*0,pass,240\)/,
    "the core copy consumes the exact indirect row count");
  assert.deepEqual(wgslReachableBindings("buildFaceBandTopologyDelta"),
    [0, 4, 5, 6, 7, 26, 61]);
  assert.deepEqual(encoded.get("emitCatalogSupport1"),
    [0, 5, 6, 7, 26, 28, 29, 30, 33, 34, 66]);
  assert.deepEqual(encoded.get("emitCatalogSupport2"), encoded.get("emitCatalogSupport1"));
  assert.deepEqual(encoded.get("markSupport1"), [0, 5, 6, 7, 66, 67]);
  assert.deepEqual(encoded.get("markSupport2"), encoded.get("markSupport1"));
  assert.deepEqual(encoded.get("scanSupportIdentityBlocks"), [0, 67]);
  assert.deepEqual(encoded.get("scatterSupport1"), [0, 5, 6, 7, 13, 62, 67, 68]);
  for (const entryPoint of [
    "emitFaceBandCatalogSupport1", "emitFaceBandCatalogSupport2",
  ]) assert.equal(wgslReachableBindings(entryPoint).filter((binding) => binding !== 0).length, 10,
    `${entryPoint} remains at WebGPU's portable ten-storage-buffer limit`);
  for (const entryPoint of ["markSupport1", "scatterSupport1"]) {
    assert.ok(wgslReachableBindings(entryPoint, octreeFaceBandSupportScatterWGSL)
      .filter((binding) => binding !== 0).length <= 10,
    `${entryPoint} remains within WebGPU's portable storage-buffer limit`);
  }

  const transition = source.slice(source.indexOf('case"transition-adjacency"'),
    source.indexOf('case"closest-point-extension"'));
  assert.equal(transition.match(/computePass\("/g)?.length, 4);
  const describe = transition.match(/run\("describeCatalogRows",\[([\s\S]*?)\],0,pass,204/)?.[1];
  assert.ok(describe);
  const describeBindings = [...describe.matchAll(/\[(\d+),/g)].map((match) => Number(match[1]));
  assert.deepEqual(describeBindings, [0, 6, 26, 27, 32, 33, 34]);
  assert.deepEqual(wgslReachableBindings("describeCatalogBandRows"), describeBindings);
  const resolve = transition.match(/run\("resolveCatalogAdjacency",\[([\s\S]*?)\],0,pass,204/)?.[1];
  assert.ok(resolve);
  const bindings = [...resolve.matchAll(/\[(\d+),/g)].map((match) => Number(match[1]));
  assert.deepEqual(bindings, [0, 6, 7, 26, 27, 28, 29, 30, 31, 32]);
  assert.deepEqual(wgslReachableBindings("resolveCatalogTransitionAdjacency"), bindings);
  assert.equal(bindings.filter((binding) => binding !== 0).length, 9);
  const publish = transition.match(/run\("publishCatalogAdjacency",\[([\s\S]*?)\],1,pass\)/)?.[1];
  assert.ok(publish);
  const publishBindings = [...publish.matchAll(/\[(\d+),/g)].map((match) => Number(match[1]));
  assert.deepEqual(publishBindings, [0, 6, 18, 27, 32, 65]);
  assert.deepEqual(wgslReachableBindings("publishCatalogTransitionAdjacency"), publishBindings);
  assert.doesNotMatch(transition,
    /enumerateSupport|resolveSupportOwners|insertSupport|captureSupport|emitDeep|transitionDeep/);
});

test("catalog adjacency validates every row and record before publication", () => {
  assert.equal(OCTREE_FACE_BAND_TRANSITION_ERROR.invalidBandDescriptor, 8);
  const prepare = wgslFunction("prepareCatalogTransitionAdjacency");
  const describe = wgslFunction("describeCatalogBandRows");
  const resolve = wgslFunction("resolveCatalogTransitionAdjacency");
  const validate = wgslFunction("validateCatalogTransitionAdjacency");
  const publish = wgslFunction("publishCatalogTransitionAdjacency");
  assert.match(prepare, /transitionControl\.ready=0u/);
  assert.match(prepare,
    /transitionControl\.coreEnd=min\(control\.initialRows,transitionControl\.rowCount\)/,
    "the immutable power prefix remains distinct from deterministic support-only rows");
  assert.match(describe, /metrics\[band\]=describeBandRow\(band\)/);
  assert.doesNotMatch(resolve, /describeBandRow|sameOrFinerDirect|sameOrCoarserDirect/);
  assert.match(resolve, /!exactCatalogRowIdentity\(band,row\)/);
  assert.match(wgslFunction("exactCatalogRowIdentity"),
    /letexactGlobal=select\(row\.globalRow>0u,row\.globalRow==band,band<transitionControl\.coreEnd\)/);
  assert.match(resolve, /letselectors=vec3u\(packed&255u,\(packed>>8u\)&255u,\(packed>>16u\)&255u\)/);
  assert.match(resolve, /TransitionAdjacency\(band,a,b,c\)/);
  assert.match(resolve,
    /cachedTransitionNeighbor\(band,header\.first,local,selectors\.x,metric\)/,
    "each immutable catalog selector resolves once per row and is reused from prior emitted tetrahedra");
  assert.match(wgslFunction("cachedTransitionNeighbor"),
    /for\(varprior=0u;prior<local;prior\+=1u\).*if\(selectors\.x==selector\)\{returnrecord\.a;\}.*returntransitionNeighbor\(band,selector,metric\)/s,
    "the local adjacency record is the exact generation-fixed selector cache");
  assert.doesNotMatch(resolve, /band>=transitionControl\.support1End/,
    "every S0/S1/S2 interpolation anchor publishes its local Delaunay adjacency");
  assert.match(validate, /rows\[band\]\.flags\|=ROW_TRANSITION_VALIDATED/);
  assert.match(validate,
    /band>=transitionControl\.support3NodeEnd.*return.*exactCatalogNeighborIdentity\(record\.a,transitionControl\.endpointEnd\).*exactCatalogNeighborIdentity\(record\.b,transitionControl\.endpointEnd\).*exactCatalogNeighborIdentity\(record\.c,transitionControl\.endpointEnd\)/s,
    "terminal S3 rows are vertex-only, while every S0/S1/S2 anchor validates its tetra vertices through the S3 endpoint prefix");
  assert.doesNotMatch(validate, /band>=transitionControl\.support1End/,
    "S2 must not lose the paper's local Delaunay interpolant at its owned regular faces");
  assert.match(wgslFunction("carryCatalogTransitionAdjacency"),
    /remapCommittedTransitionNeighbor\(prior\.a,transitionControl\.endpointEnd\).*remapCommittedTransitionNeighbor\(prior\.b,transitionControl\.endpointEnd\).*remapCommittedTransitionNeighbor\(prior\.c,transitionControl\.endpointEnd\)/s,
    "warm generations carry exact S3 vertex identities for every retained S2 tetrahedron");
  assert.match(publish,
    /transitionControl\.adjacencyCount>transitionControl\.support3NodeEnd\*MAX_TETRA/,
    "the publication bound covers every Delaunay anchor and no terminal endpoint anchor");
  assert.match(publish,
    /topologyPublishLaneRange\(min\(count,arrayLength\(&rows\)\),lane\).*topologyPublishSums\[lane\]=vec4u\(validatedRows,adjacencyCount,boundaryGhostRequests,affectedRows\).*width=128u/s,
    "publication reduces contiguous row intervals across the whole workgroup");
  assert.match(octreeFaceBandWGSL,
    /@compute @workgroup_size\(256\)fn publishCatalogTransitionAdjacency/,
    "transition tallies must not return to a capacity-sized singleton walk");
  const neighborIdentity = wgslFunction("exactCatalogNeighborIdentity");
  assert.match(neighborIdentity, /if\(neighbor==INVALID\)\{returntrue;\}/,
    "an explicit catalog boundary-ghost sentinel is a valid adjacency lane");
  assert.match(neighborIdentity,
    /returnneighbor<count&&exactCatalogRowIdentity\(neighbor,rows\[neighbor\]\)/,
    "every non-ghost lane remains range checked and exact-identity checked");
  assert.doesNotMatch(neighborIdentity, /neighbor>=count|returntrue;returntrue/,
    "no non-sentinel out-of-range neighbor may pass publication");
  assert.match(publish,
    /letclean=transitionControl\.flags==0u;writeSupportDispatch\(54u,select\(0u,count,clean\)\)/,
    "a rejected candidate must publish zero indirect work");
  assert.match(publish,
    /transitionControl\.hierarchyReady=select\(0u,VALID,clean\)/);
  assert.match(wgslFunction("gateTransitionTransfer"),
    /transitionFlags==0u&&transitionControl\.hierarchyReady==VALID/);
  assert.doesNotMatch(wgslFunction("gateTransitionTransfer"), /fail\(BAD_ROW/,
    "the readiness gate must not overwrite the producer's first failing identity with a synthetic zero");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /fn (?:enumerateSupportRequests|resolveSupportOwners|insertSupportCandidate|captureSupport\w+Boundary|buildDeepTransitionAdjacency)\(/);
});
test("paper Section 5 regular-to-power publication is fail-closed and centroid based", () => {
  assert.equal(OCTREE_FACE_BAND_POWER_PUBLICATION_ERROR.incomplete, 64);
  const wet = wgslFunction("bandRowIsWet");
  assert.match(wet,
    /\(rows\[band\]\.flags&ROW_PHI\)!=0u&&finite\(rows\[band\]\.representativePhi\)&&rows\[band\]\.representativePhi<0\.0/,
    "projected-face preservation uses only a finite current extended liquid sign");
  const interpolatePower = wgslFunction("interpolatePowerFaceVector");
  assert.match(interpolatePower,
    /if\(bandRowIsWet\(negativeBand\)\|\|bandRowIsWet\(positiveBand\)\)\{powerVelocityScratch\[index\]=vec4u\(0u,0u,0u,2u\);return;\}/,
    "one-sided extrapolation must preserve every projected face incident to liquid, including a wet/out-of-band interface face");
  assert.doesNotMatch(interpolatePower,
    /bandRowIsWet\(negativeBand\)&&bandRowIsWet\(positiveBand\)/,
    "interface faces are pressure degrees of freedom, not air-side extrapolation targets");
  assert.match(wgslFunction("projectPowerFaceVelocity"),
    /if\(candidate\.w==2u\)\{powerVelocityScratch\[index\]=vec4u\(0u,3u,0u,0u\);return;\}/,
    "preserved projected faces publish the deterministic terminal marker counted by the later transaction");
  assert.match(octreeFaceBandWGSL, /fn marchedCentroidVector\(anchor:u32,pointGrid:vec3f\)/,
    "power-face interpolation must evaluate the extended full-vector field at the physical face centroid");
  assert.match(octreeFaceBandWGSL,
    /if\(\(header\.flags&1u\)!=0u\)[\s\S]*result\+=weight\*value\.xyz/,
    "uniform regions must use the paper's regular-grid trilinear interpolant");
  assert.match(octreeFaceBandWGSL,
    /let weights=tetraWeights\(point,[\s\S]*result=weights\.x\*anchorVelocity\.xyz[\s\S]*result\+=weights\.y\*va\.xyz[\s\S]*result\/=validWeight/,
    "transition regions use normalized catalog-Delaunay barycentric interpolation");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /faces\[regularFace\]\.area\/max\(dot\(delta,delta\),0\.0625\)/,
    "regular-to-power publication must not revive inverse-distance incident-face averaging");
  assert.match(octreeFaceBandWGSL, /let pointGrid=centroid\/h/);
  assert.match(octreeFaceBandWGSL, /let value=dot\(full,normal\)/,
    "the interpolated full vector is projected onto the actual generalized face normal");
  assert.match(octreeFaceBandWGSL,
    /control\.generation!=p\.generation[^}]+powerFaceControl\[7\]!=p\.powerGeneration/,
    "fine and power generations must both match before scratch publication");
  assert.match(octreeFaceBandWGSL,
    /atomicLoad\(&transitionControl\.ready\)!=VALID\|\|transitionControl\.transferReady!=VALID/,
    "the catalog-Delaunay all-band topology must publish before regular-to-power transfer");
  assert.match(wgslFunction("powerFacePublicationReady"),
    /targets>0u&&atomicLoad\(&powerPublication\.interpolatedCount\)==targets/,
    "partial interpolation cannot open the commit gate");
  assert.match(wgslFunction("commitPowerFaceVelocity"),
    /letready=powerFacePublicationReady\(\).*if\(!ready\|\|index>=powerPublication\.faceCount[\s\S]*return;\}letcandidate=powerVelocityScratch\[index\]/,
    "power face records remain untouched unless the whole target subset is valid");
  assert.match(octreeFaceBandWGSL,
    /fn mapPowerFaceBands[\s\S]*let negativeBand=bandForGlobalRow\(powerFace\.negativeRow\)[\s\S]*powerVelocityScratch\[index\]=vec4u\(0u,0u,negativeBand,positiveBand\)/,
    "the mapping stage must publish only power-endpoint to regular-band identities");
  assert.match(wgslFunction("bandForGlobalRow"),
    /globalRow>=atomicLoad\(&control\.initialRows\).*rows\[globalRow\]\.globalRow!=globalRow.*returnglobalRow/s,
    "only immutable power-prefix ids map directly; support-suffix ids never masquerade as pressure rows");
  assert.match(octreeFaceBandWGSL,
    /fn interpolatePowerFaceVector[\s\S]*let mapping=powerVelocityScratch\[index\];let negativeBand=mapping\.z;let positiveBand=mapping\.w/,
    "the interpolation stage must consume the completed endpoint mapping instead of repeating hash lookup");

  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const prepare = source.match(/run\("preparePowerPublication",\[([\s\S]*?)\],1,pass\d*\)/)?.[1];
  assert.ok(prepare);
  for (const binding of [0, 5, 18, 32, 36, 37, 38, 39, 40, 41, 48]) {
    assert.match(prepare, new RegExp(`\\[${binding},`));
  }
  assert.match(prepare, /\[48,this\.pointFieldControl\]/,
    "regular-to-power publication must consume the complete final point-field transaction");
  assert.match(wgslFunction("preparePowerPublication"),
    /writeSupportDispatch\(54u,0u\).*writeSupportDispatch\(54u,powerPublication\.faceCount\)/s,
    "the producer publishes zero work until the exact live power-face prefix validates");
  const map = source.match(/run\("mapPowerFaceBands",\[([\s\S]*?)\],0,pass,216/)?.[1];
  assert.ok(map);
  for (const binding of [5, 6, 37, 40, 41]) {
    assert.match(map, new RegExp(`\\[${binding},`));
  }
  for (const binding of [0, 1, 12, 14, 15, 19, 35, 36, 38, 39]) {
    assert.doesNotMatch(map, new RegExp(`\\[${binding},`),
      "endpoint mapping must not bind regular-vector interpolation or unused publication inputs");
  }
  const mapBindings = [...map.matchAll(/\[(\d+),/g)].map((match) => Number(match[1]));
  assert.deepEqual(wgslReachableBindings("mapPowerFaceBands"), mapBindings);
  const interpolate = source.match(/run\("interpolatePowerFaces",\[([\s\S]*?)\],0,pass,216/)?.[1];
  assert.ok(interpolate);
  assert.match(interpolate, /\[19,this\.velocities\]/,
    "centroid interpolation must use the final S0/S1 plus support-row velocity field");
  for (const binding of [0, 1, 6, 7, 19, 27, 28, 29, 30, 39, 40, 41]) {
    assert.match(interpolate, new RegExp(`\\[${binding},`));
  }
  for (const binding of [5, 12, 14, 15, 35, 36, 37, 38]) {
    assert.doesNotMatch(interpolate, new RegExp(`\\[${binding},`),
      "catalog interpolation must stay within the portable ten-storage-buffer stage limit");
  }
  assert.equal(interpolate.match(/\[\d+,/g)?.length, 12,
    "catalog interpolation binds exactly two uniforms plus ten storage buffers");
  const project = source.match(/run\("projectPowerFaces",\[([\s\S]*?)\],0,pass,216/)?.[1];
  assert.ok(project);
  for (const binding of [0, 37, 38, 40, 41]) {
    assert.match(project, new RegExp(`\\[${binding},`));
  }
  for (const binding of [6, 7, 19, 27, 28, 29, 30, 39]) {
    assert.doesNotMatch(project, new RegExp(`\\[${binding},`),
      "normal projection must not bind the catalog interpolation arena again");
  }
  const prepareAt = source.indexOf('run("preparePowerPublication"');
  const mapAt = source.indexOf('run("mapPowerFaceBands"', prepareAt);
  const interpolateAt = source.indexOf('run("interpolatePowerFaces"', mapAt);
  const projectAt = source.indexOf('run("projectPowerFaces"', interpolateAt);
  const commitAt = source.indexOf('run("commitPowerFaces"', projectAt);
  assert.ok(prepareAt >= 0 && mapAt > prepareAt && interpolateAt > mapAt
    && projectAt > interpolateAt && commitAt > projectAt,
  "Section 5 transfer must map endpoints, interpolate the completed regular field, project, validate globally, then commit");
  assert.doesNotMatch(source, /run\("publishPowerFaces"/,
    "commit workgroups evaluate the completed aggregate once without a publication-only launch");
  assert.doesNotMatch(source, /Math\.ceil\(this\.plan\.powerFaceCapacity\/64\)/,
    "no regular-to-power stage may launch the retired capacity-sized tail");
});

test("paper Section 5 retains least-squares liquid power-cell vectors before extrapolating support", () => {
  const seedMapped = wgslFunction("seedMappedPowerRowVelocity");
  assert.match(seedMapped,
    /letrow=rows\[band\];if\(\(row\.flags&ROW_CORE\)==0u\)\{return;\}/,
    "only outside-liquid support without a mapped power row may rely on extrapolated closure");
  assert.match(seedMapped,
    /letvalue=powerRowVelocities\[row\.globalRow\]/,
    "mapped liquid cells must consume the generalized-face least-squares centre vector");
  assert.match(seedMapped,
    /rowVelocities\[band\]=value;provisionalVelocities\[band\]=value/,
    "the authoritative cell vector must reach both the interpolation and publication fields");
  assert.doesNotMatch(seedMapped, /nearest|rest=|vec4f\(0\.,0\.,0\.,1\.\)/,
    "mapped power cells must not be replaced by nearest-cell or rest-state scaffolding");

  const source = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const reconstructAt = source.indexOf('run("reconstruct"');
  const mappedAt = source.indexOf('run("seedMappedPowerRows"', reconstructAt);
  const commitAt = source.indexOf('run("commitRows"', mappedAt);
  assert.ok(reconstructAt >= 0 && mappedAt > reconstructAt && commitAt > mappedAt,
    "real power-cell vectors override reconstructed catalog rows before publication");
});

test("factor-4/factor-8 production schedule publishes and consumes the closest-point-extended air band", () => {
  const projection = compact(WebGPUOctreeProjection.prototype.encode);
  const faceBand = compact((WebGPUOctreeProjection.prototype as unknown as {
    encodeGlobalFineFaceBandPhase: (broker: PassBroker, phase: string) => void;
  }).encodeGlobalFineFaceBandPhase);
  const powerVelocity = projection.indexOf("this.encodePowerVelocityPublication(projectionBroker)");
  const faceClosestPoint = projection.indexOf("this.encodeGlobalFineFaceBand(encoder", powerVelocity);
  const pressureImpulses = projection.indexOf("this.powerSolidFaces?.encodePressureImpulses(projectionTailBroker", faceClosestPoint);
  assert.ok(powerVelocity >= 0 && faceClosestPoint > powerVelocity && pressureImpulses > faceClosestPoint,
    "native power velocity -> face-band extension must precede downstream publication");
  assert.match(faceBand,
    /this\.globalFineCurrentIsA\?this\.globalFineSourceA:this\.globalFineSourceB/,
    "face march must use the currently published fine generation, not the destination generation");
  assert.match(faceBand,
    /this\.globalFineCurrentIsA\?this\.globalFineTopologyBA:this\.globalFineTopologyAB/,
    "face march must select the topology transaction that produced the current A/B fine slot");
  assert.match(faceBand, /fineTopologyControl:fineTopology\.control/,
    "rollback generation admission must use the selected current-slot topology proof");
  assert.match(faceBand, /this\.globalFineBootstrapped/,
    "the empty bootstrap source must not be treated as a valid extrapolation band");
  assert.match(faceBand,
    /owners:this\.topology,coarsePhiDirectory:this\.powerCoarseLevelSetSchedule\.sampleSource\.directory/,
    "the face band must bind owner topology and published coarse phi");
  assert.match(faceBand, /powerTopology:this\.powerTopology\.source/,
    "the face band must reuse the exact Stage-B catalog source");
  assert.match(faceBand, /powerFaces:this\.powerFaces\.source/,
    "the completed face extension must transactionally republish onto generalized power faces");
  const recapture = faceBand.indexOf("this.powerFaceAdvection.encodeCapture(broker");
  assert.ok(recapture > faceBand.indexOf("this.globalFineFaceExtension.encodePhase(broker"),
    "the extrapolated native power field must become the next old-mesh snapshot");
  assert.doesNotMatch(faceBand, /encodePowerToAxis|faceMirror/,
    "Section 5 publication must not round-trip through Cartesian faces");
  assert.doesNotMatch(faceBand, /powerFaceTransfer\?\.encodeCapture/,
    "the unused generalized-face transfer must not sort a dead snapshot");

  const transport = compact(WebGPUFineLevelSetTransport.prototype.encode);
  assert.match(transport,
    /this\.velocityPrepass\.prepareFusedSampling[\s\S]*this\.faceBand\.prepareFusedAirSampling/,
    "the fused characteristic must publish both direct and closest-point air sampler parameters");
  assert.match(transport, /\{binding:19,resource:binding\(options\.ownerTopology\)\}/,
    "the fused air sampler must bind the current adaptive owner authority directly");
  assert.doesNotMatch(transport, /copy\(options\.ownerTopology|b\.owners/,
    "the sparse owner-page arena must never be mirrored into transport storage");
  assert.match(transport,
    /run\(this\.prepareDispatchPipeline[\s\S]*broker\.updateIndirectBuffer\(this\.dispatchMetadata/,
    "validated live work must be copied into an immutable indirect-only command buffer");
  assert.match(transport,
    /runIndirect\(this\.fused\.pipeline[\s\S]*runIndirect\(this\.summarizePipeline[\s\S]*runIndirect\(this\.commitPipeline/,
    "trace, summary, and commit must consume only GPU-authored live-prefix dispatches");
  assert.doesNotMatch(transport,
    /Math\.ceil\(this\.(?:positionCapacity|queryCapacity)\/64\)|resetControl|writeBuffer\(this\.control/,
    "capacity-sized launches and the host control reset must be deleted");
  assert.doesNotMatch(transport,
    /encodeFromPositions|encodeAirSamples|advancePipeline|preparePipeline|samplePipeline/,
    "the split per-segment transport implementation must be deleted");
});

test("fine transport directly binds producer-owned grouped authorities", () => {
  const source = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-transport.ts", import.meta.url), "utf8",
  );
  const constructorStart = source.indexOf("  constructor(");
  const encodeStart = source.indexOf("\n  encode(", constructorStart);
  const destroyStart = source.indexOf("\n  destroy(", encodeStart);
  assert.ok(constructorStart >= 0 && encodeStart > constructorStart && destroyStart > encodeStart);
  const constructor = source.slice(constructorStart, encodeStart);
  const encode = source.slice(encodeStart, destroyStart);
  assert.match(constructor,
    /directAuthority:\s*direct\.authority,\s*airAuthority:\s*air\.authority/,
    "transport must retain the two producer-owned grouped authority buffers directly");
  assert.match(encode,
    /this\.velocityPrepass\.encodeFusedAuthority\(broker,\s*options\.rowVelocities\);\s*this\.faceBand\.encodeFusedAuthority\(broker\)/,
    "each producer must publish its grouped raw authority before the fused trace");
  assert.doesNotMatch(constructor, /createCommandEncoder|catalogPack|packCatalog/,
    "transport must not own an immutable catalog mirror");
  assert.doesNotMatch(encode,
    /copy\(direct\.|copy\(air\.|copyBufferToBuffer\(direct\.|copyBufferToBuffer\(air\./,
    "transport must not retain recurring producer mirrors or copies");
});

test("fine A/B consumers retain the submitted publication during an unpublished target probe", () => {
  type PublicationHarness = {
    globalFineCurrentIsA: boolean;
    globalFinePublishedIsA: boolean;
    globalFinePublicationByEncoder: WeakMap<object, boolean>;
    powerCoarseLevelSetSchedule?: { retireSubmittedEncoder(encoder: object): void };
    retireSubmittedEncoder(encoder: object): void;
  };
  const harness = Object.create(WebGPUOctreeProjection.prototype) as PublicationHarness;
  const targetEncoder = {};
  harness.globalFineCurrentIsA = false;
  harness.globalFinePublishedIsA = true;
  harness.globalFinePublicationByEncoder = new WeakMap([[targetEncoder, false]]);
  assert.equal(harness.globalFinePublishedIsA, true,
    "encoding/reset-time probing of unpublished B must leave consumers on submitted A");
  harness.retireSubmittedEncoder(targetEncoder);
  assert.equal(harness.globalFinePublishedIsA, false,
    "only submission of the encoder carrying B finalize/restriction publishes B");

  const surface = compact(WebGPUOctreeProjection.prototype.encodeSurface);
  const register = surface.indexOf("this.globalFinePublicationByEncoder.set(encoder,publicationTargetIsA)");
  const boundary = surface.indexOf('splitProductionPhase("fineRestriction")', register);
  const advance = surface.indexOf("this.globalFineCurrentIsA=publicationTargetIsA", boundary);
  assert.ok(register >= 0 && boundary > register && advance > boundary,
    "the target parity must be attached before the fine-restriction encoder can be split/submitted");
  assert.match(compact(Object.getOwnPropertyDescriptor(
    WebGPUOctreeProjection.prototype, "globalFineLevelSetSource",
  )!.get!), /this\.globalFinePublishedIsA\?this\.globalFineSourceA:this\.globalFineSourceB/,
  "renderer and QA source selection must ignore optimistic encode parity");
});

test("row-zero boundary reconstruction accepts three independent face normals", () => {
  const normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;
  const normalMatrix = Array.from({ length: 3 }, (_, row) =>
    Array.from({ length: 3 }, (_, column) =>
      normals.reduce((sum, normal) => sum + normal[row] * normal[column], 0)));
  assert.deepEqual(normalMatrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const determinant = normalMatrix[0][0]
    * (normalMatrix[1][1] * normalMatrix[2][2] - normalMatrix[1][2] * normalMatrix[2][1])
    - normalMatrix[0][1]
    * (normalMatrix[1][0] * normalMatrix[2][2] - normalMatrix[1][2] * normalMatrix[2][0])
    + normalMatrix[0][2]
    * (normalMatrix[1][0] * normalMatrix[2][1] - normalMatrix[1][1] * normalMatrix[2][0]);
  assert.equal(determinant, 1,
    "the three positive regular faces at cell zero uniquely determine its zero vector");

  const reconstruct = wgslFunction("reconstructBandRowVelocity");
  assert.match(reconstruct,
    /varls=BandLS.*letorientation=select\(-1\.,1\.,f\.negativeRow==row\).*normal\[axis\]=orientation.*letu=orientation\*f\.velocity\[axis\]/s,
    "Stage A accumulates every accepted signed face-normal component");
  assert.match(reconstruct,
    /letdeterminant=xx\*c00\+xy\*c01\+xz\*c02.*determinant<=1e-7\*trace\*trace\*trace.*letcondition=.*condition>1e5.*letresult=vec3f/s,
    "completeness is exact rank and conditioning of the least-squares system");
  assert.doesNotMatch(reconstruct,
    /targetArea|negativeArea|positiveArea|negativeSum|positiveSum/,
    "the deleted six-side area stencil cannot reject a full-rank boundary system");
});

test("rank-deficient support closure uses bounded completed face carriers", () => {
  const carriers = [
    { area: 4, velocity: [0.17682609, -1.78302312, 0.07583368] },
    { area: 4, velocity: [0.23008147, -1.87113881, 0.05250613] },
  ] as const;
  const totalArea = carriers.reduce((sum, carrier) => sum + carrier.area, 0);
  const completed = [0, 1, 2].map((axis) =>
    carriers.reduce((sum, carrier) => sum + carrier.area * carrier.velocity[axis], 0) / totalArea);
  assert.ok(completed.every(Number.isFinite));
  assert.ok(Math.abs(completed[2] - 0.064169905) < 1e-8,
    "the full face carriers retain the component absent from the rank-two normal star");

  const gather = wgslFunction("completedIncidentVector");
  assert.match(gather,
    /count=min\(incidence\[rowIndex\],p\.axisStride\).*if\(!faceVelocityTarget\(face\)\|\|\(face\.flags&FACE_VELOCITY_VALID\)==0u.*weighted\+=face\.area\*face\.velocity\.xyz/s,
    "support completion is bounded by the fixed incidence star and consumes only admitted full-vector carriers");
  assert.doesNotMatch(gather, /rowCount|while\(|loop\{/,
    "support completion cannot scan the row arena or launch a marching search");

  const publish = wgslFunction("publishSupportClosureVector");
  assert.match(publish,
    /if\(row<transitionControl\.support1End\)\{returnfalse;\}.*value=completedIncidentVector\(row\).*provisionalVelocities\[row\]=value;rowVelocities\[row\]=value/s,
    "only the non-core closure tier may use the full-vector completion");
  const reconstruct = wgslFunction("reconstructBandRowVelocity");
  assert.match(reconstruct,
    /determinant<=1e-7\*trace\*trace\*trace\)\{if\(publishSupportClosureVector\(row,r,origin\)\)\{return;\}vectorFail\(INCOMPLETE,row\)/,
    "rank-deficient core and S1 rows remain fail-closed");
});

test("row-zero regular faces seed from the exact core-cell edge restriction", () => {
  const negativeCenter = [0.5, 0.5, 0.5] as const;
  const positiveCenters = [
    [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5],
  ] as const;
  assert.deepEqual(positiveCenters.map((positive) =>
    positive.map((component, axis) => 0.5 * (negativeCenter[axis] + component))),
  [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
  "the three row-zero regular-face centroids are exact edges of the cell-centre mesh");

  const edgeRows = wgslFunction("exactRegularFaceEdgeRows");
  assert.match(edgeRows,
    /negative\.size!=positive\.size.*midpoint=\.5\*\(negativeCenter\+positiveCenter\).*face\.centroid\.xyz-midpoint/s,
    "only equal-size faces at the exact cell-centre midpoint enter the direct restriction");
  const edgePhi = wgslFunction("exactRegularFaceEdgePhi");
  assert.match(edgePhi,
    /publishedBandScalar\(edge\.x\).*publishedBandScalar\(edge\.y\).*phi=\.5\*\(negative\.x\+positive\.x\)/s,
    "the same mesh edge linearly restricts both published endpoint phi values");
  const coarse = wgslFunction("sampleBandFaceCoarsePhi");
  assert.match(coarse,
    /sampled\.valid==0u.*exactRegularFaceEdgePhi\(face\).*edge\.y!=0\.&&edge\.x<=0\..*PhiCPT\(edge\.x,vec3f\(0\.\),1u\)/s,
    "only an exact wet edge may publish PHI_VALID without inventing an air-side closest-point gradient");
  assert.doesNotMatch(coarse, /atomicLoad\(&control\.flags\)/,
    "one diagnosed face cannot race other independent faces into unprocessed LIVE-only records");

  const restriction = wgslFunction("exactRegularFaceCoreVector");
  assert.match(restriction,
    /exactRegularFaceEdgeRows\(face\).*negative=cellVector\(coord\(negativeRow\.cell\),negativeRow\.size\).*positive=cellVector\(coord\(positiveRow\.cell\),positiveRow\.size\).*!velocityValid\(negative\)\|\|!velocityValid\(positive\).*value=\.5\*\(negative\.xyz\+positive\.xyz\)/s,
    "both endpoint power-cell vectors are required and linearly restricted without a fallback");
  const seed = wgslFunction("seedFaceCentroids");
  assert.match(seed,
    /f\.flags&\(LIVE\|PHI_VALID\).*faceCarriesLiquid\(f\).*exactRegularFaceCoreVector\(f\).*f\.flags\|=SEED\|FACE_VELOCITY_VALID/s,
    "the exact vector restriction remains behind explicit current wet-face authority");
  const extend = wgslFunction("extendFaceClosestPoints");
  assert.match(extend,
    /face\.flags&PHI_VALID.*CPT_BAD_FACE.*face\.flags&SEED.*FACE_VELOCITY_VALID.*acceptedCount/s,
    "even an exact core edge cannot bypass the published phi contract");
});

test("Section 5 final point field consumes one complete transient physical power graph", () => {
  for (const entry of ["prepareTransientBandPowerGraph", "emitTransientBandPowerGraph",
    "sampleTransientBandPowerFaces", "validateTransientBandPowerGraph", "transientPowerGraphReady",
    "prepareBandPointField", "reduceBandPointField", "finalizeBandPointField",
    "reconstructBandTransientPowerPointField", "publishBandPointField"]) {
    assert.match(octreeFaceBandWGSL, new RegExp(`fn ${entry}\\b`), `${entry} must exist in WGSL`);
  }
  const emit = wgslFunction("emitTransientBandPowerGraph");
  const prepareTransient = wgslFunction("prepareTransientBandPowerGraph");
  assert.match(prepareTransient,
    /faceSlots=transientPowerControl\.rowCount\*POINT_MAX_FACES/,
    "the transient graph publishes only its current row-slot prefix");
  assert.match(prepareTransient, /rowCount>p\.rowCapacity/,
    "a corrupt live prefix fails closed before addressing the fixed arenas");
  assert.match(emit,
    /for\(varretired=0u;retired<POINT_MAX_FACES;retired\+=1u\)\{transientPowerFaces\[base\+retired\]\.flags=0u;\}/,
    "each current row retires sparse face flags before publishing new owners");
  assert.match(emit, /neighbor>=transitionControl\.support2End/,
    "S0/S1 physical faces may use S2 carriers, but never an unclosed endpoint");
  assert.match(emit, /reverseSlot=transientReciprocalSlot\(row,neighbor\)/,
    "every interior catalog face requires the neighbor's reciprocal slot");
  const neighborLookup = wgslFunction("transientNeighborMeasurement");
  assert.match(neighborLookup, /any\(abs\(originValue-origin\)>vec3f\(2e-4\)\)/,
    "a catalog neighbor must be on the exact integer lattice before row lookup");
  assert.match(emit, /2u\|\(measured\.y<<8u\)/,
    "a rejected transient neighbor publishes its exact lattice/directory/identity cause");
  const validGeometry = wgslFunction("validTransientGeometry");
  assert.match(validGeometry,
    /finite\(g\.neighborCenter\.x\)&&finite\(g\.neighborCenter\.y\)&&finite\(g\.neighborCenter\.z\)/,
    "non-finite catalog neighbor geometry cannot be rounded into a live row");
  const reciprocal = wgslFunction("transientReciprocalSlot");
  assert.match(reciprocal, /lettolerance=max\(1e-5,wantedSize\*2e-5\)/,
    "reciprocal pairing uses the production power-face tolerance exactly");
  assert.match(emit, /if\(row>neighbor\)\{continue;\}/,
    "one owner materializes each shared interior face exactly once");
  assert.match(emit,
    /letbase=row\*POINT_MAX_FACES[\s\S]*transientPowerIncidences\[base\+slot\]=PowerIncidence\(faceIndex,1\)[\s\S]*PowerIncidence\(faceIndex,-1\)/,
    "both endpoint rows receive reciprocal signed incidence to the same face record");
  const polygon = wgslFunction("transientFacePolygon");
  assert.match(polygon,
    /clipTransientByCell\(BandPowerPolygon\(vertices,4u\),row,epsilon\)[\s\S]*clipTransientByCell\(polygon,neighbor,epsilon\)/,
    "shared geometry clips one bisector polygon by both endpoint power cells");
  assert.match(polygon,
    /letreverseGeometry=transientCatalogGeometry\(neighbor,reverseSlot\);center=\.5\*\(geometry\.centroid\+reverseGeometry\.centroid\)/,
    "the reciprocal seed plane is centred from both catalog reconstructions exactly as production sharedGeometry");
  assert.match(polygon, /epsilon=max\(1e-6,1e-5\*scale\)/,
    "transient clipping uses the same scale-relative tolerance as production sharedGeometry");
  const sample = wgslFunction("sampleTransientBandPowerFaces");
  assert.match(sample,
    /letrow=g\.x[\s\S]*row>=transientPowerControl\.rowCount[\s\S]*for\(varslot=0u;slot<POINT_MAX_FACES/,
    "sampling is row-count bounded instead of scanning the face-slot capacity");
  assert.match(sample,
    /marchedCentroidVector\(face\.negativeRow,face\.centroid\.xyz\)[\s\S]*marchedCentroidVector\(face\.positiveRow,face\.centroid\.xyz\)/,
    "both endpoint carriers are interpolated at the exact shared physical centroid");
  assert.match(sample,
    /if\(negativeValid&&positiveValid\)\{full=\.5\*\(negative\.xyz\+positive\.xyz\);\}elseif\(negativeValid\)\{full=negative\.xyz;\}elseif\(positiveValid\)\{full=positive\.xyz;\}/,
    "S1 faces may use their valid incident interpolant when the opposite S2 interpolant would require unbuilt S3 vectors");
  assert.match(sample,
    /letnegativeReason=.*letpositiveReason=.*provisionalS1ToS3Carrier\(face\.negativeRow,face\.positiveRow,negativeReason,positiveReason\).*provisionalS1ToS3Carrier\(face\.positiveRow,face\.negativeRow,positiveReason,negativeReason\)/s,
    "only after both exact endpoint interpolants fail may the face try the symmetric S1-to-S3 exceptional carrier");
  assert.match(wgslFunction("provisionalS1ToS3Carrier"),
    /s1Reason!=7u\|\|s3Reason!=1u.*\(s1\.flags&ROW_SUPPORT1\)==0u.*\(s3\.flags&ROW_SUPPORT3_NODE\)==0u.*provisionalCellVector\(coord\(s1\.cell\),s1\.size\)/s,
    "the O(1) fallback is restricted to the observed S1 missing-selector / S3 non-anchor pair");
  assert.doesNotMatch(sample, /containingPublishedRow|nearest|catalog/,
    "transient face recovery performs no neighborhood or catalog search");
  assert.doesNotMatch(sample,
    /negativeTarget&&!negativeValid|positiveTarget&&!positiveValid/,
    "a T-junction centroid is located by whichever endpoint's paper cube/tetrahedron contains it");
  assert.match(sample,
    /elseif\(positiveValid\)\{full=positive\.xyz;\}else\{letnegativeReason=[\s\S]*letpositiveReason=[\s\S]*transientFaceFail\(POINT_SAMPLE,faceIndex\)/,
    "the physical face still fails closed when neither endpoint interpolation contains its centroid");
  assert.match(sample,
    /transientPowerFaces\[faceIndex\]=face;transientFaceFail\(POINT_SAMPLE,faceIndex\)/,
    "a sampling rejection records its endpoint-validity payload on the exact failed face");
  assert.match(sample, /scalar=dot\(full,face\.normal\.xyz\)/,
    "the exact-centroid vector is projected once to the committed normal scalar");
  assert.match(octreeFaceBandWGSL, /determinant<=1e-7\*trace\*trace\*trace/);
  assert.match(octreeFaceBandWGSL, /condition>1e5/);
  assert.doesNotMatch(octreeFaceBandWGSL, /var best=anchorVelocity|bestDistance=dot\(point,point\)/,
    "authoritative interpolation has no nearest/anchor fallback");
  assert.doesNotMatch(octreeFaceBandWGSL, /fn seedOpenWorldNormal|seedOpenWorld/,
    "out-of-domain open regular faces are not members of the velocity band");
  assert.match(wgslFunction("emitBandFaces"), /if\(boundary>=p\.dims\[axis\]\)\{continue;\}/,
    "positive world-boundary faces are excluded before phi/CPT sampling");
  assert.match(wgslFunction("reconstructBandRowVelocity"),
    /origin\[axis\]\+r\.size==p\.dims\[axis\].*rows\[row\]\.padf=result\[axis\]/s,
    "the authored open boundary publishes the solved normal component without inventing a missing face");
  assert.doesNotMatch(wgslFunction("prepareBandPointField"), /powerPublication/,
    "the transient physical graph must break the former point-field/power-publication cycle");
  assert.match(wgslFunction("preparePowerPublication"),
    /atomicLoad\(&pointControl\.valid\)!=VALID|pointControl\.generation!=p\.generation/,
    "production power faces require the complete same-generation point field");
  const physical = wgslFunction("reconstructBandTransientPowerPointField");
  assert.match(physical,
    /letorientation=f32\(item\.sign\);letnormal=orientation\*face\.normal\.xyz;letu=orientation\*face\.normalVelocity/,
    "every S0/S1 row consumes signed physical normals and normal velocities from the transient CSR");
  assert.match(physical, /letweight=face\.area/,
    "the final least-squares fit uses exact physical generalized-face area");
  assert.doesNotMatch(physical, /globalRow|powerIncidenceRows|catalogFaces/,
    "wet and dry rows share one final physical graph with no authority split");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /prepareBandPointRows|prepareBandPointDispatch|accumulateBandTransientPowerLS|solveBandPowerLS|validateBandPointField|accumulateBandPowerLSBatch|accumulateBandPowerLS0|accumulateBandPowerLS1|accumulateBandPowerLS2/,
    "the point-field clean cut retains one exact row-local LS kernel and no preparation or validation passes");
  assert.doesNotMatch(octreeFaceBandWGSL, /@binding\(46\)|pointAccumulator/,
    "the deleted multi-pass point accumulator has no shader binding or backing arena");
  assert.equal("pointAccumulatorBytes" in planOctreeFaceBandGPU(64, 8, 4, 4, [16, 16, 16]), false,
    "the deleted accumulator is absent from the production allocation plan");

  const schedule = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.doesNotMatch(schedule,
    /clearBuffer\(this\.transientPower(?:Faces|Incidence|Rows|Control)\)/,
    "counted transient publication does not clear any fixed-capacity arena");
  assert.match(schedule,
    /run\("sampleTransientPower"[\s\S]*,0,pass,216\)/,
    "one invocation samples all owned slots for each exact live row");
  const provisionalAt = schedule.indexOf('run("commitRows"');
  const prepareTransientAt = schedule.indexOf('run("prepareTransientPower"', provisionalAt);
  const emitTransientAt = schedule.indexOf('run("emitTransientPower"', prepareTransientAt);
  const sampleTransientAt = schedule.indexOf('run("sampleTransientPower"', emitTransientAt);
  const validateTransientAt = schedule.indexOf('run("validateTransientPower"', sampleTransientAt);
  const preparePointAt = schedule.indexOf('run("preparePointField"', validateTransientAt);
  const reducePointAt = schedule.indexOf('run("reducePointField"', preparePointAt);
  const finalizePointAt = schedule.indexOf('run("finalizePointField"', reducePointAt);
  const physicalAt = schedule.indexOf('run("reconstructPhysicalPoint"', finalizePointAt);
  const publishPointAt = schedule.indexOf('run("publishPoint"', physicalAt);
  const preparePowerAt = schedule.indexOf('run("preparePowerPublication"', publishPointAt);
  const commitAt = schedule.indexOf('run("commitPowerFaces"', preparePowerAt);
  assert.ok(provisionalAt >= 0 && prepareTransientAt > provisionalAt && emitTransientAt > prepareTransientAt
    && sampleTransientAt > emitTransientAt && validateTransientAt > sampleTransientAt
    && preparePointAt > validateTransientAt
    && reducePointAt > preparePointAt && finalizePointAt > reducePointAt
    && physicalAt > finalizePointAt && publishPointAt > physicalAt
    && preparePowerAt > publishPointAt && commitAt > preparePowerAt,
  "paper order is regular march -> all-band physical graph -> final cell-centre LS -> production publication");
  assert.doesNotMatch(schedule, /run\("publishTransientPower"/,
    "point-field preparation publishes the immutable transient aggregate without a singleton-only launch");
  const physicalBindings = schedule.match(/run\("reconstructPhysicalPoint",\[([\s\S]*?)\],0,pass,216\)/)?.[1];
  assert.ok(physicalBindings);
  assert.equal(physicalBindings.match(/\[\d+,/g)?.length, 8,
    "all-band physical LS stays within the portable storage-binding limit");
  const emitBindings = schedule.match(/run\("emitTransientPower",\[([\s\S]*?)\],0,pass,216\)/)?.[1];
  assert.equal(emitBindings?.match(/\[\d+,/g)?.length, 11,
    "physical graph emission uses one uniform plus the portable maximum ten storage buffers");
});

test("transient Section 5 power graph accepts only the current or exact predecessor retained band", () => {
  const generation = wgslFunction("committedBandGenerationValid");
  assert.match(generation,
    /fine=p\.generation&mask.*band=control\.generation&mask.*power=p\.powerGeneration&mask.*predecessor=\(fine\+mask\)&mask.*returnband==fine\|\|\(power==fine&&band==predecessor\)/s,
    "the recurring power clock may consume exactly one retained face-band predecessor, never an older publication");
  const prepareTransient = wgslFunction("prepareTransientBandPowerGraph");
  assert.match(prepareTransient,
    /writeSupportDispatch\(54u,0u\).*control\.valid!=VALID.*!committedBandGenerationValid\(\).*transientFail\(POINT_SOURCE,0u\).*writeSupportDispatch\(54u,transientPowerControl\.rowCount\)/s,
    "an out-of-contract committed band must reject and retain a zero dispatch before transient generation publication");
  assert.match(wgslFunction("finalizeBandPointField"), /!committedBandGenerationValid\(\)/,
    "the point-field finalizer shares the same retained-band clock");
  assert.match(wgslFunction("preparePowerPublication"), /!committedBandGenerationValid\(\)/,
    "the final regular-to-power transaction shares the same retained-band clock");
  const schedule = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  assert.match(schedule,
    /run\("prepareTransientPower",\[\[0,this\.params\],\[5,this\.control\],\[18,this\.indirect\]/,
    "the source-generation gate must bind the immutable committed face-band control");
});
