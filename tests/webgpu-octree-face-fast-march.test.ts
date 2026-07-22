import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  OCTREE_FACE_BAND_FACE_BYTES,
  OCTREE_FACE_BAND_CONTROL_BYTES,
  OCTREE_FACE_BAND_ROW_BYTES,
  OCTREE_FACE_BAND_STATE_BYTES,
  OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES,
  OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
  OCTREE_FACE_BAND_TRANSITION_DETAIL,
  OCTREE_FACE_BAND_TRANSITION_ERROR,
  OCTREE_FACE_BAND_OWNER_FAILURE_STAGE,
  OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES,
  OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES,
  OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES,
  OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES,
  OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES,
  OCTREE_FACE_BAND_POWER_PUBLICATION_ERROR,
  OCTREE_FACE_BAND_ENCODE_PHASES,
  WebGPUOctreeFaceFastMarch,
  classifyOctreeFaceBandBoundaryCrossing,
  makeOctreeFaceBandAirSampleWGSL,
  octreeFaceBandCoarseGenerationPairIsValid,
  octreeFaceBandMarchKeyBefore,
  octreeFaceBandWGSL,
  planOctreeFaceBandGPU,
  planOctreeFaceBandCPT,
  planOctreeFaceBandLiveSupportDispatch,
  planOctreeFaceBandMarchHeap,
  resolveOctreeFaceBandUniformSupportRequest,
  unpackOctreeFaceBandControl,
  unpackOctreeFaceBandPointFieldControl,
  unpackOctreeFaceBandPowerPublication,
  unpackOctreeFaceBandTransientPowerControl,
  unpackOctreeFaceBandTransitionControl,
} from "../lib/webgpu-octree-face-fast-march";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
  sitesForSameOrCoarserPowerDescriptor,
} from "../lib/octree-power-descriptor";
import {
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  decodeGeneratedOctreePowerCatalog,
} from "../lib/generated/octree-power-catalog";
import {
  OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
  OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW,
} from "../lib/octree-face-fast-march";
import { WebGPUFineLevelSetTransport } from "../lib/webgpu-octree-fine-levelset-transport";
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

test("Section 5 face-band phases retain paper order and fence support key/value publications", () => {
  assert.deepEqual(OCTREE_FACE_BAND_ENCODE_PHASES, [
    "topology-build", "transition-adjacency", "fast-march", "power-publication",
  ]);
  const wrapper = compact(WebGPUOctreeFaceFastMarch.prototype.encode);
  assert.match(wrapper,
    /for\(constphaseofOCTREE_FACE_BAND_ENCODE_PHASES\)this\.encodePhase\(encoder,input,phase\)/);
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const topology = source.indexOf('case"topology-build"');
  const transitions = source.indexOf('case"transition-adjacency"', topology);
  const march = source.indexOf('case"fast-march"', transitions);
  const publication = source.indexOf('case"power-publication"', march);
  assert.ok(topology >= 0 && transitions > topology && march > transitions && publication > march);
  assert.equal(source.match(/computePass\("/g)?.length, 3,
    "topology, march, and power publication remain batched checkpoints");
  assert.equal(source.match(/encoder\.beginComputePass/g)?.length, 3,
    "transition closure owns an explicit initial pass and synchronized continuation constructor");
  assert.match(source.slice(topology, transitions),
    /run\("prepare"[\s\S]*run\("map"[\s\S]*run\("indexGlobalRows"/);
  assert.match(source.slice(topology, transitions),
    /run\("map"[\s\S]*0,pass\d*,0\)/,
    "fine-row discovery must dispatch from the live GPU worklist rather than brick capacity");
  assert.match(source.slice(transitions, march),
    /run\("prepareTransition"[\s\S]*run\("resolveTransition"[\s\S]*run\("transition"[\s\S]*run\("emit"[\s\S]*run\("sampleFacePhi"[\s\S]*run\("sampleFaceCoarsePhi"[\s\S]*run\("reducePhiFailure"[\s\S]*run\("publishPhiFailure"[\s\S]*run\("summarizeRowPhi"[\s\S]*run\("gateTransition"/);
  assert.match(source.slice(transitions, march),
    /run\("insertSupport1"[\s\S]*synchronizeTransitionStorage\(\)[\s\S]*run\("recordBandPhiEdges"[\s\S]*run\("captureSupport1"[\s\S]*synchronizeTransitionStorage\(\)[\s\S]*run\("resolveSupport1Topology"/,
    "a pass boundary makes each split hash key/value publication visible before its first consumer");
  assert.match(source.slice(march, publication),
    /run\("seedCentroids"[\s\S]*run\("initialize"[\s\S]*run\("linkCpt"[\s\S]*run\("jumpCpt"[\s\S]*run\("resolveCpt"[\s\S]*run\("prepareBfs"[\s\S]*propagateBfs[\s\S]*run\("validate"[\s\S]*run\("reconstruct"[\s\S]*run\("publish"/);
  assert.match(source.slice(publication),
    /run\("preparePowerPublication"[\s\S]*run\("mapPowerFaceBands"[\s\S]*run\("interpolatePowerFaces"[\s\S]*run\("projectPowerFaces"[\s\S]*run\("publishPowerFaces"[\s\S]*run\("commitPowerFaces"/);
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
    "the bounded spatial fallback radius must cover every generated selector");
  assert.deepEqual([...selectorSizeRatios].sort((a, b) => a - b), [0.5, 1, 2],
    "the catalog fallback enumerates every dyadic owner scale");
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
  const runtimeDelaunay = wgslFunction("surroundingOwnerDelaunayVectorMeasured");
  assert.match(runtimeDelaunay,
    /for\(varcorner=0u;corner<8u[\s\S]*dot\(delta,delta\)<radius2-tolerance/,
    "the runtime path enumerates actual surrounding owners and requires an empty circumsphere");
  assert.match(runtimeDelaunay, /containingPublishedRow\(probe\)/,
    "surrounding owners are resolved by the immutable spatial directory");
  assert.match(runtimeDelaunay, /probe>=vec3i\(p\.dims\)[\s\S]*\{continue;\}/,
    "world-boundary probes retain the legacy in-domain subset");
  assert.doesNotMatch(runtimeDelaunay, /candidate<rowCount/,
    "co-spherical repair must never scan the published row arena");
  assert.match(wgslFunction("containingPublishedRow"), /r\.cell==cell\(origin\)&&r\.size==size/,
    "spatial-directory hits must match the exact queried owner origin and size");
  assert.doesNotMatch(runtimeDelaunay, /nearest|project/,
    "the exact co-spherical repair never projects to a nearest carrier");
  assert.match(runtimeDelaunay,
    /invalidPointVectorAt\(15u,vec3f\(bitcast<f32>\(row\),0\.,0\.\)\)/,
    "a missing surrounding-owner vector preserves the exact band row in the bounded failure payload");
  assert.match(wgslFunction("oldAdvectionFail"),
    /if\(atomicLoad\(&oldAdvectionControl\[15\]\)==INVALID\)\{atomicStore\(&oldAdvectionControl\[15\],detail\);\}/,
    "old-mesh diagnostics retain the first bounded retained-band failure detail");
  const prepareRepair = wgslFunction("preparePowerFaceAdvectionBandRepair");
  assert.match(prepareRepair,
    /atomicStore\(&oldAdvectionControl\[9\],atomicLoad\(&oldAdvectionControl\[0\]\)\)[\s\S]*atomicStore\(&oldAdvectionControl\[12\],atomicLoad\(&oldAdvectionControl\[6\]\)\)/,
    "the compact interpolation verdict is snapshotted before retained-band repair resets live authority");
  assert.match(prepareRepair,
    /atomicStore\(&oldAdvectionControl\[0\],0u\)[\s\S]*atomicStore\(&oldAdvectionSeed\[6\],0u\)/,
    "the paper Section 5 repair starts a new authority transaction after preserving the first-attempt snapshot");
  assert.doesNotMatch(prepareRepair, /if\(atomicLoad\(&oldAdvectionControl\[0\]\)!=0u\)\{return;\}/,
    "an expected compact-interpolant miss must not suppress the retained-band repair");
  assert.match(prepareRepair,
    /letrequested=atomicLoad\(&oldAdvectionControl\[8\]\);if\(sp\.count!=requested\|\|arrayLength\(&powerFaceControl\)<9u\|\|sp\.count!=powerFaceControl\[1\]\)\{oldAdvectionFail\(0u,1u,\(sp\.count<<8u\)\|27u\);return;\}/,
    "repair dispatch and publication must consume one exact live generalized-face count");
  const finalizeRepair = wgslFunction("finalizePowerFaceAdvectionFromBand");
  assert.match(finalizeRepair,
    /completed==requestedFaces\)\{atomicStore\(&oldAdvectionControl\[6\],VALID\);atomicStore\(&oldAdvectionSeed\[6\],VALID\);\}/,
    "a complete repaired face set publishes both advection and downstream velocity-seed authority");
  assert.match(finalizeRepair, /else\{oldAdvectionFail\(i,1u,\(i<<8u\)\|26u\);\}/,
    "an unprocessed in-range power face records its exact index instead of collapsing to incomplete reason 23");
  assert.doesNotMatch(finalizeRepair, /oldAdvectionControl\[(9|10|11|12)\]/,
    "repair publication must retain the original compact-attempt diagnostic snapshot");
  assert.match(wgslFunction("repairPowerFaceAdvectionFromBand"),
    /if\(\(face\.flags&POWER_FACE_VALID\)==0u\)\{powerFaceCentroids\[i\]\.w=bitcast<f32>\(\(1u<<8u\)\|25u\);return;\}/,
    "an invalid face inside the published count is attributable rather than silently incomplete");
  assert.match(wgslFunction("repairPowerFaceAdvectionFromBand"),
    /letpriorStatus=bitcast<u32>\(powerFaceCentroids\[i\]\.w\);if\(priorStatus==STATUS_VALID\)\{return;\}varface=/,
    "retained-band repair preserves every face already completed by direct old-mesh interpolation");
  assert.match(wgslFunction("repairPowerFaceAdvectionFromBand"),
    /powerFaceCentroids\[i\]\.w=bitcast<f32>\(STATUS_VALID\)/,
    "successful repair stores a finite scratch marker that Metal cannot normalize to positive zero");
  assert.doesNotMatch(wgslFunction("repairPowerFaceAdvectionFromBand"), /sp\.fineGeneration==2u/,
    "retained fine-band generation 2 must not zero a recurring power-mesh characteristic");
  assert.match(wgslFunction("finalizePowerFaceAdvectionFromBand"), /status==STATUS_VALID/,
    "repair publication consumes the stable finite scratch marker");
  const repairEncoder = compact(WebGPUOctreeFaceFastMarch.prototype.encodeRepairPowerFaceAdvection);
  assert.match(repairEncoder,
    /words\.set\(\[\.\.\.input\.dimensions,positive\(input\.maximumLeafSize,"Oldface-bandmaximumleaf"\),0,0,this\.plan\.rowHashCapacity,this\.plan\.rowCapacity\]\)/,
    "repair parameters do not substitute a host capacity for the live generalized-face count");
  assert.match(repairEncoder,
    /encoder\.copyBufferToBuffer\(input\.advectionControl,8\*4,this\.repairParams,5\*4,4\)/,
    "the repair dispatch consumes the exact GPU-published face-count transaction");
  assert.equal(repairEncoder.match(/encoder\.beginComputePass/g)?.length, 4,
    "cold publication plus repair prepare, transition, and finalize each own a dependency-fenced pass");
  assert.match(repairEncoder,
    /prepare\.dispatchWorkgroups\(1\);prepare\.end\(\);constrepair=.*repair\.end\(\);constfinalize=/,
    "the live count and every per-face status publish before their consumers");
  assert.match(repairEncoder, /if\(input\.powerGeneration===1\)/,
    "cold start is selected by the power-mesh epoch, not the retained fine-band epoch");
  assert.doesNotMatch(repairEncoder, /if\(input\.fineGeneration===2\)/,
    "fine generation 2 remains the valid retained old band for early recurring characteristics");
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

test("topology discovery lets duplicate producers exit and fences split key/value publication", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const insert = octreeFaceBandWGSL.slice(
    octreeFaceBandWGSL.indexOf("fn insertRow"),
    octreeFaceBandWGSL.indexOf("fn rowOf"),
  );
  assert.match(insert, /if\(result\.old_value==key\)\{return INVALID;\}/,
    "a duplicate claimant must let the winning producer run without a cross-workgroup wait");
  assert.doesNotMatch(wgslFunction("publishedRow"), /for\(|loop\{/,
    "a consumer must not spin on another workgroup's split key/value publication");
  assert.match(source.slice(source.indexOf('case"transition-adjacency"'), source.indexOf('case"fast-march"')),
    /run\("enumerateSupport1"[\s\S]*synchronizeTransitionStorage\(\)[\s\S]*run\("resolveSupportOwners"[\s\S]*synchronizeTransitionStorage\(\)[\s\S]*run\("insertSupport1"/,
    "command-ordered passes, not cross-workgroup polling, publish each support candidate stage");
  assert.doesNotMatch(insert, /\bloop\s*\{/,
    "same-key row publication must not use an unbounded cross-workgroup wait");
  assert.match(wgslFunction("prepareFaceBand"),
    /letbandBricks=fineBandBrickCount\(\)[\s\S]*indirect\[0\]=\(bandBricks\*p\.ownersPerBrick\+63u\)\/64u/,
    "the preparation dispatch must cover every published fine-band brick used by trajectory interpolation");
  assert.match(source.slice(source.indexOf('case"topology-build"'), source.indexOf('case"transition-adjacency"')),
    /run\("prepare",\[\[0,this\.params\],\[1,input\.fine\.params\],\[3,input\.fine\.worklist\],\[5,this\.control\],\[9,input\.powerVelocityControl\],\[18,this\.indirect\],\[42,input\.fineTopologyControl\]\]/,
    "the topology preparation bind group must supply the now-live fine-prefix control binding");

  const topology = source.slice(source.indexOf('case"topology-build"'),
    source.indexOf('case"transition-adjacency"'));
  assert.equal(topology.match(/encoder\.clearBuffer/g)?.length, 5,
    "topology warmup resets its three authorities plus two tiny downstream diagnostic gates");
  assert.match(topology, /clearBuffer\(this\.transitionControl\)/);
  assert.match(topology, /clearBuffer\(this\.powerPublicationControl\)/);
  assert.doesNotMatch(topology, /this\.(?:faces|incidence|state|transitionAdjacency|powerVelocityScratch)/,
    "later Section 5 capacities must not be cleared during sparse row discovery");

  const transitions = source.slice(source.indexOf('case"transition-adjacency"'),
    source.indexOf('case"fast-march"'));
  assert.doesNotMatch(transitions, /clearBuffer\(this\.faces\)/,
    "the 64-byte sparse face arena must not be cleared when all consumers use its active prefix");
  assert.match(transitions, /clearBuffer\(this\.incidence,0,this\.plan\.rowCapacity\*4\)/,
    "only per-row incidence counters are reset; payload is overwritten before use");
  assert.match(transitions,
    /run\("captureEndpoints"[\s\S]*run\("retireFaceSlots"[\s\S]*0,pass,228\)[\s\S]*run\("emit"/,
    "endpoint capture must publish the active face-slot prefix before only its LIVE words are retired");
  assert.match(wgslFunction("retireBandFaceSlots"),
    /letcount=min\(p\.faceCapacity,transitionControl\.endpointEnd\*p\.ownedFacesPerRow\)[\s\S]*faces\[g\.x\]\.flags=0u/,
    "face retirement writes one word per active slot instead of clearing every 64-byte record");
  assert.match(octreeFaceBandWGSL, /atomicStore\(&states\[i\]\.status,UNKNOWN\)/,
    "every live face receives fresh march state after the capacity-wide state clear is removed");
});

test("closed-wall ghost policy has a dedicated parameter word and preserves face strides", () => {
  const constructor = compact(WebGPUOctreeFaceFastMarch);
  const encode = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(constructor, /label:"octreeface-bandparameters",size:80/);
  assert.match(encode, /constwords=newUint32Array\(20\)/);
  assert.match(encode, /words\[16\]=47\|\(input\.closedTop\?16:0\)/);
  assert.match(octreeFaceBandWGSL,
    /axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,pad0:u32,pad1:u32,pad2:u32/,
    "the boundary mask must not alias either regular-face stride word");
});

test("factor-4 GPU face band is compact, bounded, and has no fine velocity channel", () => {
  const plan = planOctreeFaceBandGPU(100, 20, 4, 4, undefined, [24, 18, 16]);
  assert.equal(plan.ownerCandidatesPerBrick, 1);
  assert.equal(plan.coreRowCapacity, 20);
  assert.equal(plan.rowCapacity, 24 * 18 * 16,
    "deduplicated support roles share the exact logical-domain owner bound");
  assert.equal(plan.guardRowCapacity, plan.rowCapacity - plan.support0RowCapacity);
  assert.equal(plan.guardCandidateCapacity,
    plan.rowCapacity * Math.max(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
      OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW));
  assert.equal(plan.guardCandidateBytes, plan.guardCandidateCapacity * 16,
    "the four closure stages reuse one vec4u exact-owner request arena");
  assert.equal(OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW, 12);
  assert.equal(OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, 24);
  assert.equal(plan.faceCapacity, plan.rowCapacity * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW);
  assert.equal(OCTREE_FACE_BAND_ROW_BYTES, 32);
  assert.equal(OCTREE_FACE_BAND_FACE_BYTES, 64,
    "four u32 fields, two vec4f values, and four scalar fields define the face ABI");
  assert.equal(OCTREE_FACE_BAND_STATE_BYTES, 32,
    "one vec4f velocity and four u32 fields define the march-state ABI");
  assert.equal(plan.rowBytes, plan.rowCapacity * OCTREE_FACE_BAND_ROW_BYTES,
    "the allocation must match the WGSL row ABI with representative/min/max phi");
  assert.equal(plan.bandFaceBytes, plan.faceCapacity * OCTREE_FACE_BAND_FACE_BYTES);
  assert.equal(plan.stateBytes, plan.faceCapacity * OCTREE_FACE_BAND_STATE_BYTES);
  assert.equal(plan.cptParentBytes, plan.faceCapacity * 4,
    "each race-free CPT snapshot stores one parent index per face");
  assert.equal(plan.frontierBytes, 240,
    "one map record plus live support and final row/face prefix records replace capacity-sized dispatches");
  assert.equal(plan.bandFaceBytes - plan.faceCapacity * 4, 4_976_640,
    "the UI-sized domain retires LIVE words rather than rewriting 4.98 MiB of inactive face payload");
  assert.equal(plan.allocatedBytes,
    plan.bandFaceBytes + plan.incidenceBytes + plan.stateBytes + 2 * plan.cptParentBytes,
    "legacy and production accounting both include the two CPT snapshots");
  assert.match(compact(planOctreeFaceBandGPU),
    /gpuAllocatedBytes:rowBytes\+bandFaceBytes\+incidenceBytes\+stateBytes\+2\*cptParentBytes\+hashBytes/,
    "complete GPU accounting includes both CPT buffers exactly once");
  const implementation = compact(WebGPUOctreeFaceFastMarch);
  assert.doesNotMatch(implementation, /frontierA|nextFrontier|binding\(16\).*frontier|binding\(17\).*frontier/,
    "the retired serial-heap scratch and shader bindings must not be allocated");
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
      OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
    ) * OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES,
    "reused incidence scratch covers the total support-edge bound without assuming equal catalog maxima");
  assert.equal(plan.transientPowerRowBytes,
    (plan.rowCapacity + 1) * OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES);
  assert.equal(OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES, 64);
  assert.equal(plan.transientPowerFaceBytes, 13_271_040,
    "the default 24x18x16 dam-break plan accounts for the complete 12.66 MiB physical-face arena");
  const retiredArenaBytes = plan.transientPowerFaceBytes + plan.transientPowerIncidenceBytes
    + plan.transientPowerRowBytes + OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES;
  const activeFlagBytes = plan.rowCapacity
    * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence * 4;
  assert.equal(retiredArenaBytes, 15_372_368,
    "the old publication cleared 14.66 MiB of fixed-capacity arenas per advance");
  assert.equal(activeFlagBytes, 829_440,
    "even a full live prefix retires only one four-byte flag per catalog slot");
  assert.equal(retiredArenaBytes - activeFlagBytes, 14_542_928,
    "active-prefix overwrite removes at least 13.87 MiB of recurring writes at full occupancy");
  assert.equal(plan.maximumDirectWorkgroups, Math.ceil(Math.max(plan.rowCapacity, plan.faceCapacity,
    plan.guardCandidateCapacity, plan.powerFaceCapacity, plan.transientPowerFaceCapacity) / 64));
  assert.match(compact(WebGPUOctreeFaceFastMarch),
    /this\.plan\.maximumDirectWorkgroups>device\.limits\.maxComputeWorkgroupsPerDimension/,
    "a buffer-admissible plan still fails closed when its one-dimensional dispatch would exceed the adapter limit");
});

test("factor-8 B4 face-band discovery deduplicates the eight fine bricks containing one finest cell", () => {
  const plan = planOctreeFaceBandGPU(100, 160, 4, 8, 300, [16, 16, 16]);
  assert.equal(plan.ownerCandidatesPerBrick, 1);
  assert.equal(plan.powerFaceCapacity, 300);
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(source, /input\.fine\.plan\.fineFactor!==4&&input\.fine\.plan\.fineFactor!==8/);
  assert.doesNotMatch(source, /brickResolution!==input\.fine\.plan\.fineFactor/);
});

test("transition adjacency scales with every owner candidate in wider fine bricks", () => {
  const plan = planOctreeFaceBandGPU(100, 20, 8, 4, undefined, [24, 18, 16]);
  assert.equal(plan.ownerCandidatesPerBrick, 8);
  assert.equal(plan.transitionAdjacencyCapacity,
    plan.metricRowCapacity * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra,
    "B8/factor-4 support adjacency includes every S0/S1/S2 owner");
  assert.equal(plan.transitionAdjacencyBytes,
    plan.transitionAdjacencyCapacity * OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES);
});

test("Section 5 support closure is domain-deduplicated and reuses its largest exact request arena", () => {
  const wetRows = 100, maximumFineBricks = 20, brickResolution = 8, fineFactor = 4;
  const plan = planOctreeFaceBandGPU(wetRows, maximumFineBricks, brickResolution, fineFactor,
    undefined, [24, 18, 16]);
  const domainRows = 24 * 18 * 16;
  assert.equal(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows, 36,
    "the generated catalog includes co-spherical selectors that touch only at Voronoi vertices");
  assert.equal(plan.rowCapacity, domainRows,
    "overlapping tier roles deduplicate to one exact owner-origin row");
  assert.equal(plan.transitionAdjacencyCapacity,
    plan.metricRowCapacity * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra,
    "only S0/S1/S2 own catalog adjacency; terminal S3 roles do not");
  const domainCapped = planOctreeFaceBandGPU(wetRows, maximumFineBricks,
    brickResolution, fineFactor, undefined, [4, 4, 4]);
  assert.equal(domainCapped.rowCapacity, 4 * 4 * 4,
    "the optional logical-domain bound is safe because unique dyadic owner origins cannot exceed finest cells");
  assert.equal(domainCapped.guardRowCapacity, 0);
  assert.equal(domainCapped.guardCandidateCapacity,
    4 * 4 * 4 * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
    "the reused arena retains every request from the largest catalog stage");
});

test("face march heap has an exact topology-capacity pop bound and deterministic key", () => {
  assert.deepEqual(planOctreeFaceBandMarchHeap(15_479),
    { popBound: 15_479, chunkSize: 1024, chunkBound: 16 });
  assert.deepEqual(planOctreeFaceBandMarchHeap(82_944),
    { popBound: 82_944, chunkSize: 1024, chunkBound: 81 });
  assert.throws(() => planOctreeFaceBandMarchHeap(0), /positive/);
  const ordered = [
    { phi: 2, globalFace: 7, slot: 4 },
    { phi: -1, globalFace: 9, slot: 2 },
    { phi: 1, globalFace: 3, slot: 8 },
    { phi: 1, globalFace: 3, slot: 1 },
  ].sort((a, b) => octreeFaceBandMarchKeyBefore(a, b) ? -1
    : octreeFaceBandMarchKeyBefore(b, a) ? 1 : 0);
  assert.deepEqual(ordered.map(({ globalFace, slot }) => [globalFace, slot]),
    [[3, 1], [3, 8], [9, 2], [7, 4]]);
});

test("face-band control diagnostics distinguish fail-closed causes", () => {
  assert.equal(OCTREE_FACE_BAND_CONTROL_BYTES, 128);
  const words = new Uint32Array(32);
  words.set([1 | 2 | 64 | 128, 91, 120, 315, 630, 7, 0x8000_0000, 11, 8, 300, 15, 0, 4, 27, 3, 44]);
  words.set([307, 315, 307, 2, 4, 3, 1, 11], 16);
  words.set([273, 42, 88_200, 7, 19_600, 315, 315], 24);
  words[31] = 9;
  assert.deepEqual(unpackOctreeFaceBandControl(words), {
    flags: 195, firstError: 91, rowCount: 120, faceCount: 315, incidenceCount: 630,
    generation: 7, valid: false, maximumDepth: 11, seedCount: 8, acceptedCount: 300,
    unresolvedCount: 15, sampleFailures: 4, coarsePhiFallbacks: 27, coarsePhiFailures: 3,
    bandPhiExtensions: 44,
    marchHeapHighWater: 307, marchPops: 315, marchTrials: 307, marchChunks: 2,
    marchChunkBound: 4, marchCapExhausted: 3,
    marchUnresolvedWithAcceptedPredecessor: 1, marchDisconnected: 11,
    directAnchorSuccess: 273, fullRowFallbackInvocations: 42,
    fullRowCandidateRowsTested: 88_200, surroundingOwnerFallbackInvocations: 7,
    surroundingOwnerRowsTested: 19_600, airSamplesSelected: 315, airSamplesEvaluated: 315,
    connectivityFallbacks: 9,
    capacityFailure: true, hashProbeFailure: true,
    invalidSource: false, invalidRow: false, invalidFace: false, invalidPhi: false,
    unresolved: true, incompleteVector: true, outsideFineBand: false,
  });
  words.set([0, 0xffff_ffff, 120, 315, 630, 7, 0x8000_0000, 11, 8, 315, 0, 0, 4]);
  assert.equal(unpackOctreeFaceBandControl(words).valid, true);
  assert.throws(() => unpackOctreeFaceBandControl(new Uint32Array(12)), /at least 13/);
});

test("air-band evaluation is atomic-free while preserving exact fallback and boundary sampling", () => {
  const locate = wgslFunction("locateFinalPointVectorMeasured");
  assert.match(locate, /candidateRowsTested\+=1u/,
    "each bounded local catalog candidate is counted");
  assert.match(locate, /for\(vardz=-2i;dz<=2i[\s\S]*for\(vardy=-2i;dy<=2i[\s\S]*for\(vardx=-2i;dx<=2i/,
    "catalog candidates cover the proven five-cubed selector-radius box");
  assert.match(locate, /if\(candidate>=bestRow\)\{continue;\}[\s\S]*bestRow=candidate;bestValue=value/,
    "spatial enumeration preserves the old ascending-row result");
  assert.doesNotMatch(locate, /candidate<count|min\(p\.rowCapacity/,
    "the recurring sampler must not do work proportional to allocated row capacity");
  assert.match(locate, /surroundingOwnerDelaunayVectorMeasured\(pointGrid\)/);
  assert.match(wgslFunction("locateFinalPointVector"),
    /returnlocateFinalPointVectorMeasured\(initialAnchor,pointGrid\)\.value/,
    "power publication and retained repair preserve the existing pure sampler wrapper");
  const hot = makeOctreeFaceBandAirSampleWGSL();
  assert.doesNotMatch(hot, /\batomic(?:Add|CompareExchangeWeak|Load|Max|Min|Or|Store)\b/,
    "the recurring sampler has no atomic operation");
  assert.match(hot, /@binding\(7\)var<storage,read>rowHash:array<u32>/,
    "the cold-built row hash is immutable during recurring transport");
  assert.match(hot, /@binding\(23\)var<storage,read_write>sampleStatus:array<u32>/,
    "exclusive per-query statuses use ordinary stores");
  assert.match(wgslFunction("finalPointVector", hot), /finalSignedVector\(cornerOrigin,row\.size\)/,
    "ordinary same-size cube interpolation performs the exact immutable row-hash lookup");
  assert.match(wgslFunction("finalTetraPointVector", hot), /finalSelectorVector\(anchor,selectors\.[xyz]/,
    "catalog interpolation preserves the exact selector-to-owner lookup");
  const classify = wgslFunction("classifyAirBandVelocity", hot);
  assert.match(classify, /sampleStatus\[i\]=SAMPLE_EVALUATE/);
  const boundary = wgslFunction("airSampleGrid", hot);
  assert.match(boundary, /p\.closedBoundaryMask&negativeBoundaryBit\(axis\).*grid\[axis\]=0\./,
    "Stage-B velocity constantly extends through closed negative container walls");
  assert.match(boundary, /p\.closedBoundaryMask&positiveBoundaryBit\(axis\).*openCeiling=axis==1u&&!closed.*grid\[axis\]=f32\(sp\.dims\[axis\]\)-1e-5/,
    "Stage-B velocity extends through closed positive walls and the authored open ceiling");
  const evaluate = wgslFunction("evaluateAirBandVelocity", hot);
  assert.match(evaluate, /letsample=airSampleGrid\(positions\[i\]\.xyz\)/,
    "classification and evaluation consume one identical boundary-adjusted point");
  assert.match(evaluate, /locateFinalPointVectorMeasured\(band,sample\.xyz\)\.value/);
  assert.doesNotMatch(evaluate, /\bcontrol\b/,
    "the 10-buffer evaluator must not make face-band control reachable");
  const encode = compact(WebGPUOctreeFaceFastMarch.prototype.encodeAirSamples);
  assert.doesNotMatch(encode, /sampleAggregatePipeline|aggregateEntries|clearBuffer\(statuses/,
    "the recurring sampler has no serialized diagnostic fold or counter tail");
});

test("Dawn compiles every atomic-free recurring air-sampler entry at the ten-storage limit", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for recurring air-sampler checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const module = device.createShaderModule({ code: makeOctreeFaceBandAirSampleWGSL() });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of ["classifyAirBandVelocity", "evaluateAirBandVelocity", "finalizeAirBandVelocity"]) {
    device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
  }
  assert.equal(await device.popErrorScope(), null);
  const storage = (words: Uint32Array<ArrayBuffer>, extraUsage = 0) => {
    const buffer = device.createBuffer({ size: Math.max(4, words.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage });
    device.queue.writeBuffer(buffer, 0, words);
    return buffer;
  };
  const controlWords = new Uint32Array(32); controlWords[5] = 7; controlWords[6] = 0x8000_0000;
  const pointWords = new Uint32Array(8); pointWords[3] = 7; pointWords[5] = 0x8000_0000;
  const sampleWords = new Uint32Array(12); sampleWords[5] = 4; sampleWords[9] = 7;
  const statuses = storage(new Uint32Array([0x0200_0000, 0x0100_0123, 0x8000_0000, 0x0400_0000]),
    GPUBufferUsage.COPY_SRC);
  const control = storage(controlWords), point = storage(pointWords), params = device.createBuffer({ size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, sampleWords);
  const finalize = device.createComputePipeline({ layout: "auto", compute: { module,
    entryPoint: "finalizeAirBandVelocity" } });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(finalize); pass.setBindGroup(0, device.createBindGroup({
    layout: finalize.getBindGroupLayout(0), entries: [
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
  for (const buffer of [statuses, control, point, params, readback]) buffer.destroy();
  device.destroy();
});

test("uniform catalog support still honors every Section 6.1 same/coarser owner", () => {
  assert.deepEqual(resolveOctreeFaceBandUniformSupportRequest(
    [15, 11, 3], 1, 16, 0x8000_0087), { origin: [16, 10, 4], size: 2 },
  "the production payload's +x,+z edge request resolves to its parity-aligned coarse owner");

  for (let low = 0; low < 512; low += 1) {
    const descriptor = (OCTREE_POWER_SAME_OR_COARSER_FLAG | low) >>> 0;
    const sites = sitesForSameOrCoarserPowerDescriptor(descriptor);
    const anchor = sites.find((site) => site.key === "anchor")!;
    for (let request = 0; request < 18; request += 1) {
      const direction = OCTREE_POWER_NEIGHBOR_DIRECTIONS[request];
      const probe = anchor.origin.map((value, axis) => value + direction[axis]);
      const expected = sites.find((site) => site.key !== anchor.key && site.origin.every((value, axis) =>
        probe[axis] >= value && probe[axis] < value + site.size));
      assert.ok(expected, `descriptor ${low}, request ${request}`);
      assert.deepEqual(resolveOctreeFaceBandUniformSupportRequest(
        anchor.origin, anchor.size, request, descriptor),
      { origin: expected.origin, size: expected.size }, `descriptor ${low}, request ${request}`);
    }
  }

  const resolve = wgslFunction("resolveSupportOwners");
  assert.match(resolve, /letdescriptor=metric\.reserved/,
    "the uniform fast path retains the raw Section 6.1 descriptor");
  assert.match(resolve,
    /coarseDirections=array<vec3i,6>.*descriptor&\(1u<<\(coarse\+3u\)\).*coarseSize=row\.size\*2u/,
    "parity-allowed coarse face/edge bits select a 2x owner request");
  assert.match(resolve, /letcoarseOrigin=\(coarseProbe\/i32\(coarseSize\)\)\*i32\(coarseSize\)/,
    "the requested coarse row is its exact dyadic origin");
});

test("face phi uses the paper's fine field then the redistanced dry-owner cube/Delaunay field", () => {
  const sample = wgslFunction("sampleBandFacePhi");
  const coarseSample = wgslFunction("sampleBandFaceCoarsePhi");
  assert.match(sample, /finePhiAtFaceCentroid\(face\.centroid\.xyz\)/,
    "valid fine phi has priority");
  assert.match(coarseSample,
    /\(face\.flags&\(LIVE\|PHI_VALID\)\)!=LIVE.*coarsePhiAtPoint\(face\.negativeRow,face\.centroid\.xyz\).*face\.positiveRow<transitionControl\.support2End.*coarsePhiAtPoint\(face\.positiveRow,face\.centroid\.xyz\)/,
    "only unresolved faces fall through, and only fully closed S0-S2 rows may anchor local interpolation");
  assert.match(coarseSample, /atomicAdd\(&control\.coarsePhiFallbacks,1u\)/);
  assert.match(coarseSample, /atomicAdd\(&control\.coarsePhiFailures,1u\);fail\(OUTSIDE_FINE_BAND,face\.globalFace\)/,
    "missing fine and coarse authority remains publication-fatal and visible");

  const exactCoarseCell = wgslFunction("exactCoarseCellScalar");
  assert.match(exactCoarseCell, /validCoarse\(\).*coarseSlot\(cell\(origin\),size\)/,
    "redistance seeds read only the current exact compact octree owner record");
  const coarseEntry = wgslFunction("coarseEntryRecord");
  assert.match(coarseEntry, /\(entry\.flags&9u\)!=9u.*entry\.minimumPhi>entry\.phi\|\|entry\.phi>entry\.maximumPhi/,
    "malformed or stale coarse signed distance is rejected");
  assert.doesNotMatch(exactCoarseCell, /ROW_COARSE_AIR|select\([^)]*,[^)]*,entry/,
    "exact seeds cannot synthesize a sign or capped distance");
  const coarseSeed = wgslFunction("coarseCellSeedRecord");
  assert.match(coarseSeed,
    /letexact=coarseEntryRecord\(coarseSlot\(cell\(origin\),size\)\).*letq=min\(origin\+vec3u\(size\/2u\),p\.dims-vec3u\(1u\)\).*coarseSlot\(cell\(priorOrigin\),scale\)/s,
    "a changed leaf samples the same-time spatial coarse publication at its centre");
  assert.match(coarseSeed, /if\(scale>=coarsePhi\.maximumLeafSize\)\{break;\}scale\*=2u/,
    "spatial migration is bounded by the published octree hierarchy");
  const coarseCell = wgslFunction("coarseCellScalar");
  assert.match(coarseCell, /rowOf\(cell\(origin\)\).*row\.flags&ROW_PHI.*row\.representativePhi/s,
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

  const transitionPhase = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const fineBindings = transitionPhase.match(/run\("sampleFacePhi",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  const coarseBindings = transitionPhase.match(/run\("sampleFaceCoarsePhi",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  assert.ok(fineBindings && coarseBindings);
  for (const binding of [0, 1, 2, 5, 8, 12, 24, 51]) {
    assert.match(fineBindings, new RegExp(`\\[${binding},`));
  }
  for (const binding of [0, 5, 6, 7, 12, 27, 28, 29, 30, 31]) {
    assert.match(coarseBindings, new RegExp(`\\[${binding},`));
  }
  for (const binding of [1, 25, 42]) assert.doesNotMatch(coarseBindings, new RegExp(`\\[${binding},`));
});

test("cube and Delaunay interpolants load only vertices that geometrically contribute", () => {
  for (const [functionName, selectorLoad] of [
    ["coarsePhiAtPoint", "localTetraBandScalar(anchor,local,0u,selectors.x"],
    ["centroidVector", "selectorVector(anchor,selectors.x"],
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
    ["centroidVector", "compactSignedVector"],
    ["marchedCentroidVector", "provisionalSignedVector"],
    ["finalPointVector", "finalSignedVector"],
  ] as const) {
    assert.match(wgslFunction(functionName), new RegExp(
      `letweight=[^;]+;if\\(weight==0\\.\\)\\{continue;\\}letcornerOrigin=[^;]+;let(?:v|value)=${cornerLoader}`),
    `${functionName} must not require a zero-weight trilinear corner`);
  }
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
  assert.throws(() => unpackOctreeFaceBandTransientPowerControl(new Uint32Array(8)), /at least nine/);
});

test("transition diagnostics preserve the exact pre-emission catalog failure", () => {
  const words = new Uint32Array([
    OCTREE_FACE_BAND_TRANSITION_ERROR.invalidBandDescriptor,
    17, 3582, 211, 917, 0, 0,
    OCTREE_FACE_BAND_TRANSITION_DETAIL.aboveDomain | OCTREE_FACE_BAND_TRANSITION_DETAIL.missingBandRow,
  ]);
  assert.deepEqual(unpackOctreeFaceBandTransitionControl(words), {
    flags: 8, firstError: 17, rowCount: 3582, transitionRows: 211, adjacencyCount: 917,
    ready: false, transferReady: false, invalidSource: false, capacityFailure: false,
    unresolvedAdjacency: false, invalidBandDescriptor: true, acuteGrading: false,
    detailFlags: 36, malformedGeometry: false, belowDomain: false, aboveDomain: true,
    misalignedGeometry: false, ownerMismatch: false, missingBandRow: true,
    rowOutOfRange: false, ownerSizeMismatch: false,
    coreRowCount: 3582, support1RowCount: 3582, support2RowCount: 3582,
    support3NodeRowCount: 3582, endpointRowCount: 3582, boundaryGhostRequests: 0,
    phiFailureCounts: { missingRow: 0, exactCoarseMiss: 0, invalidMetric: 0, invalidSelector: 0 },
  });
  words.set([0, 0xffff_ffff, 3582, 211, 917, 0x8000_0000, 0x8000_0000, 0]);
  assert.equal(unpackOctreeFaceBandTransitionControl(words).ready, true);
  assert.equal(unpackOctreeFaceBandTransitionControl(words).transferReady, true);
  assert.throws(() => unpackOctreeFaceBandTransitionControl(new Uint32Array(7)), /eight/);
});

test("transition diagnostics decode legacy acute captures but producer admits every catalog mask", () => {
  const words = new Uint32Array(32);
  const descriptor = (OCTREE_POWER_SAME_OR_COARSER_FLAG | (25 << 3) | 3) >>> 0;
  words[0] = OCTREE_FACE_BAND_TRANSITION_ERROR.acuteGrading;
  words[1] = 73;
  words.set([
    73, OCTREE_FACE_BAND_OWNER_FAILURE_STAGE.acuteGrading, 9123, 2,
    descriptor, 0xffff_ffff, 0, 25,
  ], 16);
  const decoded = unpackOctreeFaceBandTransitionControl(words);
  assert.equal(decoded.acuteGrading, true);
  assert.equal(decoded.invalidBandDescriptor, false);
  assert.equal(decoded.ownerFailure, undefined);
  assert.deepEqual(decoded.acuteGradingFailure, {
    band: 73, rowCell: 9123, rowSize: 2, descriptor, coarseMask: 25,
  });

  assert.match(wgslFunction("strictlyObtuseCoarseMask"), /returnfalse/,
    "co-spherical vertex-only sites make every graded same\/coarser mask catalog-valid");
  assert.doesNotMatch(octreeFaceBandWGSL, /mask==25u\|\|mask==42u/,
    "the dry-band producer must not carry the obsolete six-mask exclusion");
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

test("GPU face band closes endpoints before deterministic parallel CPT", () => {
  assert.doesNotMatch(octreeFaceBandWGSL, /\b(?:active|target|global)\b/,
    "Dawn/WebGPU reserve active, target, and global as future keywords");
  assert.match(octreeFaceBandWGSL,
    /let neighbor=rowOf\(cell\(neighborOwner\.origin\)\);if\(neighbor==INVALID\|\|neighbor>=transitionControl\.endpointEnd/,
    "a required 2:1 subface fails when its exact endpoint support row is absent");
  assert.match(octreeFaceBandWGSL, /let encoded=atomicLoad\(&rowHash\[slot\*2u\+1u\]\);return select\(INVALID,encoded-1u,encoded!=0u&&encoded!=INVALID\)/,
    "row-hash values use row+1 publication and subtract exactly once");
  assert.match(compact(wgslFunction("insertRow")),
    /for\(varprobe=0u;probe<32u;probe\+=1u\).*?for\(varretry=0u;retry<32u;retry\+=1u\).*?atomicCompareExchangeWeak.*?if\(result\.old_value==0u\)\{continue;\}.*?if\(occupied==0u\)\{fail\(HASH,cellKey\);returnINVALID;\}/,
    "a spurious weak-CAS failure must retry the same slot or fail closed instead of leaving a lookup-terminating hole");
  assert.match(octreeFaceBandWGSL,
    /ap<\(\*bestPhi\)\|\|\(ap==\(\*bestPhi\)&&\(globalFace<\(\*bestGlobal\)/,
    "fallback resolution uses closest-|phi| then stable-face-ID tie breaking");
  assert.match(octreeFaceBandWGSL, /atomicLoad\(&control\.unresolvedCount\)==0u/);
  assert.match(octreeFaceBandWGSL, /storeSampleStatus\(i,VALID\|EXTRAPOLATED\)/,
    "air-side Stage-B publication is explicitly marked extrapolated");
  assert.match(octreeFaceBandWGSL, /binding\(10\)var<storage,read>powerRowVelocities:array<vec4f>/,
    "regular face seeds consume Stage-A least-squares row vectors");
  assert.match(octreeFaceBandWGSL,
    /powerVelocityControl\[0\]==VALID&&powerVelocityControl\[2\]<=p\.powerRowCapacity&&powerVelocityControl\[7\]==p\.powerGeneration/,
    "stale or unpublished Stage-A vectors must fail the complete face-band generation");
  assert.doesNotMatch(octreeFaceBandWGSL, /AxisFace|axisFaces|axisIncidence/,
    "regular face seeds must not bypass Stage A through packed exact-axis faces");
  assert.match(octreeFaceBandWGSL, /!finite\(f\.phi\)\)\{return;\}if\(f\.phi>0\.\)\{f\.pad=1u/,
    "seed membership uses the sampled face-centroid signed distance itself");
  assert.match(octreeFaceBandWGSL, /centroidVector\(f\.negativeRow,f\.centroid\.xyz\)/,
    "regular-face seeds evaluate the full Stage-B vector at the actual face centroid");
  assert.match(octreeFaceBandWGSL,
    /let stageBStatus=loadSampleStatus\(i\);let stageBReason=stageBStatus&255u;let needsDualCompletion=\(stageBStatus&VALID\)==0u&&\(stageBReason==4u\|\|stageBReason==8u\);if\(rows\[band\]\.minimumPhi<0\.&&!needsDualCompletion\)\{return;\}/,
    "Section 5 completes true Stage-B failures without mistaking valid tetrahedron indices 4 or 8 for failure reasons");
  assert.match(octreeFaceBandWGSL,
    /let owner=ownerAt\(q\);if\(owner\.valid==0u\)[\s\S]*let band=retainedBandAnchor\(grid\)/,
    "air sampling must validate adaptive ownership before resolving the containing regular-band row");
  assert.match(octreeFaceBandWGSL,
    /band>=transitionControl\.support7NodeEnd/,
    "air sampling must remain inside the published non-endpoint support band");
  assert.match(octreeFaceBandWGSL, /control\.generation!=sp\.fineGeneration/,
    "the sampled regular-face band must match the transported fine generation");
  assert.match(octreeFaceBandWGSL,
    /band==INVALID[^}]+storeSampleStatus\(i,SAMPLE_FAILED/,
    "a missing classification row is a structural failure, not proof of liquid");
  assert.match(octreeFaceBandWGSL,
    /control\.generation!=sp\.fineGeneration[\s\S]*minimumPhi<0\.&&!needsDualCompletion[\s\S]*storeSampleStatus\(i,SAMPLE_EVALUATE\)/,
    "only an exact current liquid row may preserve Stage B; air and uncovered dual octants require published completion");
  assert.doesNotMatch(octreeFaceBandWGSL, /sampleBandRow\(vec3u\(floor\(grid\)\)/,
    "queries inside coarse owners must not use a non-origin finest-cell key");
});

test("fine bricks bound work while coarse phi remains sign-only topology metadata", () => {
  const reconstruct = wgslFunction("reconstructBandRowVelocity");
  assert.doesNotMatch(octreeFaceBandWGSL, /guardPhi/,
    "storage guard pages must never invent signed-distance authority");
  assert.doesNotMatch(octreeFaceBandWGSL, /coarseSummary|physicalCellSize\*f32\(max/,
    "a directory miss must not synthesize a signed-distance magnitude");
  assert.match(wgslFunction("coarseSignFlag"),
    /letentry=coarseCellSeedRecord\(owner\.origin,owner\.size\);if\(entry\.w==0\.\)\{returnROW_COARSE_MIXED;\}/,
    "a spatial coarse-directory miss remains unsigned until neighboring-cell redistance resolves it");
  assert.match(wgslFunction("extendBandRowPhi"), /if\(sign==0\.\).*nearestSignedNeighbor/s,
    "narrow-band closure propagates a measured neighboring sign instead of fabricating air");
  assert.match(wgslFunction("nearestSignedNeighbor"),
    /pointStatus\[rowIndex\].*transientPowerIncidences\[edge\].*letparent=item\.face.*source\.y!=0\..*signedBest=select\(-candidate,candidate,source\.x>=0\.\)/s,
    "mixed S3-S6 closure rows obtain their measured sign through the same recorded parent links as one-sided phi propagation");
  assert.match(octreeFaceBandWGSL,
    /let owner=ownerAt\(q\);if\(owner\.valid==0u\)\{fail\(SOURCE,cell\(q\)\);return;\}/,
    "missing owner topology must fail closed");
  assert.match(wgslFunction("ownerAt"),
    /!found\|\|encoded==0u\|\|encoded==INVALID[\s\S]*returninvalidOwner/,
    "missing paged owners must not be synthesized as coarse topology");
  assert.match(wgslFunction("ownerAt"), /if\(word==0u\)\{returnresidentCanonicalOwner\(q\);\}/,
    "a zero payload word in a resident page is the topology's canonical coarse owner, not a missing owner");
  assert.match(wgslFunction("ownerAt"), /owners\[7\]==0u\|\|owners\[7\]!=p\.powerGeneration/,
    "paged owner lookup must consume the exact current topology generation");
  assert.match(wgslFunction("ownerAt"), /!found\|\|encoded==0u\|\|encoded==INVALID/,
    "missing, reserved, and unpublished owner pages are structural failures");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /if\(owner\.size!=1u\)\{fail\(BAD_ROW/,
    "transition owners must reach the catalog producer instead of being rejected during discovery");
  assert.match(octreeFaceBandWGSL,
    /insertRow\(ownerCell,globalRow,ROW_COARSE\|ROW_CORE\|signFlag,owner\.size\)/,
    "row identity and seed sign remain separate from later fine signed-distance publication");
  assert.match(reconstruct,
    /abs\(negativeArea\[axis\]-targetArea\)>tolerance\|\|abs\(positiveArea\[axis\]-targetArea\)>tolerance/,
    "every provisional component requires complete accepted regular-face area on both sides");
  assert.match(reconstruct,
    /letclosure=\(r\.flags&ROW_SUPPORT3_NODE\)!=0u/,
    "only the terminal S3 interpolation-support node tier may close a missing graph side");
  assert.match(reconstruct,
    /closure&&negativeArea\[axis\]==0\.&&positiveArea\[axis\]>=targetArea[\s\S]*negativeSum\[axis\]=positiveSum\[axis\]/,
    "S3 closure copies an actual complete accepted carrier side, never a synthetic scalar");
  assert.match(reconstruct,
    /closure&&positiveArea\[axis\]==0\.&&negativeArea\[axis\]>=targetArea[\s\S]*positiveSum\[axis\]=negativeSum\[axis\]/,
    "the symmetric S3 closure is likewise limited to a wholly absent side");
  assert.match(reconstruct,
    /else\{if\(closure\)\{return;\}fail\(INCOMPLETE,row\);return;\}/,
    "an unused incomplete S3 support row remains unpublished, while every S0-S2 target still fails globally");
  assert.doesNotMatch(reconstruct, /ROW_SUPPORT3_ENDPOINT/,
    "terminal endpoint rows are not vector targets and cannot weaken reconstruction");
});

test("paper Section 5 orders LIVE regular faces by the current two-resolution phi at their actual centroids", () => {
  const emit = wgslFunction("emitBandFaces");
  assert.doesNotMatch(emit, /representativePhi|minimumPhi|maximumPhi|facePhi/,
    "face emission must never select an endpoint or coarse row distance");
  assert.match(octreeFaceBandWGSL, /const PHI_VALID:u32=4u/,
    "fine-distance authority must use an explicit flag separate from the scalar channel");
  assert.match(emit, /Face\([^;]+,0\.,[^;]+,LIVE,0u\)/,
    "an emitted face starts with a representable scalar but no fine-distance authority");
  assert.doesNotMatch(octreeFaceBandWGSL, /0x7fc00000u/,
    "face-band WGSL must not materialize a NaN constant rejected by Dawn/Tint");
  assert.match(emit, /band>=transitionControl\.support2End/,
    "only metric-resolved S0-S2 rows may own marched LIVE faces");
  assert.doesNotMatch(emit, /endpointOnly|reconstructable/,
    "deep support rows are interpolation closure, not a second face graph");

  const sampler = wgslFunction("finePhiAtFaceCentroid");
  assert.match(sampler,
    /letworld=fp\.domainOrigin\+pointGrid\*coarseWidth;letraw=\(world-fp\.domainOrigin\)\/fp\.fineWidth-vec3f\(\.5\)/,
    "the fine lattice query is formed from the actual regular-face centroid in world space");
  assert.match(sampler, /if\(weight==0\.\)\{continue;\}.*loadFineScalarExtended/s,
    "every nonzero trilinear product weight requires an authoritative fine sample");
  assert.match(sampler, /fp\.generation!=p\.generation/,
    "face phi never admits a stale or rollback-adjacent generation");
  const load = wgslFunction("loadFineScalarExtended");
  assert.match(load, /q\[axis\]!=-1.*q\[axis\]=0/s,
    "the lower world plane permits exactly one virtual fine-center layer and mirrors it evenly");
  assert.match(load, /q\[axis\]!=limit.*q\[axis\]=limit-1/s,
    "the upper world plane permits exactly one virtual fine-center layer and mirrors it evenly");
  const lookup = wgslFunction("finePage");
  assert.match(lookup,
    /metadata\[base\]!=id\|\|metadata\[base\+1u\]!=key\|\|metadata\[base\+2u\]!=fp\.generation/,
    "the hash hit, page identity, key, and generation must all agree");
  assert.match(load, /\(sampleFlags\[index\]&1u\)==0u/);

  const sampleKernel = wgslFunction("sampleBandFacePhi");
  const coarseSampleKernel = wgslFunction("sampleBandFaceCoarsePhi");
  assert.match(sampleKernel, /finePhiAtFaceCentroid\(face\.centroid\.xyz\)/,
    "fine phi is always attempted first");
  assert.match(coarseSampleKernel,
    /coarsePhiAtPoint\(face\.negativeRow,face\.centroid\.xyz\).*face\.positiveRow<transitionControl\.support2End.*coarsePhiAtPoint\(face\.positiveRow,face\.centroid\.xyz\)/,
    "fine phi is preferred and only the fully closed S0-S2 octree field supplies the outside-band interpolant");
  assert.match(coarseSampleKernel,
    /phiRecord\.cause==INVALID&&face\.positiveRow<transitionControl\.support2End/,
    "an endpoint-only row can neither evaluate nor diagnose a Delaunay phi anchor without a catalog one-ring");
  assert.match(coarseSampleKernel, /sampled\.y==0\.[\s\S]*fail\(OUTSIDE_FINE_BAND,face\.globalFace\)/,
    "missing, stale, invalid, or non-finite fine and coarse support fails the whole publication");
  assert.match(coarseSampleKernel,
    /diagnoseCoarsePhiAtPoint\(face\.negativeRow,face\.centroid\.xyz\).*PHI_DIAGNOSTIC/s,
    "a failed face retains bounded first-failure evidence without weakening the publication gate");
  assert.match(wgslFunction("reduceBandPhiFailure"),
    /atomicAdd\(&transitionControl\.phiFailureCounts,1u<<\(cause\*8u\)\).*atomicMin\(&transitionControl\.failureBand,PHI_FAILURE_TAG\|index\)/s,
    "failed faces are counted by cause and select one deterministic minimum face index");
  assert.match(wgslFunction("publishBandPhiFailure"),
    /failureStage=index.*failureRowCell=face\.globalFace.*failureOwnerSizeValid=cause\|\(detail<<8u\)/s,
    "the selected face publishes its exact interpolation record into existing readback words");
  assert.match(sampleKernel, /face\.phi=sampled\.x/,
    "the physical signed distance is retained unchanged for closest-face ordering");
  assert.match(sampleKernel, /face\.flags\|=PHI_VALID/,
    "distance authority is published only after an exact current centroid sample succeeds");
  const summary = wgslFunction("summarizeBandRowPhi");
  assert.match(summary, /row>=transitionControl\.support2End/,
    "only metric-resolved S0-S2 rows publish current-phi summaries; S3 closure rows never do");
  assert.match(summary,
    /minimum=min\(minimum,face\.phi\);maximum=max\(maximum,face\.phi\)/,
    "row min/max are reductions of real sampled incident face positions");
  assert.match(summary, /rows\[row\]\.flags\|=ROW_PHI/,
    "row distance authority is published only after every incident face validates");
  assert.match(summary, /face\.flags&\(LIVE\|PHI_VALID\)/,
    "row summaries reject every face without explicit current-distance authority");
  assert.match(wgslFunction("seedFaceCentroids"), /f\.flags&\(LIVE\|PHI_VALID\)/,
    "fast-march seeds require explicit current-distance authority");
  assert.match(wgslFunction("initializeFaceMarch"), /f\.flags&PHI_VALID/,
    "fast marching fails closed if a LIVE face was not sampled");
  assert.match(wgslFunction("initializeFaceMarch"), /states\[i\]\.pad=INVALID/,
    "every non-seed starts without a stale directed discovery offer");
  assert.match(wgslFunction("initializeFaceMarch"),
    /f\.flags&SEED.*states\[i\]\.parent=i.*atomicStore\(&states\[i\]\.status,ACCEPTED\)/s,
    "parallel initialization publishes each wet carrier as a CPT root");
  assert.match(wgslFunction("linkFaceClosestPoints"),
    /closestPredecessorTopology\(face\.negativeRow,faceIndex,&best\).*closestPredecessorTopology\(face\.positiveRow,faceIndex,&best\).*states\[faceIndex\]\.parent=best.*status,TRIAL/s,
    "every air face independently links to its deterministic decreasing-phi predecessor");
  assert.match(wgslFunction("faceHeapBefore"),
    /ap=faceArrival\(af\.phi\).*bp=faceArrival\(bf\.phi\).*ap<bp.*ap>bp.*af\.globalFace<bf\.globalFace.*af\.globalFace>bf\.globalFace.*a<b/s,
    "the CPT predecessor order is deterministic in liquid-zero/air-phi arrival, stable face id, then slot");
  assert.match(wgslFunction("jumpFaceClosestPoints"),
    /parent=cptParentInput\[faceIndex\].*ancestor=cptParentInput\[parent\].*cptParentOutput\[faceIndex\]=select\(parent,ancestor/s,
    "parallel pointer jumping reads an immutable predecessor snapshot and writes the other buffer");
  assert.doesNotMatch(wgslFunction("jumpFaceClosestPoints"), /states\[.*\]\.parent=/,
    "one jump dispatch never mutates the parent graph another lane is reading");
  assert.match(wgslFunction("resolveFaceClosestPoints"),
    /states\[root\]\.status\)!=ACCEPTED.*faces\[root\]\.flags&SEED.*states\[faceIndex\]\.velocity=states\[root\]\.velocity/s,
    "only a validated wet CPT root can publish an extrapolated carrier");
  const recorded = wgslFunction("recordedAcceptedPredecessor");
  assert.match(recorded, /sourceFace=states\[targetFace\]\.pad/);
  assert.match(recorded, /states\[sourceFace\]\.status\)!=ACCEPTED/);
  assert.match(recorded, /faceArrival\(sourceRecord\.phi\)>faceArrival\(targetRecord\.phi\)\+1e-6/,
    "the retained asymmetric offer is revalidated as accepted and causal at pop time");
  assert.match(wgslFunction("faceArrival"), /max\(phi,0\.\)/,
    "all liquid interpolation faces are zero-arrival boundary data even when their centroid is farther from the interface than the first air face");
  const consider = wgslFunction("consider");
  assert.match(consider, /targetRecord\.flags&\(LIVE\|PHI_VALID\)/);
  assert.match(consider, /accepted\.flags&\(LIVE\|PHI_VALID\)/,
    "closest-face ordering compares only explicitly sampled target and accepted faces");
  assert.match(wgslFunction("validateFaceMarch"),
    /status==TRIAL.*marchCapExhausted.*recordedAcceptedPredecessor\(i\)!=INVALID\|\|acceptedFacePredecessor\(i\)!=INVALID.*marchUnresolvedWithPredecessor.*marchDisconnected/s,
    "post-bound validation attributes both directed offers and target-local neighbors before declaring disconnection");

  const schedule = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(schedule,
    /run\("sampleFacePhi",\[\[0,this\.params\],\[1,input\.fine\.params\],\[2,input\.fine\.metadata\],\[5,this\.control\],\[8,input\.fine\.flags\],\[12,this\.faces\],\[24,input\.fine\.phi\],\[51,input\.fine\.hash\]\]/,
    "the fine sampler retains its bounded sparse-page layout");
  assert.match(schedule,
    /run\("sampleFaceCoarsePhi",\[\[0,this\.params\],\[5,this\.control\],\[6,this\.rows\],\[7,this\.rowHash\],\[12,this\.faces\],\[27,this\.transitionMetrics\],\[28,tetrahedronHeaders\],\[29,tetrahedra\],\[30,tetrahedronVertices\],\[31,this\.transitionAdjacency\],\[32,this\.transitionControl\]\]/,
    "the ordered coarse sampler consumes committed row phi, validated adjacency, and the S0-S2 prefix gate");
  assert.match(schedule,
    /run\("commitBandPhi",\[\[5,this\.control\],\[6,this\.rows\],\[19,currentPhi\],\[32,this\.transitionControl\]\]/,
    "the row-field commit binds its exact scalar-publication prefix and output globals");
  assert.match(wgslFunction("commitBandRowPhi"), /rowIndex>=transitionControl\.endpointEnd/,
    "terminal regular-face endpoints publish phi only after their exact closure edges redistance");
  assert.match(wgslFunction("recordBandPhiEndpointEdges"),
    /base%MAX_ENDPOINTS!=0u.*recordBandPhiEdgeGroup\(base,MAX_ENDPOINTS\)/,
    "endpoint parent capture uses the distinct 24-request regular-face stride");
  assert.match(wgslFunction("extendBandRowPhi"),
    /lowerSimplexCandidate\(rowIndex,sign\).*localTetraEikonal\(rowIndex,sign\)/s,
    "terminal support edges remain causal lower-dimensional simplices while transition rows use local tetrahedra");
  assert.match(wgslFunction("lowerSimplexCandidate"),
    /edge=atomicLoad\(&pointStatus\[rowIndex\]\).*best=min\(best,abs\(source\.x\)\+length\(center-bandCenter\(parent\)\)\*unit\).*edge=bitcast<u32>\(item\.sign\)/s,
    "support-only nodes traverse every exact closure-parent edge");
  const emitAt = schedule.indexOf('run("emit"');
  const sampleAt = schedule.indexOf('run("sampleFacePhi"', emitAt);
  const initializePhiAt = schedule.indexOf('run("initializeBandPhi"', sampleAt);
  const seedPhiAt = schedule.indexOf('run("seedBandPhiFaces"', initializePhiAt);
  const extendPhiAt = schedule.indexOf('run("extendBandPhi"', seedPhiAt);
  const commitPhiAt = schedule.indexOf('run("commitBandPhi"', extendPhiAt);
  const coarseSampleAt = schedule.indexOf('run("sampleFaceCoarsePhi"', commitPhiAt);
  const summaryAt = schedule.indexOf('run("summarizeRowPhi"', coarseSampleAt);
  const gateAt = schedule.indexOf('run("gateTransition"', summaryAt);
  assert.ok(emitAt >= 0 && sampleAt > emitAt && initializePhiAt > sampleAt
    && seedPhiAt > initializePhiAt && extendPhiAt > seedPhiAt && commitPhiAt > extendPhiAt
    && coarseSampleAt > commitPhiAt
    && summaryAt > coarseSampleAt && gateAt > summaryAt,
    "face topology, current phi, row summaries, then transaction gate is the only publication order");
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
  assert.match(sample, /if\(\(p\.closedBoundaryMask&plane\)==0u\)/,
    "closed and open planes are applied independently per physical world face");
  assert.match(sample, /plane!=positiveBoundaryBit\(1u\)/,
    "padf/open extension is authored only for the dam-break +y world plane");
  assert.doesNotMatch(octreeFaceBandWGSL, /accumulateBandPowerLSBatch|reflectedBoundaryPoint/,
    "the final field has no catalog-local or reflected scalar fallback");
});

test("air-side sampler bind-group ABI has one entry per binding", () => {
  const source = String(WebGPUOctreeFaceFastMarch.prototype.encodeAirSamples);
  const groups = [...source.matchAll(/const (classify|evaluate|finalize)Entries\s*=\s*\[([\s\S]*?)\];/g)]
    .map((match) => Array.from(match[2].matchAll(/binding:\s*(\d+)/g), (item) => Number(item[1])));
  assert.deepEqual(groups, [
    [0, 20, 21, 5, 6, 7, 22, 23, 26, 32, 48],
    [0, 6, 7, 19, 20, 21, 22, 23, 27, 28, 29, 30],
    [5, 20, 23, 48],
  ]);
  for (const bindings of groups) assert.equal(new Set(bindings).size, bindings.length);
  assert.equal(groups[1].filter((binding) => ![0, 20].includes(binding)).length, 10,
    "the exact evaluator stays at the portable ten-storage-buffer limit");
  assert.doesNotMatch(source, /powerFaces|faceNormals/,
    "compact-wet equivalence may override fine queries but must not claim persistent power-face authority");
});

test("row reconstruction binds the accepted face records it dereferences", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(source,
    /run\("reconstruct",\[\[0,this\.params\],\[5,this\.control\],\[6,this\.rows\],\[12,this\.faces\],\[14,this\.incidence\],\[15,this\.state\],\[19,this\.velocities\],\[32,this\.transitionControl\],\[44,this\.provisionalVelocities\],\[48,this\.pointFieldControl\]\]/,
    "auto-layout must receive the accepted face records reconstruction dereferences");
  assert.match(source,
    /run\("reconstructDeep",\[\[0,this\.params\],\[6,this\.rows\],\[12,this\.faces\],\[14,this\.incidence\],\[15,this\.state\],\[19,this\.velocities\],\[32,this\.transitionControl\],\[44,this\.provisionalVelocities\]\]/,
    "S4 carrier reconstruction must bind only its accepted-face graph and output vectors");
});

test("band phi retains fail-closed graph extension for closure-only rows", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(source, /run\("initializeBandPhi"[\s\S]*run\("seedBandPhiFaces"[\s\S]*run\("extendBandPhi"[\s\S]*run\("commitBandPhi"/);
});

test("2:1 face emission publishes bounded incidence directly", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(source,
    /run\("emit",\[\[0,this\.params\],\[5,this\.control\],\[6,this\.rows\],\[7,this\.rowHash\],\[12,this\.faces\],\[14,this\.incidence\],\[26,input\.owners\],\[32,this\.transitionControl\]\]/);
  assert.doesNotMatch(source, /run\("incidence"/);
});

test("GPU CPT binds its complete causal face graph within the portable limit", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const implementation = compact(WebGPUOctreeFaceFastMarch);
  assert.match(implementation, /jumpCpt:pipeline\("jumpFaceClosestPoints"\)/,
    "the pointer-jump WGSL entry point must have a live production pipeline");
  assert.match(source,
    /run\("linkCpt",\[\[0,this\.params\],\[5,this\.control\],\[6,this\.rows\],\[12,this\.faces\],\[14,this\.incidence\],\[15,this\.state\],\[27,this\.transitionMetrics\],\[31,this\.transitionAdjacency\],\[32,this\.transitionControl\],\[56,this\.cptParentsA\]\],0,pass\d*,228\)/,
    "CPT construction binds every resource used by the local Delaunay predecessor graph");
  assert.match(source,
    /for\(letround=0;round<this\.cptPlan\.jumpRounds;round\+=1\)\{run\("jumpCpt",\[\[0,this\.params\],\[5,this\.control\],\[12,this\.faces\],\[15,this\.state\],\[56,currentParents\],\[57,nextParents\]\],0,pass\d*,228\);\[currentParents,nextParents\]=\[nextParents,currentParents\]\}/,
    "the host dispatches the planned logarithmic number of race-free ping-pong jumps");
  assert.doesNotMatch(source, /Math\.ceil\(this\.plan\.faceCapacity\/64\)/,
    "no face stage may return to a fixed-capacity dispatch tail");
  assert.match(source,
    /run\("resolveCpt",\[\[0,this\.params\],\[5,this\.control\],\[12,this\.faces\],\[15,this\.state\],\[56,currentParents\]\]/,
    "resolution consumes the final contracted parent snapshot");
  assert.doesNotMatch(wgslFunction("resolveFaceClosestPoints"), /for\(/,
    "the resolver validates one contracted root instead of chasing an O(domain) chain");
  assert.match(source, /run\("prepareBfs"[\s\S]*propagateBfs/,
    "positive-phi plateaus use deterministic parallel BFS after the fast CPT forest");
  assert.match(source, /propagateBfs[\s\S]*propagateConnectivity[\s\S]*run\("validate"/,
    "a bounded seed-connected pass closes discrete face-centred phi minima before strict validation");
  assert.match(wgslFunction("considerBfsRow"), /states\[candidate\]\.depth>=layer/,
    "a layer never observes payload written by another invocation in the same dispatch");
  assert.match(wgslFunction("considerBfsRow"), /faceBfsCausal!=0u&&faceArrival/,
    "only the second bounded pass may cross a shallow positive-air local minimum");
  assert.match(wgslFunction("propagateFaceBfsLayer"), /connectivityFallbacks/,
    "every non-causal completion remains visible in generation diagnostics");
  assert.deepEqual(planOctreeFaceBandCPT(384), { maximumGraphDepth: 384, jumpRounds: 9 });
});

test("band-phi Jacobi and CPT remain bounded by the Section 5 narrow band", () => {
  const projection = compact(WebGPUOctreeProjection);
  assert.match(projection,
    /maximumNarrowBandGraphDepth=Math\.min\(256,Math\.max\(4,this\.interfaceRefinementBandCells\*\(this\.globalFineLevelSet\?\.plan\.fineFactor\?\?4\)\)\)/);
  assert.match(projection,
    /bandPhiRelaxationRounds=maximumNarrowBandGraphDepth/);
  assert.match(projection,
    /maximumCptGraphDepth=maximumNarrowBandGraphDepth/);
  assert.doesNotMatch(projection, /dims\.nx\+this\.dims\.ny\+this\.dims\.nz/,
    "the 2017 fine level set must not require whole-domain propagation");
  assert.match(projection,
    /rowCapacity,bandPhiRelaxationRounds,maximumCptGraphDepth,this\.powerFaces\.plan\.faceCapacity/);
  const schedule = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(schedule, /round<this\.bandPhiRelaxationRounds.*run\("extendBandPhi"/s);
  assert.doesNotMatch(schedule, /round<this\.cptPlan\.maximumGraphDepth.*run\("extendBandPhi"/s);
  assert.deepEqual(planOctreeFaceBandCPT(768), { maximumGraphDepth: 768, jumpRounds: 10 });
});

test("face emission binds only globals statically used by its auto layout", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const emit = source.match(/run\("emit",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  assert.ok(emit);
  assert.doesNotMatch(emit, /\[4,input\.siteIndex\]/,
    "site lookup is consumed while mapping rows, not while emitting the induced face graph");
  for (const binding of [0, 5, 6, 7, 12, 14, 26, 32]) assert.match(emit, new RegExp(`\\[${binding},`));
  for (const binding of [9, 10, 13]) assert.doesNotMatch(emit, new RegExp(`\\[${binding},`));
});

test("row mapping binds real topology and coarse-phi authority", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const map = source.match(/run\("map",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  assert.ok(map);
  assert.match(map, /\[25,input\.coarsePhiDirectory\]/);
  assert.match(map, /\[26,input\.owners\]/);
  assert.match(map, /\[42,input\.fineTopologyControl\]/,
    "coarse/fine pairing must consume the topology transaction that published the current fine slot");
  assert.match(octreeFaceBandWGSL,
    /fineTopologyControl\[0\]==0u&&fineTopologyControl\[4\]==1u&&fineTopologyControl\[5\]==0u&&fineTopologyControl\[7\]==0u/,
    "only a clean, published, nonrollback fine topology may seed Section 5");
  assert.match(wgslFunction("fineBandBrickCount"),
    /letseeds=fineTopologyControl\[8\].*fineTopologyControl\[2\]!=residentCount\|\|seeds>residentCount[\s\S]*returnresidentCount/s,
    "row discovery validates the interface seed prefix but covers the complete published narrow band");
  assert.match(wgslFunction("prepareFaceBand"),
    /bandBricks\*p\.ownersPerBrick/,
    "the indirect map dispatch must include every cell whose fine trajectory can request velocity");
  assert.match(wgslFunction("mapFineBricksToBandRows"),
    /letglobalRow=containing\(owner\.origin\);_=insertRow\(ownerCell,globalRow,ROW_COARSE\|ROW_CORE\|signFlag,owner\.size\)/,
    "air cells in the fine band remain regular-face march rows even without pressure unknowns");
  assert.match(octreeFaceBandWGSL, /clean&&coarseGeneration==fineGeneration/,
    "fine and coarse fields must be the same current paper generation");
  assert.doesNotMatch(octreeFaceBandWGSL, /rollbackPrior|coarseGeneration\+1u/,
    "rejected A/B scratch is never a new Section 5 authority");
  assert.doesNotMatch(map, /\[(?:8|24),/,
    "fine pages bound discovery only; their sparse phi validity must not become row authority");
  assert.doesNotMatch(map, /guard|fallback/i);
});

test("catalog-Delaunay transition adjacency has a bounded fail-closed producer ABI", () => {
  assert.equal(OCTREE_FACE_BAND_TRANSITION_ERROR.invalidBandDescriptor, 8);
  for (const [binding, declaration] of [
    [27, "metrics:array<Metric>"],
    [28, "tetraHeaders:array<TetraHeader>"],
    [29, "tetrahedra:array<u32>"],
    [30, "tetraVertices:array<TetraVertex>"],
    [31, "transitionAdjacency:array<TransitionAdjacency>"],
    [32, "transitionControl:TransitionControl"],
  ] as const) {
    assert.match(octreeFaceBandWGSL,
      new RegExp(`binding\\(${binding}\\)var<storage,[^>]+>${declaration.replace(/[<>()]/g, "\\$&")}`));
  }
  assert.match(octreeFaceBandWGSL, /fn prepareTransitionAdjacency\(/);
  assert.match(octreeFaceBandWGSL, /fn buildTransitionAdjacency\(/);
  assert.match(octreeFaceBandWGSL, /fn gateTransitionTransfer\(/);
  assert.match(octreeFaceBandWGSL,
    new RegExp(`const MAX_TETRA:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u`),
    "WGSL indexing must use the same generated catalog maximum as allocation planning");
  assert.match(octreeFaceBandWGSL,
    /let selectors=vec3u\(packed&255u,\(packed>>8u\)&255u,\(packed>>16u\)&255u\)/,
    "the producer must decode the catalog's three packed byte selectors");
  assert.match(octreeFaceBandWGSL,
    /transitionNeighbor\(band,selectors\.x,metric\)/,
    "the exact Stage-B metric must remain available to transition diagnostics");
  assert.match(wgslFunction("transitionNeighbor"),
    /inversePowerTransform\(vertex\.xyz,metric\.transformFlags&63u\)/,
    "catalog offsets must use the exact Stage-B row transform");
  assert.match(octreeFaceBandWGSL,
    /let expectedOwner=ownerAt\(neighborOrigin\);if\(!ownerContains\(expectedOwner,neighborOrigin\)\|\|expectedOwner\.size!=neighborSize\|\|any\(expectedOwner\.origin!=neighborOrigin\)\)/,
    "catalog neighbor geometry must first resolve to the exact authoritative adaptive owner");
  assert.match(octreeFaceBandWGSL,
    /let neighborCell=cell\(neighborOrigin\);let neighbor=rowOf\(neighborCell\);if\(neighbor==INVALID\)[\s\S]*?if\(neighbor>=atomicLoad\(&control\.rowCount\)\|\|neighbor>=arrayLength\(&rows\)\)[\s\S]*?if\(rows\[neighbor\]\.size!=neighborSize\)/,
    "catalog neighbors must distinguish a missing band row, row-range failure, and dyadic size mismatch");
  assert.match(octreeFaceBandWGSL,
    /atomicStore\(&transitionControl\.detailFlags,0u\)/,
    "aggregate adjacency detail bits must be reset for every transition transaction");
  assert.match(octreeFaceBandWGSL,
    /atomicStore\(&transitionControl\.failureBand,INVALID\)/,
    "the bounded first-owner-mismatch claim must reset for every transition transaction");
  assert.match(wgslFunction("recordOwnerFailure"),
    /atomicCompareExchangeWeak\(&transitionControl\.failureBand,INVALID,band\)/,
    "only one invocation may publish the diagnostic payload");
  assert.match(wgslFunction("transitionNeighbor"),
    /recordOwnerFailure\(OWNER_FAILURE_TRANSITION,[\s\S]*?adjacencyFail\(band,DETAIL_OWNER\)/,
    "strict transition-owner rejection must retain its reconstructable first failure");
  assert.match(octreeFaceBandWGSL, /TransitionAdjacency\(band,a,b,c\)/,
    "the producer must publish catalog-derived row quadruples, not Cartesian transition edges");
  assert.match(octreeFaceBandWGSL, /fn describeBandRow\(band:u32\)->Metric/,
    "dry band rows must resolve topology from their live owner neighborhood");
  assert.match(octreeFaceBandWGSL, /sameOrFinerDirect|sameOrCoarserDirect/,
    "all-band descriptors must reuse the generated direct-map catalog");
  assert.doesNotMatch(octreeFaceBandWGSL, /metrics\[row\.globalRow\]/,
    "transition topology must not depend on the wet-only compact metric arena");
  assert.match(octreeFaceBandWGSL, /transitionControl\.transferReady=VALID/,
    "the gate opens only after the Delaunay-expanded full-vector transfer is installed");
  assert.match(octreeFaceBandWGSL, /fn considerTopology\(row:u32,targetFace:u32/,
    "face propagation expands incidence through catalog-Delaunay node connectivity");
  assert.match(octreeFaceBandWGSL, /let transitionFlags=atomicLoad\(&transitionControl\.flags\)/,
    "invalid descriptors or adjacency still block publication");
});

test("Section 5 closes S0 through terminal endpoint support before building S0/S1/S2 adjacency", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const transitionPhase = source.slice(source.indexOf('case"transition-adjacency"'),
    source.indexOf('case"fast-march"'));
  for (const entrypoint of ["enumerateSupport1Requests", "enumerateSupport2Requests",
    "enumerateSupport3Requests", "resolveSupportOwners", "insertSupport1Rows",
    "insertSupport2Rows", "insertSupport3NodeRows", "enumerateSupport3EndpointRequests",
    "insertSupport3EndpointRows", "captureSupport3EndpointBoundary"] as const) wgslFunction(entrypoint);
  assert.match(transitionPhase,
    /run\("resolveTransition"[\s\S]*run\("enumerateSupport1"[\s\S]*run\("captureSupport1"[\s\S]*run\("enumerateSupport2"[\s\S]*run\("captureSupport2"[\s\S]*run\("enumerateSupport3"[\s\S]*run\("captureSupport3"[\s\S]*run\("enumerateEndpoints"[\s\S]*run\("captureEndpoints"[\s\S]*run\("transition"/,
    "every support boundary is snapshotted before the next exact dependency expansion");

  const prepare = wgslFunction("prepareTransitionAdjacency");
  assert.match(prepare,
    /transitionControl\.rowCount=min\(atomicLoad\(&control\.rowCount\),p\.rowCapacity\)/);
  assert.match(prepare, /atomicStore\(&control\.initialRows,transitionControl\.rowCount\)/,
    "the pre-closure row count is the immutable core prefix");
  assert.match(wgslFunction("buildTransitionAdjacency"),
    /band>=transitionControl\.support2End/,
    "S3 node and endpoint roles never own catalog adjacency");

  const topologyPhase = source.slice(source.indexOf('case"topology-build"'),
    source.indexOf('case"transition-adjacency"'));
  assert.match(topologyPhase, /run\("map"[\s\S]*run\("indexGlobalRows"/,
    "global-row publication completes over the core prefix before any closure row exists");
  assert.match(transitionPhase, /run\("captureEndpoints"[\s\S]*run\("indexGlobalRows"[\s\S]*run\("transition"/,
    "wet support rows publish their exact compact power-row identity only after closure is complete");
  assert.doesNotMatch(transitionPhase, /run\("transitionDeep"|run\("transitionSupport5"|run\("transitionSupport6"/,
    "S3-S6 support rows must not recursively demand Delaunay owner closure beyond the paper's narrow band");
  assert.match(transitionPhase, /run\("emit"[\s\S]*run\("emitDeep"[\s\S]*run\("sampleFacePhi"/,
    "deep-owned inward seams must close S0-S2 before regular-face phi sampling");
  const seamEmitter = wgslFunction("emitDeepBandFaces");
  assert.match(seamEmitter,
    /letband=transitionControl\.support2End\+g\.x[\s\S]*if\(neighbor>=transitionControl\.support2End\)\{continue;\}/,
    "a deep row emits only a positive-side seam whose neighbor is a marched S0-S2 target");
  assert.doesNotMatch(seamEmitter, /if\(neighbor<transitionControl\.support2End\)\{continue;\}/,
    "spatially negative deep owners must not drop the only emitter of an S0-S2 seam");
  assert.doesNotMatch(seamEmitter, /positiveBoundaryBit|Face\(band,INVALID/,
    "deep support rows do not create deep-world or recursively marched faces");
  assert.match(seamEmitter,
    /appendIncidence\(band,slot\);appendIncidence\(neighbor,slot\)/,
    "the seam is published reciprocally to its deep owner and marched target");
  assert.doesNotMatch(transitionPhase, /completeAdaptiveOwners/,
    "interface closure must not append every compact pressure owner in the domain");
  assert.match(wgslFunction("auditNarrowBandOwnerRows"),
    /band>=transitionControl\.support6NodeEnd[\s\S]*letexact=ownerAt\(origin\)[\s\S]*rowOf\(row\.cell\)!=band/,
    "the terminal audit is restricted to requested S0-S6 rows and verifies exact owner/hash identity");
  assert.match(wgslFunction("extendBandRowPhi"),
    /letclosureOnly=\(row\.flags&\(ROW_SUPPORT3_NODE\|ROW_SUPPORT4_NODE\|ROW_SUPPORT5_NODE\|ROW_SUPPORT6_NODE\|ROW_SUPPORT3_ENDPOINT\)\)!=0u;if\(!closureOnly&&rowIndex<arrayLength\(&metrics\)/,
    "deep support phi uses recorded lower-dimensional support rather than an unpublished Delaunay graph");
  assert.match(transitionPhase,
    /run\("auditNarrowBandOwners",\[\[0,this\.params\],\[6,this\.rows\],\[7,this\.rowHash\],\[26,input\.owners\]/,
    "the audit binds the already-closed narrow-band rows rather than the domain-wide compact site table");
  assert.doesNotMatch(wgslFunction("captureSupport3EndpointBoundary"), /endpointEnd!=support7NodeEnd|TRANSITION_CAPACITY/,
    "terminal endpoint-only carriers may close the bounded graph without being mistaken for capacity overflow");
});

test("support-closure stages stay within portable auto-layout bind groups", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const singleGroupBindings = (pipeline: string): number[] => {
    const call = source.match(new RegExp(`run\\("${pipeline}",\\[([\\s\\S]*?)\\],1,pass\\)`))?.[1];
    assert.ok(call, `missing encoded ${pipeline} single-workgroup bind group`);
    return [...call.matchAll(/\[(\d+),/g)].map((match) => Number(match[1]));
  };
  assert.deepEqual(singleGroupBindings("prepareSupport0Dispatch"), [18, 32, 43],
    "support dispatch capacity checks bind both their indirect output and candidate arena");
  for (const pipeline of ["captureSupport1", "captureSupport2", "captureSupport3", "captureSupport4",
    "captureSupport5", "captureSupport6", "captureSupport7"]) {
    assert.deepEqual(singleGroupBindings(pipeline), [0, 5, 18, 32, 43],
      `${pipeline} binds every resource reachable through writeSupportTierDispatch`);
  }
  const bindings = (pipeline: string): number[] => {
    const call = source.match(new RegExp(`run\\("${pipeline}",\\[([\\s\\S]*?)\\],(?:Math\\.ceil|0,pass)`))?.[1];
    assert.ok(call, `missing encoded ${pipeline} bind group`);
    return [...call.matchAll(/\[(\d+),/g)].map((match) => Number(match[1]));
  };
  assert.deepEqual(bindings("enumerateSupport1"), [6, 27, 28, 29, 32, 43]);
  assert.deepEqual(bindings("extendBandPhi"), [0, 1, 5, 6, 12, 14, 19, 27, 31, 44, 47, 53],
    "closure-only scalar support retains its portable bounded graph solve");
  const extendStorageBindings = bindings("extendBandPhi").filter((binding) => binding !== 0 && binding !== 1);
  assert.equal(extendStorageBindings.length, 10,
    "band-phi extension stays within the adapter's ten-storage-buffer shader-stage limit");
  assert.doesNotMatch(wgslFunction("extendBandRowPhi"), /transitionControl/,
    "closure roles come from the bound row record and must not pull an eleventh storage buffer into auto layout");
  assert.deepEqual(bindings("sampleFaceCoarsePhi"), [0, 5, 6, 7, 12, 27, 28, 29, 30, 31, 32],
    "the S0-S2-only Delaunay anchor gate binds its live transition prefix control");
  assert.equal(bindings("sampleFaceCoarsePhi").filter((binding) => binding !== 0).length, 10,
    "coarse face-phi interpolation remains at the ten-storage-buffer shader-stage limit");
  const resolveBindings = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase)
    .match(/constresolveOwnerBindings=\[([\s\S]*?)\];/)?.[1];
  assert.ok(resolveBindings);
  assert.deepEqual([...resolveBindings.matchAll(/\[(\d+),/g)].map((match) => Number(match[1])),
    [0, 5, 6, 26, 27, 30, 32, 43],
    "owner resolution reaches exact geometry, owner topology, and transition failure state only");
  const supportBindings = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase)
    .match(/constsupportBindings=\[([\s\S]*?)\];/)?.[1];
  assert.ok(supportBindings);
  assert.deepEqual([...supportBindings.matchAll(/\[(\d+),/g)].map((match) => Number(match[1])),
    [0, 1, 4, 5, 6, 7, 25, 27, 32, 42, 43],
    "guard insertion remains within ten storage bindings after scalar-edge capture is split out");
  const recordBindings = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase)
    .match(/constrecordPhiEdgeBindings=\[([\s\S]*?)\];/)?.[1];
  assert.ok(recordBindings);
  assert.deepEqual([...recordBindings.matchAll(/\[(\d+),/g)].map((match) => Number(match[1])),
    [0, 5, 7, 32, 43, 47, 53],
    "bounded incoming scalar edges reuse later transient-power scratch in a separate portable stage");
});

test("S2 support candidates resolve against the live pre-capture row prefix", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const enumerate = source.indexOf('run("enumerateSupport2"');
  const resolve = source.indexOf('run("resolveSupportOwners"', enumerate);
  const insert = source.indexOf('run("insertSupport2"', resolve);
  const capture = source.indexOf('run("captureSupport2"', insert);
  assert.ok(enumerate >= 0 && resolve > enumerate && insert > resolve && capture > insert,
    "S2 requests must resolve and insert before support2End is captured");
  const resolver = wgslFunction("resolveSupportOwners");
  assert.match(resolver,
    /letexistingEnd=min\(atomicLoad\(&control\.rowCount\),p\.rowCapacity\)/,
    "the resolver admits every currently published S1 anchor while support2End is still stale");
  assert.doesNotMatch(resolver, /candidate\.band>=transitionControl\.support2End/,
    "the future S2 boundary cannot gate the S2 owner-resolution stage");
});

test("live support prefixes dispatch sparse closure even when S0 reservation consumes the domain", () => {
  const plan = planOctreeFaceBandGPU(64, 24 * 18 * 16, 4, 4, 256, [24, 18, 16]);
  assert.equal(plan.support0RowCapacity, 24 * 18 * 16);
  assert.equal(plan.support1RowCapacity, 0,
    "reservation arithmetic intentionally has no remaining static S1 role budget");
  assert.deepEqual(planOctreeFaceBandLiveSupportDispatch({
    coreEnd: 12,
    support1End: 42,
    support2End: 102,
    support3NodeEnd: 150,
    support4NodeEnd: 198,
  }), [1, 1, 1, 1, 1], "live sparse prefixes, not zero static role caps, control closure work");
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.match(source, /run\("prepareSupport0Dispatch"[\s\S]*0,pass,24\)/);
  for (const [capture, offset] of [["captureSupport1", 48], ["captureSupport2", 72],
    ["captureSupport3", 96], ["captureSupport4", 120], ["captureSupport5", 144],
    ["captureSupport6", 168], ["captureSupport7", 192]] as const) assert.match(source,
    new RegExp(`run\\("${capture}"[\\s\\S]*0,pass,${offset}\\)`));
  for (const offset of [36, 60, 84, 108, 132, 156]) {
    assert.match(source, new RegExp(`run\\("resolveSupportOwners"[\\s\\S]{0,80}0,pass,${offset}\\)`),
      `candidate work at byte ${offset} must consume a GPU-authored live-prefix dispatch`);
  }
  assert.match(source, /run\("insertEndpoints"[\s\S]{0,80}0,pass,204\)/,
    "endpoint candidates consume their GPU-authored live-prefix dispatch");
  assert.doesNotMatch(source, /candidateWorkgroups/,
    "support closure never dispatches the shared candidate capacity");
  for (const offset of [36, 60, 84, 108, 132, 156, 204]) {
    assert.match(source, new RegExp(`run\\("clearSupportCandidates"[\\s\\S]{0,80}0,pass,${offset}\\)`),
      `the padded live prefix at byte ${offset} is retired without touching the capacity tail`);
  }
  assert.match(wgslFunction("enumerateSupportRequests"),
    /for\(varrequest=0u;request<MAX_GUARDS;request\+=1u\).*GuardCandidate\(INVALID,INVALID,INVALID,0u\)/,
    "each live fixed-fanout owner exclusively retires and rewrites its candidate record range");
  const capture4 = wgslFunction("captureSupport4NodeBoundary");
  assert.match(capture4,
    /writeSupportDispatch\(30u,transitionControl\.support4NodeEnd-transitionControl\.support2End\)/,
    "deep topology revisits the cumulative S3+S4 range required by buildDeepTransitionAdjacency");
  assert.match(capture4,
    /letappended=transitionControl\.support4NodeEnd-transitionControl\.support3NodeEnd[\s\S]*appended\*MAX_GUARDS/,
    "S5 candidate discovery consumes only rows newly appended by the S4 closure");
  assert.match(source,
    /run\("preparePointDispatch"[\s\S]*run\("preparePointRows"[\s\S]*0,pass\d*,24\)/,
    "the final S0+S1 point field also consumes the live published prefix rather than a zero role cap");
});

test("shared candidate arena covers sparse live-prefix migration across every enumerate tier", () => {
  const plan = planOctreeFaceBandGPU(16, 2, 4, 4, 64, [4, 1, 1]);
  assert.deepEqual([
    plan.support0RowCapacity,
    plan.support1RowCapacity,
    plan.support2RowCapacity,
    plan.support3NodeRowCapacity,
  ], [2, 2, 0, 0], "the static reservation reproduces the sparse migration case");
  assert.equal(plan.rowCapacity, 4);
  assert.equal(plan.guardCandidateCapacity, 4 * 36,
    "every physical row may temporarily belong to the largest-fanout live tier");

  const liveEnumerations = [
    { tier: "S0", rows: 1, fanout: 36 },
    { tier: "S1", rows: 3, fanout: 36 },
    { tier: "S2", rows: 4, fanout: 36 },
    { tier: "S3 endpoint", rows: 4, fanout: 24 },
  ] as const;
  for (const { tier, rows, fanout } of liveEnumerations) {
    const lastBase = (rows - 1) * fanout;
    assert.ok(lastBase >= 0 && lastBase + fanout <= plan.guardCandidateCapacity,
      `${tier} final live enumerate base must fit the shared arena`);
  }
  assert.ok(2 * 36 < 3 * 36,
    "the former static-tier arena would fail the three-row live S1 enumeration");
});

test("catalog/uniform closure inserts exact owner/coarse rows and never clamps a demand", () => {
  assert.match(octreeFaceBandWGSL,
    new RegExp(`const MAX_GUARDS:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows}u`),
    "WGSL closure indexing must share the generated catalog neighbor-row bound");

  const enumerate = wgslFunction("enumerateSupportRequests");
  assert.match(octreeFaceBandWGSL, /const UNIFORM_GUARDS:u32=26u/,
    "uniform cube interpolation closes the complete surrounding 3x3x3 owner block");
  assert.match(enumerate, /for\(varrequest=0u;request<UNIFORM_GUARDS;request\+=1u\)/,
    "uniform anchors request all 18 catalog neighbors plus eight exact body-diagonal carriers");
  assert.match(enumerate, /letselectors=array<u32,3>\(packed&255u,\(packed>>8u\)&255u,\(packed>>16u\)&255u\)/,
    "every catalog tetrahedron selector is enumerated from the generated packed record");
  assert.doesNotMatch(enumerate, /clamp\(/,
    "a catalog selector outside exact owner support is a failure, not a clamped request");

  const resolve = wgslFunction("resolveSupportOwners");
  assert.match(enumerate + resolve,
    /selector>=arrayLength\(&tetraVertices\)[\s\S]{0,160}transitionFail\(TRANSITION_SOURCE,[^)]+\);return;/,
    "an invalid catalog selector is publication-fatal, never silently skipped");
  assert.match(resolve, /letvertex=tetraVertices\[candidate\.selector\]\.v/,
    "guard demand geometry comes from the exact catalog selector vertex");
  assert.match(resolve, /direction=DESCRIPTOR_DIRECTIONS\[request\]/,
    "uniform closure uses the same canonical face/edge directions as descriptor construction");
  assert.match(resolve, /direction=BODY_DIAGONAL_DIRECTIONS\[request-18u\]/,
    "the eight cube-corner carrier requests remain exact same-size owner offsets");
  assert.match(resolve, /letexact=ownerAt\(origin\)/,
    "each catalog demand resolves through the live adaptive owner authority");
  assert.match(resolve,
    /!ownerContains\(exact,origin\)\|\|exact\.size!=size\|\|any\(exact\.origin!=origin\)/,
    "a selector demand accepts only the exact requested owner origin and size");
  assert.match(resolve, /adjacencyFail\(candidate\.band,DETAIL_OWNER\)/,
    "owner mismatch is publication-fatal rather than silently skipped");
  assert.doesNotMatch(resolve, /clamp\(/,
    "owner resolution cannot move an out-of-domain selector onto the boundary");

  const insert = wgslFunction("insertSupportCandidate");
  assert.match(insert, /coarseSignFlag\(owner\)/,
    "closure membership consumes only the published coarse sign");
  assert.doesNotMatch(insert, /representativePhi|minimumPhi|maximumPhi|coarse\.x|coarse\.y|coarse\.z/,
    "support insertion cannot publish a coarse or complement magnitude as distance");
  assert.match(insert,
    /letglobalRow=containing\(origin\).*insertRow\(candidate\.cell,globalRow,ROW_COARSE\|rowRole\|signFlag,candidate\.size\)/,
    "support rows retain their exact compact power-row identity without claiming fine phi authority");

  const endpoints = wgslFunction("enumerateSupport3EndpointRequests");
  assert.match(endpoints, /for\(varaxis=0u;axis<3u;axis\+=1u\)/);
  assert.match(endpoints, /for\(varside=0u;side<2u;side\+=1u\)/);
  assert.match(endpoints, /letsampleCount=select\(1u,4u,row\.size>1u\)/);
  assert.match(endpoints, /for\(varlocal=0u;local<sampleCount;local\+=1u\)/,
    "terminal endpoint closure enumerates the exact six-side/four-subface 2:1 incidence bound");
  assert.match(resolve + endpoints, /velocityExtendedOrigin/,
    "one-layer world-boundary requests use algebraic closed reflection or authored open extension without exterior rows");
});

test("transition producer binds the Stage-B catalog and no fine velocity surrogate", () => {
  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const transition = source.match(/run\("transition",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  assert.ok(transition);
  for (const binding of [0, 6, 7, 26, 27, 28, 29, 30, 31, 32]) {
    assert.match(transition, new RegExp(`\\[${binding},`));
  }
  for (const binding of [8, 24, 25]) {
    assert.doesNotMatch(transition, new RegExp(`\\[${binding},`),
      "fine samples and phi must not replace catalog-Delaunay adjacency authority");
  }
  assert.match(source,
    /run\("prepareTransition",\[\[0,this\.params\],\[5,this\.control\],\[32,this\.transitionControl\]\],1,pass\)/);
  assert.match(source,
    /run\("gateTransition",\[\[5,this\.control\],\[32,this\.transitionControl\]\],1,pass\)/);
  assert.match(source,
    /if\(!tetrahedronHeaders\|\|!tetrahedra\|\|!tetrahedronVertices\)\{thrownewRangeError\("Face-bandtransitionsrequirethecatalogDelaunaybuffers"\)\}/,
    "missing catalog buffers must be rejected on the CPU before encoding work");
});

test("paper Section 5 regular-to-power publication is fail-closed and centroid based", () => {
  assert.equal(OCTREE_FACE_BAND_POWER_PUBLICATION_ERROR.incomplete, 64);
  const wet = wgslFunction("bandRowIsWet");
  assert.match(wet,
    /\(rows\[band\]\.flags&ROW_PHI\)!=0u&&finite\(rows\[band\]\.representativePhi\)&&rows\[band\]\.representativePhi<0\.0/,
    "projected-face preservation uses only a finite current marched liquid sign");
  const interpolatePower = wgslFunction("interpolatePowerFaceVector");
  assert.match(interpolatePower,
    /if\(bandRowIsWet\(negativeBand\)\|\|bandRowIsWet\(positiveBand\)\)\{powerVelocityScratch\[index\]=vec4u\(0u,0u,0u,2u\);return;\}/,
    "one-sided extrapolation must preserve every projected face incident to liquid, including a wet/out-of-band interface face");
  assert.doesNotMatch(interpolatePower,
    /bandRowIsWet\(negativeBand\)&&bandRowIsWet\(positiveBand\)/,
    "interface faces are pressure degrees of freedom, not air-side extrapolation targets");
  assert.match(wgslFunction("projectPowerFaceVelocity"),
    /if\(candidate\.w==2u\)\{powerVelocityScratch\[index\]=vec4u\(0u,3u,0u,0u\);atomicAdd\(&powerPublication\.interpolatedCount,1u\);return;\}/,
    "preserved projected faces still participate in the all-or-nothing publication tally");
  assert.match(octreeFaceBandWGSL, /fn marchedCentroidVector\(anchor:u32,pointGrid:vec3f\)/,
    "power-face interpolation must evaluate the marched full-vector field at the physical face centroid");
  assert.match(octreeFaceBandWGSL,
    /if\(\(header\.flags&1u\)!=0u\)[\s\S]*result\+=weight\*value\.xyz/,
    "uniform regions must use the paper's regular-grid trilinear interpolant");
  assert.match(octreeFaceBandWGSL,
    /let weights=tetraWeights\(point,[\s\S]*weights\.x\*anchorVelocity\.xyz\+weights\.y\*va\.xyz/,
    "transition regions must use catalog-Delaunay barycentric interpolation");
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
  assert.match(octreeFaceBandWGSL,
    /atomicLoad\(&powerPublication\.interpolatedCount\)==targets\)\{atomicStore\(&powerPublication\.valid,VALID\)/,
    "partial interpolation cannot open the commit gate");
  assert.match(octreeFaceBandWGSL,
    /if\(atomicLoad\(&powerPublication\.valid\)!=VALID[^}]+return;\}let candidate=powerVelocityScratch\[index\]/,
    "power face records remain untouched unless the whole target subset is valid");
  assert.match(octreeFaceBandWGSL,
    /fn mapPowerFaceBands[\s\S]*let negativeBand=bandForGlobalRow\(powerFace\.negativeRow\)[\s\S]*powerVelocityScratch\[index\]=vec4u\(0u,0u,negativeBand,positiveBand\)/,
    "the mapping stage must publish only power-endpoint to regular-band identities");
  const globalIndex = wgslFunction("indexBandGlobalRows");
  assert.match(globalIndex,
    /lettargetRoles=ROW_CORE\|ROW_SUPPORT1\|ROW_SUPPORT2/,
    "only S0-S2 rows with a guaranteed final vector may independently target production power faces");
  assert.match(globalIndex,
    /if\(\(row\.flags&targetRoles\)==0u\)\{return;\}/,
    "S3-S6 and endpoint scalar/selector closure cannot create an unpublishable target");
  assert.doesNotMatch(globalIndex,
    /targetRoles=[^;]*ROW_SUPPORT3_NODE|targetRoles=[^;]*ROW_SUPPORT4_NODE|targetRoles=[^;]*ROW_SUPPORT5_NODE|targetRoles=[^;]*ROW_SUPPORT6_NODE|targetRoles=[^;]*ROW_SUPPORT3_ENDPOINT/,
    "best-effort deep reconstruction does not widen the authoritative Section 5 update subset");
  assert.match(octreeFaceBandWGSL,
    /fn interpolatePowerFaceVector[\s\S]*let mapping=powerVelocityScratch\[index\];let negativeBand=mapping\.z;let positiveBand=mapping\.w/,
    "the interpolation stage must consume the completed endpoint mapping instead of repeating hash lookup");

  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const prepare = source.match(/run\("preparePowerPublication",\[([\s\S]*?)\],1,pass\d*\)/)?.[1];
  assert.ok(prepare);
  for (const binding of [0, 5, 32, 36, 37, 38, 39, 40, 41, 48]) {
    assert.match(prepare, new RegExp(`\\[${binding},`));
  }
  assert.match(prepare, /\[48,this\.pointFieldControl\]/,
    "regular-to-power publication must consume the complete final point-field transaction");
  const map = source.match(/run\("mapPowerFaceBands",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  assert.ok(map);
  for (const binding of [0, 35, 37, 40, 41]) {
    assert.match(map, new RegExp(`\\[${binding},`));
  }
  for (const binding of [1, 5, 6, 12, 14, 15, 19, 36, 38, 39]) {
    assert.doesNotMatch(map, new RegExp(`\\[${binding},`),
      "endpoint mapping must not bind regular-vector interpolation or unused publication inputs");
  }
  const interpolate = source.match(/run\("interpolatePowerFaces",\[([\s\S]*?)\],Math\.ceil/)?.[1];
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
  const project = source.match(/run\("projectPowerFaces",\[([\s\S]*?)\],Math\.ceil/)?.[1];
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
  const publishAt = source.indexOf('run("publishPowerFaces"', projectAt);
  const commitAt = source.indexOf('run("commitPowerFaces"', publishAt);
  assert.ok(prepareAt >= 0 && mapAt > prepareAt && interpolateAt > mapAt
    && projectAt > interpolateAt && publishAt > projectAt && commitAt > publishAt,
  "Section 5 transfer must map endpoints, interpolate the completed regular field, project, validate globally, then commit");
});

test("paper Section 5 retains least-squares liquid power-cell vectors before extrapolating support", () => {
  const seedMapped = wgslFunction("seedMappedPowerRowVelocity");
  assert.match(seedMapped,
    /letrow=rows\[band\];if\(row\.globalRow==INVALID\)\{return;\}/,
    "only outside-liquid support without a mapped power row may rely on extrapolated closure");
  assert.match(seedMapped,
    /letvalue=powerRowVelocities\[row\.globalRow\]/,
    "mapped liquid cells must consume the generalized-face least-squares centre vector");
  assert.match(seedMapped,
    /rowVelocities\[band\]=value;provisionalVelocities\[band\]=value/,
    "the authoritative cell vector must reach both the interpolation and publication fields");
  assert.doesNotMatch(seedMapped, /nearest|rest=|vec4f\(0\.,0\.,0\.,1\.\)/,
    "mapped power cells must not be replaced by nearest-cell or rest-state scaffolding");

  const source = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  const closureAt = source.indexOf('run("completeDeepClosure"');
  const endpointAt = source.indexOf('run("initializeColdEndpoints"', closureAt);
  const mappedAt = source.indexOf('run("seedMappedPowerRows"', endpointAt);
  const publishAt = source.indexOf('run("publish"', mappedAt);
  assert.ok(closureAt >= 0 && endpointAt > closureAt && mappedAt > endpointAt && publishAt > mappedAt,
    "real power-cell vectors must override auxiliary support closure before the band transaction publishes");
});

test("factor-4/factor-8 production schedule publishes and consumes the face-marched air band", () => {
  const projection = compact(WebGPUOctreeProjection.prototype.encode);
  const faceBand = compact((WebGPUOctreeProjection.prototype as unknown as {
    encodeGlobalFineFaceBandPhase: (encoder: GPUCommandEncoder, phase: string) => void;
  }).encodeGlobalFineFaceBandPhase);
  const powerToAxis = projection.indexOf("this.encodePowerVelocityPublication(encoder)");
  const constrain = projection.indexOf("this.solidFaces?.encodePostProjectionConstraint(encoder)", powerToAxis);
  const faceMarch = projection.indexOf("this.encodeGlobalFineFaceBand(encoder", constrain);
  const divergence = projection.indexOf("this.faceMirror?.encodeProjectedDivergence(encoder)", faceMarch);
  assert.ok(powerToAxis >= 0 && constrain > powerToAxis && faceMarch > constrain && divergence > faceMarch,
    "power -> regular faces -> solid constraint -> face march must precede downstream publication");
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
    "the completed face marcher must transactionally republish onto generalized power faces");
  const republish = faceBand.indexOf("this.powerFaceSeed.encodePowerToAxis(encoder,this.powerOperator.control,true)");
  const recapture = faceBand.indexOf("this.powerFaceAdvection?.encodeCapture(encoder", republish);
  assert.ok(republish > faceBand.indexOf("this.globalFineFaceFastMarch.encodePhase(encoder") && recapture > republish,
    "the extrapolated power field must return to compact face transport and the next old-mesh snapshot");
  assert.doesNotMatch(faceBand, /powerFaceTransfer\?\.encodeCapture/,
    "the unused generalized-face transfer must not sort a dead snapshot");

  const transport = compact(WebGPUFineLevelSetTransport.prototype.encode);
  const stageB = transport.indexOf("this.velocityPrepass.encodeFromPositions");
  const airBand = transport.indexOf("this.faceBand?.encodeAirSamples", stageB);
  const advance = transport.indexOf("run(this.advancePipeline", airBand);
  assert.ok(stageB >= 0 && airBand > stageB && advance > airBand,
    "every piecewise-Euler segment must replace positive-air Stage-B fallback before advancing");
  assert.match(transport.slice(airBand, advance),
    /this\.velocityPrepass\.source\.results,this\.velocityPrepass\.source\.statuses/,
    "the face marcher must override the exact result/status pair consumed by trajectory advancement");
  assert.match(transport.slice(airBand, advance), /owners:options\.ownerTopology/,
    "every air-band query must resolve the same adaptive owner authority as row publication");
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

test("Section 5 final point field consumes one complete transient physical power graph", () => {
  for (const entry of ["prepareTransientBandPowerGraph", "emitTransientBandPowerGraph",
    "sampleTransientBandPowerFaces", "validateTransientBandPowerGraph", "publishTransientBandPowerGraph",
    "prepareBandPointField", "prepareBandPointRows", "accumulateBandTransientPowerLS",
    "solveBandPowerLS", "validateBandPointField", "publishBandPointField"]) {
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
  const neighborLookup = wgslFunction("transientNeighbor");
  assert.match(neighborLookup, /any\(abs\(originValue-origin\)>vec3f\(2e-4\)\)/,
    "a catalog neighbor must be on the exact integer lattice before row lookup");
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
  assert.doesNotMatch(sample,
    /negativeTarget&&!negativeValid|positiveTarget&&!positiveValid/,
    "a T-junction centroid is located by whichever endpoint's paper cube/tetrahedron contains it");
  assert.match(sample,
    /elseif\(positiveValid\)\{full=positive\.xyz;\}else\{transientFail\(POINT_SAMPLE,faceIndex\)/,
    "the physical face still fails closed when neither endpoint interpolation contains its centroid");
  assert.match(sample, /scalar=dot\(full,face\.normal\.xyz\)/,
    "the exact-centroid vector is projected once to the committed normal scalar");
  assert.match(octreeFaceBandWGSL, /determinant<=1e-7\*trace\*trace\*trace/);
  assert.match(octreeFaceBandWGSL, /condition>1e5/);
  assert.doesNotMatch(octreeFaceBandWGSL, /var best=anchorVelocity|bestDistance=dot\(point,point\)/,
    "authoritative interpolation has no nearest/anchor fallback");
  assert.match(octreeFaceBandWGSL,
    /fn seedOpenWorldNormal[\s\S]*pf\.normalVelocity\/n\[axis\]/,
    "the one-sided open regular face inherits the matching pre-extrapolation power-face scalar");
  assert.doesNotMatch(wgslFunction("prepareBandPointField"), /powerPublication/,
    "the transient physical graph must break the former point-field/power-publication cycle");
  assert.match(wgslFunction("preparePowerPublication"),
    /atomicLoad\(&pointControl\.valid\)!=VALID|pointControl\.generation!=p\.generation/,
    "production power faces require the complete same-generation point field");
  const physical = wgslFunction("accumulateBandTransientPowerLS");
  assert.match(physical,
    /letorientation=f32\(item\.sign\);letnormal=orientation\*face\.normal\.xyz;letu=orientation\*face\.normalVelocity/,
    "every S0/S1 row consumes signed physical normals and normal velocities from the transient CSR");
  assert.match(physical, /letweight=face\.area/,
    "the final least-squares fit uses exact physical generalized-face area");
  assert.doesNotMatch(physical, /globalRow|powerIncidenceRows|catalogFaces/,
    "wet and dry rows share one final physical graph with no authority split");
  const prepareRows = wgslFunction("prepareBandPointRows");
  assert.doesNotMatch(prepareRows, /globalRow|catalogFaces/,
    "all S0/S1 accumulators start identically before the transient graph fit");
  assert.doesNotMatch(octreeFaceBandWGSL,
    /accumulateBandPowerLSBatch|accumulateBandPowerLS0|accumulateBandPowerLS1|accumulateBandPowerLS2/,
    "no catalog-local invented scalar fallback remains callable");

  const schedule = compact(WebGPUOctreeFaceFastMarch.prototype.encodePhase);
  assert.doesNotMatch(schedule,
    /clearBuffer\(this\.transientPower(?:Faces|Incidence|Rows|Control)\)/,
    "counted transient publication does not clear any fixed-capacity arena");
  assert.match(schedule,
    /run\("sampleTransientPower"[\s\S]*Math\.ceil\(this\.plan\.rowCapacity\/64\),pass\d*\)/,
    "one invocation samples all owned slots for a live row");
  const provisionalAt = schedule.indexOf('run("publish"');
  const prepareTransientAt = schedule.indexOf('run("prepareTransientPower"', provisionalAt);
  const emitTransientAt = schedule.indexOf('run("emitTransientPower"', prepareTransientAt);
  const sampleTransientAt = schedule.indexOf('run("sampleTransientPower"', emitTransientAt);
  const validateTransientAt = schedule.indexOf('run("validateTransientPower"', sampleTransientAt);
  const publishTransientAt = schedule.indexOf('run("publishTransientPower"', validateTransientAt);
  const preparePointAt = schedule.indexOf('run("preparePointField"', publishTransientAt);
  const physicalAt = schedule.indexOf('run("accumulatePhysicalPoint"', preparePointAt);
  const solveAt = schedule.indexOf('run("solvePoint"', physicalAt);
  const publishPointAt = schedule.indexOf('run("publishPoint"', solveAt);
  const preparePowerAt = schedule.indexOf('run("preparePowerPublication"', publishPointAt);
  const commitAt = schedule.indexOf('run("commitPowerFaces"', preparePowerAt);
  assert.ok(provisionalAt >= 0 && prepareTransientAt > provisionalAt && emitTransientAt > prepareTransientAt
    && sampleTransientAt > emitTransientAt && validateTransientAt > sampleTransientAt
    && publishTransientAt > validateTransientAt && preparePointAt > publishTransientAt
    && physicalAt > preparePointAt && solveAt > physicalAt && publishPointAt > solveAt
    && preparePowerAt > publishPointAt && commitAt > preparePowerAt,
  "paper order is regular march -> all-band physical graph -> final cell-centre LS -> production publication");
  const physicalBindings = schedule.match(/run\("accumulatePhysicalPoint",\[([\s\S]*?)\],0,pass,24\)/)?.[1];
  assert.ok(physicalBindings);
  assert.equal(physicalBindings.match(/\[\d+,/g)?.length, 7,
    "all-band physical LS stays within the portable storage-binding limit");
  const emitBindings = schedule.match(/run\("emitTransientPower",\[([\s\S]*?)\],Math\.ceil/)?.[1];
  assert.equal(emitBindings?.match(/\[\d+,/g)?.length, 11,
    "physical graph emission uses one uniform plus the portable maximum ten storage buffers");
});
