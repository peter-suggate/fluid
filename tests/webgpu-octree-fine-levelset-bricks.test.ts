import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  FineLevelSetBrickOracle,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
} from "../lib/octree-fine-levelset-bricks";
import {
  WebGPUFineLevelSetBricks,
  fineLevelSetBrickSamplingWGSL,
} from "../lib/webgpu-octree-fine-levelset-bricks";
import { unpackFineLevelSetGPUTransportControl } from
  "../lib/webgpu-octree-fine-levelset-transport";
import {
  FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
  FINE_LEVELSET_JFA_LOOKUP_STRIDES,
  WebGPUFineLevelSetRedistance,
  fineLevelSetJFACPTWGSL,
  fineLevelSetJFATapAddress,
  planFineLevelSetJFAStrides,
  unpackFineLevelSetGPURedistanceControl,
} from "../lib/webgpu-octree-fine-levelset-redistance";

import { PassBroker } from "../lib/webgpu-pass-broker";
import {
  FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON,
  fineLevelSetLeafSeedWGSL,
  makeFineLevelSetTopologyWGSL,
  planFineLevelSetLeafBrickBounds,
  planFineLevelSetPageDeltaLayout,
  planFineLevelSetTopologyBand,
  WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology,
  unpackFineLevelSetGPUTopologyControl,
} from "../lib/webgpu-octree-fine-levelset-topology";

type DawnModule = { create(options: string[]): GPU; globals: Record<string, unknown> };
// Dawn's native event pump can outlive the final device callback. Keep the
// instance reachable for the process lifetime so V8 cannot reclaim it early.
const retainedDawnInstances: GPU[] = [];
const dawnModule = process.env.WEBGPU_NODE_MODULE
  ? import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href) as Promise<DawnModule>
  : undefined;
const dawnDevice = dawnModule?.then(async (dawn) => {
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  retainedDawnInstances.push(gpu);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  assert.ok(adapter.features.has("subgroups"),
    "the production M1 Max fine-JFA target requires subgroup candidate reductions");
  return adapter.requestDevice({
    requiredFeatures: ["subgroups"],
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
});
function redistanceDeltaAuthority(
  pageDelta: GPUBuffer,
  pageCapacity: number,
  dispatchBuffer: GPUBuffer = pageDelta,
) {
  return {
    pageDelta,
    pageDeltaLayout: planFineLevelSetPageDeltaLayout(pageCapacity),
    redistanceDispatches: {
      buffer: dispatchBuffer,
      dirtyOffsetBytes: 84 as const,
      supportOffsetBytes: 60 as const,
    },
  };
}

// Keep the native Dawn construction gate first. The Metal-backed Node binding
// is unstable when this very large module is compiled after V8 has reclaimed
// earlier copies of the generated WGSL in the same test isolate.
test("Dawn constructs every production fine-topology entry point", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for full topology pipeline validation",
}, async () => {
  const device = await dawnDevice!;
  const topologyWGSL = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  );
  const entryPoints = [
    "clearFinePageDelta", "clearDesiredGeneration", "clearTopologyErrors",
    "discoverInterfaceBricks", "insertExternalSeeds",
    "scanSparseSeedRecords", "scanSparseCandidateRecords", "scanSparseGroups",
    "scanSparseSuperGroups", "offsetSparseGroups", "offsetSparseRecords",
    "finalizeDesiredSeedCount", "clearSparseCandidates", "compactSparseSeeds",
    "sortSparseCandidates", "expandSparseDesiredX", "expandSparseDesiredY",
    "expandSparseDesiredZ", "compactSparseDesiredBricks", "publishDesiredBricks",
    "publishRecurringSparseBand", "scanRecurringDesiredRecords",
    "finalizeRecurringSparseBand", "scatterRecurringSparseBand",
    "classifyDesiredPageIdentities",
    "reduceTopologyErrorRecords", "reduceTopologyErrorGroups",
    "reduceTopologyErrorSuperGroups", "validateDesiredSeeds", "scanIdentityRecords",
    "scanIdentityGroups", "scanIdentitySuperGroups", "offsetIdentityGroups",
    "offsetIdentityRecords", "prepareDesiredPageIdentityAssignment",
    "compactDesiredPageIdentities", "assignDesiredPageIdentities",
    "finalizeDesiredPageIdentityAssignment", "carryDesiredSamples", "carryDesiredWorkSamples",
    "classifyFinePageDelta", "compactFineChangedKeys", "prepareFinePageDeltaExpansion",
    "classifyFineAffectedPages", "compactFineAffectedPages", "finalizeFinePageDelta",
    "snapshotDeltaPayload", "initializeDesiredSamples", "linkDesiredNeighbors",
    "finalizeDesiredGeneration", "finalizeFinePublication", "settleFinePublication",
    "settleFineWorkPayload",
  ];
  for (const entryPoint of entryPoints) {
    assert.match(topologyWGSL, new RegExp(`fn\\s+${entryPoint}\\s*\\(`));
  }
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [1, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 1 });
  const oracle = new FineLevelSetBrickOracle(plan);
  oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [0, 0, 0])], ([x]) => x - 0.5);
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.uploadGeneration(oracle.exportGPUGeneration());
  const target = owner.prepareGPUGeneration(2);
  device.pushErrorScope("validation");
  const topology = new WebGPUFineLevelSetTopology(device, current, target,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}");
  assert.equal(await device.popErrorScope(), null,
    "the production constructor's complete topology pipeline table must remain Dawn-valid");
  topology.destroy(); owner.destroy();
});

test("fine redistance diagnostics separate missing CPT seeds from Eikonal violations", () => {
  const words = new Uint32Array([2_388, 26_436_288, 5_804, 0, 0, 0xffff_ffff,
    12_345, 4_096, 4_096, 17]);
  assert.equal(FINE_LEVELSET_REDISTANCE_CONTROL_BYTES, 40);
  assert.deepEqual(unpackFineLevelSetGPURedistanceControl(words), {
    unresolvedCells: 2_388,
    resolveMissingCells: 17,
    residualViolationCells: 2_371,
    maximumResidualScaled: 26_436_288,
    seedCount: 5_804,
    committed: false,
    flags: 0,
    firstError: 0xffff_ffff,
    acceptedCells: 12_345,
    initialPages: 4_096,
    finalPages: 4_096,
  });
  assert.match(fineLevelSetJFACPTWGSL,
    /control\.resolveMissing=reduceSum0\[0\];control\.unresolved=reduceSum0\[0\]\+reduceSum1\[0\]/);
  assert.match(fineLevelSetJFACPTWGSL,
    /partials\[work\]\.validationUnresolved=reduceSum0\[0\][\s\S]*control\.unresolved=reduceSum0\[0\]\+reduceSum1\[0\]/);
});

test("fine transport diagnostics decode the exact first invalid velocity position", () => {
  const bytes = new ArrayBuffer(64);
  const words = new Uint32Array(bytes);
  const floats = new Float32Array(bytes);
  words.set([3, 2, 41, 0, 17, 5, 7, 11, 13, 19, 0x0120_0004, 0x0800_0004, 23]);
  floats.set([1.25, 2.5, 3.75], 13);

  assert.deepEqual(unpackFineLevelSetGPUTransportControl(words), {
    departureOutsideBand: 3,
    nonfiniteVelocity: 2,
    processed: 41,
    committed: false,
    extrapolatedVelocity: 17,
    maximumDisplacementFineCells: 5,
    structuredAuthorityUnavailable: 7,
    velocityUnavailable: 11,
    invalidVelocityStatus: 13,
    nonpositiveVelocityResult: 19,
    velocityStatusReasonOr: 0x0120_0004,
    firstInvalidVelocityStatus: 0x0800_0004,
    firstInvalidVelocityLocalIndex: 23,
    firstInvalidVelocityPosition: [1.25, 2.5, 3.75],
  });
});

test("fine topology takes the larger trajectory/redistance radius plus the paper one-ring", () => {
  assert.deepEqual(planFineLevelSetTopologyBand(4, {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 32,
  }), {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 32,
    safetyBrickRings: 1,
    requiredFineCells: 32,
    dilationBrickRings: 9,
  });
  assert.throws(() => planFineLevelSetTopologyBand(4, {
    maximumBacktraceFineCells: 1, interpolationSupportFineCells: 1,
    redistanceBandFineCells: 4, safetyBrickRings: 0,
  }), /at least one publication safety ring/);
});

test("fine page delta publishes exact XOR keys and separate dirty/JFA-support sets", () => {
  assert.deepEqual(planFineLevelSetPageDeltaLayout(7), {
    headerWords: 16,
    changedKeysOffsetWords: 16,
    dirtyPagesOffsetWords: 30,
    supportPagesOffsetWords: 37,
    desiredKeysOffsetWords: 44,
    addedPagesOffsetWords: 51,
    retiredPagesOffsetWords: 58,
    rollbackPagesOffsetWords: 65,
    changedCandidatesOffsetWords: 72,
    dirtyCandidatesOffsetWords: 86,
    supportCandidatesOffsetWords: 93,
    identityScanScratchOffsetWords: 100,
    identityScanBlockWords: 1,
    identityScanSuperBlockWords: 1,
    lifecycleDispatchOffsetWords: 108,
    totalWords: 129,
    totalBytes: 516,
  });
  const shader = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  ).replace(/\s+/g, "");
  assert.match(shader,
    /letold=carriedPage\(key\);letid=work;letbase=id\*10u;if\(old!=INVALID\)\{letoldBase=old\*10u;for\(varword=0u;word<10u;word\+=1u\)\{sourceC\[base\+word\]=currentMetadata\[oldBase\+word\];\}/,
    "carried keys gather into compact Morton-rank physical identities");
  assert.match(shader, /sourceD\[7u\+work\]=id;sourceD\[7u\+params\.pageCapacity\+key\]=id/,
    "compact worklist and direct logical directory publish together");
  assert.match(shader, /pageDelta\[changedCandidatesOffset\(\)\+control\[2\]\+rank\]=currentMetadata\[retired\*10u\+1u\]/,
    "retired current-only pages compact into deterministic fixed records");
  assert.doesNotMatch(shader, /pageAlreadyAssigned|@compute@workgroup_size\(1\)fnmergeDesiredPageIdentities/,
    "identity assignment has no singleton full-capacity fallback");
  assert.match(shader, /@compute@workgroup_size\(256\)fnscanIdentityRecords/,
    "identity records use a parallel fixed-width scan");
  assert.match(shader, /if\(carriedPage\(key\)==INVALID\)\{pageDelta\[changedCandidatesOffset\(\)\+item\]=key;\}/,
    "new desired pages occupy deterministic fixed candidate slots");
  assert.match(shader, /producerChangedContains\(key\)\)\{pageDelta\[changedCandidatesOffset\(\)\+work\]=key;\}/,
    "stable pages consume the transport producer's compact interface-membership delta");
  assert.doesNotMatch(shader, /transportedPayloadChanged|for\(varlocal=0u;local<params\.samplesPerBrick/,
    "topology no longer rescans every transported sample to rediscover dirty pages");
  assert.match(shader, /fncompactFineChangedKeys[\s\S]*identityTotal\(0u,desired\)/,
    "changed keys use the fixed identity scan rather than all-pairs ranking");
  assert.match(shader,
    /fnchangedNeighborRadii[\s\S]*for\(varitem=0u;item<total&&\(dirty==0u\|\|support==0u\)[\s\S]*distance<=i32\(params\.dirtyHaloRings\)[\s\S]*distance<=i32\(params\.supportHaloRings\)/,
    "the changed-prefix walk continues past a wider-support hit until the narrower dirty band is known");
  assert.doesNotMatch(shader, /for\(varz=-radius[\s\S]*sortedChangedContains/,
    "affected-page classification never searches an entire halo cube per resident page");
  assert.match(shader,
    /fncompactFineAffectedPages[\s\S]*identityPrefix\(0u,item\)[\s\S]*identityPrefix\(1u,item\)[\s\S]*identityPrefix\(2u,item\)/,
    "dirty, support, and rollback records use parallel stable-scan ranks");
  assert.match(shader,
    /fnfinalizeFinePageDelta[\s\S]*pageDelta\[0\]=dirty\+retired[\s\S]*pageDelta\[13\]=dirty/,
    "the summary transaction must cover every page whose phi fixed-pass JFA-CPT rewrites");
  assert.match(shader,
    /fnpublishFineSummaryChangedKeys[\s\S]*changedKeysOffset\(\)\+item[\s\S]*changedKeysOffset\(\)\+dirty\+item/,
    "dirty target keys and retired source keys publish as separate bounded streams");
  assert.doesNotMatch(shader,
    /fnchangedKeyAffects|for\(varitem=0u;item<changed;item\+=1u\)|fnfinalizeFinePageDelta[\s\S]*for\(varitem=/,
    "changed-by-desired search and singleton affected-list compaction are deleted");
  assert.doesNotMatch(shader,
    /markFineChangedBrickMask|dilateFineDirtyBricks|dilateFineSupportBricks|scanFineAffectedRecords|offsetFineAffectedRecords/,
    "the duplicate full-logical masks and capacity-wide affected scans stay deleted");
  assert.doesNotMatch(shader, /pageDelta:array<atomic<u32>>|appendChangedKey|markDeltaPage|StampsOffset/,
    "delta publication has no append counter or generation-stamp atomic backing");
  assert.doesNotMatch(shader, /snapshotCurrentPayload|assignDesiredPages|restoreUntouchedSamples/,
    "the recurring full-capacity snapshot/assignment/carry path is deleted");
  assert.match(shader, /fnsettleFinePublication[\s\S]*dirtyPagesOffset\(\)\+work[\s\S]*committedPhi\[index\]/,
    "only exact dirty pages update the committed signed-distance payload");
  assert.match(shader,
    /fnfinalizeFinePageDelta[\s\S]*writeIndirectDispatch\(5u,support\)[\s\S]*writeIndirectDispatch\(7u,dirty\)/,
    "topology must publish exact immutable support and dirty commands for redistance");
  assert.doesNotMatch(shader, /writeDeltaDispatch\(4u,dirty\)|writeDeltaDispatch\(7u,support\)/,
    "the writable page-delta header must not retain duplicate indirect command records");
});

test("fine topology isolates cold closure and publishes recurring sparse deltas", () => {
  const wgsl = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  ).replace(/\s+/g, "");
  const encode = WebGPUFineLevelSetTopology.prototype.encode.toString().replace(/\s+/g, "");
  const topologyHost = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-topology.ts", import.meta.url), "utf8",
  ).replace(/\s+/g, "");
  const leafSeedHost = topologyHost.slice(topologyHost.indexOf("exportclassWebGPUFineLevelSetLeafSeeds"),
    topologyHost.indexOf("exportfunctionunpackFineLevelSetGPUTopologyControl"));

  assert.doesNotMatch(wgsl, /@compute@workgroup_size\(1\)fndiscoverDesired/,
    "interface discovery must not scan every resident page in one invocation");
  assert.match(wgsl, /@compute@workgroup_size\(64\)fnclearDesiredGeneration/);
  assert.match(wgsl, /@compute@workgroup_size\(64\)fndiscoverInterfaceBricks/);
  assert.match(wgsl, /@compute@workgroup_size\(64\)fninsertExternalSeeds/);
  assert.match(wgsl, /fnsortSparseCandidates[\s\S]*workgroupBarrier/,
    "each bounded expansion is globally sorted before publication");
  assert.match(wgsl,
    /fnexpandedAxisKey[\s\S]*owner=item\/3u[\s\S]*fnexpandSparseDesiredX[\s\S]*fnexpandSparseDesiredY[\s\S]*fnexpandSparseDesiredZ/,
    "exact neighbor closure emits only three bounded records per resident key and axis");
  assert.match(wgsl,
    /fnsparseCandidateRunStart[\s\S]*fncompactSparseDesiredBricks[\s\S]*sparseCandidateRunStart\(item\)/,
    "sorted duplicate runs compact deterministically without claims");
  assert.match(wgsl,
    /fnpublishDesiredBricks[\s\S]*targetB\[7u\+work\]=sparseCandidates\[params\.sparseCandidateCapacity\+work\]/,
    "the already sorted compact result publishes directly into the canonical worklist");
  assert.match(wgsl,
    /fnpublishRecurringSparseBand[\s\S]*transportDelta\[1\]!=params\.currentGeneration[\s\S]*transportDelta\[3\]!=currentWorklist\[1\]/,
    "recurring topology fail-closes on an invalid compact producer authority");
  assert.match(wgsl,
    /fnpublishRecurringSparseBand[\s\S]*recurringProducerChanged\(item\)[\s\S]*currentMetadata\[id\*10u\+3u\]&2u[\s\S]*interfaceKey=key/,
    "recurring seeds are the compact producer-changed current-interface set");
  assert.match(wgsl,
    /fnrecurringScatterMembership[\s\S]*atomicOr\(&topologyErrors\[output\],2u\|select\(0u,1u,exact\)\)[\s\S]*fnscanRecurringDesiredRecords[\s\S]*scanIdentityBlock[\s\S]*desiredScan\[item\]=prefix[\s\S]*fnscatterRecurringSparseBand[\s\S]*output=desiredScan\[key\][\s\S]*targetB\[7u\+output\]=key/,
    "compact seeds scatter bounded halos, rank in parallel, and publish canonical keys exactly once");
  assert.match(wgsl,
    /fnscanRecurringDesiredRecords[\s\S]*wid\.x\*256u\+local[\s\S]*fnfinalizeRecurringSparseBand[\s\S]*recurringDesiredTotal\(\)[\s\S]*fnscatterRecurringSparseBand/,
    "all logical-domain sizes use the same hierarchical mark/rank/scatter publication");
  assert.doesNotMatch(wgsl, /fnrecurringSort|fnrecurringCompact|fnrecurringExpandedAxisKey/,
    "recurring topology must not retain a capacity-wide ordering branch");
  assert.match(wgsl,
    /recurringSeedLanes\[local\]=localSeeds[\s\S]*recurringSeedLanes\[local\]\+=recurringSeedLanes\[local\+width\][\s\S]*control\[1\]=seeds;control\[8\]=seeds/,
    "the exact producer-changed interface prefix is captured before sparse dilation");
  assert.doesNotMatch(wgsl,
    /publishRecurringDesiredBricks|recurringCandidate|recurringKeyAffected|recurringDesiredNow|recurringShiftedKey|sparseFringe|seedWithinDilation|topologySourceLookup|recurringSeedMembership|finalizeDesiredBricks|for\(varother=0u;other<count;other\+=1u\)/,
    "the scalar resident walk, per-output membership search, old full fringe, and quadratic rank sort are deleted");
  assert.match(wgsl, /topologyErrors:array<atomic<u32>>/,
    "the direct identity mask uses a dedicated atomic word per logical brick");
  assert.match(wgsl, /atomicStore\(&topologyErrors\[item\],0u\)[\s\S]*recurringScatterMembership/,
    "membership storage is deterministically reset before compact seeds scatter");

  assert.match(encode, /dilationBrickRings=bandPlan\.dilationBrickRings/,
    "topology must allocate transport, interpolation, redistance, and safety support before redistance");
  assert.match(encode,
    /this\.scanSparseSeedRecordsPipeline[\s\S]*this\.sortSparseCandidatesPipeline[\s\S]*this\.compactSparseDesiredPipeline[\s\S]*this\.expandSparseDesiredPipelines[\s\S]*this\.publishDesiredBricksPipeline/,
    "cold bootstrap retains deterministic sorted seeds and bounded axis expansion");
  assert.match(encode,
    /if\(publication\.kind==="bootstrap"\)[\s\S]*else\{[\s\S]*this\.publishRecurringSparseBandPipeline[\s\S]*this\.scanRecurringDesiredPipeline[\s\S]*this\.scanSparseGroupsPipeline[\s\S]*this\.finalizeRecurringSparseBandPipeline[\s\S]*this\.scatterRecurringSparseBandPipeline/,
    "recurring publication has a separate parallel mark/rank/scatter schedule");
  assert.doesNotMatch(encode,
    /compactRecurringExpansion|Expandrecurringglobalfinetopology|Publishsortedrecurringglobalfinetopology/,
    "recurring sparse closure cannot expand into a dispatch-per-ring schedule");
  assert.doesNotMatch(encode,
    /clearDesiredMaskPipeline|markDesiredSeedsPipeline|scanDesiredClosureRecordsPipeline|dilateDesired[XYZ]Pipeline/,
    "the recurring whole-lattice host schedule has no retained switch or fallback");
  assert.doesNotMatch(encode, /beginComputePass|GPUCommandEncoder/,
    "topology compute-pass ownership is cut over completely to the caller's broker");
  assert.match(encode, /broker\.compute/,
    "topology requests compute work from the caller's shared broker");
  assert.doesNotMatch(encode, /createBindGroup/,
    "recurring topology encoding must reuse immutable resource-tuple bindings");
  assert.doesNotMatch(encode, /updateIndirectBuffer\(this\.pageDelta|copyBufferToBuffer/,
    "page-delta dispatches are authored directly without copy-induced pass breaks");
  assert.match(encode,
    /broker\.fence\("finechanged-keydispatchpublication"\)[\s\S]*runIndirect\(this\.classifyAffectedPagesPipeline[\s\S]*?,3,\[0,3,4,7,15,16,21\]\)/,
    "one storage-to-indirect boundary publishes the exact dirty/support classifier");
  assert.doesNotMatch(encode,
    /clearAffectedCandidatesPipeline|markRetiredRollbackPipeline|dilateDirty|dilateSupport|scanAffectedRecordsPipeline/,
    "the retired full-mask and capacity-scan host schedule stays deleted");
  assert.doesNotMatch(encode, /dispatchWorkgroupsIndirect\(this\.pageDelta/,
    "a writable storage transaction can never alias an indirect argument in one synchronization scope");
  const finalize = WebGPUFineLevelSetTopology.prototype.encodeFinalizePublication.toString().replace(/\s+/g, "");
  assert.doesNotMatch(finalize, /createBindGroup/,
    "deferred publication finalization must reuse immutable resource-tuple bindings");
  assert.match(topologyHost,
    /this\.bindGroupCache\.find[\s\S]*entry\.binding===selected\[index\]!\.binding[\s\S]*sameResource/,
    "the cache key must include pipeline, binding number, buffer identity, offset, and size");
  assert.match(leafSeedHost,
    /this\.bindGroupCache\.find[\s\S]*candidate\.entryPoint===entryPoint[\s\S]*sameResource[\s\S]*this\.bindGroupCache\.push/,
    "fine seed publication must cache immutable resource tuples instead of allocating groups per encode");
  assert.doesNotMatch(WebGPUFineLevelSetLeafSeeds.prototype.encode.toString()
    + WebGPUFineLevelSetLeafSeeds.prototype.encodeFromAllInterfaceLeaves.toString(), /createBindGroup/);
  assert.doesNotMatch(finalize, /beginComputePass|GPUCommandEncoder/,
    "publication finalization cannot retain standalone pass ownership");
  assert.match(finalize,
    /constpass=broker\.compute[\s\S]*broker\.fence\("finedeferredsettlementdispatchpublication"\)[\s\S]*dispatchWorkgroupsIndirect\(this\.indirectDispatch,0\)[\s\S]*broker\.fence\("globalfinetopologypublicationcomplete"\)/,
    "publication validation authors one exact settlement dispatch before the semantic publication fence");
});

test("direct recurring identity scatter exactly matches per-output membership", () => {
  const topologySource = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-topology.ts", import.meta.url), "utf8",
  );
  assert.doesNotMatch(topologySource, /clearFinePageDirectory|logicalBrickCount\s*\/\s*64/,
    "recurring publication must not clear the logical-directory capacity");
  assert.match(topologySource,
    /fn targetLookup[\s\S]*sourceC\[base\+2u\]==params\.nextGeneration/,
    "stale A\/B directory words are rejected by the assigned page generation");
  const fixtures = [
    { dimensions: [24, 18, 16] as const, radius: 1,
      seeds: [0, 1, 1, 23, 24, 431, 432, 3455, 6911] },
    { dimensions: [7, 5, 3] as const, radius: 2,
      seeds: [0, 6, 34, 35, 52, 104, 104] },
    { dimensions: [2, 2, 2] as const, radius: 4,
      seeds: [0, 7] },
  ] as const;
  for (const fixture of fixtures) {
    const [nx, ny, nz] = fixture.dimensions;
    const logical = nx * ny * nz;
    const unpack = (key: number) => {
      const z = Math.floor(key / (nx * ny));
      const remainder = key - z * nx * ny;
      const y = Math.floor(remainder / nx);
      return [remainder - y * nx, y, z] as const;
    };
    const uniqueSeeds = new Set(fixture.seeds.filter((key) => key >= 0 && key < logical));
    const reference: number[] = [];
    for (let key = 0; key < logical; key += 1) {
      const point = unpack(key);
      if ([...uniqueSeeds].some((seed) => {
        const source = unpack(seed);
        return Math.max(
          Math.abs(point[0] - source[0]),
          Math.abs(point[1] - source[1]),
          Math.abs(point[2] - source[2]),
        ) <= fixture.radius;
      })) reference.push(key);
    }
    const mask = new Uint8Array(logical);
    for (const seed of fixture.seeds) {
      if (seed < 0 || seed >= logical) continue;
      const [sx, sy, sz] = unpack(seed);
      for (let z = -fixture.radius; z <= fixture.radius; z += 1) {
        for (let y = -fixture.radius; y <= fixture.radius; y += 1) {
          for (let x = -fixture.radius; x <= fixture.radius; x += 1) {
            const qx = sx + x, qy = sy + y, qz = sz + z;
            if (qx < 0 || qx >= nx || qy < 0 || qy >= ny || qz < 0 || qz >= nz) continue;
            mask[qx + nx * (qy + ny * qz)] |= 2 | Number(x === 0 && y === 0 && z === 0);
          }
        }
      }
    }
    const scattered = [...mask.keys()].filter((key) => (mask[key] & 2) !== 0);
    assert.deepEqual(scattered, reference,
      `direct scatter must preserve exact radius-${fixture.radius} membership`);
    assert.equal([...mask].filter((value) => (value & 1) !== 0).length, uniqueSeeds.size,
      "idempotent exact-seed bits must deduplicate repeated producer identities");
  }
});

test("B4 leaf seeds cover all eight factor-8 bricks per finest cell", () => {
  const factor4 = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 5, 4],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });
  const factor8 = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 5, 4],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks: 8 });
  assert.deepEqual(planFineLevelSetLeafBrickBounds(factor4, [2, 1, 1], 1), {
    first: [2, 1, 1], last: [2, 1, 1], bricksPerFinestCell: 1, brickCount: 1,
  });
  assert.deepEqual(planFineLevelSetLeafBrickBounds(factor8, [2, 1, 1], 1), {
    first: [4, 2, 2], last: [5, 3, 3], bricksPerFinestCell: 2, brickCount: 8,
  });
  assert.deepEqual(planFineLevelSetLeafBrickBounds(factor8, [1, 1, 1], 2), {
    first: [2, 2, 2], last: [5, 5, 5], bricksPerFinestCell: 2, brickCount: 64,
  });
  assert.throws(() => planFineLevelSetLeafBrickBounds(factor8, [5, 4, 3], 2), /outside/);

  assert.match(fineLevelSetLeafSeedWGSL,
    /fn leafFirst\(leaf:FineSeedLeaf\)->vec3u\{return leafOrigin\(leaf\)\*params\.header\.x\/params\.header\.y;\}[\s\S]*return min\(high\/params\.header\.y/,
    "seed publication must map leaf bounds by fineFactor / brickResolution on every axis");
  assert.match(fineLevelSetLeafSeedWGSL,
    /fn recordGreater[\s\S]*fn sortSeedRecords[\s\S]*for\(var width=2u;width<=count;width<<=1u\)/,
    "expanded leaf records must use the bounded cooperative key/owner ordering transaction");
  assert.match(fineLevelSetLeafSeedWGSL,
    /fn runStart\(index:u32\)->bool[\s\S]*fn emitSeedRuns[\s\S]*seeds\[4u\+output\]=record\.x/,
    "run-boundary compaction must publish one sorted record per unique brick key");
  assert.doesNotMatch(fineLevelSetLeafSeedWGSL,
    /claimLeaf|scanLeafSeed|emitLeaf|seedHash|insertHash|atomic(?:Load|Store|Add|Or|Min|Max|Exchange|CompareExchangeWeak)|atomic<u32>/,
    "leaf seed publication must not retain the legacy claim/hash-CAS/atomic scan path");
});

test("fine topology telemetry labels exact requirements and overflow lower bounds", () => {
  assert.throws(() => unpackFineLevelSetGPUTopologyControl([0, 7, 35, 35, 1, 0, 7, 0]),
    /requires nine words/, "the retired eight-word diagnostic ABI must stay deleted");
  const published = unpackFineLevelSetGPUTopologyControl([0, 7, 35, 35, 1, 0, 7, 0, 11]);
  assert.equal(published.requiredDesiredBricks, 35);
  assert.equal(published.requiredDesiredBricksExact, true);
  assert.equal(published.dilationBrickRings, 7);
  assert.equal(published.downstreamFinalizeReason, 0);
  assert.equal(published.interfaceSeedBricks, 11);
  const overflow = unpackFineLevelSetGPUTopologyControl([1, 7, 32, 0, 1, 1, 33, 0, 7]);
  assert.equal(overflow.desiredBricks, 32);
  assert.equal(overflow.requiredDesiredBricks, 33);
  assert.equal(overflow.requiredDesiredBricksExact, false);
  assert.equal(overflow.dilationBrickRings, 0);
  const downstream = unpackFineLevelSetGPUTopologyControl([16, 7, 32, 32, 1, 1, 0,
    FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON.redistance | FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON.volume, 7]);
  assert.equal(downstream.downstreamFinalizeReason, 6);
});

test("fine topology rollback snapshot cannot alias Section 5 transport or closest-point scratch", () => {
  const constructor = WebGPUFineLevelSetTopology.toString().replace(/\s+/g, "");
  const encode = WebGPUFineLevelSetTopology.prototype.encode.toString().replace(/\s+/g, "");
  const finalize = WebGPUFineLevelSetTopology.prototype.encodeFinalizePublication.toString().replace(/\s+/g, "");
  assert.match(constructor,
    /current\.flags===next\.flags[\s\S]*current\.workB===next\.workB[\s\S]*current\.rollbackPhi===next\.rollbackPhi[\s\S]*current\.rollbackPhi===current\.workA[\s\S]*current\.rollbackPhi===current\.workB/,
    "A/B topology must reject an aliased or generation-local rollback buffer");
  assert.match(encode, /binding:17,resource:resource\(this\.next\.rollbackPhi\)/,
    "the candidate generation writes its dedicated committed signed-distance channel");
  assert.match(encode, /binding:32,resource:resource\(this\.current\.rollbackPhi\)/,
    "rejection reads the accepted generation's protected signed-distance channel");
  assert.doesNotMatch(encode, /binding:10,resource:resource\(this\.current\.work[AB]\)/,
    "transport/distance/request scratch cannot carry rollback authority");
  assert.match(finalize, /binding:17,resource:resource\(this\.next\.rollbackPhi\)[\s\S]*binding:32,resource:resource\(this\.current\.rollbackPhi\)/,
    "a rejected downstream generation must restore the protected signed snapshot");
});

test("fine topology keeps cold failure unpublished and confines affine seeds to bootstrap", () => {
  const shader = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  ).replace(/\s+/g, "");
  assert.match(shader,
    /fncurrentFinePopulated\(\)->bool\{returncurrentFinePublished\(\)&&currentWorklist\[1\]>0u;\}/,
    "a structurally published empty A/B slot remains a cold bootstrap source");
  assert.match(shader,
    /fninsertExternalSeeds[\s\S]*if\(currentFinePopulated\(\)\|\|arrayLength\(&externalSeeds\)<4u\)\{return;\}/,
    "a published generation rejects the bootstrap-only external seed stream");
  assert.match(shader,
    /fnexternalSeedPhi[\s\S]*params\.affineSeeds==0u\|\|currentFinePopulated\(\)[\s\S]*return3\.402823e38/,
    "recurring pages initialize from coarse phi rather than a compact affine bootstrap plane");
  assert.match(shader,
    /fnexternalAffineInterfaceBrick[\s\S]*for\(varcorner=0u;corner<8u;corner\+=1u\)[\s\S]*returnminimum<=0\.0&&maximum>=0\.0/,
    "cold affine keys enter the pre-dilation prefix only when their page contains a zero crossing");
  assert.match(shader,
    /letfirst=vec3f\(brick\*params\.brickResolution\)\/f32\(params\.fineFactor\);letlast=vec3f\(\(brick\+vec3u\(1u\)\)\*params\.brickResolution\)\/f32\(params\.fineFactor\)/,
    "interface admission must include shared page boundaries instead of testing sample centres only");
  const alignedFace = 1;
  const centreBounds = [[0.125, 0.875], [1.125, 1.875]] as const;
  const supportBounds = [[0, 1], [1, 2]] as const;
  assert.ok(centreBounds.every(([first, last]) => !((first - alignedFace) <= 0 && (last - alignedFace) >= 0)),
    "the regression fixture must reproduce centre-only rejection on both sides of an aligned face");
  assert.ok(supportBounds.every(([first, last]) => (first - alignedFace) <= 0 && (last - alignedFace) >= 0),
    "both page supports must admit their shared zero face for deterministic one-ring ownership");
  assert.match(shader,
    /letkey=externalSeeds\[4u\+seed\];if\(!externalAffineInterfaceBrick\(key\)\)\{return;\}/,
    "the published-empty t0 path cannot collapse to a zero interface prefix or seed the entire affine leaf set");
  assert.match(shader,
    /letcurrentCount=select\(0u,min\(currentWorklist\[1\],params\.pageCapacity\),currentPublished\)[\s\S]*sourceD\[3\]=select\(0u,3u,currentPublished\);sourceD\[4\]=\(currentCount\+63u\)\/64u;sourceD\[5\]=1u;sourceD\[6\]=1u;control\[4\]=select\(0u,1u,currentPublished\);control\[5\]=1u/,
    "parallel rollback keeps a cold failure unpublished and only republishes a validated current generation");
  assert.match(shader,
    /fnrecurringScatterMembership[\s\S]*atomicOr\(&topologyErrors\[output\],[\s\S]*fnscanRecurringDesiredRecords[\s\S]*scanIdentityBlock[\s\S]*fnscatterRecurringSparseBand[\s\S]*atomicLoad\(&topologyErrors\[key\]\)/,
    "recurring topology uses bounded idempotent marks and hierarchical ranks instead of ordering or membership searches");
});

test("fine topology binds exactly the resources reachable from every compute entry point", () => {
  const shader = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  );
  const bindings = new Map<string, number>();
  for (const match of shader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )) bindings.set(match[2], Number(match[1]));

  const bodies = new Map<string, string>();
  for (const match of shader.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = shader.indexOf("{", match.index);
    assert.notEqual(open, -1, `function ${match[1]} must have a body`);
    let depth = 0; let close = -1;
    for (let at = open; at < shader.length; at += 1) {
      if (shader[at] === "{") depth += 1;
      if (shader[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.notEqual(close, -1, `function ${match[1]} must close its body`);
    bodies.set(match[1], shader.slice(open + 1, close));
  }
  const entryPoints = [...shader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]);
  const reachableBindings = (entryPoint: string): number[] => {
    const pending = [entryPoint]; const reached = new Set<string>(); const used = new Set<number>();
    while (pending.length > 0) {
      const name = pending.pop()!;
      if (reached.has(name)) continue;
      reached.add(name);
      const body = bodies.get(name);
      assert.notEqual(body, undefined, `reachable function ${name} must exist`);
      for (const [global, binding] of bindings) {
        if (new RegExp(`\\b${global}\\b`).test(body!)) used.add(binding);
      }
      for (const callee of bodies.keys()) {
        if (!reached.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
      }
    }
    return [...used].sort((a, b) => a - b);
  };

  const observed = new Map<string, number[]>();
  const buffer = {} as GPUBuffer;
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {} },
    createBuffer: () => ({}),
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({ entryPoint: compute.entryPoint }),
    }),
    createBindGroup: ({ layout, entries }: { layout: { entryPoint: string }; entries: { binding: number }[] }) => {
      const actual = entries.map(({ binding }) => binding).sort((a, b) => a - b);
      const prior = observed.get(layout.entryPoint);
      if (prior) assert.deepEqual(actual, prior, `${layout.entryPoint} bind contract changed between encodes`);
      observed.set(layout.entryPoint, actual); return {};
    },
  } as unknown as GPUDevice;
  const currentPayload = {
    flags: { kind: "flags" } as unknown as GPUBuffer,
    phi: { kind: "phi" } as unknown as GPUBuffer,
    workA: { kind: "workA" } as unknown as GPUBuffer,
    workB: { kind: "workB" } as unknown as GPUBuffer,
    rollbackPhi: { kind: "rollbackPhi" } as unknown as GPUBuffer,
  };
  const nextPayload = {
    flags: { kind: "nextFlags" } as unknown as GPUBuffer,
    phi: { kind: "nextPhi" } as unknown as GPUBuffer,
    workA: { kind: "nextWorkA" } as unknown as GPUBuffer,
    workB: { kind: "nextWorkB" } as unknown as GPUBuffer,
    rollbackPhi: { kind: "nextRollbackPhi" } as unknown as GPUBuffer,
  };
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [1, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 1 });
  const current = { ...currentPayload, generation: 1, generationSlot: 0 as const, params: buffer,
    metadata: buffer, worklist: buffer, plan };
  const next = { ...nextPayload, generation: 2, generationSlot: 1 as const, params: buffer,
    metadata: buffer, worklist: buffer, plan };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {},
    dispatchWorkgroupsIndirect() {}, end() {} };
  const encoder = { beginComputePass: () => pass, copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const topology = new WebGPUFineLevelSetTopology(device, current, next,
      "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}");
    const broker = new PassBroker(encoder);
    topology.encode(broker);
    topology.encodeFinalizePublication(broker, { redistance: buffer });
    const largePlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
      finestCellDimensions: [17, 17, 17], finestCellWidth: 1, fineFactor: 4,
      brickResolution: 4, maximumResidentBricks: 1 });
    const largeCurrent = { ...current, plan: largePlan };
    const largeNext = { ...next, plan: largePlan };
    new WebGPUFineLevelSetTopology(device, largeCurrent, largeNext,
      "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}").encode(broker);
    topology.encode(broker, undefined, [], undefined, false, {
      kind: "delta",
      producer: { buffer, pageCapacity: 1, candidateKeysOffsetWords: 8,
        changedKeysOffsetWords: 9 },
    });
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  assert.deepEqual([...observed.keys()].sort(), [...entryPoints].sort(),
    "host encoding must exercise every topology compute entry point");
  for (const entryPoint of entryPoints) {
    assert.deepEqual(observed.get(entryPoint), reachableBindings(entryPoint),
      `${entryPoint} bind group must equal its transitive WGSL resource reachability`);
  }
});

test("fine-brick sampling WGSL uses a flat direct lookup, exact generation validation, and explicit coarse authority", () => {
  assert.match(fineLevelSetBrickSamplingWGSL, /let directoryBase=7u\+params\.worklistCapacity/);
  assert.match(fineLevelSetBrickSamplingWGSL, /worklist\[0\]!=params\.generation/);
  assert.match(fineLevelSetBrickSamplingWGSL,
    /let physicalId=worklist\[directoryBase\+key\];let base=physicalId\*10u/);
  assert.match(fineLevelSetBrickSamplingWGSL,
    /metadata\[base\]==physicalId&&metadata\[base\+1u\]==key&&metadata\[base\+2u\]==params\.generation/);
  assert.match(fineLevelSetBrickSamplingWGSL, /Result\(coarsePhi,0u/);
  assert.doesNotMatch(fineLevelSetBrickSamplingWGSL, /while|binary|middle|low<high|hash|probe/i);
  assert.doesNotMatch(fineLevelSetBrickSamplingWGSL, /octree.*row/i);
});

test("fine redistance applies its inclusive residual tolerance at telemetry precision", () => {
  assert.match(fineLevelSetJFACPTWGSL,
    /residual=u32\([\s\S]*unresolved=select\(0u,1u,residual>u32\(p\.tolerance\*1000000\.\)\)/);
  assert.match(fineLevelSetJFACPTWGSL,
    /finalizeJFADistances[\s\S]*control\.residualScaled=reduceMaximum\[0\]/);
  assert.doesNotMatch(fineLevelSetJFACPTWGSL, /if\(residual>p\.tolerance\)/);
  assert.doesNotMatch(fineLevelSetJFACPTWGSL,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "JFA-CPT recurring control and diagnostic publication must be deterministic and atomic-free");
});

test("fine redistance construction requires the topology-authored delta ABI", () => {
  const buffer = {} as GPUBuffer;
  const source = {
    plan: { maximumResidentBricks: 7 },
  } as unknown as ConstructorParameters<typeof WebGPUFineLevelSetRedistance>[1];
  const malformed = {
    pageDelta: buffer,
    pageDeltaLayout: {
      headerWords: 16 as const,
      dirtyPagesOffsetWords: 16,
      supportPagesOffsetWords: 23,
    },
    redistanceDispatches: {
      buffer,
      dirtyOffsetBytes: 84 as const,
      supportOffsetBytes: 60 as const,
    },
  };
  assert.throws(() => new WebGPUFineLevelSetRedistance(
    {} as GPUDevice, source, malformed,
  ), /exact topology page-delta ABI/);
});

test("fine redistance is fixed-pass JFA-CPT", () => {
  assert.deepEqual(planFineLevelSetJFAStrides(21), [4, 2, 1, 1, 1]);
  assert.deepEqual(planFineLevelSetJFAStrides(21, 21), [16, 8, 4, 2, 1, 1, 1]);
  assert.deepEqual(planFineLevelSetJFAStrides(1), [1, 1, 1]);
  assert.deepEqual(planFineLevelSetJFAStrides(2), [2, 1, 1, 1]);
  assert.match(fineLevelSetJFACPTWGSL,
    /otherDistance<bestDistance\|\|\(otherDistance==bestDistance&&otherKey<bestKey\)/,
    "equal-distance cooperative reduction must choose the stable global sample key");
  assert.match(fineLevelSetJFACPTWGSL,
    /subgroupShuffleDown\(bestDistance,width\)[\s\S]*subgroupShuffleDown\(bestSeed,width\)/,
    "candidate distance and seed must be evaluated once and reduced in subgroup registers");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn resolvedDistance\(seed:u32,q:vec3u\)->f32\{if\(seed==INVALID\)\{return bandDistance\(\);\}return length/,
    "a reachable distance must remain unclamped so beyond-band seeds cannot masquerade as exact-cutoff samples");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn distanceValue\(index:u32\)->f32\{return bitcast<f32>\(workB\[index\]\);\}\s*fn resolvedSeed\(index:u32\)->u32\{return workA\[index\];\}/,
    "persistent delta state must have one parity-independent seed/distance channel contract");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL, /distanceInB/,
    "recurring JFA must not reinterpret a prior generation's distance bits as seed indices");
  assert.match(fineLevelSetJFACPTWGSL,
    /resolveClosestPointsAToB[\s\S]*var seed=workA\[index\][\s\S]*workA\[index\]=seed;workB\[index\]=bitcast<u32>\(d\)/,
    "an even flood schedule must preserve A seeds while publishing distances into B");
  assert.match(fineLevelSetJFACPTWGSL,
    /resolveClosestPointsBToCanonical[\s\S]*var seed=workB\[index\][\s\S]*workA\[index\]=seed;workB\[index\]=bitcast<u32>\(d\)/,
    "an odd flood schedule must canonicalize B seeds into A before the next delta generation");
  assert.match(fineLevelSetJFACPTWGSL,
    /seedClosestPoints[\s\S]*workA\[index\]=INVALID;workB\[index\]=INVALID;flags\[index\]&=\(1u<<SAMPLE_FLAG_BITS\)-1u/,
    "every recurring generation erases prior scratch while retaining untouched authority bits");
  assert.match(fineLevelSetJFACPTWGSL,
    /let value=bitcast<f32>\(phi\[index\]\);if\(!finite\(value\)\)\{errorFlags\|=NONFINITE/,
    "recurring redistance must reject non-finite transported phi");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL,
    /let value=bitcast<f32>\(phi\[index\]\);if\(\(flags\[index\]&VALID\)==0u/,
    "the prior generation's narrow-band clipping mask must not reject finite support samples");
  assert.match(fineLevelSetJFACPTWGSL,
    /if\(resolvedSeed\(index\)==INVALID\|\|d>bandDistance\(\)\)\{flags\[index\]=0u;\}/,
    "the closed cutoff may publish only a sample reached from a real interface seed");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL, /fn betterSeed/,
    "candidate comparison must evaluate each closest point only once");
  assert.match(fineLevelSetJFACPTWGSL,
    /flags\[index\]\|=closest<<SAMPLE_FLAG_BITS/,
    "seeding materializes subcell closest points without rewriting persistent validity/sign bits");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn materializedClosestPoint\(index:u32\)[^}]*flags\[index\]>>SAMPLE_FLAG_BITS[^}]*CP_FRACTION_MASK/,
    "flood comparisons must read the cached closest point without re-walking neighboring phi");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL,
    /fn materializedClosestPoint\([^}]*sampleIndex|fn materializedClosestPoint\([^}]*pageOf/,
    "cached closest-point lookup must not perform a sparse hash lookup");
  assert.match(fineLevelSetJFACPTWGSL,
    /if\(hasCachedClosestPoint\(index\)\)\{seed=index;\}/,
    "interface seeds must retain subcell zero-crossing samples");
  assert.match(fineLevelSetJFACPTWGSL,
    /unresolved=select\(0u,1u,seed==INVALID&&abs\(transported\)<bandDistance\(\)\)/,
    "a JFA propagation miss inside transported narrow-band support must fail publication");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn sampleIndex\(q:vec3u\)[\s\S]*supportPageOf\(packBrick\(q\/p\.brickResolution\)\)/,
    "all JFA seed/flood/validation neighbors must be restricted to the exact support publication");
  assert.match(fineLevelSetJFACPTWGSL,
    /var<workgroup>floodPageIds:array<u32,27>[\s\S]*fn prepareFloodPageIds[\s\S]*if\(lid<27u\)\{floodPageIds\[lid\]=page;\}workgroupBarrier\(\)[\s\S]*fn cachedFloodSampleIndex[\s\S]*let id=floodPageIds\[/,
    "each JFA workgroup must resolve its 27 generation-fixed support pages once");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL,
    /fn cooperativeFlood\([^}]*sampleIndex\(/,
    "the 27-tap flood must not repeat page-directory resolution in every lane");
  assert.match(fineLevelSetJFACPTWGSL,
    /@compute @workgroup_size\(256\)fn jumpFloodAToB[\s\S]*fn jumpFloodBToA/,
    "each brick flood must expose eight 32-lane candidate teams");
  assert.match(fineLevelSetJFACPTWGSL,
    /let team=lid\/subgroupSize;[\s\S]*let candidateSlot=subgroupLane[\s\S]*candidateSlot<27u[\s\S]*candidateSlot==27u/,
    "the 27 spatial taps and incumbent must occupy one deterministic 32-lane reduction");
  const cooperativeFlood = fineLevelSetJFACPTWGSL.slice(
    fineLevelSetJFACPTWGSL.indexOf("fn cooperativeFlood"),
    fineLevelSetJFACPTWGSL.indexOf("fn resolvedDistance"),
  );
  assert.doesNotMatch(cooperativeFlood,
    /floodCandidate(?:Distance|Key|Seed)|workgroupBarrier/,
    "candidate reduction must not serialize eight teams through workgroup scratch barriers");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn deltaRecordError[\s\S]*publishedPageOf\(key\)!=id[\s\S]*!support&&supportPageOf\(key\)!=id/,
    "dirty/support records must be sorted, published, generation-current, and dirty must be a support subset");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL,
    /fn sampleIndex\(q:vec3u\)[^}]*publishedPageOf/,
    "the full active worklist cannot re-enter recurring JFA through neighbor lookup");
});

test("fixed B4 JFA tap lookup is exhaustive parity with the coordinate formula", () => {
  const coordinate = (index: number) =>
    [index & 3, (index >> 2) & 3, (index >> 4) & 3] as const;
  for (const stride of FINE_LEVELSET_JFA_LOOKUP_STRIDES) {
    const radius = Math.max(1, Math.ceil(stride / 4));
    for (let local = 0; local < 64; local += 1) {
      const q = coordinate(local);
      for (let tap = 0; tap < 27; tap += 1) {
        const tapZ = Math.floor(tap / 9); const remainder = tap - tapZ * 9;
        const tapY = Math.floor(remainder / 3); const tapX = remainder - tapY * 3;
        const displaced = [
          q[0] + (tapX - 1) * stride,
          q[1] + (tapY - 1) * stride,
          q[2] + (tapZ - 1) * stride,
        ];
        const pageDelta = displaced.map((value) => Math.floor(value / 4));
        assert.ok(pageDelta.every((value) => value === 0 || Math.abs(value) === radius),
          `stride ${stride}, local ${local}, tap ${tap} escaped the 27-page halo`);
        const pageCoordinate = pageDelta.map((value) => value < 0 ? 0 : value > 0 ? 2 : 1);
        const localCoordinate = displaced.map((value, axis) => value - pageDelta[axis]! * 4);
        assert.deepEqual(fineLevelSetJFATapAddress(local, tap, stride), {
          pageSlot: pageCoordinate[0]! + 3 * (pageCoordinate[1]! + 3 * pageCoordinate[2]!),
          localIndex: localCoordinate[0]! + 4 * (localCoordinate[1]! + 4 * localCoordinate[2]!),
        }, `stride ${stride}, local ${local}, tap ${tap}`);
      }
    }
  }
  assert.throws(() => fineLevelSetJFATapAddress(-1, 0, 1), /local index/);
  assert.throws(() => fineLevelSetJFATapAddress(0, 27, 1), /tap/);
  assert.throws(() => fineLevelSetJFATapAddress(0, 0, 3), /power of two/);
});

test("JFA fixed tap path consumes topology's exact-filtered radius-one halo", () => {
  assert.match(fineLevelSetJFACPTWGSL,
    /const JFA_TAP_AXIS_CODES:array<u32,27>[\s\S]*const JFA_AXIS_ADDRESSES:array<u32,108>/,
    "the shader lookup must remain the compact separable B4 table");
  assert.match(fineLevelSetJFACPTWGSL,
    /if\(JFA_STRIDE<=4u&&hasRadiusOneHalo\(\)\)\{page=radiusOneHaloPage\(id,lid,key\);\}else\{page=supportPageOf\(key\);\}/,
    "radius-one strides must consume topology's existing halo with an absent-halo fallback");
  assert.match(fineLevelSetJFACPTWGSL,
    /metadata\[base\+1u\]==expectedKey[\s\S]*supportMask\[page\]==p\.generation/,
    "topology halo entries must retain identity and exact support-membership validation");
  const flood = fineLevelSetJFACPTWGSL.slice(
    fineLevelSetJFACPTWGSL.indexOf("fn cooperativeFlood"),
    fineLevelSetJFACPTWGSL.indexOf("fn resolvedDistance"),
  );
  assert.match(flood, /cachedFloodSampleIndex\(local,candidateSlot\)/);
  assert.doesNotMatch(flood, /q\/p\.brickResolution|q%p\.brickResolution|abs\(delta\)/,
    "candidate lanes must not reconstruct the fixed tap address");
});

test("sparse diagonal JFA landing gaps deterministically reject narrow-band publication", () => {
  const brickResolution = 4; const dimensions = [6, 6, 1] as const;
  const nearRing = [[0, 0, 0], [1, 0, 0], [0, 1, 0]] as const;
  const farRing = [[4, 4, 0], [3, 4, 0], [5, 4, 0], [4, 3, 0], [4, 5, 0]] as const;
  const residentBricks = new Set([...nearRing, ...farRing].map((q) => q.join(",")));
  const residentSamples: [number, number, number][] = [];
  for (let z = 0; z < dimensions[2] * brickResolution; z += 1) {
    for (let y = 0; y < dimensions[1] * brickResolution; y += 1) {
      for (let x = 0; x < dimensions[0] * brickResolution; x += 1) {
        if (residentBricks.has([Math.floor(x / 4), Math.floor(y / 4), Math.floor(z / 4)].join(","))) {
          residentSamples.push([x, y, z]);
        }
      }
    }
  }
  const key = (q: readonly number[]) => q.join(",");
  let reachable = new Set(residentSamples
    .filter(([x, y]) => x < 8 && y < 8)
    .map(key));
  for (const stride of planFineLevelSetJFAStrides(7)) {
    const next = new Set(reachable);
    for (const q of residentSamples) {
      let found = false;
      for (let dz = -1; dz <= 1 && !found; dz += 1) {
        for (let dy = -1; dy <= 1 && !found; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (reachable.has(key([q[0] + dx * stride, q[1] + dy * stride, q[2] + dz * stride]))) {
              found = true; break;
            }
          }
        }
      }
      if (found) next.add(key(q));
    }
    reachable = next;
  }
  assert.equal(reachable.has(key([16, 16, 0])), false,
    "the disconnected diagonal island has no resident landing chain from the interface seeds");
  assert.match(fineLevelSetJFACPTWGSL,
    /unresolved=select\(0u,1u,seed==INVALID&&abs\(transported\)<bandDistance\(\)\)/,
    "an unreachable transported narrow-band sample must reject finalization");
  assert.match(fineLevelSetJFACPTWGSL,
    /control\.committed=select\(0u,1u,control\.flags==0u&&control\.unresolved==0u/,
    "publication cannot commit while a sparse propagation gap is unresolved");
});

test("every constructed fine redistance pipeline names an existing WGSL compute entry point", () => {
  const jfaEntryPoints = [...WebGPUFineLevelSetRedistance.toString()
    .matchAll(/jfaPipeline\("([A-Za-z0-9_]+)"/g)].map((match) => match[1]);
  const jfaShaderEntryPoints = [...fineLevelSetJFACPTWGSL
    .matchAll(/@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z0-9_]+)/g)].map((match) => match[1]);
  assert.deepEqual(new Set(jfaEntryPoints), new Set(jfaShaderEntryPoints));
});

test("fine redistance binds exactly the resources reachable from each compute entry point", () => {
  const observed = new Map<string, Set<number>>();
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {} },
    createBuffer: () => ({}),
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({ entryPoint: compute.entryPoint }),
    }),
    createBindGroup: ({ layout, entries }: { layout: { entryPoint: string }; entries: { binding: number }[] }) => {
      observed.set(layout.entryPoint, new Set(entries.map(({ binding }) => binding)));
      return {};
    },
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const source = {
    generation: 1,
    metadata: buffer, worklist: buffer, flags: buffer, phi: buffer, workA: buffer, workB: buffer,
    rollbackPhi: buffer,
    plan: {
      fineFactor: 4, brickResolution: 4, brickDimensions: [1, 1, 1], sampleDimensions: [4, 4, 4],
      samplesPerBrick: 64, maximumResidentBricks: 1,
      fineCellWidth: 0.25,
    },
  };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {} };
  const encoder = { clearBuffer() {}, copyBufferToBuffer() {}, beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    for (const fineFactor of [4, 8] as const) {
      const broker = new PassBroker(encoder);
      new WebGPUFineLevelSetRedistance(device, {
        ...source, plan: { ...source.plan, fineFactor },
      } as never, redistanceDeltaAuthority(buffer, 1)).encode(
        broker, { bandCells: fineFactor === 4 ? 1 : 2 });
      broker.fence("redistance binding inspection");
    }
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  const expected: Record<string, number[]> = {
    publishSupportPageMask: [0, 2, 3, 10],
    seedClosestPoints: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    jumpFloodAToB: [0, 1, 2, 3, 4, 5, 6, 7, 10],
    jumpFloodBToA: [0, 1, 2, 3, 4, 5, 6, 7, 10],
    resolveClosestPointsAToB: [0, 2, 3, 4, 5, 6, 7, 9],
    resolveClosestPointsBToCanonical: [0, 2, 3, 4, 5, 6, 7, 9],
    validateJFADistances: [0, 1, 2, 3, 4, 5, 7, 9, 10],
    finalizeJFADistances: [0, 3, 8, 9],
    commitJFADistances: [0, 2, 3, 4, 5, 6, 7, 8],
  };
  assert.deepEqual(Object.fromEntries([...observed].map(([entryPoint, bindings]) =>
    [entryPoint, [...bindings].sort((a, b) => a - b)])), expected);
});

test("recurring fine redistance canonicalizes opposite flood parities on the same A/B buffers", () => {
  const passes: string[][] = [];
  let parameterWrites = 0;
  let currentPipeline = "";
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() { parameterWrites += 1; } },
    createBuffer: () => ({}),
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const source = {
    generation: 3,
    metadata: buffer, worklist: buffer, flags: buffer, phi: buffer,
    workA: buffer, workB: buffer, rollbackPhi: buffer,
    plan: {
      fineFactor: 4, brickResolution: 4, brickDimensions: [1, 1, 1], sampleDimensions: [4, 4, 4],
      samplesPerBrick: 64, maximumResidentBricks: 1, fineCellWidth: 0.25,
    },
  };
  const encoder = {
    clearBuffer() {},
    copyBufferToBuffer() {},
    beginComputePass() {
      const commands: string[] = []; passes.push(commands);
      return {
        setPipeline(pipeline: { entryPoint: string }) { currentPipeline = pipeline.entryPoint; },
        setBindGroup() {},
        dispatchWorkgroups() { commands.push(currentPipeline); },
        dispatchWorkgroupsIndirect() { commands.push(currentPipeline); },
        end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const redistance = new WebGPUFineLevelSetRedistance(
      device, source as never, redistanceDeltaAuthority(buffer, 1));
    for (const bandCells of [2, 1]) {
      const broker = new PassBroker(encoder);
      redistance.encode(broker, { bandCells });
      broker.fence(`redistance recurring parity ${bandCells}`);
    }
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  const recurrence = passes.map((commands) => commands.filter((entryPoint) =>
    entryPoint === "seedClosestPoints" || entryPoint.startsWith("jumpFlood")
      || entryPoint.startsWith("resolveClosestPoints")));
  assert.deepEqual(recurrence, [
    ["seedClosestPoints", "jumpFloodAToB", "jumpFloodBToA", "jumpFloodAToB",
      "jumpFloodBToA", "resolveClosestPointsAToB"],
    ["seedClosestPoints", "jumpFloodAToB", "jumpFloodBToA", "jumpFloodAToB",
      "resolveClosestPointsBToCanonical"],
  ]);
  assert.equal(parameterWrites, 2,
    "each recurring encode uploads one parameter block, independent of the JFA stride count");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn distanceValue\(index:u32\)->f32\{return bitcast<f32>\(workB\[index\]\);\}\s*fn resolvedSeed\(index:u32\)->u32\{return workA\[index\];\}/,
    "validation and commit must consume the same canonical A-seed/B-distance state after either parity");
});

test("factor-4 JFA-CPT redistance exposes every topology-bounded micro-stage", () => {
  const passes: string[][] = [];
  const copies: unknown[][] = [];
  const indirectOffsets: number[] = [];
  let currentPipeline = "";
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 }, queue: { writeBuffer() {} },
    createBuffer: () => ({}), createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const source = { generation: 1, metadata: buffer, worklist: buffer,
    flags: buffer, phi: buffer, workA: buffer, workB: buffer, rollbackPhi: buffer,
    plan: { fineFactor: 4, brickResolution: 4, brickDimensions: [1, 1, 1], sampleDimensions: [4, 4, 4],
      samplesPerBrick: 64, maximumResidentBricks: 1,
      fineCellWidth: 0.25 } };
  const encoder = {
    clearBuffer() {},
    copyBufferToBuffer(...args: unknown[]) { copies.push(args); },
    beginComputePass() {
      const commands: string[] = []; passes.push(commands);
      return {
        setPipeline(pipeline: { entryPoint: string }) { currentPipeline = pipeline.entryPoint; },
        setBindGroup() {}, dispatchWorkgroups() { commands.push(currentPipeline); },
        dispatchWorkgroupsIndirect(_buffer: GPUBuffer, offset: number) {
          commands.push(currentPipeline); indirectOffsets.push(offset);
        }, end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    // Product default: 4 interface cells * factor 4, plus factor + one
    // transport/interpolation support cell at publication = 21 fine cells.
    const broker = new PassBroker(encoder, { isolateLabels: true });
    new WebGPUFineLevelSetRedistance(
      device, source as never, redistanceDeltaAuthority(buffer, 1)).encode(broker,
      { bandCells: 21 });
    broker.fence("redistance test inspection");
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  assert.equal(passes.length, 13,
    "support, seed, seven floods, resolve, validate, finalize, and commit need exact labels");
  assert.deepEqual(copies, [],
    "JFA must consume topology's immutable command publication without recurring copies");
  assert.deepEqual(indirectOffsets, [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 84, 84],
    "support-mask publication plus seed/flood/resolve consume JFA support while validation and parallel commit touch the dirty dispatch");
  assert.doesNotMatch(WebGPUFineLevelSetRedistance.toString(),
    /updateIndirectBuffer|dispatchWorkgroupsIndirect\(this\.delta\.pageDelta/,
    "the writable page-delta transaction must never be consumed as an indirect command buffer");
  assert.deepEqual(passes.flat(), ["publishSupportPageMask",
    "seedClosestPoints", "jumpFloodAToB", "jumpFloodBToA",
    "jumpFloodAToB", "jumpFloodBToA", "jumpFloodAToB", "jumpFloodBToA",
    "jumpFloodAToB", "resolveClosestPointsBToCanonical", "validateJFADistances",
    "finalizeJFADistances", "commitJFADistances"]);
  assert.equal(passes.flat().length, 13,
    "generation-stamped direct support membership needs one bounded publication and no capacity clear");
  assert.match(fineLevelSetJFACPTWGSL, /override JFA_STRIDE:u32=1u/);
  assert.doesNotMatch(WebGPUFineLevelSetRedistance.toString(),
    /jfaParams|initializeJFAControl|reduceJFASeedStats|reduceJFAResolveStats|reduceJFAValidationStats/,
    "mutable stride buffers and the retired scalar reduction pipelines must stay deleted");
  assert.equal(passes.flat().filter((entryPoint) =>
    entryPoint === "seedClosestPoints" || entryPoint.startsWith("jumpFlood")
      || entryPoint.startsWith("resolveClosestPoints")).length, 9,
    "the 21-cell distance transform is one seed, seven floods, and one resolve dispatch");
});

test("opt-in Dawn backend reproducer: sparse diagonal JFA support gap dispatch", {
  skip: !process.env.WEBGPU_NODE_MODULE || process.env.FLUID_RUN_SPARSE_JFA_GAP !== "1"
    ? "set WEBGPU_NODE_MODULE and FLUID_RUN_SPARSE_JFA_GAP=1 for the Dawn sparse-gap reproducer" : false,
}, async () => {
  const device = await dawnDevice!; device.pushErrorScope("validation");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 6, 2],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 16 });
  const oracle = new FineLevelSetBrickOracle(plan);
  const near = packFineLevelSetBrickKey(plan, [0, 0, 0]);
  const diagonal = packFineLevelSetBrickKey(plan, [4, 4, 0]);
  // Two axial one-rings form disconnected diagonal islands. Only the near
  // island crosses zero; the far island deliberately claims transported
  // narrow-band values so a missed CPT propagation must reject publication.
  oracle.publishInterfaceAndRing([near, diagonal], ([x, y]) =>
    x > 3 && y > 3 ? 0.25 : x - 0.375);
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const source = owner.uploadGeneration(oracle.exportGPUGeneration());
  const delta = device.createBuffer({ size: (16 + 6 * plan.maximumResidentBricks) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  const deltaWords = new Uint32Array(16 + 6 * plan.maximumResidentBricks);
  const generationData = oracle.exportGPUGeneration();
  deltaWords.set([generationData.activeCount, source.generation, generationData.activeCount,
    generationData.activeCount, generationData.activeCount, 1, 1,
    generationData.activeCount, 1, 1]);
  const sortedIds = [...generationData.worklistWords.slice(7, 7 + generationData.activeCount)]
    .sort((a, b) => generationData.metadataWords[a * 10 + 1]!
      - generationData.metadataWords[b * 10 + 1]!);
  for (let i = 0; i < generationData.activeCount; i += 1) {
    const id = sortedIds[i]!;
    deltaWords[16 + i] = generationData.metadataWords[id * 10 + 1];
    deltaWords[16 + 2 * plan.maximumResidentBricks + i] = id;
    deltaWords[16 + 3 * plan.maximumResidentBricks + i] = id;
  }
  device.queue.writeBuffer(delta, 0, deltaWords);
  const deltaDispatch = device.createBuffer({ size: 96,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  const deltaDispatchWords = new Uint32Array(24);
  deltaDispatchWords.set([generationData.activeCount, 1, 1], 15);
  deltaDispatchWords.set([generationData.activeCount, 1, 1], 21);
  device.queue.writeBuffer(deltaDispatch, 0, deltaDispatchWords);
  const redistance = new WebGPUFineLevelSetRedistance(
    device, source, redistanceDeltaAuthority(delta, plan.maximumResidentBricks, deltaDispatch));
  const readback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  redistance.encode(broker, { bandCells: 7, residualTolerance: 1 });
  broker.copyBufferToBuffer(redistance.control, 0, readback, 0,
    FINE_LEVELSET_REDISTANCE_CONTROL_BYTES);
  device.queue.submit([broker.finish()]); await device.queue.onSubmittedWorkDone();
  const validationError = await device.popErrorScope(); assert.equal(validationError, null);
  await readback.mapAsync(GPUMapMode.READ);
  const control = unpackFineLevelSetGPURedistanceControl(
    new Uint32Array(readback.getMappedRange().slice(0), 0,
      FINE_LEVELSET_REDISTANCE_CONTROL_BYTES / 4));
  readback.unmap();
  assert.ok(control.seedCount > 0);
  // This is the intended result once the Dawn/Metal backend fault is cleared;
  // the deterministic test above owns the always-on fail-closed contract.
  assert.ok(control.unresolvedCells > 0,
    "a narrow-band island unreachable through resident JFA landing points must be reported");
  assert.equal(control.committed, false);
  assert.equal(control.flags, 0,
    `a propagation gap is a publication rejection, not a shader fault: ${JSON.stringify(control)}`);
  readback.destroy(); redistance.destroy(); deltaDispatch.destroy(); owner.destroy();
});

test("Dawn indexes global factor-4 bricks and returns coarse phi for missing/outside bricks", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const device = await dawnDevice!;
  const shaderModule = device.createShaderModule({ code: fineLevelSetBrickSamplingWGSL });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const plan = planFineLevelSetBricks({
    domainOrigin: [1, 2, 3], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8,
  });
  const oracle = new FineLevelSetBrickOracle(plan);
  const key = packFineLevelSetBrickKey(plan, [0, 0, 0]);
  oracle.publishInterfaceAndRing([key], ([x]) => x - 1.5);
  const page = oracle.pageForKey(key); assert.ok(page); page.phi[0] = -9;
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const source = owner.uploadGeneration(oracle.exportGPUGeneration());
  assert.equal(source.generation, 1);
  assert.equal(source.generationSlot, 1);

  const queryData = new Float32Array([
    1.125, 2.125, 3.125, 111,
    1 + 1.125, 2 + 1.125, 3 + 1.125, 123,
    0, 0, 0, 456,
  ]);
  const queries = device.createBuffer({ size: queryData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(queries, 0, queryData);
  const results = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "sampleQueries" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: source.params } },
    { binding: 1, resource: { buffer: source.worklist } },
    { binding: 2, resource: { buffer: source.metadata } },
    { binding: 3, resource: { buffer: source.flags } },
    { binding: 4, resource: { buffer: source.phi } },
    { binding: 5, resource: { buffer: queries } },
    { binding: 6, resource: { buffer: results } },
  ] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(results, 0, readback, 0, 48);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0); readback.unmap();
  const values = new Float32Array(bytes); const words = new Uint32Array(bytes);
  assert.equal(values[0], -9); assert.equal(words[1], 1);
  assert.equal(values[4], 123); assert.equal(words[5], 0);
  assert.equal(values[8], 456); assert.equal(words[9], 0);

  const nextSource = owner.prepareGPUGeneration(2);
  const topology = new WebGPUFineLevelSetTopology(device, source, nextSource,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-1.5;}");
  const topologyReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const topologyEncoder = device.createCommandEncoder();
  topology.encode(new PassBroker(topologyEncoder));
  topologyEncoder.copyBufferToBuffer(topology.control, 0, topologyReadback, 0, 36);
  topologyEncoder.copyBufferToBuffer(nextSource.worklist, 0, topologyReadback, 36, 28);
  device.queue.submit([topologyEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await topologyReadback.mapAsync(GPUMapMode.READ);
  const topologyWords = new Uint32Array(topologyReadback.getMappedRange().slice(0)); topologyReadback.unmap();
  const topologyControl = unpackFineLevelSetGPUTopologyControl(topologyWords);
  assert.equal(topologyControl.flags, 0);
  assert.equal(topologyControl.published, true);
  assert.ok(topologyControl.interfaceBricks > 0);
  assert.equal(topologyWords[9], 2);
  assert.equal(topologyWords[10], topologyControl.desiredBricks);
  assert.equal(topologyWords[11], plan.maximumResidentBricks);
  assert.equal(topologyWords[12], 3);
  assert.deepEqual([...topologyWords.slice(14, 16)], [1, 1]);
  assert.ok(topologyControl.interfaceSeedBricks > 0);

  const redistance = new WebGPUFineLevelSetRedistance(device, nextSource, topology);
  const redistanceReadback = device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const redistanceEncoder = device.createCommandEncoder();
  const redistanceBroker = new PassBroker(redistanceEncoder);
  redistance.encode(redistanceBroker, { bandCells: 2, residualTolerance: 1 });
  redistanceBroker.copyBufferToBuffer(redistance.control, 0, redistanceReadback, 0,
    FINE_LEVELSET_REDISTANCE_CONTROL_BYTES);
  redistanceBroker.copyBufferToBuffer(nextSource.phi, 0, redistanceReadback, 48, 4);
  device.queue.submit([redistanceBroker.finish()]); await device.queue.onSubmittedWorkDone();
  await redistanceReadback.mapAsync(GPUMapMode.READ);
  const redistanceBytes = redistanceReadback.getMappedRange().slice(0); redistanceReadback.unmap();
  const redistanceControl = unpackFineLevelSetGPURedistanceControl(new Uint32Array(redistanceBytes, 0,
    FINE_LEVELSET_REDISTANCE_CONTROL_BYTES / 4));
  assert.ok(redistanceControl.seedCount > 0, JSON.stringify(redistanceControl));
  assert.equal(redistanceControl.unresolvedCells, 0);
  assert.equal(redistanceControl.committed, true);
  assert.equal(redistanceControl.flags, 0);
  assert.equal(redistanceControl.firstError, 0xffff_ffff);
  assert.ok(redistanceControl.initialPages > 0);
  assert.ok(redistanceControl.finalPages >= redistanceControl.initialPages);
  assert.ok(redistanceControl.acceptedCells >= redistanceControl.seedCount);
  assert.ok(new Float32Array(redistanceBytes, 48, 1)[0] < 0);
  // A topology safety guard may be much wider than the requested redistance
  // band. The fixed band-sized iteration budget must still commit by storing
  // finite saturated distances outside that band.
  const guardPlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [20, 1, 1], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 20 });
  const guardOracle = new FineLevelSetBrickOracle(guardPlan);
  const guardKeys = Array.from({ length: guardPlan.brickDimensions[0] }, (_, x) =>
    packFineLevelSetBrickKey(guardPlan, [x, 0, 0]));
  guardOracle.publishInterfaceAndRing(guardKeys, ([x]) => x - 0.5);
  const guardOwner = new WebGPUFineLevelSetBricks(device, guardPlan);
  const guardSource = guardOwner.uploadGeneration(guardOracle.exportGPUGeneration());
  const guardDelta = device.createBuffer({ size: (16 + 6 * guardPlan.maximumResidentBricks) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  const guardDeltaWords = new Uint32Array(16 + 6 * guardPlan.maximumResidentBricks);
  const guardData = guardOracle.exportGPUGeneration();
  guardDeltaWords.set([guardData.activeCount, guardSource.generation, guardData.activeCount,
    guardData.activeCount, guardData.activeCount, 1, 1, guardData.activeCount, 1, 1]);
  for (let i = 0; i < guardData.activeCount; i += 1) {
    const id = guardData.worklistWords[7 + i];
    guardDeltaWords[16 + i] = guardData.metadataWords[id * 10 + 1];
    guardDeltaWords[16 + 2 * guardPlan.maximumResidentBricks + i] = id;
    guardDeltaWords[16 + 3 * guardPlan.maximumResidentBricks + i] = id;
  }
  device.queue.writeBuffer(guardDelta, 0, guardDeltaWords);
  const guardDispatch = device.createBuffer({ size: 96,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  const guardDispatchWords = new Uint32Array(24);
  guardDispatchWords.set([guardData.activeCount, 1, 1], 15);
  guardDispatchWords.set([guardData.activeCount, 1, 1], 21);
  device.queue.writeBuffer(guardDispatch, 0, guardDispatchWords);
  const guardRedistance = new WebGPUFineLevelSetRedistance(
    device, guardSource, redistanceDeltaAuthority(
      guardDelta, guardPlan.maximumResidentBricks, guardDispatch,
    ));
  const guardReadback = device.createBuffer({ size: 52,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const guardEncoder = device.createCommandEncoder();
  const guardBroker = new PassBroker(guardEncoder);
  guardRedistance.encode(guardBroker, { bandCells: 2, residualTolerance: 1 });
  guardBroker.copyBufferToBuffer(guardRedistance.control, 0, guardReadback, 0,
    FINE_LEVELSET_REDISTANCE_CONTROL_BYTES);
  guardBroker.copyBufferToBuffer(guardSource.phi,
    (guardPlan.maximumResidentBricks - 1) * guardPlan.samplesPerBrick * 4,
    guardReadback, 40, 4);
  guardBroker.copyBufferToBuffer(guardSource.flags,
    (guardPlan.maximumResidentBricks - 1) * guardPlan.samplesPerBrick * 4,
    guardReadback, 44, 4);
  guardBroker.copyBufferToBuffer(guardSource.flags, 0, guardReadback, 48, 4);
  device.queue.submit([guardBroker.finish()]); await device.queue.onSubmittedWorkDone();
  await guardReadback.mapAsync(GPUMapMode.READ);
  const guardBytes = guardReadback.getMappedRange().slice(0); guardReadback.unmap();
  const guardControl = unpackFineLevelSetGPURedistanceControl(new Uint32Array(
    guardBytes, 0, FINE_LEVELSET_REDISTANCE_CONTROL_BYTES / 4,
  ));
  assert.ok(guardControl.seedCount > 0, JSON.stringify(guardControl));
  assert.equal(guardControl.unresolvedCells, 0);
  assert.equal(guardControl.committed, true);
  assert.equal(new Float32Array(guardBytes, 40, 1)[0], 0.5);
  assert.equal(new Uint32Array(guardBytes, 44, 1)[0], 0,
    "the allocated outside-band guard must use coarse-octree phi fallback");
  assert.notEqual(new Uint32Array(guardBytes, 48, 1)[0] & 1, 0,
    "the actual narrow-band interface must remain fine-level-set authority");
  guardReadback.destroy(); guardRedistance.destroy(); guardDispatch.destroy(); guardOwner.destroy();

  const smallPlan = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 4,
  });
  const smallOracle = new FineLevelSetBrickOracle(smallPlan);
  smallOracle.publishInterfaceAndRing([packFineLevelSetBrickKey(smallPlan, [0, 0, 0])], ([x]) => x - 0.5);
  const smallOwner = new WebGPUFineLevelSetBricks(device, smallPlan);
  const smallCurrent = smallOwner.uploadGeneration(smallOracle.exportGPUGeneration());
  const smallNext = smallOwner.prepareGPUGeneration(2);
  const overflowTopology = new WebGPUFineLevelSetTopology(device, smallCurrent, smallNext,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}");
  const overflowReadback = device.createBuffer({ size: 40, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const overflowEncoder = device.createCommandEncoder();
  overflowTopology.encode(new PassBroker(overflowEncoder));
  overflowEncoder.copyBufferToBuffer(overflowTopology.control, 0, overflowReadback, 0, 36);
  overflowEncoder.copyBufferToBuffer(smallNext.worklist, 0, overflowReadback, 36, 4);
  device.queue.submit([overflowEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await overflowReadback.mapAsync(GPUMapMode.READ);
  const overflowWords = new Uint32Array(overflowReadback.getMappedRange().slice(0)); overflowReadback.unmap();
  const overflowControl = unpackFineLevelSetGPUTopologyControl(overflowWords);
  assert.notEqual(overflowControl.flags, 0);
  assert.ok(overflowControl.requiredDesiredBricks > smallCurrent.plan.maximumResidentBricks);
  assert.equal(overflowControl.requiredDesiredBricksExact, false);
  assert.equal(overflowControl.published, true);
  assert.equal(overflowControl.rolledBack, true);
  assert.equal(overflowWords[9], smallNext.generation,
    "a rejected candidate must retag the accepted rollback payload as the requested generation");

  const bootstrapOwner = new WebGPUFineLevelSetBricks(device, plan);
  const emptyCurrent = bootstrapOwner.initializeEmptyGPUGeneration(1);
  const bootstrapNext = bootstrapOwner.prepareGPUGeneration(2);
  const leafBuffer = new ArrayBuffer(64); const leafWords = new Uint32Array(leafBuffer);
  const leafFloats = new Float32Array(leafBuffer); leafWords[3] = 1; leafWords[4] = 2;
  // Bootstrap must copy interface values, not merely keys: this affine plane
  // crosses zero inside the first globally keyed factor-4 brick.
  leafFloats[8] = 0; leafFloats[9] = 1;
  const bootstrapLeaves = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bootstrapCandidates = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bootstrapCandidateControl = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bootstrapLeaves, 0, leafWords);
  device.queue.writeBuffer(bootstrapCandidates, 0, new Uint32Array([0, 2]));
  device.queue.writeBuffer(bootstrapCandidateControl, 0, new Uint32Array([1, 1, 1, 1]));
  const leafSeeds = new WebGPUFineLevelSetLeafSeeds(device, bootstrapNext);
  const bootstrapTopology = new WebGPUFineLevelSetTopology(device, emptyCurrent, bootstrapNext,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}");
  const bootstrapReadback = device.createBuffer({ size: 52 + 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const bootstrapEncoder = device.createCommandEncoder();
  const bootstrapBroker = new PassBroker(bootstrapEncoder);
  const seedSource = leafSeeds.encode(bootstrapBroker, { buffer: bootstrapLeaves }, { buffer: bootstrapCandidates },
    { buffer: bootstrapCandidateControl });
  bootstrapTopology.encode(bootstrapBroker, seedSource);
  bootstrapEncoder.copyBufferToBuffer(leafSeeds.buffer, 0, bootstrapReadback, 0, 8);
  bootstrapEncoder.copyBufferToBuffer(bootstrapTopology.control, 0, bootstrapReadback, 8, 36);
  bootstrapEncoder.copyBufferToBuffer(bootstrapNext.worklist, 0, bootstrapReadback, 44, 8);
  bootstrapEncoder.copyBufferToBuffer(bootstrapNext.phi, 0, bootstrapReadback, 52, 256);
  device.queue.submit([bootstrapEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await bootstrapReadback.mapAsync(GPUMapMode.READ);
  const bootstrapWords = new Uint32Array(bootstrapReadback.getMappedRange().slice(0)); bootstrapReadback.unmap();
  assert.deepEqual([...bootstrapWords.slice(0, 2)], [1, 0]);
  const bootstrapControl = unpackFineLevelSetGPUTopologyControl(bootstrapWords.slice(2, 11));
  assert.equal(bootstrapControl.flags, 0);
  // A block 1-ring is the complete 3D Chebyshev neighborhood, clipped at
  // this domain corner (2 x 2 x 2 bricks).
  assert.equal(bootstrapControl.desiredBricks, 8);
  assert.equal(bootstrapControl.dilationBrickRings, 1);
  assert.equal(bootstrapControl.requiredDesiredBricksExact, true);
  assert.equal(bootstrapControl.published, true);
  assert.deepEqual([...bootstrapWords.slice(11, 13)], [bootstrapNext.generation, 8],
    "the canonical workset header is generation then compact live count");
  const bootstrapPhi = new Float32Array(bootstrapWords.buffer, 52, 64);
  assert.ok([...bootstrapPhi].some((value) => value < 0));
  assert.ok([...bootstrapPhi].some((value) => value > 0));

  redistance.destroy(); redistanceReadback.destroy(); topology.destroy(); topologyReadback.destroy();
  overflowTopology.destroy(); overflowReadback.destroy(); smallOwner.destroy();
  leafSeeds.destroy(); bootstrapTopology.destroy(); bootstrapReadback.destroy(); bootstrapOwner.destroy();
  bootstrapLeaves.destroy(); bootstrapCandidates.destroy(); bootstrapCandidateControl.destroy();
  owner.destroy(); queries.destroy(); results.destroy(); readback.destroy();
});
