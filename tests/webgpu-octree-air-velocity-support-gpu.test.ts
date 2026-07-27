import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  decodeOctreeAirSupportGPUFirstError,
  OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE,
  OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS,
  OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX,
  OCTREE_AIR_SUPPORT_GPU_ERROR,
  OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE,
  OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS,
  OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE,
  WebGPUOctreeAirVelocitySupportProducer,
  octreeAirVelocitySupportPublicationWGSL,
  planOctreeAirVelocitySupportGPU,
} from "../lib/webgpu-octree-air-velocity-support-gpu";
import {
  OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE,
  OCTREE_AIR_SUPPORT_SELECTOR_STRIDE,
  OCTREE_AIR_SUPPORT_VALID,
} from "../lib/webgpu-octree-air-velocity-support";

const compact = (source: string) => source.replace(/\s+/g, "");

test("bounded topology errors retain stage and item in the existing word", () => {
  assert.deepEqual(decodeOctreeAirSupportGPUFirstError(
    (OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE.faceReconstruction << 24) | 1820), {
    stage: OCTREE_AIR_SUPPORT_GPU_TOPOLOGY_STAGE.faceReconstruction,
    item: 1820,
  });
  assert.match(compact(octreeAirVelocitySupportPublicationWGSL),
    /fnfailTopology\(stage:u32,item:u32\).*stage<<24u.*item&0x00ffffffu/);
});

test("GPU plan composes both support layouts and bounded candidate schedules", () => {
  const plan = planOctreeAirVelocitySupportGPU(4_096, 4_096 * 30, [16, 16, 16]);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS, 3 * 68);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE,
    OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE + OCTREE_AIR_SUPPORT_GPU_SELECTOR_SLOTS);
  assert.equal(plan.fineCandidateOffset, 4_096 * OCTREE_AIR_SUPPORT_GPU_CANDIDATE_STRIDE);
  assert.equal(plan.candidateCapacity, plan.fineCandidateOffset + plan.domainVolume);
  assert.equal(plan.domainVolume, 4_096);
  assert.equal(plan.records.capacity, plan.support.supportCapacity);
  assert.equal(plan.support.selectorStride, OCTREE_AIR_SUPPORT_SELECTOR_STRIDE);
  assert.equal(plan.support.ownerDirectoryCellCapacity, plan.domainVolume);
  assert.equal(plan.support.ownerDirectoryBytes, plan.domainVolume * 16);
  assert.ok(plan.offsets.candidates < plan.offsets.ranks);
  assert.ok(plan.offsets.ranks < plan.offsets.directoryWinners);
  assert.ok(plan.offsets.directoryWinners < plan.offsets.directoryFlags);
  assert.ok(plan.offsets.directoryFlags < plan.offsets.blockCounts);
  assert.ok(plan.offsets.blockCounts < plan.offsets.blockOffsets);
  assert.equal(plan.scratchBytes, plan.scratchWords * 4);
  assert.equal(plan.indirectBytes, 6 * 12);
  assert.equal(plan.faceCellCapacity, plan.rowCapacity + plan.support.supportCapacity);
  assert.equal(plan.faceCapacity, plan.faceCellCapacity * 12);
  assert.equal(plan.faceBytes, plan.faceCapacity * 16);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE, 1 + 30 + 24);
  assert.equal(plan.faceAdjacencyStride, OCTREE_AIR_SUPPORT_GPU_FACE_ADJACENCY_STRIDE);
  assert.equal(plan.faceAdjacencyBytes, plan.faceCellCapacity * plan.faceAdjacencyStride * 4);
  assert.equal(plan.directAirVectorBytes, plan.rowCapacity * 16);
});

test("WGSL performs deterministic mark, parallel scan, scatter, and tag resolution", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader, /letresolvedCell=cellOf\(owner\.origin\);letresolvedSize=owner\.size/);
  assert.match(shader, /atomicMin\(&scratch\[p\.directoryWinnerOffset\+resolvedCell\],item\)/);
  assert.match(shader, /atomicOr\(&scratch\[p\.directoryFlagOffset\+resolvedCell\],flags\)/);
  assert.match(shader, /fnmarkAndScanAirSupportCandidates[\s\S]*marks\[lane\]=mark[\s\S]*marks\[lane\]\+=add/);
  assert.match(shader, /fnprefixAirSupportBlocks[\s\S]*blockScan\[lane\]\+=add[\s\S]*p\.blockOffsetOffset/);
  assert.match(shader, /fnscatterAirSupportRecords[\s\S]*p\.recordOffset\+output\*8u/);
  assert.match(shader, /fnresolveAirSupportTags[\s\S]*atomicStore\(&supportArena\[c\.tagWord\],tag\)/);
  assert.doesNotMatch(shader, /workgroup_size\(1\)fn(?:clear|emit|mark|scatter|resolve)/,
    "all O(N) stages must be parallel");
});

test("producer proves generations and exact owner identities before admitting air support", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  // The producer proves internal coherence against the ACCEPTED epoch, not
  // the host's attempt stamp: when the newest candidate is rejected but a
  // clean older accepted authority survives, support deliberately rebuilds
  // against that epoch so Section 5 continues on one coherent (temporarily
  // reused) topology. Downstream consumers (supportPublicationValid) then
  // require the support epoch to equal the accepted epoch they run under.
  assert.match(shader, /if\(atomicLoad\(&accepted\.flags\)!=0u&&existingReady\)/,
    "a rejected candidate over a clean prior receipt must early-out, preserving that receipt");
  assert.match(shader, /accepted\.bank>1u/);
  assert.match(shader, /ownerPageArena\[7u\]!=accepted\.epoch/);
  assert.match(shader, /boundary!=accepted\.epoch/);
  assert.match(shader, /sw\(3u,accepted\.epoch\)/,
    "the publication must be stamped with the accepted epoch it was built from");
  assert.match(shader, /octreeOwnerPageLookup\(vec3i\(origin\)\)/);
  assert.match(shader, /if\(resolvedCell!=u32\(cell\)\|\|resolvedSize!=size\)/,
    "cube and tetrahedron vertices must both retain their exact octree-cell identity");
  assert.match(shader, new RegExp(`ERROR_TOPOLOGY:u32=${OCTREE_AIR_SUPPORT_GPU_ERROR.topology}u`));
  assert.match(shader, /publishedRow\(resolvedCell,resolvedSize\)/,
    "accepted owner identities must retain direct compact-row tags");
  assert.match(shader,
    /fnmarkFineBandAirSupportDemand.*fineWorklist\[0\]==p\.expectedFineGeneration.*fineMetadata\[id\*10u\+2u\]!=p\.expectedFineGeneration/s,
    "fine-band demand must be generated only from the exact selected fine publication");
});

test("fine-band support closes over the exact regular or transition interpolation dependencies", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const closure = shader.slice(shader.indexOf("fncloseFineBandAirSupportInterpolationDemand"),
    shader.indexOf("fnemitFineBandAirSupportCandidates"));
  assert.match(closure,
    /if\(caseId==0u&&markExactRegularNeighborhood\(owner\.origin,owner\.size,item\)\)\{return;\}/,
    "case zero may use trilinear closure only after proving its exact regular neighborhood");
  assert.match(closure, /markExactRegularNeighborhood.*for\(vartetra=0u/s,
    "a body-diagonal transition retains exact regular octants and the case-zero Delaunay fan");
  assert.match(closure,
    /letpacked=tetrahedra\[first\+tetra\].*letselector=.*letv=tetraVertices\[selector\].*letselectorCenter=center\+f32\(owner\.size\)\*inverseTransform.*markFineResolvedOwner\(selectorCenter,selectorSize,item\)/s,
    "transition demand must include the actual catalog tetrahedron selector owners");
  assert.match(shader,
    /fnmarkExactRegularNeighborhood.*letexpectedCenter=clamp\(center\+.*vec3f\(half\),vec3f\(p\.dimensions\)-vec3f\(half\)\).*letresolved=octreeOwnerPageLookup.*letmatches=resolved\.size==size.*atomicOr/s,
    "regular closure must require the exact same-size center after the consumer's boundary clamp");
  assert.match(shader,
    /fnmarkFineResolvedOwner\(expectedCenter:vec3f,expectedSize:u32,item:u32\).*owner\.size!=expectedSize\|\|any\(abs\(ownerCenter-expectedCenter\)>vec3f\(tolerance\)\).*failTopology\(2u,item\)/s,
    "fine closure publication must reject a containing owner with the wrong identity");
  assert.match(closure,
    /letsizef=f32\(owner\.size\)\*v\.w;letselectorSize=u32\(round\(sizef\)\).*selectorSize==0u\|\|abs\(sizef-f32\(selectorSize\)\)>2e-4.*letoriginf=selectorCenter-vec3f\(\.5\*sizef\);letorigin=vec3i\(round\(originf\)\).*abs\(originf-vec3f\(origin\)\)>vec3f\(2e-4\).*origin<vec3i\(0\).*origin\+vec3i\(i32\(selectorSize\)\)>vec3i\(p\.dimensions\).*markFineResolvedOwner/s,
    "transition closure may use constant boundary extension only for a geometrically proven exterior selector");
  assert.match(closure,
    /v\.w<=0\.\|\|!finiteValue\(v\.x\)\|\|!finiteValue\(v\.y\)\|\|!finiteValue\(v\.z\)\|\|!finiteValue\(v\.w\).*fail\(item,ERROR_CATALOG\)/s,
    "non-finite selector geometry must fail as a catalog error rather than masquerade as physical exterior");
  assert.doesNotMatch(closure, /maxLeaf|conservative|approximate/,
    "adaptive closure must be topology exact rather than a guessed spatial radius");
  assert.match(shader, /constQUERY_FINE:u32=0x40000000u/);
  assert.match(closure, /&QUERY_FINE\)==0u/);
  assert.doesNotMatch(closure, /markFineResolvedOwner.*QUERY_FINE/s,
    "VALUE_ONLY closure writes must not recursively become new queries in the same dispatch");
});

test("transition and regular demand slots are fixed and physical exterior stays invalid", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader, /var<workgroup>emitRowRegular:atomic<u32>/);
  // The 27-site proof runs for EVERY enabled row: eligibility is geometric
  // (boundary-clamped uniformity), not caseId==0, so wall-touching rows keep
  // the exact per-axis staggered face basis. Nonzero-case regular rows also
  // publish their retained tet fan for transition consumers.
  assert.match(shader,
    /if\(lane<27u\).*fineResolvedOwnerMatches\(expectedCenter,g\.y,itemBase\+lane\).*workgroupBarrier\(\);letregular=workgroupUniformLoad\(&emitRowRegular\)!=0u/s,
    "one row-owned workgroup must cooperatively prove the 27-site cube exactly once");
  assert.match(shader, /letneedsSelectors=!regular\|\|g\.z!=0u;/,
    "a regular wall row publishes both closures so transition sampling keeps its selectors");
  assert.match(shader, /if\(regular&&lane<27u\)\{letlocal=lane/);
  assert.match(shader, /letoccurrence=lane.*letitem=itemBase\+27u\+occurrence/s);
  assert.match(shader, /occurrence>=3u\*count/);
  assert.match(shader, /letselector=\(packed>>\(8u\*\(occurrence%3u\)\)\)&255u/);
  // No early return: a regular wall row's lane may still owe a selector
  // occurrence for the dual-closure publication after stamping its
  // out-of-domain cube tag INVALID.
  assert.match(shader, /if\(!inDomain\)\{atomicStore\(&supportArena\[tag\],INVALID\);\}else\{/);
  assert.match(shader, /any\(origin<vec3i\(0\)\).*atomicStore\(&supportArena\[tag\],INVALID\)/);
  assert.match(shader, /writeDispatch\(16u,select\(vec3u\(0u,1u,1u\),dispatchFor\(rows,1u\),clean\)\)/,
    "the indirect schedule must dispatch one 2-D-safe workgroup per row");
  assert.doesNotMatch(shader, /regularNeighborhoodExact/,
    "candidate lanes must not each repeat the complete 27-owner proof");
});

test("Section 5 chain is power-face seeded, persistently marched, and reconstructed at power centroids", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  const faceValue = shader.slice(shader.indexOf("fnprojectedAxisFaceValue"), shader.indexOf("fnneighborIdentity"));
  assert.match(faceValue, /structuredAuthority\[handleAt\][\s\S]*p\.valuesOffset\+handle[\s\S]*catalogFaces\[global\]/,
    "the seed source must be the accepted projected power-face samples with their catalog geometry");
  assert.match(faceValue, /abs\(aligned\)>=0\.999/, "only axis-normal power faces may seed an ordinary axis face");
  assert.match(faceValue, /abs\(centroid\[axis\]-patchCenter\[axis\]\)<=tolerance/,
    "the seeding power face must be coplanar with the seeded patch");
  assert.match(faceValue, /bestLocal==INVALID\|\|separation<bestDistance/,
    "nearest-face selection is strict-less so the lowest local slot index wins exact ties");
  assert.doesNotMatch(shader, /fnprojectedPowerVector/,
    "no cell-centred reconstructed vector may masquerade as a face seed");
  const seed = shader.slice(shader.indexOf("fnseedAirSupportFaces"), shader.indexOf("fnbetterFace"));
  assert.match(seed, /projectedAxisFaceValue\(faceRow,axis,patchCenter\)/);
  assert.match(seed, /adjacencyPositive\(faceRow,axis,local%4u\)/);
  assert.match(seed, /projectedAxisFaceValue\(otherRow,axis,patchCenter\)/,
    "an air-owned patch seeds from its positive liquid neighbour's coincident face");
  assert.doesNotMatch(seed, /sum\/f32\(weight\)|sum\+=|weight\+=/,
    "the seed copies one exact face value and never averages adjacent liquid cells");
  assert.match(seed, /vec4u\(bitcast<u32>\(seed\.x\),0u,item,1u\)/,
    "the seed distance origin remains the seeding face patch centre");
  const publication = shader.slice(shader.indexOf("fnresolveAirSupportFaceAdjacency"), shader.indexOf("var<workgroup>seedCounts"));
  assert.match(publication, /catalogNeighbor\(cell,header\.x\+localFace\)[\s\S]*faceRowForIdentity\(identity\)/,
    "T-junction catalog incidence must be resolved during topology publication");
  assert.match(publication, /signedFaceNeighborIdentity\(cell,axis,quadrant,false\)[\s\S]*signedFaceNeighborIdentity\(cell,axis,quadrant,true\)/,
    "all twelve signed regular-face patches must be published once");
  const march = shader.slice(shader.indexOf("fnextendFace"), shader.indexOf("fnquadrantAt"));
  assert.match(march, /adjacencyIncidentCount\(faceRow\)[\s\S]*adjacencyIncident\(faceRow,localFace\)/);
  assert.doesNotMatch(march, /tagForIdentity|faceRowForTag|octreeOwnerPageLookup|catalogNeighbor|caseHeader/,
    "the face march must be a pure indexed gather over published adjacency");
  assert.match(shader,
    /fnbetterFace\(candidate:vec4u,best:vec4u\).*candidateDistance=bitcast<f32>\(candidate\.y\).*candidateDistance<bestDistance.*candidateDistance==bestDistance&&candidate\.z<best\.z/s);
  // The carrier metric is the true Euclidean distance to the ORIGINAL seed
  // patch (candidate.z, preserved verbatim through every copy) — the
  // closest-point-transform ranking. Accumulating per-hop path length made
  // the metric an axis-graph geodesic (Manhattan-like), which under-drives
  // diagonal spreading and squares off the dam front.
  assert.match(march,
    /distance=length\(faceCenter\(item\)-faceCenter\(candidate\.z\)\)/,
    "the fast-marching-like relaxation must rank sources by Euclidean distance to the original seed, not accumulated hop length");
  assert.match(shader, /clean=errors==0u&&\(s\(35u\)\|s\(36u\)\|s\(37u\)\)==0u/,
    "publication requires a deliberate GPU-observed no-change wave");
  assert.match(march,
    /var<workgroup>relaxationChanged:atomic<u32>.*fnmarchAirSupportFacesToFixedPoint.*letaxis=wid\.x;letcount=4u\*faceRows.*for\(varlocal=lane;local<count;local\+=256u\).*storageBarrier\(\).*changed==0u\|\|tailWave>=max\(1u,count\)/s,
    "one persistent workgroup per independent velocity axis must march its compact graph to a fixed point with the |V_axis| proof bound");
  assert.doesNotMatch(march, /STRUCTURED_VELOCITY_EXTENSION_LAYERS|maximumRelaxationWaves|dimensions\.reduce/,
    "the paper march must not be replaced by a widened scene or domain wave count");
  assert.doesNotMatch(march, /sum\+=|weight\+=/,
    "marching copies one deterministic closest source and never averages");
  assert.match(shader, /letcentroid=anchorCenter\+f32\(cell\.y\)\*inverseTransform\(slot\.areaCentroid\.yzw,transform\)/);
  assert.match(shader, /regularVectorAt\(faceRow,centroid\)[\s\S]*dot\(interpolated\.xyz,normal\)/);
  assert.match(shader, /positive\.w==0u&&u32\(origin\[axis\]\)\+cell\.y==p\.dimensions\[axis\][\s\S]*p\.closedBoundaryMask/,
    "positive closed walls supply only their exact stationary-solid normal sample");
  assert.match(shader, /negativeRow!=INVALID[\s\S]*u32\(origin\[axis\]\)==0u&&\(p\.closedBoundaryMask/,
    "negative closed walls supply the matching exact stationary-solid sample");
});

test("direct air rows and support identities share one fail-closed face transaction", () => {
  const shader = compact(octreeAirVelocitySupportPublicationWGSL);
  assert.match(shader, /!publishedLiquidRow\(faceRow\)/,
    "accepted liquid row vectors remain the projected source authority");
  const reconstruct = shader.slice(shader.indexOf("fnreconstructAirSupportVectors"),
    shader.indexOf("fnfinalizeAirSupportMetadata"));
  assert.match(reconstruct, /directAirVectors\[faceRow\]=result/);
  assert.doesNotMatch(reconstruct, /rowVelocities\[/,
    "reconstruction may not expose partial direct-air vectors in the accepted bank");
  assert.match(shader, /supportVectors\[support\]=result[\s\S]*p\.recordVectorOffset\+4u\*support/);
  // An axis with no marched value on either side is a demand island (a cell
  // the degenerately dense fine band demanded with no demanded path to any
  // seeded liquid face). It takes stationary air with a counted, identified
  // provenance latch instead of rejecting: failing closed froze the whole
  // epoch on the first island (measured at dam wall contact, far air 5+
  // cells above the liquid, flags fineBandDemand|extensionClosure).
  assert.match(shader,
    /if\(positive\.w==0u&&negative\.w==0u\)\{[\s\S]*?atomicAdd\(&scratch\[41u\],1u\);[\s\S]*?atomicMin\(&scratch\[42u\],\(\(\(cell\.w>>6u\)&0xffu\)<<16u\)\|\(cell\.x<<3u\)\|axis\);[\s\S]*?result\[axis\]=0\.;continue;\}/,
    "an unreached axis must take latched stationary air, never reject the publication");
  assert.match(shader, /if\(!validVector\(interpolated\)\)\{failTopology\(8u,faceRow\);/,
    "genuinely invalid interpolation results must still fail the face transaction closed");
  assert.match(shader,
    /if\(!validVector\(interpolated\)\)\{failTopology\(8u,faceRow\);[\s\S]*atomicCompareExchangeWeak\(&recordArena\[14u\],0u,detail\)[\s\S]*returnvec4f\(f32\(local\),interpolated\.x,interpolated\.y\+16\.\*interpolated\.z,-1\.\);\}/,
    "a rejected reconstruction must retain the local catalog face in the existing vector slot");
  assert.match(reconstruct,
    /elseif\(faceRow<s\(2u\)\).*directAirVectors\[faceRow\]=result.*else\{letsupport=faceRow-s\(2u\).*supportVectors\[support\]=result/s,
    "failure-only diagnostics remain in already-owned rejected vector storage");
  assert.doesNotMatch(reconstruct, /atomicStore\(&supportArena/,
    "the aligned support-vector suffix must use non-conflicting vec4 stores");
  assert.match(shader, /var<workgroup>reconstructExpected:array<u32,256>[\s\S]*atomicAdd\(&scratch\[30u\],reconstructExpected\[0\]\)/);
  assert.match(shader, /var<workgroup>seedCounts:array<u32,256>[\s\S]*atomicAdd\(&scratch\[25u\],seedCounts\[0\]\)/);
  assert.match(shader, /expectedVectors=s\(30u\)[\s\S]*s\(28u\)==expectedVectors/);
  assert.match(shader, /fnfinalizeAirSupportMetadata[\s\S]*sw\(31u,select\(0u,1u,clean\)\)[\s\S]*supportArena\[at\+13u\],0u/);
  assert.match(shader, /fncommitAirSupportDirectRows[\s\S]*if\(s\(31u\)!=1u[\s\S]*rowVelocities\[output\]=directAirVectors\[row\]/);
  assert.match(shader, new RegExp(`atomicStore\\(&supportArena\\[p\\.airControlOffset\\+13u\\],select\\(0u,${OCTREE_AIR_SUPPORT_VALID}u,clean\\)\\);\\}$`),
    "suffix VALID must be the literal last publication store");
  assert.match(shader, /RECORD_REGULAR\|RECORD_EXTENSION/);
  assert.match(shader, /RECORD_SELECTOR\|RECORD_EXTENSION/);
});

test("host schedule uses a parallel occupancy prefix and a persistent fixed-point authority", () => {
  const encode = compact(WebGPUOctreeAirVelocitySupportProducer.prototype.encode.toString());
  assert.match(encode, /dispatchWorkgroups\(1\).*updateIndirectBuffer/);
  assert.equal((encode.match(/updateIndirectBuffer/g) ?? []).length, 2,
    "fixed-point marching must not repeatedly copy indirect schedules");
  for (const [name, offset] of [["clearAirSupportDirectory", 0], ["clearAirSupportTags", 12],
    ["emitAirSupportCandidates", 24], ["markAndScanAirSupportCandidates", 36]] as const) {
    assert.match(encode, new RegExp(`run\\("${name}",${offset}\\)`));
  }
  assert.match(encode, /updateIndirectBuffer\(this\.scratch,32\*4,this\.indirect,4\*12,2\*12\)/);
  assert.match(encode, /dispatchWorkgroupsIndirect\(this\.indirect,48\)/);
  assert.match(encode, /dispatchWorkgroupsIndirect\(this\.indirect,60\)/);
  assert.match(encode, /run\("finalizeAirSupportMetadata"\)[\s\S]*commitAirSupportDirectRows[\s\S]*run\("commitAirSupportPublication"\)/);
  assert.doesNotMatch(encode, /dispatchWorkgroups\(this\.plan\.faceCapacity/);
  assert.match(encode, /clearBuffer\(this\.scratch,32\*4,6\*4\)/,
    "copied indirect words must be recycled once as six storage-visible wave flags");
  assert.match(encode, /marchAirSupportFacesToFixedPoint.*dispatchWorkgroups\(3\).*broker\.fence\("Section5closest-facefixedpointpublished"\)/s);
  assert.equal(OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX, 12);
  assert.match(encode, /for\(letwave=0;wave<OCTREE_AIR_SUPPORT_GPU_PARALLEL_MARCH_PREFIX;wave\+=1\)/,
    "the fully parallel prefix is a fixed occupancy optimization");
  assert.doesNotMatch(encode, /maximumRelaxationWaves|prepareAirSupportContinuation|for\(letlayer/,
    "the host must not encode a scene/domain propagation bound");
});

test("entry bind sets exactly match the reachable staging transaction", () => {
  assert.deepEqual(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS, {
    beginAirSupportPublication: [0,1,3,7,8,9,10],
    clearAirSupportDirectory: [0,7], clearAirSupportCandidates: [0,2,7],
    clearAirSupportTags: [0,7,9],
    emitAirSupportCandidates: [0,2,3,4,5,6,7,9,11,18],
    markAndScanAirSupportCandidates: [0,7], prefixAirSupportBlocks: [0,7],
    scatterAirSupportRecords: [0,7,8], resolveAirSupportTags: [0,7,8,9],
    resolveAirSupportTopology: [0,3,7,8,11,12,13,14],
    markFineBandAirSupportDemand: [0,7,25,26,27,28],
    closeFineBandAirSupportInterpolationDemand: [0,2,3,4,5,6,7,11,12,13,14],
    emitFineBandAirSupportCandidates: [0,2,3,7,11],
    publishAirSupportOwnerDirectory: [0,2,3,7,8,9,11],
    prepareAirSupportFaces: [0,7],
    resolveAirSupportFaceAdjacency: [0,2,3,7,8,11,15,16,23],
    seedAirSupportFaces: [0,1,2,7,8,15,16,18,19,21,23],
    extendAirSupportFacesAtoB: [0,2,7,8,19,20,23],
    extendAirSupportFacesBtoA: [0,2,7,8,19,20,23],
    advanceAirSupportMarchWave: [7],
    marchAirSupportFacesToFixedPoint: [0,2,7,8,19,20,23],
    reconstructAirSupportVectors: [0,2,7,8,15,16,19,22,23,24],
    finalizeAirSupportMetadata: [0,2,7,8,9,22],
    commitAirSupportDirectRows: [0,2,7,17,22],
    commitAirSupportPublication: [0,7,8,9],
  });
  for (const bindings of Object.values(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS)) {
    assert.ok(bindings.filter((binding) => binding !== 0 && binding !== 11).length <= 10);
  }
});

test("producer exposes canonical banked row vectors and avoids double-counting a shared suffix arena", () => {
  const source = compact(Object.getOwnPropertyDescriptor(
    WebGPUOctreeAirVelocitySupportProducer.prototype, "source")!.get!.toString());
  assert.match(source, /canonicalRowVelocities:this\.inputs\.structured\.rowVelocities/);
  const implementation = compact(readFileSync(
    new URL("../lib/webgpu-octree-air-velocity-support-gpu.ts", import.meta.url), "utf8"));
  assert.match(implementation, /this\.allocatedBytes=this\.plan\.allocatedBytes\+256-\(inputs\.sharedArena\?this\.plan\.support\.totalBytes:0\)/);
  assert.match(implementation,
    /this\.params=Object\.freeze\(\[0,1\]\.map.*constparameterSlot=this\.parameterSlot;this\.parameterSlot=parameterSlot===0\?1:0.*queue\.writeBuffer\(params,0,this\.parameterData/s,
    "two support transactions encoded into one command buffer must retain invocation-stable uniforms");
  assert.match(implementation,
    /expectedVectors=s\(30u\).*s\(28u\)==expectedVectors/s,
    "the face march must prove every exact demanded vector was reconstructed");
  assert.doesNotMatch(implementation,
    /maximumDisplacementFineCells\s*>\s*STRUCTURED_VELOCITY_EXTENSION_LAYERS/,
    "fine-cell backtrace distance and adaptive face-graph hops are unlike units");
});

test("Dawn compiles every Section 5 producer entry point at the M1 storage-buffer ceiling", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const shaderModule = device.createShaderModule({ code: octreeAirVelocitySupportPublicationWGSL });
  const errors = (await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of Object.keys(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS)) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const error = await device.popErrorScope(); assert.equal(error, null, error?.message);
  device.destroy();
});

test("GPU plan rejects malformed capacities and domain dimensions", () => {
  assert.throws(() => planOctreeAirVelocitySupportGPU(0, 30, [16, 16, 16]), /positive/);
  assert.throws(() => planOctreeAirVelocitySupportGPU(1, 30, [16, 0, 16]), /positive/);
});
