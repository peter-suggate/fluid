import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("Losasso compact coarse-only ghost predicates are valid WGSL boolean expressions", () => {
  const source = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  const coarseOnly = source.slice(source.indexOf("octreeLosassoCoarseOnlyPhiWGSL"),
    source.indexOf("octreeLosassoCoarsePhiWGSL"));
  assert.match(coarseOnly, /let negativeNearby=.*&&.*;[\s\S]*let positiveNearby=.*&&.*;[\s\S]*if\(negativeNearby\|\|positiveNearby\)/,
    "Dawn must not have to parse mixed conjunction and disjunction operators in one expression");
  assert.match(coarseOnly, /let contactEnabled=.*;let separated=.*;let sampledAir=.*&&.*;[\s\S]*if\(contactEnabled&&\(separated\|\|sampledAir\)\)/,
    "the separated-or-air classification should give each conjunction an explicit name");
  assert.match(coarseOnly, /vec3f\(vec3u\(leaf\.originX,leaf\.originY,leaf\.originZ\)\)/,
    "WGSL requires an explicit vector conversion from integer seed origins");
  assert.match(coarseOnly,
    /let transported=trackerPhi\(centre\)[\s\S]*nextGradient\[row\]=vec4f\(gradient,1\)/,
    "compact rows must be a restriction of the transported dense coarse lattice");
});

test("Losasso ghost conditioning reconstructs its dual from geometry", () => {
  const source = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  const publish = source.slice(source.indexOf("fn publishLosassoGhostDistances"));
  assert.doesNotMatch(publish, /distance=1\.\/face\.inverseDistance/,
    "a conditioning pass must never use its own prior conditioned coefficient as geometry");
  assert.match(publish, /let dual=f32\(rowSize\)\*p\.cellSize/,
    "the staged missing neighbor has the same size, so the dual is one row width");
  assert.match(publish, /let faceToAir=\.5\*dual/);
  assert.match(publish, /theta=clamp\(-liquid\/\(airPhi-liquid\),1e-4,1\.\)/,
    "idempotence, not a raised theta floor, bounds repeated conditioning");
});

test("Losasso candidate faces stage missing air neighbors at one wet-row width", () => {
  const source = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  const positive = source.slice(source.indexOf("fn writeFace"),
    source.indexOf("fn writeNegativeBoundaryFace"));
  const negative = source.slice(source.indexOf("fn writeNegativeBoundaryFace"),
    source.indexOf("fn denseSolid"));
  assert.match(positive,
    /let neighborSize=select\(rowSize,neighbor\.size,neighborRow!=INVALID&&neighbor\.size>0u\)/,
    "a dry positive-side owner placeholder must not create a half-cell pressure dual");
  assert.match(negative, /let distance = f32\(rowSize\) \* params\.cellSize\[axis\]/,
    "a missing negative-side pressure neighbor must stage the same full dual");
});

test("Losasso coarse phi publishes through a stable arena binding", () => {
  const source = read("../lib/webgpu-octree-losasso-coarse-phi.ts");
  const encode = source.slice(source.indexOf("  encode(broker:"),
    source.indexOf("\n  fineTopologyEntry", source.indexOf("  encode(broker:")));
  assert.match(encode, /const previousArena = this\.source\.arena/);
  assert.match(encode, /const nextArena = this\.arenas\[1\]/);
  assert.doesNotMatch(encode, /this\.source as \{ arena: GPUBuffer \}/,
    "encode must not invalidate bind groups by rotating the public buffer object");
  assert.match(encode,
    /run\("ghost"[\s\S]*run\("publish", \[0, 5, 8, 14\], \[this\.readyDispatch, 12\]\)/,
    "the staged generation publishes in-pass only after old-arena warm/delta readers finish");
  assert.doesNotMatch(encode, /copyBufferToBuffer/,
    "stable arena publication must not break the compute pass for a full-buffer blit");
});

test("Losasso ready coarse phi selects retained-directory maintenance on the GPU", () => {
  const shader = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  assert.match(shader,
    /fn exactTopologyReuse\(\)->bool[\s\S]*coarseControl\[5\]==1u/);
  assert.match(shader,
    /fn prepareLosassoCoarsePhi[\s\S]*if\(reuse\)[\s\S]*publishedArena\[13\][\s\S]*return;/,
    "exact reuse must validate and retain the published directory instead of clearing staging");
  assert.match(shader,
    /readyDispatch\[0\]=select\(p\.schedule\.y,0u,reuse\)[\s\S]*readyDispatch\[3\]=select\(p\.schedule\.z,0u,reuse\)/,
    "the GPU reuse decision should zero the inactive warm and arena-copy grids");
  assert.match(shader,
    /fn restrictLosassoCoarsePhi[\s\S]*restrictLosassoCoarsePhiRow\(invocation\.x,!reuse,reuse\)/,
    "the recurring restrict dispatch should refresh retained row identities without directory insertion");
  assert.match(shader,
    /fn publishLosassoCoarsePhiArena[\s\S]*if\(exactTopologyReuse\(\)\)\{return;\}/,
    "in-place maintenance must skip the full arena publication copy");
  assert.match(shader,
    /fn publishLosassoVolumeBridge[\s\S]*if\(reuse\)[\s\S]*summaryDelta\[0\]=retainedCount/,
    "the steady volume bridge should publish current rows only, not retired-plus-current topology");
});

test("Losasso post-volume coarse phi refresh retains topology-shaped state", () => {
  const octreeSource = read("../lib/webgpu-octree.ts");
  const coarsePhiSource = read("../lib/webgpu-octree-losasso-coarse-phi.ts");
  assert.match(octreeSource,
    /losassoCoarsePhi\.encodeFieldRefresh\(redistanceBroker, pending\.target/,
    "post-volume settlement should refresh fields through the retained coarse topology");
  assert.match(octreeSource,
    /losassoCoarsePhi\.encodeFieldRefresh\(coarseBroker, publicationTarget/,
    "transported fine phi must refresh the topology already accepted at the ready boundary");
  assert.match(coarsePhiSource,
    /encodeFieldRefresh[\s\S]*prepareRefresh[\s\S]*refresh[\s\S]*volumeRefresh[\s\S]*ghost/,
    "the refresh schedule must update values, bridge records, and ghost distances");
  assert.doesNotMatch(coarsePhiSource.slice(coarsePhiSource.indexOf("encodeFieldRefresh"),
    coarsePhiSource.indexOf("private updateParams")),
    /run\("warm"|run\("publish"/,
    "same-topology refresh must not remap rows or republish the retained arena");
});

test("Losasso W7 extension closes the third diagonal MAC support layer", () => {
  const host = read("../lib/webgpu-octree-losasso-extension-band.ts");
  const shader = read("../lib/webgpu-octree-losasso-extension-band.wgsl.ts");
  assert.match(shader, /if\(page==INVALID\)\{continue;\}/,
    "a missing outer fine-phi corner must not punch a hole in MAC support");
  assert.match(shader, /exactValue\(exact\)\/total/,
    "one-sided MAC membership must renormalize the available fine-phi corners");
  assert.match(shader, /sample=finePhiCells\(clamp\(centre,pageLow,pageHigh\)\)/,
    "a completely absent diagonal stencil must use its originating B4 page");
  assert.match(shader, /dilateAirFace\(linearInvocation\(g,l\),6u,7u\)/);
  assert.match(host, /run(?:Indirect)?\("dilateLosassoAirBand7"/,
    "W4 transport plus a three-axis trilinear corner requires all three closure layers");
});

test("Losasso factor-one transport has no fine-page dependency", () => {
  const octree = read("../lib/webgpu-octree.ts");
  const exchange = read("../lib/webgpu-octree-losasso-coarse-phi.ts");
  const extension = read("../lib/webgpu-octree-losasso-extension-band.wgsl.ts");
  assert.match(octree,
    /coarseOnlySurfaceTracking && this\.coarseDynamics\.backend === "losasso"[\s\S]*encodeCoarseOnly[\s\S]*encodeCoarseExtensionBandPublication/,
    "factor one must cut over before the global-fine transaction");
  const coarseOnly = exchange.slice(exchange.indexOf("  encodeCoarseOnly("),
    exchange.indexOf("  get coarseOnlyPublishedGeneration"));
  assert.doesNotMatch(coarseOnly, /WebGPUFineLevelSetBrickSource|metadata|worklist|samples/,
    "compact transport must not smuggle a fine-page authority into factor one");
  assert.match(coarseOnly, /velocity\.control[\s\S]*input\.coarseControl/,
    "velocity interpolation and pressure row counts are distinct authorities");
  assert.match(coarseOnly,
    /this\.coarseOnlyNextGradient, input\.faceGeometry, velocity\.extendedVelocity/,
    "factor-one ghost conditioning must index the accepted pressure-face geometry");
  assert.doesNotMatch(coarseOnly,
    /this\.coarseOnlyNextGradient, velocity\.faceGeometry, velocity\.extendedVelocity/,
    "W7 extension geometry is not ordered by pressure face id");
  assert.match(read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts"),
    /else if\(liquid<0\.&&\(p\.closed&2u\)!=0u\)\{theta=1\.;distance=dual;face\.inverseDistance=1\.\/dual;flags=1u;\}/,
    "paper mode must place p_air at every non-solid missing neighbour without a fine-sample proof");
  assert.match(read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts"),
    /let rowSize=headers\[face\.negativeRow\]\.size;let dual=f32\(rowSize\)\*p\.velocityCellSize/,
    "factor-one cell-centred air must use the accepted wet-row centre distance");
  assert.match(read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts"),
    /previousGradient\[row\]\.w==1\.[\s\S]*nextGradient\[row\]=vec4f\(gradient,1\)/,
    "coarse-only transport must bootstrap an absent gradient and explicitly publish its validity");
  assert.match(exchange,
    /words\.set\(\[1, generation, Math\.ceil\(this\.plan\.arenaBytes \/ 4 \/ 64\), 0\], 12\)/,
    "factor-one must publish the complete arena, including its trailing row hash directory");
  assert.match(read("../lib/webgpu-octree-fine-seed-adapter.ts"),
    /gradient=coarseGradientAt\(row\)/,
    "recurring topology seeds must retain the coarse affine gradient");
  assert.match(extension,
    /coarsePhiDirectory\[1\]&0x40000000u[\s\S]*capacity-volume\+linearCell/,
    "coarse W7 support must sample the transported dense complement rather than pressure-row flags");
  assert.match(extension,
    /let crossing=[\s\S]*let close=[\s\S]*if\(!crossing&&!close\)/,
    "coarse W7 seeds must cover signed crossings and the adjacent one-cell phi band");
  const tracker = read("../lib/webgpu-octree-coarse-summary.ts");
  assert.match(tracker,
    /sum_i\(max\(u-a_i,0\)\^2\)=h\^2[\s\S]*sqrt\(discriminant\)/,
    "coarse redistance must solve the Euclidean upwind Eikonal equation");
  assert.match(tracker,
    /fn quantizePhi[\s\S]*let predicted=quantizePhi[\s\S]*let value=quantizePhi/,
    "coarse transport and volume correction must retain exact D4 phi pairs");
  for (const layer of [2, 3, 4, 5, 6, 7]) {
    assert.match(extension, new RegExp(`dilateLosassoAirBand${layer}`),
      `coarse support must publish deterministic graph layer ${layer}`);
  }
  const readyFlipStart = octree.indexOf("  encodeReadyTopologyFlip(");
  const readyFlip = octree.slice(readyFlipStart,
    octree.indexOf("\n  finishTopologyCandidate()", readyFlipStart));
  assert.doesNotMatch(readyFlip, /encodeCoarseExtensionBandPublication/,
    "a ready flip must retain the already-extended W7 graph after remapping wet ids");
  const surfaceStart = octree.indexOf("  encodeSurface(");
  const coarseSurfaceStart = octree.indexOf(
    "if (this.coarseOnlySurfaceTracking && this.coarseDynamics.backend === \"losasso\")",
    surfaceStart);
  const coarseSurface = octree.slice(coarseSurfaceStart,
    octree.indexOf("if (this.globalFineSeeds", coarseSurfaceStart));
  assert.match(coarseSurface,
    /if \(dt_s === 0\)[\s\S]*encodeCoarseExtensionBandPublication[\s\S]*backend\.encodeExtension\(/,
    "only the already-projected t=0 checkpoint may publish and extend a graph during surface setup");
  assert.match(coarseSurface,
    /if \(dt_s > 0\)[\s\S]*coarseOnlySummary\?\.encode/,
    "the already-published analytic t=0 tracker must not record a false missing-velocity advance");
  assert.doesNotMatch(coarseSurface,
    /const advanceSerial[\s\S]*backend\.encodeExtension\(/,
    "a live factor-one graph publication must not consume S3e before projection");
  const losassoStepStart = octree.indexOf("  private encodeLosasso(");
  const losassoStep = octree.slice(losassoStepStart,
    octree.indexOf("\n  /** Publish lazily allocated diagnostic textures", losassoStepStart));
  assert.match(losassoStep,
    /encodeProjection[\s\S]*coarseOnlySurfaceTracking[\s\S]*encodeCoarseExtensionBandPublication[\s\S]*encodeExtension[\s\S]*powerAdvancingPressureSteps \+= 1/,
    "factor one must rebuild and extend from the current projection before advancing the host serial");
});

test("Losasso factor-one permits a held surface only for its cold publication", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-coarse-summary.ts", import.meta.url), "utf8");
  assert.match(source, /losassoMode\(\)&&atomicLoad\(&state\[19\]\)==0u/,
    "only the first encoded publication may break the cold phi/velocity dependency cycle");
  assert.doesNotMatch(source, /trackerRunnable\(\)->bool\{[^}]*state\[16\]/,
    "the persistent initialized-bank bit must not excuse missing transport forever");
});

test("Losasso frontier cannot accept a live-to-empty topology transition", () => {
  const source = read("../lib/webgpu-octree.ts");
  const finalize = source.slice(source.indexOf("fn finalizeFrontier("),
    source.indexOf("fn classifyRowDelta(", source.indexOf("fn finalizeFrontier(")));
  assert.match(finalize, /&&\(previousCount==0u\|\|required>0u\)/,
    "a one-generation wetness gap must reject instead of entering the absorbing zero-row state");
});

test("Losasso V-cycle transfers preserve aggregate constants", () => {
  const source = read("../lib/webgpu-octree-losasso-vcycle-gpu.ts");
  const transfers = source.slice(source.indexOf("const transferWGSL"));
  assert.match(source, /let begin=childOffsets\[row\];let end=childOffsets\[row\+1u\]/,
    "restriction is owned by one parent over its stable child range");
  assert.doesNotMatch(source, /addFixedF32|atomicAdd\(&partials/,
    "fixed-order restriction does not need exact atomic deposits");
  assert.match(transfers, /output\[row\]\+=input\[parent\]/,
    "a constant coarse correction prolongs unchanged to every child");
  assert.doesNotMatch(transfers,
    /\(weights\[row\]\*coarseInverseVolumes\[parent\]\)\*input/);
});

test("Losasso speedup path is live-sized and fixed-order", () => {
  const backend = read("../lib/webgpu-octree-losasso-backend.ts");
  const backendShader = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  const hierarchy = read("../lib/webgpu-octree-losasso-hierarchy.ts");
  const vcycle = read("../lib/webgpu-octree-losasso-vcycle-gpu.ts");
  assert.match(backend, /dispatchWorkgroupsIndirect\(output\.faceDispatch, 0\)/,
    "face-wide publication follows the GPU-authored live face count");
  assert.match(backendShader,
    /@compute @workgroup_size\(256\)\s*fn prefixLosassoFaces/);
  assert.match(backendShader,
    /@compute @workgroup_size\(256\)\s*fn prefixLosassoIncidences/);
  assert.match(backendShader, /fn clearLosassoFaceDirectory/);
  assert.match(backendShader,
    /atomicStore\(&authority\[6\],directoryCapacity\)[\s\S]*faceDispatch\[3\]=\(directoryCapacity\+63u\)\/64u/,
    "the epoch-local face directory must be sized from live faces on the GPU");
  assert.match(backend,
    /dispatchWorkgroupsIndirect\(output\.faceDispatch, 12\)/,
    "directory maintenance must use the GPU-authored live prefix grid");
  assert.match(hierarchy, /encodeCoefficientRefresh/,
    "coefficient-only updates do not rebuild parent and CSR topology");
  assert.doesNotMatch(vcycle, /SIGNED_RADIX_256_F32_MAX_TERMS|restrictionPartials/);
  assert.match(vcycle, /octreeCompensatedF32WGSL/);
});

test("Losasso recurring migration and CG drains stay inside the compute stream", () => {
  const migration = read("../lib/webgpu-octree-losasso-velocity-migration.ts");
  const migrationShader = read("../lib/webgpu-octree-losasso-velocity-migration.wgsl.ts");
  const cg = read("../lib/webgpu-octree-pipelined-mgpcg.ts");
  const projection = read("../lib/webgpu-octree.ts");
  assert.doesNotMatch(migration, /broker\.copyBufferToBuffer/,
    "retired wet-face state must not force full-capacity blits or a pass break");
  assert.match(migration, /snapshotLosassoFaceState[\s\S]*snapshotLosassoFaceLookup/,
    "geometry and lookup payloads are split only to respect the storage-binding ceiling");
  assert.match(migrationShader,
    /fn snapshotLosassoFaceState[\s\S]*fn snapshotLosassoFaceLookup/);
  assert.match(migration,
    /dispatchWorkgroupsIndirect\(source\.faceDispatch, 0\)[\s\S]*dispatchWorkgroupsIndirect\(source\.faceDispatch, 12\)[\s\S]*dispatchWorkgroupsIndirect\(target\.faceDispatch, 0\)/,
    "snapshot and migration geometry work must follow live accepted/candidate face counts");
  assert.match(cg, /this\.usesCombinedReductionDrains \? 0 : 4/,
    "cooperative drains need no mutable-indirect transition inside an iteration");
  assert.match(cg,
    /if \(this\.usesCombinedReductionDrains\) \{[\s\S]*runAcceptedRowsIndirect\(pass, "advancePCGState"[\s\S]*runAcceptedRowsIndirect\(pass, "updateDirections"/,
    "combined iterations must use the immutable accepted-row dispatch record");
  assert.match(cg, /if \(!this\.usesCombinedReductionDrains\) \{[\s\S]*pipelined MGPCG convergence-tail publication/);
  assert.match(projection,
    /factorOneCombinedReductionDrains:\s*this\.coarseOnlySurfaceTracking && rowCapacity <= 4_096/,
    "factor-4 UI scenes and large factor-1 arenas must retain the scalable reduction schedule");
});

test("Losasso cold row publication is not suppressed by dry owner probes", () => {
  const projection = read("../lib/webgpu-octree.ts");
  const overflow = projection.slice(projection.indexOf("fn pressureOverflowed()"),
    projection.indexOf("fn axisVector(", projection.indexOf("fn pressureOverflowed()")));
  const emit = projection.slice(projection.indexOf("fn emitLeaves("),
    projection.indexOf("`;", projection.indexOf("fn emitLeaves(")));
  assert.match(overflow, /compaction\[pressureControlBase\(\)\] != 0u/);
  assert.doesNotMatch(overflow, /owners\[2\]/,
    "a rejected dry-support lookup must not cancel every valid pressure-row write");
  assert.match(emit,
    /let acceptedOwner=ownerAtIndex\(cell\)[\s\S]*leafHeaders\[row\] = LeafHeader/,
    "row emission must still validate the exact owner used to publish each header");
});

test("Losasso recurring topology publication reuses exact bind variants", () => {
  const backend = read("../lib/webgpu-octree-losasso-backend.ts");
  const readyCommit = read("../lib/webgpu-octree-losasso-ready-commit.ts");
  assert.match(backend, /private readonly bindGroupCache = new Map<GPUComputePipeline/);
  assert.match(backend,
    /const cached = variants\.find[\s\S]*variant\.buffers\[index\] === buffers\[index\]/,
    "cache identity must include every exact binding and buffer");
  assert.equal((backend.match(/this\.cachedBindGroup\(pipeline,/g) ?? []).length, 2,
    "candidate construction and ready commit should both reuse stable bind variants");
  assert.match(readyCommit,
    /let groups=\(count\+63u\)\/64u;let width=min\(groups,65535u\)[\s\S]*@builtin\(num_workgroups\)n:vec3u[\s\S]*g\.x\+g\.y\*n\.x\*64u/,
    "capacity-shaped accepted-row commits must tile across WebGPU's portable 2-D dispatch grid");
  const authorityCommit = read("../lib/webgpu-octree-losasso-authority-commit.wgsl.ts");
  assert.match(backend,
    /const dispatchLimit = this\.device\.limits\.maxComputeWorkgroupsPerDimension[\s\S]*Math\.ceil\(count \/ Math\.max\(1, width\)\)/,
    "capacity-shaped authority-bank copies must tile on the device dispatch limit");
  assert.match(authorityCommit,
    /fn linearInvocation\(g:vec3u,n:vec3u\)->u32\{return g\.x\+g\.y\*n\.x\*64u;\}/,
    "authority-bank copy kernels must consume the complete 2-D dispatch grid");
});

test("Losasso exact row reuse carries the accepted topology bank on the GPU", () => {
  const backendWGSL = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  const authorityCommitWGSL = read("../lib/webgpu-octree-losasso-authority-commit.wgsl.ts");
  const readyCommit = read("../lib/webgpu-octree-losasso-ready-commit.ts");
  assert.match(backendWGSL,
    /fn exactRowTopologyReuse[\s\S]*sourceFrontier\[base\+8u\]==0x52444c54u[\s\S]*sourceFrontier\[base\+5u\]==0u/,
    "candidate publication should consume the exact GPU row-delta reuse receipt");
  assert.match(backendWGSL, /fn rows\(\) -> u32 \{ return select\([\s\S]*topologyReused\(\)\); \}/,
    "reused topology must zero the rebuild's row-shaped work without host branching");
  assert.match(authorityCommitWGSL,
    /if\(reused\(\)\)\{acceptedSolver\[4u\]=candidateControl\[0u\]/,
    "ready publication should restamp epochs while retaining accepted topology arrays");
  assert.match(readyCommit,
    /dispatch\[0\]=select\(0u,width,valid&&!reuse\)/,
    "exact reuse must preserve accepted row headers and pressure rather than copying a candidate bank");
});

test("Losasso fused sub-L0 schedule is the resident-tier default", () => {
  const source = read("../lib/webgpu-octree-losasso-vcycle-gpu.ts");
  const hierarchy = read("../lib/webgpu-octree-losasso-hierarchy.ts");
  assert.match(source, /fn fusedSubL0\(/);
  assert.doesNotMatch(source, /FLUID_LOSASSO_FUSED_SUB_L0/);
  assert.match(source, /this\.useFusedSubL0=Boolean\(hierarchy\.fusedSubL0&&hierarchy\.levels\.length>1/);
  // The fused cycle is ONE workgroup grid-striding every sub-L0 row, so it must
  // stay size-gated: ungated, a 128-cubed hierarchy walks ~30K L1 rows serially
  // per sweep per level per iteration. Measured 2026-08-06: removing the gate
  // costs more wall than the 10x iteration reduction it accompanies wins back.
  assert.match(source, /const FUSED_SUB_L0_MAXIMUM_LEVEL_ROWS = 4_096;/,
    "the fused cycle must be bounded by the resident row tier");
  assert.match(source,
    /subLevelRows\.every\(capacity=>capacity<=FUSED_SUB_L0_MAXIMUM_LEVEL_ROWS\)/,
    "every sub-L0 level must fit the resident tier before the fused form is chosen");
  assert.match(source, /this\.useFusedSubL0\s*\?9:/,
    "the accepted fused graph has nine encoded correction dispatches");
  assert.match(hierarchy,
    /const levelRowCapacityAt = \(targetSpan: number\) => Math\.min\(rows,\s*\(nx \/ targetSpan\) \* \(ny \/ targetSpan\) \* \(nz \/ targetSpan\)\);/,
    "coarse vector bounds should come from the number of dyadic target cells");
  // A level below the row floor has no parallelism to relax with but still
  // carries tens of thousands of retained face patches. Measured 2026-08-06:
  // the bottom two levels of the 128-cubed hierarchy cost ~460 ms to take the
  // iteration count from 27 to 17.
  assert.match(hierarchy, /export const MINIMUM_COARSE_LEVEL_ROWS = 1_024;/,
    "hierarchy depth must stop where a level can no longer be relaxed in parallel");
  assert.match(hierarchy,
    /if \(levelRowCapacityAt\(targetSpan\) < MINIMUM_COARSE_LEVEL_ROWS\) break;/,
    "the depth cap must be applied when the level list is built");
  assert.match(source, /VECTOR_BASES[\s\S]*fn vectorIndex\(level:u32,row:u32\)->u32\{return VECTOR_BASES\[level\]\+row;\}/,
    "the fused cycle must index packed per-level vector sections");
});

test("Losasso exact topology reuse refreshes hierarchy fields without rebuilding geometry", () => {
  const shader = readFileSync("lib/webgpu-octree-losasso-hierarchy.wgsl.ts", "utf8");
  const hierarchy = readFileSync("lib/webgpu-octree-losasso-hierarchy.ts", "utf8");
  assert.match(shader, /fn buildLosassoParentRows[\s\S]*if\(fineControl\[5\]==1u\)\{return;\}/);
  assert.match(shader, /fn refreshLosassoReusedCoarseFaces[\s\S]*refreshCoarseFace\(g\.x,true\)/);
  assert.match(shader, /coarseControl\[0\]=fineControl\[0\];coarseControl\[5\]=1u/);
  assert.match(hierarchy, /encodeTransition\(pass,[\s\S]*encodeReusedTransitionRefresh\(pass,/);
});

test("Losasso face publication masks T-junction quadrants and stages walls closed", () => {
  const source = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  assert.match(source, /struct FacePatchPlan \{ mask: u32, subdivision: u32 \}/);
  assert.match(source, /mask \|= 1u << \(a \+ 2u \* b\)/,
    "each uncovered negative subface gets its own publication bit");
  assert.match(source, /countOneBits\(positiveFacePlan\(header, axis\)\.mask\)/);
  assert.match(source, /if \(\(negativePlan\.mask & patchBit\) != 0u\)/);
  assert.equal((source.match(/select\(1\.0, 0\.0, closedBoundary\)/g) ?? []).length, 2,
    "both wall orientations must be staged with zero aperture");
  assert.match(source,
    /face\.openFraction=select\(openSum\/f32\(span\*span\),0\.,closedBoundary\)/,
    "solid conditioning must not reopen a staged container wall");
});

test("Losasso diagnostics and tripwires are no longer backend-gated", () => {
  const uniform = read("../lib/webgpu-uniform-eulerian.ts");
  const stats = uniform.slice(uniform.indexOf("async readStats()"), uniform.indexOf("\n  destroy()"));
  assert.match(stats,
    /globalFineDiagnosticsPromise = compactFineExpected[\s\S]*readGlobalFineLevelSetDiagnostics/);
  assert.doesNotMatch(stats,
    /globalFineDiagnosticsPromise = this\.octreeProjection\?\.coarseBackend === "power2017"/);
  assert.match(stats, /if \(failures\.length > 0\) \{[\s\S]*\[losasso-step-receipt\]/,
    "every bad receipt is emitted even after the first sequence fault");

  const smoke = read("../tools/webgpu-smoke-executor.ts");
  assert.match(smoke, /const tripwireCapacity = !tripwiresDisabled && method\.id === "octree"/);
  assert.doesNotMatch(smoke,
    /const tripwireCapacity =[^\n]*!losassoCutoverLane/);
  assert.match(smoke,
    /losassoCutoverLane\s*\? \["topology", "mgpcg", "fineWorklist"\]/);
});

test("Losasso consumes the shared residual-gated hard iteration ceiling", () => {
  const source = read("../lib/webgpu-octree.ts");
  const construction = source.slice(source.indexOf("this.losassoBackend ="),
    source.indexOf("this.pressureSolverControl", source.indexOf("this.losassoBackend =")));
  assert.match(construction,
    /maximumIterations: this\.solveTailPolicy\.hardOuterIterationCeiling/);
  assert.match(construction,
    /hardIterationCeiling: this\.solveTailPolicy\.hardOuterIterationCeiling/);
  assert.doesNotMatch(construction, /maximumIterations: 32|hardIterationCeiling: 32/);
});

test("Losasso resolves ambiguous redistance seeds without changing Power", () => {
  const redistance = read("../lib/webgpu-octree-fine-levelset-redistance.ts");
  assert.match(redistance,
    /axisPermutationInvariantSeeds \? "if\(tied\)\{return 6u<<24u;\}" : ""/,
    "an equal axis-distance crossing uses the centre sentinel on the symmetric lane");
  assert.match(redistance,
    /axisPermutationInvariantSeeds = false/,
    "shared and Power construction retains the historical axis tie rule by default");

  const octree = read("../lib/webgpu-octree.ts");
  const losasso = octree.slice(octree.indexOf("const redistanceOptions ="),
    octree.indexOf("this.globalFineRedistanceA", octree.indexOf("const redistanceOptions =")));
  assert.match(losasso, /axisPermutationInvariantSeeds: true/);
});

test("Losasso coarse advection reconstructs nodes and trilinearly samples them", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-dynamics.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(source, /fn velocityAtNode\(node: vec3u, field: u32\)/);
  assert.match(source, /for \(var quadrant = 0u; quadrant < 4u; quadrant \+= 1u\)/);
  assert.match(source, /face = containingFace\(axis, unsignedCoordinate\)/);
  assert.match(source, /result\[axis\] = exactValue\(&exact\) \/ f32\(sampleCount\)/);
  assert.match(source, /fn velocityAtGrid\(gridValue: vec3f, field: u32\)/);
  assert.match(source, /for \(var corner = 0u; corner < 8u; corner \+= 1u\)/);
  assert.match(source, /addExact\(&exactX, weight \* sample\.value\.x\)/);
  assert.match(source, /let weight = symmetricProduct3\(weightVector\)/);
  assert.match(source, /quantizeTraceOffset\(-0\.5 \* gridDt \* first\.value\)/);
  assert.match(source, /let midpoint = velocityAtGrid\(origin \+ midpointOffset, field\)/,
    "characteristics should use a midpoint carrier rather than forward Euler");
});

test("Losasso velocity advection uses a bounded MacCormack correction", () => {
  const shader = read("../lib/webgpu-octree-losasso-dynamics.wgsl.ts");
  const host = read("../lib/webgpu-octree-losasso-dynamics.ts");
  const backend = read("../lib/webgpu-octree-losasso-backend.ts");
  const band = read("../lib/webgpu-octree-losasso-extension-band.ts");
  const bandShader = read("../lib/webgpu-octree-losasso-extension-band.wgsl.ts");
  const extension = read("../lib/webgpu-octree-losasso-velocity-extension.ts");
  assert.match(shader, /fn reverseLosassoFaces/);
  assert.match(shader, /traceVelocity\(centre, -params\.domainOriginDt\.w, ADVECTED_FIELD\)/);
  assert.match(shader,
    /let corrected = prediction \+ 0\.5 \* \(original\.value\[axis\] - reversed\)/);
  assert.match(shader,
    /advectedVelocity\[faceId\] = clamp\(corrected,[\s\S]*sourceStencil\.lower\[axis\],[\s\S]*sourceStencil\.upper\[axis\]\)/,
    "the error correction must stay inside the forward source stencil");
  assert.match(shader, /predictedVelocity\[faceId\] = INVALID_VELOCITY/,
    "missing sparse reverse support must disable rather than extrapolate a correction");
  assert.match(shader,
    /if \(original\.boundary \|\| departure\.boundary \|\| sourceStencil\.boundary[\s\S]*FACE_INTERFACE_NEARBY\) != 0u\) \{ return; \}/,
    "MacCormack must fall back to first order at closed walls and the liquid interface");
  assert.match(shader,
    /if \(arrival\.valid && !arrival\.boundary[\s\S]*FACE_INTERFACE_NEARBY\) == 0u\)/,
    "the reverse characteristic must not re-enable correction across a wall or surface");
  const coarsePhi = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  assert.match(coarsePhi,
    /face\.reserved&=~FACE_INTERFACE_NEARBY;[\s\S]*targetLoad\(rowBase\+8u\*face\.negativeRow\+5u,reuse\)&INTERFACE/,
    "coarse publication must stamp interface adjacency onto the existing face ABI");
  const advection = host.slice(host.indexOf("encodeAdvection("),
    host.indexOf("encodeForcesAndDivergence("));
  assert.ok(advection.indexOf("advectLosassoFaces") < advection.indexOf("reverseLosassoFaces"));
  assert.ok(advection.indexOf("reverseLosassoFaces") < advection.indexOf("correctLosassoFaces"));
  assert.match(shader, /predictedVelocity is scratch here/,
    "MacCormack should reuse the force output field rather than allocate another face array");
  assert.match(backend, /encodePredictorExtension\([\s\S]*advectedVelocity/,
    "the reverse pass must sample a complete predictor on the published W7 graph");
  assert.match(band, /this\.predictorVelocity = this\.projectedSeeds/,
    "the band predictor should reuse the projected-seed arena");
  const gather = bandShader.slice(bandShader.indexOf("fn gatherLosassoProjectedSeeds"));
  assert.match(gather, /let wet=seedWetFace\[id\]/,
    "advance-time seed gathering must remain one direct mapped read");
  assert.doesNotMatch(gather, /containingWet|wetDirectory/,
    "topology lookup must not leak into the advection-time gather");
  assert.match(band, /encodeTopologyRemap\(broker: PassBroker\)/,
    "topology changes should refresh direct seed ids once at the epoch boundary");
  const remap = bandShader.slice(bandShader.indexOf("fn clearLosassoDenseWetFaces"),
    bandShader.indexOf("fn gatherLosassoProjectedSeeds"));
  assert.match(remap, /atomicMin\(&denseWetFace\[key\],encoded\)/,
    "topology remapping should scatter into a bounded dense finest-face map");
  assert.doesNotMatch(remap, /wetDirectory|exactWet|containingWet|for\(var probe/,
    "topology remapping must not chase the wet hash directory");
  assert.match(remap, /MULTI_OWNER/,
    "a retired coarse seed spanning refined owners must remain a live averaged seed");
  const multiGather = bandShader.slice(bandShader.indexOf("fn gatherLosassoProjectedSeeds"));
  assert.match(multiGather, /wet==MULTI_OWNER[\s\S]*exactAdd\(&exact,value\)[\s\S]*exactValue\(exact\)\/max\(1\.,count\)/,
    "multi-owner seeds must area-average the finest projected faces instead of becoming zero");
  assert.match(backend,
    /publisher\.encodeReadyCommit\(broker, input\);[\s\S]*extensionBand\.encodeTopologyRemap\(broker\)/,
    "the seed remap must observe the newly accepted wet authority");
  assert.match(extension,
    /predictorVelocities = \[this\.source\.projectedVelocity,[\s\S]*this\.source\.projectedVelocity\]/,
    "the existing Jacobi scratch pair should extend the predictor in place");
});

test("Losasso wall release uses regional volume control without porous contacts", () => {
  const projection = read("../lib/webgpu-octree-losasso-projection.ts");
  const host = read("../lib/webgpu-octree.ts");
  assert.match(projection, /let releasePressure = select\(0\.0, params\.contactPressure, overhead\)/,
    "side walls should release under tension instead of remaining welded Neumann faces");
  assert.match(projection, /solved < renewalPressure, wasSeparated/,
    "released wall contact needs hysteresis around ambient pressure");
  assert.match(projection, /projected = outward \* max\(0\.0, outward \* projected\)/,
    "a released contact must reject velocity returning into the wall");
  assert.match(host,
    /measureOnlyLosasso[\s\S]*pending\.volume\.encodeMeasurement\(redistanceBroker\)[\s\S]*pending\.volume\.encode\(redistanceBroker\)/,
    "Losasso should apply regional correction by default while retaining a measure-only Dawn A\/B");
  assert.match(host, /"moving-pages"/,
    "Losasso volume replacement must exclude sleeping wall films and fragments");
});

test("Losasso coarse publication cannot resurrect an invalid retired row", () => {
  const shader = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  const restrict = shader.slice(shader.indexOf("fn restrictLosassoCoarsePhiRow"),
    shader.indexOf("@compute", shader.indexOf("fn restrictLosassoCoarsePhiRow")));
  assert.match(restrict, /flags=seed\.w&\(VALID\|FINITE\|FINE_RESTRICTED\)/);
  assert.match(restrict, /if\(centreFine\.valid!=0u\)\{flags=VALID\|FINITE/);
  assert.match(restrict, /if\(\(flags&\(VALID\|FINITE\)\)==\(VALID\|FINITE\)&&minimum<=0\./,
    "a stale or zeroed seed must remain invisible rather than publish a phantom interface");
});

test("Losasso paper-fidelity pressure and extension remain construction-time A/B choices", () => {
  const method = read("../lib/methods/octree.ts");
  const coarse = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  const projection = read("../lib/webgpu-octree-losasso-projection.ts");
  const extension = read("../lib/webgpu-octree-losasso-velocity-extension.wgsl.ts");
  assert.match(method, /losassoFreeSurfacePressure/);
  assert.match(method, /losassoVelocityExtension/);
  assert.match(coarse, /let cellCenteredAir=p\.schedule\.w==1u/);
  assert.match(coarse, /theta=1\.;distance=dual;face\.inverseDistance=1\.\/dual/);
  assert.match(projection, /params\.reserved0 == 0u/,
    "the cell-centered paper A/B must disable unilateral wall contact");
  assert.match(extension, /if \(params\.causalFront != 0u\)/);
  assert.match(extension, /if \(layer < params\.sweep\)/);
  assert.match(extension, /if \(layer > params\.sweep\)/);
});

test("Losasso inflow overrides the visual nozzle's solid aperture", () => {
  const source = read("../lib/webgpu-octree-losasso-dynamics.wgsl.ts");
  const host = read("../lib/webgpu-octree-losasso-dynamics.ts");
  const force = source.slice(source.indexOf("fn forceLosassoFaces"),
    source.indexOf("@group(0) @binding(11)"));
  assert.match(force,
    /let resolved = face\.solidNormalVelocity[\s\S]*let value = select\(resolved, inflow\.x, inflow\.y > 0\.5\)/,
    "the divergence input must receive the prescribed source even through the solid nozzle cap");
  assert.doesNotMatch(force,
    /face\.openFraction \* \(inflow\.x - face\.solidNormalVelocity\)/,
    "the source boundary condition must not be attenuated by visual solid coverage");

  const constraint = source.slice(source.indexOf("fn constrainLosassoInflowFaces"),
    source.indexOf("fn divergenceLosassoRows"));
  assert.match(constraint,
    /projectedVelocity\[faceId\] = select\(0\.0, inflow\.x, finite\(inflow\.x\)\)/,
    "post-projection restoration must preserve the exact authored source velocity");
  assert.doesNotMatch(constraint, /face\.openFraction/,
    "the visual nozzle aperture must not erase the restored source");
  assert.match(host, /constrainLosassoInflowFaces: \[0, 1, 3, 11\]/,
    "the constraint bind group must omit the solid-face buffer it no longer reads");
});

test("Losasso recurring fine transport keeps the authored nozzle source alive", () => {
  const host = read("../lib/webgpu-octree.ts");
  const transport = read("../lib/webgpu-octree-losasso-fine-transport.ts");
  const shader = read("../lib/webgpu-octree-losasso-fine-transport.wgsl.ts");
  const selection = host.slice(host.indexOf("transport as WebGPUOctreeLosassoFineTransport"),
    host.indexOf(": (transport as WebGPUFineLevelSetTransport"));
  assert.match(selection, /\.\.\.\(inflow \? \{ inflow \} : \{\}\)/,
    "the Losasso transport call must receive the same live source state as Power");
  assert.match(transport, /readonly inflow\?: SurfaceInflowState/);
  assert.match(shader,
    /fn applyInflowPhi[\s\S]*nextPhi\[index\]=applyInflowPhi\(transported\+exitCells\*p\.fineCellWidth,world\)/,
    "resident nozzle pages must have source phi restored after every accepted characteristic");
});

test("Losasso cold fine bootstrap retains the analytic liquid SDF around solids", () => {
  const source = read("../lib/webgpu-octree.ts");
  const selection = source.slice(source.indexOf("const exactAnalyticFineSeed"),
    source.indexOf("this.globalFineSeeds = new WebGPUFineLevelSetLeafSeeds"));
  assert.match(selection,
    /initialBrickSeeds_m\?\.length \?\? 0\) === 0[\s\S]*!sceneDamBreakIsOffsetFromCorner\(this\.scene\)/,
    "unseeded tank/dam liquid keeps exact phi even when solids require dense topology");
  assert.doesNotMatch(selection, /this\.analyticSparseBootstrap/,
    "rigid geometry must not make the authored liquid SDF unavailable to cold halo pages");
});

test("Losasso diagnostics do not interpret absent Power controls as failures", () => {
  const source = read("../lib/webgpu-uniform-eulerian.ts");
  assert.match(source,
    /const powerStructuredAuthority = this\.octreeProjection\?\.coarseBackend === "power2017"/);
  assert.match(source,
    /const liveFailure = powerStructuredAuthority && !!support/);
  assert.match(source,
    /const latchedFailure = powerStructuredAuthority &&/);
});
