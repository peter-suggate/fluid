import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

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
  assert.match(host, /run\("dilateLosassoAirBand7"/,
    "W4 transport plus a three-axis trilinear corner requires all three closure layers");
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

test("Losasso fused sub-L0 schedule is the multi-level default", () => {
  const source = read("../lib/webgpu-octree-losasso-vcycle-gpu.ts");
  const hierarchy = read("../lib/webgpu-octree-losasso-hierarchy.ts");
  assert.match(source, /fn fusedSubL0\(/);
  assert.doesNotMatch(source, /FLUID_LOSASSO_FUSED_SUB_L0/);
  assert.match(source,
    /this\.useFusedSubL0=Boolean\(hierarchy\.fusedSubL0&&hierarchy\.levels\.length>1\)/);
  assert.match(source, /this\.useFusedSubL0\s*\?9:/,
    "the accepted fused graph has nine encoded correction dispatches");
  assert.match(hierarchy,
    /levelRowCapacities\.push\(Math\.min\(rows,[\s\S]*\(nx \/ targetSpan\) \* \(ny \/ targetSpan\) \* \(nz \/ targetSpan\)\)\)/,
    "coarse vector bounds should come from the number of dyadic target cells");
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
  assert.match(source, /fn velocityAtNode\(node: vec3u\)/);
  assert.match(source, /for \(var quadrant = 0u; quadrant < 4u; quadrant \+= 1u\)/);
  assert.match(source, /let face = containingFace\(axis, unsignedCoordinate\)/);
  assert.match(source, /result\[axis\] = exactValue\(&exact\) \/ f32\(sampleCount\)/);
  assert.match(source, /fn velocityAtGrid\(gridValue: vec3f\)/);
  assert.match(source, /for \(var corner = 0u; corner < 8u; corner \+= 1u\)/);
  assert.match(source, /result \+= weight \* sample\.value/);
  assert.match(source, /let midpoint = velocityAtGrid/,
    "characteristics should use a midpoint carrier rather than forward Euler");
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

test("Losasso diagnostics do not interpret absent Power controls as failures", () => {
  const source = read("../lib/webgpu-uniform-eulerian.ts");
  assert.match(source,
    /const powerStructuredAuthority = this\.octreeProjection\?\.coarseBackend === "power2017"/);
  assert.match(source,
    /const liveFailure = powerStructuredAuthority && !!support/);
  assert.match(source,
    /const latchedFailure = powerStructuredAuthority &&/);
});
