import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { simulationMethods } from "../lib/methods";
import { octreeMethod } from "../lib/methods/octree";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { octreeDiagnosticShader, octreePressureCouplingShader, octreeProjectionShader, WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { quadtreeSurfaceShader, WebGPUQuadtreeSurfaceState } from "../lib/webgpu-quadtree-builder";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const uniformSolverSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("octree is a registered GPU method with dam-break defaults", () => {
  assert.ok(simulationMethods.includes(octreeMethod));
  assert.equal(octreeMethod.id, "octree");
  assert.equal(octreeMethod.backend, "webgpu");
  assert.equal(octreeMethod.presetFor("balanced").pressureIterations, 128);
  assert.equal(octreeMethod.presetFor("balanced").maximumLeafSize, "16");
  assert.equal(octreeMethod.presetFor("high").maximumLeafSize, "16");
  assert.equal(octreeMethod.presetFor("balanced").adaptivity, 1);
  assert.match(octreeMethod.detail, /no topology readbacks/);
  assert.match(octreeMethod.detail, /Chebyshev-Jacobi/);
  assert.match(octreeMethod.detail, /rigid-body coupling/);
  assert.match(octreeMethod.description, /signed-distance level set/);
  const maximumLeaf = octreeMethod.params.find((spec) => spec.key === "maximumLeafSize");
  assert.ok(maximumLeaf && maximumLeaf.kind === "select");
  assert.equal(maximumLeaf.default, "16");
  assert.deepEqual(maximumLeaf.options.map((option) => option.value), ["2", "4", "8", "16", "32"]);
  assert.match(octreeSource, /function octreeLeafSize\(value: number\): 2 \| 4 \| 8 \| 16 \| 32/);
  assert.match(octreeSource, /rounded >= 32/);
  const interfaceBand = octreeMethod.params.find((spec) => spec.key === "interfaceRefinementBandCells");
  assert.ok(interfaceBand && interfaceBand.kind === "number" && interfaceBand.tier === "fine" && interfaceBand.default === 4);
  assert.doesNotMatch(`${octreeSource}\n${uniformSolverSource}`, /airRefinementBandCells/);
  assert.match(uniformSolverSource, /interfaceRefinementBandCells: options\.octree\.interfaceRefinementBandCells \?\? 4/);
  const surfaceDetail = octreeMethod.params.find((spec) => spec.key === "surfaceDetailStrength");
  assert.ok(surfaceDetail && surfaceDetail.kind === "number" && surfaceDetail.default === 0);
  for (const quality of ["balanced", "high", "ultra"] as const) {
    assert.equal(octreeMethod.presetFor(quality).surfaceDetailStrength, 0, "dynamic refinement must be uniformly opt-in across quality presets");
  }
  const particleCorrection = octreeMethod.params.find((spec) => spec.key === "secondaryParticleSurfaceCorrection");
  assert.ok(particleCorrection && particleCorrection.kind === "number" && particleCorrection.default === 0);
  for (const quality of ["balanced", "high", "ultra"] as const) {
    assert.equal(octreeMethod.presetFor(quality).secondaryParticleSurfaceCorrection, 0, "particle feedback must be uniformly opt-in across quality presets");
  }
  const warmStart = octreeMethod.params.find((spec) => spec.key === "pressureWarmStart");
  assert.ok(warmStart && warmStart.kind === "select" && warmStart.tier === "fine" && warmStart.default === "on");
  // Options are copied field-by-field into the solver; a dropped key would
  // silently revert the UI toggle to its default.
  assert.match(octreeSource, /pressureWarmStart\?: boolean/);
  assert.match(uniformSolverSource, /pressureWarmStart: options\.octree\.pressureWarmStart/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode.replace(/\s+/g, ""), /if\(!this\.pressureWarmStart\)\{encoder\.clearBuffer\(this\.pressureA\);encoder\.clearBuffer\(this\.pressureB\);?\}/);
});

test("octree participates in the shared two-way immersed-body coupling path", () => {
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.match(methodSource, /values, onRigidLoads\) => new WebGPUUniformEulerianSolver\(device, scene, quality, onRigidLoads/);
  assert.match(methodSource, /values, onRigidLoads, onProgress\) => WebGPUUniformEulerianSolver\.createAsync/);
  assert.match(uniformSolverSource, /const activeBodies = bodies\.slice\(0, 12\)/);
  assert.doesNotMatch(uniformSolverSource, /this\.octreeProjection \? \[\] : bodies/);
  assert.match(uniformSolverSource, /if \(this\.adaptiveProjection\) this\.solidPhiGroup/);
  assert.match(uniformSolverSource, /texture: this\.adaptiveProjection\.levelSetTexture/);
  assert.match(uniformSolverSource, /this\.dispatch\(pass, this\.rigidPipeline, this\.rigidGroup\)/);
  assert.match(uniformSolverSource, /this\.rigidSystem\.encode\(encoder, delta/);
  assert.doesNotMatch(uniformSolverSource.slice(uniformSolverSource.indexOf("advanceTo(time_s"), uniformSolverSource.indexOf("async readStats()")), /mapAsync|encodeBodyImpulseReadback/);
});

test("octree voxelizes partial solid volume and reports liquid-displaced volume", () => {
  assert.match(octreeProjectionShader, /fn bodySolidFraction/);
  assert.match(octreeProjectionShader, /for \(var corner = 0u; corner < 8u/);
  assert.match(octreeProjectionShader, /return inside \/ 8\.0/);
  assert.match(octreeProjectionShader, /fn rasterizeSolids/);
  assert.match(octreeProjectionShader, /if \(candidate > fraction\) \{ fraction = candidate; owner = i32\(bodyIndex\); \}/,
    "overlapping terrain/body samples must have one maximum-coverage owner");
  assert.match(legacyUniformComputeShader, /let displacedWeight=wetFraction\*solidFraction/,
    "buoyancy volume must integrate liquid occupancy times sub-cell solid volume");
  assert.match(legacyUniformComputeShader, /if\(candidate>solidFraction\)\{solidFraction=candidate;coupledBody=bodyIndex;\}/,
    "overlapping bodies must not double-count displaced volume");
});

test("octree pressure solve uses the variational solid face constraint", () => {
  const assemble = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn assembleSystem"), octreeProjectionShader.indexOf("fn iterateLeaves"));
  assert.match(assemble, /let open = 1\.0 - clamp\(solid\.fraction/);
  assert.match(assemble, /let coefficient = open \* area \/ max\(distance/);
  assert.match(assemble, /constrainedFaceVelocity\(faceCell, axis, solid\)/);
  assert.match(octreeProjectionShader, /open \* component\(velocityAt\(faceCell\), axis\) \+ solid\.fraction \* component\(solidVelocity/);
  assert.match(octreeProjectionShader, /if \(crossesSurface \|\| crossesSolidBoundary\) \{ return true; \}/,
    "solid interfaces must force finest octree leaves");
});

test("octree retains exact rank-six coupling as an A/B path and defaults to lagged feedback", () => {
  assert.match(octreeProjectionShader, /K M\^-1 K\^T/);
  assert.match(octreeProjectionShader, /fn gatherBodyCoupling/);
  assert.match(octreeProjectionShader, /faceArea\(axis\) \* solid\.fraction \* \(p1 - p0\)/);
  assert.match(octreeProjectionShader, /let coupling = leafBodyCoupling/);
  assert.match(octreeProjectionShader, /effectiveDiagonal = header\.diagonal \+ coupling\.y/);
  assert.match(octreeProjectionShader, /fn applyBodyCoupling/);
  assert.match(octreeProjectionShader, /body\.linearVelocity\.xyz - linear/);
  assert.match(octreeProjectionShader, /body\.angularVelocity\.xyz - angular/,
    "the exact compact ladder must retain its same-step pressure-updated body velocity");
  assert.match(octreePressureCouplingShader, /params\.physical\.x \* faceArea\(axis\) \* solid\.fraction \* \(p0 - p1\)/);
  assert.match(octreePressureCouplingShader, /0\.5 \* f32\(owner\.size - 1u\)/,
    "pressure-to-body coupling must classify leaves at the same geometric centre as projection");
  assert.match(octreePressureCouplingShader, /atomicAdd\(&rigidExchange\[base/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /gatherBodyCoupling\(pressure,group\)/);
  assert.match(encode, /constuseChebyshev=this\.leafSolver==="chebyshev"/);
  assert.match(encode, /constuseLaggedRigidCoupling=useChebyshev&&this\.couplingHasDynamicBodies/);
  assert.match(encode, /if\(!useLaggedRigidCoupling\)gatherBodyCoupling\(project,finalGroup\)/);
  assert.match(encode, /this\.pressureImpulsePipeline/,
    "lagged coupling must still publish the current pressure impulse for the next batch");
});

test("octree topology is genuinely three-dimensional and 2:1 balanced", () => {
  assert.match(octreeProjectionShader, /packOrigin\(p: vec3u\)/);
  assert.match(octreeProjectionShader, /p\.z << 20u/);
  assert.match(WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString(), /Math\.ceil\(Math\.log2\(this\.maxLeafSize\)\)/);
  assert.match(octreeProjectionShader, /for \(var z = 0u; z < size/);
  assert.match(octreeProjectionShader, /for \(var y = 0u; y < size/);
  assert.match(octreeProjectionShader, /for \(var x = 0u; x < size/);
  assert.match(octreeProjectionShader, /neighborTooFine/);
  assert.match(octreeProjectionShader, /\.size \* 2u < size/);
});

test("octree refinement is graded by resident signed distance rather than bulk VOF occupancy", () => {
  assert.match(octreeProjectionShader, /levelSetIn: texture_3d<f32>/);
  assert.doesNotMatch(octreeProjectionShader, /volumeIn/, "the octree solve must not bind the diagnostic VOF field");
  assert.match(octreeProjectionShader, /let samplePhi = phi\(vec3i\(q\)\)/);
  assert.match(octreeProjectionShader, /closestSurface = min\(closestSurface, abs\(samplePhi\)\)/);
  assert.match(octreeProjectionShader, /if \(minimumSolid >= 1\.0 - 1e-5\) \{ return false; \}/,
    "fully solid bulk leaves should be allowed to stay coarse");
  assert.match(octreeProjectionShader, /let effectiveBand = baseBand \+ 8\.0 \* detailActivity/);
  assert.match(octreeProjectionShader, /return closestSurface < effectiveBand \* finestWidth;/,
    "pure air and liquid leaves should use the explicit band plus bounded local detail support");
  assert.match(octreeProjectionShader, /params\.physical\.w \* clamp\(max\(strainActivity, 2\.0 \* maximumCurvatureProxy\)/);
  assert.match(octreeProjectionShader, /params\.physical\.w > 0\.0/, "zero detail strength must skip activity sampling and preserve the baseline sizing path");
  assert.doesNotMatch(octreeProjectionShader, /closestSurface \* adaptivity < f32\(size\) \* finestWidth/);
  const refinement = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn leafNeedsRefinement"), octreeProjectionShader.indexOf("fn splitLeaf"));
  assert.doesNotMatch(refinement, /wet != liquidCell|a > 0\.001/);
  assert.match(octreeProjectionShader, /fn liquidCell\(p: vec3i\) -> bool \{ return valid\(p\) && phi\(p\) < 0\.0; \}/);
});

test("octree preserves advected level-set volume with GPU-only feedback", () => {
  assert.match(quadtreeSurfaceShader, /fn correctLevelSetVolume/);
  assert.match(quadtreeSurfaceShader, /let desiredVolume = params\.inflowTiming\.y/);
  assert.match(quadtreeSurfaceShader, /atomicLoad\(&reductions\[0\]\)/);
  assert.match(quadtreeSurfaceShader, /u32\(occupied \* open \* 256\.0 \+ 0\.5\)/, "mixed open-cell volume reduction must be unbiased");
  assert.match(quadtreeSurfaceShader, /abs\(value\) < 2\.0 \* params\.cellAndDt\.y/, "controller derivative must use the Heaviside support");
  assert.match(quadtreeSurfaceShader, /occupied \* open \* 256\.0/, "volume control must conserve liquid only in the open fraction of each cell");
  assert.match(quadtreeSurfaceShader, /surfaceSolids\[index3\(gid\)\]\.fraction/, "the controller must consume the octree's current VOS field");
  assert.match(octreeSource, /true, true, this\.solidCells/, "octree must bind its freshly rasterized solid fractions into surface-volume control");
  assert.match(octreeSource, /readSurfaceDiagnostics\(\) \{ return this\.surfaceState\.readVolumeDiagnostics\(\); \}/,
    "octree telemetry must report its authoritative level-set volume rather than the dormant VOF texture");
  assert.match(legacyUniformComputeShader, /let represented=surfaceOccupancy\(id\)\*open/, "reported liquid volume must exclude the same displaced solid fraction");
  assert.match(legacyUniformComputeShader, /for\(var step=1;step<=64;step\+=1\)/, "phi-s must reach open liquid across a newly submerged large solid in one pass");
  assert.match(legacyUniformComputeShader, /exteriorSum\+=exteriorWeight\*textureLoad\(pressureIn,exterior,0\)\.x/, "solid-interior phase must be extended from exterior fluid samples");
  assert.match(uniformSolverSource, /gridKind: "octree",[\s\S]*?volumeControl: true/, "octree diagnostics must report that GPU volume control is active");
  assert.doesNotMatch(quadtreeSurfaceShader, /params\.control\.y > 1\.5/);
  assert.match(octreeSource, /undefined, false, false, true, true/, "octree enables GPU phi-volume correction and topology-preserving transport");
  const encode = WebGPUQuadtreeSurfaceState.prototype.encode.toString();
  assert.match(encode, /monotoneLevelSetTransport[\s\S]*advectLevelSet[\s\S]*advectPredict/, "octree can select monotone phi transport without changing other surface users");
});

test("octree shared fine-grid dynamics use the resident level set as their sole liquid authority", () => {
  const projectionConstruction = uniformSolverSource.indexOf("else if (options.octree)");
  const authorityBinding = uniformSolverSource.indexOf("const surfaceAuthority = this.adaptiveProjection?.levelSetTexture ?? this.volumeA");
  assert.ok(projectionConstruction >= 0 && authorityBinding > projectionConstruction, "surface bind groups must be created after the octree level set exists");
  assert.match(uniformSolverSource, /surfaceIn: GPUTexture = this\.adaptiveProjection\?\.levelSetTexture \?\? volumeIn/);
  assert.doesNotMatch(uniformSolverSource, /surfaceIn: GPUTexture = this\.quadtreeProjection\?\.levelSetTexture/);
  assert.match(uniformSolverSource, /this\.transportConservativeVolume = !this\.octreeProjection/);
  assert.match(uniformSolverSource, /if \(this\.transportConservativeVolume\) \{\s+prep\.setPipeline\(this\.buildFluxScalesPipeline\)/);
  assert.match(uniformSolverSource, /this\.densitySharpening && this\.transportConservativeVolume/);
  assert.match(uniformSolverSource, /this\.adaptiveProjection \? 1 : 0, sigma/, "octree must interpret the shared authority texture as signed distance");
  assert.match(legacyUniformComputeShader, /fn transportConservativeVolume\(\) -> bool/);
  assert.match(legacyUniformComputeShader, /if\(transportConservativeVolume\(\)\)\{advected=advectedVolume\(id,dt\);\}/);
});

test("octree rebuild and solve stay resident on the GPU", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString();
  const solve = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(rebuild, /beginComputePass/);
  assert.match(solve, /clearBuffer/);
  assert.doesNotMatch(`${rebuild}\n${solve}`, /mapAsync|getMappedRange/);
  // Device-local copies publish row-parallel and one-workgroup-per-leaf
  // indirect args; neither is a readback or dense-field copy.
  const copies = (solve.match(/copyBufferToBuffer\([^)]*\)/g) ?? []).map((copy) => copy.replace(/\s+/g, ""));
  assert.deepEqual(copies, [
    "copyBufferToBuffer(this.compaction,8,this.solveDispatch,0,24)",
  ]);
  assert.match(octreeProjectionShader, /var<storage, read_write> owners/);
  assert.match(octreeProjectionShader, /pressureIn: array<f32>/);
  assert.match(uniformSolverSource, /if \(substep > 0 && this\.octreeProjection\) this\.octreeProjection\.encodeInlineRebuild\(encoder\)/,
    "CFL subdivision must rebuild from the level set transported by the preceding substep");
});

test("octree telemetry samples the live compacted pressure-row count", () => {
  const diagnostics = WebGPUOctreeProjection.prototype.readSolveDiagnostics.toString();
  assert.match(diagnostics, /copyBufferToBuffer\(this\.compaction,\s*0,\s*readback,\s*0,\s*8\)/);
  assert.match(diagnostics, /this\.info\.liquidDofCount\s*=\s*liquidRows/);
  assert.match(diagnostics, /this\.info\.pressureSampleCount\s*=\s*liquidRows/);
});

test("octree compacted leaf solve scans, assembles once, and iterates over rows only", () => {
  // The persistent frontier uses atomics only while evolving active topology
  // tiles. Pressure-row rank/scatter remains a prefix scan over that frontier.
  assert.match(octreeProjectionShader, /fn beginFrontier/);
  assert.match(octreeProjectionShader, /fn filterFrontier/);
  assert.match(octreeProjectionShader, /fn appendFrontierActive/);
  assert.match(octreeProjectionShader, /fn appendFrontierRetired/);
  assert.match(octreeProjectionShader, /fn finalizeFrontier/);
  assert.match(octreeProjectionShader, /fn planLeaves/);
  assert.match(octreeProjectionShader, /fn scanLeafBlocks/);
  assert.match(octreeProjectionShader, /fn emitLeaves/);
  assert.match(octreeProjectionShader, /var running = scanPairs\[lid\] - sum;/);
  const emit = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn emitLeaves"), octreeProjectionShader.indexOf("fn compactRowIndex"));
  assert.match(emit, /frontierCell\(current, slot\)/);
  assert.doesNotMatch(emit, /liquidOwner|ownerPhi|textureLoad/,
    "row emission must not reclassify the finest lattice or resample phi");
  // Assembly caches diagonal, RHS flux, and a merged neighbor table, so the
  // per-iteration kernels never touch the velocity texture or owner map.
  const assemble = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn assembleSystem"), octreeProjectionShader.indexOf("fn iterateLeaves"));
  assert.match(assemble, /neighborCoefficients\[j\] \+= coefficient/);
  assert.match(assemble, /flux \+= f32\(side\) \* area \* /);
  const iterate = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn iterateLeaves"), octreeProjectionShader.indexOf("fn leafPressureLoad"));
  assert.doesNotMatch(iterate, /velocityAt|ownerAt|textureLoad/);
  assert.match(iterate, /entry\.coefficient \* pressureIn\[entry\.cell\]/);
});

test("octree assembles coarse leaf faces cooperatively with deterministic quadrants", () => {
  const start = octreeProjectionShader.indexOf("fn assembleCoarseSystem");
  const end = octreeProjectionShader.indexOf("fn iterateLeaves", start);
  const coarse = octreeProjectionShader.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(octreeProjectionShader, /if \(owner\.size >= 8u\)[\s\S]*coarseTasks = tiles \* tiles \* tiles/);
  assert.match(coarse, /@builtin\(local_invocation_index\) lid/);
  assert.match(coarse, /workgroupUniformLoad\(&coarseTaskEligible\) == 0u/);
  assert.match(coarse, /sample = lid; sample < faceSamples; sample \+= 64u/);
  assert.match(coarse, /quadrant \* 64u \+ lid/);
  assert.match(coarse, /header\.entryStart \+ face \* 4u \+ quadrant/);
  assert.match(coarse, /header\.entryCount = 24u/);
  assert.doesNotMatch(coarse, /atomicAdd/,
    "coarse face coefficients must reduce deterministically rather than accumulate atomically");
  const encode = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /this\.assemblePipeline[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,0\)[\s\S]*this\.assembleCoarsePipeline[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,12\)/);
});

test("octree Chebyshev solve removes three quarters of global iteration boundaries", () => {
  const start = octreeProjectionShader.indexOf("fn iterateChebyshev");
  const end = octreeProjectionShader.indexOf("fn leafPressureLoad", start);
  const chebyshev = octreeProjectionShader.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(chebyshev, /params\.solve\.y/);
  assert.match(chebyshev, /params\.solve\.z/);
  assert.match(chebyshev, /header\.pad0 = bitcast<u32>\(search\)/);
  assert.match(chebyshev, /header\.pad1 = bitcast<u32>\(rho\)/);
  assert.doesNotMatch(chebyshev, /workgroupBarrier|storageBarrier|ownerAt|textureLoad/, "each pass stays row-parallel and reduction-free");
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode, /Math\.ceil\(this\.iterations\s*\/\s*4\)/);
  assert.match(encode, /this\.iterateChebyshevPipeline/);
  assert.doesNotMatch(encode, /this\.leafSolver === "chebyshev" && !this\.couplingHasDynamicBodies/,
    "dynamic bodies must not force the accelerated solve back onto the dispatch ladder");
});

test("octree megakernel keeps barriers in uniform control flow and folds parity back", () => {
  const solve = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn solveLeaves"));
  assert.match(solve, /workgroupUniformLoad\(&convergedShared\)/);
  assert.match(solve, /workgroupUniformLoad\(&rowCountShared\)/);
  assert.match(solve, /storageBarrier\(\)/);
  assert.match(solve, /pressureIn\[cell\] = pressureOut\[cell\]/);
  assert.match(solve, /tolerance2 \* normB/);
});

test("octree pressure traverses coarse-fine faces by finest subfaces", () => {
  const face = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn accumulateFace"), octreeProjectionShader.indexOf("@compute @workgroup_size(4,4,4)\nfn jacobi"));
  assert.match(face, /for \(var b = 0u; b < size/);
  assert.match(face, /for \(var a = 0u; a < size/);
  assert.match(face, /ownerAt\(outside\)/);
  assert.match(face, /let distance = (?:pressureDistance\(ownerAt\(vec3i\(origin\)\), neighbor, axis\)|0\.5 \* f32\(size \+ neighbor\.size\) \* h)/);
  if (octreeProjectionShader.includes("fn pressureDistance")) {
    assert.match(octreeProjectionShader, /let full = 0\.5 \* f32\(a\.size \+ b\.size\) \* cellWidth\(axis\)/);
  }
  assert.match(face, /area \/ max\(distance/);
});

test("octree uses the level-set crossing and prolongates pressure inside coarse leaves", () => {
  assert.match(octreeProjectionShader, /fn ownerPhi\(owner: Owner\)/);
  assert.match(octreeProjectionShader, /0\.5 \* f32\(owner\.size - 1u\)/,
    "even-sized leaf pressure samples must use the geometric centre rather than an upper fine cell");
  assert.match(octreeProjectionShader, /fn pressureDistance\(a: Owner, b: Owner, axis: u32\)/);
  assert.match(octreeProjectionShader, /abs\(liquidPhi\) \/ max\(abs\(liquidPhi\) \+ abs\(airPhi\)/,
    "the free-surface pressure boundary must lie at phi=0");
  assert.match(octreeProjectionShader, /fn reconstructGradients/);
  assert.match(octreeProjectionShader, /pressureOut\[index\(workgroupOrigin \+ vec3u\(1u,0u,0u\)\)\] = reconstructedAxisGradient/);
  const project = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn projectedComponent"), octreeProjectionShader.indexOf("fn extrapolate"));
  assert.match(project, /left\.packedOrigin != right\.packedOrigin[\s\S]*pressureDistance\(left, right, axis\)/,
    "leaf-boundary faces must retain the exact assembled variational gradient");
  assert.match(project, /else if \(leftWet && rightWet\) \{\s+fluid -= reconstructedGradient\(left, axis\)/,
    "dense faces inside a coarse leaf must no longer be invisible to pressure");
  assert.match(octreeDiagnosticShader, /mappedPressure = pressure\[ownerIndex\] \+ dot\(leafGradient\(owner\), offset\)/,
    "the pressure overlay must expose the same affine field used by projection");
  assert.match(octreeDiagnosticShader, /bitcast<u32>\(pressureUpdate\)/,
    "the pressure-ownership texture must expose the applied velocity update without a readback");
});

test("octree reconstructs coarse affine gradients with a cooperative face reduction", () => {
  const reconstruct = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("var<workgroup> gradientPartials"),
    octreeProjectionShader.indexOf("fn projectedComponent"),
  );
  assert.match(reconstruct, /@builtin\(local_invocation_index\) lid: u32/);
  assert.match(reconstruct, /let sampleCount = 6u \* coarseOwner\.size \* coarseOwner\.size/);
  assert.match(reconstruct, /for \(var sample = lid; sample < sampleCount; sample \+= 64u\)/,
    "a 32^3 leaf must distribute its 6,144 boundary visits across the 64-lane workgroup");
  assert.match(reconstruct, /coarseGradientContribution\(coarseOwner, sample\)/);
  assert.match(reconstruct, /pressureDistance\(owner, neighbor, axis\)/,
    "coarse/fine and ghost-fluid pressure distances must survive reconstruction");
  assert.match(reconstruct, /let open = 1\.0 - clamp\(solid\.fraction/,
    "the affine fit must retain the same VOS open-face weighting as projection");
  assert.match(reconstruct, /workgroupUniformLoad\(&gradientOwnerEligible\)/,
    "the workgroup reduction branch must be uniform before reaching barriers");
  assert.match(reconstruct, /for \(var stride = 32u; stride > 0u; stride >>= 1u\)/);
  assert.match(reconstruct, /coarseOwner\.size >= 4u/);
  assert.match(reconstruct, /fn reconstructSmallGradients[\s\S]*header\.size >= 8u \|\| header\.size <= 1u[\s\S]*reconstructedAxisGradient\(owner, 0u\)/,
    "size-2 and size-4 leaves should retain the row-parallel scalar path");
});

test("octree projection is driven by the persistent leaf frontier", () => {
  const small = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn projectSmallLeaves"), octreeProjectionShader.indexOf("fn projectLeaves"));
  const project = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn projectLeaves"), octreeProjectionShader.indexOf("fn extrapolateSeed"));
  assert.match(octreeProjectionShader, /@compute @workgroup_size\(256\)\s+fn projectSmallLeaves/);
  assert.match(small, /let row = compactRowIndex\(gid\)/);
  assert.match(small, /header\.size >= 8u/);
  assert.match(octreeProjectionShader, /@compute @workgroup_size\(256\)\s+fn projectLeaves/);
  assert.match(project, /let row = coarseTaskRow\(task\)/);
  assert.match(project, /let tile = coarseTaskTile\(task\)/);
  assert.match(project, /tileCoord \* 8u/);
  assert.match(project, /sample = lid; sample < samples; sample \+= 256u/);
  assert.match(octreeProjectionShader, /projectionController\(id\) != owner\.packedOrigin/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode, /this\.projectSmallLeavesPipeline/);
  assert.match(encode, /this\.projectLeavesPipeline/);
  assert.match(encode, /this\.projectSmallLeavesPipeline\)[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,\s*0\)/);
  assert.match(encode, /this\.projectLeavesPipeline\)[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,\s*12\)/);
  assert.match(octreeProjectionShader, /fn extrapolateSeed/,
    "the first dense compatibility sweep must ignore stale air validity");
});

test("octree materializes adaptive overlay fields without a readback", () => {
  assert.match(octreeDiagnosticShader, /texture_storage_3d<rg32uint, write>/);
  assert.match(octreeDiagnosticShader, /origin\.x \| \(origin\.z << 10u\) \| \(owner\.size << 20u\)/);
  assert.match(octreeDiagnosticShader, /origin\.y \| \(\(origin\.y \+ owner\.size\) << 10u\)/);
  assert.match(octreeDiagnosticShader, /textureStore\(pressureSamplesOut/);
  assert.match(octreeDiagnosticShader, /textureStore\(pressureOut/);
  assert.match(octreeDiagnosticShader, /textureStore\(divergenceOut/);
  assert.doesNotMatch(octreeDiagnosticShader, /mapAsync|getMappedRange/);
  assert.match(rendererSource, /gridKind === "quadtree-tall-cell" \|\| gpuInfo\?\.gridKind === "octree" \? 1 : 0/);
});

test("octree dense diagnostic textures are allocated only on overlay demand", () => {
  const constructor = octreeSource.slice(octreeSource.indexOf("constructor("), octreeSource.indexOf("private descriptor("));
  assert.doesNotMatch(constructor, /createTexture\(\{ label: "Octree overlay/);
  assert.match(octreeSource, /ensureDiagnosticTextures\(\): boolean/);
  assert.match(uniformSolverSource, /ensureGridDiagnosticTextures\(\)/);
  assert.match(rendererSource, /gridOverlay\?\.axis !== "off"[\s\S]{0,100}ensureGridDiagnosticTextures/);
});

test("octree compact solve dispatches cover rows with two-dimensional tiles", () => {
  assert.match(octreeProjectionShader, /compaction\[2\] = x; compaction\[3\] = y/);
  assert.match(octreeProjectionShader, /fn compactRowIndex\(gid: vec3u\)/);
  assert.match(octreeProjectionShader, /gid\.x \+ gid\.y \* compaction\[2\] \* 256u/);
  assert.doesNotMatch(octreeProjectionShader, /compaction\[2\] = min\(/);
});

test("octree materializes its live owner map on the reset frame", () => {
  assert.match(uniformSolverSource, /encodeInlineRebuild\(initialSparseScene\);\s*this\.octreeProjection\.encodeOverlayMaterialization\(initialSparseScene\);\s*this\.octreeProjection\.encodeSparseBrickWorld/);
  assert.match(octreeSource, /reset-time grid[\s\S]{0,120}zero-initialized topology storage as finest 1\^3/);
});
