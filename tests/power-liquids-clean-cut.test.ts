import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

const production = [
  source("../lib/webgpu-octree.ts"),
  source("../lib/webgpu-octree-sparse-bricks.ts"),
  source("../lib/webgpu-device-limits.ts"),
  source("../lib/webgpu-octree-fine-levelset-redistance.ts"),
  source("../lib/webgpu-octree-fine-levelset-topology.ts"),
  source("../lib/webgpu-octree-mgpcg.ts"),
  source("../lib/webgpu-octree-spgrid-vcycle.ts"),
  source("../lib/webgpu-pass-broker.ts"),
  source("../lib/webgpu-uniform-eulerian.ts"),
  source("../package.json"),
].join("\n");
const productionDocs = [
  source("../docs/UNIFIED_OCTREE_SIMULATION.md"),
  source("../docs/POWER_LIQUIDS_REPRESENTATION_V2_PLAN.md"),
].join("\n");

const closestPointDependents = [
  source("../lib/methods/types.ts"),
  source("../lib/webgpu-eulerian.ts"),
  source("../lib/webgpu-uniform-eulerian.ts"),
  source("../lib/webgpu-octree.ts"),
  source("../lib/webgpu-octree-technique-overlay.ts"),
  source("../lib/paper-pipeline-diagnostics.ts"),
  source("../tools/run-webgpu-smoke.ts"),
].join("\n");

test("power liquids has no runtime switch back to retired authority paths", () => {
  for (const retiredSwitch of [
    "FLUID_PASS_BROKER",
    "FLUID_OCTREE_INCREMENTAL",
    "FLUID_OCTREE_CHANGE_DRIVEN",
    "FLUID_OCTREE_COARSE_WORKLIST",
    "FLUID_FINE_REDISTANCE",
    "FLUID_SECTION5_ADJACENCY",
    "FLUID_OCTREE_DIRECT_PAGED_PHI",
    "FLUID_OCTREE_SURFACE_PAGES",
    "FLUID_OCTREE_OWNER_PAGES",
    "FLUID_OCTREE_COMPACT_BRICK_ATLAS",
    "FLUID_OCTREE_SPARSE_WORLD",
    "FLUID_OCTREE_FACE_TRANSFER",
  ]) assert.doesNotMatch(production, new RegExp(retiredSwitch), retiredSwitch);
  assert.doesNotMatch(production, /powerDiagramProjection|leafSolver|faceVelocityTransport/);
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"),
    /private (?:jacobi|iterate|iterateChebyshev|solve|pressureOverflow|reconstructSmallGradients|reconstructGradients|project|projectSmallLeaves|projectLeaves|extrapolate)Pipeline/);
  assert.doesNotMatch(source("../lib/methods/octree.ts"),
    /brickSparseVelocityAdvection|brickSparseTransportPreparation|brickSparseOccupancyFluxPreparation|hydrostaticSplit|secondaryParticles|surfaceDetailStrength|brickAtlas|brickPreActivation|brickSparseSurface|pressureWarmStart|pressureIterations|fencedInitialSparseAuthority|FLUID_SAFE_BRINGUP/);
  assert.equal(existsSync(new URL("../lib/webgpu-brick-atlas.ts", import.meta.url)), false,
    "the retired dense brick-atlas mirror implementation must stay deleted");
  assert.doesNotMatch(production, /brickAtlas|WebGPUFluidBrickAtlas|bulkResidencyOnly/,
    "the production sparse world retains only explicit compact bulk residency");
  assert.doesNotMatch(source("../lib/webgpu-device-limits.ts"), /float32-filterable|optionalFluidDeviceFeatures/,
    "device creation no longer requests a feature used only by the deleted atlas sampler");
  assert.doesNotMatch(source("../lib/methods/octree.ts") + source("../tools/run-webgpu-smoke.ts"),
    /Coarse octree only|coarseOnlyOctreeRequested|\["off",\s*"4",\s*"8"\]/,
    "the authoritative fine-band cutover has no coarse-only runtime switch or QA backing path");
  assert.doesNotMatch(source("../tools/run-webgpu-smoke.ts"),
    /octreeGlobalFineFactorOverride\s*===\s*"off"/,
    "the removed coarse-only value must not survive as dead diagnostic backing code");
  assert.doesNotMatch(source("../package.json"),
    /test:webgpu:octree-face-transport/,
    "the deleted Cartesian face-transport lane must not retain a named smoke command");
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"), /legacyCapacity/,
    "the physical narrow-band capacity estimate must not be inflated by the retired page heuristic");
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"), /coarseOnlyBroker|Coarse-only paper mode/,
    "the deleted coarse-only switch has no simulation branch backing it");
  assert.doesNotMatch(source("../lib/gpu-t0-presentation.ts"),
    /globalFineRequired|coarse-only mode|coarse-octree raster/,
    "the paused t=0 gate must not retain a renderer-side coarse-only escape hatch");
  assert.doesNotMatch(source("../lib/webgpu-renderer.ts"),
    /initialRasterGlobalFineRequired|globalFineRequired/,
    "the renderer must require the mandatory global-fine attachment directly");
  assert.doesNotMatch(source("../lib/webgpu-uniform-eulerian.ts"),
    /secondaryParticlesEnabled\?:|octreeHostVelocityAllocation|sparseSurfaceBand|adaptive face velocity authority|fencedInitialSparseAuthority|initialSparseAuthorityFencesEnabled|publishInitialSparseSceneBatched|combined Section 4\/5 publication/);
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"), /encodeInitialSparseAuthority\(encoder/,
    "the combined t=0 command-buffer path must stay deleted");
  assert.doesNotMatch(source("../lib/gpu-startup.ts") + source("../tools/run-webgpu-bringup-stage-worker.ts"),
    /fencedSparseAuthorityBringupEnabled|FLUID_SAFE_BRINGUP/,
    "phase-fenced t=0 publication is unconditional, not a browser or environment mode");
  assert.doesNotMatch(source("../tools/run-webgpu-smoke.ts"),
    /method\.id === "octree"[^\n]*pressureIterations|pressureIterations[^\n]*method\.id === "octree"/,
    "the shared smoke pressure override must not recreate an octree iteration ABI");
  assert.doesNotMatch(source("../lib/octree-host-allocation.ts"),
    /OctreeHostVelocityAllocationPlan|planOctreeHostVelocityAllocation|@deprecated/,
    "retired host-allocation aliases must not preserve the old public surface");
  assert.doesNotMatch(source("../lib/methods/types.ts"), /sparseSurfaceBand/);
  assert.doesNotMatch(productionDocs,
    /brickSparseVelocityAdvection|brickSparseTransportPreparation|brickSparseOccupancyFluxPreparation|brickSparseSurface|hydrostaticSplit/,
    "production documentation must not preserve deleted sparse compatibility switches");
  assert.doesNotMatch(source("../lib/webgpu-water-pipeline.ts"),
    /setSparseSurface|SparseSurfaceBandGPUSource|extractSparsePipeline|extractHybridCoarsePipeline|polygoniseSparsePipeline/);
  assert.doesNotMatch(source("../lib/webgpu-grid-overlay.ts"),
    /setSparseSurface|SparseSurfaceBandGPUSource|SparseSurfaceParams|sparseSurfaceAvailable/);
});

test("UI contracts and validation profiles expose only the native power authority", () => {
  const methodContract = source("../lib/methods/types.ts");
  const validationProfiles = source("../lib/scenes.ts");
  const visualPanel = source("../components/VisualPanel.tsx");
  const diagnosticsPanel = source("../components/DiagnosticsPanel.tsx");

  assert.doesNotMatch(methodContract,
    /SparseSurfaceBandGPUSource|OctreeFaceMirrorSource|OctreeFaceVelocitySource|OctreeSurfacePageSource|adaptiveFaceMirrorSource|adaptiveFaceVelocitySource|adaptiveSurfacePageSource|readGlobalFineDisconnectedFaceFailure/);
  assert.doesNotMatch(validationProfiles,
    /faceVelocityTransport|powerDiagramProjection|leafSolver/,
    "validation presets must not preserve values for deleted method switches");
  assert.doesNotMatch(visualPanel,
    /gridOverlayMode === "surface"|gridOverlayMode === "faces"|SPARSE SURFACE BAND|SPARSE VELOCITY FACES|Face march/);
  assert.doesNotMatch(diagnosticsPanel,
    /powerAuthorityRequested|powerDiagramFallbackReason|AXIS ROLLBACK|POWER UNAVAILABLE|adaptiveSurfacePageCapacity|sparseSurfacePageCapacity/);
});

test("retired FMM and staged MGPCG selectors cannot be reconstructed", () => {
  assert.doesNotMatch(production, /FineLevelSetRedistanceMethod|resolveFineLevelSetRedistanceMethod/);
  assert.doesNotMatch(source("../lib/webgpu-octree-fine-levelset-redistance.ts"),
    /\bFMM\b|\bfmm\b|bucket(?:ed)? fast march|activated:atomic|requests:atomic|bucket:atomic/i);
  assert.doesNotMatch(source("../lib/webgpu-octree-mgpcg.ts"),
    /staged-small-domain|staged-mgpcg|stagedExecution|restartPCG|reduceDirectionProduct|updatePressureResidual|reduceNextState|fn updateDirection/);
  assert.doesNotMatch(production,
    /powerPcgIterationCap|OCTREE_SECTION43_(?:DEFAULT|MAXIMUM)_PCG_ITERATIONS|normalizeOctreeSection43IterationCap|octreeSection43RecordedIterationCap/,
    "the retired 128-iteration configuration path cannot coexist with immutable 12/16-step lanes");
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"),
    /options\.pressureIterations|private readonly iterations|compactAuthority|legacyHashCapacity|legacyDenseAndHashBytes/,
    "the native solver must not retain dead host-authored iteration or dense-compaction compatibility bindings");
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"),
    /authoritativePowerTopology|surfaceDrivenRefinement|crossesSurface|powerDiagramAuthority/,
    "power topology must not retain the old surface-refinement authority selector or its backing branch");
  assert.doesNotMatch(source("../lib/webgpu-octree.ts"),
    /compiled\.find|compiled\[index\]\s*\?\?=|fallback slot/,
    "an unreachable pipeline must stay absent instead of being aliased to an unrelated compiled kernel");
  assert.doesNotMatch(
    source("../lib/webgpu-octree-mgpcg.ts") + source("../lib/webgpu-octree-spgrid-vcycle.ts"),
    /\bmaximumIterations\b|encodedCorrectionPassCount/,
    "the exact persistent shader must not retain a host-authored iteration binding");
  assert.equal(existsSync(new URL("../lib/webgpu-octree-surface-pages.ts", import.meta.url)), false,
    "the retired page-local PDE redistance owner must stay deleted");
  assert.doesNotMatch(source("../lib/octree-surface-allocation.ts"),
    /presentationOnly|legacyAuxiliaryBytes|denseBaselineBytes|savedBytes/,
    "the native surface planner must not expose allocation modes or metrics for the deleted dense pipeline");
});

test("the retired CPU face marcher and its graph-state budget are deleted", () => {
  assert.equal(existsSync(new URL("../lib/octree-face-fast-march.ts", import.meta.url)), false);
  assert.equal(existsSync(new URL("../lib/webgpu-sparse-surface-band.ts", import.meta.url)), false);
  const band = source("../lib/octree-face-band.ts");
  assert.doesNotMatch(band,
    /fastMarchOctreeFaceVelocity|OctreeFaceMarch(?:Face|Result)|parents|graphDistance|marchBytes/);
});

test("closest-point cutover has no face-march state, split controls, or disconnected fallback API", () => {
  assert.doesNotMatch(closestPointDependents,
    /faceBandMarchControl|faceBandSearchControl|readGlobalFineDisconnectedFaceFailure/);
  assert.doesNotMatch(closestPointDependents,
    /globalFineFaceBandMarch(?:Heap|Pops|Trials|Chunks|Cap|Unresolved|Disconnected)|ConnectivityFallbacks/);
  assert.doesNotMatch(closestPointDependents,
    /section5-face-band-fast-march|encodeGlobalFineFaceBandPhase\(encoder,\s*"fast-march"\)/);
  const overlay = source("../lib/webgpu-octree-technique-overlay.ts");
  const section5Start = overlay.indexOf("export const octreeTechniqueSection5FaceBandShader");
  const section5 = overlay.slice(section5Start,
    overlay.indexOf("export const octreeTechniqueLifecycleShader", section5Start));
  assert.doesNotMatch(section5, /struct State|var<storage,read> states|status==|\bTRIAL\b/);
  assert.doesNotMatch(
    source("../lib/webgpu-octree-face-closest-point.ts"),
    /surroundingOwnerDelaunay|SurroundingOwnerVectorMeasurement|fullRowFallback|surroundingFallback/,
    "the target resolver has one exact catalog-anchor authority and no per-query Delaunay repair",
  );
});

test("native power velocity replaces the deleted Cartesian face bridge", () => {
  const projection = source("../lib/webgpu-octree.ts");
  assert.doesNotMatch(projection, /FaceMirror|FaceTransport|encodePowerToAxis/);
  assert.match(source("../lib/webgpu-octree-fine-seed-adapter.ts"),
    /OctreePowerVelocitySource/);
  assert.doesNotMatch(source("../lib/webgpu-octree-fine-seed-adapter.ts"),
    /FaceRecord|faceControl|incidence/);
  assert.doesNotMatch(source("../lib/webgpu-octree-fine-levelset-topology.ts"), /refreshFromCoarse/);
});

test("physical power volumes launch only the GPU-published live row prefix", () => {
  const projection = source("../lib/webgpu-octree.ts");
  const assembly = projection.slice(
    projection.indexOf("private encodeNativePowerAssembly"),
    projection.indexOf("private encodeNativePowerProjection"),
  );
  assert.match(assembly,
    /dispatchWorkgroupsIndirect\(\s*faces\.source\.liveFaceDispatch,\s*OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,\s*\)/,
    "physical volume publication must consume the exact live-row indirect record");
  assert.doesNotMatch(assembly,
    /dispatchWorkgroups\(Math\.ceil\(this\.pressureCapacity\.rowCapacity\s*\/\s*64\)\)/,
    "the deleted capacity-sized physical-volume launch must not return");
  const volumeShader = projection.slice(projection.indexOf("export const octreePowerVolumeShader"));
  assert.doesNotMatch(volumeShader, /rowCapacity/,
    "the physical-volume kernel no longer needs a host-capacity scheduling uniform");
  assert.match(volumeShader, /rowCountSource\[0\]/,
    "the compact GPU row publication remains the per-invocation bounds authority");
});

test("Section 5 regular-face completion is mandatory and has no repair switch or cold duplicate", () => {
  const projection = source("../lib/webgpu-octree.ts");
  const direct = source("../lib/webgpu-octree-power-face-advection.ts");
  const completion = source("../lib/webgpu-octree-face-closest-point.ts");
  assert.doesNotMatch(direct + projection,
    /deferInterpolationFailures|deferFailures|repairFromRetainedFaceBand/,
    "recurring regular-face completion cannot be disabled or replaced by direct-only publication");
  assert.doesNotMatch(completion,
    /encodeRepairPowerFaceAdvection|publishColdPowerFaceAdvection|coldPowerAdvectionPipeline/,
    "the mandatory recurring completion has no fallback name or duplicate cold-start producer");
  assert.match(projection,
    /powerGeneration\s*>\s*1[\s\S]*encodeCompletePowerFaceAdvectionFromRegularBand/,
    "every recurring power generation must execute the paper's regular-face completion");
  assert.match(completion,
    /dispatchWorkgroupsIndirect\(input\.faces\.liveFaceDispatch,\s*0\)/,
    "completion dispatches only the device-published live generalized-face prefix");
  assert.doesNotMatch(completion,
    /copyBufferToBuffer\(input\.advectionControl|Math\.ceil\(this\.plan\.powerFaceCapacity\/64\)/,
    "completion has neither a copied-count pass break nor a capacity-sized launch");
});

test("production velocity and coarse-phi owners expose no retired duplicate GPU lanes", () => {
  const velocity = source("../lib/webgpu-octree-power-velocity.ts");
  assert.doesNotMatch(velocity,
    /WebGPUOctreePowerVelocitySampler|octreePowerVelocitySampleShader|planOctreePowerVelocitySamples|OctreePowerVelocityEncodeOptions|octreePowerVelocityPrepareShader|\bencode\(broker:/);
  assert.doesNotMatch(source("../lib/webgpu-octree-power-velocity-prepass.ts"),
    /buildPowerTrajectoryQueriesWGSL/);
  const coarse = source("../lib/webgpu-octree-coarse-levelset.ts");
  assert.doesNotMatch(coarse,
    /encodeFineCorrection|correctCoarsePhi|finalizeCoarsePhi|CoarsePhiGPUControl|rowStatus/);
  assert.match(coarse, /octreeCoarsePhiBootstrapShader/);
  assert.doesNotMatch(source("../lib/octree-power-operator.ts"), /usedFallback|fallback:\s*PowerVec3/,
    "the CPU oracle must mirror fail-closed velocity reconstruction rather than accept a substitute vector");
});

test("power descriptors have one sorted owner-page authority", () => {
  const descriptor = source("../lib/webgpu-octree-power-descriptor.ts");
  const octree = source("../lib/webgpu-octree.ts");
  assert.doesNotMatch(descriptor,
    /OctreePowerOwnerMode|ownerMode|decodeDenseOctreePowerOwner|packDenseOctreePowerOwner|decodeDenseOwner/);
  assert.match(descriptor, /fn ownerAt\(cell:vec3u\)->Owner\{\s*return pagedOwner\(cell\);\s*\}/);
  assert.doesNotMatch(octree,
    /ownerMode:|"paged"\s*:\s*"dense"|ownerPagesEnabled|decodeDenseOwner|encodeDenseOwner|planOctreeOwnerAllocation/);
  assert.match(octree, /private readonly ownerPages: WebGPUOctreeSimulationOwnerPages;/);
  assert.doesNotMatch(octree, /ownerPages\?:|ownerPages!|\(this\.ownerPages \?/);
});

test("power row producers require one exact GPU count and row-delta authority", () => {
  for (const path of [
    "../lib/webgpu-octree-power-descriptor.ts",
    "../lib/webgpu-octree-power-topology.ts",
    "../lib/webgpu-octree-power-faces.ts",
  ]) {
    const producer = source(path);
    assert.doesNotMatch(producer,
      /hostRowCount|number \| GPUBuffer|rowDelta\?:|deltaEnabled|typeof options\.rowCount/,
      `${path} must not retain host-count or optional-delta compatibility lanes`);
    assert.doesNotMatch(producer,
      /rowDelta\?\.rows|\?\? this\.hostRowCount|if \(!options\.rowDelta\)|if \(options\.rowDelta\)/,
      `${path} must bind and dispatch only the exact row-delta publication`);
    assert.match(producer, /assertOctreePowerRowDeltaLayout\(/);
    assert.match(producer, /ROW_DELTA_VALID/);
  }
  const descriptor = source("../lib/webgpu-octree-power-descriptor.ts");
  assert.match(descriptor, /readonly rowCountBuffer: GPUBuffer/);
  assert.match(descriptor, /readonly rowDelta: OctreePowerRowDeltaSource/);
  assert.doesNotMatch(descriptor, /readonly rowCount\?: number/);
  assert.match(descriptor, /rowDelta\[base\+7u\]==generation\(\)/,
    "descriptor publication must consume the exact frontier generation");
  const topology = source("../lib/webgpu-octree-power-topology.ts");
  assert.match(topology,
    /encode\(broker: PassBroker, descriptors: GPUBuffer, rowCount: GPUBuffer,[\s\S]*rowDelta: OctreePowerRowDeltaSource\)/);
  const faces = source("../lib/webgpu-octree-power-faces.ts");
  assert.match(faces, /readonly rowCount: GPUBuffer/);
  assert.match(faces, /readonly rowDelta: OctreePowerRowDeltaSource/);
  assert.doesNotMatch(faces, /runAffectedRows[\s\S]*if \(!options\.rowDelta\)/);
  assert.match(faces, /rowDelta\[base\+7u\]==params\.capacitiesGeneration\.w/,
    "face publication must consume the exact frontier generation");
});

test("descriptor and metric authority uses parallel exact-delta staging and compact commits", () => {
  const descriptor = source("../lib/webgpu-octree-power-descriptor.ts");
  const topology = source("../lib/webgpu-octree-power-topology.ts");
  for (const producer of [descriptor, topology]) {
    assert.doesNotMatch(producer,
      /rowStatusBytes|summaryBytes|carryPipeline|carryPower(?:Descriptors|Topology)|summarizePower(?:Descriptors|Topology)/,
      "the retired capacity carry backing state must remain deleted");
    assert.doesNotMatch(producer,
      /dispatchWorkgroups\(Math\.ceil\(this\.plan\.rowCapacity|dispatchDirect\(pass,\s*this\.plan\.rowCapacity/,
      "no host pass may dispatch descriptor or metric work from capacity");
    assert.doesNotMatch(producer,
      /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
      "exact row identity relocation must remain deterministic and synchronization-free");
    assert.match(producer, /oldPlusOne-1u/);
    assert.match(producer, /old!=row/);
    assert.match(producer, /commitDispatch=dispatchFor\(published\)/,
      "the commit dispatch is sourced only from the compact changed-row publication");
    assert.doesNotMatch(producer, /for\(var row=0u;row<requested/,
      "row carry and validation must remain block-parallel");
  }
  assert.match(descriptor, /stagePowerDescriptorDelta/);
  assert.match(descriptor, /prefixPowerDescriptorDelta/);
  assert.match(descriptor, /scatterPowerDescriptorDelta/);
  assert.match(topology, /stagePowerTopologyDelta/);
  assert.match(topology, /prefixPowerTopologyDelta/);
  assert.match(topology, /scatterPowerTopologyDelta/);
});

test("power publication transaction buffers match WGSL alignment exactly", () => {
  const descriptor = source("../lib/webgpu-octree-power-descriptor.ts");
  const topology = source("../lib/webgpu-octree-power-topology.ts");
  const faces = source("../lib/webgpu-octree-power-faces.ts");
  for (const producer of [descriptor, topology]) {
    assert.match(producer, /CONTROL_ARENA_BYTES = 112/);
    assert.match(producer, /size: OCTREE_POWER_(?:DESCRIPTOR|TOPOLOGY)_CONTROL_ARENA_BYTES/);
  }
  assert.match(descriptor,
    /copyBufferToBuffer\(this\.control,\s*80,\s*this\.workDispatch,\s*0,/,
    "the descriptor commit vec3u starts at the next 16-byte WGSL alignment boundary");
  assert.doesNotMatch(topology,
    /copyBufferToBuffer\(this\.control,\s*80,\s*this\.workDispatch,\s*0,/,
    "topology scatter owns its changed rows and needs no second indirect dispatch");
  assert.match(faces, /OCTREE_POWER_FACE_CONTROL_BYTES = 64/);
  assert.match(faces, /OCTREE_POWER_FACE_PARAMETER_BYTES = 112/);
  assert.match(faces, /OCTREE_POWER_FACE_BOUNDARY_QUERY_BYTES = 32/);
  assert.match(faces, /OCTREE_POWER_FACE_QUADRATURE_BYTES = 16 \+ OCTREE_POWER_FACE_QUADRATURE_SAMPLES \* 4/);
  assert.match(faces, /const workspaceBytes = \(rowCapacity \+ 1\) \* 16/);
  assert.match(faces, /size: OCTREE_POWER_FACE_CONTROL_BYTES/);
  assert.match(faces, /size: OCTREE_POWER_FACE_PARAMETER_BYTES/);
});
