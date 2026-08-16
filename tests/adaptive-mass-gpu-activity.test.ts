import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createWebgpuSparseCM12ResidentWGSL, webgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import {
  evaluateSparseCM12Performance,
  SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  type SparseCM12BenchmarkArm,
} from "../lib/methods/adaptive-mass/adaptive-mass-performance";
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import { cm12Scene } from "../lib/core/cm12-paper-scenes";
import { FINE_LEVELSET_METADATA_WORDS } from
  "../lib/core/fine-levelset-brick-abi";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createMinimalPowerDamBreak64Scene,
  createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  sparseBrickKey,
  sparseBrickAtlasStats,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  dormantReceiverDomain,
  dormantReceiverResolution,
  adaptiveMassReceiverScaleForScene,
  adaptiveMassPresentationDimensionsForScene,
  residentSupportAtlas,
  SPARSE_CM12_RECEIVER_CAPACITY_FACTOR,
  SPARSE_CM12_RECEIVER_SUPPORT_RINGS,
} from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  decodeSparseCM12FinePresentationSource,
  SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE,
  SPARSE_CM12_HOST_TEMPLATE_MUTABLE_BRICK_MAXIMUM,
  sparseCM12HostTemplateVariantsEnabled,
  sparseCM12FinePresentationPlan,
  sparseCM12TopologyPagePoolPlan,
} from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import {
  ADAPTIVE_MASS_RUNTIME_PARAM_KEYS,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";

test("resident activity measurement is GPU-owned and disjoint from accepted fields", () => {
  assert.match(webgpuSparseCM12ResidentWGSL,
    /@group\(0\)@binding\(12\)var<storage,read_write>activity:array<atomic<u32>>/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn advanceActivityClock\(\)/);
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn measureBrickActivity");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn planBrickResolution", begin,
  );
  assert.ok(begin >= 0 && end > begin, "activity kernel must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /atomicStore\(&activity\[/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "measurement must not mutate accepted physics state");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/,
    "measurement must not mutate accepted topology");
});

test("resident sharpening converts the CM12 pseudo-time to finest-cell units", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn sharpeningDelta");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn scatterSharpeningMass", begin,
  );
  assert.ok(begin >= 0 && end > begin, "sharpening kernel must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /pseudoTimeFineCells=3\.0\*p\.frame\.x\/p\.frame\.y/,
    "3 dt must be divided by finest-cell metres before using grid-coordinate distances");
  assert.match(kernel, /pseudoTimeFineCells\/beforeDistance/);
  assert.doesNotMatch(kernel, /courant\*width\/beforeDistance/,
    "local cell width must not cancel the physical density gradient");
});

test("resident sharpening implements CM12 Algorithm 2 trace and scatter", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn traceSharpeningMass");
  const end = webgpuSparseCM12ResidentWGSL.indexOf("fn preserveHorizontalD4", begin);
  assert.ok(begin >= 0 && end > begin, "sharpening return must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /maximumDistance=p\.sharpening\.x\*sourceWidth/);
  assert.match(kernel, /step<40u/);
  assert.match(kernel, /step>=u32\(p\.sharpening\.y\)/);
  assert.match(kernel, /let field=sampleSharpeningField\(position\)/);
  assert.match(kernel, /let gradient=field\.yzw/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn prepareSharpeningField/,
    "the per-cell sharpening dose must be frozen once before trace donors run");
  assert.match(kernel, /0\.5\*cellMinimumWidth\(owner\)/,
    "the trace step must adapt after crossing a 2:1 seam");
  assert.match(kernel, /gradient\/magnitude\*distance/);
  assert.match(kernel, /rho==0\.0/);
  assert.match(kernel, /rho>CM12_LIQUID_ISOVALUE/,
    "empty support and liquid-side cells must skip the composite sharpening stencil");
  assert.match(kernel, /f32\(removedFixed\)\*weight\/total/,
    "removed integrated mass must be scattered trilinearly at the traced point");
  assert.match(kernel, /offeredFixed=remainingFixed/,
    "fixed-point scatter rounding must remain exactly conservative");
  assert.match(kernel, /\+incoming\/cellVolume\(cell\)/,
    "every fixed-point deposit must be resolved into receiver density");
  assert.doesNotMatch(kernel, /uphillConductance|capacity\/incoming/,
    "the paper trace must not fall back to the old one-face/capacity shortcut");
});

test("resident face preparation preserves the dry extrapolated velocity band", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn prepareTransportFaces");
  const end = webgpuSparseCM12ResidentWGSL.indexOf("fn traceGammaAndBeta", begin);
  assert.ok(begin >= 0 && end > begin, "face preparation must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel,
    /state\[destinationCellVelocity\(\)\+4u\*cell\+3u\]>0\.5/,
    "the skip predicate must follow extrapolated-velocity validity into dry receivers");
  assert.doesNotMatch(kernel, /touchesTransport|rho>CM12_DRY_CELL_THRESHOLD/,
    "density cannot decide whether a dry receiver face carries front velocity");
});

test("resident gamma diffusion uses conservative row-owned snapshot iterations", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn scatterGammaRow");
  const end = webgpuSparseCM12ResidentWGSL.indexOf("fn conditionedDensity", begin);
  assert.ok(begin >= 0 && end > begin, "gamma transaction must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /acceptedTemplateRowInvocation\(gid\.x\)/,
    "each physical composite row must own its endpoint pairs once");
  assert.match(kernel, /atomicAdd\(&conditioning\[negative\],rhoReceipt\)/);
  assert.match(kernel, /atomicAdd\(&conditioning\[positive\],-rhoReceipt\)/);
  assert.match(kernel, /atomicAdd\(&conditioning\[p\.counts\.x\+negative\],gammaReceipt\)/);
  assert.match(kernel, /atomicAdd\(&conditioning\[p\.counts\.x\+positive\],-gammaReceipt\)/,
    "the two endpoints must receive one exactly antisymmetric fixed-point receipt");
  assert.match(kernel, /state\[outputRho\+cell\]=ownRho\+rhoReceipt\*inverseVolume/,
    "the paired density receipt must be resolved by physical cell volume");
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL,
    /diffuseGammaForwardX|diffuseGammaReverseX|averageGammaDiffusion/,
    "the six order-dependent axis sweeps must not return");

  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source,
    /closePass\(\);\s*encoder\.clearBuffer\(this\.conditioning,[\s\S]*?\);\s*stage\("gamma-diffusion"/,
    "transport receipts must be retired before their accumulator banks are recycled");
  assert.match(source,
    /dispatchAccepted\("scatterGammaSnapshot", "row"\);\s*dispatchAccepted\("finalizeGammaSnapshot", "cell"\)/);
  assert.match(source,
    /dispatchAccepted\("scatterGammaRefinement", "row"\);\s*dispatchAccepted\("finalizeGammaRefinement", "cell"\)/,
    "a second stable snapshot iteration must recover paper-step dam-front conditioning");
});

test("resident D4 preservation has disjoint density and gamma scratch", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source, /sharpeningDelta: cells\(\), symmetryGamma: cells\(\)/,
    "density and gamma symmetry values must not share storage");
  // Only the first two words are pinned: the rest of stateOffsets5 is reserved
  // space that later regions claim, and a test that pinned the zeros would fail
  // for a claim that has nothing to do with what it is guarding.
  assert.match(source,
    /u\.set\(\[l\.sharpeningDelta, l\.symmetryGamma,/,
    "stateOffsets5.y must address allocated gamma scratch rather than densityA");
});

test("resident pressure solve caches its classified epoch masks", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source, /clearBuffer\(this\.state, 4 \* this\.layout\.theta/);
  assert.match(source, /clearBuffer\(this\.state, 4 \* this\.layout\.liquid/);
  assert.match(source, /dispatchPressureCell\("applyPipelinedImage"\)/);
  assert.match(source, /dispatchAccepted\("compactPressureCells", "cell"\)/,
    "pressure must compact accepted topology to live liquid cells once per frame");
  assert.doesNotMatch(source,
    /stage\("pressure-topology",[\s\S]*?clearBuffer\(this\.conditioning/,
    "pressure compaction overwrites its accepted-workgroup census and must not clear"
      + " the full template arena");

  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn applyOperator");
  const end = webgpuSparseCM12ResidentWGSL.indexOf("fn preparePressure", begin);
  assert.ok(begin >= 0 && end > begin, "pressure operator must be inspectable");
  const operator = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.doesNotMatch(operator, /rowAccepted\(row\)/,
    "the recurring SpMVs must consume the cleared/classified theta mask");
  assert.match(operator, /let edgeOffsets=pressureTemplateWord\(15u\)/,
    "the immutable prepacked pressure-edge section must be cached per matrix row");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn isLiquid\(cell:u32\)->bool\{return state\[p\.stateOffsets2\.w\+cell\]>0\.5;\}/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn pressureCellInvocation[\s\S]*fineSamples\[4u\+invocation\]/,
    "hot PCG kernels must retain stable vector IDs through a compact invocation list");
});

test("SIM pressure controls bound live PCG work and optional early stopping", () => {
  assert.equal(adaptiveMassSolverOptions({}).pressureRelativeTolerance,
    SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE);
  assert.ok(SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE > 0,
    "production Sparse CM12 must use true-residual-driven execution");
  const configured = adaptiveMassSolverOptions({
    pressureIterations: 40,
    pressureRelativeTolerance: 0.012,
  });
  assert.equal(configured.pressureIterations, 40);
  assert.equal(configured.pressureRelativeTolerance, 0.012);
  assert.equal(adaptiveMassSolverOptions({ pressureIterations: 1 }).pressureIterations, 8);
  assert.equal(adaptiveMassSolverOptions({ pressureIterations: 999 }).pressureIterations, 256);
  assert.equal(adaptiveMassSolverOptions({ pressureRelativeTolerance: -1 })
    .pressureRelativeTolerance, 0);
  assert.equal(adaptiveMassSolverOptions({ pressureRelativeTolerance: 1 })
    .pressureRelativeTolerance, 0.1);
  assert.equal(adaptiveMassSolverOptions({}).presentationPageResolution, 4);
  assert.equal(adaptiveMassSolverOptions({
    brickFineResolution: "16", presentationPageResolution: "8",
  }).presentationPageResolution, 8);
  assert.equal(adaptiveMassSolverOptions({
    brickFineResolution: "4", presentationPageResolution: "16",
  }).presentationPageResolution, 4,
  "presentation pages cannot exceed the selected solver brick width");

  const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
    (candidate) => candidate.id === "pressure-solve",
  );
  assert.ok(stage);
  const controls = new Set(stage.controls?.flatMap((control) =>
    control.kind === "readout" ? [] : [control.param]));
  assert.ok(controls.has("pressureIterations"));
  assert.ok(controls.has("pressureRelativeTolerance"));
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("pressureIterations"));
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("pressureRelativeTolerance"));

  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source, /iteration < pressureIterations/,
    "the host dispatch budget must use the live iteration control");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn reduceGuardedTrueResidual[\s\S]*receipt\.x<=tolerance\*tolerance\*scalars\[1\]/,
    "the device must stop PCG only from a freshly recomputed b-Ap residual");
  assert.match(source,
    /\(iteration \+ 1\) % SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE/,
    "the resident solve must guard long f32 recurrences at a fixed cadence");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn pipelinedPressureActive\(\)->bool\{return scalars\[5\]>0\.5&&scalars\[14\]<0\.5;\}/,
    "the single pipelined solver must gate converged and curvature-recovery tails");
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL,
    /PRESSURE_EARLY_STOP|projectedJacobiToPressure|fn applyDirection/,
    "the immediate cutover must not retain alternate pressure pipelines");
  assert.match(source, /no fallback solver is installed/,
    "unsupported separating boundaries must fail closed during construction");
});

test("resident topology templates snapshot pooled geometry by value", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("function snapshotTemplateCell");
  const end = source.indexOf("function resampleBrick", begin);
  assert.ok(begin >= 0 && end > begin, "template cell snapshot must be inspectable");
  const snapshot = source.slice(begin, end);
  for (const field of ["brickCoordinate", "local", "minimumFine", "maximumFine",
    "centerFine", "widthsFine"] as const) {
    assert.match(snapshot, new RegExp(`${field}: \\[\\.\\.\\.source\\.${field}\\]`),
      `${field} must not alias the recycled composite-grid workspace`);
  }
  assert.match(source,
    /rows\.push\(\{ \.\.\.source, id: rows\.length, centerFine: \[\.\.\.source\.centerFine\], terms \}\)/,
    "retained template row centers must not alias the recycled workspace");
  assert.match(source, /templates\.words\[base \+ 7\]! < 8 \* coordinate\[0\]/,
    "template construction must reject spatially aliased cell ranges");
});

test("adaptive construction leaves interface resolution to the atlas initializer", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf('id: "adaptive-mass.atlas"');
  const end = source.indexOf('id: "adaptive-mass.presentation"', begin);
  const construction = source.slice(begin, end);
  assert.match(construction, /options\.resolutionMode === "all-coarse"/);
  assert.match(construction, /:\s*undefined/);
  assert.match(construction, /\.\.\.\(resolutionForBrick \? \{ resolutionForBrick \} : \{\}\)/);
  assert.doesNotMatch(construction,
    /options\.resolutionMode === "all-fine"[\s\S]*?\? \(\) => 8[\s\S]*?: \(\) => 4,/,
    "adaptive mode must not blanket-force every interface brick to 4 cubed");
  assert.doesNotMatch(source, /initializeSparseAtlasDynamics/,
    "GPU-only startup must not construct an unused CPU dynamics state");
  assert.match(construction, /grid = buildSparseAtlasCompositeGrid\(atlas\)/);
});

test("frame scheduling remains GPU-only and never waits for topology readback", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  const advanceBegin = source.indexOf("  advanceTo(");
  const advanceEnd = source.indexOf("  private finishFrameCapture(", advanceBegin);
  const advance = source.slice(advanceBegin, advanceEnd);
  assert.doesNotMatch(advance,
    /readGPUActivityPolicy|readActivitySnapshot|readCompactRemeshState|mapAsync|onSubmittedWorkDone/);
  assert.match(source, /Explicit acceptance\/debug readback; never consulted by advanceTo/);
});

test("GPU candidate planning is epoch-gated and cannot mutate accepted state", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn planBrickResolution");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn closePlannedResolution", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate planner must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /requested=atomicLoad\(&activity\[output\+8u\]\)/,
    "candidate requests must persist between topology epochs");
  assert.match(kernel,
    /if\(!brickActive\(brick\)\)\{\s*atomicStore\(&activity\[output\+8u\],current\)/,
    "dormant capacity must retain metadata until a physical receiver activates it");
  assert.match(kernel,
    /velocityFloor=select\(1u,measuredVelocityFloor,activitySignals\)/);
  assert.match(kernel, /adaptiveSurface=surface&&!enclosed/);
  assert.match(kernel,
    /select\(1u,8u,requiredSurface\|\|thinFluid\|\|receiver\)/);
  assert.doesNotMatch(kernel, /horizontalD4IsAuthoritative/,
    "D4 field symmetry must not freeze topology planning");
  assert.match(kernel,
    /else if\(!activitySignals\|\|atomicLoad\(&activity\[5\]\)!=0u\)/,
    "surface-distance mode must share the planner without inheriting activity cadence");
  assert.match(kernel,
    /if\(activitySignals&&!enclosed&&!slowSurface&&hotEpochs>=p\.activityEpochs\.y\)/);
  assert.match(kernel, /requested=min\(8u,2u\*current\)/);
  assert.match(kernel,
    /\(!activitySignals\|\|enclosed\|\|slowSurface\|\|quietEpochs>=p\.activityEpochs\.z\)&&!detail/);
  assert.match(kernel,
    /requested=select\(max\(required,current\/2u\),required,directBulk\)/);
  assert.match(kernel, /atomicStore\(&activity\[output\+8u\],requested\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/);
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("GPU selector keeps free surfaces fine and deep translating bulk coarse in activity mode", () => {
  const floorBegin = webgpuSparseCM12ResidentWGSL.indexOf("fn velocityResolutionFloor");
  const measureBegin = webgpuSparseCM12ResidentWGSL.indexOf("fn measureBrickActivity");
  const planBegin = webgpuSparseCM12ResidentWGSL.indexOf("fn planBrickResolution");
  assert.ok(floorBegin >= 0 && measureBegin > floorBegin && planBegin > measureBegin);
  const floor = webgpuSparseCM12ResidentWGSL.slice(floorBegin, measureBegin);
  assert.match(floor, /if\(!activitySignalsEnabled\(\)\)\{return 1u;\}/,
    "surface-distance mode must ignore velocity floors");
  assert.match(floor, /travelFineCells>=p\.activityThresholds\.x\)\{return 8u/);
  assert.match(floor, /travelFineCells>=p\.activityThresholds\.y\)\{return 4u/);
  assert.match(floor, /travelFineCells>=p\.activityThresholds\.z\)\{return 2u/);
  assert.match(floor, /return 1u/);
  const measurement = webgpuSparseCM12ResidentWGSL.slice(measureBegin, planBegin);
  assert.match(measurement, /densitySum\+=i32\(round\(rho\*ACTIVITY_FIXED\)\)/,
    "activity fixed point must remain dimensionless before its reduction");
  assert.doesNotMatch(measurement, /densitySum\+=i32\(round\(rho\*volume\*ACTIVITY_FIXED\)\)/,
    "a full 32-cubed macro cell must not overflow signed activity fixed point");
  assert.match(measurement,
    /densityMassFineCells=f32\(activityDensitySum\[0\]\)\/ACTIVITY_FIXED\*cellVolume\(first\)/,
    "the common macro-cell volume must be applied after the safe fixed-point reduction");
  assert.match(measurement, /p\.frame\.x\*length\(ownVelocity\)/,
    "velocity activity must be expressed as finest-cell travel per accepted step");
  assert.match(measurement,
    /normalizedVelocityActivity=velocityActivity\s*\/max\(p\.activityThresholds\.x,1e-6\)/,
    "the finest travel control must normalize emergency velocity scoring too");
  assert.match(measurement,
    /scoredVelocityActivity=select\(0\.0,normalizedVelocityActivity,surface\|\|thinFluid\)/,
    "uniformly translating flooded bulk must not score as missing spatial detail");
  assert.match(measurement,
    /dynamicActivity=select\(0\.0,max\(activityDeformation\[0\],temporal\),\s*surface\|\|thinFluid\)/,
    "hydrostatic deformation residue must not erase deep-bulk coarsening");
  assert.match(measurement,
    /detailActivity=max\(0\.0,scoredDetailError\/p\.activityDensity\.w-1\.0\)/,
    "restriction detail below the configured tolerance must score zero");
  assert.match(measurement,
    /scoredDetailError=select\(activityDetailError\[0\],0\.0,\s*surface&&!thinFluid&&shape==0\.0\)/,
    "a sharpened calm planar surface must not retain a permanent detail veto");
  assert.match(measurement, /activityDetailError\[0\]>p\.activityDensity\.w/,
    "sub-veto detail must not pin every nonuniform brick forever");
  assert.match(measurement,
    /representedThickness=clamp\(rho,0\.0,1\.0\)\*cellMinimumWidth\(cell\)/,
    "thinness must be measured in finest-cell units on the represented leaf");
  assert.match(measurement,
    /representedThickness<p\.activityThresholds\.w\s*&&\(exposedSides&oppositeSides\)==oppositeSides/,
    "a sub-two-cell slab needs exposed support on both sides of an axis");
  assert.match(measurement, /var interfaceCell=false/,
    "fractional submerged density must not become surface evidence by itself");
  assert.doesNotMatch(measurement, /surfaceCell=surfaceCell\|\|interfaceCell/,
    "wall-conditioned bulk density must not pin bottom bricks at the surface rung");
  assert.match(measurement,
    /if\(fractionalCell&&airFacing\)\{\s*interfaceCell=true;surfaceAxes\|=1u<<axis/,
    "a diffuse fractional interface must retain an interior air-facing side");
  assert.match(measurement,
    /sideHasSurfaceFluid=sideHasSurfaceFluid\s*\|\|neighborDensity>p\.activityDensity\.y/,
    "diffuse surface exposure must ignore sub-threshold transported mist");
  assert.match(measurement,
    /crossesIsovalue[\s\S]*?min\(fill,neighborDensity\)<=p\.activityDensity\.y/,
    "a submerged oscillation around the isovalue must not become a surface crossing");
  assert.match(measurement,
    /if\(crosses\)\{\s*interfaceCell=true;\s*surfaceAxes\|=1u<<rowAxis\(row\)/,
    "surface evidence must come from an accepted liquid-air crossing");
  assert.match(measurement, /if\(thinFluid\)\{reasons\|=256u;\}/,
    "thin fluid must remain independently visible in policy diagnostics");
  assert.match(measurement,
    /quiet=!thinFluid&&velocityFloor<current\s*&&select\(!surface,activityQuiet,activitySignals\)/,
    "activity histories may mark a calm surface quiet without weakening its independent floor");
  assert.match(measurement,
    /cellCenter\s*\+p\.activityTiming\.x\*p\.frame\.x\*ownVelocity/,
    "front support must use the live accepted-step lookahead");
  assert.match(measurement,
    /if\(interfaceCell\|\|cellIsThinFluid\)\{/,
    "every represented interface must retain its immediate static receiver shell");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /step%p\.activityEpochs\.x==0u/,
    "candidate topology cadence must come from the live policy uniform");
  const planEnd = webgpuSparseCM12ResidentWGSL.indexOf("fn closePlannedResolution", planBegin);
  const plan = webgpuSparseCM12ResidentWGSL.slice(planBegin, planEnd);
  assert.doesNotMatch(plan, /if\(!activitySignals\)[\s\S]*?return;/,
    "selector modes must not fork the physical resolution planner");
  assert.match(plan,
    /velocityFloor=select\(1u,measuredVelocityFloor,activitySignals\)/,
    "surface-distance mode must disable only the activity-derived velocity floor");
  assert.match(plan,
    /requiredSurface=select\(surface,adaptiveSurface,activitySignals\)/,
    "both modes must share the surface floor and differ only in activity filtering");
  assert.match(plan, /boundaryRequired=activitySignals&&cutBoundary/,
    "surface-distance mode must not retain an activity boundary floor");
  assert.match(plan,
    /!activitySignals\|\|enclosed\|\|slowSurface\|\|quietEpochs>=p\.activityEpochs\.z/,
    "surface-distance bulk must bypass activity history before coarsening");
  assert.match(plan, /directBulk=!activitySignals\|\|enclosed/,
    "surface-distance bulk must request the coarsest rung in the shared planner");
  assert.match(plan,
    /movingInternalSurface=activitySignals&&surface&&velocityFloor>1u[\s\S]*?enclosed=activitySignals&&brickDeeplyEnclosed\(brick\)&&!movingInternalSurface/,
    "a fast internal interface must veto the accepted-neighbour deep-bulk classification");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /brickDirectoryLookupAtCoordinate\(vec3u\(neighborCoordinate\)\)/,
    "deep enclosure must recognize adjacent macro-bricks rather than exact origins only");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /activityF32\(activityRecord\(neighbor\)\+4u\)<CM12_LIQUID_ISOVALUE/,
    "air-receiver residue must not hide a real exposed surface");
  assert.match(plan,
    /receiver=injectionReceiver\s*\|\|\(receiverRequested&&\(\(reasons&64u\)==0u\s*\|\|\(activitySignals&&velocityFloor>1u\)\)\)/,
    "empty and fast destinations must retain the urgent receiver floor");
  assert.match(plan, /adaptiveSurface=surface&&!enclosed/,
    "only exposed interfaces may pin the surface floor; enclosed rho ripples must recover");
  assert.match(plan,
    /select\(1u,8u,requiredSurface\|\|thinFluid\|\|receiver\)/,
    "every genuine surface, thin liquid, and moving receiver must get an 8-cubed floor");
  assert.doesNotMatch(plan, /select\(1u,4u,requiredSurface\)/,
    "calm free surfaces must never demote to the middle rung");
  assert.match(plan, /urgent=requiredSurface\|\|thinFluid\|\|receiver/,
    "a newly detected surface must jump directly back to 8 cubed");
  assert.match(plan, /slowSurface=adaptiveSurface&&!thinFluid&&velocityFloor==1u/,
    "a slow non-thin interface must remain quiet without weakening its surface floor");
  assert.match(plan, /&&!enclosed&&!slowSurface/,
    "deep or slow-surface restriction residue must remain diagnostic without changing the surface floor");
  assert.match(plan,
    /requested=select\(min\(8u,max\(required,2u\*current\)\),required,urgent\)/,
    "non-urgent measured activity must advance by one rung");
  assert.match(plan,
    /if\(p\.injectionCenter\.w!=0\.0&&!injectionReceiver\)\{[\s\S]*?activity\[output\+8u\],current/,
    "an injection topology transaction must preserve every untouched accepted brick");
  assert.match(plan,
    /if\(!brickActive\(brick\)\)\{\s*atomicStore\(&activity\[output\+8u\],current\)/,
    "dormant capacity must not schedule pointless coarsening before receiver activation");
  assert.match(plan,
    /recoveryFloor=recoveryState&15u[\s\S]*?recoveryLocked=\(recoveryState&ACTIVITY_RECOVERY_LOCK\)!=0u/,
    "candidate planning must retain the brick's pre-promotion calm level");
  assert.match(plan,
    /settledRecoveredBulk=activitySignals&&recoveryLocked&&surface[\s\S]*?quietEpochs>=p\.activityEpochs\.z/,
    "a quiet, refilled deep brick must be allowed to dismiss internal rho crossings");
  assert.match(plan, /select\(1u,recoveryFloor,recoveryRequired\)/,
    "recovery must stop exactly at the remembered calm level");
  assert.match(plan,
    /required=select\(dynamicRequired,select\(1u,4u,boundaryRequired\),enclosed\)/,
    "submerged bulk must ignore motion/history floors while retaining cut-boundary detail");
  assert.match(plan,
    /requested=select\(max\(required,current\/2u\),required,directBulk\)/,
    "enclosed liquid must jump to its coarsest request before 2:1 closure");
  assert.match(plan, /planReasons=select\(16u,2048u,directBulk\)/,
    "aggressive submerged requests must remain observable in GPU policy receipts");
  const commitBegin = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn validateAndCommitShadowTopology",
  );
  const commit = webgpuSparseCM12ResidentWGSL.slice(commitBegin);
  assert.match(commit,
    /if\(next>accepted\)\{\s*atomicStore\(&activity\[output\+38u\],recoveryFloor\|ACTIVITY_RECOVERY_LOCK\)/,
    "the first accepted promotion must freeze the calm recovery target");
  const receiver = webgpuSparseCM12ResidentWGSL.slice(
    webgpuSparseCM12ResidentWGSL.indexOf("fn brickRequestedAsReceiver"), planBegin,
  );
  assert.match(receiver, /neighborOutput\+32u/,
    "empty receivers must retain the complete immediate support receipt");
  assert.match(measurement, /output\+39u.*activitySweptSupportMask/,
    "the swept subset must remain independently observable in policy telemetry");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn activateSweptReceivers[\s\S]*?brickRequestedAsReceiver\(brick\)/,
    "activation and retained receiver resolution must share one prediction predicate");
  assert.match(plan, /else if\(thinFluid\)\{planReasons=256u;\}/);
});

test("liquid injection commits fine topology before writing the drop", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("encodeLiquidInjection(");
  const end = source.indexOf("private lastPacked", begin);
  assert.ok(begin >= 0 && end > begin, "liquid injection must be independently inspectable");
  const injection = source.slice(begin, end);
  const plan = injection.indexOf('dispatchTopology("planBrickResolution"');
  const commit = injection.indexOf('dispatchTopology("validateAndCommitShadowTopology"');
  const wet = injection.indexOf("this.pipelines.injectLiquid");
  assert.ok(plan >= 0 && commit > plan && wet > commit,
    "the intersected topology must plan, commit, and only then receive density");
  assert.match(injection,
    /copyBufferToBuffer\(this\.topologyArena,[\s\S]*?this\.acceptedIndirectArguments/,
    "the promoted accepted worklists must be snapshotted in the injection transaction");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn injectionReachesBrick[\s\S]*?injectionCenter\.xyz\+p\.injectionRadius\.xyz>=lower/,
    "drop/brick intersection must be independent of the brick's current cell resolution");
});

test("Sparse CM12 exposes normalized structural and live candidate policy controls", () => {
  const options = adaptiveMassSolverOptions({
    selectorMode: "activity",
    receiverFloor: "8", surfaceFineRings: 3, receiverSupportRings: 12,
    finestTravelCells: 0.5, fourTravelCells: 0.75, twoTravelCells: 0.9,
    frontLookaheadSteps: 9.4, thinFeatureCells: 1.75, thinFeatureDensity: 0.02,
    residencyDensity: 0.007, residencyMassFineCells: 1.5,
    topologyCadenceSteps: 3, prepareBricksPerFrame: 11,
    promoteEpochs: 4, demoteEpochs: 7,
    promoteScore: 0.7, demoteScore: 0.8, emergencyScore: 0.6,
  });
  assert.equal(options.receiverFloor, 8);
  assert.equal(options.surfaceFineRings, 3);
  assert.equal(options.receiverSupportRings, 12);
  assert.deepEqual([
    options.activityPolicy?.finestTravelCells,
    options.activityPolicy?.fourTravelCells,
    options.activityPolicy?.twoTravelCells,
  ], [0.5, 0.5, 0.5], "velocity floors must remain descending");
  assert.equal(options.activityPolicy?.frontLookaheadSteps, 9);
  assert.equal(options.activityPolicy?.thinFeatureCells, 1.75);
  assert.equal(options.activityPolicy?.residencyDensity, 0.007);
  assert.equal(options.activityPolicy?.residencyMassFineCells, 1.5);
  assert.equal(options.activityPolicy?.demoteScore, 0.7);
  assert.equal(options.activityPolicy?.emergencyScore, 0.7);
  assert.equal(options.activityPolicy?.activitySignals, true);
  assert.equal(options.activityPolicy?.prepareBricksPerFrame, 11);
  const surfaceOnly = adaptiveMassSolverOptions({ selectorMode: "surface" });
  assert.equal(surfaceOnly.activityPolicy?.activitySignals, false,
    "the live Surface distance choice must select the history-free GPU fast path");
  const defaults = adaptiveMassSolverOptions({}).activityPolicy;
  assert.equal(defaults?.activitySignals, false,
    "Surface distance must be the default adaptive criterion");
  assert.equal(defaults?.residencyDensity, 0.005,
    "the default region cutoff must reject settled dilute residue");
  assert.equal(defaults?.residencyMassFineCells, 1,
    "subcell fragments must not keep a whole sparse region populated");

  const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
    (candidate) => candidate.id === "resolution-planning",
  );
  assert.ok(stage);
  const params = new Set(stage.controls?.flatMap((control) =>
    control.kind === "readout" ? [] : [control.param]));
  for (const param of ["selectorMode", "receiverFloor", "surfaceFineRings", "receiverSupportRings",
    "finestTravelCells", "fourTravelCells", "twoTravelCells", "frontLookaheadSteps",
    "thinFeatureCells", "thinFeatureDensity", "residencyDensity",
    "residencyMassFineCells", "surfaceDensityMinimum",
    "surfaceDensityMaximum", "detailTolerance", "topologyCadenceSteps",
    "prepareBricksPerFrame",
    "promoteEpochs", "demoteEpochs", "promoteScore", "demoteScore", "emergencyScore"]) {
    assert.ok(params.has(param), `SIM activity stage is missing ${param}`);
  }
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("finestTravelCells"));
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("selectorMode"));
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("prepareBricksPerFrame"));
  assert.ok(!ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("receiverFloor" as never),
    "the accepted receiver floor must rebuild rather than pretend to update live");
  assert.match(stage.tip.summary,
    /sends deeply submerged bricks directly to the coarsest level permitted by the accepted 2:1-closed/);
});

test("GPU candidate levels close the full 1/2/4/8 ladder to 2:1", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn closePlannedResolution");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn validBrickResolution", begin,
  );
  assert.ok(begin >= 0 && end > begin, "grading closure must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /max\(neighborResolution,neighborAccepted\)\/2u/,
    "a partially published coarsening transaction must remain 2:1 against accepted neighbors");
  assert.match(kernel, /atomicMax\(&activity\[activityRecord\(brick\)\+8u\],required\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/);
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(residentSource,
    /for \(let gradingPass = 0; gradingPass < 3; gradingPass \+= 1\)/);
});

test("GPU candidate validation is isolated from accepted level and topology", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn validateCandidateResolution");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn scheduleTopologyPreparation", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate validator must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /validBrickResolution\(accepted\)/);
  assert.match(kernel, /larger>2u\*smaller/);
  assert.match(kernel, /let transition=candidate!=accepted/,
    "existing and dormant represented bricks must both be eligible for publication");
  assert.match(kernel, /atomicStore\(&activity\[output\+13u\],candidate\)/);
  assert.match(kernel, /atomicStore\(&activity\[output\+14u\]/);
  assert.doesNotMatch(kernel, /atomicStore\(&activity\[output\+12u\]/,
    "validation cannot publish the accepted logical level");
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/);
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("large-scene topology pages use a bounded GPU free list", () => {
  assert.equal(SPARSE_CM12_HOST_TEMPLATE_MUTABLE_BRICK_MAXIMUM, 2048,
    "the complete template path must cover the bounded long-dam receiver domain");
  assert.equal(sparseCM12HostTemplateVariantsEnabled(250_000, 750_000, 2048), true,
    "the bounded compatibility frontier may retain prepacked host variants");
  assert.equal(sparseCM12HostTemplateVariantsEnabled(250_000, 750_000, 2049), false,
    "mutable brick count must cap host work even when accepted topology is compact");
  assert.equal(sparseCM12HostTemplateVariantsEnabled(250_001, 1, 1), false);
  assert.equal(sparseCM12HostTemplateVariantsEnabled(1, 750_001, 1), false);
  assert.deepEqual(sparseCM12TopologyPagePoolPlan(0), {
    pageCapacity: 0, freeListWords: 0, pageWords: 8_196, descriptorWords: 0,
  });
  assert.deepEqual(sparseCM12TopologyPagePoolPlan(64), {
    pageCapacity: 32, freeListWords: 32, pageWords: 8_196,
    descriptorWords: 262_272,
  });
  assert.deepEqual(sparseCM12TopologyPagePoolPlan(100_000), {
    pageCapacity: 512, freeListWords: 512, pageWords: 8_196,
    descriptorWords: 4_196_352,
  });
  assert.match(webgpuSparseCM12ResidentWGSL, /fn acquireTopologyPage\(\)->u32/);
  assert.match(webgpuSparseCM12ResidentWGSL, /atomicCompareExchangeWeak/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn allocateCandidateTopologyPages\(@builtin\(global_invocation_id\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn synthesizeCandidateCellPages\(@builtin\(local_invocation_id\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn brickCandidatePlanningEnabled\(brick:u32\)->bool/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn deferDynamicTopologyPublication\(\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /atomicStore\(&activity\[16\],0u\)/);
  assert.match(readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url,
  ), "utf8"), /topologyPage: words\[at \+ 37\]/);
  const synthesis = webgpuSparseCM12ResidentWGSL.slice(
    webgpuSparseCM12ResidentWGSL.indexOf("fn synthesizeCandidateCellPages"),
    webgpuSparseCM12ResidentWGSL.indexOf("fn beginShadowTopology"),
  );
  assert.match(synthesis, /pageBase\+4u\+16u\*local/);
  assert.doesNotMatch(synthesis, /state\[/,
    "geometry synthesis must not mutate accepted fields before publication");
});

test("GPU candidate cell transfer is conservative and remains non-authoritative", () => {
  assert.match(webgpuSparseCM12ResidentWGSL,
    /@group\(0\)@binding\(13\)var<storage,read_write>candidateState:array<f32>/);
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn transferCandidateCells");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn prepareCandidateFaceReceipts", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate transfer must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /candidate<accepted/);
  assert.match(kernel, /momentumSum\/massSum/);
  assert.match(kernel, /beforeMomentumScale\+=abs\(momentumContribution\)/);
  assert.match(kernel, /afterMomentumScale\+=abs\(momentumContribution\)/);
  assert.match(kernel, /momentumTolerance=max\(vec3f\(1e-3\),1e-6\*vec3f\(/);
  assert.match(kernel, /all\(abs\(momentumError\)<=momentumTolerance\)/);
  assert.doesNotMatch(kernel, /momentumError\)<=vec3f\(max\(1e-3,tolerance\)\)/);
  assert.match(kernel, /candidateState\[candidateFieldIndex\(0u,brick,local\)\]=rho/);
  assert.match(kernel, /atomicStore\(&activity\[output\+18u\],bitcast<u32>\(massError\)\)/);
  assert.match(kernel, /if\(!valid\)\{atomicOr\(&activity\[7\],2u\);\}/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "candidate transfer cannot write accepted fields");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("GPU candidate face transfer area-averages authoritative exterior flux", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn transferCandidateFaces");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn writeCandidateCellsToShadow", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate face transfer must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /state\[destinationFaceVelocity\(\)\+row\]\*rowAreaValue/);
  assert.match(kernel, /candidateState\[candidateFieldIndex\(6u\+side,brick,lane\)\]/);
  assert.match(kernel, /reduceB\[0\]-reduceA\[0\]/);
  assert.match(kernel, /atomicOr\(&activity\[7\],4u\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "face transfer cannot write accepted velocity");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("swept receiver activation is GPU-published and dormant cells stay inert", () => {
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /atomicCompareExchangeWeak\(&activity\[output\+10u\],0u,1u\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /atomicAdd\(&activity\[10\],1u\);atomicAdd\(&activity\[11\],count\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /if\(!cellActive\(id\)\)\{\s*state\[destinationDensity\(\)\+id\]=0\.0/);
  const plan = residentSource.indexOf('dispatch("planBrickResolution"');
  const activate = residentSource.indexOf('dispatch("activateSweptReceivers"');
  const present = residentSource.indexOf('dispatch("classifyPresentationBricks"');
  assert.ok(plan >= 0 && activate > plan && present > activate,
    "GPU planning must publish receiver activation before presentation");
  const encodeBegin = residentSource.indexOf("  encode(\n");
  const encodeEnd = residentSource.indexOf("  /** Publish generation zero", encodeBegin);
  const encode = residentSource.slice(encodeBegin, encodeEnd);
  assert.doesNotMatch(encode, /mapAsync|readActivitySnapshot|readGPUActivityPolicy/);
});

test("GPU retirement drops only bounded dilute residue and clears stale receiver state", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn retireUnsupportedEmptyBricks");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn classifyPresentationBricks", begin,
  );
  assert.ok(begin >= 0 && end > begin, "retirement kernel must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.doesNotMatch(kernel, /horizontalD4IsAuthoritative/,
    "D4 field symmetry must not disable ordinary brick retirement");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /occupiedCell=occupiedCell\|\|rho>residencyDensity/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /return max\(CM12_DRY_CELL_THRESHOLD,p\.sharpening\.z\)/,
    "region liveness must use its own policy floor without weakening CM12 arithmetic cleanup");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /occupied=densityPresent&&densityMassFineCells>=p\.sharpening\.w/,
    "a concentrated subcell fragment must not count as a populated region");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /activity\[output\+32u\],select\(0u,activitySupportMask\[0\],occupied\)/,
    "subcell fragments must not keep neighboring regions alive through support masks");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /wet=wet\|\|rho>residencyDensityThreshold\(\)[\s\S]*massFineCells\+=max\(0\.0,rho\)\*cellVolume\(at\)[\s\S]*wet=wet&&massFineCells>=p\.sharpening\.w/,
    "presentation population must agree with physical region liveness");
  assert.match(kernel, /activity\[output\+1u\]\)&64u/,
    "mass above the region-density floor must retain its own brick");
  assert.match(kernel, /activityRecord\(neighbor\)\+32u\]\)&\(1u<<bit\)/,
    "only the neighbor's directional surface/swept bit may retain this air brick");
  assert.match(kernel, /for\(var dz=-1;dz<=1;dz\+=1\)/);
  assert.match(kernel, /for\(var dy=-1;dy<=1;dy\+=1\)/);
  assert.match(kernel, /for\(var dx=-1;dx<=1;dx\+=1\)/);
  assert.match(kernel, /atomicCompareExchangeWeak\(&activity\[output\+10u\],1u,0u\)/);
  assert.match(kernel, /residueMass\+=max\(0\.0,state\[destinationDensity\(\)\+cell\]\)\*cellVolume\(cell\)/);
  assert.match(kernel, /atomicStore\(&activity\[output\+34u\],bitcast<u32>\(residueMass\)\)/);
  assert.match(kernel, /atomicSub\(&activity\[8\],1u\)/);
  assert.match(kernel, /state\[p\.stateOffsets0\.x\+cell\]=0\.0/,
    "retirement must erase residue so later activation cannot resurrect it");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/,
    "retirement must not mutate immutable packed topology");
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const activate = residentSource.indexOf('dispatch("activateSweptReceivers"');
  const retire = residentSource.indexOf('dispatch("retireUnsupportedEmptyBricks"');
  const present = residentSource.indexOf('dispatch("classifyPresentationBricks"');
  assert.ok(activate >= 0 && retire > activate && present > retire,
    "receiver publication and support closure must precede retirement and presentation");
});

test("new Sparse CM12 receivers are fine with a full graded support ladder", () => {
  assert.equal(dormantReceiverResolution("adaptive"), 8);
  assert.equal(dormantReceiverResolution("adaptive", 1), 4);
  assert.equal(dormantReceiverResolution("adaptive", 2), 2);
  assert.equal(dormantReceiverResolution("adaptive", 3), 1);
  assert.equal(dormantReceiverResolution("adaptive", 4), 1);
  assert.equal(dormantReceiverResolution("adaptive", 1, 4), 2);
  assert.equal(dormantReceiverResolution("all-coarse"), 4);
  assert.equal(dormantReceiverResolution("all-fine"), 8);
});

test("the created-region floor can guarantee fine physical receivers", () => {
  const dimensions = [24, 24, 24] as const;
  const source = createSparseAdaptiveMassAtlas(dimensions, [{
    key: sparseBrickKey([1, 1, 1], [3, 3, 3]),
    coordinate: [1, 1, 1], resolution: 8,
    density: new Float64Array(8 ** 3).fill(1),
    gamma: new Float64Array(8 ** 3).fill(1),
  }]);
  const domain = dormantReceiverDomain(source, "adaptive", 1, 8);
  assert.equal(domain.bricks.length, 27);
  assert.ok(domain.bricks.every((brick) => brick.resolution === 8),
    "8³ created-region floor must make every pre-created receiver physically fine");
});

test("Figure 7 promotes only fluid-bearing face receivers at construction", () => {
  const scene = cm12Scene("cm12-figure-7");
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const supported = residentSupportAtlas(initial, "adaptive");
  const dormant = dormantReceiverDomain(supported, "adaptive");
  const initialStats = sparseBrickAtlasStats(initial);
  const supportStats = sparseBrickAtlasStats(supported);
  const dormantStats = sparseBrickAtlasStats(dormant);
  assert.equal(supportStats.residentBrickCount - initialStats.residentBrickCount, 8,
    "the spherical source must not create its whole 3x3x3 neighbor shell");
  assert.equal(supportStats.fineBrickCount - initialStats.fineBrickCount, 8,
    "the first transport step needs a fine immediate destination shell");
  assert.ok(dormantStats.leafCount < 100_000,
    `Figure 7 construction regressed to ${dormantStats.leafCount} packed cells`);
});

test("Figure 6 receiver capacity is live-set-shaped rather than domain-shaped", () => {
  const scene = cm12Scene("cm12-figure-6");
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const supported = residentSupportAtlas(initial, "adaptive");
  const resident = dormantReceiverDomain(supported, "adaptive");
  assert.ok(resident.bricks.length
    <= SPARSE_CM12_RECEIVER_CAPACITY_FACTOR * supported.bricks.length);
  assert.ok(resident.bricks.length < 0.5 * 16 ** 3,
    "Figure 6 must not allocate its complete logical brick domain");
  assert.equal(sparseBrickAtlasStats(resident).integratedMassFineCells, 94_276);
});

test("the 64-cubed mini dam has dormant receivers on every domain axis", () => {
  const scene = createMinimalPowerDamBreak64Scene();
  const dimensions = sceneLatticeDimensions(scene);
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
    resolutionForBrick: () => 4,
  });
  const domain = dormantReceiverDomain(initial, "adaptive");
  assert.equal(domain.bricks.length, 8 ** 3);
  assert.ok(domain.bricks.every((brick) => brick.resolution >= 4),
    "a boundary-fed dam needs a traversable 4-cubed immutable receiver pool");
  for (const coordinate of [[7, 0, 7], [0, 7, 7], [7, 7, 7]] as const) {
    const key = coordinate[0] + 8 * (coordinate[1] + 8 * coordinate[2]);
    assert.ok(domain.directory.has(key),
      `missing dormant mini-dam receiver ${coordinate.join(",")}`);
  }
});

test("Sparse CM12 receiver and compact page capacity do not grow with empty world volume", () => {
  const seeded = (dimensions: readonly [number, number, number],
    coordinate: readonly [number, number, number]) => {
    const brickDimensions = dimensions.map((value) => Math.ceil(value / 8)) as
      [number, number, number];
    return createSparseAdaptiveMassAtlas(dimensions, [{
      key: sparseBrickKey(coordinate, brickDimensions),
      coordinate,
      resolution: 8,
      density: new Float64Array(8 ** 3).fill(1),
      gamma: new Float64Array(8 ** 3).fill(1),
    }]);
  };
  const local = dormantReceiverDomain(seeded([256, 32, 32], [16, 2, 2]), "adaptive");
  const vast = dormantReceiverDomain(
    seeded([1_000_000, 32, 32], [62_500, 2, 2]), "adaptive",
  );
  assert.equal(SPARSE_CM12_RECEIVER_SUPPORT_RINGS, 9);
  assert.equal(vast.bricks.length, local.bricks.length,
    "empty world width must not add receiver records");
  assert.equal(sparseBrickAtlasStats(vast).leafCount, sparseBrickAtlasStats(local).leafCount,
    "empty world width must not add solver cells");
  const localPresentation = sparseCM12FinePresentationPlan(local);
  const vastPresentation = sparseCM12FinePresentationPlan(vast);
  assert.equal(vastPresentation.plan.maximumResidentBricks,
    localPresentation.plan.maximumResidentBricks,
    "empty world width must not add compact pages");
  assert.equal(vastPresentation.worklist.length,
    7 + vastPresentation.plan.maximumResidentBricks,
    "compact worklist must omit the logical-domain direct table");
  assert.ok(vastPresentation.plan.logicalBrickCount
    > 1_000 * vastPresentation.plan.maximumResidentBricks,
  "the test world must be materially larger than its retained page set");
});

test("compact Sparse CM12 presentation pages retain their direct source address", () => {
  const atlas = dormantReceiverDomain(createSparseAdaptiveMassAtlas([80, 24, 16], [{
    key: sparseBrickKey([4, 1, 1], [10, 3, 2]),
    coordinate: [4, 1, 1],
    resolution: 8,
    density: new Float64Array(8 ** 3).fill(1),
    gamma: new Float64Array(8 ** 3).fill(1),
  }]), "adaptive", 2);
  const publication = sparseCM12FinePresentationPlan(atlas);
  const [pagesX, pagesY] = publication.plan.brickDimensions;
  let previousKey = -1;
  for (let page = 0; page < publication.plan.maximumResidentBricks; page += 1) {
    const at = FINE_LEVELSET_METADATA_WORDS * page;
    const key = publication.metadata[at + 1]!;
    const source = decodeSparseCM12FinePresentationSource(publication.metadata[at + 3]!);
    const brick = atlas.bricks[source.brick]!;
    const octant = source.octant;
    assert.equal(source.spanBricks, 1);
    const coordinate = [2 * brick.coordinate[0] + (octant & 1),
      2 * brick.coordinate[1] + ((octant >> 1) & 1),
      2 * brick.coordinate[2] + ((octant >> 2) & 1)] as const;
    const expected = coordinate[0] + pagesX * (coordinate[1] + pagesY * coordinate[2]);
    assert.equal(key, expected, "page metadata must directly address its packed source octant");
    assert.ok(key > previousKey, "compact metadata must remain sorted for consumer lookup");
    previousKey = key;
  }
});

test("Sparse CM12 presentation publication caches coarse page stencils", () => {
  assert.match(webgpuSparseCM12ResidentWGSL,
    /var<workgroup>presentationDensityCache:array<f32,64>/,
    "coarse reconstruction must share its bounded page stencil across output lanes");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /if\(ownerScale>=u32\(cellScale\)\)[\s\S]*return state\[destinationDensity\(\)\+owner\.x\]/,
    "an already-coarse authority cell must not be resolved once per finest child");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /if\(scale==1u\)[\s\S]*let cell=range\.x\+localCell\.x/,
    "fine pages must use the source brick encoded in metadata directly");
  const publication = webgpuSparseCM12ResidentWGSL.slice(
    webgpuSparseCM12ResidentWGSL.indexOf("fn publishSparseLevelSet"),
  );
  assert.doesNotMatch(publication, /interpolatedPresentationPhi/,
    "page publication must consume its shared stencil instead of rebuilding every corner");
  const page8 = createWebgpuSparseCM12ResidentWGSL(16, 8);
  const page16 = createWebgpuSparseCM12ResidentWGSL(16, 16);
  assert.match(page8, /const PRESENTATION_PAGE_RESOLUTION:u32=8u/);
  assert.match(page8, /var<workgroup>presentationDensityCache:array<f32,216>/);
  assert.match(page16, /const PRESENTATION_SAMPLES_PER_PAGE:u32=4096u/);
  assert.match(page16, /var<workgroup>presentationDensityCache:array<f32,1000>/);
  assert.match(page16,
    /for\(var localIndex=lane;localIndex<PRESENTATION_SAMPLES_PER_PAGE;localIndex\+=64u\)/,
    "wide pages must be cooperatively published rather than widening the workgroup");
  assert.throws(() => createWebgpuSparseCM12ResidentWGSL(8, 16), /does not divide/);
});

test("Sparse CM12 scientific overlays consume accepted compact state directly", () => {
  const overlay = readFileSync(new URL("../lib/core/webgpu-grid-overlay.ts", import.meta.url),
    "utf8");
  const resident = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const renderer = readFileSync(new URL("../lib/core/webgpu-renderer.ts", import.meta.url),
    "utf8");
  const residentActivityStride = resident.match(/const ACTIVITY_RECORD_WORDS = (\d+);/)?.[1];
  const overlayActivityStride = overlay.match(
    /const SPARSE_ACTIVITY_RECORD_WORDS:u32=(\d+)u;/,
  )?.[1];
  assert.ok(residentActivityStride, "the sparse resident must declare its activity stride");
  assert.equal(overlayActivityStride, residentActivityStride,
    "the scientific overlay must walk the live sparse activity ABI without per-brick drift");
  assert.doesNotMatch(overlay, /sparseTopologyArena\[6u\]\+16u\*cell/,
    "the overlay must not decode the retired 16-word sparse cell record");
  assert.match(overlay,
    /scale=max\(1u,sparseBrickFineResolution\(\)\*sparseBrickSpan\(owner\.y\)\/resolution\)/,
    "represented sparse cells must derive their isotropic size from the live brick rung");
  assert.match(overlay, /@binding\(11\) var<storage,read> sparseTopology/);
  assert.match(overlay, /fn sparseBrickLookup\(key:u32\)->u32/);
  assert.match(overlay, /if\(sparseGridEnabled\(\)\)\{return sparseDensityAt/,
    "surface density must come from the live compact resident state");
  assert.match(overlay,
    /select\(sparseP\.stateOffsets0\.x,sparseP\.stateOffsets0\.y,sparseP\.frame\.w>0\.5\)/,
    "density visualization must select the solver's accepted ping-pong bank");
  assert.match(overlay,
    /select\(sparseP\.stateOffsets1\.x,sparseP\.stateOffsets1\.y,sparseP\.frame\.w>0\.5\)/,
    "velocity visualization must select the solver's accepted ping-pong bank");
  assert.match(overlay, /if\(sparseGridEnabled\(\)&&sparseOwner\(cell\)\.x==SPARSE_INVALID\)/,
    "unrepresented world space must remain transparent");
  assert.match(renderer, /setSparseSource\(this\.gpuFluid\.sparseAdaptiveGridSource\)/);
  assert.doesNotMatch(renderer, /readDiagnosticFields/,
    "rendering must never trigger dense QA materialization");
});

test("analytic initial volumes do not trigger an empty-domain brick scan", () => {
  const base = createSparseCM12LongDamBreakScene();
  const seed = base.fluid.initialBrickSeeds_m?.[0] ?? { x: 0, y: 0, z: 0 };
  const h = base.container.width_m / 1_000_000;
  const scene = { ...base, fluid: { ...base.fluid,
    initialBrickSeeds_m: [seed],
    initialBrickSeedsAdditive: false,
    initialLiquidVolumes: [{ shape: "box" as const,
      min_m: { x: seed.x - h, y: seed.y, z: seed.z - h },
      max_m: { x: seed.x + h, y: seed.y + h, z: seed.z + h } }],
  } };
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [1_000_000, 32, 32],
  });
  assert.ok(atlas.bricks.length > 0);
  assert.ok(atlas.bricks.length < 16,
    "local authored sources must not retain bricks from the empty million-cell corridor");
});

test("Sparse CM12 publication stays page-shaped with a leaf-scaled hashed locator", () => {
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL, /texture_storage_3d/);
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL, /fn publishPresentation/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn brickDirectoryLookup\(key:u32\)[\s\S]*originKey\*0x9e3779b1u/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /maximumSpanLog=topology\[p\.topologyOffsets2\.w\+1u\]&31u/);
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL,
    /fn brickDirectoryLookup\(key:u32\)[\s\S]*var low=0u;var high=p\.dispatch\.w/,
    "hot characteristic samples must not binary-search all resident leaves");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn publishSparseLevelSet[\s\S]*pageCount=arrayLength\(&fineMetadata\)\/4u/);
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.doesNotMatch(residentSource, /denseCount|brickDirectoryOffset|ownerOffset/);
  assert.doesNotMatch(residentSource, /const topologyArenaWords = new Uint32Array/,
    "startup must not concatenate its largest GPU uploads on the host");
  assert.match(residentSource, /function compactPressureTopology\(/,
    "pressure aliasing must copy only the CSR edge tail");
  assert.doesNotMatch(residentSource, /dispatch\("publishPresentation"/);
  assert.match(residentSource,
    /dispatch\("publishSparseLevelSet",\s*this\.globalFineLevelSetSource\.plan\.maximumResidentBricks\)/);
  const solverSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  assert.match(solverSource, /new WebGPUAdaptiveMassSparsePresentation\(device\)/);
  assert.doesNotMatch(solverSource, /new WebGPUAdaptiveMassAtlasPresentation\(/,
    "the sparse solver must not construct full-domain presentation textures");
  assert.match(solverSource, /get globalFineLevelSetSource\(\)/,
    "compact pages must be the renderer's surface authority");
});

test("canonical Sparse CM12 dam defines a tall sparse 192x96x32 traversal", () => {
  const scene = createSparseCM12LongDamBreakScene();
  assert.deepEqual(sceneLatticeDimensions(scene), [192, 96, 32]);
  assert.equal(SPARSE_CM12_LONG_DAM_METHOD_PROFILE.methodId, "adaptive-mass");
  assert.equal(SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides?.timeStep, "paper");
  assert.equal(SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides?.selectorMode, undefined,
    "scene profiles must not override the product's Surface distance default");
  assert.deepEqual([
    SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides?.finestTravelCells,
    SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides?.fourTravelCells,
    SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides?.twoTravelCells,
  ], [4, 2, 1]);
  const dam = sceneDamBreakBox(scene);
  assert.equal(dam.min.x, 0);
  assert.ok(Math.abs(dam.max.x - 1 / 6) < 1e-12);
  assert.ok(Math.abs(dam.max.y - 5 / 12) < 1e-12);
  assert.equal(dam.max.z, 1);
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [192, 96, 32],
    resolutionForBrick: () => 4,
  });
  const stats = sparseBrickAtlasStats(atlas);
  assert.equal(stats.logicalBrickCount, 1_152);
  assert.equal(stats.residentBrickCount, 80);
  assert.equal(stats.omittedEmptyBrickCount, 1_072);
  assert.equal(atlas.brickDimensions[0] - 4, 20,
    "the front must cross twenty initially dry brick columns");
});

test("receiver capacity scaling depends only on physical detail ratio", () => {
  const scale = (nominal_m: number, finest_m: number) =>
    adaptiveMassReceiverScaleForScene({
      nominalResolution: { length_m: nominal_m },
      voxelDomain: { finestCellSize_m: finest_m, brickSize_cells: 8 },
    });
  assert.deepEqual(scale(0.04, 0.04), {
    supportRings: 9, minimumCapacityScale: 1,
  });
  assert.deepEqual(scale(0.04, 0.01), {
    supportRings: 39, minimumCapacityScale: 64,
  });
  assert.deepEqual(scale(0.12, 0.03), scale(0.04, 0.01),
    "WORLD scaling must not change the receiver pool in lattice coordinates");
});

test("the authored DETAIL x2 long dam preserves its physical receiver reach", () => {
  const scene = createSparseCM12LongDamBreakScene();
  const dimensions = sceneLatticeDimensions(scene);
  assert.deepEqual(dimensions, [192, 96, 32]);
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
  });
  const supported = residentSupportAtlas(initial, "adaptive");
  const scale = adaptiveMassReceiverScaleForScene(scene);
  assert.deepEqual(scale, { supportRings: 19, minimumCapacityScale: 8 });
  const domain = dormantReceiverDomain(
    supported, "adaptive", scale.supportRings, "auto", scale.minimumCapacityScale,
  );
  assert.equal(Math.max(...domain.bricks.map((brick) => brick.coordinate[0])), 23,
    "the doubled-detail front must retain receiver capacity through the far wall");
  assert.equal(sparseBrickAtlasStats(domain).residentBrickCount, 24 * 12 * 4,
    "this finite narrow tank fits inside the physically scaled receiver apron");
});

test("mini32 performance accepts a Rung-A fine set without freezing the old one-seed count", () => {
  const arm = (methodId: "uniform" | "adaptive-mass"): SparseCM12BenchmarkArm => ({
    methodId,
    sceneId: "minimal-power-dam-break-32",
    finestDimensions: [32, 32, 32],
    dt_s: 0.004,
    constructionExcluded: true,
    endToEndFrame_ms: new Array(30).fill(methodId === "uniform" ? 10 : 9),
    cpuTraces: [],
    gpuTraces: [],
    initialTopology: methodId === "adaptive-mass" ? {
      fineBricks: 28,
      coarseBricks: 36,
      fineCoarseFaceConnectedPairs: 24,
      mixedSeamRows: 384,
    } : undefined,
    evolvedTopology: methodId === "adaptive-mass" ? [{
      fineBricks: 28,
      coarseBricks: 36,
      fineCoarseFaceConnectedPairs: 24,
      mixedSeamRows: 384,
    }] : undefined,
  });
  const verdict = evaluateSparseCM12Performance(
    arm("uniform"), arm("adaptive-mass"),
    SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  );
  assert.equal(verdict.passed, true, verdict.failures.join("\n"));
});
