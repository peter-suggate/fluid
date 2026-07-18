import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { resolveMethodValues, simulationMethods } from "../lib/methods";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { nextQuadtreeIterationBudget, quadtreeChebyshevPasses, quadtreeChebyshevSpectrum, quadtreeDispatchShader, quadtreeDivergenceShader, quadtreeIterationBudget, quadtreeMegakernelDofLimit, quadtreeMegakernelPreferred, quadtreeMegakernelRowIterationLimit, quadtreeMultigridShader, quadtreeTallCellProjectionShader, quadtreeVelocityClampShader, quadtreeVelocityExtrapolationShader, WebGPUQuadtreeTallCellProjection } from "../lib/webgpu-quadtree-tall-cell";
import { nextQuadtreeVofReconciliationActive, packedQuadtreeRootMap, quadtreeConstructionShader, quadtreeSurfaceJumpSequence, quadtreeSurfaceShader, quadtreeVofReconciliationFraction, WebGPUQuadtreeBuilder, WebGPUQuadtreeSurfaceState } from "../lib/webgpu-quadtree-builder";
import { quadtreeSegmentationPackShader, WebGPUQuadtreePackBuilder } from "../lib/webgpu-quadtree-pack-builder";
import { buildAdaptiveOpticalLayerField, buildQuadtree, buildVariationalSystem, populateTallPressureGrid } from "../lib/quadtree-tall-cell-grid";
import { capillaryStableDt_s, proactiveQuadtreeSubsteps, quadtreeMissedFrames, quadtreeRebuildRetryDelay } from "../lib/webgpu-uniform-eulerian";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";

test("the old optical-layer method is replaced by quadtree tall cells", () => {
  assert.ok(simulationMethods.includes(quadtreeTallCellMethod));
  assert.ok(!simulationMethods.some((method) => method.id === "adaptive-optical-layer"));
  assert.equal(quadtreeTallCellMethod.shortLabel, "Adaptive");
  assert.match(quadtreeTallCellMethod.detail, /T-junction/);
  assert.equal(quadtreeTallCellMethod.presetFor("balanced").preconditioner, "poly", "the parallel polynomial preconditioner is the runtime default");
  assert.equal(quadtreeTallCellMethod.presetFor("balanced").pressureSolver, "pcg", "the tolerance-driven solve is the runtime default");
  assert.equal(quadtreeTallCellMethod.presetFor("balanced").opticalLayerMode, "adaptive-motion", "the product preset exercises the 2026 adaptive layer while the fixed path remains selectable");
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "opticalLayerMode" && param.default === "adaptive-motion"));
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "opticalAlpha" && param.default === 0.5));
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "preconditioner" && param.default === "poly"));
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "preconditioner" && param.kind === "select" && param.options.some((option) => option.value === "mg")), "geometric MG is selectable without replacing the measured default");
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "vofReconciliation" && param.default === "on"), "catastrophic-loss recovery is armed by default but inactive during healthy phi transport");
});

test("quadtree Chebyshev pressure is bounded, row parallel, and opt-in", () => {
  assert.equal(quadtreeChebyshevPasses(96), 96);
  assert.equal(quadtreeChebyshevPasses(160), 160);
  assert.equal(quadtreeChebyshevPasses(240), 240);
  for (const entry of ["initializeChebyshev", "iterateChebyshevAB", "iterateChebyshevBA", "finishChebyshevFromPressure", "finishChebyshevFromBest", "reduceChebyshevResidual"]) {
    assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  }
  const polynomial = quadtreeTallCellProjectionShader.slice(
    quadtreeTallCellProjectionShader.indexOf("fn initializeChebyshevRow"),
    quadtreeTallCellProjectionShader.indexOf("fn initialize(")
  );
  assert.match(polynomial, /pressureProduct\(row, source\)/, "each pass is a cached sparse row product");
  assert.match(polynomial, /let lower = 0\.005; let upper = 4\.2/);
  assert.doesNotMatch(polynomial, /workgroupBarrier|atomic|coupleReduce|coupleApply/, "the hot polynomial pass has no global scalar dependency");

  // The live dam-break matrix reaches lambda_max ~= 3.88 after diagonal
  // scaling. Exercise that edge directly so an octree-sized upper bound can
  // never be copied back into this solver unnoticed.
  const eigenvalue = 3.88;
  const { lower, upper } = quadtreeChebyshevSpectrum;
  const theta = 0.5 * (upper + lower), delta = 0.5 * (upper - lower), sigma = theta / delta;
  let pressure = 0, previousSearch = 0, previousRho = 0;
  for (let pass = 0; pass < quadtreeChebyshevPasses(96); pass += 1) {
    const residual = 1 - eigenvalue * pressure;
    const rho = previousRho > 0 ? 1 / (2 * sigma - previousRho) : 1 / sigma;
    const search = previousRho > 0
      ? rho * previousRho * previousSearch + (2 * rho / delta) * residual
      : residual / theta;
    pressure += search;
    previousSearch = search;
    previousRho = rho;
  }
  assert.ok(Math.abs(1 - eigenvalue * pressure) < 1e-2, "the corrected polynomial damps the measured high-frequency mode by at least two orders of magnitude");

  const encode = WebGPUQuadtreeTallCellProjection.prototype.encode.toString();
  assert.match(encode, /exactCoupled=coupled&&!chebyshev/, "accelerated rigid solves omit the exact low-rank response");
  assert.match(encode, /if\(exactCoupled\)\{indirect\("coupleReduce",12\);indirect\("coupleApply",24\)\}/, "PCG retains exact same-step coupling as the reference");
  assert.match(encode, /if\(coupled\)direct\("coupleImpulse",1\)/, "both paths publish the current pressure impulse");

  const defaults = resolveMethodValues(quadtreeTallCellMethod, "balanced", {});
  assert.equal(defaults.pressureSolver, "pcg");
  assert.equal(resolveMethodValues(quadtreeTallCellMethod, "balanced", { pressureSolver: "chebyshev" }).pressureSolver, "chebyshev");
});

test("quadtree projection preserves and extrapolates near-surface air velocity", () => {
  assert.match(quadtreeTallCellProjectionShader, /ownPhi > 5\.0 \* h && otherPhi > 5\.0 \* h/, "only air beyond the aligned five-cell band is zeroed");
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /volumeIn|loadVolume/, "the adaptive pressure projection must have no VOF binding");
  const project = quadtreeTallCellProjectionShader.slice(quadtreeTallCellProjectionShader.indexOf("fn project"));
  assert.doesNotMatch(project, /volumeIn|Alpha|alpha/, "projection classification must remain level-set authoritative");
  assert.match(quadtreeVelocityExtrapolationShader, /fn extrapolateVelocity/);
  assert.match(quadtreeVelocityExtrapolationShader, /phi\(q\) < 0\.0 \|\| phi\(plus\) < 0\.0/, "known faces touch phi-negative liquid");
  assert.match(quadtreeVelocityExtrapolationShader, /min\(ownPhi, otherPhi\) < 5\.0 \* h/, "extrapolation reaches the aligned five-cell surface band");
});

test("quadtree fine-grid dynamics use the resident level set as fluid support", () => {
  assert.match(legacyUniformComputeShader, /@group\(0\) @binding\(20\) var surfaceIn/);
  assert.match(legacyUniformComputeShader, /fn levelSetAuthority\(\) -> bool/);
  assert.match(legacyUniformComputeShader, /fn liquid\(p:vec3i\)->bool\{return surfaceLiquid\(p\);\}/, "gravity and velocity-force wetness use the authoritative surface");
  assert.match(legacyUniformComputeShader, /fn transportVelocity[\s\S]*surfaceOccupancy\(id\)/, "velocity extension uses phi support");
  assert.match(legacyUniformComputeShader, /fn buildOccupancy[\s\S]*surfaceOccupancy/, "air-work culling uses phi support");
  assert.match(legacyUniformComputeShader, /let cellMass=params\.physical\.x\*h\.x\*h\.y\*h\.z\*wetFraction/, "rigid coupling uses phi-represented liquid mass");
  assert.match(legacyUniformComputeShader, /if\(surfaceLiquid\(id\)\)\{atomicMax\(&reductions\[1\]/, "front diagnostics follow the rendered level set");
});

test("quadtree publishes a post-projection divergence diagnostic field", () => {
  assert.match(quadtreeDivergenceShader, /fn computeDivergence/);
  assert.match(quadtreeDivergenceShader, /textureStore\(divergenceOut, q, vec4f\(value\)\)/);
  assert.match(quadtreeDivergenceShader, /component\(q \+ vec3i\(1, 0, 0\), 0u\) - component\(q, 0u\)/);
});

test("quadtree CFL subdivisions use the current conservative velocity bound", () => {
  assert.equal(proactiveQuadtreeSubsteps(0, 0, 9.81, 0.01, 0.02), 1);
  assert.equal(proactiveQuadtreeSubsteps(3, 0, 9.81, 0.01, 0.02), 2, "the previous projected maximum controls this frame");
  assert.equal(proactiveQuadtreeSubsteps(0, 5, 0, 0.01, 0.02), 3, "a faster inflow participates before its first reduction");
  assert.equal(proactiveQuadtreeSubsteps(100, 0, 0, 0.01, 0.02), 50, "the safety ceiling rises past the old eight-step CFL limit");
  assert.equal(proactiveQuadtreeSubsteps(1, 0, 0, 0, 0), 1, "degenerate startup inputs remain safe");
});

test("adaptive substeps also enforce the finest-cell capillary-wave bound", () => {
  assert.equal(capillaryStableDt_s(1_000, 0, 0.001), Number.POSITIVE_INFINITY, "zero surface tension has no capillary restriction");
  const coarse = capillaryStableDt_s(1_000, 0.072, 0.004);
  const fine = capillaryStableDt_s(1_000, 0.072, 0.001);
  assert.ok(fine < coarse);
  assert.ok(Math.abs(coarse / fine - 8) < 1e-10, "the bound scales with h^(3/2)");
  assert.equal(proactiveQuadtreeSubsteps(0, 0, 0, 0.02, 0.001, 64, 1_000, 0.072), Math.ceil(0.02 / fine));
  assert.equal(proactiveQuadtreeSubsteps(0, 0, 0, 0.02, 0.001, 64, 1_000, 0), 1, "sigma=0 preserves the previous CFL-only result");
});

test("quadtree blocked-frame telemetry counts missed presentation budgets, not retry polls", () => {
  assert.equal(quadtreeMissedFrames(0), 0);
  assert.equal(quadtreeMissedFrames(16), 0);
  assert.equal(quadtreeMissedFrames(17), 1);
  assert.equal(quadtreeMissedFrames(34), 2);
  assert.deepEqual([0, 1, 2, 3, 6, 20].map(quadtreeRebuildRetryDelay), [0, 2, 4, 8, 60, 60], "failed rebuilds back off while the previous topology remains usable");
});

test("VOF reconciliation is an armed catastrophic-loss circuit breaker", () => {
  assert.equal(nextQuadtreeVofReconciliationActive(false, -0.099), false);
  assert.equal(nextQuadtreeVofReconciliationActive(false, -0.101), true);
  assert.equal(nextQuadtreeVofReconciliationActive(true, -0.021), true);
  assert.equal(nextQuadtreeVofReconciliationActive(true, -0.019), false);
  assert.equal(nextQuadtreeVofReconciliationActive(false, Number.NaN), false);
  assert.equal(quadtreeVofReconciliationFraction(0, 100), 0);
  assert.equal(quadtreeVofReconciliationFraction(100, 800), 1 / 64);
  assert.equal(quadtreeVofReconciliationFraction(1000, 100), 1 / 32);
});

test("quadtree updates evaluate sizing, adaptive optical depth, subdivide, and smooth on WebGPU", () => {
  for (const entry of ["evaluateSizing", "evaluateOpticalLayer", "dilateOpticalLayerX", "dilateOpticalLayerZ", "smoothOpticalLayer", "refine", "smoothTopology", "sampleLeafProfiles"]) assert.match(quadtreeConstructionShader, new RegExp(`fn ${entry}\\b`));
  assert.match(quadtreeConstructionShader, /for \(var y = 0u; y < params\.dims\.y; y \+= 1u\)/, "sizing must vertically reduce each column on the GPU");
  assert.match(quadtreeConstructionShader, /sizingField\[index2\(q\)\] = maximum/);
  assert.match(quadtreeConstructionShader, /neighborTooFine/);
  assert.match(quadtreeConstructionShader, /demand > 1\.0 \/ testedWidth/);
  const roots = packedQuadtreeRootMap(16, 8, 8);
  assert.equal(roots.length, 128);
  assert.equal((roots[0] >>> 20) & 1023, 8);
  assert.equal((roots[15] >>> 20) & 1023, 8);
  assert.match(quadtreeConstructionShader, /if \(!nearSurface && !wet\) \{ continue; \}/, "deep liquid remains eligible for velocity-variation sizing");
  assert.doesNotMatch(quadtreeConstructionShader, /volumeIn|loadVolume/, "adaptive topology must not read VOF");
  assert.match(quadtreeSegmentationPackShader, /fn footprintWet/);
  assert.match(quadtreeSegmentationPackShader, /fn adaptiveOpticalFirst/);
  assert.match(quadtreeSegmentationPackShader, /params\.optical\.x != 0u/, "the same resident pack supports fixed and motion-adaptive A\/B modes");
  assert.match(quadtreeSegmentationPackShader, /let liquid = footprintWet\(word, y\)/, "GPU sparse packing must use footprint wetness for coarse DOFs");
});

test("quadtree topology kernels elect one invocation per leaf", () => {
  const splitWriter = quadtreeConstructionShader.slice(quadtreeConstructionShader.indexOf("fn writeSplitLeaf"), quadtreeConstructionShader.indexOf("fn refine"));
  assert.match(splitWriter, /topologyOut\[index2\(q\)\] = childForCell\(origin, size, q\)/, "a split must still materialize every fine-cell owner");

  const refine = quadtreeConstructionShader.slice(quadtreeConstructionShader.indexOf("fn refine"), quadtreeConstructionShader.indexOf("fn neighborTooFine"));
  assert.match(refine, /if \(any\(q != origin\)\) \{ return; \}[\s\S]*var demand = 0\.0/, "only the leaf origin may reduce its sizing footprint");
  assert.match(refine, /if \(demand > 1\.0 \/ testedWidth\) \{ writeSplitLeaf\(origin, size\); \}/);
  assert.doesNotMatch(refine, /topologyOut\[index\]/, "unchanged cells come from the bulk owner-map copy");

  const smooth = quadtreeConstructionShader.slice(quadtreeConstructionShader.indexOf("fn smoothTopology"), quadtreeConstructionShader.indexOf("fn sampleLeafProfiles"));
  assert.match(smooth, /if \(any\(q != origin\)\) \{ return; \}[\s\S]*neighborTooFine/, "only the leaf origin may scan its boundary");
  assert.match(smooth, /writeSplitLeaf\(origin, size\)/);
  assert.doesNotMatch(smooth, /topologyOut\[index\]/);

  assert.match(WebGPUQuadtreeBuilder.prototype.encodeConstruction.toString(), /copyBufferToBuffer/, "each sparse topology kernel must start from a complete copied owner map");
});

test("resident phi uses bounded-MacCormack transport with per-step sub-cell redistance", () => {
  for (const entry of ["advectLevelSet", "advectPredict", "advectReverse", "advectCorrect", "reduceVolume", "seedDistance", "jumpFlood", "finalizeDistance"]) assert.match(quadtreeSurfaceShader, new RegExp(`fn ${entry}\\b`));
  assert.match(quadtreeSurfaceShader, /fn centredMacVelocity/);
  assert.match(quadtreeSurfaceShader, /fn departurePoint/);
  assert.match(quadtreeSurfaceShader, /let midpoint = p - 0\.5 \* first \* dt \* cellsPerMetre/, "phi trace must be RK2 midpoint");
  assert.match(quadtreeSurfaceShader, /predicted \+ 0\.5 \* \(original - reversed\)/, "BFECC correction");
  assert.match(quadtreeSurfaceShader, /if \(corrected < lower \|\| corrected > upper\) \{ corrected = predicted; \}/, "bounded MacCormack fallback");
  assert.match(quadtreeSurfaceShader, /own\.x \+ loadVelocity\(q - vec3i\(1, 0, 0\)\)\.x/);
  assert.match(quadtreeSurfaceShader, /fn packSeedPoint/, "seeds must carry projected interface points, not cell centres");
  assert.match(quadtreeSurfaceShader, /0\.87 \* hMin\(\)/);
  assert.match(quadtreeSurfaceShader, /if \(abs\(advected\) >= 2\.5 \* h \|\| interfaceDistance >= 2\.5 \* h\)/, "the narrow band is bounded by the true interface distance, not the advected magnitude, so swept fossils are repaired");
  assert.doesNotMatch(quadtreeSurfaceShader, /sqrt\(seedDistanceSquared\(gid, word\)\) \+ 0\.5 \* h\b/, "the half-cell redistance floor must be gone");
  assert.match(quadtreeSurfaceShader, /fn volumeCorrectedPhi/);
  assert.match(quadtreeSurfaceShader, /value - params\.control\.x \* h \* params\.cellAndDt\.w/, "volume correction is a normal displacement of phi's interface");
  assert.match(quadtreeSurfaceShader, /abs\(value\) < 1\.5 \* h/);
  assert.match(quadtreeSurfaceShader, /value \/ \(4\.0 \* params\.cellAndDt\.y\)/);
  assert.match(quadtreeSurfaceShader, /atomicAdd\(&reductions\[0\]/);
  assert.match(quadtreeSurfaceShader, /isInflowVelocityCell/, "the nozzle must source fluid into the resident level set");
  // Conservative VOF is isolated behind the catastrophic-loss control. It
  // may restore missing liquid, but cannot delete phi topology or filter the
  // phi-derived redistance seeds during ordinary transport.
  assert.match(quadtreeSurfaceShader, /reconcileVolumeIn/);
  assert.match(quadtreeSurfaceShader, /params\.control\.y > 0\.5 && wet/);
  assert.match(quadtreeSurfaceShader, /let signMismatch = \(result < 0\.0\) != wet/);
  assert.match(quadtreeSurfaceShader, /let decisiveMismatch = signMismatch && abs\(result\) > 0\.5 \* h/);
  assert.match(quadtreeSurfaceShader, /\(0\.5 - alpha\) \* \(4\.0 \* params\.cellAndDt\.y\)/, "VOF repairs preserve the conservative sub-cell amount instead of stamping half-cell signs");
  assert.match(quadtreeSurfaceShader, /let wet = alpha >= 0\.5/, "emergency restoration requires majority VOF occupancy");
  const emergencyRepair = quadtreeSurfaceShader.slice(quadtreeSurfaceShader.indexOf("// VOF is not part"), quadtreeSurfaceShader.indexOf("fn cullDebris"));
  assert.doesNotMatch(emergencyRepair, /confidentlyDry|params\.control\.y > 0\.5 && !wet/, "VOF must never erase phi-wet topology");
  assert.match(quadtreeSurfaceShader, /atomicAdd\(&reductions\[3\], 1u\)/, "VOF disagreement remains diagnostic outside emergency recovery");
  assert.doesNotMatch(quadtreeSurfaceShader, /\bvolumeIn\b|\bloadVolume\b/);
  assert.match(quadtreeSurfaceShader, /fn cullDebris/);
  assert.match(quadtreeSurfaceShader, /params\.control\.z > 0\.5/, "debris hygiene stays explicitly gated");
  assert.match(quadtreeSurfaceShader, /textureLoad\(reconcileVolumeIn, q, 0\)\.x < 0\.5/);
  const seeds = quadtreeSurfaceShader.slice(quadtreeSurfaceShader.indexOf("fn seedDistance"), quadtreeSurfaceShader.indexOf("fn seedDistanceSquared"));
  assert.doesNotMatch(seeds, /reconcileVolumeIn|Alpha|alpha/, "redistance seeds must be derived only from phi sign crossings");
  assert.match(quadtreeConstructionShader, /fn effectiveWet[\s\S]*return loadAdvancedPhi\(clamp3\(q\)\) < 0\.0/);
  assert.match(quadtreeConstructionShader, /let profile = 3u \* \(leaf \* params\.dims\.y \+ y\)/);
  assert.match(quadtreeConstructionShader, /columnProfiles\[profile \+ 1u\] = minimum/);
  assert.match(quadtreeConstructionShader, /columnProfiles\[profile \+ 2u\] = maximum/);
});

test("resident phi redistance is bounded to the consumed five-cell band", () => {
  assert.deepEqual(quadtreeSurfaceJumpSequence(64, 32, 48), [4, 2, 1, 1], "large domains use capped JFA+1 instead of domain-scale jumps");
  assert.deepEqual(quadtreeSurfaceJumpSequence(4, 3, 2), [4, 2, 1], "tiny domains retain the full schedule");
  assert.ok(4 + 2 + 1 >= Math.ceil(5 + 0.87), "the capped schedule reaches every projected seed that can affect the 5h band");

  const finalize = quadtreeSurfaceShader.slice(quadtreeSurfaceShader.indexOf("fn finalizeDistance"), quadtreeSurfaceShader.indexOf("fn cullDebris"));
  assert.match(finalize, /distance = min\(5\.0 \* h, max\(2\.5 \* h, interfaceDistance\)\)/, "far-field magnitudes remain capped at 5h");
  assert.match(finalize, /result = select\(distance, -distance, advected < 0\.0\)/, "far-field signs do not depend on a global seed");
  assert.match(finalize, /if \(params\.control\.z <= 0\.5\) \{ accumulateVolume\(result, gid\); \}/, "default diagnostics are fused into finalization");

  const encode = WebGPUQuadtreeSurfaceState.prototype.encode.toString();
  assert.match(encode, /surfacePass[\s\S]*advectPredict[\s\S]*jumpFlood[\s\S]*finalizeDistance[\s\S]*surfacePass\.end/, "dependent surface stages share one ordered compute pass");
  assert.match(encode, /if\s*\(this\.debrisCulling\)[\s\S]*reduceVolume/, "the post-cull path retains an exact reduction");
});

test("pressure iterations consume precomputed face masks, fluxes, and row activity", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn refreshRows/);
  assert.match(quadtreeTallCellProjectionShader, /liquidMask \|= 1u << slot/);
  assert.match(quadtreeTallCellProjectionShader, /face\.flux = face\.weights\.x \* faceVelocity\(face\) \+ face\.solidFlux/);
  assert.match(quadtreeTallCellProjectionShader, /fn dofActive\(row: u32\) -> bool \{ return state\[stateIndex\(row, ACTIVE_FLAG\)\] != 0u; \}/);
  assert.match(quadtreeTallCellProjectionShader, /rhs \+= item\.coefficient \* face\.flux/);
  const iterationPath = quadtreeTallCellProjectionShader.slice(quadtreeTallCellProjectionShader.indexOf("fn rowProduct"), quadtreeTallCellProjectionShader.indexOf("fn cellIndex"));
  assert.doesNotMatch(iterationPath, /faceSamplePhi|faceVelocity/);
  assert.match(iterationPath, /matrixCoefficient\(entry\) \* stateF\(matrixNode\(entry\), DIRECTION\)/);
});

test("projection write-back retains fine sub-face shear while preserving coarse flux", () => {
  const project = quadtreeTallCellProjectionShader.slice(quadtreeTallCellProjectionShader.indexOf("fn project"));
  assert.match(project, /gradient = gradient - face\.mlsMean \+ solved/);
  assert.doesNotMatch(project, /value\[axis\] = \(face\.flux - face\.solidFlux\) \/ face\.weights\.x/, "projection must not box-filter each fine sample to the adaptive face mean");
  assert.match(quadtreeTallCellProjectionShader, /face\.flux = face\.weights\.x \* faceVelocity\(face\) \+ face\.solidFlux/, "the coarse constraint is integrated from the same fine samples");
});

test("WebGPU pressure path is variational PCG rather than Jacobi pressure smoothing", () => {
  for (const entry of ["initialize", "multiply", "applyStep", "finishIteration", "project"]) {
    assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(quadtreeDispatchShader, /fn updateDispatch\b/);
  assert.match(quadtreeTallCellProjectionShader, /rowProduct/);
  assert.match(quadtreeTallCellProjectionShader, /faceGradient/);
  assert.match(quadtreeTallCellProjectionShader, /matrixBaseCoefficient\(entry\) \* face\.weights\.y/);
  assert.match(quadtreeTallCellProjectionShader, /alias SolverField = u32/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionBlockIC/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionJacobi/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionLine/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionPolynomialStart/);
  for (const entry of ["applyStepPartial", "applyStepFinalize", "applyStepUpdate", "finishIterationPartial", "finishIterationFinalize", "finishIterationUpdate"]) assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /atomic(Add|Max|Min)/);
});

test("geometric MG assembles Galerkin levels and applies a symmetric V-cycle", () => {
  for (const entry of ["clearCoarseMatrix", "assembleGalerkin", "lineSmoothInitial", "restrictDefectGather", "solveCoarsest", "prolongateCorrection", "lineSmoothFinal"]) {
    assert.match(quadtreeMultigridShader, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(quadtreeMultigridShader, /entryParent\(entry\)/, "symbolic fine entries map directly into Galerkin coarse CSR");
  assert.match(quadtreeMultigridShader, /sourceF\(RHS, row\) - sourceProduct\(row\)/, "restriction consumes the post-smoothing defect");
  assert.match(quadtreeMultigridShader, /for \(var iteration = 0u; iteration < 8u/, "the coarse polynomial is fixed and therefore linear inside PCG");
});

test("parallel PCG fuses row-local setup and polynomial iteration stages", () => {
  for (const entry of [
    "initializeJacobiDirection",
    "initializePolynomialStart",
    "preconditionPolynomialUpdateDirection",
    "applyStepUpdateJacobi",
    "applyStepUpdatePolynomialStart",
    "preconditionPolynomialUpdateFinishPartial",
  ]) assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));

  const encode = WebGPUQuadtreeTallCellProjection.prototype.encode.toString();
  assert.match(encode, /initializeJacobiDirection/);
  assert.match(encode, /initializePolynomialStart/);
  assert.match(encode, /applyStepUpdatePolynomialStart/);
  assert.match(encode, /preconditionPolynomialUpdateFinishPartial/);

  const fusedFinish = quadtreeTallCellProjectionShader.slice(
    quadtreeTallCellProjectionShader.indexOf("fn preconditionPolynomialUpdateFinishPartial"),
    quadtreeTallCellProjectionShader.indexOf("fn finishIterationFinalize"),
  );
  assert.match(fusedFinish, /addStateF\(row, PRECONDITIONED/);
  assert.match(fusedFinish, /reducePartial\(lid\.x, rz, rr\)/, "the final polynomial update must feed the PCG reduction in the same dispatch");

  assert.match(quadtreeTallCellProjectionShader, /fn finishIterationFinalize\b/);
  assert.match(quadtreeTallCellProjectionShader, /publishNextDispatches\(keepSolving\)/, "the residual finalizer must publish the next iteration without another control dispatch");
  assert.match(quadtreeDispatchShader, /scalars\[10\] = select\(0\.0, 1\.0, keepSolving\)/, "the initial control dispatch must seed the fused finalizer's active flag");
  assert.match(encode, /first===0/);
  assert.match(encode, /finishIterationFinalize/);
  assert.match(encode, /finishIterationUpdate",124/, "the current iteration's direction update needs its independent indirect triple");
});

test("persistent PCG megakernel is barrier-uniform and isolated from ladder control state", () => {
  const start = quadtreeTallCellProjectionShader.indexOf("fn megakernelPreconditionedProduct");
  const end = quadtreeTallCellProjectionShader.indexOf("fn cellIndex", start);
  assert.ok(start >= 0 && end > start, "the megakernel source region must be present");
  const megakernel = quadtreeTallCellProjectionShader.slice(start, end);
  assert.match(megakernel, /fn solveMegakernel\b/);
  assert.match(megakernel, /@compute @workgroup_size\(256\)/);
  assert.match(megakernel, /workgroupUniformLoad\(&megakernelConverged\)/, "barrier-containing loop exits must be workgroup-uniform");
  assert.match(megakernel, /params\.couplingCounts\.z >> 1u/, "the hard iteration cap is uniform and independent of ladder feedback");
  assert.doesNotMatch(megakernel, /dispatchArgs|controlWord|setControlWord|publishNextDispatches|factorColumns/, "the megakernel must not read or mutate ladder dispatch controls");
  assert.match(quadtreeTallCellProjectionShader, /fn warmStartActive\(\) -> bool \{ return \(params\.couplingCounts\.z & 1u\) != 0u; \}/, "the packed hard cap must not accidentally enable warm start");

  const encode = WebGPUQuadtreeTallCellProjection.prototype.encode.toString();
  assert.match(encode, /megakernel=this\.megakernelEligible/);
  assert.match(encode, /quadtreeMegakernelPreferred/, "runtime selection must include the measured workload gate");
  assert.match(encode, /direct\("solveMegakernel",1\)/);
});

test("persistent PCG megakernel selection requires a converged low-work observation", () => {
  assert.equal(quadtreeMegakernelPreferred(5_000, undefined), false, "the first solve establishes a ladder baseline");
  assert.equal(quadtreeMegakernelPreferred(5_000, Number.NaN), false);
  assert.equal(quadtreeMegakernelPreferred(5_000, -1), false);
  assert.equal(quadtreeMegakernelPreferred(9_000, 3), true, "measured calm-scene work stays below the crossover");
  assert.equal(quadtreeMegakernelPreferred(3_000, 14), false, "iteration-heavy small systems retain the parallel ladder");
  assert.equal(quadtreeMegakernelPreferred(quadtreeMegakernelDofLimit + 1, 0), false);
  assert.equal(quadtreeMegakernelPreferred(quadtreeMegakernelRowIterationLimit, 1), true);
  assert.equal(quadtreeMegakernelPreferred(quadtreeMegakernelRowIterationLimit + 1, 1), false);
  assert.equal(quadtreeMegakernelPreferred(10_000, 2, 4), false, "higher-degree polynomial work is charged proportionally");
  assert.equal(quadtreeMegakernelPreferred(5_000, 1, 2, 4_096, 30_000), false, "the user DOF limit is honored");
  assert.equal(quadtreeMegakernelPreferred(5_000, 2, 2, 32_768, 9_000), false, "the user work limit is honored");
  assert.equal(quadtreeMegakernelPreferred(5_000, 2, 2, 32_768, 10_000), true);
});

test("quadtree advanced settings expose dynamic, forced, and disabled megakernel modes", () => {
  const mode = quadtreeTallCellMethod.params.find((spec) => spec.key === "megakernelMode");
  const dofLimit = quadtreeTallCellMethod.params.find((spec) => spec.key === "megakernelDofLimit");
  const workLimit = quadtreeTallCellMethod.params.find((spec) => spec.key === "megakernelRowIterationLimit");
  assert.equal(mode?.kind, "select");
  assert.equal(mode?.tier, "fine");
  if (mode?.kind === "select") assert.deepEqual(mode.options.map((option) => option.value), ["dynamic", "always", "off"]);
  assert.equal(dofLimit?.kind, "number");
  assert.equal(workLimit?.kind, "number");
  const defaults = resolveMethodValues(quadtreeTallCellMethod, "balanced", {});
  assert.equal(defaults.megakernelMode, "dynamic");
  assert.equal(defaults.megakernelDofLimit, quadtreeMegakernelDofLimit);
  assert.equal(defaults.megakernelRowIterationLimit, quadtreeMegakernelRowIterationLimit);
  assert.equal(resolveMethodValues(quadtreeTallCellMethod, "balanced", { megakernelMode: "always", megakernelDofLimit: 8_192 }).megakernelMode, "always");
});

test("blockic packing partitions the DOFs and drops cross-block factor couplings", () => {
  const nx = 16, ny = 12, nz = 16, h = { x: 0.5, y: 0.25, z: 0.5 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: h.x, maximumLeafSize: 1 });
  const phi = new Float32Array(nx * ny * nz).fill(-1);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) phi[x + nx * (ny - 1 + ny * z)] = 1;
  const velocity = Array.from({ length: phi.length }, (_, index) => ({ x: 0.01 * index, y: 0.1 * Math.sin(index), z: 0 }));
  const system = buildVariationalSystem(populateTallPressureGrid(quadtree, phi, ny, h, 1), { velocity }, { assembleDense: false });
  const packed = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, [], "blockic");
  const n = system.liquidSampleIds.length;
  assert.ok(n > 512, `fixture too small for multiple blocks (${n} DOFs)`);
  assert.ok(packed.blockCount >= 2);
  assert.ok(packed.factorLevelCount >= 1);
  const aux = packed.factorAuxWords;
  const blockOf = new Int32Array(n).fill(-1);
  let previousEnd = 0;
  for (let block = 0; block < packed.blockCount; block += 1) {
    const header = packed.blockTableOffset + 2 * block;
    const [start, end] = [aux[header], aux[header + 1]];
    assert.equal(start, previousEnd, "blocks must be contiguous ascending row ranges");
    assert.ok(end > start && end <= n);
    previousEnd = end;
    blockOf.fill(block, start, end);
  }
  assert.equal(previousEnd, n, "blocks must cover every DOF");
  const factorColumns = new Uint32Array(packed.factorColumns.buffer, packed.factorColumns.byteOffset, packed.factorColumns.byteLength / 4);
  const factorEntries = new Uint32Array(packed.factorEntries.buffer, packed.factorEntries.byteOffset, packed.factorEntries.byteLength / 4);
  let entryCount = 0;
  for (let column = 0; column < n; column += 1) {
    for (let entry = factorColumns[2 * column]; entry < factorColumns[2 * (column + 1)]; entry += 1) {
      assert.equal(blockOf[factorEntries[2 * entry]], blockOf[column], "factor couplings must not cross blocks");
      entryCount += 1;
    }
  }
  assert.ok(entryCount > 0, "the block factor must retain in-block couplings");
});

test("non-incomplete-Cholesky preconditioners skip the factorization during packing", () => {
  const nx = 16, ny = 12, nz = 16, h = { x: 0.5, y: 0.25, z: 0.5 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: h.x, maximumLeafSize: 1 });
  const phi = new Float32Array(nx * ny * nz).fill(-1);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) phi[x + nx * (ny - 1 + ny * z)] = 1;
  const velocity = Array.from({ length: phi.length }, () => ({ x: 0, y: 0, z: 0 }));
  const system = buildVariationalSystem(populateTallPressureGrid(quadtree, phi, ny, h, 1), { velocity }, { assembleDense: false });
  const packed = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, [], "jacobi");
  assert.equal(packed.factorEntries.byteLength, 0, "jacobi must not build an IC(0) factor");
  assert.equal(packed.factorLevelCount, 1);
  assert.equal(packed.blockCount, 0);
});

test("direct count-scan-emit packing reproduces the uncoupled variational reference", () => {
  const nx = 12, ny = 9, nz = 10, h = { x: 0.25, y: 0.2, z: 0.25 };
  const sizing = Float32Array.from({ length: nx * nz }, (_, index) => index % 7 === 0 ? 20 : 0);
  const quadtree = buildQuadtree(sizing, nx, nz, { h: h.x, maximumLeafSize: 4, adaptivityStrength: 1, smoothingDilations: 3 });
  const phi = Float32Array.from({ length: nx * ny * nz }, (_, index) => {
    const y = Math.floor(index / nx) % ny, x = index % nx;
    return (y - 4.2 + 0.08 * x) * h.y;
  });
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, 1, 0.25);
  const system = buildVariationalSystem(grid, {}, { assembleDense: false });
  const reference = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, [], "poly");
  const direct = WebGPUQuadtreeTallCellProjection.packUncoupledGrid(grid, nx, ny, nz);
  assert.equal(direct.dofCount, system.liquidSampleIds.length);
  assert.equal(direct.faceCount, system.faces.length);
  for (const key of ["faces", "rowOffsets", "rowEntries", "matrixWords", "cellProjection", "cellTopology", "factorColumns", "factorEntries", "factorAuxWords", "cellPressureSamples"] as const) {
    assert.deepEqual(Array.from(direct.packed[key]), Array.from(reference[key]), `${key} differs from the reference pack`);
  }
});

test("pressure command budgets follow convergence feedback without lowering the hard cap", () => {
  const initial = quadtreeIterationBudget(12_322, { pressureIterations: 96 });
  assert.equal(initial.hardBudget, 445);
  assert.equal(initial.encodedBudget, 445);
  const converged = nextQuadtreeIterationBudget(initial, 200, true);
  assert.equal(converged.hardBudget, 445);
  assert.equal(converged.encodedBudget, 314, "converged solves retain 20% EMA headroom plus a small margin without encoding the old 50% tail");
  const capped = nextQuadtreeIterationBudget({ ...converged, encodedBudget: 128 }, 128, false);
  assert.equal(capped.encodedBudget, 256);
  const thinLayer = quadtreeIterationBudget(12_322, { pressureIterations: 96, iterationConditioningScale: 8 });
  assert.ok(thinLayer.hardBudget > initial.hardBudget, "a dmax/dmin=8 adaptive layer retains headroom for long vertical pressure modes despite its smaller DOF count");
});

test("cubic pressure projection conservatively prolongs the solved variational face correction", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn solvedFaceGradient\(face: Face\)/);
  assert.match(quadtreeTallCellProjectionShader, /face\.weights\.y \/ face\.weights\.x/);
  assert.match(quadtreeTallCellProjectionShader, /else \{ gradient = solved; \}/, "single-subface faces retain the exact solved gradient");
  assert.match(quadtreeTallCellProjectionShader, /value\[axis\] -= fluidScale \* gradient/);
  assert.match(quadtreeTallCellProjectionShader, /let fluidScale = min\(20\.0,/, "velocity updates clamp theta at 0.05 independently of the matrix");
  assert.match(quadtreeVelocityClampShader, /let limit = 0\.9 \* params\.cell\[axis\] \/ max\(params\.cell\.w, 1e-6\)/, "a current-step CFL clamp catches projection-created spikes");
  assert.match(quadtreeVelocityClampShader, /atomicAdd\(&debugCounters\[0\], 1u\)/, "last-resort clamping is never silent");
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /cellPressure\(plus\) - cellPressure\(gid\)/);
});

test("monolithic rigid coupling and MLS mapping are wired into the solve", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn coupleReduce/);
  assert.match(quadtreeTallCellProjectionShader, /fn coupleApply/);
  assert.match(quadtreeTallCellProjectionShader, /fn coupleImpulse/);
  assert.match(quadtreeTallCellProjectionShader, /fn mapPressure/);
  assert.match(quadtreeTallCellProjectionShader, /let ghostValue = clamp\(phiAir \/ phiLiquid, -20\.0, 0\.0\)/, "MLS includes linear ghost-air pressures at the free surface");
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /own\.x == 0xffffffffu && own\.y == 0xffffffffu.*return/, "air queries participate in ghost-pressure mapping");
  assert.match(quadtreeTallCellProjectionShader, /fn refreshFaceMls/);
  assert.match(quadtreeTallCellProjectionShader, /gradient = gradient - face\.mlsMean \+ solved/, "GPU MLS keeps the solved adaptive face average");
  assert.match(quadtreeTallCellProjectionShader, /face\.flux = face\.weights\.x \* faceVelocity\(face\) \+ face\.solidFlux/);
  assert.match(quadtreeTallCellProjectionShader, /addStateF\(dof, MATRIX_DIRECTION, sum\)/);
});

test("corrected inner ghost velocity averages every replaced vertical face", () => {
  // Every vertical face, ghost or single-cell, averages its leaf's full x/z
  // footprint; only horizontal faces take the transverse-row branch.
  assert.match(quadtreeTallCellProjectionShader, /if \(axis != 1u\) \{/);
  assert.match(quadtreeTallCellProjectionShader, /for \(var y = face\.bounds\.z; y < face\.bounds\.w; y \+= 1u\)/);
  assert.match(quadtreeTallCellProjectionShader, /sum \/ max\(1\.0, count\)/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

async function withSurfaceDevice(run: (device: GPUDevice) => Promise<void>) {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "no WebGPU adapter");
  const device = await adapter.requestDevice();
  try {
    device.pushErrorScope("validation");
    await run(device);
    const validation = await device.popErrorScope();
    assert.equal(validation, null, `WebGPU validation error: ${validation?.message}`);
  } finally { device.destroy(); }
}

function writeVelocityTexture(device: GPUDevice, nx: number, ny: number, nz: number, sample: (x: number, y: number, z: number) => [number, number, number]) {
  const texture = device.createTexture({ size: [nx, ny, nz], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const rowBytes = nx * 16, pitch = Math.ceil(rowBytes / 256) * 256;
  const upload = new Uint8Array(pitch * ny * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    const row = new Float32Array(upload.buffer, pitch * (y + ny * z), nx * 4);
    for (let x = 0; x < nx; x += 1) row.set([...sample(x, y, z), 0], x * 4);
  }
  device.queue.writeTexture({ texture }, upload, { bytesPerRow: pitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
  return texture;
}

function writeScalarTexture(device: GPUDevice, values: Float32Array, nx: number, ny: number, nz: number) {
  const texture = device.createTexture({ size: [nx, ny, nz], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const rowBytes = nx * 4, pitch = Math.ceil(rowBytes / 256) * 256, upload = new Uint8Array(pitch * ny * nz), source = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) upload.set(source.subarray(rowBytes * (y + ny * z), rowBytes * (y + ny * z + 1)), pitch * (y + ny * z));
  device.queue.writeTexture({ texture }, upload, { bytesPerRow: pitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
  return texture;
}

function writeStorageBuffer(device: GPUDevice, data: ArrayBufferView, extraUsage = 0) {
  const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

async function gpuComputeExecutes(device: GPUDevice) {
  const storage = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const shaderModule = device.createShaderModule({ code: "@group(0) @binding(0) var<storage, read_write> value: array<u32>; @compute @workgroup_size(1) fn probe() { value[0] = 0x5a17u; }" });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "probe" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: storage } }] });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(storage, 0, readback, 0, 4); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ); const result = new Uint32Array(readback.getMappedRange())[0]; readback.unmap();
  storage.destroy(); readback.destroy(); return result === 0x5a17;
}

test("persistent PCG megakernel solves a known SPD system and publishes diagnostics", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU solve checks" }, async (t) => {
  await withSurfaceDevice(async (device) => {
    if (!await gpuComputeExecutes(device)) { t.skip("Dawn adapter accepted pipelines but did not execute a trivial compute dispatch"); return; }
    const shaderModule = device.createShaderModule({ label: "megakernel numerical contract", code: quadtreeTallCellProjectionShader });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "solveMegakernel" } });
    for (const degree of [1, 2]) {
      device.pushErrorScope("validation");
      const faces = new Uint32Array(28), faceFloats = new Float32Array(faces.buffer); faceFloats[24] = 1;
      const rowOffsets = new Uint32Array([0, 1, 1]);
      const rowEntries = new Uint32Array([0, new Uint32Array(new Float32Array([1]).buffer)[0]]);
      const matrix = new Uint32Array(3 + 4 * 4); matrix.set([0, 2, 4]);
      const coefficients = [2, -1, -1, 2], nodes = [0, 1, 0, 1];
      for (let entry = 0; entry < 4; entry += 1) {
        const base = 3 + 4 * entry; matrix[base] = nodes[entry];
        matrix[base + 2] = new Uint32Array(new Float32Array([coefficients[entry]]).buffer)[0];
      }
      const stateWords = new Uint32Array(16); stateWords[14] = 1; stateWords[15] = 1;
      const scalarWords = new Float32Array(32);
      const params = new Uint32Array(48), paramsF = new Float32Array(params.buffer);
      params[3] = 1; params.set([2, 1, 0, 0], 8); paramsF[12] = 1e-8;
      params.set([0, 0, 8 << 1, 1], 20); params[27] = degree;

      const faceBuffer = writeStorageBuffer(device, faces);
      const rowOffsetBuffer = writeStorageBuffer(device, rowOffsets);
      const rowEntryBuffer = writeStorageBuffer(device, rowEntries);
      const matrixBuffer = writeStorageBuffer(device, matrix);
      const stateBuffer = writeStorageBuffer(device, stateWords, GPUBufferUsage.COPY_SRC);
      const scalarBuffer = writeStorageBuffer(device, scalarWords, GPUBufferUsage.COPY_SRC);
      const paramsBuffer = device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, params);
      const factorAux = device.createTexture({ size: [1, 1], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING });
      const mappedPressure = device.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 3, resource: { buffer: faceBuffer } }, { binding: 4, resource: { buffer: rowOffsetBuffer } },
        { binding: 5, resource: { buffer: rowEntryBuffer } }, { binding: 6, resource: { buffer: matrixBuffer } },
        { binding: 8, resource: { buffer: stateBuffer } }, { binding: 9, resource: { buffer: scalarBuffer } },
        { binding: 12, resource: { buffer: paramsBuffer } }, { binding: 13, resource: factorAux.createView() },
        { binding: 17, resource: mappedPressure.createView() }
      ] });
      const readback = device.createBuffer({ size: 16 * 4 + 12 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(stateBuffer, 0, readback, 0, 16 * 4);
      encoder.copyBufferToBuffer(scalarBuffer, 0, readback, 16 * 4, 12 * 4);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const state = new Float32Array(readback.getMappedRange(0, 16 * 4));
      const scalars = new Float32Array(readback.getMappedRange(16 * 4, 12 * 4));
      const validation = await device.popErrorScope();
      assert.equal(validation, null, `degree-${degree} dispatch validation error: ${validation?.message}`);
      assert.ok(Math.abs(state[2] - 2 / 3) < 2e-5 && Math.abs(state[3] - 1 / 3) < 2e-5, `degree-${degree} best iterate was [${state[2]}, ${state[3]}], pressure [${state[0]}, ${state[1]}], scalars ${Array.from(scalars).join(",")}`);
      assert.equal(scalars[3], 1, "|b|^2 telemetry remains relative to the original RHS");
      assert.ok(Math.sqrt(scalars[7] / scalars[3]) <= 1e-4, `degree-${degree} relative residual ${Math.sqrt(scalars[7] / scalars[3])}`);
      assert.ok(scalars[9] >= 1 && scalars[9] <= 2, `degree-${degree} iteration count ${scalars[9]}`);
      assert.ok(Number.isFinite(scalars[4]) && Number.isFinite(scalars[6]), "alpha/beta telemetry is finite");
      readback.unmap();
      for (const resource of [faceBuffer, rowOffsetBuffer, rowEntryBuffer, matrixBuffer, stateBuffer, scalarBuffer, paramsBuffer, readback]) resource.destroy();
      factorAux.destroy(); mappedPressure.destroy();
    }
  });
});

test("GPU count-scan-emit rebuild reproduces the CPU topology and face graph", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU pack checks" }, async () => {
  await withSurfaceDevice(async (device) => {
    const nx = 12, ny = 9, nz = 10, h = { x: 0.25, y: 0.2, z: 0.25 };
    const sizing = Float32Array.from({ length: nx * nz }, (_, index) => index % 7 === 0 ? 20 : 0);
    const quadtree = buildQuadtree(sizing, nx, nz, { h: h.x, maximumLeafSize: 4, adaptivityStrength: 1, smoothingDilations: 3 });
    const packedCells = new Uint32Array(nx * nz);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) { const leaf = quadtree.leaves[quadtree.leafAt[x + nx * z]]; packedCells[x + nx * z] = leaf.x | (leaf.z << 10) | (leaf.size << 20); }
    const phi = Float32Array.from({ length: nx * ny * nz }, (_, index) => { const y = Math.floor(index / nx) % ny, x = index % nx; return (y - 4.2 + 0.08 * x) * h.y; });
    const grid = populateTallPressureGrid(quadtree, phi, ny, h, 1, 0.25), reference = WebGPUQuadtreeTallCellProjection.packUncoupledGrid(grid, nx, ny, nz);
    const texture = writeScalarTexture(device, phi, nx, ny, nz), builder = new WebGPUQuadtreePackBuilder(device, { nx, ny, nz }, h, 0.25);
    const gpu = await builder.build(packedCells, texture, { dofCount: reference.dofCount, faceCount: reference.faceCount, pressureSampleCount: grid.samples.length });
    assert.ok(gpu, "GPU pack overflowed a 1.75x reference-sized workspace");
    assert.deepEqual({ leaves: gpu.leafCount, samples: gpu.pressureSampleCount, dofs: gpu.dofCount, faces: gpu.faceCount, ghosts: gpu.ghostFaceCount, tall: gpu.tallSegmentCount }, {
      leaves: quadtree.leaves.length, samples: grid.samples.length, dofs: reference.dofCount, faces: reference.faceCount, ghosts: reference.ghostFaceCount, tall: reference.tallSegmentCount
    });
    assert.deepEqual(Array.from(gpu.packed.cellTopology), Array.from(reference.packed.cellTopology));
    const gpuFaceU32 = new Uint32Array(gpu.packed.faces.buffer), referenceFaceU32 = new Uint32Array(reference.packed.faces.buffer);
    const gpuFaceF32 = new Float32Array(gpu.packed.faces.buffer), referenceFaceF32 = new Float32Array(reference.packed.faces.buffer);
    // GPU scans use dense z/x leaf-owner order while the recursive CPU grid
    // assigns leaf/sample ids in tree order. Canonicalize both sparse ids by
    // their physical sample and face geometry before requiring exact content.
    const sampleKey = (words: Uint32Array, base: number, dof: number) => `${words[base + 4 * dof]},${words[base + 4 * dof + 1]},${words[base + 4 * dof + 2]}`;
    const referenceDofByKey = new Map(Array.from({ length: reference.dofCount }, (_, dof) => [sampleKey(reference.packed.factorAuxWords, reference.packed.dofSamplesBase, dof), dof] as const));
    const dofMap = Array.from({ length: gpu.dofCount }, (_, dof) => referenceDofByKey.get(sampleKey(gpu.packed.factorAuxWords, gpu.packed.dofSamplesBase, dof)) ?? -1);
    assert.ok(dofMap.every((dof) => dof >= 0) && new Set(dofMap).size === reference.dofCount, "GPU DOFs bijectively match CPU sample geometry");
    const faceKey = (u32: Uint32Array, face: number) => {
      const base = 28 * face;
      return [u32[base + 12] & 0x003fffff, ...Array.from(u32.subarray(base + 8, base + 12)), ...Array.from(u32.subarray(base + 16, base + 24))].join(",");
    };
    const referenceFaceByKey = new Map(Array.from({ length: reference.faceCount }, (_, face) => [faceKey(referenceFaceU32, face), face] as const));
    const faceMap = Array.from({ length: gpu.faceCount }, (_, face) => referenceFaceByKey.get(faceKey(gpuFaceU32, face)) ?? -1);
    const unmatchedFace = faceMap.findIndex((face) => face < 0);
    const closestReference = unmatchedFace < 0 ? -1 : Array.from({ length: reference.faceCount }, (_, face) => face).find((face) => {
      const gb = 28 * unmatchedFace, rb = 28 * face; return (gpuFaceU32[gb + 12] & 0x003fffff) === (referenceFaceU32[rb + 12] & 0x003fffff) && [8, 9, 10, 11].every((word) => gpuFaceU32[gb + word] === referenceFaceU32[rb + word]);
    }) ?? -1;
    assert.ok(unmatchedFace < 0 && new Set(faceMap).size === reference.faceCount, unmatchedFace < 0 ? "GPU face mapping is not bijective" : `GPU face ${unmatchedFace} has no CPU geometry match: ${faceKey(gpuFaceU32, unmatchedFace)}; closest CPU ${closestReference}: ${closestReference < 0 ? "none" : faceKey(referenceFaceU32, closestReference)}`);
    const remappedPressureSamples = gpu.packed.cellPressureSamples.slice();
    for (let cell = 0; cell < nx * ny * nz; cell += 1) for (let endpoint = 0; endpoint < 2; endpoint += 1) {
      const index = 4 * cell + endpoint, dof = remappedPressureSamples[index]; if (dof !== 0xffffffff) remappedPressureSamples[index] = dofMap[dof];
    }
    assert.deepEqual(Array.from(remappedPressureSamples), Array.from(reference.packed.cellPressureSamples));
    const remappedProjection = gpu.packed.cellProjection.slice();
    for (let cell = 0; cell < nx * ny * nz; cell += 1) for (let axis = 0; axis < 3; axis += 1) { const index = 4 * cell + axis, encoded = remappedProjection[index]; if (encoded > 0) remappedProjection[index] = faceMap[Math.round(encoded) - 1] + 1; }
    assert.deepEqual(Array.from(remappedProjection), Array.from(reference.packed.cellProjection));
    for (let gpuFace = 0; gpuFace < gpu.faceCount; gpuFace += 1) {
      const referenceFace = faceMap[gpuFace], gpuBase = gpuFace * 28, referenceBase = referenceFace * 28;
      for (let slot = 0; slot < 4; slot += 1) { const dof = gpuFaceU32[gpuBase + slot]; assert.equal(dof === 0xffffffff ? dof : dofMap[dof], referenceFaceU32[referenceBase + slot], `face ${gpuFace} node ${slot}`); }
      for (const word of [8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23]) assert.equal(gpuFaceU32[gpuBase + word], referenceFaceU32[referenceBase + word], `face ${gpuFace} word ${word}`);
      for (const word of [4, 5, 6, 7, 14, 26]) assert.ok(Math.abs(gpuFaceF32[gpuBase + word] - referenceFaceF32[referenceBase + word]) < 2e-6, `face ${gpuFace} float ${word}`);
    }
    const incidentRows = (rowOffsets: Uint32Array, entries: Uint8Array, row: number, mapFace: (face: number) => number) => {
      const words = new Uint32Array(entries.buffer, entries.byteOffset, entries.byteLength / 4), floats = new Float32Array(entries.buffer, entries.byteOffset, entries.byteLength / 4);
      return Array.from({ length: rowOffsets[row + 1] - rowOffsets[row] }, (_, local) => { const entry = rowOffsets[row] + local; return [mapFace(words[2 * entry]), floats[2 * entry + 1]] as const; }).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    };
    for (let gpuRow = 0; gpuRow < gpu.dofCount; gpuRow += 1) assert.deepEqual(incidentRows(gpu.packed.rowOffsets, gpu.packed.rowEntries, gpuRow, (face) => faceMap[face]), incidentRows(reference.packed.rowOffsets, reference.packed.rowEntries, dofMap[gpuRow], (face) => face));
    const matrixRow = (words: Uint32Array, dofs: number, row: number, mapDof: (dof: number) => number, mapFace: (face: number) => number) => {
      const floats = new Float32Array(words.buffer, words.byteOffset, words.length), start = words[row], end = words[row + 1];
      return Array.from({ length: end - start }, (_, local) => { const base = dofs + 1 + 4 * (start + local), packedFace = words[base + 1]; return [mapDof(words[base]), mapFace(packedFace & 0x3fffffff) | ((packedFace >>> 30) << 30), floats[base + 3]] as const; }).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    };
    for (let gpuRow = 0; gpuRow < gpu.dofCount; gpuRow += 1) assert.deepEqual(matrixRow(gpu.packed.matrixWords, gpu.dofCount, gpuRow, (dof) => dofMap[dof], (face) => faceMap[face]), matrixRow(reference.packed.matrixWords, reference.dofCount, dofMap[gpuRow], (dof) => dof, (face) => face));
    const resident = await builder.build(packedCells, texture, { dofCount: reference.dofCount, faceCount: reference.faceCount, pressureSampleCount: grid.samples.length }, true);
    assert.ok(resident?.resident, "runtime GPU pack returns directly bindable resources");
    assert.equal(resident.packed.faces.byteLength, 0, "resident runtime path does not materialize the sparse pack on the CPU");
    await device.queue.onSubmittedWorkDone();
    for (const buffer of [resident.resident.faces, resident.resident.rowOffsets, resident.resident.rowEntries, resident.resident.matrixBuffer, resident.resident.factorColumns, resident.resident.factorEntries]) buffer.destroy();
    for (const resource of [resident.resident.factorAux, resident.resident.cellProjection, resident.resident.cellTopology, resident.resident.cellPressureSamples]) resource.destroy();
    builder.destroy(); texture.destroy();
  });
});

test("GPU sparse packing consumes the same adaptive optical columns as the CPU oracle", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU pack checks" }, async () => {
  await withSurfaceDevice(async (device) => {
    const nx = 8, ny = 32, nz = 6, h = { x: 0.25, y: 0.25, z: 0.25 };
    const phi = Float32Array.from({ length: nx * ny * nz }, (_, index) => (Math.floor(index / nx) % ny - 15.5) * h.y);
    const velocity = Array.from({ length: phi.length }, (_, index) => {
      const y = Math.floor(index / nx) % ny, x = index % nx;
      return { x: x === 3 ? (y % 2 === 0 ? 8 : -8) : 0, y: 0, z: 0 };
    });
    const sizing = new Float32Array(nx * nz).fill(100), quadtree = buildQuadtree(sizing, nx, nz, { h: h.x, maximumLeafSize: 2, adaptivityStrength: 1 });
    const packedCells = new Uint32Array(nx * nz);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) { const leaf = quadtree.leaves[quadtree.leafAt[x + nx * z]]; packedCells[x + nx * z] = leaf.x | (leaf.z << 10) | (leaf.size << 20); }
    const optical = buildAdaptiveOpticalLayerField(phi, velocity, nx, ny, nz, h), grid = populateTallPressureGrid(quadtree, phi, ny, h, 1, 0.25, undefined, optical);
    const reference = WebGPUQuadtreeTallCellProjection.packUncoupledGrid(grid, nx, ny, nz);
    const opticalData = new Uint32Array(optical.columns.length);
    opticalData.set(optical.columns);
    const opticalBuffer = device.createBuffer({ label: "adaptive optical test columns", size: opticalData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(opticalBuffer, 0, opticalData);
    const texture = writeScalarTexture(device, phi, nx, ny, nz), builder = new WebGPUQuadtreePackBuilder(device, { nx, ny, nz }, h, 0.25, "adaptive-motion");
    const gpu = await builder.build(packedCells, texture, { dofCount: reference.dofCount, faceCount: reference.faceCount, pressureSampleCount: grid.samples.length }, false, opticalBuffer);
    assert.ok(gpu);
    assert.deepEqual({ samples: gpu.pressureSampleCount, dofs: gpu.dofCount, faces: gpu.faceCount, tall: gpu.tallSegmentCount }, { samples: grid.samples.length, dofs: reference.dofCount, faces: reference.faceCount, tall: reference.tallSegmentCount });
    assert.deepEqual(Array.from(gpu.packed.cellTopology), Array.from(reference.packed.cellTopology), "adaptive CPU/GPU segmentation bounds must match exactly");
    builder.destroy(); texture.destroy(); opticalBuffer.destroy();
  });
});

test("GPU motion-error construction matches the adaptive optical CPU oracle", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU optical-layer checks" }, async () => {
  await withSurfaceDevice(async (device) => {
    const nx = 8, ny = 32, nz = 6, h = { x: 0.25, y: 0.25, z: 0.25 };
    const phi = Float32Array.from({ length: nx * ny * nz }, (_, index) => (Math.floor(index / nx) % ny - 15.5) * h.y);
    const velocityValues = Array.from({ length: phi.length }, (_, index) => {
      const y = Math.floor(index / nx) % ny, x = index % nx;
      return { x: x === 3 ? (y % 2 === 0 ? 8 : -8) : 0, y: 0, z: 0 };
    });
    const expected = buildAdaptiveOpticalLayerField(phi, velocityValues, nx, ny, nz, h).columns;
    const levelSet = writeScalarTexture(device, phi, nx, ny, nz);
    const velocity = writeVelocityTexture(device, nx, ny, nz, (x, y, z) => { const value = velocityValues[x + nx * (y + ny * z)]; return [value.x, value.y, value.z]; });
    const diagnostics = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const builder = new WebGPUQuadtreeBuilder(device, { nx, ny, nz }, h, 2, 1, 1, undefined, 1, "adaptive-motion", 0.5);
    const built = await builder.build({ velocity, levelSet, explicitSizing: new Float32Array(nx * nz).fill(100), diagnosticBuffer: diagnostics, diagnosticBytes: 48, readLeafProfiles: false });
    assert.deepEqual(Array.from(built.opticalColumns), Array.from(expected));
    WebGPUQuadtreeBuilder.destroyCache(builder.cache); diagnostics.destroy(); velocity.destroy(); levelSet.destroy();
  });
});

async function readScalarTexture(device: GPUDevice, texture: GPUTexture, nx: number, ny: number, nz: number) {
  const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * ny * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: ny }, [nx, ny, nz]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(buffer.getMappedRange().slice(0));
  const out = new Float32Array(nx * ny * nz), rowFloats = bytesPerRow / 4;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) out[x + nx * (y + ny * z)] = raw[x + rowFloats * (y + ny * z)];
  buffer.destroy();
  return out;
}

test("per-step redistance preserves a tilted plane's sub-cell interface position", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU level-set checks" }, async () => {
  await withSurfaceDevice(async (device) => {
    const { WebGPUQuadtreeSurfaceState } = await import("../lib/webgpu-quadtree-builder");
    const n = 32, h = 1 / n, cell = { x: h, y: h, z: h };
    const magnitude = Math.hypot(1, 2, 0.5), normal = [1 / magnitude, 2 / magnitude, 0.5 / magnitude];
    const center = [16.3, 15.7, 16.1];
    const exact = (x: number, y: number, z: number) => ((x - center[0]) * normal[0] + (y - center[1]) * normal[1] + (z - center[2]) * normal[2]) * h;
    const phi = new Float32Array(n * n * n);
    for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) phi[x + n * (y + n * z)] = exact(x, y, z);
    const velocity = writeVelocityTexture(device, n, n, n, () => [0, 0, 0]);
    const state = new WebGPUQuadtreeSurfaceState(device, { nx: n, ny: n, nz: n }, cell, velocity, phi);
    for (let step = 0; step < 5; step += 1) {
      const encoder = device.createCommandEncoder();
      state.encode(encoder, 1 / 60);
      device.queue.submit([encoder.finish()]);
    }
    const result = await readScalarTexture(device, state.texture, n, n, n);
    let maxBandError = 0, bandSum = 0, bandCount = 0;
    for (let z = 2; z < n - 2; z += 1) for (let y = 2; y < n - 2; y += 1) for (let x = 2; x < n - 2; x += 1) {
      const truth = exact(x, y, z);
      if (Math.abs(truth) >= 1.5 * h) continue;
      const error = Math.abs(result[x + n * (y + n * z)] - truth);
      maxBandError = Math.max(maxBandError, error); bandSum += error; bandCount += 1;
    }
    // The old +0.5h redistance floor produced ~0.5h errors beside the
    // interface; sub-cell seeding must keep the band under a fifth of a cell
    // even after five consecutive redistance passes.
    assert.ok(bandCount > 500, `narrow band unexpectedly small (${bandCount})`);
    assert.ok(maxBandError < 0.2 * h, `max narrow-band redistance error ${(maxBandError / h).toFixed(3)}h >= 0.2h`);
    assert.ok(bandSum / bandCount < 0.06 * h, `mean narrow-band redistance error ${(bandSum / bandCount / h).toFixed(3)}h >= 0.06h`);
    state.destroy(); velocity.destroy();
  });
});

test("bounded MacCormack transport keeps a rotating notched column intact", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU level-set checks" }, async () => {
  await withSurfaceDevice(async (device) => {
    const { WebGPUQuadtreeSurfaceState } = await import("../lib/webgpu-quadtree-builder");
    const nx = 64, ny = 8, nz = 64, h = 1 / 64, cell = { x: h, y: h, z: h };
    const rotationCenter = [32, 32], diskCenter = [32, 48], radius = 10, notchHalfWidth = 2, notchDepth = 12;
    const inDisk = (x: number, z: number) => {
      const inCircle = Math.hypot(x - diskCenter[0], z - diskCenter[1]) <= radius;
      const inNotch = Math.abs(x - diskCenter[0]) <= notchHalfWidth && z <= diskCenter[1] - radius + notchDepth;
      return inCircle && !inNotch;
    };
    const phi = new Float32Array(nx * ny * nz);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      // Coarse initialization is fine: the first redistance rebuilds the SDF.
      phi[x + nx * (y + ny * z)] = inDisk(x, z) ? -0.5 * h : 0.5 * h;
    }
    const steps = 330, dt = 2 * Math.PI / steps;
    // Solid rotation about the y axis (omega = 1 rad/s), stored as
    // negative-face MAC samples: u_x lives at (x-1/2, y, z), u_z at
    // (x, y, z-1/2), in metres per second.
    const staggered = writeVelocityTexture(device, nx, ny, nz, (x, _y, z) => [
      -((z - rotationCenter[1]) * h), 0, ((x - rotationCenter[0]) * h)
    ]);
    const state = new WebGPUQuadtreeSurfaceState(device, { nx, ny, nz }, cell, staggered, phi);
    for (let step = 0; step < steps; step += 1) {
      const encoder = device.createCommandEncoder();
      state.encode(encoder, dt);
      device.queue.submit([encoder.finish()]);
      if (step % 32 === 31) await device.queue.onSubmittedWorkDone();
    }
    const result = await readScalarTexture(device, state.texture, nx, ny, nz);
    const y = 4;
    let initialWet = 0, wetCount = 0, intersection = 0, union = 0, notchAir = 0, notchCells = 0;
    for (let z = 4; z < nz - 4; z += 1) for (let x = 4; x < nx - 4; x += 1) {
      const expected = inDisk(x, z), wet = result[x + nx * (y + ny * z)] < 0;
      if (expected) initialWet += 1;
      if (wet) wetCount += 1;
      if (expected && wet) intersection += 1;
      if (expected || wet) union += 1;
      const inNotch = Math.abs(x - diskCenter[0]) <= notchHalfWidth - 1 && z >= diskCenter[1] - radius + 2 && z <= diskCenter[1] - radius + notchDepth - 2;
      if (inNotch) { notchCells += 1; if (!wet) notchAir += 1; }
    }
    const iou = intersection / Math.max(1, union), notchRetention = notchAir / Math.max(1, notchCells);
    // Measured on Dawn/Metal: first-order transport dissolves the column
    // completely within one revolution (0 wet cells, IoU 0); bounded
    // MacCormack arrives volume-preserving (~96% wet, IoU ~0.44 dominated by
    // a dispersive phase lag of a few cells, notch ~0.19 air). The gates sit
    // between the two schemes with margin.
    assert.ok(wetCount >= 0.7 * initialWet && wetCount <= 1.3 * initialWet, `wet cells ${wetCount} drifted beyond +-30% of ${initialWet}`);
    assert.ok(iou >= 0.35, `notched-column IoU after one revolution ${iou.toFixed(3)} < 0.35`);
    assert.ok(notchRetention >= 0.1, `notch air retention ${notchRetention.toFixed(3)} < 0.1`);
    state.destroy(); staggered.destroy();
  });
});
