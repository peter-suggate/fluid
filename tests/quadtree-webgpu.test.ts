import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { simulationMethods } from "../lib/methods";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { nextQuadtreeIterationBudget, quadtreeDispatchShader, quadtreeDivergenceShader, quadtreeIterationBudget, quadtreeTallCellProjectionShader, quadtreeVelocityClampShader, quadtreeVelocityExtrapolationShader, WebGPUQuadtreeTallCellProjection } from "../lib/webgpu-quadtree-tall-cell";
import { nextQuadtreeVofReconciliationActive, packedQuadtreeRootMap, quadtreeConstructionShader, quadtreeSurfaceShader, quadtreeVofReconciliationFraction } from "../lib/webgpu-quadtree-builder";
import { WebGPUQuadtreePackBuilder } from "../lib/webgpu-quadtree-pack-builder";
import { buildQuadtree, buildVariationalSystem, populateTallPressureGrid } from "../lib/quadtree-tall-cell-grid";
import { proactiveQuadtreeSubsteps, quadtreeMissedFrames } from "../lib/webgpu-uniform-eulerian";

test("the old optical-layer method is replaced by quadtree tall cells", () => {
  assert.ok(simulationMethods.includes(quadtreeTallCellMethod));
  assert.ok(!simulationMethods.some((method) => method.id === "adaptive-optical-layer"));
  assert.equal(quadtreeTallCellMethod.shortLabel, "Adaptive");
  assert.match(quadtreeTallCellMethod.detail, /T-junction/);
  assert.equal(quadtreeTallCellMethod.presetFor("balanced").preconditioner, "poly", "the parallel polynomial preconditioner is the runtime default");
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "preconditioner" && param.default === "poly"));
  assert.ok(quadtreeTallCellMethod.params.find((param) => param.key === "vofReconciliation" && param.default === "on"), "W0 reconciliation stays enabled until the W7 soak gate passes");
});

test("quadtree projection preserves and extrapolates near-surface air velocity", () => {
  assert.match(quadtreeTallCellProjectionShader, /ownPhi > 2\.0 \* h && otherPhi > 2\.0 \* h/, "only far-field air faces are zeroed");
  assert.match(quadtreeVelocityExtrapolationShader, /fn extrapolateVelocity/);
  assert.match(quadtreeVelocityExtrapolationShader, /phi\(q\) < 0\.0 \|\| phi\(plus\) < 0\.0/, "known faces touch phi-negative liquid");
  assert.match(quadtreeVelocityExtrapolationShader, /min\(ownPhi, otherPhi\) < 3\.0 \* h/, "extrapolation is restricted to the three-cell surface band");
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

test("quadtree blocked-frame telemetry counts missed presentation budgets, not retry polls", () => {
  assert.equal(quadtreeMissedFrames(0), 0);
  assert.equal(quadtreeMissedFrames(16), 0);
  assert.equal(quadtreeMissedFrames(17), 1);
  assert.equal(quadtreeMissedFrames(34), 2);
});

test("VOF reconciliation is an armed catastrophic-loss circuit breaker", () => {
  assert.equal(nextQuadtreeVofReconciliationActive(false, -0.099), false);
  assert.equal(nextQuadtreeVofReconciliationActive(false, -0.101), true);
  assert.equal(nextQuadtreeVofReconciliationActive(true, -0.021), true);
  assert.equal(nextQuadtreeVofReconciliationActive(true, -0.019), false);
  assert.equal(quadtreeVofReconciliationFraction(0, 100), 0);
  assert.equal(quadtreeVofReconciliationFraction(100, 800), 1 / 64);
  assert.equal(quadtreeVofReconciliationFraction(1000, 100), 1 / 32);
});

test("quadtree updates evaluate sizing, subdivide, and smooth on WebGPU", () => {
  for (const entry of ["evaluateSizing", "refine", "smoothTopology", "sampleLeafProfiles"]) assert.match(quadtreeConstructionShader, new RegExp(`fn ${entry}\\b`));
  assert.match(quadtreeConstructionShader, /for \(var y = 0u; y < params\.dims\.y; y \+= 1u\)/, "sizing must vertically reduce each column on the GPU");
  assert.match(quadtreeConstructionShader, /sizingField\[index2\(q\)\] = maximum/);
  assert.match(quadtreeConstructionShader, /neighborTooFine/);
  assert.match(quadtreeConstructionShader, /demand > 1\.0 \/ testedWidth/);
  const roots = packedQuadtreeRootMap(16, 8, 8);
  assert.equal(roots.length, 128);
  assert.equal((roots[0] >>> 20) & 1023, 8);
  assert.equal((roots[15] >>> 20) & 1023, 8);
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
  assert.match(quadtreeSurfaceShader, /value - params\.control\.x \* h \* params\.cellAndDt\.w/);
  assert.match(quadtreeSurfaceShader, /abs\(value\) < 1\.5 \* h/);
  assert.match(quadtreeSurfaceShader, /value \/ \(4\.0 \* params\.cellAndDt\.y\)/);
  assert.match(quadtreeSurfaceShader, /atomicAdd\(&reductions\[0\]/);
  assert.match(quadtreeSurfaceShader, /isInflowVelocityCell/, "the nozzle must source fluid into the resident level set");
  // The GPU port of reconcileLevelSetWithVolume: decisive wet/dry
  // disagreement with the conservative VOF is overruled toward the VOF, and
  // JFA seeding ignores phi sign changes with no VOF interface nearby. Both
  // are gated on control.y so pure-transport users see no behavior change.
  assert.match(quadtreeSurfaceShader, /reconcileVolumeIn/);
  assert.match(quadtreeSurfaceShader, /params\.control\.y > 0\.5/);
  assert.match(quadtreeSurfaceShader, /let signMismatch = \(result < 0\.0\) != wet/);
  assert.match(quadtreeSurfaceShader, /let decisiveMismatch = signMismatch && abs\(result\) > 0\.5 \* h/);
  assert.match(quadtreeSurfaceShader, /\(0\.5 - alpha\) \* \(4\.0 \* params\.cellAndDt\.y\)/, "VOF repairs preserve the conservative sub-cell amount instead of stamping half-cell signs");
  assert.match(quadtreeSurfaceShader, /params\.control\.y > 0\.5 && wet/, "the W0 safety net restores lost liquid without deleting phi-wet surface cells at a diffused VOF threshold");
  assert.match(quadtreeSurfaceShader, /atomicAdd\(&reductions\[3\], 1u\)/, "the W7 retirement gate needs a real pre-reconciliation mismatch count");
  assert.doesNotMatch(quadtreeSurfaceShader, /\bvolumeIn\b|\bloadVolume\b/);
  assert.match(quadtreeSurfaceShader, /fn cullDebris/);
  assert.match(quadtreeSurfaceShader, /params\.control\.z > 0\.5/, "debris hygiene stays explicitly gated");
  assert.match(quadtreeSurfaceShader, /textureLoad\(reconcileVolumeIn, q, 0\)\.x < 0\.5/);
  assert.match(quadtreeConstructionShader, /fn effectiveWet[\s\S]*return loadAdvancedPhi\(clamp3\(q\)\) < 0\.0/);
  assert.match(quadtreeConstructionShader, /let profile = 3u \* \(leaf \* params\.dims\.y \+ y\)/);
  assert.match(quadtreeConstructionShader, /columnProfiles\[profile \+ 1u\] = minimum/);
  assert.match(quadtreeConstructionShader, /columnProfiles\[profile \+ 2u\] = maximum/);
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
  assert.ok(converged.encodedBudget < initial.encodedBudget);
  const capped = nextQuadtreeIterationBudget({ ...converged, encodedBudget: 128 }, 128, false);
  assert.equal(capped.encodedBudget, 256);
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
