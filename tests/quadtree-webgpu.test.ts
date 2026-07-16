import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { simulationMethods } from "../lib/methods";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { nextQuadtreeIterationBudget, quadtreeDispatchShader, quadtreeIterationBudget, quadtreeTallCellProjectionShader } from "../lib/webgpu-quadtree-tall-cell";
import { packedQuadtreeRootMap, quadtreeConstructionShader, quadtreeSurfaceShader } from "../lib/webgpu-quadtree-builder";

test("the old optical-layer method is replaced by quadtree tall cells", () => {
  assert.ok(simulationMethods.includes(quadtreeTallCellMethod));
  assert.ok(!simulationMethods.some((method) => method.id === "adaptive-optical-layer"));
  assert.equal(quadtreeTallCellMethod.shortLabel, "Adaptive");
  assert.match(quadtreeTallCellMethod.detail, /T-junction/);
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
  assert.match(quadtreeSurfaceShader, /if \(abs\(advected\) >= 2\.5 \* h\)/, "the narrow band must keep the advected phi verbatim");
  assert.doesNotMatch(quadtreeSurfaceShader, /sqrt\(seedDistanceSquared\(gid, word\)\) \+ 0\.5 \* h\b/, "the half-cell redistance floor must be gone");
  assert.match(quadtreeSurfaceShader, /fn volumeCorrectedPhi/);
  assert.match(quadtreeSurfaceShader, /value - params\.control\.x \* h \* params\.cellAndDt\.w/);
  assert.match(quadtreeSurfaceShader, /abs\(value\) < 1\.5 \* h/);
  assert.match(quadtreeSurfaceShader, /value \/ \(4\.0 \* params\.cellAndDt\.y\)/);
  assert.match(quadtreeSurfaceShader, /atomicAdd\(&reductions\[0\]/);
  assert.match(quadtreeSurfaceShader, /isInflowVelocityCell/, "the nozzle must source fluid into the resident level set");
  assert.doesNotMatch(quadtreeSurfaceShader, /volumeIn|loadVolume/);
  assert.match(quadtreeConstructionShader, /fn effectiveWet[\s\S]*return loadAdvancedPhi\(clamp3\(q\)\) < 0\.0/);
  assert.match(quadtreeConstructionShader, /columnProfiles\[leaf \* params\.dims\.y \+ y\]/);
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
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionJacobi/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionLine/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionPolynomialStart/);
  for (const entry of ["applyStepPartial", "applyStepFinalize", "applyStepUpdate", "finishIterationPartial", "finishIterationFinalize", "finishIterationUpdate"]) assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /atomic(Add|Max|Min)/);
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
  assert.match(quadtreeTallCellProjectionShader, /value\[axis\] -= fluidScale \* solvedFaceGradient\(face\)/);
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /cellPressure\(plus\) - cellPressure\(gid\)/);
});

test("monolithic rigid coupling and MLS mapping are wired into the solve", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn coupleReduce/);
  assert.match(quadtreeTallCellProjectionShader, /fn coupleApply/);
  assert.match(quadtreeTallCellProjectionShader, /fn coupleImpulse/);
  assert.match(quadtreeTallCellProjectionShader, /fn mlsRowGradient/);
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
