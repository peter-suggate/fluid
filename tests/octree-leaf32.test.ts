import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { octreeMethod } from "../lib/methods/octree";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { decodeOctreeOwnerWord, encodeOctreeOwnerWord, planOctreeLeafFrontierAllocation, planOctreeOwnerAllocation } from "../lib/webgpu-octree";
import { WebGPUUniformEulerianSolver } from "../lib/webgpu-uniform-eulerian";

const modulePath = process.env.WEBGPU_NODE_MODULE;

/**
 * A calm deep pool sized for the raised 32-cubed leaf cap: 64x96x64 cells of
 * 0.025 m with the surface at 72 cells. Below the graded band the interior
 * must coarsen through 16-cubed into full 32-cubed leaves while preserving
 * strict 2:1 balance (and therefore the fixed 24-entry matrix row budget).
 */
function calmDeepScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "test-octree-leaf32-calm-deep";
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 1.6, height_m: 2.4, depth_m: 1.6, fillFraction: 0.75, top: "open", fluidWallMode: "no-slip" };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.numerics.surfaceColumnsOverride = 4096;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.005;
  return scene;
}

async function createDevice() {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredFeatures: optionalFluidDeviceFeatures(adapter.features),
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  return { device, validationErrors };
}

async function readBufferBytes(device: GPUDevice, source: GPUBuffer, byteLength: number) {
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0);
  readback.unmap();
  readback.destroy();
  return bytes;
}

interface OctreeInternals {
  octreeProjection: {
    topology: GPUBuffer;
    pressureA: GPUBuffer;
    pressureB: GPUBuffer;
    compaction: GPUBuffer;
    resources: { velocityOut: GPUTexture };
  };
}

async function readVelocityTexture(device: GPUDevice, texture: GPUTexture, dims: readonly [number, number, number]) {
  const bytesPerRow = Math.ceil((dims[0] * 16) / 256) * 256;
  const byteLength = bytesPerRow * dims[1] * dims[2];
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow, rowsPerImage: dims[1] }, dims);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap(); readback.destroy();
  return values;
}

async function runCalmDeepSolve(device: GPUDevice, maximumLeafSize: "8" | "32") {
  const scene = calmDeepScene();
  const values = Object.fromEntries(octreeMethod.params.map((parameter) => [parameter.key, "default" in parameter ? parameter.default : 0])) as Record<string, string | number | boolean>;
  values.maximumLeafSize = maximumLeafSize;
  // This test reads the legacy dense projection texture directly. Keep that
  // compatibility representation authoritative rather than the compact faces.
  values.faceVelocityTransport = "off";
  values.secondaryParticles = "off";
  const solver = octreeMethod.createSolver!(device, scene, "balanced", values, undefined) as unknown as {
    advanceTo(time_s: number, bodies: never[]): boolean;
    info: { nx: number; ny: number; nz: number };
    destroy(): void;
  };
  const dims = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
  assert.deepEqual(dims, [64, 96, 64], "the calm deep test grid must resolve exactly");
  const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));
  const dt = scene.numerics.fixedDt_s!;
  for (let step = 0; step < 2; step += 1) {
    while (!solver.advanceTo((step + 1) * dt, [])) await nextTurn();
  }
  await device.queue.onSubmittedWorkDone();
  const internals = (solver as unknown as OctreeInternals).octreeProjection;
  const count = dims[0] * dims[1] * dims[2];
  const owners = new Uint32Array(await readBufferBytes(device, internals.topology, count * 4));
  const pressureA = new Float32Array(await readBufferBytes(device, internals.pressureA, internals.pressureA.size));
  const pressureB = new Float32Array(await readBufferBytes(device, internals.pressureB, internals.pressureB.size));
  const velocity = await readVelocityTexture(device, internals.resources.velocityOut, dims);
  const compaction = new Uint32Array(await readBufferBytes(device, internals.compaction, 8));
  return { solver, dims, owners, pressureA, pressureB, velocity, liquidLeafRows: compaction[0], matrixEntries: compaction[1] };
}

test("maximum leaf 32 coarsens a calm deep interior with intact 2:1 balance and finite pressure", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU octree checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const run32 = await runCalmDeepSolve(device, "32");
    const [nx, ny, nz] = run32.dims;
    const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
    const ownerAt = (x: number, y: number, z: number) => decodeOctreeOwnerWord(run32.owners[index(x, y, z)], [x, y, z]);
    const sizeCounts = new Map<number, number>();
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const cell = index(x, y, z);
      const { origin, size } = decodeOctreeOwnerWord(run32.owners[cell], [x, y, z]);
      assert.ok([1, 2, 4, 8, 16, 32].includes(size), `cell ${x},${y},${z} has invalid leaf size ${size}`);
      assert.ok(origin[0] === Math.floor(x / size) * size && origin[1] === Math.floor(y / size) * size && origin[2] === Math.floor(z / size) * size,
        `cell ${x},${y},${z} owner origin ${origin} is not its aligned ${size}-block`);
      sizeCounts.set(size, (sizeCounts.get(size) ?? 0) + 1);
    }
    assert.ok((sizeCounts.get(32) ?? 0) > 0, "the deep calm interior must contain 32-cubed leaves");
    assert.ok((sizeCounts.get(16) ?? 0) > 0, "the graded ladder must contain 16-cubed leaves");
    assert.equal(ownerAt(32, 8, 32).size, 32, "the bottom interior tier must coarsen fully to 32");
    // Strict 2:1 balance across every face-adjacent owner pair.
    const sizeAt = (x: number, y: number, z: number) => ownerAt(x, y, z).size;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const size = sizeAt(x, y, z);
      for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const) {
        const xx = x + dx, yy = y + dy, zz = z + dz;
        if (xx >= nx || yy >= ny || zz >= nz) continue;
        const neighbor = sizeAt(xx, yy, zz);
        const ratio = Math.max(size, neighbor) / Math.min(size, neighbor);
        assert.ok(ratio <= 2, `2:1 balance violated at ${x},${y},${z}: ${size} vs ${neighbor}`);
      }
    }
    // Distinct face-neighbor leaves per leaf must fit the fixed 24-entry row
    // budget (6 faces x at most 4 subfaces under 2:1) with no saturation.
    const neighborCounts = new Map<string, Set<string>>();
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const cell = index(x, y, z);
      for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const) {
        const xx = x + dx, yy = y + dy, zz = z + dz;
        if (xx >= nx || yy >= ny || zz >= nz) continue;
        const other = index(xx, yy, zz);
        const leftOwner = decodeOctreeOwnerWord(run32.owners[cell], [x, y, z]);
        const rightOwner = decodeOctreeOwnerWord(run32.owners[other], [xx, yy, zz]);
        const left = leftOwner.origin.join(","), right = rightOwner.origin.join(",");
        if (left === right) continue;
        if (!neighborCounts.has(left)) neighborCounts.set(left, new Set());
        if (!neighborCounts.has(right)) neighborCounts.set(right, new Set());
        neighborCounts.get(left)!.add(right);
        neighborCounts.get(right)!.add(left);
      }
    }
    let maximumNeighbors = 0;
    for (const neighbors of neighborCounts.values()) maximumNeighbors = Math.max(maximumNeighbors, neighbors.size);
    assert.ok(maximumNeighbors <= 24, `a leaf has ${maximumNeighbors} distinct face neighbors, beyond the 24-entry row budget`);
    // The pressure solve must stay finite and produce a genuine field.
    let maximumMagnitude = 0;
    for (const field of [run32.pressureA, run32.pressureB]) for (const value of field) {
      assert.ok(Number.isFinite(value), "pressure must remain finite at maximum leaf 32");
      maximumMagnitude = Math.max(maximumMagnitude, Math.abs(value));
    }
    let maximumVelocity = 0;
    for (let cell = 0; cell < nx * ny * nz; cell += 1) for (let component = 0; component < 3; component += 1) {
      const value = run32.velocity[cell * 4 + component];
      assert.ok(Number.isFinite(value), "sparse frontier projection must publish finite dense-compatible velocity");
      maximumVelocity = Math.max(maximumVelocity, Math.abs(value));
    }
    assert.ok(maximumVelocity > 0 && maximumVelocity <= 50.001, "projected velocity must be nonzero and respect the solver clamp");
    assert.ok(maximumMagnitude > 0, "the calm pool must carry a non-trivial (hydrostatic) pressure field");
    assert.ok(run32.liquidLeafRows > 0, "the compacted solve must emit liquid leaf rows");
    run32.solver.destroy();

    // The same pool at maximum leaf 8 for the degrees-of-freedom comparison.
    const run8 = await runCalmDeepSolve(device, "8");
    assert.ok(run8.liquidLeafRows > run32.liquidLeafRows,
      `leaf 32 must reduce liquid pressure unknowns (leaf8 ${run8.liquidLeafRows} vs leaf32 ${run32.liquidLeafRows})`);
    console.log(JSON.stringify({
      phase: "leaf32-dof-comparison", grid: run32.dims,
      leaf8: { liquidLeafRows: run8.liquidLeafRows, matrixEntries: run8.matrixEntries },
      leaf32: { liquidLeafRows: run32.liquidLeafRows, matrixEntries: run32.matrixEntries },
      rowReduction: Number((1 - run32.liquidLeafRows / run8.liquidLeafRows).toFixed(4)),
      leafSizeCensus: Object.fromEntries([...sizeCounts.entries()].sort((a, b) => a[0] - b[0]).map(([size, cells]) => [size, cells / (size ** 3)]))
    }));
    run8.solver.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});

test("packed owner authority removes one persistent word per finest cell", () => {
  const ocean = planOctreeOwnerAllocation(320 * 96 * 80);
  assert.equal(ocean.allocatedBytes, 9_830_400);
  assert.equal(ocean.legacyDenseBytes, 19_660_800);
  assert.equal(ocean.savedBytes, 9_830_400);
  for (const size of [1, 2, 4, 8, 16, 32]) {
    const origin = [1_500_032 - size, 1_250_016 - size, 1_100_000 - size] as const;
    const cell = [origin[0] + size - 1, origin[1] + size - 1, origin[2] + size - 1] as const;
    assert.deepEqual(decodeOctreeOwnerWord(encodeOctreeOwnerWord(origin, size), cell), { origin: [...origin], size });
  }
});

test("compact frontier overflow fails closed with bounded lists and a dense origin map", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU octree checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const scene = calmDeepScene();
    const solver = new WebGPUUniformEulerianSolver(device, scene, "balanced", undefined, {
      secondaryParticles: false,
      octree: {
        pressureIterations: 8,
        faceVelocityTransport: true,
        maximumLeafSize: 32,
        adaptivity: 1,
        interfaceRefinementBandCells: 4,
        pressureRowCapacity: 1,
      },
    });
    const projection = (solver as unknown as {
      octreeProjection: {
        leafFrontier: GPUBuffer;
        info: {
          pressureRowCapacity: number;
          pressureCapacityOverflow?: boolean;
          frontierListCapacity: number;
          frontierRequiredLeaves?: number;
          frontierCapacityOverflow?: boolean;
        };
        readSolveDiagnostics(): Promise<void>;
      };
    }).octreeProjection;
    const count = solver.info.nx * solver.info.ny * solver.info.nz;
    assert.equal(projection.info.pressureRowCapacity, 256, "the explicit row override remains scan-block aligned");
    assert.equal(projection.leafFrontier.size, planOctreeLeafFrontierAllocation(count, 256, true).allocatedBytes);
    const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));
    while (!solver.advanceTo(2 * scene.numerics.fixedDt_s!, [])) await nextTurn();
    await device.queue.onSubmittedWorkDone();
    await projection.readSolveDiagnostics();
    assert.equal(projection.info.frontierListCapacity, 256);
    assert.equal(projection.info.frontierCapacityOverflow, true);
    assert.equal(projection.info.pressureCapacityOverflow, true, "frontier overflow must suppress the pressure solve");
    assert.ok((projection.info.frontierRequiredLeaves ?? 0) > projection.info.frontierListCapacity);
    solver.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});
