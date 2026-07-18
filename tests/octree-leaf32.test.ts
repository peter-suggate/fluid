import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { octreeMethod } from "../lib/methods/octree";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

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
  };
}

async function runCalmDeepSolve(device: GPUDevice, maximumLeafSize: "8" | "32") {
  const scene = calmDeepScene();
  const values = Object.fromEntries(octreeMethod.params.map((parameter) => [parameter.key, "default" in parameter ? parameter.default : 0])) as Record<string, string | number | boolean>;
  values.maximumLeafSize = maximumLeafSize;
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
  const owners = new Uint32Array(await readBufferBytes(device, internals.topology, count * 8));
  const pressureA = new Float32Array(await readBufferBytes(device, internals.pressureA, count * 4));
  const pressureB = new Float32Array(await readBufferBytes(device, internals.pressureB, count * 4));
  const compaction = new Uint32Array(await readBufferBytes(device, internals.compaction, 8));
  return { solver, dims, owners, pressureA, pressureB, liquidLeafRows: compaction[0], matrixEntries: compaction[1] };
}

test("maximum leaf 32 coarsens a calm deep interior with intact 2:1 balance and finite pressure", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU octree checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const run32 = await runCalmDeepSolve(device, "32");
    const [nx, ny, nz] = run32.dims;
    const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
    const unpack = (word: number) => [word & 1023, (word >> 10) & 1023, (word >> 20) & 1023] as const;
    const sizeCounts = new Map<number, number>();
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const cell = index(x, y, z);
      const origin = unpack(run32.owners[cell * 2]);
      const size = run32.owners[cell * 2 + 1];
      assert.ok([1, 2, 4, 8, 16, 32].includes(size), `cell ${x},${y},${z} has invalid leaf size ${size}`);
      assert.ok(origin[0] === Math.floor(x / size) * size && origin[1] === Math.floor(y / size) * size && origin[2] === Math.floor(z / size) * size,
        `cell ${x},${y},${z} owner origin ${origin} is not its aligned ${size}-block`);
      sizeCounts.set(size, (sizeCounts.get(size) ?? 0) + 1);
    }
    assert.ok((sizeCounts.get(32) ?? 0) > 0, "the deep calm interior must contain 32-cubed leaves");
    assert.ok((sizeCounts.get(16) ?? 0) > 0, "the graded ladder must contain 16-cubed leaves");
    assert.equal(run32.owners[index(32, 8, 32) * 2 + 1], 32, "the bottom interior tier must coarsen fully to 32");
    // Strict 2:1 balance across every face-adjacent owner pair.
    const sizeAt = (x: number, y: number, z: number) => run32.owners[index(x, y, z) * 2 + 1];
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
    const neighborCounts = new Map<number, Set<number>>();
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const cell = index(x, y, z);
      for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const) {
        const xx = x + dx, yy = y + dy, zz = z + dz;
        if (xx >= nx || yy >= ny || zz >= nz) continue;
        const other = index(xx, yy, zz);
        if (run32.owners[cell * 2] === run32.owners[other * 2]) continue;
        const left = run32.owners[cell * 2], right = run32.owners[other * 2];
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
