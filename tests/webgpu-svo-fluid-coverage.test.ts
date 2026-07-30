import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  planSvoFluidCoverageVolume,
  quantizeSvoFluidCoverage,
  svoFluidCoverageFromSignedDistance,
} from "../lib/svo-fluid-coverage";
import {
  svoFluidCoverageFillShader,
  svoFluidCoverageReduceShader,
  SVO_FLUID_COVERAGE_BINDINGS,
  WebGpuSvoFluidCoverage,
} from "../lib/webgpu-svo-fluid-coverage";

const modulePath = process.env.WEBGPU_NODE_MODULE;

const CELL = 0.05;
const CELL_DIAGONAL = Math.hypot(CELL, CELL, CELL);
const FIELD = [16, 16, 16] as const;
/** Water fills everything below this height, with a metric distance either side. */
const SURFACE_CELL_Y = 6.5;
const signedDistanceAt = (y: number) => (y - SURFACE_CELL_Y) * CELL;

test("the fill reads the coarse level set directly, with no octree or directory", () => {
  const fill = svoFluidCoverageFillShader(2);
  for (const [name, binding] of Object.entries(SVO_FLUID_COVERAGE_BINDINGS)) {
    assert.match(fill, new RegExp(`@binding\\(${binding}\\)`), `fill shader must declare ${name} at binding ${binding}`);
  }
  assert.match(fill, /textureLoad\(coarsePhi, vec3i\(cell\), 0\)\.x/,
    "the coarse level set is a signed distance in lane x");
  assert.match(fill, /texture_storage_3d<rgba8unorm, write>/,
    "core WebGPU guarantees write-only storage for rgba8unorm but not for r8unorm");
  assert.match(fill, /if \(any\(cell >= params\.fieldDimensions\)\) \{ return 0\.0; \}/,
    "cells past the field contribute air rather than clamping the boundary value");
  // The whole point of the rewrite: the fill no longer walks the sparse
  // publication, so none of that vocabulary may reappear here.
  assert.doesNotMatch(fill, /nodes|leaves|fluidLeafStates|publicationState|findLeaf|brickSize/,
    "resampling the coarse field must not reintroduce a traversal");
  assert.match(svoFluidCoverageReduceShader, /total \/ 8\.0/, "reduction is the mean, matching the CPU reference");
});

test("real GPU fill resamples the coarse level set and mips it", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU fluid coverage checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "no WebGPU adapter");
  const device = await adapter.requestDevice();
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });

  // r32float, exactly like the solver's resident level set. A filterable
  // binding would be rejected against it, which is why the fill declares its
  // layout explicitly instead of inferring one.
  const phi = device.createTexture({
    label: "coarse level set", size: [...FIELD], dimension: "3d", format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const values = new Float32Array(FIELD[0] * FIELD[1] * FIELD[2]);
  for (let z = 0; z < FIELD[2]; z += 1) for (let y = 0; y < FIELD[1]; y += 1) for (let x = 0; x < FIELD[0]; x += 1) {
    values[(z * FIELD[1] + y) * FIELD[0] + x] = signedDistanceAt(y);
  }
  device.queue.writeTexture(
    { texture: phi }, values,
    { bytesPerRow: FIELD[0] * 4, rowsPerImage: FIELD[1] },
    [...FIELD],
  );

  const options = {
    fieldDimensions: FIELD,
    worldOrigin_m: [0, 0, 0] as const,
    cellSize_m: [CELL, CELL, CELL] as const,
    cellsPerTexel: 1,
  };
  const coverage = new WebGpuSvoFluidCoverage(device, options, { coarsePhi: phi.createView({ dimension: "3d" }) });
  const plan = planSvoFluidCoverageVolume(options);
  assert.deepEqual([...plan.dimensions], [16, 16, 16]);

  assert.equal(coverage.visibleGeneration(), undefined, "nothing is visible before a fill is encoded");
  const encoder = device.createCommandEncoder();
  coverage.encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  // Surface binding/layout rejections here rather than as a volume of zeros.
  assert.deepEqual(validationErrors, [], "the fill must encode without validation errors");
  const live = coverage.visibleGeneration();
  assert.ok(live, "a fill must publish a generation");
  assert.equal(live.generation, 1);

  const texture = (coverage as unknown as { texture: GPUTexture }).texture;
  // 16 texels of rgba8 per row is below the 256-byte copy alignment, so the
  // staging row stride is padded.
  const bytesPerRow = 256;
  const readLevel = async (level: number, dimensions: readonly number[]) => {
    const readback = device.createBuffer({
      size: bytesPerRow * dimensions[1] * dimensions[2],
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copy = device.createCommandEncoder();
    copy.copyTextureToBuffer({ texture, mipLevel: level }, { buffer: readback, bytesPerRow, rowsPerImage: dimensions[1] }, [...dimensions]);
    device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return (x: number, y: number, z: number) => bytes[(z * dimensions[1] + y) * bytesPerRow + x * 4] / 255;
  };

  const at = await readLevel(0, plan.dimensions);
  for (let z = 0; z < plan.dimensions[2]; z += 1) {
    for (let y = 0; y < plan.dimensions[1]; y += 1) {
      for (let x = 0; x < plan.dimensions[0]; x += 1) {
        const expected = quantizeSvoFluidCoverage(svoFluidCoverageFromSignedDistance(signedDistanceAt(y), CELL_DIAGONAL));
        assert.ok(Math.abs(at(x, y, z) - expected) <= 1 / 255 + 1e-6,
          `texel ${x},${y},${z} expected ${expected.toFixed(3)}, read ${at(x, y, z).toFixed(3)}`);
      }
    }
  }
  // The interface must actually land inside the volume, or the sweep above
  // would pass against a uniformly empty or uniformly full field.
  assert.equal(at(0, 0, 0), 1, "well below the surface is fully covered");
  assert.equal(at(0, plan.dimensions[1] - 1, 0), 0, "well above the surface is empty");
  assert.ok(at(0, 6, 0) > 0 && at(0, 6, 0) < 1, "the interface row is partially covered");

  // Level one proves the reduce chain observed the fill. Each pass is its own
  // usage scope, so the storage-write to level N and the sampled read of it at
  // level N+1 are separated by a pass boundary rather than sharing one.
  const coarseDimensions = plan.dimensions.map((component) => component >> 1);
  const coarseAt = await readLevel(1, coarseDimensions);
  for (let z = 0; z < coarseDimensions[2]; z += 1) {
    for (let y = 0; y < coarseDimensions[1]; y += 1) {
      for (let x = 0; x < coarseDimensions[0]; x += 1) {
        let total = 0;
        for (let index = 0; index < 8; index += 1) {
          total += at(x * 2 + (index & 1), y * 2 + ((index >> 1) & 1), z * 2 + ((index >> 2) & 1));
        }
        assert.ok(Math.abs(coarseAt(x, y, z) - total / 8) <= 2 / 255,
          `level-1 texel ${x},${y},${z} expected ${(total / 8).toFixed(3)}, read ${coarseAt(x, y, z).toFixed(3)}`);
      }
    }
  }

  coverage.destroy();
  assert.deepEqual(validationErrors, []);
});
