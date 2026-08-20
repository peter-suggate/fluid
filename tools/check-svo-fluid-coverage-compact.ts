#!/usr/bin/env node
/**
 * Prove the fluid coverage volume reads a compact publication.
 *
 * The dense arm was the only one for a long time, and on a buffer-native solver
 * it was silently pointed at a 1x1x1 compatibility stand-in: every fetch went
 * out of bounds, returned zero, and quantized to half coverage across the whole
 * domain — a uniform fog that no gate noticed because the volume was "valid" and
 * the shader that consumed it was itself unreachable. Nothing about that failure
 * is visible from CPU state, so this check publishes a synthetic page set with a
 * known signed distance and compares the GPU volume against the CPU coverage
 * reference texel by texel.
 *
 * It also covers the two answers that are not "read the sample": a logical page
 * that was never published must resolve to air through the dry-apron fallback,
 * and a sample whose VALID bit is clear must not contribute.
 */
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  packFineLevelSetSample,
  unpackFineLevelSetPackedPhi,
} from "../lib/core/fine-levelset-packed-sample";
import {
  quantizeSvoFluidCoverage,
  svoFluidCoverageFromSignedDistance,
  SVO_FLUID_COVERAGE_LAYOUT,
} from "../lib/svo/svo-fluid-coverage";
import {
  WebGpuSvoFluidCoverage,
  type WebGpuSvoFluidCoverageCompactSource,
} from "../lib/svo/webgpu-svo-fluid-coverage";

const BRICK_RESOLUTION = 4;
const SAMPLES_PER_BRICK = BRICK_RESOLUTION ** 3;
const LATTICE = 16;
const BRICKS_PER_AXIS = LATTICE / BRICK_RESOLUTION;
const PAGE_COUNT = BRICKS_PER_AXIS ** 3;
const CELLS_PER_TEXEL = 2;
/**
 * The publication's own cell width — the metric its stored `phi` is measured in,
 * and the only thing the coverage band may be derived from.
 *
 * Deliberately different from the world cell size below. Sparse CM12 publishes a
 * lattice in index space (`fineCellWidth: 1`) while the volume is placed in
 * world metres, so the two frames genuinely differ in production. Holding them
 * equal here is what let a real inversion — world diagonal over a publication
 * phi — pass this gate byte-exact while reading as air on screen.
 */
const CELL_WIDTH_M = 0.02;
/** The world box the volume occupies: a different origin and a different scale. */
const WORLD_ORIGIN_M = [-0.64, 0, 1.28] as const;
const WORLD_CELL_SIZE_M = 0.05;
const GENERATION = 9;
/** Half the lattice height, in publication units: water fills everything below. */
const SURFACE_M = LATTICE * CELL_WIDTH_M * 0.5;
/** One page left unpublished, and one sample published with VALID clear. */
const OMITTED_KEY = 5;
const INVALID_SAMPLE_CELL = [1, 1, 1] as const;

/** Signed distance to the flat water surface; negative below, in metres. */
function signedDistanceAt(cell: readonly [number, number, number]): number {
  return (cell[1] + 0.5) * CELL_WIDTH_M - SURFACE_M;
}

function pageKeyOf(cell: readonly [number, number, number]): number {
  const page = cell.map((value) => Math.floor(value / BRICK_RESOLUTION));
  return page[0]! + BRICKS_PER_AXIS * (page[1]! + BRICKS_PER_AXIS * page[2]!);
}

interface Publication {
  worklist: Uint32Array;
  metadata: Uint32Array;
  samples: Uint32Array;
  /** Keys present in the page set, in the sorted order the lookup binary-searches. */
  keys: number[];
}

/**
 * Build the page set exactly as a compact publisher does: a validated worklist
 * header with the compact flag, key-sorted metadata, and packed samples.
 */
function publish(): Publication {
  const keys: number[] = [];
  for (let key = 0; key < PAGE_COUNT; key += 1) if (key !== OMITTED_KEY) keys.push(key);
  const count = keys.length;

  const worklist = new Uint32Array(7 + count);
  worklist[0] = GENERATION;
  worklist[1] = count;
  worklist[2] = PAGE_COUNT;
  // Bits 0-1 are the published transaction, bit 31 the compact publisher flag,
  // bits 16-20 the native fine resolution and bits 8-12 the largest span log —
  // zero here, because every page in this fixture is an ordinary 4^3 page.
  worklist[3] = 0x8000_0000 | 3 | (BRICK_RESOLUTION << 16);
  worklist[4] = Math.ceil(count / 64);
  worklist[5] = 1;
  worklist[6] = 1;
  for (let index = 0; index < count; index += 1) worklist[7 + index] = index;

  const metadata = new Uint32Array(4 * PAGE_COUNT);
  for (let index = 0; index < count; index += 1) {
    metadata[4 * index] = index;
    metadata[4 * index + 1] = keys[index]!;
    metadata[4 * index + 2] = GENERATION;
    metadata[4 * index + 3] = 0;
  }

  const samples = new Uint32Array(PAGE_COUNT * SAMPLES_PER_BRICK);
  for (let z = 0; z < LATTICE; z += 1) {
    for (let y = 0; y < LATTICE; y += 1) {
      for (let x = 0; x < LATTICE; x += 1) {
        const cell = [x, y, z] as const;
        const page = keys.indexOf(pageKeyOf(cell));
        if (page < 0) continue;
        const local = cell.map((value) => value % BRICK_RESOLUTION);
        const index = page * SAMPLES_PER_BRICK
          + local[0]! + BRICK_RESOLUTION * (local[1]! + BRICK_RESOLUTION * local[2]!);
        const valid = cell.every((value, axis) => value === INVALID_SAMPLE_CELL[axis]) ? 0 : 1;
        samples[index] = packFineLevelSetSample(signedDistanceAt(cell), valid);
      }
    }
  }
  return { worklist, metadata, samples, keys };
}

/** What the GPU volume must hold, computed the way the shader is specified to. */
function expectedCoverage(publication: Publication, dimensions: readonly number[]): Float64Array {
  const diagonal = Math.hypot(CELL_WIDTH_M, CELL_WIDTH_M, CELL_WIDTH_M);
  const expected = new Float64Array(dimensions[0]! * dimensions[1]! * dimensions[2]!);
  for (let tz = 0; tz < dimensions[2]!; tz += 1) {
    for (let ty = 0; ty < dimensions[1]!; ty += 1) {
      for (let tx = 0; tx < dimensions[0]!; tx += 1) {
        let total = 0;
        for (let z = 0; z < CELLS_PER_TEXEL; z += 1) {
          for (let y = 0; y < CELLS_PER_TEXEL; y += 1) {
            for (let x = 0; x < CELLS_PER_TEXEL; x += 1) {
              const cell = [
                tx * CELLS_PER_TEXEL + x, ty * CELLS_PER_TEXEL + y, tz * CELLS_PER_TEXEL + z,
              ] as const;
              // A page the publisher omitted is authoritative air under the dry
              // apron, and a sample without its VALID bit is not an answer.
              if (!publication.keys.includes(pageKeyOf(cell))) continue;
              if (cell.every((value, axis) => value === INVALID_SAMPLE_CELL[axis])) continue;
              // Storage is binary16, so the reference must round-trip too or the
              // comparison measures the format instead of the fill.
              const stored = unpackFineLevelSetPackedPhi(
                packFineLevelSetSample(signedDistanceAt(cell), 1),
              );
              total += svoFluidCoverageFromSignedDistance(stored, diagonal);
            }
          }
        }
        const index = tx + dimensions[0]! * (ty + dimensions[1]! * tz);
        expected[index] = total / CELLS_PER_TEXEL ** 3;
      }
    }
  }
  return expected;
}

function storageBuffer(device: GPUDevice, label: string, words: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({
    label, size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, words.buffer as ArrayBuffer);
  return buffer;
}

async function main(): Promise<void> {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
  const publication = publish();

  await acquireWebGPUExclusiveLock("wgsl-check", "svo-fluid-coverage-compact");
  try {
    const { create, globals } = await import(dawnModule) as {
      create: (flags: string[]) => GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");

    const source: WebGpuSvoFluidCoverageCompactSource = {
      kind: "compact",
      worklist: { buffer: storageBuffer(device, "fixture worklist", publication.worklist) },
      metadata: { buffer: storageBuffer(device, "fixture metadata", publication.metadata) },
      samples: { buffer: storageBuffer(device, "fixture samples", publication.samples) },
      sampleDimensions: [LATTICE, LATTICE, LATTICE],
      brickDimensions: [BRICKS_PER_AXIS, BRICKS_PER_AXIS, BRICKS_PER_AXIS],
      brickResolution: BRICK_RESOLUTION,
      samplesPerBrick: SAMPLES_PER_BRICK,
      pageCapacity: PAGE_COUNT,
      fineFactor: 1,
      fineCellWidth: CELL_WIDTH_M,
      domainOrigin: [0, 0, 0],
      generation: GENERATION,
    };
    const coverage = new WebGpuSvoFluidCoverage(device, {
      fieldDimensions: [LATTICE, LATTICE, LATTICE],
      worldOrigin_m: [...WORLD_ORIGIN_M],
      cellSize_m: [WORLD_CELL_SIZE_M, WORLD_CELL_SIZE_M, WORLD_CELL_SIZE_M],
      cellsPerTexel: CELLS_PER_TEXEL,
    }, source);
    await coverage.initializePipelines();

    // The box the volume occupies is the world's, not the publication's. Pinned
    // here because a misplaced box still samples cleanly and still reports a
    // valid frame: it simply puts the shadow somewhere no receiver is standing.
    const boxOrigin = coverage.plan.origin_m;
    const boxTexel = coverage.plan.texelSize_m;
    if (WORLD_ORIGIN_M.some((value, axis) => boxOrigin[axis] !== value)) {
      throw new Error(`coverage box origin ${boxOrigin.join(",")} is not the world origin ${WORLD_ORIGIN_M.join(",")}`);
    }
    if (boxTexel.some((value) => Math.abs(value - WORLD_CELL_SIZE_M * CELLS_PER_TEXEL) > 1e-9)) {
      throw new Error(`coverage texel ${boxTexel.join(",")} is not the world cell size times ${CELLS_PER_TEXEL}`);
    }
    const dimensions = coverage.plan.dimensions;
    const encoder = device.createCommandEncoder();
    if (!coverage.encode(encoder)) throw new Error("coverage fill was not encoded");
    // 256-byte row alignment is a copy requirement, so the readback is padded.
    const bytesPerRow = Math.ceil(dimensions[0] * SVO_FLUID_COVERAGE_LAYOUT.bytesPerTexel / 256) * 256;
    const readback = device.createBuffer({
      label: "coverage readback", size: bytesPerRow * dimensions[1] * dimensions[2],
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: (coverage as unknown as { texture: GPUTexture }).texture, mipLevel: 0 },
      { buffer: readback, bytesPerRow, rowsPerImage: dimensions[1] },
      [dimensions[0], dimensions[1], dimensions[2]],
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const scope = await device.popErrorScope();
    if (scope) throw new Error(`GPU validation: ${scope.message}`);

    const expected = expectedCoverage(publication, dimensions);
    let worst = 0, worstAt = "", wet = 0, dry = 0;
    for (let z = 0; z < dimensions[2]; z += 1) {
      for (let y = 0; y < dimensions[1]; y += 1) {
        for (let x = 0; x < dimensions[0]; x += 1) {
          const byte = bytes[z * bytesPerRow * dimensions[1] + y * bytesPerRow
            + x * SVO_FLUID_COVERAGE_LAYOUT.bytesPerTexel + SVO_FLUID_COVERAGE_LAYOUT.coverageChannel]!;
          const reference = quantizeSvoFluidCoverage(expected[x + dimensions[0] * (y + dimensions[1] * z)]!);
          const error = Math.abs(byte / 255 - reference);
          if (error > worst) { worst = error; worstAt = `(${x},${y},${z})`; }
          if (byte > 128) wet += 1; else if (byte === 0) dry += 1;
        }
      }
    }
    // One byte lane, so a single quantization step is the floor on agreement.
    const tolerance = 1.5 / 255;
    if (worst > tolerance) {
      throw new Error(`compact fill disagrees with the CPU reference by ${worst.toFixed(5)} at ${worstAt}`);
    }
    const texels = dimensions[0] * dimensions[1] * dimensions[2];
    // A fill that read nothing would be uniformly dry, and one over a stand-in
    // texture uniformly half-covered; both are excluded by needing each side.
    if (wet < texels / 8 || dry < texels / 8) {
      throw new Error(`degenerate volume: ${wet} wet and ${dry} dry of ${texels} texels`);
    }
    coverage.destroy();
    console.log(`SVO fluid coverage: compact publication resampled over ${dimensions.join("x")}`
      + ` texels (${wet} wet, ${dry} dry, worst error ${(worst * 255).toFixed(2)}/255;`
      + ` omitted page ${OMITTED_KEY} and one invalid sample both read as air)`);
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

void main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
