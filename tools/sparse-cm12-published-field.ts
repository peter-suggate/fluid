import assert from "node:assert/strict";
import { unpackFineLevelSetPackedFlags, unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

// Read the actual sparse presentation payload, independently of diagnostic density.
async function readWords(device: GPUDevice, source: GPUBuffer,
  words: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: Math.max(4, 4 * words),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, 4 * words);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

export async function readPublishedCM12Field(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver): Promise<{
    readonly values: Float32Array;readonly floorContinuation: Uint8Array;
  }> {
  const source = solver.globalFineLevelSetSource;
  const { plan } = source;
  assert.equal(plan.fineFactor, 1);
  assert.deepEqual(plan.sampleDimensions,
    [solver.info.nx, solver.info.ny, solver.info.nz]);
  const capacity = plan.maximumResidentBricks;
  const [worklist, metadata, samples] = await Promise.all([
    readWords(device, source.worklist, 7 + capacity),
    readWords(device, source.metadata, 4 * capacity),
    readWords(device, source.samples, plan.payloadCapacityBytes / 4),
  ]);
  const [nx, ny, nz] = plan.sampleDimensions;
  const r = plan.brickResolution;
  const field = new Float32Array(nx * ny * nz).fill(Number.NaN);
  const floorContinuation = new Uint8Array(nx * nz);
  for (let work = 0; work < worklist[1]!; work += 1) {
    const page = worklist[7 + work]!;
    assert.ok(page < capacity);
    const key = metadata[4 * page + 1]!;
    // Sparse CM12 publishes the signed SparseWorld brick key rather than the
    // dense bounded-domain page key used by the older octree publisher.
    const bx = (key & 0x7ff) - 1024;
    const by = ((key >>> 11) & 0x3ff) - 512;
    const bz = ((key >>> 21) & 0x7ff) - 1024;
    for (let local = 0; local < plan.samplesPerBrick; local += 1) {
      const qx = bx * r + local % r;
      const qy = by * r + Math.floor(local / r) % r;
      const qz = bz * r + Math.floor(local / (r * r));
      if (qx < 0 || qy < 0 || qz < 0 || qx >= nx || qy >= ny || qz >= nz) continue;
      const packed = samples[page * plan.samplesPerBrick + local]!;
      const flags = unpackFineLevelSetPackedFlags(packed);
      if ((flags & 1) === 0) continue;
      field[qx + nx * (qy + ny * qz)] = unpackFineLevelSetPackedPhi(packed);
      if (qy === 0 && (flags & 2) !== 0) floorContinuation[qx + nx * qz] = 1;
    }
  }
  return { values: field, floorContinuation };
}

