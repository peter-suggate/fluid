import {
  reconstructSvoGBufferWorldPosition,
  SVO_GBUFFER_FLAGS,
  SVO_GBUFFER_PRECISION,
  unpackSvoGBufferPixel,
  type SvoPackedGBufferPixel,
} from "./svo-gbuffer";
import type { SvoVec3 } from "./webgpu-svo-traversal";
import type { SparseVoxelGBufferTextures } from "./webgpu-svo-gbuffer-targets";

export const SVO_GPU_PICKING_READBACK_SLOTS = 3;
export const SVO_GPU_PICKING_BYTES_PER_ROW = 256;
export const SVO_GPU_PICKING_BUFFER_BYTES = 3 * SVO_GPU_PICKING_BYTES_PER_ROW;
export const SVO_GPU_PICKING_OFFSETS = Object.freeze({ radianceDepth: 0, packedSurface: 256, identityMedia: 512 } as const);

export interface SvoGpuPickingRequest {
  pixelX: number;
  pixelY: number;
  rayOrigin_m: SvoVec3;
  rayDirection: SvoVec3;
  rigidBodyCount: number;
  materialCount: number;
  frameToken: number;
}

export type SvoGpuPickingReadbackResult =
  | {
    status: "hit";
    bodyIndex: number;
    materialId: number;
    localTopologyGeneration: number;
    depth_m: number;
    position_m: SvoVec3;
    geometricNormal: SvoVec3;
  }
  | { status: "miss"; reason: "background" | "non-interactive-owner" }
  | { status: "busy" }
  | { status: "invalid"; reason: "coordinates" | "identity" | "depth" | "generation" | "malformed-gbuffer" };

function finiteVec3(value: SvoVec3): boolean {
  return value.length === 3 && value.every(Number.isFinite);
}

export function svoPickingPixelFromNormalized(
  normalizedX: number,
  normalizedY: number,
  width: number,
  height: number,
): readonly [number, number] | undefined {
  if (![normalizedX, normalizedY].every(Number.isFinite)
      || !Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1
      || normalizedX < 0 || normalizedX > 1 || normalizedY < 0 || normalizedY > 1) return undefined;
  return [Math.min(width - 1, Math.floor(normalizedX * width)), Math.min(height - 1, Math.floor(normalizedY * height))];
}

/** CPU decode/validation mirror, separated from WebGPU so malformed samples are deterministic in tests. */
export function decodeSvoGpuPickingSample(
  packed: SvoPackedGBufferPixel,
  request: SvoGpuPickingRequest,
): SvoGpuPickingReadbackResult {
  if (!Number.isSafeInteger(request.rigidBodyCount) || request.rigidBodyCount < 0 || request.rigidBodyCount > 0xffff
      || !Number.isSafeInteger(request.materialCount) || request.materialCount < 1 || request.materialCount > 0xffff
      || !Number.isSafeInteger(request.frameToken) || request.frameToken < 1
      || !finiteVec3(request.rayOrigin_m) || !finiteVec3(request.rayDirection)
      || !(Math.hypot(...request.rayDirection) > 1e-12)) return { status: "invalid", reason: "identity" };
  let pixel;
  try { pixel = unpackSvoGBufferPixel(packed); }
  catch { return { status: "invalid", reason: "malformed-gbuffer" }; }
  if (pixel.status === "miss") return { status: "miss", reason: "background" };
  const flags = pixel.additionalFlags ?? 0;
  if ((flags & SVO_GBUFFER_FLAGS.validSurface) === 0 || (flags & SVO_GBUFFER_FLAGS.depthValid) === 0
      || pixel.ownerId === 0xffff || pixel.ownerId >= request.rigidBodyCount) {
    return { status: "miss", reason: "non-interactive-owner" };
  }
  if (pixel.materialId >= request.materialCount) return { status: "invalid", reason: "identity" };
  if (!Number.isSafeInteger(pixel.localTopologyGeneration) || pixel.localTopologyGeneration < 1) {
    return { status: "invalid", reason: "generation" };
  }
  if (!Number.isFinite(pixel.depth_m) || pixel.depth_m <= 0 || pixel.depth_m > SVO_GBUFFER_PRECISION.maximumLinearDepth_m) {
    return { status: "invalid", reason: "depth" };
  }
  return {
    status: "hit",
    bodyIndex: pixel.ownerId,
    materialId: pixel.materialId,
    localTopologyGeneration: pixel.localTopologyGeneration,
    depth_m: pixel.depth_m,
    position_m: reconstructSvoGBufferWorldPosition(request.rayOrigin_m, request.rayDirection, pixel.depth_m),
    geometricNormal: pixel.geometricNormal,
  };
}

interface ReadbackSlot { buffer: GPUBuffer; pending: boolean }

/** Three request slots bound outstanding map work without blocking presentation or allocating per click. */
export class SparseVoxelGpuPickingReadbackRing {
  private readonly slots: ReadbackSlot[];
  private destroyed = false;

  constructor(private readonly device: GPUDevice) {
    this.slots = Array.from({ length: SVO_GPU_PICKING_READBACK_SLOTS }, (_, index) => ({
      buffer: device.createBuffer({
        label: `Sparse voxel pick readback ${index + 1}/${SVO_GPU_PICKING_READBACK_SLOTS}`,
        size: SVO_GPU_PICKING_BUFFER_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: false,
    }));
  }

  async pick(
    radianceDepth: GPUTexture,
    gBuffer: SparseVoxelGBufferTextures,
    request: SvoGpuPickingRequest,
    isFrameCurrent: () => boolean,
  ): Promise<SvoGpuPickingReadbackResult> {
    if (this.destroyed) return { status: "invalid", reason: "generation" };
    if (!Number.isSafeInteger(request.pixelX) || !Number.isSafeInteger(request.pixelY)
        || request.pixelX < 0 || request.pixelY < 0 || request.pixelX >= gBuffer.width || request.pixelY >= gBuffer.height) {
      return { status: "invalid", reason: "coordinates" };
    }
    const slot = this.slots.find((candidate) => !candidate.pending);
    if (!slot) return { status: "busy" };
    slot.pending = true;
    const origin = { x: request.pixelX, y: request.pixelY, z: 0 };
    const copySize = { width: 1, height: 1, depthOrArrayLayers: 1 };
    const encoder = this.device.createCommandEncoder({ label: "Read one sparse voxel G-buffer pick sample" });
    encoder.copyTextureToBuffer({ texture: radianceDepth, origin }, {
      buffer: slot.buffer, offset: SVO_GPU_PICKING_OFFSETS.radianceDepth,
      bytesPerRow: SVO_GPU_PICKING_BYTES_PER_ROW, rowsPerImage: 1,
    }, copySize);
    encoder.copyTextureToBuffer({ texture: gBuffer.packedSurface, origin }, {
      buffer: slot.buffer, offset: SVO_GPU_PICKING_OFFSETS.packedSurface,
      bytesPerRow: SVO_GPU_PICKING_BYTES_PER_ROW, rowsPerImage: 1,
    }, copySize);
    encoder.copyTextureToBuffer({ texture: gBuffer.identityMedia, origin }, {
      buffer: slot.buffer, offset: SVO_GPU_PICKING_OFFSETS.identityMedia,
      bytesPerRow: SVO_GPU_PICKING_BYTES_PER_ROW, rowsPerImage: 1,
    }, copySize);
    this.device.queue.submit([encoder.finish()]);
    try {
      await slot.buffer.mapAsync(GPUMapMode.READ);
      if (this.destroyed || !isFrameCurrent()) return { status: "invalid", reason: "generation" };
      const bytes = slot.buffer.getMappedRange();
      const packed: SvoPackedGBufferPixel = {
        radianceDepth: Uint16Array.from(new Uint16Array(bytes, SVO_GPU_PICKING_OFFSETS.radianceDepth, 4)),
        packedSurface: Uint32Array.from(new Uint32Array(bytes, SVO_GPU_PICKING_OFFSETS.packedSurface, 4)),
        identityMedia: Uint16Array.from(new Uint16Array(bytes, SVO_GPU_PICKING_OFFSETS.identityMedia, 4)),
        debugSidecar: new Uint32Array(4),
      };
      return decodeSvoGpuPickingSample(packed, request);
    } catch {
      return { status: "invalid", reason: "malformed-gbuffer" };
    } finally {
      try { slot.buffer.unmap(); } catch { /* Device loss or destruction. */ }
      slot.pending = false;
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const slot of this.slots) { try { slot.buffer.destroy(); } catch { /* Device loss. */ } }
  }
}
