type FluidAdapterLimits = Pick<GPUSupportedLimits,
  | "maxStorageBuffersPerShaderStage"
  | "maxStorageBufferBindingSize"
  | "maxBufferSize"
  | "maxTextureDimension3D"
  | "maxColorAttachmentBytesPerSample"
>;

/**
 * Raster-assisted primary visibility writes packed surface (16 B), identity and
 * media (8 B), exact primary geometry (16 B) and opaque identity (8 B) as four
 * depth-tested colour attachments, which is 16 B past the WebGPU default.
 */
export const FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE = 48;
const DEFAULT_COLOR_ATTACHMENT_BYTES_PER_SAMPLE = 32;

/**
 * Limits whose adapter values are required by the large/sparse fluid paths.
 *
 * WebGPU devices otherwise expose conservative defaults even when the
 * adapter supports more. Requesting the adapter's advertised values is both
 * portable (the request is already clamped to that adapter) and necessary for
 * the nine-storage-buffer sparse-band layout and buffers larger than 128 MiB.
 */
export function requiredFluidDeviceLimits(limits: FluidAdapterLimits): Record<string, GPUSize64> {
  return {
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxBufferSize: limits.maxBufferSize,
    maxTextureDimension3D: limits.maxTextureDimension3D,
    // Requested, never assumed: adapters that cannot grant this still produce a
    // valid device, and the raster-primary renderer fails closed against the
    // granted device limit rather than against this request.
    maxColorAttachmentBytesPerSample: Math.max(
      DEFAULT_COLOR_ATTACHMENT_BYTES_PER_SAMPLE,
      Math.min(
        limits.maxColorAttachmentBytesPerSample ?? DEFAULT_COLOR_ATTACHMENT_BYTES_PER_SAMPLE,
        FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE,
      ),
    ),
  };
}

export const FLUID_REDUCTION_BYTES_PER_LANE = 32;
export const FLUID_M1_MAX_REDUCTION_LANES = 128;

/**
 * Fail-closed capability gate for the sole measured M1 Max reduction shape.
 */
export function supportsFluidM1MaxReduction(
  limits: Pick<GPUSupportedLimits,
    "maxComputeInvocationsPerWorkgroup" | "maxComputeWorkgroupStorageSize">,
): boolean {
  return FLUID_M1_MAX_REDUCTION_LANES <= limits.maxComputeInvocationsPerWorkgroup
    && FLUID_M1_MAX_REDUCTION_LANES * FLUID_REDUCTION_BYTES_PER_LANE
      <= limits.maxComputeWorkgroupStorageSize;
}
