type FluidAdapterLimits = Pick<GPUSupportedLimits,
  | "maxStorageBuffersPerShaderStage"
  | "maxStorageBufferBindingSize"
  | "maxBufferSize"
  | "maxTextureDimension2D"
  | "maxTextureDimension3D"
  | "maxSampledTexturesPerShaderStage"
  | "maxStorageTexturesPerShaderStage"
  | "maxColorAttachmentBytesPerSample"
  | "maxComputeInvocationsPerWorkgroup"
  | "maxComputeWorkgroupSizeX"
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
    // The static node-mip directory is one texture row per page, so this limit
    // is the pyramid's page ceiling. The 8192 default is below what a
    // room-enclosed scene needs (hose-tank: 10361), and falling short does not
    // degrade the pyramid, it withdraws it — cone lighting then falls back to
    // exact traversal for the whole frame.
    maxTextureDimension2D: limits.maxTextureDimension2D,
    maxTextureDimension3D: limits.maxTextureDimension3D,
    // The Phase-1 voxel-light lane is the seventeenth sampled texture in the
    // deferred fragment stage. Request the adapter value; capability checks in
    // the renderer still fail closed on devices that cannot expose it.
    maxSampledTexturesPerShaderStage: limits.maxSampledTexturesPerShaderStage,
    // W2's compute resolve storage-writes the four G-buffer colour planes plus
    // a transient r32float depth plane. Request the adapter's advertised value;
    // adapters below five keep the exact fragment resolve fallback.
    maxStorageTexturesPerShaderStage: limits.maxStorageTexturesPerShaderStage,
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
    // The persistent MGPCG solve is ONE dispatch of ONE workgroup, so the width
    // of that workgroup is the only occupancy lever WGSL can express without a
    // cross-workgroup barrier. It was capped at 256 lanes -- not by the M1 Max,
    // which hosts 1024 threads per threadgroup, but because this function never
    // asked for the adapter's value and WebGPU's default is 256. A 256-lane
    // ceiling on a latency-bound kernel was therefore a limit nobody raised.
    // Requesting the advertised value is the same portable move the six limits
    // above make; every consumer still fails closed against the GRANTED limit,
    // and `WebGPUOctreePersistentMGPCG` does exactly that before any pipeline
    // exists.
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    // A wider workgroup also needs the X extent raised; its default is 256 too,
    // and a @workgroup_size the device cannot host is a pipeline-creation
    // failure rather than a silent clamp.
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
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
