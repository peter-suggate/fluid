type FluidAdapterLimits = Pick<GPUSupportedLimits,
  | "maxStorageBuffersPerShaderStage"
  | "maxStorageBufferBindingSize"
  | "maxBufferSize"
  | "maxTextureDimension3D"
>;

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
