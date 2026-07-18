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

/**
 * Optional features the fluid paths exploit when the adapter offers them.
 * float32-filterable enables hardware trilinear sampling of the r32float /
 * rgba32float brick-atlas tiles; without it the atlas helpers fall back to
 * manual trilinear interpolation.
 */
export function optionalFluidDeviceFeatures(features: GPUSupportedFeatures): GPUFeatureName[] {
  return (["float32-filterable"] as GPUFeatureName[]).filter((feature) => features.has(feature));
}
