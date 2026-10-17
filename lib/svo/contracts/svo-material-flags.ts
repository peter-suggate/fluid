/** Dependency-free flag bits shared with the surface extractor. */
export const SVO_MATERIAL_FLAGS = Object.freeze({
  opaque: 1 << 0,
  dielectric: 1 << 1,
  /** Primary surface is reconstructed from an independent imported voxel field. */
  sampledVolume: 1 << 2,
} as const);
