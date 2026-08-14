/**
 * Solver storage/work quality, independent of any one method.
 *
 * Quality controls storage and work around the scene-authored lattice.
 * Spatial resolution belongs exclusively to SceneDescription.voxelDomain.
 */
export type GPUQuality = "balanced" | "high" | "ultra";
