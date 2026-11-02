/** Production pool-party SVO path, not the historical dry harness defaults.
 * tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/benchmark-svo-main-kernels.ts
 */
process.env.FLUID_SVO_MAIN_KERNEL_STUDY ??= "artifacts/svo-main-kernels";
Object.assign(process.env, {
  FLUID_SVO_DRY_FRAME_SCENE: "backyard-pool-party",
  FLUID_SVO_DRY_FRAME_TRAVERSAL: "raster-primary",
  FLUID_SVO_DRY_FRAME_SURFACE_MESH: "1",
  FLUID_SVO_DRY_FRAME_CONE_TRACING: "raster-ao",
  FLUID_SVO_DRY_FRAME_CONE_SCALE: "1",
  FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY: "bounds",
  FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING: "1",
  FLUID_SVO_DRY_FRAME_SHADING: "split",
  FLUID_SVO_DRY_FRAME_GI: "0",
  FLUID_SVO_DRY_FRAME_TIMING: "gpu",
  FLUID_SVO_DRY_FRAME_WIDTH: process.env.FLUID_SVO_DRY_FRAME_WIDTH ?? "1280",
  FLUID_SVO_DRY_FRAME_HEIGHT: process.env.FLUID_SVO_DRY_FRAME_HEIGHT ?? "720",
});
await import("./benchmark-svo-dry-frame-gpu");
export {};
