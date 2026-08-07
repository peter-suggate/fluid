/** User-facing visibility effects layered over cone-traced global illumination. */
/**
 * How lighting visibility (shadows, AO, GI) is resolved.
 * - `cones`: hierarchical cone marches plus every stage that feeds them — the
 *   reduced-rate cone prepass, sample fan-out, and persistent world-GI cache.
 * - `exact`: no cone stage runs; shadows and AO use the bounded exact SVO
 *   visibility traversals (sharp reference shadows, costlier per pixel).
 * - `off`: no visibility work at all — direct lighting is unshadowed, AO and
 *   GI are absent. Strictly removes work relative to either other mode.
 */
export type SvoConeTracingMode = "cones" | "exact" | "off";

/**
 * How *primary* visibility is resolved, as distinct from the lighting visibility
 * `SvoConeTracingMode` selects.
 * - `raster`: hardware-rasterize the resident bricks as depth-tested proxies.
 *   Octree leaves partition space, so the depth test alone is an exact
 *   visibility oracle and the image matches `traced` pixel for pixel.
 * - `traced`: the full-screen traversal megakernel every pixel marches for
 *   itself. Kept switchable because it is the reference the raster path is
 *   measured against, and because a device too narrow for four depth-tested
 *   colour planes has to fall back to it.
 */
export type SvoPrimaryTraversalMode = "raster" | "traced";

/**
 * Proxies per pixel above which the raster primary stops paying for itself.
 *
 * `raster` emits one 36-vertex box proxy per occupied leaf brick, so its cost
 * tracks total projected proxy *area* rather than pixel count. Measured on the
 * depth-honest smoke lane (2026-08-06, `hero-garden-hose`, 800x460 = 368,000
 * pixels, median of the timed frames):
 *
 * | refinement depth | occupied bricks | proxies/pixel | `raster` | `traced` |
 * |---|---:|---:|---:|---:|
 * | 0 | 16,820  | 0.046 |  50.46 ms | **24.26 ms** |
 * | 1 | 29,720  | 0.081 |  69.57 ms | **27.12 ms** |
 * | 2 | 81,850  | 0.222 | 148.03 ms | **43.12 ms** |
 * | 3 | 267,733 | 0.728 | 246.12 ms | **56.14 ms** |
 *
 * The raster sweep fits `t ~= 88 + 0.038 * bricks^(2/3)` ms — an area law, not
 * a pixel law. That law has survived a scene change: an earlier sweep measured
 * depth 3 at 873,700 bricks / 2.374 proxies per pixel / 428.1 ms, and the same
 * fit predicts 246.0 ms at today's 267,733 bricks against 246.12 measured. So
 * **the brick count is the operating point and no absolute number here means
 * anything without it.** Raster-primary was designed and tuned at depth 0, where
 * there is one brick per 135 pixels; the crossover was never measured.
 *
 * These are clean-lane numbers (2026-08-06), interleaved and fingerprinted. An
 * earlier sweep ran under external GPU contention and read the depth-3 ratio as
 * 6.2x with a non-monotonic 3.4 / 3.1 / 4.6 ladder below it; contention inflates
 * the *longer* arm more, so it biases ratios and not just precision, and that
 * non-monotonicity was the tell.
 *
 * A ratio was the first shape of this rule, and the measurement retired it. The
 * megakernel is not merely past a crossover at depth 3, it is **4.38x** faster
 * there, and it wins at every rung measured — 2.08x / 2.57x / 3.43x / 4.38x,
 * rising monotonically with depth, because it never builds a per-brick anything:
 * one ray per pixel, and the octree descent is logarithmic in the structure the
 * raster pass enumerates linearly. A ceiling only makes sense between two curves
 * that cross. These two do not.
 *
 * `traced` is **not** flat in refinement depth, as this comment once claimed —
 * 24.26 to 56.14 ms is a 2.31x rise across the ladder. It is `raster` that grows
 * faster still, which is what opens the gap.
 *
 * So `traced` is now unconditional and this constant is retained only as the
 * evidence. What it *was* protecting — depths 0-2, where the ratio is 0.007 to
 * 0.326 — is protection against nothing: raster costs 88.5-186.9 ms across that
 * range while traced costs ~57 ms at depth 3 with 320x more bricks, so the arm
 * the ratio was steering toward raster is the slower one at every rung
 * measured. Depths 0-2 are re-timed traced by the sweep this constant links.
 *
 * Resolution was the other open question and it points the same way. Both arms
 * grow with pixel count, but raster carries a vertex/setup term that does not —
 * 31 M vertices at depth 3, paid whether the frame is 368 k pixels or 4 M — so
 * more pixels *amortize* raster's fixed tax rather than escaping it, and the
 * ratio a per-pixel ceiling computes falls exactly where raster's advantage is
 * weakest in absolute terms. Steering by that ratio had production, at 3-4 M
 * pixels, keeping the path the 800x460 lane hands to the megakernel; that was
 * the wrong direction, and it is the reason this is a default rather than a
 * threshold.
 *
 * The image is not identical and the difference is bounded and named: 321 of
 * 368,000 pixels (0.087%) differ by more than 8/255, mean absolute difference
 * 0.054/255, on sub-voxel silhouette edges where `raster` additionally runs
 * `encodeScenePrimitivePrimary` and the megakernel returns at the first voxel.
 * That tier is a real capability and restoring it under `traced` is tracked
 * separately; it is not worth 6.2x.
 */
export const SVO_PRIMARY_RASTER_PROXIES_PER_PIXEL_CEILING = 1;


/**
 * Diagnostic override for the traversal decision, read once at module load.
 *
 * Same idiom as the near-field band's switches (`svo-render-tuning.ts:368`):
 * the acceptance lanes select the traversal through the shared rule, so an A/B
 * over it would otherwise mean editing a constant between arms, and the one
 * measurement rule this program has is that arms are interleaved rather than
 * run in blocks. Forces the arm outright — including forcing `raster` onto a
 * depth the rule would hand to the megakernel, which is the direction that
 * reproduces the pre-change frame.
 */
const traversalEnvironment = typeof process !== "undefined" ? process.env : undefined;
const traversalOverrideRaw = traversalEnvironment?.["FLUID_SVO_PRIMARY_TRAVERSAL"] ?? "";
if (traversalOverrideRaw !== "" && traversalOverrideRaw !== "raster" && traversalOverrideRaw !== "traced") {
  throw new RangeError(`FLUID_SVO_PRIMARY_TRAVERSAL must be raster or traced, got ${traversalOverrideRaw}`);
}
export const SVO_PRIMARY_TRAVERSAL_OVERRIDE: SvoPrimaryTraversalMode | undefined =
  traversalOverrideRaw === "" ? undefined : traversalOverrideRaw;

/** The structural facts the traversal decision is allowed to depend on. */
export interface SvoPrimaryTraversalScale {
  /** Occupied leaf bricks, i.e. the proxies `raster` would emit. */
  readonly leafBricks?: number;
  /** Pixels of the primary's own render target, after `resolutionScale`. */
  readonly targetPixels?: number;
  /** Authored environment refinement depth; consulted only when the brick count is unknown. */
  readonly environmentRefinementDepth?: number;
}

/**
 * The primary traversal a frame should actually run: `traced`, now, always.
 *
 * The megakernel is the primary. See
 * {@link SVO_PRIMARY_RASTER_PROXIES_PER_PIXEL_CEILING} for the paired-worktree
 * measurement that decided it — 6.2x on a frozen scene with the renderer as the
 * only variable — and for why the ratio this function used to compute steered
 * exactly the wrong way at production resolution.
 *
 * `scale` is retained and ignored. Every call site already computes it, it is
 * the input any future rule would need, and a rule that reads no inputs is
 * better spelled as one that has them and does not consult them than as a
 * signature change that would have to be undone. `raster` remains fully
 * compiled and reachable through {@link SVO_PRIMARY_TRAVERSAL_OVERRIDE}; it is
 * the arm that reproduces the pre-change frame and the one that still carries
 * the analytic scene-primitive tier.
 */
export function resolveSvoPrimaryTraversal(
  requested: SvoPrimaryTraversalMode,
  scale: SvoPrimaryTraversalScale = {},
): SvoPrimaryTraversalMode {
  void scale;
  if (SVO_PRIMARY_TRAVERSAL_OVERRIDE) return SVO_PRIMARY_TRAVERSAL_OVERRIDE;
  if (requested !== "raster") return requested;
  return "traced";
}

/** Observable lifecycle of the requested primary seam-closure pass. */
export type SvoSilhouetteRefinementStatus = Readonly<{
  state: "enabled" | "disabled" | "not-applicable" | "compiling" | "failed";
  /** Exact compile/capability reason; never authorizes another render path. */
  detail?: string;
}>;

/** Effective lighting-visibility path, including a visible exact fallback. */
export type SvoLightingVisibilityStatus = Readonly<{
  state: "cones" | "exact" | "off";
  /** True only when cones were requested but their complete hierarchy is unavailable. */
  fallback?: boolean;
  detail?: string;
}>;

export type SvoLightingOptions = Readonly<{
  shadowsEnabled: boolean;
  ambientOcclusionEnabled: boolean;
  /**
   * Experimental full-rate seam treatment for reduced SVO presentation.
   * This is an explicit render choice: disabling it never happens as a
   * substitute for a compiling or failed refinement pipeline. Omitted means
   * disabled.
   */
  silhouetteRefinementEnabled?: boolean;
  /** Omitted means `cones`. */
  coneTracingMode?: SvoConeTracingMode;
  /** Omitted means `raster`. Switching it rebuilds the dry-scene pipeline. */
  primaryTraversal?: SvoPrimaryTraversalMode;
}>;

/** The presentation preset aims for the finished image; each effect remains independently switchable. */
export const DEFAULT_SVO_LIGHTING_OPTIONS: SvoLightingOptions = Object.freeze({
  shadowsEnabled: true,
  ambientOcclusionEnabled: true,
  silhouetteRefinementEnabled: false,
  coneTracingMode: "cones",
  primaryTraversal: "raster",
});
