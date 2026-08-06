import type { SvoConeLightingScale } from "./webgpu-svo-dry-scene";

/** Full-resolution reconstruction policy for reduced visibility and opaque radiance. */
export type SvoConeRadianceReconstruction =
  | "nearest" | "gated-linear" | "joint-bilateral" | "wide-relight" | "full-res-relight";

export const SVO_CONE_RADIANCE_RECONSTRUCTION_CODES = Object.freeze({
  nearest: 0,
  "gated-linear": 1,
  "joint-bilateral": 2,
  "wide-relight": 3,
  "full-res-relight": 4,
} satisfies Record<SvoConeRadianceReconstruction, number>);

/** Audited compile-time ceiling for primary-ray leaf continuation. */
export const SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT = 256;

/**
 * Ceiling on the analytic band's budget: the authored record ceiling itself.
 *
 * Mirrors `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` deliberately rather than
 * importing it — this module is on the browser's tuning path and has no other
 * reason to pull in the candidate BVH — and a budget above the record ceiling
 * is indistinguishable from an unbounded one anyway.
 */
export const SVO_NEAR_FIELD_BAND_MAXIMUM_RECORDS = 16_384;

/** Additional authored-environment subdivision beyond the legacy SVO brick plan. */
export const SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM = 1;

/**
 * Extra octree levels dense environment regions may descend into.
 *
 * Distinct from `environmentBrickRefinementLevels`, which *coarsens* toward the
 * solver's lattice. This spends depth below it, and only on a scene the solver
 * does not own — the domain planner claims every brick of a simulated
 * container, and a solver brick pins its node. Two is the measured point where
 * the hero garden's busiest brick reaches the 64 candidates the hierarchy binds;
 * beyond that nothing more splits.
 */
export const SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM = 3;

/**
 * The depth a dry document is authored at when nobody says otherwise.
 *
 * The ceiling, deliberately: the authored sets are exact analytic SDFs and the
 * scenery generators resolve their legibility floors against this leaf, so the
 * depth is what decides whether a feature is drawn or aliased. At the hero
 * garden's 6.25 mm lattice a stepping-stone tread is 74-108 mm — 12-17 voxels
 * at depth 0, which is what made them read as faceted plates — and 0.78 mm
 * leaves put them at 95-138.
 *
 * **The ceiling was measured before it was made the default**, on
 * `hero-garden-hose` through `tmp/sp18/census.ts`, because an older reading of
 * this ladder said depth 3 was unreachable (4.6M pages, a 9.99 GB staging
 * buffer at depth 2, a Node OOM at depth 3). That reading is stale — the
 * page-reduction stack landed and moved the exponent from ~8 to ~4:
 *
 * | depth | leaf | node-mip pages | pyramid | levels | CPU plan |
 * |---|---|---|---|---|---|
 * | 2 | 1.5625 mm |  107,115 | 0.25 GiB | 9  | ~22 s |
 * | 3 | 0.78125 mm | 528,237 | 1.22 GiB | 10 | ~28 s |
 *
 * Both complete. The CPU figures exclude the census harness's own
 * `createTallCellLayout` (130 s of its wall clock), which the live SVO path does
 * not call. Ten levels also clears the cone hierarchy's twelve, past which
 * derived lighting withdraws for roughly 15x.
 *
 * **It is still a request rather than a guarantee.** Those numbers are one
 * scene on one machine, and a larger domain scales with volume.
 * `WebGPULiveSvoScene.create` degrades to the finest depth that fits rather than
 * throwing — a throw there bricks the tab — and reports the rung it settled on
 * through `builtRefinementDepth`, which the renderer surfaces in its status
 * line. A scene that comes up coarser than this has said so.
 */
export const SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT = SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM;

/**
 * The voxel a set is actually drawn into, given the lattice and the depth.
 *
 * `webgpu-octree-sparse-bricks.ts` resolves this at construction —
 * `refineScale = 2 ** refinementDepth`, `renderCellSize = cellSize / refineScale`,
 * and `refinementDepth` is zero unless `scene.systems.fluid === false` because a
 * solver brick pins its node and leaves nowhere to descend. This is that rule
 * as a function, so a *scene* can be built at the size it will be voxelized at
 * rather than at the size its solver would run on.
 *
 * It matters because authored detail is measured in voxels. A heightfield is
 * sampled once per finest column, a floret has to span about three voxels before
 * it stops being aliasing, and both of those are questions about the number this
 * returns and not about `finestCellSize_m`. Hand it to
 * `voxelDomain.detailCellSize_m` before construction; setting the lattice
 * afterwards is too late, because every generator has already expanded.
 */
export function svoSceneryDetailCellSize_m(
  finestCellSize_m: number,
  options: { readonly environmentRefinementDepth?: number; readonly fluid?: boolean } = {},
): number {
  if (!(finestCellSize_m > 0) || !Number.isFinite(finestCellSize_m)) {
    throw new RangeError("Scenery detail cell size needs a positive finite lattice");
  }
  const depth = options.fluid === false
    ? Math.min(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
      Math.max(0, Math.trunc(options.environmentRefinementDepth ?? 0)))
    : 0;
  return finestCellSize_m / 2 ** depth;
}

/**
 * The refinement depth a document is *authored* at — the only one there is.
 *
 * This used to be two numbers. `SvoRenderTuning` carried a depth that divided
 * the tree's cell by `2^depth`, and the document carried
 * `voxelDomain.detailCellSize_m`, which is what `buildEnvironmentProxyCatalog`
 * hands every generator as its legibility floor. Nothing held them in step, so
 * dragging the render panel's slider built a finer *tree* over a set that had
 * been expanded at the coarse size: smoother silhouettes, no new detail, and no
 * indication which of the two you were looking at. Measured on
 * `hero-garden-hose`, whose stepping-stone treads are 74-108 mm — 12-17 voxels
 * at the authored 6.25 mm — while the tree claimed a 0.78 mm leaf.
 *
 * So the depth is now read *off the document* and stored nowhere else. It moves
 * only through `SimulationController.rebuildSceneAtLattice`, which re-runs the
 * scene's own factory at the finer lattice — the one edit that can re-bake a
 * heightfield and re-resolve a generator's floors. The exact inverse of
 * {@link svoSceneryDetailCellSize_m}, so a round trip is the identity.
 *
 * Zero on a fluid scene whatever the document says, for the same reason the
 * forward rule returns the coarse cell there: the domain planner claims every
 * brick of a simulated container and a solver brick pins its node, so there is
 * nowhere to descend and no finer document to be authored at.
 */
export function svoSceneryRefinementDepth(
  domain: { readonly finestCellSize_m: number; readonly detailCellSize_m?: number },
  options: { readonly fluid?: boolean } = {},
): number {
  const finest = domain.finestCellSize_m;
  const detail = domain.detailCellSize_m ?? finest;
  if (options.fluid !== false) return 0;
  if (!(finest > 0) || !Number.isFinite(finest) || !(detail > 0) || !Number.isFinite(detail)) return 0;
  return Math.min(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
    Math.max(0, Math.round(Math.log2(finest / detail))));
}

/**
 * How the primary decides to stop descending the hierarchy.
 *
 * `screen-space` is the shipping policy: a node stops when its projected
 * footprint falls under `lodScreenSpacePixels` — distance-adaptive, so far
 * geometry costs less without anything authored per scene.
 *
 * `fixed-level` pins descent to `lodFixedLevel` for every ray regardless of
 * projection. It exists to make the hierarchy legible: with a fixed level the
 * frame shows exactly what a given depth can represent, which is what tells a
 * missing-detail bug apart from a level the tree never built.
 */
export type SvoLodMode = "screen-space" | "fixed-level";

/**
 * Widest useful screen-space threshold.
 *
 * The predicate measures the enclosing sphere's diameter, so 64 already
 * dissolves a hero foreground prop into single boxes. Higher values are not
 * more informative, they are just slower to drag back from.
 */
export const SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM = 64;

/** Deepest addressable octree level, matching the Morton key's 21-bit budget. */
export const SVO_LOD_FIXED_LEVEL_MAXIMUM = 21;

/**
 * What surface the primary shades, once a voxel has been hit.
 *
 * The voxel AABB is the traversal primitive under both arms; this only decides
 * what the hit *means* once the DDA has stopped on a solid cell.
 *
 * `voxel-face` is what shipped: the hit is the cell face the ray crossed and the
 * normal is one of six axis directions, read back off the cell bounds by
 * `dryVoxelFaceNormal`. Every surface therefore terraces at every lattice, and
 * the six-way quantisation is also what banded the interior of flat faces —
 * the G-buffer carries distinct geometric and shading normal slots and this arm
 * fills both with the same face.
 *
 * `trilinear` reconstructs the scene-geometry lane's signed distance across the
 * eight voxel centres surrounding the hit and takes the analytic gradient of
 * that reconstruction as the shading normal. The field is already stored, at the
 * voxel centre, in metres, and is differentiated exactly this way by the derived
 * builder's `safeNormal`; the only thing that was missing was a binding. The
 * stencil is clamped inside the leaf, matching `safeNormal`, so a dual cell that
 * straddles an 8-cell brick face falls back to a one-sided difference rather
 * than resolving the neighbouring leaf.
 *
 * `analytic` is the shipping arm, and it asks the surface rather than a stored
 * sample of it. A solid cell already carries the owner of the primitive that
 * filled it, that primitive's exact signed-distance function is already bound in
 * the same shader, and the ground's heightfield is already sampled there for the
 * marched path — so the exact outward normal at the hit point is one record
 * fetch away, with no lattice in it at all.
 *
 * ## Why the stored field could not answer this
 *
 * The scene-geometry lane is not a distance field, in two separate ways, and
 * both of them reach the frame as normals that are wrong in lattice-locked
 * patches — long straight bands on a flat surface, contour rings on a bowl:
 *
 *  - the ground writes `y - h(x,z)` (`webgpu-sparse-scene-proxies.ts:2066`), a
 *    *vertical* pseudo-distance. Its gradient is the correct normal on a gentle
 *    slope and collapses toward +Y on a steep one, which is measurable as
 *    vertical stripes down the pond coping's wall;
 *  - every other voxel gets the `min` over only the primitives binned into *its
 *    own brick*, so the field carries no guarantee of continuity across a brick
 *    face — while the stencil deliberately reads across brick faces, because a
 *    stencil clamped inside the leaf banded the surface a different way.
 *
 * Measured on `water-box-dam-break`, whose whole environment is six flat white
 * boxes: depth, owner and visibility are all exact, and `trilinear` alone turns
 * the room into a debris field of rectangular slabs. See
 * `SVO_SURFACE_ANALYTIC_EIKONAL_TOLERANCE` for what the fallback does about it.
 *
 * `trilinear` is retained as the arm that produced that evidence, and because it
 * is still the answer for a cell whose owner names nothing analytic.
 *
 * The geometric normal, the hit distance, and hit-versus-miss are identical
 * under all three arms — this changes shading only, which is what lets
 * `voxel-face` stand as the bit-exact reference the acceptance gates rest on.
 */
export type SvoSurfaceReconstruction = "voxel-face" | "trilinear" | "analytic";

/**
 * How far the reconstructed gradient's magnitude may sit from 1 before the
 * fallback declines it.
 *
 * A signed distance field satisfies the eikonal equation, `|grad d| = 1`, so the
 * gradient the stencil already computes carries its own validity certificate and
 * it costs nothing to read: the length is taken either way, and today it is only
 * compared against `1e-12`. Where the lane is a real distance the magnitude is 1
 * to within quantisation; where it is `y - h`, or a `min` over a neighbour
 * brick's different candidate set, it is not, and the cell's own face is the
 * better answer than a confident wrong one.
 *
 * The band is deliberately generous. `f16` metres over a 6.25 mm cell quantise
 * the difference by well under a percent, and a real surface crossing the dual
 * cell at an angle still reads 1; what it has to catch is the order-of-magnitude
 * departure a foreign field produces, not a few degrees of quantisation noise.
 */
export const SVO_SURFACE_ANALYTIC_EIKONAL_TOLERANCE = 0.35;

/**
 * The ground gradient's step, in cells.
 *
 * The ground's normal is `(-dh/dx, 1, -dh/dz)`, and the only free choice is how
 * far apart to sample `h`. One cell is the right width for the same reason the
 * lattice is the lattice: finer aliases, coarser erases relief the tree was
 * refined to hold. The 20 mm `terrainNormalAt` uses predates the fine lattices
 * and averages the hero garden's ground over three voxels.
 *
 * Width is not what a vertical face needs, and this was measured rather than
 * assumed. On the pond coping's wall: one cell stripes it, four cells smooth it
 * and grow a dark fringe along the arris instead, and a height-aware split
 * between "on the graph" and "under it" made the striping worse. A heightfield
 * column is multi-valued at a wall and no sampling of `h` recovers the surface;
 * `dryTerrainSurfaceNormal` limits the difference so the *edge* is right and
 * declines to guess the face, and the face wants geometry.
 */
export const SVO_SURFACE_TERRAIN_GRADIENT_CELLS = 1;

/** Runtime-adjustable sparse-presentation controls. Shader loops retain hard caps;
 * these values only lower work or adjust quality inside those audited bounds. */
export interface SvoRenderTuning {
  readonly resolutionScale: number;
  readonly environmentBrickRefinementLevels: number;
  /**
   * Whether the environment refinement rule may stop at a node its surface
   * crosses flatly, instead of spending every level the depth above allows.
   *
   * The test is second order — the residual against the node centre's own
   * tangent plane, and for the ground the spread of its column heights — so the
   * depth it saves is saved precisely where curvature is lowest. That is also
   * where it shows: the primary shades a leaf as one of six axis-aligned voxel
   * faces, so an exempted node arrives as a flat facet with a straight octree
   * boundary against the refined slope beside it, and a broad shallow dome
   * reads as a grid of them. Off by default for that reason; on, it is the
   * cheaper tree. See `lib/svo-environment-refinement.ts`.
   */
  readonly environmentPlanarRefinementExemption: boolean;
  readonly coneLightingScale: SvoConeLightingScale;
  readonly coneRadianceReconstruction: SvoConeRadianceReconstruction;
  /** Reuse an exact primary G-buffer while an eligible camera and scene remain unchanged. */
  readonly stationaryPrimaryReuseEnabled: boolean;
  /** What the primary shades once it has hit a voxel. See `SvoSurfaceReconstruction`. */
  readonly surfaceReconstruction: SvoSurfaceReconstruction;
  /** Which predicate stops the descent. See `SvoLodMode`. */
  readonly lodMode: SvoLodMode;
  /**
   * Projected footprint, in pixels, below which a node is drawn as a proxy
   * rather than descended into.
   *
   * Authored against `SVO_SCREEN_SPACE_TERMINATION_CONTRACT`'s reference
   * viewport height and rescaled to the live target, which keeps the threshold
   * angular: raising DPR or resolution scale must not silently ask for more
   * detail than the author requested.
   *
   * Zero means exact traversal with no approximation anywhere, and that is the
   * invariant the acceptance gates rest on — a lane that wants the reference
   * image asks for zero and gets it, not "very nearly zero".
   */
  readonly lodScreenSpacePixels: number;
  /** Level descent stops at under `fixed-level`. Ignored by `screen-space`. */
  readonly lodFixedLevel: number;
  readonly primaryLeafVisits: number;
  readonly coneStepBudget: number;
  readonly maximumShadedLights: number;
  readonly stableAreaLightSamples: number;
  readonly movingAreaLightSamples: number;
  readonly stableAoSamples: number;
  readonly movingAoSamples: number;
  readonly visibilityNodeVisits: number;
  readonly visibilityLeafVisits: number;
  readonly visibilityWorkItems: number;
  readonly visibilityIntersections: number;
  readonly shadowBiasCells: number;
  readonly shadowStrength: number;
  readonly aoRadiusScale: number;
  readonly aoStrength: number;
  readonly aoConeAperture: number;
  readonly shadowConeAperture: number;
  /** Artistic exposure applied only to gathered diffuse radiance. */
  readonly giBounceStrength: number;
  /** Broad visibility recovered from the diffuse GI cones. */
  readonly giOcclusionStrength: number;
  /** Analytic diffuse-environment contribution while GI is active. */
  readonly giEnvironmentStrength: number;
  /** Live analytic direct-light contribution, independent of derived-GI readiness. */
  readonly giDirectStrength: number;
  readonly giConeAperture: number;
  readonly giConeCount: number;
  readonly coneNormalEscapeCells: number;
  readonly coneEmitterClearanceCells: number;
  /**
   * Near-field analytic band: projected voxel size, in pixels, above which an
   * authored record is still drawn analytically through the coverage arena.
   *
   * Zero disables the band and every record stays analytic, which is the exact
   * historical image. A positive threshold hands everything below it to the
   * voxel primary, where the owning record's own surface is still what resolves
   * the hit — the accelerator changes, the authority does not.
   *
   * See `lib/svo-scene-primitive-band.ts` for the measure and the budget.
   */
  readonly nearFieldBandPixels: number;
  /** Exit threshold as a fraction of the entry threshold. One removes hysteresis. */
  readonly nearFieldBandHysteresis: number;
  /** Hard ceiling on analytic records per frame. Zero leaves the band unbounded. */
  readonly nearFieldBandBudget: number;
}

/**
 * Diagnostic overrides for the band, read once at module load.
 *
 * The acceptance lanes for this policy (`tools/benchmark-svo-dry-frame-gpu.ts`,
 * `tools/run-svo-dry-render-smoke.ts`) build their tuning by spreading
 * `DEFAULT_SVO_RENDER_TUNING`, so an A/B over the band would otherwise mean
 * editing a constant between arms — and the one measurement rule this program
 * has is that arms are interleaved rather than run in blocks. These are read
 * exactly like the octree solver's diagnostic switches and default to the
 * authored production values.
 */
const bandEnvironment = typeof process !== "undefined" ? process.env : undefined;
const bandOverride = (name: string, fallback: number): number => {
  const raw = bandEnvironment?.[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be a non-negative finite number`);
  return value;
};

export const DEFAULT_SVO_RENDER_TUNING: SvoRenderTuning = Object.freeze({
  resolutionScale: 0.72,
  environmentBrickRefinementLevels: 1,
  // Off: the levels it declines to spend are the low-curvature ones, and a
  // coarse leaf there is a flat axis-aligned facet on exactly the smooth
  // surfaces — a mound cap, a pond basin — where one is most visible.
  environmentPlanarRefinementExemption: false,
  // The balanced tier keeps the accepted 2x2 visibility error bar. The 4x4
  // rate remains available in the performance preset, while full-resolution
  // relighting preserves material and edge detail at either reduced rate.
  coneLightingScale: 0.5,
  coneRadianceReconstruction: "full-res-relight",
  stationaryPrimaryReuseEnabled: false,
  // The smooth surface is the shaded view's whole point, so it is the default and
  // `voxel-face` is the reference arm rather than the other way round. This moves
  // the frame hash and the hero fidelity baseline by construction; both are
  // re-blessed against this default, and a lane that wants the pre-existing image
  // asks for `voxel-face` and gets it bit-exactly.
  surfaceReconstruction: "analytic",
  lodMode: "screen-space",
  // Off by default.
  //
  // The machinery stays compiled and the slider stays live — this is a runtime
  // threshold, and zero is the one value the contract defines as *exact*:
  // no node is ever collapsed, so the frame is the reference image rather than
  // an approximation of it. Raising the slider still works for anyone who wants
  // the saving back.
  //
  // Three pixels shipped, and the approximation it bought was not paying for
  // itself. A sub-threshold brick is drawn as one voxel, so far geometry
  // arrives as axis-aligned facets — and once the primary started shading a
  // smooth reconstructed normal (see `SvoSurfaceReconstruction`) that tier
  // became the only part of the frame still terracing, which made it the most
  // visible thing in the image rather than the least.
  //
  // Note that `fixed-level` is inert regardless: `packLodParams` writes the mode
  // and level through the u32 view of a vec4f lane, so `dryLodMode()` never
  // returns `fixed-level` and `dryLodFixedLevel()` is always zero. See the
  // comment there — it is a real bug, deliberately left until someone wants that
  // debug view back.
  lodScreenSpacePixels: 0,
  // The deepest level, so entering `fixed-level` without moving the slider is
  // as close to a no-op as the mode allows. A mode switch that changes the
  // image on its own would make the debugging tool the thing under suspicion.
  lodFixedLevel: SVO_LOD_FIXED_LEVEL_MAXIMUM,
  primaryLeafVisits: 48,
  coneStepBudget: 48,
  maximumShadedLights: 8,
  stableAreaLightSamples: 2,
  movingAreaLightSamples: 1,
  stableAoSamples: 4,
  movingAoSamples: 1,
  visibilityNodeVisits: 96,
  visibilityLeafVisits: 24,
  visibilityWorkItems: 768,
  visibilityIntersections: 4,
  shadowBiasCells: 0.02,
  shadowStrength: 1,
  aoRadiusScale: 1,
  aoStrength: 1,
  aoConeAperture: 0.62,
  shadowConeAperture: 0.065,
  // GI is deliberately image-forward by default: the exact key light remains
  // crisp, while bounce and broad cone visibility visibly shape the scene.
  giBounceStrength: 1.8,
  // Multi-bounce compensation already restores much of the energy hidden by
  // broad cone occlusion. Retaining the old .82 contrast double-darkened room
  // corners and amplified 8-bit opacity contours into visible bands.
  giOcclusionStrength: 0.65,
  giEnvironmentStrength: 0.85,
  // Direct light is evaluated from the current scene-light arena every frame.
  // Derived radiance contains emitted energy, not a baked replacement for the
  // authored lights, so enabling GI must not silently remove ten percent of it.
  giDirectStrength: 1,
  giConeAperture: 1.05,
  giConeCount: 4,
  coneNormalEscapeCells: 0.5,
  coneEmitterClearanceCells: 3,
  // Off by default until the band's silhouette evidence is authored per scene:
  // zero reproduces the historical analytic set exactly, so enabling the voxel
  // primary underneath it (which only ever seeds a tighter depth) cannot be
  // confused with the band's own image effect.
  nearFieldBandPixels: bandOverride("FLUID_SVO_BAND_PIXELS", 0),
  nearFieldBandHysteresis: bandOverride("FLUID_SVO_BAND_HYSTERESIS", 0.8),
  nearFieldBandBudget: bandOverride("FLUID_SVO_BAND_BUDGET", 256),
});

export const SVO_RENDER_TUNING_PRESETS = Object.freeze({
  performance: Object.freeze({
    ...DEFAULT_SVO_RENDER_TUNING,
    resolutionScale: 0.5,
    coneLightingScale: 0.25 as const,
    primaryLeafVisits: 24,
    coneStepBudget: 20,
    giConeCount: 3,
    giBounceStrength: 1.35,
    giOcclusionStrength: 0.6,
    maximumShadedLights: 3,
    stableAreaLightSamples: 1,
    stableAoSamples: 2,
    visibilityNodeVisits: 48,
    visibilityLeafVisits: 12,
    visibilityWorkItems: 320,
  }),
  balanced: DEFAULT_SVO_RENDER_TUNING,
  quality: Object.freeze({
    ...DEFAULT_SVO_RENDER_TUNING,
    resolutionScale: 1,
    coneLightingScale: 0.5 as const,
    primaryLeafVisits: 128,
    coneStepBudget: 48,
    giConeCount: 4,
    visibilityNodeVisits: 128,
    visibilityLeafVisits: 32,
    visibilityWorkItems: 1024,
  }),
});

export type SvoRenderTuningPreset = keyof typeof SVO_RENDER_TUNING_PRESETS;

const bounded = (value: number, minimum: number, maximum: number) =>
  Math.max(minimum, Math.min(maximum, Number.isFinite(value) ? value : minimum));
const integer = (value: number, minimum: number, maximum: number) =>
  Math.round(bounded(value, minimum, maximum));

export function normalizeSvoRenderTuning(value: SvoRenderTuning): SvoRenderTuning {
  const coneLightingScale = value.coneLightingScale === 0.125 || value.coneLightingScale === 0.25 || value.coneLightingScale === 0.5
    ? value.coneLightingScale : 1;
  const coneRadianceReconstruction = value.coneRadianceReconstruction === "nearest"
    || value.coneRadianceReconstruction === "gated-linear"
    || value.coneRadianceReconstruction === "joint-bilateral"
    || value.coneRadianceReconstruction === "wide-relight"
    || value.coneRadianceReconstruction === "full-res-relight"
    ? value.coneRadianceReconstruction : "full-res-relight";
  return {
    resolutionScale: bounded(value.resolutionScale, 0.35, 1),
    environmentBrickRefinementLevels: integer(
      value.environmentBrickRefinementLevels,
      0,
      SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
    ),
    environmentPlanarRefinementExemption: Boolean(value.environmentPlanarRefinementExemption),
    coneLightingScale,
    coneRadianceReconstruction,
    stationaryPrimaryReuseEnabled: Boolean(value.stationaryPrimaryReuseEnabled),
    // Unknown normalises to the default rather than to the reference arm: a
    // typo in a stored URL should give the shipping image, not silently pin the
    // frame to the arm the gates keep for comparison.
    surfaceReconstruction: value.surfaceReconstruction === "voxel-face"
      || value.surfaceReconstruction === "trilinear" || value.surfaceReconstruction === "analytic"
      ? value.surfaceReconstruction : DEFAULT_SVO_RENDER_TUNING.surfaceReconstruction,
    lodMode: value.lodMode === "fixed-level" ? "fixed-level" : "screen-space",
    // Not `integer`: the threshold is a continuous angular measure, and rounding
    // it would quantise the one control that sweeps the quality/cost curve.
    lodScreenSpacePixels: bounded(
      value.lodScreenSpacePixels ?? DEFAULT_SVO_RENDER_TUNING.lodScreenSpacePixels,
      0,
      SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
    ),
    lodFixedLevel: integer(
      value.lodFixedLevel ?? DEFAULT_SVO_RENDER_TUNING.lodFixedLevel,
      0,
      SVO_LOD_FIXED_LEVEL_MAXIMUM,
    ),
    primaryLeafVisits: integer(value.primaryLeafVisits, 1, SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT),
    coneStepBudget: integer(value.coneStepBudget, 1, 48),
    maximumShadedLights: integer(value.maximumShadedLights, 1, 8),
    stableAreaLightSamples: integer(value.stableAreaLightSamples, 1, 2),
    movingAreaLightSamples: integer(value.movingAreaLightSamples, 1, 2),
    stableAoSamples: integer(value.stableAoSamples, 1, 4),
    movingAoSamples: integer(value.movingAoSamples, 1, 4),
    visibilityNodeVisits: integer(value.visibilityNodeVisits, 1, 128),
    visibilityLeafVisits: integer(value.visibilityLeafVisits, 1, 32),
    visibilityWorkItems: integer(value.visibilityWorkItems, 16, 1024),
    visibilityIntersections: integer(value.visibilityIntersections, 1, 4),
    shadowBiasCells: bounded(value.shadowBiasCells, 0, 0.25),
    shadowStrength: bounded(value.shadowStrength, 0, 1),
    aoRadiusScale: bounded(value.aoRadiusScale, 0.1, 3),
    aoStrength: bounded(value.aoStrength, 0, 1),
    aoConeAperture: bounded(value.aoConeAperture, 0.1, 1.4),
    shadowConeAperture: bounded(value.shadowConeAperture, 0.01, 0.25),
    giBounceStrength: bounded(value.giBounceStrength ?? DEFAULT_SVO_RENDER_TUNING.giBounceStrength, 0, 4),
    giOcclusionStrength: bounded(value.giOcclusionStrength ?? DEFAULT_SVO_RENDER_TUNING.giOcclusionStrength, 0, 1),
    giEnvironmentStrength: bounded(value.giEnvironmentStrength ?? DEFAULT_SVO_RENDER_TUNING.giEnvironmentStrength, 0, 2),
    giDirectStrength: bounded(value.giDirectStrength ?? DEFAULT_SVO_RENDER_TUNING.giDirectStrength, 0, 2),
    giConeAperture: bounded(value.giConeAperture ?? DEFAULT_SVO_RENDER_TUNING.giConeAperture, 0.4, 1.4),
    giConeCount: integer(value.giConeCount ?? DEFAULT_SVO_RENDER_TUNING.giConeCount, 3, 4),
    coneNormalEscapeCells: bounded(value.coneNormalEscapeCells, 0, 2),
    coneEmitterClearanceCells: bounded(value.coneEmitterClearanceCells, 0, 8),
    nearFieldBandPixels: bounded(value.nearFieldBandPixels ?? DEFAULT_SVO_RENDER_TUNING.nearFieldBandPixels, 0, 4096),
    nearFieldBandHysteresis: bounded(value.nearFieldBandHysteresis ?? DEFAULT_SVO_RENDER_TUNING.nearFieldBandHysteresis, 0.25, 1),
    nearFieldBandBudget: integer(value.nearFieldBandBudget ?? DEFAULT_SVO_RENDER_TUNING.nearFieldBandBudget, 0, SVO_NEAR_FIELD_BAND_MAXIMUM_RECORDS),
  };
}

export function svoRenderTuningKey(value: SvoRenderTuning): string {
  return Object.values(value).join(":");
}
