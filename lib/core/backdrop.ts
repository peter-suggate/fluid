/**
 * The backdrop: rolling hills and sparse vegetation standing *around* the set.
 *
 * A document field of its own (`SceneDescription.backdrop`), and deliberately
 * not scenery. Everything in `scene.scenery` is enumerated by the editor's
 * picking, the `fluid-collider` scan, the WASM slice, the CM12/adaptive stamps
 * and the set world's sparse domain — and a hillside eighty metres across would
 * blow every one of those budgets while contributing nothing any of them wants.
 * The backdrop ground is drawn by the SVO renderer as a tiled height field out
 * to the haze horizon, which no solver, bake or collider scan ever reads; see
 * `docs/backdrop-svo-plan.md` and `lib/svo/features/backdrop/`.
 *
 * Descriptive parameters only, never baked geometry: the expander is pure and
 * seeded, so the document stays a few hundred bytes and survives the render
 * worker's structured clone for free.
 *
 * The one quantity the backdrop shares with the set — the seam height at the
 * footprint edge — is *not* stored here. It is derived at build time from the
 * set's own ground (`sceneSvoGroundPlane`), so the two cannot drift apart when
 * the vessel is re-authored.
 */
export interface BackdropDescription {
  kind: "rolling-hills";
  /** Unsigned 32-bit seed for the hill waves and every vegetation placement. */
  seed: number;
  /**
   * Where the valley's mean rise is complete, measured from the footprint. The
   * ground itself does not end here: it runs on to the haze horizon.
   */
  outerRadius_m: number;
  /**
   * E-folding distance of the fade into the horizon colour, measured from the
   * footprint edge. The backdrop owns its atmosphere: its ground fades as
   * `1 - exp(-d / hazeDistance_m)` into the sky at the horizon, so the horizon
   * is one continuous gradient, and the ground is traced out to where the haze
   * leaves less than 1/256 of it (`svoBackdropContentRadius`).
   */
  hazeDistance_m: number;
  hills: BackdropHills;
  /**
   * Inner tile levels stored as real octree voxels (terrain plus seeded
   * set-style scatter) rather than walked as a height field, so they cast and
   * receive shadows, AO and cone GI like the set. Ring `l` has cells of the
   * scene cell times `2^l`; 0 keeps the whole backdrop a walked height field.
   */
  detailRings?: number;
  vegetation: {
    trees: BackdropVegetationClass;
    shrubs: BackdropVegetationClass;
    tufts: BackdropVegetationClass;
    pebbles: BackdropVegetationClass;
  };
}

export interface BackdropHills {
  /** Peak-to-trough height of the rolling waves once fully ramped in. */
  amplitude_m: number;
  /** Wavelength of the first (broadest) octave. */
  wavelength_m: number;
  /** Octaves of seeded directional waves; each halves the wavelength. */
  octaves: number;
  /** Level ring around the footprint, at exactly the seam height. */
  flatRing_m: number;
  /** Distance beyond the flat ring over which the waves ease in from zero. */
  rampWidth_m: number;
  /**
   * Mean rise of the ground at `outerRadius_m`, so the set reads as sitting in
   * the bottom of a shallow valley and the far rim hides the world's edge.
   */
  valleyRise_m: number;
}

/**
 * One kind of vegetation. Placement is uniform in density over the annulus
 * `[radius_m[0], radius_m[1]]` from the set centre (so the item count grows
 * with area, and most items stand far out); size is drawn from `size_m` with a
 * bias toward the large end farther out.
 */
export interface BackdropVegetationClass {
  /** Expected items per square metre inside the annulus. */
  density_m2: number;
  /** Inner and outer placement radius from the set centre. */
  radius_m: readonly [number, number];
  /** Height (trees, shrubs, tufts) or diameter (pebbles), min and max. */
  size_m: readonly [number, number];
}

/** Hard caps that turn an accidental density into a loud error, not a hang. */
export const BACKDROP_MAXIMUM_OUTER_RADIUS_M = 500;
export const BACKDROP_MAXIMUM_OCTAVES = 6;
export const BACKDROP_MAXIMUM_HAZE_DISTANCE_M = 2000;
export const BACKDROP_MAXIMUM_ITEMS_PER_CLASS = 20_000;
/** Stored rings are aligned on 100 mm solver bricks; ring 4 (200 mm cells) would not hold the seam lattice. */
export const BACKDROP_MAXIMUM_DETAIL_RINGS = 4;

const VEGETATION_CLASSES = ["trees", "shrubs", "tufts", "pebbles"] as const;

function finitePositive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function finiteNonNegative(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/**
 * Explicit validation, one message per broken field. A backdrop is either
 * wholly valid or the document is refused — there is no partial expansion.
 */
export function validateBackdrop(backdrop: BackdropDescription): string[] {
  const errors: string[] = [];
  if (!backdrop || typeof backdrop !== "object") return ["Backdrop must be an object"];
  if (backdrop.kind !== "rolling-hills") errors.push(`Unknown backdrop kind ${String(backdrop.kind)}`);
  if (!Number.isSafeInteger(backdrop.seed) || backdrop.seed < 0 || backdrop.seed > 0xffff_ffff) {
    errors.push("Backdrop seed must be an unsigned 32-bit integer");
  }
  if (!finitePositive(backdrop.outerRadius_m) || backdrop.outerRadius_m > BACKDROP_MAXIMUM_OUTER_RADIUS_M) {
    errors.push(`Backdrop outer radius must be positive and at most ${BACKDROP_MAXIMUM_OUTER_RADIUS_M} m`);
  }
  if (!finitePositive(backdrop.hazeDistance_m) || backdrop.hazeDistance_m > BACKDROP_MAXIMUM_HAZE_DISTANCE_M) {
    errors.push(`Backdrop haze distance must be positive and at most ${BACKDROP_MAXIMUM_HAZE_DISTANCE_M} m`);
  }
  const hills = backdrop.hills;
  if (!hills || typeof hills !== "object") errors.push("Backdrop hills are required");
  else {
    if (!finiteNonNegative(hills.amplitude_m)) errors.push("Backdrop hill amplitude must be non-negative and finite");
    if (!finitePositive(hills.wavelength_m)) errors.push("Backdrop hill wavelength must be positive and finite");
    if (!Number.isInteger(hills.octaves) || hills.octaves < 1 || hills.octaves > BACKDROP_MAXIMUM_OCTAVES) {
      errors.push(`Backdrop hill octaves must be an integer from 1 to ${BACKDROP_MAXIMUM_OCTAVES}`);
    }
    // Positive, not merely non-negative: past the ring the ground's curvature
    // carries the footprint distance's own, which is 1/d round a corner, and
    // the virtual voxels' conservative cell law bounds it by 1/flatRing.
    if (!finitePositive(hills.flatRing_m)) errors.push("Backdrop flat ring must be positive and finite");
    if (!finitePositive(hills.rampWidth_m)) errors.push("Backdrop hill ramp width must be positive and finite");
    if (!finiteNonNegative(hills.valleyRise_m)) errors.push("Backdrop valley rise must be non-negative and finite");
    if (finitePositive(backdrop.outerRadius_m) && finiteNonNegative(hills.flatRing_m)
      && hills.flatRing_m >= backdrop.outerRadius_m) {
      errors.push("Backdrop flat ring must end inside the outer radius");
    }
  }
  if (backdrop.detailRings !== undefined && (!Number.isInteger(backdrop.detailRings)
    || backdrop.detailRings < 0 || backdrop.detailRings > BACKDROP_MAXIMUM_DETAIL_RINGS)) {
    errors.push(`Backdrop detail rings must be an integer from 0 to ${BACKDROP_MAXIMUM_DETAIL_RINGS}`);
  }
  const vegetation = backdrop.vegetation;
  if (!vegetation || typeof vegetation !== "object") errors.push("Backdrop vegetation is required");
  else for (const name of VEGETATION_CLASSES) {
    const entry = vegetation[name];
    if (!entry || typeof entry !== "object") { errors.push(`Backdrop vegetation ${name} is required`); continue; }
    if (!finiteNonNegative(entry.density_m2)) errors.push(`Backdrop ${name} density must be non-negative and finite`);
    const radius = entry.radius_m, size = entry.size_m;
    if (!Array.isArray(radius) || radius.length !== 2 || !finiteNonNegative(radius[0]) || !finitePositive(radius[1])
      || radius[0] >= radius[1]) {
      errors.push(`Backdrop ${name} radius must be an increasing pair of non-negative distances`);
    } else if (finitePositive(backdrop.outerRadius_m) && radius[1] > backdrop.outerRadius_m) {
      errors.push(`Backdrop ${name} must stand inside the outer radius`);
    }
    if (!Array.isArray(size) || size.length !== 2 || !finitePositive(size[0]) || !finitePositive(size[1]) || size[0] > size[1]) {
      errors.push(`Backdrop ${name} size must be a non-decreasing pair of positive lengths`);
    }
    if (Array.isArray(radius) && radius.length === 2 && finiteNonNegative(entry.density_m2)
      && finiteNonNegative(radius[0]) && finitePositive(radius[1])) {
      const expected = entry.density_m2 * Math.PI * (radius[1] ** 2 - radius[0] ** 2);
      if (expected > BACKDROP_MAXIMUM_ITEMS_PER_CLASS) {
        errors.push(`Backdrop ${name} would place ${Math.round(expected)} items; the cap is ${BACKDROP_MAXIMUM_ITEMS_PER_CLASS}`);
      }
    }
  }
  return errors;
}
