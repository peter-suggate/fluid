/** Candidate finite-distance geometry bounds, not an approved production cap.
 * The direct cap experiment failed the moved-inflow comparison (see research).
 * A distance reconstruction/validity contract is required before using it.
 * The cap is
 * 16 times the largest lattice spacing, not a physical boundary or a liquid
 * cutoff. Negative phi and conservative V must always remain resident.
 *
 * Redistancing starts below 4 hMax, can move its query by four lattice cells
 * per axis, and samples a +/- .25-cell gradient with trilinear interpolation.
 * For an exact distance field these reads are inside (4 + sqrt(3)*5.25) hMax,
 * less than 16 hMax. Evolved fields still require measured support validation;
 * this geometric bound alone is not a proof that a page can be retired.
 */
export const UNIFORM_PHI_BAND_CELLS = 16;
export const UNIFORM_PHI_REDISTANCE_READ_REACH = 5.25;

export function uniformPhiBandLimit(spacing: readonly [number, number, number], cells = UNIFORM_PHI_BAND_CELLS): number {
  if (!Number.isSafeInteger(cells) || cells < UNIFORM_PHI_BAND_CELLS
    || !spacing.every(h => Number.isFinite(h) && h > 0)) throw new RangeError("Invalid finite phi band");
  // Match the GPU's parameter upload and multiplication rounding.
  const limit = Math.fround(cells * Math.max(...spacing.map(Math.fround)));
  if (!Number.isFinite(limit) || limit <= 0) throw new RangeError("Phi band is not representable as f32");
  return limit;
}

/** Conservative read halo for one advection + redistance composition. The
 * displacement must bound actual characteristic queries, including midpoint
 * sampling and force/source allowances. It is not a lagged CPU velocity sample.
 * Contact continuation, phi/V agreement, extension, pressure and correction add
 * their own support. This helper does not bound those compound operators.
 */
export function uniformPhiReadHalo(displacementCells: readonly [number, number, number]): readonly [number, number, number] {
  if (!displacementCells.every(d => Number.isFinite(d) && d >= 0)) throw new RangeError("Invalid characteristic displacement");
  return displacementCells.map(d => Math.ceil(d + UNIFORM_PHI_REDISTANCE_READ_REACH)) as [number, number, number];
}
