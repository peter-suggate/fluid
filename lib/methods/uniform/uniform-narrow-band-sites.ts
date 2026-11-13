/**
 * The particle band's surface sampling density (method value surfaceSamples).
 *
 * The samples define the surface only in the band's surface layer: cells whose
 * centre is above 1.5 cells deep (where particleDensity "graded" keeps eight).
 * Beneath it they only carry momentum. surfaceSamples sets the surface layer's
 * density alone:
 *   8 (default): eight unit samples a cell on a jittered 2x2x2 sub-lattice,
 *     the solver as it was; every generator emits the same text.
 *   64: sixty-four unit samples a cell on a jittered 4x4x4 sub-lattice.
 *
 * The deep band keeps its own density: eight samples a cell (four under
 * particleDensity "graded", or a collar under collarDensity "deep") on the
 * 2x2x2 lattice. A cell holds S units of liquid (S = surfaceSamples), so a
 * deep sample carries S/8 (or S/4) units: the same liquid, fewer samples.
 * A deep sample that rises into the surface layer splits (nbFinish: the
 * crowded-cell split, halving a step), and surface samples that sink are
 * merged by the crowded-cell cap (nbMergeRun), the machinery of graded two-
 * unit seeds. Both need crowdedCells "merge": 64 throws without it.
 *
 * Scaled with S, so the physics is unchanged:
 *   - a sample's radius r = cbrt(m)/(2n), n = cbrt(S): unit samples rest 1/n
 *     apart and a whole-cell sample (m = S) has radius 1/2 at any S;
 *   - the merge bound NB_MASS_MAX = S, the deep cell bound NB_CELL_MASS = 2S;
 *   - the reseed test (a cell's mass under S), the transfer's expected mass
 *     (S a full cell), the union's mass bound (S a cell);
 *   - particleCellCap in surface-layer cells (x S/8; deep cells keep the
 *     rail's value, their density is unchanged), sparseNeighbours (counts of
 *     samples of the reading sample's own mass: x S/(8m));
 *   - the union's sphere radius 0.625 + 1/(2n) (0.875 at n = 2) and its
 *     shell's tangential offset 1/(2n^2), which keep the union's surface where
 *     a flat seeded layer puts it at n = 2;
 *   - the particle records (records rule and per-fine-cell budget) by the
 *     surface layer's share of the band (narrowBandSurfaceSampleScale).
 * 27 (3x3x3) is not offered: the graded deep cell's four samples and the
 * band's eight cannot carry 27 units in whole numbers.
 */

export const NARROW_BAND_SURFACE_SAMPLES = [8, 64] as const;
export type NarrowBandSurfaceSamples = typeof NARROW_BAND_SURFACE_SAMPLES[number];

/** surfaceSamples: absent is 8; 8 or 64 (number or string); anything else throws. */
export function narrowBandSurfaceSamples(value: unknown): NarrowBandSurfaceSamples {
  if (value === undefined) return 8;
  const n = typeof value === "string" && /^\d+$/.test(value) ? Number(value) : value;
  if (n === 8 || n === 64) return n;
  throw new Error(`Narrow-band FLIP surfaceSamples must be one of ${NARROW_BAND_SURFACE_SAMPLES.join(", ")}: ${JSON.stringify(value)}`);
}
/** More than eight a cell holds the deep band's liquid in samples of several units, which only crowdedCells "merge" carries. */
export function narrowBandSurfaceSamplesCheck(samples: NarrowBandSurfaceSamples, crowdedCells: "retire" | "merge"): void {
  if (samples !== 8 && crowdedCells !== "merge") throw new Error(`Narrow-band FLIP surfaceSamples ${samples} keeps the deep band at eight samples of several units a cell, which needs crowdedCells "merge" (got "${crowdedCells}")`);
}
/** Sites an axis of the surface sub-lattice. */
export const narrowBandSiteAxis = (samples: NarrowBandSurfaceSamples): 2 | 4 => samples === 8 ? 2 : 4;
/** Bits of a sample's mass (mass - 1) in before.w and the compact flag word. */
export const narrowBandMassBits = (samples: NarrowBandSurfaceSamples): number => Math.log2(samples);
/** A sample's radius a mass m = 0..S: cbrt(m)/(2n). */
export function narrowBandSampleRadii(samples: NarrowBandSurfaceSamples): number[] {
  const n = narrowBandSiteAxis(samples);
  return Array.from({ length: samples + 1 }, (_, m) => Math.cbrt(m) / (2 * n));
}
/** NB_SAMPLE_RADIUS and nbSampleRadius. At 8 the table the solver always compiled. */
export function narrowBandSampleRadiusWGSL(samples: NarrowBandSurfaceSamples): string {
  const table = samples === 8 ? "0.0,0.25,0.31498,0.36056,0.39685,0.42750,0.45428,0.47823,0.5"
    : narrowBandSampleRadii(samples).map(r => r === 0 ? "0.0" : r.toFixed(5)).join(",");
  return `const NB_SAMPLE_RADIUS=array<f32,${samples + 1}>(${table});
fn nbSampleRadius(m:f32)->f32{return NB_SAMPLE_RADIUS[min(u32(m),${samples}u)];}`;
}
/** The particle union's sphere radius, the shell's depth and tangential offset
 * squared, and the lattice spacing squared, for n sites an axis
 * (uniform-narrow-band-surface.wgsl.ts). n = 2: 0.875, sqrt(0.875^2-0.125), 0.125, 0.25. */
export function narrowBandUnionShape(n: 2 | 4): { radius: number; seedDepth: number; offset2: number; spacing2: number } {
  const radius = 0.625 + 1 / (2 * n), offset2 = 1 / (2 * n * n);
  return { radius, seedDepth: Math.sqrt(radius ** 2 - offset2), offset2, spacing2: 1 / (n * n) };
}
/** The records a band cell needs at this density over the records at eight:
 * the surface layer (taken as two cells: the cell and a half under the
 * surface and the seeded part of the cell above it) at S, the rest of the band
 * at eight. 1 at S = 8. */
export function narrowBandSurfaceSampleScale(samples: NarrowBandSurfaceSamples, bandDepth: number): number {
  if (samples === 8) return 1;
  const surface = Math.min(2, bandDepth);
  return (surface * samples + (bandDepth - surface) * 8) / (bandDepth * 8);
}

// ---- CPU model of the seeding rules (tests/uniform-narrow-band-sites.test.ts).

/** umCorner(k, n): site k of an n^3 lattice, x fastest. */
export const narrowBandLatticeCorner = (k: number, n: number): [number, number, number] => [k % n, Math.floor(k / n) % n, Math.floor(k / (n * n))];
/** A surface-layer site's unjittered origin in its cell and the extent its jitter spans: c + (corner + jitter)/n. */
export function narrowBandSurfaceSite(k: number, samples: NarrowBandSurfaceSamples, jitter: readonly [number, number, number] = [0, 0, 0]): [number, number, number] {
  const n = narrowBandSiteAxis(samples), c = narrowBandLatticeCorner(k, n);
  return [(c[0] + jitter[0]) / n, (c[1] + jitter[1]) / n, (c[2] + jitter[2]) / n];
}
/** A deep cell's j-th site of `target` (8, or 4: the even corners 0, 3, 5, 6) on the 2x2x2 lattice. */
export const narrowBandCoarseSite = (j: number, target: 4 | 8): number => target === 8 ? j : [0, 3, 5, 6][j & 3]!;
/** The samples a cell seeds full (nbSeedTarget) and the units each carries. */
export function narrowBandSeedPlan(depth: number, graded: boolean, samples: NarrowBandSurfaceSamples, collarDeep = false): { target: number; mass: number; fine: boolean } {
  const target = collarDeep || (graded && depth <= -1.5) ? 4 : depth <= -1.5 ? 8 : samples;
  return { target, mass: samples / target, fine: target === samples };
}
