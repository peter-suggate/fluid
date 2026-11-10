/**
 * The particle band's collar: the heat-one tiles around hot ones
 * (activitySpread), seeded as transfer support without authority to erase
 * liquid (uniform-narrow-band-activity.wgsl.ts). Two method values narrow it
 * (docs/plans/paddleboats-60fps-2026-11-10/progress-lane-d.md has the
 * argument and what is verified):
 *
 * collarSpread
 *   "neighbours" (default): a tile's target heat is the most of its own heat
 *     and half of each of its 26 neighbours'.
 *   "liquid": a tile takes its neighbours' half heat only where the head
 *     census found liquid in it (UNIFORM_STAGE_IMPORTANCE.wet, copied into
 *     NB_ACTIVITY_SCORES); a dry tile keeps its own heat alone. Every wet tile
 *     takes the target it takes today. A dry tile has no vertex at or under
 *     the surface, so no seed site in it passes seedSite's depth test: its
 *     collar seeded nothing. What the dry collar did was cap the heat of a
 *     sample arriving there at one (nbFinish), which only a tile with target
 *     heat did. Under "liquid" every grid-supported arrival is capped at one
 *     whatever its tile's target, so no sample leaves the heated tiles with
 *     erasure heat, however far it travels in a step.
 *
 * collarDensity
 *   "band" (default): a collar cell is seeded as any band cell: eight unit
 *     samples a cell to a cell and a half deep, four of two units below
 *     (particleDensity "graded").
 *   "deep": a non-source cell whose tile's target heat is in (0, 1] takes
 *     the deep density at every depth: four samples of two units on the even
 *     sites (0, 3, 5, 6) of the cell's 2x2x2 lattice. The cell's mass is the
 *     same eight units, so the transfer's coverage (mass over nbExpectedMass)
 *     is unchanged. The crowded-cell split (nbFinish) reads the same target, so
 *     a collar sample does not split where it was seeded, and splits back to
 *     the band's density once its tile is hot. Needs crowdedCells "merge" (the
 *     only mode in which a sample carries mass).
 *
 * Unknown values fail: the parsers below throw.
 */

export const NARROW_BAND_COLLAR_SPREADS = ["neighbours", "liquid"] as const;
export type NarrowBandCollarSpread = typeof NARROW_BAND_COLLAR_SPREADS[number];
export const NARROW_BAND_COLLAR_DENSITIES = ["band", "deep"] as const;
export type NarrowBandCollarDensity = typeof NARROW_BAND_COLLAR_DENSITIES[number];

function choice<T extends string>(key: string, options: readonly T[], value: unknown): T {
  if (value === undefined) return options[0]!;
  if (typeof value === "string" && (options as readonly string[]).includes(value)) return value as T;
  throw new Error(`Narrow-band FLIP ${key} must be one of ${options.join(", ")}: ${JSON.stringify(value)}`);
}
/** collarSpread: absent is "neighbours"; anything else not listed throws. */
export const narrowBandCollarSpread = (value: unknown): NarrowBandCollarSpread => choice("collarSpread", NARROW_BAND_COLLAR_SPREADS, value);
/** collarDensity: absent is "band"; anything else not listed throws. */
export const narrowBandCollarDensity = (value: unknown): NarrowBandCollarDensity => choice("collarDensity", NARROW_BAND_COLLAR_DENSITIES, value);
/** collarDensity "deep" holds a cell's eight units in four samples, which only crowdedCells "merge" carries. */
export function narrowBandCollarCheck(density: NarrowBandCollarDensity, crowdedCells: "retire" | "merge"): void {
  if (density === "deep" && crowdedCells !== "merge") throw new Error(`Narrow-band FLIP collarDensity "deep" seeds samples of two units, which need crowdedCells "merge" (got "${crowdedCells}")`);
}

// ---- WGSL fragments. Each default returns the text the solver compiled before these values, exactly.

/** activitySpread's neighbour loop over tile t at p (the loop that writes heat). */
export function narrowBandCollarSpreadWGSL(spread: NarrowBandCollarSpread, wetBit: number): string {
  const body = `for(var z=-R;z<=R;z++){for(var y=-R;y<=R;y++){for(var x=-R;x<=R;x++){
   let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
   let value=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+umTileAt(vec3u(q))]));
   heat=max(heat,value*select(0.5,1.0,x==0&&y==0&&z==0));
  }}}`;
  if (spread === "neighbours") return body.replaceAll("-R", "-1").replaceAll("<=R", "<=1");
  return `// collarSpread "liquid": a tile the head census found dry reads its own heat alone.
  let reach=select(0i,1i,(atomicLoad(&bins[NB_ACTIVITY_SCORES+2u*t+1u])&${wetBit >>> 0}u)!=0u);
  ${body.replaceAll("-R", "-reach").replaceAll("<=R", "<=reach")}`;
}
/** nbFinish's arrival heat: a grid-supported sample's heat where it ends (arrived: its tile's target heat). */
export function narrowBandCollarArrivalWGSL(spread: NarrowBandCollarSpread): string {
  return spread === "neighbours" ? "if(arrived>0.0){heat=min(heat,max(arrived,1.0));}"
    : "// collarSpread \"liquid\": erasure heat stays in heated tiles wherever a sample ends.\n  heat=min(heat,max(arrived,1.0));";
}
/** The seed target of a cell at centre depth `depth` whose tile's target heat is `heat`
 * (WGSL expressions); `source`: a WGSL bool, a source cell (which keeps the band's density). */
export function narrowBandSeedTargetWGSL(density: NarrowBandCollarDensity, depth: string, heat: string, source = "false"): string {
  return density === "band" ? `nbSeedTarget(${depth})` : `nbCollarSeedTarget(${heat},${depth},${source})`;
}
/** Helpers the "deep" density's targets call, after nbSeedTarget. */
export function narrowBandCollarDensityWGSL(density: NarrowBandCollarDensity): string {
  return density === "band" ? "" : `// collarDensity "deep": a collar cell (target heat in (0, 1], not a source's) seeds four
// samples of two units at every depth, the band's deep density.
fn nbCollarSeedTarget(heat:f32,depth:f32,source:bool)->u32{return select(nbSeedTarget(depth),4u,heat>0.0&&heat<=1.0&&!source);}
fn nbSeedSiteAt(j:u32,count:u32)->u32{if(count==8u){return j;}return array<u32,4>(0u,3u,5u,6u)[j&3u];}
`;
}

// ---- CPU model of the same rules (tests/uniform-narrow-band-collar.test.ts).

/** A synthetic tile grid: own heat (NB_ACTIVITY_HEAT) and the census's wet bit a tile; absent tiles hold no storage. */
export interface CollarTileGrid { readonly n: readonly [number, number, number]; readonly heat: ArrayLike<number>; readonly wet: ArrayLike<boolean>; readonly present?: ArrayLike<boolean> }
/** activitySpread's target heat a tile, as narrowBandCollarSpreadWGSL writes it. */
export function narrowBandCollarTargets(grid: CollarTileGrid, spread: NarrowBandCollarSpread): Float64Array {
  const [nx, ny, nz] = grid.n, at = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const present = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && (grid.present?.[at(x, y, z)] ?? true);
  const out = new Float64Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (!present(x, y, z)) continue;
    const reach = spread === "neighbours" || grid.wet[at(x, y, z)] ? 1 : 0;
    let heat = 0;
    for (let c = -reach; c <= reach; c++) for (let b = -reach; b <= reach; b++) for (let a = -reach; a <= reach; a++) {
      if (!present(x + a, y + b, z + c)) continue;
      heat = Math.max(heat, grid.heat[at(x + a, y + b, z + c)]! * (a === 0 && b === 0 && c === 0 ? 1 : 0.5));
    }
    out[at(x, y, z)] = heat;
  }
  return out;
}
/** nbFinish's arrival heat (non-ballistic) under each spread. */
export function narrowBandCollarArrival(spread: NarrowBandCollarSpread, heat: number, arrived: number): number {
  return spread === "neighbours" ? (arrived > 0 ? Math.min(heat, Math.max(arrived, 1)) : heat) : Math.min(heat, Math.max(arrived, 1));
}
/** nbSeedTarget (graded: particleDensity "graded") and the collar's override. */
export function narrowBandSeedTarget(depth: number, graded: boolean, density: NarrowBandCollarDensity = "band", heat = 2, source = false): 4 | 8 {
  const band = graded && depth <= -1.5 ? 4 : 8;
  return density === "deep" && heat > 0 && heat <= 1 && !source ? 4 : band;
}
/** The deep density's sites: the even corners of the cell's 2x2x2 lattice. */
export const NARROW_BAND_DEEP_SITES = [0, 3, 5, 6] as const;
export const narrowBandSeedSite = (j: number, target: 4 | 8) => target === 8 ? j : NARROW_BAND_DEEP_SITES[j & 3]!;
/** umCorner(k, 2). */
export const narrowBandCorner = (k: number): [number, number, number] => [k & 1, (k >> 1) & 1, k >> 2];
