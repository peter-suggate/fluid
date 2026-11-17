/**
 * The brick engine's scene adapter (lane E): what a SceneDescription becomes
 * (lattice, spacing, step, gravity, pools, initial liquid), or the explicit
 * list of what the engine cannot run in it yet.
 *
 * M2 runs high-resolution-dam-break only:
 * - a closed box container (its shell is the lattice's walls, which the
 *   engine closes) of whole pages, cubic cells;
 * - any initial liquid (brick-initial.ts: the closed form or the old engine's t = 0 φ as a page table), at rest
 *   or with the scene's initial velocity;
 * - scheduled drops (lane F's sources, brick-source.ts); no other solids, bodies or an inflow.
 * Density, viscosity and surface tension pass to lane F's stages, which
 * apply the forces (as the old engine's forces kernel) or refuse them.
 * Anything else is a gap, and the method runs the scene on the old engine.
 */
import type { SceneDescription } from "../../core/model";
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { uniformFixedStep_s } from "../uniform/uniform-paper";
import type { BrickCapacities } from "./brick-abi";
import { nbfVelocitySupported } from "./velocity-formulas.wgsl";
import { brickRigidBodyGaps } from "./bodies-host";
import type { BrickInitialPhi, BrickLiquidBox, Vec3 } from "./brick-engine";
import { bootParticleCount, brickInitialLiquid, brickInitialVelocity } from "./brick-initial";
import { brickSources, type BrickSources } from "./brick-source";
import { BRICK_BOOT_BAND, bootPhiAt, type BrickEngineOptions } from "./brick-engine";

/** Scenes the brick engine takes: M2's dam, and the first enabled batch (lane E, 2026-11-17: 60 steps against the old
 * engine from a frozen 0a81b19e, each brick |peak| and |final| volume drift within the old engine's worst repeat plus
 * max(1%, its run noise), a clean smoke; progress-lane-e.md). Named fails stay off until their causes are fixed. */
export const BRICK_ENGINE_SCENES: readonly string[] = [
  "high-resolution-dam-break",
  "minimal-power-dam-break-32", "minimal-power-dam-break-64", "falling-water-torus",
  "nbflip-analytic-rest", "nbflip-analytic-standing-wave", "nbflip-analytic-dam-break", "nbflip-analytic-mound",
  // The 100-cell droplets (lane E r29, 300 steps on their authored pools): both engines lose the tension-free film below
  // resolution (64^3: brick -14%, old -77%; fill-256-100: brick -90.5%, old -92.9%). power-droplet-128 is off: the brick
  // ends at -90.7% against the old engine's -40.1% (|peak| 90.7 against 78.0). Its sub-resolution film loses particles
  // (747 -> 441 by step 280, under 4 a wet brick) where the old engine's film reaches the walls and thickens again.
  "power-droplet-64", "power-fill-256-100",
  // The 800-cell fill (r29): 4.01 / -2.39 at 60 steps against the bar's 9.67 / 2.67; over 300 steps on its authored
  // pools the film loses -77.4% where the old engine loses -87.4%. power-fill-256-6400 is off: +4.64 / +4.28 at 60
  // steps (two runs) against the bar's 4.01 (old -3.01 / -2.97), the moving-liquid volume gain (F's named open fail).
  "power-fill-256-800",
  "cm12-figure-5", "cm12-figure-6", "cm12-figure-9", "mass-conserving-figure-9-dam-break", "coarse-first-pool-impact",
  // A splash scene (lane E r23, 2026-11-17, the batch rule's splash clause), on the pool rule's splash term.
  "nbflip-analytic-tilted-free-fall",
  // Re-judged after F's sym8 (r26, the splash clause: 2 brick / 2 old): peak -24.86 / -24.95 at 60 steps and final
  // -0.63 / -0.81 at 120, against the bar's 25.55 / 1.18; the long run (300 steps) F 316/405, no latch.
  "nbflip-analytic-free-fall",
  // A splash scene (r29): peak 5.51 / 5.73 at 60 steps and final -5.56 / -6.26 at 120, against the bar's 7.29 / 18.43
  // (old 6.28 / 6.29 and 15.85 / 17.14); the long run (480 steps) F 2545/3670, no latch.
  "nbflip-stair-cube",
  // Splash (r23): it gains +2.05% at 120 steps, where the old engine loses 10.3% (on the full box's page pool; its
  // authored pool exhausts at frame 79). Re-judged after F's sym8 lands, which is expected to remove the gain.
  "cm12-figure-7-256",
  // The rigid coupling oracles (lane P, M5; 16^3 padded to 32^3), on their authored pools.
  "rigid-hydrostatic", "rigid-float", "rigid-sink",
];

/** The air an open top keeps above the container's pages, in cells: one page. */
export const BRICK_OPEN_TOP_HEADROOM = 32;

export interface BrickSceneSetup {
  readonly sceneId: string;
  /** Cells an axis: the container's cells rounded up to whole pages (sceneFluidLatticeDimensions, the old engine's
   * fluid box). Where that pads past the container, `box` is the container's cells, and the cells past it are the box
   * rule's wall (brick-abi.md §2), as the old engine's voxel shell and padding close them. */
  readonly lattice: Vec3;
  /** The container's cells (whole bricks) where the lattice is padded past them; absent: the lattice. Under an open top
   * its y is the lattice's: the headroom above the container is in the box (BRICK_OPEN_TOP_HEADROOM). */
  readonly box?: Vec3;
  /** An open top (container.top "open"): the lattice holds BRICK_OPEN_TOP_HEADROOM cells of air above the container,
   * and liquid that reaches its top brick layer latches (brick-set.wgsl.ts setMark, FAIL.REACH info bit 31). */
  readonly openTop?: true;
  /** NB_PHI_FAR in h: the old engine's far φ, its page-rounded lattice's largest extent (fluidLatticeFarPhi); the lattice's own when absent. */
  readonly farPhi?: number;
  readonly h_m: number;
  readonly dt_s: number;
  readonly gravity_mps2: Vec3;
  readonly density_kg_m3: number;
  readonly viscosity_Pa_s: number;
  readonly surfaceTension_N_m: number;
  readonly capacities: BrickCapacities;
  /** The particle band's depth in cells (scene.fluid.brickTiers.particleBandDepth, 3 when absent). */
  readonly particleBandDepth: number;
  /** Initial liquid in cells, the closed form (closed-wall faces continued through the wall) ... */
  readonly liquid: readonly BrickLiquidBox[];
  /** ... or the page table of the old engine's t = 0 φ (brick-initial.ts; `liquid` then empty). */
  readonly initial?: BrickInitialPhi;
  /** The boot's F bricks and tiered page-only candidates (brick-initial.ts brickInitialLiquid), for the pool rule. */
  readonly bootBricks?: readonly number[];
  readonly bootPages?: readonly number[];
  /** The liquid's initial velocity on its tiles (brick-initial.ts brickInitialVelocity); absent at rest. */
  readonly initialVelocity?: { readonly velocity: Vec3; readonly tiles: readonly number[] };
  /** The lattice's low corner (m) and the container's cells. */
  readonly origin_m: Vec3;
  /** The scheduled drops in source form (brick-source.ts brickSources); absent: none. */
  readonly sources?: BrickSources;
}

/** The scene's setup, or every reason the brick engine cannot run it. */
export function brickSceneSetup(scene: SceneDescription, values: Readonly<Record<string, unknown>>): { setup?: BrickSceneSetup; gaps: string[] } {
  const gaps: string[] = [];
  const id = scene.sceneId ?? "(unnamed)";
  if (!BRICK_ENGINE_SCENES.includes(id)) gaps.push(`scene "${id}" is not enabled on the brick engine (brick-scene.ts BRICK_ENGINE_SCENES lists the ${BRICK_ENGINE_SCENES.length} it runs)`);
  const c = scene.container, f = scene.fluid;
  if (c.walls === "none" || c.boundary === "free-space") gaps.push("an open world (no container walls) needs the moving window (M6)");
  if (c.shape === "sphere") gaps.push("a spherical container needs static solids (M3)");
  const openTop = c.top === "open";
  if (c.top !== "closed" && !openTop) gaps.push(`container.top "${c.top}" is not a brick engine top (closed or open)`);
  if (solidVoxelEditsForScene(scene).length > 0) gaps.push("solid voxels beyond the container shell need solidStamp (M3)");
  // The static hooks read the voxel mask (brick-static.wgsl.ts); a ground field or a procedural boundary sampled as a
  // field (umUnitBoxOpen's terrain, umBoundaryTouchesBox) is not read yet, so it must not run as open water.
  if (scene.terrain) gaps.push("terrain needs the static solid's ground field (M3)");
  if (scene.solidBoundary && scene.solidBoundary.collision !== "voxels") gaps.push("a procedural boundary sampled as a field needs boundary records (M3)");
  // Rigid bodies (lane P, M5: bodies-host.ts): the body stages need K slots, and the host's own refusals.
  gaps.push(...brickRigidBodyGaps(scene));
  // Scheduled drops are lane F's sources (brick-source.ts); an inflow is not ported yet.
  if (f.inflow) gaps.push("an inflow needs lane F's inflow source (M4)");
  // The pool rule sizes from the t = 0 liquid alone: a drop scene would latch POOL_F or PARTICLES as its drops land.
  if (f.scheduledDrops?.length && !f.brickPools) gaps.push("scheduled drops need authored brickPools (the pool rule sizes the t = 0 liquid only)");
  // F's stages refuse the forces they do not implement (velocity-formulas nbfVelocitySupported): a gap here, so "auto"
  // never selects a scene whose stages would throw at creation.
  try { nbfVelocitySupported({ viscosity: f.dynamicViscosity_Pa_s, surfaceTension: f.surfaceTension_N_m, noSlip: false }); }
  catch (e) { gaps.push(`${(e as Error).message.replace(/^NB-FLIP brick engine: /, "")} (F)`); }
  // Padding (M3): the census and the C level set run whole 32^3 pages (setCensus, levelset-coarse 0x4c430001), so the
  // lattice is the container's cells rounded up to whole pages (the old engine's fluid box), and the box rule closes
  // the cells past the container at the box (brick-abi.md §2), as the old engine's shell does (its padding lies behind
  // the shell, so no liquid reaches it). A container that is not whole bricks would cut a brick and a 4h tile: that
  // needs solidStamp's cut slots.
  // An open top (M3): the old engine's frame holds AIR pages above the box, and its x and z walls rise to the frame's
  // top. Here the lattice holds BRICK_OPEN_TOP_HEADROOM cells (a page) more above the container's pages, in the box:
  // the air there is the free surface's (p = 0), the walls rise with it, and the lattice's top is a lid that liquid
  // never meets, because liquid in its top brick layer latches first (the old engine's page-edge fatal, explicit).
  const cells = sceneLatticeDimensions(scene), old = sceneFluidLatticeDimensions(scene);
  const paged: Vec3 = openTop ? [old[0]!, old[1]! + BRICK_OPEN_TOP_HEADROOM, old[2]!] : [old[0]!, old[1]!, old[2]!];
  const walls: Vec3 = openTop ? [cells[0]!, paged[1], cells[2]!] : [cells[0]!, cells[1]!, cells[2]!];
  const padded = walls.some((n, a) => n !== paged[a]), lattice = paged, box = padded ? walls : undefined;
  if (walls.some(n => n % 8 !== 0)) gaps.push(`the container's ${cells.join("x")} cells are not whole bricks: padding inside a brick needs solidStamp's cuts (M3)`);
  // Padded to whole pages: F's walls (512e7a9f), the release readers' (7b9e12e7) and P's pressure walls (fecb32a9)
  // are at the box.
  const hs = [c.width_m / cells[0], c.height_m / cells[1], c.depth_m / cells[2]];
  if (hs.some(h => Math.abs(h - hs[0]!) > 1e-9 * hs[0]!)) gaps.push(`cells are not cubic (${hs.map(h => h.toPrecision(6)).join(", ")} m)`);
  const fixed = uniformFixedStep_s(values.timeStep);
  const dt = fixed ?? scene.numerics.fixedDt_s;
  if (!(dt !== undefined && dt > 0)) gaps.push("the brick engine needs a fixed step (no substeps)");
  if (gaps.length) return { gaps };
  // The initial liquid (brick-initial.ts): a dam-break box or a tank-fill slab keeps its closed form; any other
  // description is the old engine's own t = 0 φ as a page table.
  const n = lattice, g = f.gravity_m_s2, farPhi = Math.max(...old), band = f.brickTiers?.particleBandDepth ?? 3;
  const L: Vec3 = [n[0]!, n[1]!, n[2]!], B: Vec3 | undefined = box ? [box[0]!, box[1]!, box[2]!] : undefined, inBox = B ? { box: B } : {};
  const start = brickInitialLiquid(scene, { lattice: L, ...inBox, farPhi, h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z] });
  const origin_m: Vec3 = [-0.5 * c.width_m, 0, -0.5 * c.depth_m];
  const sources = brickSources(scene, { lattice: L, ...inBox, h_m: hs[0]!, origin_m, dt_s: dt! });
  // Pools: the scene's authored brickPools, else the sizing rule (brickPoolRule) from the boot. Either way they are fixed:
  // an overrun latches FAIL.POOL / PARTICLES and is never grown.
  const extent = brickInitialExtent({ lattice: L, ...inBox, farPhi, liquid: start.liquid, initial: start.initial });
  // The boot's candidates (its F bricks and page-only keys) fill an f-long list (BrickEngine.create).
  const p = f.brickPools ?? brickPoolRule(L, extent, Math.max(start.bricks.length + start.pages.length, extent.resident), bootParticleCount({ lattice: L, ...inBox, farPhi, liquid: start.liquid, initial: start.initial }, start.bricks, band));
  // The body stages take a K slot a body brick (A5): a scene with bodies authors brickPools.k from its NEAR_BODY peak.
  if ((scene.rigidBodies?.length ?? 0) > 0 && !(p.k > 0)) return { gaps: [...gaps, "rigid bodies need K slots: author brickPools.k from the NEAR_BODY peak"] };
  return {
    gaps,
    setup: {
      sceneId: id, lattice: L, ...inBox, ...(openTop ? { openTop: true as const } : {}), farPhi, h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z], density_kg_m3: f.density_kg_m3, viscosity_Pa_s: f.dynamicViscosity_Pa_s, surfaceTension_N_m: f.surfaceTension_N_m,
      capacities: { f: p.f, c: p.c, b: p.b, k: p.k, particles: p.particles, ...("t" in p && p.t !== undefined ? { t: p.t } : {}) }, particleBandDepth: band,
      liquid: start.liquid, initial: start.initial, initialVelocity: brickInitialVelocity(scene, L, B), bootBricks: start.bricks, bootPages: start.pages, origin_m,
      ...(sources ? { sources } : {}),
    },
  };
}

/** The initial liquid's volume in cells (vertices with φ < 0 inside the container) and its top (one past the highest
 * such vertex's y), from the boot's own φ (bootPhiAt: the closed form or the page table). A brick whose centre φ is
 * past its half-diagonal (√3·4 < 7 cells) is wholly in or out and counts without its cells.
 * `resident` is the tiered boot's first residency: the liquid bricks within BRICK_BOOT_BAND of the surface (a vertex
 * with φ > −band; bootTieredSet's sources) dilated by two bricks a side (Chebyshev), within the window. The engine's
 * first residency is that set (the h ring's tiles, the particle and seed classes: ≤ 16 cells past a liquid tile; the
 * free fall's step-1 F box, 210 bricks, is exactly it), where the boot's own brick list (100) is not. The deeper
 * liquid is C from the boot (lane F, the still-tank item C, 2026-11-17).
 * `free` is the liquid above its rest depth (cells whose centre lies above V / floor cells): the liquid with
 * potential energy to spend, the only liquid that can spread (brickPoolRule's LF).
 */
export function brickInitialExtent(o: Pick<BrickEngineOptions, "lattice" | "box" | "farPhi" | "liquid" | "initial">): { cells: number; top: number; drop: number; resident: number; free: number } {
  // Over the container's cells (the box: a padded lattice's cells past it are wall).
  const at = bootPhiAt(o), n = o.box ?? o.lattice, nb = n.map(v => Math.floor(v / 8) + 1), liquid: [number, number, number][] = [];
  const rows = new Float64Array(n[1]!);
  let cells = 0, top = 0, bot = Infinity;
  for (let bz = 0; 8 * bz < n[2]; bz++) for (let by = 0; 8 * by < n[1]; by++) for (let bx = 0; 8 * bx < n[0]; bx++) {
    const lo = [8 * bx, 8 * by, 8 * bz], hi = lo.map((v, a) => Math.min(v + 8, n[a]!)), centre = at([lo[0]! + 4, lo[1]! + 4, lo[2]! + 4]);
    if (centre >= 7) continue;
    if (centre <= -7 && hi.every((v, a) => v - lo[a]! === 8)) {
      cells += 512; top = Math.max(top, hi[1]!); bot = Math.min(bot, lo[1]!); for (let y = lo[1]!; y < hi[1]!; y++) rows[y] += 64;
      // Wholly liquid: within the band where its centre is (the half-diagonal past −band reaches φ > −band).
      if (centre > -BRICK_BOOT_BAND - 7) liquid.push([bx, by, bz]);
      continue;
    }
    let wet = false, near = false;
    for (let z = lo[2]!; z < hi[2]!; z++) for (let y = lo[1]!; y < hi[1]!; y++) for (let x = lo[0]!; x < hi[0]!; x++) {
      const d = at([x, y, z]);
      if (d < 0) { cells++; rows[y]++; top = Math.max(top, y + 1); bot = Math.min(bot, y); wet = true; if (d > -BRICK_BOOT_BAND) near = true; }
    }
    if (wet && near) liquid.push([bx, by, bz]);
  }
  const resident = new Set<number>();
  for (const [bx, by, bz] of liquid) for (let dz = -2; dz <= 2; dz++) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const x = bx + dx, y = by + dy, z = bz + dz;
    if (x >= 0 && y >= 0 && z >= 0 && x < nb[0]! && y < nb[1]! && z < nb[2]!) resident.add(x + nb[0]! * (y + nb[1]! * z));
  }
  // The drop: the liquid's lowest cell above its rest depth (its cells spread over the box's floor), in cells.
  const rest = cells / (n[0]! * n[2]!), drop = cells > 0 ? bot - rest : 0;
  let free = 0;
  for (let y = 0; y < n[1]!; y++) if (y + 0.5 > rest) free += rows[y]!;
  return { cells, top, drop, resident: resident.size, free };
}

/**
 * The pool sizing rule (progress-lane-e.md, 2026-11-17, measured against nine scenes' peaks) for a scene with no
 * authored brickPools. n is the lattice in cells; the liquid's cells V, top, drop and free come from the boot; bootF is
 * the boot's F residency (brickInitialExtent's `resident`, at least the tiered boot's candidates: its bricks and
 * page-only keys, bootTieredSet) and bootP the boot seeds' bound (bootParticleCount).
 * - The window is the hard cap: WF = Π(n/8 + 1), WC = Π(n/32 + 1).
 * - Energy: no liquid rises past twice its initial top plus a ring: EF = floorF·min(n_y/8 + 1, ⌈2 top/8⌉ + 2),
 *   EC = floorC·min(n_y/32 + 1, ⌈2 top/32⌉ + 1), floor* the window's x·z.
 * - Spread: a film 1/16 cell thick over the floor, four brick layers deep (SF = 4 min(floorF, ⌈V/4⌉)); every brick
 *   of the liquid that can spread at least 1/16 full (LF = ⌈free/32⌉, free the liquid above its rest depth: tiered, the
 *   liquid below it stays C unless the liquid above reaches it, and a still tank has none; lane F, the still-tank item
 *   C, 2026-11-17; free = V where absent); the boot's candidates with a quarter's margin; 128 bricks at least.
 *   A small liquid with no tension thins past 1/16 cell as it spreads (the 100-cell droplets: F 162 / 418 / 452 at
 *   64^3 / 128^3 / 256^3 by step 300, every brick a wet floor brick or its ring; the rule's 128 latched POOL_F near
 *   step 100). The spread is time-bound, not closed-form, so such a scene authors brickPools from its long run.
 * - Splash (lane E, 2026-11-17; EMPIRICAL, fitted to five measured peaks, not derived): a scene whose liquid starts
 *   two bricks or more above its rest depth (drop ≥ 16 cells, brickInitialExtent) splashes over the floor on impact.
 *   SPF = 10·min(floorF, ⌈1.5 V/64⌉): eight brick layers with the quarter's margin, over the floor or, for a small
 *   liquid, the area V covers as a film 2/3 cell thick. Its particles: 10 V, the seed target (8 a cell) with the
 *   margin, since nearly every splashing cell is band. The peaks (160 steps, no latch; progress-lane-e.md):
 *     scene        floorF  peakF  layers  SPF (after caps)  peak particles  10 V
 *     cm12-fig-7     289   2437    8.4   2890 (1.19x)       0.29M      0.33M
 *     cm12-fig-7-256 1089  8948    8.2   10890 (1.22x)      2.05M      2.68M
 *     stair-cube     625   2600    -     3670 (1.41x)       0.15M      0.16M
 *     free fall       45    289    6.4   405 = WF (1.40x)
 *     tilted fall     45    238    5.3   405 = WF (1.70x)
 *   The layers do not grow with the drop in cells (fig-7 at 128 and 256: drop 69 and 137 cells, 8.3 layers both), so
 *   the drop is only the trigger. Stair is the weakest point: the only V-limited case, which sets the 1.5. A splash
 *   past the fit latches FAIL.POOL / PARTICLES explicitly; pools never grow.
 * - f = min(WF, EF, max(SF, LF, ⌈1.25 bootF⌉, 128, SPF)); c = min(WC, EC, f); particles = max(3 bootP, 1024, the
 *   splash's 10 V); b = 16; k = 0 (no static cut is bound yet: solidStamp's KCAP rule comes with the first mask
 *   scene).
 * - t (setTransfer's listed tiles a step) = min(512 c, max(2 f, 4096)): setTransfer lists a tile of a C_ALL page at
 *   most once a step, so 512 c is exact; the layout's 2 f default (the dam's) under-counts a small f whose 4h tiles
 *   change (falling-water-torus listed 251 > 2·125 at step 3). Its xfer scratch is 3 KB a tile, so 4096 is 12.6 MB.
 */
export const BRICK_SPLASH_DROP = 16;
export function brickPoolRule(n: Vec3, liquid: { cells: number; top: number; drop: number; free?: number }, bootF: number, bootP: number): BrickCapacities {
  const nb = n.map(v => Math.floor(v / 8)), np = n.map(v => Math.floor(v / 32));
  const floorF = (nb[0]! + 1) * (nb[2]! + 1), floorC = (np[0]! + 1) * (np[2]! + 1);
  const WF = floorF * (nb[1]! + 1), WC = floorC * (np[1]! + 1);
  const EF = floorF * Math.min(nb[1]! + 1, Math.ceil(2 * liquid.top / 8) + 2), EC = floorC * Math.min(np[1]! + 1, Math.ceil(2 * liquid.top / 32) + 1);
  const V = liquid.cells, SF = 4 * Math.min(floorF, Math.ceil(V / 4)), LF = Math.ceil((liquid.free ?? V) / 32);
  const splash = liquid.drop >= BRICK_SPLASH_DROP, SPF = splash ? 10 * Math.min(floorF, Math.ceil(1.5 * V / 64)) : 0, SPP = splash ? Math.ceil(10 * V) : 0;
  const f = Math.min(WF, EF, Math.max(SF, LF, Math.ceil(1.25 * bootF), 128, SPF));
  const c = Math.min(WC, EC, f);
  return { f, c, b: 16, k: 0, particles: Math.max(3 * bootP, 1024, SPP), t: Math.min(512 * c, Math.max(2 * f, 4096)) };
}
