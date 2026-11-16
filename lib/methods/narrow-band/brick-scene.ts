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
 * - no other solids, bodies or sources.
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
import type { BrickInitialPhi, BrickLiquidBox, Vec3 } from "./brick-engine";
import { bootParticleCount, brickInitialLiquid, brickInitialVelocity } from "./brick-initial";
import { bootPhiAt, type BrickEngineOptions } from "./brick-engine";

/** Scenes the brick engine takes at this milestone (M2: the dam). */
export const BRICK_ENGINE_SCENES: readonly string[] = ["high-resolution-dam-break"];

export interface BrickSceneSetup {
  readonly sceneId: string;
  /** Cells an axis: the container's own (whole bricks). The storage rounds up to pages behind it; the cells past the
   * container are the box rule's wall (brick-static.wgsl.ts), as the old engine's voxel shell and padding close them. */
  readonly lattice: Vec3;
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
}

/** The scene's setup, or every reason the brick engine cannot run it. */
export function brickSceneSetup(scene: SceneDescription, values: Readonly<Record<string, unknown>>): { setup?: BrickSceneSetup; gaps: string[] } {
  const gaps: string[] = [];
  const id = scene.sceneId ?? "(unnamed)";
  if (!BRICK_ENGINE_SCENES.includes(id)) gaps.push(`scene "${id}" is not enabled on the brick engine (M2 enables ${BRICK_ENGINE_SCENES.join(", ")})`);
  const c = scene.container, f = scene.fluid;
  if (c.walls === "none" || c.boundary === "free-space") gaps.push("an open world (no container walls) needs the moving window (M6)");
  if (c.shape === "sphere") gaps.push("a spherical container needs static solids (M3)");
  if (c.top !== "closed") gaps.push("an open top needs an air boundary above the lattice (M3)");
  if (solidVoxelEditsForScene(scene).length > 0) gaps.push("solid voxels beyond the container shell need solidStamp (M3)");
  // The static hooks read the voxel mask (brick-static.wgsl.ts); a ground field or a procedural boundary sampled as a
  // field (umUnitBoxOpen's terrain, umBoundaryTouchesBox) is not read yet, so it must not run as open water.
  if (scene.terrain) gaps.push("terrain needs the static solid's ground field (M3)");
  if (scene.solidBoundary && scene.solidBoundary.collision !== "voxels") gaps.push("a procedural boundary sampled as a field needs boundary records (M3)");
  if ((scene.rigidBodies?.length ?? 0) > 0) gaps.push("rigid bodies need lane P's bodies (M5)");
  if (f.inflow || (f.scheduledDrops?.length ?? 0) > 0) gaps.push("sources need lane F's sources (M4)");
  // F's stages refuse the forces they do not implement (velocity-formulas nbfVelocitySupported): a gap here, so "auto"
  // never selects a scene whose stages would throw at creation.
  try { nbfVelocitySupported({ viscosity: f.dynamicViscosity_Pa_s, surfaceTension: f.surfaceTension_N_m, noSlip: false }); }
  catch (e) { gaps.push(`${(e as Error).message.replace(/^NB-FLIP brick engine: /, "")} (F)`); }
  // Padding (M3 stage 1): the lattice is the container's cells, and the box rule closes the cells past it, as the old
  // engine's shell does (its padding lies behind the shell, so no liquid reaches it). A container that is not whole
  // bricks would cut a brick and a 4h tile: that needs solidStamp's cut slots.
  const cells = sceneLatticeDimensions(scene), lattice = cells, paged = sceneFluidLatticeDimensions(scene);
  if (cells.some(n => n % 8 !== 0)) gaps.push(`the container's ${cells.join("x")} cells are not whole bricks: padding inside a brick needs solidStamp's cuts (M3)`);
  // The census and the C level set run whole 32^3 pages (setCensus, levelset-coarse 0x4c430001). Padding the lattice to
  // whole pages needs F's and P's walls at the box rule's box instead of the lattice's (M3): until then, refused here.
  else if (cells.some(n => n % 32 !== 0)) gaps.push(`the container's ${cells.join("x")} cells are not whole pages: padding to 32-cell pages needs F's and P's walls at the box (M3)`);
  const hs = [c.width_m / cells[0], c.height_m / cells[1], c.depth_m / cells[2]];
  if (hs.some(h => Math.abs(h - hs[0]!) > 1e-9 * hs[0]!)) gaps.push(`cells are not cubic (${hs.map(h => h.toPrecision(6)).join(", ")} m)`);
  const fixed = uniformFixedStep_s(values.timeStep);
  const dt = fixed ?? scene.numerics.fixedDt_s;
  if (!(dt !== undefined && dt > 0)) gaps.push("the brick engine needs a fixed step (no substeps)");
  if (gaps.length) return { gaps };
  // The initial liquid (brick-initial.ts): a dam-break box or a tank-fill slab keeps its closed form; any other
  // description is the old engine's own t = 0 φ as a page table.
  const n = lattice, g = f.gravity_m_s2, farPhi = Math.max(...paged), band = f.brickTiers?.particleBandDepth ?? 3;
  const start = brickInitialLiquid(scene, { lattice: [n[0]!, n[1]!, n[2]!], farPhi, h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z] });
  // Pools: the scene's authored brickPools, else the sizing rule (brickPoolRule) from the boot. Either way they are fixed:
  // an overrun latches FAIL.POOL / PARTICLES and is never grown.
  const p = f.brickPools ?? brickPoolRule([n[0]!, n[1]!, n[2]!], brickInitialExtent({ lattice: [n[0]!, n[1]!, n[2]!], farPhi, liquid: start.liquid, initial: start.initial }),
    start.bricks.length, bootParticleCount({ lattice: [n[0]!, n[1]!, n[2]!], farPhi, liquid: start.liquid, initial: start.initial }, start.bricks, band));
  return {
    gaps,
    setup: {
      sceneId: id, lattice: [n[0]!, n[1]!, n[2]!], farPhi, h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z], density_kg_m3: f.density_kg_m3, viscosity_Pa_s: f.dynamicViscosity_Pa_s, surfaceTension_N_m: f.surfaceTension_N_m,
      capacities: { f: p.f, c: p.c, b: p.b, k: p.k, particles: p.particles, ...("t" in p && p.t !== undefined ? { t: p.t } : {}) }, particleBandDepth: band,
      liquid: start.liquid, initial: start.initial, initialVelocity: brickInitialVelocity(scene, [n[0]!, n[1]!, n[2]!]), bootBricks: start.bricks, bootPages: start.pages, origin_m: [-0.5 * c.width_m, 0, -0.5 * c.depth_m],
    },
  };
}

/** The initial liquid's volume in cells (vertices with φ < 0 inside the container) and its top (one past the highest
 * such vertex's y), from the boot's own φ (bootPhiAt: the closed form or the page table). A brick whose centre φ is
 * past its half-diagonal (√3·4 < 7 cells) is wholly in or out and counts without its cells. */
export function brickInitialExtent(o: Pick<BrickEngineOptions, "lattice" | "farPhi" | "liquid" | "initial">): { cells: number; top: number } {
  const at = bootPhiAt(o), n = o.lattice;
  let cells = 0, top = 0;
  for (let bz = 0; 8 * bz < n[2]; bz++) for (let by = 0; 8 * by < n[1]; by++) for (let bx = 0; 8 * bx < n[0]; bx++) {
    const lo = [8 * bx, 8 * by, 8 * bz], hi = lo.map((v, a) => Math.min(v + 8, n[a]!)), centre = at([lo[0]! + 4, lo[1]! + 4, lo[2]! + 4]);
    if (centre >= 7) continue;
    if (centre <= -7 && hi.every((v, a) => v - lo[a]! === 8)) { cells += 512; top = Math.max(top, hi[1]!); continue; }
    for (let z = lo[2]!; z < hi[2]!; z++) for (let y = lo[1]!; y < hi[1]!; y++) for (let x = lo[0]!; x < hi[0]!; x++) {
      if (at([x, y, z]) < 0) { cells++; top = Math.max(top, y + 1); }
    }
  }
  return { cells, top };
}

/**
 * The pool sizing rule (progress-lane-e.md, 2026-11-17, measured against nine scenes' peaks) for a scene with no
 * authored brickPools. n is the lattice in cells; the liquid's cells V and top come from the boot; bootF is the boot's F
 * bricks and bootP the boot seeds' bound (bootParticleCount).
 * - The window is the hard cap: WF = Π(n/8 + 1), WC = Π(n/32 + 1).
 * - Energy: no liquid rises past twice its initial top plus a ring: EF = floorF·min(n_y/8 + 1, ⌈2 top/8⌉ + 2),
 *   EC = floorC·min(n_y/32 + 1, ⌈2 top/32⌉ + 1), floor* the window's x·z.
 * - Spread: a film 1/16 cell thick over the floor, four brick layers deep (SF = 4 min(floorF, ⌈V/4⌉)); every liquid
 *   brick at least 1/16 full (LF = ⌈V/32⌉); the boot's bricks with a quarter's margin; 128 bricks at least.
 * - f = min(WF, EF, max(SF, LF, ⌈1.25 bootF⌉, 128)); c = min(WC, EC, f); particles = 3 bootP (at least one bank's
 *   1024); b = 16; k = 0 (no static cut is bound yet: solidStamp's KCAP rule comes with the first mask scene).
 * - t (setTransfer's listed tiles a step) = min(512 c, max(2 f, 4096)): setTransfer lists a tile of a C_ALL page at
 *   most once a step, so 512 c is exact; the layout's 2 f default (the dam's) under-counts a small f whose 4h tiles
 *   change (falling-water-torus listed 251 > 2·125 at step 3). Its xfer scratch is 3 KB a tile, so 4096 is 12.6 MB.
 */
export function brickPoolRule(n: Vec3, liquid: { cells: number; top: number }, bootF: number, bootP: number): BrickCapacities {
  const nb = n.map(v => Math.floor(v / 8)), np = n.map(v => Math.floor(v / 32));
  const floorF = (nb[0]! + 1) * (nb[2]! + 1), floorC = (np[0]! + 1) * (np[2]! + 1);
  const WF = floorF * (nb[1]! + 1), WC = floorC * (np[1]! + 1);
  const EF = floorF * Math.min(nb[1]! + 1, Math.ceil(2 * liquid.top / 8) + 2), EC = floorC * Math.min(np[1]! + 1, Math.ceil(2 * liquid.top / 32) + 1);
  const V = liquid.cells, SF = 4 * Math.min(floorF, Math.ceil(V / 4)), LF = Math.ceil(V / 32);
  const f = Math.min(WF, EF, Math.max(SF, LF, Math.ceil(1.25 * bootF), 128));
  const c = Math.min(WC, EC, f);
  return { f, c, b: 16, k: 0, particles: Math.max(3 * bootP, 1024), t: Math.min(512 * c, Math.max(2 * f, 4096)) };
}
