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
import { brickInitialLiquid, brickInitialVelocity } from "./brick-initial";

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
  if (!f.brickPools) gaps.push("scene.fluid.brickPools is not authored (brick-abi.md §9)");
  // F's stages refuse the forces they do not implement (velocity-formulas nbfVelocitySupported): a gap here, so "auto"
  // never selects a scene whose stages would throw at creation.
  try { nbfVelocitySupported({ viscosity: f.dynamicViscosity_Pa_s, surfaceTension: f.surfaceTension_N_m, noSlip: false }); }
  catch (e) { gaps.push(`${(e as Error).message.replace(/^NB-FLIP brick engine: /, "")} (F)`); }
  // Padding (M3 stage 1): the lattice is the container's cells, and the box rule closes the cells past it, as the old
  // engine's shell does (its padding lies behind the shell, so no liquid reaches it). A container that is not whole
  // bricks would cut a brick and a 4h tile: that needs solidStamp's cut slots.
  const cells = sceneLatticeDimensions(scene), lattice = cells, paged = sceneFluidLatticeDimensions(scene);
  if (cells.some(n => n % 8 !== 0)) gaps.push(`the container's ${cells.join("x")} cells are not whole bricks: padding inside a brick needs solidStamp's cuts (M3)`);
  const hs = [c.width_m / cells[0], c.height_m / cells[1], c.depth_m / cells[2]];
  if (hs.some(h => Math.abs(h - hs[0]!) > 1e-9 * hs[0]!)) gaps.push(`cells are not cubic (${hs.map(h => h.toPrecision(6)).join(", ")} m)`);
  const fixed = uniformFixedStep_s(values.timeStep);
  const dt = fixed ?? scene.numerics.fixedDt_s;
  if (!(dt !== undefined && dt > 0)) gaps.push("the brick engine needs a fixed step (no substeps)");
  if (gaps.length) return { gaps };
  // The initial liquid (brick-initial.ts): a dam-break box or a tank-fill slab keeps its closed form; any other
  // description is the old engine's own t = 0 φ as a page table.
  const n = lattice, p = f.brickPools!, g = f.gravity_m_s2, farPhi = Math.max(...paged);
  const start = brickInitialLiquid(scene, { lattice: [n[0]!, n[1]!, n[2]!], farPhi, h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z] });
  return {
    gaps,
    setup: {
      sceneId: id, lattice: [n[0]!, n[1]!, n[2]!], farPhi, h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z], density_kg_m3: f.density_kg_m3, viscosity_Pa_s: f.dynamicViscosity_Pa_s, surfaceTension_N_m: f.surfaceTension_N_m,
      capacities: { f: p.f, c: p.c, b: p.b, k: p.k, particles: p.particles }, particleBandDepth: f.brickTiers?.particleBandDepth ?? 3,
      liquid: start.liquid, initial: start.initial, initialVelocity: brickInitialVelocity(scene, [n[0]!, n[1]!, n[2]!]), bootBricks: start.bricks, bootPages: start.pages, origin_m: [-0.5 * c.width_m, 0, -0.5 * c.depth_m],
    },
  };
}
