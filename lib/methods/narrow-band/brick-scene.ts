/**
 * The brick engine's scene adapter (lane E): what a SceneDescription becomes
 * (lattice, spacing, step, gravity, pools, initial liquid), or the explicit
 * list of what the engine cannot run in it yet.
 *
 * M2 runs high-resolution-dam-break only:
 * - a closed box container (its shell is the lattice's walls, which the
 *   engine closes) of whole pages, cubic cells;
 * - the dam-break initial condition at rest;
 * - no other solids, bodies or sources.
 * Density, viscosity and surface tension pass to lane F's stages, which
 * apply the forces (as the old engine's forces kernel) or refuse them.
 * Anything else is a gap, and the method runs the scene on the old engine.
 */
import type { SceneDescription } from "../../core/model";
import { sceneDamBreakBox } from "../../core/initial-fluid";
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { uniformFixedStep_s } from "../uniform/uniform-paper";
import type { BrickCapacities } from "./brick-abi";
import type { BrickLiquidBox, Vec3 } from "./brick-engine";

/** Scenes the brick engine takes at this milestone (M2: the dam). */
export const BRICK_ENGINE_SCENES: readonly string[] = ["high-resolution-dam-break"];

export interface BrickSceneSetup {
  readonly sceneId: string;
  /** Cells an axis (whole pages). */
  readonly lattice: Vec3;
  readonly h_m: number;
  readonly dt_s: number;
  readonly gravity_mps2: Vec3;
  readonly density_kg_m3: number;
  readonly viscosity_Pa_s: number;
  readonly surfaceTension_N_m: number;
  readonly capacities: BrickCapacities;
  /** The particle band's depth in cells (scene.fluid.brickTiers.particleBandDepth, 3 when absent). */
  readonly particleBandDepth: number;
  /** Initial liquid in cells; closed-wall faces are continued through the wall. */
  readonly liquid: readonly BrickLiquidBox[];
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
  if ((scene.rigidBodies?.length ?? 0) > 0) gaps.push("rigid bodies need lane P's bodies (M5)");
  if (f.inflow || (f.scheduledDrops?.length ?? 0) > 0) gaps.push("sources need lane F's sources (M4)");
  if (f.initialCondition !== "dam-break" || f.initialLiquidVolumes?.length || f.initialBrickSeeds_m?.length) gaps.push("only the dam-break initial condition is supported (M2)");
  const v0 = f.initialVelocity_m_s;
  if (v0 && (v0.x !== 0 || v0.y !== 0 || v0.z !== 0)) gaps.push("an initial liquid velocity is not supported (M2)");
  if (!f.brickPools) gaps.push("scene.fluid.brickPools is not authored (brick-abi.md §9)");
  const cells = sceneLatticeDimensions(scene), lattice = sceneFluidLatticeDimensions(scene);
  if (cells.some((n, a) => n !== lattice[a])) gaps.push(`the container's ${cells.join("x")} cells are not whole pages (${lattice.join("x")}): padding needs static solids (M3)`);
  const hs = [c.width_m / cells[0], c.height_m / cells[1], c.depth_m / cells[2]];
  if (hs.some(h => Math.abs(h - hs[0]!) > 1e-9 * hs[0]!)) gaps.push(`cells are not cubic (${hs.map(h => h.toPrecision(6)).join(", ")} m)`);
  const fixed = uniformFixedStep_s(values.timeStep);
  const dt = fixed ?? scene.numerics.fixedDt_s;
  if (!(dt !== undefined && dt > 0)) gaps.push("the brick engine needs a fixed step (no substeps)");
  if (gaps.length) return { gaps };
  // The dam's box in cells. A face on a closed wall drops its constraint
  // (damBreakSignedDistanceAtNode): the box runs on through the wall, here
  // 2 max(n) cells past it, so no dropped face is ever the maximum inside.
  const dam = sceneDamBreakBox(scene), n = lattice, far = 2 * Math.max(...n);
  const lo = [dam.min.x, dam.min.y, dam.min.z].map((v, a) => v * n[a]!), hi = [dam.max.x, dam.max.y, dam.max.z].map((v, a) => v * n[a]!);
  const tol = 1e-9 * Math.max(...n);
  const box: BrickLiquidBox = {
    lo: lo.map(v => v <= tol ? -far : v) as unknown as Vec3,
    hi: hi.map((v, a) => v >= n[a]! - tol ? n[a]! + far : v) as unknown as Vec3,
  };
  const p = f.brickPools!, g = f.gravity_m_s2;
  return {
    gaps,
    setup: {
      sceneId: id, lattice: [n[0]!, n[1]!, n[2]!], h_m: hs[0]!, dt_s: dt!, gravity_mps2: [g.x, g.y, g.z], density_kg_m3: f.density_kg_m3, viscosity_Pa_s: f.dynamicViscosity_Pa_s, surfaceTension_N_m: f.surfaceTension_N_m,
      capacities: { f: p.f, c: p.c, b: p.b, k: p.k, particles: p.particles }, particleBandDepth: f.brickTiers?.particleBandDepth ?? 3, liquid: [box], origin_m: [-0.5 * c.width_m, 0, -0.5 * c.depth_m],
    },
  };
}
