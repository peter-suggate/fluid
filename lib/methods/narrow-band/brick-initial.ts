/**
 * The brick engine's initial liquid from any scene description (lane F): what bootPhi writes at t = 0, and the
 * boot's bricks, pages and seed count for lane E's pool rule.
 *
 * Two forms, the old engine's own two (initial-liquid-lattice.ts; its analyticInitialPhi and host-rasterised seed):
 * - Closed form (BrickLiquidBox[]): a plain dam-break box or a tank-fill slab and nothing else. A face on a closed
 *   wall drops its constraint (damBreakSignedDistanceAtNode): the box runs on through the wall, 2 max(n) cells past
 *   it, so no dropped face is ever the maximum inside. The dam keeps this form, bitwise as before.
 * - The page table (BrickInitialPhi): every other description (height fields, brick seeds, initial liquid volumes,
 *   box unions, an empty tank), sampled by the old engine's fluidLatticePagedVertexPhi, verbatim: the base fill, the
 *   height field, the seeded bricks (additive or not), the authored volumes, the box-union transform, and (once the
 *   engine runs solids) buried vertices and the wall continuation. The table holds the pages initialLiquidCellBounds
 *   meets, grown by one page (every reach-set brick and boot page lies in them); a page off it is far air, as the old
 *   engine's unstored pages are. Its size follows the liquid's pages, and it is boot-only (BrickEngine.boot).
 */
import type { SceneDescription } from "../../core/model";
import { sceneDamBreakBox } from "../../core/initial-fluid";
import { fluidLatticePagedCellVolumes, fluidLatticePagedVertexPhi, initialLiquidCellBounds, latticeBoxPageKeys } from "../../core/initial-liquid-lattice";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { sceneHasTerrain } from "../../core/terrain";
import { proceduralSolidField } from "../../core/procedural-solid";
import { phiFar } from "./brick-abi";
import { type BrickEngineOptions, type BrickInitialPhi, type BrickLiquidBox, type Vec3, bootBricks, bootCornerPages, bootPhiAt, initialPhiPages } from "./brick-engine";

/** Whether the scene's t = 0 liquid is the closed form: a plain dam-break box or tank-fill slab (fillFraction > 0),
 * nothing else, and no closed cell (voxels, terrain, a procedural solid) to bury a vertex. */
export function closedFormLiquid(scene: SceneDescription): boolean {
  const f = scene.fluid, c = scene.container;
  return scene.systems?.fluid !== false && !f.initialHeightField && !f.initialBrickSeeds_m?.length && !f.initialLiquidVolumes?.length
    && c.boundary !== "free-space" && c.walls !== "none" && c.fillFraction > 0
    && solidVoxelEditsForScene(scene).length === 0 && !sceneHasTerrain(scene) && !proceduralSolidField(scene);
}

/** The closed form's box in cells (closedFormLiquid scenes): the dam's box, or the tank's slab, run on through the walls. */
function closedFormBox(scene: SceneDescription, lattice: Vec3): BrickLiquidBox {
  const n = lattice, far = 2 * Math.max(...n), tol = 1e-9 * Math.max(...n);
  const dam = scene.fluid.initialCondition === "dam-break" ? sceneDamBreakBox(scene)
    : { min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: scene.container.fillFraction, z: 1 } };
  const lo = [dam.min.x, dam.min.y, dam.min.z].map((v, a) => v * n[a]!), hi = [dam.max.x, dam.max.y, dam.max.z].map((v, a) => v * n[a]!);
  // An open top is no wall: a slab or box reaching it keeps its top face (damBreakSignedDistanceAtNode).
  const wall = (a: number) => a !== 1 || scene.container.top === "closed";
  return {
    lo: lo.map(v => v <= tol ? -far : v) as unknown as Vec3,
    hi: hi.map((v, a) => v >= n[a]! - tol && wall(a) ? n[a]! + far : v) as unknown as Vec3,
  };
}

/** The old engine's t = 0 φ (fluidLatticePagedVertexPhi, metres) on the brick engine's pages, in h. */
export function brickInitialPhi(scene: SceneDescription, lattice: Vec3, h_m: number, farPhi = phiFar(lattice)): BrickInitialPhi {
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  if (cells.some((v, a) => v !== lattice[a])) throw new Error(`Brick engine: the initial φ table needs the container's lattice (${cells.join("x")} cells, the engine's ${lattice.join("x")})`);
  const P = initialPhiPages(lattice), far = farPhi;
  // The liquid's pages, grown by one page an axis.
  const bounds = initialLiquidCellBounds(scene, cells) ?? [[[0, 0, 0], [...cells]]];
  const liquid = latticeBoxPageKeys(bounds, 32, P), keys = new Set<number>();
  for (const k of liquid) {
    const p = [k % P[0], Math.floor(k / P[0]) % P[1], Math.floor(k / (P[0] * P[1]))];
    for (let j = 0; j < 27; j++) {
      const q = [p[0]! + (j % 3) - 1, p[1]! + (Math.floor(j / 3) % 3) - 1, p[2]! + Math.floor(j / 9) - 1];
      if (q.every((v, a) => v >= 0 && v < P[a]!)) keys.add(q[0]! + P[0] * (q[1]! + P[1] * q[2]!));
    }
  }
  const pages = [...keys].sort((a, b) => a - b);
  // The old function's pages are ceil(n / 32) an axis with 33^3 vertices (the high plane included): the brick
  // engine's high-plane page (index n / 32) reads its last page's plane 32.
  const Q = cells.map(n => Math.ceil(n / 32)), qOf = (v: number, a: number) => Math.min(v >> 5, Q[a]! - 1);
  const stored = new Set<number>();
  for (const k of pages) {
    const p = [k % P[0], Math.floor(k / P[0]) % P[1], Math.floor(k / (P[0] * P[1]))];
    stored.add(Math.min(p[0]!, Q[0]! - 1) + Q[0]! * (Math.min(p[1]!, Q[1]! - 1) + Q[1]! * Math.min(p[2]!, Q[2]! - 1)));
  }
  const old = fluidLatticePagedVertexPhi(scene, 32, (x, y, z) => stored.has(x + Q[0]! * (y + Q[1]! * z)));
  const phi = new Float32Array(32768 * pages.length), inv = 1 / h_m;
  pages.forEach((k, i) => {
    const p = [k % P[0], Math.floor(k / P[0]) % P[1], Math.floor(k / (P[0] * P[1]))];
    for (let l = 0; l < 32768; l++) {
      const v = [32 * p[0]! + (l & 31), 32 * p[1]! + ((l >> 5) & 31), 32 * p[2]! + (l >> 10)];
      if (v.some((x, a) => x > lattice[a]!)) { phi[32768 * i + l] = far; continue; }
      const q = v.map(qOf), page = old.at.get(q[0]! + Q[0]! * (q[1]! + Q[1]! * q[2]!));
      if (!page) throw new Error(`Brick engine: the initial φ page ${q.join(",")} was not sampled`);
      const w = v.map((x, a) => x - 32 * q[a]!);
      phi[32768 * i + l] = Math.fround(page[w[0]! + 33 * (w[1]! + 33 * w[2]!)]! * inv);
    }
  });
  return { pages, phi };
}

export interface BrickInitialLiquid {
  /** The closed form (empty with `initial`). */
  readonly liquid: readonly BrickLiquidBox[];
  readonly initial?: BrickInitialPhi;
  /** The boot's F bricks (bootBricks: the reach set of the liquid's vertices, at rest), ascending keys. */
  readonly bricks: readonly number[];
  /** Tiered, the boot's page-only candidates besides its bricks' pages (bootCornerPages). */
  readonly pages: readonly number[];
}

/** The scene's initial liquid for the engine, and the boot's bricks and pages. */
export function brickInitialLiquid(scene: SceneDescription, o: Pick<BrickEngineOptions, "lattice" | "farPhi" | "h_m" | "dt_s" | "gravity_mps2">): BrickInitialLiquid {
  const form = closedFormLiquid(scene) ? { liquid: [closedFormBox(scene, o.lattice)] } : { liquid: [], initial: brickInitialPhi(scene, o.lattice, o.h_m, o.farPhi) };
  const bricks = bootBricks({ ...o, ...form });
  return { ...form, bricks, pages: bootCornerPages(o.lattice, bricks) };
}

/**
 * The records the boot's initial seeding writes at most (pSeed cfg.w = 1, particles-formulas nbfSeedCellDepthOk,
 * nbSeedTarget): every cell of a boot brick whose centre depth d (the mean of its corners' φ) passes the gate,
 * !(d > 0.9 || d < -band - 0.5), at 8 sites, 4 graded below d = -1.5. An upper bound: the site tests (depth, an h
 * owner) only remove sites. For lane E's pool rule; its run-time margin is E's.
 */
export function bootParticleCount(o: Pick<BrickEngineOptions, "lattice" | "farPhi" | "liquid" | "initial">, bricks: readonly number[], bandDepth: number, graded = true): number {
  const at = bootPhiAt(o);
  let n = 0;
  for (const k of bricks) {
    const b = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023];
    for (let l = 0; l < 512; l++) {
      const c = [8 * b[0]! + (l & 7), 8 * b[1]! + ((l >> 3) & 7), 8 * b[2]! + (l >> 6)];
      if (c.some((x, a) => x >= o.lattice[a]!)) continue;
      let d = 0;
      for (let j = 0; j < 8; j++) d += at([c[0]! + (j & 1), c[1]! + ((j >> 1) & 1), c[2]! + (j >> 2)]);
      d /= 8;
      if (d > 0.9 || d < -bandDepth - 0.5) continue;
      n += graded && d <= -1.5 ? 4 : 8;
    }
  }
  return n;
}

/**
 * The liquid's initial velocity (scene.fluid.initialVelocity_m_s), as the old engine sets it
 * (uniform-volume-initial.ts uniformInitialLiquidVelocityPages): the velocity on every face of each 4h tile that
 * holds liquid (fluidLatticePagedCellVolumes' volume > 0) and of the tiles around it (3^3), zero beyond; the
 * extension carries it on into the air. The tiles as keys t.x + T.x (t.y + T.y t.z) (T = lattice / 4), ascending.
 * Undefined: the liquid starts at rest.
 */
export function brickInitialVelocity(scene: SceneDescription, lattice: Vec3): BrickEngineOptions["initialVelocity"] {
  const v = scene.fluid.initialVelocity_m_s;
  if (!v || (v.x === 0 && v.y === 0 && v.z === 0)) return undefined;
  if (![v.x, v.y, v.z].every(Number.isFinite)) throw new Error(`Initial liquid velocity must be finite: ${JSON.stringify(v)}`);
  // The paged volumes cover the old engine's fluid box (sceneFluidLatticeDimensions: the container's cells padded to
  // whole pages), from the container's cell 0; its liquid lies in the container's cells, the brick engine's lattice.
  const cells = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  if (cells.some((n, a) => n !== lattice[a])) throw new Error(`Brick engine: the initial velocity needs the container's lattice (${cells.join("x")} cells, the engine's ${lattice.join("x")})`);
  const { volume } = fluidLatticePagedCellVolumes(scene, 32), [px, py] = volume.pages, T = lattice.map(n => n >> 2);
  const wet = new Set<number>();
  for (const [key, page] of volume.at) {
    const o = [32 * (key % px), 32 * (Math.floor(key / px) % py), 32 * Math.floor(key / (px * py))];
    for (let i = 0; i < 32768; i++) if (page[i]! > 0) {
      const t = [(o[0]! + (i & 31)) >> 2, (o[1]! + ((i >> 5) & 31)) >> 2, (o[2]! + (i >> 10)) >> 2];
      if (t.some((x, a) => x >= T[a]!)) throw new Error(`Brick engine: initial liquid in the padding past the container's ${lattice.join("x")} cells (tile ${t.join(",")})`);
      wet.add(t[0]! + T[0]! * (t[1]! + T[1]! * t[2]!));
    }
  }
  const near = new Set<number>();
  for (const k of wet) {
    const t = [k % T[0]!, Math.floor(k / T[0]!) % T[1]!, Math.floor(k / (T[0]! * T[1]!))];
    for (let j = 0; j < 27; j++) {
      const q = [t[0]! + (j % 3) - 1, t[1]! + (Math.floor(j / 3) % 3) - 1, t[2]! + Math.floor(j / 9) - 1];
      if (q.every((x, a) => x >= 0 && x < T[a]!)) near.add(q[0]! + T[0]! * (q[1]! + T[1]! * q[2]!));
    }
  }
  return { velocity: [v.x, v.y, v.z], tiles: [...near].sort((a, b) => a - b) };
}
