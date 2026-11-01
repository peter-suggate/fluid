import assert from "node:assert/strict";
import type { SceneDescription, Vec3 } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { boxSolidVoxelShell } from "../lib/core/solid-world";

/** The scaling acceptance test's box (probe-only; no scene changes): the
 * scene's box `factor` times as wide on x, the extra width all past its +x
 * wall, with everything in it where it was. The x origin is the box's centre,
 * so world x moves by -(factor-1)/2 of the old width to keep each liquid box
 * and refinement region on the same cells; solid voxels are in cells from the
 * -x wall and stay as they are, but the box's own shell (boxSolidVoxelShell,
 * which a scene document adds) moves to the wider box's faces. The page pool
 * and everything else are kept.
 * The lattice must grow by whole pages (32 cells), so the pages the water
 * and walls meet are the same pages. Only scene parts this can carry are
 * admitted: no terrain, bodies, inflow, drops or non-box liquid, and no scenery
 * fluid collider but under the box's floor (the render room's stage). */
export function widenSceneBox(scene: SceneDescription, factor: number): SceneDescription {
  assert.ok(Number.isInteger(factor) && factor >= 1, "the box width factor must be a whole number >= 1");
  if (factor === 1) return scene;
  const f = scene.fluid;
  assert.ok(!scene.terrain && !scene.rigidBodies?.length && !scene.solidBoundary && !f.refinementKeyframes?.length,
    "widenSceneBox: no terrain, rigid bodies, solid boundary or refinement keyframes");
  // Scenery is the render room; a fluid collider in it must lie wholly under the box's floor.
  for (const node of scene.scenery?.nodes ?? []) {
    const n = node as { kind: string; tags?: readonly string[]; place?: { position: Vec3 }; halfSize?: Vec3 };
    if (!n.tags?.includes("fluid-collider")) continue;
    assert.ok(n.kind === "box" && n.place && n.halfSize && n.place.position.y + n.halfSize.y <= 0, "widenSceneBox: a scenery fluid collider only under the box's floor");
  }
  assert.ok(!f.inflow && !f.scheduledDrops?.length && !f.initialHeightField && !f.initialDamBreakDimensions_m && !f.initialBrickSeeds_m?.length && !(scene.container.fillFraction > 0),
    "widenSceneBox: the liquid must be initial liquid boxes alone");
  const [nx] = sceneLatticeDimensions(scene);
  assert.equal(nx % 32, 0, "widenSceneBox: the box's x cells must be whole pages");
  const width = scene.container.width_m, shift = -(factor - 1) * width / 2;
  // The shell's patches at the old size, by their cells: each replaced by its place on the wider box.
  const cells = sceneLatticeDimensions(scene), key = (v: { minimum: readonly number[]; maximumExclusive: readonly number[] }) => `${v.minimum.join(",")}:${v.maximumExclusive.join(",")}`;
  const top = scene.container.top, old = boxSolidVoxelShell(cells, { top }), wide = boxSolidVoxelShell([factor * nx, cells[1], cells[2]], { top });
  const at = new Map(old.map((v, i) => [key(v), i]));
  const shell = scene.solidVoxels.filter(v => v.operation === "fill" && at.has(key(v)));
  assert.ok(shell.length === 0 || shell.length === old.length, "widenSceneBox: the box's shell is whole or absent");
  const solidVoxels = scene.solidVoxels.map(v => v.operation === "fill" && at.has(key(v)) ? { ...wide[at.get(key(v))!]!, materialId: v.materialId } : v);
  const moved = (p: Vec3): Vec3 => ({ ...p, x: p.x + shift });
  const out: SceneDescription = { ...scene, solidVoxels, container: { ...scene.container, width_m: factor * width }, fluid: {
    ...f,
    initialLiquidVolumes: f.initialLiquidVolumes?.map(v => {
      assert.equal(v.shape, "box", "widenSceneBox: initial liquid boxes only");
      const b = v as Extract<typeof v, { shape: "box" }>;
      return { ...b, min_m: moved(b.min_m), max_m: moved(b.max_m) };
    }),
    refinementRegions: f.refinementRegions?.map(r => ({ ...r, min_m: moved(r.min_m), max_m: moved(r.max_m) })),
  } };
  assert.equal(sceneLatticeDimensions(out)[0], factor * nx, "widenSceneBox: the wider lattice must be factor times the cells");
  return out;
}
