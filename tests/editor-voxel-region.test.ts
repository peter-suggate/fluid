import assert from "node:assert/strict";
import test from "node:test";
import type { EditorEntityContext } from "../lib/core/editor-entity";
import type { EditorAction } from "../lib/core/editor-action";
import { targetActionsAt, targetAtRay } from "../lib/core/editor-probe-catalog";
import { projectSolidVoxelClearRegion, withSolidVoxelClearRegion } from "../lib/core/editor-solid-voxel";
import {
  voxelDragAnchor,
  voxelRegionContains,
  voxelRegionEntity,
  voxelRegionExtent,
  VOXEL_REGION_SELECTION,
} from "../lib/core/editor-voxel-region";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";

function context(
  scene: SceneDescription,
  voxelRegion?: EditorEntityContext["voxelRegion"],
): EditorEntityContext {
  return { scene, bodies: [], pickingAvailable: true, voxelRegion };
}

/** A slab of solids and a ray that meets its −X face, as a sweep would start. */
function slabScene() {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [{
    operation: "fill", minimum: [3, 4, 2], maximumExclusive: [7, 8, 6], materialId: 2,
  }];
  const [hx, hy, hz] = sceneCellSizes_m(scene);
  const originX = -0.5 * scene.container.width_m;
  const originZ = -0.5 * scene.container.depth_m;
  const rayAt = (j: number, k: number) => ({
    origin: { x: originX + 2 * hx, y: (j + 0.5) * hy, z: originZ + (k + 0.5) * hz },
    direction: { x: 1, y: 0, z: 0 },
  });
  return { scene, rayAt };
}

test("a press on a voxel gives a drag anchor locked to the face the ray met", () => {
  const { scene, rayAt } = slabScene();
  const target = targetAtRay(context(scene), rayAt(4, 2));
  assert.equal(target.kind, "solid-voxel");
  const anchor = voxelDragAnchor(scene, target);
  assert.deepEqual(anchor?.coordinate, [3, 4, 2]);
  // Face-locked: a sweep runs across the surface being looked at, never into
  // the solid behind it, so X is the axis the drag may not move.
  assert.equal(anchor?.faceAxis, 0);
  assert.equal(anchor?.faceSign, -1);
});

// The literal spec case: a sweep begun on a voxel keeps extending when the drag
// runs onto the wall, because the projection is onto the anchor's face plane and
// does not care what is standing on it.
test("a sweep grows across the face plane and covers the cells it passed", () => {
  const { scene, rayAt } = slabScene();
  const anchor = voxelDragAnchor(scene, targetAtRay(context(scene), rayAt(4, 2)))!;
  const region = projectSolidVoxelClearRegion(scene, rayAt(7, 5), anchor);
  assert.ok(region, "the drag ray must land on the anchor's face plane");
  assert.deepEqual([...region.minimum], [3, 4, 2]);
  assert.deepEqual([...region.maximumExclusive], [4, 8, 6]);
  assert.deepEqual(voxelRegionExtent(region), [1, 4, 4]);
  assert.ok(voxelRegionContains(region, [3, 6, 4]));
  assert.ok(!voxelRegionContains(region, [4, 6, 4]));
});

// A bare wall is just the face of the cell layer that lines it, which is what
// lets one gesture serve both and what "extend the tank wall drag-select" meant.
test("a bare tank wall yields an anchor on the layer that lines it", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [];
  const c = scene.container;
  const target = targetAtRay(context(scene), {
    origin: { x: 0, y: 0.5 * c.height_m, z: -2 * c.depth_m },
    direction: { x: 0, y: 0, z: 1 },
  });
  assert.equal(target.kind, "tank-wall");
  const anchor = voxelDragAnchor(scene, target);
  assert.ok(anchor, "a wall must be able to start a sweep");
  assert.equal(anchor.faceAxis, 2);
  const [, , hz] = sceneCellSizes_m(scene);
  const last = Math.floor(c.depth_m / hz) - 1;
  assert.equal(anchor.coordinate[2], last, "the cell inside the +Z wall, not the one past it");
});

// The whole point of the change: the release makes a selection, and the verb
// that edits the document lives on it. This asserts the verb still produces the
// exact patch the armed CLEAR SOLIDS tool used to write on release.
test("the region's Clear verb writes the same patch the armed tool wrote", () => {
  const { scene } = slabScene();
  const region = { minimum: [3, 4, 2], maximumExclusive: [4, 8, 6] } as const;
  const [action, ...rest] = voxelRegionEntity.actions!(context(scene, region), {
    point_m: { x: 0, y: 0, z: 0 }, normal: { x: 0, y: 0, z: 0 },
    selection: VOXEL_REGION_SELECTION,
  });
  assert.equal(rest.length, 0, "one verb, so the ring is not padded with variants");
  assert.ok(action);
  assert.equal(action.id, "clear-solids");
  const effect = action.effect;
  assert.ok(effect?.kind === "scene");
  assert.equal(effect.reseed, true,
    "solids are in the solver's seed key, so the run must restart from a defined t=0");
  const written = effect.scene.solidVoxels;
  assert.deepEqual(written, withSolidVoxelClearRegion(scene.solidVoxels ?? [], region));
  const cleared = solidWorldForScene({ ...scene, solidVoxels: written });
  assert.equal(sampleSolidWorld(cleared, [3, 6, 4]).solidFraction, 0);
  assert.equal(sampleSolidWorld(cleared, [4, 6, 4]).solidFraction, 1);

  // The Delete key and the flyout button go through `remove`, which must be the
  // same edit — two routes to one verb, not two verbs that drift apart.
  const entity = voxelRegionEntity.find!(context(scene, region), VOXEL_REGION_SELECTION.id);
  assert.deepEqual(entity?.remove?.().solidVoxels, written);
});

test("Clear is withheld when the swept box holds no solids at all", () => {
  const { scene } = slabScene();
  const empty = { minimum: [40, 40, 40], maximumExclusive: [42, 42, 42] } as const;
  assert.deepEqual(voxelRegionEntity.actions!(context(scene, empty), {
    point_m: { x: 0, y: 0, z: 0 }, normal: { x: 0, y: 0, z: 0 },
    selection: VOXEL_REGION_SELECTION,
  }), []);
});

// How the region's ring is reached at all: a region has no surface of its own,
// so the voxels inside it carry it. Right-clicking one opens the region's ring
// rather than a bare cell's.
test("a voxel inside the standing region names it, and one outside names the tank", () => {
  const { scene, rayAt } = slabScene();
  const region = { minimum: [3, 4, 2], maximumExclusive: [4, 8, 6] } as const;
  const inside = targetAtRay(context(scene, region), rayAt(5, 3));
  assert.deepEqual(inside.selection, VOXEL_REGION_SELECTION);
  assert.ok(targetActionsAt(context(scene, region), inside).some((a) => a.id === "clear-solids"));

  // Outside every region a solid still belongs to something: the vessel whose
  // solid world it is a cell of. A click has to land somewhere, and only a drag
  // makes a region — so "no region here" must not mean "nothing here".
  const outside = targetAtRay(context(scene, { minimum: [90, 90, 90], maximumExclusive: [91, 91, 91] }),
    rayAt(5, 3));
  assert.deepEqual(outside.selection, { kind: "tank", id: "tank" });
  assert.ok(!targetActionsAt(context(scene), outside).some((a) => a.id === "clear-solids"));
});

// A pixel is traceable wherever it is drawn, and a solid was the one surface
// nothing in the document named — so nothing could offer the probe for it.
//
// Reached recursively, because *where* the wedge sits is the ring's business,
// not this test's: a vessel groups its instruments under `Inspect` and offers it
// there, while a target with no instrument group of its own gets it as the ring's
// last wedge. What is asserted here is only that it is reachable.
test("a voxel and a wall both offer the ray probe", () => {
  const offersRay = (actions: readonly EditorAction[]): boolean => actions.some(
    (action) => action.id === "trace-ray" || offersRay(action.children ?? []));

  const { scene, rayAt } = slabScene();
  const voxel = targetAtRay(context(scene), rayAt(4, 2));
  assert.ok(offersRay(targetActionsAt(context(scene), voxel)));

  const bare = cloneScene(defaultScene);
  bare.solidVoxels = [];
  const wall = targetAtRay(context(bare), {
    origin: { x: 0, y: 0.5 * bare.container.height_m, z: -2 * bare.container.depth_m },
    direction: { x: 0, y: 0, z: 1 },
  });
  assert.ok(offersRay(targetActionsAt(context(bare), wall)));
});
