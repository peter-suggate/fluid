import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import { parseScene, serializeScene, type SceneDescription } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { createPaneSession } from "../lib/core/session/session";
import { validateSolidPatch } from "../lib/core/solid-patch/compose";
import { subcellBounds, subcellPatch } from "../lib/core/solid-patch/ellipsoid-bounds";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForScene, type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { mirrorControls } from "../lib/core/voxel-editor/controls";
import { mirroredPatches } from "../lib/core/voxel-editor/geometry";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { shapeKindControl, shapeLengthControl, shapeTool } from "../lib/core/voxel-editor/plugins/shape";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { sessionToolHost } from "../lib/core/voxel-editor/session-host";
import { visibleToolSurface } from "../lib/core/voxel-editor/surface";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { packSvoSculptCuts, sampleSvoSculptCuts } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { svoSculptChangeBounds } from "../lib/svo/features/scene-publication/svo-sculpt-change-bounds";
import { svoRenderResidual } from "../lib/svo/features/scene-publication/svo-planar-boundary";

const SPHERE = 0, BOX = 1, CYLINDER = 2;

/** A slab of voxels, top face at y = 4 cells, and a ray straight down onto it at fractional cells (x, z). */
function slab() {
  const scene = createEmptyScene();
  scene.scenery = { ...scene.scenery!, nodes: [{ ...scene.scenery!.nodes.find(node => node.kind === "room-shell")!, faces: [] }] };
  scene.solidVoxels = [{ operation: "fill", minimum: [2, 1, 2], maximumExclusive: [30, 4, 30], materialId: 2 }];
  const h = sceneCellSizes_m(scene);
  const ray = (x: number, z = 8.625) => ({ origin: { x: -scene.container.width_m / 2 + x * h[0], y: 2,
    z: -scene.container.depth_m / 2 + z * h[2] }, direction: { x: 0, y: -1, z: 0 } });
  return { scene, ray, h };
}

/** Two thin scenery walls across z, as a tank's: the near one's face at z = 1.3 m. */
function tank() {
  const scene = createEmptyScene({ extents_m: { x: 12.8, y: .8, z: 2.4 }, finestCellSize_m: .025 });
  scene.scenery = { ...scene.scenery!, nodes: [
    { ...scene.scenery!.nodes.find(node => node.kind === "room-shell")!, faces: [] },
    ...[-1, 1].map(sign => ({
      id: `tank-wall/${sign}`, kind: "box" as const,
      place: { units: "metres" as const, position: { x: 0, y: .4, z: sign * 1.25 } },
      halfSize: { x: 6.5, y: .4, z: .05 },
      material: { colorLinear: [.8, .75, .6] as [number, number, number], surface: "ceramic" as const },
    })),
  ] };
  const ray = (x = 0) => ({ origin: { x, y: .4, z: 2 }, direction: { x: 0, y: 0, z: -1 } });
  return { scene, ray };
}

const centreOf = (patch: SolidWorldVoxelPatch) => {
  const b = subcellBounds(patch);
  return b.minimum.map((v, a) => (v + b.maximumExclusive[a]!) / 2);
};
const sizeOf = (patch: SolidWorldVoxelPatch) => {
  const b = subcellBounds(patch);
  return b.maximumExclusive.map((v, a) => v - b.minimum[a]!);
};

test("Shape replaces Center and Box on the rail, arms on U, and keeps its controls tiered", () => {
  assert.equal(voxelTools.get("shape"), shapeTool);
  assert.equal(voxelTools.get("center"), undefined);
  assert.equal(voxelTools.get("box"), undefined);
  assert.equal(shapeTool.execution, "live");
  assert.equal(shapeTool.ui.key, "u");
  assert.equal(shapeLengthControl.tier, "advanced");
  assert.equal(shapeLengthControl.when!(toolValues(shapeTool, { shape: SPHERE })), false);
  assert.equal(shapeLengthControl.when!(toolValues(shapeTool, { shape: CYLINDER })), true);
});

test("every shape is centred on the exact pressed point, to an eighth of a cell, with no axis snapping", () => {
  const { scene, ray } = slab(), start = ray(10.625, 8.375);
  for (const [shape, kind] of [[SPHERE, "ellipsoid"], [BOX, "cuboid"], [CYLINDER, "cylinder"]] as const) {
    const gesture = shapeTool.begin({ scene, ray: start, values: toolValues(shapeTool, { shape, operation: 0 }) })!;
    const click = gesture.update(start)!;
    assert.equal(click.patches.length, 1);
    click.patches.forEach(validateSolidPatch);
    assert.equal(click.patches[0]!.kind, kind);
    assert.deepEqual(centreOf(click.patches[0]!), [10.625, 4, 8.375], `${kind} click stands on the press`);
    assert.deepEqual(sizeOf(click.patches[0]!), [1, 1, 1], `${kind}: a click is one voxel across`);
    const grown = gesture.update(ray(13, 8.375))!;
    assert.equal(grown.replaces, true, "a new radius replaces the stroke rather than appending to it");
    grown.patches.forEach(validateSolidPatch);
    assert.deepEqual(centreOf(grown.patches[0]!), [10.625, 4, 8.375], `${kind} grows about the press`);
    assert.deepEqual(sizeOf(grown.patches[0]!), [4.75, 4.75, 4.75], `${kind} radius is the drag`);
    assert.equal(gesture.update(ray(13.01, 8.375)), grown, "motion below the radius step reuses the proposal");
    assert.equal(grown.patches[0]!.materialId !== undefined, true);
    if (kind === "cylinder") assert.equal(grown.patches[0]!.axis, 1, "a cylinder stands along the surface normal");
  }
  const long = shapeTool.begin({ scene, ray: start, values: toolValues(shapeTool, { shape: CYLINDER, length: 3 }) })!
    .update(ray(12, 8.375))!.patches[0]!;
  assert.deepEqual(sizeOf(long), [2.75, 3, 2.75]);
});

test("an attached shape is a surface the shared pick finds, and leaves every other surface exactly as it was", () => {
  const { scene, ray, h } = slab();
  for (const shape of [SPHERE, BOX, CYLINDER]) {
    const update = shapeTool.begin({ scene, ray: ray(10.625), values: toolValues(shapeTool, { shape }) })!.update(ray(12))!;
    const changed = sceneWithSolidStroke(scene, update.patches);
    const top = visibleToolSurface(changed, ray(10.625))!;
    // The pick is the voxel residual's, a whole cell at a time: the shape's top layer (it reaches y = 5.375).
    assert.ok(Math.abs(top.point.y - 6 * h[1]) < 1e-6, `shape ${shape} stands up out of the slab`);
    assert.deepEqual(visibleToolSurface(changed, ray(20)), visibleToolSurface(scene, ray(20)), "a ray beside it is untouched");
    assert.deepEqual(parseScene(serializeScene(changed)).solidVoxels, changed.solidVoxels);
    const after = solidWorldForScene(changed), before = solidWorldForScene(scene);
    for (const cell of [[20, 3, 20], [5, 2, 5], [10, 1, 15]] as const) {
      assert.deepEqual(sampleSolidWorld(after, cell), sampleSolidWorld(before, cell), "cells outside the shape keep their detail");
    }
  }
});

test("the cuboid and the subcell cylinder rasterize at their fractional faces", () => {
  const box = subcellPatch([10.25, 2.75, 8.25], [12.75, 5.25, 10.75], "fill", "cuboid");
  const world = solidWorldForScene({ ...createEmptyScene(), solidVoxels: [{ ...box, materialId: 2 }] });
  // A face a quarter-cell into a layer covers it by three quarters; the cell inside is whole.
  assert.ok(Math.abs(sampleSolidWorld(world, [10, 4, 9]).solidFraction - .75) < .01);
  assert.equal(sampleSolidWorld(world, [11, 4, 9]).solidFraction, 1);
  assert.ok(Math.abs(sampleSolidWorld(world, [11, 5, 9]).solidFraction - .25) < .01);
  assert.equal(sampleSolidWorld(world, [13, 4, 9]).solidFraction, 0);
  const post = { ...subcellPatch([10, 2, 10], [14, 4.5, 14], "fill", "cylinder"), axis: 1 as const, materialId: 2 };
  const posts = solidWorldForScene({ ...createEmptyScene(), solidVoxels: [post] });
  assert.equal(sampleSolidWorld(posts, [12, 3, 12]).solidFraction, 1);
  assert.ok(Math.abs(sampleSolidWorld(posts, [12, 4, 12]).solidFraction - .5) < .01, "the cap stands half a cell into its layer");
});

test("Shape erase cuts scenery and its collision together, keeps the authored model, and survives save, load and undo", async () => {
  const { scene, ray } = tank();
  const hit = visibleToolSurface(scene, ray())!;
  assert.equal(hit.scenery, "tank-wall/1");
  const values = toolValues(shapeTool, { shape: SPHERE, operation: 1 });
  const gesture = shapeTool.begin({ scene, values, ray: ray() })!;
  const update = gesture.update(ray(.15))!;
  assert.ok(update.patches.every(p => p.operation === "clear" && p.affectsScenery === true));
  const changed = sceneWithSolidStroke(scene, update.patches);
  assert.equal(changed.scenery, scene.scenery, "the edit never substitutes or resamples the authored model");
  // Hit testing: the hole is real to the shared pick.
  const through = visibleToolSurface(changed, ray())!;
  assert.ok(through.point.z < 1.25, "the press point now looks into the cut");
  // Untouched fidelity: a ray beside the cut meets exactly what it met before.
  assert.deepEqual(visibleToolSurface(changed, ray(.5)), visibleToolSurface(scene, ray(.5)));
  // Collision: the physics wall has the same opening, and no hidden copy surfaces.
  const beforeWorld = solidWorldForScene(scene), afterWorld = solidWorldForScene(changed);
  assert.ok(sampleSolidWorld(beforeWorld, [256, 16, 96]).solidFraction > 0);
  assert.equal(sampleSolidWorld(afterWorld, [256, 16, 96]).solidFraction, 0);
  assert.equal(svoRenderResidual(changed, afterWorld).excluded.size, svoRenderResidual(scene, beforeWorld).excluded.size);
  const primitives = buildEnvironmentProxyCatalog(scene, scene.environment ?? "stage").primitives;
  assert.deepEqual(svoSculptChangeBounds(scene, scene.solidVoxels, changed.solidVoxels, primitives).addedPrimitiveBounds, [],
    "a local cut cannot voxelize the wall");
  // Save and load keep the subcell shape exactly.
  const restored = parseScene(serializeScene(changed));
  assert.deepEqual(restored.solidVoxels, changed.solidVoxels);
  assert.deepEqual(restored.scenery, parseScene(serializeScene(scene)).scenery);
  assert.deepEqual(visibleToolSurface(restored, ray()), through);
  // Undo, through the live transaction a host runs: one entry, standing on the original document.
  const session = createPaneSession("a");
  session.scene.getState().setScene(scene);
  const original = session.scene.getState().scene;
  let runtime: SceneDescription = original;
  const host = sessionToolHost(session, async (next, before) => { assert.equal(before, runtime); runtime = next; });
  const transaction = beginToolTransaction(shapeTool, host, ray(), values)!;
  await transaction.update(ray(.05));
  await transaction.update(ray(.15));
  assert.deepEqual(session.scene.getState().scene.solidVoxels, changed.solidVoxels, "the live edit is the larger radius alone");
  await transaction.finish();
  assert.equal(session.history.getState().past.length, 1);
  assert.deepEqual(session.history.getState().past[0]!.scene.solidVoxels, original.solidVoxels, "the one entry is the document before the stroke");
  assert.deepEqual(visibleToolSurface(session.history.getState().past[0]!.scene, ray()), hit, "undo is the original surface");
});

test("a cuboid cut reaches scenery as the box record at its subcell bounds; unknown kinds fail", () => {
  const lattice = { origin_m: [0, 0, 0] as const, cellSize_m: [.5, .5, .5] as const };
  const cut = { ...subcellPatch([1.25, 1.25, 1.25], [2.75, 2.75, 2.75], "clear", "cuboid"), affectsScenery: true };
  const words = packSvoSculptCuts([cut], lattice);
  assert.equal(words[0], 1);
  assert.ok(Math.abs(sampleSvoSculptCuts(words, [1, 1, 1]).distance + .375) < 1e-6, "the centre is 3/8 m inside");
  assert.ok(Math.abs(sampleSvoSculptCuts(words, [.625, 1, 1]).distance) < 1e-6, "the face stands at its subcell bound");
  assert.throws(() => validateSolidPatch({ operation: "fill", kind: "frustum", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1] }),
    /Unknown solid patch kind "frustum"/);
  assert.throws(() => validateSolidPatch({ operation: "fill", kind: "cuboid", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1],
    payload: "[0.5,0,0,0.5,0,0]" }), /Invalid cuboid subcell bounds/);
});

test("live replacement rewinds and restands in one exchange, and cancelling rolls it all back", async () => {
  const { scene, ray } = slab();
  const session = createPaneSession("b");
  session.scene.getState().setScene(scene);
  const publications: SceneDescription[] = [];
  const host = sessionToolHost(session, async (next) => { publications.push(next); });
  const transaction = beginToolTransaction(shapeTool, host, ray(10.625), toolValues(shapeTool, { shape: BOX, operation: 0 }))!;
  await transaction.update(ray(10.625));
  const small = session.scene.getState().scene;
  assert.equal(small.solidVoxels.length, scene.solidVoxels.length + 1);
  await transaction.update(ray(12));
  assert.deepEqual(publications.slice(1).map(next => next.solidVoxels.length),
    [scene.solidVoxels.length, scene.solidVoxels.length + 1], "back to the base, then the new size over it");
  assert.equal(publications[1], scene, "the rewind is to the very base document");
  await transaction.finish(true);
  assert.equal(session.scene.getState().scene, scene);
  assert.equal(session.history.getState().past.length, 0, "a cancelled shape records nothing");
});

test("mirroring a subcell cuboid or cylinder swaps its insets with its sides", () => {
  const { scene } = slab();
  const values = { [mirrorControls[0].id]: 1 };
  for (const kind of ["cuboid", "cylinder"] as const) {
    const patch = subcellPatch([14.125, 2, 6], [17.125, 5, 9], "fill", kind);
    const [, image] = mirroredPatches({ scene, values }, [patch]);
    assert.deepEqual(image!.minimum, patch.minimum, kind);
    assert.equal(subcellBounds(image!).minimum[0], 14.875, kind);
  }
});

test("Paint on scenery is refused at the press, and Shape lists its own kind control", () => {
  const { scene, ray } = tank();
  assert.throws(() => shapeTool.begin({ scene, ray: ray(), values: toolValues(shapeTool, { operation: 2 }) }),
    /scenery keeps its own materials/);
  assert.ok(shapeTool.ui.controls.includes(shapeKindControl.control));
});

test("Shape centres on the scene focus: unlocked it is the pick itself, and a Shift+Y lock holds the press height in open air", async () => {
  const { FocusSession } = await import("../lib/core/editor-focus");
  const { scene, ray, h } = slab();
  const values = toolValues(shapeTool, { shape: SPHERE });
  const bare = shapeTool.begin({ scene, ray: ray(10.625), values })!.update(ray(12));
  const hosted = shapeTool.begin({ scene, ray: ray(10.625), values,
    focus: new FocusSession({ constraint: () => undefined }) })!.update(ray(12));
  assert.deepEqual(hosted, bare);
  const anchor = { x: ray(10.625).origin.x, y: 4 * h[1], z: ray(10.625).origin.z };
  assert.equal(shapeTool.begin({ scene, ray: ray(40), values }), undefined, "unlocked, open air has no centre");
  const air = shapeTool.begin({ scene, ray: ray(40), values,
    focus: new FocusSession({ constraint: () => ["x", "z"], fallback: anchor }) })!.update(ray(40))!;
  assert.deepEqual(centreOf(air.patches[0]!), [40, 4, 8.625], "the centre stands on the lock, at the anchor's height");
});
