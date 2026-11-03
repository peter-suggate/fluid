import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import { parseScene, serializeScene } from "../lib/core/model";
import { sceneWithSolidStroke, solidWorldForScene, sampleSolidWorld } from "../lib/core/solid-world";
import { visibleToolSurface } from "../lib/core/voxel-editor/surface";
import { penTool } from "../lib/core/voxel-editor/plugins/pen";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { toolSurface } from "../lib/core/voxel-editor/geometry";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { svoSculptChangeBounds } from "../lib/svo/features/scene-publication/svo-sculpt-change-bounds";
import { buildSvoPlanarBoundaryCatalog, svoRenderResidual } from "../lib/svo/features/scene-publication/svo-planar-boundary";

const ray = { origin: { x: 0, y: .4, z: 2 }, direction: { x: 0, y: 0, z: -1 } };
test("tank wall carves at the visible face, opens collision, and retains all authored detail on save and undo", () => {
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
  const hit = visibleToolSurface(scene, ray)!;
  assert.ok(Math.abs(hit.point.z - 1.3) < 1e-6);
  const values = toolValues(penTool, { operation: 1, size: 16 });
  const voxelHit = toolSurface({ scene, values }, ray)!;
  assert.equal(voxelHit.faceAxis, 2); assert.equal(voxelHit.faceSign, 1);
  const gesture = penTool.begin({ scene, values, ray })!;
  const patches = gesture.update(ray)!.patches;
  assert.equal(gesture.update(ray)!.patches, patches, "holding the brush cannot tunnel through to the wall behind");
  const changed = sceneWithSolidStroke(scene, patches);
  assert.equal(changed.scenery, scene.scenery, "the edit never substitutes or resamples the authored model");
  const after = visibleToolSurface(changed, ray)!;
  assert.ok(after.point.z < -1, "a through hole reveals the far wall");
  assert.notEqual(after.scenery, hit.scenery);
  const nearby = { ...ray, origin: { ...ray.origin, x: .5 } };
  assert.deepEqual(visibleToolSurface(changed, nearby), visibleToolSurface(scene, nearby), "untouched first-hit position and normal stay exact");
  const cell = [256, 16, 96] as const;
  const beforeWorld = solidWorldForScene(scene), afterWorld = solidWorldForScene(changed);
  assert.ok(sampleSolidWorld(beforeWorld, cell).solidFraction > 0);
  assert.equal(sampleSolidWorld(afterWorld, cell).solidFraction, 0, "the physics wall has the same opening");
  const residual = svoRenderResidual(changed, afterWorld);
  assert.equal(residual.excluded.size, svoRenderResidual(scene, beforeWorld).excluded.size, "sculpting never reveals a hidden collision copy");
  const primitives = buildEnvironmentProxyCatalog(scene, scene.environment ?? "stage").primitives;
  const changes = svoSculptChangeBounds(scene, scene.solidVoxels, changed.solidVoxels, primitives);
  assert.equal(changes.relistPlanar, false);
  assert.deepEqual(changes.addedPrimitiveBounds, [], "a tiny cut cannot voxelize the entire wall");
  assert.ok(buildSvoPlanarBoundaryCatalog(primitives, p => ({ materialId: p.ownerIndex + 32, ownerId: p.ownerIndex })).sources.length > 0);
  const restored = parseScene(serializeScene(changed));
  assert.deepEqual(restored.scenery, parseScene(serializeScene(scene)).scenery);
  assert.ok(visibleToolSurface(restored, ray)!.point.z < -1);
  assert.deepEqual(visibleToolSurface(scene, ray), hit, "undo is the original unmodified surface");
});

test("all surface tools hit foreground scenery before voxels, regardless of the old walls switch", async () => {
  const { createEmptyScene } = await import("../lib/core/empty-scene");
  const { penTool } = await import("../lib/core/voxel-editor/plugins/pen");
  const { sceneCellSizes_m } = await import("../lib/core/scene-lattice");
  const scene = createEmptyScene();
  scene.scenery = { ...scene.scenery!, nodes: [{ ...scene.scenery!.nodes.find(node => node.kind === "room-shell")!, faces: [] }, {
    id: "foreground", kind: "box", place: { units: "metres", position: { x: 0, y: .5, z: 0 } },
    halfSize: { x: .4, y: .05, z: .4 }, material: { colorLinear: [.8, .2, .1], surface: "ceramic" },
  }] };
  scene.solidVoxels = [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [64, 2, 64] }];
  const input = { origin: { x: 0, y: 1, z: 0 }, direction: { x: 0, y: -5, z: 0 } };
  for (const walls of [0, 1]) {
    const hit = visibleToolSurface(scene, input)!;
    assert.equal(hit.scenery, "foreground"); assert.ok(Math.abs(hit.point.y - .55) < 1e-6);
    const values = { ...toolValues(penTool, { operation: 1, size: 1 }), shell: walls };
    const surface = toolSurface({ scene, values }, input)!;
    assert.ok(surface.coordinate[1] * sceneCellSizes_m(scene)[1] > .4);
    const patches = penTool.begin({ scene, values, ray: input })!.update(input)!.patches;
    assert.ok(patches.every(p => p.minimum[1] * sceneCellSizes_m(scene)[1] > .4), "no edit is placed on the hidden background");
    assert.ok(patches.every(p => p.operation === "clear" && p.affectsScenery));
  }
  assert.equal(visibleToolSurface(scene, { ...input, direction: { x: 0, y: 1, z: 0 } }), undefined, "a miss cannot invent a target behind the view");
});

test("unknown authoring geometry is rejected instead of silently becoming uneditable", async () => {
  const { createEmptyScene } = await import("../lib/core/empty-scene");
  const { validateSceneryGraph } = await import("../lib/core/scenery-graph");
  const scene = createEmptyScene();
  const graph = { ...scene.scenery!, nodes: [...scene.scenery!.nodes, { id: "render-only", kind: "triangle-decoration" }] };
  assert.ok(validateSceneryGraph(graph as never).some(e => e.includes("unsupported editable geometry")));
  assert.throws(() => buildEnvironmentProxyCatalog({ ...scene, scenery: graph as never }, scene.environment ?? "stage"), /unsupported editable geometry/);
});

test("visible voxels occlude scenery and a render-only cutaway never intercepts a tool ray", async () => {
  const { createEmptyScene } = await import("../lib/core/empty-scene");
  const scene = createEmptyScene({ extents_m: { x: 1, y: 1, z: 1 }, finestCellSize_m: .125 });
  scene.container.walls = "none";
  scene.scenery = { ...scene.scenery!, nodes: [{ ...scene.scenery!.nodes.find(node => node.kind === "room-shell")!, faces: [] }] };
  scene.solidVoxels = [
    { operation: "fill", minimum: [1, 0, 1], maximumExclusive: [7, 1, 7] },
    { operation: "fill", minimum: [1, 1, 1], maximumExclusive: [2, 5, 7] },
    { operation: "fill", minimum: [6, 1, 1], maximumExclusive: [7, 5, 7] },
    { operation: "fill", minimum: [1, 1, 1], maximumExclusive: [7, 5, 2] },
    { operation: "fill", minimum: [1, 1, 6], maximumExclusive: [7, 5, 7] },
  ];
  const input = { origin: { x: 0, y: .3, z: 2 }, direction: { x: 0, y: 0, z: -1 } };
  assert.equal(visibleToolSurface(scene, input)!.point.z, .375);
  const cutaway = { ...scene, cutaway: { eye_m: input.origin } };
  assert.ok(visibleToolSurface(cutaway, input)!.point.z < 0, "only the far visible wall can receive the stroke");
  const withScenery = { ...scene, scenery: { ...scene.scenery, nodes: [...scene.scenery.nodes, {
    id: "behind", kind: "box" as const, place: { units: "metres" as const, position: { x: 0, y: .3, z: 0 } },
    halfSize: { x: .1, y: .1, z: .1 }, material: { colorLinear: [1, 1, 1] as [number, number, number] },
  }] } };
  const hit = visibleToolSurface(withScenery, input)!;
  assert.equal(hit.scenery, undefined);
  assert.equal(hit.point.z, .375, "a scenery object behind a voxel wall cannot steal its hit");
});
