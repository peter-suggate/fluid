import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { meshAssetHeader, registerMeshAssetDistanceField } from "../lib/core/imported-assets/asset-library";
import { createEmptyScene, minimalRoomSceneryGraph } from "../lib/core/empty-scene";
import type { EditorRay } from "../lib/core/editor-entity";
import { scenerySurfaceAt } from "../lib/core/editor-scenery";
import type { SceneDescription } from "../lib/core/model";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { SVO_SCULPT_CUT_MAXIMUM_RECORDS } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { SOLID_PATCH_ELLIPSOID_KIND } from "../lib/core/solid-patch/kind";
import { penDiameterControl, penTool } from "../lib/core/voxel-editor/plugins/pen";
import { adjustableControl, toolValues } from "../lib/core/voxel-editor/plugin";
import { voxelTools } from "../lib/core/voxel-editor/registry";

function scene(): SceneDescription {
  const result = createEmptyScene();
  result.scenery = { ...minimalRoomSceneryGraph, nodes: [minimalRoomSceneryGraph.nodes[0]!] };
  result.solidVoxels = [{ operation: "fill", minimum: [2, 1, 2], maximumExclusive: [30, 4, 30], materialId: 2 }];
  return result;
}
function ray(scene: SceneDescription, x: number, z = 8): EditorRay {
  const h = sceneCellSizes_m(scene);
  return { origin: { x: -scene.container.width_m / 2 + (x + .5) * h[0], y: 2,
    z: -scene.container.depth_m / 2 + (z + .5) * h[2] }, direction: { x: 0, y: -1, z: 0 } };
}
const values = (size = 1, mode = 0) => toolValues(penTool, { size, mode });

test("pen is a live tool with adjustable diameter and direct Draw/Carve controls", () => {
  assert.equal(voxelTools.get("pen"), penTool);
  assert.equal(penTool.execution, "live");
  assert.equal(adjustableControl(penTool, false), penDiameterControl);
  assert.equal(adjustableControl(penTool, true), penDiameterControl);
  assert.equal(values(500).size, 64);
  assert.equal(values(0).size, 1);
  const modes = penTool.ui.controls.find(c => c.id === "mode");
  assert.ok(modes?.kind === "choice");
  assert.deepEqual(modes.options.map(option => option.label), ["Draw", "Carve"]);
});

test("pen fills fast-motion gaps with surface samples and never creates a selection", () => {
  const base = scene(), start = ray(base, 4);
  const gesture = penTool.begin({ scene: base, ray: start, values: values() })!;
  assert.equal(gesture.surface, true);
  assert.equal(gesture.advance, undefined);
  const first = gesture.update(start)!;
  assert.equal(first.selection, undefined);
  assert.deepEqual(first.patches[0]!.minimum, [4, 4, 8]);
  const moved = gesture.update(ray(base, 12))!;
  assert.deepEqual(moved.patches.slice(0, first.patches.length), first.patches, "updates retain their accepted prefix");
  assert.deepEqual(moved.patches.map(p => p.minimum[0]), [4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.ok(moved.patches.every(p => p.minimum[1] === 4 && p.operation === "fill"));
  assert.equal(moved.selection, undefined);
  assert.equal(moved.highlight.kind, "box", "the nib alone is outlined, never the accumulated region");
  assert.deepEqual(gesture.update(ray(base, 12))!.patches, moved.patches, "stationary pointer doesn't duplicate stamps");
  assert.equal(base.solidVoxels.length, 1, "geometry plugin leaves publication to the live host");
});

test("each interpolated pen sample follows the actual stepped surface", () => {
  const base = scene();
  base.solidVoxels = [...base.solidVoxels, { operation: "fill", minimum: [16, 4, 2], maximumExclusive: [30, 8, 30], materialId: 2 }];
  const start = ray(base, 12), gesture = penTool.begin({ scene: base, ray: start, values: values() })!;
  gesture.update(start);
  const result = gesture.update(ray(base, 20))!;
  assert.ok(result.patches.some(p => p.minimum[0] < 16));
  assert.ok(result.patches.some(p => p.minimum[0] >= 16));
  for (const patch of result.patches) assert.equal(patch.minimum[1], patch.minimum[0] < 16 ? 4 : 8);
});

test("rounded Draw and Carve stamps have adjustable diameter and opposite modifier", () => {
  const base = scene(), start = ray(base, 10);
  for (const mode of [0, 1]) for (const invert of [false, true]) {
    const gesture = penTool.begin({ scene: base, ray: start, values: values(5, mode), invert })!;
    const result = gesture.update(start)!;
    const patch = result.patches[0]!;
    assert.equal(patch.kind, SOLID_PATCH_ELLIPSOID_KIND);
    assert.deepEqual(patch.maximumExclusive.map((v, a) => v - patch.minimum[a]!), [5, 5, 5]);
    assert.equal(patch.operation, (mode === 1) !== invert ? "clear" : "fill");
    assert.equal(result.selection, undefined);
    assert.equal(result.highlight.kind, "paths");
  }
  const carve = penTool.begin({ scene: base, ray: start, values: values(1, 1) })!.update(start)!;
  assert.deepEqual(carve.patches[0]!.minimum, [10, 3, 8], "single-cell carving lands inside the surface");
});

test("empty space never becomes a ground-plane stroke and a miss breaks interpolation", () => {
  const base = scene(), start = ray(base, 4);
  const empty = { ...base, solidVoxels: [] };
  assert.equal(penTool.begin({ scene: empty, ray: start, values: values() }), undefined);
  const gesture = penTool.begin({ scene: base, ray: start, values: values() })!;
  const first = gesture.update(start)!;
  const miss = gesture.update({ ...start, direction: { x: 0, y: 1, z: 0 } })!;
  assert.deepEqual(miss.patches, first.patches);
  assert.deepEqual(miss.highlight, { kind: "paths", paths: [] });
  assert.deepEqual(gesture.update(ray(base, 20))!.patches.map(p => p.minimum[0]), [4, 20]);
});

test("interpolation between disconnected surfaces does not stamp the empty gap", () => {
  const base = scene();
  base.solidVoxels = [
    { operation: "fill", minimum: [4, 1, 2], maximumExclusive: [8, 4, 30] },
    { operation: "fill", minimum: [20, 1, 2], maximumExclusive: [24, 4, 30] },
  ];
  const start = ray(base, 5), gesture = penTool.begin({ scene: base, ray: start, values: values() })!;
  gesture.update(start);
  const result = gesture.update(ray(base, 22))!;
  assert.ok(result.patches.every(p => p.minimum[0] < 8 || p.minimum[0] >= 20));
  assert.ok(result.patches.length > 2, "reachable parts of each surface still get intermediate samples");
});

test("Draw and Carve clamp to the nearest scenery surface", () => {
  const base = scene();
  base.scenery = { ...base.scenery!, nodes: [minimalRoomSceneryGraph.nodes[0]!, { kind: "box", id: "block", place: { units: "metres", position: { x: 0, y: .6, z: 0 } },
    halfSize: { x: .2, y: .1, z: .2 }, material: { colorLinear: [1, 1, 1] } }] };
  const start = { origin: { x: .025, y: 2, z: .025 }, direction: { x: 0, y: -1, z: 0 } };
  const hit = scenerySurfaceAt(base, start)!;
  assert.ok(Math.abs(hit.position_m.y - .7) < 1e-5);
  assert.deepEqual(scenerySurfaceAt(base, { ...start, direction: { x: 0, y: -7, z: 0 } }), hit,
    "surface bounds use physical distances for non-unit/interpolated rays too");
  const draw = penTool.begin({ scene: base, ray: start, values: values() })!.update(start)!;
  assert.equal(draw.patches[0]!.minimum[1], 14);
  const carve = penTool.begin({ scene: base, ray: start, values: values(1, 1) })!.update(start)!;
  assert.equal(carve.patches[0]!.minimum[1], 13);
  assert.equal(carve.patches[0]!.affectsScenery, true);
  assert.equal(draw.patches[0]!.affectsScenery, undefined);
});

test("over-budget pen samples leave the accepted cumulative prefix intact", () => {
  const base = scene(), start = ray(base, 4), gesture = penTool.begin({ scene: base, ray: start, values: values(64) })!;
  const first = gesture.update(start)!;
  assert.throws(() => gesture.update(ray(base, 20)), /Stroke is full/);
  assert.deepEqual(gesture.update(start)!.patches, first.patches);
});


test("consecutive scenery strokes pick the newly carved cut face then reveal the ground", () => {
  const base = scene();
  base.scenery = { ...base.scenery!, nodes: [minimalRoomSceneryGraph.nodes[0]!, { kind: "box", id: "block",
    place: { units: "metres", position: { x: 0, y: .6, z: 0 } }, halfSize: { x: .2, y: .1, z: .2 },
    material: { colorLinear: [1, 1, 1] } }] };
  const start = { origin: { x: .025, y: 2, z: .025 }, direction: { x: 0, y: -1, z: 0 } };
  const first = penTool.begin({ scene: base, ray: start, values: values(5, 1) })!.update(start)!;
  const cut = sceneWithSolidStroke(base, first.patches);
  const cap = scenerySurfaceAt(cut, start)!;
  assert.ok(cap.position_m.y > .54 && cap.position_m.y < .56, `cut face at ${cap.position_m.y}`);
  assert.ok(cap.normal.y > .99, "retained cut wall normal points into the cavity");
  assert.equal(cap.nodeId, "block");
  assert.ok(Math.abs(scenerySurfaceAt(base, start)!.position_m.y - .7) < 1e-5, "old document's uncut pick is unchanged");
  const second = penTool.begin({ scene: cut, ray: start, values: values(3, 1) })!.update(start)!;
  assert.ok(second.patches[0]!.minimum[1] < first.patches[0]!.minimum[1], "new stroke sinks into the exposed cut wall");
  const through = sceneWithSolidStroke(cut, second.patches);
  assert.equal(scenerySurfaceAt(through, start), undefined, "a fully opened hole does not pick the old face or back face");
  const next = penTool.begin({ scene: through, ray: start, values: values(1, 1) })!.update(start)!;
  assert.equal(next.patches[0]!.minimum[1], 3, "the terrain below becomes the nearest editable surface");
  const fromCavity = scenerySurfaceAt(cut, { origin: { x: .025, y: .65, z: .025 }, direction: { x: 0, y: -1, z: 0 } })!;
  assert.ok(Math.abs(fromCavity.position_m.y - cap.position_m.y) < 1e-4, "a ray beginning inside the cavity finds its near cut wall");
});

test("legacy voxel clears do not carve scenery and uncut rays retain exact picks", () => {
  const base = scene();
  base.scenery = { ...base.scenery!, nodes: [minimalRoomSceneryGraph.nodes[0]!, { kind: "box", id: "block",
    place: { units: "metres", position: { x: 0, y: .6, z: 0 } }, halfSize: { x: .2, y: .1, z: .2 },
    material: { colorLinear: [1, 1, 1] } }] };
  const start = { origin: { x: .025, y: 2, z: .025 }, direction: { x: 0, y: -1, z: 0 } };
  const expected = scenerySurfaceAt(base, start);
  const legacy = sceneWithSolidStroke(base, [{ operation: "clear", minimum: [0, 0, 0], maximumExclusive: [32, 32, 32] }]);
  assert.deepEqual(scenerySurfaceAt(legacy, start), expected);
  const distant = sceneWithSolidStroke(base, [{ operation: "clear", minimum: [2, 2, 2], maximumExclusive: [3, 3, 3], affectsScenery: true }]);
  assert.deepEqual(scenerySurfaceAt(distant, start), expected);
});

test("pen refuses scenery cuts beyond the shared arena capacity without losing prior samples", () => {
  const base = scene();
  base.solidVoxels = [...base.solidVoxels, ...Array.from({ length: SVO_SCULPT_CUT_MAXIMUM_RECORDS - 1 }, () => ({
    operation: "clear" as const, minimum: [2, 2, 2] as const, maximumExclusive: [3, 3, 3] as const, affectsScenery: true,
  }))];
  const start = ray(base, 8), gesture = penTool.begin({ scene: base, ray: start, values: values(1, 1) })!;
  const first = gesture.update(start)!;
  assert.throws(() => gesture.update(ray(base, 9)), /Scenery sculpt capacity reached.*4096/);
  assert.deepEqual(gesture.update(start)!.patches, first.patches);
});


test("a pen carve on a native imported paver exposes a pickable interior cut face", () => {
  const asset = "honed-travertine-paver", header = meshAssetHeader(asset);
  const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL("../public/assets/pool-party/honed-travertine-paver.fvx.gz", import.meta.url))), true);
  registerMeshAssetDistanceField(asset, { header: volume.header, bricks: new Map([...volume.bricks].map(
    ([key, { origin, distance }]) => [key, { origin, distance }])) });
  const center = header.origin.map((v, i) => v + header.dimensions[i]! * header.cellSize / 2);
  const base = createEmptyScene({ extents_m: { x: 4, y: 4, z: 4 }, finestCellSize_m: .025 });
  base.solidVoxels = [];
  base.scenery = { ...base.scenery!, nodes: [base.scenery!.nodes[0]!, { kind: "field-program", id: "paver",
    place: { units: "metres", position: { x: center[0]!, y: 1 + center[1]!, z: center[2]! } },
    program: { ops: [], result: 0, sampledVolume: { asset } }, material: { colorLinear: [1, 1, 1] } }] };
  const start = { origin: { x: .0125, y: 3, z: .0125 }, direction: { x: 0, y: -1, z: 0 } };
  const original = scenerySurfaceAt(base, start)!;
  assert.equal(original.nodeId, "paver");
  const carve = penTool.begin({ scene: base, ray: start, values: values(1, 1) })!.update(start)!;
  assert.equal(carve.patches[0]!.affectsScenery, true);
  const cut = sceneWithSolidStroke(base, carve.patches);
  const cap = scenerySurfaceAt(cut, start)!;
  assert.ok(cap.position_m.y < original.position_m.y - .001, `original ${original.position_m.y}, cap ${cap.position_m.y}`);
  assert.ok(Math.abs(cap.position_m.y - 1.025) < .0002, `cap follows the interior cut plane: ${cap.position_m.y}`);
  assert.ok(cap.normal.y > .99);
  const second = penTool.begin({ scene: cut, ray: start, values: values(1, 1) })!.update(start)!;
  assert.ok(second.patches[0]!.minimum[1] < carve.patches[0]!.minimum[1], "another stroke starts on the cut face, not the original mesh");
});
