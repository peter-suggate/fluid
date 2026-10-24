import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { ellipsoidBounds, subcellEllipsoidPatch } from "../lib/core/solid-patch/ellipsoid-bounds";
import { validateSolidPatch } from "../lib/core/solid-patch/compose";
import { mirroredPatches } from "../lib/core/voxel-editor/geometry";
import { materialControl, mirrorControls } from "../lib/core/voxel-editor/controls";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { penTool } from "../lib/core/voxel-editor/plugins/pen";
import { centerTool } from "../lib/core/voxel-editor/plugins/center";
import { voxelBrushTool } from "../lib/core/voxel-editor/plugins/voxel";
import { lineTool } from "../lib/core/voxel-editor/plugins/line";
import { patternTool } from "../lib/core/voxel-editor/plugins/pattern";

function fixture() {
  const scene = createEmptyScene();
  scene.scenery = { ...scene.scenery!, nodes: [scene.scenery!.nodes[0]!] };
  scene.solidVoxels = [{ operation: "fill", minimum: [2, 1, 2], maximumExclusive: [30, 4, 30], materialId: 2 }];
  const h = sceneCellSizes_m(scene);
  const ray = (x: number, z = 8.625) => ({ origin: { x: -scene.container.width_m / 2 + x * h[0], y: 2,
    z: -scene.container.depth_m / 2 + z * h[2] }, direction: { x: 0, y: -1, z: 0 } });
  return { scene, ray };
}

test("Pen Draw uses the same subcell sphere as Carve and retains the chosen material", () => {
  const { scene, ray } = fixture(), start = ray(10.625);
  const material = materialControl.options.find(option => option.value !== materialControl.initial)!.value;
  const draw = penTool.begin({ scene, ray: start, values: toolValues(penTool, { size: 5, mode: 0, material }) })!;
  const first = draw.update(start)!;
  const carve = penTool.begin({ scene, ray: start, values: toolValues(penTool, { size: 5, mode: 1 }) })!.update(start)!;
  assert.deepEqual(ellipsoidBounds(first.patches[0]!), ellipsoidBounds(carve.patches[0]!));
  assert.equal(first.patches[0]!.materialId, material);
  assert.equal(first.patches[0]!.affectsScenery, undefined);
  assert.equal(draw.update(start), first, "unchanged pen samples reuse the nib outline too");
  assert.equal(draw.update(start)!.patches, first.patches, "unchanged pen samples reuse their immutable patch list");
  assert.notDeepEqual(draw.update(ray(10.875))!.patches, first.patches);
});

test("Center spheres stay on the press point and grow at subcell radii", () => {
  const { scene, ray } = fixture(), start = ray(10.625);
  const gesture = centerTool.begin({ scene, ray: start, values: toolValues(centerTool, { shape: 0 }) })!;
  const click = gesture.update(start)!;
  click.patches.forEach(validateSolidPatch);
  assert.deepEqual(click.patches[0]!.maximumExclusive.map((v, a) => v - click.patches[0]!.minimum[a]!), [1, 1, 1]);
  const grown = gesture.update(ray(13))!, bounds = ellipsoidBounds(grown.patches[0]!);
  grown.patches.forEach(validateSolidPatch);
  assert.deepEqual(bounds.minimum.map((v, a) => (v + bounds.maximumExclusive[a]!) / 2), [10.625, 4, 8.625]);
  assert.deepEqual(bounds.maximumExclusive.map((v, a) => v - bounds.minimum[a]!), [4.75, 4.75, 4.75]);
  assert.equal(gesture.update(ray(13.01)), grown, "motion below radius precision reuses geometry and outline");
  assert.notEqual(gesture.update(ray(13.25)), grown);
});

test("mirroring preserves both asymmetric subcell spheres sharing one integer box", () => {
  const { scene } = fixture();
  const patch = subcellEllipsoidPatch([14.125, 2, 6], [17.125, 5, 9], "clear");
  const images = mirroredPatches({ scene, values: { [mirrorControls[0].id]: 1 } }, [patch]);
  assert.equal(images.length, 2);
  assert.deepEqual(images[0]!.minimum, images[1]!.minimum);
  assert.equal(ellipsoidBounds(images[1]!).minimum[0], 14.875);
  const symmetric = subcellEllipsoidPatch([14.125, 2, 6], [17.875, 5, 9], "clear");
  assert.equal(mirroredPatches({ scene, values: { [mirrorControls[0].id]: 1 } }, [symmetric]).length, 1);
});

test("Voxel misses break interpolation and unchanged brush proposals reuse their preview", () => {
  const { scene, ray } = fixture(), start = ray(4.5);
  const gesture = voxelBrushTool.begin({ scene, ray: start, values: toolValues(voxelBrushTool) })!;
  const first = gesture.update(start)!;
  assert.equal(gesture.update(ray(4.6)), first);
  gesture.update({ ...start, direction: { x: 0, y: 1, z: 0 } });
  const moved = gesture.update(ray(20.5))!;
  assert.deepEqual(moved.patches.map(p => p.minimum[0]), [4, 20], "re-entry does not bridge the off-surface excursion");
  assert.notEqual(moved, first);
});

test("Line and Pattern rebuild only when their lattice placement changes", () => {
  const { scene, ray } = fixture(), start = ray(4.5);
  const clipboard = { minimum: [0, 0, 0] as const, block: { size: [1, 1, 1] as const,
    solidFraction: new Uint8Array([255]), signedDistanceQ8: new Int16Array([-128]), materialId: new Uint16Array([2]) } };
  for (const tool of [lineTool, patternTool]) {
    const gesture = tool.begin({ scene, ray: start, values: toolValues(tool), clipboard })!;
    const first = gesture.update(ray(8.5))!;
    assert.equal(gesture.update(ray(8.6)), first, tool.id);
    const moved = gesture.update(ray(9.5))!;
    assert.notEqual(moved, first, tool.id);
    assert.notDeepEqual(moved.patches, first.patches, tool.id);
  }
});
