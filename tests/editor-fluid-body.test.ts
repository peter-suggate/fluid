import assert from "node:assert/strict";
import test from "node:test";
import { validateScene, type SceneDescription, type Vec3 } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { sceneCellSizes_m } from "../lib/scene-lattice";
import { projectToViewport } from "../lib/webgpu-camera";
import {
  dragFluidBodyBox,
  fluidBodyEdgeSegment,
  shapeHandleAtPointer,
  fluidBodyBox,
  fluidBodyBoxPatch,
  fluidBodyBoxVolume_m3,
  fluidBodyHandleAxis,
  fluidBodyHandleById,
  fluidBodyHandles,
  fluidBodyLimits,
  scaleFluidBodyBox,
  scaleFluidBodyVolume,
} from "../lib/editor-fluid-body";
import { damBreakBoxContains, sceneDamBreakBox } from "../lib/initial-fluid";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

function applied(scene: SceneDescription, box: ReturnType<typeof fluidBodyBox>): SceneDescription {
  assert.ok(box);
  return { ...scene, ...fluidBodyBoxPatch(scene, box) };
}

test("the body box carries a handle for every face, edge, and corner", () => {
  const box = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
  const handles = fluidBodyHandles(box);
  assert.equal(handles.length, 26);
  assert.equal(handles.filter((handle) => handle.kind === "face").length, 6);
  assert.equal(handles.filter((handle) => handle.kind === "edge").length, 12);
  assert.equal(handles.filter((handle) => handle.kind === "corner").length, 8);
  assert.equal(new Set(handles.map((handle) => handle.id)).size, 26, "handle ids must be unique");
  const maxX = fluidBodyHandleById(box, "+00");
  assert.deepEqual(maxX?.position_m, { x: 1, y: 1, z: 0 });
  assert.deepEqual(maxX?.sides, { x: "max" });
  const corner = fluidBodyHandleById(box, "-+-");
  assert.deepEqual(corner?.position_m, { x: -1, y: 2, z: -1 });
  assert.deepEqual(corner?.sides, { x: "min", y: "max", z: "min" });
});

test("faces drag along their own normal; edges and corners do not claim one", () => {
  const box = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
  assert.deepEqual(fluidBodyHandleAxis(fluidBodyHandleById(box, "0+0")!), { x: 0, y: 1, z: 0 });
  assert.deepEqual(fluidBodyHandleAxis(fluidBodyHandleById(box, "00-")!), { x: 0, y: 0, z: 1 });
  assert.equal(fluidBodyHandleAxis(fluidBodyHandleById(box, "++0")!), undefined);
  assert.equal(fluidBodyHandleAxis(fluidBodyHandleById(box, "+++")!), undefined);
});

test("a face drag moves only the side it owns", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const handle = fluidBodyHandleById(box, "+00")!;
  const target = { x: box.min.x + 0.5 * (scene.container.width_m), y: 99, z: 99 };
  const dragged = dragFluidBodyBox(box, handle, target, scene);
  assert.equal(dragged.min.x, box.min.x);
  assert.deepEqual([dragged.min.y, dragged.max.y], [box.min.y, box.max.y]);
  assert.deepEqual([dragged.min.z, dragged.max.z], [box.min.z, box.max.z]);
  assert.ok(dragged.max.x > box.max.x, "the +x face should have moved outward");
});

test("a corner drag moves all three of its sides", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const handle = fluidBodyHandleById(box, "+++")!;
  const target = { x: limits.max.x, y: limits.max.y, z: limits.max.z };
  const dragged = dragFluidBodyBox(box, handle, target, scene);
  assert.deepEqual(dragged.min, box.min, "the opposite corner is an anchor");
  assert.ok(dragged.max.x > box.max.x && dragged.max.y > box.max.y && dragged.max.z > box.max.z);
});

test("drags snap to the finest lattice, so a moved handle always moves the water", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX] = sceneCellSizes_m(scene);
  const limits = fluidBodyLimits(scene);
  const handle = fluidBodyHandleById(box, "+00")!;
  const dragged = dragFluidBodyBox(box, handle, { x: box.max.x + cellX * 1.4, y: 0, z: 0 }, scene);
  const offset = (dragged.max.x - limits.min.x) / cellX;
  assert.ok(Math.abs(offset - Math.round(offset)) < 1e-9, `${offset} should be a whole number of cells`);
});

test("a side pushed through the body stops at one cell rather than inverting it", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX] = sceneCellSizes_m(scene);
  const handle = fluidBodyHandleById(box, "+00")!;
  const dragged = dragFluidBodyBox(box, handle, { x: box.min.x - 10, y: 0, z: 0 }, scene);
  assert.ok(dragged.max.x > dragged.min.x);
  assert.ok(Math.abs((dragged.max.x - dragged.min.x) - cellX) < 1e-9);
});

test("handles never push the body outside the container", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const dragged = dragFluidBodyBox(box, fluidBodyHandleById(box, "+++")!, { x: 99, y: 99, z: 99 }, scene);
  const pulled = dragFluidBodyBox(dragged, fluidBodyHandleById(dragged, "---")!, { x: -99, y: -99, z: -99 }, scene);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(pulled.min[axis] >= limits.min[axis] - 1e-9, `min ${axis}`);
    assert.ok(pulled.max[axis] <= limits.max[axis] + 1e-9, `max ${axis}`);
  }
  assert.deepEqual(validateScene(applied(scene, pulled)), []);
});

test("a box still in the corner stays anchored, keeping the closed-form GPU seed", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const grown = dragFluidBodyBox(box, fluidBodyHandleById(box, "0+0")!, { x: 0, y: scene.container.height_m, z: 0 }, scene);
  const patch = fluidBodyBoxPatch(scene, grown);
  assert.equal(patch.fluid.initialDamBreakOrigin_m, undefined,
    "an anchored reservoir must not author an origin — that would cost the analytic bootstrap");
  assert.deepEqual(validateScene({ ...scene, ...patch }), []);
});

test("a box dragged off the corner authors an origin the seeding honours", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX] = sceneCellSizes_m(scene);
  const moved = dragFluidBodyBox(box, fluidBodyHandleById(box, "-00")!, { x: box.min.x + 2 * cellX, y: 0, z: 0 }, scene);
  const next = applied(scene, moved);
  const origin = next.fluid.initialDamBreakOrigin_m;
  assert.ok(origin);
  assert.ok(Math.abs(origin.x - 2 * cellX) < 1e-9);
  assert.deepEqual(validateScene(next), []);

  // The seeding test must now exclude the cells the body was pulled off.
  const dam = sceneDamBreakBox(next);
  const nx = Math.round(next.container.width_m / next.voxelDomain.finestCellSize_m);
  assert.equal(damBreakBoxContains(dam, 0.5 / nx, 0.5, 0.5), false, "the vacated column must be dry");
  assert.equal(damBreakBoxContains(dam, (2.5) / nx, 0.5 * (dam.min.y + dam.max.y) / 1, 0.5), false);
  const inside = 0.5 * (dam.min.x + dam.max.x);
  assert.equal(damBreakBoxContains(dam, inside, 0.5 * (dam.min.y + dam.max.y), 0.5 * (dam.min.z + dam.max.z)), true);
});

test("an unmoved corner-anchored reservoir seeds exactly as it did before origins existed", () => {
  const scene = preset("water-box-dam-break");
  const dam = sceneDamBreakBox(scene);
  assert.deepEqual(dam.min, { x: 0, y: 0, z: 0 });
  const size = { x: dam.max.x, y: dam.max.y, z: dam.max.z };
  for (const fraction of [0.01, 0.2, 0.5, 0.99]) {
    assert.equal(
      damBreakBoxContains(dam, fraction, fraction, fraction),
      fraction <= size.x && fraction <= size.y && fraction <= size.z,
      `fraction ${fraction} must match the legacy corner test`);
  }
});

test("reshaping a filled tank keeps exactly the water it had", () => {
  const scene = preset("water-box-tank-fill");
  assert.equal(scene.fluid.initialCondition, "tank-fill");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const c = scene.container;
  assert.ok(Math.abs(fluidBodyBoxVolume_m3(box)
    - c.width_m * c.height_m * c.depth_m * c.fillFraction) < 1e-9);
  const next = applied(scene, box);
  assert.equal(next.fluid.initialCondition, "dam-break");
  assert.ok(Math.abs(next.container.fillFraction - c.fillFraction) < 1e-9,
    "the conversion must not change how much water there is");
  assert.deepEqual(validateScene(next), []);
});

test("grow and shrink scale about the centre and stay inside the tank", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const smaller = scaleFluidBodyBox(box, 0.5, scene);
  assert.ok(fluidBodyBoxVolume_m3(smaller) < fluidBodyBoxVolume_m3(box));
  const bigger = scaleFluidBodyBox(box, 2, scene);
  assert.ok(fluidBodyBoxVolume_m3(bigger) > fluidBodyBoxVolume_m3(box));
  for (const candidate of [smaller, bigger]) {
    for (const axis of ["x", "y", "z"] as const) {
      assert.ok(candidate.min[axis] >= limits.min[axis] - 1e-9);
      assert.ok(candidate.max[axis] <= limits.max[axis] + 1e-9);
    }
    assert.deepEqual(validateScene(applied(scene, candidate)), []);
  }
});

test("the water control doubles the water, not the edge lengths", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const before = fluidBodyBoxVolume_m3(box);
  const doubled = scaleFluidBodyVolume(box, 2, scene);
  // Snapping to whole cells keeps this from being exact, but ×2 must read as
  // twice the water beside a litres readout — never eight times it.
  const ratio = fluidBodyBoxVolume_m3(doubled) / before;
  assert.ok(ratio > 1.5 && ratio < 2.6, `volume ratio ${ratio} should be about 2`);
  const halved = scaleFluidBodyVolume(box, 0.5, scene);
  const shrunkRatio = fluidBodyBoxVolume_m3(halved) / before;
  assert.ok(shrunkRatio > 0.3 && shrunkRatio < 0.7, `volume ratio ${shrunkRatio} should be about 0.5`);
});

test("repeated growth saturates at the container instead of failing", () => {
  const scene = preset("water-box-dam-break");
  let box = fluidBodyBox(scene);
  assert.ok(box);
  for (let step = 0; step < 6; step += 1) box = scaleFluidBodyBox(box, 2, scene);
  const limits = fluidBodyLimits(scene);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(box.min[axis] - limits.min[axis]) < 1e-9);
    assert.ok(Math.abs(box.max[axis] - limits.max[axis]) < 1e-9);
  }
  assert.deepEqual(validateScene(applied(scene, box)), []);
});

test("a render-only scene has no shapeable body", () => {
  const scene = preset("water-box-dam-break");
  assert.equal(fluidBodyBox({ ...scene, systems: { fluid: false } }), undefined);
});

test("the box round-trips: authoring it and reading it back returns the same box", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX, , cellZ] = sceneCellSizes_m(scene);
  const moved = dragFluidBodyBox(
    dragFluidBodyBox(box, fluidBodyHandleById(box, "-0-")!,
      { x: box.min.x + cellX, y: 0, z: box.min.z + cellZ } as Vec3, scene),
    fluidBodyHandleById(box, "+0+")!,
    { x: box.max.x + cellX, y: 0, z: box.max.z + cellZ } as Vec3, scene);
  const next = applied(scene, moved);
  const reread = fluidBodyBox(next);
  assert.ok(reread);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(reread.min[axis] - moved.min[axis]) < 1e-9, `min ${axis}`);
    assert.ok(Math.abs(reread.max[axis] - moved.max[axis]) < 1e-9, `max ${axis}`);
  }
});

test("an edge handle spans the whole edge it grabs", () => {
  const box = { min: { x: -1, y: 0, z: -2 }, max: { x: 1, y: 3, z: 2 } };
  // "+0-" fixes max x and min z, leaving y free: the edge runs the box's height.
  const handle = fluidBodyHandleById(box, "+0-")!;
  const segment = fluidBodyEdgeSegment(box, handle);
  assert.ok(segment);
  assert.deepEqual(segment.from, { x: 1, y: 0, z: -2 });
  assert.deepEqual(segment.to, { x: 1, y: 3, z: -2 });
});

test("only edges have a segment; faces and corners are points", () => {
  const box = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
  assert.equal(fluidBodyEdgeSegment(box, fluidBodyHandleById(box, "+00")!), undefined);
  assert.equal(fluidBodyEdgeSegment(box, fluidBodyHandleById(box, "+++")!), undefined);
  for (const handle of fluidBodyHandles(box).filter((entry) => entry.kind === "edge")) {
    const segment = fluidBodyEdgeSegment(box, handle);
    assert.ok(segment, `${handle.id} must span an edge`);
    // Exactly one axis varies along an edge — the one the handle leaves free.
    const varying = (["x", "y", "z"] as const).filter((axis) => segment.from[axis] !== segment.to[axis]);
    assert.equal(varying.length, 1);
    assert.equal(handle.sides[varying[0]!], undefined);
  }
});

test("every edge segment lies on the box it came from", () => {
  const box = { min: { x: -1, y: 0, z: -2 }, max: { x: 1, y: 3, z: 2 } };
  for (const handle of fluidBodyHandles(box).filter((entry) => entry.kind === "edge")) {
    const segment = fluidBodyEdgeSegment(box, handle)!;
    for (const end of [segment.from, segment.to]) {
      for (const axis of ["x", "y", "z"] as const) {
        assert.ok(end[axis] === box.min[axis] || end[axis] === box.max[axis],
          `${handle.id} endpoint is off the box on ${axis}`);
      }
    }
  }
});

test("an edge is grabbable along its length, not just at its midpoint", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const camera = { azimuth_rad: 0.72, elevation_rad: 0.42, distance_m: 2.65, target_m: { x: 0, y: 0.38, z: 0 } };
  const [width, height] = [1200, 800];
  const candidates = [{ target: "fluid" as const, box }];

  // Find an edge whose whole span projects in front of the camera, then aim a
  // quarter of the way along it — nowhere near the midpoint a square would use.
  const edge = fluidBodyHandles(box).filter((handle) => handle.kind === "edge")
    .map((handle) => ({ handle, segment: fluidBodyEdgeSegment(box, handle)! }))
    .map(({ handle, segment }) => ({
      handle,
      ends: [segment.from, segment.to].map((point) => projectToViewport(point, camera, width, height)),
    }))
    .find(({ ends }) => ends.every((end) => end.visible && end.depth_m > 1e-6));
  assert.ok(edge, "fixture must present at least one fully visible edge");

  const [from, to] = edge.ends.map((end) => ({ x: end.leftFraction * width, y: end.topFraction * height }));
  const quarter = { x: from!.x + 0.25 * (to!.x - from!.x), y: from!.y + 0.25 * (to!.y - from!.y) };
  const pick = shapeHandleAtPointer(candidates, camera, width, height, quarter);
  assert.ok(pick, "a point on the drawn edge must grab something");
  assert.equal(pick.handleId, edge.handle.id, "and it must be the edge under the pointer");
});
