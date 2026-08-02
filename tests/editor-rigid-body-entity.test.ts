import assert from "node:assert/strict";
import test from "node:test";
import {
  rigidBodyBox,
  rigidBodyBoxPatch,
  rigidBodyEntity,
  rigidBodyResizePolicy,
  RIGID_MINIMUM_HALF_SIZE_M,
} from "../lib/editor-rigid-body";
import { boxSize, resizeBox, WORLD_FRAME, type EditorFrame } from "../lib/editor-entity";
import { createBodyDescription } from "../lib/rigid-body";
import { getScenePreset } from "../lib/scenes";
import { validateScene, type RigidBodyDescription, type RigidShape, type SceneDescription } from "../lib/model";
import { context, sides } from "./helpers/editor-entities";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

function withBody(shape: RigidShape): { scene: SceneDescription; body: RigidBodyDescription } {
  const base = preset("water-box-dam-break");
  const body = createBodyDescription(shape, 1, base.container.height_m);
  return { scene: { ...base, rigidBodies: [body] }, body };
}

test("each shape's box is the bounding box its dimensions actually describe", () => {
  const sphere = createBodyDescription("sphere", 1, 1);
  const half = boxSize(rigidBodyBox(sphere));
  assert.equal(half.x / 2, sphere.dimensions_m.x, "a sphere's x is its radius");
  assert.equal(half.x, half.y);
  assert.equal(half.y, half.z);

  const box = createBodyDescription("box", 1, 1);
  assert.deepEqual(boxSize(rigidBodyBox(box)), box.dimensions_m, "a box's dimensions are full extents");

  const cylinder = createBodyDescription("cylinder", 1, 1);
  const cylinderSize = boxSize(rigidBodyBox(cylinder));
  assert.equal(cylinderSize.x, 2 * cylinder.dimensions_m.x, "x is a radius");
  assert.equal(cylinderSize.y, cylinder.dimensions_m.y, "y is a full height");

  const capsule = createBodyDescription("capsule", 1, 1);
  const capsuleSize = boxSize(rigidBodyBox(capsule));
  assert.equal(capsuleSize.y, capsule.dimensions_m.y + 2 * capsule.dimensions_m.x,
    "a capsule's bounding height carries a cap at each end");
});

test("a dragged sphere stays a sphere, and a cylinder keeps its circular section", () => {
  for (const shape of ["sphere", "box", "cylinder", "capsule"] as const) {
    const { scene, body } = withBody(shape);
    const box = rigidBodyBox(body);
    const grown = resizeBox(box, sides("+00"), { x: box.max.x * 3, y: 0, z: 0 },
      rigidBodyResizePolicy(body));
    const patch = rigidBodyBoxPatch(scene, body, WORLD_FRAME, grown);
    const next = patch.rigidBodies[0]!;
    if (shape === "sphere") {
      assert.equal(next.dimensions_m.x, next.dimensions_m.y, "a sphere has one radius");
      assert.equal(next.dimensions_m.y, next.dimensions_m.z);
    }
    if (shape === "cylinder" || shape === "capsule") {
      assert.equal(next.dimensions_m.x, next.dimensions_m.z,
        `a ${shape} cross section must stay circular`);
      assert.equal(next.dimensions_m.y, body.dimensions_m.y,
        `a radial drag must not change a ${shape}'s height`);
    }
    assert.deepEqual(validateScene({ ...scene, ...patch }), [],
      `${shape} must stay a valid scene`);
  }
});

test("widening a capsule keeps the cylinder it had rather than eating it", () => {
  // Read naively the bounding half-height is the half-length plus one cap, so
  // growing the radius at a fixed bounding height shortens the body from both
  // ends and arrives at a sphere — a capsule that refuses to get fatter.
  const { scene, body } = withBody("capsule");
  const box = rigidBodyBox(body);
  const fatter = resizeBox(box, sides("+00"), { x: 3 * box.max.x, y: 0, z: 0 },
    rigidBodyResizePolicy(body));
  const next = rigidBodyBoxPatch(scene, body, WORLD_FRAME, fatter).rigidBodies[0]!;
  assert.ok(next.dimensions_m.x > body.dimensions_m.x, "the radius grew");
  assert.equal(next.dimensions_m.y, body.dimensions_m.y, "and the cylinder is untouched");

  // A y drag is what sets the length, and shrinking it past the radius is a
  // sphere rather than a negative capsule.
  const shorter = resizeBox(box, sides("0+0"), { x: 0, y: 0, z: 0 }, rigidBodyResizePolicy(body));
  const squat = rigidBodyBoxPatch(scene, body, WORLD_FRAME, shorter).rigidBodies[0]!;
  assert.ok(squat.dimensions_m.y >= 0);
  assert.ok(squat.dimensions_m.y < body.dimensions_m.y);
});

test("a free-axis drag moves the face, so the body's position follows its centre", () => {
  // Only a box has free axes; every other shape is symmetric about its origin.
  const { scene, body } = withBody("box");
  const box = rigidBodyBox(body);
  const grown = resizeBox(box, sides("+00"), { x: box.max.x + 0.2, y: 0, z: 0 },
    rigidBodyResizePolicy(body));
  const next = rigidBodyBoxPatch(scene, body, WORLD_FRAME, grown).rigidBodies[0]!;
  assert.ok(Math.abs(next.dimensions_m.x - (body.dimensions_m.x + 0.2)) < 1e-9);
  assert.ok(Math.abs(next.position_m.x - 0.1) < 1e-9,
    "the anchored -x side must stay put, which moves the centre by half the growth");
  assert.equal(next.position_m.y, 0, "and the untouched axes do not move");
});

test("a body cannot be shrunk out of existence", () => {
  const { scene, body } = withBody("sphere");
  const box = rigidBodyBox(body);
  const collapsed = resizeBox(box, sides("+00"), { x: 0, y: 0, z: 0 }, rigidBodyResizePolicy(body));
  const next = rigidBodyBoxPatch(scene, body, WORLD_FRAME, collapsed).rigidBodies[0]!;
  assert.ok(next.dimensions_m.x >= RIGID_MINIMUM_HALF_SIZE_M - 1e-9);
  assert.deepEqual(validateScene({ ...scene, ...next && scene }), []);
});

test("the handles sit on the body the user can see, not the one the document remembers", () => {
  // Mid-run a body is nowhere near its authored position, and handles drawn on
  // the authored one would be unusable.
  const { scene, body } = withBody("sphere");
  const live = { x: 0.31, y: 0.62, z: -0.14 };
  const entity = rigidBodyEntity.find(
    context(scene, [{ id: body.id, position_m: live, orientation: body.orientation }]), body.id);
  assert.ok(entity);
  assert.deepEqual(entity.frame.origin_m, live);
  assert.equal(entity.simulatedBodyId, body.id,
    "and it declares that it is drawn from the solver rather than the document");
});

test("a rotated body's handles resize along its own axes", () => {
  const { scene, body } = withBody("box");
  // A quarter turn about y: the body's +x now faces the world's -z.
  const orientation = { w: Math.SQRT1_2, x: 0, y: Math.SQRT1_2, z: 0 };
  const frame: EditorFrame = { origin_m: { x: 0, y: 0, z: 0 }, orientation };
  const box = rigidBodyBox(body);
  const grown = resizeBox(box, sides("+00"), { x: box.max.x + 0.2, y: 0, z: 0 },
    rigidBodyResizePolicy(body));
  const next = rigidBodyBoxPatch(scene, body, frame, grown).rigidBodies[0]!;
  assert.ok(Math.abs(next.dimensions_m.x - (body.dimensions_m.x + 0.2)) < 1e-9,
    "the body's own x is what grew");
  assert.ok(Math.abs(next.position_m.z + 0.1) < 1e-9,
    `the centre moved along world -z, got z=${next.position_m.z}`);
  assert.ok(Math.abs(next.position_m.x) < 1e-9, "and not along world x");
});

test("a body offers every handle: twenty-six to resize and four to move", () => {
  const { scene, body } = withBody("sphere");
  const entity = rigidBodyEntity.find(context(scene), body.id)!;
  assert.equal(entity.handles.filter((handle) => handle.space === "entity").length, 26);
  assert.equal(entity.handles.filter((handle) => handle.space === "world").length, 4,
    "a centre and three arrows");
  assert.equal(entity.handles.filter((handle) => handle.kind === "center").length, 1);
});

test("a body is picked by its bounds, nearest first", () => {
  const base = preset("water-box-dam-break");
  const near = { ...createBodyDescription("sphere", 1, 1), id: "near", position_m: { x: 0, y: 0.3, z: 0 } };
  const far = { ...createBodyDescription("sphere", 2, 1), id: "far", position_m: { x: 0, y: 0.3, z: 0.6 } };
  const scene = { ...base, rigidBodies: [far, near] };
  const hit = rigidBodyEntity.pick!(context(scene),
    { origin: { x: 0, y: 0.3, z: -3 }, direction: { x: 0, y: 0, z: 1 } });
  assert.equal(hit?.selection.id, "near", "the listed order must not decide what is in front");

  const missed = rigidBodyEntity.pick!(context(scene),
    { origin: { x: 0, y: 9, z: -3 }, direction: { x: 0, y: 0, z: 1 } });
  assert.equal(missed, undefined);
});

test("a body can be removed, and the removal is a whole scene", () => {
  const { scene, body } = withBody("sphere");
  const entity = rigidBodyEntity.find(context(scene), body.id)!;
  assert.ok(entity.remove);
  const next = entity.remove();
  assert.equal(next.rigidBodies.length, 0);
  assert.deepEqual(validateScene(next), []);
});
