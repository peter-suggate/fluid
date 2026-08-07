import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  decodeGPURigidBodyPoses,
  GPU_RIGID_BODY_CAPACITY,
  GPU_RIGID_RENDER_FLOATS,
} from "../lib/webgpu-rigid-body";
import { CLICK_SLOP_PX, pointerStayedWithinClickSlop } from "../lib/editor-tools";
import { drawnBodies, useDiagnosticsStore } from "../lib/stores/diagnostics-store";
import { editorBodyPoses, editorEntityContext } from "../lib/editor-entity-catalog";
import { initializeRigidBody } from "../lib/rigid-body";

/** One `RenderBody` record: position+radius, half+shape, orientation, colour. */
function renderRecord(values: Float32Array, index: number, position: readonly [number, number, number], orientation: readonly [number, number, number, number]) {
  const offset = index * GPU_RIGID_RENDER_FLOATS;
  values.set(position, offset);
  values.set(orientation, offset + 8);
}

// Poses are authored in f32-exact values so the assertions read as equalities
// rather than tolerances; the buffer itself is f32.
test("GPU render records decode to the poses the frame was drawn with", () => {
  const values = new Float32Array(GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS);
  renderRecord(values, 0, [0.125, 0.25, 0.375], [1, 0, 0, 0]);
  renderRecord(values, 1, [-0.5, 0.0625, 0.75], [0, 0, 1, 0]);
  const poses = decodeGPURigidBodyPoses(values, 2);
  assert.equal(poses.length, 2);
  assert.deepEqual(poses[0].position_m, { x: 0.125, y: 0.25, z: 0.375 });
  assert.deepEqual(poses[1].position_m, { x: -0.5, y: 0.0625, z: 0.75 });
  assert.deepEqual(poses[1].orientation, { w: 0, x: 0, y: 1, z: 0 });
});

test("a body's pose is read from its own record, not the roster's first", () => {
  // The regression this guards: a drag that starts from the wrong body's centre
  // holds the dragged body that far from the cursor for the whole gesture.
  const values = new Float32Array(GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS);
  renderRecord(values, 0, [0, 1.5, 0], [1, 0, 0, 0]);
  renderRecord(values, 2, [0, 0.0625, 0], [1, 0, 0, 0]);
  assert.equal(decodeGPURigidBodyPoses(values, 3)[2].position_m.y, 0.0625);
});

test("a malformed record stops the decode rather than reporting a pose at the origin", () => {
  const values = new Float32Array(GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS);
  renderRecord(values, 0, [0.2, 0.4, 0.1], [1, 0, 0, 0]);
  renderRecord(values, 1, [Number.NaN, 0, 0], [1, 0, 0, 0]);
  const poses = decodeGPURigidBodyPoses(values, 2);
  assert.equal(poses.length, 1, "the body behind the bad record falls back to the host mirror");
  assert.equal(decodeGPURigidBodyPoses(values, GPU_RIGID_BODY_CAPACITY + 4).length, 1);
});

test("the editor reads a body where the frame drew it, not where it was commanded", () => {
  const description = initializeRigidBody({
    id: "crate", name: "Crate", shape: "box",
    dimensions_m: { x: 0.1, y: 0.1, z: 0.1 }, density_kg_m3: 800,
    position_m: { x: 0, y: 1.25, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.2, friction: 0.4,
  });
  useDiagnosticsStore.getState().set({ bodies: [description], bodyPoses: {} });
  // Before the renderer has published anything, the commanded pose is all there
  // is — which is also the CPU backend's steady state.
  assert.equal(drawnBodies()[0].position_m.y, 1.25);

  useDiagnosticsStore.getState().set({ bodyPoses: {
    crate: { position_m: { x: 0, y: 0.05, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 } },
  } });
  assert.equal(drawnBodies()[0].position_m.y, 0.05, "the crate has settled; the roster has not heard");
  assert.equal(useDiagnosticsStore.getState().bodies[0].position_m.y, 1.25,
    "the command channel the solver reads must not be rewritten from the run");
  assert.equal(editorBodyPoses()[0].position_m.y, 0.05);
  assert.equal(editorEntityContext().bodies[0].position_m.y, 0.05);

  // A body the renderer has not published yet keeps its commanded pose rather
  // than vanishing from the editor's view of the scene.
  useDiagnosticsStore.getState().set({ bodyPoses: {} });
  assert.equal(drawnBodies()[0].position_m.y, 1.25);
});

test("a press and a release at the same point is a click, whatever it was over", () => {
  assert.equal(pointerStayedWithinClickSlop(0, 0), true);
  assert.equal(pointerStayedWithinClickSlop(CLICK_SLOP_PX, 0), true);
  assert.equal(pointerStayedWithinClickSlop(CLICK_SLOP_PX + 1, 0), false);
  assert.equal(pointerStayedWithinClickSlop(CLICK_SLOP_PX, CLICK_SLOP_PX), false);
});

test("grabbing a body neither selects it on the press nor starts from the host mirror", () => {
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const beginBodyDrag = viewport.slice(viewport.indexOf("const beginBodyDrag"), viewport.indexOf("const TERRAIN_HANDLE_TOLERANCE_PX"));
  assert.ok(beginBodyDrag.includes("simulation.dragBody"), "the grab opens a kinematic drag");
  assert.equal(beginBodyDrag.includes("selectBody"), false,
    "selecting on the press puts a bounding box around every throw; the release decides");
  // The pick is the only place the drawn pose is available, so the drag must
  // take its centre from there rather than from the diagnostics roster.
  const pointerDown = viewport.slice(viewport.indexOf("const pointerDown"), viewport.indexOf("const pointerMove"));
  assert.match(pointerDown, /beginBodyDrag\([^)]*picked\.position_m/,
    "the throw starts from the picked pose");
});
