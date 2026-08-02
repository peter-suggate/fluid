import assert from "node:assert/strict";
import test from "node:test";
import {
  aimInflow,
  createInflowAt,
  inflowDirection,
  inflowEntity,
  INFLOW_SELECTION_ID,
  inflowSpeed_m_s,
  moveInflow,
  setInflowRadius,
  INFLOW_ARROW_SECONDS,
  INFLOW_MAXIMUM_SPEED_M_S,
  INFLOW_MINIMUM_RADIUS_M,
  INFLOW_MINIMUM_SPEED_M_S,
} from "../lib/editor-inflow";
import { cloneScene, defaultScene, validateScene, type SceneDescription } from "../lib/model";

function scene(): SceneDescription {
  const next = cloneScene(defaultScene);
  next.container.top = "open";
  return next;
}

test("a hose placed on a surface sprays away from it and validates", () => {
  const base = scene();
  const inflow = createInflowAt({ x: 0, y: 0.1, z: 0 }, { x: 0, y: 1, z: 0 }, base);
  const direction = inflowDirection(inflow);
  assert.ok(direction.y < -0.99, "a nozzle on the floor aims its jet downward into the tank");
  assert.ok(inflow.center_m.y > 0.1, "the nozzle body sits off the surface it was placed on");
  assert.deepEqual(validateScene({ ...base, fluid: { ...base.fluid, inflow } }), []);
});

test("placement clamps the nozzle inside the container", () => {
  const base = scene();
  const c = base.container;
  const inflow = createInflowAt({ x: c.width_m, y: c.height_m * 2, z: -c.depth_m }, { x: 1, y: 1, z: -1 }, base);
  assert.ok(Math.abs(inflow.center_m.x) <= c.width_m / 2 + 1e-9);
  assert.ok(inflow.center_m.y >= 0 && inflow.center_m.y <= c.height_m + 1e-9);
  assert.ok(Math.abs(inflow.center_m.z) <= c.depth_m / 2 + 1e-9);
  assert.deepEqual(validateScene({ ...base, fluid: { ...base.fluid, inflow } }), []);
});

test("the arrow tip encodes direction and speed and survives a round trip", () => {
  const base = scene();
  const inflow = createInflowAt({ x: 0, y: 0.2, z: 0 }, { x: 0, y: 1, z: 0 }, base);
  // The arrow is an ordinary handle now, so this reads it where the pointer
  // would find it rather than through a bespoke accessor.
  const entity = inflowEntity.find(
    { scene: { ...base, fluid: { ...base.fluid, inflow } }, bodies: [] }, INFLOW_SELECTION_ID)!;
  const nozzle = entity.handles.find((handle) => handle.id === "aim")!;
  const offset = Math.hypot(
    nozzle.position_m.x - inflow.center_m.x,
    nozzle.position_m.y - inflow.center_m.y,
    nozzle.position_m.z - inflow.center_m.z,
  );
  assert.ok(Math.abs(offset - INFLOW_ARROW_SECONDS * inflowSpeed_m_s(inflow)) < 1e-9);

  const aimed = aimInflow(inflow, { x: inflow.center_m.x + INFLOW_ARROW_SECONDS * 3, y: inflow.center_m.y, z: inflow.center_m.z });
  assert.ok(Math.abs(inflowSpeed_m_s(aimed) - 3) < 1e-9, "arrow length sets speed");
  assert.ok(inflowDirection(aimed).x > 0.99, "arrow direction sets the jet axis");
});

test("aiming refuses degenerate velocities that validateScene would reject", () => {
  const base = scene();
  const inflow = createInflowAt({ x: 0, y: 0.2, z: 0 }, { x: 0, y: 1, z: 0 }, base);
  assert.deepEqual(aimInflow(inflow, { ...inflow.center_m }), inflow, "a tip on the nozzle is inert");

  const crawling = aimInflow(inflow, { x: inflow.center_m.x + 1e-4, y: inflow.center_m.y, z: inflow.center_m.z });
  assert.ok(inflowSpeed_m_s(crawling) >= INFLOW_MINIMUM_SPEED_M_S - 1e-12, "speed floors instead of reaching zero");
  assert.deepEqual(validateScene({ ...base, fluid: { ...base.fluid, inflow: crawling } }), []);

  const blasting = aimInflow(inflow, { x: inflow.center_m.x + 1000, y: inflow.center_m.y, z: inflow.center_m.z });
  assert.ok(inflowSpeed_m_s(blasting) <= INFLOW_MAXIMUM_SPEED_M_S + 1e-9);
});

test("moving and resizing stay inside the authoring constraints", () => {
  const base = scene();
  const c = base.container;
  const inflow = createInflowAt({ x: 0, y: 0.2, z: 0 }, { x: 0, y: 1, z: 0 }, base);
  const moved = moveInflow(inflow, { x: 99, y: -99, z: 99 }, c);
  assert.equal(moved.center_m.x, c.width_m / 2);
  assert.equal(moved.center_m.y, 0);
  assert.equal(moved.center_m.z, c.depth_m / 2);
  assert.deepEqual(validateScene({ ...base, fluid: { ...base.fluid, inflow: moved } }), []);

  assert.equal(setInflowRadius(inflow, -1, c).radius_m, INFLOW_MINIMUM_RADIUS_M);
  assert.equal(setInflowRadius(inflow, 99, c).radius_m, 0.5 * Math.min(c.width_m, c.depth_m));
});
