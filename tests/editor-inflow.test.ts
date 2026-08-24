import assert from "node:assert/strict";
import test from "node:test";

import { createPaperScenario } from "../lib/core/paper-scenarios";
import { entityAtRay } from "../lib/core/editor-entity-catalog";
import {
  INFLOW_SELECTION_ID,
  inflowEntity,
  inflowFlowRate_L_s,
  inflowSpeed_m_s,
  setInflowFlowRate_L_s,
} from "../lib/core/editor-inflow";

test("hose flow is adjustable independently of launch speed", () => {
  const scene = createPaperScenario("hose-tank");
  const inflow = scene.fluid.inflow!;
  const next = setInflowFlowRate_L_s(inflow, 30, scene.container);

  assert.ok(Math.abs(inflowFlowRate_L_s(next) - 30) < 1e-9);
  assert.equal(inflowSpeed_m_s(next), inflowSpeed_m_s(inflow));
  assert.deepEqual(next.velocity_m_s, inflow.velocity_m_s);
  assert.ok(next.radius_m < inflow.radius_m);
});

test("selected hose exposes its volume flow in the scene UI", () => {
  const scene = createPaperScenario("hose-tank");
  const entity = inflowEntity.find({ scene, bodies: [] }, INFLOW_SELECTION_ID);
  const field = entity?.fields?.find((candidate) => candidate.id === "flow");

  assert.ok(field);
  assert.equal(field.label, "Flow");
  assert.equal(field.unit, "L/s");
  assert.ok(Math.abs(field.value - 1000 * Math.PI * 0.08 ** 2 * 2.6) < 1e-9);

  const patch = field.apply(24);
  const next = patch.fluid?.inflow;
  assert.ok(next);
  assert.ok(Math.abs(inflowFlowRate_L_s(next) - 24) < 1e-9);
  assert.equal(inflowSpeed_m_s(next), 2.6);
});

test("the hose remains selectable through its static nozzle shell", () => {
  const scene = createPaperScenario("hose-tank");
  const nozzle = scene.rigidBodies[0]!;
  const hit = entityAtRay({
    scene,
    bodies: [{ id: nozzle.id, position_m: nozzle.position_m,
      orientation: nozzle.orientation }],
  }, {
    origin: { x: -2, y: scene.fluid.inflow!.center_m.y, z: 0 },
    direction: { x: 1, y: 0, z: 0 },
  });

  assert.deepEqual(hit?.selection, { kind: "inflow", id: INFLOW_SELECTION_ID });
});
