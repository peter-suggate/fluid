import assert from "node:assert/strict";
import test from "node:test";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { inflowStrength } from "../lib/initial-fluid";
import { validateScene } from "../lib/model";
import { createPaperScenario, paperScenarios } from "../lib/paper-scenarios";
import { advanceRigidBodies, initializeRigidBodies } from "../lib/rigid-body";
import { createTallCellLayout } from "../lib/tall-cell-grid";

test("paper-derived scenarios are valid, deterministic, and uniquely identified", () => {
  assert.deepEqual(paperScenarios.map((scenario) => scenario.paperFigure), ["Figure 3", "Figure 4", "Figure 6"]);
  for (const metadata of paperScenarios) {
    const first = createPaperScenario(metadata.id), second = createPaperScenario(metadata.id);
    assert.deepEqual(first, second);
    assert.deepEqual(validateScene(first), []);
    assert.equal(new Set(first.rigidBodies.map((body) => body.id)).size, first.rigidBodies.length);
  }
});

test("inflow ramp is bounded and zero outside its configured interval", () => {
  assert.equal(inflowStrength(-1, 0, 2, 0.5), 0);
  assert.equal(inflowStrength(0, 0, 2, 0.5), 0);
  assert.equal(inflowStrength(0.25, 0, 2, 0.5), 0.5);
  assert.equal(inflowStrength(1, 0, 2, 0.5), 1);
  assert.equal(inflowStrength(1.75, 0, 2, 0.5), 0.5);
  assert.equal(inflowStrength(2, 0, 2, 0.5), 0);
});

test("hose source injects represented liquid into the CPU oracle", () => {
  const scene = createPaperScenario("hose-tank");
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.05;
  const solver = new EulerianFluidSolver(scene, 5000), initialMarkers = solver.markers.length;
  for (let step = 0; step < 8; step += 1) solver.step(0.05);
  assert.ok(solver.markers.length > initialMarkers, `${solver.markers.length} <= ${initialMarkers}`);
  assert.equal(solver.diagnostics.nanCount, 0);
});

test("hose layout retains a regular band spanning the receiving surface and nozzle", () => {
  const scene = createPaperScenario("hose-tank"), layout = createTallCellLayout(scene, "balanced");
  const inflow = scene.fluid.inflow!, nozzle = Math.floor(inflow.center_m.y / scene.container.height_m * layout.fineNy);
  const x = Math.max(0, Math.min(layout.nx - 1, Math.floor((inflow.center_m.x / scene.container.width_m + 0.5) * layout.nx)));
  const z = Math.max(0, Math.min(layout.nz - 1, Math.floor((inflow.center_m.z / scene.container.depth_m + 0.5) * layout.nz)));
  const base = layout.columnBases[x + layout.nx * z];
  assert.ok(base <= nozzle && nozzle < base + layout.settings.regularLayers, `nozzle ${nozzle} is outside [${base}, ${base + layout.settings.regularLayers})`);
  assert.equal(layout.packedNy, layout.settings.regularLayers + 2);
  assert.ok(layout.planning.storedRegularLayers > layout.planning.requestedRegularLayers, "the disconnected inlet should select a larger construction-time B_y");
  let expectedVolume = 0;
  for (let y = 0; y < layout.fineNy; y += 1) if ((y + 0.5) / layout.fineNy <= scene.container.fillFraction) expectedVolume += layout.nx * layout.nz;
  assert.equal(layout.initialVolumeCellSum, expectedVolume, "fractional tall cells must preserve the shallow pool volume");
});

test("paper dam break retains tall columns across its vertical liquid face", () => {
  const layout = createTallCellLayout(createPaperScenario("dam-break-boxes"), "balanced");
  assert.ok(layout.settings.regularLayers >= layout.planning.requestedRegularLayers);
  assert.equal(layout.planning.ordinaryGridFallback, false);
  assert.ok(layout.columnBases.some((base) => base >= 2));
  assert.ok(layout.columnBases.every((base) => base <= layout.settings.maximumTallHeight));
});

test("paper sphere obstacle remains fixed while dynamic bodies advance", () => {
  const scene = createPaperScenario("sphere-jet"), bodies = initializeRigidBodies(scene.rigidBodies), before = { ...bodies[0].position_m };
  advanceRigidBodies(bodies, scene, 1 / 30);
  assert.deepEqual(bodies[0].position_m, before);
  assert.deepEqual(bodies[0].linearVelocity_m_s, { x: 0, y: 0, z: 0 });
  assert.equal(bodies[0].inverseMass_kg, 0);
});

test("sphere jet exits the hose at high speed", () => {
  const inflow = createPaperScenario("sphere-jet").fluid.inflow!;
  const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
  assert.ok(speed >= 1.2, `expected at least 1.2 m/s, received ${speed} m/s`);
  assert.ok(inflow.velocity_m_s.x > 0, "sphere jet must travel out of the left-hand hose");
});
