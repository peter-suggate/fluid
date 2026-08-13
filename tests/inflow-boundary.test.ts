import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { averageInflowStrength, createInflowGridBoundary, inflowBoundaryWGSL, inflowOutletCenter, integratedInflowVolume } from "../lib/inflow-boundary";
import { createPaperScenario } from "../lib/paper-scenarios";
import { createTallCellLayout } from "../lib/tall-cell-grid";

const uniformShader = readFileSync(
  new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");

test("hose outlet is a normalized face flux rather than a painted volume", () => {
  const scene = createPaperScenario("hose-tank"), inflow = scene.fluid.inflow!;
  const layout = createTallCellLayout(scene, "balanced");
  const boundary = createInflowGridBoundary(inflow, scene.container, [layout.nx, layout.fineNy, layout.nz]);
  const outlet = inflowOutletCenter(inflow);
  const nozzle = scene.rigidBodies.find((body) => body.id === "paper-hose-nozzle")!;
  assert.ok(Math.abs(outlet.x - (nozzle.position_m.x + 0.5 * nozzle.dimensions_m.y)) < 1e-12, "inflow outlet must meet the visible nozzle tip");
  assert.deepEqual(outlet, { x: -0.38, y: 0.55, z: 0 });
  assert.equal(boundary.axis, 0);
  assert.equal(boundary.receiverIndex, boundary.faceIndex + 1);
  const normalizedRate = boundary.rawProjectedArea_m2 * boundary.apertureScale * Math.abs(inflow.velocity_m_s.x);
  assert.ok(Math.abs(normalizedRate - boundary.flowRate_m3_s) < 1e-12, `${normalizedRate} != ${boundary.flowRate_m3_s}`);
  assert.ok(boundary.apertureScale > 0.9 && boundary.apertureScale < 1.1, String(boundary.apertureScale));
});

test("projected aperture normalization is orientation independent", () => {
  const scene = createPaperScenario("hose-tank"), inflow = scene.fluid.inflow!;
  inflow.velocity_m_s = { x: -0.37, y: 0.81, z: 0.22 };
  inflow.center_m = { x: 0.2, y: 0.35, z: -0.1 };
  const boundary = createInflowGridBoundary(inflow, scene.container, [61, 46, 41]);
  assert.equal(boundary.axis, 1);
  assert.equal(boundary.receiverIndex, boundary.faceIndex + 1);
  const normalizedRate = boundary.rawProjectedArea_m2 * boundary.apertureScale * Math.abs(inflow.velocity_m_s.y);
  assert.ok(Math.abs(normalizedRate - boundary.flowRate_m3_s) < 1e-12, `${normalizedRate} != ${boundary.flowRate_m3_s}`);
});

test("step-averaged ramp integrates the configured tap volume", () => {
  const inflow = { ...createPaperScenario("hose-tank").fluid.inflow!, ramp_s: 0.35 };
  assert.ok(Math.abs(averageInflowStrength(inflow, 0, inflow.ramp_s) - 0.5) < 1e-12);
  assert.equal(averageInflowStrength(inflow, inflow.ramp_s, 1), 1);
  const time = 3.144, expectedStrengthIntegral = time - 0.5 * inflow.ramp_s;
  assert.ok(Math.abs(averageInflowStrength(inflow, 0, time) * time - expectedStrengthIntegral) < 1e-12);
  const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
  const expectedVolume = Math.PI * inflow.radius_m ** 2 * speed * expectedStrengthIntegral;
  assert.ok(Math.abs(integratedInflowVolume(inflow, 0, time) - expectedVolume) < 1e-9);
});

test("the conservative boundary-face flux stays loop-free", () => {
  assert.doesNotMatch(inflowBoundaryWGSL, /\bfor\s*\(/, "the shared aperture must not expand quadrature loops into transport pipelines");
});

test("large-CFL inflow rasterizes the full swept plug instead of saturating one receiver layer", () => {
  assert.match(inflowBoundaryWGSL, /fn inflowSweptPlugSource/);
  assert.match(inflowBoundaryWGSL,
    /let overlap=max\(0\.0,min\(axial\+halfAxial,speed\*dt\)-max\(axial-halfAxial,0\.0\)\)/,
    "the source must cover every cell interval crossed during the timestep");
  assert.doesNotMatch(inflowBoundaryWGSL,
    /fn inflowSweptPlugSource[\s\S]*q\[axis\]!=inflowReceiverIndex/,
    "the swept source must not collapse back to one receiver plane");
  assert.match(inflowBoundaryWGSL,
    /fn isInflowSweptVelocityCell[\s\S]*upstreamSupport\|\|inflowSweptPlugSource\(q,params\.dimsDt\.w\)>0\.0/,
    "the prescribed velocity must carry every newly rasterized plug cell away from the nozzle");
  assert.match(uniformShader, /return applyInflowSweptVelocity\(id,v\)/,
    "the swept reservoir velocity must be applied before pressure projection");
  assert.match(uniformShader, /v=applyInflowVelocity\(id,v\);textureStore\(velocityOut/,
    "post-projection enforcement must remain limited to the actual nozzle face");
  assert.match(uniformShader,
    /let inflowSource=min\(inflowSweptPlugSource\(id,params\.dimsDt\.w\),max\(0\.0,1\.0-rhoNext\)\)/,
    "recirculating liquid in the virtual nozzle must not be overfilled");
  assert.match(uniformShader,
    /if\(inflowSource>0\.0\)\{gammaNext=max\(gammaNext,1\.0\);\}/,
    "newly wetted reservoir cells need the unit cumulative-gamma state used at startup");
});
