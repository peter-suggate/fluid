import assert from "node:assert/strict";
import test from "node:test";
import type { FluidPipelineContext } from "../lib/core/fluid-pipeline";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";

const context = (densityTransport: string): FluidPipelineContext => ({
  values: { densityTransport, gammaDiffusion: "on", surfaceSharpening: "on" },
  info: null, sceneId: "any-scene", bodyCount: 0, hasTerrain: false, hasInflow: false, running: false,
});
function stages(value: FluidPipelineContext) {
  return Object.fromEntries(ADAPTIVE_MASS_FLUID_PIPELINE.stages.map(stage =>
    [stage.id, { ...stage, ...stage.presentation?.(value) }]));
}

test("current spatial field panel exposes density authority and distinguishes bypass from scalar publication", () => {
  const ctx = context("current-map"), rows = stages(ctx);
  const transport = rows["conservative-transport"]!;
  assert.equal(transport.label, "Current spatial field transport");
  assert.match(transport.tip.summary, /same field supplies native liquid amounts/);
  assert.ok(transport.controls?.some(control => control.kind === "param-choice" && control.param === "densityTransport"));
  assert.equal(rows["gamma-diffusion"]!.state(ctx), "unavailable");
  assert.equal(rows["gamma-diffusion"]!.toggle, undefined);
  assert.equal(rows["surface-sharpening"]!.state(ctx), "on");
  assert.equal(rows["surface-sharpening"]!.label, "Scalar mask publication");
  assert.equal(rows["surface-sharpening"]!.toggle, undefined);
  assert.deepEqual(rows["surface-sharpening"]!.controls, []);
});

test("native CM12 stage copy, toggles, and timestamp ownership remain unchanged", () => {
  const ctx = context("native-cm12"), rows = stages(ctx);
  assert.equal(rows["conservative-transport"]!.label, "Mass + gamma + momentum transport");
  assert.equal(rows["conservative-transport"]!.chip(ctx), "trace · scatter · gather");
  for (const id of ["gamma-diffusion", "surface-sharpening"]) {
    assert.equal(rows[id]!.state(ctx), "on");
    assert.ok(rows[id]!.toggle);
  }
  const current = stages(context("current-map"));
  for (const id of Object.keys(rows)) assert.deepEqual(current[id]!.phaseLabels, rows[id]!.phaseLabels);
});
