import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { sceneDocument, sceneCardForDefinition, defineScene } from "../lib/core/scene-definition";
import { sceneRenderTuning } from "../lib/core/scene-render-profile";
import { parseScene, validateScene } from "../lib/core/model";
import { parseUIQueryState, parseMethodQueryState, serializeQueryState } from "../lib/core/url-state";
import { sceneOverrideClearPlan } from "../lib/core/scene-overrides";
import { DEFAULT_SVO_RENDER_TUNING } from "../lib/svo/pipeline/svo-render-tuning";

const definition = getSceneDefinition("forest-waterfall");

test("forest render defaults travel through definition, preset, card and saved document without changing fluid pitch", () => {
  assert.deepEqual(definition.renderProfile, { environmentRefinementDepth: 0 });
  assert.deepEqual(getScenePreset(definition.id).renderProfile, definition.renderProfile);
  const scene = sceneDocument(definition);
  assert.deepEqual(scene.renderProfile, definition.renderProfile);
  assert.deepEqual(sceneCardForDefinition(definition).open().scene.renderProfile, definition.renderProfile);
  assert.deepEqual(parseScene(JSON.stringify(scene)).renderProfile, definition.renderProfile);
  assert.equal(scene.voxelDomain.finestCellSize_m, .125);
  assert.equal(sceneRenderTuning(scene.renderProfile).environmentRefinementDepth, 0);
  assert.equal(scene.voxelDomain.finestCellSize_m / 2 ** sceneRenderTuning(scene.renderProfile).environmentRefinementDepth, .125,
    "the finer fluid does not refine the previous app environment lattice again");
  assert.equal(sceneRenderTuning(), DEFAULT_SVO_RENDER_TUNING, "other scenes retain their global defaults");
  assert.equal(sceneRenderTuning(scene.renderProfile, DEFAULT_SVO_RENDER_TUNING), DEFAULT_SVO_RENDER_TUNING,
    "an explicitly chosen global default is still an override");
});

test("forest URL defaults and explicit refinement overrides survive canonical serialization", () => {
  const profile = definition.renderProfile!;
  const normal = parseUIQueryState("", definition.id);
  assert.equal(normal.svoRenderTuning.environmentRefinementDepth, 0);
  assert.deepEqual(normal.svoRenderTuning, sceneRenderTuning(profile));
  assert.equal(parseUIQueryState("svoRefinementDepth=invalid", definition.id).svoRenderTuning.environmentRefinementDepth, 0);
  assert.equal(parseUIQueryState("svoRefinementDepth=99", definition.id).svoRenderTuning.environmentRefinementDepth, 0);
  assert.equal(parseUIQueryState("", "hero-garden-hose").svoRenderTuning.environmentRefinementDepth, DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth);
  const scene = sceneDocument(definition), method = parseMethodQueryState("");
  for (const depth of [0, 1, 2]) {
    const ui = parseUIQueryState(`svoRefinementDepth=${depth}`, definition.id);
    assert.equal(ui.svoRenderTuning.environmentRefinementDepth, depth);
    const serialized = serializeQueryState("foreign=preserved", { scene, presetId: definition.id }, method, ui);
    const query = new URLSearchParams(serialized);
    assert.equal(query.get("foreign"), "preserved");
    assert.equal(query.get("svoRefinementDepth"), depth === 0 ? null : String(depth));
    assert.equal(parseUIQueryState(serialized, definition.id).svoRenderTuning.environmentRefinementDepth, depth);
  }
  const reset = sceneOverrideClearPlan(["svoRefinementDepth"], { scene, presetId: definition.id });
  assert.equal(reset.svoRenderTuning.environmentRefinementDepth, 0);
  assert.deepEqual(reset.reload, []);
});

test("malformed authored renderer profiles fail definition and scene validation", () => {
  for (const depth of [-1, .5, 4, Infinity]) {
    assert.throws(() => defineScene({ ...definition, renderProfile: { environmentRefinementDepth: depth } }), /refinement depth/);
    const scene = sceneDocument(definition);
    scene.renderProfile = { environmentRefinementDepth: depth };
    assert.ok(validateScene(scene).some(error => error.includes("refinement depth")));
  }
});

test("authored shaded water preserves explicit surface overrides and restores its scene default", () => {
  const stair = getSceneDefinition("nbflip-stair-cube"), scene = sceneDocument(stair);
  assert.equal(parseUIQueryState("", stair.id).fluidSurfaceRenderMode, "shaded");
  assert.equal(parseUIQueryState("", stair.id).fluidParticleDisplay, "off");
  for (const display of ["off", "motion", "speed", "fine"] as const) {
    const ui = parseUIQueryState(`fluidParticles=${display}`, stair.id);
    const query = serializeQueryState("", { scene, presetId: stair.id }, parseMethodQueryState(""), ui);
    assert.equal(new URLSearchParams(query).get("fluidParticles"), display === "off" ? null : display);
    assert.equal(parseUIQueryState(query, stair.id).fluidParticleDisplay, display);
  }
  assert.equal(sceneOverrideClearPlan(["fluidParticles"], { scene, presetId: stair.id }).ui.fluidParticleDisplay, "off");
  assert.equal(parseUIQueryState("fluidSurface=invalid", stair.id).fluidSurfaceRenderMode, "shaded");
  assert.equal(parseUIQueryState("", definition.id).fluidSurfaceRenderMode, "simple");
  for (const mode of ["shaded", "wireframe", "simple"] as const) {
    const ui = parseUIQueryState(`fluidSurface=${mode}`, stair.id);
    const query = serializeQueryState("", { scene, presetId: stair.id }, parseMethodQueryState(""), ui);
    assert.equal(new URLSearchParams(query).get("fluidSurface"), mode === "shaded" ? null : mode);
    assert.equal(parseUIQueryState(query, stair.id).fluidSurfaceRenderMode, mode);
  }
  const reset = sceneOverrideClearPlan(["fluidSurface"], { scene, presetId: stair.id });
  assert.equal(reset.ui.fluidSurfaceRenderMode, "shaded");
  assert.deepEqual(reset.reload, []);
  assert.throws(() => parseScene(JSON.stringify({ ...scene, renderProfile: { fluidSurfaceRenderMode: "invalid" } })), /surface render mode/);
});
