/**
 * The open top's headroom (fluid.brickTiers.openTopHeadroom, 2026-11-18): the cells of air the brick lattice keeps
 * above the container, whole 32-cell pages, one page when absent. churning-tank's paddles throw spray past one page
 * (FAIL.REACH at frame 591), so it authors three. CPU only.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import { validateScene } from "../../core/model";
import { BRICK_OPEN_TOP_HEADROOM, brickSceneSetup } from "./brick-scene";

test("brick headroom: the lattice holds the authored headroom above an open top, one page when absent", () => {
  const tank = sceneDocument(getSceneDefinition("bulk-rest-tank"));
  assert.equal(tank.fluid.brickTiers?.openTopHeadroom, undefined);
  assert.equal(brickSceneSetup(tank, { timeStep: "sixtieth" }).setup!.lattice[1], 288 + BRICK_OPEN_TOP_HEADROOM);
  const tall = structuredClone(tank); tall.fluid.brickTiers = { openTopHeadroom: 96 };
  const setup = brickSceneSetup(tall, { timeStep: "sixtieth" }).setup!;
  assert.deepEqual([setup.lattice, setup.openTop, setup.farPhi], [[256, 288 + 96, 128], true, 288], "the far φ stays the old engine's");
  const churn = sceneDocument(getSceneDefinition("churning-tank"));
  assert.equal(churn.fluid.brickTiers?.openTopHeadroom, 96);
  assert.equal(brickSceneSetup(churn, { timeStep: "sixtieth" }).setup?.lattice[1], 64 + 96);
});

test("brick headroom: validation takes whole pages only", () => {
  const tank = sceneDocument(getSceneDefinition("bulk-rest-tank"));
  for (const [v, ok] of [[32, true], [96, true], [0, false], [40, false], [16, false], [32.5, false]] as const) {
    const s = structuredClone(tank); s.fluid.brickTiers = { openTopHeadroom: v };
    assert.equal(validateScene(s).some(e => e.includes("openTopHeadroom")), !ok, `${v}`);
  }
});
