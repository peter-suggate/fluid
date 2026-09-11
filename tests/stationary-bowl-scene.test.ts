import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition, defaultScenePresetId } from "../lib/core/scenes";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { initialHeightFieldFractionAtCell } from "../lib/core/initial-height-field";
import { initialLiquidFractionAtCell } from "../lib/core/initial-fluid";
import { parseQueryState } from "../lib/core/url-state";
import { initializeSparseBrickAtlasFromScene } from "../lib/methods/adaptive-volume/sparse-brick-atlas";

const dims = [48, 32, 40] as const;
test("stationary bowl is a selectable scene with its own still-water profile", () => {
  const definition = getSceneDefinition("stationary-bowl"), scene = sceneDocument(definition);
  assert.notEqual(defaultScenePresetId, "stationary-bowl", "adding a study must not change the startup scene");
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid.initialHeightField, scene.fluid.initialHeightField);
  const query = parseQueryState("?scene=stationary-bowl");
  assert.equal(query.methodId, "adaptive-volume");
  assert.deepEqual(query.scene.fluid.initialHeightField, scene.fluid.initialHeightField);
  assert.deepEqual(scene.fluid.gravity_m_s2, { x: 0, y: 0, z: 0 });
  assert.equal(definition.methodProfile?.overrides?.gammaDiffusion, "on");
  assert.equal(definition.methodProfile?.overrides?.surfaceSharpening, "on");
  const changed = { ...scene.fluid.initialHeightField!, baseHeight_m: .9 };
  const edited = parseQueryState(`?scene=stationary-bowl&scene.fluid.initialHeightField=${encodeURIComponent(JSON.stringify(changed))}`);
  assert.deepEqual(edited.scene.fluid.initialHeightField, changed);
  assert.equal(scene.fluid.initialHeightField!.kind, "quadratic");
  if (scene.fluid.initialHeightField!.kind === "quadratic") scene.fluid.initialHeightField!.curvatureX_mInv = -1;
  assert.ok(validateScene(scene).some(e => e.includes("height field")));
});

test("bowl source agrees with the original diagnostic volume quadrature", () => {
  const scene = sceneDocument(getSceneDefinition("stationary-bowl"));
  for (const [x,z] of [[0,0],[8,8],[23,19],[31,24],[47,39]]) for (let y = 0; y < 32; y++) {
    let expected = 0;
    for (let iz = 0; iz < 8; iz++) for (let ix = 0; ix < 8; ix++) {
      const top = 17.3 + .003 * ((x! + (ix + .5) / 8 - 24) ** 2 + .7 * (z! + (iz + .5) / 8 - 20) ** 2);
      expected += Math.max(0, Math.min(1, top-y)) / 64;
    }
    assert.ok(Math.abs(initialHeightFieldFractionAtCell(scene,x!,y,z!,dims)! - expected) < 1e-12);
    assert.ok(Math.abs(initialLiquidFractionAtCell(scene,x!,y,z!,dims,false) - expected) < 1e-12);
  }
});

test("ordinary coarse atlas initialization retains the complete curved bowl volume", () => {
  const scene = sceneDocument(getSceneDefinition("stationary-bowl"));
  const atlas = initializeSparseBrickAtlasFromScene(scene, { finestDimensions: dims,
    brickFineResolution: 8, maximumMacroSpanBricks: 1, coarseFirstCurvatureTolerance: .25 });
  let actual = 0;
  for (const brick of atlas.bricks) {
    assert.equal(brick.resolution, 2, "scene enforces width 4");
    actual += brick.density.reduce((sum,rho) => sum+rho,0) * 4**3;
  }
  // Exact area moment of the original midpoint quadrature, integrated in y.
  const expected = 48*40*(17.3+.003*((48**2/12-1/768)+.7*(40**2/12-1/768)));
  assert.ok(Math.abs(actual-expected)<1e-7, `seed volume ${actual} != ${expected}`);
});
