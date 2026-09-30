import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition, defaultScenePresetId } from "../lib/core/scenes";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { initialHeightFieldFractionAtCell } from "../lib/core/initial-height-field";
import { initialLiquidFractionAtCell } from "../lib/core/initial-fluid";
import { parseQueryState } from "../lib/core/url-state";

const dims = [48, 32, 40] as const;

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
