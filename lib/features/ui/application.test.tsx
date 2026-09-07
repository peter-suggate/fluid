import "../../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { applicationViews, FeatureSlot } from "./FeatureSlot";
import { composeFeatureUI } from "./composition";

test("every advertised application placement has a registered implementation", () => {
  for (const method of ["adaptive-mass", "uniform"]) {
    for (const fluid of [true, false]) {
      for (const placement of composeFeatureUI(method, fluid).placements) {
        assert.equal(typeof applicationViews[`${placement.feature}/${placement.control}`], "function",
          `${placement.feature}/${placement.control} in ${placement.slot}`);
      }
    }
  }
});

test("installed scene and panel slots render from the active session", () => {
  for (const slot of ["scene.visibility", "scene.physics", "scene.adaptivity", "fluid.material", "sim.adaptivity", "frame.options", "frame.lighting", "frame.reconstruction", "frame.reconstruction"]) {
    assert.doesNotThrow(() => renderToStaticMarkup(createElement(FeatureSlot, { slot })), slot);
  }
});
