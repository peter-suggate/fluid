import assert from "node:assert/strict";
import test from "node:test";
import { sparseCM12PresentationColumnHeightMode } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

test("presentation column mode defaults to adaptive auto", () => {
  assert.equal(sparseCM12PresentationColumnHeightMode({}), "auto");
});

test("legacy column-height booleans remain compatible", () => {
  assert.equal(sparseCM12PresentationColumnHeightMode({
    presentationColumnHeightEnabled: true,
  }), "on");
  assert.equal(sparseCM12PresentationColumnHeightMode({
    presentationColumnHeightEnabled: false,
  }), "off");
});

test("an explicit mode takes precedence over the legacy boolean", () => {
  assert.equal(sparseCM12PresentationColumnHeightMode({
    presentationColumnHeightMode: "auto",
    presentationColumnHeightEnabled: false,
  }), "auto");
  assert.equal(sparseCM12PresentationColumnHeightMode({
    presentationColumnHeightMode: "off",
    presentationColumnHeightEnabled: true,
  }), "off");
});
