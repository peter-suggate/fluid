import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  createSvoDrySceneFragmentWGSL,
  SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL,
  SVO_DRY_SPLIT_GEOMETRY_FORMAT,
  SVO_DRY_SPLIT_IDENTITY_FORMAT,
  SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL,
  svoDryPrimaryCoherenceDecision,
} from "../lib/webgpu-svo-dry-scene";

test("split dry rendering keeps inline as the exact default and adds two isolated entry points", () => {
  const inline = createSvoDrySceneFragmentWGSL(1, "hybrid", "off");
  const split = createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "split");
  assert.doesNotMatch(inline, /dryVisibilityMain|dryLightingMain|drySplitGeometryWrite/);
  assert.match(split, /@fragment fn dryVisibilityMain/);
  assert.match(split, /@fragment fn dryLightingMain/);
  assert.match(split, /texture_storage_2d<rgba32float,write>/);
  assert.match(split, /textureStore\(drySplitGeometryWrite/);
  const visibility = split.slice(split.indexOf("@fragment fn dryVisibilityMain"), split.indexOf("@fragment fn dryLightingMain"));
  assert.doesNotMatch(visibility, /shadeDryOpaque|dryLightVisibility|dryContactVisibility/,
    "visibility must not carry lighting, shadow, or cone work");
  const lighting = split.slice(split.indexOf("@fragment fn dryLightingMain"));
  assert.doesNotMatch(lighting, /traceOpaqueScene\(/,
    "lighting must never repeat primary SVO traversal, including behind glass");
  assert.match(lighting, /let glass=traceGlass\([^]*var color=shadeDryOpaque/,
    "cheap analytic glass keeps the inline evaluation order");
  assert.equal(SVO_DRY_SPLIT_GEOMETRY_FORMAT, "rgba32float");
  assert.equal(SVO_DRY_SPLIT_IDENTITY_FORMAT, "rg32uint");
  assert.equal(SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL, 48,
    "24 bytes are stored then read across the split boundary");
  assert.equal(SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL, 24);
});

test("static-primary coherence is exact-keyed and fails closed", () => {
  assert.equal(svoDryPrimaryCoherenceDecision("off", true, "frame", "frame"), "trace");
  assert.equal(svoDryPrimaryCoherenceDecision("static-primary", false, "frame", "frame"), "trace");
  assert.equal(svoDryPrimaryCoherenceDecision("static-primary", true, undefined, undefined), "trace");
  assert.equal(svoDryPrimaryCoherenceDecision("static-primary", true, "next", "old"), "trace");
  assert.equal(svoDryPrimaryCoherenceDecision("static-primary", true, "frame", "frame"), "reuse");
});

test("reduced cone prepass reserves group one and moves the split bridge to group two", () => {
  const split = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split");
  assert.match(split, /@group\(1\) @binding\(0\) var dryPrepassVisibilityTexture0/);
  assert.match(split, /@group\(2\) @binding\(0\) var drySplitGeometryWrite/);
  assert.match(split, /dryPrepassResolve\(input\.position\.xy,opaque\.t,opaque\.normal\)/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("inline and split dry-scene shaders compile on the target backend", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU split-render checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    for (const scale of [1, 0.5] as const) {
      const module = device.createShaderModule({
        label: `Split dry shader validation x${scale}`,
        code: createSvoDrySceneFragmentWGSL(scale, "hybrid", "off", "split"),
      });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    }
  } finally {
    device.destroy();
  }
});
