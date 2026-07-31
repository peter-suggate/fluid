import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  createSvoDrySceneFragmentWGSL,
  SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL,
  SVO_DRY_SPLIT_GEOMETRY_FORMAT,
  SVO_DRY_SPLIT_IDENTITY_FORMAT,
  SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL,
  SVO_DRY_RASTER_RIGID_BODY_THRESHOLD,
  svoDryRasterGlassRecordRange,
  svoDryRasterGlassShader,
  svoDryPrimaryCoherenceDecision,
  svoDryRigidPrimaryStrategy,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_THIN_GLASS_RECORD_WORDS } from "../lib/svo-thin-glass";

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
  assert.match(visibility, /let glass=traceGlass\([^]*let glassKey=select\(0u,glass\.recordIndex\+1u,glassVisible\)/,
    "primary visibility must retain the exact winning pane rather than discarding its search result");
  assert.doesNotMatch(lighting, /traceGlass\(/,
    "deferred lighting must not repeat the full pane search");
  assert.match(lighting, /let glassKey=\(packedOpaqueMaterial>>16u\)&0x1ffu[^]*svoThinGlassIntersect\(record,ro,rd,0\.0,opaque\.t/,
    "only a visible pane performs one indexed exact intersection in deferred lighting");
  assert.equal(SVO_DRY_SPLIT_GEOMETRY_FORMAT, "rgba32float");
  assert.equal(SVO_DRY_SPLIT_IDENTITY_FORMAT, "rg32uint");
  assert.equal(SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL, 48,
    "24 bytes are stored then read across the split boundary");
  assert.equal(SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL, 24);
});

test("raster glass discovery is an exact split-only layered hit buffer", () => {
  assert.throws(
    () => createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "inline", 0, false, true),
    /requires split shading/,
  );
  const split = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split", 0, false, true);
  const visibility = split.slice(split.indexOf("@fragment fn dryVisibilityMain"), split.indexOf("@fragment fn dryLightingMain"));
  const lighting = split.slice(split.indexOf("@fragment fn dryLightingMain"));
  assert.doesNotMatch(visibility, /traceGlass\(/,
    "coverage-scaled discovery must remove the all-panes search from every primary pixel");
  assert.match(split, /@binding\(6\) var drySplitGlassKeyRead:texture_2d<u32>/);
  assert.match(lighting, /let glassKey=textureLoad\(drySplitGlassKeyRead,coordinate,0\)\.x/);
  assert.match(lighting, /svoThinGlassIntersect\(record,ro,rd,0\.0,opaque\.t/,
    "the winning raster candidate must still receive the canonical analytic intersection");
  assert.match(svoDryRasterGlassShader, /@builtin\(instance_index\) recordIndex:u32/);
  assert.match(svoDryRasterGlassShader, /svoThinGlassIntersect\(record,ray\[0\],ray\[1\],0\.0,opaqueDepth/);
  assert.match(svoDryRasterGlassShader, /@location\(0\) glassKey:u32/);
  assert.doesNotMatch(svoDryRasterGlassShader, /SvoGBufferTargets|packedSurface|identityMedia/,
    "glass discovery must not pay to rewrite the opaque G-buffer beneath transmission");
});

test("raster rigid discovery removes bodies only from split primary visibility", () => {
  assert.throws(
    () => createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "inline", 0, false, false, true),
    /Raster rigid discovery requires split shading/,
  );
  const split = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split", 0, false, true, true);
  const visibility = split.slice(split.indexOf("@fragment fn dryVisibilityMain"), split.indexOf("@fragment fn dryLightingMain"));
  assert.match(visibility, /let opaque=traceStaticSolidScene\(ro,rd\)/);
  assert.doesNotMatch(visibility, /let opaque=traceOpaqueScene\(ro,rd\)/,
    "fullscreen primary visibility must not retain the rigid loop");
  assert.match(split, /dryPrepassBoundaryMain[^]*traceOpaqueScene\(ray\[0\],ray\[1\]\)/,
    "uncertain reduced-rate boundary pixels keep the complete current-frame scene fallback");
});

test("rigid primary strategy switches precompiled paths at the measured body-count crossover", () => {
  assert.equal(SVO_DRY_RASTER_RIGID_BODY_THRESHOLD, 4);
  assert.equal(svoDryRigidPrimaryStrategy(1, true), "analytic");
  assert.equal(svoDryRigidPrimaryStrategy(3, true), "analytic");
  assert.equal(svoDryRigidPrimaryStrategy(4, true), "raster");
  assert.equal(svoDryRigidPrimaryStrategy(12, true), "raster");
  assert.equal(svoDryRigidPrimaryStrategy(12, false), "analytic");
});

test("raster glass draw skips only a contiguous compositor-owned prefix", () => {
  const records = (paneIds: readonly number[]): Uint32Array => {
    const packed = new Uint32Array(paneIds.length * SVO_THIN_GLASS_RECORD_WORDS);
    paneIds.forEach((paneId, index) => { packed[index * SVO_THIN_GLASS_RECORD_WORDS + 16] = paneId; });
    return packed;
  };
  assert.deepEqual(svoDryRasterGlassRecordRange(records([0x1000, 0x1001, 7, 8]), 0x1000, 2),
    { firstRecord: 2, recordCount: 2 });
  assert.deepEqual(svoDryRasterGlassRecordRange(records([7, 8]), 0x1000, 2),
    { firstRecord: 0, recordCount: 2 });
  assert.deepEqual(svoDryRasterGlassRecordRange(records([0x1000, 7, 0x1001]), 0x1000, 2),
    { firstRecord: 0, recordCount: 3 }, "interleaved ownership must fail closed to the complete draw");
  assert.deepEqual(svoDryRasterGlassRecordRange(new Uint32Array(SVO_THIN_GLASS_RECORD_WORDS - 1), 0x1000, 2),
    { firstRecord: 0, recordCount: 0 }, "partial records must never reach an instanced draw");
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
  assert.match(split, /@group\(1\) @binding\(0\) var dryPrepassVisibilityKeyTexture/);
  assert.match(split, /@group\(2\) @binding\(0\) var drySplitGeometryWrite/);
  assert.match(split, /dryPrepassResolve\(input\.position\.xy,opaque\.t,opaque\.normal,opaque\)/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("inline, split, and raster-glass dry-scene shaders compile on the target backend", {
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
      for (const rasterGlass of [false, true]) {
        for (const rasterRigid of [false, true]) {
          const module = device.createShaderModule({
            label: `Split dry shader validation x${scale}, raster glass ${rasterGlass}, raster rigid ${rasterRigid}`,
            code: createSvoDrySceneFragmentWGSL(scale, "hybrid", "off", "split", 0, false, rasterGlass, rasterRigid),
          });
          const info = await module.getCompilationInfo();
          assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
        }
      }
    }
    const rasterModule = device.createShaderModule({ label: "Raster glass shader validation", code: svoDryRasterGlassShader });
    const rasterInfo = await rasterModule.getCompilationInfo();
    assert.deepEqual(rasterInfo.messages.filter(({ type }) => type === "error"), []);
  } finally {
    device.destroy();
  }
});
