import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SVO_GBUFFER_FLAGS,
  SVO_GBUFFER_PRODUCERS,
  SVO_GBUFFER_PRODUCER_MASK,
  SVO_GBUFFER_PRODUCER_SHIFT,
  svoGBufferProducerFlags,
  svoGBufferProducerOf,
} from "../lib/svo-gbuffer";
import {
  DEFAULT_SVO_RENDER_DIAGNOSTICS,
  SVO_RENDER_STAGE_CLAIMANT_LEGEND,
  SVO_RENDER_STAGE_DEFINITIONS,
  SVO_RENDER_STAGE_GROUPS,
  SVO_RENDER_STAGE_SEQUENTIAL_LEGEND,
  SVO_RENDER_STAGE_VIEWS,
  normalizeSvoRenderDiagnostics,
  svoRenderStageCode,
  svoRenderStageUsesLightSlot,
} from "../lib/svo-render-diagnostics";
import { useUIStore } from "../lib/stores/ui-store";
import {
  SVO_RENDER_STAGE_OVERLAY_CONTRACT,
  createSvoRenderStageOverlayWGSL,
  svoRenderStageOverlayBindGroupLayoutEntries,
} from "../lib/webgpu-svo-stage-overlay";
import { createSvoDrySceneFragmentWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { svoRigidRasterShader } from "../lib/webgpu-svo-rigid-raster";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
const overlayShader = createSvoRenderStageOverlayWGSL();

if (typeof globalThis.GPUShaderStage === "undefined") {
  Object.defineProperty(globalThis, "GPUShaderStage", {
    configurable: true, writable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  });
}

test("stage-view controls are bounded and retain a composited-frame default", () => {
  assert.deepEqual(DEFAULT_SVO_RENDER_DIAGNOSTICS, {
    stageView: "off", lightSlot: 0, maximumTraversalDepth: 21, maximumNodeVisits: 256,
  });
  assert.equal(svoRenderStageCode("off"), 0);
  assert.equal(svoRenderStageCode("pass-claimant"), 1);
  assert.deepEqual(normalizeSvoRenderDiagnostics({
    stageView: "owner-identity", lightSlot: 99, maximumTraversalDepth: 99, maximumNodeVisits: 0,
  }), { stageView: "owner-identity", lightSlot: 7, maximumTraversalDepth: 21, maximumNodeVisits: 1 });
  assert.ok(svoRenderStageUsesLightSlot("cone-light-visibility"));
  assert.ok(!svoRenderStageUsesLightSlot("cone-ambient-visibility"));

  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);
  useUIStore.getState().setSvoStageView("pass-claimant");
  useUIStore.getState().setSvoStageLightSlot(42);
  useUIStore.getState().setSvoMaximumTraversalDepth(0);
  useUIStore.getState().setSvoMaximumNodeVisits(999);
  assert.deepEqual({
    view: useUIStore.getState().svoStageView,
    slot: useUIStore.getState().svoStageLightSlot,
    depth: useUIStore.getState().svoMaximumTraversalDepth,
    visits: useUIStore.getState().svoMaximumNodeVisits,
  }, { view: "pass-claimant", slot: 7, depth: 1, visits: 256 });
  useUIStore.setState(initial, true);
});

test("every stage view names the plane it decodes and covers the frame graph", () => {
  assert.equal(new Set(SVO_RENDER_STAGE_VIEWS).size, SVO_RENDER_STAGE_VIEWS.length);
  assert.deepEqual(Object.keys(SVO_RENDER_STAGE_DEFINITIONS), [...SVO_RENDER_STAGE_VIEWS]);
  for (const stageView of SVO_RENDER_STAGE_VIEWS) {
    const definition = SVO_RENDER_STAGE_DEFINITIONS[stageView];
    assert.equal(definition.view, stageView);
    assert.ok(definition.label.length > 0);
    assert.ok(definition.description.length > 20, `${stageView} needs an explanatory sentence`);
    assert.ok(definition.plane.length > 0, `${stageView} must name the plane it reads`);
    assert.ok(definition.legend.length >= 3, `${stageView} needs a readable legend`);
    assert.ok(SVO_RENDER_STAGE_GROUPS.includes(definition.group));
  }
  // Every group except the presentation passthrough carries real views.
  for (const group of SVO_RENDER_STAGE_GROUPS) {
    const members = SVO_RENDER_STAGE_VIEWS.filter((stageView) => SVO_RENDER_STAGE_DEFINITIONS[stageView].group === group);
    assert.ok(members.length > 0, `${group} must contribute at least one view`);
  }
  assert.deepEqual(SVO_RENDER_STAGE_SEQUENTIAL_LEGEND.map(({ color }) => color), [
    "#08051f", "#2447ff", "#00d9ff", "#00ff85", "#eaff00", "#ff8500", "#ff174d", "#ffffff",
  ]);
});

test("the producing-pass tag occupies spare flag bits and never collides", () => {
  const allocated = Object.values(SVO_GBUFFER_FLAGS);
  const tagMask = SVO_GBUFFER_PRODUCER_MASK << SVO_GBUFFER_PRODUCER_SHIFT;
  for (const flag of allocated) {
    assert.equal(flag & tagMask, 0, `flag ${flag} overlaps the producing-pass tag`);
  }
  // The tag has to fit the sixteen-bit flags field the metadata word packs.
  assert.ok(tagMask <= 0xffff);
  for (const producer of Object.values(SVO_GBUFFER_PRODUCERS)) {
    assert.equal(svoGBufferProducerOf(svoGBufferProducerFlags(producer)), producer);
  }
  assert.equal(svoGBufferProducerOf(SVO_GBUFFER_FLAGS.validSurface | SVO_GBUFFER_FLAGS.hardFeature),
    SVO_GBUFFER_PRODUCERS.unspecified);
});

test("each primary pass tags itself so the claimant view is exact", () => {
  const raster = createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, true, true);
  // Terrain, brick and primitive fragments share one surface writer, so each
  // names its producer at its own call and the writer turns it into flags.
  assert.match(raster, /fn dryRasterPrimarySurface\(opaque:DryHit,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32\)/);
  assert.match(raster, /svoGBufferProducerFlags\(producer\)/);
  assert.match(raster, /traceTerrain[^]*dryRasterPrimarySurface\(terrain,camera\[0\],rd,camera\[1\],SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND\)/);
  assert.match(raster, /dryRasterPrimarySurface\(payload,ro,rd,camera\[1\],SVO_GBUFFER_PRODUCER_BRICK\)/);
  assert.match(raster, /dryRasterPrimarySurface\(exact,ro,rd,camera\[1\],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE\)/);
  assert.match(raster, /svoGBufferProducerFlags\(SVO_GBUFFER_PRODUCER_TRACED\)/);
  // A pixel no pass claimed is a miss, and the background pass owns that too.
  assert.match(raster, /fn dryRasterPrimaryMiss\(\)[^]*svoGBufferProducerFlags\(SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND\)/);
  assert.match(svoRigidRasterShader, /svoGBufferProducerFlags\(SVO_GBUFFER_PRODUCER_RIGID\)/);
  assert.match(svoDrySceneShader, /svoGBufferProducerFlags\(SVO_GBUFFER_PRODUCER_GLASS\)/);
  // The claimant legend is indexed positionally by the overlay shader.
  assert.deepEqual(SVO_RENDER_STAGE_CLAIMANT_LEGEND.map(({ label }) => label), [
    "Sky / miss", "Terrain", "Brick raster", "Scene primitive",
    "Rigid impostor", "Glass discovery", "Traced primary", "Untagged",
  ]);
});

test("the dry shader carries no per-pixel cost counters at any scale", () => {
  const counters = /dryCostOverlay|dryCostRamp|dryNormalizedCost|dryPrimaryNodeVisits|dryPrimaryLeafVisits|dryPrimaryEmptyBrickSkips|dryPrimaryVoxelWorkItems|dryPrimaryExactTests|dryPrimaryMaximumDepth|dryShadowNodeVisits|dryShadowLeafVisits|dryShadowWorkItems|dryMipSteps|dryTraversalFailure/;
  for (const scale of [1, 0.5, 0.25] as const) {
    assert.doesNotMatch(createSvoDrySceneFragmentWGSL(scale), counters,
      `scale ${scale} must not keep an invocation-private tally nothing reads`);
  }
  assert.doesNotMatch(createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, true, true), counters);
  // The traversal-limit controls are a separate knob and stay.
  assert.match(svoDrySceneShader, /fn dryDiagnosticMaximumNodeVisits\(\)->u32/);
  assert.match(svoDrySceneShader, /fn dryDiagnosticMaximumDepth\(\)->u32/);
});

test("the overlay reads published planes and never writes one", () => {
  const bindings = svoRenderStageOverlayBindGroupLayoutEntries();
  assert.equal(bindings.length, Object.keys(SVO_RENDER_STAGE_OVERLAY_CONTRACT.bindings).length);
  // Sampled textures per fragment stage default to sixteen; the plane set has
  // to stay under it or the pass cannot be created on a baseline device.
  const sampled = bindings.filter((entry) => entry.texture);
  assert.ok(sampled.length <= 16, "stage planes must fit the baseline sampled-texture limit");
  for (const entry of bindings) {
    assert.equal(entry.visibility, GPUShaderStage.FRAGMENT);
    assert.equal(entry.storageTexture, undefined, "the overlay must not bind a writable plane");
    if (entry.buffer) assert.equal(entry.buffer.type, "uniform");
  }
  assert.doesNotMatch(overlayShader, /textureStore/, "the overlay is strictly a reader");
  for (const stageView of SVO_RENDER_STAGE_VIEWS) {
    if (stageView === "off") continue;
    assert.match(overlayShader, new RegExp(`mode==${svoRenderStageCode(stageView)}u`),
      `${stageView} must be decoded by the overlay shader`);
  }
  // The palettes are generated from the published legends, so a swatch in the
  // panel and the pixels it names cannot drift apart.
  const brick = SVO_RENDER_STAGE_CLAIMANT_LEGEND[2].color;
  const channel = (index: number) => (Number.parseInt(brick.slice(1 + index * 2, 3 + index * 2), 16) / 255).toFixed(4);
  assert.ok(overlayShader.includes(`vec3f(${channel(0)},${channel(1)},${channel(2)})`),
    "claimant hues must be emitted from the legend, not re-authored in WGSL");
});

test("the renderer encodes the overlay after the composite and only on request", () => {
  assert.match(rendererSource, /if \(stageViewActive\) requested\.push\("svo-stage-overlay"\)/);
  const drawStart = rendererSource.indexOf("const rasterResult = this.waterPipeline.encode(");
  const overlayAt = rendererSource.indexOf("this.svoStageOverlay.encode(", drawStart);
  assert.ok(drawStart > 0 && overlayAt > drawStart,
    "the stage overlay must be encoded after the optical composite it replaces");
  assert.match(rendererSource, /stageView: ui\.svoStageView|activeSvoDiagnostics\.stageView !== "off"/);
  assert.doesNotMatch(rendererSource, /svoCostOverlayCode/);
});

test("render panel and viewport present stage views with an in-scene legend", () => {
  assert.match(panelSource, /Render pipeline stages/);
  assert.match(panelSource, /WHAT EACH PASS ACTUALLY WROTE/);
  assert.match(panelSource, /SVO_RENDER_STAGE_DEFINITIONS\[candidate\]/);
  assert.match(panelSource, /view === svoStageView && view !== "off" \? "off" : view/);
  assert.match(panelSource, /Maximum traversal depth/);
  assert.match(panelSource, /Maximum node visits/);
  assert.match(viewportSource, /data-testid="svo-stage-legend"/);
  assert.match(viewportSource, /SVO_RENDER_STAGE_DEFINITIONS\[svoStageView\]/);
  assert.match(viewportSource, /stageView: ui\.svoStageView/);
  assert.match(viewportSource, /lightSlot: ui\.svoStageLightSlot/);
});

test("render panel contains rendering controls only; solver fields stay in performance", () => {
  assert.doesNotMatch(panelSource, /useMethodStore|gridOverlay|paperPipeline|finePublicationGate/);
  assert.doesNotMatch(panelSource, />Solver grid<|Paper pipeline inspector|CFL load|Projected divergence/);
  assert.match(panelSource, /RENDER OBSERVATORY/);
  assert.match(panelSource, /GLOBAL SVO VIEW/);
});
