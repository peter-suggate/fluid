import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_SVO_RENDER_TUNING,
} from "../lib/svo-render-tuning";
import {
  measureRenderPipelineNode,
  RENDER_PIPELINE_NODES,
  renderPipelinePhaseDurations,
  type RenderPipelineContext,
} from "../lib/render-pipeline-graph";
import {
  disabledRenderStagesFrom,
  disabledRenderStagesKey,
  NO_DISABLED_RENDER_STAGES,
  RENDER_STAGE_SWITCH_IDS,
  type RenderStageSwitchId,
} from "../lib/render-stage-switches";
import type { PerformanceTrace } from "../lib/performance-trace";

const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const renderWorkerSource = readFileSync(new URL("../lib/webgpu-render-worker.ts", import.meta.url), "utf8");
const renderWorkerClientSource = readFileSync(new URL("../lib/webgpu-render-worker-client.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
const pipelineSource = readFileSync(new URL("../components/RenderPipeline.tsx", import.meta.url), "utf8");

const context = (disabled: readonly RenderStageSwitchId[] = []): RenderPipelineContext => ({
  disabledStages: disabledRenderStagesFrom(disabled),
  coneTracingMode: "cones",
  shadowsEnabled: true,
  ambientOcclusionEnabled: true,
  seamClosureEnabled: true,
  globalIlluminationEnabled: true,
  tuning: DEFAULT_SVO_RENDER_TUNING,
  sceneHasFluid: true,
  refinementDepth: 0,
  leafVoxel_mm: 25,
  rendererActive: true,
  stageView: "off",
  rasterPrimaryActive: true,
});

test("every node in the frame graph has a switch that reaches the encoder", () => {
  for (const node of RENDER_PIPELINE_NODES) {
    assert.equal(node.toggleable, true, `${node.id} must be switchable`);
    // Exactly one mechanism per node. A stage id is an encode-time ablation; the
    // four nodes without one are switched by a contract the shaders compile
    // against, and having both would be two sources of truth for one bit.
    assert.equal(Boolean(node.stage) !== Boolean(node.switchedBy), true,
      `${node.id} must declare exactly one of stage / switchedBy`);
    if (node.stage) {
      assert.equal(node.stage, node.id,
        "a stage switch is named for its node, so a lamp and a gate are the same name");
      assert.ok(RENDER_STAGE_SWITCH_IDS.includes(node.stage));
    }
  }
  // The four exceptions, named, so adding a fifth is a deliberate act.
  assert.deepEqual(
    RENDER_PIPELINE_NODES.filter((node) => !node.stage).map((node) => node.id),
    ["stationary-reuse", "seam-closure", "cone-visibility", "gi-composition"]);
});

test("every stage switch is honoured at an encode site", () => {
  // Where each gate lives. A switch the panel offers and no encoder reads would
  // report a saving the frame never made.
  const sites: Record<RenderStageSwitchId, string> = {
    "sparse-world-build": rendererSource,
    "primary-traversal": drySceneSource,
    "thin-glass": drySceneSource,
    "scene-primitive": drySceneSource,
    "rigid-impostor": drySceneSource,
    "voxel-light-cache": drySceneSource,
    "world-gi-cache": drySceneSource,
    "reduced-shade": drySceneSource,
    "deferred-lighting": drySceneSource,
    "water-interfaces": waterSource,
    caustics: waterSource,
    "optical-composite": waterSource,
    "inspection-overlays": rendererSource,
    present: rendererSource,
  };
  for (const stage of RENDER_STAGE_SWITCH_IDS) {
    assert.match(sites[stage], new RegExp(`disabledStages\\.has\\("${stage}"\\)`),
      `${stage} must be read where the frame is encoded`);
  }
  // The set reaches both presentation pipelines through their own channel, and
  // never through the one that rebuilds shaders.
  assert.match(rendererSource, /this\.svoDryScenePipeline\?\.setDisabledStages\(disabledStages\)/);
  assert.match(rendererSource, /this\.waterPipeline\.setDisabledStages\(disabledStages\)/);
  const lightingSetterStart = drySceneSource.indexOf("setLightingOptions(options: SparseVoxelDrySceneLightingOptions)");
  assert.ok(lightingSetterStart > 0);
  const lightingSetter = drySceneSource.slice(
    lightingSetterStart, drySceneSource.indexOf("get silhouetteRefinementStatus", lightingSetterStart));
  assert.doesNotMatch(lightingSetter, /disabledStages/,
    "an ablation must not reach the code that recompiles bundles and rewrites params");
});

test("a withheld stage invalidates the frames traced under the previous set", () => {
  // Reuse hands back a G-buffer traced before the switch moved, and the
  // world-GI cache keeps entries until something explicitly clears them.
  assert.match(drySceneSource, /setDisabledStages\(disabled: DisabledRenderStages\): void \{[\s\S]*?this\.clearReusableFrame\(\);/);
  assert.match(drySceneSource, /setDisabledStages\(disabled: DisabledRenderStages\): void \{[\s\S]*?this\.primaryVisibilityCacheKey = undefined;/);
  assert.match(drySceneSource, /if \(staleWorldGi\) this\.worldGiCacheDirty = true;/);
  // Averaging is per trace context, so the two pipelines must not pool.
  assert.match(rendererSource, /without-\$\{disabledRenderStagesKey\(disabledStages\)/);
  assert.equal(disabledRenderStagesKey(NO_DISABLED_RENDER_STAGES), "");
  // Canonical order, so one set has one spelling however it was clicked.
  assert.equal(
    disabledRenderStagesKey(disabledRenderStagesFrom(["present", "caustics"])),
    disabledRenderStagesKey(disabledRenderStagesFrom(["caustics", "present"])));
});

test("a withheld stage keeps the clears its consumers read", () => {
  // The difference between an ablation and a bug: the attachment is defined
  // afterwards, so the image degrades in a stated way instead of presenting
  // whatever the last frame that did run happened to leave behind.
  assert.match(drySceneSource, /label: "Sparse voxel primary visibility · withheld"[\s\S]{0,600}depthLoadOp: "clear"/);
  assert.match(drySceneSource, /if \(!this\.disabledStages\.has\("deferred-lighting"\)\) \{[\s\S]{0,900}lighting\.draw\(3\);\s*\n\s*\}\s*\n\s*lighting\.end\(\)/);
  assert.match(waterSource, /if \(!this\.disabledStages\.has\("optical-composite"\)\) \{[\s\S]{0,200}composite\.end\(\)/);
  // Present acquires and clears the swap chain either way. A frame that never
  // wrote it would leave an older one on screen, which reads as "free".
  assert.match(rendererSource, /getCurrentTexture\(\)[\s\S]{0,240}if \(!disabledStages\.has\("present"\)\)/);
  // Re-enabling has to redo what was retained precisely because nothing
  // invalidated it.
  assert.match(waterSource, /setDisabledStages\(disabled: DisabledRenderStages\) \{[\s\S]*?this\.dryInterfaceClearsEncoded = false;[\s\S]*?this\.causticsValid = false;/);
});

test("the panel routes every lamp, and states what is withheld", () => {
  assert.match(panelSource, /if \(node\?\.stage\) \{\s*\n\s*setRenderStageDisabled\(node\.stage, !disabledStages\.has\(node\.stage\)\);/,
    "a node with a stage must switch that stage and nothing else");
  for (const [id, flag] of [
    ["stationary-reuse", /updateTuning\("stationaryPrimaryReuseEnabled"/],
    ["seam-closure", /setSilhouetteRefinementEnabled\(!silhouetteRefinementEnabled\)/],
    ["cone-visibility", /setSvoConeTracingMode\(/],
    ["gi-composition", /setSvoGlobalIlluminationEnabled\(!svoGlobalIlluminationEnabled\)/],
  ] as const) {
    assert.match(panelSource, new RegExp(`id === "${id}"\\)[^\\n]*`), `${id} must be routed`);
    assert.match(panelSource, flag, `${id} must move its own contract`);
  }
  // Withheld stages are invisible from the viewport — the image just looks
  // wrong — so the count is stated and is itself the way back.
  assert.match(panelSource, /data-testid="withheld-stage-count"/);
  assert.match(panelSource, /setRenderStageDisabled\(stage, false\)/,
    "the count must restore every withheld stage");
});

test("every node can report a live cost, and a withheld pass reads as zero", () => {
  // No node may be structurally incapable of a number: it either owns trace
  // phases, or is a term inside a pass that does, or provably spends no time.
  for (const node of RENDER_PIPELINE_NODES) {
    assert.ok(node.phaseLabels.length > 0 || node.costInsideNode || node.spendsNoFrameTime,
      `${node.id} would show an em dash forever`);
  }
  const trace = {
    sampleId: 1, domain: "gpu", lane: "presentation", context: "test", capturedAt_ms: 0,
    total_ms: 20,
    phases: [
      { id: "dry-scene", label: "SVO deferred dry lighting", duration_ms: 8 },
      { id: "svo-primary", label: "SVO primary visibility", duration_ms: 12 },
    ],
  } as unknown as PerformanceTrace;
  const durations = renderPipelinePhaseDurations(trace);
  const node = (id: string) => RENDER_PIPELINE_NODES.find((candidate) => candidate.id === id)!;

  const primary = measureRenderPipelineNode(node("primary-traversal"), durations, 20, "on");
  assert.equal(primary.kind, "measured");
  assert.equal(primary.duration_ms, 12);
  assert.equal(primary.share, 0.6);

  // The distinction the switch exists for: a pass the encoder skipped is a
  // measured zero, and must never print the same em dash as a missing trace.
  const withheld = measureRenderPipelineNode(node("caustics"), durations, 20, "off");
  assert.equal(withheld.kind, "withheld");
  assert.equal(withheld.duration_ms, 0);
  assert.equal(measureRenderPipelineNode(node("caustics"), new Map(), 0, "off").kind, "unmeasured",
    "without a trace there is nothing to call a zero");

  // GI composition is a branch inside the deferred draw, so it has no phase of
  // its own and reports the pass it runs inside rather than a dash.
  const shared = measureRenderPipelineNode(node("gi-composition"), durations, 20, "on");
  assert.equal(shared.kind, "shared");
  assert.equal(shared.insideNode, "deferred-lighting");
  assert.equal(shared.duration_ms, 8);

  const gate = measureRenderPipelineNode(node("stationary-reuse"), durations, 20, "armed");
  assert.equal(gate.kind, "structural");
  assert.equal(gate.duration_ms, 0);

  const gated = measureRenderPipelineNode(node("world-gi-cache"), durations, 20, "on");
  assert.equal(gated.kind, "idle");
  assert.equal(gated.duration_ms, 0,
    "a detailed trace makes a stage that encoded no pass a known zero, never a blank");
});

test("the render panel's timing mode reaches the worker-owned renderer", () => {
  assert.match(renderWorkerClientSource,
    /instrumentationMode: usePerformanceInstrumentationStore\.getState\(\)\.mode/,
    "the window must carry its live timing mode across the worker boundary");
  assert.match(renderWorkerSource,
    /setMode\(message\.instrumentationMode\)[\s\S]{0,300}runtime\.draw\(/,
    "the worker must apply the mode before draw decides whether to record timestamps");
  assert.doesNotMatch(rendererSource, /hardwarePresentationTraceInvalid/,
    "an early invalid sample must not permanently disable the later per-stage timestamps");
  assert.match(panelSource,
    /\.map\(\(report\) => report\.presentationStages\)/,
    "the panel must consume the renderer's per-pass hardware trace independently of the frame total");
  assert.match(rendererSource,
    /new GPUPassTimestampRecorder\([\s\S]{0,100}presentation pass timestamps/,
    "each real pass must carry its own beginning/end timestamp pair");
  assert.match(rendererSource,
    /const shouldTracePresentation = measurementInstrumentationEnabled\s*&& !this\.presentationTracePending;/,
    "the next frame must be sampled as soon as the one in-flight readback completes");
  assert.doesNotMatch(rendererSource, /lastPresentationTraceAt_ms/,
    "the old quarter-second presentation timing throttle must not leave dead timer state");
  assert.doesNotMatch(rendererSource, /presentationEncodingTrace/,
    "rp-trunk-cost must never substitute CPU encoding latency for GPU stage cost");
  assert.doesNotMatch(panelSource, /encodingMean|cpu-encode/,
    "the panel must show unavailable when GPU stage cost is unknowable, not a zero-like proxy");
});

test("the cost reads on the trunk, at the junction the node taps", () => {
  assert.match(pipelineSource, /className=\{`rp-trunk-cost is-\$\{cost\.kind\}`\}/,
    "the figure must carry its kind so a withheld zero cannot be read as a measurement");
  assert.match(pipelineSource, /x=\{side === "left" \? TRUNK_X \+ 6 : TRUNK_X - 6\}/,
    "the figure sits past the junction, on the side of the trunk the node is not on");
  assert.doesNotMatch(pipelineSource, /rp-node-cost/,
    "one place for the cost; a second copy in the card head is clutter that can disagree");
});

test("switching a node changes its state, and only its state", () => {
  for (const node of RENDER_PIPELINE_NODES) {
    if (!node.stage) continue;
    assert.equal(node.state(context()), node.id === "inspection-overlays" ? "armed" : "on",
      `${node.id} must read as live on a default pipeline`);
    assert.equal(node.state(context([node.stage])), "off", `${node.stage} must move its own node`);
    for (const other of RENDER_PIPELINE_NODES) {
      if (other.id === node.id) continue;
      assert.equal(other.state(context([node.stage])), other.state(context()),
        `withholding ${node.stage} must not restate ${other.id}`);
    }
  }
});

test("the raster-only tiers say unavailable rather than claiming a saving", () => {
  // On the megakernel primary these three passes are not reached at all, so an
  // ablation on them would report a saving the frame never makes. Their
  // switches still exist and are live on the arm that runs them.
  const traced: RenderPipelineContext = { ...context(), rasterPrimaryActive: false };
  for (const id of ["thin-glass", "scene-primitive", "rigid-impostor"] as const) {
    const node = RENDER_PIPELINE_NODES.find((candidate) => candidate.id === id)!;
    assert.equal(node.state(traced), "unavailable", `${id} is not reachable on the traced primary`);
    assert.equal(node.state({ ...traced, disabledStages: disabledRenderStagesFrom([id]) }), "unavailable",
      `${id} must not pretend a switch withheld a pass the frame never encodes`);
    assert.equal(node.state(context()), "on", `${id} must be live on the raster primary`);
    assert.equal(node.state(context([id])), "off", `${id} must be switchable on the raster primary`);
  }
});
