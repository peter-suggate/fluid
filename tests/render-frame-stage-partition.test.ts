import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  RENDER_FRAME_STAGES,
  RENDER_FRAME_STAGE_TRACE,
  RenderFrameSeamRecorder,
  mergeRenderFrameManifests,
  type RenderFrameStageId,
  type RenderFrameStageOwner,
} from "../lib/core/render-frame-stages";
import {
  RENDER_PIPELINE_NODES,
  measureRenderPipelineBand,
  measureRenderPipelineNode,
  renderPipelineEncodings,
  renderPipelineStageDurations,
  renderPipelineUnownedPhases,
} from "../lib/core/render-pipeline-graph";

/**
 * The RENDER panel puts a figure on a row only when the stages that row owns
 * are the stages the encoders actually close. The registry makes a missing,
 * renamed or unassigned stage a type error; what the types cannot see — the
 * order the seams close in, an encoder closing a stage it does not own, a pass
 * chain encoded after the last seam — is pinned here against the encoders'
 * source, alongside the attribution rule the whole ABI was written for.
 */

/** Where each owner's seams live. One file per owner, by construction. */
const OWNER_SOURCE: Record<RenderFrameStageOwner, string> = {
  world: "lib/svo/webgpu-svo-sparse-bricks.ts",
  renderer: "lib/core/webgpu-renderer.ts",
  water: "lib/core/webgpu-water-pipeline.ts",
  svo: "lib/svo/webgpu-svo-dry-scene.ts",
};

/** Stage ids closed in one encoder, in the order the source closes them. */
const seamsClosedIn = (path: string): readonly string[] => {
  const source = readFileSync(new URL(`../${path}`, import.meta.url), "utf8");
  return [...source.matchAll(/(?:closeStage|seam\?\.|tracePhase\?\.)\("([a-z0-9-]+)"\)/g)]
    .map((match) => match[1]);
};

/** First-closed order, which is the order the frame reaches the seams. */
const firstClosedOrder = (seams: readonly string[]): readonly string[] =>
  seams.filter((stage, index) => seams.indexOf(stage) === index);

const isSubsequence = (candidate: readonly string[], of: readonly string[]): boolean => {
  let cursor = 0;
  for (const item of candidate) {
    cursor = of.indexOf(item, cursor) + 1;
    if (cursor === 0) return false;
  }
  return true;
};

test("every seam an encoder closes is a declared stage of that encoder's own", () => {
  for (const [owner, path] of Object.entries(OWNER_SOURCE) as [RenderFrameStageOwner, string][]) {
    for (const stage of seamsClosedIn(path)) {
      const declared = RENDER_FRAME_STAGE_TRACE[stage as RenderFrameStageId];
      assert.ok(declared, `${path} closes "${stage}", which is not in the stage ABI`);
      assert.equal(declared.owner, owner,
        `${path} closes "${stage}", which belongs to ${declared.owner}`);
    }
  }
});

test("every declared stage is closed by its encoder", () => {
  const closed = new Set(Object.values(OWNER_SOURCE).flatMap((path) => seamsClosedIn(path)));
  const orphans = RENDER_FRAME_STAGES.filter((stage) => !closed.has(stage));
  assert.deepEqual(orphans, [],
    "a stage no encoder closes is a row that can only ever read unmeasured");
});

test("each encoder closes its seams in ABI order", () => {
  for (const path of Object.values(OWNER_SOURCE)) {
    const order = firstClosedOrder(seamsClosedIn(path));
    assert.ok(isSubsequence(order, RENDER_FRAME_STAGES),
      `${path} closes seams in an order the ABI does not declare: ${order.join(" → ")}`);
  }
});

test("stage trace labels are unique, so a captured phase names one stage", () => {
  const labels = RENDER_FRAME_STAGES.map((stage) => RENDER_FRAME_STAGE_TRACE[stage].phase.label);
  assert.equal(new Set(labels).size, labels.length, "two stages share a trace label");
});

test("the pipeline rows partition the stage ABI", () => {
  const owned = RENDER_PIPELINE_NODES.flatMap((node) => node.stages);
  assert.equal(new Set(owned).size, owned.length, "two rows claim the same stage");
  assert.deepEqual([...owned].sort(), [...RENDER_FRAME_STAGES].sort(),
    "every stage must be reported by exactly one row");
  for (const node of RENDER_PIPELINE_NODES) {
    if (node.stages.length > 0) continue;
    assert.ok(node.spendsNoFrameTime || node.costInsideNode,
      `${node.id} owns no stage, so it must be structural or report inside another row`);
  }
});

test("the recorder attributes each pass to the seam that closes over it", () => {
  const recorder = new RenderFrameSeamRecorder();
  const encoder = recorder.instrument(fakeEncoder());
  encoder.beginComputePass();
  encoder.beginComputePass();
  recorder.close("world-topology-publish");
  recorder.close("world-proxy-voxelize");
  encoder.beginRenderPass({ colorAttachments: [] });
  recorder.close("present");

  const manifest = recorder.manifest();
  assert.deepEqual(manifest.stages, [
    { stage: "world-topology-publish", computePasses: 2, renderPasses: 0 },
    { stage: "world-proxy-voxelize", computePasses: 0, renderPasses: 0 },
    { stage: "present", computePasses: 0, renderPasses: 1 },
  ]);
  assert.equal(manifest.unclaimed, 0);
});

test("a pass encoded after the last seam is unclaimed, never charged to a row", () => {
  const recorder = new RenderFrameSeamRecorder();
  const encoder = recorder.instrument(fakeEncoder());
  encoder.beginComputePass();
  recorder.close("fluid-coverage");
  encoder.beginComputePass();
  assert.equal(recorder.manifest().unclaimed, 1);
});

test("a stage closed twice in a frame is summed, not replaced", () => {
  const recorder = new RenderFrameSeamRecorder();
  const encoder = recorder.instrument(fakeEncoder());
  encoder.beginComputePass();
  recorder.close("scene-primitive-visibility");
  encoder.beginComputePass();
  encoder.beginComputePass();
  recorder.close("scene-primitive-visibility");
  assert.deepEqual(recorder.manifest().stages,
    [{ stage: "scene-primitive-visibility", computePasses: 3, renderPasses: 0 }]);
});

test("merging frames keeps the maximum, so an intermittent stage stays live", () => {
  const merged = mergeRenderFrameManifests([
    { stages: [{ stage: "world-proxy-voxelize", computePasses: 0, renderPasses: 0 }], unclaimed: 0 },
    { stages: [{ stage: "world-proxy-voxelize", computePasses: 7, renderPasses: 0 }], unclaimed: 0 },
    { stages: [{ stage: "world-proxy-voxelize", computePasses: 0, renderPasses: 0 }], unclaimed: 0 },
  ]);
  assert.deepEqual(merged?.stages,
    [{ stage: "world-proxy-voxelize", computePasses: 7, renderPasses: 0 }]);
});

test("a stage that encoded nothing reports a true zero, not a band's wall", () => {
  // The regression this ABI exists for: a settled world encodes no maintenance
  // pass, the source band's fence wall is tens of milliseconds of queue and
  // callback latency, and the panel used to hand all of it to the one row it
  // could not price.
  const worldBuild = node("sparse-world-build");
  const encodings = renderPipelineEncodings({
    stages: RENDER_FRAME_STAGES.map((stage) => ({ stage, computePasses: 0, renderPasses: 0 })),
    unclaimed: 0,
  });
  const cost = measureRenderPipelineNode(worldBuild, new Map(), 40, "on", encodings);
  assert.equal(cost.kind, "withheld");
  assert.equal(cost.duration_ms, 0);
});

test("without a manifest the same row is unmeasured, which is what a wall may claim", () => {
  const cost = measureRenderPipelineNode(node("sparse-world-build"), new Map(), 40, "on");
  assert.equal(cost.kind, "unmeasured");
});

test("a row whose passes are render passes is unpriced, and can carry the band wall", () => {
  const encodings = renderPipelineEncodings({
    stages: [{ stage: "primary-traversal", computePasses: 0, renderPasses: 1 }],
    unclaimed: 0,
  });
  const cost = measureRenderPipelineNode(node("primary-traversal"), new Map(), 40, "on", encodings);
  assert.equal(cost.kind, "unpriced");
  assert.equal(cost.unpricedRenderPasses, 1);
});

test("a band sums its rows' own stages and nothing else", () => {
  const durations = new Map<RenderFrameStageId, number>([
    ["world-topology-publish", 1.5],
    ["world-proxy-voxelize", 2.5],
    ["world-derived-lighting", 3],
    ["fluid-coverage", 0.07],
  ]);
  const band = measureRenderPipelineBand("source", durations, 40);
  assert.equal(band.kind, "measured");
  assert.ok(Math.abs((band.duration_ms ?? 0) - 7.07) < 1e-9);
});

test("stage durations come from the trace by label, and unowned labels are reported", () => {
  const durations = renderPipelineStageDurations({
    sampleId: 1,
    domain: "gpu",
    lane: "presentation",
    context: "test",
    capturedAt_ms: 0,
    total_ms: 3,
    phases: [
      { id: "svo-primary", label: RENDER_FRAME_STAGE_TRACE["primary-traversal"].phase.label, duration_ms: 2 },
      { id: "svo-primary", label: RENDER_FRAME_STAGE_TRACE["primary-traversal"].phase.label, duration_ms: 1 },
      { id: "svo-primary", label: "A pass nobody declared", duration_ms: 5 },
    ],
  });
  assert.equal(durations.get("primary-traversal"), 3);
  assert.deepEqual(renderPipelineUnownedPhases({
    sampleId: 1,
    domain: "gpu",
    lane: "presentation",
    context: "test",
    capturedAt_ms: 0,
    total_ms: 5,
    phases: [{ id: "svo-primary", label: "A pass nobody declared", duration_ms: 5 }],
  }), ["A pass nobody declared"]);
});

const node = (id: string) => {
  const found = RENDER_PIPELINE_NODES.find((candidate) => candidate.id === id);
  assert.ok(found, `${id} must be a pipeline row`);
  return found;
};

/** Enough of a `GPUCommandEncoder` for the recorder's proxy to count against. */
const fakeEncoder = (): GPUCommandEncoder => ({
  beginComputePass: () => ({}),
  beginRenderPass: () => ({}),
  finish: () => ({}),
} as unknown as GPUCommandEncoder);
