import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { AnyStageLens } from "../lib/core/stage-lens";
import {
  ADAPTIVE_MASS_ADVANCE_PHASE,
  ADAPTIVE_MASS_FLUID_PIPELINE,
  ADAPTIVE_MASS_GPU_WORK_CHUNKS,
  ADAPTIVE_MASS_RESIDENT_STAGE_PHASE,
  adaptiveMassPressureTopologyChip,
} from "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import {
  SPARSE_CM12_LENSES,
  SPARSE_CM12_STAGE_LENSES,
} from "../lib/methods/adaptive-mass/sparse-cm12-stage-lenses";
import { SPARSE_CM12_STAGES } from "../lib/methods/adaptive-mass/sparse-cm12-stages";
import {
  SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES,
  SPARSE_CM12_RESIDENT_STAGES,
} from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

/**
 * The SIM panel puts a figure on a stage only when that stage's declared seam
 * label is the label the encoder actually emits. The stage registry makes a
 * missing or renamed stage a type error; what the types cannot see — the
 * order the encoder closes its seams, a dispatch that escapes the partition,
 * two seams sharing a label — is pinned here against the encoder's source.
 */

const residentSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");

const residentStageSource = (): string => {
  const begin = residentSource.indexOf("  encode(\n    encoder: GPUCommandEncoder,");
  const end = residentSource.indexOf("    this.stageLenses?.endFrame(encoder);", begin);
  assert.ok(begin >= 0 && end > begin, "the resident advance must be inspectable");
  return residentSource.slice(begin, end);
};

/** Each stage callback's body, keyed by stage id, in encode order. */
const residentStageBodies = (): Map<string, string> => {
  const body = residentStageSource();
  const heads = [...body.matchAll(/\n {4}stage\("([a-z0-9-]+)"/g)];
  const bodies = new Map<string, string>();
  heads.forEach((head, index) => {
    const start = head.index ?? 0;
    const stop = index + 1 < heads.length ? heads[index + 1].index ?? body.length : body.length;
    bodies.set(head[1], body.slice(start, stop));
  });
  return bodies;
};

test("the resident encoder closes every declared stage exactly once, in order", () => {
  assert.deepEqual([...residentStageBodies().keys()], [...SPARSE_CM12_RESIDENT_STAGES],
    "encode order and the stage ABI must be the same list");
});

test("the stage registry lists the stages in encode order", () => {
  // `satisfies` pins the key set; insertion order is what the diagram reads down.
  assert.deepEqual(Object.keys(SPARSE_CM12_STAGES), [...SPARSE_CM12_RESIDENT_STAGES]);
});

test("no dispatch escapes the stage partition", () => {
  const body = residentStageSource();
  assert.doesNotMatch(body, /\n {4}dispatch\(/,
    "a dispatch outside a stage callback would be timed under its neighbour");
});

test("each stage closes exactly the sub-seams its ABI declares, in order", () => {
  const bodies = residentStageBodies();
  for (const stage of SPARSE_CM12_RESIDENT_STAGES) {
    const closed = [...(bodies.get(stage) ?? "").matchAll(/closeSubstage\("([a-z0-9-]+)"\)/g)]
      .map((match) => match[1]);
    assert.deepEqual(closed, [...SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[stage]],
      `${stage} closes a different sub-seam list than it declares`);
  }
});

test("truth-sensitive timing rows inventory the shader work their UI describes", () => {
  const bodies = residentStageBodies();
  for (const stage of ["activity-measurement", "resolution-planning"] as const) {
    const manifest = SPARSE_CM12_STAGES[stage].timedWork;
    assert.ok(manifest, `${stage} must publish timed-work detail to the UI`);
    const body = bodies.get(stage) ?? "";
    const dispatched = [...body.matchAll(
      /\bdispatch[A-Za-z]*\("([A-Za-z0-9]+)"/g,
    )].map((match) => match[1]).sort();
    assert.doesNotMatch(body, /\.setPipeline\(/,
      `${stage} bypasses the dispatch helpers and therefore its UI manifest`);
    const declared = manifest.groups.flatMap((group) => group.entryPoints).sort();
    assert.deepEqual(declared, dispatched,
      `${stage} shader dispatches changed; update its UI timed-work manifest`);
    assert.equal(new Set(declared).size, declared.length,
      `${stage} timed-work manifest lists one shader twice`);
    const commandCopies = [...body.matchAll(/encoder\.copyBufferToBuffer\(/g)].length;
    const declaredCommandCopies = "commandCopies" in manifest
      ? manifest.commandCopies : 0;
    assert.equal(declaredCommandCopies, commandCopies,
      `${stage} command-copy count changed; update its UI timed-work manifest`);
  }
});

test("every seam the encoder emits has a label of its own", () => {
  const labels = ADAPTIVE_MASS_GPU_WORK_CHUNKS.map((chunk) => chunk.phase.label);
  assert.equal(new Set(labels).size, labels.length,
    "two seams sharing a label would sum into one unattributable figure");
  for (const stage of SPARSE_CM12_RESIDENT_STAGES) {
    assert.equal(ADAPTIVE_MASS_RESIDENT_STAGE_PHASE[stage], SPARSE_CM12_STAGES[stage].phase);
  }
});

test("the SIM diagram has one node per resident stage, in encode order", () => {
  assert.deepEqual(ADAPTIVE_MASS_FLUID_PIPELINE.stages.map((stage) => stage.id),
    [...SPARSE_CM12_RESIDENT_STAGES]);
});

test("adaptivity timing labels describe the complete bracketed work", () => {
  const stages = new Map(ADAPTIVE_MASS_FLUID_PIPELINE.stages.map((stage) => [stage.id, stage]));
  assert.equal(stages.get("activity-measurement")?.label, "Activity census + frontier");
  assert.match(stages.get("activity-measurement")?.tip.timing ?? "", /10 shader entry points/);
  assert.equal(stages.get("resolution-planning")?.label, "Candidate topology build");
  assert.match(stages.get("resolution-planning")?.tip.timing ?? "",
    /13 shader entry points \+ 4 command-buffer copies/);
  assert.equal(stages.get("brick-retirement")?.label, "Post-commit activity mask");
});

test("the SIM diagram names every advance seam, and only those", () => {
  const emitted = ADAPTIVE_MASS_GPU_WORK_CHUNKS.map((chunk) => chunk.phase.label);
  const named = ADAPTIVE_MASS_FLUID_PIPELINE.stages.flatMap((stage) => stage.phaseLabels);
  assert.equal(new Set(named).size, named.length, "no two nodes may claim one seam");
  assert.deepEqual([...named].sort(), [...emitted].sort());
});

test("every timestamp phase is owned by the stage whose seam emits it", () => {
  for (const chunk of ADAPTIVE_MASS_GPU_WORK_CHUNKS) {
    assert.equal(chunk.rollupStage, chunk.residentStage);
    const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
      (candidate) => candidate.id === chunk.rollupStage);
    assert.ok(stage, `${chunk.id} names missing stage ${chunk.rollupStage}`);
    assert.ok(stage.phaseLabels.includes(chunk.phase.label),
      `${chunk.id} is absent from its ${chunk.rollupStage} node`);
  }
  for (const stage of SPARSE_CM12_RESIDENT_STAGES) {
    const own = ADAPTIVE_MASS_GPU_WORK_CHUNKS.filter((chunk) => chunk.residentStage === stage);
    assert.equal(own.filter((chunk) => chunk.kind === "stage").length, 1);
    assert.equal(own.filter((chunk) => chunk.kind === "substage").length,
      SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[stage].length);
    assert.equal(own.at(-1)?.kind, "stage", `${stage}'s own seam closes last`);
  }
});

test("every diagram node owns a seam; none borrows one", () => {
  for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
    assert.ok(stage.phaseLabels.length > 0, `${stage.id} would read as unmeasured forever`);
    assert.equal(stage.costInsideStage, undefined,
      `${stage.id} owns a seam and must not also borrow one`);
  }
});

test("every diagram node sits in a declared band", () => {
  const bands = new Set(ADAPTIVE_MASS_FLUID_PIPELINE.bands.map((band) => band.id));
  for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
    assert.ok(bands.has(stage.band), `${stage.id} is in the orphan band ${stage.band}`);
  }
  const populated = new Set(ADAPTIVE_MASS_FLUID_PIPELINE.stages.map((stage) => stage.band));
  for (const band of bands) assert.ok(populated.has(band), `band ${band} draws nothing`);
});

test("a stage's lens is the lens its diagram node carries", () => {
  // The overlay opens `stage-lens/<stage>` from the node; a lens declared
  // under one id and drawn under another is a ◎ that opens nothing.
  for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
    const declared: AnyStageLens | null =
      SPARSE_CM12_STAGE_LENSES[stage.id as keyof typeof SPARSE_CM12_STAGE_LENSES];
    assert.ok(stage.lens === (declared ?? undefined),
      `${stage.id} draws a lens it does not declare`);
    if (stage.lens) assert.equal(stage.lens.stage, stage.id);
  }
  assert.deepEqual(SPARSE_CM12_LENSES,
    ADAPTIVE_MASS_FLUID_PIPELINE.stages.flatMap((stage) => stage.lens ? [stage.lens] : []));
});

test("pressure topology says when its attribution receipt is missing", () => {
  const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
    (candidate) => candidate.id === "pressure-topology",
  );
  assert.ok(stage);
  const context = { info: null } as Parameters<typeof stage.chip>[0];
  assert.equal(stage.chip(context), adaptiveMassPressureTopologyChip(null));
  assert.match(stage.chip(context), /awaiting paired diagnostics receipt/);
});

test("the CPU-lane brackets stay outside the diagram", () => {
  // Host planning and submission are real CPU intervals but no GPU stage owns
  // them; a node naming either would read 0 ms under a hardware partition.
  const named = new Set(ADAPTIVE_MASS_FLUID_PIPELINE.stages.flatMap((stage) => stage.phaseLabels));
  for (const phase of [
    ADAPTIVE_MASS_ADVANCE_PHASE.commandEncoding,
    ADAPTIVE_MASS_ADVANCE_PHASE.upload,
    ADAPTIVE_MASS_ADVANCE_PHASE.queueCompletion,
  ]) assert.ok(!named.has(phase.label), `${phase.label} is not GPU stage time`);
});
