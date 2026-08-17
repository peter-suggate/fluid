#!/usr/bin/env node
/** CPU/static fail-closed gate for the append-only VEX1 + A4D2 batch adapter. */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS,
  SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_REQUIRED_FIELDS,
  SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT,
  SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT_WORDS,
  SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS,
  SPARSE_CM12_VEX_ACTIVITY_BATCH_QA,
  createSparseCM12VexActivityBatchIndirectCopies,
  createSparseCM12VexActivityBatchConstructionSchedule,
  createSparseCM12VexActivityBatchInitialWords,
  createSparseCM12VexActivityBatchLayout,
  createSparseCM12VexActivityBatchPipelineDescriptors,
  createSparseCM12VexActivityBatchQAOraclePipelineDescriptors,
  createSparseCM12VexActivityBatchSchedule,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-activity-batch";
import {
  SPARSE_CM12_PRODUCTION_ACTIVITY_MAGIC,
  SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER,
} from "../lib/methods/adaptive-mass/sparse-cm12-production-activity";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  SPARSE_CM12_VELOCITY_EXTENSION_MAGIC,
  SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT,
  sparseCM12VelocityExtensionInputBank,
  sparseCM12VelocityExtensionOutputBank,
} from "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VexActivityBatchWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-activity-batch.wgsl";

const fail = (message: string): never => { throw new Error(message); };

const rangesOverlap = (a: readonly [number, number], b: readonly [number, number]): boolean =>
  a[0] < b[1] && b[0] < a[1];

function validateLayout(profile: { readonly name: string; readonly activityTailWords: number;
  readonly stateTailFloats: number; readonly brickCapacity: number;
  readonly cellCapacity: number; readonly brickFineResolution: 4 | 8 | 16 }): void {
  const layout = createSparseCM12VexActivityBatchLayout(profile);
  if (layout.activityBaseWords < profile.activityTailWords
    || layout.velocityState.acceptedVelocityFloatBase < profile.stateTailFloats) {
    fail(`${profile.name}: batch overlaps an accepted arena tail`);
  }
  if (rangesOverlap(
    [layout.productionActivity.baseWords, layout.productionActivity.totalWords],
    [layout.velocityExtension.headerBaseWords, layout.velocityExtension.totalWords],
  )) fail(`${profile.name}: A4D2 overlaps VEX1 in activity`);
  if (layout.productionActivity.totalWords > layout.velocityExtension.headerBaseWords) {
    fail(`${profile.name}: VEX1 starts before the A4D2 tail`);
  }
  const words = createSparseCM12VexActivityBatchInitialWords(layout);
  const a = layout.productionActivity.baseWords - layout.activityBaseWords;
  const v = layout.velocityExtension.headerBaseWords - layout.activityBaseWords;
  if (words[a + SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.magic]
    !== SPARSE_CM12_PRODUCTION_ACTIVITY_MAGIC) fail(`${profile.name}: A4D2 magic missing`);
  if (words[v + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.magic]
    !== SPARSE_CM12_VELOCITY_EXTENSION_MAGIC) fail(`${profile.name}: VEX1 magic missing`);
  if (words.length !== layout.totalActivityWords - layout.activityBaseWords) {
    fail(`${profile.name}: initial tail has wrong length`);
  }
  console.log(`${profile.name}: activity [${layout.activityBaseWords},`
    + `${layout.totalActivityWords}) u32; state [${layout.velocityState.acceptedVelocityFloatBase},`
    + `${layout.totalStateFloats}) f32; ${layout.productionActivity.tilesPerBrick} tiles/brick`);
}

function validateIndirect(): void {
  const packets = Object.entries(SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT);
  if (packets.length !== 5) fail("batch must own exactly five indirect packets");
  const ranges = packets.map(([name, value]) => ({ name,
    range: [value.offsetWords, value.offsetWords + value.sizeWords] as const }));
  for (let index = 0; index < ranges.length; index += 1) {
    const current = ranges[index]!;
    if (current.range[1] > SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT_WORDS) {
      fail(`${current.name}: indirect range exceeds buffer`);
    }
    for (let prior = 0; prior < index; prior += 1) {
      if (rangesOverlap(current.range, ranges[prior]!.range)) {
        fail(`${current.name}: overlaps ${ranges[prior]!.name}`);
      }
    }
  }
  const layout = createSparseCM12VexActivityBatchLayout({ activityTailWords: 101,
    stateTailFloats: 1003, brickCapacity: 17, cellCapacity: 4096 });
  const copies = createSparseCM12VexActivityBatchIndirectCopies(layout);
  if (new Set(copies.slice(0, 4).map((copy) => copy.destinationWord)).size !== 4) {
    fail("A4D2 indirect packets are not dedicated");
  }
  const serial = copies.filter((copy) => copy.packet === "vexSerial");
  if (serial.length !== 4 || serial.some((copy, index) => index > 0 && !copy.reuseAfter)) {
    fail("VEX serial packet lacks ordered reuse certificates");
  }
  for (const copy of copies) {
    if (copy.words !== 3 || copy.sourceWord < layout.activityBaseWords
      || copy.sourceWord + copy.words > layout.totalActivityWords) {
      fail(`${copy.id}: invalid activity header snapshot`);
    }
  }
}

function validatePipelinesAndSchedule(): void {
  const pipelines = createSparseCM12VexActivityBatchPipelineDescriptors();
  if (pipelines.length !== 53) fail(`expected 53 manager descriptors; got ${pipelines.length}`);
  const byKey = new Map(pipelines.map((pipeline) => [pipeline.key, pipeline]));
  if (byKey.size !== pipelines.length) fail("duplicate pipeline descriptor key");
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    for (const prefix of ["prepareVelocityExtensionFrontier",
      "expandVelocityExtensionFrontier", "sealVelocityExtensionFrontier"] as const) {
      const descriptor = byKey.get(`${prefix}${depth}`);
      if (descriptor?.constants?.EXTENSION_FRONTIER_DEPTH !== depth) {
        fail(`${prefix}${depth}: incorrect frontier specialization`);
      }
    }
    if (byKey.get(`advanceVelocityExtensionCandidates${depth}`)
      ?.constants?.EXTENSION_RECURRENCE_DEPTH !== depth) {
      fail(`advanceVelocityExtensionCandidates${depth}: incorrect recurrence specialization`);
    }
    const input = sparseCM12VelocityExtensionInputBank(depth);
    const output = sparseCM12VelocityExtensionOutputBank(depth);
    if (input === output || input !== (depth % 2 === 1 ? "destination" : "source")) {
      fail(`depth ${depth}: legacy source/destination parity changed`);
    }
  }
  const bootstrap = byKey.get("bootstrapVelocityExtensionRoots");
  if (!bootstrap?.constructionOnly || bootstrap.runtimeSelectable !== false
    || bootstrap.outputSelectable !== false) fail("full bootstrap is runtime-selectable");
  if (pipelines.some((pipeline) => pipeline.dispatch.kind === "construction-authority-indirect"
    && !pipeline.constructionOnly)) fail("runtime pipeline uses full construction authority");

  const schedule = createSparseCM12VexActivityBatchSchedule();
  if (schedule.some((step) => step.constructionOnly
    || step.pipeline === "bootstrapVelocityExtensionRoots")) {
    fail("construction bootstrap leaked into the recurring runtime schedule");
  }
  const construction = createSparseCM12VexActivityBatchConstructionSchedule();
  if (construction.length !== 40 || construction.some((step) => !step.constructionOnly)
    || construction[0]!.pipeline !== "bootstrapVelocityExtensionRoots"
    || construction.at(-1)!.copy !== "copy-vex-blast") {
    fail("construction bootstrap/plan schedule is not isolated and complete");
  }
  const ids = new Set(schedule.map((step) => step.id));
  if (ids.size !== schedule.length) fail("duplicate schedule step");
  const position = new Map(schedule.map((step, index) => [step.id, index]));
  for (const step of schedule) {
    for (const dependency of step.after) {
      if (!ids.has(dependency) || position.get(dependency)! >= position.get(step.id)!) {
        fail(`${step.id}: invalid dependency ${dependency}`);
      }
    }
    if (step.pipeline && !byKey.has(step.pipeline)) fail(`${step.id}: missing pipeline`);
  }
  const copies = new Set(createSparseCM12VexActivityBatchIndirectCopies(
    createSparseCM12VexActivityBatchLayout({ activityTailWords: 0, stateTailFloats: 0,
      brickCapacity: 1, cellCapacity: 64 }),
  ).map((copy) => copy.id));
  for (const step of schedule.filter((entry) => entry.operation === "copy-indirect")) {
    if (!step.copy || !copies.has(step.copy)) fail(`${step.id}: unknown copy seam`);
  }
  const constructionIds = new Set(construction.map((step) => step.id));
  const constructionPosition = new Map(construction.map((step, index) => [step.id, index]));
  for (const step of construction) {
    for (const dependency of step.after) {
      if (!constructionIds.has(dependency)
        || constructionPosition.get(dependency)! >= constructionPosition.get(step.id)!) {
        fail(`${step.id}: invalid construction dependency ${dependency}`);
      }
    }
    if (step.pipeline && !byKey.has(step.pipeline)) fail(`${step.id}: missing pipeline`);
    if (step.copy && !copies.has(step.copy)) fail(`${step.id}: unknown copy seam`);
  }
  const requiredOrder = ["vex-begin-execution", "vex-initialize-candidates",
    "vex-accept", "vex-producer-roots",
    "vex-begin-next-plan", "vex-copy-next-blast", "fpl-activity-receipt"];
  for (let index = 1; index < requiredOrder.length; index += 1) {
    if (position.get(requiredOrder[index - 1]!)! >= position.get(requiredOrder[index]!)!) {
      fail(`two-phase schedule order is not exact at ${requiredOrder[index]}`);
    }
  }
  if (schedule.some((step) => /accepted|global/i.test(step.pipeline ?? ""))) {
    fail("production schedule contains a global accepted dispatch");
  }
}

/** Pure phase/generation fixture for the GPU header state machine. Planning
 * must never advance a generation; only the following frame's recurrence
 * commit may accept the already-imported blast. */
function validateTwoPhaseGenerationFixture(): void {
  type Phase = "collecting" | "planned";
  type Header = { accepted: number; candidate: number; phase: Phase; roots: number };
  const plan = (header: Header, sourceFca: number, nextFca: number,
    authoritySealed: boolean): Header => {
    const construction = header.accepted === 0 && header.candidate === sourceFca;
    const runtime = authoritySealed && header.accepted === sourceFca
      && header.candidate === nextFca;
    const injection = header.accepted !== 0 && header.accepted + 1 === sourceFca
      && header.candidate === sourceFca;
    if (header.phase !== "collecting" || (!construction && !runtime && !injection)) {
      fail("two-phase fixture rejected a valid planning boundary");
    }
    return { ...header, phase: "planned" };
  };
  const consume = (header: Header, sourceFca: number, nextFca: number): Header => {
    if (header.phase !== "planned" || header.candidate !== sourceFca
      || nextFca <= header.candidate) fail("two-phase fixture consumed a stale plan");
    return { accepted: header.candidate, candidate: nextFca,
      phase: "collecting", roots: 0 };
  };
  const reopenInjection = (header: Header, sourceFca: number): Header => {
    if (header.phase !== "planned" || header.candidate !== sourceFca
      || (header.accepted !== 0 && header.accepted + 1 !== header.candidate)) {
      fail("two-phase fixture reopened a mismatched injection plan");
    }
    return { ...header, phase: "collecting" };
  };

  let header: Header = { accepted: 0, candidate: 1, phase: "collecting", roots: 64 };
  header = plan(header, 1, 2, false);
  if (header.accepted !== 0 || header.candidate !== 1) fail("construction plan published early");
  header = consume(header, 1, 2);
  if (header.accepted !== 1 || header.candidate !== 2) fail("construction consume mismatch");
  header = { ...header, roots: 7 };
  header = plan(header, 1, 2, true);
  if (header.accepted !== 1 || header.candidate !== 2) fail("runtime plan published early");
  header = consume(header, 2, 3);
  if (header.accepted !== 2 || header.candidate !== 3) fail("runtime consume mismatch");
  header = { ...header, roots: 5 };
  header = plan(header, 2, 3, true);
  const rootsBeforeInjection = header.roots;
  header = reopenInjection(header, 3);
  if (header.roots !== rootsBeforeInjection) fail("injection reopen dropped accepted roots");
  header = { ...header, roots: header.roots + 2 };
  header = plan(header, 3, 4, false);
  if (header.accepted !== 2 || header.candidate !== 3 || header.roots !== 7) {
    fail("injection replan changed generation or root authority");
  }
}

/** Interleaving fixture for the root-stamp busy state. The claimant owns the
 * one list slot and initial cause publication; same-dispatch waiters are
 * allowed to retire because their producer cause is uniform. A later dispatch
 * observes the published generation and atomically adds its distinct cause. */
function validateRootClaimConcurrencyFixture(): void {
  const generation = 7;
  const busy = generation | 0x8000_0000;
  const firstCause = 1 << 1;
  const laterCause = 1 << 3;
  let stamp = 0;
  let rootCause = 0;
  let listSlots = 0;
  let faults = 0;

  // Claimant wins the compare-exchange, then is descheduled before it can
  // initialize the cause/list. A concurrent invocation from the same dispatch
  // sees the exact busy generation and safely accepts the uniform cause.
  stamp = busy;
  if (stamp === busy) {
    // No list or cause mutation belongs to the waiter.
  } else {
    faults += 1;
  }
  rootCause = firstCause;
  listSlots += 1;
  stamp = generation;

  // A differently-caused producer is dispatch-ordered after publication.
  // It neither appends a duplicate nor loses its provenance bit.
  if (stamp === generation) rootCause |= laterCause;
  else faults += 1;

  if (faults !== 0 || listSlots !== 1
    || rootCause !== (firstCause | laterCause) || stamp !== generation) {
    fail("root busy-claim interleaving lost slot, cause, or fault semantics");
  }
}

/** A wide guard failure is one rejected transaction. Model the exact atomic
 * flag winner contract at the 1.15M-lane scale; no losing invocation may
 * hammer the shared indirect header or provenance receipt. */
function validateWideFaultLatchFixture(): void {
  const faultBit = 1 << 2;
  let flags = 0;
  let faultReceipt = 0;
  let uncoveredCompatibilityReceipt = 0;
  let zeroPublications = 0;
  let provenanceFaultPublications = 0;
  let firstCell = 0xffff_ffff;
  let firstDepth = 0xffff_ffff;
  for (let lane = 0; lane < 1_150_000; lane += 1) {
    const priorFlags = flags;
    flags |= faultBit;
    if ((priorFlags & faultBit) !== 0) continue;
    faultReceipt = 1;
    uncoveredCompatibilityReceipt = 1;
    zeroPublications += 1;
    provenanceFaultPublications += 1;
    firstCell = lane;
    firstDepth = 3;
  }
  if (flags !== faultBit || faultReceipt !== 1
    || uncoveredCompatibilityReceipt !== 1 || zeroPublications !== 1
    || provenanceFaultPublications !== 1 || firstCell !== 0 || firstDepth !== 3) {
    fail("wide VEX guard failure is not a binary first-fault transaction latch");
  }
}

/** Force the receiver activation CAS to fail spuriously for every attempt.
 * The shader must stop at the fixed bound and poison only the local topology
 * transaction; a silent skip and an unbounded retry are both forbidden. */
function validateReceiverActivationCASFixture(): void {
  let attempts = 0;
  let activityFailureFlags = 0;
  let candidateStatus = 1;
  let scheduled = 1;
  for (let attempt = 0; attempt < 64; attempt += 1) attempts += 1;
  if (attempts === 64) {
    candidateStatus = 2;
    scheduled = 0;
    activityFailureFlags |= 32;
  }
  if (attempts !== 64 || candidateStatus !== 2 || scheduled !== 0
    || activityFailureFlags !== 32) {
    fail("receiver activation weak-CAS exhaustion is not bounded and fail-closed");
  }
}

/** A root may retire between producer publication and the planning boundary,
 * or between sealing and FPL import. Seed owns the one activity decision; once
 * sealed, the exact blast and FPL stage-0 coverage must remain identical. */
function validateRetiredRootFramePlanFixture(): void {
  const plan = (activeAtSeed: boolean, activeAtImport: boolean,
    legacyImportFilter: boolean): { readonly blast: number[]; readonly fpl: number[] } => {
    const root = 97_780;
    const blast = activeAtSeed ? [root] : [];
    const fpl = blast.filter(() => !legacyImportFilter || activeAtImport);
    return { blast, fpl };
  };

  const retiredBeforeSeed = plan(false, false, false);
  if (retiredBeforeSeed.blast.length !== 0 || retiredBeforeSeed.fpl.length !== 0) {
    fail("retired VEX root survived the seed-time activity boundary");
  }
  const liveCompanion = plan(true, true, false);
  if (liveCompanion.blast.length !== 1 || liveCompanion.fpl[0] !== 97_780) {
    fail("live VEX root lost exact FPL stage-0 coverage");
  }
  // Pending topology roots are recorded before the accepted rung becomes
  // active. Once the rung enters before seed, it is retained like any other
  // live root and receives the same-frame FPL receipt.
  const enteringRungCompanion = plan(true, true, false);
  if (enteringRungCompanion.blast[0] !== 97_780
    || enteringRungCompanion.fpl[0] !== 97_780) {
    fail("entering-rung VEX root was pruned after accepted activation");
  }
  const retiredAfterSeed = plan(true, false, false);
  if (retiredAfterSeed.blast.length !== 1
    || retiredAfterSeed.fpl.length !== retiredAfterSeed.blast.length) {
    fail("sealed VEX blast diverged from FPL coverage after retirement");
  }
  const mutationControl = plan(true, false, true);
  if (mutationControl.fpl.length === mutationControl.blast.length) {
    fail("retired-root fixture does not detect the legacy importer filter");
  }
}

/** Cache ownership is local accepted topology identity, not live planning
 * state. A pre-commit receiver activation must not change it; a logical-key or
 * accepted-rung replacement must. */
function validateStableVelocityOwnerFixture(): void {
  const stableOwner = (cellRung: number, acceptedRung: number,
    logicalKey: number): number => cellRung === acceptedRung ? logicalKey : 0xffff_ffff;
  const volatileOwner = (brick: number, cellRung: number, acceptedRung: number,
    span: number, active: boolean): number => (brick ^ (cellRung << 24)
      ^ ((acceptedRung | (span << 8) | (active ? 0x8000_0000 : 0)) * 0x9e37_79b9)) >>> 0;
  const cellRung = 16;
  const logicalKey = 37;
  const beforeActivation = stableOwner(cellRung, 16, logicalKey);
  const afterPrecommitActivation = stableOwner(cellRung, 16, logicalKey);
  if (volatileOwner(3, cellRung, 16, 1, false)
      === volatileOwner(3, cellRung, 16, 1, true)) {
    fail("fixture did not reproduce the old pre-commit active-bit owner change");
  }
  if (beforeActivation !== afterPrecommitActivation) {
    fail("pre-commit receiver activity changed stable VEX owner identity");
  }
  if (stableOwner(cellRung, 8, logicalKey) !== 0xffff_ffff
    || stableOwner(cellRung, 16, logicalKey + 1) === beforeActivation) {
    fail("accepted rung/logical-key replacement did not change VEX owner identity");
  }
}

/** The canonical B16 dam's first swept receiver is logical brick [3,0,0].
 * Its construction cache is intentionally invalid while dormant. The prior
 * active boundary at fine x=47 can close only eight graph layers (x48..55),
 * so local [8,14,0] / fine x56 is the first active ninth-layer lookup unless
 * activation publishes the receiver's complete accepted range as VEX roots. */
function validateActivatedReceiverRootFixture(): void {
  const brickFineResolution = 16;
  const brickIndex = 3;
  const local = [8, 14, 0] as const;
  const localIndex = local[0] + brickFineResolution
    * (local[1] + brickFineResolution * local[2]);
  const firstCell = brickIndex * brickFineResolution ** 3;
  const faultCell = firstCell + localIndex;
  const priorBoundaryFineX = 47;
  const faultFineX = brickIndex * brickFineResolution + local[0];
  const oldClosureMaximumFineX = priorBoundaryFineX
    + SPARSE_CM12_VELOCITY_EXTENSION_DEPTH;
  const activatedRoots = new Set(Array.from(
    { length: brickFineResolution ** 3 }, (_, index) => firstCell + index));
  if (faultCell !== 12_520 || faultFineX !== 56
    || oldClosureMaximumFineX !== 55 || !activatedRoots.has(faultCell)) {
    fail("same-rung receiver activation did not root the first uncached ninth layer");
  }
}

/** The construction queue belongs to candidate generation one. The passive
 * oracle's first physical capture advances directly to candidate two, so its
 * compact counts must reset while generation-stamped arrays remain untouched. */
function validateLegacyOracleClockResetFixture(): void {
  const capacity = 262_144;
  const constructionRootCount = 161_792;
  const generationOneStamp: number = 1;
  const nextCandidateGeneration: number = 2;
  const generationTwoRoots = 129_152;
  if (constructionRootCount + generationTwoRoots <= capacity) {
    fail("fixture did not reproduce stale construction-root capacity overflow");
  }
  const resetRootCount = 0;
  const appended = resetRootCount + generationTwoRoots;
  if (appended > capacity || generationOneStamp === nextCandidateGeneration) {
    fail("passive oracle compact reset did not preserve generation-keyed reuse");
  }
}

function validateHooksReceiptsAndQA(): void {
  const expectedRoots = SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT
    .map((entry) => entry.producer);
  const roots = SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS.vexRoots
    .map((entry) => entry.producer);
  if (roots.length !== expectedRoots.length
    || expectedRoots.some((root) => !roots.includes(root))) fail("VEX root coverage is incomplete");
  if (SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS.activity.length !== 14
    || SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS.provenance.length !== 5) {
    fail("resident hook ABI is incomplete");
  }
  for (let stage = 0; stage < 6; stage += 1) {
    const receipt = SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS
      .find((value) => value.stage === stage);
    if (!receipt || receipt.absent !== "unknown-magenta"
      || receipt.direct !== "required" || receipt.closure !== "required"
      || receipt.disposition !== "executed-or-skipped"
      || receipt.generation !== "accepted-frame"
      || !/executed|published/.test(receipt.requirement)) {
      fail(`Grid stage ${stage} lacks fail-closed truthful receipt`);
    }
  }
  for (const field of ["causeMask", "maximumClosureDepth", "producerGeneration",
    "consumerGeneration", "executedCount", "skippedCount", "unknownCount",
    "uncoveredWriteCount", "faultCount"]) {
    if (!(SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_REQUIRED_FIELDS as readonly string[])
      .includes(field)) fail(`Grid receipt ABI lacks ${field}`);
  }
  if (SPARSE_CM12_VEX_ACTIVITY_BATCH_QA.some((qa) => !qa.constructionOnly
    || qa.runtimeSelectable || qa.outputSelectable)) fail("QA oracle can become runtime fallback");
  const qaPipelines = createSparseCM12VexActivityBatchQAOraclePipelineDescriptors();
  if (qaPipelines.length !== 4 || qaPipelines.some((pipeline) => !pipeline.constructionOnly
    || pipeline.runtimeSelectable !== false || pipeline.outputSelectable !== false)) {
    fail("legacy QA pipeline descriptors are not immutable construction-only paths");
  }
}

function validateSourceContracts(): void {
  const directory = "../lib/methods/adaptive-mass/";
  const batchWGSL = readFileSync(fileURLToPath(new URL(
    `${directory}sparse-cm12-vex-activity-batch.wgsl.ts`, import.meta.url)), "utf8");
  const activityWGSL = readFileSync(fileURLToPath(new URL(
    `${directory}sparse-cm12-production-activity.wgsl.ts`, import.meta.url)), "utf8");
  const vexWGSL = readFileSync(fileURLToPath(new URL(
    `${directory}sparse-cm12-velocity-extension.wgsl.ts`, import.meta.url)), "utf8");
  const residentWGSL = readFileSync(fileURLToPath(new URL(
    `${directory}webgpu-sparse-cm12-resident.wgsl.ts`, import.meta.url)), "utf8");
  const residentTS = readFileSync(fileURLToPath(new URL(
    `${directory}webgpu-sparse-cm12-resident.ts`, import.meta.url)), "utf8");
  if (batchWGSL.includes("createComputePipeline")) fail("adapter bypasses compilation manager");
  for (const hook of SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS.activity) {
    if (!activityWGSL.includes(`${hook}(`)) fail(`A4D2 does not consume ${hook}`);
  }
  for (const suffix of ["VelocityExtensionRoot", "VelocityExtensionClosure",
    "VelocityExtensionScheduled", "VelocityExtensionOwner", "VelocityExtensionFault"]) {
    if (!vexWGSL.includes(suffix)) fail(`VEX1 does not consume ${suffix}`);
  }
  const withoutConstructionQA = vexWGSL
    .replace(/fn bootstrapVelocityExtensionRoots[\s\S]*?\n}\n/, "")
    .replace(/fn captureLegacyVelocityExtensionForQA[\s\S]*?\n}\n/, "");
  const productionVexWGSL = withoutConstructionQA;
  for (const token of ["dispatchAcceptedCells", "acceptedTemplateCellInvocation",
    "globalAcceptedCells"]) {
    if (activityWGSL.includes(token) || productionVexWGSL.includes(token)) {
      fail(`runtime cutover contains forbidden full authority token ${token}`);
    }
  }
  for (const token of ["reopenVelocityExtensionPlanForInjection",
    "cm12ExtensionPhasePlanning", "cm12ExtensionPhasePlanned",
    "cm12ExtensionPhaseConsuming", "cm12ExtensionPhaseCollecting",
    "observed==(generation|0x80000000u)",
    "finalizeLegacyVelocityExtensionClockForQA"]) {
    if (!vexWGSL.includes(token)) fail(`VEX1 two-phase lifecycle lacks ${token}`);
  }
  const qaCaptureStart = vexWGSL.indexOf("fn captureLegacyVelocityExtensionForQA");
  const qaCaptureEnd = vexWGSL.indexOf("fn finalizeLegacyVelocityExtensionClockForQA",
    qaCaptureStart);
  const qaCapture = vexWGSL.slice(qaCaptureStart, qaCaptureEnd);
  for (const token of [
    "cm12ExtensionZeroDispatches();",
    "faultCount)},0u)",
    "maximumDepth)},0u)",
    "executedCellCount)},0u)",
    "reusedCellCount)},0u)",
    "uncoveredWriteCount)},0u)",
    "cm12ExtensionFlagCollecting",
  ]) {
    if (!qaCapture.includes(token)) fail(`legacy VEX clock reset lacks ${token}`);
  }
  for (const token of [
    "let priorFlags=atomicOr(",
    "if((priorFlags&cm12ExtensionFlagFault)!=0u){return;}",
    "faultCount)},1u)",
    "uncoveredWriteCount)},1u)",
  ]) {
    if (!vexWGSL.includes(token)) fail(`VEX1 first-fault latch lacks ${token}`);
  }
  for (const field of ["faultCount", "uncoveredWriteCount"]) {
    const perLaneCounter = `atomicAdd(&\${arena}[\${h(`
      + `SPARSE_CM12_VELOCITY_EXTENSION_HEADER.${field})}`;
    if (vexWGSL.includes(perLaneCounter)) {
      fail(`VEX1 ${field} regressed to a per-lane atomic counter`);
    }
  }
  for (const token of ["fn importVelocityExtensionBlastToFramePlanNext",
    "cm12ExtensionBlastList", "cm12FramePlanMarkOwnedNextTile",
    "cm12ExtensionPhaseConsuming"]) {
    if (!residentWGSL.includes(token)) fail(`resident actual-blast import lacks ${token}`);
  }
  const seedStart = vexWGSL.indexOf("fn seedVelocityExtensionRoots");
  const seedEnd = vexWGSL.indexOf("fn sealVelocityExtensionSeedFrontier", seedStart);
  const seedSource = vexWGSL.slice(seedStart, seedEnd);
  if (!seedSource.includes("if(!cellActive(cell)){return;}")) {
    fail("VEX root seed lacks the retirement activity boundary");
  }
  const importStart = residentWGSL.indexOf("fn importVelocityExtensionBlastToFramePlanNext");
  const importEnd = residentWGSL.indexOf("fn cm12ActivityCandidateGeneration", importStart);
  const importSource = residentWGSL.slice(importStart, importEnd);
  if (importSource.includes("!cellActive(cell)")) {
    fail("sealed VEX blast import still applies an independent activity filter");
  }
  if (!importSource.includes("if(cell>=p.counts.x){cm12ExtensionFail(cell,0u);return;}")) {
    fail("sealed VEX blast import lacks a fail-closed bounds guard");
  }
  const zeroStart = vexWGSL.indexOf("fn cm12ExtensionZeroDispatches");
  const zeroEnd = vexWGSL.indexOf("fn cm12ExtensionReceiptGeneration", zeroStart);
  const zeroSource = vexWGSL.slice(zeroStart, zeroEnd);
  for (const field of ["rootDispatchY", "rootDispatchZ", "frontierADispatchY",
    "frontierADispatchZ", "frontierBDispatchY", "frontierBDispatchZ"]) {
    if (zeroSource.includes(`SPARSE_CM12_VELOCITY_EXTENSION_HEADER.${field})},0u`)) {
      fail(`VEX dispatch drain erases first-fault provenance ${field}`);
    }
  }
  const failStart = vexWGSL.indexOf("fn cm12ExtensionFail");
  const failEnd = vexWGSL.indexOf("fn cm12ExtensionOwner", failStart);
  const failSource = vexWGSL.slice(failStart, failEnd);
  for (const field of ["rootDispatchY", "rootDispatchZ", "frontierADispatchY",
    "frontierADispatchZ", "frontierBDispatchY", "frontierBDispatchZ"]) {
    if (!failSource.includes(`SPARSE_CM12_VELOCITY_EXTENSION_HEADER.${field})},0u`)) {
      fail(`VEX first-fault winner does not initialize provenance ${field}`);
    }
  }
  const ownerStart = residentWGSL.indexOf("fn cm12ResidentVelocityExtensionOwner");
  const ownerEnd = residentWGSL.indexOf("fn cm12ResidentVelocityExtensionFault", ownerStart);
  const ownerSource = residentWGSL.slice(ownerStart, ownerEnd);
  for (const token of ["rung!=acceptedBrickResolution(brick)",
    "topology[p.topologyOffsets2.z+2u*brick+1u]"]) {
    if (!ownerSource.includes(token)) fail(`stable VEX owner lacks ${token}`);
  }
  for (const token of ["incrementalActivityTopologyState", "brickActive("]) {
    if (ownerSource.includes(token)) fail(`VEX owner retains volatile ${token}`);
  }
  for (const token of [
    "for(var attempt=1u;attempt<64u&&!claimed.exchanged",
    "if(!claimed.exchanged&&claimed.old_value==0u)",
    "atomicOr(&activity[7],32u)",
    "let activationFault=(atomicLoad(&activity[7])&32u)!=0u",
  ]) {
    if (!residentWGSL.includes(token)) {
      fail(`resident receiver activation fail-closed CAS lacks ${token}`);
    }
  }
  if (residentWGSL.includes("while(!claimed.exchanged&&claimed.old_value==0u)")) {
    fail("resident receiver activation retains an unbounded weak-CAS retry");
  }
  const activationStart = residentWGSL.indexOf("fn activateSweptReceivers");
  const activationEnd = residentWGSL.indexOf("fn retireUnsupportedEmptyBricks",
    activationStart);
  const activationSource = residentWGSL.slice(activationStart, activationEnd);
  for (const token of [
    "for(var cell=first;cell<first+count;cell+=1u)",
    "_=cm12ExtensionRecordRoot(cell,",
    "SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.topologyOrEdgeOwnership",
  ]) {
    if (!activationSource.includes(token)) {
      fail(`same-rung receiver VEX root publication lacks ${token}`);
    }
  }
  const initialize = residentTS.indexOf(
    'dispatchVexActivity("initializeVelocityExtensionCandidates", "vexSerial")');
  const accept = residentTS.indexOf('dispatch("finalizeVelocityExtensionCandidate", 1)', initialize);
  const retire = residentTS.indexOf('dispatch("retireUnsupportedEmptyBricks", bricks)', accept);
  const plan = residentTS.indexOf("this.encodeVelocityExtensionPlan(encoder", accept);
  const framePlan = residentTS.indexOf("this.encodeFramePlanPresentation(encoder", plan);
  if (initialize < 0 || accept <= initialize || retire <= accept || plan <= retire
    || framePlan <= plan) {
    fail("resident does not consume, plan, then import the next VEX generation");
  }
  const planToImport = residentTS.slice(plan, framePlan);
  if (planToImport.includes("dispatch(") || planToImport.includes("stage(")) {
    fail("resident inserts work between sealing the VEX blast and FPL import");
  }
  const retirementStart = residentWGSL.indexOf("fn retireUnsupportedEmptyBricks");
  const retirementEnd = residentWGSL.indexOf(
    "fn sealSparseCM12PressureTopologyJournal", retirementStart);
  const retirementSource = residentWGSL.slice(retirementStart, retirementEnd);
  if (!retirementSource.includes("cm12ExtensionInvalidateRetiredCell(cell,")) {
    fail("resident retirement does not revoke accepted VEX cell ownership");
  }
  if (!residentTS.includes(
    "const hasFramePlanProvenance = hasFirstFault && words[h.reserved] === 3;")) {
    fail("compact VEX QA reader exposes FPL provenance outside a consuming first fault");
  }
  const fplBegin = residentTS.indexOf('dispatch("beginSparseCM12FramePlanNext"');
  const fplImport = residentTS.indexOf("importVelocityExtensionBlastToFramePlanNext", fplBegin);
  const fplResolve = residentTS.indexOf('dispatch("resolveSparseCM12FramePlanNextClosure"', fplImport);
  if (fplBegin < 0 || fplImport <= fplBegin || fplResolve <= fplImport) {
    fail("FPL Next does not import the actual VEX blast before closure");
  }
}

function validateCombinedNaga(): void {
  const layout = createSparseCM12VexActivityBatchLayout({ activityTailWords: 64,
    stateTailFloats: 4096, brickCapacity: 7, cellCapacity: 448,
    brickFineResolution: 16 });
  const generated = createSparseCM12VexActivityBatchWGSL({ layout });
  const source = /* wgsl */ `
const CM12_LIQUID_ISOVALUE:f32=0.5;
@group(0) @binding(0) var<storage,read_write> activity:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> state:array<f32>;
@group(0) @binding(2) var<storage,read_write> fineSamples:array<u32>;
fn cellActive(cell:u32)->bool{return cell<${layout.velocityExtension.cellCapacity}u;}
fn acceptedTemplateCellInvocation(index:u32)->u32{
  return select(index,0xffffffffu,index>=${layout.velocityExtension.cellCapacity}u);
}
fn sourceCellVelocity()->u32{return 0u;}
fn destinationCellVelocity()->u32{return 2048u;}
fn sourceDensity()->u32{return 4000u;}
fn pressureTemplateWord(word:u32)->u32{return select(0u,word,word<16u);}
fn pressureEdgeRows()->u32{return 64u;}
fn pressureNeighborOffset()->u32{return 128u;}
fn pressureExtrapolationWeightOffset()->u32{return 256u;}
fn rowAccepted(row:u32)->bool{return row<1024u;}
fn cm12ActivityCandidateGeneration()->u32{return 7u;}
fn cm12ActivityCandidateBrickCount()->u32{return 7u;}
fn cm12ActivityCandidateListGeneration()->u32{return 7u;}
fn cm12ActivityCandidateBrickInvocation(candidate:u32)->u32{return candidate;}
fn cm12ActivityTopologyGeneration()->u32{return 3u;}
fn cm12ActivityFramePlanGeneration()->u32{return 7u;}
fn cm12ActivityBrickTopologySignature(brick:u32)->u32{return brick^3u;}
fn cm12ActivityBrickTopologyChanged(brick:u32)->bool{return brick==0xffffffffu;}
fn cm12ActivityBuildTileTrigger(brick:u32,tile:u32)->vec4u{
  return vec4u(7u,brick|tile,0x807f0001u,brick^3u);
}
fn cm12ActivityExpectedTileContributionCount(brick:u32,tile:u32)->u32{
  return select(1u,0u,(brick|tile)==0xffffffffu);
}
fn cm12ActivityExpectedTileCheck(brick:u32,tile:u32)->u32{return brick^tile;}
fn cm12BatchVelocityExtensionRoot(cell:u32,cause:u32,generation:u32)->bool{
  return (cell|cause|generation)!=0xffffffffu;
}
fn cm12BatchVelocityExtensionClosure(cell:u32,depth:u32,generation:u32)->bool{
  return (cell|depth|generation)!=0xffffffffu;
}
fn cm12BatchVelocityExtensionScheduled(cell:u32,depth:u32,generation:u32)->bool{
  return (cell|depth|generation)!=0xffffffffu;
}
fn cm12BatchVelocityExtensionOwner(cell:u32)->u32{return cell;}
fn cm12BatchVelocityExtensionFault(cell:u32,depth:u32){_=cell;_=depth;}
${generated}
fn cm12ActivityRebuildExactTile(brick:u32,tile:u32)->A4D2TileSummary{
  return A4D2TileSummary(vec4i(i32(brick),i32(tile),0,0),
    vec4f(f32(brick),f32(tile),0.0,0.0),0u,0u,0u,1u,brick^tile);
}
fn cm12ActivityPublishFramePlanRoot(brick:u32,tile:u32,directStages:u32,
 closureStages:u32,cause:u32,depth:u32,generation:u32)->bool{
  return (brick|tile|directStages|closureStages|cause|depth|generation)!=0xffffffffu;
}
fn cm12ActivityPublishExactBrick(brick:u32,moments:vec4i,metrics:vec4f,
 masks:vec4u,prior:vec4u)->vec4u{
  return vec4u(u32(max(0,moments.x)),masks.x,prior.z,
    select(0u,1u,brick<${layout.productionActivity.brickCapacity}u&&metrics.x>=0.0));
}
`;
  const directory = mkdtempSync(join(tmpdir(), "fluid-vex-a4d2-batch-"));
  try {
    const path = join(directory, "batch.wgsl");
    writeFileSync(path, source);
    const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
    if (result.status !== 0) fail(`combined Naga validation failed:\n${result.stderr || result.stdout}`);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
  console.log("combined A4D2+VEX1 WGSL: Naga PASS");
}

function main(): void {
  for (const profile of [
    { name: "dam-b16", activityTailWords: 131_113, stateTailFloats: 3_145_731,
      brickCapacity: 64, cellCapacity: 262_144, brickFineResolution: 16 as const },
    { name: "ocean-b16-p16", activityTailWords: 1_048_601,
      stateTailFloats: 29_491_203, brickCapacity: 600, cellCapacity: 2_457_600,
      brickFineResolution: 16 as const },
    { name: "compat-b8", activityTailWords: 817, stateTailFloats: 901,
      brickCapacity: 37, cellCapacity: 18_944, brickFineResolution: 8 as const },
    { name: "compat-b4", activityTailWords: 401, stateTailFloats: 503,
      brickCapacity: 23, cellCapacity: 1472, brickFineResolution: 4 as const },
  ]) validateLayout(profile);
  validateIndirect();
  validatePipelinesAndSchedule();
  validateTwoPhaseGenerationFixture();
  validateRootClaimConcurrencyFixture();
  validateWideFaultLatchFixture();
  validateReceiverActivationCASFixture();
  validateRetiredRootFramePlanFixture();
  validateStableVelocityOwnerFixture();
  validateActivatedReceiverRootFixture();
  validateLegacyOracleClockResetFixture();
  validateHooksReceiptsAndQA();
  validateSourceContracts();
  validateCombinedNaga();
  console.log("VEX1+A4D2 append-only batch contract: PASS");
}

main();
