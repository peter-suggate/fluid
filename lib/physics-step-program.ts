/**
 * Declarative physics step program.
 *
 * One solver advance is a fixed sequence of encode stages with explicit data
 * dependencies. Historically that sequence lived implicitly in `advanceTo`'s
 * control flow, and the two drivers (the Dawn smoke harness and the web UI
 * renderer) could only be compared by reading code. This module makes the
 * program explicit data:
 *
 * - every stage declares what it reads and writes, including `carriedReads` —
 *   values produced by the PREVIOUS step (the pipelined hand-offs, e.g. the
 *   topology candidate built at a step's tail and committed at the next
 *   step's head);
 * - the declared order is validated against the dependencies (a stage may not
 *   read a value no earlier stage of this step wrote unless it is carried);
 * - the program serializes to stable JSON (`physicsStepProgramJSON`) so the
 *   contract can be persisted and diffed (`docs/physics-step-program.json`);
 * - `stepSequenceDeviations` compares an executed stage-id sequence against
 *   the program, so BOTH drivers can assert — cheaply, every step, profiled
 *   or not — that the solver ran exactly the declared procedure.
 *
 * The encode closures themselves stay in the solver; a stage id is the
 * binding between this declaration and the code that implements it. What the
 * drivers may vary is documented in `driverContract`: everything else is the
 * solver's, inside one command buffer, in this order.
 */

export interface PhysicsStepStage {
  /** Stable stage id; the solver reports these to the sequence recorder. */
  readonly id: string;
  readonly label: string;
  /** Roll-up phase matching the GPU timestamp taxonomy. */
  readonly phase: string;
  /** Values this stage consumes that an earlier stage of this step produced. */
  readonly reads: readonly string[];
  /** Values produced by the previous step's stages (pipelined hand-offs). */
  readonly carriedReads: readonly string[];
  readonly writes: readonly string[];
  /** Present only under a condition the driver does not control (e.g. rigid
   * bodies in the scene). Optional stages may be skipped, never reordered. */
  readonly optional?: boolean;
  /**
   * Conductor-conditional stage (P0.5 /
   * docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A5.1). Names a predicate declared in
   * `PhysicsStepProgram.predicates`. The recorder accepts this stage's ABSENCE
   * for a step only when the driver recorded that predicate TRUE for the same
   * step, and the recorded predicate value becomes part of the step record.
   *
   * The stage may always be present: encoding more than predicted is legal,
   * encoding less is what needs evidence. That asymmetry is the whole safety
   * argument for host-shaped launches — see `PhysicsStepPredicate`.
   */
  readonly condition?: string;
}

/**
 * A driver-evaluated predicate that may delete a stage's encode.
 *
 * A predicate is only sound if its staleness is CONSERVATIVE: reading an old
 * snapshot may cause the driver to encode work that turned out unnecessary,
 * never to delete work that turned out live. `zeroWorkWitness` names the GPU
 * counter that must read zero in the step's OWN snapshot record; the
 * predicted-work check (`physicsStepPredictionFailures`) fails the step
 * otherwise, which is what keeps every Part-B skip honest.
 */
export interface PhysicsStepPredicate {
  readonly id: string;
  readonly description: string;
  readonly zeroWorkWitness: PhysicsStepWorkWitness;
  /** Prose statement of why a stale TRUE cannot delete live work. */
  readonly stalenessArgument: string;
}

/** GPU counters a predicate can be held to, decoded from the step record. */
export type PhysicsStepWorkWitness =
  | "executed-solve-iterations"
  | "spgrid-level-delta"
  | "topology-epoch-advance"
  | "fine-active-bricks"
  | "air-support-error";

export interface PhysicsStepProgram {
  readonly id: string;
  readonly description: string;
  /** Driver obligations that keep both lanes running the same procedure. */
  readonly driverContract: readonly string[];
  /** Predicates stages may name in `condition`. Empty until Part B lands.
   * Optional so a program with no conductor-conditional stage need not carry
   * the field; a `condition` naming an undeclared predicate is a validation
   * failure either way. */
  readonly predicates?: readonly PhysicsStepPredicate[];
  readonly stages: readonly PhysicsStepStage[];
}

const stage = (
  id: string,
  label: string,
  phase: string,
  reads: readonly string[],
  carriedReads: readonly string[],
  writes: readonly string[],
  optional?: boolean,
  condition?: string,
): PhysicsStepStage => Object.freeze({
  id, label, phase, reads: Object.freeze([...reads]),
  carriedReads: Object.freeze([...carriedReads]), writes: Object.freeze([...writes]),
  ...(optional ? { optional: true } : {}),
  ...(condition ? { condition } : {}),
});

/**
 * The octree structured lane's advance: what `WebGPUUniformEulerianSimulation
 * .advanceTo` encodes, in order, into ONE command buffer submitted by ONE
 * `queue.submit`. Resource names are logical values, not buffer identities.
 */
export const OCTREE_STEP_PROGRAM: PhysicsStepProgram = Object.freeze({
  id: "octree-structured-step",
  description:
    "One fixed-dt advance of the octree structured lane. The active topology "
    + "epoch is immutable for the whole step: the previous step's validated "
    + "candidate is committed at the head, and this step's candidate is built "
    + "at the tail for the next step to commit.",
  driverContract: Object.freeze([
    "A driver requests whole advances only; stage selection, ordering, and substepping belong to the solver.",
    "One advance encodes exactly one command buffer and one queue.submit; drivers never split or merge steps.",
    "Drivers may choose only WHEN to request an advance and how many advances are in flight; both are forbidden from affecting the VALUES the physics reads.",
    "Launch-shape carve-out (P0.5): a driver may delete an encode it can prove does zero work, declared as a conductor-conditional stage whose predicate the driver records per step. It may never delete a live encode, and every predicate must be conservative under staleness (a stale read encodes more, never less).",
    "dt policy: the target clock advances in whole fixed steps (controller collapse / harness checkpoint), so every advance sees dt = maxDt exactly.",
    "Diagnostics consume the end-of-step snapshot record (step-snapshot stage); no consumer may race live solver buffers with an independent readback encoder.",
    "No host scheduling decision inside the step may depend on a GPU readback (hostSchedulingUsesReadback = false).",
    "Performance instrumentation must be sequence-neutral: traced and untraced advances submit the same command graph, and scheduling inputs must exist identically whether instrumentation is on or off.",
  ]),
  // No stage is conductor-conditional yet: P0.5 lands the mechanism, Part B
  // (B1 fine-seed chain, B2 air-support publication, B3 coarse-phi entry
  // point, B5 settled tier) declares the predicates that use it. A stage's
  // `condition` must name an id declared here.
  predicates: Object.freeze([]),
  stages: Object.freeze([
    stage("ready-topology-flip",
      "Commit the previous tail's validated topology candidate (or retain the accepted epoch on rejection) and rebuild Section 5 air support against the committed epoch",
      "power-topology",
      [],
      ["candidate-epoch", "candidate-structured-velocity", "candidate-boundary-controls"],
      ["accepted-epoch", "accepted-structured-velocity", "accepted-boundary-controls", "air-support"]),
    stage("surface-transport",
      "Fine and coarse level-set advection with the previous substep's projected, closest-point-extended velocity; fine narrow-band topology, redistance, restriction; fine generation advances",
      "fine-sdf-advection",
      ["accepted-epoch", "accepted-structured-velocity", "air-support"],
      ["projected-velocity"],
      ["fine-level-set", "fine-generation", "coarse-level-set"]),
    stage("pressure-projection",
      "Power descriptor topology, structured advection + boundary RHS, physical-volume capture, pressure row assembly, MGPCG solve, structured projection + CPT seeds, projection tail publication",
      "pressure-solve",
      ["accepted-epoch", "accepted-structured-velocity", "accepted-boundary-controls", "fine-level-set", "coarse-level-set"],
      [],
      ["projected-velocity", "pressure", "projection-energy-stats", "solve-stats"]),
    stage("inactive-topology-candidate",
      "Build and cross-validate the next step's coupled topology/velocity/boundary candidate from this step's surface and projection; a poisoned candidate leaves the next flip retaining the current epoch",
      "coarse-grid",
      ["fine-level-set", "projected-velocity", "pressure"],
      [],
      ["candidate-epoch", "candidate-structured-velocity", "candidate-boundary-controls"]),
    stage("rigid-exchange",
      "Rigid-body impulse exchange and integration",
      "other",
      ["projected-velocity", "pressure"],
      [],
      ["rigid-state"],
      true),
    stage("sparse-brick-world",
      "Publish the final substep's resident fields into the shared sparse-brick render world",
      "adaptive-publication",
      ["fine-level-set", "projected-velocity"],
      [],
      ["render-world"]),
    stage("step-snapshot",
      "Copy the accepted structured/boundary controls, fine worklist header, stats words, topology-epoch state, SPGrid per-level setup delta, air-support failure words, and fine-transport governor schedule into the step-coherent snapshot ring (the only sanctioned diagnostics source)",
      "other",
      ["accepted-structured-velocity", "accepted-boundary-controls", "fine-generation", "solve-stats", "projection-energy-stats", "accepted-epoch", "candidate-epoch", "air-support", "fine-level-set"],
      [],
      ["step-snapshot-record"]),
  ]),
});

/** A stage the driver or the scene may legally leave out of a step. */
function isSkippable(entry: PhysicsStepStage): boolean {
  return entry.optional === true || entry.condition !== undefined;
}

/** Structural validation: unique ids, every read satisfied by an earlier
 * stage of the same step (carried reads are satisfied by the previous step),
 * declared predicates for every `condition`, and — the P0.5 addition — no
 * unguarded consumer of a skippable stage's output inside the same step. */
export function validatePhysicsStepProgram(program: PhysicsStepProgram): readonly string[] {
  const failures: string[] = [];
  const seen = new Set<string>();
  const produced = new Set<string>();
  const producedAnywhere = new Set(program.stages.flatMap((entry) => [...entry.writes]));
  const predicates = new Set((program.predicates ?? []).map((entry) => entry.id));
  const predicateIds = new Set<string>();
  for (const predicate of program.predicates ?? []) {
    if (predicateIds.has(predicate.id)) failures.push(`duplicate predicate id: ${predicate.id}`);
    predicateIds.add(predicate.id);
  }
  /** value name -> the condition guarding the stage that writes it. */
  const skippableWrites = new Map<string, string>();
  for (const entry of program.stages) {
    if (seen.has(entry.id)) failures.push(`duplicate stage id: ${entry.id}`);
    seen.add(entry.id);
    if (entry.condition !== undefined && !predicates.has(entry.condition)) {
      failures.push(`stage ${entry.id} names undeclared predicate "${entry.condition}"`);
    }
    for (const read of entry.reads) {
      if (!produced.has(read)) {
        failures.push(`stage ${entry.id} reads "${read}" before any earlier stage writes it`);
      }
      // A skipped producer must not silently hand a consumer last step's
      // value: either the consumer is skipped under the same predicate, or it
      // declares the value carried (an explicit cross-step hand-off).
      const guard = skippableWrites.get(read);
      if (guard !== undefined && entry.condition !== guard && !isSkippable(entry)) {
        failures.push(
          `stage ${entry.id} reads "${read}" from skippable stage output without sharing its guard`);
      }
    }
    for (const carried of entry.carriedReads) {
      if (!producedAnywhere.has(carried)) {
        failures.push(`stage ${entry.id} carries "${carried}" that no stage of the program produces`);
      }
    }
    for (const write of entry.writes) {
      produced.add(write);
      if (isSkippable(entry)) skippableWrites.set(write, entry.condition ?? entry.id);
      else skippableWrites.delete(write);
    }
  }
  return failures;
}

/** Stable, diffable serialization for persisting the contract as JSON. */
export function physicsStepProgramJSON(program: PhysicsStepProgram): string {
  return `${JSON.stringify(program, null, 2)}\n`;
}

/** Predicate values the driver recorded for one step, id -> value. */
export type PhysicsStepConditionValues = Readonly<Record<string, boolean>>;

/**
 * Compare an executed stage-id sequence against the program. Optional stages
 * may be absent; a conductor-conditional stage may be absent only when
 * `conditions` records its predicate true; nothing may be reordered,
 * repeated, or unknown. Both drivers can afford this every step: it is an
 * array walk over ≤8 strings.
 */
export function stepSequenceDeviations(
  executed: readonly string[],
  program: PhysicsStepProgram,
  conditions: PhysicsStepConditionValues = {},
): readonly string[] {
  const failures: string[] = [];
  const declared = program.stages;
  // Absence is a deviation unless the stage is scene-optional, or its declared
  // predicate was recorded true for THIS step. An unrecorded predicate reads
  // false: silence can never authorize deleting an encode.
  const absenceAllowed = (candidate: PhysicsStepStage): boolean => {
    if (candidate.optional) return true;
    if (candidate.condition === undefined) return false;
    return conditions[candidate.condition] === true;
  };
  const absenceFailure = (candidate: PhysicsStepStage, before?: string): string => {
    const where = before === undefined ? "was never executed" : `was skipped before ${before}`;
    return candidate.condition === undefined
      ? `required stage ${candidate.id} ${where}`
      : `conditional stage ${candidate.id} ${where} without predicate ${candidate.condition}`;
  };
  let cursor = 0;
  for (const id of executed) {
    let matched = false;
    while (cursor < declared.length) {
      const candidate = declared[cursor];
      cursor += 1;
      if (candidate.id === id) { matched = true; break; }
      if (!absenceAllowed(candidate)) failures.push(absenceFailure(candidate, id));
    }
    if (!matched) failures.push(`unexpected or out-of-order stage: ${id}`);
  }
  while (cursor < declared.length) {
    const candidate = declared[cursor];
    cursor += 1;
    if (!absenceAllowed(candidate)) failures.push(absenceFailure(candidate));
  }
  for (const id of Object.keys(conditions)) {
    if (!(program.predicates ?? []).some((entry) => entry.id === id)) {
      failures.push(`driver recorded undeclared predicate: ${id}`);
    }
  }
  return failures;
}

/**
 * Allocation-light per-step recorder the solver fills while encoding.
 *
 * `recordCondition` is the driver's half of the launch-shape carve-out: it
 * states, for this step, which predicate it evaluated and from which snapshot
 * step it read the evidence. Both travel into the step record, so a skip is
 * always attributable to a specific observation.
 */
export class StepSequenceRecorder {
  private readonly executed: string[] = [];
  private readonly conditions: PhysicsStepConditionRecord[] = [];
  record(id: string) { this.executed.push(id); }
  /** Declare a conductor predicate's value and the snapshot step it came from
   * (`sourceStep` -1 when no snapshot had mapped yet). */
  recordCondition(id: string, value: boolean, sourceStep = -1) {
    this.conditions.push(Object.freeze({ id, value, sourceStep }));
  }
  /** Predicate records accumulated so far this step. */
  get recordedConditions(): readonly PhysicsStepConditionRecord[] { return this.conditions; }
  /** The predicate values as the sequence check consumes them. */
  conditionValues(): PhysicsStepConditionValues {
    const values: Record<string, boolean> = {};
    for (const entry of this.conditions) values[entry.id] = entry.value;
    return values;
  }
  /** Stage ids the program declares that this step did not encode. */
  skippedStageIds(program: PhysicsStepProgram): readonly string[] {
    const executed = new Set(this.executed);
    return program.stages
      .filter((entry) => entry.condition !== undefined && !executed.has(entry.id))
      .map((entry) => entry.id);
  }
  /** Validate and reset for the next step; returns deviations (empty = conformant). */
  finishStep(program: PhysicsStepProgram): readonly string[] {
    const failures = stepSequenceDeviations(this.executed, program, this.conditionValues());
    this.executed.length = 0;
    this.conditions.length = 0;
    return failures;
  }
}

/** One predicate the driver evaluated, with the evidence it read it from. */
export interface PhysicsStepConditionRecord {
  readonly id: string;
  readonly value: boolean;
  /** Snapshot step the value was read from; -1 when none had mapped. */
  readonly sourceStep: number;
}

/**
 * What the driver predicted when it shaped step N's encode (P0.5 / A5.2).
 *
 * Recorded synchronously at encode time and resolved asynchronously when step
 * N's own snapshot maps — a lag-k check, where k is the pipeline depth. It is
 * a pure audit: nothing in the step consults it, so it cannot feed back into
 * physics.
 */
export interface PhysicsStepPrediction {
  readonly step: number;
  /** Outer solve iterations this step's command buffer actually encoded. */
  readonly encodedSolveBudget: number;
  readonly conditions: readonly PhysicsStepConditionRecord[];
  /** Conductor-conditional stages this step deleted. */
  readonly skippedStageIds: readonly string[];
}

/**
 * GPU counters for step N, decoded from step N's own snapshot record. Every
 * field is optional: `undefined` means the record segment was not copied, and
 * an uncopied segment is NEVER evidence of zero work.
 */
export interface PhysicsStepWorkCounters {
  readonly step: number;
  readonly executedSolveIterations?: number;
  readonly solveConverged?: boolean;
  readonly topologyFlipReady?: boolean;
  readonly topologyEpochError?: number;
  readonly spgridLevelDirty?: readonly number[];
  readonly fineActiveBricks?: number;
  readonly airSupportErrorFlags?: number;
}

function witnessIsZeroWork(
  witness: PhysicsStepWorkWitness,
  counters: PhysicsStepWorkCounters,
): boolean | undefined {
  switch (witness) {
    case "executed-solve-iterations":
      return counters.executedSolveIterations === undefined
        ? undefined : counters.executedSolveIterations === 0;
    case "spgrid-level-delta":
      return counters.spgridLevelDirty === undefined
        ? undefined : counters.spgridLevelDirty.every((word) => word === 0);
    case "topology-epoch-advance":
      return counters.topologyFlipReady === undefined
        ? undefined : counters.topologyFlipReady === false;
    case "fine-active-bricks":
      return counters.fineActiveBricks === undefined
        ? undefined : counters.fineActiveBricks === 0;
    case "air-support-error":
      return counters.airSupportErrorFlags === undefined
        ? undefined : counters.airSupportErrorFlags === 0;
  }
}

/**
 * The lag-k honesty check: did the GPU agree that the deleted work was zero,
 * and did the encoded budget cover what the solve actually ran?
 *
 * Empty means the prediction held. Non-empty is a hard failure in harness
 * lanes and an ALERT in the browser diagnostics card — never a silent
 * correction, because a prediction that missed once means the predicate is
 * unsound, not that this step was unlucky.
 */
export function physicsStepPredictionFailures(
  prediction: PhysicsStepPrediction,
  counters: PhysicsStepWorkCounters,
  program: PhysicsStepProgram,
): readonly string[] {
  const failures: string[] = [];
  if (prediction.step !== counters.step) {
    failures.push(`prediction step ${prediction.step} compared against counters for step ${counters.step}`);
    return failures;
  }
  const executed = counters.executedSolveIterations;
  if (executed !== undefined) {
    if (executed > prediction.encodedSolveBudget) {
      failures.push(`solve executed ${executed} iterations against an encoded budget of ${prediction.encodedSolveBudget}`);
    } else if (executed >= prediction.encodedSolveBudget && counters.solveConverged === false) {
      // The seed-publish cliff: an exhausted budget publishes the seed
      // pressure and nothing else fails the step.
      failures.push(`solve exhausted its encoded budget of ${prediction.encodedSolveBudget} without converging`);
    }
  }
  const byId = new Map(program.stages.map((entry) => [entry.id, entry]));
  const predicates = new Map((program.predicates ?? []).map((entry) => [entry.id, entry]));
  const recorded = new Map(prediction.conditions.map((entry) => [entry.id, entry]));
  for (const stageId of prediction.skippedStageIds) {
    const declared = byId.get(stageId);
    if (!declared) { failures.push(`skipped unknown stage: ${stageId}`); continue; }
    if (declared.condition === undefined) {
      failures.push(`stage ${stageId} was skipped but declares no predicate`);
      continue;
    }
    const condition = recorded.get(declared.condition);
    if (!condition || !condition.value) {
      failures.push(`stage ${stageId} was skipped without recording predicate ${declared.condition}`);
      continue;
    }
    const predicate = predicates.get(declared.condition);
    if (!predicate) {
      failures.push(`predicate ${declared.condition} is not declared by the program`);
      continue;
    }
    const zero = witnessIsZeroWork(predicate.zeroWorkWitness, counters);
    if (zero === undefined) {
      failures.push(`stage ${stageId} was skipped but the record carries no ${predicate.zeroWorkWitness} witness`);
    } else if (!zero) {
      failures.push(`stage ${stageId} was skipped while ${predicate.zeroWorkWitness} reports live work`);
    }
  }
  return failures;
}

/**
 * Small keyed ring of pending predictions. The browser can have up to
 * `maxPendingAdvances` steps in flight and a snapshot maps some steps later,
 * so the ledger must hold at least that many; entries are consumed by step
 * number, and an evicted entry is simply not checked.
 */
export class PhysicsStepPredictionLedger {
  private readonly entries = new Map<number, PhysicsStepPrediction>();

  constructor(private readonly capacity = 16) {}

  record(prediction: PhysicsStepPrediction) {
    this.entries.set(prediction.step, prediction);
    while (this.entries.size > Math.max(1, this.capacity)) {
      const oldest = this.entries.keys().next();
      if (oldest.done) break;
      this.entries.delete(oldest.value);
    }
  }

  /** Consume the prediction for a mapped snapshot's step, if still held. */
  take(step: number): PhysicsStepPrediction | undefined {
    const entry = this.entries.get(step);
    if (entry) this.entries.delete(step);
    return entry;
  }

  get pending() { return this.entries.size; }
  clear() { this.entries.clear(); }
}
