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
}

export interface PhysicsStepProgram {
  readonly id: string;
  readonly description: string;
  /** Driver obligations that keep both lanes running the same procedure. */
  readonly driverContract: readonly string[];
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
): PhysicsStepStage => Object.freeze({
  id, label, phase, reads: Object.freeze([...reads]),
  carriedReads: Object.freeze([...carriedReads]), writes: Object.freeze([...writes]),
  ...(optional ? { optional: true } : {}),
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
    "Drivers may choose only WHEN to request an advance and how many advances are in flight; both are forbidden from affecting stage order or step content.",
    "dt policy: the target clock advances in whole fixed steps (controller collapse / harness checkpoint), so every advance sees dt = maxDt exactly.",
    "Diagnostics consume the end-of-step snapshot record (step-snapshot stage); no consumer may race live solver buffers with an independent readback encoder.",
    "No host scheduling decision inside the step may depend on a GPU readback (hostSchedulingUsesReadback = false).",
    "Performance instrumentation must be sequence-neutral: traced and untraced advances submit the same command graph, and scheduling inputs must exist identically whether instrumentation is on or off.",
  ]),
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
      "Copy the accepted structured/boundary controls, fine worklist header, and stats words into the step-coherent snapshot ring (the only sanctioned diagnostics source)",
      "other",
      ["accepted-structured-velocity", "accepted-boundary-controls", "fine-generation", "solve-stats", "projection-energy-stats"],
      [],
      ["step-snapshot-record"]),
  ]),
});

/** Structural validation: unique ids, and every read satisfied by an earlier
 * stage of the same step (carried reads are satisfied by the previous step). */
export function validatePhysicsStepProgram(program: PhysicsStepProgram): readonly string[] {
  const failures: string[] = [];
  const seen = new Set<string>();
  const produced = new Set<string>();
  const producedAnywhere = new Set(program.stages.flatMap((entry) => [...entry.writes]));
  for (const entry of program.stages) {
    if (seen.has(entry.id)) failures.push(`duplicate stage id: ${entry.id}`);
    seen.add(entry.id);
    for (const read of entry.reads) {
      if (!produced.has(read)) {
        failures.push(`stage ${entry.id} reads "${read}" before any earlier stage writes it`);
      }
    }
    for (const carried of entry.carriedReads) {
      if (!producedAnywhere.has(carried)) {
        failures.push(`stage ${entry.id} carries "${carried}" that no stage of the program produces`);
      }
    }
    for (const write of entry.writes) produced.add(write);
  }
  return failures;
}

/** Stable, diffable serialization for persisting the contract as JSON. */
export function physicsStepProgramJSON(program: PhysicsStepProgram): string {
  return `${JSON.stringify(program, null, 2)}\n`;
}

/**
 * Compare an executed stage-id sequence against the program. Optional stages
 * may be absent; nothing may be reordered, repeated, or unknown. Both drivers
 * can afford this every step: it is an array walk over ≤8 strings.
 */
export function stepSequenceDeviations(
  executed: readonly string[],
  program: PhysicsStepProgram,
): readonly string[] {
  const failures: string[] = [];
  const declared = program.stages;
  let cursor = 0;
  for (const id of executed) {
    let matched = false;
    while (cursor < declared.length) {
      const candidate = declared[cursor];
      cursor += 1;
      if (candidate.id === id) { matched = true; break; }
      if (!candidate.optional) {
        failures.push(`required stage ${candidate.id} was skipped before ${id}`);
      }
    }
    if (!matched) failures.push(`unexpected or out-of-order stage: ${id}`);
  }
  while (cursor < declared.length) {
    const candidate = declared[cursor];
    cursor += 1;
    if (!candidate.optional) failures.push(`required stage ${candidate.id} was never executed`);
  }
  return failures;
}

/** Allocation-light per-step recorder the solver fills while encoding. */
export class StepSequenceRecorder {
  private readonly executed: string[] = [];
  record(id: string) { this.executed.push(id); }
  /** Validate and reset for the next step; returns deviations (empty = conformant). */
  finishStep(program: PhysicsStepProgram): readonly string[] {
    const failures = stepSequenceDeviations(this.executed, program);
    this.executed.length = 0;
    return failures;
  }
}
