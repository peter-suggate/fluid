/**
 * CPU-only proof harness for sparsifying the static geometric low-flux limiter.
 *
 * This retains the frozen f32 proposal/commit-bank experiment documented by
 * the Rust cutover source manifest. It is a GPU optimization proof harness,
 * not production physics, and does not implement the moving-solid FISTA branch.
 */

export interface StaticLowFluxFace2D {
  readonly negativeCell: number;
  readonly positiveCell: number;
  readonly lowFlux: number;
}

export interface StaticLowFluxIncidence2D {
  readonly face: number;
  readonly negative: boolean;
}

export interface StaticLowFluxProblem2D {
  readonly volumes: Float32Array;
  readonly capacities: Float32Array;
  readonly sourceRates?: Float32Array;
  readonly dt: number;
  readonly faces: readonly StaticLowFluxFace2D[];
  /** Production GV_CELL_FACE order for each accepted cell. */
  readonly incidences: readonly (readonly StaticLowFluxIncidence2D[])[];
  readonly solidMotionActive?: boolean;
  readonly maximumPasses?: number;
}

export type StaticLowFluxMode2D = "dense-copy" | "dense-ping-pong" | "active-frontier";

export interface StaticLowFluxPass2D {
  readonly pass: number;
  readonly evaluatedCells: readonly number[];
  readonly invalidCount: number;
  readonly firstInvalid: number;
  readonly changedCells: readonly number[];
}

export interface StaticLowFluxResult2D {
  readonly mode: StaticLowFluxMode2D;
  readonly converged: boolean;
  readonly passes: number;
  readonly factors: Float32Array;
  readonly lowFlux: Float32Array;
  readonly lowStateVolume: Float32Array;
  readonly passReceipts: readonly StaticLowFluxPass2D[];
  readonly updateCellVisits: number;
  readonly bankCopyCellVisits: number;
  readonly frontierCommitCellVisits: number;
  /** Includes the mandatory full-domain seed pass. */
  readonly peakFrontier: number;
  readonly peakFrontierAfterSeed: number;
}

const f = Math.fround;
const add = (a: number, b: number): number => f(f(a) + f(b));
const mul = (a: number, b: number): number => f(f(a) * f(b));
const div = (a: number, b: number): number => f(f(a) / f(b));

function delta(flux: number, negative: boolean): number {
  return negative ? f(-flux) : f(flux);
}

function receiver(face: StaticLowFluxFace2D): number {
  return face.lowFlux >= 0 ? face.positiveCell : face.negativeCell;
}

function donor(face: StaticLowFluxFace2D): number {
  return face.lowFlux >= 0 ? face.negativeCell : face.positiveCell;
}

function nextLowerPositiveF32(value: number): number {
  const words = new Uint32Array(1);
  const floats = new Float32Array(words.buffer);
  floats[0] = value;
  if (words[0]! > 0) words[0]!--;
  return floats[0]!;
}

function volumeValid(volume: number, capacity: number): boolean {
  const margin = mul(9.5367431640625e-7, capacity);
  return capacity >= 0 && Number.isFinite(capacity)
    && volume >= -margin && volume <= add(capacity, margin);
}

function startingVolume(problem: StaticLowFluxProblem2D, cell: number): number {
  return add(problem.volumes[cell]!, mul(problem.dt, problem.sourceRates?.[cell] ?? 0));
}

function validate(problem: StaticLowFluxProblem2D): void {
  const count = problem.volumes.length;
  if (problem.solidMotionActive) {
    throw new RangeError("active-frontier proof is static-only; use the production dense moving-solid limiter");
  }
  if (problem.capacities.length !== count || problem.incidences.length !== count
    || problem.sourceRates && problem.sourceRates.length !== count) {
    throw new RangeError("static low-flux cell arrays disagree");
  }
  for (let cell = 0; cell < count; cell += 1) {
    if (!volumeValid(problem.volumes[cell]!, problem.capacities[cell]!)) {
      throw new RangeError(`invalid starting volume at cell ${cell}`);
    }
  }
  for (let face = 0; face < problem.faces.length; face += 1) {
    const entry = problem.faces[face]!;
    if (!Number.isFinite(entry.lowFlux)
      || entry.negativeCell >= count || entry.positiveCell >= count
      || entry.negativeCell < -1 || entry.positiveCell < -1) {
      throw new RangeError(`invalid low-flux face ${face}`);
    }
  }
}

function incomingByCell(problem: StaticLowFluxProblem2D): Float32Array {
  const incoming = new Float32Array(problem.volumes.length);
  for (let cell = 0; cell < incoming.length; cell += 1) {
    let total = 0;
    for (const incidence of problem.incidences[cell]!) {
      const original = problem.faces[incidence.face]!.lowFlux;
      total = add(total, Math.max(0, delta(original, incidence.negative)));
    }
    incoming[cell] = total;
  }
  return incoming;
}

interface Proposal {
  readonly next: number;
  readonly invalid: boolean;
}

function propose(problem: StaticLowFluxProblem2D, incoming: Float32Array,
  current: Float32Array, cell: number): Proposal {
  const volume = startingVolume(problem, cell);
  const capacity = problem.capacities[cell]!;
  let signedDelta = 0;
  let outgoing = 0;
  for (const incidence of problem.incidences[cell]!) {
    const face = problem.faces[incidence.face]!;
    const factorOwner = receiver(face);
    const factor = factorOwner >= 0 ? current[factorOwner]! : 1;
    const flux = mul(face.lowFlux, factor);
    signedDelta = add(signedDelta, delta(flux, incidence.negative));
    const originalDelta = delta(face.lowFlux, incidence.negative);
    if (originalDelta < 0) outgoing = add(outgoing, mul(-originalDelta, factor));
  }
  const candidate = add(volume, signedDelta);
  const allowed = volumeValid(volume, capacity) ? Math.max(capacity, volume) : capacity;
  const previous = current[cell]!;
  let next = previous;
  if (candidate > allowed && incoming[cell]! > 0) {
    next = Math.min(previous,
      div(Math.max(0, add(f(allowed - volume), outgoing)), incoming[cell]!));
    next = Math.min(next, Math.max(0,
      f(previous - div(f(candidate - allowed), incoming[cell]!))));
    if (next === previous && previous > 0) next = nextLowerPositiveF32(previous);
  }
  return { next: f(next), invalid: !volumeValid(candidate, capacity) };
}

function finalLowFlux(problem: StaticLowFluxProblem2D,
  factors: Float32Array): Float32Array {
  return Float32Array.from(problem.faces, face => {
    const owner = receiver(face);
    return mul(face.lowFlux, owner >= 0 ? factors[owner]! : 1);
  });
}

function gatheredLowState(problem: StaticLowFluxProblem2D,
  fluxes: Float32Array): Float32Array {
  return Float32Array.from(problem.volumes, (_value, cell) => {
    let amount = startingVolume(problem, cell);
    for (const incidence of problem.incidences[cell]!) {
      amount = add(amount, delta(fluxes[incidence.face]!, incidence.negative));
    }
    return amount;
  });
}

function finish(mode: StaticLowFluxMode2D, problem: StaticLowFluxProblem2D,
  converged: boolean, factors: Float32Array, receipts: StaticLowFluxPass2D[],
  updateCellVisits: number, bankCopyCellVisits: number, frontierCommitCellVisits: number,
  peakFrontier: number, peakFrontierAfterSeed: number): StaticLowFluxResult2D {
  const lowFlux = finalLowFlux(problem, factors);
  return { mode, converged, passes: receipts.length, factors: factors.slice(), lowFlux,
    lowStateVolume: gatheredLowState(problem, lowFlux), passReceipts: receipts,
    updateCellVisits, bankCopyCellVisits, frontierCommitCellVisits,
    peakFrontier, peakFrontierAfterSeed };
}

function solveDense(problem: StaticLowFluxProblem2D,
  mode: "dense-copy" | "dense-ping-pong"): StaticLowFluxResult2D {
  validate(problem);
  const count = problem.volumes.length;
  const incoming = incomingByCell(problem);
  let current = new Float32Array(count).fill(1);
  let proposed = new Float32Array(count);
  const receipts: StaticLowFluxPass2D[] = [];
  let updateCellVisits = 0;
  let bankCopyCellVisits = 0;
  const maximumPasses = problem.maximumPasses ?? 1024;
  for (let pass = 0; pass < maximumPasses; pass += 1) {
    const changed: number[] = [];
    let invalidCount = 0;
    let firstInvalid = -1;
    for (let cell = 0; cell < count; cell += 1) {
      const value = propose(problem, incoming, current, cell);
      proposed[cell] = value.next;
      if (value.next !== current[cell]) changed.push(cell);
      if (value.invalid) {
        invalidCount += 1;
        if (firstInvalid < 0) firstInvalid = cell;
      }
    }
    updateCellVisits += count;
    receipts.push({ pass, evaluatedCells: Array.from({ length: count }, (_, cell) => cell),
      invalidCount, firstInvalid, changedCells: changed });
    if (invalidCount === 0) return finish(mode, problem, true, current, receipts,
      updateCellVisits, bankCopyCellVisits, 0, count, receipts.length > 1 ? count : 0);
    if (pass + 1 >= maximumPasses) return finish(mode, problem, false, current, receipts,
      updateCellVisits, bankCopyCellVisits, 0, count, receipts.length > 1 ? count : 0);
    if (mode === "dense-copy") {
      current.set(proposed);
      bankCopyCellVisits += count;
    } else {
      [current, proposed] = [proposed, current];
    }
  }
  throw new Error("unreachable dense low-flux exit");
}

export function solveStaticLowFluxDenseCopy2D(problem: StaticLowFluxProblem2D):
  StaticLowFluxResult2D {
  return solveDense(problem, "dense-copy");
}

export function solveStaticLowFluxDensePingPong2D(problem: StaticLowFluxProblem2D):
  StaticLowFluxResult2D {
  return solveDense(problem, "dense-ping-pong");
}

export function solveStaticLowFluxFrontier2D(problem: StaticLowFluxProblem2D):
  StaticLowFluxResult2D {
  validate(problem);
  const count = problem.volumes.length;
  const incoming = incomingByCell(problem);
  const current = new Float32Array(count).fill(1);
  const invalid = new Uint8Array(count);
  let active = Array.from({ length: count }, (_, cell) => cell);
  const receipts: StaticLowFluxPass2D[] = [];
  let updateCellVisits = 0;
  let frontierCommitCellVisits = 0;
  let peakFrontier = count;
  let peakFrontierAfterSeed = 0;
  const maximumPasses = problem.maximumPasses ?? 1024;
  for (let pass = 0; pass < maximumPasses; pass += 1) {
    const proposed = new Float32Array(active.length);
    const changed: number[] = [];
    for (let index = 0; index < active.length; index += 1) {
      const cell = active[index]!;
      const value = propose(problem, incoming, current, cell);
      proposed[index] = value.next;
      invalid[cell] = value.invalid ? 1 : 0;
      if (value.next !== current[cell]) changed.push(cell);
    }
    updateCellVisits += active.length;
    let invalidCount = 0;
    let firstInvalid = -1;
    for (let cell = 0; cell < count; cell += 1) if (invalid[cell]) {
      invalidCount += 1;
      if (firstInvalid < 0) firstInvalid = cell;
    }
    receipts.push({ pass, evaluatedCells: active.slice(), invalidCount, firstInvalid,
      changedCells: changed });
    if (invalidCount === 0) return finish("active-frontier", problem, true, current,
      receipts, updateCellVisits, 0, frontierCommitCellVisits,
      peakFrontier, peakFrontierAfterSeed);
    if (pass + 1 >= maximumPasses) return finish("active-frontier", problem, false,
      current, receipts, updateCellVisits, 0, frontierCommitCellVisits, peakFrontier,
      peakFrontierAfterSeed);

    for (let index = 0; index < active.length; index += 1) {
      current[active[index]!] = proposed[index]!;
    }
    frontierCommitCellVisits += active.length;

    const next = new Set<number>();
    for (let cell = 0; cell < count; cell += 1) if (invalid[cell]) next.add(cell);
    for (const changedReceiver of changed) {
      next.add(changedReceiver);
      for (const incidence of problem.incidences[changedReceiver]!) {
        const face = problem.faces[incidence.face]!;
        if (receiver(face) !== changedReceiver) continue;
        const predecessor = donor(face);
        if (predecessor >= 0) next.add(predecessor);
      }
    }
    active = [...next].sort((a, b) => a - b);
    peakFrontier = Math.max(peakFrontier, active.length);
    peakFrontierAfterSeed = Math.max(peakFrontierAfterSeed, active.length);
  }
  throw new Error("unreachable frontier low-flux exit");
}

export function lowFluxLimiterExecution2D(problem: Pick<StaticLowFluxProblem2D,
  "solidMotionActive">): "active-frontier" | "dense-moving-solid" {
  return problem.solidMotionActive ? "dense-moving-solid" : "active-frontier";
}

export function staticLowFluxProblem2D(values: {
  readonly volumes: readonly number[];
  readonly capacities: readonly number[];
  readonly faces: readonly StaticLowFluxFace2D[];
  readonly sourceRates?: readonly number[];
  readonly dt?: number;
  readonly maximumPasses?: number;
}): StaticLowFluxProblem2D {
  const count = values.volumes.length;
  const incidences: StaticLowFluxIncidence2D[][] = Array.from({ length: count }, () => []);
  values.faces.forEach((face, index) => {
    if (face.negativeCell >= 0) incidences[face.negativeCell]!.push({ face: index, negative: true });
    if (face.positiveCell >= 0) incidences[face.positiveCell]!.push({ face: index, negative: false });
  });
  return { volumes: Float32Array.from(values.volumes),
    capacities: Float32Array.from(values.capacities),
    sourceRates: values.sourceRates ? Float32Array.from(values.sourceRates) : undefined,
    dt: values.dt ?? 0, faces: values.faces.map(face => ({ ...face, lowFlux: f(face.lowFlux) })),
    incidences, maximumPasses: values.maximumPasses };
}
