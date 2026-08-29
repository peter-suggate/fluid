import { CLOCK_EPSILON_S, collapseGPUFixedSteps, commitGPUCompletion, gpuCanAcceptNextStep } from "./gpu-clock";

/**
 * The pane that always exists. Single-pane mode is compare mode with one
 * session, so there is no second code path: pane A is registered at
 * construction and can never be unregistered.
 */
export const PRIMARY_PANE_ID = "a";

/**
 * Lockstep pins the transport window to one advance per pane. Two panes each
 * running two deep would let one drift a whole frame ahead of the other
 * *inside* the barrier's tolerance, which is precisely the thing the mode
 * exists to prevent; the throughput lost is the price of the mode.
 */
export const LOCKSTEP_IN_FLIGHT_DEPTH = 1;

interface PaneClockState {
  /** Last GPU state this pane reported complete. */
  completedTime_s: number;
  /** Work this pane admitted to its queue but has not yet reported complete. */
  submittedTime_s?: number;
  /** True once this pane has reported the drain of the current pause. */
  drained?: boolean;
  /** Step this pane's own scene + method pins, when it is not pane A's. */
  declaredStep_s?: number;
}

/** One pane's clock as the host sees it. */
export interface PaneClockReport {
  readonly id: string;
  readonly completedTime_s: number;
  readonly step_s: number;
}

/**
 * The host clock for N panes.
 *
 * There is one target clock and one transport. Each pane reports its own GPU
 * completions; the host's completed time is the *minimum* over the panes, so
 * the slowest pane owns the clock. With one pane registered the arithmetic is
 * exactly the single-pane transport that shipped before compare mode existed —
 * accumulate wall time, collapse whole fixed steps, admit up to the renderer's
 * throughput depth. With more than one, the target advances by exactly one dt
 * and only once every pane has reached it, so no pane can ever be more than
 * one step ahead of the slowest.
 *
 * This owns no store and no device: it is the arithmetic of the barrier, and
 * is driven entirely by the controller.
 */
export class PaneClockHost {
  private readonly panes = new Map<string, PaneClockState>([
    [PRIMARY_PANE_ID, { completedTime_s: 0 }],
  ]);
  private targetTime_s = 0;
  private accumulator_s = 0;
  /** Pane A's effective step, restated by the controller on every tick. */
  private primaryStep_s = 0;

  // ---- registration ------------------------------------------------------

  paneIds(): readonly string[] { return [...this.panes.keys()]; }
  paneCount(): number { return this.panes.size; }
  hasPane(id: string): boolean { return this.panes.has(id); }

  /** True while more than one pane is registered: the barrier is in force. */
  lockstep(): boolean { return this.panes.size > 1; }

  /**
   * Admit a pane to the barrier. A pane joining mid-run inherits the host's
   * current floor rather than starting at zero, so it cannot stall the clock
   * with completions it was never asked for; entering compare mode resets the
   * host anyway, which is where the two panes actually get a common t=0.
   */
  registerPane(id: string): boolean {
    if (this.panes.has(id)) return false;
    this.panes.set(id, { completedTime_s: this.completedTime() });
    return true;
  }

  /** Pane A is the session; it is never unregistered. */
  unregisterPane(id: string): boolean {
    if (id === PRIMARY_PANE_ID) return false;
    return this.panes.delete(id);
  }

  // ---- clocks ------------------------------------------------------------

  /** The target every pane's transport is advancing toward. */
  targetTime(): number { return this.targetTime_s; }

  /** The slowest pane's completion: the time the product may claim to be at. */
  completedTime(): number {
    let minimum = Number.POSITIVE_INFINITY;
    for (const pane of this.panes.values()) minimum = Math.min(minimum, pane.completedTime_s);
    return Number.isFinite(minimum) ? minimum : 0;
  }

  paneCompletedTime(id: string): number | undefined {
    return this.panes.get(id)?.completedTime_s;
  }

  /** Wall-clock debt not yet turned into steps. */
  pendingTime_s(): number { return this.accumulator_s; }

  reports(): readonly PaneClockReport[] {
    return this.paneIds().map((id) => ({
      id,
      completedTime_s: this.panes.get(id)?.completedTime_s ?? 0,
      step_s: this.paneStep_s(id),
    }));
  }

  // ---- step size ---------------------------------------------------------

  /**
   * Declare a pane's own step. Pane A's step arrives with every `advance`
   * (it is the controller's `effectiveSimulationStep_s`), so this is how a
   * second pane whose diff changes `fixedDt_s` or pins a method step tells the
   * host it is not on A's clock. `undefined` puts the pane back on A's step.
   */
  setPaneDt(id: string, dt_s: number | undefined): void {
    const pane = this.panes.get(id);
    if (!pane) return;
    pane.declaredStep_s = dt_s !== undefined && Number.isFinite(dt_s) && dt_s > 0 ? dt_s : undefined;
  }

  /** The step a pane runs at: its own when it declared one, else pane A's. */
  paneStep_s(id: string): number {
    return this.panes.get(id)?.declaredStep_s ?? this.primaryStep_s;
  }

  /**
   * The host steps at the smallest step any pane declares, and lets a
   * larger-dt pane skip the steps it does not need. Visually that is no longer
   * a paired step, which is why `panesDtDiffer` exists for the diff strip to
   * say so.
   */
  stepSize_s(): number {
    let step = this.primaryStep_s;
    for (const id of this.panes.keys()) {
      const pane = this.paneStep_s(id);
      if (pane > 0 && (step <= 0 || pane < step)) step = pane;
    }
    return step;
  }

  /** True when some registered pane is not on pane A's step. */
  panesDtDiffer(): boolean {
    const primary = this.paneStep_s(PRIMARY_PANE_ID);
    for (const id of this.panes.keys()) {
      if (Math.abs(this.paneStep_s(id) - primary) > CLOCK_EPSILON_S) return true;
    }
    return false;
  }

  // ---- admission ---------------------------------------------------------

  /**
   * The barrier. Identical to the single-pane admission rule read against the
   * host minimum: the target may stand at most one step beyond the slowest
   * pane's completed state.
   */
  canAcceptNextStep(): boolean {
    return gpuCanAcceptNextStep(this.targetTime_s, this.completedTime());
  }

  /**
   * Turn wall-clock elapsed time into target-clock steps. Returns the steps
   * taken so the caller can apply per-step host work exactly once.
   */
  advance(elapsed_s: number, primaryStep_s: number): number {
    this.primaryStep_s = primaryStep_s;
    const dt = this.stepSize_s();
    this.accumulator_s += Math.max(0, Number.isFinite(elapsed_s) ? elapsed_s : 0);
    const collapsed = collapseGPUFixedSteps(this.accumulator_s, dt);
    if (collapsed.steps <= 0) return 0;
    if (!this.lockstep()) {
      // Single pane: collapse every owed tick at once. Exact, because no host
      // fluid/body evolution occurs at the intermediate ticks.
      this.accumulator_s = collapsed.remainder_s;
      this.targetTime_s += collapsed.steps * dt;
      this.clearSubmittedFloors();
      return collapsed.steps;
    }
    if (!this.canAcceptNextStep()) return 0;
    // One step, and only one: the debt beyond it is dropped rather than
    // banked, so a pane that recovers cannot sprint through a backlog and
    // break the pairing the mode exists for.
    this.accumulator_s = collapsed.remainder_s;
    this.targetTime_s += dt;
    this.clearSubmittedFloors();
    return 1;
  }

  /** STEP: one paired step for every pane, refused while the barrier is shut. */
  step(primaryStep_s: number): boolean {
    this.primaryStep_s = primaryStep_s;
    if (!this.canAcceptNextStep()) return false;
    this.targetTime_s += this.stepSize_s();
    this.clearSubmittedFloors();
    return true;
  }

  // ---- pane callbacks ----------------------------------------------------

  /**
   * Record one pane's GPU completion. Returns true when the host minimum moved
   * and the product's published time is therefore stale.
   */
  completeAdvance(time_s: number, paneId: string = PRIMARY_PANE_ID): boolean {
    const pane = this.panes.get(paneId);
    if (!pane) return false;
    const completed = commitGPUCompletion(this.targetTime_s, pane.completedTime_s, time_s);
    if (completed === pane.completedTime_s) return false;
    const before = this.completedTime();
    pane.completedTime_s = completed;
    return this.completedTime() !== before;
  }

  /**
   * Drop host-side debt when a pane's transport pauses, retaining only work
   * already admitted to a queue. With two panes the target falls back to the
   * furthest-submitted pane, so nothing that was already encoded is orphaned
   * behind a target that rewound past it.
   *
   * Which is why, in lockstep, the rewind waits for *every* pane to report.
   * The floor is the maximum over the panes and that maximum is unknown until
   * the last of them has said what it drained; a rewind taken on the first
   * report can land below an advance another pane has already encoded. A solver
   * refuses to re-encode a time it has already submitted, so that advance is
   * never re-issued, the completion the barrier is waiting for never arrives,
   * and the clock dies for both panes. Debt is dropped either way — owing wall
   * time across a pause is never right — only the rewind is deferred.
   */
  schedulingPaused(submittedTime_s?: number, paneId: string = PRIMARY_PANE_ID): void {
    const pane = this.panes.get(paneId);
    if (!pane) return;
    pane.drained = true;
    if (submittedTime_s !== undefined && Number.isFinite(submittedTime_s)) {
      pane.submittedTime_s = Math.max(pane.submittedTime_s ?? 0, submittedTime_s);
    }
    this.accumulator_s = 0;
    for (const candidate of this.panes.values()) {
      if (!candidate.drained) return;
    }
    let floor = 0;
    for (const candidate of this.panes.values()) {
      floor = Math.max(floor, candidate.completedTime_s, candidate.submittedTime_s ?? 0);
    }
    // A reset can pause while a renderer still owns the previous solver.
    if (floor > this.targetTime_s + CLOCK_EPSILON_S) return;
    this.targetTime_s = floor;
  }

  // ---- lifecycle ---------------------------------------------------------

  /** Reset every pane to its own t = 0. Registrations and declared steps survive. */
  reset(): void {
    this.targetTime_s = 0;
    this.accumulator_s = 0;
    for (const pane of this.panes.values()) {
      pane.completedTime_s = 0;
      pane.submittedTime_s = undefined;
      pane.drained = false;
    }
  }

  /** Bound host debt to one step, e.g. after the shared step size changed. */
  clampPendingTime(limit_s: number): void {
    this.accumulator_s = Math.min(this.accumulator_s, limit_s);
  }

  /** Forget wall-clock debt: nothing may be owed across a stall or a reset. */
  dropPendingTime(): void { this.accumulator_s = 0; }

  /** The target moved, so every retained drain floor describes a past pause. */
  private clearSubmittedFloors(): void {
    for (const pane of this.panes.values()) {
      pane.submittedTime_s = undefined;
      pane.drained = false;
    }
  }
}
