/**
 * Declarative initialization work for GPU resources.
 *
 * The runner is the sole owner of progress accounting. Callers describe the
 * work they are about to perform as tasks; they never pass hand-maintained
 * completed/total counters. Tasks may be appended after a planning/allocation
 * task discovers a device-dependent sub-plan, but a batch is registered in
 * full before any task in that batch starts.
 */

export type GPUInitializationPhase =
  | "planning"
  | "allocation"
  | "solver-pipelines"
  | "adaptive-topology"
  | "secondary-particles"
  | "upload"
  | "warmup"
  | "attach";

export interface GPUInitializationSnapshot {
  phase: GPUInitializationPhase;
  taskId: string;
  label: string;
  completed: number;
  total: number;
}

export interface GPUInitializationTask {
  id: string;
  phase: GPUInitializationPhase;
  label: string;
  dependencies?: readonly string[];
  /** Number of independently reportable units owned by this task. The task
   * remains the dependency boundary; this only exposes honest milestones
   * inside constructors or browser GPU calls that cannot be split safely. */
  workUnits?: number;
  /** Force a paint immediately before this task. Phase changes and long
   * same-phase batches already paint automatically. */
  paintBeforeRun?: boolean;
  run(signal: AbortSignal, report?: (label: string, completedWorkUnits?: number) => void): void | Promise<void>;
}

export type GPUInitializationSnapshotReporter = (snapshot: GPUInitializationSnapshot) => void;

const yieldForPaint = () => new Promise<void>((resolve) => {
  // Resuming a promise directly inside requestAnimationFrame still runs its
  // continuation before the browser paints that frame. Put the task onto the
  // following macrotask so the status React just committed is actually
  // visible before shader compilation or a large GPU allocation begins. A
  // dedicated worker has no UI paint to wait for, and worker rAF is not a
  // bootstrap scheduler: browsers may withhold it until the OffscreenCanvas
  // has presented once. Yield one macrotask there so initialization cannot
  // deadlock before its first frame.
  if (typeof document !== "undefined" && typeof requestAnimationFrame === "function") requestAnimationFrame(() => setTimeout(resolve, 0));
  else setTimeout(resolve, 0);
});

/**
 * Per-task startup timing, behind `FLUID_GPU_INIT_CENSUS=1`.
 *
 * Every benchmark arm is a fresh process, so construction is paid once per arm
 * and lands entirely inside the A/B loop's wall clock -- yet nothing reported
 * where it went. The progress records this runner already emits carry a
 * completed/total counter and no time at all, so a task that costs three
 * seconds is indistinguishable from one that costs three milliseconds.
 *
 * Same doctrine as the resident-memory census: off by default, cumulative, and
 * attributed to the task id that asked for the work rather than to a phase
 * total, because the phase totals are what made this invisible.
 */
const initializationCensusEnabled = typeof process !== "undefined"
  && process.env?.FLUID_GPU_INIT_CENSUS === "1";

/** Compile independent pipeline tasks concurrently. On by default; set
 * `FLUID_GPU_PARALLEL_PIPELINE_COMPILE=0` to restore the strictly serial
 * runner, which is the definition of the result either way -- a batch only ever
 * contains tasks that cannot depend on one another. */
const parallelPipelineCompilationEnabled = typeof process === "undefined"
  || process.env?.FLUID_GPU_PARALLEL_PIPELINE_COMPILE !== "0";

export interface GPUInitializationCensusEntry {
  taskId: string;
  phase: GPUInitializationPhase;
  elapsed_ms: number;
}

const initializationCensus: GPUInitializationCensusEntry[] = [];

/** Every task timed so far this process, in completion order. */
export const readInitializationCensus = (): readonly GPUInitializationCensusEntry[] =>
  initializationCensus;

/** Slowest tasks first, with the share of total construction each one owns. */
export const formatInitializationCensus = (
  entries: readonly GPUInitializationCensusEntry[] = initializationCensus,
): string => {
  if (entries.length === 0) return "no GPU initialization tasks were timed";
  const total = entries.reduce((sum, entry) => sum + entry.elapsed_ms, 0);
  const byPhase = new Map<GPUInitializationPhase, number>();
  for (const entry of entries) {
    byPhase.set(entry.phase, (byPhase.get(entry.phase) ?? 0) + entry.elapsed_ms);
  }
  const lines = [`GPU initialization: ${total.toFixed(1)} ms across ${entries.length} tasks`];
  for (const [phase, elapsed] of [...byPhase].sort((a, b) => b[1] - a[1])) {
    lines.push(`  ${(elapsed / total * 100).toFixed(1).padStart(5)}%  ${elapsed.toFixed(1).padStart(9)} ms  phase ${phase}`);
  }
  lines.push("  --- slowest tasks ---");
  for (const entry of [...entries].sort((a, b) => b.elapsed_ms - a.elapsed_ms).slice(0, 15)) {
    lines.push(`  ${(entry.elapsed_ms / total * 100).toFixed(1).padStart(5)}%  ${entry.elapsed_ms.toFixed(1).padStart(9)} ms  ${entry.taskId}`);
  }
  return lines.join("\n");
};

/** Bound the amount of initialization work that can run between visible
 * progress updates without paying one full animation frame per small shader. */
const TASKS_PER_PAINT = 8;

export class GPUInitializationTaskRunner {
  private readonly registered = new Set<string>();
  private readonly completed = new Set<string>();
  private completedWorkUnits = 0;
  private total = 0;
  private lastPaintedPhase?: GPUInitializationPhase;
  private tasksSincePaint = TASKS_PER_PAINT;

  constructor(
    private readonly report: GPUInitializationSnapshotReporter,
    private readonly signal: AbortSignal,
  ) {}

  get completedCount() { return this.completed.size; }
  get totalCount() { return this.total; }

  private register(tasks: readonly GPUInitializationTask[]) {
    for (const task of tasks) {
      if (!task.id) throw new Error("GPU initialization task IDs must be non-empty");
      if (this.registered.has(task.id)) throw new Error(`Duplicate GPU initialization task: ${task.id}`);
      this.registered.add(task.id);
    }
    this.total += tasks.reduce((sum, task) => sum + Math.max(1, Math.floor(task.workUnits ?? 1)), 0);
  }

  async run(tasks: readonly GPUInitializationTask[]) {
    this.register(tasks);
    for (let index = 0; index < tasks.length; index += 1) {
      const task = tasks[index]!;
      // Shader compilation is the one phase whose tasks are pure producers:
      // they build pipelines from already-allocated resources and touch no
      // shared ordering. Every other phase stays strictly serial -- allocation
      // in particular, because buffer creation order decides device addresses
      // and this runner's output is compared across A/B arms.
      //
      // A batch is only formed from CONSECUTIVE compile tasks whose declared
      // dependencies are already complete before the batch starts, so the
      // dependency contract the serial path enforces is unchanged; the tasks in
      // a batch simply cannot depend on each other.
      if (parallelPipelineCompilationEnabled && task.phase === "solver-pipelines") {
        const batch: GPUInitializationTask[] = [];
        while (index + batch.length < tasks.length) {
          const candidate = tasks[index + batch.length]!;
          if (candidate.phase !== "solver-pipelines" || candidate.paintBeforeRun) break;
          if ((candidate.dependencies ?? []).some((id) => !this.completed.has(id))) break;
          batch.push(candidate);
        }
        if (batch.length > 1) {
          await this.runCompileBatch(batch);
          index += batch.length - 1;
          continue;
        }
      }
      if (this.signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      this.assertDependenciesComplete(task);
      this.reportTask(task, task.label);
      // Paint every phase transition immediately, then periodically within a
      // long phase. Pipeline compilation stays sequential, but a large set no
      // longer adds one full animation frame per tiny async pipeline task.
      // Callers can retain the old before-task fence for exceptional heavy
      // synchronous work with `paintBeforeRun`.
      const shouldPaint = task.paintBeforeRun
        || task.phase !== this.lastPaintedPhase
        || this.tasksSincePaint >= TASKS_PER_PAINT;
      if (shouldPaint) {
        await yieldForPaint();
        this.lastPaintedPhase = task.phase;
        this.tasksSincePaint = 0;
      }
      if (this.signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      const taskStarted = initializationCensusEnabled ? performance.now() : 0;
      await task.run(this.signal, (label, completedWorkUnits) => this.reportTask(task, label, completedWorkUnits));
      if (initializationCensusEnabled) {
        initializationCensus.push({ taskId: task.id, phase: task.phase,
          elapsed_ms: performance.now() - taskStarted });
      }
      this.tasksSincePaint += 1;
      this.completed.add(task.id);
      this.completedWorkUnits += Math.max(1, Math.floor(task.workUnits ?? 1));
      this.reportTask(task, task.label);
    }
  }

  /** Compile a batch of mutually independent pipeline tasks concurrently.
   *
   * Completion is recorded in the batch's declared order regardless of which
   * task finished first, so `completed` and the progress counter read exactly
   * as they would have serially. Only the wall clock differs. */
  private async runCompileBatch(batch: readonly GPUInitializationTask[]) {
    if (this.signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
    for (const task of batch) this.assertDependenciesComplete(task);
    this.reportTask(batch[0]!, batch[0]!.label);
    await yieldForPaint();
    this.lastPaintedPhase = batch[0]!.phase;
    this.tasksSincePaint = 0;
    if (this.signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
    const started = initializationCensusEnabled ? performance.now() : 0;
    await Promise.all(batch.map((task) => Promise.resolve(
      task.run(this.signal, (label, completedWorkUnits) => this.reportTask(task, label, completedWorkUnits)))));
    if (initializationCensusEnabled) {
      // The batch is one wall-clock interval, so attributing it per task would
      // invent numbers. Charge it to the batch and name its members.
      initializationCensus.push({
        taskId: `[parallel ${batch.length}] ${batch.map((task) => task.id).join(" ")}`,
        phase: batch[0]!.phase, elapsed_ms: performance.now() - started,
      });
    }
    this.tasksSincePaint += batch.length;
    for (const task of batch) {
      this.completed.add(task.id);
      this.completedWorkUnits += Math.max(1, Math.floor(task.workUnits ?? 1));
      this.reportTask(task, task.label);
    }
  }

  private assertDependenciesComplete(task: GPUInitializationTask) {
    for (const dependency of task.dependencies ?? []) {
      if (!this.completed.has(dependency)) {
        throw new Error(`GPU initialization task ${task.id} ran before ${dependency}`);
      }
    }
  }

  private reportTask(task: GPUInitializationTask, label: string, completedWithinTask = 0) {
    const workUnits = Math.max(1, Math.floor(task.workUnits ?? 1));
    this.report({ phase: task.phase, taskId: task.id, label,
      completed: this.completedWorkUnits + Math.max(0, Math.min(workUnits - 1, Math.floor(completedWithinTask))),
      total: this.total });
  }

}

export function isGPUInitializationAbort(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
