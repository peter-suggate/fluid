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
    for (const task of tasks) {
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
      await task.run(this.signal, (label, completedWorkUnits) => this.reportTask(task, label, completedWorkUnits));
      this.tasksSincePaint += 1;
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
