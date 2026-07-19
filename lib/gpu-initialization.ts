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
  run(signal: AbortSignal): void | Promise<void>;
}

export type GPUInitializationSnapshotReporter = (snapshot: GPUInitializationSnapshot) => void;

const yieldForPaint = () => new Promise<void>((resolve) => {
  // Resuming a promise directly inside requestAnimationFrame still runs its
  // continuation before the browser paints that frame. Put the task onto the
  // following macrotask so the status React just committed is actually
  // visible before shader compilation or a large GPU allocation begins.
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => setTimeout(resolve, 0));
  else setTimeout(resolve, 0);
});

export class GPUInitializationTaskRunner {
  private readonly registered = new Set<string>();
  private readonly completed = new Set<string>();
  private total = 0;

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
    this.total += tasks.length;
  }

  async run(tasks: readonly GPUInitializationTask[]) {
    this.register(tasks);
    for (const task of tasks) {
      if (this.signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      for (const dependency of task.dependencies ?? []) {
        if (!this.completed.has(dependency)) throw new Error(`GPU initialization task ${task.id} ran before ${dependency}`);
      }
      this.report({
        phase: task.phase,
        taskId: task.id,
        label: task.label,
        completed: this.completed.size,
        total: this.total,
      });
      // Let React commit the new stage before a synchronous allocation or CPU
      // packing task begins. Expensive CPU planners should still be split or
      // moved to a worker, but they can no longer run before their stage is
      // visible.
      await yieldForPaint();
      if (this.signal.aborted) throw new DOMException("GPU initialization superseded", "AbortError");
      await task.run(this.signal);
      this.completed.add(task.id);
      this.report({
        phase: task.phase,
        taskId: task.id,
        label: task.label,
        completed: this.completed.size,
        total: this.total,
      });
    }
  }
}

export function isGPUInitializationAbort(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
