/**
 * Ownership of asynchronous work. Each `start` supersedes the previous run by
 * aborting its signal, so a continuation that resumes after an `await` checks
 * `run.signal.aborted` (or calls `run.current()`) instead of comparing one of
 * many ad hoc generation counters. `dispose` aborts the live run and refuses
 * new ones.
 *
 * A run is a plain AbortSignal underneath: pass it to fetch, worker requests
 * or anything else that already accepts one.
 */
export interface Run {
  readonly signal: AbortSignal;
  /** Monotonic per scope; for diagnostics and keyed caches, never for guards. */
  readonly serial: number;
  /** True while this run is the scope's current run and the scope is open. */
  current(): boolean;
}

export class RunScope {
  private controller?: AbortController;
  private serial = 0;
  private closed = false;

  /** Abort the current run (if any) and begin a new one. */
  start(reason = "superseded"): Run {
    if (this.closed) throw new Error("RunScope.start: scope disposed");
    this.controller?.abort(reason);
    const controller = new AbortController();
    this.controller = controller;
    const serial = ++this.serial;
    return { signal: controller.signal, serial, current: () => !controller.signal.aborted };
  }

  /** The live run, if one is open. */
  get active(): Run | undefined {
    const controller = this.controller;
    if (!controller || controller.signal.aborted) return undefined;
    const serial = this.serial;
    return { signal: controller.signal, serial, current: () => !controller.signal.aborted };
  }

  /** Abort the live run without starting another. */
  cancel(reason = "cancelled"): void {
    this.controller?.abort(reason);
    this.controller = undefined;
  }

  dispose(): void {
    this.cancel("disposed");
    this.closed = true;
  }

  get disposed(): boolean { return this.closed; }
}

/**
 * What a run created — a controller, a device — or nothing, when the run ended
 * while it was being created. A superseded run's creation is destroyed here
 * rather than adopted by a run that never asked for it.
 */
export async function ownedBy<T>(run: Run, created: Promise<T>, destroy: (value: T) => unknown): Promise<T | undefined> {
  const value = await created;
  if (run.current()) return value;
  await destroy(value);
  return undefined;
}
