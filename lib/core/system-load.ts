/**
 * What the machine is doing while the studio waits on it.
 *
 * A browser exposes no utilisation counter for a CPU core or for the GPU, so
 * every reading here is something this page can measure for itself, named for
 * what it is rather than for the counter it stands in for:
 *
 * - **Thread blocked** — the share of a window a thread could not answer a
 *   timer, from how late a fixed heartbeat fires. A thread compiling a shader
 *   module or encoding a large upload reads high; a thread awaiting an async
 *   pipeline reads near zero, because it is free.
 * - **GPU queue busy** — the share of a window this device's queue held
 *   submitted, unfinished work, from submit to `onSubmittedWorkDone`. It counts
 *   this tab's work only, and it includes completion-callback latency, so it
 *   reads as an upper bound on time the GPU spent on us.
 * - **CPU pressure** — the platform's own four-level verdict on the whole
 *   machine (Compute Pressure API), where the browser provides one.
 *
 * Samples are windows, not instants: each `sample()` covers the time since the
 * one before it and starts the next.
 */

/** The heartbeat period. Lateness under one period is not visible. */
export const LOAD_HEARTBEAT_MS = 50;
/** A heartbeat this late is a stall a reader would have felt as a hitch. */
export const LOAD_STALL_MS = 100;
/** How often the render worker reports, and how many reports a sparkline holds. */
export const LOAD_SAMPLE_INTERVAL_MS = 500;
export const LOAD_HISTORY_LENGTH = 40;

export interface ThreadLoadSample {
  /** 0–1: share of the window the thread could not service a timer. */
  readonly blocked: number;
  /** Heartbeats that arrived more than `LOAD_STALL_MS` late. */
  readonly stalls: number;
  readonly longestStall_ms: number;
}

export interface GPUQueueLoadSample {
  /** 0–1: share of the window the queue held unfinished work from this device. */
  readonly busy: number;
  readonly submits: number;
}

export type CPUPressureState = "nominal" | "fair" | "serious" | "critical";

/** One worker report: its own thread, and the queue of the device it owns. */
export interface WorkerLoadSample {
  readonly thread: ThreadLoadSample;
  readonly gpu?: GPUQueueLoadSample;
}

/** A sample, stamped on the main thread's clock when it arrived. */
export interface LoadReading {
  readonly at_ms: number;
  readonly uiThread?: ThreadLoadSample;
  readonly worker?: WorkerLoadSample;
}

type Clock = () => number;

/**
 * Lateness of a fixed-period timer, accumulated into windows.
 *
 * The timer is the probe: it asks for a callback every period, and any time it
 * arrives late is time the thread was doing something else it could not set
 * aside. Driven by `beat(now)` so a test can run it on a fake clock.
 */
export class ThreadStallMeter {
  private last_ms: number;
  private windowStart_ms: number;
  private blocked_ms = 0;
  private stalls = 0;
  private longest_ms = 0;

  constructor(private readonly period_ms = LOAD_HEARTBEAT_MS, now: Clock = () => performance.now()) {
    this.last_ms = now();
    this.windowStart_ms = this.last_ms;
  }

  beat(now_ms: number): void {
    const late_ms = Math.max(0, now_ms - this.last_ms - this.period_ms);
    this.last_ms = now_ms;
    this.blocked_ms += late_ms;
    if (late_ms > LOAD_STALL_MS) {
      this.stalls += 1;
      this.longest_ms = Math.max(this.longest_ms, late_ms);
    }
  }

  sample(now_ms: number): ThreadLoadSample {
    // Time since the last beat that is already past due is blocking happening
    // now; leaving it out would report a thread frozen mid-window as idle.
    const overdue_ms = Math.max(0, now_ms - this.last_ms - this.period_ms);
    const window_ms = Math.max(1, now_ms - this.windowStart_ms);
    const sample = {
      blocked: Math.min(1, (this.blocked_ms + overdue_ms) / window_ms),
      stalls: this.stalls,
      longestStall_ms: Math.max(this.longest_ms, overdue_ms),
    };
    this.windowStart_ms = now_ms;
    this.blocked_ms = -overdue_ms;
    this.stalls = 0;
    this.longest_ms = 0;
    return sample;
  }
}

/** Starts a heartbeat on this thread's timers; returns the meter and its stop. */
export function startThreadStallMeter(period_ms = LOAD_HEARTBEAT_MS) {
  const meter = new ThreadStallMeter(period_ms);
  const timer = setInterval(() => meter.beat(performance.now()), period_ms);
  return { meter, stop: () => clearInterval(timer) };
}

/**
 * Busy time of one device queue, from its submits and their completions.
 *
 * At most one completion watch is outstanding: `onSubmittedWorkDone` resolves
 * once everything submitted before it has finished, so one watch armed on the
 * first submit and re-armed while newer submits are pending finds the moment
 * the queue drains, without a promise per submit.
 */
export class GPUQueueOccupancy {
  private submitted = 0;
  private watching = false;
  private busySince_ms?: number;
  private windowStart_ms: number;
  private busy_ms = 0;
  private submits = 0;

  constructor(private readonly drained: () => Promise<unknown>, private readonly now: Clock = () => performance.now()) {
    this.windowStart_ms = now();
  }

  submit(): void {
    this.submits += 1;
    this.submitted += 1;
    this.busySince_ms ??= this.now();
    if (!this.watching) this.watch();
  }

  private watch(): void {
    this.watching = true;
    const target = this.submitted;
    const settle = () => {
      this.watching = false;
      if (this.submitted !== target) { this.watch(); return; }
      if (this.busySince_ms !== undefined) {
        this.busy_ms += this.now() - Math.max(this.busySince_ms, this.windowStart_ms);
        this.busySince_ms = undefined;
      }
    };
    // A lost device rejects; the queue is no longer busy on our account either way.
    this.drained().then(settle, settle);
  }

  sample(): GPUQueueLoadSample {
    const now_ms = this.now();
    const open_ms = this.busySince_ms === undefined ? 0 : now_ms - Math.max(this.busySince_ms, this.windowStart_ms);
    const window_ms = Math.max(1, now_ms - this.windowStart_ms);
    const sample = { busy: Math.min(1, (this.busy_ms + open_ms) / window_ms), submits: this.submits };
    this.windowStart_ms = now_ms;
    this.busy_ms = 0;
    this.submits = 0;
    return sample;
  }
}

let observedQueue: GPUQueueOccupancy | undefined;

/**
 * Wraps this queue's `submit` so every submission, from every owner of the
 * device, is counted — the solver's advances submit on their own and would
 * otherwise be invisible to a meter that only saw the presentation's.
 */
export function observeGPUQueue(queue: GPUQueue): GPUQueueOccupancy {
  const occupancy = new GPUQueueOccupancy(() => queue.onSubmittedWorkDone());
  const submit = queue.submit.bind(queue);
  queue.submit = (commandBuffers: Iterable<GPUCommandBuffer>) => {
    submit(commandBuffers);
    occupancy.submit();
  };
  observedQueue = occupancy;
  return occupancy;
}

/** The queue most recently observed in this realm, if a device exists yet. */
export function observedGPUQueue(): GPUQueueOccupancy | undefined {
  return observedQueue;
}
