/**
 * Long synchronous builds, made interruptible without leaving the thread.
 *
 * The render worker services `draw`, `set-render-scene` and `shutdown` off its
 * message queue, which it can only reach by returning to the event loop. A
 * scene build that is one uninterrupted synchronous block therefore freezes the
 * viewport for its whole duration — 40 s at environment refinement depth 3 on
 * `hero-garden-hose` — and a rebuild queued behind it runs to completion too,
 * because the only abort check is between retry-ladder rungs and the signal
 * that would trip it arrives on the queue the build is blocking.
 *
 * A build expressed as a generator fixes both at once. It `yield`s at points
 * where its own partial state is coherent; this driver decides *whether* each
 * one becomes a real event-loop yield, so the granularity of the interrupt is
 * a policy here rather than a property of the producer. Between slices the
 * driver checks the abort signal, and an aborted build is unwound through
 * `Generator.return`, which runs every `finally` on the suspended stack.
 *
 * Nothing here is browser-specific: `MessageChannel` is a global in Node 15+
 * and in every worker realm, so a headless lane measures the same schedule the
 * viewport gets.
 */

/** The default slice. Two frames at 60 Hz, so a yield is never the reason a frame is late. */
export const COOPERATIVE_BUILD_SLICE_MS = 8;

export interface CooperativeBuildOptions {
  /**
   * Supersession. Checked at every slice boundary, never mid-slice, so an
   * abandoned build always stops at a point its producer declared coherent.
   */
  signal?: AbortSignal;
  /**
   * Longest the thread may be held between event-loop yields, in milliseconds.
   *
   * This is an upper bound on worker message latency only to the extent the
   * generator offers yield points that often; a single `yield`-free device call
   * that costs more than the slice is still a stall of its own length. Raising
   * it trades responsiveness for fewer task hops.
   */
  sliceBudget_ms?: number;
  /** Observed slice statistics, for lanes that assert on responsiveness. */
  onSlice?: (slice: { yields: number; elapsed_ms: number }) => void;
}

/**
 * One macrotask hop.
 *
 * `MessageChannel` rather than `setTimeout` on purpose: a nested timer is
 * clamped to 4 ms after five levels, which on a build with thousands of slices
 * is minutes of pure clamp, and a posted message is not clamped at all. It also
 * lands in the same task source as the messages the worker is trying to
 * service, so those are dequeued ahead of the resumption rather than racing it.
 */
function yieldToEventLoop(): Promise<void> {
  return new Promise<void>((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      channel.port2.close();
      resolve();
    };
    channel.port2.postMessage(0);
  });
}

/** The one error a superseded build raises, recognised by `isCooperativeBuildAbort`. */
export function cooperativeBuildAbortError(): Error {
  return new DOMException("GPU initialization superseded", "AbortError");
}

export function isCooperativeBuildAbort(error: unknown): boolean {
  return error instanceof DOMException ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}

/**
 * Run a generator to completion, yielding to the event loop whenever it has
 * held the thread longer than the slice budget.
 *
 * On abort the generator is closed with `return()` before the rejection
 * propagates, so a producer that owns partially allocated resources gets the
 * chance to release them in a `finally` — and a producer that instead keeps
 * every allocation reachable from a field it has already assigned (which is the
 * contract `OctreeSparseBrickWorld` uses) can be destroyed by its caller.
 */
export async function driveCooperativeBuild<T>(
  steps: Generator<unknown, T, undefined>,
  options: CooperativeBuildOptions = {},
): Promise<T> {
  const budget_ms = options.sliceBudget_ms ?? COOPERATIVE_BUILD_SLICE_MS;
  let sliceStart_ms = performance.now();
  let sliceYields = 0;
  for (;;) {
    if (options.signal?.aborted) {
      steps.return(undefined as T);
      throw cooperativeBuildAbortError();
    }
    const step = steps.next();
    if (step.done) return step.value;
    sliceYields += 1;
    const elapsed_ms = performance.now() - sliceStart_ms;
    if (elapsed_ms < budget_ms) continue;
    options.onSlice?.({ yields: sliceYields, elapsed_ms });
    await yieldToEventLoop();
    sliceStart_ms = performance.now();
    sliceYields = 0;
  }
}

/**
 * Drive a generator with no yielding at all, for callers that are already
 * synchronous and cannot become otherwise.
 *
 * This is what keeps the interruptible form a superset: the same generator body
 * is the definition of the build for both the worker's async path and the
 * constructors, benchmarks and tests that still build in one shot.
 */
export function completeCooperativeBuild<T>(steps: Generator<unknown, T, undefined>): T {
  for (;;) {
    const step = steps.next();
    if (step.done) return step.value;
  }
}
