/**
 * Native Dawn instances own the ProcessEvents runner used by asynchronous
 * WebGPU compilation. Node may otherwise collect a locally scoped GPU before
 * the device's final native callbacks have retired, which can race process
 * shutdown. Keep every harness-owned instance strongly reachable for the
 * lifetime of its isolated Node process.
 */
export interface NodeDawnProvider {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
}

const PROCESS_DAWN_GPU_KEEPALIVE = new Set<GPU>();

export function createProcessRetainedDawnGPU(
  dawn: NodeDawnProvider,
  options: string[],
): GPU {
  const gpu = dawn.create(options);
  PROCESS_DAWN_GPU_KEEPALIVE.add(gpu);
  return gpu;
}

/**
 * Dawn's Node binding polls ProcessEvents through `setImmediate` for as long
 * as an asynchronous operation is pending, creating a native function and an
 * Immediate for every tick. A frame that waits 50 ms on the GPU spins several
 * thousand of them: about 4 MB of garbage a frame on the figure-9 teaser,
 * twelve scavenges a frame and a 40-50 ms major collection every ~45 frames.
 * None of it is the solver's and a browser has no such loop, so a benchmark
 * that measures frame times paces the poll: each tick first waits
 * `interval_ms`, which bounds the latency it adds to a completion. The wait
 * is a counted spin. A sleep of that length leaves the thread slow when it
 * wakes (the teaser's frame encode took 4.9 ms against 3.1), and a spin that
 * reads the clock makes a heap number every turn (two scavenges a frame).
 * The binding makes a new function for every tick, so a tick is recognised by
 * its single argument and not by identity; another caller that passes one
 * is delayed by the same interval and nothing else. What is left is the
 * binding's own function and Node's Immediate per tick, about 0.5 KB.
 * Call once per process, before the first device is requested.
 */
export function paceDawnEventPolling(interval_ms = 0.25): void {
  const immediate = globalThis.setImmediate, cell = new Int32Array(new SharedArrayBuffer(4));
  const spin = (turns: number) => { let sum = 0; for (let i = 0; i < turns; i++) sum = (sum + Atomics.load(cell, 0)) | 0; return sum; };
  // Timed once it is optimized: the turns of one interval.
  for (let i = 0; i < 8; i++) spin(1 << 20);
  const start = performance.now(); spin(1 << 22);
  const turns = Math.ceil(interval_ms * (1 << 22) / (performance.now() - start));
  // One trampoline for every tick: a closure per tick would be garbage of its own.
  const waiting: (() => void)[] = [];
  const tick = () => { const callback = waiting.shift(); spin(turns); callback?.(); };
  function paced(this: unknown, callback: () => void) {
    if (arguments.length !== 1) return Reflect.apply(immediate, this, arguments);
    waiting.push(callback); return immediate(tick);
  }
  globalThis.setImmediate = Object.assign(paced, immediate) as unknown as typeof setImmediate;
}
