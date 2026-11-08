/**
 * Pipelines an owner binds only in some configurations or on some frames.
 *
 * Construction requests them at background priority and does not await them.
 * The manager starts them only after every critical and visible request, and
 * after construction releases its hold (holdGPUBackgroundCompilation). A frame
 * that may bind one awaits it before its first queue write (require), which
 * raises the request to critical. There is no alternate path: a frame either
 * waits for the pipeline or does not need it, and taking one that is not ready
 * (get) throws.
 */

import { requestGPUCompilationAt, type GPUCompilationScopedRequest } from "./gpu-compilation-manager";

interface BackgroundEntry {
  readonly request: GPUCompilationScopedRequest<Promise<GPUComputePipeline>>;
  /** Settles (never rejects) once the request has. */
  readonly settled: Promise<void>;
  pipeline?: GPUComputePipeline;
  failure?: Error;
}

export class GPUBackgroundPipelines<Key extends string = string> {
  private readonly entries = new Map<Key, BackgroundEntry>();
  private unsettled = 0;
  private failures = 0;

  /**
   * Request one pipeline at background priority. `compile` must issue its
   * pipeline request synchronously, before its first await (the managed
   * device's createComputePipelineAsync does, through any wrapper that
   * forwards the call).
   */
  add(key: Key, compile: () => Promise<GPUComputePipeline>): void {
    if (this.entries.has(key)) throw new Error(`Background pipeline ${key} is requested twice`);
    const request = requestGPUCompilationAt("background", compile);
    this.unsettled++;
    const entry: BackgroundEntry = {
      request,
      settled: request.result.then(
        pipeline => { entry.pipeline = pipeline; },
        (error: unknown) => {
          entry.failure = error instanceof Error ? error : new Error(String(error));
          this.failures++;
        },
      ).then(() => { this.unsettled--; }),
    };
    this.entries.set(key, entry);
  }

  has(key: string): key is Key { return this.entries.has(key as Key); }

  /** Every pipeline compiled: no frame has anything to wait for. */
  get allReady(): boolean { return this.unsettled === 0 && this.failures === 0; }

  /** The compiled pipeline. Throws while it compiles (the frame did not await require) or after it failed. */
  get(key: Key): GPUComputePipeline {
    const entry = this.entries.get(key);
    if (!entry) throw new Error(`Background pipeline ${key} was never requested`);
    if (entry.failure) throw entry.failure;
    if (!entry.pipeline) throw new Error(`Background pipeline ${key} is still compiling: an advance that may bind it must await its owner's pending pipelines first`);
    return entry.pipeline;
  }

  /**
   * undefined when every key is compiled. Otherwise raise the pending ones to
   * critical priority and return a promise that resolves once all are
   * compiled, or rejects with the first failure. A failed background compile
   * rejects every call, needed or not: a broken pipeline fails the next
   * advance explicitly rather than only the frame that first binds it.
   */
  require(keys: readonly Key[]): Promise<void> | undefined {
    if (this.failures > 0) {
      for (const [key, entry] of this.entries) if (entry.failure) return Promise.reject(new Error(`Background pipeline ${key} failed to compile`, { cause: entry.failure }));
    }
    let waits: Promise<void>[] | undefined;
    for (const key of keys) {
      const entry = this.entries.get(key);
      if (!entry) throw new Error(`Background pipeline ${key} was never requested`);
      if (entry.pipeline) continue;
      if (!entry.failure) entry.request.raise("critical");
      (waits ??= []).push(entry.settled);
    }
    if (!waits) return undefined;
    return Promise.all(waits).then(() => {
      for (const key of keys) {
        const failure = this.entries.get(key)!.failure;
        if (failure) throw failure;
      }
    });
  }
}

/** Join owners' pending-pipeline promises: undefined when none is pending. */
export function joinPendingPipelines(...pending: (Promise<void> | undefined)[]): Promise<void> | undefined {
  let waits: Promise<void>[] | undefined;
  for (const wait of pending) if (wait) (waits ??= []).push(wait);
  if (!waits) return undefined;
  return waits.length === 1 ? waits[0] : Promise.all(waits).then(() => undefined);
}
