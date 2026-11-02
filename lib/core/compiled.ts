/**
 * A lazily built resource (a pipeline, a bundle, a prewarm) with one failure
 * policy for the whole app: a failed build is remembered with its error until
 * the next `ensure`, which retries. It replaces the mix of sticky rejected
 * promises, separate `failed` flags and clear-for-retry fields.
 *
 *   idle ─ensure→ building ─→ ready
 *                    └──────→ failed ─ensure→ building
 *   reset from any → idle (an in-flight build's result is dropped)
 */
export type CompiledState<T> =
  | { readonly kind: "idle" }
  | { readonly kind: "building"; readonly promise: Promise<T> }
  | { readonly kind: "ready"; readonly value: T }
  | { readonly kind: "failed"; readonly error: unknown };

export class Compiled<T> {
  private current: CompiledState<T> = { kind: "idle" };
  private generation = 0;

  constructor(private readonly build: () => Promise<T>) {}

  get state(): CompiledState<T> { return this.current; }
  /** The built value, for hot paths that must not await. */
  get value(): T | undefined { return this.current.kind === "ready" ? this.current.value : undefined; }
  get ready(): boolean { return this.current.kind === "ready"; }
  get error(): unknown { return this.current.kind === "failed" ? this.current.error : undefined; }

  /** Build if needed, join a build in flight, or retry after a failure. */
  ensure(): Promise<T> {
    const current = this.current;
    if (current.kind === "ready") return Promise.resolve(current.value);
    if (current.kind === "building") return current.promise;
    const generation = ++this.generation;
    const promise = this.build().then(value => {
      if (generation === this.generation) this.current = { kind: "ready", value };
      return value;
    }, (error: unknown) => {
      if (generation === this.generation) this.current = { kind: "failed", error };
      throw error;
    });
    this.current = { kind: "building", promise };
    return promise;
  }

  /** Forget the value or failure; a build in flight still settles its own callers. */
  reset(): void {
    this.generation += 1;
    this.current = { kind: "idle" };
  }
}
