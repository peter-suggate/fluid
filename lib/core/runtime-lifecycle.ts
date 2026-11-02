/**
 * The lifecycle of a long-lived runtime owner: a renderer, a worker client, a
 * solver frame pipeline. It replaces families of sticky booleans (`disposed`,
 * `deviceLost`, `runtimeFailure`, `stopped`, `failed`) whose guards each checked
 * a different subset.
 *
 * The phase only moves forward. `fail` and `lose` are first-wins: the first
 * terminal transition is the one the owner reports, and every later one is
 * refused, so a failure can never be overwritten by a later ready or hidden by
 * a write that bypassed the reporting path. `dispose` is the one move out of a
 * failure: a failed or lost owner still releases its resources, and keeps the
 * reason it ended with.
 */
export type RuntimePhase =
  | { readonly kind: "starting" }
  | { readonly kind: "live" }
  | { readonly kind: "failed"; readonly reason: string; readonly cause?: unknown }
  | { readonly kind: "lost"; readonly reason: string }
  | { readonly kind: "disposed"; readonly after?: Extract<RuntimePhase, { kind: "failed" | "lost" }> };

export type RuntimeTerminalPhase = Extract<RuntimePhase, { kind: "failed" | "lost" | "disposed" }>;

export class RuntimeLifecycle {
  private current: RuntimePhase = { kind: "starting" };

  /**
   * @param onTerminal runs once, synchronously, on the first terminal
   * transition. Owners publish status and abort outstanding work here.
   */
  constructor(private readonly onTerminal?: (phase: RuntimeTerminalPhase) => void) {}

  get phase(): RuntimePhase { return this.current; }
  /** True while the owner is starting or live: the one check hot paths make. */
  get open(): boolean { return this.current.kind === "starting" || this.current.kind === "live"; }
  get live(): boolean { return this.current.kind === "live"; }
  get disposed(): boolean { return this.current.kind === "disposed"; }
  /** The failed or lost reason, if the owner ended that way (before or after disposal). */
  get failure(): string | undefined {
    const phase = this.current.kind === "disposed" ? this.current.after : this.current;
    return phase?.kind === "failed" || phase?.kind === "lost" ? phase.reason : undefined;
  }

  /** starting → live. Refused (false) once terminal. */
  ready(): boolean {
    if (this.current.kind !== "starting") return this.current.kind === "live";
    this.current = { kind: "live" };
    return true;
  }

  fail(reason: string, cause?: unknown): boolean { return this.end({ kind: "failed", reason, cause }); }
  lose(reason: string): boolean { return this.end({ kind: "lost", reason }); }
  /**
   * → disposed from any phase but disposed. Reports through `onTerminal` only
   * when the owner was still open; a failure was already reported.
   */
  dispose(): boolean {
    const phase = this.current;
    if (phase.kind === "disposed") return false;
    if (phase.kind === "failed" || phase.kind === "lost") { this.current = { kind: "disposed", after: phase }; return true; }
    return this.end({ kind: "disposed" });
  }

  /** Throws when the owner has ended; for entry points that must not run late. */
  assertOpen(operation: string): void {
    if (this.open) return;
    const phase = this.current;
    throw new Error(`${operation}: runtime ${phase.kind}${"reason" in phase ? ` (${phase.reason})` : ""}`);
  }

  private end(phase: RuntimeTerminalPhase): boolean {
    if (!this.open) return false;
    this.current = phase;
    this.onTerminal?.(phase);
    return true;
  }
}
