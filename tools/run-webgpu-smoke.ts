/** Thin CLI entrypoint; all execution and diagnostics live in focused modules. */
// The catalog installs itself on import, and the executor resolves methods
// through the registry — so this side-effect import must precede it.
import "../lib/methods";
await import("../lib/harness/webgpu-smoke-executor");
export {};
