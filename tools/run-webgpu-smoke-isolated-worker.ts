import "../lib/methods";

/**
 * The supervisor sends SIGTERM on timeout and only escalates to SIGKILL after a
 * grace period. Exit with the timeout status rather than the default signal
 * disposition so the supervisor's report names the cause.
 */
for (const signal of ["SIGTERM", "SIGINT"] as const) {
  process.once(signal, () => process.exit(124));
}

await import("./run-webgpu-smoke");
