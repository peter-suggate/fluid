import { writeSync } from "node:fs";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
  releaseWebGPUExclusiveLockSync,
} from "./webgpu-smoke-isolation";

async function acquireExclusiveGPUProcessLock(): Promise<void> {
  const evidence = await acquireWebGPUExclusiveLock(
    "dawn-smoke",
    "tools/run-webgpu-smoke.ts",
  );
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "acquired", ...evidence }));
}

/**
 * The supervisor sends SIGTERM on timeout and only escalates to SIGKILL after a
 * grace period. Without a handler the default disposition kills us before the
 * `finally` below can run, so the lock leaked and the next run refused to
 * start. `writeSync` bypasses the stdout stream, whose queued writes are
 * discarded on termination when stdout is a pipe.
 */
for (const signal of ["SIGTERM", "SIGINT"] as const) {
  process.once(signal, () => {
    releaseWebGPUExclusiveLockSync();
    writeSync(1, `${JSON.stringify({
      phase: "webgpu-exclusive-lock", state: "released", reason: signal, pid: process.pid,
    })}\n`);
    process.exit(124);
  });
}

await acquireExclusiveGPUProcessLock();
try {
  await import("./run-webgpu-smoke");
} finally {
  await releaseWebGPUExclusiveLock();
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "released", pid: process.pid }));
}
