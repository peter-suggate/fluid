import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

async function acquireExclusiveGPUProcessLock(): Promise<void> {
  const evidence = await acquireWebGPUExclusiveLock(
    "dawn-smoke",
    "tools/run-webgpu-smoke.ts",
  );
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "acquired", ...evidence }));
}

await acquireExclusiveGPUProcessLock();
try {
  await import("./run-webgpu-smoke");
} finally {
  await releaseWebGPUExclusiveLock();
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "released", pid: process.pid }));
}
