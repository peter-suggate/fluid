import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { WEBGPU_EXCLUSIVE_LOCK } from "./webgpu-smoke-isolation";

async function acquireExclusiveGPUProcessLock(): Promise<void> {
  try {
    await mkdir(WEBGPU_EXCLUSIVE_LOCK);
  } catch (error) {
    let owner = "unknown owner";
    try { owner = await readFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* diagnostic only */ }
    throw new Error(`Refusing concurrent GPU smoke; ${WEBGPU_EXCLUSIVE_LOCK} already exists (${owner}). Remove it only after confirming its owner PID is gone and no Dawn or browser GPU run is active.`, { cause: error });
  }

  const evidence = {
    pid: process.pid,
    parentPid: process.ppid,
    startedAt: new Date().toISOString(),
    kind: "dawn-smoke",
    target: "tools/run-webgpu-smoke.ts",
  };
  try {
    await writeFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, JSON.stringify(evidence));
  } catch (error) {
    await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "acquired", ...evidence }));
}

await acquireExclusiveGPUProcessLock();
try {
  await import("./run-webgpu-smoke");
} finally {
  await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
  console.log(JSON.stringify({ phase: "webgpu-exclusive-lock", state: "released", pid: process.pid }));
}
