import { mkdir, readFile, rm, writeFile } from "node:fs/promises";

export const WEBGPU_EXCLUSIVE_LOCK = "/tmp/fluid-webgpu-exclusive.lock";
export const DEFAULT_WEBGPU_SMOKE_TIMEOUT_MS = 120_000;
export const MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS = 60_000;
export const MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS = 240_000;
export const WEBGPU_SMOKE_TERMINATE_GRACE_MS = 2_000;
export const WEBGPU_SMOKE_KILL_REAP_MS = 2_000;

export interface WebGPUExclusiveLockOwner {
  readonly pid: number;
  readonly parentPid: number;
  readonly startedAt: string;
  readonly kind: string;
  readonly target: string;
}

/**
 * Atomically acquire the process-wide Dawn/browser exclusion lock.
 *
 * GPU entrypoints acquire this before importing Dawn. A lock left behind by a
 * hard crash is intentional owner evidence; callers must release it only from
 * an ordinary `finally` path.
 */
export async function acquireWebGPUExclusiveLock(
  kind: string,
  target: string,
): Promise<WebGPUExclusiveLockOwner> {
  try {
    await mkdir(WEBGPU_EXCLUSIVE_LOCK);
  } catch (error) {
    let owner = "unknown owner";
    try { owner = await readFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* diagnostic only */ }
    throw new Error(`Refusing concurrent GPU execution; ${WEBGPU_EXCLUSIVE_LOCK} already exists (${owner}). Remove it only after confirming its owner PID is gone and no Dawn or browser GPU run is active.`, { cause: error });
  }

  const evidence: WebGPUExclusiveLockOwner = Object.freeze({
    pid: process.pid,
    parentPid: process.ppid,
    startedAt: new Date().toISOString(),
    kind,
    target,
  });
  try {
    await writeFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, JSON.stringify(evidence));
  } catch (error) {
    await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
    throw error;
  }
  return evidence;
}

export async function releaseWebGPUExclusiveLock(): Promise<void> {
  await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
}

/** Parse the deliberately narrow wall-clock envelope before loading Dawn. */
export function parseWebGPUSmokeTimeout(value: string | undefined): number {
  if (value === undefined) return DEFAULT_WEBGPU_SMOKE_TIMEOUT_MS;
  const timeout = Number(value);
  if (!Number.isInteger(timeout)
    || timeout < MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS
    || timeout > MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS) {
    throw new Error(`FLUID_WEBGPU_SMOKE_TIMEOUT_MS must be an integer from ${MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS} to ${MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS}`);
  }
  return timeout;
}
