import { rmSync } from "node:fs";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";

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

export interface WebGPUExclusiveLockHolder {
  /** Absent when the lock exists but has not been stamped (or was hand-made). */
  readonly owner?: WebGPUExclusiveLockOwner;
  /**
   * Whether the owning process still exists. This is the distinction a caller
   * must act on: a live owner means wait for it or stop it, while a dead one is
   * the crash evidence this lock deliberately preserves for a human to clear.
   * An unstamped lock counts as live -- it is most likely a run mid-acquisition.
   */
  readonly alive: boolean;
  /** One line naming the holder, for an error a reader can act on. */
  readonly description: string;
}

const ownerIsRunning = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM means the pid exists and belongs to somebody else, which for our
    // purposes is every bit as much a live GPU run as one of our own.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
};

/**
 * Who holds the exclusive lock right now, or `undefined` when it is free.
 *
 * Read this before starting long GPU work. Acquisition failure is reported by
 * whichever process loses the race, deep inside its startup and only through
 * its exit status, so a supervisor that spends minutes on a capture wants to
 * find out first -- and to be able to say who it is waiting for. The lock is an
 * absolute path, so the holder may well be a different checkout of this repo.
 */
export async function readWebGPUExclusiveLockHolder(
  lockPath: string = WEBGPU_EXCLUSIVE_LOCK,
): Promise<WebGPUExclusiveLockHolder | undefined> {
  try {
    await stat(lockPath);
  } catch {
    return undefined;
  }
  let owner: WebGPUExclusiveLockOwner | undefined;
  try {
    owner = JSON.parse(await readFile(`${lockPath}/owner.json`, "utf8")) as WebGPUExclusiveLockOwner;
  } catch { /* an unstamped lock still excludes us; it just cannot say who by */ }
  if (owner === undefined || typeof owner.pid !== "number") {
    return { alive: true, description: "an owner that has not stamped owner.json yet" };
  }
  const alive = ownerIsRunning(owner.pid);
  return {
    owner,
    alive,
    description: `pid ${owner.pid} (${owner.kind}, ${owner.target})`
      + ` started ${owner.startedAt}${alive ? "" : ", no longer running"}`,
  };
}

export async function releaseWebGPUExclusiveLock(): Promise<void> {
  await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
}

/**
 * Release the lock from a signal handler.
 *
 * A supervisor SIGTERM is an ordinary termination, not a crash: the async
 * `finally` in the worker never runs, so without this the timeout path leaks
 * the lock and the *next* run fails with "Refusing concurrent GPU execution".
 * Only the un-reapable SIGKILL path should leave the directory behind as owner
 * evidence, which is what `run-webgpu-smoke-isolated.ts` already reports.
 */
export function releaseWebGPUExclusiveLockSync(): void {
  rmSync(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
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
