export const WEBGPU_EXCLUSIVE_LOCK = "/tmp/fluid-webgpu-exclusive.lock";
export const DEFAULT_WEBGPU_SMOKE_TIMEOUT_MS = 120_000;
export const MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS = 60_000;
export const MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS = 240_000;
export const WEBGPU_SMOKE_TERMINATE_GRACE_MS = 2_000;
export const WEBGPU_SMOKE_KILL_REAP_MS = 2_000;

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
