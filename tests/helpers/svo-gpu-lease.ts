import { acquireWebGPUExclusiveLock } from "../../lib/harness/webgpu-smoke-isolation";

/** Wait inside the loaded fixture too: another runner can take the lease while
 * this file's imports load after the Dawn runner's availability check. */
export async function acquireSvoTestLease(target: string): Promise<void> {
  const deadline = Date.now() + 900_000;
  for (;;) {
    try { await acquireWebGPUExclusiveLock("dawn-test", target); return; }
    catch (error) {
      if (!(error instanceof Error) || !error.message.includes("Refusing concurrent GPU execution") || Date.now() >= deadline) throw error;
      await new Promise(resolve => setTimeout(resolve, 25));
    }
  }
}
