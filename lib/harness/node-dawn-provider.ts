/**
 * Native Dawn instances own the ProcessEvents runner used by asynchronous
 * WebGPU compilation. Node may otherwise collect a locally scoped GPU before
 * the device's final native callbacks have retired, which can race process
 * shutdown. Keep every harness-owned instance strongly reachable for the
 * lifetime of its isolated Node process.
 */
export interface NodeDawnProvider {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
}

const PROCESS_DAWN_GPU_KEEPALIVE = new Set<GPU>();

export function createProcessRetainedDawnGPU(
  dawn: NodeDawnProvider,
  options: string[],
): GPU {
  const gpu = dawn.create(options);
  PROCESS_DAWN_GPU_KEEPALIVE.add(gpu);
  return gpu;
}
