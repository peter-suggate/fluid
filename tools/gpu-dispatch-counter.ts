/** Compute dispatches by pass label, counted at the device boundary (the
 * scaling acceptance test: a step's work must not follow the box).
 *
 * Wraps a device so that, while enabled, every compute pass of every command
 * encoder it creates counts its dispatches: per pass label, the direct and
 * indirect dispatch calls and the direct ones' workgroups. Indirect arguments
 * live on the GPU and are not read. Nothing else changes.
 *
 *   const counter = gpuDispatchCounter(device); // build the solver on counter.device
 *   counter.enabled = true; ...advance...; const counts = counter.take();
 */
export interface GPUDispatchCount { direct: number; indirect: number; workgroups: number }
export interface GPUDispatchCounter {
  readonly device: GPUDevice;
  enabled: boolean;
  /** The counts by pass label since the last take, sorted by label; resets them. */
  take(): Record<string, GPUDispatchCount>;
}

export function gpuDispatchCounter(device: GPUDevice): GPUDispatchCounter {
  let counts = new Map<string, GPUDispatchCount>();
  const entry = (label: string) => { let c = counts.get(label); if (!c) counts.set(label, c = { direct: 0, indirect: 0, workgroups: 0 }); return c; };
  const wrapPass = (pass: GPUComputePassEncoder, label: string): GPUComputePassEncoder => new Proxy(pass, { get(target, key) {
    if (key === "dispatchWorkgroups") return (x: number, y = 1, z = 1) => { const c = entry(label); c.direct++; c.workgroups += x * y * z; target.dispatchWorkgroups(x, y, z); };
    if (key === "dispatchWorkgroupsIndirect") return (buffer: GPUBuffer, offset: number) => { entry(label).indirect++; target.dispatchWorkgroupsIndirect(buffer, offset); };
    const value = Reflect.get(target, key);
    return typeof value === "function" ? value.bind(target) : value;
  } });
  const wrapEncoder = (encoder: GPUCommandEncoder): GPUCommandEncoder => new Proxy(encoder, { get(target, key) {
    if (key === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
      const pass = target.beginComputePass(descriptor);
      return counter.enabled ? wrapPass(pass, descriptor?.label ?? "(unlabelled pass)") : pass;
    };
    const value = Reflect.get(target, key);
    return typeof value === "function" ? value.bind(target) : value;
  } });
  const counter: GPUDispatchCounter = {
    enabled: false,
    device: new Proxy(device, { get(target, key) {
      if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => wrapEncoder(target.createCommandEncoder(descriptor));
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    } }),
    take() {
      const out = Object.fromEntries([...counts].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0));
      counts = new Map(); return out;
    },
  };
  return counter;
}
