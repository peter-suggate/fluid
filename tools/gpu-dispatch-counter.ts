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
  /** The same dispatches by "pass label | pipeline" (the pipeline's descriptor
   * label, else its module's, and its entry point), since the last
   * takeKernels; resets them. Pipelines created before the counter wrapped
   * the device are "(unnamed pipeline)". */
  takeKernels(): Record<string, GPUDispatchCount>;
}

export function gpuDispatchCounter(device: GPUDevice): GPUDispatchCounter {
  let counts = new Map<string, GPUDispatchCount>(), kernels = new Map<string, GPUDispatchCount>();
  const names = new WeakMap<object, string>();
  const nameOf = (descriptor: GPUComputePipelineDescriptor) => `${descriptor.label || descriptor.compute.module.label || "(unlabelled module)"} / ${descriptor.compute.entryPoint ?? "(default entry)"}`;
  const at = (m: Map<string, GPUDispatchCount>, label: string) => { let c = m.get(label); if (!c) m.set(label, c = { direct: 0, indirect: 0, workgroups: 0 }); return c; };
  const entry = (label: string) => at(counts, label);
  const wrapPass = (pass: GPUComputePassEncoder, label: string): GPUComputePassEncoder => { let pipeline = "(no pipeline)"; return new Proxy(pass, { get(target, key) {
    if (key === "setPipeline") return (p: GPUComputePipeline) => { pipeline = names.get(p) ?? "(unnamed pipeline)"; target.setPipeline(p); };
    if (key === "dispatchWorkgroups") return (x: number, y = 1, z = 1) => { for (const c of [entry(label), at(kernels, `${label} | ${pipeline}`)]) { c.direct++; c.workgroups += x * y * z; } target.dispatchWorkgroups(x, y, z); };
    if (key === "dispatchWorkgroupsIndirect") return (buffer: GPUBuffer, offset: number) => { entry(label).indirect++; at(kernels, `${label} | ${pipeline}`).indirect++; target.dispatchWorkgroupsIndirect(buffer, offset); };
    const value = Reflect.get(target, key);
    return typeof value === "function" ? value.bind(target) : value;
  } }); };
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
      if (key === "createComputePipeline") return (descriptor: GPUComputePipelineDescriptor) => { const p = target.createComputePipeline(descriptor); names.set(p, nameOf(descriptor)); return p; };
      if (key === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => { const p = await target.createComputePipelineAsync(descriptor); names.set(p, nameOf(descriptor)); return p; };
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    } }),
    take() {
      const out = Object.fromEntries([...counts].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0));
      counts = new Map(); return out;
    },
    takeKernels() {
      const out = Object.fromEntries([...kernels].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0));
      kernels = new Map(); return out;
    },
  };
  return counter;
}
