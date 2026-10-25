/** Benchmark-only shader substitution. Both arms use production resources;
 * explicitly named pipelines can differ and candidate-only passes can be omitted. */
export interface ComputeShaderReference {
  pipelines: string[];
  replacements: { search: string; replacement: string }[];
  omitPasses?: string[];
}

export function computeShaderReference(device: GPUDevice, reference: ComputeShaderReference) {
  const modules = new WeakMap<GPUShaderModule, GPUShaderModule>();
  const pipelines = new Map<GPUComputePipeline, GPUComputePipeline>();
  const matched = new Set<string>();
  let exactRequests: GPUBuffer | undefined;
  const wrapped = new Proxy(device, {
    get(target, property) {
      if (property === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
        const buffer = target.createBuffer(descriptor);
        if (descriptor.label === "Unrepresented visibility receivers") exactRequests = buffer;
        return buffer;
      };
      if (property === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(descriptor);
        let code = descriptor.code;
        for (const change of reference.replacements) {
          if (code.includes(change.search)) matched.add(change.search);
          code = code.replaceAll(change.search, change.replacement);
        }
        if (code !== descriptor.code) modules.set(module, target.createShaderModule({ ...descriptor, code, label: `${descriptor.label} reference` }));
        return module;
      };
      if (property === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(descriptor);
        if (reference.pipelines.includes(descriptor.label ?? "")) {
          const module = modules.get(descriptor.compute.module);
          if (!module) throw new Error(`No reference substitution for ${descriptor.label}`);
          pipelines.set(pipeline, await target.createComputePipelineAsync({ ...descriptor,
            label: `${descriptor.label} reference`, compute: { ...descriptor.compute, module } }));
        }
        return pipeline;
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  return { device: wrapped, omittedPasses: reference.omitPasses ?? [], async exactReceiverCount(): Promise<number | undefined> {
    if (!exactRequests) return undefined;
    const read = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(exactRequests, 0, read, 0, 4);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      return new Uint32Array(read.getMappedRange())[0];
    } finally { read.destroy(); }
  }, encode(encoder: GPUCommandEncoder): GPUCommandEncoder {
    for (const change of reference.replacements) if (!matched.has(change.search)) throw new Error("Reference shader search did not match");
    for (const label of reference.pipelines) if (![...pipelines.keys()].some(p => p.label === label)) throw new Error(`Reference pipeline not compiled: ${label}`);
    return new Proxy(encoder, {
      get(target, property) {
        if (property === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
          if (reference.omitPasses?.includes(descriptor?.label ?? "")) {
            return new Proxy({ label: descriptor?.label }, { get: (target, property) => property === "label" ? target.label : () => {} }) as GPUComputePassEncoder;
          }
          const pass = target.beginComputePass(descriptor);
          return new Proxy(pass, {
            get(target, property) {
              if (property === "setPipeline") return (pipeline: GPUComputePipeline) => target.setPipeline(pipelines.get(pipeline) ?? pipeline);
              const value = Reflect.get(target, property, target);
              return typeof value === "function" ? value.bind(target) : value;
            },
          });
        };
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  } };
}
