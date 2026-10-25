/** Benchmark-only command replay: omit one pipeline's draws from the production
 * bundle. Bindings, attachments, culling and all other draws are unchanged.
 * Differences are marginal costs, not additive per-object GPU timings.
 * An optional shader reference replays every draw with one alternate pipeline
 * for same-device interleaved comparisons. Never used by production rendering. */
export interface PrimaryShaderReference { pipeline: string; search: string; replacement: string }
/** Capture immutable asset indices, or supply a previous ordering for replay.
 * Reference indices must address the same vertex buffer and LOD ranges. */
export interface PrimaryIndexReference {
  replay: boolean;
  read: (label: string, candidate: Uint32Array) => Uint32Array | undefined;
}
export function primaryDrawAblation(device: GPUDevice, reference?: PrimaryShaderReference, indexReference?: PrimaryIndexReference) {
  const variants = new WeakMap<GPURenderBundle, Map<string, GPURenderBundle>>();
  const referenceIndices = new WeakMap<GPUBuffer, GPUBuffer>();
  const referenceModules = new WeakMap<GPUShaderModule, GPUShaderModule>();
  const referencePipelines = new WeakMap<GPURenderPipeline, GPURenderPipeline>();
  const labels = new Set<string>();
  const wrappedDevice = new Proxy(device, {
    get(target, property) {
      if (indexReference && property === 'createBuffer') return (descriptor: GPUBufferDescriptor) => {
        const buffer = target.createBuffer(descriptor);
        if (descriptor.mappedAtCreation && descriptor.label?.endsWith(' native surface indices')) {
          const unmap = buffer.unmap.bind(buffer);
          const getMappedRange = buffer.getMappedRange.bind(buffer);
          let mapped: ArrayBuffer | undefined;
          buffer.getMappedRange = (offset, size) => {
            const range = getMappedRange(offset, size);
            if ((offset ?? 0) !== 0 || range.byteLength !== Number(descriptor.size)) throw new Error('Index capture requires the full mapped range');
            mapped = range;
            return range;
          };
          buffer.unmap = () => {
            // WebGPU forbids requesting an overlapping mapped range twice.
            const words = new Uint32Array(mapped ?? getMappedRange());
            const original = indexReference.read(descriptor.label!, words);
            if (original) {
              if (original.byteLength !== words.byteLength) throw new Error(`Reference index size mismatch: ${descriptor.label}`);
              const previous = target.createBuffer({ ...descriptor, label: `${descriptor.label} reference` });
              new Uint32Array(previous.getMappedRange()).set(original);
              previous.unmap();
              referenceIndices.set(buffer, previous);
              const destroy = buffer.destroy.bind(buffer);
              buffer.destroy = () => { previous.destroy(); destroy(); };
            }
            buffer.getMappedRange = getMappedRange;
            buffer.unmap = unmap;
            unmap();
          };
        }
        return buffer;
      };
      if (reference && property === 'createShaderModule') return (descriptor: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(descriptor);
        if (descriptor.code.includes(reference.search)) referenceModules.set(module, target.createShaderModule({
          ...descriptor, label: `${descriptor.label} reference`, code: descriptor.code.replaceAll(reference.search, reference.replacement),
        }));
        return module;
      };
      if (reference && property === 'createRenderPipelineAsync') return async (descriptor: GPURenderPipelineDescriptor) => {
        const pipeline = await target.createRenderPipelineAsync(descriptor);
        if (descriptor.label === reference.pipeline) {
          const module = descriptor.fragment && referenceModules.get(descriptor.fragment.module);
          if (!module) throw new Error(`Reference shader replacement did not match ${reference.pipeline}`);
          referencePipelines.set(pipeline, await target.createRenderPipelineAsync({ ...descriptor,
            label: `${descriptor.label} reference`, fragment: { ...descriptor.fragment!, module },
            vertex: { ...descriptor.vertex, module: referenceModules.get(descriptor.vertex.module) ?? descriptor.vertex.module },
          }));
        }
        return pipeline;
      };
      if (property === 'createRenderBundleEncoder') return (descriptor: GPURenderBundleEncoderDescriptor) => {
        const original = target.createRenderBundleEncoder(descriptor);
        if (descriptor.label !== 'Cached primary draw commands') return original;
        const commands: { name: string; args: unknown[] }[] = [];
        const pipelines = new Set<string>();
        return new Proxy(original, {
          get(bundle, name) {
            if (name === 'finish') return (finish?: GPURenderBundleDescriptor) => {
              const result = bundle.finish(finish);
              if (reference && !commands.some(command => command.name === 'setPipeline'
                && referencePipelines.has(command.args[0] as GPURenderPipeline))) {
                throw new Error(`Reference pipeline not drawn: ${reference.pipeline}`);
              }
              if (indexReference?.replay && !commands.some(command => command.name === 'setIndexBuffer'
                && referenceIndices.has(command.args[0] as GPUBuffer))) {
                throw new Error('Reference index buffer not drawn');
              }
              const alternatives = new Map<string, GPURenderBundle>();
              for (const omitted of [...pipelines, ...(reference || indexReference?.replay ? ['@reference'] : [])]) {
                labels.add(omitted);
                const replay = target.createRenderBundleEncoder(descriptor);
                let pipeline = '';
                for (const command of commands) {
                  if (command.name === 'setPipeline') pipeline = (command.args[0] as GPURenderPipeline).label;
                  if (command.name.startsWith('draw') && pipeline === omitted) continue;
                  const args = omitted === '@reference' && command.name === 'setPipeline'
                    ? [referencePipelines.get(command.args[0] as GPURenderPipeline) ?? command.args[0]]
                    : omitted === '@reference' && command.name === 'setIndexBuffer'
                      ? [referenceIndices.get(command.args[0] as GPUBuffer) ?? command.args[0], ...command.args.slice(1)] : command.args;
                  Reflect.get(replay, command.name).apply(replay, args);
                }
                alternatives.set(omitted, replay.finish({ label: omitted === '@reference' ? 'Primary reference' : `Primary without ${omitted}` }));
              }
              variants.set(result, alternatives);
              return result;
            };
            const value = Reflect.get(bundle, name, bundle);
            if (typeof value !== 'function') return value;
            return (...args: unknown[]) => {
              commands.push({ name: String(name), args });
              if (name === 'setPipeline') pipelines.add((args[0] as GPURenderPipeline).label);
              return value.apply(bundle, args);
            };
          },
        });
      };
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return { device: wrappedDevice, labels, omit(encoder: GPUCommandEncoder, pipeline: string): GPUCommandEncoder {
    return new Proxy(encoder, {
      get(target, property) {
        if (property === 'beginRenderPass') return (descriptor: GPURenderPassDescriptor) => {
          const pass = target.beginRenderPass(descriptor);
          if (descriptor.label !== 'Voxel surface mesh rasterization') return pass;
          return new Proxy(pass, {
            get(draw, name) {
              if (name === 'executeBundles') return (bundles: Iterable<GPURenderBundle>) =>
                draw.executeBundles(Array.from(bundles, bundle => variants.get(bundle)?.get(pipeline) ?? bundle));
              const value = Reflect.get(draw, name, draw);
              return typeof value === 'function' ? value.bind(draw) : value;
            },
          });
        };
        const value = Reflect.get(target, property, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
  } };
}
