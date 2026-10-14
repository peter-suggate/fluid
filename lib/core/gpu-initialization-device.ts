import { aliasGPUCompilationDevice, gpuCompilationManagerFor, managedGPUDevice } from "./gpu-compilation-manager";

/** A build owns its allocations even when a constructor throws before its
 * result can be assigned. After commit the same capability becomes the live
 * solver's device; later scene cancellation cannot invalidate that solver. */
export function gpuInitializationDevice(device: GPUDevice, signal: AbortSignal) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal.aborted) abort();
  else signal.addEventListener("abort", abort, { once: true });
  const resources = new Set<GPUBuffer | GPUTexture | GPUQuerySet>();
  const methods = new Map<string | symbol, unknown>();
  const manager = gpuCompilationManagerFor(device);
  // Harness callers may supply a raw device. All compilation resources still
  // pass through the same manager, including modules and structural layouts.
  device = managedGPUDevice(device);
  let committed = false;
  const assertActive = () => {
    if (!committed && controller.signal.aborted)
      throw new DOMException("GPU initialization superseded", "AbortError");
  };
  const scoped = new Proxy(device, {
    get(target, property) {
      const cached = methods.get(property);
      if (cached) return cached;
      const value = Reflect.get(target, property, target);
      if (typeof value !== "function") return value;
      let method: unknown;
      if (property === "createComputePipelineAsync") {
        method = (descriptor: GPUComputePipelineDescriptor) => manager.compileComputePipeline(descriptor,
          committed ? {} : { signal: controller.signal });
      } else if (property === "createRenderPipelineAsync") {
        method = (descriptor: GPURenderPipelineDescriptor) => manager.compileRenderPipeline(descriptor,
          committed ? {} : { signal: controller.signal });
      } else if (typeof property === "string" && property.startsWith("create")) {
        method = (...args: unknown[]) => {
          assertActive();
          const resource = Reflect.apply(value, target, args);
          if (!committed && ["createBuffer", "createTexture", "createQuerySet"].includes(property)) resources.add(resource as GPUBuffer | GPUTexture | GPUQuerySet);
          return resource;
        };
      } else method = value.bind(target);
      methods.set(property, method);
      return method;
    },
  });
  aliasGPUCompilationDevice(device, scoped);
  return {
    device: scoped,
    commit() {
      assertActive();
      committed = true;
      resources.clear();
      signal.removeEventListener("abort", abort);
    },
    rollback() {
      controller.abort();
      signal.removeEventListener("abort", abort);
      for (const resource of resources) {
        try { resource.destroy(); } catch { /* The device may already be lost. */ }
      }
      resources.clear();
    },
  };
}
