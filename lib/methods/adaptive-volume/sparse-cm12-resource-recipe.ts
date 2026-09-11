/** Serializable GPU construction commands. A CPU-only preparation worker records
 * resource creation; the advancing worker realizes it cooperatively on its device.
 * No GPUDevice or live simulation resource crosses a worker boundary. */
export interface CM12ResourceReference { readonly cm12Resource: number }
export interface CM12ResourceOperation {
  readonly target: "device" | "queue" | number;
  readonly method: string;
  readonly args: readonly unknown[];
  readonly result?: number;
}
export interface CM12ResourceRecipe {
  readonly operations: readonly CM12ResourceOperation[];
  readonly state: unknown;
  /** Temporary shader/encoder handles die after their last construction use. */
  readonly releaseAfter?: readonly (readonly number[])[];
}
const resourceId = Symbol("cm12Resource");
const creationMethods = new Set(["createBuffer", "createBindGroupLayout", "createPipelineLayout",
  "createBindGroup", "createShaderModule", "createComputePipelineAsync", "createCommandEncoder"]);
const encoderResults = new Set(["beginComputePass", "finish"]);
const encoderMethods = new Set(["setPipeline", "setBindGroup", "dispatchWorkgroups",
  "dispatchWorkgroupsIndirect", "end", "copyBufferToBuffer", "clearBuffer",
  "pushDebugGroup", "popDebugGroup", "insertDebugMarker"]);

export function createCM12ResourceRecorder(limits: GPUSupportedLimits,
  external: readonly { size: number; usage: number; label?: string }[] = []) {
  const operations: CM12ResourceOperation[] = [];
  const resources = new Map<number, object>();
  let nextId = external.length;
  const encode = (value: unknown, seen = new Map<object, unknown>()): unknown => {
    if (value === null || typeof value !== "object") {
      if (typeof value === "function") throw new Error("CM12 resource recipe cannot contain a callback");
      return value;
    }
    const id = (value as Record<symbol, number>)[resourceId];
    if (id !== undefined) return { cm12Resource: id };
    if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return value;
    if (seen.has(value)) return seen.get(value);
    if (value instanceof Map) {
      const result = new Map(); seen.set(value, result);
      for (const [key, entry] of value) result.set(encode(key, seen), encode(entry, seen));
      return result;
    }
    if (value instanceof Set) {
      const result = new Set(); seen.set(value, result);
      for (const entry of value) result.add(encode(entry, seen));
      return result;
    }
    const result: Record<string, unknown> | unknown[] = Array.isArray(value) ? [] : {};
    seen.set(value, result);
    for (const [key, entry] of Object.entries(value)) {
      if (typeof entry === "function" || entry instanceof Promise) continue;
      (result as Record<string, unknown>)[key] = encode(entry, seen);
    }
    return result;
  };
  const record = (target: CM12ResourceOperation["target"], method: string,
    args: unknown[], result?: number) => operations.push({ target, method,
      args: encode(args) as unknown[], result });
  const resource = (id: number, descriptor: Record<string, unknown> = {}) => {
    let mapped: ArrayBuffer | undefined;
    const object = new Proxy({ [resourceId]: id, ...descriptor }, { get(target, property) {
      if (property === resourceId) return id;
      if (property === "getCompilationInfo") return async () => ({ messages: [] });
      if (property === "getMappedRange") return () => mapped ??= new ArrayBuffer(Number(descriptor.size));
      if (property === "unmap") return () => {
        if (mapped) record("queue", "writeBuffer", [object, 0, mapped]);
        mapped = undefined;
      };
      if (property === "destroy") return () => record(id, "destroy", []);
      if (typeof property === "string" && encoderResults.has(property)) return (...args: unknown[]) => {
        const result = nextId++; record(id, property, args, result); return resource(result);
      };
      if (typeof property === "string" && encoderMethods.has(property))
        return (...args: unknown[]) => record(id, property, args);
      return Reflect.get(target, property);
    } });
    resources.set(id, object); return object;
  };
  external.forEach((descriptor, id) => resource(id, descriptor));
  const queue = {
    writeBuffer(buffer: unknown, offset: number, data: ArrayBuffer | ArrayBufferView,
      dataOffset = 0, size?: number) {
      const elementBytes = ArrayBuffer.isView(data) && "BYTES_PER_ELEMENT" in data
        ? Number(data.BYTES_PER_ELEMENT) : 1;
      const bytes = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
      const start = dataOffset * elementBytes;
      const copy = bytes.slice(start, size === undefined ? undefined : start + size * elementBytes);
      record("queue", "writeBuffer", [buffer, offset, copy]);
    },
    submit(commands: unknown[]) { record("queue", "submit", [commands]); },
    onSubmittedWorkDone: async () => {},
  };
  const device = new Proxy({ limits, features: new Set<string>(), queue,
    lost: new Promise(() => {}), addEventListener() {}, removeEventListener() {} }, {
    get(target, property) {
      if (typeof property === "string" && creationMethods.has(property)) return (descriptor: Record<string, unknown>) => {
        const id = nextId++;
        const uploadedDescriptor = property === "createBuffer" ? { ...descriptor, mappedAtCreation: false,
          usage: Number(descriptor.usage) | (descriptor.mappedAtCreation ? 8 /* COPY_DST */ : 0) } : descriptor;
        record("device", property, [uploadedDescriptor], id);
        const result = resource(id, descriptor);
        return property.endsWith("Async") ? Promise.resolve(result) : result;
      };
      return Reflect.get(target, property);
    },
  }) as unknown as GPUDevice;
  return { device, externalResources: [...resources.values()],
    finish(state: unknown): CM12ResourceRecipe {
      const encodedState = encode(state);
      const references = (value: unknown, visit: (id: number) => void, seen = new Set<object>()) => {
        if (!value || typeof value !== "object" || seen.has(value)) return;
        seen.add(value);
        if ("cm12Resource" in value) { visit((value as CM12ResourceReference).cm12Resource); return; }
        if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return;
        if (value instanceof Map) { for (const [key, entry] of value) { references(key, visit, seen); references(entry, visit, seen); } return; }
        if (value instanceof Set) { for (const entry of value) references(entry, visit, seen); return; }
        for (const entry of Object.values(value)) references(entry, visit, seen);
      };
      const lastUse = new Map<number, number>();
      operations.forEach((operation, index) => {
        if (operation.result !== undefined) lastUse.set(operation.result, index);
        if (typeof operation.target === "number") lastUse.set(operation.target, index);
        references(operation.args, id => lastUse.set(id, index));
      });
      references(encodedState, id => lastUse.delete(id));
      const releaseAfter: number[][] = Array.from({length: operations.length}, () => []);
      for (const [id, index] of lastUse) releaseAfter[index]!.push(id);
      return { operations, state: encodedState, releaseAfter };
    } };
}

/** Realize a recipe with bounded upload chunks and cooperative CPU slices.
 * Driver pipeline compilation remains asynchronous and manager-owned. */
export async function realizeCM12ResourceRecipe(device: GPUDevice, recipe: CM12ResourceRecipe,
  external: readonly object[] = [], options: { compilationDevice?: GPUDevice; maximumSliceMs?: number; uploadChunkBytes?: number;
    signal?: AbortSignal; onSlice?: (milliseconds: number, operation?: string) => void } = {}): Promise<{
      state: unknown; destroy(): void;
    }> {
  const { gpuCompilationManagerFor } = await import("../../core/gpu-compilation-manager");
  const compiler = gpuCompilationManagerFor(options.compilationDevice ?? device);
  const resources = new Map<number, unknown>(external.map((resource, id) => [id, resource]));
  const buffers: GPUBuffer[] = [];
  const decode = (value: unknown, seen = new Map<object, unknown>()): unknown => {
    if (value === null || typeof value !== "object") return value;
    if ("cm12Resource" in value) {
      const id = (value as CM12ResourceReference).cm12Resource;
      if (!resources.has(id)) throw new Error(`CM12 resource ${id} has no prepared dependency`);
      return resources.get(id);
    }
    if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return value;
    if (seen.has(value)) return seen.get(value);
    if (value instanceof Map) {
      const result = new Map(); seen.set(value, result);
      for (const [key, entry] of value) result.set(decode(key, seen), decode(entry, seen));
      return result;
    }
    if (value instanceof Set) {
      const result = new Set(); seen.set(value, result);
      for (const entry of value) result.add(decode(entry, seen));
      return result;
    }
    const result: Record<string, unknown> | unknown[] = Array.isArray(value) ? [] : {};
    seen.set(value, result);
    for (const [key, entry] of Object.entries(value)) (result as Record<string, unknown>)[key] = decode(entry, seen);
    return result;
  };
  let sliceStart = performance.now();
  let sliceOperation = "begin";
  const checkpoint = async () => {
    options.signal?.throwIfAborted();
    const elapsed = performance.now() - sliceStart;
    if (elapsed < (options.maximumSliceMs ?? 2)) return;
    options.onSlice?.(elapsed, sliceOperation);
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    sliceStart = performance.now();
  };
  try {
    for (const [operationIndex, operation] of recipe.operations.entries()) {
      await checkpoint();
      const args = decode(operation.args) as unknown[];
      sliceOperation = `${operation.method}: ${(args[0] as {label?: string})?.label ?? ""}`;
      if (operation.target === "queue" && operation.method === "writeBuffer") {
        const data = args[2] as ArrayBuffer | Uint8Array;
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const chunk = options.uploadChunkBytes ?? 1024 * 1024;
        for (let offset = 0; offset < bytes.byteLength; offset += chunk) {
          device.queue.writeBuffer(args[0] as GPUBuffer, Number(args[1]) + offset,
            bytes.buffer as ArrayBuffer, bytes.byteOffset + offset, Math.min(chunk, bytes.byteLength - offset));
          await checkpoint();
        }
        for (const id of recipe.releaseAfter?.[operationIndex] ?? []) resources.delete(id);
        continue;
      }
      const target = operation.target === "device" ? device : operation.target === "queue" ? device.queue
        : resources.get(operation.target);
      let result: unknown;
      if (operation.method === "createShaderModule") result = compiler.createShaderModule(args[0] as GPUShaderModuleDescriptor);
      else if (operation.method === "createComputePipelineAsync") {
        result = await compiler.compileComputePipeline(args[0] as GPUComputePipelineDescriptor, { priority: "background" });
        sliceStart = performance.now();
      } else {
        const method = (target as Record<string, (...args: unknown[]) => unknown>)[operation.method];
        if (typeof method !== "function") throw new Error(`CM12 resource method ${operation.method} is unavailable`);
        result = method.apply(target, args);
      }
      if (operation.result !== undefined) resources.set(operation.result, result);
      if (operation.method === "createBuffer") buffers.push(result as GPUBuffer);
      for (const id of recipe.releaseAfter?.[operationIndex] ?? []) resources.delete(id);
    }
    return { state: decode(recipe.state), destroy() { for (const buffer of buffers) buffer.destroy(); } };
  } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
}
