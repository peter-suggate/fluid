/** Snapshot WebGPU dictionaries before deferred compilation. Sequences may be
 * one-shot iterators; GPU objects must retain identity, not be cloned as {}. */
export function snapshotGPUDescriptor<T>(value: T, property = ""): T {
  if (value === null || typeof value !== "object") return value;
  if (property === "module" || property === "layout") return value;
  if (property === "bindGroupLayouts") return Array.from(value as unknown as Iterable<unknown>) as T;
  if (Symbol.iterator in value) {
    return Array.from(value as unknown as Iterable<unknown>, item => snapshotGPUDescriptor(item)) as T;
  }
  return Object.fromEntries(Object.entries(value).map(([key, item]) =>
    [key, snapshotGPUDescriptor(item, key)])) as T;
}

/** Exact, collision-free keys. Only the descriptor's diagnostic label is
 * ignored (a specialization constant named `label` is still significant).
 * Unknown/raw GPU objects are keyed by identity, never guessed equivalent.
 * Managed modules/layouts have already been interned by their descriptors. */
export class GPUCompilationDescriptorKeys {
  private readonly identities = new WeakMap<object, number>();
  private nextIdentity = 0;

  private identity(value: object): number {
    let id = this.identities.get(value);
    if (id === undefined) { id = this.nextIdentity++; this.identities.set(value, id); }
    return id;
  }

  key(descriptor: object): string {
    const { label: _label, ...state } = descriptor as Record<string, unknown>;
    return JSON.stringify(this.value(state));
  }

  private value(value: unknown, property = ""): unknown {
    if (value === undefined) return ["undefined"];
    if (Object.is(value, -0)) return ["number", "-0"];
    if (typeof value === "number" && !Number.isFinite(value)) return ["number", String(value)];
    if (value === null || typeof value !== "object") return value;
    if (property === "module" || property === "layout") return ["gpu", this.identity(value)];
    if (property === "bindGroupLayouts") {
      return ["layouts", ...Array.from(value as Iterable<object | null | undefined>,
        layout => layout == null ? null : this.identity(layout))];
    }
    if (Array.isArray(value)) return ["array", ...value.map(item => this.value(item))];
    return ["dictionary", ...Object.keys(value).sort()
      .filter(key => (value as Record<string, unknown>)[key] !== undefined)
      .map(key => [key, this.value((value as Record<string, unknown>)[key], key)])];
  }
}

export function normalizedBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayoutDescriptor {
  return { ...descriptor, entries: Array.from(descriptor.entries, entry => ({
    ...entry,
    ...(entry.buffer ? { buffer: { type: "uniform", hasDynamicOffset: false, minBindingSize: 0, ...entry.buffer } } : {}),
    ...(entry.sampler ? { sampler: { type: "filtering", ...entry.sampler } } : {}),
    ...(entry.texture ? { texture: { sampleType: "float", viewDimension: "2d", multisampled: false, ...entry.texture } } : {}),
    ...(entry.storageTexture ? { storageTexture: { access: "write-only", viewDimension: "2d", ...entry.storageTexture } } : {}),
  } as GPUBindGroupLayoutEntry)).sort((a, b) => a.binding - b.binding) };
}
