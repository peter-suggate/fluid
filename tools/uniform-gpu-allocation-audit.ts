/** Requested GPU resource bytes, measured at the device boundary. Driver heap
 * overhead and temporary compiler allocations are outside WebGPU's visibility. */
export function auditUniformGPUAllocations(device: GPUDevice, capBytes = Number.POSITIVE_INFINITY) {
  const records: { kind: "buffer" | "texture"; label: string; bytes: number; live: boolean }[] = [];
  let liveBytes = 0, peakBytes = 0;
  const check = (label: string | undefined, bytes: number) => {
    if (liveBytes + bytes > capBytes) throw new Error(`Allocation audit: ${label ?? ""} (${bytes} bytes) takes requested bytes past the cap ${capBytes} (live ${liveBytes})`);
  };
  const track = <T extends GPUBuffer | GPUTexture>(resource: T, kind: "buffer" | "texture", bytes: number): T => {
    const record = { kind, label: resource.label, bytes, live: true };
    records.push(record); liveBytes += bytes; peakBytes = Math.max(peakBytes, liveBytes);
    const destroy = resource.destroy.bind(resource);
    resource.destroy = () => { if (record.live) { record.live = false; liveBytes -= bytes; } destroy(); };
    return resource;
  };
  const auditedDevice = new Proxy(device, { get(target, key) {
    if (key === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
      check(descriptor.label, descriptor.size);
      return track(target.createBuffer(descriptor), "buffer", descriptor.size);
    };
    if (key === "createTexture") return (descriptor: GPUTextureDescriptor) => {
      const bytesPerTexel: Partial<Record<GPUTextureFormat, number>> = {
        r32float: 4, r32uint: 4, r32sint: 4, rg32float: 8, rg32uint: 8,
        rgba32float: 16, rgba32uint: 16, rgba32sint: 16,
        rgba16float: 8, r16float: 2, rg16float: 4, rgba8unorm: 4,
        r8unorm: 1, rg8unorm: 2, rg11b10ufloat: 4,
      };
      const size = bytesPerTexel[descriptor.format];
      if (size === undefined || (descriptor.mipLevelCount ?? 1) !== 1 || (descriptor.sampleCount ?? 1) !== 1)
        throw new Error(`Allocation audit needs an exact byte rule for ${descriptor.label ?? ""}: ${descriptor.format}`);
      // Normalize an iterable once: callers may supply a single-use iterator.
      const extent = Symbol.iterator in descriptor.size
        ? [...descriptor.size as Iterable<number>]
        : undefined;
      const dimensions = extent
        ? { width: extent[0]!, height: extent[1] ?? 1, depthOrArrayLayers: extent[2] ?? 1 }
        : descriptor.size as GPUExtent3DDict;
      const bytes = dimensions.width * (dimensions.height ?? 1) * (dimensions.depthOrArrayLayers ?? 1) * size;
      check(descriptor.label, bytes);
      return track(target.createTexture({ ...descriptor, size: dimensions }), "texture", bytes);
    };
    const value = Reflect.get(target, key);
    return typeof value === "function" ? value.bind(target) : value;
  }});
  return { device: auditedDevice, snapshot() {
    const groups = new Map<string, { kind: string; label: string; count: number; bytes: number }>();
    for (const r of records.filter(r => r.live)) {
      const key = `${r.kind}:${r.label}`;
      const group = groups.get(key) ?? { kind: r.kind, label: r.label, count: 0, bytes: 0 };
      group.count++; group.bytes += r.bytes; groups.set(key, group);
    }
    return { liveBytes, peakBytes, resources: [...groups.values()].sort((a,b) => b.bytes-a.bytes) };
  }};
}
