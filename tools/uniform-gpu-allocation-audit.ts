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
        depth32float: 4, r32float: 4, r32uint: 4, r32sint: 4, rg32float: 8, rg32uint: 8,
        rgba32float: 16, rgba32uint: 16, rgba32sint: 16,
        rgba16float: 8, r16float: 2, rg16float: 4, rgba8unorm: 4, "rgba8unorm-srgb": 4,
        r8unorm: 1, rg8unorm: 2, rg11b10ufloat: 4,
      };
      const size = bytesPerTexel[descriptor.format];
      if (size === undefined || (descriptor.sampleCount ?? 1) !== 1)
        throw new Error(`Allocation audit needs an exact byte rule for ${descriptor.label ?? ""}: ${descriptor.format}`);
      // Normalize an iterable once: callers may supply a single-use iterator.
      const extent = Symbol.iterator in descriptor.size
        ? [...descriptor.size as Iterable<number>]
        : undefined;
      const dimensions = extent
        ? { width: extent[0]!, height: extent[1] ?? 1, depthOrArrayLayers: extent[2] ?? 1 }
        : descriptor.size as GPUExtent3DDict;
      let bytes = 0;
      for (let mip=0; mip<(descriptor.mipLevelCount??1); mip++) {
        const depth=descriptor.dimension === "3d" ? Math.max(1,(dimensions.depthOrArrayLayers??1)>>mip) : (dimensions.depthOrArrayLayers??1);
        bytes += Math.max(1,dimensions.width>>mip)*Math.max(1,(dimensions.height??1)>>mip)*depth*size;
      }
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

/** Allocation classes for byte reports, by resource label (the first match wins). */
export const UNIFORM_GPU_ALLOCATION_CLASSES: readonly (readonly [string, RegExp])[] = [
  ["particles", /particle|FLIP (cell starts|bins|next|slab)/i],
  ["pressure", /pressure|arena|multigrid|brick/i],
  ["h detail fields", /velocity|volume|surface|gamma|phi|detail|unit tap/i],
  ["ownership/layout", /owner|layout|tile|hanging|seam|census|mixed/i],
  ["page pool", /page|pool|block/i],
  ["scratch", /scratch|stage|conditioning/i],
];
export const uniformGPUAllocationClass = (label: string) => UNIFORM_GPU_ALLOCATION_CLASSES.find(([, r]) => r.test(label))?.[0] ?? "other";
