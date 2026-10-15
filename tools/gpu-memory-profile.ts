/** Requested GPU resource bytes, excluding driver tiling/alignment and caches.
 * Tracks explicit destruction without wrapping resource identities passed to Dawn. */
export class GPUMemoryProfile {
  readonly device: GPUDevice;
  private readonly live = new Map<object, { label: string; kind: string; bytes: number }>();
  private bytes = 0;
  private peakBytes = 0;
  constructor(raw: GPUDevice) {
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createBuffer") return (d: GPUBufferDescriptor) => {
        const b = target.createBuffer(d); this.track(b, "buffer", b.size); return b;
      };
      if (key === "createTexture") return (d: GPUTextureDescriptor) => {
        const t = target.createTexture(d);
        const components = t.format.startsWith("rgba") ? 4 : t.format.startsWith("rg") ? 2 : 1;
        const bits = /(?:rgba|rg|r)(8|16|32)/.exec(t.format)?.[1];
        if (!bits) throw new Error(`Memory profile needs byte size for ${t.format}`);
        let texels = 0;
        for (let m = 0; m < t.mipLevelCount; m++) texels += Math.max(1, t.width >> m) * Math.max(1, t.height >> m) * (t.dimension === "3d" ? Math.max(1, t.depthOrArrayLayers >> m) : t.depthOrArrayLayers);
        this.track(t, "texture", texels * components * Number(bits) / 8 * t.sampleCount); return t;
      };
      const v = Reflect.get(target, key, target); return typeof v === "function" ? v.bind(target) : v;
    } });
  }
  private track(resource: GPUBuffer | GPUTexture, kind: string, bytes: number): void {
    this.live.set(resource, { label: resource.label, kind, bytes }); this.bytes += bytes;
    this.peakBytes = Math.max(this.peakBytes, this.bytes);
    const destroy = resource.destroy.bind(resource);
    resource.destroy = () => { if (this.live.delete(resource)) this.bytes -= bytes; destroy(); };
  }
  get allocatedBytes(): number { return this.bytes; }
  snapshot() {
    return { bytes: this.bytes, peakBytes: this.peakBytes, resources: [...this.live.values()].sort((a, b) => b.bytes - a.bytes) };
  }
}
