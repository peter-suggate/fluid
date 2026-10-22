/** Logical texture payload bytes, excluding driver tiling/alignment. Depth24
 * formats use a four-byte depth plane estimate; physical storage is opaque. */
export function gpuTexturePayloadBytes(t: Pick<GPUTexture, "format" | "width" | "height" | "depthOrArrayLayers" | "dimension" | "mipLevelCount" | "sampleCount">): number {
  let blockWidth = 1, blockHeight = 1, blockBytes: number;
  const scalar = /^(r|rg|rgba)(8|16|32)(?:unorm|snorm|uint|sint|float)(?:-srgb)?$/.exec(t.format);
  if (scalar) blockBytes = ({ r: 1, rg: 2, rgba: 4 }[scalar[1]!]!) * Number(scalar[2]) / 8;
  else if (/^(?:bgra8unorm(?:-srgb)?|rgb10a2(?:uint|unorm)|rg11b10ufloat|rgb9e5ufloat|depth24plus|depth32float)$/.test(t.format)) blockBytes = 4;
  else if (t.format === "stencil8") blockBytes = 1;
  else if (t.format === "depth16unorm") blockBytes = 2;
  else if (t.format === "depth24plus-stencil8" || t.format === "depth32float-stencil8") blockBytes = 8;
  else if (/^(?:bc|etc2|eac)/.test(t.format)) {
    blockWidth = blockHeight = 4;
    blockBytes = /^(?:bc1-|bc4-|etc2-rgb8|etc2-rgb8a1|eac-r11)/.test(t.format) ? 8 : 16;
  } else {
    const astc = /^astc-(\d+)x(\d+)-/.exec(t.format);
    if (!astc) throw new Error(`Memory profile needs byte size for ${t.format}`);
    blockWidth = Number(astc[1]); blockHeight = Number(astc[2]); blockBytes = 16;
  }
  let bytes = 0;
  for (let mip = 0; mip < t.mipLevelCount; mip++) {
    const width = Math.max(1, Math.floor(t.width / 2 ** mip));
    const height = Math.max(1, Math.floor(t.height / 2 ** mip));
    const depth = t.dimension === "3d" ? Math.max(1, Math.floor(t.depthOrArrayLayers / 2 ** mip)) : t.depthOrArrayLayers;
    bytes += Math.ceil(width / blockWidth) * Math.ceil(height / blockHeight) * depth * blockBytes * t.sampleCount;
  }
  return bytes;
}

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
        this.track(t, "texture", gpuTexturePayloadBytes(t)); return t;
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
    const resources = [...this.live.values()].sort((a, b) => b.bytes - a.bytes);
    const bufferBytes = resources.reduce((sum, r) => sum + (r.kind === "buffer" ? r.bytes : 0), 0);
    return { bytes: this.bytes, peakBytes: this.peakBytes, bufferBytes, textureBytes: this.bytes - bufferBytes,
      resources };
  }
}
