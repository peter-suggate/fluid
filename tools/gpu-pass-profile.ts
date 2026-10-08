/** Diagnostic pass timestamps. Preserves pass boundaries and dispatch order. */
export class GPUPassProfile {
  readonly device: GPUDevice;
  private active = false;
  private rows: { label: string; dispatches: number }[] = [];
  private readonly queries: GPUQuerySet;
  constructor(private readonly raw: GPUDevice, private readonly capacity = 4096) {
    this.queries = raw.createQuerySet({ type: "timestamp", count: capacity });
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
        const encoder = target.createCommandEncoder(descriptor);
        return new Proxy(encoder, { get: (e, property) => {
          if (property === "beginComputePass") return (d?: GPUComputePassDescriptor) => {
            if (!this.active) return e.beginComputePass(d);
            if (d?.timestampWrites) throw new Error("Disable other timestamp instrumentation");
            const index = this.rows.length * 2;
            if (index + 1 >= capacity) throw new Error("Pass timestamp capacity exceeded");
            const row = { label: d?.label ?? "unlabelled", dispatches: 0 }; this.rows.push(row);
            const pass = e.beginComputePass({ ...d, timestampWrites: { querySet: this.queries,
              beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
            return new Proxy(pass, { get: (p, k) => {
              const v = Reflect.get(p, k, p);
              if (k === "dispatchWorkgroups" || k === "dispatchWorkgroupsIndirect") return (...args: unknown[]) => {
                row.dispatches++; return Reflect.apply(v, p, args);
              };
              return typeof v === "function" ? v.bind(p) : v;
            } });
          };
          const v = Reflect.get(e, property, e); return typeof v === "function" ? v.bind(e) : v;
        } });
      };
      const v = Reflect.get(target, key, target); return typeof v === "function" ? v.bind(target) : v;
    } });
  }
  start(): void { this.rows = []; this.active = true; }
  async finish() {
    this.active = false;
    const count = this.rows.length * 2;
    if (!count) throw new Error("No passes captured");
    const output = this.raw.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.raw.createBuffer({ size: count * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const e = this.raw.createCommandEncoder(); e.resolveQuerySet(this.queries, 0, count, output, 0);
      e.copyBufferToBuffer(output, 0, read, 0, count * 8); this.raw.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const times = new BigUint64Array(read.getMappedRange());
      if (times.every(t => t === 0n)) throw new Error("GPU timestamps were not written");
      // idle_ms is the time since the previous pass ended: the copies and clears encoded between the two.
      const passes = this.rows.map((r, i) => ({ ...r, ms: Number(times[2 * i + 1] - times[2 * i]) / 1e6,
        idle_ms: i ? Number(times[2 * i] - times[2 * i - 1]) / 1e6 : 0 }));
      read.unmap(); return passes;
    } finally { output.destroy(); read.destroy(); }
  }
  destroy(): void { this.queries.destroy(); }
}
