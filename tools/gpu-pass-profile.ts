/** Diagnostic pass timestamps. Preserves pass boundaries and dispatch order.
 * A timestamped pass costs about 34 microseconds of GPU idle on Metal that an
 * untimed one does not (5 at most): a profiled frame's idle_ms, about 2.4 ms
 * over the figure-9 teaser's 76 passes, is the profile's and not the frame's. */
export class GPUPassProfile {
  readonly device: GPUDevice;
  private active = false;
  private rows: { label: string; dispatches: number }[] = [];
  private readonly queries: GPUQuerySet;
  private readonly pipelineNames = new WeakMap<GPUComputePipeline, string>();
  /** splitDispatches is diagnostic only: additional pass boundaries change
   * dispatch overhead, so its whole-frame time is not a performance result. */
  constructor(private readonly raw: GPUDevice, private readonly capacity = 4096, private readonly splitDispatches = false) {
    this.queries = raw.createQuerySet({ type: "timestamp", count: capacity });
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createComputePipeline") return (descriptor: GPUComputePipelineDescriptor) => {
        const pipeline = target.createComputePipeline(descriptor);
        this.pipelineNames.set(pipeline, descriptor.compute.entryPoint ?? descriptor.label ?? "unnamed");
        return pipeline;
      };
      if (key === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(descriptor);
        this.pipelineNames.set(pipeline, descriptor.compute.entryPoint ?? descriptor.label ?? "unnamed");
        return pipeline;
      };
      if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
        const encoder = target.createCommandEncoder(descriptor);
        return new Proxy(encoder, { get: (e, property) => {
          if (property === "beginComputePass") return (d?: GPUComputePassDescriptor) => {
            if (!this.active) return e.beginComputePass(d);
            if (d?.timestampWrites) throw new Error("Disable other timestamp instrumentation");
            if (this.splitDispatches) {
              let pipeline: GPUComputePipeline | undefined;
              const bindings = new Map<number, { group: GPUBindGroup | null | undefined; offsets?: number[] }>();
              const dispatch = (indirect: boolean, args: number[] | [GPUBuffer, number]) => {
                if (!pipeline) throw new Error("Dispatch without a pipeline");
                const index = this.rows.length * 2;
                if (index + 1 >= capacity) throw new Error("Kernel timestamp capacity exceeded");
                this.rows.push({ label: `${d?.label ?? "unlabelled"} / ${this.pipelineNames.get(pipeline) ?? pipeline.label}`, dispatches: 1 });
                const pass = e.beginComputePass({ ...d, timestampWrites: { querySet: this.queries,
                  beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
                pass.setPipeline(pipeline);
                for (const [index, binding] of bindings) pass.setBindGroup(index, binding.group, binding.offsets);
                Reflect.apply(indirect ? pass.dispatchWorkgroupsIndirect : pass.dispatchWorkgroups, pass, args);
                pass.end();
              };
              return {
                setPipeline: (value: GPUComputePipeline) => { pipeline = value; },
                setBindGroup: (index: number, group: GPUBindGroup | null | undefined, offsets?: Iterable<number>, start?: number, length?: number) => {
                  const values = offsets === undefined ? undefined : Array.from(offsets);
                  bindings.set(index, { group, offsets: start === undefined ? values : values?.slice(start, start + length!) });
                },
                dispatchWorkgroups: (...args: number[]) => dispatch(false, args),
                dispatchWorkgroupsIndirect: (buffer: GPUBuffer, offset: number) => dispatch(true, [buffer, offset]),
                insertDebugMarker() {}, pushDebugGroup() {}, popDebugGroup() {}, end() {},
              };
            }
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
