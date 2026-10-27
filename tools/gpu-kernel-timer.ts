/** Per-dispatch GPU time attribution for a probe (profiling only).
 *
 * Wraps a device so that, while enabled, every compute pass of every command
 * encoder it creates is split into one pass per dispatch, each with its own
 * timestamp pair, labelled `<pass label> / <pipeline label>`. Splitting
 * serialises dispatches and adds an encoder break each (a 4-byte copy, so
 * Metal does not fold the passes and drop their samples), so the sum runs
 * above the clean frame: read the table for where the time goes, and score a
 * change on clean wall time, never on these sums.
 *
 *   const timer = gpuKernelTimer(device); // build the solver on timer.device
 *   timer.enabled = true; ...advance...; const ms = await timer.collect();
 */
export interface GPUKernelTimer {
  readonly device: GPUDevice;
  enabled: boolean;
  /** Durations (ms) by label of every pass recorded since the last collect, after the queue drains. */
  collect(): Promise<{ byLabel: Map<string, number>; dispatches: number; untimed: number }>;
}

const QUERIES = 4096;

export function gpuKernelTimer(device: GPUDevice): GPUKernelTimer {
  const pending: { read: GPUBuffer; resolve: GPUBuffer; querySet: GPUQuerySet; labels: string[] }[] = [];
  let untimed = 0;
  // Each pipeline's name: its descriptor's label and entry point.
  const names = new WeakMap<GPUComputePipeline, string>();
  const named = (p: GPUComputePipeline, d: GPUComputePipelineDescriptor) => { const c = d.compute.constants ? Object.entries(d.compute.constants).filter(([k]) => /^um(Fast|Merged|Cubic|Free|Solid|Twin|Regular|H)/.test(k)).map(([k, v]) => `${k}=${v}`).join(",") : ""; names.set(p, `${d.label ?? p.label ?? "?"}:${d.compute.entryPoint ?? "main"}${c ? `{${c}}` : ""}`); return p; };
  const breakSource = device.createBuffer({ label: "kernel timer break source", size: 4, usage: GPUBufferUsage.COPY_SRC });
  const breakTarget = device.createBuffer({ label: "kernel timer break target", size: 4, usage: GPUBufferUsage.COPY_DST });
  const wrapEncoder = (encoder: GPUCommandEncoder): GPUCommandEncoder => {
    let querySet: GPUQuerySet | undefined;
    const labels: string[] = [];
    const open = (descriptor: GPUComputePassDescriptor | undefined): GPUComputePassEncoder => {
      if (2 * labels.length + 2 > QUERIES) { untimed++; return encoder.beginComputePass(descriptor); }
      querySet ??= device.createQuerySet({ type: "timestamp", count: QUERIES });
      const i = labels.length; labels.push(`${descriptor?.label ?? "pass"} / (no dispatch)`);
      encoder.copyBufferToBuffer(breakSource, 0, breakTarget, 0, 4);
      return encoder.beginComputePass({ ...descriptor, timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
    };
    const beginComputePass = (descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder => {
      // Replaying these query indices on each split overwrites the caller's
      // whole-pass measurement with the final dispatch. Leave that pass intact.
      if (descriptor?.timestampWrites) { untimed++; return encoder.beginComputePass(descriptor); }
      let current = open(descriptor), dispatched = false, index = labels.length - 1;
      let pipeline: GPUComputePipeline | undefined;
      const groups = new Map<number, unknown[]>();
      const dispatch = (run: () => void) => {
        if (dispatched) {
          current.end(); current = open(descriptor); index = labels.length - 1;
          if (pipeline) current.setPipeline(pipeline);
          for (const [, args] of groups) (current.setBindGroup as (...a: unknown[]) => void)(...args);
        }
        if (index >= 0 && labels[index]!.endsWith("(no dispatch)")) labels[index] = `${descriptor?.label ?? "pass"} / ${(pipeline && names.get(pipeline)) ?? pipeline?.label ?? "pipeline"}`;
        run(); dispatched = true;
      };
      return new Proxy(current, { get(_target, key) {
        if (key === "setPipeline") return (p: GPUComputePipeline) => { pipeline = p; current.setPipeline(p); };
        if (key === "setBindGroup") return (...args: unknown[]) => { groups.set(args[0] as number, args); (current.setBindGroup as (...a: unknown[]) => void)(...args); };
        if (key === "dispatchWorkgroups") return (x: number, y?: number, z?: number) => dispatch(() => current.dispatchWorkgroups(x, y, z));
        if (key === "dispatchWorkgroupsIndirect") return (b: GPUBuffer, o: number) => dispatch(() => current.dispatchWorkgroupsIndirect(b, o));
        if (key === "end") return () => current.end();
        const value = Reflect.get(current, key);
        return typeof value === "function" ? value.bind(current) : value;
      } });
    };
    return new Proxy(encoder, { get(target, key) {
      if (key === "beginComputePass") return beginComputePass;
      if (key === "finish") return (descriptor?: GPUCommandBufferDescriptor) => {
        if (labels.length && querySet) {
          const bytes = 16 * labels.length;
          const resolveBuffer = device.createBuffer({ label: "kernel timer resolve", size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
          const read = device.createBuffer({ label: "kernel timer read", size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          target.resolveQuerySet(querySet, 0, 2 * labels.length, resolveBuffer, 0);
          target.copyBufferToBuffer(resolveBuffer, 0, read, 0, bytes);
          pending.push({ read, resolve: resolveBuffer, querySet, labels });
        }
        return target.finish(descriptor);
      };
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    } });
  };
  const timer: GPUKernelTimer = {
    enabled: false,
    device: new Proxy(device, { get(target, key) {
      if (key === "createComputePipeline") return (d: GPUComputePipelineDescriptor) => named(target.createComputePipeline(d), d);
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => named(await target.createComputePipelineAsync(d), d);
      if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
        const encoder = target.createCommandEncoder(descriptor);
        return timer.enabled ? wrapEncoder(encoder) : encoder;
      };
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    } }),
    async collect() {
      const byLabel = new Map<string, number>(); let dispatches = 0;
      const batch = pending.splice(0);
      try {
        await device.queue.onSubmittedWorkDone();
        for (const { read, labels } of batch) {
          await read.mapAsync(GPUMapMode.READ);
          const t = new BigUint64Array(read.getMappedRange());
          for (let i = 0; i < labels.length; i++) {
            const begin = t[2 * i]!, end = t[2 * i + 1]!;
            if (end < begin || begin === 0n) continue;
            byLabel.set(labels[i]!, (byLabel.get(labels[i]!) ?? 0) + Number(end - begin) / 1e6); dispatches++;
          }
          read.unmap();
        }
      } finally {
        for (const { read, resolve, querySet } of batch) { read.destroy(); resolve.destroy(); querySet.destroy(); }
      }
      const out = { byLabel, dispatches, untimed }; untimed = 0; return out;
    },
  };
  return timer;
}
