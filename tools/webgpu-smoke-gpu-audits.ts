import type {
  GPUDataFlowEncoderSession,
  GPUDataFlowPassRecorder,
} from "./webgpu-data-flow-manifest";

export type GPUCommandAuditBucket = { calls: number; bytes: number };
export interface GPUCommandAuditReport {
  writeBuffer: GPUCommandAuditBucket;
  writeTexture: GPUCommandAuditBucket;
  clearBuffer: GPUCommandAuditBucket;
  copyBufferToBuffer: GPUCommandAuditBucket;
  bufferAllocations: GPUCommandAuditBucket;
  bindGroups: number;
  commandEncoders: number;
  commandBuffers: number;
  computePasses: number;
  dispatches: number;
  indirectDispatches: number;
  submissions: number;
  submittedCommandBuffers: number;
  completionFences: number;
  writeBufferByLabel: Record<string, GPUCommandAuditBucket>;
  clearBufferByLabel: Record<string, GPUCommandAuditBucket>;
  copyBufferToBufferByLabel: Record<string, GPUCommandAuditBucket>;
  bufferAllocationsByLabel: Record<string, GPUCommandAuditBucket>;
  commandEncodersByLabel: Record<string, GPUCommandAuditBucket>;
  computePassesByLabel: Record<string, GPUCommandAuditBucket>;
  dispatchesByPassLabel: Record<string, GPUCommandAuditBucket>;
  indirectDispatchesByPassLabel: Record<string, GPUCommandAuditBucket>;
}

export interface GPUFineTimestampBucket {
  samples: number;
  total_ms: number;
  mean_ms: number;
  minimum_ms: number;
  maximum_ms: number;
}

export interface GPUFineTimestampReport {
  measuredAdvances: number;
  measuredPasses: number;
  invalidPasses: number;
  summedPass_ms: number;
  byLabel: Record<string, GPUFineTimestampBucket>;
}

export interface GPUPassTimestampReport {
  capturedCommandBuffers: number;
  measuredPasses: number;
  invalidPasses: number;
  capacityOverflows: number;
  summedPass_ms: number;
  /** False means Dawn was free to merge passes into shared Metal encoders, so
   * each label is really its encoder's total charged to the encoder's last
   * pass. Only an isolated report attributes time to a single pass, and only
   * an unisolated wall clock states what the frame costs. */
  encoderIsolated: boolean;
  /** Every labelled `compute()` got its own pass, so a label's dispatches are
   * exactly the dispatches recorded under that label. */
  labelIsolated: boolean;
  /** Optional prefixes retained before query slots are allocated. */
  labelPrefixes?: readonly string[];
  /** Wall span of the captured command buffers on the GPU timeline: last
   * timestamp minus first. */
  span_ms: number;
  /** `summedPass_ms / span_ms`. One means the passes tile the span, which is
   * the only state in which a label's ms is that label's cost. Above one means
   * brackets overlap, below one means unbracketed GPU time. */
  coverageRatio: number;
  byLabel: Record<string, GPUFineTimestampBucket>;
}

interface GPUPassTimestampCapture {
  buckets: Array<[string, number | undefined]>;
  capacityOverflows: number;
  span_ms: number;
}

export class GPUPassTimestampEncoderSession {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  private readonly labels: string[] = [];
  private queryCount = 0;
  private capacityOverflows = 0;

  constructor(private readonly device: GPUDevice, private readonly capacity: number,
    private readonly labelPrefixes: readonly string[] = []) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: capacity });
    this.resolveBuffer = device.createBuffer({
      label: "Algorithm pass timestamps resolve",
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: "Algorithm pass timestamps readback",
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  computePassDescriptor(descriptor?: GPUComputePassDescriptor): GPUComputePassDescriptor | undefined {
    const label = descriptor?.label?.trim() || "<unlabeled compute pass>";
    if (this.labelPrefixes.length > 0
      && !this.labelPrefixes.some((prefix) => label.startsWith(prefix))) return descriptor;
    if (descriptor?.timestampWrites) {
      // Never overwrite the solver's semantic recorder. Diagnostic runs turn
      // that recorder off, but retaining this guard makes the two facilities
      // composable and fail-closed.
      return descriptor;
    }
    if (this.queryCount + 2 > this.capacity) {
      this.capacityOverflows += 1;
      return descriptor;
    }
    const beginningOfPassWriteIndex = this.queryCount;
    const endOfPassWriteIndex = this.queryCount + 1;
    this.queryCount += 2;
    this.labels.push(label);
    return {
      ...descriptor,
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex,
        endOfPassWriteIndex,
      },
    };
  }

  resolve(encoder: GPUCommandEncoder): void {
    if (this.queryCount === 0) return;
    const bytes = this.queryCount * 8;
    encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, bytes);
  }

  async read(): Promise<GPUPassTimestampCapture> {
    try {
      if (this.queryCount === 0) {
        return { buckets: [], capacityOverflows: this.capacityOverflows, span_ms: 0 };
      }
      const bytes = this.queryCount * 8;
      await this.readBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
      const timestamps = new BigUint64Array(this.readBuffer.getMappedRange(0, bytes).slice(0));
      // The union of every pass bracket. Compared against the sum of the
      // brackets it states whether this capture is an attribution at all: a sum
      // materially above the span means passes overlapped on the GPU timeline,
      // and a per-pass number that overlaps its neighbours cannot be a cost.
      let earliest: bigint | undefined, latest: bigint | undefined;
      for (let index = 0; index < this.queryCount; index += 1) {
        const timestamp = timestamps[index] ?? 0n;
        if (timestamp === 0n) continue;
        if (earliest === undefined || timestamp < earliest) earliest = timestamp;
        if (latest === undefined || timestamp > latest) latest = timestamp;
      }
      return {
        buckets: this.labels.map((label, index) => {
          const begin = timestamps[2 * index] ?? 0n;
          const end = timestamps[2 * index + 1] ?? 0n;
          return [label, begin > 0n && end >= begin ? Number(end - begin) / 1e6 : undefined];
        }),
        capacityOverflows: this.capacityOverflows,
        span_ms: earliest !== undefined && latest !== undefined ? Number(latest - earliest) / 1e6 : 0,
      };
    } finally {
      if (this.readBuffer.mapState === "mapped") this.readBuffer.unmap();
      this.querySet.destroy();
      this.resolveBuffer.destroy();
      this.readBuffer.destroy();
    }
  }
}

/** Captures a bounded number of complete command buffers after `start()`. The
 * smoke starts it only after construction/t=0 publication, so the first buffer
 * is a real mini-dam recurring advance rather than shader warmup. */
export class GPUPassTimestampAudit {
  private enabled = false;
  private claimedCommandBuffers = 0;
  private skippedCommandBuffers = 0;
  private readonly submitted = new WeakMap<GPUCommandBuffer, GPUPassTimestampEncoderSession>();
  private readonly reads: Promise<GPUPassTimestampCapture>[] = [];

  constructor(
    private readonly device: GPUDevice,
    private readonly maximumCommandBuffers = 1,
    private readonly queryCapacity = 2048,
    private readonly encoderIsolated = false,
    private readonly labelIsolated = false,
    private readonly skipCommandBuffers = 0,
    private readonly labelPrefixes: readonly string[] = [],
  ) {}

  start(): void { this.enabled = true; }

  createEncoderSession(): GPUPassTimestampEncoderSession | undefined {
    if (!this.enabled || this.claimedCommandBuffers >= this.maximumCommandBuffers) return undefined;
    // The first recurring command buffer is the coldest one in the run: on the
    // mini lane its GPU span measured 92.7 ms against a 39.1 ms mean advance.
    // Skipping forward buys a representative buffer at no cost but a later
    // capture.
    if (this.skippedCommandBuffers < this.skipCommandBuffers) {
      this.skippedCommandBuffers += 1;
      return undefined;
    }
    this.claimedCommandBuffers += 1;
    return new GPUPassTimestampEncoderSession(this.device, this.queryCapacity, this.labelPrefixes);
  }

  attach(commandBuffer: GPUCommandBuffer, session: GPUPassTimestampEncoderSession): void {
    this.submitted.set(commandBuffer, session);
  }

  afterSubmit(commandBuffers: readonly GPUCommandBuffer[]): void {
    for (const commandBuffer of commandBuffers) {
      const session = this.submitted.get(commandBuffer);
      if (session) this.reads.push(session.read());
    }
  }

  async report(): Promise<GPUPassTimestampReport> {
    const captures = await Promise.all(this.reads);
    const aggregates = new Map<string, Omit<GPUFineTimestampBucket, "mean_ms">>();
    let invalidPasses = 0;
    for (const capture of captures) for (const [label, duration] of capture.buckets) {
      if (duration === undefined || !Number.isFinite(duration) || duration < 0) {
        invalidPasses += 1;
        continue;
      }
      const bucket = aggregates.get(label) ?? {
        samples: 0, total_ms: 0, minimum_ms: Number.POSITIVE_INFINITY, maximum_ms: 0,
      };
      bucket.samples += 1;
      bucket.total_ms += duration;
      bucket.minimum_ms = Math.min(bucket.minimum_ms, duration);
      bucket.maximum_ms = Math.max(bucket.maximum_ms, duration);
      aggregates.set(label, bucket);
    }
    const byLabel = Object.fromEntries(Array.from(aggregates.entries())
      .sort((left, right) => right[1].total_ms - left[1].total_ms || left[0].localeCompare(right[0]))
      .map(([label, bucket]) => [label, {
        ...bucket,
        mean_ms: bucket.total_ms / Math.max(1, bucket.samples),
      }]));
    const summedPass_ms = Array.from(aggregates.values()).reduce((sum, bucket) => sum + bucket.total_ms, 0);
    const span_ms = captures.reduce((sum, capture) => sum + capture.span_ms, 0);
    return {
      capturedCommandBuffers: captures.length,
      measuredPasses: Array.from(aggregates.values()).reduce((sum, bucket) => sum + bucket.samples, 0),
      invalidPasses,
      capacityOverflows: captures.reduce((sum, capture) => sum + capture.capacityOverflows, 0),
      summedPass_ms,
      encoderIsolated: this.encoderIsolated,
      labelIsolated: this.labelIsolated,
      ...(this.labelPrefixes.length > 0 ? { labelPrefixes: this.labelPrefixes } : {}),
      span_ms,
      coverageRatio: span_ms > 0 ? summedPass_ms / span_ms : 0,
      byLabel,
    };
  }
}

export class GPUCommandAudit {
  private writeBuffer = { calls: 0, bytes: 0 };
  private writeTexture = { calls: 0, bytes: 0 };
  private clearBuffer = { calls: 0, bytes: 0 };
  private copyBufferToBuffer = { calls: 0, bytes: 0 };
  private bufferAllocations = { calls: 0, bytes: 0 };
  private bindGroups = 0;
  private commandEncoders = 0;
  private commandBuffers = 0;
  private computePasses = 0;
  private dispatches = 0;
  private indirectDispatches = 0;
  private submissions = 0;
  private submittedCommandBuffers = 0;
  private completionFences = 0;
  private readonly writeBufferByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly clearBufferByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly copyBufferToBufferByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly bufferAllocationsByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly commandEncodersByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly computePassesByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly dispatchesByPassLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly indirectDispatchesByPassLabel = new Map<string, GPUCommandAuditBucket>();

  private label(value: { label?: string } | undefined, fallback: string): string {
    return value?.label?.trim() || fallback;
  }
  private add(map: Map<string, GPUCommandAuditBucket>, label: string, bytes = 0): void {
    const bucket = map.get(label) ?? { calls: 0, bytes: 0 };
    bucket.calls += 1; bucket.bytes += bytes; map.set(label, bucket);
  }
  private record(bucket: GPUCommandAuditBucket, bytes: number): void {
    bucket.calls += 1; bucket.bytes += bytes;
  }
  reset(): void {
    for (const bucket of [this.writeBuffer, this.writeTexture, this.clearBuffer,
      this.copyBufferToBuffer, this.bufferAllocations]) { bucket.calls = 0; bucket.bytes = 0; }
    this.bindGroups = 0; this.commandEncoders = 0; this.commandBuffers = 0;
    this.computePasses = 0; this.dispatches = 0; this.indirectDispatches = 0;
    this.submissions = 0; this.submittedCommandBuffers = 0; this.completionFences = 0;
    for (const map of [this.writeBufferByLabel, this.clearBufferByLabel, this.bufferAllocationsByLabel,
      this.copyBufferToBufferByLabel, this.commandEncodersByLabel, this.computePassesByLabel,
      this.dispatchesByPassLabel, this.indirectDispatchesByPassLabel]) map.clear();
  }
  recordWriteBuffer(buffer: GPUBuffer, bytes: number): void {
    this.record(this.writeBuffer, bytes); this.add(this.writeBufferByLabel, this.label(buffer, "<unlabeled buffer>"), bytes);
  }
  recordWriteTexture(bytes: number): void { this.record(this.writeTexture, bytes); }
  recordClearBuffer(buffer: GPUBuffer, bytes: number): void {
    this.record(this.clearBuffer, bytes); this.add(this.clearBufferByLabel, this.label(buffer, "<unlabeled buffer>"), bytes);
  }
  recordCopyBuffer(source: GPUBuffer, destination: GPUBuffer, bytes: number): void {
    this.record(this.copyBufferToBuffer, bytes);
    this.add(this.copyBufferToBufferByLabel,
      `${this.label(source, "<unlabeled buffer>")} -> ${this.label(destination, "<unlabeled buffer>")}`, bytes);
  }
  recordBufferAllocation(descriptor: GPUBufferDescriptor): void {
    const bytes = Number(descriptor.size); this.record(this.bufferAllocations, bytes);
    this.add(this.bufferAllocationsByLabel, descriptor.label?.trim() || "<unlabeled buffer>", bytes);
  }
  recordBindGroup(): void { this.bindGroups += 1; }
  recordCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): void {
    this.commandEncoders += 1;
    this.add(this.commandEncodersByLabel, descriptor?.label?.trim() || "<unlabeled encoder>");
  }
  recordCommandBuffer(): void { this.commandBuffers += 1; }
  recordComputePass(descriptor?: GPUComputePassDescriptor): void {
    this.computePasses += 1;
    this.add(this.computePassesByLabel, descriptor?.label?.trim() || "<unlabeled compute pass>");
  }
  recordDispatch(passLabel: string, indirect: boolean): void {
    this.dispatches += 1; this.add(this.dispatchesByPassLabel, passLabel);
    if (indirect) { this.indirectDispatches += 1; this.add(this.indirectDispatchesByPassLabel, passLabel); }
  }
  recordSubmit(commandBufferCount: number): void {
    this.submissions += 1; this.submittedCommandBuffers += commandBufferCount;
  }
  recordFence(): void { this.completionFences += 1; }
  private object(map: Map<string, GPUCommandAuditBucket>): Record<string, GPUCommandAuditBucket> {
    return Object.fromEntries(Array.from(map.entries()).sort((left, right) =>
      right[1].bytes - left[1].bytes || right[1].calls - left[1].calls || left[0].localeCompare(right[0])));
  }
  snapshot(): GPUCommandAuditReport {
    return {
      writeBuffer: { ...this.writeBuffer }, writeTexture: { ...this.writeTexture },
      clearBuffer: { ...this.clearBuffer }, copyBufferToBuffer: { ...this.copyBufferToBuffer },
      bufferAllocations: { ...this.bufferAllocations }, bindGroups: this.bindGroups,
      commandEncoders: this.commandEncoders, commandBuffers: this.commandBuffers,
      computePasses: this.computePasses, dispatches: this.dispatches,
      indirectDispatches: this.indirectDispatches, submissions: this.submissions,
      submittedCommandBuffers: this.submittedCommandBuffers, completionFences: this.completionFences,
      writeBufferByLabel: this.object(this.writeBufferByLabel),
      clearBufferByLabel: this.object(this.clearBufferByLabel),
      copyBufferToBufferByLabel: this.object(this.copyBufferToBufferByLabel),
      bufferAllocationsByLabel: this.object(this.bufferAllocationsByLabel),
      commandEncodersByLabel: this.object(this.commandEncodersByLabel),
      computePassesByLabel: this.object(this.computePassesByLabel),
      dispatchesByPassLabel: this.object(this.dispatchesByPassLabel),
      indirectDispatchesByPassLabel: this.object(this.indirectDispatchesByPassLabel),
    };
  }
}

export function writtenByteLength(data: GPUAllowSharedBufferSource, dataOffset = 0, size?: number): number {
  if (size !== undefined) return size;
  const byteLength = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
  return Math.max(0, byteLength - dataOffset);
}

function auditComputePass(pass: GPUComputePassEncoder, audit: GPUCommandAudit | undefined,
  passLabel: string, dataFlow?: GPUDataFlowPassRecorder): GPUComputePassEncoder {
  return new Proxy(pass, { get(target, property) {
    if (property === "setPipeline") return (pipeline: GPUComputePipeline) => {
      dataFlow?.setPipeline(pipeline);
      return target.setPipeline(pipeline);
    };
    if (property === "setBindGroup") return (
      index: number,
      bindGroup: GPUBindGroup | null,
      dynamicOffsets?: Iterable<number>,
      dynamicOffsetsDataStart?: number,
      dynamicOffsetsDataLength?: number,
    ) => {
      const offsets = dynamicOffsets === undefined ? undefined : Array.from(dynamicOffsets);
      const start = dynamicOffsetsDataStart ?? 0;
      const length = dynamicOffsetsDataLength ?? Math.max(0, (offsets?.length ?? 0) - start);
      dataFlow?.setBindGroup(index, bindGroup, offsets?.slice(start, start + length));
      if (dynamicOffsetsDataStart !== undefined && dynamicOffsets instanceof Uint32Array) {
        return target.setBindGroup(
          index, bindGroup, dynamicOffsets,
          dynamicOffsetsDataStart,
          dynamicOffsetsDataLength ?? Math.max(0, dynamicOffsets.length - dynamicOffsetsDataStart),
        );
      }
      if (dynamicOffsets !== undefined) return target.setBindGroup(index, bindGroup, dynamicOffsets);
      return target.setBindGroup(index, bindGroup);
    };
    if (property === "dispatchWorkgroups") return (...args: Parameters<GPUComputePassEncoder["dispatchWorkgroups"]>) => {
      audit?.recordDispatch(passLabel, false);
      dataFlow?.direct(args[0], args[1], args[2]);
      return Reflect.apply(target.dispatchWorkgroups, target, args);
    };
    if (property === "dispatchWorkgroupsIndirect") return (...args: Parameters<GPUComputePassEncoder["dispatchWorkgroupsIndirect"]>) => {
      audit?.recordDispatch(passLabel, true);
      dataFlow?.indirect(args[0], Number(args[1]));
      return Reflect.apply(target.dispatchWorkgroupsIndirect, target, args);
    };
    const value = Reflect.get(target, property, target);
    return typeof value === "function" ? value.bind(target) : value;
  } }) as GPUComputePassEncoder;
}

export function auditCommandEncoder(
  encoder: GPUCommandEncoder,
  audit?: GPUCommandAudit,
  dataFlow?: GPUDataFlowEncoderSession,
  passTimestamps?: GPUPassTimestampEncoderSession,
  attachPassTimestamps?: (
    commandBuffer: GPUCommandBuffer,
    session: GPUPassTimestampEncoderSession,
  ) => void,
): GPUCommandEncoder {
  return new Proxy(encoder, { get(target, property) {
    if (property === "clearBuffer") return (buffer: GPUBuffer, offset = 0, size?: number) => {
      const bytes = size ?? Math.max(0, buffer.size - offset); audit?.recordClearBuffer(buffer, bytes);
      return target.clearBuffer(buffer, offset, size);
    };
    if (property === "copyBufferToBuffer") return (source: GPUBuffer, sourceOffset: number, destination: GPUBuffer,
      destinationOffset: number, size: number) => {
      audit?.recordCopyBuffer(source, destination, size);
      return target.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
    };
    if (property === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
      audit?.recordComputePass(descriptor);
      const label = descriptor?.label?.trim() || "<unlabeled compute pass>";
      const pass = target.beginComputePass(passTimestamps?.computePassDescriptor(descriptor) ?? descriptor);
      const flowPass = dataFlow?.beginPass(label);
      return audit || flowPass ? auditComputePass(pass, audit, label, flowPass) : pass;
    };
    if (property === "finish") return (descriptor?: GPUCommandBufferDescriptor) => {
      audit?.recordCommandBuffer();
      passTimestamps?.resolve(target);
      const commandBuffer = target.finish(descriptor);
      if (passTimestamps) attachPassTimestamps?.(commandBuffer, passTimestamps);
      return commandBuffer;
    };
    const value = Reflect.get(target, property, target);
    return typeof value === "function" ? value.bind(target) : value;
  } }) as GPUCommandEncoder;
}

