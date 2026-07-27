/**
 * Opt-in logical GPU activity telemetry.
 *
 * This module deliberately records shader-visible logical identities and
 * caller-supplied progress ticks. WGSL exposes neither a portable shader clock
 * nor a physical core identity, so this ABI must never be interpreted as either.
 */

export type GPULogicalActivityMode = "disabled" | "workgroup" | "subgroup";
export type GPULogicalActivityEvidence = "measured" | "reconstructed" | "unknown";

export const GPU_LOGICAL_ACTIVITY_MAGIC = 0x464c4143; // "FLAC"
export const GPU_LOGICAL_ACTIVITY_VERSION = 1;
export const GPU_LOGICAL_ACTIVITY_UNKNOWN_U32 = 0xffffffff;
export const GPU_LOGICAL_ACTIVITY_HEADER_WORDS = 8;
export const GPU_LOGICAL_ACTIVITY_RECORD_WORDS = 16;
export const GPU_LOGICAL_ACTIVITY_RECORD_BYTES = GPU_LOGICAL_ACTIVITY_RECORD_WORDS * 4;
export const GPU_LOGICAL_ACTIVITY_MAX_CAPACITY = 1_000_000;

export const GPU_LOGICAL_ACTIVITY_FLAGS = {
  workgroupIdPresent: 1 << 0,
  subgroupIdPresent: 1 << 1,
  subgroupIdReconstructed: 1 << 2,
  laneIdPresent: 1 << 3,
  laneCountMeasured: 1 << 4,
  laneCountReconstructed: 1 << 5,
  activeMaskMeasured: 1 << 6,
} as const;

const GPU_LOGICAL_ACTIVITY_KNOWN_FLAGS = Object.values(GPU_LOGICAL_ACTIVITY_FLAGS)
  .reduce((mask, flag) => mask | flag, 0);

export interface GPULogicalActivityShaderConfig {
  mode: GPULogicalActivityMode;
  /** Bind group reserved by the instrumented pipeline layout. */
  group?: number;
  /** Storage-buffer binding reserved by the instrumented pipeline layout. */
  binding?: number;
  /** Set when the surrounding module already contains `enable subgroups;`. */
  subgroupsAlreadyEnabled?: boolean;
}

export interface GPULogicalActivityWorkgroupCall {
  taskId: string;
  checkpointId: string;
  /** A caller-defined logical tick expression. Omit to retain sequence only. */
  tick?: string;
  workgroupId: string;
  localInvocationIndex: string;
  workgroupLaneCount: string;
}

export interface GPULogicalActivitySubgroupCall {
  taskId: string;
  checkpointId: string;
  /** A caller-defined logical tick expression. Omit to retain sequence only. */
  tick?: string;
  workgroupId: string;
  subgroupId: string;
  subgroupIdEvidence: "measured" | "reconstructed";
  subgroupLane: string;
  subgroupSize: string;
  /** Must be evaluated by every lane in uniform subgroup control flow. */
  active: string;
}

function checkedBinding(value: number | undefined, name: string): number {
  if (!Number.isInteger(value) || value === undefined || value < 0) {
    throw new RangeError(`GPU logical activity ${name} must be a non-negative integer`);
  }
  return value;
}

/**
 * Generate the complete binding and helper ABI for an instrumented WGSL module.
 * Disabled mode returns the empty string, adding no declarations or bindings.
 */
export function gpuLogicalActivityWGSL(config: GPULogicalActivityShaderConfig): string {
  if (config.mode === "disabled") return "";
  const group = checkedBinding(config.group, "group");
  const binding = checkedBinding(config.binding, "binding");
  const subgroupEnable = config.mode === "subgroup" && !config.subgroupsAlreadyEnabled
    ? "enable subgroups;\n\n"
    : "";
  const subgroupHelper = config.mode === "subgroup" ? /* wgsl */ `
fn fluidGpuLogicalActivitySubgroup(
  taskId: u32,
  checkpointId: u32,
  tick: u32,
  workgroupId: vec3u,
  subgroupId: u32,
  subgroupIdReconstructed: bool,
  subgroupLane: u32,
  subgroupSize: u32,
  activePredicate: bool,
) {
  let activeMask = subgroupBallot(activePredicate);
  if (subgroupLane != 0u) { return; }
  let activeLaneCount = countOneBits(activeMask.x) + countOneBits(activeMask.y)
    + countOneBits(activeMask.z) + countOneBits(activeMask.w);
  var flags = FLUID_GPU_ACTIVITY_WORKGROUP_ID_PRESENT
    | FLUID_GPU_ACTIVITY_SUBGROUP_ID_PRESENT
    | FLUID_GPU_ACTIVITY_LANE_ID_PRESENT
    | FLUID_GPU_ACTIVITY_LANE_COUNT_MEASURED
    | FLUID_GPU_ACTIVITY_ACTIVE_MASK_MEASURED;
  if (subgroupIdReconstructed) { flags |= FLUID_GPU_ACTIVITY_SUBGROUP_ID_RECONSTRUCTED; }
  fluidGpuLogicalActivityAppend(
    taskId, checkpointId, tick, workgroupId, subgroupId, subgroupLane,
    subgroupSize, activeLaneCount, activeMask, flags,
  );
}
` : "";
  return subgroupEnable + /* wgsl */ `
const FLUID_GPU_ACTIVITY_UNKNOWN: u32 = 0xffffffffu;
const FLUID_GPU_ACTIVITY_WORKGROUP_ID_PRESENT: u32 = 1u;
const FLUID_GPU_ACTIVITY_SUBGROUP_ID_PRESENT: u32 = 2u;
const FLUID_GPU_ACTIVITY_SUBGROUP_ID_RECONSTRUCTED: u32 = 4u;
const FLUID_GPU_ACTIVITY_LANE_ID_PRESENT: u32 = 8u;
const FLUID_GPU_ACTIVITY_LANE_COUNT_MEASURED: u32 = 16u;
const FLUID_GPU_ACTIVITY_LANE_COUNT_RECONSTRUCTED: u32 = 32u;
const FLUID_GPU_ACTIVITY_ACTIVE_MASK_MEASURED: u32 = 64u;

struct FluidGpuLogicalActivityRecord {
  sequence: u32,
  taskId: u32,
  checkpointId: u32,
  tick: u32,
  workgroupX: u32,
  workgroupY: u32,
  workgroupZ: u32,
  subgroupId: u32,
  laneId: u32,
  logicalLaneCount: u32,
  activeLaneCount: u32,
  activeMask0: u32,
  activeMask1: u32,
  activeMask2: u32,
  activeMask3: u32,
  flags: u32,
}

struct FluidGpuLogicalActivityBuffer {
  magic: u32,
  version: u32,
  capacity: u32,
  recordWords: u32,
  writeCount: atomic<u32>,
  overflow: atomic<u32>,
  captureId: u32,
  flags: u32,
  records: array<FluidGpuLogicalActivityRecord>,
}

@group(${group}) @binding(${binding})
var<storage, read_write> fluidGpuLogicalActivity: FluidGpuLogicalActivityBuffer;

fn fluidGpuLogicalActivityAppend(
  taskId: u32,
  checkpointId: u32,
  tick: u32,
  workgroupId: vec3u,
  subgroupId: u32,
  laneId: u32,
  logicalLaneCount: u32,
  activeLaneCount: u32,
  activeMask: vec4u,
  flags: u32,
) {
  let sequence = atomicAdd(&fluidGpuLogicalActivity.writeCount, 1u);
  let physicalCapacity = arrayLength(&fluidGpuLogicalActivity.records);
  if (sequence >= fluidGpuLogicalActivity.capacity || sequence >= physicalCapacity) {
    atomicStore(&fluidGpuLogicalActivity.overflow, 1u);
    return;
  }
  fluidGpuLogicalActivity.records[sequence] = FluidGpuLogicalActivityRecord(
    sequence, taskId, checkpointId, tick,
    workgroupId.x, workgroupId.y, workgroupId.z,
    subgroupId, laneId, logicalLaneCount, activeLaneCount,
    activeMask.x, activeMask.y, activeMask.z, activeMask.w, flags,
  );
}

fn fluidGpuLogicalActivityWorkgroup(
  taskId: u32,
  checkpointId: u32,
  tick: u32,
  workgroupId: vec3u,
  localInvocationIndex: u32,
  workgroupLaneCount: u32,
) {
  if (localInvocationIndex != 0u) { return; }
  fluidGpuLogicalActivityAppend(
    taskId, checkpointId, tick, workgroupId,
    FLUID_GPU_ACTIVITY_UNKNOWN, 0u, workgroupLaneCount,
    FLUID_GPU_ACTIVITY_UNKNOWN, vec4u(0u),
    FLUID_GPU_ACTIVITY_WORKGROUP_ID_PRESENT
      | FLUID_GPU_ACTIVITY_LANE_ID_PRESENT
      | FLUID_GPU_ACTIVITY_LANE_COUNT_RECONSTRUCTED,
  );
}
${subgroupHelper}`.trimStart();
}

/** Disabled mode emits no call expression, so templates remain pollution-free. */
export function gpuLogicalActivityWorkgroupCallWGSL(
  config: GPULogicalActivityShaderConfig,
  call: GPULogicalActivityWorkgroupCall,
): string {
  if (config.mode === "disabled") return "";
  const tick = call.tick ?? "FLUID_GPU_ACTIVITY_UNKNOWN";
  return `fluidGpuLogicalActivityWorkgroup(${call.taskId}, ${call.checkpointId}, ${tick}, ${call.workgroupId}, ${call.localInvocationIndex}, ${call.workgroupLaneCount});`;
}

/** Disabled and workgroup-only variants emit no subgroup call expression. */
export function gpuLogicalActivitySubgroupCallWGSL(
  config: GPULogicalActivityShaderConfig,
  call: GPULogicalActivitySubgroupCall,
): string {
  if (config.mode !== "subgroup") return "";
  const tick = call.tick ?? "FLUID_GPU_ACTIVITY_UNKNOWN";
  const reconstructed = call.subgroupIdEvidence === "reconstructed" ? "true" : "false";
  return `fluidGpuLogicalActivitySubgroup(${call.taskId}, ${call.checkpointId}, ${tick}, ${call.workgroupId}, ${call.subgroupId}, ${reconstructed}, ${call.subgroupLane}, ${call.subgroupSize}, ${call.active});`;
}

/** A stable cache-key suffix forces separately compiled pipeline variants. */
export function gpuLogicalActivityShaderVariantKey(
  baseKey: string,
  config: GPULogicalActivityShaderConfig,
): string {
  if (config.mode === "disabled") return `${baseKey}|gpu-logical-activity:v${GPU_LOGICAL_ACTIVITY_VERSION}:disabled`;
  const group = checkedBinding(config.group, "group");
  const binding = checkedBinding(config.binding, "binding");
  return `${baseKey}|gpu-logical-activity:v${GPU_LOGICAL_ACTIVITY_VERSION}:${config.mode}:g${group}:b${binding}`;
}

/**
 * Prefix an already assembled module with the ABI and return its variant key.
 * In disabled mode `code` is byte-for-byte the input source.
 */
export function buildGPULogicalActivityShaderVariant(input: {
  source: string;
  baseKey: string;
  config: GPULogicalActivityShaderConfig;
}): { code: string; cacheKey: string; instrumentationWGSL: string } {
  const instrumentationWGSL = gpuLogicalActivityWGSL(input.config);
  if (!instrumentationWGSL) {
    return {
      code: input.source,
      cacheKey: gpuLogicalActivityShaderVariantKey(input.baseKey, input.config),
      instrumentationWGSL,
    };
  }
  // WGSL enable/requires directives must precede declarations. Keep the
  // activity extension directive with the module's existing directives, then
  // insert the ABI declarations immediately after them. Disabled mode above
  // remains byte-for-byte identical.
  const subgroupDirective = instrumentationWGSL.startsWith("enable subgroups;\n\n")
    ? "enable subgroups;\n"
    : "";
  const declarations = subgroupDirective
    ? instrumentationWGSL.slice("enable subgroups;\n\n".length)
    : instrumentationWGSL;
  const directiveEnd = wgslDirectivePrefixEnd(input.source);
  return {
    code: `${subgroupDirective}${input.source.slice(0, directiveEnd)}${declarations}\n${input.source.slice(directiveEnd)}`,
    cacheKey: gpuLogicalActivityShaderVariantKey(input.baseKey, input.config),
    instrumentationWGSL,
  };
}

function wgslDirectivePrefixEnd(source: string): number {
  let cursor = 0;
  const skipTrivia = () => {
    while (cursor < source.length) {
      if (/\s/.test(source[cursor]!)) { cursor += 1; continue; }
      if (source.startsWith("//", cursor)) {
        const newline = source.indexOf("\n", cursor + 2);
        cursor = newline < 0 ? source.length : newline + 1;
        continue;
      }
      if (source.startsWith("/*", cursor)) {
        const close = source.indexOf("*/", cursor + 2);
        cursor = close < 0 ? source.length : close + 2;
        continue;
      }
      break;
    }
  };
  skipTrivia();
  while (source.startsWith("enable", cursor) || source.startsWith("requires", cursor)) {
    const semicolon = source.indexOf(";", cursor);
    if (semicolon < 0) break;
    cursor = semicolon + 1;
    skipTrivia();
  }
  return cursor;
}

function checkedCapacity(capacity: number): number {
  if (!Number.isInteger(capacity) || capacity <= 0 || capacity > GPU_LOGICAL_ACTIVITY_MAX_CAPACITY) {
    throw new RangeError(`GPU logical activity capacity must be in [1, ${GPU_LOGICAL_ACTIVITY_MAX_CAPACITY}]`);
  }
  return capacity;
}

export function gpuLogicalActivityBufferByteSize(capacity: number): number {
  return (GPU_LOGICAL_ACTIVITY_HEADER_WORDS + checkedCapacity(capacity) * GPU_LOGICAL_ACTIVITY_RECORD_WORDS) * 4;
}

/** Header upload for a newly cleared storage/readback buffer. */
export function createGPULogicalActivityBufferImage(capacity: number, captureId: number): Uint32Array {
  const checked = checkedCapacity(capacity);
  if (!Number.isInteger(captureId) || captureId < 0 || captureId > 0xffffffff) {
    throw new RangeError("GPU logical activity capture ID must be a u32");
  }
  const image = new Uint32Array(GPU_LOGICAL_ACTIVITY_HEADER_WORDS + checked * GPU_LOGICAL_ACTIVITY_RECORD_WORDS);
  image.set([
    GPU_LOGICAL_ACTIVITY_MAGIC,
    GPU_LOGICAL_ACTIVITY_VERSION,
    checked,
    GPU_LOGICAL_ACTIVITY_RECORD_WORDS,
    0,
    0,
    captureId,
    0,
  ]);
  return image;
}

export type GPULogicalActivityRecorderState =
  | "recording"
  | "staged"
  | "reading"
  | "complete"
  | "destroyed";

export interface GPULogicalActivityRecorderOptions {
  mode: GPULogicalActivityMode;
  capacity: number;
  captureId: number;
  label?: string;
}

export type GPULogicalActivityRecorderReadResult = GPULogicalActivityDecodeResult | {
  ok: false;
  code: "recorder-state" | "readback-failed";
  message: string;
};

/**
 * One-shot owner for an enabled logical-activity capture.
 *
 * `create` returns undefined in disabled mode, so production shader variants
 * also allocate no telemetry resources. The owner never submits or waits on the
 * queue: the caller stages the copy into its existing command encoder, submits
 * normally, and then awaits `read`.
 */
export class GPULogicalActivityRecorder {
  private readonly storageBuffer: GPUBuffer;
  private readBuffer?: GPUBuffer;
  private currentState: GPULogicalActivityRecorderState = "recording";

  static create(
    device: GPUDevice,
    options: GPULogicalActivityRecorderOptions,
  ): GPULogicalActivityRecorder | undefined {
    if (options.mode === "disabled") return undefined;
    return new GPULogicalActivityRecorder(device, options);
  }

  private constructor(
    private readonly device: GPUDevice,
    private readonly options: GPULogicalActivityRecorderOptions,
  ) {
    const initialImage = createGPULogicalActivityBufferImage(options.capacity, options.captureId);
    this.byteSize = initialImage.byteLength;
    this.storageBuffer = device.createBuffer({
      label: `${options.label ?? "GPU logical activity"} storage`,
      size: this.byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Uploading the complete image makes initialization independent of the
    // implementation's buffer-zeroing strategy and clears every record slot.
    device.queue.writeBuffer(
      this.storageBuffer,
      0,
      initialImage.buffer as ArrayBuffer,
      initialImage.byteOffset,
      initialImage.byteLength,
    );
  }

  readonly byteSize: number;

  get state(): GPULogicalActivityRecorderState { return this.currentState; }
  get capacity(): number { return this.options.capacity; }
  get captureId(): number { return this.options.captureId; }

  private isDestroyed(): boolean { return this.currentState === "destroyed"; }

  /** Create the extra bind group expected by the instrumented pipeline layout. */
  createBindGroup(layout: GPUBindGroupLayout, binding: number): GPUBindGroup {
    if (this.currentState !== "recording") {
      throw new Error(`GPU logical activity bind group requested while ${this.currentState}`);
    }
    const checked = checkedBinding(binding, "binding");
    return this.device.createBindGroup({
      label: `${this.options.label ?? "GPU logical activity"} bind group`,
      layout,
      entries: [{ binding: checked, resource: { buffer: this.storageBuffer } }],
    });
  }

  /** Convenience form for the group/binding encoded in a shader variant. */
  createBindGroupForPipeline(
    pipeline: GPUComputePipeline | GPURenderPipeline,
    config: GPULogicalActivityShaderConfig,
  ): GPUBindGroup {
    if (config.mode === "disabled") throw new Error("Disabled shader variants have no logical activity bind group");
    const group = checkedBinding(config.group, "group");
    const binding = checkedBinding(config.binding, "binding");
    return this.createBindGroup(pipeline.getBindGroupLayout(group), binding);
  }

  /** Append the one-shot readback copy to the frame's existing encoder. */
  finish(encoder: GPUCommandEncoder): void {
    if (this.currentState !== "recording") {
      throw new Error(`GPU logical activity capture cannot finish while ${this.currentState}`);
    }
    const readBuffer = this.device.createBuffer({
      label: `${this.options.label ?? "GPU logical activity"} readback`,
      size: this.byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(this.storageBuffer, 0, readBuffer, 0, this.byteSize);
    this.readBuffer = readBuffer;
    this.currentState = "staged";
  }

  /** Map and fail-closed decode after the command buffer containing `finish` is submitted. */
  async read(): Promise<GPULogicalActivityRecorderReadResult> {
    if (this.currentState !== "staged" || !this.readBuffer) {
      return {
        ok: false,
        code: "recorder-state",
        message: `GPU logical activity read requested while ${this.currentState}`,
      };
    }
    const readBuffer = this.readBuffer;
    this.currentState = "reading";
    try {
      await readBuffer.mapAsync(GPUMapMode.READ, 0, this.byteSize);
      if (this.isDestroyed()) {
        return { ok: false, code: "recorder-state", message: "GPU logical activity recorder was destroyed during readback" };
      }
      return decodeGPULogicalActivity(new Uint32Array(readBuffer.getMappedRange(0, this.byteSize)));
    } catch (error) {
      return {
        ok: false,
        code: "readback-failed",
        message: `GPU logical activity readback failed: ${error instanceof Error ? error.message : String(error)}`,
      };
    } finally {
      if (readBuffer.mapState === "mapped") readBuffer.unmap();
      readBuffer.destroy();
      if (this.readBuffer === readBuffer) this.readBuffer = undefined;
      if (!this.isDestroyed()) this.currentState = "complete";
    }
  }

  destroy(): void {
    if (this.currentState === "destroyed") return;
    this.currentState = "destroyed";
    this.readBuffer?.destroy();
    this.readBuffer = undefined;
    this.storageBuffer.destroy();
  }
}

export interface GPULogicalActivityEvent {
  sequence: number;
  taskId: number;
  checkpointId: number;
  /** Undefined means this capture carries sequence order only. */
  tick?: number;
  workgroupId: readonly [number, number, number];
  workgroupEvidence: "measured";
  subgroupId?: number;
  subgroupEvidence: GPULogicalActivityEvidence;
  laneId?: number;
  logicalLaneCount?: number;
  logicalLaneCountEvidence: GPULogicalActivityEvidence;
  activeLaneCount?: number;
  activeLaneMask?: readonly [number, number, number, number];
  activeLaneEvidence: GPULogicalActivityEvidence;
}

export interface GPULogicalActivityCapture {
  captureId: number;
  capacity: number;
  /** True when the recorder filled and only its complete prefix was retained. */
  overflowed: boolean;
  /** Number of append attempts beyond the retained prefix. */
  droppedEventCount: number;
  events: readonly GPULogicalActivityEvent[];
}

export type GPULogicalActivityDecodeErrorCode =
  | "short-buffer"
  | "bad-header"
  | "bad-size"
  | "overflow"
  | "malformed-record";

export type GPULogicalActivityDecodeResult =
  | { ok: true; capture: GPULogicalActivityCapture }
  | { ok: false; code: GPULogicalActivityDecodeErrorCode; message: string };

function popcount32(value: number): number {
  value >>>= 0;
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function maskHasBitsBeyond(mask: readonly number[], laneCount: number): boolean {
  if (laneCount >= 128) return false;
  const fullWords = Math.floor(laneCount / 32);
  const partialBits = laneCount % 32;
  for (let word = fullWords + (partialBits > 0 ? 1 : 0); word < 4; word += 1) {
    if ((mask[word] ?? 0) !== 0) return true;
  }
  if (partialBits === 0) return false;
  const allowed = 0xffffffff >>> (32 - partialBits);
  return (((mask[fullWords] ?? 0) & ~allowed) >>> 0) !== 0;
}

/**
 * Decode the complete records retained by the recorder. An overflowing append
 * counter does not invalidate the already-written prefix: sequence numbers
 * reserve unique slots, and readback occurs only after queue completion. The
 * returned capture explicitly reports truncation so later time remains unknown.
 */
export function decodeGPULogicalActivity(
  source: ArrayBuffer | ArrayBufferView | ArrayLike<number>,
): GPULogicalActivityDecodeResult {
  let words: Uint32Array;
  if (source instanceof ArrayBuffer) {
    if (source.byteLength % 4 !== 0) return { ok: false, code: "bad-size", message: "Capture byte length is not u32-aligned" };
    words = new Uint32Array(source);
  }
  else if (ArrayBuffer.isView(source)) {
    if (source.byteOffset % 4 !== 0 || source.byteLength % 4 !== 0) {
      return { ok: false, code: "bad-size", message: "Capture view is not u32-aligned" };
    }
    words = new Uint32Array(source.buffer, source.byteOffset, source.byteLength / 4);
  } else words = Uint32Array.from(source);
  if (words.length < GPU_LOGICAL_ACTIVITY_HEADER_WORDS) {
    return { ok: false, code: "short-buffer", message: "Capture is shorter than its ABI header" };
  }
  const [magic, version, capacity, recordWords, writeCount, overflow, captureId, headerFlags] = words;
  if (magic !== GPU_LOGICAL_ACTIVITY_MAGIC || version !== GPU_LOGICAL_ACTIVITY_VERSION
    || recordWords !== GPU_LOGICAL_ACTIVITY_RECORD_WORDS || headerFlags !== 0) {
    return { ok: false, code: "bad-header", message: "Capture ABI header is invalid or unsupported" };
  }
  if (capacity === 0 || capacity > GPU_LOGICAL_ACTIVITY_MAX_CAPACITY) {
    return { ok: false, code: "bad-header", message: "Capture capacity is invalid" };
  }
  const expectedWords = GPU_LOGICAL_ACTIVITY_HEADER_WORDS + capacity * GPU_LOGICAL_ACTIVITY_RECORD_WORDS;
  if (words.length !== expectedWords) {
    return { ok: false, code: "bad-size", message: "Capture size does not match its declared capacity" };
  }
  if (overflow > 1 || (overflow === 0) !== (writeCount <= capacity)) {
    return { ok: false, code: "bad-header", message: "Capture overflow state is inconsistent" };
  }
  const retainedCount = Math.min(writeCount, capacity);
  const events: GPULogicalActivityEvent[] = [];
  for (let index = 0; index < retainedCount; index += 1) {
    const offset = GPU_LOGICAL_ACTIVITY_HEADER_WORDS + index * GPU_LOGICAL_ACTIVITY_RECORD_WORDS;
    const record = words.subarray(offset, offset + GPU_LOGICAL_ACTIVITY_RECORD_WORDS);
    const [sequence, taskId, checkpointId, tick, wx, wy, wz, subgroupId, laneId,
      logicalLaneCount, activeLaneCount, mask0, mask1, mask2, mask3, flags] = record;
    const malformed = (message: string): GPULogicalActivityDecodeResult => ({
      ok: false, code: "malformed-record", message: `Record ${index}: ${message}`,
    });
    if (sequence !== index) return malformed("append sequence is incomplete or out of order");
    if ((flags & ~GPU_LOGICAL_ACTIVITY_KNOWN_FLAGS) !== 0
      || (flags & GPU_LOGICAL_ACTIVITY_FLAGS.workgroupIdPresent) === 0) {
      return malformed("identity flags are invalid");
    }
    const subgroupPresent = (flags & GPU_LOGICAL_ACTIVITY_FLAGS.subgroupIdPresent) !== 0;
    const subgroupReconstructed = (flags & GPU_LOGICAL_ACTIVITY_FLAGS.subgroupIdReconstructed) !== 0;
    if (subgroupPresent === (subgroupId === GPU_LOGICAL_ACTIVITY_UNKNOWN_U32)
      || (!subgroupPresent && subgroupReconstructed)) {
      return malformed("subgroup identity is inconsistent");
    }
    const lanePresent = (flags & GPU_LOGICAL_ACTIVITY_FLAGS.laneIdPresent) !== 0;
    const laneMeasured = (flags & GPU_LOGICAL_ACTIVITY_FLAGS.laneCountMeasured) !== 0;
    const laneReconstructed = (flags & GPU_LOGICAL_ACTIVITY_FLAGS.laneCountReconstructed) !== 0;
    if (laneMeasured && laneReconstructed) return malformed("lane count has conflicting evidence");
    const laneCountPresent = laneMeasured || laneReconstructed;
    if (!lanePresent || !laneCountPresent || logicalLaneCount === 0
      || logicalLaneCount === GPU_LOGICAL_ACTIVITY_UNKNOWN_U32
      || laneId === GPU_LOGICAL_ACTIVITY_UNKNOWN_U32 || laneId >= logicalLaneCount) {
      return malformed("logical lane identity is invalid");
    }
    const activeMeasured = (flags & GPU_LOGICAL_ACTIVITY_FLAGS.activeMaskMeasured) !== 0;
    const mask = [mask0, mask1, mask2, mask3] as const;
    if (activeMeasured) {
      if (logicalLaneCount > 128 || activeLaneCount > logicalLaneCount
        || activeLaneCount === GPU_LOGICAL_ACTIVITY_UNKNOWN_U32
        || maskHasBitsBeyond(mask, logicalLaneCount)
        || mask.reduce((sum, word) => sum + popcount32(word), 0) !== activeLaneCount) {
        return malformed("active-lane ballot is invalid");
      }
    } else if (activeLaneCount !== GPU_LOGICAL_ACTIVITY_UNKNOWN_U32 || mask.some((word) => word !== 0)) {
      return malformed("unknown active lanes carry unsupported values");
    }
    events.push({
      sequence, taskId, checkpointId,
      tick: tick === GPU_LOGICAL_ACTIVITY_UNKNOWN_U32 ? undefined : tick,
      workgroupId: [wx, wy, wz],
      workgroupEvidence: "measured",
      subgroupId: subgroupPresent ? subgroupId : undefined,
      subgroupEvidence: subgroupPresent ? (subgroupReconstructed ? "reconstructed" : "measured") : "unknown",
      laneId,
      logicalLaneCount,
      logicalLaneCountEvidence: laneMeasured ? "measured" : "reconstructed",
      activeLaneCount: activeMeasured ? activeLaneCount : undefined,
      activeLaneMask: activeMeasured ? mask : undefined,
      activeLaneEvidence: activeMeasured ? "measured" : "unknown",
    });
  }
  return {
    ok: true,
    capture: {
      captureId,
      capacity,
      overflowed: overflow !== 0,
      droppedEventCount: Math.max(0, writeCount - retainedCount),
      events,
    },
  };
}

export interface GPULogicalActivityTimeLocation {
  time_ms: number;
  /** Evidence belongs to the time projection, not to a shader clock. */
  evidence: Exclude<GPULogicalActivityEvidence, "unknown">;
}

export type GPULogicalActivityTimeLocator = (
  event: GPULogicalActivityEvent,
) => GPULogicalActivityTimeLocation | undefined;

export interface GPULogicalActivityBin {
  start_ms: number;
  end_ms: number;
  eventCount: number;
  taskCount: number;
  checkpointCount: number;
  workgroupCount: number;
  subgroupObservationCount: number;
  activeLanes: {
    sampleCount: number;
    sum: number;
    maximum: number;
    evidence: GPULogicalActivityEvidence;
  };
  timeEvidence: Exclude<GPULogicalActivityEvidence, "unknown">;
}

export interface GPULogicalActivityBinning {
  binWidth_ms: 1;
  bins: readonly GPULogicalActivityBin[];
  unknownTimeEventCount: number;
  timeEvidence: GPULogicalActivityEvidence;
}

function weakerEvidence(
  left: Exclude<GPULogicalActivityEvidence, "unknown">,
  right: Exclude<GPULogicalActivityEvidence, "unknown">,
) {
  return left === "reconstructed" || right === "reconstructed" ? "reconstructed" : "measured";
}

/**
 * Place logical heartbeat events into fixed 1 ms bins using an external time
 * projection. Without one, time remains explicitly unknown and no bins are
 * invented from append order.
 */
export function binGPULogicalActivity1ms(
  capture: GPULogicalActivityCapture,
  locateTime?: GPULogicalActivityTimeLocator,
): GPULogicalActivityBinning {
  type MutableBin = GPULogicalActivityBin & { tasks: Set<number>; checkpoints: Set<string>; workgroups: Set<string> };
  const bins = new Map<number, MutableBin>();
  let unknownTimeEventCount = 0;
  let overallEvidence: GPULogicalActivityEvidence = "unknown";
  for (const event of capture.events) {
    const location = locateTime?.(event);
    if (!location || !Number.isFinite(location.time_ms)) {
      unknownTimeEventCount += 1;
      continue;
    }
    overallEvidence = overallEvidence === "unknown"
      ? location.evidence
      : weakerEvidence(overallEvidence, location.evidence);
    const start_ms = Math.floor(location.time_ms);
    let bin = bins.get(start_ms);
    if (!bin) {
      bin = {
        start_ms, end_ms: start_ms + 1,
        eventCount: 0, taskCount: 0, checkpointCount: 0, workgroupCount: 0,
        subgroupObservationCount: 0,
        activeLanes: { sampleCount: 0, sum: 0, maximum: 0, evidence: "unknown" },
        timeEvidence: location.evidence,
        tasks: new Set(), checkpoints: new Set(), workgroups: new Set(),
      };
      bins.set(start_ms, bin);
    } else bin.timeEvidence = weakerEvidence(bin.timeEvidence, location.evidence);
    bin.eventCount += 1;
    bin.tasks.add(event.taskId);
    bin.checkpoints.add(`${event.taskId}:${event.checkpointId}`);
    bin.workgroups.add(event.workgroupId.join(","));
    if (event.subgroupId !== undefined) bin.subgroupObservationCount += 1;
    if (event.activeLaneCount !== undefined) {
      bin.activeLanes.sampleCount += 1;
      bin.activeLanes.sum += event.activeLaneCount;
      bin.activeLanes.maximum = Math.max(bin.activeLanes.maximum, event.activeLaneCount);
      bin.activeLanes.evidence = event.activeLaneEvidence;
    }
  }
  const output = [...bins.values()].sort((left, right) => left.start_ms - right.start_ms).map((bin) => {
    bin.taskCount = bin.tasks.size;
    bin.checkpointCount = bin.checkpoints.size;
    bin.workgroupCount = bin.workgroups.size;
    return {
      start_ms: bin.start_ms,
      end_ms: bin.end_ms,
      eventCount: bin.eventCount,
      taskCount: bin.taskCount,
      checkpointCount: bin.checkpointCount,
      workgroupCount: bin.workgroupCount,
      subgroupObservationCount: bin.subgroupObservationCount,
      activeLanes: bin.activeLanes,
      timeEvidence: bin.timeEvidence,
    };
  });
  return {
    binWidth_ms: 1,
    bins: output,
    unknownTimeEventCount,
    timeEvidence: capture.events.length === 0 || unknownTimeEventCount === capture.events.length
      ? "unknown"
      : unknownTimeEventCount > 0 ? "unknown" : overallEvidence,
  };
}

/** Convenience projection for caller-defined logical ticks; always reconstructed. */
export function reconstructGPULogicalActivityTicks(input: {
  originTick: number;
  origin_ms: number;
  millisecondsPerTick: number;
}): GPULogicalActivityTimeLocator {
  if (!Number.isFinite(input.originTick) || !Number.isFinite(input.origin_ms)
    || !Number.isFinite(input.millisecondsPerTick) || input.millisecondsPerTick <= 0) {
    throw new RangeError("Logical tick reconstruction requires finite origin and positive scale");
  }
  return (event) => event.tick === undefined ? undefined : ({
    time_ms: input.origin_ms + (event.tick - input.originTick) * input.millisecondsPerTick,
    evidence: "reconstructed",
  });
}
