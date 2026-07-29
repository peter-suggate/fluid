import type { PassBroker } from "./webgpu-pass-broker";

export interface OctreePressureHistoryRemapSource {
  readonly rowDelta: GPUBuffer;
  readonly rowDeltaControlOffsetWords: number;
  readonly rowDeltaNewToOldOffsetWords: number;
  readonly currentCandidatePressure: GPUBuffer;
  readonly candidateHistory: GPUBuffer;
  readonly rowCapacity: number;
}

/** Publishes the row-remapped temporal secant p[n] - p[n-1]. */
export class WebGPUOctreePressureHistoryRemap {
  readonly encodedDispatchCount = 1;
  private readonly params: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly groups = new Map<GPUBuffer, GPUBindGroup>();
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly source: OctreePressureHistoryRemapSource,
  ) {
    if (!Number.isSafeInteger(source.rowCapacity) || source.rowCapacity < 1
      || !Number.isSafeInteger(source.rowDeltaControlOffsetWords)
      || source.rowDeltaControlOffsetWords < 0
      || !Number.isSafeInteger(source.rowDeltaNewToOldOffsetWords)
      || source.rowDeltaNewToOldOffsetWords < 0) {
      throw new RangeError("Pressure history remap layout is invalid");
    }
    const vectorBytes = source.rowCapacity * 4;
    if (source.currentCandidatePressure.size < vectorBytes
      || source.candidateHistory.size < vectorBytes
      || source.rowDelta.size < (source.rowDeltaNewToOldOffsetWords + source.rowCapacity) * 4) {
      throw new RangeError("Pressure history remap buffers are smaller than their row layout");
    }
    this.params = device.createBuffer({
      label: "Temporal pressure history remap parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      source.rowCapacity, source.rowDeltaNewToOldOffsetWords,
      source.rowDeltaControlOffsetWords, 0,
    ]));
    const module = device.createShaderModule({
      label: "Temporal pressure history row-identity remap",
      code: octreePressureHistoryRemapShader,
    });
    this.pipeline = device.createComputePipeline({
      label: "Temporal pressure history · remap previous solve",
      layout: "auto",
      compute: { module, entryPoint: "remapPressureHistory" },
    });
  }

  encode(broker: PassBroker, previousPressure: GPUBuffer): void {
    const pass = broker.compute({ label: "Remap temporal pressure history by row identity" });
    this.encodeIntoPass(pass, previousPressure);
    broker.fence("temporal pressure history remap complete");
  }

  encodeIntoPass(pass: GPUComputePassEncoder, previousPressure: GPUBuffer): void {
    if (this.destroyed) throw new Error("Pressure history remap is destroyed");
    if (previousPressure.size < this.source.rowCapacity * 4) {
      throw new RangeError("Previous pressure is smaller than the history remap capacity");
    }
    let group = this.groups.get(previousPressure);
    if (!group) {
      group = this.device.createBindGroup({
        label: "Temporal pressure history remap bindings",
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: this.source.rowDelta } },
          { binding: 2, resource: { buffer: previousPressure } },
          { binding: 3, resource: { buffer: this.source.currentCandidatePressure } },
          { binding: 4, resource: { buffer: this.source.candidateHistory } },
        ],
      });
      this.groups.set(previousPressure, group);
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.source.rowCapacity / 64));
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
  }
}

export const octreePressureHistoryRemapShader = /* wgsl */ `
struct Params { rowCapacity: u32, newToOldOffset: u32, controlOffset: u32, padding: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rowDelta: array<u32>;
@group(0) @binding(2) var<storage, read> previousPressure: array<f32>;
@group(0) @binding(3) var<storage, read> currentCandidatePressure: array<f32>;
@group(0) @binding(4) var<storage, read_write> candidateHistory: array<f32>;
fn finite(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
@compute @workgroup_size(64)
fn remapPressureHistory(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  let count = min(rowDelta[params.controlOffset], params.rowCapacity);
  if (row >= count) { return; }
  let encoded = rowDelta[params.newToOldOffset + row];
  // Bit 31 marks the row as affected; the +1 predecessor identity occupies
  // the low 31 bits and remains valid for both clean and dirty carried rows.
  let value = encoded & 0x7fffffffu;
  let oldRow = value - 1u;
  var history = 0.0;
  if (value != 0u && oldRow < arrayLength(&previousPressure)) {
    let carried = previousPressure[oldRow];
    let current = currentCandidatePressure[row];
    if (finite(carried) && finite(current)) { history = current - carried; }
  }
  candidateHistory[row] = history;
}
`;
