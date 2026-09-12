import { boxFraction, compileMap, cellOrigin, type MapSpec, type V3 } from "./geometry";
import { remapOverlapWGSL } from "./overlap.wgsl";

export type RemapShape = "slab" | "oblique" | "full";
export interface RemapSettings { travel: number; deformation: number; shape: RemapShape }
export interface RemapFrame {
  step: number; volumes: Float32Array; initialVolume: number; volume: number;
  relativeDrift: number; capacityError: number; donorError: number; boundError: number;
  candidates: number; pieces: number; cpuMs: number; executionMs: number;
}
export const REMAP_SIZE = 8;
const total = (values: ArrayLike<number>) => Array.from(values).reduce((s, v) => s + v, 0);
export function initialRemapVolumes(shape: RemapShape): Float32Array {
  return Float32Array.from({ length: REMAP_SIZE ** 3 }, (_, i) => {
    const p = cellOrigin(i, REMAP_SIZE);
    if (shape === "full") return 1;
    if (shape === "slab") return Math.max(0, Math.min(p[0] + 1, 4.75) - Math.max(p[0], 2.25));
    return boxFraction([2, -1, 3], 2.15 * REMAP_SIZE, p, p.map(v => v + 1) as V3);
  });
}
export function remapStepSpec(settings: RemapSettings): MapSpec {
  if (!Number.isFinite(settings.travel) || Math.abs(settings.travel) > 25.5
    || !Number.isFinite(settings.deformation) || settings.deformation < 0 || settings.deformation > 1) {
    throw new Error("Remap scene parameters are outside the verified workload");
  }
  return { size: REMAP_SIZE, shift: [settings.travel, 0, 0], shears: settings.deformation === 0 ? [] : [
    { axis: 0, dependent: 1, amplitude: 0.65 * settings.deformation, knots: REMAP_SIZE },
    { axis: 1, dependent: 2, amplitude: 0.55 * settings.deformation, knots: REMAP_SIZE },
    { axis: 2, dependent: 0, amplitude: 0.75 * settings.deformation, knots: REMAP_SIZE },
  ] };
}

/** Small prescribed-flow session shared by the UI worker and Dawn integration
 * probe. Every step reconstructs the LAST accepted volumes, applies one map,
 * validates raw transfers, then publishes. This is not an animation that
 * repeatedly resamples the authored initial shape.
 */
export class GeometricRemapSession {
  private constructor(private device: GPUDevice, private layout: GPUBindGroupLayout,
    private intersect: GPUComputePipeline, private gather: GPUComputePipeline) {}
  private values = initialRemapVolumes("slab");
  private initialVolume = total(this.values);
  private stepIndex = 0;
  private generation = 0;
  private busy = false;

  static async create(device: GPUDevice): Promise<GeometricRemapSession> {
    const module = device.createShaderModule({ label: "Geometric remap lab", code: remapOverlapWGSL });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => m.message).join("\n"));
    const layout = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: (binding % 2 ? "storage" : "read-only-storage") as GPUBufferBindingType },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const intersect = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "intersect" } });
    const gather = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "gather" } });
    return new GeometricRemapSession(device, layout, intersect, gather);
  }
  reset(shape: RemapShape): RemapFrame {
    this.generation++; this.values = initialRemapVolumes(shape);
    this.initialVolume = total(this.values); this.stepIndex = 0;
    return { step: 0, volumes: this.values.slice(), initialVolume: this.initialVolume,
      volume: this.initialVolume, relativeDrift: 0, capacityError: 0, donorError: 0,
      boundError: 0, candidates: 0, pieces: 0, cpuMs: 0, executionMs: 0 };
  }
  async advance(settings: RemapSettings): Promise<RemapFrame> {
    if (this.busy) throw new Error("A remap step is already in flight");
    this.busy = true;
    const buffers: GPUBuffer[] = [], generation = this.generation;
    const device = this.device, cells = this.values.length;
    const allocate = (label: string, size: number, usage: GPUBufferUsageFlags) => {
      const b = device.createBuffer({ label, size, usage }); buffers.push(b); return b;
    };
    try {
      const start = performance.now(), before = this.values, initialVolume = this.initialVolume;
      const map = compileMap(remapStepSpec(settings), false, before);
      const count = map.overlaps.length, packed = new Float32Array(16 * count), ranges = new Uint32Array(2 * cells);
      map.overlaps.forEach((o, i) => {
        o.vertices.forEach((v, k) => packed.set([...v.p, v.liquid], 16 * i + 4 * k));
        if (i === 0 || map.overlaps[i - 1]!.receiver !== o.receiver) ranges[2 * o.receiver] = i;
        ranges[2 * o.receiver + 1] = i + 1;
      });
      const cpuMs = performance.now() - start;
      const input = allocate("Remap input", packed.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const raw = allocate("Remap transfers", count * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const segments = allocate("Remap segments", ranges.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const received = allocate("Remap candidate", cells * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = allocate("Remap validation", count * 8 + cells * 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(input, 0, packed); device.queue.writeBuffer(segments, 0, ranges);
      const group = device.createBindGroup({ layout: this.layout, entries: [input, raw, segments, received]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const executionStart = performance.now(), encoder = device.createCommandEncoder();
      for (const [i, pipeline] of [this.intersect, this.gather].entries()) {
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil((i ? cells : count) / (i ? 64 : 32))); pass.end();
      }
      encoder.copyBufferToBuffer(raw, 0, readback, 0, count * 8);
      encoder.copyBufferToBuffer(received, 0, readback, count * 8, cells * 8);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      if (generation !== this.generation) throw new Error("Remap cancelled by reset");
      const executionMs = performance.now() - executionStart, data = new Float32Array(readback.getMappedRange());
      const donorBulk = new Float64Array(cells), donorLiquid = new Float64Array(cells), candidate = new Float32Array(cells);
      let donorError = 0, capacityError = 0, boundError = 0;
      for (let i = 0; i < count; i++) {
        const bulk = data[2 * i]!, liquid = data[2 * i + 1]!;
        if (!Number.isFinite(bulk + liquid) || bulk < 0 || liquid < -1e-6 || liquid > bulk + 1e-6) {
          throw new Error(`Invalid raw overlap ${i}; last accepted state retained`);
        }
        donorBulk[map.overlaps[i]!.donor]! += bulk;
        donorLiquid[map.overlaps[i]!.donor]! += liquid;
      }
      for (let i = 0; i < cells; i++) {
        const bulk = data[2 * count + 2 * i]!, liquid = data[2 * count + 2 * i + 1]!;
        if (!Number.isFinite(bulk + liquid)) throw new Error("Nonfinite remap candidate");
        capacityError = Math.max(capacityError, Math.abs(bulk - 1));
        donorError = Math.max(donorError, Math.abs(donorBulk[i]! - 1), Math.abs(donorLiquid[i]! - before[i]!));
        boundError = Math.max(boundError, -liquid, liquid - 1); candidate[i] = liquid;
      }
      const volume = total(candidate), relativeDrift = (volume - initialVolume) / Math.max(1, initialVolume);
      const stepDrift = Math.abs(volume - total(before)) / Math.max(1, total(before));
      if (capacityError > 2e-5 || donorError > 2e-5 || boundError > 1e-6 || stepDrift > 2e-6 || Math.abs(relativeDrift) > 1e-5) {
        throw new Error(`Remap rejected: capacity ${capacityError.toExponential(2)}, donor ${donorError.toExponential(2)}, drift ${relativeDrift.toExponential(2)}. Last accepted state retained.`);
      }
      this.values = candidate; this.stepIndex++;
      return { step: this.stepIndex, volumes: candidate.slice(), initialVolume,
        volume, relativeDrift, capacityError, donorError, boundError, candidates: count,
        pieces: map.pieceCount, cpuMs, executionMs };
    } finally {
      for (const b of buffers) { if (b.mapState === "mapped") b.unmap(); b.destroy(); }
      this.busy = false;
    }
  }
}
