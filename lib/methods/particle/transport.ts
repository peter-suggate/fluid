import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { MethodParamValues } from "../../core/method-contract";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";
import type { MacTransport, MacTransportContext } from "../mac-shared/transport";
import { APIC_BINDINGS, APIC_SHADER, type ApicEntry } from "./shader";
import { ParticleBinScan } from "./scan";
import { seedApicParticles } from "./seed";
import { particleTransferOptions } from "./parameters";

export type ApicInfo = GPUEulerianInfo & {
  apicTransferMode?: number;
  apicFlipRatio?: number;
  apicParticleCount?: number;
  apicInitialVolume_m3?: number;
  apicMaterialVolume_m3?: number;
  apicEscapedVolume_m3?: number;
  apicMaterialDrift?: number;
  apicKineticEnergy_J?: number;
  macFramePressureIterations?: number;
  macFrameEncode_ms?: number;
  macFrameSlots?: number;
  apicAffineSpeedBound_m_s?: number;
};

export class ApicTransport implements MacTransport {
  allocatedBytes = 0;
  readonly diagnostics: Record<string, number>;
  readonly debug: Record<string, unknown>;
  readonly particleSource: GPUFluidParticleSource;
  private readonly owned: GPUBuffer[] = [];
  private readonly resources: Map<number, GPUBindingResource>;
  private readonly count: number;
  private readonly volume: number;
  private readonly cells: number;
  private readonly vertices: number;
  private readonly settings: GPUBuffer;
  private transferMode = 0;
  private flipRatio = 0.95;
  private scan!: ParticleBinScan;
  private commands!: Record<ApicEntry, { pipeline: GPUComputePipeline; group: GPUBindGroup }>;

  private constructor(private readonly context: MacTransportContext) {
    const { device, dimensions, h } = context;
    this.cells = dimensions[0] * dimensions[1] * dimensions[2];
    this.vertices = (dimensions[0] + 1) * (dimensions[1] + 1) * (dimensions[2] + 1);
    const capacity = Math.min(1_000_000, Math.floor(Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize) / 80));
    const seed = seedApicParticles(context.initialPhi, context.solids, dimensions, h, capacity);
    this.count = seed.count; this.volume = seed.volume_m3;
    const initial = context.scene.fluid.initialVelocity_m_s ?? { x: 0, y: 0, z: 0 };
    for (let i = 0; i < this.count; i++) seed.data.set([initial.x, initial.y, initial.z], i * 20 + 4);
    const particles = this.buffer("particles (80 bytes each)", seed.data.byteLength);
    device.queue.writeBuffer(particles, 0, seed.data);
    const settings = this.settings = this.buffer("settings", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const words = new ArrayBuffer(16); new Uint32Array(words)[0] = this.count;
    new Float32Array(words).set([0.55, 1.5, this.volume], 1); device.queue.writeBuffer(settings, 0, words);
    this.resources = new Map([
      [0, { buffer: context.parameters }], [1, { buffer: particles }],
      [2, { buffer: this.buffer("bin counts", this.cells * 4) }], [3, { buffer: this.buffer("bin offsets", this.cells * 4) }],
      [4, { buffer: context.solidBuffer }], [5, { buffer: context.velocity }],
      [6, { buffer: context.transferredVelocity }], [7, { buffer: context.phi }],
      [8, { buffer: context.scalars }], [9, { buffer: this.buffer("particle health and outflow", 8) }], [10, { buffer: settings }],
      [12, { buffer: this.buffer("bin scatter cursors", this.cells * 4) }],
      [13, { buffer: this.buffer("spatially sorted gather particles", seed.data.byteLength) }],
      [14, { buffer: this.buffer("parallel particle reduction", Math.max(1, Math.ceil(this.count / 256)) * 32) }],
      [15, { buffer: this.buffer("FLIP pre-force velocity", this.cells * 16) }],
      [11, { buffer: context.faceVolumes ?? this.buffer("particle face volumes", this.cells * 16) }],
    ]);
    this.diagnostics = { apicParticleCount: this.count, apicInitialVolume_m3: this.volume, apicMaterialVolume_m3: this.volume, apicEscapedVolume_m3: 0 };
    // Tank-local metres; eight particles seed a cell, so spheres are a little
    // under a quarter cell in radius. A retired particle keeps its lane at w = 0.
    this.particleSource = { buffer: particles, strideFloats: 20, capacity: this.count,
      positionScale_m: [1, 1, 1], radius_m: 0.22 * Math.min(...h) };
    this.debug = { apicParticles: particles, apicParticleCapacity: this.count,
      apicBinCounts: (this.resources.get(2) as GPUBufferBinding).buffer,
      apicBinOffsets: (this.resources.get(3) as GPUBufferBinding).buffer,
      apicSortedParticles: (this.resources.get(13) as GPUBufferBinding).buffer };
    this.applyRuntimeValues(context.values ?? {});
  }
  static async create(context: MacTransportContext): Promise<ApicTransport> {
    const result = new ApicTransport(context);
    try {
      const bundle = await gpuCompilationManagerFor(context.device).acquire({ id: "particle-apic", revision: 6,
        modules: { apic: { source: APIC_SHADER } },
        compute: Object.fromEntries(Object.keys(APIC_BINDINGS).map(entry => [entry, { layout: "auto" as const, compute: { module: "apic", entryPoint: entry } }])), render: {},
      }, { signal: context.signal });
      result.commands = Object.fromEntries(Object.entries(APIC_BINDINGS).map(([entry, bindings]) => {
        const pipeline = bundle.compute[entry];
        return [entry, { pipeline, group: context.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: bindings.map(binding => ({ binding, resource: result.resources.get(binding)! })) }) }];
      })) as typeof result.commands;
      result.scan = await ParticleBinScan.create(context.device, result.cells,
        (result.resources.get(2) as GPUBufferBinding).buffer, (result.resources.get(3) as GPUBufferBinding).buffer,
        context.scalars, context.signal);
      result.allocatedBytes += result.scan.allocatedBytes;
      // Enable the initialization-only transfer/reconstruction before the first beginFrame.
      context.device.queue.writeBuffer(context.scalars, 21 * 4, new Float32Array([1]));
      return result;
    } catch (error) { result.destroy(); throw error; }
  }
  private buffer(label: string, size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const buffer = this.context.device.createBuffer({ label: `APIC ${label}`, size, usage });
    this.owned.push(buffer); this.allocatedBytes += size; return buffer;
  }
  applyRuntimeValues(values: MethodParamValues): void {
    const { transferMode, flipRatio } = particleTransferOptions(values);
    this.transferMode = transferMode; this.flipRatio = flipRatio;
    Object.assign(this.diagnostics, { apicTransferMode: transferMode, apicFlipRatio: flipRatio });
    const words = new ArrayBuffer(8); new Uint32Array(words)[0] = transferMode; new Float32Array(words)[1] = flipRatio;
    this.context.device.queue.writeBuffer(this.settings, 16, words);
  }
  private run(encoder: GPUCommandEncoder, entry: ApicEntry, count: number): void {
    if (!count) return;
    const { pipeline, group } = this.commands[entry], groups = Math.ceil(count / 64);
    const pass = encoder.beginComputePass({ label: `APIC ${entry}` });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535)); pass.end();
  }
  private bins(encoder: GPUCommandEncoder): void {
    this.run(encoder, "clearBins", this.cells); this.run(encoder, "binParticles", this.count);
    this.scan.encode(encoder); this.run(encoder, "scatterParticles", this.count);
  }
  encodeInitial(encoder: GPUCommandEncoder): void {
    this.bins(encoder); this.run(encoder, "surface", this.vertices);
  }
  encodeTransfer(encoder: GPUCommandEncoder): void {
    // Bins already describe the particle positions adopted at the previous step.
    this.run(encoder, "transfer", this.cells);
  }
  encodeMove(encoder: GPUCommandEncoder): void {
    this.run(encoder, "moveParticles", this.count); this.bins(encoder); this.run(encoder, "surface", this.vertices);
  }
  encodeStats(encoder: GPUCommandEncoder): void {
    this.run(encoder, "ledgerPartial", Math.ceil(this.count / 256) * 64);
    this.run(encoder, "ledger", 64);
  }
  updateStats(result: Float32Array, info: ApicInfo): void {
    Object.assign(info, { apicTransferMode: this.transferMode, apicFlipRatio: this.flipRatio,
      apicParticleCount: result[31], apicInitialVolume_m3: this.volume, apicAffineSpeedBound_m_s: result[5],
      apicMaterialVolume_m3: result[27], apicEscapedVolume_m3: result[28],
      apicMaterialDrift: this.volume > 0 ? (result[27] + result[28]) / this.volume - 1 : 0, apicKineticEnergy_J: result[30] });
  }
  destroy(): void { this.scan?.destroy(); for (const buffer of this.owned) buffer.destroy(); }
}
