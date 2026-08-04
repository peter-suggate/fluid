import type { PassBroker } from "./webgpu-pass-broker";
import { octreeLosassoVelocityMigrationWGSL } from
  "./webgpu-octree-losasso-velocity-migration.wgsl";

export interface WebGPUOctreeLosassoVelocityMigrationSource {
  readonly control: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly axisFaceDirectory: GPUBuffer;
  readonly extendedVelocity: GPUBuffer;
}

/** Snapshots the old wet authority, then exactly area-averages it onto new dyadic faces. */
export class WebGPUOctreeLosassoVelocityMigration {
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly oldControl: GPUBuffer;
  private readonly oldGeometry: GPUBuffer;
  private readonly oldDirectory: GPUBuffer;
  private readonly oldVelocity: GPUBuffer;
  private snapshotPipeline?: GPUComputePipeline;
  private migrationPipeline?: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly input: {
    readonly dimensions: readonly [number, number, number];
    readonly maximumLeafSize: number;
    readonly faceCapacity: number;
    readonly directoryCapacity: number;
  }) {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(16, size), usage: storage });
    this.params = device.createBuffer({ label: "Losasso velocity migration parameters", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const words = new Uint32Array(8); words.set([...input.dimensions, input.maximumLeafSize], 0);
    words.set([input.faceCapacity, input.directoryCapacity], 4);
    device.queue.writeBuffer(this.params, 0, words);
    this.oldControl = make("Losasso retired wet-face control", 32);
    this.oldGeometry = make("Losasso retired wet-face geometry", input.faceCapacity * 16);
    this.oldDirectory = make("Losasso retired wet-face directory", input.directoryCapacity * 8);
    this.oldVelocity = make("Losasso retired lagged wet velocity", input.faceCapacity * 4);
    this.allocatedBytes = this.params.size + this.oldControl.size + this.oldGeometry.size
      + this.oldDirectory.size + this.oldVelocity.size;
  }
  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso lagged velocity migration", run: () => this.initialize() }];
  }
  async initialize(): Promise<void> { this.assertLive(); if (this.migrationPipeline) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso lagged wet velocity migration shader",
      code: octreeLosassoVelocityMigrationWGSL });
    [this.snapshotPipeline, this.migrationPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ label: "Snapshot Losasso wall separation state",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "snapshotLosassoFaceState" } }),
      this.device.createComputePipelineAsync({ label: "Migrate Losasso lagged wet velocity",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "migrateLosassoLaggedVelocity" } }),
    ]); }
  encodeSnapshot(broker: PassBroker, source: WebGPUOctreeLosassoVelocityMigrationSource): void {
    this.assertReady();
    broker.copyBufferToBuffer(source.control, 0, this.oldControl, 0, Math.min(32, source.control.size));
    broker.copyBufferToBuffer(source.faceGeometry, 0, this.oldGeometry, 0,
      Math.min(this.oldGeometry.size, source.faceGeometry.size));
    broker.copyBufferToBuffer(source.axisFaceDirectory, 0, this.oldDirectory, 0,
      Math.min(this.oldDirectory.size, source.axisFaceDirectory.size));
    broker.copyBufferToBuffer(source.extendedVelocity, 0, this.oldVelocity, 0,
      Math.min(this.oldVelocity.size, source.extendedVelocity.size));
    // Pack the sole lagged wall-state bit into the retired geometry word. This
    // keeps the migration entry point at WebGPU's portable eight-storage-buffer
    // ceiling instead of carrying a ninth full face-record binding.
    const pipeline = this.snapshotPipeline!;
    const pass = broker.compute({ label: "Losasso topology - snapshot wall separation state" });
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 1, resource: { buffer: this.oldControl } },
        { binding: 2, resource: { buffer: this.oldGeometry } },
        { binding: 3, resource: { buffer: source.faces } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(this.input.faceCapacity / 64));
  }
  encodeMigration(broker: PassBroker, target: WebGPUOctreeLosassoVelocityMigrationSource): void {
    this.assertReady(); const pipeline = this.migrationPipeline!;
    const buffers = [this.params, this.oldControl, this.oldGeometry,
      this.oldDirectory, this.oldVelocity, target.control, target.faceGeometry,
      target.faces, target.extendedVelocity];
    const bindings = [0, 1, 2, 4, 5, 6, 7, 8, 9] as const;
    const pass = broker.compute({ label: "Losasso topology - migrate lagged wet velocity" });
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, index) => ({
        binding: bindings[index]!, resource: { buffer },
      })) }));
    pass.dispatchWorkgroups(Math.ceil(this.input.faceCapacity / 64));
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.params, this.oldControl, this.oldGeometry,
      this.oldDirectory, this.oldVelocity]) buffer.destroy();
    this.snapshotPipeline = undefined; this.migrationPipeline = undefined; }
  private assertReady(): void { this.assertLive(); if (!this.snapshotPipeline || !this.migrationPipeline) throw new Error("Losasso velocity migration is not initialized"); }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso velocity migration is destroyed"); }
}
