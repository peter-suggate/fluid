import type { PassBroker } from "./webgpu-pass-broker";
import { octreeLosassoVelocityMigrationWGSL } from
  "./webgpu-octree-losasso-velocity-migration.wgsl";

export interface WebGPUOctreeLosassoVelocityMigrationSource {
  readonly control: GPUBuffer;
  /** GPU-authored `{ceil(liveFaceCount / 64), 1, 1}` dispatch record. */
  readonly faceDispatch: GPUBuffer;
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
  private geometrySnapshotPipeline?: GPUComputePipeline;
  private lookupSnapshotPipeline?: GPUComputePipeline;
  private migrationPipeline?: GPUComputePipeline;
  private snapshotGroups?: {
    readonly source: WebGPUOctreeLosassoVelocityMigrationSource;
    readonly geometry: GPUBindGroup;
    readonly lookup: GPUBindGroup;
  };
  private migrationGroup?: {
    readonly target: WebGPUOctreeLosassoVelocityMigrationSource;
    readonly group: GPUBindGroup;
  };
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
    [this.geometrySnapshotPipeline, this.lookupSnapshotPipeline,
      this.migrationPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ label: "Snapshot Losasso wet-face geometry",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "snapshotLosassoFaceState" } }),
      this.device.createComputePipelineAsync({ label: "Snapshot Losasso wet-face lookup state",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "snapshotLosassoFaceLookup" } }),
      this.device.createComputePipelineAsync({ label: "Migrate Losasso lagged wet velocity",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "migrateLosassoLaggedVelocity" } }),
    ]); }
  encodeSnapshot(broker: PassBroker, source: WebGPUOctreeLosassoVelocityMigrationSource): void {
    this.assertReady();
    let cached = this.snapshotGroups;
    if (!cached || cached.source.control !== source.control || cached.source.faces !== source.faces
      || cached.source.faceGeometry !== source.faceGeometry
      || cached.source.axisFaceDirectory !== source.axisFaceDirectory
      || cached.source.extendedVelocity !== source.extendedVelocity) {
      const geometryPipeline = this.geometrySnapshotPipeline!;
      const lookupPipeline = this.lookupSnapshotPipeline!;
      cached = {
        source,
        geometry: this.device.createBindGroup({ layout: geometryPipeline.getBindGroupLayout(0), entries: [
          { binding: 1, resource: { buffer: this.oldControl } },
          { binding: 2, resource: { buffer: this.oldGeometry } },
          { binding: 3, resource: { buffer: source.faces } },
          { binding: 6, resource: { buffer: source.control } },
          { binding: 7, resource: { buffer: source.faceGeometry } },
        ] }),
        lookup: this.device.createBindGroup({ layout: lookupPipeline.getBindGroupLayout(0), entries: [
          { binding: 1, resource: { buffer: this.oldControl } },
          { binding: 4, resource: { buffer: this.oldDirectory } },
          { binding: 5, resource: { buffer: this.oldVelocity } },
          { binding: 10, resource: { buffer: source.axisFaceDirectory } },
          { binding: 11, resource: { buffer: source.extendedVelocity } },
        ] }),
      };
      this.snapshotGroups = cached;
    }
    // Preserve the old authority inside the compute stream. Splitting geometry
    // from lookup payloads keeps each entry point below the portable eight
    // storage-buffer limit and avoids four copy commands plus their pass break.
    const pass = broker.compute({ label: "Losasso topology - snapshot wall separation state" });
    pass.setPipeline(this.geometrySnapshotPipeline!); pass.setBindGroup(0, cached.geometry);
    pass.dispatchWorkgroupsIndirect(source.faceDispatch, 0);
    pass.setPipeline(this.lookupSnapshotPipeline!); pass.setBindGroup(0, cached.lookup);
    pass.dispatchWorkgroupsIndirect(source.faceDispatch, 12);
  }
  encodeMigration(broker: PassBroker, target: WebGPUOctreeLosassoVelocityMigrationSource): void {
    this.assertReady(); const pipeline = this.migrationPipeline!;
    const buffers = [this.params, this.oldControl, this.oldGeometry,
      this.oldDirectory, this.oldVelocity, target.control, target.faceGeometry,
      target.faces, target.extendedVelocity];
    const bindings = [0, 1, 2, 4, 5, 6, 7, 8, 9] as const;
    let cached = this.migrationGroup;
    if (!cached || cached.target.control !== target.control || cached.target.faces !== target.faces
      || cached.target.faceGeometry !== target.faceGeometry
      || cached.target.axisFaceDirectory !== target.axisFaceDirectory
      || cached.target.extendedVelocity !== target.extendedVelocity) {
      cached = { target, group: this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, index) => ({
          binding: bindings[index]!, resource: { buffer },
        })) }) };
      this.migrationGroup = cached;
    }
    const pass = broker.compute({ label: "Losasso topology - migrate lagged wet velocity" });
    pass.setPipeline(pipeline); pass.setBindGroup(0, cached.group);
    pass.dispatchWorkgroupsIndirect(target.faceDispatch, 0);
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.params, this.oldControl, this.oldGeometry,
      this.oldDirectory, this.oldVelocity]) buffer.destroy();
    this.geometrySnapshotPipeline = undefined; this.lookupSnapshotPipeline = undefined;
    this.migrationPipeline = undefined; this.snapshotGroups = undefined; this.migrationGroup = undefined; }
  private assertReady(): void { this.assertLive();
    if (!this.geometrySnapshotPipeline || !this.lookupSnapshotPipeline || !this.migrationPipeline) {
      throw new Error("Losasso velocity migration is not initialized");
    } }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso velocity migration is destroyed"); }
}
