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
  /** Per-candidate-face publication state: 0 missing, 1 migrated, 2 nodal. */
  readonly candidateStatus: GPUBuffer;
  /** generation,faces,migrated,nodal,errors,valid-generation,sweeps,partial-final */
  readonly receipt: GPUBuffer;
  private preparePipeline?: GPUComputePipeline;
  private geometrySnapshotPipeline?: GPUComputePipeline;
  private lookupSnapshotPipeline?: GPUComputePipeline;
  private migrationPipeline?: GPUComputePipeline;
  private separationPipeline?: GPUComputePipeline;
  private coveragePreparePipeline?: GPUComputePipeline;
  private completionPipeline?: GPUComputePipeline;
  private coveragePipeline?: GPUComputePipeline;
  private finishPipeline?: GPUComputePipeline;
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
    this.candidateStatus = make("Losasso candidate migrated-face status", input.faceCapacity * 4);
    this.receipt = make("Losasso candidate velocity migration receipt", 32);
    this.allocatedBytes = this.params.size + this.oldControl.size + this.oldGeometry.size
      + this.oldDirectory.size + this.oldVelocity.size + this.candidateStatus.size
      + this.receipt.size;
  }
  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso lagged velocity migration", run: () => this.initialize() }];
  }
  async initialize(): Promise<void> { this.assertLive(); if (this.migrationPipeline) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso lagged wet velocity migration shader",
      code: octreeLosassoVelocityMigrationWGSL });
    [this.preparePipeline, this.geometrySnapshotPipeline, this.lookupSnapshotPipeline,
      this.migrationPipeline, this.separationPipeline, this.coveragePreparePipeline,
      this.completionPipeline,
      this.coveragePipeline, this.finishPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ label: "Prepare Losasso wet-face migration",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "prepareLosassoLaggedVelocityMigration" } }),
      this.device.createComputePipelineAsync({ label: "Snapshot Losasso wet-face geometry",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "snapshotLosassoFaceState" } }),
      this.device.createComputePipelineAsync({ label: "Snapshot Losasso wet-face lookup state",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "snapshotLosassoFaceLookup" } }),
      this.device.createComputePipelineAsync({ label: "Migrate Losasso lagged wet velocity",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "migrateLosassoLaggedVelocity" } }),
      this.device.createComputePipelineAsync({ label: "Migrate Losasso face separation",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "migrateLosassoFaceSeparation" } }),
      this.device.createComputePipelineAsync({ label: "Prepare Losasso velocity coverage verdict",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "prepareLosassoVelocityMigrationCoverage" } }),
      this.device.createComputePipelineAsync({ label: "Complete Losasso new faces from nodes",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "completeLosassoLaggedVelocityFromNodes" } }),
      this.device.createComputePipelineAsync({ label: "Count Losasso velocity migration coverage",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "countLosassoVelocityMigrationCoverage" } }),
      this.device.createComputePipelineAsync({ label: "Validate Losasso velocity migration",
        layout: "auto", compute: { module: shaderModule,
          entryPoint: "finishLosassoLaggedVelocityMigration" } }),
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
      target.extendedVelocity, this.candidateStatus];
    const bindings = [0, 1, 2, 4, 5, 6, 7, 9, 12] as const;
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
    const prepare = this.preparePipeline!;
    const prepareGroup = this.device.createBindGroup({ layout: prepare.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: target.control } },
      { binding: 9, resource: { buffer: target.extendedVelocity } },
      { binding: 12, resource: { buffer: this.candidateStatus } },
      { binding: 13, resource: { buffer: this.receipt } },
    ] });
    const pass = broker.compute({ label: "Losasso topology - prepare lagged wet velocity" });
    pass.setPipeline(prepare); pass.setBindGroup(0, prepareGroup);
    pass.dispatchWorkgroupsIndirect(target.faceDispatch, 0);
    pass.setPipeline(pipeline); pass.setBindGroup(0, cached.group);
    pass.dispatchWorkgroupsIndirect(target.faceDispatch, 0);
    const separation = this.separationPipeline!;
    const separationGroup = this.device.createBindGroup({
      layout: separation.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.oldControl } },
        { binding: 2, resource: { buffer: this.oldGeometry } },
        { binding: 4, resource: { buffer: this.oldDirectory } },
        { binding: 6, resource: { buffer: target.control } },
        { binding: 7, resource: { buffer: target.faceGeometry } },
        { binding: 8, resource: { buffer: target.faces } },
      ],
    });
    pass.setPipeline(separation); pass.setBindGroup(0, separationGroup);
    pass.dispatchWorkgroupsIndirect(target.faceDispatch, 0);
  }
  encodeNodalCompletion(broker: PassBroker, target: WebGPUOctreeLosassoVelocityMigrationSource,
    graph: { readonly control: GPUBuffer; readonly nodeDirectory: GPUBuffer;
      readonly nodalVelocity: GPUBuffer; readonly constraints: GPUBuffer }): void {
    this.assertReady(); const pipeline = this.completionPipeline!;
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: target.control } },
      { binding: 7, resource: { buffer: target.faceGeometry } },
      { binding: 9, resource: { buffer: target.extendedVelocity } },
      { binding: 12, resource: { buffer: this.candidateStatus } },
      { binding: 13, resource: { buffer: this.receipt } },
      { binding: 14, resource: { buffer: graph.control } },
      { binding: 15, resource: { buffer: graph.nodeDirectory } },
      { binding: 16, resource: { buffer: graph.nodalVelocity } },
      { binding: 17, resource: { buffer: graph.constraints } },
    ] });
    const reset = this.coveragePreparePipeline!;
    const resetGroup = this.device.createBindGroup({ layout: reset.getBindGroupLayout(0), entries: [
      { binding: 13, resource: { buffer: this.receipt } },
    ] });
    const pass = broker.compute({ label: "Losasso topology - complete new faces from nodes" });
    pass.setPipeline(reset); pass.setBindGroup(0, resetGroup); pass.dispatchWorkgroups(1);
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(target.faceDispatch, 0);
    const coverage = this.coveragePipeline!;
    const coverageGroup = this.device.createBindGroup({ layout: coverage.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: target.control } },
      { binding: 12, resource: { buffer: this.candidateStatus } },
      { binding: 13, resource: { buffer: this.receipt } },
    ] });
    pass.setPipeline(coverage); pass.setBindGroup(0, coverageGroup);
    pass.dispatchWorkgroupsIndirect(target.faceDispatch, 0);
    const finish = this.finishPipeline!;
    const finishGroup = this.device.createBindGroup({ layout: finish.getBindGroupLayout(0), entries: [
      { binding: 6, resource: { buffer: target.control } },
      { binding: 13, resource: { buffer: this.receipt } },
    ] });
    pass.setPipeline(finish); pass.setBindGroup(0, finishGroup); pass.dispatchWorkgroups(1);
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.params, this.oldControl, this.oldGeometry,
      this.oldDirectory, this.oldVelocity, this.candidateStatus, this.receipt]) buffer.destroy();
    this.preparePipeline = undefined;
    this.geometrySnapshotPipeline = undefined; this.lookupSnapshotPipeline = undefined;
    this.migrationPipeline = undefined; this.separationPipeline = undefined;
    this.coveragePreparePipeline = undefined; this.completionPipeline = undefined;
    this.coveragePipeline = undefined;
    this.finishPipeline = undefined; this.snapshotGroups = undefined; this.migrationGroup = undefined; }
  private assertReady(): void { this.assertLive();
    if (!this.preparePipeline || !this.geometrySnapshotPipeline || !this.lookupSnapshotPipeline
      || !this.migrationPipeline || !this.separationPipeline || !this.completionPipeline
      || !this.coveragePreparePipeline
      || !this.coveragePipeline || !this.finishPipeline) {
      throw new Error("Losasso velocity migration is not initialized");
    } }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso velocity migration is destroyed"); }
}
