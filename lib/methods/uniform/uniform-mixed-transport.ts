import { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import type { UniformScratchArena } from "./uniform-scratch-arena";
import { uniformMixedTransportWGSL } from "./uniform-mixed-transport.wgsl";
import type { UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

const widths = [1, 2, 4] as const;
const entries = ["clear", "build", "decode", "rowsFallback", "rowsDivide", "gather", "restrictVolume", "copyVolume"] as const;
type Entry = typeof entries[number];

/** Conservative volume stage, not a separate simulation. Persistent volume and
 * trace fields and the large transient arena belong to the native Uniform host.
 * Only ownership/worklists, counts and two-bit sampling flags are allocated here.
 */
export class UniformMixedTransportStage {
  readonly ownership: UniformMixedOwnership;
  get topologyLayout(): GPUBindGroupLayout { return this.ownership.bindLayout; }
  get topologyGroup(): GPUBindGroup { return this.ownership.bindGroup; }
  readonly allocatedBytes: number;
  get dispatchX(): number { return this.ownership.dispatchX; }
  private readonly pipelines = new Map<Entry, GPUComputePipeline[]>();
  private readonly resourcesLayout: GPUBindGroupLayout;
  private readonly restrictGroup: GPUBindGroup;
  private readonly transportGroup: GPUBindGroup;
  private readonly sampling: GPUBuffer;
  private readonly cells: number;

  get layout(): UniformMixedLayout { return this.ownership.layout; }

  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout,
    arena: UniformScratchArena, fineVolume: GPUTexture,
    restrictedVolume: GPUTexture, departures: GPUTexture,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid) {
    this.cells = layout.tiles.length * 64;
    // Fine indexing keeps the borrowed slice ABI stable across ownership changes.
    if (arena.edgeBytes < this.cells * 40 || arena.donorOffset + this.cells * 28 > arena.byteLength)
      throw new Error("Native Uniform scratch cannot hold mixed transport slices");
    this.ownership = new UniformMixedOwnership(device, layout);
    this.sampling = device.createBuffer({ label: "Uniform mixed departure sampling widths", size: layout.tiles.byteLength,
      usage: GPUBufferUsage.STORAGE });
    this.allocatedBytes = this.ownership.allocatedBytes + this.sampling.size;
    this.resourcesLayout = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      ...(sourceParams?[{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
    ] });
    const group = (input: GPUTexture, output: GPUTexture) => device.createBindGroup({ layout: this.resourcesLayout, entries: [
      { binding: 0, resource: { buffer: this.sampling } },
      { binding: 1, resource: { buffer: arena.buffer, offset: 0, size: this.cells * 40 } },
      { binding: 2, resource: { buffer: arena.buffer, offset: arena.donorOffset, size: this.cells * 24 } },
      { binding: 3, resource: { buffer: arena.buffer, offset: arena.donorOffset + this.cells * 24, size: this.cells * 4 } },
      { binding: 4, resource: input.createView() }, { binding: 5, resource: output.createView() },
      { binding: 6, resource: departures.createView() },
      ...(sourceParams?[{binding:7,resource:{buffer:sourceParams,size:176}}]:[]),
    ] });
    this.restrictGroup = group(fineVolume, restrictedVolume);
    this.transportGroup = group(restrictedVolume, fineVolume);
  }

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({ label: "Uniform graded conservative transport", code: uniformMixedTransportWGSL(this.layout,!!this.sourceParams,!!this.solid) });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.topologyLayout, this.resourcesLayout, ...(this.solid?[this.solid.bindLayout]:[])] });
    for (const entryPoint of entries) this.pipelines.set(entryPoint, await Promise.all(widths.map(umCellWidth =>
      this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint, constants: { umCellWidth, umDispatchX: this.dispatchX } } }))));
  }

  /** Reused by the native trace pipeline so every stage visits the same owners. */
  dispatch(pass: GPUComputePassEncoder, pipelines: readonly GPUComputePipeline[]): void {
    this.ownership.dispatch(pass, pipelines);
  }

  private run(encoder: GPUCommandEncoder, entry: Entry): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed transport has not been initialized");
    const pass = encoder.beginComputePass({ label: `Uniform mixed ${entry}` });
    pass.setBindGroup(0, this.topologyGroup);
    pass.setBindGroup(1, entry === "restrictVolume" || entry === "copyVolume" ? this.restrictGroup : this.transportGroup);
    if (this.solid) pass.setBindGroup(2, this.solid.bindGroup);
    this.dispatch(pass, pipelines); pass.end();
  }

  encodeCopy(encoder: GPUCommandEncoder): void { this.run(encoder, "copyVolume"); }

  encodeRestriction(encoder: GPUCommandEncoder): void { this.run(encoder, "restrictVolume"); }

  /** Call only after tracing: extension shares the edge/donor backing. */
  encodeTransport(encoder: GPUCommandEncoder): void {
    this.run(encoder, "clear"); this.run(encoder, "build"); this.run(encoder, "decode");
    for (let round = 0; round < 3; round++) {
      this.run(encoder, round === 0 ? "rowsFallback" : "rowsDivide"); this.run(encoder, "decode");
    }
    this.run(encoder, "gather");
  }

  destroy(): void { this.ownership.destroy(); this.sampling.destroy(); }
}
