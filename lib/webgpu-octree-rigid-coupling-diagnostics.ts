import type { PassBroker } from "./webgpu-pass-broker";

export interface WebGPUOctreeRigidCouplingDiagnosticInput {
  readonly authority: GPUBuffer;
  readonly leafHeaders: GPUBuffer;
  readonly solidCells: GPUBuffer;
  readonly dimensions: readonly [number, number, number];
}

/** Four u32 words: sealed wet rows, accepted generation, accepted errors, accepted row count. */
export const OCTREE_RIGID_COUPLING_DIAGNOSTIC_BYTES = 16;

export const octreeRigidCouplingDiagnosticWGSL = /* wgsl */ `
struct Params { dimensions:vec3u, rowCapacity:u32 }
struct LeafHeader {
  cell:u32, entryStart:u32, entryCount:u32, size:u32,
  diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f
}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> authority:array<u32>;
@group(0) @binding(2) var<storage,read> leafHeaders:array<LeafHeader>;
@group(0) @binding(3) var<storage,read> solidCells:array<u32>;
@group(0) @binding(4) var<storage,read_write> diagnostics:array<atomic<u32>>;
fn denseIndex(cell:vec3u)->u32 {
  return cell.x + params.dimensions.x * (cell.y + params.dimensions.y * cell.z);
}
fn solidFraction(cell:vec3u)->f32 {
  let index = denseIndex(cell);
  if (2u * index >= arrayLength(&solidCells)) { return 0.; }
  return bitcast<f32>(solidCells[2u * index]);
}
@compute @workgroup_size(1)
fn clearRigidCouplingDiagnostics() {
  atomicStore(&diagnostics[0], 0u);
  atomicStore(&diagnostics[1], authority[0]);
  atomicStore(&diagnostics[2], authority[4]);
  atomicStore(&diagnostics[3], authority[1]);
}
@compute @workgroup_size(64)
fn countSealedWetRows(@builtin(global_invocation_id) gid:vec3u) {
  let row = gid.x;
  if (arrayLength(&authority) <= 4u || authority[3] != 1u
      || row >= min(authority[1], params.rowCapacity) || row >= arrayLength(&leafHeaders)) { return; }
  let header = leafHeaders[row];
  if (header.size == 0u) { return; }
  let plane = params.dimensions.x * params.dimensions.y;
  let origin = vec3u(header.cell % params.dimensions.x,
    (header.cell / params.dimensions.x) % params.dimensions.y, header.cell / plane);
  if (any(origin + vec3u(header.size) > params.dimensions)) { return; }
  var sealed = true;
  for (var z=0u; z<header.size && sealed; z+=1u) {
    for (var y=0u; y<header.size && sealed; y+=1u) {
      for (var x=0u; x<header.size; x+=1u) {
        if (solidFraction(origin + vec3u(x,y,z)) < 0.999999) { sealed = false; break; }
      }
    }
  }
  if (sealed) { atomicAdd(&diagnostics[0], 1u); }
}
`;

/** Observational tripwire: accepted wet owners must never be wholly solid. */
export class WebGPUOctreeRigidCouplingDiagnostics {
  readonly allocatedBytes = 32;
  readonly diagnosticBuffer: GPUBuffer;
  private readonly params: GPUBuffer;
  private pipelines?: Readonly<{ clear: GPUComputePipeline; count: GPUComputePipeline }>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    readonly input: WebGPUOctreeRigidCouplingDiagnosticInput,
    readonly rowCapacity: number) {
    if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
      throw new RangeError("Rigid coupling diagnostic row capacity must be positive");
    }
    this.params = device.createBuffer({ label: "Rigid coupling diagnostic constants", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, Uint32Array.of(
      input.dimensions[0], input.dimensions[1], input.dimensions[2], rowCapacity));
    this.diagnosticBuffer = device.createBuffer({ label: "Rigid coupling diagnostic readback source",
      size: OCTREE_RIGID_COUPLING_DIAGNOSTIC_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile rigid coupling tripwire", run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipelines) return;
    const module = this.device.createShaderModule({ label: "Rigid coupling diagnostic shader",
      code: octreeRigidCouplingDiagnosticWGSL });
    const make = (entryPoint: string) => this.device.createComputePipelineAsync({
      label: entryPoint, layout: "auto", compute: { module, entryPoint },
    });
    const [clear, count] = await Promise.all([
      make("clearRigidCouplingDiagnostics"), make("countSealedWetRows"),
    ]);
    this.pipelines = Object.freeze({ clear, count });
  }

  encode(broker: PassBroker): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Rigid coupling diagnostic pipelines are not initialized");
    const buffers = [this.params, this.input.authority, this.input.leafHeaders,
      this.input.solidCells, this.diagnosticBuffer];
    const run = (pipeline: GPUComputePipeline, bindings: readonly number[], groups: number) => {
      const pass = broker.compute({ label: pipeline.label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) =>
          ({ binding, resource: { buffer: buffers[binding]! } })),
      }));
      pass.dispatchWorkgroups(groups);
    };
    run(this.pipelines.clear, [1, 4], 1);
    run(this.pipelines.count, [0, 1, 2, 3, 4], Math.ceil(this.rowCapacity / 64));
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
    this.diagnosticBuffer.destroy();
    this.pipelines = undefined;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Rigid coupling diagnostics are destroyed");
  }
}
