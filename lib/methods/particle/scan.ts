import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";

// Hierarchical exclusive scan: bounded work per lane at every level. No CPU
// readback, single-workgroup global scan, or fixed-capacity particle buckets.
const SOURCE = /* wgsl */ `
@group(0) @binding(0) var<uniform> size:vec4u;
@group(0) @binding(1) var<storage,read> input:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<storage,read_write> sums:array<u32>;
@group(0) @binding(4) var<storage,read> scalars:array<f32>;
var<workgroup> scratch:array<u32,128>;
var<workgroup> running:u32;
@compute @workgroup_size(128) fn scan(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(lane==0u){running=select(0u,1u,scalars[12]==0.0&&scalars[21]>0.0);}workgroupBarrier();
 if(workgroupUniformLoad(&running)==0u){return;}
 let block=group.x+65535u*group.y;let i=block*128u+lane;
 var value=0u;if(i<size.x){value=input[i];}scratch[lane]=value;workgroupBarrier();
 for(var stride=1u;stride<128u;stride*=2u){var add=0u;if(lane>=stride){add=scratch[lane-stride];}
  workgroupBarrier();scratch[lane]+=add;workgroupBarrier();}
 if(i<size.x){output[i]=scratch[lane]-value;}
 if(lane==127u){sums[block]=scratch[lane];}
}
@compute @workgroup_size(128) fn add(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+65535u*128u*id.y;
 if(i<size.x&&scalars[12]==0.0&&scalars[21]>0.0){output[i]+=input[i/128u];}
}
`;

export class ParticleBinScan {
  allocatedBytes = 0;
  private owned: GPUBuffer[] = [];
  private levels: { count: number; scan: GPUBindGroup; add?: GPUBindGroup }[] = [];
  private pipelines!: Record<string, GPUComputePipeline>;
  private constructor(private readonly device: GPUDevice) {}
  static async create(device: GPUDevice, count: number, input: GPUBuffer, output: GPUBuffer, scalars: GPUBuffer, signal?: AbortSignal) {
    const result = new ParticleBinScan(device);
    try {
      const bundle = await gpuCompilationManagerFor(device).acquire({ id: "apic-bin-scan", revision: 1,
        modules: { scan: { source: SOURCE } }, compute: Object.fromEntries(["scan", "add"].map(entryPoint =>
          [entryPoint, { layout: "auto" as const, compute: { module: "scan", entryPoint } }])), render: {} }, { signal });
      result.pipelines = bundle.compute;
      const resources: { count: number; params: GPUBuffer; input: GPUBuffer; output: GPUBuffer; sums: GPUBuffer }[] = [];
      for (;;) {
        const groups = Math.ceil(count / 128), params = result.buffer(16, true), sums = result.buffer(groups * 4);
        device.queue.writeBuffer(params, 0, new Uint32Array([count, 0, 0, 0]));
        resources.push({ count, params, input, output, sums });
        if (groups === 1) break;
        count = groups; input = sums; output = result.buffer(count * 4);
      }
      result.levels = resources.map((r, i) => ({ count: r.count,
        scan: device.createBindGroup({ layout: bundle.compute.scan.getBindGroupLayout(0), entries:
          [r.params, r.input, r.output, r.sums, scalars].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
        add: i + 1 < resources.length ? device.createBindGroup({ layout: bundle.compute.add.getBindGroupLayout(0), entries:
          [[0, r.params], [1, resources[i + 1].output], [2, r.output], [4, scalars]].map(([binding, buffer]) =>
            ({ binding: binding as number, resource: { buffer: buffer as GPUBuffer } })) }) : undefined,
      }));
      return result;
    } catch (error) { result.destroy(); throw error; }
  }
  private buffer(size: number, uniform = false): GPUBuffer {
    const buffer = this.device.createBuffer({ label: "APIC hierarchical bin scan", size,
      usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST });
    this.owned.push(buffer); this.allocatedBytes += size; return buffer;
  }
  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "APIC bin prefix scan" });
    const run = (entry: string, count: number, group: GPUBindGroup) => {
      const groups = Math.ceil(count / 128); pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    };
    for (const l of this.levels) run("scan", l.count, l.scan);
    for (let i = this.levels.length - 2; i >= 0; i--) run("add", this.levels[i].count, this.levels[i].add!);
    pass.end();
  }
  destroy(): void { this.owned.forEach(buffer => buffer.destroy()); }
}
