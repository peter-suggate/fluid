/** Preserve display energy before a solver reuses its grid-velocity scratch.
 * This pass only reads simulation state; its output is consumed by rendering. */
export class GPUFluidParticleMotion {
  readonly energy: GPUBuffer;
  private pipeline?: GPUComputePipeline;
  private group?: GPUBindGroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly particles: GPUBuffer,
    private readonly count: GPUBuffer,
    private readonly strideFloats: number,
    capacity: number,
  ) {
    this.energy = device.createBuffer({
      label: "Fluid particle display energy", size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({
      label: "Fluid particle display energy", code: /* wgsl */`
@group(0) @binding(0) var<storage,read> particles:array<f32>;
@group(0) @binding(1) var<storage,read> count:array<u32>;
@group(0) @binding(2) var<storage,read_write> energy:array<f32>;
@compute @workgroup_size(128)
fn capture(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u) {
  let n=min(count[0],arrayLength(&energy));
  for(var i=gid.x;i<n;i+=128u*groups.x) {
    let at=i*${this.strideFloats}u;
    let velocity=vec3f(particles[at+4u],particles[at+5u],particles[at+6u]);
    let mean=vec3f(particles[at+8u],particles[at+9u],particles[at+10u]);
    let fluctuation=velocity-mean;
    energy[i]=dot(fluctuation,fluctuation);
  }
}`,
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      label: "Fluid particle display energy", layout: "auto",
      compute: { module, entryPoint: "capture" },
    });
    this.group = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.particles } },
        { binding: 1, resource: { buffer: this.count } },
        { binding: 2, resource: { buffer: this.energy } },
      ],
    });
  }

  capture(encoder: GPUCommandEncoder): void {
    if (!this.pipeline || !this.group) throw new Error("Particle display energy is not initialized");
    const pass = encoder.beginComputePass({ label: "Fluid particle display energy" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(256);
    pass.end();
  }

  destroy(): void { this.energy.destroy(); }
}
