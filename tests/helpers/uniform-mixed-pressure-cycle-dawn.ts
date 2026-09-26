import { createCm12NumericsWGSL } from "../../lib/core/cm12-numerics";
import { DEFAULT_UNIFORM_CM11A_SCHEDULE, UNIFORM_CM11A_COARSE_SWEEP_CAP } from "../../lib/methods/uniform/pressure-policy";
import { uniformCoarseSolverWGSL, uniformPressureStateWGSL, UNIFORM_CM11A_COARSE_HEADER_BYTES, UNIFORM_CM11A_COARSE_ROW_BYTES } from "../../lib/methods/uniform/uniform-coarse-solver.wgsl";
import { uniformMixedPressureLevel, type UniformMixedLayout } from "../../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureCycles } from "../../lib/methods/uniform/uniform-mixed-pressure-cycles";

interface Level {
  ownership: UniformMixedOwnership;
  pressure: GPUBuffer; slopes: GPUBuffer; frozen: GPUBuffer; rhs: [GPUBuffer, GPUBuffer]; residual: GPUBuffer;
  minimum: GPUBuffer[]; phi?: GPUBuffer;
}

/** Device-only cycle fixture: production traversal/operators with a small
 * native coarse solve. This harness owns fields and readback; production
 * traversal borrows them and allocates no fields. No live solver is installed.
 */
export class MixedPressureCycleDawn {
  readonly levels: Level[];
  private readonly owned: GPUBuffer[] = [];
  private cycles!: UniformMixedPressureCycles;
  private readonly backup: GPUBuffer;
  private readonly state: GPUBuffer;
  private coarsePipeline?: GPUComputePipeline;
  private coarseEmpty?: GPUBindGroup;
  private readonly coarseGroups = new Map<GPUBuffer, GPUBindGroup>();

  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout, private readonly constrained = false, private readonly surface = false) {
    this.levels = [layout, uniformMixedPressureLevel(layout, 2), uniformMixedPressureLevel(layout, 4)].map((l, level) => ({
      ownership: new UniformMixedOwnership(device, l), pressure: this.buffer(l.cellCount * 4), slopes: this.buffer(l.cellCount * 16),
      frozen: this.buffer(l.cellCount * 4), rhs: [this.buffer(l.cellCount * 4), this.buffer(l.cellCount * 4)], residual: this.buffer(l.cellCount * 4),
      phi: surface ? this.buffer(l.cellCount * 4) : undefined,
      minimum: constrained ? Array.from({ length: level === 0 ? 2 : 1 }, () => this.buffer(l.cellCount * 4)) : [],
    }));
    this.backup = this.buffer(layout.cellCount * 4);
    this.state = this.buffer(UNIFORM_CM11A_COARSE_HEADER_BYTES + this.levels[2]!.ownership.layout.cellCount * UNIFORM_CM11A_COARSE_ROW_BYTES);
  }
  private buffer(size: number): GPUBuffer {
    const result = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.owned.push(result); return result;
  }
  private async module(code: string): Promise<GPUShaderModule> {
    const module = this.device.createShaderModule({ code }), info = await module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    return module;
  }
  async initialize(): Promise<void> {
    await this.initializeCoarse();
    const view=(buffer:GPUBuffer):GPUBufferBinding=>({buffer});
    const borrowedDevice=new Proxy(this.device,{get(target,key){
      if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("Mixed pressure traversal allocated a field");};
      const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
    }});
    this.cycles=new UniformMixedPressureCycles(borrowedDevice,this.levels.map(l=>({
      ownership:l.ownership,pressure:view(l.pressure),slopes:view(l.slopes),rhs:[view(l.rhs[0]),view(l.rhs[1])],
      frozen:view(l.frozen),residual:view(l.residual),minimum:l.minimum.length?l.minimum.map(view):undefined,phi:l.phi?view(l.phi):undefined,
    })),view(this.backup),(encoder,rhs)=>this.coarse(encoder,rhs.buffer),[true,true]);
    await this.cycles.initialize();
  }
  private async initializeCoarse(): Promise<void> {
    const layout = this.levels[2]!.ownership.layout, d = layout.tileDimensions, h = layout.lattice.cellSize_m.map(v => v * 4);
    // Only resource access is adapted. Preserve the native strided projected
    // coarse solver, double-single arithmetic, stopping test and sweep cap.
    const replacements = [
      ["textureLoad(mgRhsIn,id,0).x", "coarseRhs[lane]"],
      ["textureLoad(mgMinimumIn,id,0).x", this.constrained ? "coarseMinimum[lane]" : "-3.402823e38"],
      ["textureStore(mgPressureOut,id,vec4f(mgState.rows[lane].p+mgState.rows[lane].low));", "coarsePressure[lane]=mgState.rows[lane].p+mgState.rows[lane].low;"],
    ];
    let coarse = uniformCoarseSolverWGSL;
    for (const [from, to] of replacements) {
      if (coarse.split(from!).length !== 2) throw new Error("Native coarse fixture adapter needs review");
      coarse = coarse.replace(from!, to!);
    }
    const module = await this.module(/* wgsl */ `
@group(1) @binding(0) var<storage,read_write> coarsePressure:array<f32>;
@group(1) @binding(1) var<storage,read> coarseRhs:array<f32>;
${this.constrained ? "@group(1) @binding(2) var<storage,read> coarseMinimum:array<f32>;" : ""}
${this.surface ? "@group(1) @binding(3) var<storage,read> coarsePhi:array<f32>;" : ""}
${uniformPressureStateWGSL}
struct MG {levelDims:vec4u,spacing:vec4f,control:vec4u};
const mg=MG(vec4u(${d.join(",")},0),vec4f(${h.map(n => n.toFixed(8)).join(",")},0),vec4u(0,0,1,${UNIFORM_CM11A_COARSE_SWEEP_CAP}));
struct Params {dimsDt:vec4f,physical:vec4f,boundary:vec4f};
const params=Params(vec4f(0,0,0,1),vec4f(1,0,0,0),vec4f(0));
const mgTolerance=vec4f(0);var<workgroup> mgCycleStopped:u32;
fn mgSkipCycle()->bool{return false;}
fn mgP(p:vec3i)->f32{return coarsePressure[u32(p.x)+mg.levelDims.x*(u32(p.y)+mg.levelDims.y*u32(p.z))];}
fn mgPhi(p:vec3i)->f32{return ${this.surface ? "coarsePhi[u32(p.x)+mg.levelDims.x*(u32(p.y)+mg.levelDims.y*u32(p.z))]" : "-1.0"};}
fn mgTopology(p:vec3i)->vec4f{return vec4f(1.0);}
fn mgInterior(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(0))&&all(p<vec3i(d));}
fn mgD4Sum6(v:array<f32,6>)->f32{return ((v[0]+v[1])+(v[4]+v[5]))+(v[2]+v[3]);}
${createCm12NumericsWGSL()}
${coarse}
`);
    this.coarsePipeline = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "mgSolveCoarsest" } });
    this.coarseEmpty = this.device.createBindGroup({ layout: this.coarsePipeline.getBindGroupLayout(0), entries: [] });
    for (const rhs of this.levels[2]!.rhs) this.coarseGroups.set(rhs, this.device.createBindGroup({ layout: this.coarsePipeline.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: this.levels[2]!.pressure } }, { binding: 1, resource: { buffer: rhs } }, { binding: 13, resource: { buffer: this.state } },
      ...(this.surface ? [{ binding: 3, resource: { buffer: this.levels[2]!.phi! } }] : []),
      ...(this.constrained ? [{ binding: 2, resource: { buffer: this.levels[2]!.minimum[0]! } }] : []),
    ] }));
  }
  private coarse(encoder: GPUCommandEncoder, rhs: GPUBuffer): void {
    const pass = encoder.beginComputePass(); pass.setPipeline(this.coarsePipeline!); pass.setBindGroup(0, this.coarseEmpty!);
    pass.setBindGroup(1, this.coarseGroups.get(rhs)!); pass.dispatchWorkgroups(1); pass.end();
  }
  async solve(rhs: Float32Array<ArrayBuffer>, minimum?: Float32Array<ArrayBuffer>, phi?: Float32Array<ArrayBuffer>): Promise<{ pressure: Float32Array; slopes: Float32Array; residuals: number[]; coarseExhausted: number }> {
    const n = this.levels[0]!.ownership.layout.cellCount, s = DEFAULT_UNIFORM_CM11A_SCHEDULE, cycles = s.fullCycles + s.vCycles;
    if (rhs.length !== n) throw new Error("RHS size mismatch");
    if (this.constrained !== !!minimum || (minimum && minimum.length !== n)) throw new Error("Minimum size/mode mismatch");
    if (this.surface !== !!phi || (phi && phi.length !== n))
      throw new Error("Phi size/mode mismatch");
    if(phi)this.device.queue.writeBuffer(this.levels[0]!.phi!,0,phi);
    if (minimum) this.device.queue.writeBuffer(this.levels[0]!.minimum[0]!, 0, minimum);
    this.device.queue.writeBuffer(this.levels[0]!.rhs[0], 0, rhs);
    const readback = this.device.createBuffer({ size: 4 * (n * (cycles + 5) + 28), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder(); encoder.clearBuffer(this.state); encoder.clearBuffer(this.levels[0]!.pressure);
      this.cycles.encodeSurfaceRestriction(encoder);
      for (let cycle = 0; cycle < cycles; cycle++) {
        if (cycle < s.fullCycles) this.cycles.encodeFullCycle(encoder); else this.cycles.encodeVCycle(encoder);
        if (this.constrained) this.cycles.encodeMeasure(encoder); else this.cycles.encodeResidual(encoder);
        encoder.copyBufferToBuffer(this.levels[0]!.residual, 0, readback, cycle * n * 4, n * 4);
      }
      encoder.copyBufferToBuffer(this.levels[0]!.pressure, 0, readback, cycles * n * 4, n * 4);
      encoder.copyBufferToBuffer(this.levels[0]!.slopes, 0, readback, (cycles + 1) * n * 4, n * 16);
      encoder.copyBufferToBuffer(this.state, 0, readback, (cycles + 5) * n * 4, 112);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange(), values = new Float32Array(bytes), words = new Uint32Array(bytes);
      const residuals = Array.from({ length: cycles }, (_, cycle) => Math.max(...values.subarray(cycle * n, (cycle + 1) * n).map(Math.abs)));
      const pressure = values.slice(cycles * n, (cycles + 1) * n), coarseExhausted = words[(cycles + 5) * n + 3]!;
      const slopes = values.slice((cycles + 1) * n, (cycles + 5) * n);
      readback.unmap(); return { pressure, slopes, residuals, coarseExhausted };
    } finally { if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); }
  }
  destroy(): void { this.owned.forEach(b => b.destroy()); this.levels.forEach(l => l.ownership.destroy()); }
}
