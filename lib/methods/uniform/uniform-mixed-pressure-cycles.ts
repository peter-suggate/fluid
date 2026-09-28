import type { WebGPUUniformPressureMultigrid } from "./webgpu-uniform-pressure-multigrid";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { mixedCellWidth } from "./uniform-mixed-layout";
import { uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { UNIFORM_MIXED_THETA_MIN } from "./uniform-mixed-pressure-surface.wgsl";
import type { UniformMixedPressureTopology } from "./uniform-mixed-pressure-topology.wgsl";

type Continuation = ReturnType<WebGPUUniformPressureMultigrid["prepareMixedContinuation"]>;

/** The all-4h pressure root: owner-indexed fields (owner index = tile key,
 * then the six boundary planes) that RHS assembly, projection and the band
 * read and write. */
export interface UniformMixedPressureCycleLevel {
  ownership: UniformMixedOwnership;
  pressure: GPUBufferBinding;
  /** The RHS, then a Full-Cycle's correction RHS. */
  rhs: readonly [GPUBufferBinding, GPUBufferBinding];
  frozen: GPUBufferBinding;
  residual: GPUBufferBinding;
  /** The bound, then a Full-Cycle's shifted bound. */
  minimum: readonly [GPUBufferBinding, GPUBufferBinding];
  phi: GPUBufferBinding;
  /** Static solids: the all-4h (open, V+) record, directly after phi. */
  topology?: UniformMixedPressureTopology;
}

/** One compute pass shared by consecutive mixed pressure dispatches. A
 * dispatch is its own WebGPU usage scope, so dependent dispatches may share a
 * pass; encoder commands (copies, clears, foreign passes) end it first. */
export class UniformMixedPressurePasses {
  private open?: GPUComputePassEncoder;
  constructor(private readonly encoder: GPUCommandEncoder, readonly label: string) {}
  get pass(): GPUComputePassEncoder { return this.open ??= this.encoder.beginComputePass({ label: this.label }); }
  /** The encoder with no pass open. */
  get commands(): GPUCommandEncoder { this.end(); return this.encoder; }
  end(): void { this.open?.end(); this.open = undefined; }
}

type Entry = "setup" | "restrictV" | "restrictInner" | "restrictFull" | "prolongAdd" | "prolongAssign" | "prolongBackup" | "measure";

/** CM11a cycles of the all-4h pressure root, whose correction the native
 * hierarchy solves from its n/4 level (entered through its haloed
 * (T+2)^3 fields). The mixed operator stays authoritative: every residual,
 * the measure and the correction bounds are the mixed rows (ghost-fluid
 * theta at UNIFORM_MIXED_THETA_MIN, wall slots as unknowns), which the native
 * n/4 operator does not reproduce; the native V is the correction's solver.
 *
 * Every kernel walks the native lattice, one lane per native cell: interior
 * cells are the owners (owner = tile key, neighbour = key +- stride) and face
 * halo cells are the boundary slots, so there are no topology loads and no
 * intermediate level. One kernel computes the residual straight into the
 * native RHS with the correction bound and a zero correction; one adds (or
 * assigns) the native correction back. The pressure layout is all-4h with no
 * seams, so every owner is a regular row and reconstruction slopes are zero.
 * All fields are borrowed. Convergence validation and publication remain the
 * frame's job. */
export class UniformMixedPressureCycles {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly group: GPUBindGroup;
  private readonly pipelines = new Map<Entry, GPUComputePipeline>();
  private readonly workgroups: [number, number, number];
  private readonly words: Record<"p" | "b0" | "b1" | "m0" | "m1" | "backup" | "residual" | "np" | "nb" | "nmin" | "nphi" | "nv", number>;
  private readonly topologyBase?: number;
  constructor(private readonly device: GPUDevice, readonly level: UniformMixedPressureCycleLevel, backup: GPUBufferBinding,
    private readonly native: Continuation, private readonly uniformGroup: GPUBindGroup, private readonly openTop: boolean) {
    const layout = level.ownership.layout, t = layout.lattice.dimensions.map(n => n / 4);
    if (layout.tiles.some(word => mixedCellWidth(word) !== 4) || layout.tiles.length !== t[0]! * t[1]! * t[2]!)
      throw new Error("Mixed pressure root requires the uniform all-4h layout");
    const count = uniformMixedPressureStorage(layout).count;
    const nativeFields = [native.pressure, native.rhs, native.minimum, native.phi, native.topology];
    const arena = nativeFields[0]!.buffer?.buffer;
    if (!arena || nativeFields.some(f => f.buffer?.buffer !== arena)) throw new Error("Mixed pressure root requires the shared native scratch arena");
    if (nativeFields.some(f => f.dimensions.some((n, a) => n !== t[a]! + 2))) throw new Error("Mixed and native pressure extents differ");
    const mixed = [level.pressure, level.rhs[0], level.rhs[1], level.minimum[0], level.minimum[1], backup, level.residual];
    if (mixed.some(f => f.buffer !== arena || (f.size ?? f.buffer.size - (f.offset ?? 0)) < 4 * count)) throw new Error("Mixed pressure root fields must be arena views of the live words");
    // Two disjoint arena ranges: binding either whole would alias the other.
    const range = (views: readonly { offset: number; size: number }[]) => {
      const offset = Math.floor(Math.min(...views.map(v => v.offset)) / 256) * 256;
      return { offset, size: Math.max(...views.map(v => v.offset + v.size)) - offset };
    };
    const mixedRange = range(mixed.map(f => ({ offset: f.offset ?? 0, size: 4 * count })));
    const nativeRange = range(nativeFields.map(f => ({ offset: f.buffer!.offset ?? 0, size: f.buffer!.size! })));
    if (mixedRange.offset < nativeRange.offset + nativeRange.size && nativeRange.offset < mixedRange.offset + mixedRange.size)
      throw new Error("Mixed pressure root and native continuation fields overlap");
    const [p, b0, b1, m0, m1, backupWord, residual] = mixed.map(f => ((f.offset ?? 0) - mixedRange.offset) / 4) as number[];
    const [np, nb, nmin, nphi, nv] = nativeFields.map(f => ((f.buffer!.offset ?? 0) - nativeRange.offset) / 4) as number[];
    this.words = { p: p!, b0: b0!, b1: b1!, m0: m0!, m1: m1!, backup: backupWord!, residual: residual!, np: np!, nb: nb!, nmin: nmin!, nphi: nphi!, nv: nv! };
    // The static solid record rides the phi binding from this f32 index.
    let phi: GPUBufferBinding = { buffer: level.phi.buffer, offset: level.phi.offset ?? 0, size: 4 * layout.tiles.length };
    if (level.topology) {
      const base = this.topologyBase = Math.ceil(layout.tiles.length * 4 / 256) * 64, topology = level.topology.buffer;
      if (topology.buffer !== level.phi.buffer || (topology.offset ?? 0) !== (level.phi.offset ?? 0) + 4 * base || (topology.size ?? 0) < 16 * count)
        throw new Error("Mixed pressure root topology must directly follow its phi range");
      phi = { ...phi, size: 4 * base + 16 * count };
    }
    this.resources = device.createBindGroupLayout({ entries: [0, 1, 2].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
    this.group = device.createBindGroup({ layout: this.resources, entries: [
      { binding: 0, resource: { buffer: arena, ...mixedRange } }, { binding: 1, resource: { buffer: arena, ...nativeRange } }, { binding: 2, resource: phi }] });
    this.workgroups = t.map(n => Math.ceil((n + 2) / 4)) as [number, number, number];
  }
  async initialize(): Promise<void> {
    const layout = this.level.ownership.layout, h = layout.lattice.cellSize_m, t = layout.lattice.dimensions.map(n => n / 4), w = this.words;
    const solid = this.topologyBase !== undefined, open = this.openTop;
    const code = /* wgsl */ `
@group(0) @binding(2) var<storage,read_write> umSupport:array<u32>;
@group(1) @binding(0) var<storage,read_write> mixed:array<f32>;
@group(1) @binding(1) var<storage,read_write> native:array<f32>;
@group(1) @binding(2) var<storage,read_write> phi:array<f32>;
const UM_T=vec3u(${t.map(n => `${n}u`).join(",")});const UM_N=UM_T+vec3u(2u);const UM_CELLS=${layout.tiles.length}u;
const UM_H=vec3f(${h.map(n => n.toFixed(8)).join(",")});const UM_MIN_H=min(UM_H.x,min(UM_H.y,UM_H.z));
const UM_THETA_MIN:f32=${UNIFORM_MIXED_THETA_MIN};const UM_OPEN_TOP=${open};
const UM_P=${w.p}u;const UM_B0=${w.b0}u;const UM_B1=${w.b1}u;const UM_M0=${w.m0}u;const UM_M1=${w.m1}u;const UM_BACKUP=${w.backup}u;const UM_RES=${w.residual}u;
const MG_P=${w.np}u;const MG_B=${w.nb}u;const MG_MIN=${w.nmin}u;const MG_PHI=${w.nphi}u;const MG_V=${w.nv}u;
// restrict: the row's RHS and bound, and whether it opens a Full-Cycle.
override UM_RHS:u32=UM_B0;override UM_MIN:u32=UM_M0;override UM_FULL:bool=false;
// prolong: 0 add, 1 assign, 2 add then restore the Full-Cycle backup.
override UM_PROLONG:u32=0u;
// The mixed pressure schedule's slot gate (support 9n+24, 0 open).
fn umSlotClosed()->bool{return umSupport[9u*UM_CELLS+24u]!=0u;}
fn umNative(p:vec3u)->u32{return p.x+UM_N.x*(p.y+UM_N.y*p.z);}
fn umKey(c:vec3u)->u32{return c.x+UM_T.x*(c.y+UM_T.y*c.z);}
fn umStride(axis:u32)->u32{return select(select(UM_T.x*UM_T.y,UM_T.x,axis==1u),1u,axis==0u);}
// umBoundaryIndex of the all-4h layout.
fn umSlot(c:vec3u,axis:u32,side:u32)->u32{
 let d=UM_T;
 if(axis==0u){return UM_CELLS+side*d.y*d.z+c.y+d.y*c.z;}
 if(axis==1u){return UM_CELLS+2u*d.y*d.z+side*d.x*d.z+c.x+d.x*c.z;}
 return UM_CELLS+2u*(d.y*d.z+d.x*d.z)+side*d.x*d.y+c.x+d.x*c.y;
}
// A native cell: 1 an owner (interior), 2 a boundary slot (face halo), 0 an
// edge or corner, which couples to nothing.
struct UMCell{kind:u32,c:vec3u,key:u32,slot:u32,axis:u32,side:u32}
fn umCell(p:vec3u)->UMCell{
 let low=p>vec3u(0u);let high=p<UM_N-vec3u(1u);
 let inside=vec3<bool>(low.x&&high.x,low.y&&high.y,low.z&&high.z);
 let c=clamp(p,vec3u(1u),UM_N-vec3u(2u))-vec3u(1u);let key=umKey(c);
 var cell=UMCell(0u,c,key,key,0u,0u);
 if(all(inside)){cell.kind=1u;return cell;}
 if(u32(inside.x)+u32(inside.y)+u32(inside.z)!=2u){return cell;}
 for(var axis=0u;axis<3u;axis++){if(!inside[axis]){cell.axis=axis;cell.side=select(0u,1u,p[axis]>0u);}}
 cell.kind=2u;cell.slot=umSlot(c,cell.axis,cell.side);return cell;
}
fn umP(i:u32)->f32{return mixed[UM_P+i];}
fn umLiquid(k:u32)->bool{return phi[k]<0.0;}
${solid ? `fn umTopo(i:u32)->vec4f{let b=${this.topologyBase}u+4u*i;return vec4f(phi[b],phi[b+1u],phi[b+2u],phi[b+3u]);}` : ""}
fn umSum6(v:array<f32,6>)->f32{return ((v[0]+v[1])+(v[4]+v[5]))+(v[2]+v[3]);}
// umPressureSurfaceTheta / umPressureTheta of two 4h owners.
fn umSurfaceTheta(liquidPhi:f32,airPhi:f32,spacing:f32)->f32{
 let depth=max(abs(liquidPhi),UM_THETA_MIN*spacing);
 return clamp(depth/(depth+abs(airPhi)),UM_THETA_MIN,1.0);
}
fn umTheta(a:u32,b:u32)->f32{
 let la=umLiquid(a);if(la==umLiquid(b)){return 1.0;}
 var liquid=a;var air=b;if(!la){liquid=b;air=a;}
 return umSurfaceTheta(phi[liquid],phi[air],4.0*UM_MIN_H);
}
fn umOpen(axis:u32,side:u32)->bool{return UM_OPEN_TOP&&axis==1u&&side==1u;}
// umBoundaryCoefficient.
fn umWall(k:u32,s:u32,axis:u32,side:u32)->f32{
 if(!umLiquid(k)){return 0.0;}
 let distance=4.0*UM_H[axis];var fraction=0.5;var theta=1.0;
 if(umOpen(axis,side)){fraction=1.0;theta=umSurfaceTheta(phi[k],2.0*UM_MIN_H,4.0*UM_MIN_H);}
 ${solid ? "fraction=umTopo(s).x;if(fraction<=1e-6){return 0.0;}" : ""}
 return fraction/(distance*distance*theta);
}
// The regular row's (diagonal, A p): umPressureRegularTerms.
fn umRow(c:vec3u,k:u32)->vec2f{
 var diagonal:array<f32,6>;var values:array<f32,6>;let own=umP(k);
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let at=2u*axis+side;
  if(select(c[axis]==0u,c[axis]+1u==UM_T[axis],side==1u)){
   let s=umSlot(c,axis,side);let weight=umWall(k,s,axis,side);
   var p=0.0;if(!umOpen(axis,side)){p=umP(s);}
   diagonal[at]=weight;values[at]=weight*own-weight*p;
  }else{
   let n=select(k-umStride(axis),k+umStride(axis),side==1u);
   let distance=4.0*UM_H[axis];
   ${solid ? `let volume=select(umTopo(n)[axis+1u],umTopo(k)[axis+1u],side==1u);
   let coefficient=select(volume/(distance*distance*umTheta(k,n)),0.0,volume<=1e-6);` : "let coefficient=1.0/(distance*distance*umTheta(k,n));"}
   let other=select(0.0,umP(n),umLiquid(n));
   diagonal[at]=coefficient;values[at]=coefficient*(own-other);
  }
 }}
 return vec2f(umSum6(diagonal),umSum6(values));
}
// A slot row couples to its owner alone; the open lid has no slot row.
fn umSlotCoefficient(cell:UMCell)->f32{return select(umWall(cell.key,cell.slot,cell.axis,cell.side),0.0,umOpen(cell.axis,cell.side));}
fn umResidual(cell:UMCell,rhs:u32)->f32{
 if(cell.kind==1u){return select(0.0,mixed[rhs+cell.key]-umRow(cell.c,cell.key).y,umLiquid(cell.key));}
 return mixed[rhs+cell.slot]-umSlotCoefficient(cell)*(umP(cell.slot)-umP(cell.key));
}
fn umNonfinite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)==0x7f800000u;}
// The CM11a projected residual of one row; a maximal sentinel on failure.
fn umProjected(p:f32,r:f32,low:f32,diagonal:f32)->f32{
 let gap=max(0.0,p-low);
 let projected=max(select(abs(r),gap*diagonal,r<0.0&&-r>=gap*diagonal),max(0.0,low-p)*diagonal);
 return select(3.402823e38,projected,!umNonfinite(p)&&!umNonfinite(r)&&!umNonfinite(projected)&&projected>=0.0);
}

// Per solve: the native n/4 phi and topology the continuation setup extends
// and bakes (walls at the halo, V=1/2 on closed domain faces).
@compute @workgroup_size(4,4,4) fn setup(@builtin(global_invocation_id) p:vec3u){
 if(any(p>=UM_N)){return;}let at=umNative(p);let cell=umCell(p);
 var distance:f32=${2 * Math.min(...h)};var topology=vec4f(0.0);
 if(cell.kind==1u){
  distance=phi[cell.key];${solid ? "topology=umTopo(cell.key);" : `topology=vec4f(1.0);
  for(var axis=0u;axis<3u;axis++){if(p[axis]==UM_N[axis]-2u){topology[axis+1u]=select(0.5,1.0,UM_OPEN_TOP&&axis==1u);}}`}
 }else if(cell.kind==2u){
  for(var axis=0u;axis<3u;axis++){if(p[axis]==0u){topology[axis+1u]=${solid ? "umTopo(cell.slot).x" : "0.5"};}}
  if(UM_OPEN_TOP&&p.y==UM_N.y-1u){topology.x=1.0;}
 }
 native[MG_PHI+at]=distance;for(var component=0u;component<4u;component++){native[MG_V+4u*at+component]=topology[component];}
}
// The correction problem: its RHS is the mixed residual, its bound the row's
// bound less the pressure, its initial value zero. A Full-Cycle also keeps
// the pressure, the shifted bound and the residual for its closing V-cycle.
@compute @workgroup_size(4,4,4) fn restrictRoot(@builtin(global_invocation_id) p:vec3u){
 if(umSlotClosed()||any(p>=UM_N)){return;}let at=umNative(p);let cell=umCell(p);
 var b=0.0;var lower=0.0;
 if(cell.kind!=0u){
  let s=cell.slot;let own=umP(s);b=umResidual(cell,UM_RHS);lower=mixed[UM_MIN+s]-own;
  if(UM_FULL){mixed[UM_BACKUP+s]=own;mixed[UM_M1+s]=lower;mixed[UM_B1+s]=b;}
 }
 native[MG_P+at]=0.0;native[MG_B+at]=b;native[MG_MIN+at]=lower;
}
@compute @workgroup_size(4,4,4) fn prolongRoot(@builtin(global_invocation_id) p:vec3u){
 if(umSlotClosed()||any(p>=UM_N)){return;}let cell=umCell(p);if(cell.kind==0u){return;}
 let s=cell.slot;let e=native[MG_P+umNative(p)];
 if(UM_PROLONG==1u){mixed[UM_P+s]=e;}
 else if(UM_PROLONG==2u){mixed[UM_P+s]=(mixed[UM_P+s]+e)+mixed[UM_BACKUP+s];}
 else{mixed[UM_P+s]+=e;}
}
// The acceptance norm's rows, owners and wall slots.
@compute @workgroup_size(4,4,4) fn measure(@builtin(global_invocation_id) p:vec3u){
 if(umSlotClosed()||any(p>=UM_N)){return;}let cell=umCell(p);let s=cell.slot;
 if(cell.kind==2u){
  let coefficient=umSlotCoefficient(cell);let q=umP(s);
  mixed[UM_RES+s]=umProjected(q,mixed[UM_B0+s]-coefficient*(q-umP(cell.key)),mixed[UM_M0+s],coefficient);return;
 }
 if(cell.kind!=1u){return;}
 let q=umP(s);
 if(!umLiquid(s)){mixed[UM_RES+s]=select(0.0,3.402823e38,umNonfinite(q));return;}
 let row=umRow(cell.c,s);
 mixed[UM_RES+s]=umProjected(q,mixed[UM_B0+s]-row.y,mixed[UM_M0+s],row.x);
}
`;
    const module = this.device.createShaderModule({ label: "Uniform mixed pressure root", code });
    const info = await module.getCompilationInfo(), errors = info.messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.level.ownership.bindLayout, this.resources] });
    const variants: Record<Entry, [string, Record<string, number>]> = {
      setup: ["setup", {}], measure: ["measure", {}],
      restrictV: ["restrictRoot", {}], restrictInner: ["restrictRoot", { UM_RHS: w.b1, UM_MIN: w.m1 }], restrictFull: ["restrictRoot", { UM_FULL: 1 }],
      prolongAdd: ["prolongRoot", {}], prolongAssign: ["prolongRoot", { UM_PROLONG: 1 }], prolongBackup: ["prolongRoot", { UM_PROLONG: 2 }],
    };
    for (const [entry, [entryPoint, constants]] of Object.entries(variants) as [Entry, [string, Record<string, number>]][])
      this.pipelines.set(entry, await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint, constants } }));
  }
  private dispatch(passes: UniformMixedPressurePasses, entry: Entry): void {
    const pipeline = this.pipelines.get(entry); if (!pipeline) throw new Error("Mixed pressure root is not initialized");
    const pass = passes.pass;
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.level.ownership.bindGroup); pass.setBindGroup(1, this.group);
    pass.dispatchWorkgroups(...this.workgroups);
  }
  /** One batch shares a pass with the native continuation; only the
   * native setup's tile-list clears end it. */
  private batch(encoder: GPUCommandEncoder, label: string, body: (passes: UniformMixedPressurePasses, native: (kind: "v" | "full", setup?: boolean | "setup") => void) => void): void {
    const passes = new UniformMixedPressurePasses(encoder, `Uniform mixed pressure ${label}`);
    body(passes, (kind, setup = false) => this.native.encode(encoder, this.uniformGroup, kind, setup, passes)); passes.end();
  }
  /** Per solve, after RHS assembly and before the initial measure: the native
   * n/4 geometry and the continuation's setup. Ungated. */
  encodeSetup(encoder: GPUCommandEncoder): void {
    this.batch(encoder, "setup", (passes, native) => { this.dispatch(passes, "setup"); native("v", "setup"); });
  }
  encodeMeasure(encoder: GPUCommandEncoder): void { this.batch(encoder, "measure", passes => this.dispatch(passes, "measure")); }
  encodeVCycle(encoder: GPUCommandEncoder): void {
    this.batch(encoder, "V-cycle", (passes, native) => { this.dispatch(passes, "restrictV"); native("v"); this.dispatch(passes, "prolongAdd"); });
  }
  /** Algorithm 3 at the root: the native Full-Cycle on the residual under the
   * shifted bound, a closing V-cycle, then the saved pressure added back. */
  encodeFullCycle(encoder: GPUCommandEncoder): void {
    this.batch(encoder, "Full-cycle", (passes, native) => {
      this.dispatch(passes, "restrictFull"); native("full"); this.dispatch(passes, "prolongAssign");
      this.dispatch(passes, "restrictInner"); native("v"); this.dispatch(passes, "prolongBackup");
    });
  }
}
