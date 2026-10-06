import type { WebGPUUniformPressureMultigrid } from "./webgpu-uniform-pressure-multigrid";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformBufferedWork } from "./uniform-buffered-work";
import { mixedCellWidth } from "./uniform-mixed-layout";
import { uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { uniformMixedPressureLiquidWGSL, uniformMixedSurfaceThetaWGSL } from "./uniform-mixed-pressure-surface.wgsl";
import type { UniformMixedPressureTopology } from "./uniform-mixed-pressure-topology.wgsl";

type Continuation = ReturnType<WebGPUUniformPressureMultigrid["prepareMixedContinuation"]>;

/** One smoothing visit of the root solve: the weights of its simultaneous
 * (Jacobi) iterations, in order. Seven undamped iterations carry the
 * smoothing and two half-weight ones remove the alternating-sign error
 * undamped Jacobi only flips. Every level below the mixed operator runs it
 * before and after its coarse correction. */
export const UNIFORM_MIXED_ROOT_SMOOTHING: readonly number[] = [1, 1, 1, 1, 1, 1, 1, 0.5, 0.5];
/** Iterations of a visit one tiled launch carries (its tile's halo is this
 * wide, and the kernel's workgroup arrays are sized by it). */
const ROOT_LAUNCH_ITERATIONS = 3;
/** Lanes of a tile's smoothing workgroup below the root lattice: a launch
 * there waits for a lane's chain of loads, not for its total work. The root
 * lattice's launches are bound by their work (and, in a closed slot, by the
 * threads they start), so its workgroups are a lane to an owned cell. */
const ROOT_TILE_LANES = 256;
/** The coarsest level's solve: this many (1, 1/2) weighted iteration pairs.
 * It is a correction's coarsest correction; more pairs do not buy cycles. */
const ROOT_COARSEST_PAIRS = 8;
/** Lanes of the one workgroup that runs the small levels: a lane to a row of
 * the largest of them. */
const ROOT_COARSE_LANES = 1024;
/** A level of more cells than this is run a tile to a workgroup; the rest
 * share the one-workgroup launch (a workgroup is one core's worth of the
 * GPU, so it only pays where a launch's fixed cost exceeds the level's work). */
const ROOT_SINGLE_MAX_CELLS = 1000;
/** Threads a capacity slot's tiled launch starts at most. A slot the plan
 * holds beyond the cycles it expects is closed in all but a few frames, and a
 * closed launch costs its fixed 5 microseconds plus the threads it starts
 * (each reads the gate and returns); a launch this narrow still completes
 * when the slot does open, since every workgroup strides the jobs. Narrower
 * buys nothing closed and costs the open slot its parallelism (pool 128^3,
 * a V slot: +0.16 ms at 16384, +1.5 at 4096, +6.5 at 1024). */
const ROOT_CAPACITY_THREADS = 16384;

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
  /** Static solids: the all-4h (open, V+) record. */
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

/** How a root-lattice smoothing launch stores its last iterate: in the other
 * native parity, or straight into the mixed pressure (added, assigned, or
 * added with the Full-Cycle backup restored). */
type RootEnd = 0 | 1 | 2 | 3;

/** CM11a cycles of the all-4h pressure root, whose correction the native
 * hierarchy solves from its n/4 level (entered through its haloed
 * (T+2)^3 fields). The mixed operator stays authoritative: every residual,
 * the measure and the correction bounds are the mixed rows (umPressureTheta's
 * ghost-fluid surface rule, wall slots as unknowns), which the native
 * n/4 operator does not reproduce; the native V is the correction's solver.
 *
 * The native setup (topology pyramid, baked coefficients, the cycle list)
 * is the continuation's; the traversal is this class's own, in few launches
 * because these levels are small and a launch costs more than its work on
 * most of them. A level of more than ROOT_SINGLE_MAX_CELLS cells is tiled: a
 * smoothing visit is ROOT_LAUNCH_ITERATIONS Jacobi iterations to a launch
 * (each tile's workgroup loads the tile and a halo as wide as its iteration
 * count into workgroup memory and relaxes a region that shrinks a cell an
 * iteration, so the stored values are those of one launch per iteration),
 * and its descent and ascent are a launch each. The levels below share one
 * launch of one workgroup (their visits, descents, the coarsest solve and
 * ascents, with a storage barrier wherever a pass boundary was); a visit
 * there is a lane to a row, which keeps the row's coefficients and passes
 * its iterates through workgroup memory, so only the visit's last iterate
 * is stored. The last launch of a traversal adds its iterate to the mixed
 * pressure itself.
 *
 * A tiled launch's jobs are its level's tiles (the root lattice's listed
 * ones), and workgroup w of a launch of W runs jobs w, w+W, ...: the width
 * is the host's parallelism estimate (uniformBufferedWork of the list count
 * two frames old, see observeWork), never a bound on the work. A capacity
 * slot launches narrower still (ROOT_CAPACITY_THREADS).
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
  /** A launch: its fixed workgroup grid, or (tiled) its level's tile count,
   * its workgroup's lanes and whether its jobs are the cycle list's. */
  private readonly pipelines = new Map<string, { pipeline: GPUComputePipeline; groups?: readonly number[]; tiles: number; lanes: number; listed: boolean }>();
  /** Levels run a tile to a workgroup (the leading ones); the rest are the
   * one-workgroup launch's. */
  private readonly tiled: number;
  /** The native levels, root lattice first, as words of the native range. */
  private readonly levels: { d: readonly number[]; p: number[]; b: number[]; m: number[]; c: number }[];
  /** A visit's launches: the iterations each carries. */
  private readonly visit: { first: number; count: number }[];
  private readonly tileLanes: number;
  private boundPass?: GPUComputePassEncoder;
  private readonly lanes: number;
  /** The largest one-workgroup level's row count. */
  private readonly rows: number;
  private readonly workgroups: [number, number, number];
  /** The continuation's cycle list: the root lattice's kernels walk its tiles. */
  private readonly listed?: { list: GPUBuffer };
  /** The listed launches' width (see observeWork); the lattice without one. */
  private work?: number;
  private readonly words: Record<"p" | "b0" | "b1" | "m0" | "m1" | "backup" | "residual" | "np" | "nb" | "nmin" | "nphi" | "nv" | "np1", number>;
  private readonly solid: boolean;
  constructor(private readonly device: GPUDevice, readonly level: UniformMixedPressureCycleLevel, backup: GPUBufferBinding,
    private readonly native: Continuation, private readonly uniformGroup: GPUBindGroup, private readonly openTop: boolean) {
    const layout = level.ownership.layout, t = layout.lattice.dimensions.map(n => n / 4);
    if (layout.tiles.some(word => mixedCellWidth(word) !== 4) || layout.tiles.length !== t[0]! * t[1]! * t[2]!)
      throw new Error("Mixed pressure root requires the uniform all-4h layout");
    const count = uniformMixedPressureStorage(layout).count;
    const nativeFields = [native.pressure, native.rhs, native.minimum, native.phi, native.topology,
      ...(native.cycle ? [native.cycle.pressure] : [])];
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
    if (native.levels.length < 2) throw new Error("Mixed pressure root requires a native level below its lattice");
    const levelFields = native.levels.flatMap(l => [...l.pressure, ...l.rhs, ...l.minimum, l.coefficients]);
    if (levelFields.some(f => f.buffer?.buffer !== arena)) throw new Error("Mixed pressure root requires the shared native scratch arena");
    const nativeRange = range([...nativeFields, ...levelFields].map(f => ({ offset: f.buffer!.offset ?? 0, size: f.buffer!.size! })));
    const nativeWord = (f: { buffer?: GPUBufferBinding }) => ((f.buffer!.offset ?? 0) - nativeRange.offset) / 4;
    this.levels = native.levels.map(l => ({ d: l.dimensions, p: l.pressure.map(nativeWord), b: l.rhs.map(nativeWord),
      m: l.minimum.map(nativeWord), c: nativeWord(l.coefficients) }));
    this.tileLanes = Math.min(ROOT_TILE_LANES, device.limits.maxComputeInvocationsPerWorkgroup);
    const smoothing = UNIFORM_MIXED_ROOT_SMOOTHING, M = ROOT_LAUNCH_ITERATIONS;
    if (!smoothing.length || smoothing.some(w => !(w > 0 && w <= 1))) throw new Error("Mixed pressure root smoothing weights must lie in (0, 1]");
    this.visit = Array.from({ length: Math.ceil(smoothing.length / M) }, (_, k) => ({
      first: k * M, count: Math.min(M, smoothing.length - k * M) }));
    this.lanes = Math.min(ROOT_COARSE_LANES, device.limits.maxComputeInvocationsPerWorkgroup);
    // A one-workgroup level is a lane to a row (see umLSmooth).
    const cells = this.levels.map(l => l.d[0]! * l.d[1]! * l.d[2]!);
    const single = cells.findIndex(n => n <= Math.min(ROOT_SINGLE_MAX_CELLS, this.lanes));
    this.tiled = Math.max(1, single < 0 ? this.levels.length - 1 : single);
    this.rows = Math.max(...cells.slice(this.tiled));
    if (this.rows > this.lanes) throw new Error(`Mixed pressure root: a ${this.rows}-row level does not fit one workgroup of ${this.lanes} lanes`);
    if (mixedRange.offset < nativeRange.offset + nativeRange.size && nativeRange.offset < mixedRange.offset + mixedRange.size)
      throw new Error("Mixed pressure root and native continuation fields overlap");
    const [p, b0, b1, m0, m1, backupWord, residual] = mixed.map(f => ((f.offset ?? 0) - mixedRange.offset) / 4) as number[];
    const [np, nb, nmin, nphi, nv, np1] = nativeFields.map(f => ((f.buffer!.offset ?? 0) - nativeRange.offset) / 4) as number[];
    this.words = { p: p!, b0: b0!, b1: b1!, m0: m0!, m1: m1!, backup: backupWord!, residual: residual!, np: np!, nb: nb!, nmin: nmin!, nphi: nphi!, nv: nv!,
      np1: np1 ?? 0 };
    const phi: GPUBufferBinding = { buffer: level.phi.buffer, offset: level.phi.offset ?? 0, size: 4 * layout.tiles.length };
    // The static solid record, read in place from its own binding.
    const topology = level.topology?.buffer;
    if (topology && (topology.size ?? topology.buffer.size - (topology.offset ?? 0)) < 16 * count)
      throw new Error("Mixed pressure root topology is smaller than its owners");
    this.solid = !!topology;
    this.workgroups = t.map(n => Math.ceil((n + 2) / 4)) as [number, number, number];
    if (native.cycle) {
      const tiles = this.workgroups.reduce((n, d) => n * d, 1);
      if (native.cycle.tiles !== tiles) throw new Error("Mixed pressure cycle list does not cover the native continuation lattice");
      this.listed = { list: native.cycle.list };
    }
    this.resources = device.createBindGroupLayout({ entries: [0, 1, 2, ...(this.listed ? [3] : []), ...(topology ? [4] : [])].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding >= 3 ? "read-only-storage" as const : "storage" as const } })) });
    this.group = device.createBindGroup({ layout: this.resources, entries: [
      { binding: 0, resource: { buffer: arena, ...mixedRange } }, { binding: 1, resource: { buffer: arena, ...nativeRange } }, { binding: 2, resource: phi },
      ...(this.listed ? [{ binding: 3, resource: { buffer: this.listed.list } }] : []),
      ...(topology ? [{ binding: 4, resource: topology }] : [])] });
  }
  async initialize(): Promise<void> {
    const layout = this.level.ownership.layout, h = layout.lattice.cellSize_m, t = layout.lattice.dimensions.map(n => n / 4), w = this.words;
    const solid = this.solid, open = this.openTop, listed = this.listed !== undefined;
    const levels = this.levels, count = levels.length, weights = UNIFORM_MIXED_ROOT_SMOOTHING, M = ROOT_LAUNCH_ITERATIONS;
    const vec = (v: readonly number[]) => `vec${v.length}u(${v.map(n => `${n}u`).join(",")})`;
    const code = /* wgsl */ `
@group(0) @binding(2) var<storage,read_write> umSupport:array<u32>;
@group(1) @binding(0) var<storage,read_write> mixed:array<f32>;
@group(1) @binding(1) var<storage,read_write> native:array<f32>;
@group(1) @binding(2) var<storage,read_write> phi:array<f32>;
const UM_T=vec3u(${t.map(n => `${n}u`).join(",")});const UM_N=UM_T+vec3u(2u);const UM_CELLS=${layout.tiles.length}u;
const UM_H=vec3f(${h.map(n => n.toFixed(8)).join(",")});const UM_MIN_H=min(UM_H.x,min(UM_H.y,UM_H.z));
const UM_OPEN_TOP=${open};
const UM_P=${w.p}u;const UM_B0=${w.b0}u;const UM_B1=${w.b1}u;const UM_M0=${w.m0}u;const UM_M1=${w.m1}u;const UM_BACKUP=${w.backup}u;const UM_RES=${w.residual}u;
const MG_P=${w.np}u;const MG_B=${w.nb}u;const MG_MIN=${w.nmin}u;const MG_PHI=${w.nphi}u;const MG_V=${w.nv}u;
${listed ? `const MG_P1=${w.np1}u;
@group(1) @binding(3) var<storage,read> umList:array<u32>;
` : ""}
// A tiled launch's jobs: level l's 4^3 tiles${listed ? " (the root lattice's listed ones)" : ""}.
// Workgroup w of a launch of W runs jobs w, w+W, ..., so the width is the
// host's parallelism estimate and no job is ever beyond a launch.
fn umTiles(l:u32)->vec3u{let d=UM_LD;return (d[l]+vec3u(3u))/4u;}
fn umJobs(l:u32)->u32{${listed ? "if(l==0u){return umList[0];}" : ""}let d=umTiles(l);return d.x*d.y*d.z;}
fn umJobOrigin(l:u32,job:u32)->vec3i{
 var tile=job;${listed ? "if(l==0u){tile=umList[4u+job];}" : ""}
 let d=umTiles(l);return 4*vec3i(vec3u(tile%d.x,(tile/d.x)%d.y,tile/(d.x*d.y)));
}
fn umLaneCell(lane:u32)->vec3i{return vec3i(i32(lane%4u),i32((lane/4u)%4u),i32(lane/16u));}
// restrict: the row's RHS and bound, and whether it opens a Full-Cycle.
override UM_RHS:u32=UM_B0;override UM_MIN:u32=UM_M0;override UM_FULL:bool=false;
// The mixed pressure schedule's slot gate (support 9n+24, 0 open).
fn umSlotClosed()->bool{return umSupport[9u*UM_CELLS+24u]!=0u;}
fn umNative(p:vec3u)->u32{return p.x+UM_N.x*(p.y+UM_N.y*p.z);}
// The residency certificate the transfer copies into this ownership's support
// (uniformMixedResidencyWord): a flag per page of 4^3 owners.
const UM_PD=(UM_T+vec3u(3u))/4u;const UM_RESIDENCY=9u*UM_CELLS+28u;
fn umAbsent(c:vec3u)->bool{let q=c/4u;return umSupport[UM_RESIDENCY+4u+q.x+UM_PD.x*(q.y+UM_PD.y*q.z)]==0u;}
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
// umPressureLiquid of a 4h owner.
fn umLiquid(k:u32)->bool{return ${uniformMixedPressureLiquidWGSL("phi[k]", "4.0*UM_MIN_H")};}
${solid ? `@group(1) @binding(4) var<storage,read> umRecord:array<vec4f>;
fn umTopo(i:u32)->vec4f{return umRecord[i];}` : ""}
fn umSum6(v:array<f32,6>)->f32{return ((v[0]+v[1])+(v[4]+v[5]))+(v[2]+v[3]);}
// umPressureSurfaceTheta / umPressureTheta of two 4h owners.
fn umSurfaceTheta(liquidPhi:f32,airPhi:f32)->f32{return ${uniformMixedSurfaceThetaWGSL("liquidPhi", "airPhi")};}
fn umTheta(a:u32,b:u32)->f32{
 let la=umLiquid(a);if(la==umLiquid(b)){return 1.0;}
 var liquid=a;var air=b;if(!la){liquid=b;air=a;}
 return umSurfaceTheta(phi[liquid],phi[air]);
}
fn umOpen(axis:u32,side:u32)->bool{return UM_OPEN_TOP&&axis==1u&&side==1u;}
// umBoundaryCoefficient.
fn umWall(k:u32,s:u32,axis:u32,side:u32)->f32{
 if(!umLiquid(k)){return 0.0;}
 let distance=4.0*UM_H[axis];var fraction=0.5;var theta=1.0;
 if(umOpen(axis,side)){fraction=1.0;theta=umSurfaceTheta(phi[k],2.0*UM_MIN_H);}
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
 // Branch, not select: an air row must not assemble its stencil.
 if(cell.kind==1u){if(!umLiquid(cell.key)){return 0.0;}return mixed[rhs+cell.key]-umRow(cell.c,cell.key).y;}
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
 // An absent page's owners and their wall slots: the rows RHS assembly writes
 // for certified far air (V=0, air: zero RHS and pressure, the bound). RHS
 // assembly and projection stride the resident pages only, and these arena
 // words are shared scratch, rewritten by other stages every frame.
 if(cell.kind!=0u&&umAbsent(cell.c)){
  var bound=-3.402823e38;
  if(cell.kind==2u){bound=select(0.0,-3.402823e38,umOpen(cell.axis,cell.side));}
  ${solid ? "else if(umTopo(cell.key).x<=1e-5){bound=0.0;}" : ""}
  mixed[UM_P+cell.slot]=0.0;mixed[UM_B0+cell.slot]=0.0;mixed[UM_M0+cell.slot]=bound;
 }
 var distance:f32=${2 * Math.min(...h)};var topology=vec4f(0.0);
 if(cell.kind==1u){
  distance=phi[cell.key];${solid ? "topology=umTopo(cell.key);" : `topology=vec4f(1.0);
  for(var axis=0u;axis<3u;axis++){if(p[axis]==UM_N[axis]-2u){topology[axis+1u]=select(0.5,1.0,UM_OPEN_TOP&&axis==1u);}}`}
 }else if(cell.kind==2u){
  for(var axis=0u;axis<3u;axis++){if(p[axis]==0u){topology[axis+1u]=${solid ? "umTopo(cell.slot).x" : "0.5"};}}
  if(UM_OPEN_TOP&&p.y==UM_N.y-1u){topology.x=1.0;}
 }
 native[MG_PHI+at]=distance;for(var component=0u;component<4u;component++){native[MG_V+4u*at+component]=topology[component];}
${listed ? `// The far-field seeds of the listed cycle kernels: every value a dense
 // cycle leaves at a row outside the list that anything listed reads. Air
 // rows restrict a zero residual (air slots carry a zero RHS), take a zero
 // correction, and keep the bound restrictRoot would give them at p = 0
 // (-FLT_MAX absorbs any finite p where it is read); their measured residual
 // is zero. The seeded bound is also what the native list builder reads to
 // list every interior-bounded tile.
 var lower=0.0;if(cell.kind!=0u){lower=mixed[UM_M0+cell.slot];mixed[UM_RES+cell.slot]=0.0;}
 native[MG_P+at]=0.0;native[MG_P1+at]=0.0;native[MG_B+at]=0.0;native[MG_MIN+at]=lower;` : ""}
}
// The correction problem: its RHS is the mixed residual, its bound the row's
// bound less the pressure, its initial value zero. A Full-Cycle also keeps
// the pressure, the shifted bound and the residual for its closing V-cycle.
fn restrictAt(p:vec3u){
 let at=umNative(p);let cell=umCell(p);
 var b=0.0;var lower=0.0;
 if(cell.kind!=0u){
  let s=cell.slot;let own=umP(s);b=umResidual(cell,UM_RHS);lower=mixed[UM_MIN+s]-own;
  if(UM_FULL){mixed[UM_BACKUP+s]=own;mixed[UM_M1+s]=lower;mixed[UM_B1+s]=b;}
 }
 native[MG_P+at]=0.0;native[MG_B+at]=b;native[MG_MIN+at]=lower;
}
// The acceptance norm's rows, owners and wall slots.
fn measureAt(p:vec3u){
 let cell=umCell(p);let s=cell.slot;
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
${["restrict", "measure"].map(name => `
@compute @workgroup_size(64) fn ${name === "measure" ? name : `${name}Root`}(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSlotClosed()){return;}
 let count=umJobs(0u);
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  let p=vec3u(umJobOrigin(0u,job)+umLaneCell(lane));if(all(p<UM_N)){${name}At(p);}
 }
}`).join("")}

// ---- The correction's traversal. Level 0 is the root lattice (UM_N); the
// rest are the native levels below it. Each field is a word offset of native.
const UM_LAST=${count - 1}u;
const UM_LD=array<vec3u,${count}>(${levels.map(l => vec(l.d)).join(",")});
const UM_LP=array<vec2u,${count}>(${levels.map(l => vec(l.p)).join(",")});
const UM_LB=array<vec2u,${count}>(${levels.map(l => vec(l.b)).join(",")});
const UM_LM=array<vec2u,${count}>(${levels.map(l => vec(l.m)).join(",")});
const UM_LC=array<u32,${count}>(${levels.map(l => `${l.c}u`).join(",")});
// p, b, m: both parities of the pressure, RHS and bound (the Full-Cycle's
// own RHS and bound below the root lattice are the second); c: the baked
// (+x,+y,+z) coefficients and liquid mask, four words a cell.
struct UMLevel{d:vec3u,p:vec2u,b:vec2u,m:vec2u,c:u32}
fn umLevel(l:u32)->UMLevel{let d=UM_LD;let p=UM_LP;let b=UM_LB;let m=UM_LM;let c=UM_LC;return UMLevel(d[l],p[l],b[l],m[l],c[l]);}
const UM_WN=${weights.length}u;
fn umWeight(i:u32)->f32{let w=array<f32,${weights.length}>(${weights.map(w => w.toFixed(7)).join(",")});return w[i];}
fn umLAt(d:vec3u,q:vec3i)->u32{return u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));}
fn umLIn(d:vec3u,q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(d));}
fn umLCell(d:vec3u,j:u32)->vec3i{return vec3i(i32(j%d.x),i32((j/d.x)%d.y),i32(j/(d.x*d.y)));}
fn umStep(n:u32)->vec3i{let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));return e[n];}
// Bit 0 is the row's own liquid flag, bits 1..6 its -x,+x,-y,+y,-z,+z neighbours'.
fn umLMask(L:UMLevel,at:u32)->u32{return u32(native[L.c+4u*at+3u]);}
fn umSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
// A row's six face coefficients: its own +faces, its -neighbours' +faces.
fn umLFaces(L:UMLevel,q:vec3i,at:u32)->array<f32,6>{
 var a:array<f32,6>;
 for(var n=0u;n<6u;n++){
  let axis=n/2u;a[n]=0.0;
  if((n&1u)==1u){a[n]=native[L.c+4u*at+axis];}
  else{let r=q+umStep(n);if(umLIn(L.d,r)){a[n]=native[L.c+4u*umLAt(L.d,r)+axis];}}
 }
 return a;
}
// One weighted simultaneous update of a liquid row from its neighbours'
// iterate (only liquid neighbours enter), projected onto its bound.
fn umLRelax(L:UMLevel,q:vec3i,at:u32,mask:u32,old:f32,nb:array<f32,6>,rhs:u32,minimum:f32,weight:f32)->f32{
 let a=umLFaces(L,q,at);var s:array<f32,6>;
 for(var n=0u;n<6u;n++){s[n]=select(0.0,a[n]*nb[n],((mask>>(n+1u))&1u)!=0u);}
 let diagonal=umSum6(a);let sum=umSum6(s);
 let p=select(0.0,(sum+native[rhs+at])/diagonal,diagonal>0.0);
 return max(mix(old,p,weight),minimum);
}
// b - A p of a liquid row; an air row has no unknown and carries zero.
fn umLResidual(L:UMLevel,q:vec3i,par:u32,rhs:u32)->f32{
 let at=umLAt(L.d,q);let mask=umLMask(L,at);if((mask&1u)==0u){return 0.0;}
 let a=umLFaces(L,q,at);let centre=native[L.p[par]+at];var t:array<f32,6>;
 for(var n=0u;n<6u;n++){
  var other=0.0;if(((mask>>(n+1u))&1u)!=0u){other=native[L.p[par]+umLAt(L.d,q+umStep(n))];}
  t[n]=a[n]*(centre-other);
 }
 return native[rhs+at]-umSum6(t);
}
// A coarse cell's fine child; an axis that did not halve has both offsets on
// one cell, which the eight-tap folds then weight correctly.
fn umChild(fine:vec3u,coarse:vec3u,c:vec3i,o:vec3i)->vec3i{
 let fp=vec3i(fine)-vec3i(2);let cp=max(vec3i(coarse)-vec3i(2),vec3i(1));let stride=max(fp/cp,vec3i(1));
 return clamp(stride*(c-vec3i(1))+o*(stride-vec3i(1)),vec3i(-1),fp)+vec3i(1);
}
fn umCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}
// Cell-centred trilinear prolongation of the coarse pressure; taps in the
// coarse halo are outside the pressure grid and the rest renormalise.
fn umTrilinear(fine:vec3u,C:UMLevel,par:u32,id:vec3i)->f32{
 let fp=vec3f(vec3i(fine)-vec3i(2));let cp=vec3f(vec3i(C.d)-vec3i(2));let scale=cp/max(fp,vec3f(1.0));
 let q=(vec3f(id)-vec3f(0.5))*scale+vec3f(0.5);let base=vec3i(floor(q));let f=fract(q);
 var values:array<f32,8>;var weights:array<f32,8>;
 for(var corner=0u;corner<8u;corner++){
  let o=umCorner(corner);let p=base+o;values[corner]=0.0;weights[corner]=0.0;
  if(any(p<vec3i(1))||any(p>=vec3i(C.d)-vec3i(1))){continue;}
  let w=select(1.0-f.x,f.x,o.x==1)*select(1.0-f.y,f.y,o.y==1)*select(1.0-f.z,f.z,o.z==1);
  values[corner]=w*native[C.p[par]+umLAt(C.d,p)];weights[corner]=w;
 }
 let value=umSum8(values);let total=umSum8(weights);
 return select(0.0,value/total,total>0.0);
}

// ---- A tiled level's smoothing launch: UM_BCOUNT iterations of the visit
// (from its UM_BFIRST-th) on every tile of level UM_BL (the root lattice's
// listed tiles), a workgroup to a tile at a time. UM_BRHS: the level's
// V-cycle RHS and bound, or its Full-Cycle ones.
override UM_BL:u32=0u;override UM_BRHS:u32=0u;override UM_BSRC:u32=0u;override UM_BFIRST:u32=0u;override UM_BCOUNT:u32=${M}u;override UM_BEND:u32=0u;override UM_BLANES:u32=64u;
var<workgroup> umGo:u32;var<workgroup> umAny:u32;var<workgroup> umFlag:array<u32,64>;
// The tile and its halo, iterate by iterate: sides 4+2m, then 2+2m, then 2m.
var<workgroup> umBlockA:array<f32,${(4 + 2 * M) ** 3}>;var<workgroup> umBlockB:array<f32,${(2 + 2 * M) ** 3}>;
fn umBlock(k:i32,i:i32)->f32{if((k&1)==0){return umBlockA[i];}return umBlockB[i];}
fn umBlockEnd(q:vec3i,at:u32,value:f32,to:u32){
 if(UM_BEND==0u){native[to+at]=value;return;}
 let cell=umCell(vec3u(q));if(cell.kind==0u){return;}
 let s=cell.slot;
 if(UM_BEND==2u){mixed[UM_P+s]=value;}
 else if(UM_BEND==3u){mixed[UM_P+s]=(mixed[UM_P+s]+value)+mixed[UM_BACKUP+s];}
 else{mixed[UM_P+s]+=value;}
}
@compute @workgroup_size(UM_BLANES) fn smoothTiles(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 let R=umLevel(UM_BL);let rhs=R.b[UM_BRHS];let low=R.m[UM_BRHS];let src=R.p[UM_BSRC];let to=R.p[UM_BSRC^1u];
 // A closed slot has no jobs.
 if(lane==0u){umGo=select(umJobs(UM_BL),0u,umSlotClosed());}
 let count=workgroupUniformLoad(&umGo);
 // Jobs read one parity and store the other (or the mixed pressure), so a
 // workgroup's jobs are as independent as two workgroups'.
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  let origin=umJobOrigin(UM_BL,job);
  // The tile's own cells are the first 64 lanes'.
  let own=origin+umLaneCell(lane);let inside=lane<64u&&umLIn(R.d,own);
  var at=0u;var liquid=0u;if(inside){at=umLAt(R.d,own);liquid=umLMask(R,at)&1u;}
  if(lane<64u){umFlag[lane]=liquid;}workgroupBarrier();
  if(lane==0u){var any=0u;for(var i=0u;i<64u;i++){any|=umFlag[i];}umAny=any;}
  if(workgroupUniformLoad(&umAny)==0u){
   // A tile without liquid only projects: no row of it reads a neighbour.
   if(inside){umBlockEnd(own,at,max(native[src+at],native[low+at]),to);}
  }else{
   let m=i32(UM_BCOUNT);let s0=4+2*m;let b0=origin-vec3i(m);
   for(var j=i32(lane);j<s0*s0*s0;j+=i32(UM_BLANES)){
    let q=b0+vec3i(j%s0,(j/s0)%s0,j/(s0*s0));
    // A cell outside the lattice is never liquid and nothing reads it.
    var v=0.0;if(umLIn(R.d,q)){v=native[src+umLAt(R.d,q)];}
    umBlockA[j]=v;
   }
   workgroupBarrier();
   for(var k=1;k<=m;k++){
    let sp=6+2*(m-k);let sn=sp-2;let bn=origin-vec3i(m-k);let weight=umWeight(UM_BFIRST+u32(k-1));let last=k==m;
    for(var j=i32(lane);j<sn*sn*sn;j+=i32(UM_BLANES)){
     let o=vec3i(j%sn,(j/sn)%sn,j/(sn*sn));let q=bn+o;let jp=(o.x+1)+sp*((o.y+1)+sp*(o.z+1));
     let old=umBlock(k-1,jp);var next=old;
     if(umLIn(R.d,q)){
      let qa=umLAt(R.d,q);let minimum=native[low+qa];let mask=umLMask(R,qa);next=max(old,minimum);
      if((mask&1u)!=0u){
       let nb=array<f32,6>(umBlock(k-1,jp-1),umBlock(k-1,jp+1),umBlock(k-1,jp-sp),umBlock(k-1,jp+sp),umBlock(k-1,jp-sp*sp),umBlock(k-1,jp+sp*sp));
       next=umLRelax(R,q,qa,mask,old,nb,rhs,minimum,weight);
      }
      if(last){umBlockEnd(q,qa,next,to);}
     }
     if(!last){if((k&1)==1){umBlockB[j]=next;}else{umBlockA[j]=next;}}
    }
    workgroupBarrier();
   }
  }
 }
}

// ---- The small levels (UM_SINGLE and below), in one workgroup. Each helper
// is a former pass, ended by a storage barrier.
const UM_SINGLE=${this.tiled}u;const UM_ROWS=${this.rows}u;
override UM_LANES:u32=256u;override UM_CRHS:u32=0u;override UM_CFULL:bool=false;
// A visit (or, for the coarsest level, its solve) from parity start; the
// parity it ends on. A lane owns one row: it reads the row's coefficients,
// RHS and bound once, the iterates pass between lanes through workgroup
// memory, and only the visit's last is stored.
var<workgroup> umIterate:array<f32,${2 * this.rows}>;
fn umLSmooth(l:u32,start:u32,rhs:u32,low:u32,solve:bool,lane:u32)->u32{
 let L=umLevel(l);let own=lane<L.d.x*L.d.y*L.d.z;
 let iterations=select(UM_WN,${2 * ROOT_COARSEST_PAIRS}u,solve);
 var old=0.0;var minimum=0.0;var b=0.0;var diagonal=0.0;var mask=0u;
 var a:array<f32,6>;var at:array<u32,6>;
 if(own){
  old=native[L.p[start]+lane];minimum=native[low+lane];mask=umLMask(L,lane);
  if((mask&1u)!=0u){
   let q=umLCell(L.d,lane);a=umLFaces(L,q,lane);diagonal=umSum6(a);b=native[rhs+lane];
   for(var n=0u;n<6u;n++){at[n]=lane;if(((mask>>(n+1u))&1u)!=0u){at[n]=umLAt(L.d,q+umStep(n));}}
  }
  umIterate[lane]=old;
 }
 workgroupBarrier();
 var cur=0u;
 for(var i=0u;i<iterations;i++){
  var weight=select(1.0,0.5,(i&1u)==1u);if(!solve){weight=umWeight(i);}
  if(own){
   var next=max(old,minimum);
   if((mask&1u)!=0u){
    var s:array<f32,6>;
    for(var n=0u;n<6u;n++){s[n]=select(0.0,a[n]*umIterate[cur*UM_ROWS+at[n]],((mask>>(n+1u))&1u)!=0u);}
    let p=select(0.0,(umSum6(s)+b)/diagonal,diagonal>0.0);
    next=max(mix(old,p,weight),minimum);
   }
   umIterate[(cur^1u)*UM_ROWS+lane]=next;old=next;
  }
  workgroupBarrier();cur^=1u;
 }
 let end=start^(iterations&1u);
 if(own){native[L.p[end]+lane]=old;}
 storageBarrier();
 return end;
}
// A V-cycle's descent from level l: each child's residual restricted into
// the next level's RHS (the average of the eight; air rows carry zero), its
// correction cleared, and its bound the largest of the children's bounds
// less their pressure (bound-active residuals must not enter an
// unconstrained coarse solve).
fn umLTransitionAt(l:u32,par:u32,rhs:u32,low:u32,j:u32){
 let F=umLevel(l);let C=umLevel(l+1u);
 let c=umLCell(C.d,j);var terms:array<f32,8>;var lower=-3.402823e38;
 for(var corner=0u;corner<8u;corner++){
  let q=umChild(F.d,C.d,c,umCorner(corner));let at=umLAt(F.d,q);
  terms[corner]=umLResidual(F,q,par,rhs);lower=max(lower,native[low+at]-native[F.p[par]+at]);
 }
 native[C.b.x+j]=umSum8(terms)/8.0;native[C.p.x+j]=0.0;native[C.m.x+j]=lower;
}
fn umLTransition(l:u32,par:u32,rhs:u32,low:u32,lane:u32){
 let C=umLevel(l+1u);let cells=C.d.x*C.d.y*C.d.z;
 for(var j=lane;j<cells;j+=UM_LANES){umLTransitionAt(l,par,rhs,low,j);}
 storageBarrier();
}
// A Full-Cycle's descent from level l: its RHS and bound are corrections
// already, restricted unmasked into the next level's Full-Cycle fields.
fn umLRestrictAt(l:u32,rhs:u32,low:u32,j:u32){
 let F=umLevel(l);let C=umLevel(l+1u);
 let c=umLCell(C.d,j);var terms:array<f32,8>;var lower=-3.402823e38;
 for(var corner=0u;corner<8u;corner++){
  let at=umLAt(F.d,umChild(F.d,C.d,c,umCorner(corner)));
  terms[corner]=native[rhs+at];lower=max(lower,native[low+at]);
 }
 native[C.b.y+j]=umSum8(terms)/8.0;native[C.m.y+j]=lower;
}
fn umLRestrict(l:u32,rhs:u32,low:u32,lane:u32){
 let C=umLevel(l+1u);let cells=C.d.x*C.d.y*C.d.z;
 for(var j=lane;j<cells;j+=UM_LANES){umLRestrictAt(l,rhs,low,j);}
 storageBarrier();
}
// Either descent out of a tiled level UM_TL, a workgroup to a tile of the
// level below at a time: a V-cycle's from the iterate in parity UM_TSRC, or a
// Full-Cycle's (UM_TFULL). UM_TRHS: the level's V-cycle RHS and bound, or its
// Full-Cycle ones.
override UM_TL:u32=0u;override UM_TSRC:u32=0u;override UM_TRHS:u32=0u;override UM_TFULL:bool=false;
@compute @workgroup_size(64) fn descendTiles(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSlotClosed()){return;}
 let F=umLevel(UM_TL);let C=umLevel(UM_TL+1u);let rhs=F.b[UM_TRHS];let low=F.m[UM_TRHS];
 let count=umJobs(UM_TL+1u);
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  let c=umJobOrigin(UM_TL+1u,job)+umLaneCell(lane);if(!umLIn(C.d,c)){continue;}
  let j=umLAt(C.d,c);
  if(UM_TFULL){umLRestrictAt(UM_TL,rhs,low,j);}else{umLTransitionAt(UM_TL,UM_TSRC,rhs,low,j);}
 }
}
// Level l+1's pressure (parity coarse) onto level l: added to parity src, or
// assigned; stored in parity to.
fn umLProlongAt(l:u32,src:u32,to:u32,coarse:u32,add:bool,q:vec3i){
 let F=umLevel(l);let C=umLevel(l+1u);let at=umLAt(F.d,q);
 var v=umTrilinear(F.d,C,coarse,q);if(add){v=native[F.p[src]+at]+v;}
 native[F.p[to]+at]=v;
}
fn umLProlong(l:u32,src:u32,to:u32,coarse:u32,add:bool,lane:u32){
 let F=umLevel(l);let cells=F.d.x*F.d.y*F.d.z;
 for(var j=lane;j<cells;j+=UM_LANES){umLProlongAt(l,src,to,coarse,add,umLCell(F.d,j));}
 storageBarrier();
}
// The same onto a tiled level UM_PL, a workgroup to a tile at a time (the
// root lattice's listed tiles: its far field keeps the zero correction its
// seeds hold).
override UM_PL:u32=0u;override UM_PSRC:u32=0u;override UM_PTO:u32=0u;override UM_PCOARSE:u32=0u;override UM_PADD:bool=true;
@compute @workgroup_size(64) fn ascendTiles(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSlotClosed()){return;}
 let F=umLevel(UM_PL);let count=umJobs(UM_PL);
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  let q=umJobOrigin(UM_PL,job)+umLaneCell(lane);
  if(umLIn(F.d,q)){umLProlongAt(UM_PL,UM_PSRC,UM_PTO,UM_PCOARSE,UM_PADD,q);}
 }
}
// A V-cycle of level a and everything below it, on (rhs, low) at level a
// and each level's V-cycle fields below; the parity level a's result is in.
fn umLV(a:u32,rhsA:u32,lowA:u32,start:u32,lane:u32)->u32{
 var par:array<u32,${count}>;par[a]=start;
 for(var l=a;l<UM_LAST;l++){
  let L=umLevel(l);let rhs=select(L.b.x,rhsA,l==a);let low=select(L.m.x,lowA,l==a);
  par[l]=umLSmooth(l,par[l],rhs,low,false,lane);
  umLTransition(l,par[l],rhs,low,lane);par[l+1u]=0u;
 }
 {
  let L=umLevel(UM_LAST);let rhs=select(L.b.x,rhsA,UM_LAST==a);let low=select(L.m.x,lowA,UM_LAST==a);
  par[UM_LAST]=umLSmooth(UM_LAST,par[UM_LAST],rhs,low,true,lane);
 }
 for(var i=UM_LAST;i>a;i--){
  let l=i-1u;let L=umLevel(l);let rhs=select(L.b.x,rhsA,l==a);let low=select(L.m.x,lowA,l==a);
  umLProlong(l,par[l],par[l]^1u,par[i],true,lane);par[l]^=1u;
  par[l]=umLSmooth(l,par[l],rhs,low,false,lane);
 }
 return par[a];
}
// The small levels' part of a traversal: level UM_SINGLE's V-cycle from the
// zero correction a descent left in its parity 0, on its V-cycle RHS and
// bound or (UM_CRHS) its Full-Cycle ones. UM_CFULL: a Full-Cycle's opening
// instead (Algorithm 3 on these levels): the level's Full-Cycle RHS and bound
// restricted to the coarsest, solved there, and each level up assigned the
// prolonged solution and given a V-cycle. The result's parity is static
// (singleParity).
@compute @workgroup_size(UM_LANES) fn coarse(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){umGo=select(1u,0u,umSlotClosed());}
 if(workgroupUniformLoad(&umGo)==0u){return;}
 if(UM_CFULL){
  for(var l=UM_SINGLE;l<UM_LAST;l++){let L=umLevel(l);umLRestrict(l,L.b.y,L.m.y,lane);}
  let B=umLevel(UM_LAST);let cells=B.d.x*B.d.y*B.d.z;
  for(var j=lane;j<cells;j+=UM_LANES){native[B.p.x+j]=0.0;}
  storageBarrier();
  var top=umLSmooth(UM_LAST,0u,B.b.y,B.m.y,true,lane);
  for(var i=UM_LAST;i>UM_SINGLE;i--){
   let l=i-1u;let L=umLevel(l);
   umLProlong(l,0u,0u,top,false,lane);
   top=umLV(l,L.b.y,L.m.y,0u,lane);
  }
 }else{
  let L=umLevel(UM_SINGLE);let top=umLV(UM_SINGLE,L.b[UM_CRHS],L.m[UM_CRHS],0u,lane);
 }
}
`;
    const module = this.device.createShaderModule({ label: "Uniform mixed pressure root", code });
    const info = await module.getCompilationInfo(), errors = info.messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.level.ownership.bindLayout, this.resources] });
    // A fixed workgroup grid, or a tiled launch's level and lanes.
    type Variant = [string, Record<string, number>, readonly number[] | { level: number; lanes: number }];
    const root = { level: 0, lanes: 64 };
    const variants: Record<string, Variant> = {
      setup: ["setup", {}, this.workgroups], measure: ["measure", {}, root],
      restrictV: ["restrictRoot", {}, root], restrictInner: ["restrictRoot", { UM_RHS: w.b1, UM_MIN: w.m1 }, root], restrictFull: ["restrictRoot", { UM_FULL: 1 }, root],
    };
    // Every launch the traversals issue (see cycle() and opening()).
    const add = (entry: string, constants: Record<string, number>) => {
      const level = constants.UM_BL ?? constants.UM_PL ?? 0;
      variants[entry] ??= entry.startsWith("coarse") ? ["coarse", { UM_LANES: this.lanes, ...constants }, [1]]
        : entry.startsWith("descend") ? ["descendTiles", constants, { level: constants.UM_TL! + 1, lanes: 64 }]
        : entry.startsWith("ascend") ? ["ascendTiles", constants, { level, lanes: 64 }]
        : ["smoothTiles", { UM_BLANES: level ? this.tileLanes : 64, ...constants }, { level, lanes: level ? this.tileLanes : 64 }];
    };
    for (const end of [1, 2, 3] as const) this.cycle(0, 0, 0, end, add);
    this.opening(add);
    await Promise.all(Object.entries(variants).map(async ([entry, [entryPoint, constants, shape]]) => {
      const pipeline = await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint, constants } });
      this.pipelines.set(entry, "level" in shape
        ? { pipeline, tiles: levels[shape.level]!.d.reduce((n, d) => n * Math.ceil(d / 4), 1), lanes: shape.lanes, listed: listed && shape.level === 0 }
        : { pipeline, groups: shape, tiles: 0, lanes: 0, listed: false });
    }));
  }
  /** The parity the one-workgroup launch leaves level `tiled`'s result in
   * from parity 0: the coarsest solve is an even count of iterations; a
   * V-cycle is two visits and the prolongation's store. */
  private get singleParity(): number {
    return this.tiled === this.levels.length - 1 ? 0 : 1;
  }
  /** A V-cycle of level `l` (a tiled one, or the one-workgroup launch's
   * first) from the iterate in parity `start`, on the level's V-cycle RHS and
   * bound or (`set` 1) its Full-Cycle ones: a visit, the level below, a
   * visit. The root lattice's last launch stores into the mixed pressure
   * (`end`). Returns the result's parity. */
  private cycle(l: number, set: number, start: number, end: RootEnd, launch: (entry: string, constants: Record<string, number>) => void): number {
    if (l === this.tiled) {
      if (start !== 0) throw new Error("The one-workgroup V-cycle starts from parity 0");
      launch(`coarse.${set}`, { UM_CRHS: set }); return this.singleParity;
    }
    let parity = start;
    const visit = (last: RootEnd) => this.visit.forEach((v, k) => {
      const to = k === this.visit.length - 1 ? last : 0;
      launch(`smooth.${l}.${set}.${parity}.${v.first}.${v.count}.${to}`, { UM_BL: l, UM_BRHS: set, UM_BSRC: parity, UM_BFIRST: v.first, UM_BCOUNT: v.count, UM_BEND: to });
      parity ^= 1;
    });
    visit(0);
    launch(`descend.${l}.${set}.${parity}`, { UM_TL: l, UM_TRHS: set, UM_TSRC: parity });
    const below = this.cycle(l + 1, 0, 0, 0, launch);
    launch(`ascend.${l}.${parity}.${below}.add`, { UM_PL: l, UM_PSRC: parity, UM_PTO: parity ^ 1, UM_PCOARSE: below, UM_PADD: 1 }); parity ^= 1;
    visit(l === 0 ? end : 0);
    return parity;
  }
  /** A Full-Cycle's opening below the root lattice (Algorithm 3): the root
   * RHS and bound restricted level by level, the coarsest solved, and each
   * level up assigned the prolonged solution and given a V-cycle; the root
   * lattice's parity 0 is assigned the result. */
  private opening(launch: (entry: string, constants: Record<string, number>) => void): void {
    for (let l = 0; l < this.tiled; l++) launch(`descend.${l}.full`, { UM_TL: l, UM_TRHS: l ? 1 : 0, UM_TFULL: 1 });
    launch("coarse.full", { UM_CFULL: 1 });
    let top = this.singleParity;
    for (let l = this.tiled - 1; l >= 0; l--) {
      launch(`ascend.${l}.${top}.assign`, { UM_PL: l, UM_PSRC: 0, UM_PTO: 0, UM_PCOARSE: top, UM_PADD: 0 });
      if (l > 0) top = this.cycle(l, 1, 0, 0, launch);
    }
  }
  /** capacity: a launch of a slot the plan holds in reserve (see
   * ROOT_CAPACITY_THREADS). */
  private dispatch(passes: UniformMixedPressurePasses, entry: string, capacity = false): void {
    const launch = this.pipelines.get(entry); if (!launch) throw new Error("Mixed pressure root is not initialized");
    const pass = passes.pass;
    pass.setPipeline(launch.pipeline);
    // Both groups are the same for every launch of the root: once a pass.
    if (this.boundPass !== pass) { this.boundPass = pass; pass.setBindGroup(0, this.level.ownership.bindGroup); pass.setBindGroup(1, this.group); }
    if (launch.groups) { pass.dispatchWorkgroups(...(launch.groups as [number, number?, number?])); return; }
    let width = launch.listed ? Math.min(launch.tiles, this.work ?? launch.tiles) : launch.tiles;
    if (capacity) width = Math.min(width, Math.max(1, Math.floor(ROOT_CAPACITY_THREADS / launch.lanes)));
    const x = Math.min(width, this.device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(x, Math.ceil(width / x));
  }
  /** The listed launches' width from a completed frame's list count
   * (uniformBufferedWork: parallelism only, every kernel strides the live
   * list). */
  observeWork(listed: number): void {
    const tiles = this.workgroups.reduce((n, d) => n * d, 1);
    this.work = uniformBufferedWork(this.work ?? tiles, listed, tiles);
  }
  /** The cycle list's count (this solve's) into a frame receipt: one word,
   * zero without a list. */
  encodeWorkReceipt(encoder: GPUCommandEncoder, target: GPUBuffer, offset: number): void {
    if (this.listed) encoder.copyBufferToBuffer(this.listed.list, 0, target, offset, 4);
    else encoder.clearBuffer(target, offset, 4);
  }
  private batch(encoder: GPUCommandEncoder, label: string, body: (passes: UniformMixedPressurePasses) => void): void {
    const passes = new UniformMixedPressurePasses(encoder, `Uniform mixed pressure ${label}`);
    body(passes); passes.end();
  }
  /** Per solve, after RHS assembly and before the initial measure: the native
   * n/4 geometry and the continuation's setup. Ungated. */
  encodeSetup(encoder: GPUCommandEncoder): void {
    // One pass with the native setup; only its tile-list clears end it.
    this.batch(encoder, "setup", passes => { this.dispatch(passes, "setup"); this.boundPass = undefined; this.native.encodeSetup(encoder, this.uniformGroup, passes); });
  }
  /** capacity (here and on the cycles): the slot is one the plan holds in
   * reserve, launched narrow. */
  encodeMeasure(encoder: GPUCommandEncoder, capacity = false): void { this.batch(encoder, "measure", passes => this.dispatch(passes, "measure", capacity)); }
  encodeVCycle(encoder: GPUCommandEncoder, capacity = false): void {
    this.batch(encoder, "V-cycle", passes => { this.dispatch(passes, "restrictV", capacity); this.cycle(0, 0, 0, 1, entry => this.dispatch(passes, entry, capacity)); });
  }
  /** Algorithm 3 at the root: the Full-Cycle on the residual under the
   * shifted bound, a closing V-cycle, then the saved pressure added back. */
  encodeFullCycle(encoder: GPUCommandEncoder, capacity = false): void {
    this.batch(encoder, "Full-cycle", passes => {
      const launch = (entry: string) => this.dispatch(passes, entry, capacity);
      launch("restrictFull"); this.opening(launch); this.cycle(0, 0, 0, 2, launch);
      launch("restrictInner"); this.cycle(0, 0, 0, 3, launch);
    });
  }
}
