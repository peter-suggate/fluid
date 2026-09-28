import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {UNIFORM_MIXED_THETA_MIN} from "./uniform-mixed-pressure-surface.wgsl";
import {uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

/** The band's local multigrid: V-cycles h -> 2h (2^3 aggregates per tile) ->
 * 4h (one aggregate per tile), red-black Gauss-Seidel at every level. */
export interface UniformPressureBandSchedule {
 cycles:number;
 /** Red-black h sweeps before and after each coarse correction. */
 fineSweeps:number;
 /** Red-black 2h sweeps before and after the 4h correction. */
 middleSweeps:number;
 /** Red-black sweeps of the per-tile 4h level. */
 coarseSweeps:number;
 /** Each aggregate level's inter-aggregate couplings (and the diagonal share
  * they carry) are scaled by this: piecewise-constant aggregation is twice too
  * stiff for smooth modes (0.5 would recover the rediscretised interior
  * operator). Dirichlet terms stay exact -- scaling them over-corrects small
  * Dirichlet-dominated blobs and stalls the cycle. 0.6 from replaying dumped
  * band systems: a wide shallow pool (hero garden) and a thick long-dam slab
  * converge ~0.2 per cycle at 0.6 but only ~0.5 at 0.5 (0.4 stalls); the
  * cm12 figure-12 surface band is ~0.1 at 0.5 and ~0.2 at 0.6. */
 coarseScale:number;
}
export const UNIFORM_PRESSURE_BAND_SCHEDULE:UniformPressureBandSchedule={cycles:4,fineSweeps:3,middleSweeps:2,coarseSweeps:16,coarseScale:0.6};

/** Workgroups of the tile classification passes: a fixed grid-stride launch
 * over the simulation's h tile worklist. */
const LIST_GROUPS=512;
/** Every band pass strides the band's own compact slot list from a fixed
 * direct launch: one slot per group, eight slots (2h aggregates) per group,
 * or 64 slots (4h aggregates) per group, up to these group counts. Groups
 * past the live count exit at once; an indirect launch would cost Dawn a
 * validation pass per dispatch, more than those empty groups. */
const SLOT_GROUPS=1024,MIDDLE_GROUPS=256,COARSE_GROUPS=64;
/** Lanes of the single-workgroup 4h aggregate solve. */
const COARSE_SOLVE_LANES=256;
/** Band rows are field-major over CAP*64 rows: rhs, diagonal, face kinds,
 * the six face coefficients, u* per face, then with static solids the CM11a
 * V per face. */
const ROW_FIELDS=15,SOLID_ROW_FIELDS=21;
/** 2h aggregates: correction, residual, diagonal, six couplings. 4h
 * aggregates add their six neighbour slots (+1) and their red-black colour. */
const MIDDLE_FIELDS=9,COARSE_FIELDS=16;
/** Index header words: count, overflow, final residual, fatal, completed
 * cycles, three spare, then the residual before each cycle (and after the
 * last) in HISTORY words. The tile list and per-tile slot+1 map follow. */
const HEADER=24,HISTORY_WORD=8,HISTORY=16;

export interface UniformPressureBandFields {
 /** Simulation-indexed pressure phi (the simulation authority's output),
  * read before the split authority rewrites the same view in pressure
  * indexing; afterwards, the coarse pressure phi. */
 phi:GPUBufferBinding;
 correction:GPUTexture;
 vertexPhi:GPUTexture;
 /** The forced field u* in simulation ownership (before the split). */
 forced:{velocity:GPUTexture;negative:GPUBuffer};
 /** The 4h-projected field transferred to simulation ownership, and the
  * texture its copy lands in for the band projection's reads. */
 velocity:GPUTexture;negative:GPUBuffer;copy:GPUTexture;
 /** Accepted 4h pressure, indexed by the all-4h pressure ownership. */
 coarsePressure:GPUBufferBinding;
 /** h.xyz, dt; density, openTop, dt/density, min h. */
 params:GPUBuffer;
 /** The stage grids' band section: the live band pressures land at this
  * word of the buffer (uniform-stage-grids). */
 presentation:{buffer:GPUBuffer;word:number};
}

/** The h pressure band of the two-stage solve: the global solve runs on
 * all-4h ownership, then every h simulation tile holding a liquid pressure row
 * re-solves pressure at h. A tile the simulation holds at h is h in the bulk
 * too: its velocity is only 4h-divergence-free until this solve, and without
 * it V drifts per cell. Its inner boundary is Neumann with the projected 4h
 * face velocity (exactly conservative: a 4h face's flux is the sum of its h
 * faces'), air is ghost-fluid Dirichlet, walls keep the native one-cell halo.
 * The start is the liquid-weighted trilinear 4h pressure; V-cycles of a local
 * multigrid (h, 2h and 4h aggregates of the band's own tiles, homogeneous
 * Neumann wherever the h rows are) converge it however large the band is.
 * No readback on the advance path: the GPU builds a compact tile list and
 * every solver pass is an indirect launch over it.
 * With static solids every face carries the native CM11a dual-cell V (the
 * coefficient V/(h^2 theta), the divergence V u; V=0 faces are closed), rows
 * inside a solid are p_min=0 rows, and positive faces take the native
 * embedded contact release. Every cut tile with a liquid row is a band tile,
 * so every Neumann face is uncut (V=1): the receipt's fatal word (header 3)
 * reports a Neumann face with V<1 (2) or a cut tile the simulation holds
 * coarse (1), both certificate violations the frame throws on. */
export class UniformPressureBand {
 readonly allocatedBytes:number;
 /** Header (see HEADER), then the tile list and a per-tile slot+1 map. */
 readonly index:GPUBuffer;
 private readonly rows:GPUBuffer;
 private readonly coarse:GPUBuffer;
 /** Two words per tile: which of an h tile's air cells are open air. */
 private readonly open:GPUBuffer;
 private readonly solve:GPUBuffer;
 /** Band tiles one frame can hold: every tile the simulation can hold at h.
  * More is a fatal receipt, never a fallback. */
 readonly capacity:number;
 private readonly layouts=new Map<string,GPUBindGroupLayout>();
 private readonly groups=new Map<string,GPUBindGroup>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** The h pressure iterate: 64 cells per band slot, then the wall halo. */
 get iterate():GPUBuffer{return this.solve;}
 /** Byte offset in index of the per-tile slot+1 map. */
 get slotMapOffset():number{return 4*(HEADER+this.capacity);}
 private readonly rowFields:number;
 constructor(private readonly device:GPUDevice,private readonly simulation:UniformMixedOwnership,private readonly pressure:UniformMixedOwnership,private readonly fields:UniformPressureBandFields,
  readonly schedule:UniformPressureBandSchedule=UNIFORM_PRESSURE_BAND_SCHEDULE,
  /** Static solids, with the all-4h record (its cut-tile flags). */
  private readonly solid?:UniformMixedSolid){
  const layout=simulation.layout,d=layout.lattice.dimensions,tiles=layout.tiles.length;
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("The pressure band's global stage must be all-4h");
  if(solid&&!solid.coarse)throw new Error("The solid pressure band needs the all-4h solid record");
  if(schedule.cycles<1||schedule.cycles>=HISTORY)throw new Error(`Pressure band cycles must be 1..${HISTORY-1}`);
  // The simulation ownership reserves every tile at h (its capacity is fine).
  this.capacity=tiles;
  this.rowFields=solid?SOLID_ROW_FIELDS:ROW_FIELDS;
  const rows=this.capacity*64,halo=2*(d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!);
  const limit=device.limits.maxStorageBufferBindingSize;
  if(4*this.rowFields*rows>limit)throw new Error(`Pressure band rows (${4*this.rowFields*rows} bytes) exceed the storage binding limit ${limit}`);
  const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  this.index=device.createBuffer({label:"Uniform pressure band tiles",size:(HEADER+this.capacity+tiles)*4,usage:storage});
  this.rows=device.createBuffer({label:"Uniform pressure band rows",size:4*this.rowFields*rows,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  this.coarse=device.createBuffer({label:"Uniform pressure band aggregates",size:4*(8*MIDDLE_FIELDS+COARSE_FIELDS)*this.capacity,usage:GPUBufferUsage.STORAGE});
  this.solve=device.createBuffer({label:"Uniform pressure band iterate",size:(rows+halo)*4,usage:storage});
  this.open=device.createBuffer({label:"Uniform pressure band open air",size:tiles*8,usage:storage});
  this.allocatedBytes=this.index.size+this.rows.size+this.coarse.size+this.solve.size+this.open.size;
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const buffer=(binding:number,type:GPUBufferBindingType="storage")=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const params=buffer(0,"uniform");
  // At most six storage buffers here: topology and solids hold four.
  const solver=[params,buffer(2),buffer(9),buffer(10),buffer(18)];
  const entries:Record<string,GPUBindGroupLayoutEntry[]>={
   open:[params,buffer(3,"read-only-storage"),buffer(5),texture(6)],
   list:[params,buffer(2),buffer(3,"read-only-storage"),buffer(5,"read-only-storage"),texture(6)],
   copy:[params,buffer(2),texture(22),{binding:23,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}],
   prep:[params,buffer(2),buffer(3,"read-only-storage"),texture(4),buffer(5,"read-only-storage"),texture(6),texture(7),buffer(8,"read-only-storage"),buffer(9)],
   init:[params,buffer(2),buffer(9),buffer(10),texture(11),buffer(12,"read-only-storage"),buffer(13,"read-only-storage"),buffer(14,"read-only-storage")],
   solver,
   project:[...solver,texture(15),{binding:16,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},buffer(17)],
   present:[params,buffer(2),buffer(10),buffer(21)],
  };
  for(const [name,list] of Object.entries(entries))this.layouts.set(name,device.createBindGroupLayout({label:`Uniform pressure band ${name}`,entries:list}));
  const f=fields,cells=layout.cellCount,faces=d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!;
  const scalars=(view:GPUBufferBinding,count:number):GPUBufferBinding=>{
   const offset=view.offset??0;if((view.size??view.buffer.size-offset)<4*count)throw new Error("Pressure band field view is too small");return {buffer:view.buffer,offset,size:4*count};
  };
  const coarse=pressure.layout.cellCount;
  const P={0:{buffer:f.params,size:32}},solverResources={...P,2:{buffer:this.index},9:{buffer:this.rows},10:{buffer:this.solve},18:{buffer:this.coarse}};
  const resources:Record<string,Record<number,GPUBindingResource>>={
   open:{...P,3:scalars(f.phi,cells),5:{buffer:this.open},6:f.vertexPhi.createView()},
   list:{...P,2:{buffer:this.index},3:scalars(f.phi,cells),5:{buffer:this.open},6:f.vertexPhi.createView()},
   copy:{...P,2:{buffer:this.index},22:f.velocity.createView(),23:f.copy.createView()},
   prep:{...P,2:{buffer:this.index},3:scalars(f.phi,cells),4:f.correction.createView(),5:{buffer:this.open},6:f.vertexPhi.createView(),7:f.forced.velocity.createView(),8:{buffer:f.forced.negative,size:4*faces},9:{buffer:this.rows}},
   init:{...P,2:{buffer:this.index},9:{buffer:this.rows},10:{buffer:this.solve},11:f.velocity.createView(),12:scalars(f.coarsePressure,coarse),13:scalars(f.phi,coarse),14:pressure.presentation},
   solver:solverResources,
   project:{...solverResources,15:f.copy.createView(),16:f.velocity.createView(),17:{buffer:f.negative,size:4*faces}},
   present:{...P,2:{buffer:this.index},10:{buffer:this.solve},21:{buffer:f.presentation.buffer}},
  };
  if(f.presentation.buffer.size<4*(f.presentation.word+64*this.capacity))throw new Error("The stage grids cannot hold the pressure band");
  for(const [name,map] of Object.entries(resources))this.groups.set(name,device.createBindGroup({label:`Uniform pressure band ${name}`,layout:this.layouts.get(name)!,
   entries:Object.entries(map).map(([binding,resource])=>({binding:+binding,resource}))}));
 }
 async initialize():Promise<void>{
  const layout=this.simulation.layout,S=!!this.solid,schedule=this.schedule;
  const header=uniformMixedTopologyWGSL(layout,0)+uniformMixedFaceAddressWGSL+uniformMixedSolidWGSL(S?2:undefined,this.solid?.coarse?.count)+/* wgsl */`
struct BandParams {hDt:vec4f,policy:vec4f}
const CAP:u32=${this.capacity}u;const N:u32=CAP*64u;const M:u32=CAP*8u;
const LIST:u32=${HEADER}u;const SLOTS:u32=${HEADER}u+CAP;const HISTORY:u32=${HISTORY_WORD}u;const CYCLE:u32=4u;
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;
fn bLocal(lane:u32)->vec3u{return vec3u(lane%4u,(lane/4u)%4u,lane/16u);}
fn bMiddleLocal(a:u32)->vec3u{return vec3u(a&1u,(a>>1u)&1u,a>>2u);}
fn bFaceKind(kinds:u32,f:u32)->u32{return (kinds>>(3u*f))&7u;}
fn bLiquid(kinds:u32)->bool{return (kinds&0x80000000u)!=0u;}
// A row inside a solid: the CM11a p_min=0 row.
fn bInside(kinds:u32)->bool{return (kinds&0x40000000u)!=0u;}
fn bSign(f:u32)->i32{return select(-1,1,(f&1u)==1u);}
// The native one-cell wall halo at h, after every band cell.
fn bHalo(p:vec3i,axis:u32,sign:i32)->u32{
 let q=vec3u(p);let side=select(0u,1u,sign>0);var slot=0u;
 if(axis==0u){slot=side*UM_D.y*UM_D.z+q.y+UM_D.y*q.z;}
 else if(axis==1u){slot=2u*UM_D.y*UM_D.z+side*UM_D.x*UM_D.z+q.x+UM_D.x*q.z;}
 else{slot=2u*(UM_D.y*UM_D.z+UM_D.x*UM_D.z)+side*UM_D.x*UM_D.y+q.x+UM_D.x*q.y;}
 return N+slot;
}
`;
  const indexed=(atomic:boolean)=>/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(2) var<storage,read_write> index:array<${atomic?"atomic<u32>":"u32"}>;
fn bIndex(i:u32)->u32{return ${atomic?"atomicLoad(&index[i])":"index[i]"};}
fn bCount()->u32{return min(bIndex(0u),CAP);}
fn bTile(s:u32)->vec3u{return umTileCoord(bIndex(LIST+s));}
// Slot + 1 of the band tile at tile coordinate c, 0 outside the band.
fn bSlotAt(c:vec3i)->u32{if(any(c<vec3i(0))||any(c>=vec3i(UM_T))){return 0u;}return bIndex(SLOTS+umTileAt(vec3u(c)));}
// Band cell index + 1 at an h position, 0 outside the band.
fn bCellAt(q:vec3i)->u32{
 if(any(q<vec3i(0))||any(q>=vec3i(UM_D))){return 0u;}
 let u=vec3u(q);let slot=bIndex(SLOTS+umTileAt(u/4u));if(slot==0u){return 0u;}
 let l=u%4u;return (slot-1u)*64u+l.x+4u*(l.y+4u*l.z)+1u;
}
`;
  const theta=/* wgsl */`
// umPressureSurfaceTheta at h spacing.
fn bTheta(liquidPhi:f32,airPhi:f32)->f32{
 let depth=max(abs(liquidPhi),${UNIFORM_MIXED_THETA_MIN}*params.policy.w);return clamp(depth/(depth+abs(airPhi)),${UNIFORM_MIXED_THETA_MIN},1.0);
}
fn bPairTheta(a:f32,b:f32)->f32{if((a<0.0)==(b<0.0)){return 1.0;}if(a<0.0){return bTheta(a,b);}return bTheta(b,a);}
`;
  // The band's free surface is the 4h surface refined within one 4h owner:
  // h air is Dirichlet where its 4h pressure owner (the tile) is air, or
  // where it reaches such air without leaving its own owner. Other h air --
  // pockets and wall slivers the 4h solve counted as liquid -- stays liquid
  // here too: the 4h fluxes on the band's Neumann edge were solved with it
  // liquid, and a p=0 pocket inside them floats the whole tile's pressure
  // level to zero (walls release, liquid peels off them).
  const owner=/* wgsl */`
@group(1) @binding(3) var<storage,read> phi:array<f32>;
@group(1) @binding(6) var vertexPhi:texture_3d<f32>;
fn bCentrePhi(q:vec3i)->f32{
 var s=0.0;for(var k=0u;k<8u;k++){s+=textureLoad(vertexPhi,q+vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u)),0).x;}return 0.125*s;
}
// The all-4h pressure phase of tile t: its owner's centre phi, the mean of
// the eight corner vertices (UniformMixedSurfaceGeometry).
fn bOwnerAir(t:u32)->bool{
 let origin=vec3i(umTileCoord(t)*4u);var s=0.0;
 for(var k=0u;k<8u;k++){s+=textureLoad(vertexPhi,origin+4*vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u)),0).x;}return s>=0.0;
}
// The simulation's h tile worklist.
fn bHTile(j:u32)->u32{return umTopology[UM_TILES+j];}
`;
  const surface=owner+/* wgsl */`
@group(1) @binding(5) var<storage,read> openAir:array<u32>;
// Open air: Dirichlet at h. Only meaningful for air cells of h tiles.
fn bOpen(q:vec3i)->bool{
 let t=umTileAt(vec3u(q)/4u);let l=vec3u(q)%4u;let bit=l.x+4u*(l.y+4u*l.z);
 return (openAir[2u*t+bit/32u]&(1u<<(bit%32u)))!=0u;
}
// An h cell's band pressure phi: its own, or a liquid marker for closed air.
// With solids only an open cell can be a pocket: closed cells keep the
// authority's continuation (liquid beside open liquid, else air).
fn bPhiH(q:vec3i)->f32{let own=phi[umOwnerAt(q).index];return select(own,-0.5*params.policy.w,own>=0.0&&!bOpen(q)${S?"&&umCellOpen(q)>1e-5":""});}
// Face classification of row assembly. An h neighbour's band phi decides;
// inside a coarse owner, a coarse liquid owner is Neumann (its flux is the
// transferred 4h face), and a coarse air owner is Neumann too where the h
// cell is itself liquid (the 4h solve owns its flux), else air at the h
// cell's own centre phi.
struct BNeighbour {phi:f32,neumann:bool}
fn bNeighbour(q:vec3i)->BNeighbour{
 let n=umOwnerAt(q);let other=phi[n.index];
 if(n.width==1u){return BNeighbour(bPhiH(q),false);}
 if(other<0.0){return BNeighbour(other,true);}
 let own=bCentrePhi(q);return BNeighbour(own,own<0.0);
}
`;
  const rows=/* wgsl */`
@group(1) @binding(9) var<storage,read_write> rows:array<f32>;
fn bRhs(c:u32)->f32{return rows[c];}
fn bDiagonal(c:u32)->f32{return rows[N+c];}
fn bKinds(c:u32)->u32{return bitcast<u32>(rows[2u*N+c]);}
fn bCoefficient(c:u32,f:u32)->f32{return rows[(3u+f)*N+c];}
fn bForced(c:u32,f:u32)->f32{return rows[(9u+f)*N+c];}
${S?"fn bVolume(c:u32,f:u32)->f32{return rows[(15u+f)*N+c];}":""}
`;
  const solve=/* wgsl */`
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
// Off-diagonal sum of an h row: band neighbours and the wall halo.
fn bOff(cell:u32,p:vec3i,kinds:u32)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){
  let axis=f/2u;let sign=bSign(f);let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){var q=p;q[axis]+=sign;off+=bCoefficient(cell,f)*solve[bCellAt(q)-1u];}
  else if(kind==K_WALL){off+=bCoefficient(cell,f)*solve[bHalo(p,axis,sign)];}
 }
 return off;
}
// The wall halo follows its row: the native Neumann ghost, clamped at 0.
fn bFollowHalo(cell:u32,p:vec3i,kinds:u32,value:f32){
 for(var f=0u;f<6u;f++){
  if(bFaceKind(kinds,f)!=K_WALL){continue;}
  let axis=f/2u;let sign=bSign(f);
  solve[bHalo(p,axis,sign)]=max(value+f32(sign)*params.policy.x*bForced(cell,f)*params.hDt[axis]/params.hDt.w,0.0);
 }
}
// A liquid row's residual; a p_min=0 row at its bound only counts a positive one.
fn bResidual(cell:u32,p:vec3i,kinds:u32)->f32{
 let r=bRhs(cell)+bOff(cell,p,kinds)-bDiagonal(cell)*solve[cell];
 return select(r,max(r,0.0),bInside(kinds)&&solve[cell]<=0.0);
}
`;
  const aggregates=/* wgsl */`
@group(1) @binding(18) var<storage,read_write> coarse:array<f32>;
// 2h aggregate a (0..8) of slot s is cell 8s+a; field k at k*M. The per-tile
// 4h aggregate of slot s: field k at 9M+k*CAP+s (9..14 neighbour slot+1, 15 colour).
fn bM(k:u32,c:u32)->u32{return k*M+c;}
fn bC(k:u32,s:u32)->u32{return 9u*M+k*CAP+s;}
// The 2h aggregate + 1 across face f of aggregate c, 0 outside the band.
fn bMiddleNeighbour(c:u32,f:u32)->u32{
 let s=c/8u;let axis=f/2u;var g=vec3i(bTile(s)*2u+bMiddleLocal(c%8u));g[axis]+=bSign(f);
 if(g[axis]<0||g[axis]>=2*i32(UM_T[axis])){return 0u;}
 let slot=bSlotAt(g/2);if(slot==0u){return 0u;}
 let l=vec3u(g%2);return (slot-1u)*8u+l.x+2u*l.y+4u*l.z+1u;
}
fn bCoarseNeighbour(s:u32,f:u32)->u32{let axis=f/2u;var c=vec3i(bTile(s));c[axis]+=bSign(f);return bSlotAt(c);}
fn bMiddleOff(c:u32)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){let n=bMiddleNeighbour(c,f);if(n!=0u){off+=coarse[bM(3u+f,c)]*coarse[bM(0u,n-1u)];}}
 return off;
}
`;
  const slots=/* wgsl */`@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) groups:vec3u`;
  const band=header+indexed(false)+rows+solve+aggregates;
  const sources:Record<string,string>={
   open:header+owner+/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(5) var<storage,read_write> openAir:array<u32>;
var<workgroup> state:array<u32,64>;
var<workgroup> changed:u32;
// Open air per h tile: seeded across the tile's faces from air in 4h-air
// owners (or the open top), then filled through the tile's own air cells.
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var j=group.x;j<umCounts.x;j+=${LIST_GROUPS}u){
  let t=bHTile(j);
  let p=vec3i(umTileCoord(t)*4u+bLocal(lane));let l=vec3i(bLocal(lane));
  let ownerAir=bOwnerAir(t);
  let air=phi[umOwnerAt(p).index]>=0.0;var open=air&&ownerAir;
  if(air&&!ownerAir){
   for(var f=0u;f<6u;f++){
    let axis=f/2u;var q=p;q[axis]+=bSign(f);
    if(q[axis]<0||q[axis]>=i32(UM_D[axis])){if(axis==1u&&f==3u&&params.policy.y>0.5){open=true;}continue;}
    let tq=umTileAt(vec3u(q)/4u);if(tq==t||!bOwnerAir(tq)){continue;}
    let n=umOwnerAt(q);
    if(select(phi[n.index]>=0.0&&bCentrePhi(q)>=0.0,phi[n.index]>=0.0,n.width==1u)){open=true;}
   }
  }
  state[lane]=select(select(0u,1u,air),2u,open);
  if(lane==0u){changed=1u;}
  workgroupBarrier();
  // 0 liquid, 1 closed air, 2 open air; closed air next to open air opens.
  for(var round=0u;round<64u&&workgroupUniformLoad(&changed)!=0u;round++){
   if(lane==0u){changed=0u;}
   workgroupBarrier();
   if(state[lane]==1u){
    for(var f=0u;f<6u;f++){
     let axis=f/2u;var m=l;m[axis]+=bSign(f);if(m[axis]<0||m[axis]>3){continue;}
     if(state[u32(m.x+4*(m.y+4*m.z))]==2u){state[lane]=2u;changed=1u;break;}
    }
   }
   workgroupBarrier();
  }
  if(lane<2u){var word=0u;for(var b=0u;b<32u;b++){if(state[32u*lane+b]==2u){word|=1u<<b;}}openAir[2u*t+lane]=word;}
  workgroupBarrier();
 }
}`,
   list:header+surface+/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(2) var<storage,read_write> index:array<atomic<u32>>;
var<workgroup> member:atomic<u32>;
// Every h simulation tile with a liquid pressure row is a band tile: its
// h velocity is divergence-free only after the h solve.
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var j=group.x;j<umCounts.x;j+=${LIST_GROUPS}u){
  let t=bHTile(j);
  if(lane==0u){atomicStore(&member,0u);}
  workgroupBarrier();
  if(bPhiH(vec3i(umTileCoord(t)*4u+bLocal(lane)))<0.0){atomicStore(&member,1u);}
  if(workgroupUniformLoad(&member)!=0u&&lane==0u){
   let slot=atomicAdd(&index[0],1u);
   if(slot<CAP){atomicStore(&index[LIST+slot],t);atomicStore(&index[SLOTS+t],slot+1u);}else{atomicStore(&index[1],1u);}
  }
 }${S?`
 // A cut tile the simulation holds at 4h breaks the uncut-Neumann certificate.
 for(var j=group.x*64u+lane;j<umCounts.z;j+=${LIST_GROUPS*64}u){if(umSolidCut(umTopology[UM_TILES+umCounts.x+umCounts.y+j])){atomicOr(&index[3],1u);}}`:""}
}`,
   prep:header+indexed(false)+theta+surface+/* wgsl */`
@group(1) @binding(9) var<storage,read_write> rows:array<f32>;
@group(1) @binding(4) var correction:texture_3d<f32>;
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(8) var<storage,read> negative:array<f32>;
fn bForcedField(p:vec3i,axis:u32,sign:i32)->f32{
 var a=p;if(sign<0){a[axis]-=1;}
 if(a[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(p),axis)];}
 return textureLoad(velocity,a,0)[axis];
}
// Row assembly, baked once per solve.
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+lane;
  let own=bPhiH(p);let liquid=own<0.0;
  var kinds=select(0u,0x80000000u,liquid)${S?"|select(0u,0x40000000u,umCellInsideSolid(p))":""};var diagonal=0.0;var divergence=0.0;
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];var q=p;q[axis]+=sign;
   let forced=bForcedField(p,axis,sign);var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
   ${S?"var low=p;if(sign<0){low[axis]-=1;}let volume=umPressureFaceV(low,axis);":""}
   if(q[axis]<0||q[axis]>=i32(UM_D[axis])){
    if(axis==1u&&sign>0&&params.policy.y>0.5){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }else{
    let n=bNeighbour(q);
    if(bCellAt(q)!=0u){kind=K_BAND;coefficient=1.0/(h*h*bPairTheta(own,n.phi));}
    else if(n.neumann||n.phi<0.0){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*bPairTheta(own,n.phi));}
   }
   ${S?`// CM11a: V scales the coefficient and the flux; the Neumann edge is uncut.
   if(kind==K_NEUMANN){if(volume<0.99999){index[3]=2u;}}
   else if(volume<=1e-6){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
   rows[(15u+f)*N+cell]=volume;`:""}
   kinds|=kind<<(3u*f);rows[(9u+f)*N+cell]=forced;rows[(3u+f)*N+cell]=coefficient;
   if(kind!=K_NEUMANN){diagonal+=coefficient;divergence+=f32(sign)*fraction*forced/h;}
  }
  rows[cell]=select(0.0,-params.policy.x*(divergence-textureLoad(correction,p,0).x)/params.hDt.w,liquid);
  rows[N+cell]=diagonal;rows[2u*N+cell]=bitcast<f32>(kinds);
 }
}`,
   // Galerkin aggregation (piecewise-constant prolongation over liquid rows,
   // restriction its transpose): faces inside an aggregate cancel, wall
   // halos follow their row (no correction flux), Neumann faces carry none.
   middleBake:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 for(var c=group.x*64u+lane;c<bCount()*8u;c+=groups.x*64u){
  let s=c/8u;let a=bMiddleLocal(c%8u);let origin=vec3i(bTile(s)*4u);
  var diagonal=0.0;var couple=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var k=0u;k<8u;k++){
   let l=a*2u+bMiddleLocal(k);let cell=s*64u+l.x+4u*(l.y+4u*l.z);let kinds=bKinds(cell);
   if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);let coefficient=bCoefficient(cell,f);
    if(kind==K_GHOST||kind==K_OPEN){diagonal+=coefficient;continue;}
    if(kind!=K_BAND){continue;}
    let axis=f/2u;var m=vec3i(l);m[axis]+=bSign(f);
    // A face to a band row outside the liquid rows is Dirichlet (p=0 there),
    // inside the aggregate too; between liquid rows of one aggregate it
    // cancels.
    let n=bCellAt(origin+m)-1u;
    if(!bLiquid(bKinds(n))||bDiagonal(n)<=0.0){diagonal+=coefficient;continue;}
    if(all(m>=vec3i(0))&&all(m<vec3i(4))&&all(vec3u(m)/2u==a)){continue;}
    couple[f]+=coefficient;
   }
  }
  // Dirichlet terms stay exact; only couplings between aggregates scale.
  let scale=${schedule.coarseScale};var coupled=0.0;
  for(var f=0u;f<6u;f++){coarse[bM(3u+f,c)]=scale*couple[f];coupled+=scale*couple[f];}
  coarse[bM(2u,c)]=diagonal+coupled;
 }
}`,
   coarseBake:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x*64u+lane;s<bCount();s+=groups.x*64u){
  var diagonal=0.0;var couple=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var a=0u;a<8u;a++){
   let c=8u*s+a;let at=bMiddleLocal(a);diagonal+=coarse[bM(2u,c)];
   for(var f=0u;f<6u;f++){
    let axis=f/2u;let inside=select(at[axis]==1u,at[axis]==0u,bSign(f)>0);
    let w=coarse[bM(3u+f,c)];diagonal-=w;if(!inside){couple[f]+=w;}
   }
  }
  // diagonal is now the aggregate's Dirichlet part.
  let scale=${schedule.coarseScale};var coupled=0.0;
  for(var f=0u;f<6u;f++){coarse[bC(3u+f,s)]=scale*couple[f];coupled+=scale*couple[f];coarse[bC(9u+f,s)]=bitcast<f32>(bCoarseNeighbour(s,f));}
  coarse[bC(2u,s)]=diagonal+coupled;let t=bTile(s);coarse[bC(15u,s)]=bitcast<f32>((t.x+t.y+t.z)&1u);
 }
}`,
   init:header+indexed(false)+rows+solve+/* wgsl */`
@group(1) @binding(11) var velocity:texture_3d<f32>;
@group(1) @binding(12) var<storage,read> coarsePressure:array<f32>;
@group(1) @binding(13) var<storage,read> coarsePhi:array<f32>;
@group(1) @binding(14) var<storage,read> coarseTopology:array<u32>;
fn bCoarse(c:vec3i)->vec2f{
 let i=coarseTopology[umTileAt(vec3u(clamp(c,vec3i(0),vec3i(UM_T)-vec3i(1))))]&0x3fffffffu;
 return select(vec2f(0.0),vec2f(coarsePressure[i],1.0),coarsePhi[i]<0.0);
}
@compute @workgroup_size(64) fn main(${slots}){
 if(group.x==0u&&lane==0u){index[CYCLE]=0u;}
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+lane;
  let kinds=bKinds(cell);
  if(!bLiquid(kinds)){solve[cell]=0.0;continue;}
  // Liquid-weighted trilinear 4h pressure at the h centre.
  let y=(vec3f(p)+0.5)/4.0-0.5;let b=vec3i(floor(y));let w=y-floor(y);var sum=vec2f(0.0);
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let weights=select(1.0-w,w,bit==vec3u(1u));
   let c=bCoarse(b+vec3i(bit));sum+=weights.x*weights.y*weights.z*vec2f(c.x*c.y,c.y);
  }
  var start=select(0.0,sum.x/sum.y,sum.y>0.0);if(bInside(kinds)){start=max(start,0.0);}solve[cell]=start;
  // Neumann faces carry the projected 4h flux into the row.
  var rhs=bRhs(cell);
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);
   if(bFaceKind(kinds,f)==K_NEUMANN){var a=p;if(sign<0){a[axis]-=1;}rhs-=params.policy.x*f32(sign)*textureLoad(velocity,a,0)[axis]/(params.hDt[axis]*params.hDt.w);}
  }
  rows[cell]=rhs;bFollowHalo(cell,p,kinds,start);
 }
}`,
   sweep:band+/* wgsl */`
override bColour:u32=0u;
// One red-black Gauss-Seidel half sweep at h; the wall halos follow their row.
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+lane;
  if((u32(p.x+p.y+p.z)&1u)!=bColour){continue;}
  let kinds=bKinds(cell);let diagonal=bDiagonal(cell);
  if(!bLiquid(kinds)||diagonal<=0.0){continue;}
  var next=(bRhs(cell)+bOff(cell,p,kinds))/diagonal;if(bInside(kinds)){next=max(next,0.0);}solve[cell]=next;
  bFollowHalo(cell,p,kinds,next);
 }
}`,
   // h residual into the 2h aggregates (zeroing their correction), and its
   // largest liquid row into the cycle's history word.
   restrict:header+indexed(true)+rows+solve+aggregates+/* wgsl */`
var<workgroup> worst:atomic<u32>;
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){atomicStore(&worst,0u);}
 workgroupBarrier();
 var largest=0.0;
 for(var c=group.x*64u+lane;c<bCount()*8u;c+=groups.x*64u){
  let s=c/8u;let a=bMiddleLocal(c%8u);let origin=vec3i(bTile(s)*4u);var sum=0.0;
  for(var k=0u;k<8u;k++){
   let l=a*2u+bMiddleLocal(k);let cell=s*64u+l.x+4u*(l.y+4u*l.z);let kinds=bKinds(cell);
   if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
   let r=bResidual(cell,origin+vec3i(l),kinds);sum+=r;largest=max(largest,abs(r));
  }
  coarse[bM(1u,c)]=sum;coarse[bM(0u,c)]=0.0;
 }
 atomicMax(&worst,bitcast<u32>(largest*params.policy.z));
 workgroupBarrier();
 if(lane==0u){atomicMax(&index[HISTORY+min(atomicLoad(&index[CYCLE]),${HISTORY-1}u)],atomicLoad(&worst));}
}`,
   middleSweep:band+/* wgsl */`
override bColour:u32=0u;
@compute @workgroup_size(64) fn main(${slots}){
 for(var c=group.x*64u+lane;c<bCount()*8u;c+=groups.x*64u){
  let g=bTile(c/8u)*2u+bMiddleLocal(c%8u);if(((g.x+g.y+g.z)&1u)!=bColour){continue;}
  let diagonal=coarse[bM(2u,c)];if(diagonal<=0.0){continue;}
  coarse[bM(0u,c)]=(coarse[bM(1u,c)]+bMiddleOff(c))/diagonal;
 }
}`,
   // 2h residual into the tile aggregate (zeroing its correction); the
   // first lane counts the cycle.
   middleRestrict:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 if(group.x==0u&&lane==0u){index[CYCLE]+=1u;}
 for(var s=group.x*64u+lane;s<bCount();s+=groups.x*64u){
  var sum=0.0;
  for(var a=0u;a<8u;a++){let c=8u*s+a;if(coarse[bM(2u,c)]>0.0){sum+=coarse[bM(1u,c)]+bMiddleOff(c)-coarse[bM(2u,c)]*coarse[bM(0u,c)];}}
  coarse[bC(1u,s)]=sum;coarse[bC(0u,s)]=0.0;
 }
}`,
   // Every red-black sweep of the 4h aggregates in one workgroup: one
   // launch instead of two per sweep.
   coarseSolve:band+/* wgsl */`
@compute @workgroup_size(${COARSE_SOLVE_LANES}) fn main(@builtin(local_invocation_index) lane:u32){
 let n=bCount();
 for(var sweep=0u;sweep<${schedule.coarseSweeps}u;sweep++){
  for(var colour=0u;colour<2u;colour++){
   for(var s=lane;s<n;s+=${COARSE_SOLVE_LANES}u){
    if(bitcast<u32>(coarse[bC(15u,s)])!=colour){continue;}
    let diagonal=coarse[bC(2u,s)];if(diagonal<=0.0){continue;}
    var off=0.0;for(var f=0u;f<6u;f++){let m=bitcast<u32>(coarse[bC(9u+f,s)]);if(m!=0u){off+=coarse[bC(3u+f,s)]*coarse[bC(0u,m-1u)];}}
    coarse[bC(0u,s)]=(coarse[bC(1u,s)]+off)/diagonal;
   }
   storageBarrier();
  }
 }
}`,
   coarseProlong:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 for(var c=group.x*64u+lane;c<bCount()*8u;c+=groups.x*64u){if(coarse[bM(2u,c)]>0.0){coarse[bM(0u,c)]+=coarse[bC(0u,c/8u)];}}
}`,
   middleProlong:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let l=bLocal(lane);let p=vec3i(bTile(s)*4u+l);let cell=s*64u+lane;let kinds=bKinds(cell);
  if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
  let a=l/2u;var next=solve[cell]+coarse[bM(0u,8u*s+a.x+2u*a.y+4u*a.z)];if(bInside(kinds)){next=max(next,0.0);}
  solve[cell]=next;bFollowHalo(cell,p,kinds,next);
 }
}`,
   measure:header+indexed(true)+rows+solve+/* wgsl */`
var<workgroup> worst:atomic<u32>;
// The largest band row residual, in divergence units (1/s), over liquid rows.
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){atomicStore(&worst,0u);}
 workgroupBarrier();
 var largest=0.0;
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+lane;
  let kinds=bKinds(cell);if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
  largest=max(largest,abs(bResidual(cell,p,kinds))*params.policy.z);
 }
 atomicMax(&worst,bitcast<u32>(largest));
 workgroupBarrier();
 if(lane==0u){let r=atomicLoad(&worst);atomicMax(&index[2],r);atomicMax(&index[HISTORY+min(atomicLoad(&index[CYCLE]),${HISTORY-1}u)],r);}
}`,
   project:band+/* wgsl */`
@group(1) @binding(15) var field:texture_3d<f32>;
@group(1) @binding(16) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(17) var<storage,read_write> boundary:array<f32>;
// umProject on one face of band cell 'cell' at p; 'kept' is the transferred
// 4h value, which a Neumann face keeps.
fn bFace(cell:u32,p:vec3i,f:u32,kept:f32)->f32{
 let kinds=bKinds(cell);let kind=bFaceKind(kinds,f);
 if(kind==K_NEUMANN){return kept;}
 ${S?"if(kind==K_CLOSED){return 0.0;}":""}
 let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];let liquid=bLiquid(kinds);
 // The gradient coefficient: the row coefficient without its V.
 let forced=bForced(cell,f);let coefficient=bCoefficient(cell,f)${S?"/bVolume(cell,f)":""};let own=solve[cell];let scale=params.policy.z;
 if(kind==K_WALL){if(!liquid){return forced;}return forced-scale*f32(sign)*(solve[bHalo(p,axis,sign)]-own)/h;}
 if(kind==K_OPEN){if(!liquid){return forced;}return forced+scale*f32(sign)*own*coefficient*h;}
 var other=0.0;var otherLiquid=false;
 if(kind==K_BAND){var q=p;q[axis]+=sign;let n=bCellAt(q)-1u;other=solve[n];otherLiquid=bLiquid(bKinds(n));}
 if(!liquid&&!otherLiquid){return 0.0;}
 return forced-scale*f32(sign)*(other-own)*coefficient*h;
}
fn bReleased(cell:u32,p:vec3i,f:u32,value:f32)->bool{
 let axis=f/2u;let sign=bSign(f);
 ${S?`let volume=bVolume(cell,f);
 if(sign>0){
  // Native embedded/wall contact where open and closed cells meet, on the
  // separating side's pressure.
  var q=p;q[axis]+=1;let own=umCellOpen(p)>1e-5;
  if(own==(umCellOpen(q)>1e-5)||volume<=1e-6){return false;}
  var pressure=0.0;
  if(q[axis]>=i32(UM_D[axis])){pressure=select(0.0,solve[bHalo(p,axis,sign)],bLiquid(bKinds(cell))&&!(axis==1u&&params.policy.y>0.5));}
  else if(own){let b=bCellAt(q);if(b!=0u&&bLiquid(bKinds(b-1u))){pressure=solve[b-1u];}}
  else if(bLiquid(bKinds(cell))){pressure=solve[cell];}
  return pressure<=0.0&&select(1.0,-1.0,own)*value*params.hDt.w>1e-4*params.hDt[axis];
 }
 if(umCellOpen(p)<=1e-5||volume<=1e-6){return false;}`:""}
 if(bFaceKind(bKinds(cell),f)!=K_WALL){return false;}
 let pressure=select(0.0,solve[bHalo(p,axis,sign)],bLiquid(bKinds(cell)));
 return pressure<=0.0&&-f32(sign)*value*params.hDt.w>1e-4*params.hDt[axis];
}
@compute @workgroup_size(128) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let origin=vec3i(bTile(s)*4u);
  if(lane<64u){
   // A band cell's own texel: its three positive faces and wall releases.
   let p=origin+vec3i(bLocal(lane));let cell=s*64u+lane;var texel=textureLoad(field,p,0);var released=0u;
   for(var axis=0u;axis<3u;axis++){
    let value=bFace(cell,p,2u*axis+1u,texel[axis]);texel[axis]=value;
    if(bReleased(cell,p,2u*axis+1u,value)){released|=1u<<axis;}
    if(p[axis]==0){
     let low=bFace(cell,p,2u*axis,0.0);boundary[umNegativeBoundaryIndex(vec3u(p),axis)]=low;
     if(bReleased(cell,p,2u*axis,low)){released|=1u<<(axis+3u);}
    }
   }
   texel.w=f32(released);textureStore(output,p,texel);
  }else if(lane<112u){
   // A non-band texel below this tile holds band faces in its positive
   // components. The first band tile along x, then y, then z writes it.
   let axis=(lane-64u)/16u;let k=(lane-64u)%16u;var l=vec3i(0);
   l[axis]=-1;l[(axis+1u)%3u]=i32(k%4u);l[(axis+2u)%3u]=i32(k/4u);let q=origin+l;
   if(q[axis]<0||bCellAt(q)!=0u){continue;}
   var first=true;
   for(var a=0u;a<axis;a++){var r=q;r[a]+=1;if(bCellAt(r)!=0u){first=false;}}
   if(!first){continue;}
   var texel=textureLoad(field,q,0);
   for(var a=0u;a<3u;a++){var r=q;r[a]+=1;let b=bCellAt(r);if(b!=0u){texel[a]=bFace(b-1u,r,2u*a,texel[a]);}}
   textureStore(output,q,texel);
  }
 }
}`,
   // The band projection's reads: each band tile's texels and the texel
   // layer below it, instead of the whole velocity field.
   copy:header+indexed(false)+/* wgsl */`
@group(1) @binding(22) var velocity:texture_3d<f32>;
@group(1) @binding(23) var copied:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(128) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let origin=vec3i(bTile(s)*4u);var q=origin+vec3i(bLocal(lane));
  if(lane>=64u){
   if(lane>=112u){continue;}
   let axis=(lane-64u)/16u;let k=(lane-64u)%16u;var l=vec3i(0);
   l[axis]=-1;l[(axis+1u)%3u]=i32(k%4u);l[(axis+2u)%3u]=i32(k/4u);q=origin+l;
   if(q[axis]<0){continue;}
  }
  textureStore(copied,q,textureLoad(velocity,q,0));
 }
}`,
   present:header+indexed(false)+/* wgsl */`
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
@group(1) @binding(21) var<storage,read_write> presented:array<f32>;
// The live band pressures into the stage grids' band section.
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){let cell=s*64u+lane;presented[${this.fields.presentation.word}u+cell]=solve[cell];}
}`,
  };
  const coloured=new Set(["sweep","middleSweep"]);
  for(const [name,code] of Object.entries(sources)){
   const module=this.device.createShaderModule({label:`Uniform pressure band ${name}`,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(`Pressure band ${name}: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
   const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.layouts.get(this.layoutOf(name))!,...(this.solid?[this.solid.coarse!.bindLayout]:[])]});
   for(const colour of coloured.has(name)?[0,1]:[undefined])this.pipelines.set(colour===undefined?name:`${name}${colour}`,await this.device.createComputePipelineAsync({layout:pipelineLayout,
    compute:{module,entryPoint:"main",constants:colour===undefined?{umDispatchX:this.simulation.dispatchX}:{umDispatchX:this.simulation.dispatchX,bColour:colour}}}));
  }
 }
 private layoutOf(name:string):string{const base=name.replace(/[01]$/,"");return ["open","list","copy","prep","init","project","present"].includes(base)?base:"solver";}
 /** launch: a fixed group count, or a slot-list stride (one group per slot,
  * per 8 slots, per 64 slots) capped by the capacity. */
 private dispatch(pass:GPUComputePassEncoder,name:string,launch:number|"slots"|"middle"|"coarse"):void{
  const pipeline=this.pipelines.get(name);if(!pipeline)throw new Error("Pressure band is not initialized");
  pass.setPipeline(pipeline);pass.setBindGroup(0,this.simulation.bindGroup);pass.setBindGroup(1,this.groups.get(this.layoutOf(name))!);if(this.solid)pass.setBindGroup(2,this.solid.coarse!.bindGroup);
  const c=this.capacity;
  pass.dispatchWorkgroups(typeof launch==="number"?launch:launch==="slots"?Math.min(c,SLOT_GROUPS):launch==="middle"?Math.min(Math.ceil(c/8),MIDDLE_GROUPS):Math.min(Math.ceil(c/64),COARSE_GROUPS));
 }
 private sweep(pass:GPUComputePassEncoder,name:string,launch:"slots"|"middle"|"coarse",count:number):void{
  for(let i=0;i<count;i++){this.dispatch(pass,`${name}0`,launch);this.dispatch(pass,`${name}1`,launch);}
 }
 /** Before the split, with the simulation authority's phi and correction and
  * the forced field in simulation ownership: list band tiles, assemble rows
  * and bake the aggregate operators. */
 encodePrepare(encoder:GPUCommandEncoder):void{
  encoder.clearBuffer(this.index,0,4*HEADER);encoder.clearBuffer(this.index,this.slotMapOffset);
  const pass=encoder.beginComputePass({label:"Uniform pressure band list and rows"});
  this.dispatch(pass,"open",LIST_GROUPS);this.dispatch(pass,"list",LIST_GROUPS);
  this.dispatch(pass,"prep","slots");this.dispatch(pass,"middleBake","middle");this.dispatch(pass,"coarseBake","coarse");pass.end();
 }
 /** After the 4h projection reaches simulation ownership: start from the 4h
  * pressure, run the V-cycles, and project the band faces into the velocity field. */
 encodeSolve(encoder:GPUCommandEncoder):void{
  const s=this.schedule,pass=encoder.beginComputePass({label:"Uniform pressure band solve"});
  this.dispatch(pass,"init","slots");
  for(let cycle=0;cycle<s.cycles;cycle++){
   this.sweep(pass,"sweep","slots",s.fineSweeps);
   this.dispatch(pass,"restrict","middle");this.sweep(pass,"middleSweep","middle",s.middleSweeps);
   this.dispatch(pass,"middleRestrict","coarse");this.dispatch(pass,"coarseSolve",1);
   this.dispatch(pass,"coarseProlong","middle");this.sweep(pass,"middleSweep","middle",s.middleSweeps);
   this.dispatch(pass,"middleProlong","slots");this.sweep(pass,"sweep","slots",s.fineSweeps);
  }
  this.dispatch(pass,"measure","slots");
  pass.end();
  const project=encoder.beginComputePass({label:"Uniform pressure band projection"});this.dispatch(project,"copy","slots");this.dispatch(project,"project","slots");this.dispatch(project,"present","slots");project.end();
 }
 /** The eight header words, for the frame's receipt. */
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.index,0,target,offset,32);}
 /** Diagnostics: the residual history words (before each cycle, then after the last). */
 static readonly historyWord=HISTORY_WORD;
 destroy():void{for(const b of [this.index,this.rows,this.coarse,this.open,this.solve])b.destroy();}
}
