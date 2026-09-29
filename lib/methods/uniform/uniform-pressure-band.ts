import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {UNIFORM_MIXED_THETA_MIN} from "./uniform-mixed-pressure-surface.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

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
/** The h half sweeps run one colour's 32 cells of a slot per 32-lane group. */
const CELL_GROUPS=2048;
/** The single-workgroup 4h aggregate solve: up to this many lanes (the
 * device's workgroup limit), and the aggregates whose operator rows it holds
 * in registers. A larger band keeps the rows in storage, visits them in a
 * colour-ordered list, and keeps as many corrections as the device's
 * workgroup memory holds in workgroup memory, the rest in storage: the same
 * red-black updates in the same one launch, whatever the band's size. */
const COARSE_SOLVE_LANES=1024,COARSE_SOLVE_SLOTS=1024;
/** Band rows are field-major over CAP*64 rows: rhs, diagonal, face kinds,
 * the six face coefficients, u* per face, then with static solids the CM11a
 * V per face. A slot's 64 rows (and iterate values) are colour-major: each
 * red-black half sweep reads 32 contiguous rows per field. */
const ROW_FIELDS=15,SOLID_ROW_FIELDS=21;
/** 2h aggregates: correction, residual, diagonal, six couplings. 4h
 * aggregates add their six neighbour slots (+1), their red-black colour and
 * the large coarse solve's colour-ordered slot list. */
const MIDDLE_FIELDS=9,COARSE_FIELDS=17;
/** Index header words: count, overflow, final residual, fatal, completed
 * cycles, closed (the frame's verdict: nonzero when the pressure schedule's
 * last gate rejected the frame), converged (the convergence word), one spare,
 * then the residual after each cycle's pre-smoothing (and after the last
 * cycle) in HISTORY words. The tile list and per-tile slot+1 map follow. */
const HEADER=24,CLOSED_WORD=5,HISTORY_WORD=8,HISTORY=16;

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
 /** h.xyz, dt; density, openTop, dt/density, min h; the residual target
  * (1/s, the h-equivalent pressure tolerance), then 1/h^2 per axis (f32):
  * a face whose coefficient is exactly that is flagged, and its sweeps skip
  * the coefficient load. */
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
 * every solver pass is a fixed direct launch striding it.
 * With static solids every face carries the native CM11a dual-cell V (the
 * coefficient V/(h^2 theta), the divergence V u; V=0 faces are closed), rows
 * inside a solid are p_min=0 rows, and positive faces take the native
 * embedded contact release. Every cut tile with a liquid row is a band tile,
 * so every Neumann face is uncut (V=1): the receipt's fatal word (header 3)
 * reports a Neumann face with V<1 (2) or a liquid row in a cut tile the
 * simulation holds coarse (1), both certificate violations the frame throws
 * on. Solid promotion is liquid-conditional, so a dry cut tile may be 4h. */
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
  if(schedule.middleSweeps<1)throw new Error("The pressure band's 2h level needs a sweep: its first red half sweep carries the restriction and the prolongation");
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
  const P={0:{buffer:f.params,size:48}},solverResources={...P,2:{buffer:this.index},9:{buffer:this.rows},10:{buffer:this.solve},18:{buffer:this.coarse}};
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
  // The coarse solve's workgroup-memory corrections: all the device holds
  // beside its four scalars (Tint pads each to 16 bytes), never fewer than
  // the register path's.
  const limits=this.device.limits,lanes=Math.min(COARSE_SOLVE_LANES,limits.maxComputeInvocationsPerWorkgroup,limits.maxComputeWorkgroupSizeX),held=Math.ceil(COARSE_SOLVE_SLOTS/lanes);
  const shared=Math.min(this.capacity,4*Math.floor((limits.maxComputeWorkgroupStorageSize-128)/16));
  if(shared<Math.min(this.capacity,held*lanes))throw new Error(`Pressure band coarse solve needs ${4*held*lanes} bytes of workgroup memory`);
  const header=uniformMixedTopologyWGSL(layout,0)+uniformMixedFaceAddressWGSL+uniformMixedSolidWGSL(S?2:undefined,this.solid?.coarse?.count)+/* wgsl */`
struct BandParams {hDt:vec4f,policy:vec4f,solve:vec4f}
const CAP:u32=${this.capacity}u;const N:u32=CAP*64u;const M:u32=CAP*8u;
const LIST:u32=${HEADER}u;const SLOTS:u32=${HEADER}u+CAP;const HISTORY:u32=${HISTORY_WORD}u;const CYCLE:u32=4u;const DONE:u32=6u;
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;
fn bLocal(lane:u32)->vec3u{return vec3u(lane%4u,(lane/4u)%4u,lane/16u);}
// A slot's row of local cell l: colour-major, then the half sweep's lane order.
fn bRow(l:vec3u)->u32{return ((l.x+l.y+l.z)&1u)*32u+(l.x>>1u)+2u*l.y+8u*l.z;}
// Face f's coefficient is 1/h^2 (kinds bit 18+f).
fn bUnit(kinds:u32,f:u32)->bool{return (kinds&(1u<<(18u+f)))!=0u;}
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
// A closed band (the verdict rejected the frame) strides no slots: its solve,
// projection and presentation do nothing. Row assembly precedes the verdict.
fn bCount()->u32{return select(min(bIndex(0u),CAP),0u,bIndex(${CLOSED_WORD}u)!=0u);}
// Convergence: a cycle's restriction measured a residual at or below the
// target. Each history word is 0 (cleared with the header) until its cycle's
// restriction has run, and final once it has.
fn bMet(r:u32)->bool{return r!=0u&&bitcast<f32>(r)<=params.solve.x;}
fn bConvergedBefore(k:u32)->bool{for(var j=0u;j<min(k,${HISTORY-1}u);j++){if(bMet(bIndex(HISTORY+j))){return true;}}return false;}
fn bConverged()->bool{return bConvergedBefore(${schedule.cycles}u);}
// Converged launches stride zero slots: the same fixed launches, no work.
fn bLive()->u32{return select(bCount(),0u,bConverged());}
fn bTile(s:u32)->vec3u{return umTileCoord(bIndex(LIST+s));}
// Slot + 1 of the band tile at tile coordinate c, 0 outside the band.
fn bSlotAt(c:vec3i)->u32{if(any(c<vec3i(0))||any(c>=vec3i(UM_T))){return 0u;}return bIndex(SLOTS+umTileAt(vec3u(c)));}
// Band cell index + 1 at an h position, 0 outside the band.
fn bCellAt(q:vec3i)->u32{
 if(any(q<vec3i(0))||any(q>=vec3i(UM_D))){return 0u;}
 let u=vec3u(q);let slot=bIndex(SLOTS+umTileAt(u/4u));if(slot==0u){return 0u;}
 return (slot-1u)*64u+bRow(u%4u)+1u;
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
// transferred 4h face). A coarse air owner is Neumann too where the h cell
// is itself liquid and the band cell's own 4h owner is liquid: the 4h solve
// projected that face (ghost fluid) and owns its flux. Between two 4h air
// owners the 4h solve projected nothing -- both hold p=0 and the face was
// zeroed -- so the band sees air there (Dirichlet) at the air owner's
// centre phi. A calm 4h surface tile's rim owners (air centre, liquid below
// it) otherwise closed the band's lid: the liquid under them stopped.
struct BNeighbour {phi:f32,neumann:bool}
fn bNeighbour(q:vec3i,ownerLiquid:bool)->BNeighbour{
 let n=umOwnerAt(q);let other=phi[n.index];
 if(n.width==1u){let b=bPhiH(q);return BNeighbour(b,b<0.0);}
 if(other<0.0){return BNeighbour(other,true);}
 let own=bCentrePhi(q);if(own<0.0&&ownerLiquid){return BNeighbour(own,true);}
 return BNeighbour(select(own,other,own<0.0),false);
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
// A slot's six face-neighbour slots (+1), loaded once per slot by lanes 0..5
// so no tap reloads the tile map.
var<workgroup> bNear:array<u32,6>;
fn bLoadNear(s:u32,lane:u32){if(lane<6u){let axis=lane/2u;var c=vec3i(bTile(s));c[axis]+=bSign(lane);bNear[lane]=bSlotAt(c);}}
// Band cell + 1 across face f of local cell l of slot s, 0 outside the band.
fn bCellNear(s:u32,l:vec3u,f:u32)->u32{
 let axis=f/2u;var m=vec3i(l);m[axis]+=bSign(f);var slot=s+1u;
 if(m[axis]<0||m[axis]>3){slot=bNear[f];m[axis]=(m[axis]+4)%4;}
 if(slot==0u){return 0u;}
 return (slot-1u)*64u+bRow(vec3u(m))+1u;
}
// bOff and bResidual with the neighbour slots from bNear.
fn bOffNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){
  let axis=f/2u;let sign=bSign(f);let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){var w=params.solve[1u+axis];if(!bUnit(kinds,f)){w=bCoefficient(cell,f);}off+=w*solve[bCellNear(s,l,f)-1u];}
  else if(kind==K_WALL){off+=bCoefficient(cell,f)*solve[bHalo(p,axis,sign)];}
 }
 return off;
}
fn bResidualNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 let r=bRhs(cell)+bOffNear(cell,s,l,p,kinds)-bDiagonal(cell)*solve[cell];
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
 // A cut tile the simulation holds at 4h carries no solid terms: promotion
 // keeps it dry. A liquid row there breaks the uncut-Neumann certificate.
 for(var j=group.x*64u+lane;j<umCounts.y;j+=${LIST_GROUPS*64}u){
  let t=umTopology[UM_TILES+umCounts.x+j];
  if(umSolidStaticCut(t)&&phi[umOwnerAt(vec3i(umTileCoord(t)*4u)).index]<0.0){atomicOr(&index[3],1u);}
 }`:""}
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
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));
  let own=bPhiH(p);let liquid=own<0.0;let ownerLiquid=!bOwnerAir(umTileAt(vec3u(p)/4u));
  var kinds=select(0u,0x80000000u,liquid)${S?"|select(0u,0x40000000u,umCellInsideSolid(p))":""};var diagonal=0.0;var divergence=0.0;
  ${S?`// Native divergenceAtWithCapacity: V u + (V_i - V) u_s, u_s the moving wall's.
  let bodies=liquid&&umBodyCount()>0u;let capacity=select(0.0,umCellOpen(p),bodies);`:""}
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];var q=p;q[axis]+=sign;
   let forced=bForcedField(p,axis,sign);var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
   ${S?"var low=p;if(sign<0){low[axis]-=1;}let volume=umPressureFaceV(low,axis);":""}
   if(q[axis]<0||q[axis]>=i32(UM_D[axis])){
    if(axis==1u&&sign>0&&params.policy.y>0.5){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }else{
    let n=bNeighbour(q,ownerLiquid);
    if(bCellAt(q)!=0u){kind=K_BAND;coefficient=1.0/(h*h*bPairTheta(own,n.phi));}
    else if(n.neumann){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*bPairTheta(own,n.phi));}
   }
   ${S?`// CM11a: V scales the coefficient and the flux; the Neumann edge is uncut.
   if(kind==K_NEUMANN){if(volume<0.99999){index[3]=2u;}}
   else if(volume<=1e-6){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
   rows[(15u+f)*N+cell]=volume;`:""}
   kinds|=(kind<<(3u*f))|select(0u,1u<<(18u+f),kind==K_BAND&&coefficient==params.solve[1u+axis]);rows[(9u+f)*N+cell]=forced;rows[(3u+f)*N+cell]=coefficient;
   if(kind!=K_NEUMANN){diagonal+=coefficient;divergence+=f32(sign)*fraction*forced/h;}
   ${S?"if(bodies&&kind!=K_NEUMANN&&kind!=K_WALL&&kind!=K_OPEN){divergence+=f32(sign)*(capacity-fraction)*umSolidFaceVelocity(low,axis)/h;}":""}
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
   let l=a*2u+bMiddleLocal(k);let cell=s*64u+bRow(l);let kinds=bKinds(cell);
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
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));
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
var<workgroup> bSweepCount:u32;
// One red-black Gauss-Seidel half sweep at h; the wall halos follow their row.
// A 32-lane group relaxes one slot's 32 cells of this colour: tile origins are
// even, so the colour is the local parity.
@compute @workgroup_size(32) fn main(${slots}){
 if(lane==0u){bSweepCount=bLive();}
 let n=workgroupUniformLoad(&bSweepCount);
 for(var s=group.x;s<n;s+=groups.x){
  workgroupBarrier();bLoadNear(s,lane);workgroupBarrier();
  let y=(lane>>1u)&3u;let z=lane>>3u;let l=vec3u(2u*(lane&1u)+((y+z+bColour)&1u),y,z);
  let p=vec3i(bTile(s)*4u+l);let cell=s*64u+bColour*32u+lane;
  let kinds=bKinds(cell);let diagonal=bDiagonal(cell);
  if(bLiquid(kinds)&&diagonal>0.0){
   var next=(bRhs(cell)+bOffNear(cell,s,l,p,kinds))/diagonal;if(bInside(kinds)){next=max(next,0.0);}solve[cell]=next;
   bFollowHalo(cell,p,kinds,next);
  }
 }
}`,
   // h residual into the 2h aggregates, and its largest liquid row into
   // cycle bCycle's history word. With every 2h correction zero, the first
   // red half sweep of the 2h level is local: it runs here.
   restrict:header+indexed(true)+rows+solve+aggregates+/* wgsl */`
override bCycle:u32=0u;
var<workgroup> worst:atomic<u32>;
var<workgroup> bRestrictCount:u32;
var<workgroup> bCellResidual:array<f32,64>;
// A lane per h cell; lanes 0..7 then sum their 2h aggregate's eight cells.
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){atomicStore(&worst,0u);bRestrictCount=select(bCount(),0u,bConvergedBefore(bCycle));}
 let n=workgroupUniformLoad(&bRestrictCount);
 var largest=0.0;
 for(var s=group.x;s<n;s+=groups.x){
  workgroupBarrier();bLoadNear(s,lane);workgroupBarrier();
  let l=bLocal(lane);let cell=s*64u+bRow(l);let kinds=bKinds(cell);var r=0.0;
  if(bLiquid(kinds)&&bDiagonal(cell)>0.0){r=bResidualNear(cell,s,l,vec3i(bTile(s)*4u+l),kinds);largest=max(largest,abs(r));}
  bCellResidual[lane]=r;workgroupBarrier();
  if(lane<8u){
   let a=bMiddleLocal(lane);var sum=0.0;
   for(var k=0u;k<8u;k++){let q=a*2u+bMiddleLocal(k);sum+=bCellResidual[q.x+4u*(q.y+4u*q.z)];}
   let c=8u*s+lane;let g=bTile(s)*2u+a;var e=0.0;
   if(((g.x+g.y+g.z)&1u)==0u){let diagonal=coarse[bM(2u,c)];if(diagonal>0.0){e=(sum+0.0)/diagonal;}}
   coarse[bM(1u,c)]=sum;coarse[bM(0u,c)]=e;
  }
 }
 atomicMax(&worst,bitcast<u32>(largest*params.policy.z));
 workgroupBarrier();
 if(lane==0u){atomicMax(&index[HISTORY+bCycle],atomicLoad(&worst));}
}`,
   // With bProlong (red only), the 4h correction is prolonged on the fly: a
   // red update never reads its own value, and black values are prolonged
   // before any black update reads them.
   middleSweep:band+/* wgsl */`
override bColour:u32=0u;
override bProlong:bool=false;
var<workgroup> bMiddleCount:u32;
// The group's eight slots' face-neighbour slots (+1), loaded once.
var<workgroup> bMiddleSlots:array<u32,48>;
// The 2h aggregate + 1 across face f of aggregate c, the i-th slot of the group.
fn bMiddleNear(i:u32,c:u32,f:u32)->u32{
 let axis=f/2u;var m=vec3i(bMiddleLocal(c%8u));m[axis]+=bSign(f);var slot=c/8u+1u;
 if(m[axis]<0||m[axis]>1){slot=bMiddleSlots[6u*i+f];m[axis]=(m[axis]+2)%2;}
 if(slot==0u){return 0u;}
 return (slot-1u)*8u+u32(m.x+2*m.y+4*m.z)+1u;
}
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){bMiddleCount=bLive();}
 let n=workgroupUniformLoad(&bMiddleCount);
 for(var base=group.x*8u;base<n;base+=groups.x*8u){
  workgroupBarrier();
  if(lane<48u){let s=base+lane/6u;var slot=0u;if(s<n){slot=bCoarseNeighbour(s,lane%6u);}bMiddleSlots[lane]=slot;}
  workgroupBarrier();
  let c=base*8u+lane;
  if(c<n*8u){
   let g=bTile(c/8u)*2u+bMiddleLocal(c%8u);let diagonal=coarse[bM(2u,c)];
   if(((g.x+g.y+g.z)&1u)==bColour&&diagonal>0.0){
    var off=0.0;
    for(var f=0u;f<6u;f++){
     let m=bMiddleNear(lane/8u,c,f);
     if(m!=0u){var v=coarse[bM(0u,m-1u)];if(bProlong&&coarse[bM(2u,m-1u)]>0.0){v+=coarse[bC(0u,(m-1u)/8u)];}off+=coarse[bM(3u+f,c)]*v;}
    }
    coarse[bM(0u,c)]=(coarse[bM(1u,c)]+off)/diagonal;
   }
  }
 }
}`,
   // 2h residual into the tile aggregate (zeroing its correction).
   middleRestrict:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 let n=bLive();
 for(var s=group.x*64u+lane;s<n;s+=groups.x*64u){
  var sum=0.0;
  for(var a=0u;a<8u;a++){let c=8u*s+a;if(coarse[bM(2u,c)]>0.0){sum+=coarse[bM(1u,c)]+bMiddleOff(c)-coarse[bM(2u,c)]*coarse[bM(0u,c)];}}
  coarse[bC(1u,s)]=sum;coarse[bC(0u,s)]=0.0;
 }
}`,
   // Every red-black sweep of the 4h aggregates in one workgroup: one
   // launch instead of two per sweep, chosen on the GPU from the live count.
   // Up to SOLVE_SLOTS aggregates sweep with their operator rows in
   // registers; a larger band reads its rows from storage and keeps the
   // first SHARED corrections in workgroup memory, the rest in storage.
   coarseSolve:band+/* wgsl */`
const L:u32=${lanes}u;const HELD:u32=${held}u;const SOLVE_SLOTS:u32=L*HELD;const SHARED:u32=${shared}u;
var<workgroup> bSolveCount:u32;
var<workgroup> bRedCount:u32;
var<workgroup> bRed:atomic<u32>;
var<workgroup> bBlack:atomic<u32>;
var<workgroup> bCorrection:array<f32,SHARED>;
fn bCorrectionAt(s:u32)->f32{if(s<SHARED){return bCorrection[s];}return coarse[bC(0u,s)];}
@compute @workgroup_size(${lanes}) fn main(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){bSolveCount=bLive();}
 let n=workgroupUniformLoad(&bSolveCount);
 if(n<=SOLVE_SLOTS){
  var diagonal:array<f32,HELD>;var residual:array<f32,HELD>;var colour:array<u32,HELD>;var weight:array<f32,${6*held}>;var near:array<u32,${6*held}>;
  for(var k=0u;k<HELD;k++){
   let s=lane+k*L;diagonal[k]=0.0;
   if(s<n){
    diagonal[k]=coarse[bC(2u,s)];residual[k]=coarse[bC(1u,s)];colour[k]=bitcast<u32>(coarse[bC(15u,s)]);bCorrection[s]=coarse[bC(0u,s)];
    for(var f=0u;f<6u;f++){weight[6u*k+f]=coarse[bC(3u+f,s)];near[6u*k+f]=bitcast<u32>(coarse[bC(9u+f,s)]);}
   }
  }
  workgroupBarrier();
  for(var sweep=0u;sweep<${schedule.coarseSweeps}u;sweep++){
   for(var c=0u;c<2u;c++){
    for(var k=0u;k<HELD;k++){
     let s=lane+k*L;
     if(s>=n||colour[k]!=c||diagonal[k]<=0.0){continue;}
     var off=0.0;for(var f=0u;f<6u;f++){let m=near[6u*k+f];if(m!=0u){off+=weight[6u*k+f]*bCorrection[m-1u];}}
     bCorrection[s]=(residual[k]+off)/diagonal[k];
    }
    workgroupBarrier();
   }
  }
  for(var k=0u;k<HELD;k++){let s=lane+k*L;if(s<n){coarse[bC(0u,s)]=bCorrection[s];}}
  return;
 }
 // Colour-ordered list of the live rows (field 16): red from the front,
 // black from the back. Order within a colour is free: its updates are
 // independent.
 for(var s=lane;s<n;s+=L){
  if(s<SHARED){bCorrection[s]=coarse[bC(0u,s)];}
  if(coarse[bC(2u,s)]<=0.0){continue;}
  var at=0u;if(bitcast<u32>(coarse[bC(15u,s)])==0u){at=atomicAdd(&bRed,1u);}else{at=n-1u-atomicAdd(&bBlack,1u);}
  coarse[bC(16u,at)]=bitcast<f32>(s);
 }
 workgroupBarrier();storageBarrier();
 if(lane==0u){bRedCount=atomicLoad(&bRed);bSolveCount=n-atomicLoad(&bBlack);}
 let red=workgroupUniformLoad(&bRedCount);let black=workgroupUniformLoad(&bSolveCount);
 for(var sweep=0u;sweep<${schedule.coarseSweeps}u;sweep++){
  for(var colour=0u;colour<2u;colour++){
   let first=select(0u,black,colour==1u);let last=select(red,n,colour==1u);
   for(var j=first+lane;j<last;j+=L){
    let s=bitcast<u32>(coarse[bC(16u,j)]);
    var off=0.0;for(var f=0u;f<6u;f++){let m=bitcast<u32>(coarse[bC(9u+f,s)]);if(m!=0u){off+=coarse[bC(3u+f,s)]*bCorrectionAt(m-1u);}}
    let next=(coarse[bC(1u,s)]+off)/coarse[bC(2u,s)];
    if(s<SHARED){bCorrection[s]=next;}else{coarse[bC(0u,s)]=next;}
   }
   workgroupBarrier();storageBarrier();
  }
 }
 for(var s=lane;s<min(n,SHARED);s+=L){coarse[bC(0u,s)]=bCorrection[s];}
}`,
   middleProlong:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 let n=bLive();
 for(var s=group.x;s<n;s+=groups.x){
  let l=bLocal(lane);let p=vec3i(bTile(s)*4u+l);let cell=s*64u+bRow(l);let kinds=bKinds(cell);
  if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
  let a=l/2u;var next=solve[cell]+coarse[bM(0u,8u*s+a.x+2u*a.y+4u*a.z)];if(bInside(kinds)){next=max(next,0.0);}
  solve[cell]=next;bFollowHalo(cell,p,kinds,next);
 }
}`,
   // After bCycles encoded cycles.
   measure:header+indexed(true)+rows+solve+/* wgsl */`
override bCycles:u32=1u;
var<workgroup> worst:atomic<u32>;
var<workgroup> bMeasureCount:u32;
// The largest band row residual, in divergence units (1/s), over liquid rows,
// into history word bCycles. A converged solve keeps its restriction's
// residual (nothing moved since): the receipt takes that cycle and sets the
// convergence word.
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){atomicStore(&worst,0u);bMeasureCount=select(bCount(),0u,bConvergedBefore(bCycles));}
 let n=workgroupUniformLoad(&bMeasureCount);
 var largest=0.0;
 for(var s=group.x;s<n;s+=groups.x){
  workgroupBarrier();bLoadNear(s,lane);workgroupBarrier();
  let l=bLocal(lane);let p=vec3i(bTile(s)*4u+l);let cell=s*64u+bRow(l);
  let kinds=bKinds(cell);if(bLiquid(kinds)&&bDiagonal(cell)>0.0){largest=max(largest,abs(bResidualNear(cell,s,l,p,kinds))*params.policy.z);}
 }
 atomicMax(&worst,bitcast<u32>(largest));
 workgroupBarrier();
 if(lane==0u){let r=atomicLoad(&worst);atomicMax(&index[2],r);atomicMax(&index[HISTORY+bCycles],r);}
 if(group.x==0u&&lane==0u){
  var cycle=bCycles;for(var j=bCycles;j>0u;j--){if(bMet(bIndex(HISTORY+j-1u))){cycle=j-1u;}}
  atomicStore(&index[CYCLE],cycle);
  if(cycle<bCycles){atomicStore(&index[DONE],1u);atomicMax(&index[2],bIndex(HISTORY+cycle));}
 }
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
 let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];let liquid=bLiquid(kinds);
 // A closed face moves with its wall (pressureFaceData's u_s; zero when static).
 ${S?"if(kind==K_CLOSED){var low=p;if(sign<0){low[axis]-=1;}return umSolidFaceVelocity(low,axis);}":""}
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
  // Separation is relative to the wall's own motion (native wall[axis]).
  return pressure<=0.0&&select(1.0,-1.0,own)*(value-umSolidFaceVelocity(p,axis))*params.hDt.w>1e-4*params.hDt[axis];
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
   let p=origin+vec3i(bLocal(lane));let cell=s*64u+bRow(bLocal(lane));var texel=textureLoad(field,p,0);var released=0u;
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
 for(var s=group.x;s<bCount();s+=groups.x){presented[${this.fields.presentation.word}u+s*64u+lane]=solve[s*64u+bRow(bLocal(lane))];}
}`,
  };
  const range=(n:number,from=0)=>Array.from({length:n},(_,i)=>i+from);
  // Specialised launches: red-black colours, each cycle's restriction, the
  // measure after each encodable cycle count.
  const variants:Record<string,[string,Record<string,number>][]>={
   sweep:[0,1].map(c=>[`${c}`,{bColour:c}]),middleSweep:[...[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),["P",{bColour:0,bProlong:1}]],
   restrict:range(schedule.cycles).map(k=>[`@${k}`,{bCycle:k}]),measure:range(schedule.cycles,1).map(k=>[`@${k}`,{bCycles:k}]),
  };
  await Promise.all(Object.entries(sources).map(async([name,code])=>{
   const module=this.device.createShaderModule({label:`Uniform pressure band ${name}`,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(`Pressure band ${name}: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
   const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.layouts.get(this.layoutOf(name))!,...(this.solid?[this.solid.coarse!.bindLayout]:[])]});
   await Promise.all((variants[name]??[["",{}]]).map(async([suffix,constants])=>this.pipelines.set(name+suffix,await uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout:pipelineLayout,
    compute:{module,entryPoint:"main",constants:{umDispatchX:this.simulation.dispatchX,...constants,...s}}})))));
  }));
 }
 /** Pipeline and group 1 of each launch name, resolved on first use. */
 private readonly launches=new Map<string,{pipeline:GPUComputePipeline;group:GPUBindGroup}>();
 /** The pass the band's static groups were last bound in, and its group 1. */
 private boundPass?:GPUComputePassEncoder;
 private boundGroup?:GPUBindGroup;
 private layoutOf(name:string):string{const base=name.replace(/([01P]|@\d+)$/,"");return ["open","list","copy","prep","init","project","present"].includes(base)?base:"solver";}
 /** launch: a fixed group count, or a slot-list stride (one 32-lane group
  * per slot colour, one group per slot, per 8 slots, per 64 slots) capped by
  * the capacity. */
 private dispatch(pass:GPUComputePassEncoder,name:string,launch:number|"cells"|"slots"|"middle"|"coarse"):void{
  let bound=this.launches.get(name);
  if(!bound){const pipeline=this.pipelines.get(name);if(!pipeline)throw new Error("Pressure band is not initialized");
   bound={pipeline,group:this.groups.get(this.layoutOf(name))!};this.launches.set(name,bound);}
  // Groups 0 and 2 are the same for every band launch: bind them once per
  // pass, and group 1 only when the launch's layout changes.
  if(this.boundPass!==pass){this.boundPass=pass;this.boundGroup=undefined;pass.setBindGroup(0,this.simulation.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.coarse!.bindGroup);}
  pass.setPipeline(this.solid?.select(bound.pipeline)??bound.pipeline);if(this.boundGroup!==bound.group){this.boundGroup=bound.group;pass.setBindGroup(1,bound.group);}
  const c=this.capacity;
  pass.dispatchWorkgroups(typeof launch==="number"?launch:launch==="cells"?Math.min(c,CELL_GROUPS):launch==="slots"?Math.min(c,SLOT_GROUPS):launch==="middle"?Math.min(Math.ceil(c/8),MIDDLE_GROUPS):Math.min(Math.ceil(c/64),COARSE_GROUPS));
 }
 private sweep(pass:GPUComputePassEncoder,name:string,launch:"cells"|"middle"|"coarse",count:number):void{
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
  * pressure, run the V-cycles, and project the band faces into the velocity
  * field. Every launch is fixed: the GPU's band count picks the work. */
 encodeSolve(encoder:GPUCommandEncoder):void{
  const s=this.schedule;const pass=encoder.beginComputePass({label:"Uniform pressure band solve"});
  this.dispatch(pass,"init","slots");
  for(let cycle=0;cycle<s.cycles;cycle++){
   this.sweep(pass,"sweep","cells",s.fineSweeps);
   this.dispatch(pass,`restrict@${cycle}`,"slots");
   this.dispatch(pass,"middleSweep1","middle");this.sweep(pass,"middleSweep","middle",s.middleSweeps-1);
   this.dispatch(pass,"middleRestrict","coarse");
   this.dispatch(pass,"coarseSolve",1);
   this.dispatch(pass,"middleSweepP","middle");this.dispatch(pass,"middleSweep1","middle");this.sweep(pass,"middleSweep","middle",s.middleSweeps-1);
   this.dispatch(pass,"middleProlong","slots");this.sweep(pass,"sweep","cells",s.fineSweeps);
  }
  this.dispatch(pass,`measure@${s.cycles}`,"slots");
  pass.end();
  const project=encoder.beginComputePass({label:"Uniform pressure band projection"});this.dispatch(project,"copy","slots");this.dispatch(project,"project","slots");this.dispatch(project,"present","slots");project.end();
 }
 /** The eight header words, for the frame's receipt. */
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.index,0,target,offset,32);}
 /** Diagnostics: the residual history words (before each cycle, then after the last). */
 static readonly historyWord=HISTORY_WORD;
 /** The header word the frame's verdict closes the band with. */
 static readonly closedWord=CLOSED_WORD;
 destroy():void{for(const b of [this.index,this.rows,this.coarse,this.open,this.solve])b.destroy();}
}
