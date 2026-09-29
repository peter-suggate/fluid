import { UNIFORM_MIXED_HANGING_RECORD } from "./uniform-mixed-velocity-sampling.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { UNIFORM_MIXED_FUSED_REGULAR_TILES } from "./uniform-mixed-topology.wgsl";

type OwnershipUploadTarget="topology"|"counts"|"support";
interface OwnershipDerivation{
  readonly dispatchX:number;
  /** Every upload of one ownership generation, in encode order. */
  readonly writes:readonly (readonly [OwnershipUploadTarget,number,Uint32Array<ArrayBuffer>])[];
  readonly seamCounts:number[];
  readonly hangingSlots:number;
  readonly slots:Uint32Array<ArrayBuffer>;
}
/** A relayout derives the same arrays for the remap target and the frame's
 * ownership from one layout object; derive them once. queue.writeBuffer
 * copies at call time, so cached arrays can be uploaded repeatedly. */
const derivations=new WeakMap<UniformMixedLayout,OwnershipDerivation>();
function deriveOwnership(layout:UniformMixedLayout,dispatchX:number):OwnershipDerivation{
  const cached=derivations.get(layout);
  if(cached&&cached.dispatchX===dispatchX)return cached;
  const writes:[OwnershipUploadTarget,number,Uint32Array<ArrayBuffer>][]=[];
  const words=new Uint32Array(layout.metadataBytes/4);let offset=0;
  for(const part of [layout.tiles,layout.fineTiles,layout.coarseTiles,layout.stencils]){words.set(part,offset);offset+=part.length;}
  writes.push(["topology",0,words]);
  // A uniform loop bound prevents explosive Metal sampler unrolling.
  // umCounts: h tiles, 4h tiles, an unused word, the loop bound.
  writes.push(["counts",0,new Uint32Array([layout.fineTiles.length,layout.coarseTiles.length,0,8])]);
  // Standalone stages conservatively visit everything until a frame census.
  writes.push(["support",0,new Uint32Array(layout.tiles.length*4).fill(3)]);
  // Chessboard distance to a non-fine tile, fixed for this ownership
  // generation. A frame can certify an entire characteristic's footprint
  // with one distance test instead of rediscovering ownership at every tap.
  const n=layout.tiles.length,[dx,dy,dz]=layout.tileDimensions,distance=new Uint32Array(n).fill(0xffffffff),queue=new Uint32Array(n);
  let tail=0;
  for(let t=0;t<n;t++)if((layout.tiles[t]!&0x80000000)===0){distance[t]=0;queue[tail++]=t;}
  for(let at=0;at<tail;at++){
    const t=queue[at]!,px=t%dx,py=Math.floor(t/dx)%dy,pz=Math.floor(t/(dx*dy)),next=distance[t]!+1;
    for(let z=Math.max(0,pz-1);z<=Math.min(dz-1,pz+1);z++)for(let y=Math.max(0,py-1);y<=Math.min(dy-1,py+1);y++)for(let x=Math.max(0,px-1);x<=Math.min(dx-1,px+1);x++){
      const key=x+dx*(y+dy*z);
      if(distance[key]!>next){distance[key]=next;queue[tail++]=key;}
    }
  }
  // 4n: the certificate header (speed, regular and general h counts; every
  // h tile is general until the frame plan certifies).
  const header=new Uint32Array(16);header[2]=layout.fineTiles.length;
  const regular=(tile:number)=>(layout.stencils[2*tile]!>>>27)===(layout.stencils[2*tile+1]!>>>27);
  const seamLists=[layout.fineTiles,layout.coarseTiles].map(tiles=>[...tiles].filter(tile=>!regular(tile)));
  // Regular 4h tiles run one lane per owner, 64 owners per merged job.
  const regularCoarse=[...layout.coarseTiles].filter(regular);
  writes.push(["support",n*16,header]);
  writes.push(["support",(4*n+16)*4,distance]);
  writes.push(["support",(6*n+16)*4,Uint32Array.from(layout.fineTiles)]);
  const seamCounts=seamLists.map(list=>list.length);
  // 7n+16: seam h and seam 4h counts, two zero words, then both seam lists.
  writes.push(["support",(7*n+16)*4,new Uint32Array([...seamCounts,0,0,...seamLists.flat()])]);
  // 8n+20: the regular 4h count, three zero words, then its list.
  writes.push(["support",(8*n+20)*4,new Uint32Array([regularCoarse.length,0,0,0,...regularCoarse])]);
  const slotted=[...new Set([...seamLists[0]!,...seamLists[1]!])];
  const slots=new Uint32Array(2*n).fill(0xffffffff);slotted.forEach((tile,slot)=>{slots[tile]=slot;slots[n+slot]=tile;});
  const derivation={dispatchX,writes,seamCounts,hangingSlots:slotted.length,slots};
  derivations.set(layout,derivation);
  return derivation;
}

/** Workgroups of a certified launch at most (grid-stride over its jobs). A
 * launch covers its bound: surplus workgroups exit at once, while a capped
 * grid chains jobs of uneven trace cost through one workgroup and loses the
 * hardware's balancing (f7 advect, momentum, traceCells 13.6 -> 12.2 ms at
 * 1024). The WebGPU per-dimension limit. */
const CERTIFIED_GRID=65535;

/** Ownership tiers: 0 = h (width 1), 1 = 4h (width 4). */
export type UniformMixedTier=0|1;
const TIERS=[0,1] as const;
/** Owners per tile of a tier: 64 h owners or one 4h owner. */
function tierOwners(tier:UniformMixedTier):number{return 64>>(6*tier);}

/** Per-tier pipelines for a stage on the ungraded h/4h ownership: tier 0 (h)
 * and tier 1 (4h). */
export async function compileMixedTiers(compile:(width:1|4)=>Promise<GPUComputePipeline>):Promise<GPUComputePipeline[]>{
  return Promise.all([compile(1),compile(4)]);
}

/** One GPU-built ownership generation (UniformMixedLayoutBuilder), in the
 * exact buffer layout update() uploads. */
export interface UniformMixedBuiltOwnership{
  readonly layout:UniformMixedLayout;
  readonly seamCounts:readonly number[];
  readonly hangingSlots:number;
  readonly source:{readonly topology:GPUBuffer;readonly support:GPUBuffer;readonly slots:GPUBuffer;readonly counts:{readonly buffer:GPUBuffer;readonly offset:number}};
}

/** Shared owner/worklist ABI for transport, face operations and pressure.
 * Two fixed-size buffers support live region edits without pipeline rebuilds.
 */
export class UniformMixedOwnership {
  readonly bindLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  /** Hanging fine-tap cache group, bound only by its producer and samplers.
   * Rebuilt when a live edit outgrows it, so consumers read it at encode. */
  readonly hangingLayout: GPUBindGroupLayout;
  private hangingGroupCurrent!: GPUBindGroup;
  get hangingGroup(): GPUBindGroup {
    if(!this.sampled)throw new Error("This ownership samples no velocity: it has no hanging tap cache");
    return this.hangingGroupCurrent;
  }
  get allocatedBytes(): number { return this.topology.size + this.counts.size + this.support.size+this.speeds.size+(this.hanging?.size??0); }
  readonly dispatchX: number;
  private readonly topology: GPUBuffer;
  /** Stable read-only view for consumers of the accepted ownership generation. */
  readonly presentation: GPUBufferBinding;
  private readonly counts: GPUBuffer;
  /** Per tile seed and three separable support planes, rebuilt at frame entry. */
  readonly support: GPUBuffer;
  /** Per-tile extended speed and its box maximum (frame plan certificate). */
  readonly speeds: GPUBuffer;
  /** Per-frame velocity tap cache (uniformMixedHangingTapWGSL): every seam
   * tile owns a slot. */
  private hanging?:GPUBuffer;
  /** Slots in the hanging tap cache, filled one workgroup each. */
  hangingSlots=0;
  private seamCounts=[0,0];

  private frameHeld=false;
  private currentLayout: UniformMixedLayout;
  get layout(): UniformMixedLayout { return this.currentLayout; }

  /** sampled: velocity samplers run on this ownership, so it keeps the
   * hanging tap cache. Pressure levels and remap targets never sample. */
  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout, private readonly sampled = true) {
    this.currentLayout = layout;
    this.dispatchX = device.limits.maxComputeWorkgroupsPerDimension;
    this.bindLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.hangingLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    this.topology = device.createBuffer({ label: "Uniform mixed owners and tier worklists", size: layout.metadataBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE });
    this.presentation={buffer:this.topology,size:this.topology.size};
    this.counts = device.createBuffer({ label: "Uniform mixed work counts", size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM });
    // 9n+24: the mixed pressure schedule's slot gate (umSlotClosed), zero
    // except on a pressure level inside a closed slot. Builders stop at 9n+24.
    this.support = device.createBuffer({label:"Uniform shared frame support and certified work",size:(layout.tiles.length*9+28)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.speeds = device.createBuffer({label:"Uniform local speed certificate",size:layout.tiles.length*8,usage:GPUBufferUsage.STORAGE});
    this.update(layout);
    this.bindGroup = device.createBindGroup({ layout: this.bindLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } }, { binding: 1, resource: { buffer: this.counts } },
      { binding: 2, resource: { buffer: this.support } },
    ] });
  }

  /** Call between submitted frames, after remapping fields from the previous
   * ownership. Queue writes are ordered after earlier submitted GPU work. */
  update(layout: UniformMixedLayout): void {
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const prior=this.currentLayout.lattice;
    if(layout.metadataBytes!==this.topology.size || layout.lattice.dimensions.some((n,a)=>n!==prior.dimensions[a])
      || layout.lattice.cellSize_m.some((n,a)=>n!==prior.cellSize_m[a])
      || (["x","y","z"] as const).some(a=>layout.lattice.origin_m[a]!==prior.origin_m[a]))
      throw new Error("Live ownership edits cannot change the simulation lattice");
    const derived=deriveOwnership(layout,this.dispatchX);
    for(const [target,offset,data] of derived.writes)this.device.queue.writeBuffer(this[target],offset,data);
    this.seamCounts=derived.seamCounts;
    this.hangingSlots=derived.hangingSlots;
    if(this.sampled){this.reserveHanging(layout.tiles.length,derived.hangingSlots);this.device.queue.writeBuffer(this.hanging!,0,derived.slots);}
    this.currentLayout=layout;
  }

  /** Adopt a generation built on the GPU (UniformMixedLayoutBuilder). The
   * source buffers hold exactly what update() would upload for `layout`;
   * `seamCounts` and `hangingSlots` come from the builder's receipt. Copies
   * are encoded, so they order with the caller's remap passes. Regions the
   * frame plan rebuilds every frame (support 5n+16..6n+16) are not copied. */
  adopt(encoder:GPUCommandEncoder,built:UniformMixedBuiltOwnership):void{
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const layout=built.layout,prior=this.currentLayout.lattice,n=layout.tiles.length;
    if(layout.metadataBytes!==this.topology.size || layout.lattice.dimensions.some((d,a)=>d!==prior.dimensions[a])
      || layout.lattice.cellSize_m.some((d,a)=>d!==prior.cellSize_m[a])
      || (["x","y","z"] as const).some(a=>layout.lattice.origin_m[a]!==prior.origin_m[a]))
      throw new Error("Live ownership edits cannot change the simulation lattice");
    const s=built.source;
    encoder.copyBufferToBuffer(s.topology,0,this.topology,0,this.topology.size);
    encoder.copyBufferToBuffer(s.counts.buffer,s.counts.offset,this.counts,0,16);
    encoder.copyBufferToBuffer(s.support,0,this.support,0,(5*n+16)*4);
    encoder.copyBufferToBuffer(s.support,(6*n+16)*4,this.support,(6*n+16)*4,(3*n+8)*4);
    this.seamCounts=[...built.seamCounts];
    this.hangingSlots=built.hangingSlots;
    if(this.sampled){this.reserveHanging(n,built.hangingSlots);encoder.copyBufferToBuffer(s.slots,0,this.hanging!,0,2*n*4);}
    this.currentLayout=layout;
  }

  private reserveHanging(n:number,slotted:number):void{
    this.hangingSlots=slotted;
    const hangingBytes=(2*n+UNIFORM_MIXED_HANGING_RECORD*slotted)*4;
    if(!this.hanging||this.hanging.size<hangingBytes){
      this.hanging?.destroy();
      this.hanging=this.device.createBuffer({label:"Uniform mixed velocity tap cache",size:Math.max(hangingBytes,(2*n+UNIFORM_MIXED_HANGING_RECORD*Math.ceil(slotted*1.5))*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      this.hangingGroupCurrent=this.device.createBindGroup({layout:this.hangingLayout,entries:[{binding:0,resource:{buffer:this.hanging}}]});
    }
  }

  /** Hold ownership and its stencil/support storage through every asynchronous
   * pressure receipt. Edits can only publish after the frame finishes. */
  acquireFrame():()=>void {
    if(this.frameHeld)throw new Error("Ownership already belongs to an active frame");
    this.frameHeld=true;let released=false;
    return ()=>{if(!released){released=true;this.frameHeld=false;}};
  }

  /** One workgroup covers 64 h owners or 64 4h tiles. */
  dispatch(pass: GPUComputePassEncoder, pipelines: readonly GPUComputePipeline[]): void {
    for (const tier of TIERS) this.dispatchTier(pass, pipelines[tier]!, tier);
  }
  /** Fine work is split by a conservative whole-characteristic certificate.
   * Both lists use the same state, ownership generation and numerical stage.
   * tileGroups gives expensive face/vertex kernels one workgroup per tile. */
  /** Both pipelines are uniformMixedCertifiedEntriesWGSL entries (regular:
   * umCertifiedJobs 1; merged: 2, or 3 when it packs seam 4h tiles four per
   * job); each strides a fixed grid over the jobs the frame plan certified,
   * bounded by this layout's tiles. general: umCertifiedJobs 1 over the
   * general h list (umPlannedFine 2), with merged compiled umMergedCoarse. */
  dispatchCertified(pass:GPUComputePassEncoder,merged:GPUComputePipeline,regular:GPUComputePipeline,general?:GPUComputePipeline):void{
    const layout=this.currentLayout;
    pass.setPipeline(regular);pass.dispatchWorkgroups(Math.min(CERTIFIED_GRID,layout.fineTiles.length));
    if(general){pass.setPipeline(general);pass.dispatchWorkgroups(Math.min(CERTIFIED_GRID,layout.fineTiles.length));}
    pass.setPipeline(merged);pass.dispatchWorkgroups(Math.max(1,Math.min(CERTIFIED_GRID,layout.tiles.length)));
  }

  /** Frozen interface work is shared by pressure and face stages. */
  dispatchSeams(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileGroups:boolean|readonly boolean[]=false):void{
    for(const tier of TIERS){
      const groups=Math.ceil(this.seamCounts[tier]!/((typeof tileGroups==="boolean"?tileGroups:tileGroups[tier])?1:64/tierOwners(tier)));
      if(!groups)continue;
      pass.setPipeline(pipelines[tier]!);
      pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
    }
  }

  dispatchRegular(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],skipFused=false,tiers:readonly UniformMixedTier[]=TIERS):void{
    const counts=this.tierCounts();
    for(const tier of tiers)if(counts[tier]!>this.seamCounts[tier]!&&!(skipFused&&this.fusedRegularTier(tier)))this.dispatchTier(pass,pipelines[tier]!,tier);
  }
  /** umRegularCoarseOwner's regular 4h list, one lane per owner, 64 per
   * group, unless the tier is fused. */
  dispatchRegularCoarse(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{
    if(this.fusedRegularTier(1))return;
    const groups=Math.ceil((this.layout.coarseTiles.length-this.seamCounts[1]!)/64);
    if(!groups)return;
    pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
  }
  /** Mirrors umFusedRegularTier: small regular tiers ride the fused launch. */
  fusedRegularTier(tier:UniformMixedTier):boolean{
    const count=this.tierCounts()[tier];
    return count>this.seamCounts[tier]!&&count<=UNIFORM_MIXED_FUSED_REGULAR_TILES;
  }
  /** One tile workgroup per interface tile of every tier (umFusedOwner),
   * optionally followed by the tiles of small regular tiers. */
  fusedJobs(regular=false):number{
    const counts=this.tierCounts();
    let groups=this.seamCounts[0]!+this.seamCounts[1]!;
    if(regular)for(const tier of TIERS)if(this.fusedRegularTier(tier))groups+=counts[tier];
    return groups;
  }
  /** Rows of the fused job set (seams and small regular tiers): 64/w^3 per job. */
  fusedRows():number{
    const counts=this.tierCounts();
    let rows=0;
    for(const tier of TIERS)rows+=(this.seamCounts[tier]!+(this.fusedRegularTier(tier)?counts[tier]:0))*tierOwners(tier);
    return rows;
  }
  /** One lane per fused row (umFusedRow). */
  dispatchFusedRows(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{
    const groups=Math.ceil(this.fusedRows()/64);
    if(!groups)return;
    pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
  }
  /** quad: a uniformMixedFaceTileDispatchWGSL fused pipeline, which packs
   * the seam 4h tiles four per job. */
  dispatchFused(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,regular=false,quad=false):void{
    const groups=this.fusedJobs(regular)-(quad?this.seamCounts[1]!-Math.ceil(this.seamCounts[1]!/4):0);
    if(!groups)return;
    pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
  }

  /** Face kernels with large shared samplers can compile once for all widths.
   * The same tier worklists are packed into a single owner launch. */
  dispatchAll(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline): void {
    const groups = Math.ceil(this.layout.cellCount / 64);
    if (!groups) return;
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  /** h and 4h tile counts of the current generation, indexed by tier. */
  private tierCounts():readonly [number,number]{return [this.layout.fineTiles.length,this.layout.coarseTiles.length];}

  /** Pressure colours visit one tier; do not launch idle work for the others. */
  dispatchTier(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, tier: UniformMixedTier): void {
    const count = this.tierCounts()[tier] * tierOwners(tier);
    if (!count) return;
    const groups = Math.ceil(count / 64);
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  destroy(): void { this.topology.destroy(); this.counts.destroy(); this.support.destroy();this.speeds.destroy();this.hanging?.destroy(); }
}
