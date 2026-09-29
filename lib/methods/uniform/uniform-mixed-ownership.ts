import { UNIFORM_MIXED_HANGING_RECORD } from "./uniform-mixed-velocity-sampling.wgsl";
import type { UniformMixedCapacity, UniformMixedLayout } from "./uniform-mixed-layout";
import { UNIFORM_MIXED_FUSED_REGULAR_TILES, UNIFORM_MIXED_JOBS, type UniformMixedJobKind } from "./uniform-mixed-topology.wgsl";

type OwnershipUploadTarget="topology"|"counts"|"support";
interface OwnershipDerivation{
  readonly dispatchX:number;
  /** Every upload of one ownership generation, in encode order. */
  readonly writes:readonly (readonly [OwnershipUploadTarget,number,Uint32Array<ArrayBuffer>])[];
  readonly seamCounts:number[];
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
  // 9n+25: no hanging overflow (checked on the host below).
  writes.push(["support",(9*n+25)*4,new Uint32Array(1)]);
  const slotted=[...new Set([...seamLists[0]!,...seamLists[1]!])];
  const slots=new Uint32Array(2*n).fill(0xffffffff);slotted.forEach((tile,slot)=>{slots[tile]=slot;slots[n+slot]=tile;});
  if(slotted.length>uniformMixedHangingCapacity(n))throw new Error(`Mixed ownership has ${slotted.length} seam tiles, over the hanging tap cache's ${uniformMixedHangingCapacity(n)} slots`);
  const derivation={dispatchX,writes,seamCounts,slots};
  derivations.set(layout,derivation);
  return derivation;
}

/** Hanging tap cache slots of an n-tile lattice: one per seam tile, budgeted
 * at half the tiles. The builder flags a generation with more seam tiles
 * (support 9n+25) and the frame receipt fails on it. */
export function uniformMixedHangingCapacity(n:number):number{return Math.min(n,Math.max(1024,Math.ceil(n/2)));}

const J=UNIFORM_MIXED_JOBS;
/** Kinds whose jobs are those of the pipeline's umCellWidth tier. */
const TIER_KINDS=new Set<UniformMixedJobKind>([J.tier,J.regular,J.regularUnfused,J.regularFused,J.seamLanes,J.seamTiles]);
/** The job kind and tier width each certified pipeline was compiled for. */
const jobKinds=new WeakMap<GPUComputePipeline,{kind:UniformMixedJobKind;width:number}>();

/** Ownership tiers: 0 = h (width 1), 1 = 4h (width 4). */
export type UniformMixedTier=0|1;
const TIERS=[0,1] as const;
/** Owners per tile of a tier: 64 h owners or one 4h owner. */
function tierOwners(tier:UniformMixedTier):number{return 64>>(6*tier);}

/** One GPU-built ownership generation (UniformMixedLayoutBuilder), in the
 * exact buffer layout update() uploads. The host never sees its layout. */
export interface UniformMixedBuiltOwnership{
  readonly source:{readonly topology:GPUBuffer;readonly support:GPUBuffer;readonly slots:GPUBuffer;readonly counts:{readonly buffer:GPUBuffer;readonly offset:number}};
}

/** Shared owner/worklist ABI for transport, face operations and pressure.
 * Two fixed-size buffers support live region edits without pipeline rebuilds.
 */
export class UniformMixedOwnership {
  readonly bindLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  /** Hanging fine-tap cache group, bound only by its producer and samplers,
   * allocated once at uniformMixedHangingCapacity slots. */
  readonly hangingLayout: GPUBindGroupLayout;
  private hangingGroupCurrent?: GPUBindGroup;
  get hangingGroup(): GPUBindGroup {
    if(!this.hangingGroupCurrent)throw new Error("This ownership samples no velocity: it has no hanging tap cache");
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
  private seamCounts=[0,0];

  private frameHeld=false;
  /** The host layout, until a GPU-built generation replaces it. */
  private hostLayout?: UniformMixedLayout;
  /** Host layout: construction and CPU (region) edits only. Throws once the
   * GPU owns the layout; launches and buffers use `capacity`. */
  get layout(): UniformMixedLayout {
    if(!this.hostLayout)throw new Error("The GPU owns this ownership's layout: the host has only its capacity");
    return this.hostLayout;
  }
  readonly capacity: UniformMixedCapacity;
  /** Tiles of the lattice: the capacity every launch and buffer is sized by. */
  get tileCount():number{return this.capacity.tileCount;}
  /** sampled: velocity samplers run on this ownership, so it keeps the
   * hanging tap cache. Pressure levels and remap targets never sample.
   * fixed: the layout never changes (pressure levels), so launches take
   * exactly its jobs instead of a capacity grid. */
  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout, private readonly sampled = true, private readonly fixed = false) {
    this.hostLayout = layout;
    this.capacity={lattice:layout.lattice,tileDimensions:layout.tileDimensions,tileCount:layout.tiles.length,metadataBytes:layout.metadataBytes};
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
    if(sampled){
      const n=layout.tiles.length;
      this.hanging=device.createBuffer({label:"Uniform mixed velocity tap cache",size:(2*n+UNIFORM_MIXED_HANGING_RECORD*uniformMixedHangingCapacity(n))*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      this.hangingGroupCurrent=device.createBindGroup({layout:this.hangingLayout,entries:[{binding:0,resource:{buffer:this.hanging}}]});
    }
    this.update(layout);
    this.bindGroup = device.createBindGroup({ layout: this.bindLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } }, { binding: 1, resource: { buffer: this.counts } },
      { binding: 2, resource: { buffer: this.support } },
    ] });
  }

  /** Call between submitted frames, after remapping fields from the previous
   * ownership. Queue writes are ordered after earlier submitted GPU work. */
  update(layout: UniformMixedLayout): void {
    if(this.fixed&&layout!==this.hostLayout)throw new Error("A fixed ownership never changes layout");
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const prior=this.capacity.lattice;
    if(layout.metadataBytes!==this.topology.size || layout.lattice.dimensions.some((n,a)=>n!==prior.dimensions[a])
      || layout.lattice.cellSize_m.some((n,a)=>n!==prior.cellSize_m[a])
      || (["x","y","z"] as const).some(a=>layout.lattice.origin_m[a]!==prior.origin_m[a]))
      throw new Error("Live ownership edits cannot change the simulation lattice");
    const derived=deriveOwnership(layout,this.dispatchX);
    for(const [target,offset,data] of derived.writes)this.device.queue.writeBuffer(this[target],offset,data);
    this.seamCounts=derived.seamCounts;
    if(this.hanging)this.device.queue.writeBuffer(this.hanging,0,derived.slots);
    this.hostLayout=layout;
  }

  /** Adopt a generation built on the GPU (UniformMixedLayoutBuilder): the
   * source buffers hold exactly what update() would upload. Copies are
   * encoded, so they order with the caller's remap passes; the host learns
   * nothing of the layout. Regions the frame plan rebuilds every frame
   * (support 5n+16..6n+16) are not copied. */
  adopt(encoder:GPUCommandEncoder,built:UniformMixedBuiltOwnership):void{
    if(this.fixed)throw new Error("A fixed ownership never changes layout");
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const n=this.tileCount;
    const s=built.source;
    encoder.copyBufferToBuffer(s.topology,0,this.topology,0,this.topology.size);
    encoder.copyBufferToBuffer(s.counts.buffer,s.counts.offset,this.counts,0,16);
    encoder.copyBufferToBuffer(s.support,0,this.support,0,(5*n+16)*4);
    encoder.copyBufferToBuffer(s.support,(6*n+16)*4,this.support,(6*n+16)*4,(3*n+8)*4);
    // 9n+25: the builder's hanging overflow flag (frame receipt).
    encoder.copyBufferToBuffer(s.support,(9*n+25)*4,this.support,(9*n+25)*4,4);
    if(this.hanging)encoder.copyBufferToBuffer(s.slots,0,this.hanging,0,2*n*4);
    this.hostLayout=undefined;
  }



  /** Hold ownership and its stencil/support storage through every asynchronous
   * pressure receipt. Edits can only publish after the frame finishes. */
  acquireFrame():()=>void {
    if(this.frameHeld)throw new Error("Ownership already belongs to an active frame");
    this.frameHeld=true;let released=false;
    return ()=>{if(!released){released=true;this.frameHeld=false;}};
  }

  /** Compile a certified entry (uniformMixedCertifiedEntriesWGSL) as job
   * kind `kind`: the ownership launches it only as that kind. */
  async pipeline(layout:GPUPipelineLayout,module:GPUShaderModule,entryPoint:string,kind:UniformMixedJobKind,constants:Record<string,number>={}):Promise<GPUComputePipeline>{
    const pipeline=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.dispatchX,...constants,umCertifiedJobs:kind}}});
    jobKinds.set(pipeline,{kind,width:constants.umCellWidth??1});
    return pipeline;
  }
  /** One launch of a certified pipeline over the GPU job count of `kind`
   * (umCertifiedJobCount). A live ownership strides a fixed grid sized by its
   * capacity, so no launch depends on the host knowing the layout; a fixed
   * one (pressure levels) launches one group per job of its own layout. */
  private launch(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,kind:UniformMixedJobKind,tier:UniformMixedTier=0):void{
    const record=jobKinds.get(pipeline);
    if(!record||record.kind!==kind||(TIER_KINDS.has(kind)&&record.width!==(tier?4:1)))
      throw new Error(`Mixed launch of job kind ${kind} (tier ${tier}) got a pipeline compiled as ${record?`kind ${record.kind}, width ${record.width}`:"an uncertified entry"}`);
    // A live grid is the capacity bound, not a smaller stride: surplus groups
    // exit at once and every job keeps its own workgroup (load balancing).
    const groups=this.fixed?this.layoutJobs(kind,tier):Math.max(1,this.capacityJobs(kind,tier));
    if(!groups)return;
    pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.min(groups,this.dispatchX));
  }
  /** The most jobs any layout of this lattice gives `kind`: every tile h. */
  private capacityJobs(kind:UniformMixedJobKind,tier:UniformMixedTier):number{
    const n=this.tileCount,lanes=Math.ceil(n/64);
    if(kind===J.regularCoarse)return lanes;
    return TIER_KINDS.has(kind)&&kind!==J.seamTiles&&tier===1?lanes:n;
  }
  /** umCertifiedJobCount on the host, for a fixed ownership's own layout. */
  private layoutJobs(kind:UniformMixedJobKind,tier:UniformMixedTier):number{
    const counts=this.tierCounts(),seams=this.seamCounts,count=counts[tier],seam=seams[tier]!;
    const groups=tier?Math.ceil(count/64):count,fused=(t:UniformMixedTier)=>counts[t]>seams[t]!&&counts[t]<=UNIFORM_MIXED_FUSED_REGULAR_TILES;
    switch(kind){
      case J.all:return counts[0]+Math.ceil(counts[1]/64);
      case J.tier:return groups;
      case J.regular:return count>seam?groups:0;
      case J.regularUnfused:return count>seam&&!fused(tier)?groups:0;
      case J.regularFused:return fused(tier)?groups:0;
      case J.regularCoarse:return fused(1)?0:Math.ceil((counts[1]-seams[1]!)/64);
      case J.seamLanes:return tier?Math.ceil(seams[1]!/64):seams[0]!;
      case J.seamTiles:return seam;
      case J.fused:return seams[0]!+seams[1]!;
      case J.fusedRegular:case J.fusedRegularQuad:{
        let jobs=seams[0]!+seams[1]!;for(const t of TIERS)if(fused(t))jobs+=counts[t];
        return kind===J.fusedRegularQuad?jobs-seams[1]!+Math.ceil(seams[1]!/4):jobs;
      }
      default:return this.capacityJobs(kind,tier);
    }
  }

  /** Both tiers (tier kinds). */
  dispatch(pass: GPUComputePassEncoder, pipelines: readonly GPUComputePipeline[]): void {
    for (const tier of TIERS) this.launch(pass, pipelines[tier]!, J.tier, tier);
  }
  /** Fine work is split by a conservative whole-characteristic certificate.
   * Both lists use the same state, ownership generation and numerical stage.
   * merged: umCertifiedJobs merged or mergedQuad; regular: planned. */
  dispatchCertified(pass:GPUComputePassEncoder,merged:GPUComputePipeline,regular:GPUComputePipeline):void{
    this.launch(pass,regular,J.planned);
    this.launch(pass,merged,jobKinds.get(merged)?.kind===J.mergedQuad?J.mergedQuad:J.merged);
  }

  /** Frozen interface work is shared by pressure and face stages. tileGroups:
   * one tile per job (seamTiles) instead of 64 owners (seamLanes). */
  dispatchSeams(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileGroups:boolean|readonly boolean[]=false):void{
    for(const tier of TIERS)this.launch(pass,pipelines[tier]!,(typeof tileGroups==="boolean"?tileGroups:tileGroups[tier])?J.seamTiles:J.seamLanes,tier);
  }
  /** Tiers with regular tiles; skipFused: not those riding the fused launch. */
  dispatchRegular(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],skipFused=false,tiers:readonly UniformMixedTier[]=TIERS):void{
    for(const tier of tiers)this.launch(pass,pipelines[tier]!,skipFused?J.regularUnfused:J.regular,tier);
  }
  /** A small regular tier that the fused launch would carry, launched alone. */
  dispatchRegularFused(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,tier:UniformMixedTier):void{
    this.launch(pass,pipeline,J.regularFused,tier);
  }
  /** umRegularCoarseOwner's regular 4h list, one lane per owner, 64 per
   * group, unless the tier is fused. */
  dispatchRegularCoarse(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{
    this.launch(pass,pipeline,J.regularCoarse);
  }
  /** One tile job per interface tile of every tier (umFusedOwner), optionally
   * followed by the tiles of small regular tiers. quad: a
   * uniformMixedFaceTileDispatchWGSL fused pipeline, which packs the seam 4h
   * tiles four per job. */
  dispatchFused(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,regular=false,quad=false):void{
    if(quad&&!regular)throw new Error("Quad-packed fused launches include the small regular tiers");
    this.launch(pass,pipeline,quad?J.fusedRegularQuad:regular?J.fusedRegular:J.fused);
  }

  /** Face kernels with large shared samplers can compile once for all widths.
   * The same tier worklists are packed into a single owner launch. */
  dispatchAll(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline): void {
    this.launch(pass,pipeline,J.all);
  }

  /** h and 4h tile counts of a fixed ownership's layout, indexed by tier. */
  private tierCounts():readonly [number,number]{return [this.layout.fineTiles.length,this.layout.coarseTiles.length];}

  /** Pressure colours visit one tier; do not launch idle work for the others. */
  dispatchTier(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, tier: UniformMixedTier): void {
    this.launch(pass,pipeline,J.tier,tier);
  }

  destroy(): void { this.topology.destroy(); this.counts.destroy(); this.support.destroy();this.speeds.destroy();this.hanging?.destroy(); }
}
