import { UNIFORM_MIXED_HANGING_RECORD } from "./uniform-mixed-velocity-sampling.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { UNIFORM_MIXED_FUSED_REGULAR_TILES } from "./uniform-mixed-topology.wgsl";

type OwnershipUploadTarget="topology"|"counts"|"support"|"certifiedDispatch";
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
  for(const part of [layout.tiles,layout.fineTiles,layout.transitionTiles,layout.coarseTiles,layout.stencils]){words.set(part,offset);offset+=part.length;}
  writes.push(["topology",0,words]);
  // A uniform loop bound prevents explosive Metal sampler unrolling.
  writes.push(["counts",0,new Uint32Array([layout.fineTiles.length,layout.transitionTiles.length,layout.coarseTiles.length,8])]);
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
  const header=new Uint32Array(16);header[2]=layout.fineTiles.length;
  header.set([Math.min(layout.fineTiles.length,dispatchX),Math.ceil(layout.fineTiles.length/dispatchX),1],8);
  const regular=(tile:number)=>(layout.stencils[2*tile]!>>>27)===(layout.stencils[2*tile+1]!>>>27);
  const seamLists=[layout.fineTiles,layout.transitionTiles,layout.coarseTiles].map(tiles=>[...tiles].filter(tile=>!regular(tile)));
  // Regular 2h/4h tiles run one lane per owner, 64 owners per merged job.
  const regularLists=[layout.transitionTiles,layout.coarseTiles].map(tiles=>[...tiles].filter(regular));
  // Merged jobs: general h, the seam 2h and seam 4h tiles, packed owners.
  const merged=layout.fineTiles.length+seamLists[1]!.length+seamLists[2]!.length+Math.ceil((8*regularLists[0]!.length+regularLists[1]!.length)/64);
  header.set([Math.min(merged,dispatchX),Math.ceil(merged/dispatchX),1],12);
  writes.push(["support",n*16,header]);
  writes.push(["certifiedDispatch",0,header.slice(4,16)]);
  writes.push(["support",(4*n+16)*4,distance]);
  writes.push(["support",(6*n+16)*4,Uint32Array.from(layout.fineTiles)]);
  const seamCounts=seamLists.map(list=>list.length);
  writes.push(["support",(7*n+16)*4,new Uint32Array([...seamCounts,0,...seamLists.flat()])]);
  writes.push(["support",(8*n+20)*4,new Uint32Array([regularLists[0]!.length,regularLists[1]!.length,0,0,...regularLists.flat()])]);
  const slotted=[...new Set([...layout.transitionTiles,...seamLists[0]!,...seamLists[2]!])];
  const slots=new Uint32Array(2*n).fill(0xffffffff);slotted.forEach((tile,slot)=>{slots[tile]=slot;slots[n+slot]=tile;});
  const derivation={dispatchX,writes,seamCounts,hangingSlots:slotted.length,slots};
  derivations.set(layout,derivation);
  return derivation;
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
  get allocatedBytes(): number { return this.topology.size + this.counts.size + this.support.size+this.speeds.size+this.certifiedDispatch.size+(this.hanging?.size??0)+(this.records?.size??0); }
  readonly dispatchX: number;
  private readonly topology: GPUBuffer;
  /** Stable read-only view for consumers of the accepted ownership generation. */
  readonly presentation: GPUBufferBinding;
  private readonly counts: GPUBuffer;
  /** Per tile seed and three separable support planes, rebuilt at frame entry. */
  readonly support: GPUBuffer;
  /** Per-tile extended speed and its box maximum (frame plan certificate). */
  readonly speeds: GPUBuffer;
  readonly certifiedDispatch:GPUBuffer;
  /** Per-frame velocity tap cache (uniformMixedHangingTapWGSL): every 2h
   * tile and every seam tile owns a slot. */
  private hanging?:GPUBuffer;
  /** Slots in the hanging tap cache, filled one workgroup each. */
  hangingSlots=0;
  /** Pressure seam records (uniform-mixed-pressure-records.wgsl), allocated
   * only for levels that smooth; stages borrow them like any field. */
  private records?:GPUBuffer;
  private recordGroupCurrent?:GPUBindGroup;
  private seamCounts=[0,0,0];

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
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
    this.presentation={buffer:this.topology,size:this.topology.size};
    this.counts = device.createBuffer({ label: "Uniform mixed work counts", size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM });
    this.support = device.createBuffer({label:"Uniform shared frame support and certified work",size:(layout.tiles.length*9+24)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.speeds = device.createBuffer({label:"Uniform local speed certificate",size:layout.tiles.length*8,usage:GPUBufferUsage.STORAGE});
    this.certifiedDispatch=device.createBuffer({label:"Uniform certified fine dispatch",size:48,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
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
    encoder.copyBufferToBuffer(s.support,(4*n+4)*4,this.certifiedDispatch,0,48);
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

  /** One workgroup covers 64 h owners, eight 2h tiles, or 64 4h tiles. */
  dispatch(pass: GPUComputePassEncoder, pipelines: readonly GPUComputePipeline[], indirect?: GPUBuffer): void {
    for (const tier of [0, 1, 2] as const) this.dispatchTier(pass, pipelines[tier]!, tier, indirect);
  }
  /** Fine work is split by a conservative whole-characteristic certificate.
   * Both lists use the same state, ownership generation and numerical stage.
   * tileGroups gives expensive face/vertex kernels one workgroup per tile. */
  dispatchCertified(pass:GPUComputePassEncoder,merged:GPUComputePipeline,regular:GPUComputePipeline):void{
    pass.setPipeline(regular);pass.dispatchWorkgroupsIndirect(this.certifiedDispatch,0);
    pass.setPipeline(merged);pass.dispatchWorkgroupsIndirect(this.certifiedDispatch,32);
  }

  /** Frozen interface work is shared by pressure and face stages. */
  dispatchSeams(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileGroups=false):void{
    for(const tier of [0,1,2] as const){
      const groups=Math.ceil(this.seamCounts[tier]!/(tileGroups?1:1<<(tier*3)));
      if(!groups)continue;
      pass.setPipeline(pipelines[tier]!);
      pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
    }
  }

  dispatchRegular(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],skipFused=false):void{
    const counts=[this.layout.fineTiles.length,this.layout.transitionTiles.length,this.layout.coarseTiles.length];
    for(const tier of [0,1,2] as const)if(counts[tier]!>this.seamCounts[tier]!&&!(skipFused&&this.fusedRegularTier(tier)))this.dispatchTier(pass,pipelines[tier]!,tier);
  }
  /** dispatchRegular's tiers, each sized by `indirect` at tier*12. */
  dispatchRegularIndirect(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],indirect:GPUBuffer,skipFused=false):void{
    const counts=[this.layout.fineTiles.length,this.layout.transitionTiles.length,this.layout.coarseTiles.length];
    for(const tier of [0,1,2] as const)if(counts[tier]!>this.seamCounts[tier]!&&!(skipFused&&this.fusedRegularTier(tier))){
      pass.setPipeline(pipelines[tier]!);pass.dispatchWorkgroupsIndirect(indirect,tier*12);
    }
  }
  /** Mirrors umFusedRegularTier: small regular tiers ride the fused launch. */
  fusedRegularTier(tier:0|1|2):boolean{
    const count=[this.layout.fineTiles.length,this.layout.transitionTiles.length,this.layout.coarseTiles.length][tier]!;
    return count>this.seamCounts[tier]!&&count<=UNIFORM_MIXED_FUSED_REGULAR_TILES;
  }
  /** One tile workgroup per interface tile of every tier (umFusedOwner),
   * optionally followed by the tiles of small regular tiers. */
  fusedJobs(regular=false):number{
    const counts=[this.layout.fineTiles.length,this.layout.transitionTiles.length,this.layout.coarseTiles.length];
    let groups=this.seamCounts[0]!+this.seamCounts[1]!+this.seamCounts[2]!;
    if(regular)for(const tier of [0,1,2] as const)if(this.fusedRegularTier(tier))groups+=counts[tier]!;
    return groups;
  }
  /** Rows of the fused job set (seams and small regular tiers): 64/w^3 per job. */
  fusedRows():number{
    const counts=[this.layout.fineTiles.length,this.layout.transitionTiles.length,this.layout.coarseTiles.length];
    let rows=0;
    for(const tier of [0,1,2] as const)rows+=(this.seamCounts[tier]!+(this.fusedRegularTier(tier)?counts[tier]!:0))*(64>>(3*tier));
    return rows;
  }
  /** One lane per fused row (umFusedRow). */
  dispatchFusedRows(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{
    const groups=Math.ceil(this.fusedRows()/64);
    if(!groups)return;
    pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
  }
  dispatchFused(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,regular=false):void{
    const groups=this.fusedJobs(regular);
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

  /** Pressure colours visit one tier; do not launch idle work for the others. */
  dispatchTier(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, tier: 0 | 1 | 2, indirect?: GPUBuffer, tileGroups=false): void {
    const count = tier === 0 ? this.layout.fineTiles.length * 64
      : tier === 1 ? this.layout.transitionTiles.length * 8 : this.layout.coarseTiles.length;
    if (!count) return;
    const ownersPerGroup=tileGroups?64/(1<<(tier*3)):64;
    const groups = Math.ceil(count / ownersPerGroup);
    pass.setPipeline(pipeline);
    if(indirect)pass.dispatchWorkgroupsIndirect(indirect,tier*12);
    else pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  /** A storage group (hangingLayout) holding at least `bytes` of records. */
  recordGroup(bytes:number):GPUBindGroup{
    if(!this.records||this.records.size<bytes){
      this.records?.destroy();
      this.records=this.device.createBuffer({label:"Uniform mixed pressure seam records",size:Math.max(16,Math.ceil(bytes*1.5/4)*4),usage:GPUBufferUsage.STORAGE});
      this.recordGroupCurrent=this.device.createBindGroup({layout:this.hangingLayout,entries:[{binding:0,resource:{buffer:this.records}}]});
    }
    return this.recordGroupCurrent!;
  }
  destroy(): void { this.topology.destroy(); this.counts.destroy(); this.support.destroy();this.speeds.destroy();this.certifiedDispatch.destroy();this.hanging?.destroy();this.records?.destroy(); }
}
