import { uniformMixedHangingBytes, uniformMixedHangingSlotCapacity } from "./uniform-mixed-velocity-sampling.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import type { RefinementRegionLattice } from "../../core/refinement-regions";
import { UNIFORM_MIXED_FUSED_REGULAR_TILES, uniformMixedOverflowWord, uniformMixedPageCount, uniformMixedResidencyWord, uniformMixedSupportWords } from "./uniform-mixed-topology.wgsl";

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
  const n=layout.tiles.length;
  // 4n: the certificate header (speed, regular and general h counts; every
  // h tile is general until the frame plan certifies).
  const header=new Uint32Array(16);header[2]=layout.fineTiles.length;
  const regular=(tile:number)=>(layout.stencils[2*tile]!>>>27)===(layout.stencils[2*tile+1]!>>>27);
  const seamLists=[layout.fineTiles,layout.coarseTiles].map(tiles=>[...tiles].filter(tile=>!regular(tile)));
  // Regular 4h tiles run one lane per owner, 64 owners per merged job.
  const regularCoarse=[...layout.coarseTiles].filter(regular);
  writes.push(["support",n*16,header]);
  writes.push(["support",(6*n+16)*4,Uint32Array.from(layout.fineTiles)]);
  const seamCounts=seamLists.map(list=>list.length);
  // 7n+16: seam h and seam 4h counts, two zero words, then both seam lists.
  writes.push(["support",(7*n+16)*4,new Uint32Array([...seamCounts,0,0,...seamLists.flat()])]);
  // 8n+20: the regular 4h count, three zero words, then its list.
  writes.push(["support",(8*n+20)*4,new Uint32Array([regularCoarse.length,0,0,0,...regularCoarse])]);
  writes.push(["support",uniformMixedResidencyWord(n)*4,allResident(layout.lattice)]);
  const slotted=[...new Set([...seamLists[0]!,...seamLists[1]!])];
  const slots=new Uint32Array(2*n).fill(0xffffffff);slotted.forEach((tile,slot)=>{slots[tile]=slot;slots[n+slot]=tile;});
  const derivation={dispatchX,writes,seamCounts,hangingSlots:slotted.length,slots};
  derivations.set(layout,derivation);
  return derivation;
}

/** The residency region with every page resident (uniformMixedResidencyWord):
 * count, violation bits, radius, reserved, page flags, ascending page list. */
const residentRegions=new Map<string,Uint32Array<ArrayBuffer>>();
function allResident(lattice:UniformMixedLayout["lattice"]):Uint32Array<ArrayBuffer>{
  const key=lattice.dimensions.join(",");let words=residentRegions.get(key);
  if(!words){const pages=uniformMixedPageCount(lattice);words=new Uint32Array(4+2*pages);words[0]=pages;words.fill(1,4,4+pages);for(let p=0;p<pages;p++)words[4+pages+p]=p;residentRegions.set(key,words);}
  return words;
}

/** Workgroups of a certified launch at most (grid-stride over its jobs). A
 * launch covers its bound: surplus workgroups exit at once, while a capped
 * grid chains jobs of uneven trace cost through one workgroup and loses the
 * hardware's balancing (f7 advect, momentum, traceCells 13.6 -> 12.2 ms at
 * 1024). The WebGPU per-dimension limit. */
const CERTIFIED_GRID=65535;
/** Workgroups of a counted launch at most (grid-stride over a GPU count):
 * enough to fill the GPU, few enough that an idle launch stays cheap. */
const COUNTED_GRID=4096;

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

/** What an ownership generation cannot change: fixed for the ownership's
 * lifetime, so a GPU adoption never needs the host. */
export interface UniformMixedCapacity{
  readonly lattice:RefinementRegionLattice;
  readonly tileDimensions:readonly [number,number,number];
  /** Tiles of the lattice (UM_TILES). */
  readonly tiles:number;
  readonly metadataBytes:number;
  /** Preallocated hanging tap cache slots (UM_HANGING_SLOTS). */
  readonly hangingSlots:number;
}

/** One GPU-built ownership generation (UniformMixedLayoutBuilder), in the
 * exact buffer layout update() uploads. */
export interface UniformMixedBuiltOwnership{
  readonly layout:UniformMixedLayout;
  readonly seamCounts:readonly number[];
  readonly hangingSlots:number;
  readonly source:UniformMixedGenerationBuffers;
}
/** The GPU buffers of one built generation (UniformMixedLayoutBuilder.
 * generation): topology, support, hanging slot table and the 16-byte counts. */
export interface UniformMixedGenerationBuffers{readonly topology:GPUBuffer;readonly support:GPUBuffer;readonly slots:GPUBuffer;readonly counts:{readonly buffer:GPUBuffer;readonly offset:number}}

/** Shared owner/worklist ABI for transport, face operations and pressure.
 * Two fixed-size buffers support live region edits without pipeline rebuilds.
 */
export class UniformMixedOwnership {
  readonly bindLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  /** Hanging fine-tap cache group, bound only by its producer and samplers.
   * Preallocated at capacity: stable for the ownership's lifetime. */
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
   * tile owns a slot, up to capacity.hangingSlots. */
  private hanging?:GPUBuffer;
  /** Host mirror: slots in use, filled one workgroup each. */
  hangingSlots=0;
  private seamCounts=[0,0];
  readonly capacity:UniformMixedCapacity;
  /** Byte offset in `support` of the sticky overflow word
   * (uniformMixedOverflowWord): nonzero is fatal. Copy it out for a
   * non-blocking diagnostic read; nothing on the host clears it. */
  readonly overflowOffset:number;

  private frameHeld=false;
  private currentLayout: UniformMixedLayout;
  /** Host mirror of the current membership (tiles, lists, stencils). The one
   * path to it: a GPU adoption will make this throw. Use `capacity` for
   * anything a generation cannot change. */
  get membership(): UniformMixedLayout { return this.hostMirror(); }
  get layout(): UniformMixedLayout { return this.membership; }

  /** sampled: velocity samplers run on this ownership, so it keeps the
   * hanging tap cache. Pressure levels and remap targets never sample. */
  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout, private readonly sampled = true) {
    this.currentLayout = layout;
    const n=layout.tiles.length;
    this.capacity={lattice:layout.lattice,tileDimensions:layout.tileDimensions,tiles:n,metadataBytes:layout.metadataBytes,hangingSlots:uniformMixedHangingSlotCapacity(n)};
    this.overflowOffset=uniformMixedOverflowWord(n)*4;
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
    // 9n+28: the residency certificate (uniformMixedResidencyWord), all
    // resident after update(); only the census narrows it.
    this.support = device.createBuffer({label:"Uniform shared frame support and certified work",size:uniformMixedSupportWords(n,layout.lattice)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.speeds = device.createBuffer({label:"Uniform local speed certificate",size:layout.tiles.length*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    if(sampled){
      const bytes=uniformMixedHangingBytes(n);
      if(bytes>device.limits.maxStorageBufferBindingSize)throw new Error(`Uniform mixed hanging tap cache needs ${bytes} bytes; the device binds at most ${device.limits.maxStorageBufferBindingSize}`);
      this.hanging=device.createBuffer({label:"Uniform mixed velocity tap cache",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
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
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const prior=this.currentLayout.lattice;
    if(layout.metadataBytes!==this.topology.size || layout.lattice.dimensions.some((n,a)=>n!==prior.dimensions[a])
      || layout.lattice.cellSize_m.some((n,a)=>n!==prior.cellSize_m[a])
      || (["x","y","z"] as const).some(a=>layout.lattice.origin_m[a]!==prior.origin_m[a]))
      throw new Error("Live ownership edits cannot change the simulation lattice");
    const derived=deriveOwnership(layout,this.dispatchX);
    for(const [target,offset,data] of derived.writes)this.device.queue.writeBuffer(this[target],offset,data);
    this.seamCounts=derived.seamCounts;
    this.reserveHanging(derived.hangingSlots);
    if(this.sampled)this.device.queue.writeBuffer(this.hanging!,0,derived.slots);
    this.currentLayout=layout;
    this.mirrorCurrent=true;
  }

  /** Every page resident again (between frames): a frame with no census
   * that frame must not skip a page an earlier census certified. */
  resetResidency():void{
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    this.device.queue.writeBuffer(this.support,uniformMixedResidencyWord(this.capacity.tiles)*4,allResident(this.capacity.lattice));
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
    if(n!==this.capacity.tiles)throw new Error("A built generation must match the ownership's capacity");
    this.reserveHanging(built.hangingSlots);
    this.copyGeneration(encoder,built.source);
    // A host-adopted generation's later frames run without a census audit.
    this.resetResidency();
    this.seamCounts=[...built.seamCounts];
    this.currentLayout=layout;
    this.mirrorCurrent=true;
  }
  /** Adopt a GPU-built generation with no host object: buffers only, encoded
   * at the head of the frame that runs on it (so no frame hold is checked:
   * the copies are ordered inside that frame's encoder, ahead of every pass
   * that reads the ownership). The host mirror (membership, seamCounts,
   * hangingSlots) is left behind: only capacity and GPU-counted launches
   * stay valid. A hanging overflow is latched by the builder, not here. */
  adoptGpu(encoder:GPUCommandEncoder,source:UniformMixedGenerationBuffers):void{
    this.copyGeneration(encoder,source);
    this.mirrorCurrent=false;
  }
  /** False once a GPU generation was adopted without a host mirror. */
  mirrorCurrent=true;
  /** The host-sized launches below size grids from the host mirror: after a
   * GPU adoption they would launch the old generation's work, so they fail
   * instead. Use the GPU-counted launches. */
  private hostMirror():UniformMixedLayout{
    if(!this.mirrorCurrent)throw new Error("A GPU-adopted ownership has no host mirror: use a GPU-counted launch");
    return this.currentLayout;
  }
  private copyGeneration(encoder:GPUCommandEncoder,s:UniformMixedGenerationBuffers):void{
    const n=this.capacity.tiles;
    encoder.copyBufferToBuffer(s.topology,0,this.topology,0,this.topology.size);
    encoder.copyBufferToBuffer(s.counts.buffer,s.counts.offset,this.counts,0,16);
    encoder.copyBufferToBuffer(s.support,0,this.support,0,(5*n+16)*4);
    encoder.copyBufferToBuffer(s.support,(6*n+16)*4,this.support,(6*n+16)*4,(3*n+8)*4);
    if(this.sampled)encoder.copyBufferToBuffer(s.slots,0,this.hanging!,0,2*n*4);
  }

  /** Host check where the host still knows the count; a GPU-built
   * generation raises UNIFORM_MIXED_OVERFLOW_HANGING instead. */
  private reserveHanging(slotted:number):void{
    if(this.sampled&&slotted>this.capacity.hangingSlots)throw new Error(`Uniform mixed ownership needs ${slotted} hanging tap slots; the cache holds ${this.capacity.hangingSlots}`);
    this.hangingSlots=slotted;
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
    const layout=this.hostMirror();
    pass.setPipeline(regular);pass.dispatchWorkgroups(Math.min(CERTIFIED_GRID,layout.fineTiles.length));
    if(general){pass.setPipeline(general);pass.dispatchWorkgroups(Math.min(CERTIFIED_GRID,layout.fineTiles.length));}
    pass.setPipeline(merged);pass.dispatchWorkgroups(Math.max(1,Math.min(CERTIFIED_GRID,layout.tiles.length)));
  }

  /** Frozen interface work is shared by pressure and face stages. */
  dispatchSeams(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileGroups:boolean|readonly boolean[]=false):void{
    this.hostMirror();
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
    const groups=Math.ceil((this.hostMirror().coarseTiles.length-this.seamCounts[1]!)/64);
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
    const groups = Math.ceil(this.hostMirror().cellCount / 64);
    if (!groups) return;
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  /** h and 4h tile counts of the current generation, indexed by tier. */
  private tierCounts():readonly [number,number]{const layout=this.hostMirror();return [layout.fineTiles.length,layout.coarseTiles.length];}

  /** Pressure colours visit one tier; do not launch idle work for the others. */
  dispatchTier(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, tier: UniformMixedTier): void {
    const count = this.tierCounts()[tier] * tierOwners(tier);
    if (!count) return;
    const groups = Math.ceil(count / 64);
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  // GPU-counted launches: fixed grids from capacity alone, each striding the
  // jobs its pipeline counts on the GPU (uniformMixedCountedEntriesWGSL
  // entries with umCountedJobs = UNIFORM_MIXED_COUNTED.*). No membership is
  // read, so these stay valid after a GPU adoption; a count of zero still
  // pays the launch. The grid only prices the launch: any size is correct.

  /** umCertifiedJobs entries (as dispatchCertified), bounded by capacity. */
  dispatchCertifiedCounted(pass:GPUComputePassEncoder,merged:GPUComputePipeline,regular:GPUComputePipeline,general?:GPUComputePipeline):void{
    const groups=Math.min(CERTIFIED_GRID,this.capacity.tiles);
    pass.setPipeline(regular);pass.dispatchWorkgroups(groups);
    if(general){pass.setPipeline(general);pass.dispatchWorkgroups(groups);}
    pass.setPipeline(merged);pass.dispatchWorkgroups(groups);
  }
  /** One counted launch whose jobs never exceed `jobs`. */
  dispatchCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,jobs:number=this.capacity.tiles):void{
    pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.max(1,Math.min(COUNTED_GRID,jobs)));
  }
  /** owners (tileJobs false) or tiles (true) launches of a tier's list: tier,
   * planned, interface (dispatchSeams) or regular (dispatchRegular; the
   * fused gate is compiled in, umFusedRegularGate). */
  dispatchTierCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,tier:UniformMixedTier,tileJobs=false):void{
    this.dispatchCounted(pass,pipeline,tileJobs?this.capacity.tiles:Math.ceil(this.capacity.tiles*tierOwners(tier)/64));
  }
  /** dispatchTierCounted for each tier with pipelines[tier]. */
  dispatchTiersCounted(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileJobs:boolean|readonly boolean[]=false,tiers:readonly UniformMixedTier[]=TIERS):void{
    for(const tier of tiers)this.dispatchTierCounted(pass,pipelines[tier]!,tier,typeof tileJobs==="boolean"?tileJobs:tileJobs[tier]!);
  }
  /** regularCoarse: umRegularCoarseOwner, 64 owners per job. */
  dispatchRegularCoarseCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline,Math.ceil(this.capacity.tiles/64));}
  /** all: umAllOwner, 64 owners per job (dispatchAll). */
  dispatchAllCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline);}
  /** fused or fusedQuad: one job per fused tile (dispatchFused). */
  dispatchFusedCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline);}
  /** hanging: one job per hanging tap slot, at most capacity.hangingSlots. */
  dispatchHangingCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline,this.capacity.hangingSlots);}

  destroy(): void { this.topology.destroy(); this.counts.destroy(); this.support.destroy();this.speeds.destroy();this.hanging?.destroy(); }
}
