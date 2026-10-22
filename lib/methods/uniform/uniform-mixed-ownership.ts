import { uniformMixedHangingBytes, uniformMixedHangingSlotCapacity } from "./uniform-mixed-velocity-sampling.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import type { RefinementRegionLattice } from "../../core/refinement-regions";
import { UNIFORM_MIXED_FUSED_REGULAR_TILES, uniformMixedActiveWord, uniformMixedPageCount, uniformMixedResidencyWord, uniformMixedSupportWords } from "./uniform-mixed-topology.wgsl";
import { uniformDetailPick, uniformDetailSupportWords, uniformDetailTableWords } from "./uniform-detail-fields";
import { uniformBufferedWork, type UniformWorkEdit } from "./uniform-buffered-work";
import { uniformBlockNeighbourWords, type UniformLatticeAddress } from "./uniform-lattice-address";

/** Counts copied into the existing frame receipt: tier counts (4 words),
 * seam counts (2), regular coarse owners, resident pages. */
export const UNIFORM_WORK_RECEIPT_WORDS=8;
type WorkKind="fine"|"coarse"|"all"|"regularCoarse"|"fused"|"merged"|"coarseMerged"|"sharpen"|"hanging"|"fineAndSeams"|"seamFine"|"seamCoarse"|"seams";

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
  // n tile words (by key), the h then the 4h list from n, two stencil words a key from 2n, a blend mask a key from 4n.
  const words=new Uint32Array(layout.metadataBytes/4),keys=layout.tiles.length;
  words.set(layout.tiles,0);words.set(layout.fineTiles,keys);words.set(layout.coarseTiles,keys+layout.fineTiles.length);
  words.set(layout.stencils,2*keys);words.set(layout.blendMasks,4*keys);
  writes.push(["topology",0,words]);
  // A uniform loop bound prevents explosive Metal sampler unrolling.
  // umCounts: h tiles, 4h tiles, an unused word, the loop bound.
  writes.push(["counts",0,new Uint32Array([layout.fineTiles.length,layout.boxOwners??layout.coarseTiles.length,0,8])]);
  // Standalone stages conservatively visit everything until a frame census. The keys of the AIR and the SOLID block
  // hold no tile and no census writes them: unsupported in every plane, so a stage that resolves an unstored tile
  // to them (umTileAt) reads no slot of theirs.
  const n=layout.tiles.length,planes=new Uint32Array(4*n).fill(3);
  for(let plane=0;plane<4;plane++){planes.fill(0,plane*n,plane*n+512);planes.fill(0,(plane+1)*n-512,(plane+1)*n);}
  writes.push(["support",0,planes]);
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
  writes.push(["support",uniformMixedResidencyWord(n)*4,allResident(n)]);
  // Every region active until a region heat publishes its list (umActiveCoarse).
  writes.push(["support",uniformMixedActiveWord(n)*4,allActive(n)]);
  const slotted=[...new Set([...seamLists[0]!,...seamLists[1]!])];
  const slots=new Uint32Array(2*n).fill(0xffffffff);slotted.forEach((tile,slot)=>{slots[tile]=slot;slots[n+slot]=tile;});
  const derivation={dispatchX,writes,seamCounts,hangingSlots:slotted.length,slots};
  derivations.set(layout,derivation);
  return derivation;
}

/** The residency region of `tiles` tile keys with every page resident
 * (uniformMixedResidencyWord): count, violation bits, radius, reserved, page
 * flags, ascending page list. Every flag is set, those of block 0 and of the
 * last block too, so a coordinate nothing stores reads resident like any
 * other; the list is the pages of the blocks that can hold one, and its
 * reader skips a block that holds none (umTileLive). */
const residentRegions=new Map<number,Uint32Array<ArrayBuffer>>();
const activeRegions=new Map<number,Uint32Array<ArrayBuffer>>();
function allActive(tiles:number):Uint32Array<ArrayBuffer>{
  let words=activeRegions.get(tiles);
  if(!words){const pages=uniformMixedPageCount(tiles);words=new Uint32Array(1+pages);words[0]=pages;for(let p=0;p<pages;p++)words[1+p]=p;activeRegions.set(tiles,words);}
  return words;
}
function allResident(tiles:number):Uint32Array<ArrayBuffer>{
  let words=residentRegions.get(tiles);
  if(!words){const pages=uniformMixedPageCount(tiles),held=pages-16;words=new Uint32Array(4+2*pages);words[0]=held;words.fill(1,4,4+pages);for(let p=0;p<held;p++)words[4+pages+p]=8+p;residentRegions.set(tiles,words);}
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

/** What an ownership generation cannot change, so a GPU adoption never
 * needs the host. The lattice words are fixed for the ownership's lifetime;
 * the h-tile capacity (hangingSlots, fineTiles, owners) changes only between
 * frames, by the host (UniformMixedOwnership.reserveFine). */
export interface UniformMixedCapacity{
  readonly lattice:RefinementRegionLattice;
  /** Key spaces and texture extents of the lattice. */
  readonly address:UniformLatticeAddress;
  readonly tileDimensions:readonly [number,number,number];
  /** Tile keys of the lattice (UM_TILES): the extent of every per-tile table. Its tiles are address.boxTiles. */
  readonly tiles:number;
  readonly metadataBytes:number;
  /** Preallocated hanging tap cache slots (umHangingSlots()). */
  readonly hangingSlots:number;
  /** h tiles the owner-indexed storage holds (C <= address.poolTiles): the
   * host's allocation, never a GPU count. A generation with more is fatal. */
  readonly fineTiles:number;
  /** Owners at that capacity: 63*fineTiles+address.boxTiles (an h tile is
   * 64 owners in place of one); the pressure root's, a row a tile key. */
  readonly owners:number;
}

/** The GPU buffers of one built generation (UniformMixedLayoutBuilder.
 * generation): topology, support, hanging slot table and the 16-byte counts.
 * changes: the tiles whose width the build changed and their dilation
 * (uniformMixedChangedTilesWords), what the relayout head's stages visit. */
export interface UniformMixedGenerationBuffers{readonly topology:GPUBuffer;readonly support:GPUBuffer;readonly slots:GPUBuffer;readonly counts:{readonly buffer:GPUBuffer;readonly offset:number};readonly changes:GPUBuffer}

/** Shared owner/worklist ABI for transport, face operations and pressure.
 * Two fixed-size buffers support live region edits without pipeline rebuilds.
 */
export class UniformMixedOwnership {
  readonly bindLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  /** Hanging fine-tap cache group, bound only by its producer and samplers.
   * Preallocated at capacity and replaced with it (reserveFine): fetch it
   * at encode time. */
  readonly hangingLayout: GPUBindGroupLayout;
  private hangingGroupCurrent!: GPUBindGroup;
  get hangingGroup(): GPUBindGroup {
    if(!this.sampled)throw new Error("This ownership samples no velocity: it has no hanging tap cache");
    return this.hangingGroupCurrent;
  }
  get allocatedBytes(): number { return this.topology.size + this.counts.size + this.support.size+(this.speedsBuffer?.size??0)+(this.regionVelocityBuffer?.size??0)+(this.hanging?.size??0); }
  readonly dispatchX: number;
  private readonly topology: GPUBuffer;
  /** Stable read-only view for consumers of the accepted ownership generation. */
  readonly presentation: GPUBufferBinding;
  private readonly counts: GPUBuffer;
  /** Per tile seed and three separable support planes, rebuilt at frame entry. */
  readonly support: GPUBuffer;
  /** Per-tile extended speed and its box maximum (frame plan certificate):
   * only an ownership velocity runs on (sampled) has a frame plan. */
  private readonly speedsBuffer?: GPUBuffer;
  private readonly regionVelocityBuffer?: GPUBuffer;
  /** Current extended velocity extrema per 16h region: ordered-float max(u.xyz),
   * then max(-u.xyz). Zero means no sample; decode only after the certificate. */
  get regionVelocity(): GPUBuffer {
    if(!this.regionVelocityBuffer)throw new Error("This ownership samples no velocity: it has no region velocity evidence");
    return this.regionVelocityBuffer;
  }
  get speeds(): GPUBuffer {
    if(!this.speedsBuffer)throw new Error("This ownership samples no velocity: it has no speed certificate");
    return this.speedsBuffer;
  }
  /** Per-frame velocity tap cache (uniformMixedHangingTapWGSL): every seam
   * tile owns a slot, up to capacity.hangingSlots. */
  private hanging?:GPUBuffer;
  /** Host mirror: slots in use, filled one workgroup each. */
  hangingSlots=0;
  private seamCounts=[0,0];
  private held:UniformMixedCapacity;
  get capacity():UniformMixedCapacity{return this.held;}
  /** Topology tail in words, plus the runtime window's uniform target in bytes.
   * The page pool publishes both together on initialize, seed and commit. */
  get pageMirror():{readonly buffer:GPUBuffer;readonly offset:number;readonly headerUniform:{readonly buffer:GPUBuffer;readonly offset:number}}{return {buffer:this.topology,offset:this.held.metadataBytes/4,headerUniform:{buffer:this.counts,offset:48}};}
  /** Bumped by reserveFine: whoever holds a value or a buffer sized by the
   * h-tile capacity compares it and follows. */
  capacityRevision=0;
  private readonly work:Partial<Record<WorkKind,number>>={};

  /** Same buffered direct-launch policy across stages. Evidence is lagged;
   * actual GPU lists remain authoritative and every counted kernel strides
   * them in full. Never use these estimates as storage/admission bounds.
   * edit: request edits in builds the receipt does not report yet
   * (UniformWorkEdit). The h and 4h counts grow by its tiles at most. A
   * tile outside its reach keeps its class, and a tile is at most one job
   * of a count (and 1/192 of another in sharpen's), so the seam counts grow
   * by the reach and a rounding. The small-tier term is not monotone and
   * takes its bound; the resident pages follow the liquid and take every page. */
  observeWork(words:ArrayLike<number>,reserve=0,edit?:UniformWorkEdit):void{
    if(words.length!==UNIFORM_WORK_RECEIPT_WORDS)throw new Error("Invalid ownership work receipt");
    if(edit&&(!Number.isSafeInteger(edit.tiles)||!Number.isSafeInteger(edit.reach)||edit.tiles<0||edit.reach<edit.tiles))throw new Error(`Invalid ownership work edit: ${edit.tiles} tiles, reach ${edit.reach}`);
    const tiles=edit?.tiles??0,reach=edit?edit.reach+Math.ceil(edit.reach/192)+1:0;
    const fine=words[0]!,coarse=words[1]!,seamFine=words[4]!,seamCoarse=words[5]!;
    const small=edit?2*UNIFORM_MIXED_FUSED_REGULAR_TILES:(fine<=UNIFORM_MIXED_FUSED_REGULAR_TILES?fine:0)+(coarse<=UNIFORM_MIXED_FUSED_REGULAR_TILES?coarse:0);
    const pages=edit?uniformMixedPageCount(this.capacity.tiles):words[7]!;
    // Ceilings in integer steps: a fractional intermediate is a heap number until the call is optimized.
    const regular=(words[6]!+63)>>>6,coarseGroups=(coarse+63)>>>6,sharpen=fine*64+coarse+191;
    this.observe("fine",fine+tiles,reserve);this.observe("coarse",coarse+tiles,reserve);this.observe("all",fine+tiles+(coarseGroups>pages?coarseGroups:pages),reserve);
    this.observe("regularCoarse",(words[6]!+(edit?edit.reach:0)+63)>>>6,reserve);this.observe("fused",seamFine+seamCoarse+small+reach,reserve);this.observe("hanging",seamFine+seamCoarse+reach,reserve);
    this.observe("merged",fine+seamCoarse+regular+reach,reserve);this.observe("coarseMerged",seamCoarse+regular+reach,reserve);
    this.observe("sharpen",(sharpen-sharpen%192)/192+seamCoarse+reach,reserve);
    this.observe("fineAndSeams",fine+seamCoarse+reach,reserve);this.observe("seamFine",seamFine+reach,reserve);this.observe("seamCoarse",seamCoarse+reach,reserve);
    this.observe("seams",seamFine+seamCoarse+regular+reach,reserve);
  }
  private observe(kind:WorkKind,count:number,reserve:number):void{this.work[kind]=uniformBufferedWork(this.work[kind]??1,count,this.capacity.tiles,reserve);}
  /** No evidence of the layout to come: every launch at its ceiling. */
  forgetWork():void{for(const kind of Object.keys(this.work) as WorkKind[])delete this.work[kind];}
  encodeWorkReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{
    const n=this.capacity.tiles;
    encoder.copyBufferToBuffer(this.counts,0,target,offset,16);
    encoder.copyBufferToBuffer(this.support,4*(7*n+16),target,offset+16,8);
    encoder.copyBufferToBuffer(this.support,4*(8*n+20),target,offset+24,4);
    encoder.copyBufferToBuffer(this.support,4*uniformMixedResidencyWord(n),target,offset+28,4);
  }

  private frameHeld=false;
  private currentLayout: UniformMixedLayout;
  /** Host mirror of the current membership (tiles, lists, stencils). The one
   * path to it: a GPU adoption will make this throw. Use `capacity` for
   * anything a generation cannot change. */
  get membership(): UniformMixedLayout { return this.hostMirror(); }
  get layout(): UniformMixedLayout { return this.membership; }

  /** sampled: velocity samplers run on this ownership, so it keeps the
   * hanging tap cache. Pressure levels and remap targets never sample.
   * fineTiles: the h-tile capacity (every tile the pool can store unless
   * given). The construction layout is exempt from it: it only seeds a
   * remap (the t=0 fields are h data), and no frame can be acquired on it
   * while it is over capacity. */
  // fixedWorldGeometry promises the seed's world policy and authored bounds
  // remain unchanged for this ownership's lifetime. Standalone page pools,
  // which can be reseeded, keep the default runtime header interpretation.
  constructor(private readonly device: GPUDevice, layout: UniformMixedLayout, private readonly sampled = true, fineTiles = layout.address.poolTiles, readonly fixedWorldGeometry = false) {
    this.currentLayout = layout;
    const address=layout.address,n=address.tileKeys;
    if(!Number.isInteger(fineTiles)||fineTiles<0||fineTiles>address.poolTiles)throw new Error(`Uniform mixed ownership cannot hold ${fineTiles} h tiles: its page pool stores at most ${address.poolTiles}`);
    this.held={lattice:layout.lattice,address,tileDimensions:address.tiles,tiles:n,metadataBytes:layout.metadataBytes,hangingSlots:uniformMixedHangingSlotCapacity(n,fineTiles),fineTiles,owners:layout.boxOwners??63*fineTiles+address.boxTiles};
    this.dispatchX = device.limits.maxComputeWorkgroupsPerDimension;
    this.bindLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.hangingLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    // The tail after the tile words: the page table and block records
    // (UM_PAGE_TABLE, UM_BLOCK_RECORD), written here with the address's seed
    // and left alone by every generation (copyGeneration stops at the tile
    // words): `pages` is where a page pool mirrors them. Then the block
    // neighbour rows and the wall faces of that seed (UM_BLOCK_NEIGHBOURS,
    // UM_WALL_FACES), which UniformBlockNeighbours rebuilds wherever a
    // commit rewrites the table. Then the detail-storage table, zero (raw
    // fields) until a UniformDetailStorage writes it (writeDetail).
    const pages=address.seed.tail,neighbours=uniformBlockNeighbourWords(address.seed.table,address.seed.blocks);
    this.detailBytes=layout.metadataBytes+pages.byteLength+neighbours.byteLength;
    this.topology = device.createBuffer({ label: "Uniform mixed owners and tier worklists", size: this.detailBytes+4*uniformDetailTableWords(layout.lattice.dimensions),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE });
    this.presentation={buffer:this.topology,size:this.topology.size};
    device.queue.writeBuffer(this.topology,layout.metadataBytes,pages);
    device.queue.writeBuffer(this.topology,layout.metadataBytes+pages.byteLength,neighbours);
    // The tail's last word locates the detail table (lib/core/uniform-detail-abi.ts).
    device.queue.writeBuffer(this.topology,this.topology.size-4,Uint32Array.of(this.detailBytes/4));
    this.counts = device.createBuffer({ label: "Uniform mixed work counts, geometry and window", size: 176,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE });
    device.queue.writeBuffer(this.counts,48,pages.subarray(64**3,64**3+32));
    device.queue.writeBuffer(this.counts, 16, new Uint32Array([...address.cells, n]));
    device.queue.writeBuffer(this.counts, 32, new Float32Array([...layout.lattice.cellSize_m, Math.min(...layout.lattice.cellSize_m)]));
    // 9n+24: the mixed pressure schedule's slot gate (umSlotClosed), zero
    // except on a pressure level inside a closed slot. Builders stop at 9n+24.
    // 9n+28: the residency certificate (uniformMixedResidencyWord), all
    // resident after update(); only the census narrows it.
    this.support = device.createBuffer({label:"Uniform shared frame support and certified work",size:(uniformMixedSupportWords(n)+uniformDetailSupportWords())*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    if(sampled){
      this.speedsBuffer=device.createBuffer({label:"Uniform local speed certificate",size:n*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      this.regionVelocityBuffer=device.createBuffer({label:"Uniform 16h current extended velocity extrema",size:24*(n/64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      const bytes=uniformMixedHangingBytes(n,fineTiles);
      if(bytes>device.limits.maxStorageBufferBindingSize)throw new Error(`Uniform mixed hanging tap cache needs ${bytes} bytes; the device binds at most ${device.limits.maxStorageBufferBindingSize}`);
      this.hanging=device.createBuffer({label:"Uniform mixed velocity tap cache",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      this.hangingGroupCurrent=device.createBindGroup({layout:this.hangingLayout,entries:[{binding:0,resource:{buffer:this.hanging}}]});
    }
    this.write(layout);
    this.bindGroup = device.createBindGroup({ layout: this.bindLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } }, { binding: 1, resource: { buffer: this.counts } },
      { binding: 2, resource: { buffer: this.support } },
    ] });
  }

  /** Call between submitted frames, after remapping fields from the previous
   * ownership. Queue writes are ordered after earlier submitted GPU work. */
  update(layout: UniformMixedLayout): void {
    this.assertHolds(layout);
    this.write(layout);
  }
  /** Fail before anything is encoded for a layout over the h-tile capacity
   * (a host layout's count is exact; a GPU-built one raises
   * UNIFORM_MIXED_RELAYOUT_FATAL.fineCapacity instead). */
  assertHolds(layout: UniformMixedLayout): void {
    if(layout.fineTiles.length>this.capacity.fineTiles)throw new Error(`Uniform mixed ownership needs ${layout.fineTiles.length} h tiles; its owner-indexed storage holds ${this.capacity.fineTiles}`);
  }
  /** True while the generation holds (or may hold) more h tiles than
   * capacity: the construction seed, or the one a reserveFine shrank under. */
  private overCapacity=false;
  /** Between frames: hold `fineTiles` h tiles from now on. The hanging tap
   * cache is re-created at its slot count with the slot tables (words
   * [0, 2T)) copied; the taps themselves are per frame. Owner-indexed
   * buffers belong to their stages (UniformMixedFrame.reserveFine). No
   * shader holds a capacity constant, so nothing compiles. A generation
   * over a smaller capacity must be replaced (update) before the next frame.
   * confirmed: the caller holds a build receipt showing the GPU-adopted
   * generation, and every one built since, fits `fineTiles`. */
  reserveFine(fineTiles:number,confirmed=false):void{
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    const n=this.held.tiles,before=this.held.fineTiles,most=this.held.address.poolTiles;
    if(!Number.isInteger(fineTiles)||fineTiles<0||fineTiles>most)throw new Error(`Uniform mixed ownership cannot hold ${fineTiles} h tiles: its page pool stores at most ${most}`);
    if(fineTiles===before)return;
    if(this.hanging){
      const bytes=uniformMixedHangingBytes(n,fineTiles);
      if(bytes>this.device.limits.maxStorageBufferBindingSize)throw new Error(`Uniform mixed hanging tap cache needs ${bytes} bytes; the device binds at most ${this.device.limits.maxStorageBufferBindingSize}`);
      if(bytes!==this.hanging.size){
        const next=this.device.createBuffer({label:"Uniform mixed velocity tap cache",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
        const encoder=this.device.createCommandEncoder({label:"Uniform mixed hanging slot tables"});
        encoder.copyBufferToBuffer(this.hanging,0,next,0,8*n);this.device.queue.submit([encoder.finish()]);
        // The frames encoded on the old cache are submitted: destroy applies after them.
        this.hanging.destroy();this.hanging=next;
        this.hangingGroupCurrent=this.device.createBindGroup({layout:this.hangingLayout,entries:[{binding:0,resource:{buffer:next}}]});
      }
    }
    this.held={...this.held,hangingSlots:uniformMixedHangingSlotCapacity(n,fineTiles),fineTiles,owners:this.currentLayout.boxOwners??63*fineTiles+this.held.address.boxTiles};
    // A GPU-adopted generation's h tiles reach the host as receipts: growth keeps it, a smaller capacity needs one.
    this.overCapacity=this.mirrorCurrent?this.currentLayout.fineTiles.length>fineTiles:this.overCapacity||(fineTiles<before&&!confirmed);
    this.capacityRevision++;
  }
  private write(layout: UniformMixedLayout): void {
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    this.overCapacity=layout.fineTiles.length>this.capacity.fineTiles;
    const prior=this.currentLayout.lattice;
    if(layout.metadataBytes!==this.capacity.metadataBytes || layout.lattice.dimensions.some((n,a)=>n!==prior.dimensions[a])
      || layout.lattice.cellSize_m.some((n,a)=>n!==prior.cellSize_m[a])
      || (["x","y","z"] as const).some(a=>layout.lattice.origin_m[a]!==prior.origin_m[a]))
      throw new Error("Live ownership edits cannot change the simulation lattice");
    if(layout.address!==this.capacity.address)throw new Error("Uniform mixed ownership: a layout of another lattice address (the solver builds one and every layout carries it)");
    const derived=deriveOwnership(layout,this.dispatchX);
    for(const [target,offset,data] of derived.writes)this.device.queue.writeBuffer(this[target],offset,data);
    this.seamCounts=derived.seamCounts;
    this.observeWork([layout.fineTiles.length,layout.coarseTiles.length,0,0,...derived.seamCounts,
      layout.coarseTiles.length-derived.seamCounts[1]!,uniformMixedPageCount(this.capacity.tiles)]);
    this.reserveHanging(derived.hangingSlots);
    if(this.sampled)this.device.queue.writeBuffer(this.hanging!,0,derived.slots);
    this.currentLayout=layout;
    this.mirrorCurrent=true;
    this.revision++;
  }
  /** Bumped by every generation this ownership takes (update, adopt,
   * adoptGpu): a stage that patches what changed since its last encode
   * checks that exactly one adoption intervened. */
  revision=0;

  /** Byte offset of the detail-storage table in the topology buffer. */
  private readonly detailBytes:number;
  /** The detail-storage table (uniform-detail-fields.ts), past the page
   * table and block records: queue-ordered, so it takes effect for work submitted after it. */
  writeDetail(words:Uint32Array<ArrayBuffer>):void{
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    if(this.detailBytes+words.byteLength>this.topology.size-4)throw new Error("Uniform detail table exceeds the topology tail");
    this.device.queue.writeBuffer(this.topology,this.detailBytes,words);
  }
  /** The same table's words from a GPU buffer, ordered inside an encoder
   * (the detail directory a relayout changed, ahead of the passes that
   * address fields through it). */
  encodeDetail(encoder:GPUCommandEncoder,source:GPUBuffer,sourceOffset:number,word:number,words:number):void{
    if(this.detailBytes+4*(word+words)>this.topology.size-4)throw new Error("Uniform detail table exceeds the topology tail");
    encoder.copyBufferToBuffer(source,sourceOffset,this.topology,this.detailBytes+4*word,4*words);
  }
  /** Every page resident again (between frames): a frame with no census
   * that frame must not skip a page an earlier census certified. */
  resetResidency():void{
    if(this.frameHeld)throw new Error("Ownership is immutable during an active frame");
    this.device.queue.writeBuffer(this.support,uniformMixedResidencyWord(this.capacity.tiles)*4,allResident(this.capacity.tiles));
  }
  /** Adopt a GPU-built generation with no host object: buffers only, encoded
   * at the head of the frame that runs on it (so no frame hold is checked:
   * the copies are ordered inside that frame's encoder, ahead of every pass
   * that reads the ownership). The host mirror (membership, seamCounts,
   * hangingSlots) is left behind: only capacity and GPU-counted launches
   * stay valid. A hanging overflow is latched by the builder, not here.
   * Support words [0, 5n+16) are not copied: the support planes [0, 4n)
   * (the builder's constant 3), the certificate header [4n, 4n+16) and the
   * unused list-0 slot [4n+16, 5n+16). No stage reads them on a remap
   * target, and the frame that adopts rebuilds the header and planes
   * (UniformMixedFramePlan.encode clears [0, n) and the header, seeds, and
   * dilates [n, 4n) over every tile) before any reader. */
  adoptGpu(encoder:GPUCommandEncoder,source:UniformMixedGenerationBuffers):void{
    this.copyGeneration(encoder,source);
    this.revision++;
    this.mirrorCurrent=false;this.overCapacity=false;
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
    encoder.copyBufferToBuffer(s.topology,0,this.topology,0,this.capacity.metadataBytes);
    encoder.copyBufferToBuffer(s.counts.buffer,s.counts.offset,this.counts,0,16);
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
    if(this.overCapacity)throw new Error(`Uniform mixed ownership holds a generation over its h-tile capacity (the all-h seed, or one a smaller capacity left behind); no frame runs on more than ${this.capacity.fineTiles} h tiles`);
    this.frameHeld=true;const hold=this.frameHolds[this.frameHold=(this.frameHold+1)%this.frameHolds.length]!;hold.released=false;
    return hold.release;
  }
  /** acquireFrame's releases, built once and handed out in turn: each releases
   * its own hold once, so a release kept from one of the last frames cannot
   * drop a later frame's hold. */
  private readonly frameHolds=[0,1,2,3].map(()=>{const hold={released:true,release:()=>{if(!hold.released){hold.released=true;this.frameHeld=false;}}};return hold;});
  private frameHold=0;

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
    pass.setPipeline(uniformDetailPick(regular));pass.dispatchWorkgroups(Math.min(CERTIFIED_GRID,layout.fineTiles.length));
    if(general){pass.setPipeline(uniformDetailPick(general));pass.dispatchWorkgroups(Math.min(CERTIFIED_GRID,layout.fineTiles.length));}
    pass.setPipeline(uniformDetailPick(merged));pass.dispatchWorkgroups(Math.max(1,Math.min(CERTIFIED_GRID,layout.tiles.length)));
  }

  /** Frozen interface work is shared by pressure and face stages. */
  dispatchSeams(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileGroups:boolean|readonly boolean[]=false):void{
    this.hostMirror();
    for(const tier of TIERS){
      const groups=Math.ceil(this.seamCounts[tier]!/((typeof tileGroups==="boolean"?tileGroups:tileGroups[tier])?1:64/tierOwners(tier)));
      if(!groups)continue;
      pass.setPipeline(uniformDetailPick(pipelines[tier]!));
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
    pass.setPipeline(uniformDetailPick(pipeline));pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
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
    pass.setPipeline(uniformDetailPick(pipeline));pass.dispatchWorkgroups(Math.min(groups,this.dispatchX),Math.ceil(groups/this.dispatchX));
  }

  /** Face kernels with large shared samplers can compile once for all widths.
   * The same tier worklists are packed into a single owner launch. */
  dispatchAll(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline): void {
    const groups = Math.ceil(this.hostMirror().cellCount / 64);
    if (!groups) return;
    pass.setPipeline(uniformDetailPick(pipeline));
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  /** h and 4h tile counts of the current generation, indexed by tier. */
  private tierCounts():readonly [number,number]{const layout=this.hostMirror();return [layout.fineTiles.length,layout.coarseTiles.length];}

  /** Pressure colours visit one tier; do not launch idle work for the others. */
  dispatchTier(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, tier: UniformMixedTier): void {
    const count = this.tierCounts()[tier] * tierOwners(tier);
    if (!count) return;
    const groups = Math.ceil(count / 64);
    pass.setPipeline(uniformDetailPick(pipeline));
    pass.dispatchWorkgroups(Math.min(groups, this.dispatchX), Math.ceil(groups / this.dispatchX));
  }

  // GPU-counted launches: buffered direct grids, each striding the
  // jobs its pipeline counts on the GPU (uniformMixedCountedEntriesWGSL
  // entries with umCountedJobs = UNIFORM_MIXED_COUNTED.*). No membership is
  // read synchronously, so these stay valid after a GPU adoption. Completed
  // frame counts size parallelism with headroom, not the worklist: any
  // positive grid is correct, even when the current work outruns its estimate.

  /** No h tile can exist: the h-tile capacity is zero. A layout holds at
   * most its capacity in h tiles under every policy (a host layout is
   * refused above it, a GPU build admits up to it, and no frame is encoded
   * over it), so this is exact and known on the host with no readback. */
  get coarseOnly():boolean{return this.held.fineTiles===0;}
  /** A launch over `kind` that has no job for certain. With no h tile there
   * is no seam either: the h, seam and hanging lists are empty, and the
   * fused list too unless the 4h tier is small enough to join it
   * (umFusedRegularTier). Such a launch is not issued. A counted or claimed
   * entry runs its body once per job and nothing else
   * (uniformMixedCertifiedEntriesWGSL), so the launch left out wrote nothing. */
  private idle(kind:WorkKind):boolean{
    if(!this.coarseOnly)return false;
    return kind==="fine"||kind==="seamFine"||kind==="seamCoarse"||kind==="hanging"||kind==="fineAndSeams"
      ||(kind==="fused"&&this.held.tiles>UNIFORM_MIXED_FUSED_REGULAR_TILES);
  }

  /** Also serves workgroup-claimed queues: their atomic claim loop, like a
   * counted grid-stride loop, consumes the complete live queue at any width.
   * jobsPerGroup converts the evidence units to the kernel's packed jobs. */
  dispatchBuffered(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,kind:WorkKind,maximum:number,jobsPerGroup=1):void{
    if(this.idle(kind))return;
    pass.setPipeline(uniformDetailPick(pipeline));pass.dispatchWorkgroups(Math.max(1,Math.min(maximum,Math.ceil((this.work[kind]??this.capacity.tiles)/jobsPerGroup))));
  }
  /** One counted launch whose jobs never exceed `jobs`. */
  dispatchCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,jobs:number=this.capacity.tiles,kind?:WorkKind):void{
    if(kind&&this.idle(kind))return;
    pass.setPipeline(uniformDetailPick(pipeline));pass.dispatchWorkgroups(Math.max(1,Math.min(COUNTED_GRID,jobs,kind?this.work[kind]??jobs:jobs)));
  }
  /** owners (tileJobs false) or tiles (true) launches of a tier's list: tier,
   * planned, interface (dispatchSeams) or regular (dispatchRegular; the
   * fused gate is compiled in, umFusedRegularGate). */
  dispatchTierCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,tier:UniformMixedTier,tileJobs=false,seams=false,maximum=COUNTED_GRID):void{
    const kind:WorkKind=seams?(tier===0?"seamFine":"seamCoarse"):(tier===0?"fine":"coarse");
    if(this.idle(kind))return;
    const groups=Math.ceil((this.work[kind]??this.capacity.tiles)/(tileJobs||tier===0?1:64));
    this.dispatchCounted(pass,pipeline,Math.min(groups,maximum));
  }
  /** dispatchTierCounted for each tier with pipelines[tier]. */
  dispatchTiersCounted(pass:GPUComputePassEncoder,pipelines:readonly GPUComputePipeline[],tileJobs:boolean|readonly boolean[]=false,tiers:readonly UniformMixedTier[]=TIERS,seams=false):void{
    for(const tier of tiers)this.dispatchTierCounted(pass,pipelines[tier]!,tier,typeof tileJobs==="boolean"?tileJobs:tileJobs[tier]!,seams);
  }
  /** regularCoarse: umRegularCoarseOwner, 64 owners per job. */
  dispatchRegularCoarseCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline,Math.ceil(this.capacity.tiles/64),"regularCoarse");}
  /** all: umAllOwner, 64 owners per job (dispatchAll). */
  dispatchAllCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline,this.capacity.tiles,"all");}
  /** fused or fusedQuad: one job per fused tile (dispatchFused). */
  dispatchFusedCounted(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{this.dispatchCounted(pass,pipeline,this.capacity.tiles,"fused");}

  destroy(): void { this.topology.destroy(); this.counts.destroy(); this.support.destroy();this.speedsBuffer?.destroy();this.regionVelocityBuffer?.destroy();this.hanging?.destroy(); }
}
