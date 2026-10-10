import { uniformSceneConstantsWGSL } from "./uniform-scene-constants";
/** Detail fields hold a tile-resolution base block and a lattice-sized h
 * texture when fine ownership is reserved. Shader accessors read the base
 * at canonical texels and keep both homes current inside the detail ring. */
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedDetailViolationWord, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { UNIFORM_DETAIL_4H_LOAD, UNIFORM_DETAIL_BASED, UNIFORM_DETAIL_CANONICAL_LOAD, UNIFORM_DETAIL_FACE_COMPACT, UNIFORM_DETAIL_HEADER_WORDS, UNIFORM_DETAIL_PACKED, UNIFORM_DETAIL_RING_4H_LOAD, registerUniformDetailBase } from "../../core/uniform-detail-abi";

export type UniformDetailClass="cell"|"face"|"vertex"|"atlas";
type Dims=readonly [number,number,number];
const CLASSES=["cell","face","vertex","atlas"] as const;

const MIN_PATCH_EDGE=16;
/** Table words ahead of the directory (lib/core/uniform-detail-abi.ts). */
const HEADER=UNIFORM_DETAIL_HEADER_WORDS;
/** Sticky bits in umSupport[UM_DETAIL_VIOLATION], one per class. */
export const UNIFORM_DETAIL_VIOLATION={cell:1,face:2,vertex:4,atlas:8} as const;
/** Words the topology tail reserves: the table for the smallest patch edge,
 * then the locator (the table's first word index: a consumer that does not
 * know the lattice finds the table from the buffer's last word). */
export function uniformDetailTableWords(dims:readonly number[]):number{return HEADER+dims.reduce((n,d)=>n*Math.ceil(d/MIN_PATCH_EDGE),1)+1;}

export interface UniformDetailClassLayout{
 /** Atlas slot pitch: P, or P+1 for packed vertex slots. */
 readonly spacing:number;
 /** Base block: t³ cells, (2t)³ faces (2c + (l==3) per axis), (t+1)³ vertices. */
 readonly base:Dims;
 /** Packed: the base block comes first, folded into z slabs (slice z of the
  * base at fold column and row of slab z/per); the atlas starts at slab
  * atlasZ. Nothing here depends on capacity. */
 readonly atlasZ:number;readonly fold:number;readonly per:number;

 readonly extent:Dims;
}
export interface UniformDetailLayout{
 readonly dims:Dims;readonly tiles:Dims;readonly patchEdge:number;readonly patchGrid:Dims;readonly patches:number;

 readonly placement:"packed";
 /** Slots of one atlas layer (x, y), fixed by the lattice: capacity grows by
  * whole layers along z, so no slot moves and no address constant changes. */
 readonly layer:readonly [number,number];
 /** Layers that hold every patch. */
 readonly maxLayers:number;
 readonly classes:Readonly<Record<UniformDetailClass,UniformDetailClassLayout>>;

 readonly domain:UniformDetailDomain;

}

/** The halo each class needs (storage design, "Slice step 2: results"). */

/** Is a profiled load site arithmetic at `halo`? It ran, no base-homed work
 * item read an h texel through it, no tap found no texel (a profile recorded
 * before a texel became reachable would otherwise leave its site arithmetic),
 * and no tap left the box by more than the halo now covers. */

export function uniformDetailSupportWords():number{return 0;}

const haloOf=(l:UniformDetailLayout,kind:UniformDetailClass)=>0;

export interface UniformDetailDomain{
 /** A load of a base texel that is not canonical raises a violation bit
  * (UNIFORM_DETAIL_LOAD_VIOLATION), as a non-canonical store always does. */
 readonly checked:boolean;
 /** The four-texels-per-tile face base. */
 readonly compactFaces:boolean;
 /** A/B: per-tile cell fields follow the capacity like every other field. */
 readonly unpinned?:boolean;
 // The ring: beside the base block, the h store holds a 4h tile's canonical
 // texels only for the tiles of the detail ring (UNIFORM_MIXED_DETAIL_RING:
 // an h tile within three tiles). A 4h store outside it lands in the base
 // alone and no kernel loads the h texture there (certified by `poison`).
 // The consumers outside the solver (overlays, particles, extraction,
 // readbacks) take a 4h tile's values from the base block
 // (uniform-detail-abi.ts), so nothing is published for them.

 readonly poison?:boolean;

 readonly survey?:boolean;

 readonly untwinned?:boolean;

 readonly unwrapped?:boolean;
}
/** The texel value a poisoned h texel holds (-2^100, exact). */
export const UNIFORM_DETAIL_POISON=-(2**100);
/** Sticky bits of a checked domain placement: an h
 * texture load outside the ring; a load that returned the poison; a
 * non-canonical store outside the ring. */
export const UNIFORM_DETAIL_RING_VIOLATION=512,UNIFORM_DETAIL_POISON_VIOLATION=1024,UNIFORM_DETAIL_RING_STORE_VIOLATION=2048;
/** Site word bits of a domain survey or poison run (siteWords): the site
 * loaded an h texture; outside the ring; the poison; a non-canonical store
 * outside the ring. */
export const UNIFORM_DETAIL_RING_SITE={ran:1,outside:2,poison:4,store:8,homes:16} as const;
/** Checked: a canonical load read its base block while the h
 * texture, inside the ring, held another value (the two homes disagree). */
export const UNIFORM_DETAIL_HOME_VIOLATION=4096;
/** Sticky bits beside UNIFORM_DETAIL_VIOLATION under a checked domain placement. */
export const UNIFORM_DETAIL_LOAD_VIOLATION={cell:16,face:32,vertex:64} as const;
/** Sticky bit of a checked domain placement: an h-only load (UNIFORM_DETAIL_H_LOAD) ran on a base block. */
export const UNIFORM_DETAIL_H_VIOLATION=256;
function domainLayout(dims:Dims,domain:UniformDetailDomain):UniformDetailLayout{
 if(dims.some(n=>!Number.isInteger(n)||n<4||n%4))throw new Error(`Uniform detail storage needs a lattice of whole tiles, not ${dims}`);
 const tiles=dims.map(n=>n/4) as unknown as Dims,patchEdge=2**Math.ceil(Math.log2(Math.max(MIN_PATCH_EDGE,...dims)));
 const classes=Object.fromEntries(CLASSES.map(kind=>{
  const base=(kind==="cell"?tiles:kind==="vertex"?tiles.map(t=>t+1):[tiles[0],tiles[1],4*tiles[2]]) as unknown as Dims;
  return [kind,{spacing:patchEdge,base,atlasZ:0,fold:1,per:1,extent:base}];
 })) as unknown as Record<UniformDetailClass,UniformDetailClassLayout>;
 return {dims,tiles,patchEdge,patchGrid:[1,1,1],patches:1,placement:"packed",layer:[1,1],maxLayers:1,classes,domain};
}

/** Directory word of a resident patch: its slot coordinate, 10 bits per axis. */
const entry=(slot:Dims)=>(0x80000000|slot[0]|slot[1]<<10|slot[2]<<20)>>>0;

interface Field{readonly storage:UniformDetailStorage;readonly kind:UniformDetailClass;readonly logical:Dims;readonly handle:GPUTexture}
const fields=new WeakMap<GPUTexture,Field>();
/** The detail field a texture is (handle or physical), if a UniformDetailStorage made it. */
export function uniformDetailField(texture:GPUTexture):Field|undefined{return fields.get(texture);}
/** A bind group over detail fields. The storage re-creates it when the
 * fields' physical textures are replaced, so holders keep this object and
 * set `.group` at encode time, never the GPUBindGroup. */
export interface UniformDetailGroup{
 group:GPUBindGroup;
 /** The same group with every field as its base block (domain placement;
  * elsewhere `group`). For a launch whose every load and store is a
  * canonical texel: a 4h job with no h tile in its stencil reads and writes
  * tile-resolution storage at every capacity, as it does at C = 0, by the
  * same code. A store is written through to the h texture while one exists. */
 base:GPUBindGroup;
}
export interface UniformDetailGroupDescriptor{label?:string;layout:GPUBindGroupLayout;entries:Iterable<{binding:number;resource:GPUBindingResource|GPUTexture}>}
/** createBindGroup for a group that may name detail fields: a texture
 * resource (a field handle or any other texture) binds its default view, a
 * field's being its current physical texture's. */
export function uniformDetailGroup(device:GPUDevice,descriptor:UniformDetailGroupDescriptor):UniformDetailGroup{
 const entries=[...descriptor.entries],twins=twinLayouts.get(descriptor.layout);let storage:UniformDetailStorage|undefined;
 for(const e of entries){const f=fields.get(e.resource as GPUTexture);if(f)storage=f.storage;}
 const domain=!!storage?.layout.domain;
 const build=(base=false)=>device.createBindGroup({label:descriptor.label,layout:descriptor.layout,entries:entries.flatMap(e=>{
  const f=fields.get(e.resource as GPUTexture);
  const primary={binding:e.binding,resource:f?(base?f.storage.baseOf(f.handle)!:f.storage.physical(f.handle)).createView():"createView" in e.resource?(e.resource as GPUTexture).createView():e.resource as GPUBindingResource};
  if(!twins?.formats.has(e.binding))return [primary];
  const sampled=!twins.formats.get(e.binding);
  return [primary,{binding:e.binding+UNIFORM_DETAIL_TWIN_BINDING,resource:(f?f.storage.twin(f.handle,sampled,base):twinDummy(device,twins!,e.binding)).createView()}];
 })});
 const group=build(),bound={group,base:domain?build(true):group};
 storage?.track(bound,domain?build:()=>build());return bound;
}
/** A field binding's twin sits this far above it (domain placement): the
 * field's base block (a stored binding's: while its h generation is the
 * bound texture, else a 1³ stand-in). The accessors store a canonical texel
 * to both, so a kernel reads a 4h value from the base or the h texture,
 * whichever it binds, with no test in the load. */
export const UNIFORM_DETAIL_TWIN_BINDING=64;
interface TwinLayout{
 /** Twinned bindings: the storage format, undefined for a sampled binding. */
 readonly formats:ReadonlyMap<number,GPUTextureFormat|undefined>;
 /** Stand-ins of the twinned storage bindings that hold no field. */
 readonly dummies:Map<number,GPUTexture>;
}
const twinLayouts=new WeakMap<GPUBindGroupLayout,TwinLayout>();
const sampledDummies=new WeakMap<GPUDevice,GPUTexture>();
function twinDummy(device:GPUDevice,twins:TwinLayout,binding:number):GPUTexture{
 const format=twins.formats.get(binding);
 if(!format){
  let dummy=sampledDummies.get(device);
  if(!dummy)sampledDummies.set(device,dummy=device.createTexture({label:"Uniform detail twin stand-in",size:[1,1,1],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING}));
  return dummy;
 }
 // Writable: one per binding, so no two bindings of a dispatch alias.
 let dummy=twins.dummies.get(binding);
 if(!dummy)twins.dummies.set(binding,dummy=device.createTexture({label:"Uniform detail twin stand-in",size:[1,1,1],dimension:"3d",format,usage:GPUTextureUsage.STORAGE_BINDING}));
 return dummy;
}
/** createBindGroupLayout for a layout uniformDetailGroup binds and a
 * rewritten shader reads: every 3D float texture entry (a field's formats)
 * gets its twin entry. A shader declares the twins of its field bindings
 * only (uniformDetailShader), so a twin of anything else is bound and never
 * read. skip: bindings that take no twin. */
export function uniformDetailBindLayout(device:GPUDevice,descriptor:GPUBindGroupLayoutDescriptor,skip:readonly number[]=[]):GPUBindGroupLayout{
 const entries=[...descriptor.entries],twins:GPUBindGroupLayoutEntry[]=[],formats=new Map<number,GPUTextureFormat|undefined>();
 for(const e of entries){
  if(skip.includes(e.binding))continue;
  const sampled=e.texture?.viewDimension==="3d"&&(e.texture.sampleType===undefined||e.texture.sampleType==="float"||e.texture.sampleType==="unfilterable-float");
  const stored=e.storageTexture?.viewDimension==="3d"&&(e.storageTexture.format==="r32float"||e.storageTexture.format==="rgba32float");
  if(!sampled&&!stored)continue;
  if(entries.some(o=>o.binding===e.binding+UNIFORM_DETAIL_TWIN_BINDING))throw new Error(`Uniform detail layout ${descriptor.label??""}: binding ${e.binding+UNIFORM_DETAIL_TWIN_BINDING} is a twin's`);
  twins.push(sampled?{binding:e.binding+UNIFORM_DETAIL_TWIN_BINDING,visibility:e.visibility,texture:{...e.texture}}:{binding:e.binding+UNIFORM_DETAIL_TWIN_BINDING,visibility:e.visibility,storageTexture:{...e.storageTexture!}});
  formats.set(e.binding,stored?e.storageTexture!.format:undefined);
 }
 const layout=device.createBindGroupLayout({...descriptor,entries:[...entries,...twins]});
 if(twins.length)twinLayouts.set(layout,{formats,dummies:new Map()});
 return layout;
}
/** Logical extent of a field (vertex fields n+1), else the physical one:
 * bind-time lattice asserts use this, never the texture's size. */
export function uniformDetailExtent(texture:GPUTexture):[number,number,number]{
 const field=fields.get(texture);return field?[...field.logical]:[texture.width,texture.height,texture.depthOrArrayLayers];
}

/** Live storages per device: the rewrite finds a shader's layout by its
 * lattice (UM_D). Two live layouts on one lattice and device are ambiguous. */
const live=new WeakMap<GPUDevice,Set<UniformDetailStorage>>();
function liveLayout(device:GPUDevice,dims:readonly number[]):UniformDetailLayout|undefined{
 const found=[...live.get(device)??[]].filter(s=>s.layout.dims.every((n,a)=>n===dims[a]));
 if(found.some(s=>s.signature!==found[0]!.signature))throw new Error(`Uniform detail storage: two live layouts on lattice ${dims}`);
 return found[0]?.layout;
}

const DOMAIN_FLAGS=["checked","compact","parity","unpinned","poison","survey","untwinned","unwrapped"];

/** Residency header words of the GPU state (UniformDetailStorage): slot
 * capacity, resident patches, sticky fatal bits (1: a relayout needed more
 * slots than capacity), the resident high-water mark, the last relayout's
 * admitted and retiring patches, the largest demand that overflowed, and
 * the every-patch-resident switch. */
const D={capacity:0,resident:1,fatal:2,peak:3,admitted:4,retired:5,demand:6,all:7,words:8} as const;
export const UNIFORM_DETAIL_RECEIPT_BYTES=4*D.words;
/** Fatal bit of the domain placement: a relayout put a tile of the detail
 * ring outside the h store's window (UniformDetailStorage.window). */
const WINDOW_FATAL=2;
/** Words past the state's lists: the window's tile box (low x y z, high x y z, inclusive). */
const WINDOW_WORDS=8;
/** Tiles the h store holds around an h tile: the detail ring's reach
 * (uniform-mixed-layout.ts mixedStencils, the builder's `mirror` entry). */
const WINDOW_RING=3;
/** Workgroups of each transfer launch: their lanes stride the GPU lists. */

/** Fields one transfer launch binds (the default storage-texture limit). */
const BATCH=4;
type TransferKind="cell"|"vertex"|"face";

/** The ring kernels of one launch: up to BATCH fields of one format. */
interface RingBatch{readonly handles:readonly GPUTexture[];readonly layout:GPUBindGroupLayout;readonly enter:GPUComputePipeline;readonly poison?:GPUComputePipeline;readonly poisonAll?:GPUComputePipeline}
interface DomainKernels{

 readonly ring?:{readonly batches:readonly RingBatch[];readonly shareLayout:GPUBindGroupLayout;readonly share:GPUComputePipeline};
 readonly layouts:Record<"r32float"|"rgba32float",GPUBindGroupLayout>;
 /** refresh: h to base, the canonical texels. */
 readonly refresh:Record<TransferKind,GPUComputePipeline>;
}
/** The field factory: owns every h-sized texture of one solver and their
 * residency. Packed residency lives on the GPU (the state buffer: header,
 * the directory, the slot table and the relayout's lists): each relayout
 * admits the patches of its target generation's h tiles and their
 * 26-connected one-tile closure before the remap (encodeAdmit: new patches
 * are filled with what their tiles held non-resident, then the directory is
 * published into every attached topology) and retires the rest after it
 * (encodeRetire: canonical texels restricted into the base, slots freed),
 * with no readback. A relayout that needs more slots than capacity sets a
 * sticky fatal bit the frame receipt throws on. */
export class UniformDetailStorage{
 readonly layout:UniformDetailLayout;

 private readonly handles:GPUTexture[]=[];
 private readonly physicals=new Map<GPUTexture,GPUTexture>();
 private readonly stubs=new Map<GPUTexture,GPUTexture>();
 private readonly ownerships=new Set<UniformMixedOwnership>();
 /** Host mirror of the directory (0: not resident): current until the GPU
  * admits or retires, then again after sync(). */
 private readonly directory:Uint32Array;
 private mirrored=true;
 /** Atlas layers allocated, and their slots (the allocator's capacity). */
 private layers:number;
 private slots:number;
 get capacity():number{return this.slots;}

 private readonly fixed:boolean;
 readonly signature:string;
 private readonly residentAll:boolean;
 private readonly state?:GPUBuffer;
 private readonly ringDimensions?:GPUBuffer;
 private readonly words:{readonly dir:number;readonly slots:number;readonly admit:number;readonly retire:number;readonly need:number;readonly total:number};
 /** The last frame receipt's residency (noteReceipt). */
 private receipt?:{capacity:number;resident:number;peak:number};
 /** Patches resident after the last host relayout; undefined once a GPU
  * relayout admits on its own. */
 private expected?:Set<number>;
 constructor(private readonly device:GPUDevice,dims:Dims){
  const domain:UniformDetailDomain={checked:false,compactFaces:true,unpinned:false};
  const identity=false,local=undefined,patch=undefined;
  this.layout=domainLayout(dims,domain);
  const l=this.layout,per=l.layer[0]*l.layer[1];
  this.cells=l.dims;this.box=[0,0,0,l.tiles[0]-1,l.tiles[1]-1,l.tiles[2]-1];

  this.residentAll=false;
  this.fixed=true;
  this.slots=l.patches;
  if(!Number.isInteger(this.slots)||this.slots<0||this.slots>l.patches)throw new Error(`Uniform detail capacity ${this.slots} must hold 0..${l.patches} patches`);
  this.layers=Math.ceil(this.slots/per);
  this.signature=`${l.placement}:${l.patchEdge}:${l.layer}:${local??""}${domain?`:domain:${domain.checked}:${domain.compactFaces}:${!!domain.unpinned}:${!!domain.poison}:${!!domain.survey}:${!!domain.untwinned}:${!!domain.unwrapped}`:""}${""}`;
  let set=live.get(device);if(!set)live.set(device,set=new Set());set.add(this);liveLayout(device,dims);
  this.directory=new Uint32Array(l.patches);
  const maxSlots=l.maxLayers*per,dir=D.words,slots=dir+l.patches,admit=slots+maxSlots,retire=admit+l.patches,need=retire+l.patches;
  this.words={dir,slots,admit,retire,need,total:need+l.patches};

  if(domain){this.ringDimensions=device.createBuffer({label:"Uniform detail ring dimensions",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(this.ringDimensions,0,new Uint32Array([...l.tiles,this.words.total]));}
  if(l.patches>device.limits.maxComputeWorkgroupsPerDimension)throw new Error(`Uniform detail storage scans ${l.patches} patches, past one dispatch dimension`);
  if(this.residentAll)for(let i=0;i<l.patches;i++)this.directory[i]=entry(this.slotCoord(i));
  this.state=device.createBuffer({label:"Uniform detail residency",size:4*(this.words.total+WINDOW_WORDS),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.writeState();this.setBox(this.box);
 }

 private slotCoord(slot:number):Dims{const [x,y]=this.layout.layer;return [slot%x,Math.floor(slot/x)%y,Math.floor(slot/(x*y))];}
 private slotIndex(e:number):number{const [x,y]=this.layout.layer;return (e&1023)+x*(((e>>>10)&1023)+y*((e>>>20)&1023));}
 /** The host directory becomes the GPU's: header, directory and slot table. */
 private writeState():void{
  const w=this.words,words=new Uint32Array(w.admit);let resident=0;
  this.directory.forEach((e,p)=>{if(!e)return;resident++;words[w.slots+this.slotIndex(e)]=p+1;});
  words.set([this.slots,resident,0,resident,0,0,0,this.residentAll?1:0]);words.set(this.directory,w.dir);
  this.device.queue.writeBuffer(this.state!,0,words);
 }
 private extent(kind:UniformDetailClass,resident=this.resident):Dims{
  const l=this.layout,c=l.classes[kind];
  return resident?this.storeExtent(kind,this.cells):c.base;
 }

 // The h store's window (domain placement).
 /** Cells of the h store along each axis: the lattice's, or a power of two
  * below it, on which the store wraps (logical texel p is at
  * p & (cells - 1)). A wrapped store holds the texels of a tile box no wider
  * than itself wherever that box lies, so a box that moves keeps the store
  * and every texel's place in it. */
 private cells:Dims;
 /** The tile box (inclusive) whose texels the store holds on its wrapped
  * axes: the host's bound on the h tiles, with the detail ring around it.
  * The lattice's range on an axis that does not wrap. */
 private box:readonly number[];
 private get windowed():boolean{const c=this.cells,d=this.layout.dims;return c[0]!<d[0]!||c[1]!<d[1]!||c[2]!<d[2]!;}

 /** The last texel of a tile box (inclusive) on an axis: its top tile's.
  * A vertex's home is the tile below it (min(q >> 2, T - 1)), so the plane
  * above the box is another tile's, except the lattice's top plane. */
 private boxEnd(kind:UniformDetailClass,box:readonly number[],a:number):number{
  return 4*box[a+3]!+3+(kind==="vertex"&&box[a+3]===this.layout.tiles[a]!-1?1:0);
 }
 /** A kind's h store for `cells`: a wrapped axis is `cells` texels (a box of
  * w tiles has 4w texels of each kind, and the lattice's top plane of
  * vertices where it reaches it, which `cells` holds), any other the lattice's. */
 private storeExtent(kind:UniformDetailClass,cells:readonly number[]):Dims{
  const n=this.layout.dims;return cells.map((c,a)=>c<n[a]!?c:kind==="vertex"?n[a]!+1:n[a]!) as unknown as Dims;
 }
 private setBox(box:readonly number[]):void{
  const T=this.layout.tiles,n=this.layout.dims;
  this.box=[0,1,2].map(a=>this.cells[a]!<n[a]!?box[a]!:0).concat([0,1,2].map(a=>this.cells[a]!<n[a]!?box[a+3]!:T[a]!-1));
  if(this.state)this.device.queue.writeBuffer(this.state,4*this.words.total,Uint32Array.from(this.box));
 }
 /** Dynamic detail keeps lattice-sized h storage; only ring membership varies. */
 window(_tiles?:readonly number[]):void{

  this.cells=this.layout.dims;
  const T=this.layout.tiles;
  this.setBox([0,0,0,T[0]-1,T[1]-1,T[2]-1]);
 }
 private allocate(label:string,kind:UniformDetailClass,format:GPUTextureFormat,resident=this.resident):GPUTexture{
  const extent=this.extent(kind,resident);
  if(extent.some(n=>n>this.device.limits.maxTextureDimension3D))throw new RangeError(`Uniform detail field ${label} needs ${extent}, past the device's 3D texture limit`);
  return this.device.createTexture({label,size:[...extent],dimension:"3d",format,
   usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
 }
 /** A field: the returned texture is its handle (see Field). Bind it
  * through uniformDetailGroup; read and write it through this storage. */
 createField(label:string,kind:UniformDetailClass,format:"r32float"|"rgba32float",perTile=false):GPUTexture{
  // An atlas-only field has slots of its own under haloed patches; elsewhere it shares the face placement.
  if(kind==="atlas")kind="face";
  if(this.transfers)throw new Error(`Uniform detail field ${label} was created after the transfer kernels bound the fields`);
  if(perTile&&kind!=="cell")throw new Error(`Uniform detail field ${label}: only a cell field is per tile`);
  const pin=perTile;
  if(((kind==="face")!==(format==="rgba32float")))throw new Error(`Uniform detail field ${label}: the domain placement transfers scalar cell and vertex fields and vec4 face fields`);
  const physical=this.allocate(label,kind,format,pin?false:undefined);let handle=physical;
  {handle=this.device.createTexture({label,size:[1,1,1],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING});handle.destroy();}
  const d=this.layout.dims,field:Field={storage:this,kind,logical:kind==="vertex"?[d[0]+1,d[1]+1,d[2]+1]:d,handle};
  fields.set(handle,field);fields.set(physical,field);
  this.handles.push(handle);this.physicals.set(handle,physical);if(pin)this.pinned.add(handle);
  this.bases.set(handle,physical);
  return handle;
 }
 /** Domain: every field's base block. It lives at every capacity and holds
  * the current value of every canonical texel (a tile's origin cell, its
  * origin and +face anchors, a tile corner), of an h tile as of a 4h one:
  * the home of the 4h values. While C > 0 the h texture beside it is the
  * whole logical field (every h texel, the non-canonical texels kernels
  * write in a 4h tile beside the seam, and the canonical texels again), so
  * a reader may take a canonical texel from either. Every store of a
  * canonical texel writes both (the accessor twins, upload, copy; a kernel
  * with no twin is followed by encodeBaseRefresh). */
 private readonly bases=new Map<GPUTexture,GPUTexture>();
 /** A field's base block under the domain placement (else undefined). */
 baseOf(texture:GPUTexture):GPUTexture|undefined{return this.bases.get(fields.get(texture)?.handle??texture);}
 /** What a field's twin binding takes. sampled: its base block, at every
  * capacity (two sampled bindings may share a texture). Stored: its other
  * home (the base block beside the h texture; base: the h texture beside
  * the base block), a 1³ stand-in while it has one home (no h generation
  * resident, or the field is pinned). */
 twin(texture:GPUTexture,sampled:boolean,base=false):GPUTexture{
  const handle=fields.get(texture)!.handle;
  if(sampled)return this.bases.get(handle)!;
  if(!this.resident||this.pinned.has(handle))return this.stub(handle);
  return base?this.physicals.get(handle)!:this.bases.get(handle)!;
 }
 /** Two fields of one class: the textures copy whole, bases included. */
 copy(encoder:GPUCommandEncoder,from:GPUTexture,to:GPUTexture):void{
  const a=this.physical(from),b=this.physical(to);
  if(a.width!==b.width||a.height!==b.height||a.depthOrArrayLayers!==b.depthOrArrayLayers)throw new Error(`Uniform detail copy: ${from.label} and ${to.label} are not of one class`);
  encoder.copyTextureToTexture({texture:a},{texture:b},[a.width,a.height,a.depthOrArrayLayers]);
  const x=this.bases.get(fields.get(from)!.handle),y=this.bases.get(fields.get(to)!.handle);
  if(x&&y&&x!==a)encoder.copyTextureToTexture({texture:x},{texture:y},[x.width,x.height,x.depthOrArrayLayers]);
 }
 /** Every distinct texture the storage holds for its fields. */
 private textures():GPUTexture[]{return this.owned(new Set([...this.physicals.values(),...this.stubs.values(),...this.bases.values()]));}
 /** Domain: per-tile cell fields (createField's perTile). Every kernel
  * stores and loads one only at tile origins, whatever the owners, so its
  * base block is the field at every occupancy: never admitted or retired. */
 private readonly pinned=new Set<GPUTexture>();
 /** The fields whose generation follows the h-tile capacity. */
 private get following():GPUTexture[]{return this.handles.filter(h=>!this.pinned.has(h));}
 /** The texture that holds a field's texels now: valid until the storage
  * next grows, so resolve it at each use. */
 physical(texture:GPUTexture):GPUTexture{
  const f=fields.get(texture);if(f?.storage!==this)throw new Error(`${texture.label} is not a field of this detail storage`);
  return this.physicals.get(f.handle)!;
 }

 stub(texture:GPUTexture):GPUTexture{

  const handle=fields.get(texture)!.handle;let stub=this.stubs.get(handle);
  if(!stub)this.stubs.set(handle,stub=this.device.createTexture({label:`${handle.label} stub`,size:[1,1,1],dimension:"3d",format:handle.format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}));
  return stub;
 }
 private denseBinding?:{layout:GPUBindGroupLayout;group:GPUBindGroup};
 /** The residency state as uniformDetailDenseShader's directory group. */
 get dense():{layout:GPUBindGroupLayout;group:GPUBindGroup}{
  if(!this.state)throw new Error("Identity detail storage has no directory");
  if(!this.denseBinding){
   const size=16*Math.ceil((D.words+this.layout.patches)/4);
   if(size>this.device.limits.maxUniformBufferBindingSize)throw new Error(`Uniform detail directory of ${this.layout.patches} patches exceeds one uniform binding`);
   const layout=this.device.createBindGroupLayout({label:"Uniform detail directory",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
   this.denseBinding={layout,group:this.device.createBindGroup({label:"Uniform detail directory",layout,entries:[{binding:0,resource:{buffer:this.state,size}}]})};
  }
  return this.denseBinding;
 }
 private bound:{ref:WeakRef<UniformDetailGroup>;build:(base?:boolean)=>GPUBindGroup}[]=[];
 private prune=64;
 /** uniformDetailGroup's registration: rebuilt when the fields are replaced. */
 track(group:UniformDetailGroup,build:(base?:boolean)=>GPUBindGroup):void{
  if(this.bound.length>=this.prune){this.bound=this.bound.filter(b=>b.ref.deref());this.prune=Math.max(64,2*this.bound.length);}
  this.bound.push({ref:new WeakRef(group),build});
 }
 /** Allocate atlas layers for `slots` patch slots. The base block and the
  * resident slots keep their texels (the atlas grows in z), so each field
  * is copied as it is into a deeper texture and every group over a field,
  * and the transfer kernels', is re-created. Between frames only. */

 /** Every tracked group over the fields' current physical textures. */
 private rebind():void{this.bound=this.bound.filter(b=>{const g=b.ref.deref();if(g){g.group=b.build();g.base=b.build(true);}return !!g;});}
 /** Before a host relayout's admit: slots for its patches beside the
  * resident ones (both are live across the remap). */
 reserve(_layout:UniformMixedLayout):void{this.follow();
 }
 /** A GPU relayout admits on its own, and its demand reaches the host only
  * in a later receipt: every patch gets a slot. */
 reserveAll():void{this.expected=undefined;}
 get allocatedBytes():number{
  return this.textures().reduce((n,t)=>n+t.width*t.height*t.depthOrArrayLayers*(t.format==="rgba32float"?16:4),0)+(this.state?.size??0)+(this.ringDimensions?.size??0)
   +(0);
 }
 /** Resident patches: the host directory's while it is current, else the
  * last frame receipt's. */
 get residentPatches():number{

  return this.mirrored||!this.receipt?this.directory.reduce((n,e)=>n+(e?1:0),0):this.receipt.resident;
 }
 /** The table every attached ownership carries (mode 0: raw fields). */
 table():Uint32Array<ArrayBuffer>{
  const l=this.layout,words=new Uint32Array(HEADER+l.patches);

  // Domain: one patch over the lattice, its slot the h texture itself
  // (atlas texel = logical texel); directory word zero: the base blocks.
  // Haloed patches: another mode word, so a consumer through the ABI (which has no halo offset) reads no table.
  words.set([UNIFORM_DETAIL_PACKED,Math.log2(l.patchEdge),...l.patchGrid,...l.tiles]);
  CLASSES.forEach((kind,k)=>{const c=l.classes[kind];words.set([c.spacing,c.atlasZ,c.fold,c.per,...c.base,kind==="face"?UNIFORM_DETAIL_FACE_COMPACT:0],8+8*k);});
  // Domain: the tile box the h store holds (the lattice) and the two-texture flag (uniform-detail-abi.ts).
  words.set([0,0,0,l.tiles[0]-1,l.tiles[1]-1,l.tiles[2]-1,UNIFORM_DETAIL_BASED],32);
  words.set(this.directory,HEADER);return words;
 }
 /** Carry this storage's table in an ownership's topology tail (group 0 of
  * every stage that touches a field through it). */
 attach(ownership:UniformMixedOwnership):void{this.ownerships.add(ownership);storageOf.set(ownership,this);ownership.writeDetail(this.table());}
 /** The table to every attached ownership (between frames, in queue order with the submits around it). */
 private publishTable():void{const table=this.table();for(const o of this.ownerships)o.writeDetail(table);}

 /** The field handles. */
 get fieldTextures():readonly GPUTexture[]{return this.handles;}

 destroy():void{
  live.get(this.device)?.delete(this);this.state?.destroy();this.ringDimensions?.destroy();

  for(const t of this.textures())t.destroy();
 }
 kindOf(texture:GPUTexture):UniformDetailClass{const f=fields.get(texture);if(f?.storage!==this)throw new Error(`${texture.label} is not a field of this detail storage`);return f.kind;}
 private logicalOf(texture:GPUTexture):Dims{return fields.get(texture)!.logical;}

 // Host mirror of the WGSL addressing (udCell/udFace/udVertex/udUnit).
 private baseTexel(kind:UniformDetailClass,b:Dims):[number,number,number]{
  const c=this.layout.classes[kind],s=b[2]%c.per;
  return [b[0]+c.base[0]*(s%c.fold),b[1]+c.base[1]*Math.floor(s/c.fold),Math.floor(b[2]/c.per)];
 }
 /** Physical texel of logical p and its kind: 1 atlas, 0 base, 2 hanging
  * vertex (texel: its home tile), -1 none. */
 locate(kind:UniformDetailClass,p:Dims,directory?:Uint32Array):[number,number,number,number]{
  if(!directory&&!this.mirrored)throw new Error("Uniform detail storage: the GPU changed residency since the host directory; sync() first");
  directory??=this.directory;
  const l=this.layout,shift=Math.log2(l.patchEdge),g=l.patchGrid;
  const home=p.map((v,a)=>kind==="vertex"?Math.min(v>>2,l.tiles[a]!-1):v>>2) as unknown as Dims;
  const patch=home.map(v=>v>>(shift-2)) as unknown as Dims;
  const e=directory[patch[0]+g[0]*(patch[1]+g[1]*patch[2])]!;
  if(e){const c=l.classes[kind],E=c.spacing,r=haloOf(l,kind),slot=[e&1023,(e>>>10)&1023,(e>>>20)&1023];return [slot[0]!*E+r+p[0]-(patch[0]<<shift),slot[1]!*E+r+p[1]-(patch[1]<<shift),c.atlasZ+slot[2]!*E+r+p[2]-(patch[2]<<shift),1];}
  if(kind==="atlas"){const box=undefined;return box?[box[0],box[1],box[2],1]:[0,0,0,-1];}
  if(kind==="vertex"){
   if(p.every(v=>v%4===0))return [...this.baseTexel("vertex",p.map(v=>v>>2) as unknown as Dims),0];
   // Haloed patches: a vertex on a resident patch's face (or in its halo) lives in that box.
   const box=undefined;
   return box?[box[0],box[1],box[2],1]:[...home,2];
  }
  if(kind==="cell")return [...this.baseTexel("cell",home),0];
  const j=p.map(v=>(v&3)===3?1:0);
  return j[0]!+j[1]!+j[2]!>1?[0,0,0,-1]:[p[0]>>2,p[1]>>2,4*(p[2]>>2)+j[0]!+2*j[1]!+3*j[2]!,0];
 }
 /** Haloed patches: p's texel in every resident box (interior plus halo)
  * that holds it, the home patch's first. */

 private places(kind:UniformDetailClass,p:Dims):[number,number,number][]{
  {const at=this.locate(kind,p);return at[3]===0||at[3]===1?[[at[0],at[1],at[2]]]:[];}
 }
 /** Can a non-resident tile hold texel p (the store-side canonical set)? */
 private static canonical(kind:UniformDetailClass,p:Dims):boolean{
  const l=p.map(v=>v&3);
  if(kind==="cell"||kind==="vertex")return l.every(v=>v===0);
  if(kind==="face")return l.every(v=>v===0||v===3)&&l.filter(v=>v===3).length<=1;
  return false;
 }
 /** Whole-field upload of logical values (tight, x fastest, vec4 for
  * faces), as a raw texture would hold them: resident patches take every
  * texel, a non-resident tile its canonical texels' values. Packed uploads
  * before the first layout stay pending: install() deposits them. */
 upload(texture:GPUTexture,values:Float32Array):void{
  const [lx,ly,lz]=this.logicalOf(texture),comp=texture.format==="rgba32float"?4:1;this.kindOf(texture);
  if(values.length!==comp*lx*ly*lz)throw new Error(`${texture.label}: upload needs ${comp*lx*ly*lz} values, not ${values.length}`);

  if(!this.installed){this.pending.set(texture,values);return;}
  this.write(texture,this.image(texture,values));
  if(this.twinned(texture))this.write(texture,this.image(texture,values,true),true);
  // QA: the upload wrote the h texels of every tile.

 }
 /** Domain: the field's h generation is resident, its base a second texture. */
 private twinned(texture:GPUTexture):boolean{return this.resident&&!this.pinned.has(fields.get(texture)!.handle);}
 private installed=false;
 private readonly pending=new Map<GPUTexture,Float32Array>();
 get settled():boolean{return this.installed;}
 private image(texture:GPUTexture,values?:Float32Array,base=false):Float32Array<ArrayBuffer>{
  const kind=this.kindOf(texture),[lx,ly,lz]=this.logicalOf(texture),comp=texture.format==="rgba32float"?4:1,t=base?this.bases.get(fields.get(texture)!.handle)!:this.physical(texture),W=t.width,H=t.height;
  // Domain, resident: the physical texture is the logical one (base: its base block beside it).
  if(values&&!base&&this.twinned(texture)&&!this.windowed)return values as Float32Array<ArrayBuffer>;
  const image=new Float32Array(W*H*t.depthOrArrayLayers*comp);
  if(values&&!base&&this.twinned(texture)){
   // A wrapped store: the box's texels at their wrapped places.
   const b=this.box,Z=t.depthOrArrayLayers,ex=this.boxEnd(kind,b,0),ey=this.boxEnd(kind,b,1),ez=this.boxEnd(kind,b,2);
   for(let z=4*b[2]!;z<=ez;z++)for(let y=4*b[1]!;y<=ey;y++)for(let x=4*b[0]!;x<=ex;x++){
    const dst=(x%W+W*(y%H+H*(z%Z)))*comp,src=(x+lx*(y+ly*z))*comp;
    for(let c=0;c<comp;c++)image[dst+c]=values[src+c]!;
   }
   return image;
  }
  if(values){
   // Base blocks: each texel's canonical source, not a walk of the lattice.
   const compact=this.layout.domain.compactFaces;
   for(let z=0;z<t.depthOrArrayLayers;z++)for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    let p:Dims;
    if(kind!=="face")p=[4*x,4*y,4*z];
    else if(compact){const m=z&3;p=[4*x+(m===1?3:0),4*y+(m===2?3:0),4*(z>>2)+(m===3?3:0)];}
    else{if((x&1)+(y&1)+(z&1)>1)continue;p=[4*(x>>1)+3*(x&1),4*(y>>1)+3*(y&1),4*(z>>1)+3*(z&1)];}
    const dst=(x+W*(y+H*z))*comp,src=(p[0]+lx*(p[1]+ly*p[2]))*comp;
    for(let c=0;c<comp;c++)image[dst+c]=values[src+c]!;
   }
   return image;
  }

  if(values)for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++){
   const p:Dims=[x,y,z],at=this.locate(kind,p);if(at[3]<0||at[3]===2)continue;
   if(at[3]===0&&!UniformDetailStorage.canonical(kind,p))continue;
   const dst=(at[0]+W*(at[1]+H*at[2]))*comp,src=(x+lx*(y+ly*z))*comp;
   for(let c=0;c<comp;c++)image[dst+c]=values[src+c]!;
  }
  return image;
 }
 private write(texture:GPUTexture,image:Float32Array<ArrayBuffer>,base=false):void{
  const comp=texture.format==="rgba32float"?4:1,t=base?this.bases.get(fields.get(texture)!.handle)!:this.physical(texture);
  this.device.queue.writeTexture({texture:t},image,{bytesPerRow:t.width*comp*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
 }

 install(layout:UniformMixedLayout,live:{readonly volume:GPUTexture;readonly velocity:GPUTexture;readonly velocityScratch:GPUTexture}):boolean{
  if(this.installed)throw new Error("Uniform detail storage installs its first layout once");
  this.installed=true;

  let all:boolean;
  {
   // Nothing is deposited yet: an h capacity takes the uploads as they are.
   all=this.held()>0;if(all){this.installing=true;try{this.admit();}finally{this.installing=false;}}
  }
  if(this.pending.has(live.velocityScratch))throw new Error("The scratch velocity has no initial upload");
  for(const [texture,values] of this.pending){
   const image=this.image(texture,values);
   if(!all&&texture===live.volume)this.coarsenCells(layout,values,image,texture);
   if(!all&&texture===live.velocity){
    this.coarsenFaces(layout,values,image,texture);
    const scratch=this.image(live.velocityScratch);this.coarsenFaces(layout,values,scratch,live.velocityScratch);this.write(live.velocityScratch,scratch);
   }
   this.write(texture,image);
   // An h capacity: the base takes the upload's canonical texels beside it.
   if(this.twinned(texture))this.write(texture,this.image(texture,values,true),true);
  }
  this.pending.clear();
  return all;
 }
 private coarse(layout:UniformMixedLayout,visit:(o:Dims)=>void):void{
  const [tx,ty]=this.layout.tiles;
  layout.tiles.forEach((word,t)=>{if(!(word&0x80000000))visit([4*(t%tx),4*(Math.floor(t/tx)%ty),4*Math.floor(t/(tx*ty))]);});
 }
 private coarsenCells(layout:UniformMixedLayout,values:Float32Array,image:Float32Array,texture:GPUTexture):void{
  const [lx,ly]=this.logicalOf(texture),W=this.physical(texture).width,H=this.physical(texture).height;
  this.coarse(layout,o=>{
   let sum=0;for(let k=0;k<64;k++)sum=Math.fround(sum+values[(o[0]+(k&3))+lx*((o[1]+((k>>2)&3))+ly*(o[2]+(k>>4)))]!);
   for(const at of this.places("cell",o))image[at[0]+W*(at[1]+H*at[2])]=Math.fround(sum/64);
  });
 }
 private coarsenFaces(layout:UniformMixedLayout,values:Float32Array,image:Float32Array,texture:GPUTexture):void{
  const [lx,ly]=this.logicalOf(texture),W=this.physical(texture).width,H=this.physical(texture).height,d=this.layout.dims;
  const at=(q:readonly number[])=>4*(q[0]!+lx*(q[1]!+ly*q[2]!));
  // The 16 h owners of a plane of the tile: axis a at local coordinate c.
  const plane=(o:Dims,a:number,c:number)=>{const u=(a+1)%3,v=(a+2)%3,out:number[][]=[];
   for(let y=0;y<4;y++)for(let x=0;x<4;x++){const q=[...o];q[a]=o[a]!+c;q[u]=o[u]!+x;q[v]=o[v]!+y;out.push(q);}return out;};
  const held=(qs:number[][],bit:number)=>qs.every(q=>(Math.round(values[at(q)+3]!)>>bit)&1);
  this.coarse(layout,o=>{
   let walls=0;for(let b=0;b<3;b++)if(o[b]===0&&held(plane(o,b,0),b+3))walls|=1<<(b+3);
   for(let a=0;a<3;a++){
    if(o[a]+3>=d[a]!)continue;
    const qs=plane(o,a,3);let sum=0;for(const q of qs)sum=Math.fround(sum+values[at(q)+a]!);
    const anchor=[...o];anchor[a]+=3;
    for(const t of this.places("face",anchor as unknown as Dims)){const i=4*(t[0]+W*(t[1]+H*t[2]));image.fill(0,i,i+4);image[i+a]=Math.fround(sum/16);image[i+3]=walls|(held(qs,a)?1<<a:0);}
   }
  });
 }

 // The domain placement (UniformDetailDomain): one generation of every field at a time.
 /** The fields are their logical h textures (else their base blocks). */
 private resident=false;
 /** The ownership whose h-tile capacity residency follows (prepare). */
 private simulation?:UniformMixedOwnership;
 private transfers?:DomainKernels;
 /** The capacity is zero under a resident generation: the remap that
  * coarsens its last h tiles still reads it (encodeRetire). */
 private retiring=false;
 /** That remap is encoded: commitRetire drops the generation. */
 private retired=false;
 private held():number{
  if(!this.simulation)throw new Error("Uniform detail storage: the domain placement follows a prepared ownership's capacity");
  return this.simulation.capacity.fineTiles;
 }
 /** Vertex fields whose base block is a texture the caller owns (adoptBase). */
 private readonly adopted=new Map<GPUTexture,GPUTexture>();
 /** Makes a caller's (t+1)³ texture the base block of a vertex field: the
  * field itself while C = 0, the target of its restriction at retirement,
  * and the caller's to keep current while the h generation is resident
  * (the published coarse vertex phi). Before the first layout. The texture
  * stays the caller's raw texture to every reader (uniformDetailField does
  * not map it): the field is read through its handle. */
 adoptBase(handle:GPUTexture,base:GPUTexture):void{
  const field=fields.get(handle),need=GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
  if(field?.kind!=="vertex"||this.installed||this.resident)throw new Error("Uniform detail storage: only a domain placement adopts a vertex base, before its first layout");
  const e=this.extent("vertex",false);
  if(base.format!=="r32float"||base.width!==e[0]||base.height!==e[1]||base.depthOrArrayLayers!==e[2]||(base.usage&need)!==need)
   throw new Error(`Uniform detail storage: an adopted vertex base is an r32float ${e.join("x")} texture with storage, sampled and both copy usages`);
  const own=this.physicals.get(handle)!;
  this.physicals.set(handle,base);this.bases.set(handle,base);this.adopted.set(handle,base);own.destroy();this.rebind();
 }
 /** The textures of a generation the storage owns (an adopted base is its caller's). */
 private owned(textures:Iterable<GPUTexture>):GPUTexture[]{const kept=new Set(this.adopted.values());return [...textures].filter(t=>!kept.has(t));}
 /** Whether the fields are base blocks (the domain placement at C = 0). */
 get baseOnly():boolean{return !this.resident;}
 /** Bytes of the field textures at an h-tile capacity (the frame's
  * admission check): only the domain placement depends on it. */
 fieldBytesAt(fineTiles:number):number{
  return this.handles.reduce((n,h)=>{
   const t=this.physicals.get(h)!,texel=t.format==="rgba32float"?16:4,size=(e:readonly number[])=>e[0]!*e[1]!*e[2]!*texel;

   // The base at every capacity, the h generation beside it above zero.
   const kind=fields.get(h)!.kind;
   return n+size(this.extent(kind,false))+(fineTiles>0&&!this.pinned.has(h)?size(this.extent(kind,true)):0);
  },0);
 }
 /** Between frames, after the frame's reserveFine: residency follows the
  * capacity. Leaving zero admits here, before the remap that needs the h
  * texels; returning to zero only marks the generation, which the remap
  * still reads (encodeRetire, commitRetire). */
 private follow():void{
  const want=this.held()>0;
  if(want&&!this.resident)this.admit();
  // A retirement that was never committed: the h generation stays.
  if(want)this.retired=false;
  this.retiring=!want&&this.resident;
 }

 /** The previous generation's textures, destroyed once the work submitted
  * on them is done. */
 private release(old:GPUTexture[]):void{const gone=this.owned(old);void this.device.queue.onSubmittedWorkDone().then(()=>{for(const t of gone)t.destroy();});}
 /** One launch over a base block's texels (base: the target of a refresh). */
 private transfer(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline,source:GPUTexture,target:GPUTexture,base:GPUTexture):void{
  pass.setPipeline(pipeline);
  pass.setBindGroup(0,this.device.createBindGroup({layout:this.kernelsOfDomain().layouts[target.format as "r32float"|"rgba32float"],entries:[{binding:0,resource:source.createView()},{binding:1,resource:target.createView()}]}));
  pass.dispatchWorkgroups(Math.ceil(base.width/4),Math.ceil(base.height/4),Math.ceil(base.depthOrArrayLayers/4));
 }
 /** C leaves zero: every following field becomes its h texture, beside its
  * base. Nothing is moved here: no tile is in the ring yet, and the
  * relayout that brings tiles into it seeds their canonical texels from
  * the bases (encodeAdmit). Every other texel starts zero and is written
  * by the kernel that makes it state (the remap for a tile it refines, the
  * seam kernels for a 4h tile beside one) before any reader; at the first
  * layout the h textures take its uploads (install). */
 private admit():void{
  for(const handle of this.following){
   const base=this.bases.get(handle)!,field=fields.get(handle)!,after=this.allocate(handle.label,field.kind,base.format as GPUTextureFormat,true);
   this.physicals.set(handle,after);fields.set(after,field);registerUniformDetailBase(after,base);
  }
  this.resident=true;this.directory[0]=entry([0,0,0]);this.writeState();this.publishTable();this.rebind();
  // QA: a fresh h generation is poison throughout (the first layout's uploads fill it instead).

 }
 private installing=false;
 private ringGroups=new WeakMap<GPUBuffer,{key:GPUTexture;groups:GPUBindGroup[]}>();
 private other():UniformMixedOwnership{
  const others=[...this.ownerships].filter(o=>o!==this.simulation);
  if(others.length!==1)throw new Error(`Uniform detail storage: the ring hands its bit to one other ownership (the pressure root), not ${others.length}`);
  return others[0]!;
 }
 /** One launch per batch, a lane per tile: the fields' base blocks and h textures, the held and target topologies. */
 private encodeRing(encoder:GPUCommandEncoder,target:GPUBuffer,pick:(b:RingBatch)=>GPUComputePipeline|undefined,label:string):void{
  const ring=this.kernelsOfDomain().ring,held=this.simulation!.presentation.buffer;
  if(!ring)throw new Error("Uniform detail storage: no ring kernels (the storage is not prepared)");
  const first=this.physicals.get(ring.batches[0]!.handles[0]!)!;let bound=this.ringGroups.get(target);
  if(!bound||bound.key!==first){
   const other=this.other().presentation.buffer;
   bound={key:first,groups:ring.batches.map((b,n)=>this.device.createBindGroup({label:"Uniform detail ring",layout:b.layout,entries:[
    ...b.handles.flatMap((h,i)=>[{binding:i,resource:this.bases.get(h)!.createView()},{binding:BATCH+i,resource:this.physicals.get(h)!.createView()}]),
    {binding:2*BATCH,resource:{buffer:held}},{binding:2*BATCH+1,resource:{buffer:target}},{binding:2*BATCH+4,resource:{buffer:this.ringDimensions!}},...(n===0?[{binding:2*BATCH+2,resource:{buffer:other}},{binding:2*BATCH+3,resource:{buffer:this.state!}}]:[])]}))};
   this.ringGroups.set(target,bound);
  }
  const tiles=this.layout.tiles[0]*this.layout.tiles[1]*this.layout.tiles[2],pass=encoder.beginComputePass({label});
  ring.batches.forEach((b,n)=>{const p=pick(b);if(!p)throw new Error(`Uniform detail storage: ${label} is not compiled`);pass.setPipeline(p);pass.setBindGroup(0,bound!.groups[n]!);pass.dispatchWorkgroups(Math.ceil(tiles/64));});
  pass.end();
 }

 private poisonTarget?:GPUBuffer;

 encodeSettle(encoder:GPUCommandEncoder,share:boolean):void{

  if(share){
   const ring=this.kernelsOfDomain().ring!,tiles=this.layout.tiles[0]*this.layout.tiles[1]*this.layout.tiles[2];
   const pass=encoder.beginComputePass({label:"Uniform detail ring share"});pass.setPipeline(ring.share);
   pass.setBindGroup(0,this.shareGroup??=this.device.createBindGroup({label:"Uniform detail ring share",layout:ring.shareLayout,entries:[{binding:2,resource:{buffer:this.ringDimensions!}},{binding:0,resource:{buffer:this.simulation!.presentation.buffer}},{binding:1,resource:{buffer:this.other().presentation.buffer}}]}));
   pass.dispatchWorkgroups(Math.ceil(tiles/64));pass.end();
  }

 }
 private shareGroup?:GPUBindGroup;
 /** C returned to zero under a resident generation: the caller's encoder
  * holds the remap that coarsened its last h tiles. Nothing is moved: the
  * bases already hold every canonical texel. The fields stay the h textures
  * until commitRetire. */
 private encodeRestriction(_encoder:GPUCommandEncoder):void{
  if(this.retiring)this.retired=true;
 }
 /** After the encoder of encodeRetire is submitted: the bases are the
  * fields, every group binds them, and the h textures go once the work
  * that read them is done. Without a pending retirement, nothing. */
 commitRetire():void{
  if(!this.retired)return;
  this.retired=false;this.retiring=false;const old:GPUTexture[]=[];
  for(const h of this.following){old.push(this.physicals.get(h)!);this.physicals.set(h,this.bases.get(h)!);}
  this.resident=false;this.directory[0]=0;this.writeState();this.publishTable();this.rebind();this.release(old);
 }
 /** The bases from the h generation, canonical texel for canonical texel:
  * after kernels that store through no twin (the dense t=0 authority and
  * publication). Nothing while the fields are their bases. */
 encodeBaseRefresh(encoder:GPUCommandEncoder,written:readonly GPUTexture[]):void{
  if(!this.resident)return;
  if(this.windowed)throw new Error("Uniform detail storage: the dense t=0 kernels address the lattice, and the h store is a window of it");
  // Only the fields those kernels stored: every other field's h texture
  // holds no current canonical texel outside the ring, and its base is
  // current already.
  const wrote=new Set(written.map(t=>fields.get(t)?.handle));
  const k=this.kernelsOfDomain().refresh,pass=encoder.beginComputePass({label:"Uniform detail base refresh"});
  for(const h of this.following){if(!wrote.has(h))continue;const base=this.bases.get(h)!;this.transfer(pass,k[fields.get(h)!.kind as TransferKind],this.physicals.get(h)!,base,base);}
  pass.end();
  // QA: those kernels wrote the h texels of every tile.

 }
 private kernelsOfDomain():DomainKernels{return this.transfers!;}
 /** refresh (h to base) and the ring kernels are built with the storage:
  * any state with h tiles may run them. */
 private async compileDomain(domain:UniformDetailDomain):Promise<DomainKernels>{
  const device=this.device,C=GPUShaderStage.COMPUTE;
  const layout=(format:"r32float"|"rgba32float")=>device.createBindGroupLayout({label:`Uniform detail domain transfer ${format}`,entries:[
   {binding:0,visibility:C,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:1,visibility:C,storageTexture:{access:"write-only",format,viewDimension:"3d"}}]});
  const head=(format:string)=>`@group(0) @binding(0) var src:texture_3d<f32>;\n@group(0) @binding(1) var dst:texture_storage_3d<${format},write>;\n`;
  // One lane per base texel: the base is dst of a refresh.
  const kernel=(name:string,base:"src"|"dst",body:string)=>`@compute @workgroup_size(4,4,4) fn ${name}(@builtin(global_invocation_id) g:vec3u){\n if(any(g>=textureDimensions(${base}))){return;}\n${body}\n}\n`;
  const scalar=head("r32float")+/* wgsl */`
// A tile's origin cell, a tile corner: base texel g is h texel 4g.
${kernel("refresh","dst"," textureStore(dst,vec3i(g),textureLoad(src,vec3i(4u*g),0));")}`;
  const face=head("rgba32float")+(`
// Base texel m of a tile is its origin (0) or its +face anchor of axis m-1.
fn ddFace(g:vec3u)->vec3i{let m=g.z&3u;let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));return vec3i(4u*vec3u(g.xy,g.z>>2u)+3u*j);}
${kernel("refresh","dst"," textureStore(dst,vec3i(g),textureLoad(src,ddFace(g),0));")}`);
  const layouts={r32float:layout("r32float"),rgba32float:layout("rgba32float")},compiler=gpuCompilationManagerFor(device);
  const modules={r32float:device.createShaderModule({label:"Uniform detail domain scalar transfer",code:scalar}),rgba32float:device.createShaderModule({label:"Uniform detail domain face transfer",code:face})};
  const pipeline=(format:"r32float"|"rgba32float",entryPoint:string)=>compiler.compileComputePipeline({label:`Uniform detail domain ${entryPoint} ${format}`,
   layout:device.createPipelineLayout({bindGroupLayouts:[layouts[format]]}),compute:{module:modules[format],entryPoint}},{priority:"visible"});
  const pair=async(entryPoint:string):Promise<Record<TransferKind,GPUComputePipeline>>=>{
   const [scalarPipeline,facePipeline]=await Promise.all([pipeline("r32float",entryPoint),pipeline("rgba32float",entryPoint)]);
   return {cell:scalarPipeline,vertex:scalarPipeline,face:facePipeline};
  };
  return {layouts,refresh:await pair("refresh"),ring:await this.compileRing(domain)};
 }
 /** The ring kernels, a lane per tile. enter:
  * a tile the target generation brings into the ring takes its canonical
  * texels from the base blocks (the h texture held nothing current for it).
  * The first batch also hands the target's ring bit to the other ownership
  * (the all-4h pressure root, whose kernels store these fields too). */
 private async compileRing(domain:UniformDetailDomain):Promise<DomainKernels["ring"]>{
  const device=this.device,C=GPUShaderStage.COMPUTE,T=this.layout.tiles,N=T[0]*T[1]*T[2],compiler=gpuCompilationManagerFor(device);
  if(Math.ceil(N/64)>device.limits.maxComputeWorkgroupsPerDimension)throw new Error(`Uniform detail storage: the ring kernels launch ${Math.ceil(N/64)} workgroups, past one dispatch dimension`);
  const following=this.following,sets:GPUTexture[][]=[];
  for(const format of ["rgba32float","r32float"] as const){const of=following.filter(h=>h.format===format);for(let i=0;i<of.length;i+=BATCH)sets.push(of.slice(i,i+BATCH));}
  const head=`@group(0) @binding(${2*BATCH}) var<storage,read> drHeldWords:array<u32>;
@group(0) @binding(${2*BATCH+1}) var<storage,read> drNextWords:array<u32>;
@group(0) @binding(${2*BATCH+4}) var<uniform> drDimensions:vec4u;
fn drTiles()->vec3u{return drDimensions.xyz;}
fn drCount()->u32{return drDimensions.x*drDimensions.y*drDimensions.z;}
const DR_P:f32=${UNIFORM_DETAIL_POISON_WGSL};
// Logical texel p in an h store of extent d, which wraps on an axis shorter than the lattice.
fn drAt(p:vec3u,d:vec3u)->vec3i{return vec3i(p&select(d-vec3u(1u),vec3u(0xffffffffu),d>=4u*drTiles()));}
fn drHeld(t:u32)->bool{return (drHeldWords[2u*drCount()+2u*t+1u]&1u)!=0u;}
fn drNext(t:u32)->bool{return (drNextWords[2u*drCount()+2u*t+1u]&1u)!=0u;}
fn drCoord(t:u32)->vec3u{return vec3u(t%drTiles().x,(t/drTiles().x)%drTiles().y,t/(drTiles().x*drTiles().y));}
`;
  const batches=await Promise.all(sets.map(async(handles,n):Promise<RingBatch>=>{
   const format=handles[0]!.format as "r32float"|"rgba32float";
   const layout=device.createBindGroupLayout({label:`Uniform detail ring ${n}`,entries:[
    ...handles.flatMap((_,i)=>[{binding:i,visibility:C,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}},{binding:BATCH+i,visibility:C,storageTexture:{access:"write-only" as const,format,viewDimension:"3d" as const}}]),
    {binding:2*BATCH+4,visibility:C,buffer:{type:"uniform" as const}},
    {binding:2*BATCH,visibility:C,buffer:{type:"read-only-storage" as const}},{binding:2*BATCH+1,visibility:C,buffer:{type:"read-only-storage" as const}},
    ...(n===0?[{binding:2*BATCH+2,visibility:C,buffer:{type:"storage" as const}},{binding:2*BATCH+3,visibility:C,buffer:{type:"storage" as const}}]:[])]});
   let seed="",fill="";
   handles.forEach((h,i)=>{
    const kind=fields.get(h)!.kind;
    if(kind==="cell"){
     seed+=` textureStore(dst${i},drAt(4u*c,textureDimensions(dst${i})),textureLoad(src${i},vec3i(c),0));\n`;
     fill+=` for(var k=0u;k<64u;k++){textureStore(dst${i},drAt(4u*c+vec3u(k&3u,(k>>2u)&3u,k>>4u),textureDimensions(dst${i})),vec4f(DR_P));}\n`;
    }else if(kind==="vertex"){
     // A corner's home is min(g, T - 1): the far corners belong to the last tile of each axis.
     seed+=` for(var k=0u;k<8u;k++){let e=vec3u(k&1u,(k>>1u)&1u,k>>2u);if(any((e==vec3u(1u))&(c!=drTiles()-vec3u(1u)))){continue;}textureStore(dst${i},drAt(4u*(c+e),textureDimensions(dst${i})),textureLoad(src${i},vec3i(c+e),0));}\n`;
     fill+=` for(var k=0u;k<125u;k++){let e=vec3u(k%5u,(k/5u)%5u,k/25u);if(any((e==vec3u(4u))&(c!=drTiles()-vec3u(1u)))){continue;}textureStore(dst${i},drAt(4u*c+e,textureDimensions(dst${i})),vec4f(DR_P));}\n`;
    }else{
     seed+=` for(var m=0u;m<4u;m++){let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));textureStore(dst${i},drAt(4u*c+3u*j,textureDimensions(dst${i})),textureLoad(src${i},vec3i(vec3u(c.xy,4u*c.z+m)),0));}\n`;
     fill+=` for(var k=0u;k<64u;k++){textureStore(dst${i},vec3i(4u*c+vec3u(k&3u,(k>>2u)&3u,k>>4u)),vec4f(DR_P));}\n`;
    }
   });
   const code=handles.map((_,i)=>`@group(0) @binding(${i}) var src${i}:texture_3d<f32>;\n@group(0) @binding(${BATCH+i}) var dst${i}:texture_storage_3d<${format},write>;\n`).join("")+head
    +(n===0?`@group(0) @binding(${2*BATCH+2}) var<storage,read_write> drShared:array<u32>;\n@group(0) @binding(${2*BATCH+3}) var<storage,read_write> drState:array<atomic<u32>>;\n`:"")+`
@compute @workgroup_size(64) fn enter(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=drCount()){return;}
 let near=drNext(t);
${n===0?` let word=2u*drCount()+2u*t+1u;let w=drShared[word];let m=(w&0xfffffffeu)|u32(near);if(m!=w){drShared[word]=m;}
 // A ring tile outside the h store's window has no texel there: fatal at the frame receipt.
 if(near){let o=drCoord(t);let lo=vec3u(atomicLoad(&drState[drDimensions.w]),atomicLoad(&drState[drDimensions.w+1u]),atomicLoad(&drState[drDimensions.w+2u]));let hi=vec3u(atomicLoad(&drState[drDimensions.w+3u]),atomicLoad(&drState[drDimensions.w+4u]),atomicLoad(&drState[drDimensions.w+5u]));
  if(any(o<lo)||any(o>hi)){atomicOr(&drState[${D.fatal}u],${WINDOW_FATAL}u);}}
`:""} if(!near||drHeld(t)){return;}
 let c=drCoord(t);
${seed}}
fn drFill(c:vec3u){
${fill}}
@compute @workgroup_size(64) fn poison(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=drCount()||drHeld(t)){return;}
 drFill(drCoord(t));
}
@compute @workgroup_size(64) fn poisonAll(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=drCount()){return;}
 drFill(drCoord(t));
}
`;
   const module=device.createShaderModule({label:`Uniform detail ring ${n}`,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const pipeline=(entryPoint:string)=>compiler.compileComputePipeline({label:`Uniform detail ring ${entryPoint} ${n}`,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}},{priority:"visible"});
   const [enter,poison,poisonAll]=await Promise.all([pipeline("enter"),undefined,undefined]);
   return {handles,layout,enter,poison,poisonAll};
  }));
  const shareLayout=device.createBindGroupLayout({label:"Uniform detail ring share",entries:[{binding:2,visibility:C,buffer:{type:"uniform"}},{binding:0,visibility:C,buffer:{type:"read-only-storage"}},{binding:1,visibility:C,buffer:{type:"storage"}}]});
  const share=await compiler.compileComputePipeline({label:"Uniform detail ring share",layout:device.createPipelineLayout({bindGroupLayouts:[shareLayout]}),compute:{module:device.createShaderModule({label:"Uniform detail ring share",code:`
@group(0) @binding(2) var<uniform> drDimensions:vec4u;
@group(0) @binding(0) var<storage,read> drFrom:array<u32>;
@group(0) @binding(1) var<storage,read_write> drTo:array<u32>;
@compute @workgroup_size(64) fn share(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=drDimensions.x*drDimensions.y*drDimensions.z){return;}
 let word=2u*drDimensions.x*drDimensions.y*drDimensions.z+2u*t+1u;let w=drTo[word];let m=(w&0xfffffffeu)|(drFrom[word]&1u);if(m!=w){drTo[word]=m;}
}`}),entryPoint:"share"}},{priority:"visible"});
  return {batches,shareLayout,share};
 }

 async prepare(ownership:UniformMixedOwnership):Promise<void>{if(this.transfers)return;this.simulation=ownership;this.transfers=await this.compileDomain(this.layout.domain!);}
 /** The transfer kernels' groups over the fields' physical textures, in
  * batches of BATCH per class (spare bindings take 1³ dummies), and the
  * face staging for the allocated slots. */

 /** Before a remap, in its encoder: admit the patches of the target
  * generation (target: its tile words, one per tile from word 0) beside the
  * resident ones, fill them under the directory the fields were written
  * with, then publish the union directory. */
 encodeAdmit(encoder:GPUCommandEncoder,target:GPUBuffer):void{if(this.resident)this.encodeRing(encoder,target,b=>b.enter,"Uniform detail ring enter");}
 /** After the remap and its phi resolve, in their encoder: restrict the
  * patches the last encodeAdmit found unneeded into the base, free their
  * slots and publish the directory without them. */
 encodeRetire(encoder:GPUCommandEncoder):void{this.encodeRestriction(encoder);}
 /** The residency header into a frame receipt (UNIFORM_DETAIL_RECEIPT_BYTES). */
 encodeReceipt(encoder:GPUCommandEncoder,readback:GPUBuffer,offset:number):void{
  if(this.state)encoder.copyBufferToBuffer(this.state,0,readback,offset,UNIFORM_DETAIL_RECEIPT_BYTES);
  else encoder.clearBuffer(readback,offset,UNIFORM_DETAIL_RECEIPT_BYTES);
 }
 /** A frame receipt's residency words: an overflowed relayout is fatal. */
 noteReceipt(words:Uint32Array):void{

  if(words[D.fatal]!&WINDOW_FATAL)throw new Error(`Uniform detail storage: a relayout put a detail-ring tile outside the h store's window (tiles ${this.box.slice(0,3)} to ${this.box.slice(3)}, store ${this.cells} cells): the host's bound on the h tiles did not hold`);
  if(words[D.fatal])throw new Error(`Uniform detail storage needs ${words[D.demand]} patch slots; capacity is ${words[D.capacity]}`);
  this.receipt={capacity:words[D.capacity]!,resident:words[D.resident]!,peak:words[D.peak]!};
 }
 /** Refresh the host directory from the GPU (diagnostics and readbacks). */

 /** Haloed patches, admission before adoption (the frame's updateLayout,
  * ahead of reserveFine): a layout whose patches outrun a fixed pool, or
  * (slice step 2: no promotion or retirement) are not the resident ones, is
  * refused with nothing changed. */

 /** A field's physical texels (tight rows). */

 /** Logical readback (tight, as upload takes it): loads as the accessors do. */
 async read(texture:GPUTexture):Promise<Float32Array>{
  const encoder=this.device.createCommandEncoder({label:"Uniform detail field readback"}),done=this.capture(encoder,texture);
  this.device.queue.submit([encoder.finish()]);return done();
 }
 /** A field's logical values at this point of an encoder (a mid-frame
  * diagnostic capture): the physical texels and the directory they are
  * placed by. Resolve the result after the encoder is submitted. */
 capture(encoder:GPUCommandEncoder,texture:GPUTexture):()=>Promise<Float32Array>{
  const kind=this.kindOf(texture),[lx,ly,lz]=this.logicalOf(texture),comp=texture.format==="rgba32float"?4:1;
  const identity=false,source=this.physical(texture);
  const W=source.width,H=source.height,Z=source.depthOrArrayLayers,row=W*comp*4,pitch=Math.ceil(row/256)*256,patches=this.layout.patches;
  const usage=GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ;
  // Domain: residency is the host's, so the directory of this encode is known here.
  const hosted=(this.pinned.has(fields.get(texture)!.handle)?new Uint32Array(1):this.directory.slice());
  const staging=this.device.createBuffer({size:pitch*H*Z,usage}),placed=hosted?undefined:this.device.createBuffer({size:4*patches,usage});
  encoder.copyTextureToBuffer({texture:source},{buffer:staging,bytesPerRow:pitch,rowsPerImage:H},[W,H,Z]);
  // A wrapped store: the box whose texels it holds at this encode.
  const wrapped=this.twinned(texture)&&this.windowed?[...this.box]:undefined;
  // Outside the ring a canonical texel lives in the base alone, so every one is read from there.
  const base=this.twinned(texture)?this.bases.get(fields.get(texture)!.handle)!:undefined;
  const bw=base?.width??0,bh=base?.height??0,bd=base?.depthOrArrayLayers??0,brow=bw*comp*4,bpitch=Math.ceil(brow/256)*256;
  const beside=base?this.device.createBuffer({size:bpitch*bh*bd,usage}):undefined;
  if(base)encoder.copyTextureToBuffer({texture:base},{buffer:beside!,bytesPerRow:bpitch,rowsPerImage:bh},[bw,bh,bd]);
  if(placed)encoder.copyBufferToBuffer(this.state!,4*this.words.dir,placed,0,4*patches);
  return async()=>{
   let image:Float32Array,directory:Uint32Array|undefined=hosted;
   try{
    await Promise.all([staging.mapAsync(GPUMapMode.READ),placed?.mapAsync(GPUMapMode.READ),beside?.mapAsync(GPUMapMode.READ)]);
    const src=new Uint8Array(staging.getMappedRange()),dst=new Uint8Array(row*H*Z);
    for(let r=0;r<H*Z;r++)dst.set(src.subarray(r*pitch,r*pitch+row),r*row);
    image=new Float32Array(dst.buffer);if(placed)directory=new Uint32Array(placed.getMappedRange()).slice();
    if(wrapped){
     // The logical field: the box's texels from their wrapped places, zero elsewhere (no h tile is there; the canonical texels follow from the base).
     const full=new Float32Array(lx*ly*lz*comp),b=wrapped,ex=this.boxEnd(kind,b,0),ey=this.boxEnd(kind,b,1),ez=this.boxEnd(kind,b,2);
     for(let z=4*b[2]!;z<=ez;z++)for(let y=4*b[1]!;y<=ey;y++)for(let x=4*b[0]!;x<=ex;x++){
      const from=(x%W+W*(y%H+H*(z%Z)))*comp,to=(x+lx*(y+ly*z))*comp;
      for(let c=0;c<comp;c++)full[to+c]=image[from+c]!;
     }
     image=full;
    }
    if(beside){
     const packed=new Uint8Array(beside.getMappedRange()),tight=new Uint8Array(brow*bh*bd);
     for(let r=0;r<bh*bd;r++)tight.set(packed.subarray(r*bpitch,r*bpitch+brow),r*brow);
     const held=new Float32Array(tight.buffer),compact=this.layout.domain!.compactFaces;
     for(let z=0;z<bd;z++)for(let y=0;y<bh;y++)for(let x=0;x<bw;x++){
      let p:Dims;
      if(kind!=="face")p=[4*x,4*y,4*z];
      else if(compact){const m=z&3;p=[4*x+(m===1?3:0),4*y+(m===2?3:0),4*(z>>2)+(m===3?3:0)];}
      else{if((x&1)+(y&1)+(z&1)>1)continue;p=[4*(x>>1)+3*(x&1),4*(y>>1)+3*(y&1),4*(z>>1)+3*(z&1)];}
      const to=(p[0]+lx*(p[1]+ly*p[2]))*comp,from=(x+bw*(y+bh*z))*comp;
      for(let c=0;c<comp;c++)image[to+c]=held[from+c]!;
     }
    }
   }finally{for(const b of [staging,placed,beside]){if(b?.mapState==="mapped")b.unmap();b?.destroy();}}
   return this.logical(kind,[lx,ly,lz],comp,image,[W,H],directory!);
  };
 }
 private logical(kind:UniformDetailClass,[lx,ly,lz]:Dims,comp:number,image:Float32Array,[W,H]:readonly [number,number],directory:Uint32Array):Float32Array{
  return this.logicalDomain(kind,[lx,ly,lz],comp,image,[W,H],directory[0]!==0);
 }
 /** logical() for the domain placement, the same values without a locate()
  * per texel (a lane reads n³ fields at every sampled frame): the h
  * generation is the image; a base block broadcasts its texels as the
  * accessors load them. */
 private logicalDomain(kind:UniformDetailClass,[lx,ly,lz]:Dims,comp:number,image:Float32Array,[W,H]:readonly [number,number],resident:boolean):Float32Array{
  if(resident)return image;
  const out=new Float32Array(lx*ly*lz*comp);if(kind==="atlas")return out;
  const f=Math.fround,compact=this.layout.domain!.compactFaces,[tx,ty,tz]=this.layout.tiles,v=new Float64Array(8);
  for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++){
   const dst=(x+lx*(y+ly*z))*comp;let src:number;
   if(kind==="cell")src=(x>>2)+W*((y>>2)+H*(z>>2));
   else if(kind==="face"){
    const jx=(x&3)===3?1:0,jy=(y&3)===3?1:0,jz=(z&3)===3?1:0;
    if(compact){if(jx+jy+jz>1)continue;src=(x>>2)+W*((y>>2)+H*(4*(z>>2)+jx+2*jy+3*jz));}
    else src=2*(x>>2)+jx+W*(2*(y>>2)+jy+H*(2*(z>>2)+jz));
   }else if(!((x|y|z)&3))src=(x>>2)+W*((y>>2)+H*(z>>2));
   else{
    // Hanging vertex: trilinear in its home tile's corners, summed in umVertexSum8 order.
    const hx=Math.min(x>>2,tx-1),hy=Math.min(y>>2,ty-1),hz=Math.min(z>>2,tz-1),ux=(x-4*hx)/4,uy=(y-4*hy)/4,uz=(z-4*hz)/4;
    for(let k=0;k<8;k++){
     const cx=k&1,cy=(k>>1)&1,cz=k>>2,weight=f(f((cx?ux:1-ux)*(cy?uy:1-uy))*(cz?uz:1-uz));
     v[k]=weight>0?f(weight*image[(hx+cx+W*(hy+cy+H*(hz+cz)))*comp]!):0;
    }
    out[dst]=f(f(f(v[0]!+v[5]!)+f(v[1]!+v[4]!))+f(f(v[2]!+v[7]!)+f(v[3]!+v[6]!)));continue;
   }
   src*=comp;for(let c=0;c<comp;c++)out[dst+c]=image[src+c]!;
  }
  return out;
 }
}

/** Field class by binding name in the mixed shaders. `output` and `field`
 * depend on their texel type (scalar cell vs face velocity; the phi resolve
 * field is a vertex field). Raw names are tile-resolution caches or independent render fields. */
const CLASS_BY_NAME:Record<string,UniformDetailClass>={
 volume:"cell",outputVolume:"cell",phase:"cell",centerPhi:"cell",targetFill:"cell",correction:"cell",fineCenterPhi:"cell",
 fineVolume:"cell",pressureCenterPhi:"cell",pressureTarget:"cell",curvature:"cell",
 velocity:"face",physical:"face",extended:"face",advected:"face",fineVelocity:"face",departure:"face",departures:"face",
 copied:"face",bodyVelocity:"face",topology:"face",topologyOut:"face",
 phi:"vertex",outputPhi:"vertex",vertexPhi:"vertex",bodyPhi:"vertex",
 unitVelocity:"atlas",unitVelocityOut:"atlas",
};
const RAW_NAMES=new Set(["coarse","coarseOut","coarseExtended","coarseVelocity","origins","originsOut","surfacePhi","surfaceOpen"]);
function classify(name:string,type:string):UniformDetailClass|undefined{
 if(RAW_NAMES.has(name))return undefined;
 if(name==="output")return type.includes("rgba")?"face":"cell";
 if(name==="field")return type.startsWith("texture_storage")?"vertex":"face";
 const kind=CLASS_BY_NAME[name];
 if(!kind)throw new Error(`Uniform detail rewrite: unclassified 3D texture ${name}`);
 return kind;
}

/** Address helpers for a packed layout: constants baked in, the directory
 * (topology tail) the only run-time read. Each returns a physical texel and
 * w = 1 atlas, 0 base, 2 hanging vertex (xyz: its home tile), -1 none (an
 * atlas-only field of a non-resident tile). */

const UNIFORM_DETAIL_POISON_WGSL="-0x1p+100f";

/** The domain placement's accessors (UniformDetailDomain). b = 1 while the
 * bound texture is the field's base block, read from its extent, so the
 * address is arithmetic on a run-time value with no branch in a load:
 *  cell, vertex  q >> 2b
 *  face          q >> b: a canonical face's base parity texel 2c + (l==3)
 *                (compact: z = (q.z >> 2b)(1+3b) + b(j.x+2j.y+3j.z), on the
 *                logically clamped q, since j of an outside q is not its
 *                edge texel's)
 *  atlas-only    zero while b
 * The h store may be a window of the lattice (UniformDetailStorage.window):
 * on an axis where it is shorter than the lattice it is a power of two and
 * wraps, so an h address is the logically clamped q masked by the store's
 * extent (m: all ones on a base and on an axis the store spans). The base
 * block is told from the h store by its extent, which no h store has. A
 * canonical load (udLoad4_) reads a base and keeps the raw load: outside the
 * lattice it is left to the device's clamp, which each base map commutes
 * with. A store
 * to a base is range-checked here (a clamped store would land on a canonical
 * texel) and dropped unless canonical, with a violation bit (dense: silently,
 * a lattice launch deposits exactly the texels that are state). dims: the
 * lattice (vec3u). */
function detailDomainWGSL(kinds:ReadonlyMap<string,{kind:UniformDetailClass;type:string}>,domain:UniformDetailDomain,dims:string,support:string|undefined,atomic:boolean,deposit:boolean,hOnly?:ReadonlySet<string>,twins?:ReadonlyMap<string,string>,based?:ReadonlySet<string>,
 /** The mixed kernels' ring: the topology namespaces of the fields a kernel
  * loads (pf, group 0) and stores (target: the remap's adopted generation,
  * else pf), and whether every call carries a site index. */
 ring?:{readonly pf:string;readonly target:string;readonly sites:boolean},
 /** Fields with a UNIFORM_DETAIL_CANONICAL_LOAD. */
 either?:ReadonlySet<string>,
 /** Fields with a UNIFORM_DETAIL_GUARD_LOAD. */
 guards?:ReadonlySet<string>,
 /** Mixed modules use direct h-store accessors. */
 fold=false):string{
 const j="(((q&vec3i(3))+vec3i(1))>>vec3u(2u))";
 // ringed: a canonical store reaches the h texture only inside the ring.
 // watch (checked): every h texture load is tested against the ring (and the poison).
 const ringed=!!ring,watch=false,sites=watch&&ring!.sites;
 const site=sites?",site:u32":"",pass=sites?",site":"";
 let ringCode="";
 if(ring){
  // A texel's home tile: min(q >> 2, T - 1), every class (q in the lattice).
  const fn=(name:string,p:string)=>`fn ${name}(q:vec3i)->bool{let t=vec3u(min(q>>vec3u(2u),vec3i(${p}UM_T)-vec3i(1)));return (${p}umTopology[2u*${p}UM_TILES+2u*(t.x+${p}UM_T.x*(t.y+${p}UM_T.y*t.z))+1u]&1u)!=0u;}\n`;
  ringCode+=fn("udRingStore",ring.target);
  // A remap loads fields of the generation it leaves and of the one it adopts (whose entering tiles are seeded).
  ringCode+=ring.target===ring.pf?"fn udRingLoad(q:vec3i)->bool{return udRingStore(q);}\n":fn("udRingHeld",ring.pf)+"fn udRingLoad(q:vec3i)->bool{return udRingHeld(q)||udRingStore(q);}\n";
 }
 if(watch){
  const word=`${ring!.pf}umSupport[${ring!.pf}UM_DETAIL_VIOLATION+4u+site]`,R=UNIFORM_DETAIL_RING_SITE;
  const mark=!sites?"":atomic?`if((atomicLoad(&${word})&bits)!=bits){atomicOr(&${word},bits);}`:`if((${word}&bits)!=bits){${word}=${word}|bits;}`;
  ringCode+=`fn udWatch(q:vec3i,top:vec3i,v:f32${site}){
 var bits=${R.ran}u;if(!udRingLoad(clamp(q,vec3i(0),top))){bits|=${R.outside}u;}${""}
 ${mark}${`if((bits&${R.outside}u)!=0u){udViolation(${UNIFORM_DETAIL_RING_VIOLATION}u);}if((bits&${R.poison}u)!=0u){udViolation(${UNIFORM_DETAIL_POISON_VIOLATION}u);}`}
}
fn udWatchStore(${sites?"site:u32":""}){
 let bits=${R.store}u;${mark}${`udViolation(${UNIFORM_DETAIL_RING_STORE_VIOLATION}u);`}
}
fn udWatchHomes(q:vec3i,a:vec4f,h:vec4f${site}){
 var bits=${R.ran}u;if(udRingLoad(q)&&any(bitcast<vec4u>(a)!=bitcast<vec4u>(h))){bits|=${R.homes}u;}
 ${mark}${`if((bits&${R.homes}u)!=0u){udViolation(${UNIFORM_DETAIL_HOME_VIOLATION}u);}`}
}
`;
 }
 let out=/* wgsl */`
fn udViolation(bits:u32){${!support?"":atomic?`atomicOr(&${support},bits);`:`${support}=${support}|bits;`}}
fn udMask(d:vec3u,b:u32,lattice:vec3u)->vec3i{return vec3i(select(d-vec3u(1u),vec3u(0x7fffffffu),(d>=lattice)|vec3<bool>(b!=0u)));}
// A store's texel a of logical q. Out of the lattice the device drops a store (the address stays out of the texture on an axis that does not wrap); a wrapped axis drops it here, before the mask.
fn udPut(a:vec3i,q:vec3i,top:vec3i,m:vec3i)->vec3i{return select(a&m,vec3i(-1),(m!=vec3i(0x7fffffff))&(vec3u(q)>vec3u(top)));}
fn udAligned(q:vec3i)->bool{return all((q&vec3i(3))==vec3i(0));}
fn udFaceCanonical(q:vec3i)->bool{let l=q&vec3i(3);let j=${j};return all((l==vec3i(0))|(l==vec3i(3)))&&j.x+j.y+j.z<=1;}
${`fn udFace(q:vec3i,b:u32)->vec3i{let c=q>>vec3u(2u*b);let j=${j};return vec3i(c.xy,c.z*i32(1u+3u*b)+(j.x+2*j.y+3*j.z)*i32(b));}\n`}${ringCode}`;
 // fold: each accessor is written in both store forms and picks one by the
 // Mixed kernels use direct accessors for the lattice-sized h store.
 const one=(name:string,kind:UniformDetailClass,type:string,direct:boolean,sfx:string):string=>{
  let out="";
  const storage=type.startsWith("texture_storage"),level=storage?"":",0";
  const lattice=`(${dims})`,D=kind==="vertex"?`(${dims}+vec3u(1u))`:lattice,top=`vec3i(${D})-vec3i(1)`;
  // The field's base block (an atlas-only name binds a face field).
  const block=kind==="cell"?`(${dims}>>vec3u(2u))`:kind==="vertex"?`((${dims}>>vec3u(2u))+vec3u(1u))`:`((${dims}>>vec3u(2u))*vec3u(1u,1u,4u))`;
  const based_=(t:string)=>direct?`u32(textureDimensions(${t}).x<${D}.x)`:`u32(all(textureDimensions(${t})==${block}))`,extent=based_(name);
  const mode=direct?`let b=${extent};`:`let d=textureDimensions(${name});let b=u32(all(d==${block}));let m=udMask(d,b,${lattice});`;
  const canonical=(v:string)=>kind==="face"?`udFaceCanonical(${v})`:`udAligned(${v})`;
  const clamped=(v:string)=>`clamp(${v},vec3i(0),${top})`;
  // The bound texture's texel of logical v, and an h store's (name: bound or twin).
  // An out-of-lattice load as the device clamps it (unsigned: a negative coordinate reads the top edge), made here where a mask follows.
  const fit=(v:string)=>direct?v:`vec3i(min(vec3u(${v}),vec3u(${top})))`,mask=direct?"":"&m";
  const at=(v:string)=>(kind!=="face"?`(${fit(v)}>>vec3u(2u*b))`:`udFace(${clamped(v)},b)`)+mask;
  const stored=(t:string,v:string)=>direct?v:`${fit(v)}&udMask(textureDimensions(${t}),0u,${lattice})`;
  // A base block's texel of v, by b (the raw load: no window, the device's clamp).
  const raw=(v:string)=>kind!=="face"?`${v}>>vec3u(2u*b)`:`udFace(${v},b)`;
  // The bound texture's texel of a store at v.
  const put=(v:string)=>direct?kind==="atlas"?v:raw(v):`udPut(${kind==="atlas"?v:raw(v)},${v},${top},m)`;
  const home=(v:string)=>kind!=="face"?`${v}>>vec3u(2u)`:`udFace(${v},1u)`;
  // A stored field's twin is its other home (the base beside an h texture,
  // the h texture beside a base, a 1³ stand-in while there is one home):
  // a canonical v's texel there, by the bound texture's b.
  const twin=twins?.get(name),other=(v:string)=>`(${kind!=="face"?`(${v}>>vec3u(2u*(1u-b)))`:`udFace(${v},1u-b)`}${direct?"":`&udMask(textureDimensions(${twin}),1u-b,${lattice})`})`;
  if(!storage||type.includes("read_write")){
   const check="";
   const load=`textureLoad(${name},${at("q")}${level})`;
   out+=`fn udLoad_${name}${sfx}(q:vec3i${site})->vec4f{
 ${mode}
 ${check}${kind==="atlas"?`return select(textureLoad(${name},${fit("q")}${mask}${level}),vec4f(0.0),b!=0u);`
   :watch?`let v=${load};if(b==0u){udWatch(q,${top},v.x${pass});}return v;`:`return ${load};`}
}
`;
  }
  // A guard load (UNIFORM_DETAIL_GUARD_LOAD): the plain load, untested.
  if(guards?.has(name)){
   if(kind==="atlas")throw new Error(`Uniform detail rewrite: ${name} is atlas-only and has no guard load`);
   out+=`fn udLoadG_${name}${sfx}(q:vec3i)->vec4f{
 ${mode}
 return textureLoad(${name},${at("q")}${level});
}
`;
  }
  // A canonical-or-h load (UNIFORM_DETAIL_CANONICAL_LOAD): beside an h
  // texture, a canonical texel inside the lattice comes from the base
  // block (the sampled twin; a read_write binding's store twin, a base
  // while the bound texture is h), any other from the h texture. On a
  // base, and for a texture that is not a field, it is the plain load.
  if(either?.has(name)){
   if(!twin)throw new Error(`Uniform detail rewrite: ${name} has a canonical-or-h load and no twin`);
   // The twin is a base: a stored binding's is not its 1³ stand-in, a sampled one's is not the bound texture again.
   const twinBase=storage?`textureDimensions(${twin}).x>1u`:direct?`textureDimensions(${twin}).x<${D}.x`:`any(textureDimensions(${twin})!=d)`;
   out+=`fn udLoadC_${name}${sfx}(q:vec3i${site})->vec4f{
 ${mode}
 if(b==0u&&${twinBase}&&all(q>=vec3i(0))&&all(q<=${top})&&${canonical("q")}){${watch?`let v=textureLoad(${twin},${home("q")}${level});udWatchHomes(q,v,textureLoad(${name},q${mask}${level})${pass});return v;`:`return textureLoad(${twin},${home("q")}${level});`}}
 return udLoad_${name}${sfx}(q${pass});
}
`;
  }
  // A canonical load (UNIFORM_DETAIL_4H_LOAD): the base block, bound as the
  // sampled twin at every capacity (b by its extent, as any load's: a
  // texture that is not a field binds itself there).
  if(based?.has(name)){
   const c=`clamp(q,vec3i(0),${top})`,b=based_(twin!);
   const both=watch&&kind!=="atlas";
   out+=`fn udLoad4_${name}${sfx}(q:vec3i${site})->vec4f{
 let b=${b};
 ${""}${both?`let v=textureLoad(${twin},${raw(kind==="face"?c:"q")},0);if(b!=0u&&${extent}==0u){udWatchHomes(${c},v,textureLoad(${name},${stored(name,"q")},0)${pass});}return v;`:`return textureLoad(${twin},${raw(kind==="face"?c:"q")},0);`}
}
`;
  }
  if(hOnly?.has(name))out+=`fn udLoadH_${name}${sfx}(q:vec3i${site})->vec4f{
 ${""}${watch&&kind!=="atlas"?`let v=textureLoad(${name},${stored(name,"q")}${level});udWatch(q,${top},v.x${pass});return v;`:`return textureLoad(${name},${stored(name,"q")}${level});`}
}
`;
  if(storage){
   const fail=(bits:number)=>deposit?"return;":`udViolation(${bits}u);return;`;
   const inside=`all(q>=vec3i(0))&&all(q<=${top})`;
   // The ring. On a base: the h twin takes the texel inside the
   // ring. On an h texture: a canonical texel goes to the base twin, and to
   // the h texture only inside the ring; any other texel is the h texture's.
   if(twin&&kind!=="atlas"&&(ringed)){
    // The lattice-sized store's form stores an h texture's canonical texel
    // to both homes untested: the test drops a store only outside the
    // ring, where the h texel exists and is never loaded, and costs a
    // topology read at every canonical store inside it. (A base's h twin
    // keeps the test: there it saves the store.) Not under the QA fills.
    const gate=ringed&&!(direct&&!watch);
    out+=`fn udStore_${name}${sfx}(q:vec3i,value:vec4f${site}){
 ${mode}
 if(b!=0u){
  if(any(q<vec3i(0))||any(q>${top})){return;}if(!${canonical("q")}){${fail(UNIFORM_DETAIL_VIOLATION[kind])}}
  textureStore(${name},${put("q")},value);
  if(textureDimensions(${twin}).x>1u${ringed?"&&udRingStore(q)":""}){textureStore(${twin},${other("q")},value);}
  return;
 }
 ${watch&&ringed
  ?`if(textureDimensions(${twin}).x>1u&&${inside}){if(${canonical("q")}){textureStore(${twin},${other("q")},value);if(!udRingStore(q)){return;}}else if(!udRingStore(q)){udWatchStore(${sites?"site":""});}}`
  :`if(textureDimensions(${twin}).x>1u&&${inside}&&${canonical("q")}){textureStore(${twin},${other("q")},value);${gate?"if(!udRingStore(q)){return;}":""}}`}
 textureStore(${name},${put("q")},value);
}
`;
    return out;
   }
   out+=`fn udStore_${name}${sfx}(q:vec3i,value:vec4f${site}){
 ${mode}
 ${kind==="atlas"?`if(b!=0u){${fail(UNIFORM_DETAIL_VIOLATION.atlas)}}\n textureStore(${name},${put("q")},value);`
   :`if(b!=0u){if(any(q<vec3i(0))||any(q>${top})){return;}if(!${canonical("q")}){${fail(UNIFORM_DETAIL_VIOLATION[kind])}}}\n textureStore(${name},${put("q")},value);${
   // With two homes, a canonical texel is stored to both: a load takes it
   // from whichever texture its kernel binds, with no test.
   twin?`\n if(textureDimensions(${twin}).x>1u&&all(q>=vec3i(0))&&all(q<=${top})&&${canonical("q")}){textureStore(${twin},${other("q")},value);}`:""}`}
}
`;
  }
  return out;
 };
 for(const [name,{kind,type}] of kinds)out+=one(name,kind,type,fold,"");
 return out;
}
type DetailKinds=ReadonlyMap<string,{kind:UniformDetailClass;type:string}>;
/** Haloed patches: the other resident boxes that can hold texel `p` of home
 * patch n: one direction per axis, so the non-empty submasks of the near
 * axes (one for a texel near a face, three near an edge, seven at a corner).
 * A loop, not blocks: Metal flattens every call, and this text is paid at
 * every site. */

/** Haloed patches: address helpers. The directory is the only table read.
 * udFind_<class>(p): the texel of logical p with no home in hand; w = 0 the
 * base (canonical: current by write-through), 1 a resident box (its home
 * patch's, else for a vertex or atlas texel a neighbour's halo), -1 no texel
 * (xyz: the base address). */

/** Haloed patches: the write-through store of one field. Every resident box
 * that holds q, then the base when canonical; returns whether any did. */

/** Haloed patches: the dense kernels' accessors (no home): a load finds its
 * texel from the directory and, where a tile has none, takes what packed
 * storage loads there (the tile's value, its parity face, the corners'
 * interpolant); a store that lands nowhere is dropped. */

/** A work item's owner comes from one of these (by job, lane or tile); an
 * owner looked up by position is a neighbour, never the home. */
const PATCH_OWNER=/fn (\w+)\(([^)]*)\)\s*->\s*(\w*)UMOwner\s*\{/g,PATCH_NEIGHBOUR=/(OwnerAt|VertexAuthority|StageOwner)$/;
/** Haloed patches: the mixed kernels' rewrite (UniformDetailPatch). The
 * work item's home is resolved once, where it takes its owner; with none
 * resolved the box is empty and every load is a base or directory load. */

function rewriteCalls(source:string,names:ReadonlySet<string>,hOnly?:Set<string>,site?:(field:string,store:boolean,mark?:string,text?:string,scope?:string)=>{suffix:string;argument:string},based?:Set<string>,scope?:string,canonical?:Set<string>):string{
 const pattern=/(\/\*(?:4?h|c|g)\*\/\s*)?texture(Load|Store)\(\s*(\w+)\s*,/g;
 let cursor=0,output="",match:RegExpExecArray|null;
 while((match=pattern.exec(source))){
  const field=match[3]!;
  if(!names.has(field)){if(match[1])throw new Error(`Uniform detail rewrite: ${field} is marked ${match[1].trim()} and is not a field`);continue;}
  let at=pattern.lastIndex,depth=0;const commas:number[]=[];
  for(;at<source.length;at++){
   const c=source[at];if(c==="("||c==="[")depth++;
   else if(c==="]")depth--;
   else if(c===")"){if(depth===0)break;depth--;}
   else if(c===","&&depth===0)commas.push(at);
  }
  if(at===source.length)throw new Error(`Uniform detail rewrite cannot parse a ${field} access`);
  const store=match[2]==="Store",coarse=!!match[1]?.startsWith("/*4h"),either=!!match[1]?.startsWith("/*c"),guard=!!match[1]?.startsWith("/*g")&&!!canonical,h=!!match[1]&&!coarse&&!either&&!match[1].startsWith("/*g")&&!!hOnly,four=coarse&&!!based,can=either&&!!canonical;
  if(match[1]&&store)throw new Error(`Uniform detail rewrite: only a load of ${field} is marked ${match[1].trim()}`);
  if(h)hOnly!.add(field);
  if(four)based!.add(field);
  if(can)canonical!.add(field);
  if(commas.length>1||(store&&commas.length!==1))throw new Error(`Uniform detail rewrite: unexpected ${field} ${match[2]} arguments`);
  const end=commas.length?commas[0]!:at;
  if(!store&&commas.length&&!/^\s*0[ui]?\s*$/.test(source.slice(end+1,at)))throw new Error(`Uniform detail rewrite: ${field} loads only level 0`);
  // The enclosing function, for a site's report (a nested call inherits it).
  const within=!site?undefined:scope??[...source.slice(0,match.index).matchAll(/\bfn\s+(\w+)\s*\(/g)].pop()?.[1]??"";
  const form=guard?undefined:site?.(field,store,h?"h":four?"4h":can?"c":"",source.slice(match.index,at+1),within);
  const coordinate=rewriteCalls(source.slice(pattern.lastIndex,end).trim(),names,hOnly,site,based,within,canonical);
  const value=store?`,${rewriteCalls(source.slice(end+1,at).trim(),names,hOnly,site,based,within,canonical)}`:"";
  output+=source.slice(cursor,match.index)+`ud${match[2]}${h?"H":four?"4":can?"C":guard?"G":""}${form?.suffix??""}_${field}(vec3i(${coordinate})${value}${form?.argument??""})`;
  cursor=at+1;pattern.lastIndex=cursor;
 }
 return output+source.slice(cursor);
}

/** The central rewrite: every load/store of an h field in a mixed shader
 * (any group) goes through its class accessor, addressed by the table in
 * the group-0 topology (whatever its namespace prefix). A shader without
 * fields is returned unchanged. */
export function uniformDetailShader(code:string,device?:GPUDevice):string{
 const declared=[...code.matchAll(/var\s+(\w+)\s*:\s*(texture(?:_storage)?_3d<[^>]+>)/g)];
 const kinds=new Map<string,{kind:UniformDetailClass;type:string}>();
 for(const [,name,type] of declared){const kind=classify(name!,type!);if(kind)kinds.set(name!,{kind,type:type!});}
 if(!kinds.size)return code;
 const topology=/@group\(0\)\s*@binding\(0\)\s*var<storage,\s*read>\s*(\w*)umTopology\s*:/.exec(code);
 if(!topology)throw new Error("Uniform detail rewrite: fields without a group-0 mixed topology");
 const pf=topology[1]!;
 if(/fn\s+\w+\s*\([^)]*:\s*texture/.test(code))throw new Error("Uniform detail rewrite: a texture-typed function parameter bypasses the field accessors");
 for(const name of kinds.keys())if(new RegExp(`textureDimensions\\(\\s*${name}\\b`).test(code))throw new Error(`Uniform detail rewrite: textureDimensions(${name}) is not the logical lattice`);
 const atomic=new RegExp(`var<storage,\\s*read_write>\\s*${pf}umSupport\\s*:\\s*array<atomic<u32>>`).test(code);
 const dims=/const UM_D=vec3u\((\d+)u,(\d+)u,(\d+)u\)/.exec(code);
 if(!dims)throw new Error("Uniform detail rewrite: no mixed lattice constant");
 const layout=device?liveLayout(device,dims.slice(1).map(Number)):undefined;
 if(!layout)return code;
 {
  // Survey and poison runs number every load and store but the canonical (base) loads.
  const domain=layout.domain;
  code=code.replaceAll(UNIFORM_DETAIL_RING_4H_LOAD,UNIFORM_DETAIL_4H_LOAD);
  const sites=false;
  const hOnly=new Set<string>(),based=new Set<string>(),either=new Set<string>();
  const calls=rewriteCalls(code,new Set(kinds.keys()),hOnly,undefined,based,undefined,either);
  // A canonical-or-h load of a sampled binding reads the base twin a canonical load does.
  for(const name of either){const field=kinds.get(name)!;if(field.kind==="atlas")throw new Error(`Uniform detail rewrite: ${name} is atlas-only and has no canonical texel`);if(!field.type.startsWith("texture_storage"))based.add(name);}
  // Every stored field binding but an atlas-only one (no canonical texel)
  // declares its twin, and every sampled one with a canonical load.
  const twins=new Map<string,string>();let declarations="";
  for(const [,group,binding,name,type] of code.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)\s*var\s+(\w+)\s*:\s*(texture(?:_storage)?_3d<[^>]+>)/g)){
   const field=kinds.get(name!);if(!field||field.kind==="atlas"||(type!.startsWith("texture_storage")?based.has(name!):!based.has(name!))){
    if(field&&based.has(name!))throw new Error(`Uniform detail rewrite: ${name} has a canonical load and is ${field.kind==="atlas"?"atlas-only":"a stored binding"}`);
    continue;
   }
   if(twins.has(name!))throw new Error(`Uniform detail rewrite: ${name} is declared twice`);
   twins.set(name!,`udb_${name}`);declarations+=`@group(${group}) @binding(${Number(binding)+UNIFORM_DETAIL_TWIN_BINDING}) var udb_${name}:${type};\n`;
  }
  for(const [name,{kind,type}] of kinds)if((based.has(name)||(kind!=="atlas"&&type.startsWith("texture_storage")))&&!twins.has(name))throw new Error(`Uniform detail rewrite: ${name} is not declared as @group(g) @binding(b) var, so its twin has no binding`);
  // A 1³ stand-in is told from a base by its x extent.
  if(twins.size&&layout.tiles[0]<2)throw new Error("Uniform detail rewrite: the domain placement needs two tiles along x");
  // A remap stores the fields of the generation it adopts: the unprefixed topology beside the one it leaves.
  const target=pf&&/var<storage,\s*read>\s*umTopology\s*:/.test(code)?"":pf;
   // The h store spans the lattice; each accessor uses its direct address.
  return calls+declarations+detailDomainWGSL(kinds,layout.domain,`${pf}UM_D`,`${pf}umSupport[${pf}UM_DETAIL_VIOLATION]`,atomic,false,hOnly,twins,based,{pf,target,sites},either,new Set([...kinds.keys()].filter(name=>calls.includes(`udLoadG_${name}(`))),true);
 }

}
/** A dense reference shader (the t=0 authority and publication) through the
 * same accessors: `classes` names its field bindings, the directory is the
 * head of the residency state, a uniform at `group` (those kernels' own
 * layout has no storage binding to spare). A store outside the canonical texels is
 * dropped, so a lattice launch deposits exactly the texels that are state. */
export function uniformDetailDenseShader(code:string,storage:UniformDetailStorage,classes:Readonly<Record<string,UniformDetailClass>>,group:number):string{
 const l=storage.layout;
 const kinds=new Map<string,{kind:UniformDetailClass;type:string}>();
 for(const [,name,type] of code.matchAll(/var\s+(\w+)\s*:\s*(texture(?:_storage)?_3d<[^>]+>)/g)){const kind=classes[name!];if(kind)kinds.set(name!,{kind,type:type!});}
 for(const name of Object.keys(classes))if(!kinds.has(name))throw new Error(`Uniform detail dense rewrite: no 3D texture ${name}`);
 if(/fn\s+\w+\s*\([^)]*:\s*texture/.test(code))throw new Error("Uniform detail dense rewrite: a texture-typed function parameter bypasses the field accessors");
 for(const name of kinds.keys())if(new RegExp(`texture(?:Dimensions|SampleLevel)\\(\\s*${name}\\b`).test(code))throw new Error(`Uniform detail dense rewrite: ${name} is read past the accessors`);
 const dims="vec3u(params.dimsDt.xyz)";
 return rewriteCalls(code,new Set(kinds.keys()))+detailDomainWGSL(kinds,l.domain!,dims,undefined,false,true);
}
/** createShaderModule through the central rewrite. */
export function uniformDetailModule(device:GPUDevice,descriptor:GPUShaderModuleDescriptor):GPUShaderModule{
 return device.createShaderModule({...descriptor,code:uniformSceneConstantsWGSL(uniformDetailShader(descriptor.code,device))});
}
const storageOf=new WeakMap<UniformMixedOwnership,UniformDetailStorage>();
/** Every pipeline is compiled once, with direct h-store accessors. */
export function uniformDetailPipeline(device:GPUDevice,_ownership:UniformMixedOwnership,descriptor:GPUComputePipelineDescriptor):Promise<GPUComputePipeline>{
 return device.createComputePipelineAsync(descriptor);
}
export function uniformDetailPick(pipeline:GPUComputePipeline):GPUComputePipeline{return pipeline;}
