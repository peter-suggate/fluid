import { UNIFORM_ADDRESS_PLAN_ENTRIES } from "./uniform-address-plan";
import { UNIFORM_WINDOW_HEADER, uniformWindowAddressWGSL, uniformWindowFixedFrame } from "./uniform-window-address";
import { uniformSceneConstantsWGSL } from "./uniform-scene-constants";
/** Detail fields hold a tile-resolution base block and, when fine ownership
 * is reserved, an h store of every cell (a vertex field's: every vertex).
 * Cell and face textures are grids of blocks (uniform-lattice-address.ts);
 * shader accessors and the host images here take logical coordinates. They
 * read the base at canonical texels and keep both homes current inside the
 * detail ring. */
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedDetailViolationWord, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_PAGE_TABLE_WORDS, uniformBlockStored, uniformCellStored, uniformCellTexel, uniformEachTile, uniformLatticeAddressWGSL, uniformLatticeRunWGSL, uniformTileStored, uniformTileTexel, uniformVertexSlot, uniformVertexStored, uniformVertexTexel, uniformWideVertexTexel, type UniformLatticeAddress } from "./uniform-lattice-address";
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
 /** Base block: a tile texture (cells), one with a tile's four texels stacked along z (faces), a 4h vertex texture of 9³ blocks (vertices). */
 readonly base:Dims;
 /** Packed: the base block comes first, folded into z slabs (slice z of the
  * base at fold column and row of slab z/per); the atlas starts at slab
  * atlasZ. Nothing here depends on capacity. */
 readonly atlasZ:number;readonly fold:number;readonly per:number;

 readonly extent:Dims;
}
export interface UniformDetailLayout{
 readonly dims:Dims;readonly tiles:Dims;readonly patchEdge:number;readonly patchGrid:Dims;readonly patches:number;
 /** The lattice's key spaces and texture sizes. */
 readonly address:UniformLatticeAddress;

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
function domainLayout(address:UniformLatticeAddress,domain:UniformDetailDomain):UniformDetailLayout{
 const dims=address.cells,tiles=address.tiles,patchEdge=2**Math.ceil(Math.log2(Math.max(MIN_PATCH_EDGE,...dims)));
 const classes=Object.fromEntries(CLASSES.map(kind=>{
  // A cell base is a tile texture; a face base stacks a tile's four texels along z; a vertex base is a 4h vertex texture.
  const base:Dims=kind==="vertex"?address.wideVertexTexture:kind==="cell"?address.tileTexture:address.faceBaseTexture;
  return [kind,{spacing:patchEdge,base,atlasZ:0,fold:1,per:1,extent:base}];
 })) as unknown as Record<UniformDetailClass,UniformDetailClassLayout>;
 return {dims,tiles,patchEdge,patchGrid:[1,1,1],patches:1,placement:"packed",layer:[1,1],maxLayers:1,classes,domain,address};
}

interface Field{readonly storage:UniformDetailStorage;readonly kind:UniformDetailClass;readonly logical:Dims;readonly handle:GPUTexture;readonly empty:number}
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
/** Pipeline constants of a launch whose every field load and store runs in
 * an h tile's stencil (a job of the h list) and that binds the group's h
 * side (UniformDetailGroup.group): an h tile implies C > 0, where each field
 * of a following generation is its h texture and its twin the base block,
 * so the accessors take the texture class as a constant, not by the bound
 * extent (UNIFORM_DETAIL_H_LOAD's invariant, for the whole launch). Not for
 * a launch with a pinned (per-tile) field, which is a base at every C. */
export const UNIFORM_DETAIL_H_STORE:Readonly<Record<string,number>>={udHStore:1};
/** Pipeline constants of a launch whose vertex loads of either class resolve through one slot walk (udVertex): one
 * page-table probe loop for the h store's 33-wide blocks and the base's 9-wide ones, not one of each. Faster where a
 * launch's vertex loads mix the classes at run time (the merged deferred advect, the depth seeds), slower elsewhere. */
export const UNIFORM_DETAIL_MERGED_SLOTS:Readonly<Record<string,number>>={udMergedSlots:1};
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
/** The class a field was created as (createField), by its handle or by any
 * texture that holds it; undefined of a texture that is no detail field. */
export function uniformDetailClass(texture:GPUTexture):UniformDetailClass|undefined{return fields.get(texture)?.kind;}

/** Live storages per device: the rewrite finds a shader's layout by its
 * tile key count (UM_TILES). Two live layouts of one count on a device are ambiguous. */
const live=new WeakMap<GPUDevice,Set<UniformDetailStorage>>();
function liveLayout(device:GPUDevice,tiles:number):UniformDetailLayout|undefined{
 const found=[...live.get(device)??[]].filter(s=>s.layout.address.tileKeys===tiles);
 if(found.some(s=>s.signature!==found[0]!.signature))throw new Error(`Uniform detail storage: two live layouts of ${tiles} tile keys`);
 return found[0]?.layout;
}

const DOMAIN_FLAGS=["checked","compact","parity","unpinned","poison","survey","untwinned","unwrapped"];

/** Workgroups of each transfer launch: their lanes stride the GPU lists. */

/** Fields one transfer launch binds (the default storage-texture limit). */
const BATCH=4;
type TransferKind="cell"|"vertex"|"face";

/** The ring kernels of one launch: up to BATCH fields of one format. */
/** The ring's pass descriptors, built once. */
const RING_ENTER_PASS:GPUComputePassDescriptor={label:"Uniform detail ring enter"},RING_SHARE_PASS:GPUComputePassDescriptor={label:"Uniform detail ring share"};
interface RingBatch{readonly handles:readonly GPUTexture[];readonly layout:GPUBindGroupLayout;readonly enter:GPUComputePipeline}
interface DomainKernels{

 readonly ring?:{readonly batches:readonly RingBatch[];readonly shareLayout:GPUBindGroupLayout;readonly share:GPUComputePipeline};
 readonly layouts:Record<"r32float"|"rgba32float",GPUBindGroupLayout>;
 /** refresh: h to base, the canonical texels. */
 readonly refresh:Record<TransferKind,GPUComputePipeline>;
}
/** The field factory: owns every h-sized texture of one solver and their
 * residency. Every field has a base block; while the simulation ownership
 * holds an h-tile capacity each following field has an h store beside it,
 * the whole lattice (`resident`). The table in every attached topology
 * carries that as its directory word. A relayout seeds the canonical texels
 * of the tiles it brings into the detail ring from the bases (encodeAdmit),
 * with no readback. */
export class UniformDetailStorage{
 readonly layout:UniformDetailLayout;

 private readonly handles:GPUTexture[]=[];
 private readonly physicals=new Map<GPUTexture,GPUTexture>();
 private readonly stubs=new Map<GPUTexture,GPUTexture>();
 private readonly ownerships=new Set<UniformMixedOwnership>();
 readonly signature:string;
 private readonly ringDimensions:GPUBuffer;
 /** address: the solver's lattice address. */
 constructor(private readonly device:GPUDevice,address:UniformLatticeAddress){
  const domain:UniformDetailDomain={checked:false,compactFaces:true,unpinned:false};
  this.layout=domainLayout(address,domain);
  const l=this.layout;
  this.signature=`${l.placement}:${l.patchEdge}:${l.layer}::domain:${domain.checked}:${domain.compactFaces}:${!!domain.unpinned}:${!!domain.poison}:${!!domain.survey}:${!!domain.untwinned}:${!!domain.unwrapped}`;
  let set=live.get(device);if(!set)live.set(device,set=new Set());set.add(this);liveLayout(device,l.address.tileKeys);
  this.ringDimensions=device.createBuffer({label:"Uniform detail ring dimensions",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(this.ringDimensions,0,new Uint32Array([...l.tiles,l.address.blocks]));
 }

 /** A field's texture size: its h store (the block grid in cells; a vertex field's, the box's vertices) or its base block. */
 private extent(kind:UniformDetailClass,resident=this.resident):Dims{
  const l=this.layout,a=l.address;
  return resident?kind==="vertex"?a.vertexTexture:a.cellTexture:l.classes[kind].base;
 }
 private allocate(label:string,kind:UniformDetailClass,format:GPUTextureFormat,resident=this.resident):GPUTexture{
  const extent=this.extent(kind,resident);
  if(extent.some(n=>n>this.device.limits.maxTextureDimension3D))throw new RangeError(`Uniform detail field ${label} needs ${extent}, past the device's 3D texture limit`);
  return this.device.createTexture({label,size:[...extent],dimension:"3d",format,
   usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
 }
 /** A field: the returned texture is its handle (see Field). Bind it
  * through uniformDetailGroup; read and write it through this storage.
  * empty: what the field holds where nothing is stored (uniformDetailField
  * of the handle carries it). Every texel of each of its textures holds it
  * from the texture's making, and an upload's image starts from it: the
  * texels of block 0 and of the last block, of a block no page has, of a
  * skirt another page's plane holds and of a partial page past the box keep
  * it, since a kernel stores only where a vertex is stored. Only a vertex
  * field's is not zero. */
 createField(label:string,kind:UniformDetailClass,format:"r32float"|"rgba32float",perTile=false,empty=0):GPUTexture{
  // An atlas-only field has slots of its own under haloed patches; elsewhere it shares the face placement.
  if(kind==="atlas")kind="face";
  if(this.transfers)throw new Error(`Uniform detail field ${label} was created after the transfer kernels bound the fields`);
  if(perTile&&kind!=="cell")throw new Error(`Uniform detail field ${label}: only a cell field is per tile`);
  if(empty!==0&&kind!=="vertex")throw new Error(`Uniform detail field ${label}: only a vertex field holds a value where nothing is stored`);
  const pin=perTile;
  if(((kind==="face")!==(format==="rgba32float")))throw new Error(`Uniform detail field ${label}: the domain placement transfers scalar cell and vertex fields and vec4 face fields`);
  const physical=this.allocate(label,kind,format,pin?false:undefined);let handle=physical;
  {handle=this.device.createTexture({label,size:[1,1,1],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING});handle.destroy();}
  const d=this.layout.dims,field:Field={storage:this,kind,logical:kind==="vertex"?[d[0]+1,d[1]+1,d[2]+1]:d,handle,empty};
  fields.set(handle,field);fields.set(physical,field);this.writeEmpty(physical,empty);
  this.handles.push(handle);this.physicals.set(handle,physical);if(pin)this.pinned.add(handle);
  this.bases.set(handle,physical);this.resized();
  return handle;
 }
 /** A field's empty value to every texel of a texture of it just made, a layer a write. */
 private writeEmpty(t:GPUTexture,empty:number):void{
  if(empty===0)return;
  const layer=new Float32Array(t.width*t.height).fill(empty);
  for(let z=0;z<t.depthOrArrayLayers;z++)this.device.queue.writeTexture({texture:t,origin:[0,0,z]},layer,{bytesPerRow:4*t.width,rowsPerImage:t.height},[t.width,t.height,1]);
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
  if(!stub)this.resized();
  if(!stub)this.stubs.set(handle,stub=this.device.createTexture({label:`${handle.label} stub`,size:[1,1,1],dimension:"3d",format:handle.format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}));
  return stub;
 }
 private bound:{ref:WeakRef<UniformDetailGroup>;build:(base?:boolean)=>GPUBindGroup}[]=[];
 private prune=64;
 /** uniformDetailGroup's registration: rebuilt when the fields are replaced. */
 track(group:UniformDetailGroup,build:(base?:boolean)=>GPUBindGroup):void{
  if(this.bound.length>=this.prune){this.bound=this.bound.filter(b=>b.ref.deref());this.prune=Math.max(64,2*this.bound.length);}
  this.bound.push({ref:new WeakRef(group),build});
 }
 /** Every tracked group over the fields' current physical textures. */
 private rebind():void{this.bound=this.bound.filter(b=>{const g=b.ref.deref();if(g){g.group=b.build();g.base=b.build(true);}return !!g;});}
 /** Before a host relayout's admit: residency follows the capacity. */
 reserve(_layout:UniformMixedLayout):void{this.follow();
 }
 get allocatedBytes():number{
  return this.bytes??=this.textures().reduce((n,t)=>n+t.width*t.height*t.depthOrArrayLayers*(t.format==="rgba32float"?16:4),0)+this.ringDimensions.size;
 }
 /** allocatedBytes, summed once per set of textures: createField, stub,
  * adoptBase, admit and commitRetire are what change the set (resized). */
 private bytes?:number;
 /** Bumped with every change of the textures allocatedBytes sums: a reader
  * that holds the total compares this and reads again. */
 allocationRevision=0;
 private resized():void{this.bytes=undefined;this.allocationRevision++;}
 /** The table every attached ownership carries (uniform-detail-abi.ts). */
 table():Uint32Array<ArrayBuffer>{
  const l=this.layout,words=new Uint32Array(HEADER+1);
  words.set([UNIFORM_DETAIL_PACKED,Math.log2(l.patchEdge),...l.patchGrid,...l.tiles]);
  CLASSES.forEach((kind,k)=>{const c=l.classes[kind];words.set([c.spacing,c.atlasZ,c.fold,c.per,...c.base,kind==="face"?UNIFORM_DETAIL_FACE_COMPACT:0],8+8*k);});
  // The lattice's tile box, the two-texture flag and the pool's blocks.
  words.set([0,0,0,l.tiles[0]-1,l.tiles[1]-1,l.tiles[2]-1,UNIFORM_DETAIL_BASED,l.address.blocks],32);
  // The directory word: the fields have an h store.
  words[HEADER]=this.resident?1:0;return words;
 }
 /** Carry this storage's table in an ownership's topology tail (group 0 of
  * every stage that touches a field through it). */
 attach(ownership:UniformMixedOwnership):void{this.ownerships.add(ownership);storageOf.set(ownership,this);ownership.writeDetail(this.table());}
 /** The table to every attached ownership (between frames, in queue order with the submits around it). */
 private publishTable():void{const table=this.table();for(const o of this.ownerships)o.writeDetail(table);}

 /** The field handles. */
 get fieldTextures():readonly GPUTexture[]{return this.handles;}

 destroy():void{
  live.get(this.device)?.delete(this);this.ringDimensions.destroy();

  for(const t of this.textures())t.destroy();
 }
 kindOf(texture:GPUTexture):UniformDetailClass{const f=fields.get(texture);if(f?.storage!==this)throw new Error(`${texture.label} is not a field of this detail storage`);return f.kind;}
 private logicalOf(texture:GPUTexture):Dims{return fields.get(texture)!.logical;}

 private readonly texel:[number,number,number]=[0,0,0];
 /** A cell or face field's h store: the texel index of a cell. */
 private cellIndex(x:number,y:number,z:number,W:number,H:number):number{const t=uniformCellTexel(this.layout.address,x,y,z,this.texel);return t[0]+W*(t[1]+H*t[2]);}
 /** A cell or face field's base block: the texel index of a tile. A face
  * base stacks a tile's four texels along z: m = 0 its origin, else the
  * +face anchor of axis m - 1. */
 /** The h store's texel index of a vertex, and a base block's of a tile corner (in tile units). */
 private vertexIndex(x:number,y:number,z:number,W:number,H:number):number{const t=uniformVertexTexel(this.layout.address,x,y,z,this.texel);return t[0]+W*(t[1]+H*t[2]);}
 private cornerIndex(x:number,y:number,z:number,W:number,H:number):number{const t=uniformWideVertexTexel(this.layout.address,x,y,z,this.texel);return t[0]+W*(t[1]+H*t[2]);}
 private baseIndex(kind:"cell"|"face",x:number,y:number,z:number,m:number,W:number,H:number):number{const t=uniformTileTexel(this.layout.address,x,y,z,this.texel);return t[0]+W*(t[1]+H*(kind==="cell"?t[2]:4*t[2]+m));}
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
  const h=!base&&this.twinned(texture);
  const image=new Float32Array(W*H*t.depthOrArrayLayers*comp),a=this.layout.address,empty=fields.get(texture)!.empty;
  // The image starts as the field's empty value: only stored texels take an upload's.
  if(empty!==0)image.fill(empty);
  if(!values)return image;
  if(kind==="vertex"&&h){
   // The h store: a page's rows, each whole where one block holds the row
   // (the vertex at 0 in the first page's skirt). A row no one block holds
   // goes vertex by vertex.
   const first:[number,number,number,number]=[0,0,0,0],last:[number,number,number,number]=[0,0,0,0];
   for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;){
    const end=Math.min(lx,x===0?33:x+32),src=x+lx*(y+ly*z);
    uniformVertexSlot(a,x,y,z,5,first);uniformVertexSlot(a,end-1,y,z,5,last);
    if(uniformBlockStored(a,first[3])&&first[3]===last[3]&&last[0]-first[0]===end-1-x)image.set(values.subarray(src,src+end-x),this.vertexIndex(x,y,z,W,H));
    else for(let v=x;v<end;v++)if(uniformVertexStored(a,v,y,z))image[this.vertexIndex(v,y,z,W,H)]=values[src+v-x]!;
    x=end;
   }
  }else if(kind==="vertex"){
   // A vertex base: tile corner g is vertex 4g.
   const [tx,ty,tz]=this.layout.tiles;
   for(let z=0;z<=tz;z++)for(let y=0;y<=ty;y++)for(let x=0;x<=tx;x++)if(uniformVertexStored(a,4*x,4*y,4*z))image[this.cornerIndex(x,y,z,W,H)]=values[4*x+lx*(4*y+ly*4*z)]!;
  }else if(h){
   // The h store: a page's rows, each whole.
   const [dx,dy,dz]=this.layout.dims;
   for(let z=0;z<dz;z++)for(let y=0;y<dy;y++)for(let x=0;x<dx;x+=32){
    if(!uniformCellStored(a,x,y,z))continue;
    const src=(x+lx*(y+ly*z))*comp;image.set(values.subarray(src,src+Math.min(32,dx-x)*comp),this.cellIndex(x,y,z,W,H)*comp);
   }
  }else{
   // A base block: each tile's origin texel, and a face field's three +face anchors.
   const [tx,ty,tz]=this.layout.tiles,k=kind==="cell"?"cell":"face",n=k==="cell"?1:4;
   for(let z=0;z<tz;z++)for(let y=0;y<ty;y++)for(let x=0;x<tx;x++)for(let m=0;m<n;m++){
    if(!uniformTileStored(a,x,y,z))continue;
    const dst=this.baseIndex(k,x,y,z,m,W,H)*comp,src=(4*x+(m===1?3:0)+lx*(4*y+(m===2?3:0)+ly*(4*z+(m===3?3:0))))*comp;
    for(let c=0;c<comp;c++)image[dst+c]=values[src+c]!;
   }
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
  const address=this.layout.address,c:[number,number,number]=[0,0,0];
  uniformEachTile(address,(t,x,y,z)=>{if(!(layout.tiles[t]!&0x80000000))visit([4*x,4*y,4*z]);});
 }
 private coarsenCells(layout:UniformMixedLayout,values:Float32Array,image:Float32Array,texture:GPUTexture):void{
  const [lx,ly]=this.logicalOf(texture),W=this.physical(texture).width,H=this.physical(texture).height;
  this.coarse(layout,o=>{
   let sum=0;for(let k=0;k<64;k++)sum=Math.fround(sum+values[(o[0]+(k&3))+lx*((o[1]+((k>>2)&3))+ly*(o[2]+(k>>4)))]!);
   image[this.baseIndex("cell",o[0]>>2,o[1]>>2,o[2]>>2,0,W,H)]=Math.fround(sum/64);
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
    const i=4*this.baseIndex("face",o[0]>>2,o[1]>>2,o[2]>>2,a+1,W,H);image.fill(0,i,i+4);image[i+a]=Math.fround(sum/16);image[i+3]=walls|(held(qs,a)?1<<a:0);
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
 /** Makes a caller's 4h vertex texture the base block of a vertex field: the
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
  this.physicals.set(handle,base);this.bases.set(handle,base);this.adopted.set(handle,base);own.destroy();this.writeEmpty(base,field.empty);this.rebind();this.resized();
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
   this.physicals.set(handle,after);fields.set(after,field);registerUniformDetailBase(after,base);this.writeEmpty(after,field.empty);
  }
  this.resident=true;this.publishTable();this.rebind();this.resized();
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
 private encodeRing(encoder:GPUCommandEncoder,target:GPUBuffer):void{
  const ring=this.kernelsOfDomain().ring,held=this.simulation!.presentation.buffer;
  if(!ring)throw new Error("Uniform detail storage: no ring kernels (the storage is not prepared)");
  const first=this.physicals.get(ring.batches[0]!.handles[0]!)!;let bound=this.ringGroups.get(target);
  if(!bound||bound.key!==first)bound=this.bindRing(ring,first,held,target);
  const groups=(this.layout.address.tileKeys+63)>>>6,pass=encoder.beginComputePass(RING_ENTER_PASS);
  for(let n=0;n<ring.batches.length;n++){const p=ring.batches[n]!.enter;if(!p)throw new Error("Uniform detail storage: Uniform detail ring enter is not compiled");pass.setPipeline(p);pass.setBindGroup(0,bound.groups[n]!);pass.dispatchWorkgroups(groups);}
  pass.end();
 }

 /** encodeRing's groups over the fields' textures now (apart from it: a closure there would cost every encode a context). */
 private bindRing(ring:NonNullable<DomainKernels["ring"]>,first:GPUTexture,held:GPUBuffer,target:GPUBuffer){
  const other=this.other().presentation.buffer;
  const bound={key:first,groups:ring.batches.map((b,n)=>this.device.createBindGroup({label:"Uniform detail ring",layout:b.layout,entries:[
   ...b.handles.flatMap((h,i)=>[{binding:i,resource:this.bases.get(h)!.createView()},{binding:BATCH+i,resource:this.physicals.get(h)!.createView()}]),
   {binding:2*BATCH,resource:{buffer:held}},{binding:2*BATCH+1,resource:{buffer:target}},{binding:2*BATCH+4,resource:{buffer:this.ringDimensions}},...(n===0?[{binding:2*BATCH+2,resource:{buffer:other}}]:[])]}))};
  this.ringGroups.set(target,bound);return bound;
 }

 encodeSettle(encoder:GPUCommandEncoder,share:boolean):void{

  if(share){
   const ring=this.kernelsOfDomain().ring!,tiles=this.layout.address.tileKeys;
   const pass=encoder.beginComputePass(RING_SHARE_PASS);pass.setPipeline(ring.share);
   pass.setBindGroup(0,this.shareGroup??=this.device.createBindGroup({label:"Uniform detail ring share",layout:ring.shareLayout,entries:[{binding:2,resource:{buffer:this.ringDimensions}},{binding:0,resource:{buffer:this.simulation!.presentation.buffer}},{binding:1,resource:{buffer:this.other().presentation.buffer}}]}));
   pass.dispatchWorkgroups((tiles+63)>>>6);pass.end();
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
  this.resident=false;this.publishTable();this.rebind();this.resized();this.release(old);
 }
 /** The bases from the h generation, canonical texel for canonical texel:
  * after kernels that store through no twin (the dense t=0 authority and
  * publication). Nothing while the fields are their bases. */
 encodeBaseRefresh(encoder:GPUCommandEncoder,written:readonly GPUTexture[]):void{
  if(!this.resident)return;
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
// A tile's origin cell: base texel g is h texel 4g.
${kernel("refresh","dst"," textureStore(dst,vec3i(g),textureLoad(src,vec3i(4u*g),0));")}
// A tile corner: texel l of a base's 9³ block is texel 4l of the h store's 33³ block, skirts included.
${kernel("refreshVertex","dst"," let c=g/9u;textureStore(dst,vec3i(g),textureLoad(src,vec3i(33u*c+4u*(g-9u*c)),0));")}`;
  const face=head("rgba32float")+(`
// Base texel m of a tile is its origin (0) or its +face anchor of axis m-1.
fn ddFace(g:vec3u)->vec3i{let m=g.z&3u;let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));return vec3i(4u*vec3u(g.xy,g.z>>2u)+3u*j);}
${kernel("refresh","dst"," textureStore(dst,vec3i(g),textureLoad(src,ddFace(g),0));")}`);
  const layouts={r32float:layout("r32float"),rgba32float:layout("rgba32float")},compiler=gpuCompilationManagerFor(device);
  const modules={r32float:device.createShaderModule({label:"Uniform detail domain scalar transfer",code:scalar}),rgba32float:device.createShaderModule({label:"Uniform detail domain face transfer",code:face})};
  const pipeline=(format:"r32float"|"rgba32float",entryPoint:string)=>compiler.compileComputePipeline({label:`Uniform detail domain ${entryPoint} ${format}`,
   layout:device.createPipelineLayout({bindGroupLayouts:[layouts[format]]}),compute:{module:modules[format],entryPoint}},{priority:"visible"});
  const pair=async(entryPoint:string):Promise<Record<TransferKind,GPUComputePipeline>>=>{
   const [scalarPipeline,vertexPipeline,facePipeline]=await Promise.all([pipeline("r32float",entryPoint),pipeline("r32float",`${entryPoint}Vertex`),pipeline("rgba32float",entryPoint)]);
   return {cell:scalarPipeline,vertex:vertexPipeline,face:facePipeline};
  };
  return {layouts,refresh:await pair("refresh"),ring:await this.compileRing(domain)};
 }
 /** The ring kernels, a lane per tile. enter:
  * a tile the target generation brings into the ring takes its canonical
  * texels from the base blocks (the h texture held nothing current for it).
  * The first batch also hands the target's ring bit to the other ownership
  * (the all-4h pressure root, whose kernels store these fields too). */
 private async compileRing(domain:UniformDetailDomain):Promise<DomainKernels["ring"]>{
  const device=this.device,C=GPUShaderStage.COMPUTE,N=this.layout.address.tileKeys,compiler=gpuCompilationManagerFor(device);
  if(Math.ceil(N/64)>device.limits.maxComputeWorkgroupsPerDimension)throw new Error(`Uniform detail storage: the ring kernels launch ${Math.ceil(N/64)} workgroups, past one dispatch dimension`);
  const following=this.following,sets:GPUTexture[][]=[];
  // The ring kernels' lattice is their uniform's (the box in tiles, the
  // blocks); table: the topology whose tail holds the pages.
  const seam=(table:string)=>`${uniformWindowAddressWGSL(`${table}`,"5u*drCount()","(drCount()/512u)")}
${uniformLatticeAddressWGSL({constant:false,paged:{blocks:"drDimensions.w"},fine:`${table}[arrayLength(&${table})-1u-drDimensions.w+b]`})}`;
  for(const format of ["rgba32float","r32float"] as const){const of=following.filter(h=>h.format===format);for(let i=0;i<of.length;i+=BATCH)sets.push(of.slice(i,i+BATCH));}
  const head=`@group(0) @binding(${2*BATCH}) var<storage,read> drHeldWords:array<u32>;
@group(0) @binding(${2*BATCH+1}) var<storage,read> drNextWords:array<u32>;
@group(0) @binding(${2*BATCH+4}) var<uniform> drDimensions:vec4u;
fn drTiles()->vec3u{return drDimensions.xyz;}
fn drCount()->u32{return 512u*drDimensions.w;}
// The pages from the held topology: a GPU generation's buffer is its tile
// words alone (no page tail), and no relayout moves a page.
${seam("drHeldWords")}
fn drHeld(t:u32)->bool{return (drHeldWords[2u*drCount()+2u*t+1u]&1u)!=0u;}
fn drNext(t:u32)->bool{return (drNextWords[2u*drCount()+2u*t+1u]&1u)!=0u;}
`;
  const batches=await Promise.all(sets.map(async(handles,n):Promise<RingBatch>=>{
   const format=handles[0]!.format as "r32float"|"rgba32float";
   const layout=device.createBindGroupLayout({label:`Uniform detail ring ${n}`,entries:[
    ...handles.flatMap((_,i)=>[{binding:i,visibility:C,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}},{binding:BATCH+i,visibility:C,storageTexture:{access:"write-only" as const,format,viewDimension:"3d" as const}}]),
    {binding:2*BATCH+4,visibility:C,buffer:{type:"uniform" as const}},
    {binding:2*BATCH,visibility:C,buffer:{type:"read-only-storage" as const}},{binding:2*BATCH+1,visibility:C,buffer:{type:"read-only-storage" as const}},
    ...(n===0?[{binding:2*BATCH+2,visibility:C,buffer:{type:"storage" as const}}]:[])]});
   let seed="";
   handles.forEach((h,i)=>{
    const kind=fields.get(h)!.kind;
    if(kind==="cell"){
     seed+=` textureStore(dst${i},umCellTexel(vec3i(4u*c)),textureLoad(src${i},umTileTexel(vec3i(c)),0));\n`;
    }else if(kind==="vertex"){
     // A corner's home is min(g, T - 1): the far corners belong to the last tile of each axis.
     seed+=` for(var k=0u;k<8u;k++){let e=vec3u(k&1u,(k>>1u)&1u,k>>2u);if(any((e==vec3u(1u))&(c!=drTiles()-vec3u(1u)))){continue;}textureStore(dst${i},umVertexTexel(vec3i(4u*(c+e))),textureLoad(src${i},umWideVertexTexel(vec3i(c+e)),0));}\n`;
    }else{
     // The base stacks a tile's four texels along z.
     seed+=` let b${i}=vec3u(umTileTexel(vec3i(c)));for(var m=0u;m<4u;m++){let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));textureStore(dst${i},umCellTexel(vec3i(4u*c+3u*j)),textureLoad(src${i},vec3i(vec3u(b${i}.xy,4u*b${i}.z+m)),0));}\n`;
    }
   });
   const code=handles.map((_,i)=>`@group(0) @binding(${i}) var src${i}:texture_3d<f32>;\n@group(0) @binding(${BATCH+i}) var dst${i}:texture_storage_3d<${format},write>;\n`).join("")+head
    +(n===0?`@group(0) @binding(${2*BATCH+2}) var<storage,read_write> drShared:array<u32>;\n`:"")+`
@compute @workgroup_size(64) fn enter(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(!umTileLive(t)){return;}
 let near=drNext(t);
${n===0?` let word=2u*drCount()+2u*t+1u;let w=drShared[word];let m=(w&0xfffffffeu)|u32(near);if(m!=w){drShared[word]=m;}
`:""} if(!near||drHeld(t)){return;}
 let c=umTileCoord(t);
${seed}}
`;
   const module=device.createShaderModule({label:`Uniform detail ring ${n}`,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const pipeline=(entryPoint:string)=>compiler.compileComputePipeline({label:`Uniform detail ring ${entryPoint} ${n}`,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}},{priority:"visible"});
   return {handles,layout,enter:await pipeline("enter")};
  }));
  const shareLayout=device.createBindGroupLayout({label:"Uniform detail ring share",entries:[{binding:2,visibility:C,buffer:{type:"uniform"}},{binding:0,visibility:C,buffer:{type:"read-only-storage"}},{binding:1,visibility:C,buffer:{type:"storage"}}]});
  const share=await compiler.compileComputePipeline({label:"Uniform detail ring share",layout:device.createPipelineLayout({bindGroupLayouts:[shareLayout]}),compute:{module:device.createShaderModule({label:"Uniform detail ring share",code:`
@group(0) @binding(2) var<uniform> drDimensions:vec4u;
@group(0) @binding(0) var<storage,read> drFrom:array<u32>;
@group(0) @binding(1) var<storage,read_write> drTo:array<u32>;
fn drCount()->u32{return 512u*drDimensions.w;}
${seam("drFrom")}
@compute @workgroup_size(64) fn share(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(!umTileLive(t)){return;}
 let word=2u*drCount()+2u*t+1u;let w=drTo[word];let m=(w&0xfffffffeu)|(drFrom[word]&1u);if(m!=w){drTo[word]=m;}
}`}),entryPoint:"share"}},{priority:"visible"});
  return {batches,shareLayout,share};
 }

 async prepare(ownership:UniformMixedOwnership):Promise<void>{if(this.transfers)return;this.simulation=ownership;this.transfers=await this.compileDomain(this.layout.domain!);}
 /** Before a remap, in its encoder: the tiles the target generation
  * (target: its topology) brings into the detail ring take their canonical
  * texels from the bases. */
 encodeAdmit(encoder:GPUCommandEncoder,target:GPUBuffer):void{if(this.resident)this.encodeRing(encoder,target);}
 /** After the remap and its phi resolve, in their encoder: a pending
  * retirement is marked for commitRetire. */
 encodeRetire(encoder:GPUCommandEncoder):void{this.encodeRestriction(encoder);}
 /** Logical readback (tight, as upload takes it): loads as the accessors do. */
 async read(texture:GPUTexture):Promise<Float32Array>{
  const encoder=this.device.createCommandEncoder({label:"Uniform detail field readback"}),done=this.capture(encoder,texture);
  this.device.queue.submit([encoder.finish()]);return done();
 }
 /** A field's logical values at this point of an encoder (a mid-frame
  * diagnostic capture): the physical texels and the residency they are
  * placed by. Resolve the result after the encoder is submitted. */
 capture(encoder:GPUCommandEncoder,texture:GPUTexture):()=>Promise<Float32Array>{
  const kind=this.kindOf(texture),[lx,ly,lz]=this.logicalOf(texture),comp=texture.format==="rgba32float"?4:1;
  const source=this.physical(texture);
  const W=source.width,H=source.height,Z=source.depthOrArrayLayers,row=W*comp*4,pitch=Math.ceil(row/256)*256;
  const usage=GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ;
  // Residency is the host's, so this encode's is known here: the source is the field's h store.
  const resident=this.twinned(texture);
  const staging=this.device.createBuffer({size:pitch*H*Z,usage});
  encoder.copyTextureToBuffer({texture:source},{buffer:staging,bytesPerRow:pitch,rowsPerImage:H},[W,H,Z]);
  // Outside the ring a canonical texel lives in the base alone, so every one is read from there.
  const base=resident?this.bases.get(fields.get(texture)!.handle)!:undefined;
  const bw=base?.width??0,bh=base?.height??0,bd=base?.depthOrArrayLayers??0,brow=bw*comp*4,bpitch=Math.ceil(brow/256)*256;
  const beside=base?this.device.createBuffer({size:bpitch*bh*bd,usage}):undefined;
  if(base)encoder.copyTextureToBuffer({texture:base},{buffer:beside!,bytesPerRow:bpitch,rowsPerImage:bh},[bw,bh,bd]);
  return async()=>{
   let image:Float32Array,held:Float32Array|undefined;
   try{
    await Promise.all([staging.mapAsync(GPUMapMode.READ),beside?.mapAsync(GPUMapMode.READ)]);
    const src=new Uint8Array(staging.getMappedRange()),dst=new Uint8Array(row*H*Z);
    for(let r=0;r<H*Z;r++)dst.set(src.subarray(r*pitch,r*pitch+row),r*row);
    image=new Float32Array(dst.buffer);
    if(beside){
     const packed=new Uint8Array(beside.getMappedRange()),tight=new Uint8Array(brow*bh*bd);
     for(let r=0;r<bh*bd;r++)tight.set(packed.subarray(r*bpitch,r*bpitch+brow),r*brow);
     held=new Float32Array(tight.buffer);
    }
   }finally{for(const b of [staging,beside]){if(b?.mapState==="mapped")b.unmap();b?.destroy();}}
   return kind==="vertex"?this.logicalVertex([lx,ly,lz],image,[W,H],resident,held,[bw,bh]):this.logical(kind,[lx,ly,lz],comp,image,[W,H],resident,held,[bw,bh]);
  };
 }
 /** A cell or face field's logical values from its physical images. With an
  * h store: each texel from it, a canonical one (a tile's origin, a face
  * field's +face anchors) from the base beside it, where it lives alone
  * outside the ring. A base block alone broadcasts its texels as the
  * accessors load them. */
 private logical(kind:UniformDetailClass,[lx,ly,lz]:Dims,comp:number,image:Float32Array,[W,H]:readonly [number,number],resident:boolean,held:Float32Array|undefined,[bw,bh]:readonly [number,number]):Float32Array{
  const out=new Float32Array(lx*ly*lz*comp);if(kind==="atlas")return out;
  const k=kind==="cell"?"cell":"face";
  for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++){
   const jx=(x&3)===3?1:0,jy=(y&3)===3?1:0,jz=(z&3)===3?1:0,j=jx+jy+jz,m=jx+2*jy+3*jz;
   // Canonical: every local coordinate 0, or (a face field) one of them 3.
   const canonical=((x&3)%3|(y&3)%3|(z&3)%3)===0&&(k==="face"?j<=1:j===0);
   let from=image,src:number;
   if(resident){
    if(canonical&&held){from=held;src=this.baseIndex(k,x>>2,y>>2,z>>2,m,bw,bh);}else src=this.cellIndex(x,y,z,W,H);
   }else{
    if(k==="face"&&j>1)continue;
    src=this.baseIndex(k,x>>2,y>>2,z>>2,k==="face"?m:0,W,H);
   }
   const dst=(x+lx*(y+ly*z))*comp;src*=comp;for(let c=0;c<comp;c++)out[dst+c]=from[src+c]!;
  }
  return out;
 }
 /** A vertex field's logical values from its physical images: each vertex
  * from the h store, a tile corner from the base beside it; a base block
  * alone gives the corners and each hanging vertex's trilinear interpolant. */
 private logicalVertex([lx,ly,lz]:Dims,image:Float32Array,[W,H]:readonly [number,number],resident:boolean,held:Float32Array|undefined,[bw,bh]:readonly [number,number]):Float32Array{
  const out=new Float32Array(lx*ly*lz),f=Math.fround,[tx,ty,tz]=this.layout.tiles,v=new Float64Array(8);
  if(resident){
   for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++)out[x+lx*(y+ly*z)]=held&&!((x|y|z)&3)?held[this.cornerIndex(x>>2,y>>2,z>>2,bw,bh)]!:image[this.vertexIndex(x,y,z,W,H)]!;
   return out;
  }
  for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++){
   const dst=x+lx*(y+ly*z);
   if(!((x|y|z)&3)){out[dst]=image[this.cornerIndex(x>>2,y>>2,z>>2,W,H)]!;continue;}
   // Hanging vertex: trilinear in its home tile's corners, summed in umVertexSum8 order.
   const hx=Math.min(x>>2,tx-1),hy=Math.min(y>>2,ty-1),hz=Math.min(z>>2,tz-1),ux=(x-4*hx)/4,uy=(y-4*hy)/4,uz=(z-4*hz)/4;
   for(let k=0;k<8;k++){
    const cx=k&1,cy=(k>>1)&1,cz=k>>2,weight=f(f((cx?ux:1-ux)*(cy?uy:1-uy))*(cz?uz:1-uz));
    v[k]=weight>0?f(weight*image[this.cornerIndex(hx+cx,hy+cy,hz+cz,W,H)]!):0;
   }
   out[dst]=f(f(f(v[0]!+v[5]!)+f(v[1]!+v[4]!))+f(f(v[2]!+v[7]!)+f(v[3]!+v[6]!)));
  }
  return out;
 }
}

/** Field class by binding name in the mixed shaders. }
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
const RAW_NAMES=new Set(["coarse","coarseOut","coarseExtended","coarseVelocity","origins","originsOut","surfacePhi","surfaceOpen","unitHalo","unitHaloOut","unitHaloFiltered","coarseFiltered","unitHaloHalfOut"]);
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

/** The domain placement's accessors (UniformDetailDomain), addressed through
 * the lattice address functions of the topology the kernel loads
 * (uniform-lattice-address.ts). q is a logical coordinate. b = 1 while the
 * bound texture is the field's base block, told by its extent: a base block
 * is narrower than any h store. The bound texture's texel of q:
 *  cell          umCellTexel(q) in an h store (its block at its h slot), in
 *                a base umCellTexelOf(q, false) >> 2 (its block at its own)
 *  vertex        umVertexTexel(q) in an h store, umWideVertexTexel(q >> 2)
 *                in a base: a block is 33 vertices against 9
 *  face          udFace: on a base, the tile's texel c with z = 4 c.z +
 *                j.x+2j.y+3j.z, of the nearest stored q (j of an outside q
 *                is not its edge texel's)
 *  atlas-only    the h texel; zero while b
 * A load outside the stored coordinates reads what the address functions
 * give there (the nearest stored texel by an unsigned clamp), which each
 * base map but a face's commutes with. A store outside them is dropped: its
 * texel is negative. A canonical load (udLoad4_) reads a base and keeps the
 * raw load. A store to a base is range-checked here and dropped unless
 * canonical, with a violation bit (dense: silently, a lattice launch
 * deposits exactly the texels that are state). call: "()" where the
 * module's address extents are functions.
 * Runs (vertex fields; emitted where the code calls them by name): a box of
 * vertices q..q+n on every axis that one block holds (umVertexRun) is
 * addressed by one table read. udRun_<field>(q,n) gives the bound texture's
 * texel of q and, in w, the shift from a vertex's offset from q to its
 * texel's offset (0 in an h store; 2 in a base, where q must be a tile
 * corner and the texels are the tile corners at and below each vertex), or
 * w < 0 where there is no run and each vertex is loaded as before.
 * udTexel_<field>(a) loads the bound texture at such a texel. Of an
 * atlas field (no runs) it is udLoad_ by the h texel a = umCellTexel(q)
 * (zero while the bound texture is a base): one page's cells are one
 * block's, so a cell's texel steps to its in-page neighbours' directly.
 * udLoad8_<field>(q,s) is udLoad_<field> of the eight vertices q + s{0,1}³,
 * x fastest, by a run where there is one, else by the two pages' words where
 * they span two pages on one axis. udRun4_, udTexel4_, udLoad48_ are
 * the same of the base twin a canonical load reads (udLoad4_). */
function detailDomainWGSL(kinds:ReadonlyMap<string,{kind:UniformDetailClass;type:string}>,support:string|undefined,atomic:boolean,deposit:boolean,hOnly?:ReadonlySet<string>,twins?:ReadonlyMap<string,string>,based?:ReadonlySet<string>,
 /** The mixed kernels' ring: the topology namespaces of the fields a kernel
  * loads (pf, group 0) and stores (target: the remap's adopted generation,
  * else pf), and whether every call carries a site index. */
 ring?:{readonly pf:string;readonly target:string;readonly sites:boolean},
 /** Fields with a UNIFORM_DETAIL_CANONICAL_LOAD. */
 either?:ReadonlySet<string>,
 /** Fields with a UNIFORM_DETAIL_GUARD_LOAD. */
 guards?:ReadonlySet<string>,
 /** Run accessors the code calls, by name (udRun_phi, udLoad48_phi, ...). */
 asked?:ReadonlySet<string>,
 call=""):string{
 const j="(((q&vec3i(3))+vec3i(1))>>vec3u(2u))";
 // ringed: a canonical store reaches the h texture only inside the ring.
 // watch (checked): every h texture load is tested against the ring (and the poison).
 const ringed=!!ring,watch=false,sites=watch&&ring!.sites;
 const site=sites?",site:u32":"",pass=sites?",site":"";
 let ringCode="";
 if(ring){
  // A texel's home tile q >> 2, every class: a tile with nothing stored is outside the ring.
  const fn=(name:string,p:string)=>`fn ${name}(q:vec3i)->bool{let k=${p}umTileKey(q>>vec3u(2u));if(!${p}umKeyStored(k)){return false;}return (${p}umTopology[2u*${p}UM_TILES+2u*k+1u]&1u)!=0u;}\n`;
  ringCode+=fn("udRingStore",ring.target);
  // A remap loads fields of the generation it leaves and of the one it adopts (whose entering tiles are seeded).
  ringCode+=ring.target===ring.pf?"fn udRingLoad(q:vec3i)->bool{return udRingStore(q);}\n":fn("udRingHeld",ring.pf)+"fn udRingLoad(q:vec3i)->bool{return udRingHeld(q)||udRingStore(q);}\n";
 }
 if(watch){
  const word=`${ring!.pf}umSupport[${ring!.pf}UM_DETAIL_VIOLATION+4u+site]`,R=UNIFORM_DETAIL_RING_SITE;
  const mark=!sites?"":atomic?`if((atomicLoad(&${word})&bits)!=bits){atomicOr(&${word},bits);}`:`if((${word}&bits)!=bits){${word}=${word}|bits;}`;
  ringCode+=`fn udWatch(q:vec3i,v:f32${site}){
 var bits=${R.ran}u;if(!udRingLoad(q)){bits|=${R.outside}u;}${""}
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
// A launch whose every field load runs in an h tile's stencil, on the h group (UNIFORM_DETAIL_H_STORE).
override udHStore:bool=false;
// A launch whose vertex loads take one slot walk for both classes (UNIFORM_DETAIL_MERGED_SLOTS).
override udMergedSlots:bool=false;
fn udViolation(bits:u32){${!support?"":atomic?`atomicOr(&${support},bits);`:`${support}=${support}|bits;`}}
fn udAligned(q:vec3i)->bool{return all((q&vec3i(3))==vec3i(0));}
fn udFaceCanonical(q:vec3i)->bool{let l=q&vec3i(3);let j=${j};return all((l==vec3i(0))|(l==vec3i(3)))&&j.x+j.y+j.z<=1;}
${`fn udFace(q:vec3i,b:u32)->vec3i{let c=q>>vec3u(2u*b);let j=${j};return vec3i(c.xy,c.z*i32(1u+3u*b)+(j.x+2*j.y+3*j.z)*i32(b));}\n`}${ringCode}`;
 /** The address functions of a field's class. load: the texel of a logical
  * coordinate in the h store (b "0u") or in the texture b tells (0 the h
  * store, 1 a base block); store: the same of a stored coordinate, negative
  * of any other; present: the coordinate is stored; size: the h store's
  * extent. No box: a coordinate resolves through the page table. */
 interface DetailAddress{load(v:string,b:string):string;store(v:string,b:string):string;present(v:string):string;readonly size:string}
 const pf=ring?.pf??"",H="0u";
 // A cell or face field: its h store is a cell texture, its base a tile texture (a face base stacks four texels per tile
 // along z). A base block's texel is a map of the h store's.
 const cellAddress=(face:boolean):DetailAddress=>{
  // An h store places a block at its h slot, a base at its own: the texel b tells, of the h lattice, mapped.
  const base=(texel:string,b:string)=>b===H?texel:face?`udFace(${texel},${b})`:`(${texel}>>vec3u(2u*${b}))`;
  const of=(f:string,v:string,b:string)=>b===H?`${pf}${f}(${v})`:`${pf}${f}Of(${v},${b}==0u)`;
  return {load:(v,b)=>base(of("umCellTexel",v,b),b),store:(v,b)=>base(of("umCellStoreTexel",v,b),b),present:v=>`${pf}umCellPresent(${v})`,
   size:`${pf}UM_CELL_TEXTURE${call}`};
 };
 // A vertex field: a block of its h store holds a page's vertices, 33 on an axis; a block of its base the page's tile
 // corners, 9 on an axis. Each has its address function: neither texel is a map of the other.
 const vertexAddress:DetailAddress={
  load:(v,b)=>b===H?`${pf}umVertexTexel(${v})`:b==="1u"?`${pf}umWideVertexTexel((${v})>>vec3u(2u))`:`udVertex(${v},${b})`,
  store:(v,b)=>b===H?`${pf}umVertexStoreTexel(${v})`:b==="1u"?`${pf}umWideVertexStoreTexel((${v})>>vec3u(2u))`:`udVertexStore(${v},${b})`,
  present:v=>`${pf}umVertexPresent(${v})`,
  size:`${pf}UM_VERTEX_TEXTURE${call}`};
 if([...kinds.values()].some(k=>k.kind==="vertex"))out+=`fn udVertex(q:vec3i,b:u32)->vec3i{
 if(udMergedSlots){let bits=select(5u,3u,b!=0u);let s=udVertexSlot(select(q,q>>vec3u(2u),b!=0u),bits);let c=udVertexCoord(min(s.w,${pf}UM_BLOCKS${call}-1u),b);return (c<<vec3u(bits))+c+vec3i(s.xyz);}
 if(b==0u){return ${pf}umVertexTexel(q);}return ${pf}umWideVertexTexel(q>>vec3u(2u));
}
fn udVertexStore(q:vec3i,b:u32)->vec3i{
 if(udMergedSlots){let bits=select(5u,3u,b!=0u);let s=udVertexSlot(select(q,q>>vec3u(2u),b!=0u),bits);if(!${pf}umBlockStored(s.w)){return vec3i(-1);}let c=udVertexCoord(s.w,b);return (c<<vec3u(bits))+c+vec3i(s.xyz);}
 if(b==0u){return ${pf}umVertexStoreTexel(q);}return ${pf}umWideVertexStoreTexel(q>>vec3u(2u));
}
// Where a vertex texture places block w: an h store (b 0) at its h slot's place, a base at its own.
fn udVertexCoord(w:u32,b:u32)->vec3i{if(b==0u){return ${pf}umFineCoord(w);}return ${pf}umBlockCoord(w);}
// umVertexSlot (bits 5) or umWideVertexSlot (bits 3) of v, by one walk.
fn udVertexSlot(v:vec3i,bits:u32)->vec4u{
 let mask=(1<<bits)-1;
 let c=v-vec3i(1);let page=c>>vec3u(bits);let own=vec3u(c&vec3i(mask))+vec3u(1u);
 let b=${pf}umPageWord(page);if(${pf}umBlockStored(b)){return vec4u(own,b);}
 let plane=(v&vec3i(mask))==vec3i(0);
 for(var k=1u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
  if(any(step&!plane)){continue;}
  let s=${pf}umPageWord(page+vec3i(step));
  if(${pf}umBlockStored(s)){return vec4u(select(own,vec3u(0u),step),s);}
 }
 return vec4u(own,b);
}
`;
 // udVertex of every vertex q..q+n by one table read: xyz udVertex(q,b), w the shift from an offset o to its texel's
 // (udVertex(q+o,b) = xyz + (o >> w)); w < 0 where they are not one block's (or q is no tile corner of a base).
 // An atlas field's texel load (udTexel_) needs no run.
 const runs=[...asked??[]].filter(name=>kinds.get(name.slice(name.indexOf("_")+1))?.kind!=="atlas");
 if(runs.length)out+=uniformLatticeRunWGSL(pf);
 if(runs.some(name=>name.startsWith("ud")))out+=`fn udVertexRun(q:vec3i,n:u32,b:u32)->vec4i{
 if(b==0u){let s=${pf}umVertexRun(q,n);if(s.x==0u){return vec4i(-1);}return vec4i(${pf}umVertexRunTexel(s),0);}
 let s=${pf}umWideVertexRun(q>>vec3u(2u),n>>2u);if(s.x==0u||!udAligned(q)){return vec4i(-1);}return vec4i(${pf}umWideVertexRunTexel(s),2);
}
`;
 const one=(name:string,kind:UniformDetailClass,type:string,sfx:string):string=>{
  let out="";
  const storage=type.startsWith("texture_storage"),level=storage?"":",0";
  // An atlas-only name binds a face field.
  const A=kind==="vertex"?vertexAddress:cellAddress(kind==="face");
  const inside=(v:string)=>A.present(v),outside=(v:string)=>`!${A.present(v)}`;
  // A base block is narrower than the h store. Under udHStore the bound field is its h texture and a twin its base.
  const based_=(t:string)=>`select(u32(textureDimensions(${t}).x<${A.size}.x),${t.startsWith("udb_")?"1u":"0u"},udHStore)`,extent=based_(name);
  const mode=`let b=${extent};`;
  const canonical=(v:string)=>kind==="face"?`udFaceCanonical(${v})`:`udAligned(${v})`;
  // The bound texture's texel of a load at v.
  const at=(v:string)=>A.load(v,"b");
  // The h store's texel of a load at v, and the bound texture's by b (the raw load).
  const stored=(v:string)=>A.load(v,H);
  const raw=(v:string)=>A.load(v,"b");
  // The bound texture's texel of a store at v, negative where nothing is stored; a store at a v that may not be stored.
  const put=(v:string)=>A.store(v,kind==="atlas"?H:"b");
  const lay=(v:string)=>`let a=${put(v)};if(a.x<0){return;}textureStore(${name},a,value);`;
  const home=(v:string)=>A.load(v,"1u");
  // A stored field's twin is its other home (the base beside an h texture,
  // the h texture beside a base, a 1³ stand-in while there is one home):
  // a canonical stored v's texel there, by the bound texture's b.
  const twin=twins?.get(name),other=(v:string)=>A.store(v,"(1u-b)");
  if(!storage||type.includes("read_write")){
   const check="";
   const load=`textureLoad(${name},${at("q")}${level})`;
   out+=`fn udLoad_${name}${sfx}(q:vec3i${site})->vec4f{
 ${mode}
 ${check}${kind==="atlas"?`return select(textureLoad(${name},${stored("q")}${level}),vec4f(0.0),b!=0u);`
   :watch?`let v=${load};if(b==0u){udWatch(q,v.x${pass});}return v;`:`return ${load};`}
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
   const twinBase=storage?`textureDimensions(${twin}).x>1u`:`textureDimensions(${twin}).x<${A.size}.x`;
   out+=`fn udLoadC_${name}${sfx}(q:vec3i${site})->vec4f{
 ${mode}
 if(b==0u&&${twinBase}&&${inside("q")}&&${canonical("q")}){${watch?`let v=textureLoad(${twin},${home("q")}${level});udWatchHomes(q,v,textureLoad(${name},${stored("q")}${level})${pass});return v;`:`return textureLoad(${twin},${home("q")}${level});`}}
 return udLoad_${name}${sfx}(q${pass});
}
`;
  }
  // A canonical load (UNIFORM_DETAIL_4H_LOAD): the base block, bound as the
  // sampled twin at every capacity (b by its extent, as any load's: a
  // texture that is not a field binds itself there).
  if(based?.has(name)){
   const b=based_(twin!);
   const both=watch&&kind!=="atlas";
   out+=`fn udLoad4_${name}${sfx}(q:vec3i${site})->vec4f{
 let b=${b};
 ${""}${both?`let v=textureLoad(${twin},${raw("q")},0);if(b!=0u&&${extent}==0u){udWatchHomes(q,v,textureLoad(${name},${stored("q")},0)${pass});}return v;`:`return textureLoad(${twin},${raw("q")},0);`}
}
`;
  }
  if(hOnly?.has(name))out+=`fn udLoadH_${name}${sfx}(q:vec3i${site})->vec4f{
 ${""}${watch&&kind!=="atlas"?`let v=textureLoad(${name},${stored("q")}${level});udWatch(q,v.x${pass});return v;`:`return textureLoad(${name},${stored("q")}${level});`}
}
`;
  // The run accessors the code calls: of the bound texture, and (4) of the base twin a canonical load reads. The
  // eight-vertex load's last loop (two or three page crossings) is udLoad_ (udLoad4_) of each vertex, unchanged.
  // The eight vertices of class b (a base: (q + o) >> 2 on its 4h lattice) whose own pages (the pages of the cells
  // below them) and the pages a slot walk may step to span two pages on exactly one axis: both pages' words are read
  // once and each vertex's texel is a select of them, udVertex's texel. A vertex's own page is page or page + cross;
  // of page's, only one on the face between them walks, to page + cross, whose skirt (local 0) holds it where page
  // is not stored and page + cross is. Inline: a helper returning the two blocks measured slower.
  const pair=(texture:string,lv:string)=>` let sh=select(0u,2u,b!=0u);let bits=select(5u,3u,b!=0u);let mask=(1<<bits)-1;
 let page=((q>>vec3u(sh))-vec3i(1))>>vec3u(bits);let cross=vec3i((((q+vec3i(i32(s)))>>vec3u(sh))>>vec3u(bits))!=page);
 if(cross.x+cross.y+cross.z==1){
  let w0=${pf}umPageWord(page);let w1=${pf}umPageWord(page+cross);let walk=!${pf}umBlockStored(w0)&&${pf}umBlockStored(w1);
  let c0=udVertexCoord(min(w0,${pf}UM_BLOCKS${call}-1u),b);let c1=udVertexCoord(min(w1,${pf}UM_BLOCKS${call}-1u),b);
  let base0=(c0<<vec3u(bits))+c0+vec3i(1);let base1=(c1<<vec3u(bits))+c1+vec3i(1);
  for(var k=0;k<8;k++){
   let at=((q+i32(s)*vec3i(k&1,(k>>1)&1,k>>2))>>vec3u(sh))-vec3i(1);let l=at&vec3i(mask);
   let upper=any(((at>>vec3u(bits))-page)!=vec3i(0));let skirt=walk&&any((l==vec3i(mask))&(cross!=vec3i(0)));
   v[k]=textureLoad(${texture},select(select(base0+l,base1+select(l,vec3i(-1),cross!=vec3i(0)),skirt),base1+l,upper)${lv}).x;
  }
  return v;
 }`;
  for(const [four,texture,b,lv] of [["",name,extent,level],["4",twin,twin?based_(twin):"",",0"]] as const){
   const run=`udRun${four}_${name}`,texel=`udTexel${four}_${name}`,eight=`udLoad${four}8_${name}`,corner="vec3i(k&1,(k>>1)&1,k>>2)";
   if(!asked?.has(run)&&!asked?.has(texel)&&!asked?.has(eight))continue;
   if(kind==="atlas"&&!four&&!asked.has(run)&&!asked.has(eight)){out+=`fn ${texel}${sfx}(a:vec3i)->vec4f{return select(textureLoad(${name},a${level}),vec4f(0.0),${extent}!=0u);}\n`;continue;}
   if(kind!=="vertex"||!texture||watch)throw new Error(`Uniform detail rewrite: ${name} has no run accessor ${four?"of a base twin":""}`);
   if(asked.has(run))out+=`fn ${run}${sfx}(q:vec3i,n:u32)->vec4i{return udVertexRun(q,n,${b});}\n`;
   if(asked.has(texel))out+=`fn ${texel}${sfx}(a:vec3i)->vec4f{return textureLoad(${texture},a${lv});}\n`;
   if(asked.has(eight))out+=`fn ${eight}${sfx}(q:vec3i,s:u32)->array<f32,8>{
 let b=${b};let g=udVertexRun(q,s,b);var v:array<f32,8>;
 if(g.w>=0){let e=i32(s>>u32(g.w));for(var k=0;k<8;k++){v[k]=textureLoad(${texture},g.xyz+e*${corner}${lv}).x;}return v;}
${pair(texture!,lv)}
 for(var k=0;k<8;k++){v[k]=textureLoad(${texture},udVertex(q+i32(s)*${corner},b)${lv}).x;}
 return v;
}
`;
  }
  if(storage){
   const fail=(bits:number)=>deposit?"return;":`udViolation(${bits}u);return;`;
   const within=inside("q");
   // The ring. On a base: the h twin takes the texel inside the
   // ring. On an h texture: a canonical texel goes to the base twin, and to
   // the h texture only inside the ring; any other texel is the h texture's.
   if(twin&&kind!=="atlas"&&(ringed)){
    // An h texture's canonical texel is stored to both homes untested: the
    // test drops a store only outside the
    // ring, where the h texel exists and is never loaded, and costs a
    // topology read at every canonical store inside it. (A base's h twin
    // keeps the test: there it saves the store.) Not under the QA fills.
    const gate=watch;
    out+=`fn udStore_${name}${sfx}(q:vec3i,value:vec4f${site}){
 ${mode}
 if(b!=0u){
  if(${outside("q")}){return;}if(!${canonical("q")}){${fail(UNIFORM_DETAIL_VIOLATION[kind])}}
  textureStore(${name},${put("q")},value);
  if(textureDimensions(${twin}).x>1u${ringed?"&&udRingStore(q)":""}){textureStore(${twin},${other("q")},value);}
  return;
 }
 ${watch&&ringed
  ?`if(textureDimensions(${twin}).x>1u&&${within}){if(${canonical("q")}){textureStore(${twin},${other("q")},value);if(!udRingStore(q)){return;}}else if(!udRingStore(q)){udWatchStore(${sites?"site":""});}}`
  :`if(textureDimensions(${twin}).x>1u&&${within}&&${canonical("q")}){textureStore(${twin},${other("q")},value);${gate?"if(!udRingStore(q)){return;}":""}}`}
 ${lay("q")}
}
`;
    return out;
   }
   out+=`fn udStore_${name}${sfx}(q:vec3i,value:vec4f${site}){
 ${mode}
 ${kind==="atlas"?`if(b!=0u){${fail(UNIFORM_DETAIL_VIOLATION.atlas)}}\n ${lay("q")}`
   :`if(b!=0u){if(${outside("q")}){return;}if(!${canonical("q")}){${fail(UNIFORM_DETAIL_VIOLATION[kind])}}}\n ${lay("q")}${
   // With two homes, a canonical texel is stored to both: a load takes it
   // from whichever texture its kernel binds, with no test.
   twin?`\n if(textureDimensions(${twin}).x>1u&&${inside("q")}&&${canonical("q")}){textureStore(${twin},${other("q")},value);}`:""}`}
}
`;
  }
  return out;
 };
 for(const [name,{kind,type}] of kinds)out+=one(name,kind,type,"");
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
 const keys=new RegExp(`const ${pf}UM_TILES:u32=(\\d+)u;`).exec(code);
 if(!keys)throw new Error("Uniform detail rewrite: no mixed tile key count");
 const layout=device?liveLayout(device,Number(keys[1])):undefined;
 if(!layout)return code;
 {
  // Survey and poison runs number every load and store but the canonical (base) loads.
  const domain=layout.domain;
  code=code.replaceAll(UNIFORM_DETAIL_RING_4H_LOAD,UNIFORM_DETAIL_4H_LOAD);
  const sites=false;
  const hOnly=new Set<string>(),based=new Set<string>(),either=new Set<string>();
  const calls=rewriteCalls(code,new Set(kinds.keys()),hOnly,undefined,based,undefined,either);
  // The run accessors the code calls by name; one of the base twin makes its field based, as a canonical load does.
  const asked=new Set<string>();
  for(const [,accessor,four,field] of calls.matchAll(/\b(ud(?:Run|Texel)(4?)_(\w+))\(/g)){asked.add(accessor!);if(four)based.add(field!);}
  for(const [,accessor,four,field] of calls.matchAll(/\b(udLoad(4?)8_(\w+))\(/g)){asked.add(accessor!);if(four)based.add(field!);}
  for(const name of asked){const kind=kinds.get(name.slice(name.indexOf("_")+1))?.kind;if(kind!=="vertex"&&!(kind==="atlas"&&name.startsWith("udTexel_")))throw new Error(`Uniform detail rewrite: ${name} is not a vertex field's accessor`);}
  // The seam's run forms read directly (a key run): the module carries them too.
  if(/\bumVertexRun\(/.test(calls))asked.add("umVertexRun");
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
  // A remap stores the fields of the generation it adopts: the unprefixed topology beside the one it leaves.
  const target=pf&&/var<storage,\s*read>\s*umTopology\s*:/.test(code)?"":pf;
  let rewritten=calls+declarations+detailDomainWGSL(kinds,`${pf}umSupport[${pf}UM_DETAIL_VIOLATION]`,atomic,false,hOnly,twins,based,{pf,target,sites},either,new Set([...kinds.keys()].filter(name=>calls.includes(`udLoadG_${name}(`))),asked);
  if(code.includes(`fn ${pf}umPlannedVertexSlot(`))rewritten=rewritten.replace("fn udVertexSlot(v:vec3i,bits:u32)->vec4u{",`fn udVertexSlot(v:vec3i,bits:u32)->vec4u{let plan=${pf}umPlannedVertexSlot(v,bits);if(plan.w!=0xffffffffu){return plan;}`);
  return rewritten;
 }

}
/** A dense reference shader (the t=0 authority and publication) through the
 * same accessors: `classes` names its field bindings. A store outside the canonical texels is
 * dropped, so a lattice launch deposits exactly the texels that are state. */
export function uniformDetailDenseShader(code:string,classes:Readonly<Record<string,UniformDetailClass>>):string{
 const kinds=new Map<string,{kind:UniformDetailClass;type:string}>();
 for(const [,name,type] of code.matchAll(/var\s+(\w+)\s*:\s*(texture(?:_storage)?_3d<[^>]+>)/g)){const kind=classes[name!];if(kind)kinds.set(name!,{kind,type:type!});}
 for(const name of Object.keys(classes))if(!kinds.has(name))throw new Error(`Uniform detail dense rewrite: no 3D texture ${name}`);
 if(/fn\s+\w+\s*\([^)]*:\s*texture/.test(code))throw new Error("Uniform detail dense rewrite: a texture-typed function parameter bypasses the field accessors");
 for(const name of kinds.keys())if(new RegExp(`texture(?:Dimensions|SampleLevel)\\(\\s*${name}\\b`).test(code))throw new Error(`Uniform detail dense rewrite: ${name} is read past the accessors`);
 // The module's address extents are functions of its lattice.
 return rewriteCalls(code,new Set(kinds.keys()))+detailDomainWGSL(kinds,undefined,false,true,undefined,undefined,undefined,undefined,undefined,undefined,undefined,"()");
}
// The window overrides declared by each rewritten module. The opt-in lives
// on a solver ownership, never on a GPU device shared by several scenes.
const addressPlanPrefixes=new WeakMap<GPUShaderModule,readonly string[]>();
const fixedWorldPrefixes=new WeakMap<GPUShaderModule,readonly string[]>();
const runtimeWorldPrefixes=new WeakMap<GPUShaderModule,ReadonlySet<string>>();
/** createShaderModule through the central rewrite. Hot sampling modules can
 * retain immutable world/frame overrides: specialization removes repeated
 * bounds/address branches. The ownership must still promise fixed geometry;
 * moving frames and mutable worlds retain their runtime paths. */
export function uniformDetailModule(device:GPUDevice,descriptor:GPUShaderModuleDescriptor & {specializeWorldGeometry?:boolean}):GPUShaderModule{
 const {specializeWorldGeometry=false,...shader}=descriptor;
 const code=uniformSceneConstantsWGSL(uniformDetailShader(shader.code,device),specializeWorldGeometry);
 const module=device.createShaderModule({...shader,code});
 addressPlanPrefixes.set(module,[...code.matchAll(/\boverride (\w*)UseAddressPlan:bool=false;/g)].map(match=>match[1]!));
 const prefixes=[...code.matchAll(/\boverride (\w*)FixedWorldGeometry:bool=false;/g)].map(match=>match[1]!);
 if(prefixes.length){
  fixedWorldPrefixes.set(module,prefixes);
  runtimeWorldPrefixes.set(module,new Set(prefixes.filter(prefix=>!code.includes(`override ${prefix}WindowWord24:`))));
 }
 return module;
}
const storageOf=new WeakMap<UniformMixedOwnership,UniformDetailStorage>();
/** Every pipeline is compiled once. */
export function uniformDetailPipeline(device:GPUDevice,ownership:UniformMixedOwnership,descriptor:GPUComputePipelineDescriptor):Promise<GPUComputePipeline>{
 if(UNIFORM_ADDRESS_PLAN_ENTRIES.has(descriptor.compute.entryPoint??"")){
  const constants:Record<string,number>={};for(const prefix of addressPlanPrefixes.get(descriptor.compute.module)??[])constants[`${prefix}UseAddressPlan`]=1;
  descriptor={...descriptor,compute:{...descriptor.compute,constants:{...constants,...descriptor.compute.constants}}};
 }
 const prefixes=ownership.fixedWorldGeometry?fixedWorldPrefixes.get(descriptor.compute.module):undefined;
 if(prefixes?.length){
  const seed=ownership.capacity.address.seed,at=seed.layout.tail.window,frame=uniformWindowFixedFrame(seed.tail[at+UNIFORM_WINDOW_HEADER.policy]!);
  const word=(i:number)=>seed.tail[at+i]!|0,last=word(UNIFORM_WINDOW_HEADER.windowLast)>>>0;
  const inside=[0,1,2].every(a=>{const f=word(UNIFORM_WINDOW_HEADER.frame+a),lo=word(UNIFORM_WINDOW_HEADER.solidLow+a),hi=word(UNIFORM_WINDOW_HEADER.solidHigh+a);return lo>=f&&hi<=f+last+1;});
  const constants:Record<string,number>={};
  for(const prefix of prefixes){
   constants[`${prefix}FixedWorldGeometry`]=1;
   if(runtimeWorldPrefixes.get(descriptor.compute.module)?.has(prefix))continue;
   for(let i=24;i<32;i++)constants[`${prefix}WindowWord${i}`]=seed.tail[at+i]!;
   if(frame){
    constants[`${prefix}FixedFrame`]=1;for(let i=0;i<3;i++)constants[`${prefix}WindowWord${i}`]=seed.tail[at+UNIFORM_WINDOW_HEADER.frame+i]!;
    // A world box inside the fixed window: a page off the window is off the box, so SOLID whatever the window test says.
    if(inside)constants[`${prefix}WindowWord${UNIFORM_WINDOW_HEADER.windowLast}`]=0xffffffff;
   }
  }
  descriptor={...descriptor,compute:{...descriptor.compute,constants:{...constants,...descriptor.compute.constants}}};
 }
 return device.createComputePipelineAsync(descriptor);
}
export function uniformDetailPick(pipeline:GPUComputePipeline):GPUComputePipeline{return pipeline;}
