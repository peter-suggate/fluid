/** Detail storage for the h-sized fields of the mixed Uniform frame
 * (docs/plans/uniform-4h-first-storage-design-2026-10-03.md). Every h field is
 * one physical texture: a base block (the H value of every tile) plus an
 * atlas of patch slots (h texels of resident patches only). A patch is
 * P³ h = (P/4)³ tiles. The table (header + patch directory) sits past the
 * tile words of the mixed topology buffer (UM_DETAIL), so no binding is added.
 * uniformDetailShader rewrites every textureLoad/textureStore of a field into
 * class accessors at shader assembly, with the layout baked in as constants
 * and only the directory word read at run time. Identity placement (every
 * patch resident at its logical place) needs no translation and compiles to
 * the raw loads.
 * The default placement is the domain placement (UniformDetailDomain, since
 * 4 October 2026): each field is one texture, its base block while no h tile
 * is reserved and its logical h texture otherwise, with no atlas and no
 * directory read. The patch atlas above and identity are QA arms.
 * "Detail storage" is unrelated to the far-air page residency
 * (umResidentAllOwner). */
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedDetailViolationWord, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { UniformPipelineNeeds, unprepared } from "./uniform-pipeline-needs";
import { UNIFORM_DETAIL_4H_LOAD, UNIFORM_DETAIL_BASED, UNIFORM_DETAIL_CANONICAL_LOAD, UNIFORM_DETAIL_FACE_COMPACT, UNIFORM_DETAIL_HEADER_WORDS, UNIFORM_DETAIL_PACKED, UNIFORM_DETAIL_RING_4H_LOAD, registerUniformDetailBase } from "../../core/uniform-detail-abi";

export type UniformDetailClass="cell"|"face"|"vertex"|"atlas";
type Dims=readonly [number,number,number];
const CLASSES=["cell","face","vertex","atlas"] as const;
export const UNIFORM_DETAIL_PATCH_EDGE=32;
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
 /** Texture x and y, and z with no slot (identity: the logical extent). */
 readonly extent:Dims;
}
export interface UniformDetailLayout{
 readonly dims:Dims;readonly tiles:Dims;readonly patchEdge:number;readonly patchGrid:Dims;readonly patches:number;
 /** identity: every patch resident at its logical place (physical =
  * logical); packed: slots for resident patches only. */
 readonly placement:"identity"|"packed";
 /** Slots of one atlas layer (x, y), fixed by the lattice: capacity grows by
  * whole layers along z, so no slot moves and no address constant changes. */
 readonly layer:readonly [number,number];
 /** Layers that hold every patch. */
 readonly maxLayers:number;
 readonly classes:Readonly<Record<UniformDetailClass,UniformDetailClassLayout>>;
 /** QA spike: patch-local addressing (UniformDetailOptions.local). */
 readonly local?:UniformDetailLocal;
 /** The domain placement, the default (UniformDetailOptions.domain). */
 readonly domain?:UniformDetailDomain;
 /** QA: haloed patches (UniformDetailOptions.patch). */
 readonly patch?:UniformDetailPatch;
}
/** QA, slice step 2 of the storage design ("Slice step 2: contract"): real
 * patches with halos. A patch is resident when it holds an h tile (no
 * closure). Each field is its base block, unfolded at the texture origin,
 * then slots of pitch P+2r (vertex P+1+2r): the patch interior at offset r
 * and a halo of r texels. A work item reads its home entry once (udHome,
 * from umOrigin) into a box and an origin per class; a load is
 * select(base(p), origin+p, p in box). Every store writes every replica:
 * the texel in each resident box that holds it and, when canonical, the
 * base, so a halo and the base are current after every store and no refresh
 * launch exists. The storage holds its first layout only. */
export interface UniformDetailPatch{
 /** Halo texels per side, per class. Vertex and atlas hold the seam ring
  * (one tile): the hanging vertices and unit taps of a 4h tile beside an h tile. */
 readonly halo:UniformDetailHalo;
 /** guard: every load in the directory form (box test, canonical test, one
  * directory read). profile: the same, each site recording what its taps
  * did (UNIFORM_DETAIL_SITE), a lost store fatal only when checked. trust: the profile's clean sites
  * (setUniformDetailPatchProfile) in the arithmetic form, the rest guarded. */
 readonly mode:"guard"|"profile"|"trust";
 /** A load with no texel (not in the home box, not canonical, in no
  * resident patch), or an arithmetic load that needed the directory, raises
  * UNIFORM_DETAIL_LOAD_VIOLATION. */
 readonly checked:boolean;
}
export interface UniformDetailHalo{readonly cell:number;readonly face:number;readonly vertex:number;readonly atlas:number}
/** Bits of a load or store site's profile word (patch mode "profile"). */
export const UNIFORM_DETAIL_SITE={
 /** In the home patch's interior, in its halo (and how deep: 2, 3, 4 or more texels). */
 interior:1,halo:2,halo2:0x40,halo3:0x80,halo4:0x8000000,
 /** Canonical, out of the home box, read from the base: by a patch-homed, a base-homed work item. */
 basePatchHome:4,baseBaseHome:8,
 /** Base-homed work item, non-canonical tap: in a resident patch, nowhere. */
 seam:0x10,seamNowhere:0x20,
 /** Patch-homed, out of the box, non-canonical, in a resident patch: bit excess-1, excess 1..8 texels past the box. */
 beyond:0x100,
 /** The same with no texel anywhere. */
 nowhere:0x10000,
 storeSlow:0x1000000,storeNowhere:0x2000000,storeFast:0x4000000,
} as const;
/** Support words past the violation words that hold the site profile. */
const PATCH_SITE_WORDS=8192;
/** The halo each class needs (storage design, "Slice step 2: results"). */
export const UNIFORM_DETAIL_PATCH_HALO:UniformDetailHalo={cell:1,face:1,vertex:4,atlas:4};
const PATCH_TABLE=0x55445032;
export interface UniformDetailSite{readonly key:string;readonly module:string;readonly field:string;readonly kind:UniformDetailClass;readonly store:boolean;
 /** The accessor compiled there: P profiled, G arithmetic, F guarded, "" a store. */
 readonly form:string}
export interface UniformDetailPatchProfile{readonly halo:UniformDetailHalo;readonly words:ReadonlyMap<string,number>}
let patchSites:UniformDetailSite[]=[];let patchSiteIndex=new Map<string,number>();
let patchProfile:UniformDetailPatchProfile|undefined;
/** QA: the sites the patch rewrite has numbered since the last setUniformDetailStorageForQA. */
export function uniformDetailPatchSites():readonly UniformDetailSite[]{return patchSites;}
/** QA: the profile a later `patch:trust` storage compiles from. */
export function setUniformDetailPatchProfile(profile?:UniformDetailPatchProfile):void{patchProfile=profile;}
/** Is a profiled load site arithmetic at `halo`? It ran, no base-homed work
 * item read an h texel through it, no tap found no texel (a profile recorded
 * before a texel became reachable would otherwise leave its site arithmetic),
 * and no tap left the box by more than the halo now covers. */
export function uniformDetailSiteClean(word:number|undefined,kind:UniformDetailClass,profiled:UniformDetailHalo,halo:UniformDetailHalo):boolean{
 if(!word||word&(UNIFORM_DETAIL_SITE.seam|UNIFORM_DETAIL_SITE.seamNowhere|UNIFORM_DETAIL_SITE.nowhere))return false;
 const beyond=(word>>>8)&255;
 return beyond===0||profiled[kind]+(32-Math.clz32(beyond))<=halo[kind];
}
/** Support words an ownership adds for the site profile (QA; production: 0). */
export function uniformDetailSupportWords():number{return qaOptions?.patch?.mode==="profile"||qaOptions?.domain?.survey||qaOptions?.domain?.poison?PATCH_SITE_WORDS:0;}
function patchLayout(dims:Dims,patchEdge:number,patch:UniformDetailPatch,maxDepth:number):UniformDetailLayout{
 if(patchEdge<16||!Number.isInteger(Math.log2(patchEdge)))throw new Error(`Uniform detail patch edge ${patchEdge} must be a power of two, 16 or more`);
 if(dims.some(n=>!Number.isInteger(n)||n<4||n%4))throw new Error(`Uniform detail storage needs a lattice of whole tiles, not ${dims}`);
 const h=patch.halo;if([h.cell,h.face,h.vertex,h.atlas].some(r=>!Number.isInteger(r)||r<0||r>8||4*r>patchEdge))throw new Error(`Uniform detail patch halo ${JSON.stringify(h)} must be 0..8 texels and at most a quarter of the patch edge`);
 const tiles=dims.map(n=>n/4) as unknown as Dims,patchGrid=dims.map(n=>Math.ceil(n/patchEdge)) as unknown as Dims,patches=patchGrid[0]*patchGrid[1]*patchGrid[2];
 for(const layer of [[Math.ceil(patchGrid[0]/2),Math.ceil(patchGrid[1]/2)],[patchGrid[0],patchGrid[1]]] as const){
  const maxLayers=Math.ceil(patches/(layer[0]*layer[1]));
  const classes=Object.fromEntries(CLASSES.map(kind=>{
   const r=h[kind],spacing=patchEdge+2*r+(kind==="vertex"?1:0);
   const base:Dims=kind==="cell"?tiles:kind==="vertex"?tiles.map(t=>t+1) as unknown as Dims:tiles.map(t=>2*t) as unknown as Dims;
   return [kind,{spacing,base,atlasZ:base[2],fold:1,per:1,extent:[Math.max(layer[0]*spacing,base[0]),Math.max(layer[1]*spacing,base[1]),base[2]]}];
  })) as unknown as Record<UniformDetailClass,UniformDetailClassLayout>;
  const depth=Math.max(...CLASSES.map(k=>classes[k].atlasZ+maxLayers*classes[k].spacing));
  if(depth<=maxDepth&&layer[0]<=1023&&layer[1]<=1023&&maxLayers<=1024)return {dims,tiles,patchEdge,patchGrid,patches,placement:"packed",layer,maxLayers,classes,patch};
 }
 throw new Error(`Uniform detail patches of ${dims} at patch edge ${patchEdge} exceed the 3D texture depth ${maxDepth}`);
}
const haloOf=(l:UniformDetailLayout,kind:UniformDetailClass)=>l.patch?l.patch.halo[kind]:0;
/** QA: the patch layout at its degenerate size, one patch = the lattice,
 * with residency on the h-tile capacity C of the simulation ownership
 * (UniformMixedCapacity.fineTiles). A field is ONE texture at a time: at
 * C = 0 its unfolded base block, at C > 0 its logical h texture. The
 * accessors tell the two apart from the bound texture's extent, so one
 * shader set runs at every occupancy and the texture swap is the mode
 * switch. Base currency, per class:
 *  cell   t^3, texel c = the tile's origin cell 4c. The authority at C = 0;
 *         while C > 0 it does not exist (freed after the admission fill,
 *         allocated and written by the retirement restriction).
 *  face   (2t)^3, texel 2c + (l==3) per axis: the tile's origin and its
 *         three +face anchors o+3e_a, the other four texels zero (compact:
 *         (t, t, 4t), texel (c.x, c.y, 4c.z + j.x + 2j.y + 3j.z), the four
 *         canonical texels only). Same lifetime as the cell base.
 *  vertex (t+1)^3, texel g = the tile corner 4g. Same lifetime; while C > 0
 *         the only 4h vertex data is UniformCoarseVertexPhi's published
 *         texture, current at each published revision, never read by the
 *         solver.
 *  atlas-only names (unitVelocity): no base value; at C = 0 a load is zero
 *         and a store a violation. */
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
 /** QA (checked): the h texels of every tile outside the ring hold
  * UNIFORM_DETAIL_POISON (filled at admission and after each relayout), and
  * a load that returns it raises UNIFORM_DETAIL_POISON_VIOLATION. */
 readonly poison?:boolean;
 /** QA (checked): ring and poison findings are recorded per load site
  * (siteWords) and raise no violation, so one run lists every site. */
 readonly survey?:boolean;
 /** QA, measurement only: a store on an h texture never writes its base twin. */
 readonly untwinned?:boolean;
 /** QA, the comparison arm of the window: the h store is the lattice whatever
  * tile box the host bounds its h tiles by (UniformDetailStorage.window). */
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
  const base=(kind==="cell"?tiles:kind==="vertex"?tiles.map(t=>t+1):domain.compactFaces?[tiles[0],tiles[1],4*tiles[2]]:tiles.map(t=>2*t)) as unknown as Dims;
  return [kind,{spacing:patchEdge,base,atlasZ:0,fold:1,per:1,extent:base}];
 })) as unknown as Record<UniformDetailClass,UniformDetailClassLayout>;
 return {dims,tiles,patchEdge,patchGrid:[1,1,1],patches:1,placement:"packed",layer:[1,1],maxLayers:1,classes,domain};
}
/** Where a patch-local invocation takes its home entry from: the directory
 * (one storage read), a value already in hand (emulated by a uniform word:
 * no added load), a workgroup variable invocation 0 fills, or a constant. */
export type UniformDetailLocal="directory"|"record"|"workgroup"|"constant";
export function uniformDetailLayout(dims:Dims,patchEdge=UNIFORM_DETAIL_PATCH_EDGE,placement:"identity"|"packed"="identity",maxDepth=2048):UniformDetailLayout{
 if(patchEdge<16||!Number.isInteger(Math.log2(patchEdge)))throw new Error(`Uniform detail patch edge ${patchEdge} must be a power of two, 16 or more`);
 if(dims.some(n=>!Number.isInteger(n)||n<4||n%4))throw new Error(`Uniform detail storage needs a lattice of whole tiles, not ${dims}`);
 const tiles=dims.map(n=>n/4) as unknown as Dims;
 const patchGrid=dims.map(n=>Math.ceil(n/patchEdge)) as unknown as Dims;
 const patches=patchGrid[0]*patchGrid[1]*patchGrid[2],identity=placement==="identity";
 const build=(layer:readonly [number,number])=>{
  const maxLayers=identity?patchGrid[2]:Math.ceil(patches/(layer[0]*layer[1]));
  const classes=Object.fromEntries(CLASSES.map(kind=>{
   const spacing=kind==="vertex"&&!identity?patchEdge+1:patchEdge;
   // An atlas-only field lives in a face block (its base unused): the remap borrows one as a face scratch.
   const base:Dims=kind==="cell"?tiles:kind==="vertex"?tiles.map(t=>t+1) as unknown as Dims:tiles.map(t=>2*t) as unknown as Dims;
   // Identity keeps every patch resident: the base is never addressed, not allocated.
   if(identity)return [kind,{spacing,base,atlasZ:0,fold:1,per:1,extent:kind==="vertex"?dims.map(n=>n+1):dims}];
   const x=Math.max(layer[0]*spacing,base[0],1),y=Math.max(layer[1]*spacing,base[1],1);
   const fold=base[0]?Math.floor(x/base[0]):1,per=fold*(base[1]?Math.floor(y/base[1]):1);
   const atlasZ=base[2]?Math.ceil(base[2]/per):0;
   return [kind,{spacing,base,atlasZ,fold,per,extent:[x,y,atlasZ]}];
  })) as unknown as Record<UniformDetailClass,UniformDetailClassLayout>;
  const depth=identity?0:Math.max(...CLASSES.map(k=>classes[k].atlasZ+maxLayers*classes[k].spacing));
  return {dims,tiles,patchEdge,patchGrid,patches,placement,layer,maxLayers,classes,depth};
 };
 if(identity){const {depth:_,...layout}=build([patchGrid[0],patchGrid[1]]);return layout;}
 // A quarter of a patch plane per layer where every patch still fits the
 // texture depth, else the whole plane.
 for(const layer of [[Math.ceil(patchGrid[0]/2),Math.ceil(patchGrid[1]/2)],[patchGrid[0],patchGrid[1]]] as const){
  const {depth,...layout}=build(layer);
  if(depth<=maxDepth&&layer[0]<=1023&&layer[1]<=1023&&layout.maxLayers<=1024)return layout;
 }
 throw new Error(`Uniform detail storage of ${dims} at patch edge ${patchEdge} exceeds the 3D texture depth ${maxDepth}`);
}
/** Directory word of a resident patch: its slot coordinate, 10 bits per axis. */
const entry=(slot:Dims)=>(0x80000000|slot[0]|slot[1]<<10|slot[2]<<20)>>>0;

/** handle: the texture createField returned, the field's identity for its
 * holders. Packed, it holds no texels (a destroyed 1³ texture: binding it is
 * a validation error); they live in a physical texture the storage replaces
 * when the atlas grows. Both the handle and its physical textures map here. */
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
 for(const e of entries){const f=fields.get(e.resource as GPUTexture);if(f&&f.storage.layout.placement!=="identity")storage=f.storage;}
 const domain=!!storage?.layout.domain;
 const build=(base=false)=>device.createBindGroup({label:descriptor.label,layout:descriptor.layout,entries:entries.flatMap(e=>{
  const f=fields.get(e.resource as GPUTexture);
  const primary={binding:e.binding,resource:f?(base?f.storage.baseOf(f.handle)!:f.storage.physical(f.handle)).createView():"createView" in e.resource?(e.resource as GPUTexture).createView():e.resource as GPUBindingResource};
  if(!twins?.formats.has(e.binding))return [primary];
  const sampled=!twins.formats.get(e.binding);
  return [primary,{binding:e.binding+UNIFORM_DETAIL_TWIN_BINDING,resource:f?.storage.layout.domain?f.storage.twin(f.handle,sampled,base).createView():sampled?primary.resource:twinDummy(device,twins,e.binding).createView()}];
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

export interface UniformDetailOptions{
 readonly patchEdge?:number;
 /** QA: identity placement (physical = logical, raw loads, every patch
  * resident), the comparison arm of every accessor placement. */
 readonly placement?:"identity";
 /** QA: a fixed slot capacity, never grown (the overflow fatal). */
 readonly capacity?:number;
 /** QA: every patch resident from the start. */
 readonly resident?:"all";
 /** QA spike, measurement only: patch-local addressing. One resident patch
  * covers the lattice; each invocation resolves its home entry once and
  * every access is origin + p (no directory read per access, no residency
  * branch, no hanging-vertex path). */
 readonly local?:UniformDetailLocal;
 /** The domain placement (UniformDetailDomain): production’s default. */
 readonly domain?:UniformDetailDomain;
 /** QA: haloed patches (UniformDetailPatch); takes patchEdge and capacity. */
 readonly patch?:UniformDetailPatch;
}
/** The storage a solver gets without QA options: the domain placement with
 * compact faces (base blocks at zero detail, the logical h lattice once any h
 * tile is reserved). `identity` remains the comparison arm. */
const DEFAULT_OPTIONS:UniformDetailOptions={domain:{checked:false,compactFaces:true,unpinned:false}};
let qaOptions:UniformDetailOptions|undefined;
/** QA only (probes and lanes, before they build a solver): the storage every
 * later solver gets. Production never calls it. */
export function setUniformDetailStorageForQA(options?:UniformDetailOptions):void{qaOptions=options;patchSites=[];patchSiteIndex=new Map();}
/** A QA spec: `identity[:edge]` (the comparison arm), `packed[:edge]` (the
 * patch atlas, grown on demand; QA since the domain placement became the
 * default), `compact:<slots>[:edge]` (fixed capacity), `compact:all[:edge]`
 * (every patch resident), `local[:directory|record|workgroup|constant]` (the
 * patch-local spike) or `domain[:flag...]` (UniformDetailDomain, the default
 * as plain `domain`: compact faces unless `parity`; `checked`; `unpinned`;
 * `poison`, `survey`, `untwinned`; `unwrapped` for the lattice-sized h
 * store, the window's comparison arm). */
const DOMAIN_FLAGS=["checked","compact","parity","unpinned","poison","survey","untwinned","unwrapped"];
export function uniformDetailOptionsFromSpec(spec:string):UniformDetailOptions{
 const [mode,a,b]=spec.split(":");
 const edge=(v:string|undefined)=>v===undefined?{}:{patchEdge:Number(v)};
 if(mode==="identity"&&b===undefined)return {placement:"identity",...edge(a)};
 if(mode==="packed"&&b===undefined)return edge(a);
 if(mode==="compact"&&a==="all")return {resident:"all",...edge(b)};
 if(mode==="compact"&&a!==undefined)return {capacity:Number(a),...edge(b)};
 if(mode==="domain"){
  const flags=spec.split(":").slice(1),has=(f:string)=>flags.includes(f);
  if(flags.every(f=>DOMAIN_FLAGS.includes(f))&&new Set(flags).size===flags.length&&!(has("compact")&&has("parity")))
   return {domain:{checked:has("checked")||has("poison")||has("survey"),compactFaces:!has("parity"),unpinned:has("unpinned"),
    poison:has("poison"),survey:has("survey"),untwinned:has("untwinned"),unwrapped:has("unwrapped")}};
 }
 if(mode==="patch"){
  let patchEdge:number|undefined,capacity:number|undefined,halo:UniformDetailHalo=UNIFORM_DETAIL_PATCH_HALO,run:UniformDetailPatch["mode"]="guard",checked=false,all=false;
  for(const f of spec.split(":").slice(1)){
   if(/^\d+$/.test(f))patchEdge=Number(f);
   else if(/^r\d(\.\d\.\d\.\d)?$/.test(f)){const v=f.slice(1).split(".").map(Number);halo={cell:v[0]!,face:v[1]??v[0]!,vertex:v[2]??v[0]!,atlas:v[3]??v[0]!};}
   else if(f==="all")all=true;
   else if(/^cap\d+$/.test(f))capacity=Number(f.slice(3));
   else if(f==="guard"||f==="profile"||f==="trust")run=f;
   else if(f==="checked")checked=true;
   else throw new Error(`Uniform detail storage spec ${spec}: expected patch[:edge][:r<n>|:r<cell>.<face>.<vertex>.<atlas>][:cap<slots>|:all][:guard|:profile|:trust][:checked]`);
  }
  return {...(patchEdge===undefined?{}:{patchEdge}),...(capacity===undefined?{}:{capacity}),...(all?{resident:"all" as const}:{}),patch:{halo,mode:run,checked}};
 }
 if(mode==="local"&&b===undefined&&(a===undefined||a==="directory"||a==="record"||a==="workgroup"||a==="constant"))return {local:a??"directory"};
 throw new Error(`Uniform detail storage spec ${spec}: expected identity[:edge], packed[:edge], compact:<slots>[:edge], compact:all[:edge], local[:where], patch[:...] or domain[:${DOMAIN_FLAGS.join("][:")}]`);
}
export function uniformDetailStorageOptions(qa?:UniformDetailOptions):UniformDetailOptions{return qa??qaOptions??DEFAULT_OPTIONS;}
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
/** Syncs a resident h store waits, four times the size its box needs, before it is made smaller. */
const WINDOW_CALM=16;
/** Workgroups of each transfer launch: their lanes stride the GPU lists. */
const TRANSFER_GRID=256;
/** Fields one transfer launch binds (the default storage-texture limit). */
const BATCH=4;
type TransferKind="cell"|"vertex"|"face";
interface Kernels{
 readonly ownership:UniformMixedOwnership;readonly residency:GPUBindGroupLayout;readonly slots:GPUBindGroup;
 readonly need:GPUComputePipeline;readonly allocate:GPUComputePipeline;readonly release:GPUComputePipeline;
 readonly cell:{fill:GPUComputePipeline;retire:GPUComputePipeline};readonly vertex:{fill:GPUComputePipeline;retire:GPUComputePipeline};
 readonly face:{fillGather:GPUComputePipeline;fillScatter:GPUComputePipeline;retireGather:GPUComputePipeline;retireScatter:GPUComputePipeline};
 readonly layouts:{state:GPUBindGroupLayout;staged:GPUBindGroupLayout;inPlace:GPUBindGroupLayout;sampled:GPUBindGroupLayout;written:GPUBindGroupLayout};
 /** Bound to the fields' physical textures: rebuilt when they are replaced (bindKernels). */
 batches:{kind:TransferKind;state:GPUBindGroup;storage:GPUBindGroup;sampled?:GPUBindGroup}[];
 owned:(GPUTexture|GPUBuffer)[];
}
/** The ring kernels of one launch: up to BATCH fields of one format. */
interface RingBatch{readonly handles:readonly GPUTexture[];readonly layout:GPUBindGroupLayout;readonly enter:GPUComputePipeline;readonly poison?:GPUComputePipeline;readonly poisonAll?:GPUComputePipeline}
interface DomainKernels{
 /** The ring: enter seeds the tiles a relayout brings into the ring
  * (and, in its first batch, hands the ring bit to the other ownership);
  * share is that hand-over alone; poison (QA) fills what is outside it. */
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
 /** Field handles in creation order, and each one's physical texture
  * (identity: itself). */
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
 /** QA: the capacity never grows. */
 private readonly fixed:boolean;
 readonly signature:string;
 private readonly residentAll:boolean;
 private readonly state?:GPUBuffer;
 private readonly words:{readonly dir:number;readonly slots:number;readonly admit:number;readonly retire:number;readonly need:number;readonly total:number};
 /** The last frame receipt's residency (noteReceipt). */
 private receipt?:{capacity:number;resident:number;peak:number};
 /** Patches resident after the last host relayout; undefined once a GPU
  * relayout admits on its own. */
 private expected?:Set<number>;
 constructor(private readonly device:GPUDevice,dims:Dims,options:UniformDetailOptions={}){
  const identity=options.placement==="identity";
  if((identity||options.resident==="all")&&options.capacity!==undefined)throw new Error("Detail storage: identity and resident all take no capacity");
  const local=options.local,domain=options.domain,patch=options.patch;
  if(patch&&(identity||local||domain))throw new Error("Detail storage: haloed patches take a patch edge and a capacity or resident all only");
  if(local&&(identity||options.capacity!==undefined||options.patchEdge!==undefined))throw new Error("Detail storage: the patch-local spike takes no placement, capacity or patch edge");
  if(domain&&(identity||local||options.capacity!==undefined||options.patchEdge!==undefined||options.resident!==undefined))throw new Error("Detail storage: the domain placement takes no other option");
  // Patch-local: one patch covers the lattice, so every access is in the home patch.
  const packed=patch?patchLayout(dims,options.patchEdge??UNIFORM_DETAIL_PATCH_EDGE,patch,device.limits.maxTextureDimension3D):domain?domainLayout(dims,domain):uniformDetailLayout(dims,local?2**Math.ceil(Math.log2(Math.max(32,...dims))):options.patchEdge,identity?"identity":"packed",device.limits.maxTextureDimension3D);
  this.layout=local?{...packed,local}:packed;
  const l=this.layout,per=l.layer[0]*l.layer[1];
  this.cells=l.dims;this.box=[0,0,0,l.tiles[0]-1,l.tiles[1]-1,l.tiles[2]-1];
  if(local&&l.patches!==1)throw new Error(`Detail storage: the patch-local spike needs one patch, not ${l.patches}`);
  this.residentAll=options.resident==="all"||!!local;
  this.fixed=this.residentAll||options.capacity!==undefined||!!domain;
  this.slots=identity||this.residentAll||domain?l.patches:options.capacity??0;
  if(!Number.isInteger(this.slots)||this.slots<0||this.slots>l.patches)throw new Error(`Uniform detail capacity ${options.capacity} must hold 0..${l.patches} patches`);
  this.layers=identity?l.maxLayers:Math.ceil(this.slots/per);
  this.signature=`${l.placement}:${l.patchEdge}:${l.layer}:${local??""}${domain?`:domain:${domain.checked}:${domain.compactFaces}:${!!domain.unpinned}:${!!domain.poison}:${!!domain.survey}:${!!domain.untwinned}:${!!domain.unwrapped}`:""}${patch?`:patch:${JSON.stringify(patch)}`:""}`;
  let set=live.get(device);if(!set)live.set(device,set=new Set());set.add(this);liveLayout(device,dims);
  this.directory=new Uint32Array(l.patches);
  const maxSlots=l.maxLayers*per,dir=D.words,slots=dir+l.patches,admit=slots+maxSlots,retire=admit+l.patches,need=retire+l.patches;
  this.words={dir,slots,admit,retire,need,total:need+l.patches};
  if(identity)return;
  if(l.patches>device.limits.maxComputeWorkgroupsPerDimension)throw new Error(`Uniform detail storage scans ${l.patches} patches, past one dispatch dimension`);
  if(this.residentAll)for(let i=0;i<l.patches;i++)this.directory[i]=entry(this.slotCoord(i));
  this.state=device.createBuffer({label:"Uniform detail residency",size:4*(this.words.total+WINDOW_WORDS),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.writeState();this.setBox(this.box);
 }
 patchCoord(patch:number):Dims{const g=this.layout.patchGrid;return [patch%g[0],Math.floor(patch/g[0])%g[1],Math.floor(patch/(g[0]*g[1]))];}
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
  if(l.domain)return resident?this.storeExtent(kind,this.cells):c.base;
  return l.placement==="identity"?c.extent:[c.extent[0],c.extent[1],Math.max(1,c.atlasZ+this.layers*c.spacing)];
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
 private calm=0;
 private get windowed():boolean{const c=this.cells,d=this.layout.dims;return c[0]!<d[0]!||c[1]!<d[1]!||c[2]!<d[2]!;}

 // The store's forms (domain placement). A stage pipeline whose module
 // loads fields is compiled per form, by the constant udWindow: general
 // (the wrapped address, exact for any store) and direct (the lattice-sized
 // store's raw address). One is built at first; the other when the store
 // changes class, in the background, as a solid-free twin is: a store that
 // comes to span the lattice runs the general form until the direct one is
 // built, and a store that is to wrap spans the lattice until the general
 // one is. Both are adopted between frames (settle).
 private readonly twins:StoreTwin[]=[];
 /** The form a new pipeline compiles in while the store does not wrap. */
 private direct=false;
 private preparing?:{readonly direct:boolean;readonly done:Promise<void>};
 private prepared?:{readonly direct:boolean;readonly twins:readonly StoreTwin[];readonly pipelines:readonly GPUComputePipeline[]};
 private prepareError?:{readonly error:unknown};
 /** The last window() asked for a store that spans the lattice, and the frames it has since. */
 private spans=true;
 private spanned=0;
 /** Setup, before the stages compile: direct while the h tiles are not the
  * host's to bound (Dynamic, Full, bodies), so the store will span the lattice. */
 prefer(direct:boolean):void{if(this.layout.domain)this.direct=direct;}
 private compile(twin:StoreTwin,direct:boolean):Promise<GPUComputePipeline>{
  const d=twin.descriptor;
  return this.device.createComputePipelineAsync(direct?{...d,compute:{...d.compute,constants:{...d.compute.constants,udWindow:0}}}:d);
 }
 /** uniformDetailPipeline: the key of a pipeline built in the form the store holds. */
 pipeline(descriptor:GPUComputePipelineDescriptor):Promise<GPUComputePipeline>{
  const twin=new StoreTwin(this,descriptor),direct=this.direct&&!this.windowed;
  this.twins.push(twin);
  return this.compile(twin,direct).then(pipeline=>{if(direct)twin.direct=pipeline;else twin.general=pipeline;return twin.key;});
 }
 /** uniformDetailPick. Fatal when a wrapped store would run a pipeline with no general form. */
 pick(twin:StoreTwin):GPUComputePipeline{
  const pipeline=(this.windowed?undefined:twin.direct)??twin.general;
  if(!pipeline)throw new Error(`Uniform pipelines: ${twin.descriptor.label??twin.descriptor.compute.entryPoint} was dispatched on a wrapped h store before its general form was prepared`);
  return pipeline;
 }
 /** Builds one form for every pipeline that lacks it, off the frame; settle() adopts them together. */
 private prepareForm(direct:boolean):void{
  if(this.preparing||this.prepared)return;
  const twins=this.twins.filter(twin=>!(direct?twin.direct:twin.general));if(!twins.length)return;
  const done=Promise.all(twins.map(twin=>this.compile(twin,direct))).then(pipelines=>{this.prepared={direct,twins,pipelines};},error=>{this.prepareError={error};}).finally(()=>{this.preparing=undefined;});
  this.preparing={direct,done};
 }
 /** Once a frame, after the layout's sync and before the frame encodes:
  * adopts a finished build, and asks for the direct form once the store has
  * spanned the lattice by its request (not as a window still to shrink) for
  * WINDOW_CALM frames. Throws a build's failure. */
 settle():void{
  if(!this.layout.domain)return;
  const failed=this.prepareError;if(failed){this.prepareError=undefined;throw failed.error;}
  const built=this.prepared;
  if(built){this.prepared=undefined;built.twins.forEach((twin,i)=>{if(built.direct)twin.direct=built.pipelines[i];else twin.general=built.pipelines[i];});}
  if(!this.resident||this.windowed||!this.spans||this.twins.every(twin=>twin.direct)){this.spanned=0;return;}
  if(++this.spanned>=WINDOW_CALM){this.spanned=0;this.direct=true;this.prepareForm(true);}
 }
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
 /** No kind's h store may have the extent of its base block: the accessors
  * and the consumers tell a field's two textures apart by it. A wrapped axis
  * is doubled until none does (a lattice-sized store never does). */
 private distinct(cells:number[]):Dims{
  const l=this.layout,n=l.dims;
  const clash=()=>(["cell","face","vertex"] as const).some(kind=>{const e=this.storeExtent(kind,cells),b=l.classes[kind].base;return e.every((v,a)=>v===b[a]);});
  while(clash()){const a=cells.findIndex((c,i)=>c<n[i]!);cells[a]=Math.min(2*cells[a]!,n[a]!);}
  return cells as unknown as Dims;
 }
 /** The store that holds a tile box (inclusive): per axis the power of two
  * that holds its texels (boxEnd), the lattice where that is no smaller. */
 private windowCells(box:readonly number[]):Dims{
  const l=this.layout,d=l.domain!,n=l.dims;
  // The poison fill addresses every tile's texels.
  if(d.poison||d.unwrapped)return n;
  return this.distinct(n.map((full,a)=>{let c=8;while(c<this.boxEnd("vertex",box,a)+1-4*box[a]!)c*=2;return Math.min(c,full);}));
 }
 private setBox(box:readonly number[]):void{
  const T=this.layout.tiles,n=this.layout.dims;
  this.box=[0,1,2].map(a=>this.cells[a]!<n[a]!?box[a]!:0).concat([0,1,2].map(a=>this.cells[a]!<n[a]!?box[a+3]!:T[a]!-1));
  if(this.state)this.device.queue.writeBuffer(this.state,4*this.words.total,Uint32Array.from(this.box));
 }
 /** Domain, between frames and before the reserve that may admit: the tile
  * box (low inclusive, high exclusive) that bounds the h tiles of the
  * generation held and of every build not yet adopted; none: any tile.
  * The h store holds that box and the detail ring around it. Not resident:
  * the next admission allocates it. Resident: a box the store cannot hold
  * grows it at once (the texels of the tiles both boxes hold are copied, a
  * tile's worth of work per tile); one a quarter its size shrinks it after
  * WINDOW_CALM such syncs; any other only moves the box. A relayout that
  * puts a ring tile outside the box is fatal at its frame receipt. */
 window(tiles?:readonly number[]):void{
  const l=this.layout;if(!l.domain)return;
  const T=l.tiles,box=tiles&&[0,1,2].every(a=>tiles[a]!<tiles[a+3]!)
   ?[0,1,2].map(a=>Math.max(0,tiles[a]!-WINDOW_RING)).concat([0,1,2].map(a=>Math.min(T[a]!-1,tiles[a+3]!-1+WINDOW_RING)))
   :[0,0,0,T[0]-1,T[1]-1,T[2]-1];
  let want=this.windowCells(box);const held=this.cells;
  this.spans=want.every((c,a)=>c===l.dims[a]);
  // A store wraps once every pipeline has its general form; until then it spans the lattice.
  if(!this.spans){this.direct=false;if(this.twins.some(twin=>!twin.general)){this.prepareForm(false);want=l.dims;}}
  if(!this.resident){this.cells=want;this.calm=0;this.setBox(box);return;}
  const grown=this.distinct(want.map((c,a)=>Math.max(c,held[a]!)));
  if(grown.some((c,a)=>c!==held[a])){this.calm=0;this.rewrap(grown,box);return;}
  const volume=(c:Dims)=>c[0]*c[1]*c[2];
  if(4*volume(want)<=volume(held)){if(++this.calm>=WINDOW_CALM){this.calm=0;this.rewrap(want,box);return;}}
  else this.calm=0;
  this.setBox(box);
 }
 /** Every following field's h store at other extents, with the texels of
  * the tiles both boxes hold at their places in it. Between frames. */
 private rewrap(cells:Dims,box:readonly number[]):void{
  const n=this.layout.dims,old=this.cells,held=this.box,gone:GPUTexture[]=[];
  const encoder=this.device.createCommandEncoder({label:"Uniform detail window"});
  this.cells=cells;
  // On an axis that stops wrapping every tile is kept; on one that starts, the new box's.
  const lo=[0,1,2].map(a=>Math.max(box[a]!,held[a]!)),hi=[0,1,2].map(a=>Math.min(box[a+3]!,held[a+3]!));
  for(const handle of this.following){
   const before=this.physicals.get(handle)!,field=fields.get(handle)!,after=this.allocate(handle.label,field.kind,before.format,true);
   if(lo.every((v,a)=>v<=hi[a]!)){
    // Per axis: the runs of kept texels that wrap in neither store.
    const runs=[0,1,2].map(a=>{
     const end=this.boxEnd(field.kind,[0,0,0,...hi],a),from=old[a]!<n[a]!?old[a]!:Infinity,to=cells[a]!<n[a]!?cells[a]!:Infinity,out:[number,number,number][]=[];
     for(let p=4*lo[a]!;p<=end;){
      const stop=Math.min(end+1,from===Infinity?Infinity:(Math.floor(p/from)+1)*from,to===Infinity?Infinity:(Math.floor(p/to)+1)*to);
      out.push([from===Infinity?p:p%from,to===Infinity?p:p%to,stop-p]);p=stop;
     }
     return out;
    });
    for(const x of runs[0]!)for(const y of runs[1]!)for(const z of runs[2]!)
     encoder.copyTextureToTexture({texture:before,origin:[x[0],y[0],z[0]]},{texture:after,origin:[x[1],y[1],z[1]]},[x[2],y[2],z[2]]);
   }
   this.physicals.set(handle,after);fields.set(after,field);registerUniformDetailBase(after,this.bases.get(handle)!);gone.push(before);
  }
  this.device.queue.submit([encoder.finish()]);this.release(gone);this.setBox(box);this.rebind();this.verify();
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
  if(kind==="atlas"&&!this.layout.patch)kind="face";
  if(this.kernels)throw new Error(`Uniform detail field ${label} was created after the transfer kernels bound the fields`);
  if(perTile&&kind!=="cell")throw new Error(`Uniform detail field ${label}: only a cell field is per tile`);
  const pin=perTile&&!!this.layout.domain&&!this.layout.domain.unpinned;
  if(this.layout.domain&&(kind==="atlas"||(kind==="face")!==(format==="rgba32float")))throw new Error(`Uniform detail field ${label}: the domain placement transfers scalar cell and vertex fields and vec4 face fields`);
  const physical=this.allocate(label,kind,format,pin?false:undefined);let handle=physical;
  if(this.layout.placement!=="identity"){handle=this.device.createTexture({label,size:[1,1,1],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING});handle.destroy();}
  const d=this.layout.dims,field:Field={storage:this,kind,logical:kind==="vertex"?[d[0]+1,d[1]+1,d[2]+1]:d,handle};
  fields.set(handle,field);fields.set(physical,field);
  this.handles.push(handle);this.physicals.set(handle,physical);if(pin)this.pinned.add(handle);
  if(this.layout.domain)this.bases.set(handle,physical);
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
 /** A live 1³ stand-in of a field, for a bind group whose kernels never
  * touch it (the dense reference groups, which packed storage never runs
  * on a field). Identity: the field. */
 stub(texture:GPUTexture):GPUTexture{
  if(this.layout.placement==="identity")return this.physical(texture);
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
 private grow(slots:number):void{
  const l=this.layout,per=l.layer[0]*l.layer[1],layers=Math.min(l.maxLayers,Math.ceil(slots/per));
  if(layers<=this.layers)return;
  this.layers=layers;this.slots=Math.min(l.patches,layers*per);
  const encoder=this.device.createCommandEncoder({label:"Uniform detail storage growth"}),replaced:GPUTexture[]=[];
  for(const handle of this.handles){
   const before=this.physicals.get(handle)!,field=fields.get(handle)!,after=this.allocate(before.label,field.kind,before.format);
   encoder.copyTextureToTexture({texture:before},{texture:after},[before.width,before.height,before.depthOrArrayLayers]);
   this.physicals.set(handle,after);fields.set(after,field);replaced.push(before);
  }
  this.device.queue.submit([encoder.finish()]);for(const t of replaced)t.destroy();
  this.device.queue.writeBuffer(this.state!,4*D.capacity,Uint32Array.of(this.slots));
  this.bindKernels();this.rebind();
 }
 /** Every tracked group over the fields' current physical textures. */
 private rebind():void{this.bound=this.bound.filter(b=>{const g=b.ref.deref();if(g){g.group=b.build();g.base=this.layout.domain?b.build(true):g.group;}return !!g;});}
 /** Before a host relayout's admit: slots for its patches beside the
  * resident ones (both are live across the remap). */
 reserve(layout:UniformMixedLayout):void{
  if(this.layout.placement==="identity")return;
  if(this.layout.domain){this.follow();return;}
  const target=this.patchesFor(layout);
  if(this.layout.patch){
   if(target.size!==this.expected?.size||[...target].some(p=>!this.expected!.has(p)))throw new Error(`Uniform detail patches (slice step 2) hold their first layout: ${this.expected?.size} patches resident, the new layout needs ${target.size}`);
   return;
  }
  const need=this.expected?new Set([...this.expected,...target]).size:Math.min(this.layout.patches,this.slots+target.size);
  if(!this.fixed)this.grow(need);
  this.expected=target;
 }
 /** A GPU relayout admits on its own, and its demand reaches the host only
  * in a later receipt: every patch gets a slot. */
 reserveAll():void{
  if(this.layout.placement==="identity")return;
  if(this.layout.domain){this.follow();return;}
  if(this.layout.patch)throw new Error("Uniform detail patches (slice step 2) hold their first layout: no GPU relayout");
  if(!this.fixed)this.grow(this.layout.patches);
  this.expected=undefined;
 }
 get allocatedBytes():number{
  return this.textures().reduce((n,t)=>n+t.width*t.height*t.depthOrArrayLayers*(t.format==="rgba32float"?16:4),0)+(this.state?.size??0)
   +(this.kernels?.owned.reduce((n,r)=>n+("size" in r?r.size:0),0)??0);
 }
 /** Resident patches: the host directory's while it is current, else the
  * last frame receipt's. */
 get residentPatches():number{
  if(this.layout.placement==="identity")return this.layout.patches;
  return this.mirrored||!this.receipt?this.directory.reduce((n,e)=>n+(e?1:0),0):this.receipt.resident;
 }
 /** The table every attached ownership carries (mode 0: raw fields). */
 table():Uint32Array<ArrayBuffer>{
  const l=this.layout,words=new Uint32Array(HEADER+l.patches);
  if(l.placement==="identity")return words;
  // Domain: one patch over the lattice, its slot the h texture itself
  // (atlas texel = logical texel); directory word zero: the base blocks.
  // Haloed patches: another mode word, so a consumer through the ABI (which has no halo offset) reads no table.
  words.set([l.patch?PATCH_TABLE:UNIFORM_DETAIL_PACKED,Math.log2(l.patchEdge),...l.patchGrid,...l.tiles]);
  CLASSES.forEach((kind,k)=>{const c=l.classes[kind];words.set([c.spacing,c.atlasZ,c.fold,c.per,...c.base,kind==="face"&&l.domain?.compactFaces?UNIFORM_DETAIL_FACE_COMPACT:0],8+8*k);});
  // Domain: the tile box the h store holds (the lattice) and the two-texture flag (uniform-detail-abi.ts).
  if(l.domain)words.set([0,0,0,l.tiles[0]-1,l.tiles[1]-1,l.tiles[2]-1,UNIFORM_DETAIL_BASED],32);
  words.set(this.directory,HEADER);return words;
 }
 /** Carry this storage's table in an ownership's topology tail (group 0 of
  * every stage that touches a field through it). */
 attach(ownership:UniformMixedOwnership):void{this.ownerships.add(ownership);storageOf.set(ownership,this);ownership.writeDetail(this.table());}
 /** The table to every attached ownership (between frames, in queue order with the submits around it). */
 private publishTable():void{const table=this.table();for(const o of this.ownerships)o.writeDetail(table);}
 /** Patches of a layout's detail: h tiles and their 26-connected one-tile
  * closure, rounded up to patches (haloed patches: h tiles only). */
 patchesFor(layout:UniformMixedLayout):Set<number>{
  if(this.residentAll)return new Set(Array.from({length:this.layout.patches},(_,i)=>i));
  const l=this.layout,[tx,ty,tz]=l.tiles,per=l.patchEdge/4,g=l.patchGrid,set=new Set<number>();
  for(let t=0;t<layout.tiles.length;t++){
   if(!(layout.tiles[t]!&0x80000000))continue;
   const x=t%tx,y=Math.floor(t/tx)%ty,z=Math.floor(t/(tx*ty)),c=l.patch?0:1;
   for(let dz=-c;dz<=c;dz++)for(let dy=-c;dy<=c;dy++)for(let dx=-c;dx<=c;dx++){
    const qx=x+dx,qy=y+dy,qz=z+dz;if(qx<0||qy<0||qz<0||qx>=tx||qy>=ty||qz>=tz)continue;
    set.add(Math.floor(qx/per)+g[0]*(Math.floor(qy/per)+g[1]*Math.floor(qz/per)));
   }
  }
  return set;
 }
 /** The field handles. */
 get fieldTextures():readonly GPUTexture[]{return this.handles;}
 /** Unregister. Identity textures belong to their solver's destroy. */
 destroy():void{
  live.get(this.device)?.delete(this);this.state?.destroy();for(const r of this.kernels?.owned??[])r.destroy();
  if(this.layout.placement==="identity")return;
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
  if(kind==="atlas"){const box=l.patch?this.boxes(kind,p,directory)[0]:undefined;return box?[box[0],box[1],box[2],1]:[0,0,0,-1];}
  if(kind==="vertex"){
   if(p.every(v=>v%4===0))return [...this.baseTexel("vertex",p.map(v=>v>>2) as unknown as Dims),0];
   // Haloed patches: a vertex on a resident patch's face (or in its halo) lives in that box.
   const box=l.patch?this.boxes(kind,p,directory)[0]:undefined;
   return box?[box[0],box[1],box[2],1]:[...home,2];
  }
  if(kind==="cell")return [...this.baseTexel("cell",home),0];
  const j=p.map(v=>(v&3)===3?1:0);
  if(l.domain?.compactFaces)return j[0]!+j[1]!+j[2]!>1?[0,0,0,-1]:[p[0]>>2,p[1]>>2,4*(p[2]>>2)+j[0]!+2*j[1]!+3*j[2]!,0];
  return [...this.baseTexel("face",p.map((v,a)=>2*(v>>2)+j[a]!) as unknown as Dims),0];
 }
 /** Haloed patches: p's texel in every resident box (interior plus halo)
  * that holds it, the home patch's first. */
 private boxes(kind:UniformDetailClass,p:Dims,directory:Uint32Array=this.directory):[number,number,number][]{
  const l=this.layout,P=l.patchEdge,shift=Math.log2(P),g=l.patchGrid,r=haloOf(l,kind),c=l.classes[kind],E=c.spacing,out:[number,number,number][]=[];
  const n=p.map((v,a)=>Math.min(v>>2,l.tiles[a]!-1)>>(shift-2)),local=p.map((v,a)=>v-(n[a]!<<shift));
  const s=local.map(v=>((kind==="vertex"?v<=r:v<r)?-1:0)+(v>=P-r?1:0));
  const visit=(d:readonly number[])=>{
   const m=n.map((v,a)=>v+d[a]!);if(m.some((v,a)=>v<0||v>=g[a]!))return;
   const e=directory[m[0]!+g[0]*(m[1]!+g[1]*m[2]!)]!;if(!e)return;
   const slot=[e&1023,(e>>>10)&1023,(e>>>20)&1023];
   out.push([slot[0]!*E+r+p[0]-(m[0]!<<shift),slot[1]!*E+r+p[1]-(m[1]!<<shift),c.atlasZ+slot[2]!*E+r+p[2]-(m[2]!<<shift)]);
  };
  visit([0,0,0]);
  for(let m=1;m<8;m++){const d=[m&1,(m>>1)&1,m>>2];if(d.every((b,a)=>!b||s[a]))visit(d.map((b,a)=>b*s[a]!));}
  return out;
 }
 /** Every physical texel that holds logical p: its authority (locate), and
  * under haloed patches each replica and the base texel when canonical. */
 private places(kind:UniformDetailClass,p:Dims):[number,number,number][]{
  if(!this.layout.patch){const at=this.locate(kind,p);return at[3]===0||at[3]===1?[[at[0],at[1],at[2]]]:[];}
  const out=this.boxes(kind,p);
  if(kind!=="atlas"&&UniformDetailStorage.canonical(kind,p))out.push(this.baseTexel(kind,(kind==="face"?p.map(v=>2*(v>>2)+((v&3)===3?1:0)):p.map(v=>v>>2)) as unknown as Dims));
  return out;
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
  if(this.layout.placement==="identity"){
   this.device.queue.writeTexture({texture:this.physical(texture)},values as Float32Array<ArrayBuffer>,{bytesPerRow:lx*comp*4,rowsPerImage:ly},[lx,ly,lz]);return;
  }
  if(!this.installed){this.pending.set(texture,values);return;}
  this.write(texture,this.image(texture,values));
  if(this.twinned(texture))this.write(texture,this.image(texture,values,true),true);
  // QA: the upload wrote the h texels of every tile.
  if(this.layout.domain?.poison&&this.twinned(texture)&&this.transfers?.ring){const e=this.device.createCommandEncoder({label:"Uniform detail poison (upload)"});this.encodePoison(e);this.device.queue.submit([e.finish()]);}
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
  if(values&&this.layout.domain){
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
  if(values&&this.layout.patch){
   // The base: each texel's canonical source. Then every resident box, interior and halo, as the raw texture holds them.
   const l=this.layout,c=l.classes[kind],b=c.base,E=c.spacing,r=haloOf(l,kind),shift=Math.log2(l.patchEdge);
   const put=(dst:number,p:readonly number[])=>{const src=(p[0]!+lx*(p[1]!+ly*p[2]!))*comp;for(let k=0;k<comp;k++)image[dst*comp+k]=values[src+k]!;};
   if(kind!=="atlas")for(let z=0;z<b[2];z++)for(let y=0;y<b[1];y++)for(let x=0;x<b[0];x++){
    if(kind==="face"){if((x&1)+(y&1)+(z&1)>1)continue;put(x+W*(y+H*z),[4*(x>>1)+3*(x&1),4*(y>>1)+3*(y&1),4*(z>>1)+3*(z&1)]);}
    else put(x+W*(y+H*z),[4*x,4*y,4*z]);
   }
   this.directory.forEach((e,patch)=>{
    if(!e)return;
    const n=this.patchCoord(patch),slot=[e&1023,(e>>>10)&1023,(e>>>20)&1023];
    for(let k=0;k<E;k++)for(let j=0;j<E;j++)for(let i=0;i<E;i++){
     const p=[(n[0]<<shift)-r+i,(n[1]<<shift)-r+j,(n[2]<<shift)-r+k];
     if(p[0]!<0||p[1]!<0||p[2]!<0||p[0]!>=lx||p[1]!>=ly||p[2]!>=lz)continue;
     put(slot[0]!*E+i+W*(slot[1]!*E+j+H*(c.atlasZ+slot[2]!*E+k)),p);
    }
   });
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
 /** The first layout (the frame starts all-h, then installs it). Identity:
  * nothing to do. Packed: the atlas grows to the layout's patches, the host
  * assigns their slots, hands the
  * directory to the GPU and deposits the pending uploads. Returns whether
  * the GPU remap from all-h must run: with every patch resident it can;
  * otherwise the all-h fields are not on the GPU, so the deposit is that
  * remap, emulated (UniformMixedRemap's coarsening from all-h: a 4h tile's
  * volume is the z,y,x mean of its 64 cells, a +face anchor the mean of its
  * 16 h faces with release bits held by all 16, in both velocity fields;
  * phi keeps every aligned vertex) and the caller installs the ownership
  * without it. */
 install(layout:UniformMixedLayout,live:{readonly volume:GPUTexture;readonly velocity:GPUTexture;readonly velocityScratch:GPUTexture}):boolean{
  if(this.installed)throw new Error("Uniform detail storage installs its first layout once");
  this.installed=true;
  if(this.layout.placement==="identity")return true;
  let all:boolean;
  if(this.layout.domain){
   // Nothing is deposited yet: an h capacity takes the uploads as they are.
   all=this.held()>0;if(all){this.installing=true;try{this.admit();}finally{this.installing=false;}}
  }else{
   const next=this.patchesFor(layout);all=next.size===this.layout.patches;
   const added=[...next].filter(p=>!this.directory[p]).sort((a,b)=>a-b),used=new Set<number>();
   for(const e of this.directory)if(e)used.add(this.slotIndex(e));
   if(!this.fixed)this.grow(used.size+added.length);
   this.expected=next;
   if(used.size+added.length>this.slots)throw new Error(`Uniform detail storage needs ${used.size+added.length} patch slots; capacity is ${this.slots}`);
   let slot=0;
   for(const p of added){while(used.has(slot))slot++;used.add(slot);this.directory[p]=entry(this.slotCoord(slot));}
   this.writeState();
   this.publishTable();
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
  if(!this.layout.domain||field?.kind!=="vertex"||this.installed||this.resident)throw new Error("Uniform detail storage: only a domain placement adopts a vertex base, before its first layout");
  const e=this.extent("vertex",false);
  if(base.format!=="r32float"||base.width!==e[0]||base.height!==e[1]||base.depthOrArrayLayers!==e[2]||(base.usage&need)!==need)
   throw new Error(`Uniform detail storage: an adopted vertex base is an r32float ${e.join("x")} texture with storage, sampled and both copy usages`);
  const own=this.physicals.get(handle)!;
  this.physicals.set(handle,base);this.bases.set(handle,base);this.adopted.set(handle,base);own.destroy();this.rebind();
 }
 /** The textures of a generation the storage owns (an adopted base is its caller's). */
 private owned(textures:Iterable<GPUTexture>):GPUTexture[]{const kept=new Set(this.adopted.values());return [...textures].filter(t=>!kept.has(t));}
 /** Whether the fields are base blocks (the domain placement at C = 0). */
 get baseOnly():boolean{return !!this.layout.domain&&!this.resident;}
 /** Bytes of the field textures at an h-tile capacity (the frame's
  * admission check): only the domain placement depends on it. */
 fieldBytesAt(fineTiles:number):number{
  return this.handles.reduce((n,h)=>{
   const t=this.physicals.get(h)!,texel=t.format==="rgba32float"?16:4,size=(e:readonly number[])=>e[0]!*e[1]!*e[2]!*texel;
   if(!this.layout.domain)return n+size([t.width,t.height,t.depthOrArrayLayers]);
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
  this.retiring=!want&&this.resident;this.verify();
 }
 /** QA (checked): the currency rule as a fact about the textures. Every
  * field is one texture of the current generation (h while C > 0 or until
  * its restriction commits, else its base), and base blocks beside a
  * resident generation exist only as the targets of that restriction. A
  * group still holding the other generation fails device validation at its
  * next submit: that generation is destroyed. */
 private verify():void{
  if(!this.layout.domain?.checked)return;
  const fail=(what:string)=>{throw new Error(`Uniform detail storage: ${what} (capacity ${this.held()} h tiles, ${this.resident?"resident":"base only"}${this.retiring?", retiring":""})`);};
  if(this.resident!==(this.held()>0||this.retiring))fail("the field generation does not follow the h-tile capacity");
  if(this.retired&&!this.retiring)fail("a retirement is pending on a generation that is not retiring");
  for(const h of this.handles){
   const h4=this.resident&&!this.pinned.has(h),t=this.physicals.get(h)!,e=this.extent(fields.get(h)!.kind,h4);
   if(t.width!==e[0]||t.height!==e[1]||t.depthOrArrayLayers!==e[2])fail(`${h.label} is ${t.width}x${t.height}x${t.depthOrArrayLayers}, not its ${h4?"h texture":"base block"}`);
   const b=this.bases.get(h),base=this.extent(fields.get(h)!.kind,false);
   if(!b||b.width!==base[0]||b.height!==base[1]||b.depthOrArrayLayers!==base[2]||(!h4&&b!==t))fail(`${h.label} has no base block of its own beside its field`);
  }
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
  this.resident=true;this.directory[0]=entry([0,0,0]);this.writeState();this.publishTable();this.rebind();this.verify();
  // QA: a fresh h generation is poison throughout (the first layout's uploads fill it instead).
  if(this.layout.domain!.poison&&this.transfers?.ring&&!this.installing){const e=this.device.createCommandEncoder({label:"Uniform detail poison (admission)"});this.encodePoison(e,true);this.device.queue.submit([e.finish()]);}
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
    {binding:2*BATCH,resource:{buffer:held}},{binding:2*BATCH+1,resource:{buffer:target}},...(n===0?[{binding:2*BATCH+2,resource:{buffer:other}},{binding:2*BATCH+3,resource:{buffer:this.state!}}]:[])]}))};
   this.ringGroups.set(target,bound);
  }
  const tiles=this.layout.tiles[0]*this.layout.tiles[1]*this.layout.tiles[2],pass=encoder.beginComputePass({label});
  ring.batches.forEach((b,n)=>{const p=pick(b);if(!p)throw new Error(`Uniform detail storage: ${label} is not compiled`);pass.setPipeline(p);pass.setBindGroup(0,bound!.groups[n]!);pass.dispatchWorkgroups(Math.ceil(tiles/64));});
  pass.end();
 }
 /** QA: the poison into every h texel outside the ring (all: every texel). */
 private encodePoison(encoder:GPUCommandEncoder,all=false):void{
  // No tile enters here: the target binding is the held topology's stand-in.
  this.encodeRing(encoder,this.poisonTarget??=this.device.createBuffer({label:"Uniform detail poison stand-in",size:4,usage:GPUBufferUsage.STORAGE}),b=>all?b.poisonAll:b.poison,all?"Uniform detail poison (all)":"Uniform detail poison");
 }
 private poisonTarget?:GPUBuffer;
 /** After a relayout's remap and phi resolve, in their encoder. share (a
  * host relayout, whose first has no encodeAdmit): the other ownership
  * takes the ring bit of the adopted generation. QA: the tiles the
  * relayout left outside the ring are poisoned. */
 encodeSettle(encoder:GPUCommandEncoder,share:boolean):void{
  if(!this.layout.domain)return;
  if(share){
   const ring=this.kernelsOfDomain().ring!,tiles=this.layout.tiles[0]*this.layout.tiles[1]*this.layout.tiles[2];
   const pass=encoder.beginComputePass({label:"Uniform detail ring share"});pass.setPipeline(ring.share);
   pass.setBindGroup(0,this.shareGroup??=this.device.createBindGroup({label:"Uniform detail ring share",layout:ring.shareLayout,entries:[{binding:0,resource:{buffer:this.simulation!.presentation.buffer}},{binding:1,resource:{buffer:this.other().presentation.buffer}}]}));
   pass.dispatchWorkgroups(Math.ceil(tiles/64));pass.end();
  }
  if(this.layout.domain.poison&&this.resident)this.encodePoison(encoder);
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
  this.resident=false;this.directory[0]=0;this.writeState();this.publishTable();this.rebind();this.release(old);this.verify();
 }
 /** The bases from the h generation, canonical texel for canonical texel:
  * after kernels that store through no twin (the dense t=0 authority and
  * publication). Nothing while the fields are their bases. */
 encodeBaseRefresh(encoder:GPUCommandEncoder,written:readonly GPUTexture[]):void{
  if(!this.layout.domain||!this.resident)return;
  if(this.windowed)throw new Error("Uniform detail storage: the dense t=0 kernels address the lattice, and the h store is a window of it");
  // Only the fields those kernels stored: every other field's h texture
  // holds no current canonical texel outside the ring, and its base is
  // current already.
  const wrote=new Set(written.map(t=>fields.get(t)?.handle));
  const k=this.kernelsOfDomain().refresh,pass=encoder.beginComputePass({label:"Uniform detail base refresh"});
  for(const h of this.following){if(!wrote.has(h))continue;const base=this.bases.get(h)!;this.transfer(pass,k[fields.get(h)!.kind as TransferKind],this.physicals.get(h)!,base,base);}
  pass.end();
  // QA: those kernels wrote the h texels of every tile.
  if(this.layout.domain.poison)this.encodePoison(encoder);
 }
 private kernelsOfDomain():DomainKernels{return unprepared(this.transfers,"a domain detail transfer (the storage is not prepared)");}
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
  const face=head("rgba32float")+(domain.compactFaces?/* wgsl */`
// Base texel m of a tile is its origin (0) or its +face anchor of axis m-1.
fn ddFace(g:vec3u)->vec3i{let m=g.z&3u;let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));return vec3i(4u*vec3u(g.xy,g.z>>2u)+3u*j);}
${kernel("refresh","dst"," textureStore(dst,vec3i(g),textureLoad(src,ddFace(g),0));")}`:/* wgsl */`
// A tile's origin and its three +face anchors (parity 2c + (l==3) per axis); its other four base texels are zero.
${kernel("refresh","dst",` let j=g&vec3u(1u);
 textureStore(dst,vec3i(g),select(vec4f(0.0),textureLoad(src,vec3i(4u*(g>>vec3u(1u))+3u*j),0),j.x+j.y+j.z<=1u));`)}`);
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
const DR_T=vec3u(${T.map(t=>`${t}u`).join(",")});const DR_N:u32=${N}u;const DR_P:f32=${UNIFORM_DETAIL_POISON_WGSL};
// Logical texel p in an h store of extent d, which wraps on an axis shorter than the lattice.
fn drAt(p:vec3u,d:vec3u)->vec3i{return vec3i(p&select(d-vec3u(1u),vec3u(0xffffffffu),d>=4u*DR_T));}
fn drHeld(t:u32)->bool{return (drHeldWords[2u*DR_N+2u*t+1u]&1u)!=0u;}
fn drNext(t:u32)->bool{return (drNextWords[2u*DR_N+2u*t+1u]&1u)!=0u;}
fn drCoord(t:u32)->vec3u{return vec3u(t%DR_T.x,(t/DR_T.x)%DR_T.y,t/(DR_T.x*DR_T.y));}
`;
  const batches=await Promise.all(sets.map(async(handles,n):Promise<RingBatch>=>{
   const format=handles[0]!.format as "r32float"|"rgba32float";
   const layout=device.createBindGroupLayout({label:`Uniform detail ring ${n}`,entries:[
    ...handles.flatMap((_,i)=>[{binding:i,visibility:C,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}},{binding:BATCH+i,visibility:C,storageTexture:{access:"write-only" as const,format,viewDimension:"3d" as const}}]),
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
     seed+=` for(var k=0u;k<8u;k++){let e=vec3u(k&1u,(k>>1u)&1u,k>>2u);if(any((e==vec3u(1u))&(c!=DR_T-vec3u(1u)))){continue;}textureStore(dst${i},drAt(4u*(c+e),textureDimensions(dst${i})),textureLoad(src${i},vec3i(c+e),0));}\n`;
     fill+=` for(var k=0u;k<125u;k++){let e=vec3u(k%5u,(k/5u)%5u,k/25u);if(any((e==vec3u(4u))&(c!=DR_T-vec3u(1u)))){continue;}textureStore(dst${i},drAt(4u*c+e,textureDimensions(dst${i})),vec4f(DR_P));}\n`;
    }else{
     seed+=domain.compactFaces?` for(var m=0u;m<4u;m++){let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));textureStore(dst${i},drAt(4u*c+3u*j,textureDimensions(dst${i})),textureLoad(src${i},vec3i(vec3u(c.xy,4u*c.z+m)),0));}\n`
      :` for(var m=0u;m<4u;m++){let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));textureStore(dst${i},drAt(4u*c+3u*j,textureDimensions(dst${i})),textureLoad(src${i},vec3i(2u*c+j),0));}\n`;
     fill+=` for(var k=0u;k<64u;k++){textureStore(dst${i},vec3i(4u*c+vec3u(k&3u,(k>>2u)&3u,k>>4u)),vec4f(DR_P));}\n`;
    }
   });
   const code=handles.map((_,i)=>`@group(0) @binding(${i}) var src${i}:texture_3d<f32>;\n@group(0) @binding(${BATCH+i}) var dst${i}:texture_storage_3d<${format},write>;\n`).join("")+head
    +(n===0?`@group(0) @binding(${2*BATCH+2}) var<storage,read_write> drShared:array<u32>;\n@group(0) @binding(${2*BATCH+3}) var<storage,read_write> drState:array<atomic<u32>>;\n`:"")+`
@compute @workgroup_size(64) fn enter(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=DR_N){return;}
 let near=drNext(t);
${n===0?` let word=2u*DR_N+2u*t+1u;let w=drShared[word];let m=(w&0xfffffffeu)|u32(near);if(m!=w){drShared[word]=m;}
 // A ring tile outside the h store's window has no texel there: fatal at the frame receipt.
 if(near){let o=drCoord(t);let lo=vec3u(atomicLoad(&drState[${this.words.total}u]),atomicLoad(&drState[${this.words.total+1}u]),atomicLoad(&drState[${this.words.total+2}u]));let hi=vec3u(atomicLoad(&drState[${this.words.total+3}u]),atomicLoad(&drState[${this.words.total+4}u]),atomicLoad(&drState[${this.words.total+5}u]));
  if(any(o<lo)||any(o>hi)){atomicOr(&drState[${D.fatal}u],${WINDOW_FATAL}u);}}
`:""} if(!near||drHeld(t)){return;}
 let c=drCoord(t);
${seed}}
fn drFill(c:vec3u){
${fill}}
@compute @workgroup_size(64) fn poison(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=DR_N||drHeld(t)){return;}
 drFill(drCoord(t));
}
@compute @workgroup_size(64) fn poisonAll(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=DR_N){return;}
 drFill(drCoord(t));
}
`;
   const module=device.createShaderModule({label:`Uniform detail ring ${n}`,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const pipeline=(entryPoint:string)=>compiler.compileComputePipeline({label:`Uniform detail ring ${entryPoint} ${n}`,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}},{priority:"visible"});
   const [enter,poison,poisonAll]=await Promise.all([pipeline("enter"),domain.poison?pipeline("poison"):undefined,domain.poison?pipeline("poisonAll"):undefined]);
   return {handles,layout,enter,poison,poisonAll};
  }));
  const shareLayout=device.createBindGroupLayout({label:"Uniform detail ring share",entries:[{binding:0,visibility:C,buffer:{type:"read-only-storage"}},{binding:1,visibility:C,buffer:{type:"storage"}}]});
  const share=await compiler.compileComputePipeline({label:"Uniform detail ring share",layout:device.createPipelineLayout({bindGroupLayouts:[shareLayout]}),compute:{module:device.createShaderModule({label:"Uniform detail ring share",code:`
@group(0) @binding(0) var<storage,read> drFrom:array<u32>;
@group(0) @binding(1) var<storage,read_write> drTo:array<u32>;
@compute @workgroup_size(64) fn share(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=${N}u){return;}
 let word=${2*N}u+2u*t+1u;let w=drTo[word];let m=(w&0xfffffffeu)|(drFrom[word]&1u);if(m!=w){drTo[word]=m;}
}`}),entryPoint:"share"}},{priority:"visible"});
  return {batches,shareLayout,share};
 }

 private kernels?:Kernels;
 /** Compiles the residency and transfer kernels and binds every field
  * (async, after the last createField and before the first layout change;
  * identity placement has none). ownership: the simulation's, whose
  * topology tail holds the directory the fills read. */
 async prepare(ownership:UniformMixedOwnership,needs=new UniformPipelineNeeds()):Promise<void>{
  if(this.layout.placement==="identity"||this.kernels||this.layout.patch)return;
  if(this.layout.domain){
   const domain=this.layout.domain;this.simulation=ownership;
   this.transfers??=await this.compileDomain(domain);
   // Admission compiles nothing: the ring kernels above seed what a relayout brings into the ring.
   await needs.declare(["transfer"],async()=>{});return;
  }
  const device=this.device,l=this.layout,w=this.words,g=l.patchGrid,P=l.patchEdge,per=P/4,C=GPUShaderStage.COMPUTE;
  const residency=device.createBindGroupLayout({entries:[{binding:0,visibility:C,buffer:{type:"storage"}},{binding:1,visibility:C,buffer:{type:"read-only-storage"}}]});
  const slotsLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:C,buffer:{type:"storage"}}]});
  const stateLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:C,buffer:{type:"read-only-storage"}}]});
  const stagedLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:C,buffer:{type:"read-only-storage"}},{binding:1,visibility:C,buffer:{type:"storage"}}]});
  const each=<T,>(make:(binding:number)=>T)=>Array.from({length:BATCH},(_,binding)=>make(binding));
  const inPlace=device.createBindGroupLayout({entries:each(binding=>({binding,visibility:C,storageTexture:{access:"read-write" as const,format:"r32float" as const,viewDimension:"3d" as const}}))});
  const sampled=device.createBindGroupLayout({entries:each(binding=>({binding,visibility:C,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}}))});
  const written=device.createBindGroupLayout({entries:each(binding=>({binding,visibility:C,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}}))});
  const patchCoord=`fn dsPatch(i:u32)->vec3u{return vec3u(i%${g[0]}u,(i/${g[0]}u)%${g[1]}u,i/${g[0]*g[1]}u);}`;
  const lists=`const DS_DIR:u32=${w.dir}u;const DS_ADMIT:u32=${w.admit}u;const DS_RETIRE:u32=${w.retire}u;const DS_PER:u32=${per}u;const DS_P:u32=${P}u;`;
  const residencyCode=/* wgsl */`
@group(0) @binding(0) var<storage,read_write> ds:array<u32>;
@group(0) @binding(1) var<storage,read> tileWords:array<u32>;
${lists}${patchCoord}
const DS_T=vec3u(${l.tiles.map(t=>`${t}u`).join(",")});const DS_PATCHES:u32=${l.patches}u;
const DS_SLOTS:u32=${w.slots}u;const DS_NEED:u32=${w.need}u;const DS_LX:u32=${l.layer[0]}u;const DS_LY:u32=${l.layer[1]}u;
fn dsSlot(e:u32)->u32{return (e&1023u)+DS_LX*(((e>>10u)&1023u)+DS_LY*((e>>20u)&1023u));}
var<workgroup> dsNeeded:atomic<u32>;
// A patch is needed when a tile of it, or within one tile of it, is h in
// the target generation. One workgroup per patch; lanes share its tiles.
@compute @workgroup_size(64) fn need(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&dsNeeded,ds[${D.all}u]);}
 workgroupBarrier();
 let origin=vec3i(dsPatch(group.x)*DS_PER)-vec3i(1);let side=DS_PER+2u;
 for(var i=lane;i<side*side*side;i+=64u){
  let q=origin+vec3i(vec3u(i%side,(i/side)%side,i/(side*side)));
  if(any(q<vec3i(0))||any(q>=vec3i(DS_T))){continue;}
  if((tileWords[u32(q.x)+DS_T.x*(u32(q.y)+DS_T.y*u32(q.z))]&0x80000000u)!=0u){atomicStore(&dsNeeded,1u);break;}
 }
 workgroupBarrier();
 if(lane==0u){ds[DS_NEED+group.x]=atomicLoad(&dsNeeded);}
}
// One invocation: needed patches without a slot take the lowest free ones
// (in patch order) and join the admitted list; resident patches no longer
// needed join the retiring list and keep their slot until release.
@compute @workgroup_size(1) fn allocate(){
 let capacity=ds[${D.capacity}u];var resident=ds[${D.resident}u];var cursor=0u;var admitted=0u;var retired=0u;var fresh=0u;
 for(var p=0u;p<DS_PATCHES;p++){
  let wanted=ds[DS_NEED+p]!=0u;let e=ds[DS_DIR+p];
  if(wanted&&e==0u){
   fresh++;
   while(cursor<capacity&&ds[DS_SLOTS+cursor]!=0u){cursor++;}
   if(cursor>=capacity){continue;}
   ds[DS_SLOTS+cursor]=p+1u;
   ds[DS_DIR+p]=0x80000000u|(cursor%DS_LX)|(((cursor/DS_LX)%DS_LY)<<10u)|((cursor/(DS_LX*DS_LY))<<20u);
   ds[DS_ADMIT+admitted]=p;admitted++;cursor++;
  }else if(!wanted&&e!=0u){ds[DS_RETIRE+retired]=p;retired++;}
 }
 if(resident+fresh>capacity){ds[${D.fatal}u]|=1u;ds[${D.demand}u]=max(ds[${D.demand}u],resident+fresh);}
 resident+=admitted;
 ds[${D.resident}u]=resident;ds[${D.peak}u]=max(ds[${D.peak}u],resident);ds[${D.admitted}u]=admitted;ds[${D.retired}u]=retired;
}
// After the retiring patches' restriction: their slots are free again.
@compute @workgroup_size(1) fn release(){
 let retired=ds[${D.retired}u];
 for(var i=0u;i<retired;i++){let p=ds[DS_RETIRE+i];ds[DS_SLOTS+dsSlot(ds[DS_DIR+p])]=0u;ds[DS_DIR+p]=0u;}
 ds[${D.resident}u]-=retired;ds[${D.admitted}u]=0u;ds[${D.retired}u]=0u;
}
`;
  // Transfers: group 0 is the simulation ownership (its tail: the directory
  // before this relayout), group 1 the state (the new directory and lists).
  const common=uniformMixedTopologyWGSL(ownership.capacity,0)+detailAddressWGSL(l,"umTopology",`UM_DETAIL+${HEADER}u`,"UM_T","umSupport[UM_DETAIL_VIOLATION]",false)+/* wgsl */`
@group(1) @binding(0) var<storage,read> ds:array<u32>;
${lists}${patchCoord}
const DS_SHIFT:u32=${Math.log2(P)-2}u;
fn dsEntry(index:u32)->vec4u{return vec4u(dsPatch(index),ds[DS_DIR+index]);}
`;
  // Lanes stride a list of `per` tasks per listed patch.
  const stride=(count:number,per:string,body:string)=>/* wgsl */`(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let total=ds[${count}u]*${per};
 for(var i=group.x*64u+lane;i<total;i+=groups.x*64u){
  let n=i/${per};let local=i%${per};
${body}
 }
}`;
  const every=(line:(k:number)=>string)=>each(line).join("");
  const scalar=(k:number)=>`@group(2) @binding(${k}) var f${k}:texture_storage_3d<r32float,read_write>;\n`;
  const cellCode=common+every(scalar)+/* wgsl */`
const DS_E3:u32=DS_P*DS_P*DS_P;const DS_T3:u32=DS_PER*DS_PER*DS_PER;
// A new patch's cells take their tile's base value.
@compute @workgroup_size(64) fn fill${stride(D.admitted,"DS_E3",`  let e=dsEntry(ds[DS_ADMIT+n]);let p=e.xyz*DS_P+umCorner(local,DS_P);
  if(any(p>=UM_D)){continue;}
  let dst=udAtlas_cell(p,e).xyz;let src=udBase_cell(p>>vec3u(2u)).xyz;
${every(k=>`  textureStore(f${k},dst,textureLoad(f${k},src));\n`)}`)}
// A retiring patch's tiles keep their origin cell in the base.
@compute @workgroup_size(64) fn retire${stride(D.retired,"DS_T3",`  let e=dsEntry(ds[DS_RETIRE+n]);let tile=e.xyz*DS_PER+umCorner(local,DS_PER);
  if(any(tile>=UM_T)){continue;}
  let src=udAtlas_cell(tile*4u,e).xyz;let dst=udBase_cell(tile).xyz;
${every(k=>`  textureStore(f${k},dst,textureLoad(f${k},src));\n`)}`)}
`;
  const vertexType="texture_storage_3d<r32float,read_write>";
  const vertexCode=common+every(scalar)+every(k=>detailAccessorWGSL(`f${k}`,"vertex",vertexType,"UM_D+vec3u(1u)","umCounts.w"))+/* wgsl */`
const DS_E:u32=DS_P+1u;const DS_E3:u32=DS_E*DS_E*DS_E;const DS_V:u32=DS_PER+1u;const DS_V3:u32=DS_V*DS_V*DS_V;
fn dsHome(p:vec3u)->vec3u{return min(p>>vec3u(2u),UM_T-vec3u(1u))>>vec3u(DS_SHIFT);}
// A new patch's vertices take what the old directory loads there: the base
// at tile corners, the corners' trilinear interpolant elsewhere.
@compute @workgroup_size(64) fn fill${stride(D.admitted,"DS_E3",`  let e=dsEntry(ds[DS_ADMIT+n]);let p=e.xyz*DS_P+umCorner(local,DS_E);
  if(any(p>UM_D)||any(dsHome(p)!=e.xyz)){continue;}
  let dst=udAtlas_vertex(p,e).xyz;
${every(k=>`  textureStore(f${k},dst,vec4f(udLoad_f${k}(vec3i(p)).x,0.0,0.0,1.0));\n`)}`)}
// A retiring patch's tile corners return to the base.
@compute @workgroup_size(64) fn retire${stride(D.retired,"DS_V3",`  let e=dsEntry(ds[DS_RETIRE+n]);let v=e.xyz*DS_PER+umCorner(local,DS_V);let p=v*4u;
  if(any(v>UM_T)||any(dsHome(p)!=e.xyz)){continue;}
  let src=udAtlas_vertex(p,e).xyz;let dst=udBase_vertex(v).xyz;
${every(k=>`  textureStore(f${k},dst,textureLoad(f${k},src));\n`)}`)}
`;
  // Faces cannot be read and written in place (rgba32float): each patch
  // stages its base-face block, (2 per tile)³ texels.
  const faceHead=(bind:(k:number)=>string)=>common+/* wgsl */`
@group(1) @binding(1) var<storage,read_write> staging:array<vec4f>;
${every(bind)}
const DS_E3:u32=DS_P*DS_P*DS_P;const DS_B:u32=2u*DS_PER;const DS_B3:u32=DS_B*DS_B*DS_B;
`;
  const gatherCode=faceHead(k=>`@group(2) @binding(${k}) var g${k}:texture_3d<f32>;\n`)+/* wgsl */`
@compute @workgroup_size(64) fn fillGather${stride(D.admitted,"DS_B3",`  let e=dsEntry(ds[DS_ADMIT+n]);let b=e.xyz*DS_B+umCorner(local,DS_B);
  if(any(b>=2u*UM_T)){continue;}
  let src=udBase_face(b).xyz;
${every(k=>`  staging[${BATCH}u*i+${k}u]=textureLoad(g${k},src,0);\n`)}`)}
// A retiring patch's canonical faces (origin and the three +face anchors of
// each tile) return to the base; its other base texels are zero.
@compute @workgroup_size(64) fn retireGather${stride(D.retired,"DS_B3",`  let e=dsEntry(ds[DS_RETIRE+n]);let b=umCorner(local,DS_B);let j=b&vec3u(1u);let tile=e.xyz*DS_PER+(b>>vec3u(1u));
  if(any(tile>=UM_T)){continue;}
  let src=udAtlas_face(tile*4u+3u*j,e).xyz;let canonical=j.x+j.y+j.z<=1u;
${every(k=>`  staging[${BATCH}u*i+${k}u]=select(vec4f(0.0),textureLoad(g${k},src,0),canonical);\n`)}`)}
`;
  const scatterCode=faceHead(k=>`@group(2) @binding(${k}) var s${k}:texture_storage_3d<rgba32float,write>;\n`)+/* wgsl */`
// A new patch's faces take what the old directory loads there: its tile's
// base texel of the same parity (2c + (l==3) per axis).
@compute @workgroup_size(64) fn fillScatter${stride(D.admitted,"DS_E3",`  let e=dsEntry(ds[DS_ADMIT+n]);let p=e.xyz*DS_P+umCorner(local,DS_P);
  if(any(p>=UM_D)){continue;}
  let dst=udAtlas_face(p,e).xyz;
  let b=2u*((p>>vec3u(2u))-e.xyz*DS_PER)+select(vec3u(0u),vec3u(1u),(p&vec3u(3u))==vec3u(3u));
  let s=${BATCH}u*(n*DS_B3+b.x+DS_B*(b.y+DS_B*b.z));
${every(k=>`  textureStore(s${k},dst,staging[s+${k}u]);\n`)}`)}
@compute @workgroup_size(64) fn retireScatter${stride(D.retired,"DS_B3",`  let e=dsEntry(ds[DS_RETIRE+n]);let b=e.xyz*DS_B+umCorner(local,DS_B);
  if(any(b>=2u*UM_T)){continue;}
  let dst=udBase_face(b).xyz;
${every(k=>`  textureStore(s${k},dst,staging[${BATCH}u*i+${k}u]);\n`)}`)}
`;
  const compiler=gpuCompilationManagerFor(device);
  const module=(label:string,code:string)=>device.createShaderModule({label:`Uniform detail ${label}`,code});
  const modules={residency:module("residency",residencyCode),cell:module("cell transfer",cellCode),vertex:module("vertex transfer",vertexCode),gather:module("face gather",gatherCode),scatter:module("face scatter",scatterCode)};
  const pipeline=(part:keyof typeof modules,entryPoint:string,groups:GPUBindGroupLayout[])=>compiler.compileComputePipeline({label:`Uniform detail ${part} ${entryPoint}`,
   layout:device.createPipelineLayout({bindGroupLayouts:groups}),compute:{module:modules[part],entryPoint}},{priority:"visible"});
  const o=ownership.bindLayout;
  const [need,allocate,release,cellFill,cellRetire,vertexFill,vertexRetire,fillGather,fillScatter,retireGather,retireScatter]=await Promise.all([
   pipeline("residency","need",[residency]),pipeline("residency","allocate",[slotsLayout]),pipeline("residency","release",[slotsLayout]),
   pipeline("cell","fill",[o,stateLayout,inPlace]),pipeline("cell","retire",[o,stateLayout,inPlace]),
   pipeline("vertex","fill",[o,stateLayout,inPlace]),pipeline("vertex","retire",[o,stateLayout,inPlace]),
   pipeline("gather","fillGather",[o,stagedLayout,sampled]),pipeline("scatter","fillScatter",[o,stagedLayout,written]),
   pipeline("gather","retireGather",[o,stagedLayout,sampled]),pipeline("scatter","retireScatter",[o,stagedLayout,written]),
  ]);
  this.kernels={ownership,residency,slots:device.createBindGroup({layout:slotsLayout,entries:[{binding:0,resource:{buffer:this.state!}}]}),
   need,allocate,release,cell:{fill:cellFill,retire:cellRetire},vertex:{fill:vertexFill,retire:vertexRetire},
   face:{fillGather,fillScatter,retireGather,retireScatter},layouts:{state:stateLayout,staged:stagedLayout,inPlace,sampled,written},batches:[],owned:[]};
  this.bindKernels();
 }
 /** The transfer kernels' groups over the fields' physical textures, in
  * batches of BATCH per class (spare bindings take 1³ dummies), and the
  * face staging for the allocated slots. */
 private bindKernels():void{
  const k=this.kernels;if(!k)return;
  const device=this.device,l=this.layout,B=l.patchEdge/2,L=k.layouts;
  for(const r of k.owned)r.destroy();
  const owned:(GPUTexture|GPUBuffer)[]=[],batches:Kernels["batches"]=[];
  const dummy=(format:GPUTextureFormat)=>{const t=device.createTexture({label:"Uniform detail spare transfer binding",size:[1,1,1],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});owned.push(t);return t;};
  const of=(kind:UniformDetailClass)=>this.handles.filter(t=>this.kindOf(t)===kind).map(t=>this.physicals.get(t)!);
  const faces=of("face");
  let staging:GPUBuffer|undefined;
  if(faces.length){
   // A relayout lists at most the allocated slots' patches.
   const size=16*BATCH*Math.max(1,this.layers*l.layer[0]*l.layer[1])*B*B*B;
   if(size>device.limits.maxStorageBufferBindingSize)throw new Error(`Uniform detail face staging needs ${size} bytes; the device binds at most ${device.limits.maxStorageBufferBindingSize}`);
   staging=device.createBuffer({label:"Uniform detail face staging",size,usage:GPUBufferUsage.STORAGE});owned.push(staging);
  }
  const state=device.createBindGroup({layout:L.state,entries:[{binding:0,resource:{buffer:this.state!}}]});
  const staged=staging&&device.createBindGroup({layout:L.staged,entries:[{binding:0,resource:{buffer:this.state!}},{binding:1,resource:{buffer:staging}}]});
  for(const kind of ["cell","vertex","face"] as const){
   const list=of(kind),format=kind==="face"?"rgba32float":"r32float";
   for(let at=0;at<list.length;at+=BATCH){
    const set=list.slice(at,at+BATCH);while(set.length<BATCH)set.push(dummy(format));
    const views=(layout:GPUBindGroupLayout)=>device.createBindGroup({layout,entries:set.map((t,binding)=>({binding,resource:t.createView()}))});
    batches.push(kind==="face"?{kind,state:staged!,storage:views(L.written),sampled:views(L.sampled)}:{kind,state,storage:views(L.inPlace)});
   }
  }
  k.batches=batches;k.owned=owned;
 }
 private kernel():Kernels{
  if(!this.kernels)throw new Error("Uniform detail residency kernels were not prepared");return this.kernels;
 }
 private readonly residencyGroups=new WeakMap<GPUBuffer,GPUBindGroup>();
 private publish(encoder:GPUCommandEncoder):void{
  for(const o of this.ownerships)o.encodeDetail(encoder,this.state!,4*this.words.dir,HEADER,this.layout.patches);
 }
 /** Before a remap, in its encoder: admit the patches of the target
  * generation (target: its tile words, one per tile from word 0) beside the
  * resident ones, fill them under the directory the fields were written
  * with, then publish the union directory. */
 encodeAdmit(encoder:GPUCommandEncoder,target:GPUBuffer):void{
  // Domain: admission is the host's, when the capacity leaves zero (follow).
  // Haloed patches: the layout's patches are the resident ones (reserve).
  if(this.layout.domain){
   // The tiles the target brings into the ring take their canonical texels from the bases.
   if(this.resident)this.encodeRing(encoder,target,b=>b.enter,"Uniform detail ring enter");
   return;
  }
  if(this.layout.placement==="identity"||this.layout.patch)return;
  const k=this.kernel();this.mirrored=false;
  let residency=this.residencyGroups.get(target);
  if(!residency)this.residencyGroups.set(target,residency=this.device.createBindGroup({layout:k.residency,entries:[{binding:0,resource:{buffer:this.state!}},{binding:1,resource:{buffer:target}}]}));
  const pass=encoder.beginComputePass({label:"Uniform detail admit"});
  pass.setBindGroup(0,residency);
  pass.setPipeline(k.need);pass.dispatchWorkgroups(this.layout.patches);
  pass.setBindGroup(0,k.slots);pass.setPipeline(k.allocate);pass.dispatchWorkgroups(1);
  pass.setBindGroup(0,k.ownership.bindGroup);
  for(const b of k.batches){
   pass.setBindGroup(1,b.state);
   if(b.kind==="face"){
    pass.setBindGroup(2,b.sampled!);pass.setPipeline(k.face.fillGather);pass.dispatchWorkgroups(TRANSFER_GRID);
    pass.setBindGroup(2,b.storage);pass.setPipeline(k.face.fillScatter);pass.dispatchWorkgroups(TRANSFER_GRID);
   }else{pass.setBindGroup(2,b.storage);pass.setPipeline(k[b.kind].fill);pass.dispatchWorkgroups(TRANSFER_GRID);}
  }
  pass.end();
  this.publish(encoder);
 }
 /** After the remap and its phi resolve, in their encoder: restrict the
  * patches the last encodeAdmit found unneeded into the base, free their
  * slots and publish the directory without them. */
 encodeRetire(encoder:GPUCommandEncoder):void{
  if(this.layout.placement==="identity")return;
  if(this.layout.domain){this.encodeRestriction(encoder);return;}
  if(this.layout.patch)return;
  const k=this.kernel();
  const pass=encoder.beginComputePass({label:"Uniform detail retire"});
  pass.setBindGroup(0,k.ownership.bindGroup);
  for(const b of k.batches){
   pass.setBindGroup(1,b.state);
   if(b.kind==="face"){
    pass.setBindGroup(2,b.sampled!);pass.setPipeline(k.face.retireGather);pass.dispatchWorkgroups(TRANSFER_GRID);
    pass.setBindGroup(2,b.storage);pass.setPipeline(k.face.retireScatter);pass.dispatchWorkgroups(TRANSFER_GRID);
   }else{pass.setBindGroup(2,b.storage);pass.setPipeline(k[b.kind].retire);pass.dispatchWorkgroups(TRANSFER_GRID);}
  }
  pass.setBindGroup(0,k.slots);pass.setPipeline(k.release);pass.dispatchWorkgroups(1);
  pass.end();
  this.publish(encoder);
 }
 /** The residency header into a frame receipt (UNIFORM_DETAIL_RECEIPT_BYTES). */
 encodeReceipt(encoder:GPUCommandEncoder,readback:GPUBuffer,offset:number):void{
  if(this.state)encoder.copyBufferToBuffer(this.state,0,readback,offset,UNIFORM_DETAIL_RECEIPT_BYTES);
  else encoder.clearBuffer(readback,offset,UNIFORM_DETAIL_RECEIPT_BYTES);
 }
 /** A frame receipt's residency words: an overflowed relayout is fatal. */
 noteReceipt(words:Uint32Array):void{
  if(this.layout.placement==="identity")return;
  if(this.layout.domain&&words[D.fatal]!&WINDOW_FATAL)throw new Error(`Uniform detail storage: a relayout put a detail-ring tile outside the h store's window (tiles ${this.box.slice(0,3)} to ${this.box.slice(3)}, store ${this.cells} cells): the host's bound on the h tiles did not hold`);
  if(words[D.fatal])throw new Error(`Uniform detail storage needs ${words[D.demand]} patch slots; capacity is ${words[D.capacity]}`);
  this.receipt={capacity:words[D.capacity]!,resident:words[D.resident]!,peak:words[D.peak]!};
 }
 /** Refresh the host directory from the GPU (diagnostics and readbacks). */
 async sync():Promise<void>{
  if(this.layout.placement==="identity"||this.mirrored)return;
  const w=this.words,staging=this.device.createBuffer({size:4*w.slots,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(this.state!,0,staging,0,4*w.slots);
   this.device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
   const words=new Uint32Array(staging.getMappedRange());
   if(words[D.fatal])throw new Error(`Uniform detail storage needs ${words[D.demand]} patch slots; capacity is ${words[D.capacity]}`);
   this.directory.set(words.subarray(w.dir,w.slots));this.mirrored=true;
  }finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
 }
 /** Haloed patches, admission before adoption (the frame's updateLayout,
  * ahead of reserveFine): a layout whose patches outrun a fixed pool, or
  * (slice step 2: no promotion or retirement) are not the resident ones, is
  * refused with nothing changed. */
 refusal(layout:UniformMixedLayout):string|undefined{
  if(!this.layout.patch)return undefined;
  const need=this.patchesFor(layout),held=this.expected;
  if(this.fixed&&need.size>this.slots)return `the layout's h tiles need ${need.size} detail patches of ${this.layout.patchEdge}³ h; the pool holds ${this.slots}`;
  if(held&&(need.size!==held.size||[...need].some(p=>!held.has(p))))return `the layout's h tiles need ${need.size} detail patches other than the ${held.size} resident (haloed patches hold their first layout)`;
  return undefined;
 }
 /** A field's physical texels (tight rows). */
 private async physicalImage(texture:GPUTexture):Promise<Float32Array>{
  const source=this.physical(texture),comp=texture.format==="rgba32float"?4:1,W=source.width,H=source.height,Z=source.depthOrArrayLayers,row=W*comp*4,pitch=Math.ceil(row/256)*256;
  const staging=this.device.createBuffer({size:pitch*H*Z,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=this.device.createCommandEncoder({label:"Uniform detail physical readback"});
  encoder.copyTextureToBuffer({texture:source},{buffer:staging,bytesPerRow:pitch,rowsPerImage:H},[W,H,Z]);this.device.queue.submit([encoder.finish()]);
  try{
   await staging.mapAsync(GPUMapMode.READ);const src=new Uint8Array(staging.getMappedRange()),dst=new Uint8Array(row*H*Z);
   for(let r=0;r<H*Z;r++)dst.set(src.subarray(r*pitch,r*pitch+row),r*row);
   return new Float32Array(dst.buffer);
  }finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
 }
 /** QA, haloed patches: the write-through rule as a fact about the texture.
  * Every halo texel whose logical texel has an authority (its home patch's
  * interior, else the base when canonical, else the first resident box)
  * against it, and every base texel under a resident patch against the
  * patch's. Bit compares. */
 async audit(texture:GPUTexture):Promise<{replicas:number;stale:number;bases:number;staleBases:number;first?:string}>{
  const l=this.layout,kind=this.kindOf(texture);if(!l.patch)throw new Error("Uniform detail audit: haloed patches only");
  const [lx,ly,lz]=this.logicalOf(texture),comp=texture.format==="rgba32float"?4:1,t=this.physical(texture),W=t.width,H=t.height;
  const bits=new Uint32Array((await this.physicalImage(texture)).buffer),c=l.classes[kind],E=c.spacing,r=haloOf(l,kind),shift=Math.log2(l.patchEdge);
  const out={replicas:0,stale:0,bases:0,staleBases:0,first:undefined as string|undefined};
  const differ=(a:number,b:number)=>{for(let k=0;k<comp;k++)if(bits[a*comp+k]!==bits[b*comp+k])return true;return false;};
  const index=(at:readonly number[])=>at[0]!+W*(at[1]!+H*at[2]!);
  this.directory.forEach((e,patch)=>{
   if(!e)return;
   const n=this.patchCoord(patch),slot=[e&1023,(e>>>10)&1023,(e>>>20)&1023];
   for(let k=0;k<E;k++)for(let j=0;j<E;j++)for(let i=0;i<E;i++){
    const p:Dims=[(n[0]<<shift)-r+i,(n[1]<<shift)-r+j,(n[2]<<shift)-r+k];
    if(p[0]<0||p[1]<0||p[2]<0||p[0]>=lx||p[1]>=ly||p[2]>=lz)continue;
    const here=slot[0]!*E+i+W*(slot[1]!*E+j+H*(c.atlasZ+slot[2]!*E+k)),at=this.locate(kind,p);
    if(at[3]===1){
     const authority=index(at);if(authority===here)continue;
     out.replicas++;if(differ(here,authority)){out.stale++;out.first??=`${texture.label} ${kind} ${p} in patch ${n}`;}
    }else if(at[3]===0&&UniformDetailStorage.canonical(kind,p)){out.replicas++;if(differ(here,index(at))){out.stale++;out.first??=`${texture.label} ${kind} ${p} in patch ${n} against the base`;}}
   }
  });
  if(kind!=="atlas")for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++){
   const p:Dims=[x,y,z];if(!UniformDetailStorage.canonical(kind,p))continue;
   const at=this.locate(kind,p);if(at[3]!==1)continue;
   const base=index(this.baseTexel(kind,(kind==="face"?p.map(v=>2*(v>>2)+((v&3)===3?1:0)):p.map(v=>v>>2)) as unknown as Dims));
   out.bases++;if(differ(base,index(at))){out.staleBases++;out.first??=`${texture.label} ${kind} base of ${p}`;}
  }
  return out;
 }
 /** QA, haloed patches in profile mode: each numbered site's word, over every attached ownership. */
 async siteWords():Promise<Uint32Array>{
  const count=Math.min(patchSites.length,PATCH_SITE_WORDS),out=new Uint32Array(count);
  if(!(this.layout.patch?.mode==="profile"||this.layout.domain?.survey||this.layout.domain?.poison)||!count)return out;
  for(const o of this.ownerships){
   const staging=this.device.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=this.device.createCommandEncoder();
   encoder.copyBufferToBuffer(o.support,4*(uniformMixedDetailViolationWord(o.capacity.tiles,o.capacity.lattice)+4),staging,0,4*count);this.device.queue.submit([encoder.finish()]);
   try{await staging.mapAsync(GPUMapMode.READ);new Uint32Array(staging.getMappedRange()).forEach((w,i)=>{out[i]=out[i]!|w;});}
   finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
  }
  return out;
 }
 /** QA: the currency rule on the GPU's textures. For every following field,
  * the canonical texels of ring tiles and of the rest: how many the h
  * texture and the base block both hold, and how many differ bitwise. The
  * ring keeps the first difference zero. */
 async currency():Promise<{field:string;ring:[number,number];outside:[number,number]}[]>{
  const domain=this.layout.domain;if(!domain||!this.resident||!this.simulation)return [];
  const device=this.device,C=GPUShaderStage.COMPUTE,T=this.layout.tiles,N=T[0]*T[1]*T[2],following=this.following;
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:C,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:1,visibility:C,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
   {binding:2,visibility:C,buffer:{type:"read-only-storage"}},{binding:3,visibility:C,buffer:{type:"storage"}}]});
  const body:Record<TransferKind,string>={
   cell:" drTally(r,textureLoad(auBase,vec3i(c),0),textureLoad(auField,auAt(4u*c),0));",
   vertex:" for(var k=0u;k<8u;k++){let e=vec3u(k&1u,(k>>1u)&1u,k>>2u);if(any((e==vec3u(1u))&(c!=DR_T-vec3u(1u)))){continue;}drTally(r,textureLoad(auBase,vec3i(c+e),0),textureLoad(auField,auAt(4u*(c+e)),0));}",
   face:` for(var m=0u;m<4u;m++){let j=vec3u(u32(m==1u),u32(m==2u),u32(m==3u));drTally(r,textureLoad(auBase,vec3i(${domain.compactFaces?"vec3u(c.xy,4u*c.z+m)":"2u*c+j"}),0),textureLoad(auField,auAt(4u*c+3u*j),0));}`};
  const tally=device.createBuffer({size:16*following.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:tally.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const compiler=gpuCompilationManagerFor(device);
  // One pipeline per field (its tally slot is a constant).
  const pipelines=await Promise.all(following.map((h,i)=>{
   const kind=fields.get(h)!.kind as TransferKind,module=device.createShaderModule({code:`
@group(0) @binding(0) var auBase:texture_3d<f32>;
@group(0) @binding(1) var auField:texture_3d<f32>;
@group(0) @binding(2) var<storage,read> auHeld:array<u32>;
@group(0) @binding(3) var<storage,read_write> auTally:array<atomic<u32>>;
const DR_T=vec3u(${T.map(t=>`${t}u`).join(",")});const DR_N:u32=${N}u;
// Only the ring's tallies mean anything (outside it the h store holds nothing current, and a window has no texel).
fn auAt(p:vec3u)->vec3i{let d=textureDimensions(auField);return vec3i(p&select(d-vec3u(1u),vec3u(0xffffffffu),d>=4u*DR_T));}
fn drTally(r:u32,a:vec4f,b:vec4f){atomicAdd(&auTally[${4*i}u+r],1u);if(any(bitcast<vec4u>(a)!=bitcast<vec4u>(b))){atomicAdd(&auTally[${4*i}u+r+1u],1u);}}
@compute @workgroup_size(64) fn audit(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=DR_N){return;}
 let r=select(2u,0u,(auHeld[2u*DR_N+2u*t+1u]&1u)!=0u);let c=vec3u(t%DR_T.x,(t/DR_T.x)%DR_T.y,t/(DR_T.x*DR_T.y));
${body[kind]}
}`});
   return compiler.compileComputePipeline({label:`Uniform detail currency audit ${i}`,layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"audit"}},{priority:"visible"});
  }));
  const encoder=device.createCommandEncoder({label:"Uniform detail currency audit"});
  following.forEach((h,i)=>{
   const pass=encoder.beginComputePass();pass.setPipeline(pipelines[i]!);
   pass.setBindGroup(0,device.createBindGroup({layout,entries:[{binding:0,resource:this.bases.get(h)!.createView()},{binding:1,resource:this.physicals.get(h)!.createView()},{binding:2,resource:{buffer:this.simulation!.presentation.buffer}},{binding:3,resource:{buffer:tally}}]}));
   pass.dispatchWorkgroups(Math.ceil(N/64));pass.end();
  });
  encoder.copyBufferToBuffer(tally,0,read,0,tally.size);device.queue.submit([encoder.finish()]);
  try{
   await read.mapAsync(GPUMapMode.READ);const w=new Uint32Array(read.getMappedRange());
   return following.map((h,i)=>({field:h.label,ring:[w[4*i]!,w[4*i+1]!] as [number,number],outside:[w[4*i+2]!,w[4*i+3]!] as [number,number]}));
  }finally{if(read.mapState==="mapped")read.unmap();read.destroy();tally.destroy();}
 }
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
  const identity=this.layout.placement==="identity",source=this.physical(texture);
  const W=source.width,H=source.height,Z=source.depthOrArrayLayers,row=W*comp*4,pitch=Math.ceil(row/256)*256,patches=this.layout.patches;
  const usage=GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ;
  // Domain: residency is the host's, so the directory of this encode is known here.
  const hosted=this.layout.domain?(this.pinned.has(fields.get(texture)!.handle)?new Uint32Array(1):this.directory.slice()):undefined;
  const staging=this.device.createBuffer({size:pitch*H*Z,usage}),placed=identity||hosted?undefined:this.device.createBuffer({size:4*patches,usage});
  encoder.copyTextureToBuffer({texture:source},{buffer:staging,bytesPerRow:pitch,rowsPerImage:H},[W,H,Z]);
  // A wrapped store: the box whose texels it holds at this encode.
  const wrapped=this.layout.domain&&this.twinned(texture)&&this.windowed?[...this.box]:undefined;
  // Outside the ring a canonical texel lives in the base alone, so every one is read from there.
  const base=this.layout.domain&&this.twinned(texture)?this.bases.get(fields.get(texture)!.handle)!:undefined;
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
   return identity?image:this.logical(kind,[lx,ly,lz],comp,image,[W,H],directory!);
  };
 }
 private logical(kind:UniformDetailClass,[lx,ly,lz]:Dims,comp:number,image:Float32Array,[W,H]:readonly [number,number],directory:Uint32Array):Float32Array{
  if(this.layout.domain)return this.logicalDomain(kind,[lx,ly,lz],comp,image,[W,H],directory[0]!==0);
  const out=new Float32Array(lx*ly*lz*comp);
  const texel=(at:readonly number[])=>(at[0]!+W*(at[1]!+H*at[2]!))*comp;
  for(let z=0;z<lz;z++)for(let y=0;y<ly;y++)for(let x=0;x<lx;x++){
   const p:Dims=[x,y,z],at=this.locate(kind,p,directory),dst=(x+lx*(y+ly*z))*comp;
   if(at[3]===-1)continue;
   if(at[3]!==2){for(let c=0;c<comp;c++)out[dst+c]=image[texel(at)+c]!;continue;}
   // Hanging vertex of a non-resident tile: trilinear in its home corners,
   // summed in umVertexSum8 order.
   const o=at.slice(0,3).map(v=>4*v),t=p.map((v,a)=>(v-o[a]!)/4),values=new Array<number>(8).fill(0);
   for(let k=0;k<8;k++){
    const corner=[k&1,(k>>1)&1,k>>2],w=corner.map((c,a)=>c?t[a]!:1-t[a]!);const weight=Math.fround(Math.fround(w[0]!*w[1]!)*w[2]!);
    if(weight>0)values[k]=Math.fround(weight*image[texel(this.locate("vertex",o.map((v,a)=>v+4*corner[a]!) as unknown as Dims,directory))]!);
   }
   const f=Math.fround,v=values;
   out[dst]=f(f(f(v[0]!+v[5]!)+f(v[1]!+v[4]!))+f(f(v[2]!+v[7]!)+f(v[3]!+v[6]!)));
  }
  return out;
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
function detailAddressWGSL(l:UniformDetailLayout,table:string|((index:string)=>string),start:string,tiles:string,support:string|undefined,atomic:boolean):string{
 const word=typeof table==="string"?(index:string)=>`${table}[${index}]`:table;
 const shift=Math.log2(l.patchEdge);
 const g=l.patchGrid,c=(k:UniformDetailClass)=>l.classes[k];
 const atlas=(k:UniformDetailClass)=>`fn udAtlas_${k}(p:vec3u,e:vec4u)->vec4i{
 let slot=vec3u(e.w&1023u,(e.w>>10u)&1023u,(e.w>>20u)&1023u);
 return vec4i(vec3i(slot*${c(k).spacing}u+vec3u(0u,0u,${c(k).atlasZ}u)+p-(e.xyz<<vec3u(${shift}u))),1);
}
`;
 const baseFn=(k:UniformDetailClass)=>{const b=c(k);return `fn udBase_${k}(b:vec3u)->vec4i{
 let s=b.z%${b.per}u;return vec4i(vec3i(vec3u(b.x+${b.base[0]}u*(s%${b.fold}u),b.y+${b.base[1]}u*(s/${b.fold}u),b.z/${b.per}u)),0);
}
`;};
 return /* wgsl */`
fn udViolation(bits:u32){${!support?"":atomic?`atomicOr(&${support},bits);`:`${support}=${support}|bits;`}}
fn udEntry(tile:vec3u)->vec4u{
 let pc=tile>>vec3u(${shift-2}u);return vec4u(pc,${word(`${start}+pc.x+${g[0]}u*(pc.y+${g[1]}u*pc.z)`)});
}
${(["cell","face","vertex"] as const).map(atlas).join("")}${(["cell","face","vertex"] as const).map(baseFn).join("")}
fn udCell(p:vec3u)->vec4i{
 let tile=p>>vec3u(2u);let e=udEntry(tile);
 if(e.w!=0u){return udAtlas_cell(p,e);}
 return udBase_cell(tile);
}
fn udFace(p:vec3u)->vec4i{
 let tile=p>>vec3u(2u);let e=udEntry(tile);
 if(e.w!=0u){return udAtlas_face(p,e);}
 return udBase_face(2u*tile+select(vec3u(0u),vec3u(1u),(p&vec3u(3u))==vec3u(3u)));
}
fn udVertex(p:vec3u)->vec4i{
 let home=min(p>>vec3u(2u),${tiles}-vec3u(1u));let e=udEntry(home);
 if(e.w!=0u){return udAtlas_vertex(p,e);}
 if(all((p&vec3u(3u))==vec3u(0u))){return udBase_vertex(p>>vec3u(2u));}
 return vec4i(vec3i(home),2);
}
fn udUnit(p:vec3u)->vec4i{let e=udEntry(p>>vec3u(2u));if(e.w!=0u){return udAtlas_face(p,e);}return vec4i(-1);}
fn udCellCanonical(p:vec3u)->bool{return all((p&vec3u(3u))==vec3u(0u));}
fn udFaceCanonical(p:vec3u)->bool{
 let l=p&vec3u(3u);let three=select(vec3u(0u),vec3u(1u),l==vec3u(3u));
 return all((l==vec3u(0u))|(l==vec3u(3u)))&&three.x+three.y+three.z<=1u;
}
`;
}
const ADDRESS={cell:"udCell",face:"udFace",vertex:"udVertex",atlas:"udUnit"} as const;
const UNIFORM_DETAIL_POISON_WGSL="-0x1p+100f";
/** Load/store accessors of one field. dims: logical extent expression (vec3u).
 * Measured on Metal (dam 64³, every patch resident, frame +19-26% over
 * identity): the per-access directory load is ~+10-15% of the frame (a
 * uniform-buffer array costs the same as the storage tail), the residency
 * and hanging branches ~+9% (the two overlap), the address arithmetic
 * ~+3-5%. No run-time arrangement avoids the load: a last-patch cache, a
 * home-patch private, a flag-selected or loop-gated read and one window
 * lookup per sampler stencil are all null or slower. Only an entry known
 * at compile time is cheap. */
function detailAccessorWGSL(name:string,kind:UniformDetailClass,type:string,dims:string,bound:string,deposit=false):string{
 const storage=type.startsWith("texture_storage"),readable=!storage||type.includes("read_write"),writable=storage;
 const level=storage?"":",0";let out="";
 const clamp=`let p=vec3u(clamp(q,vec3i(0),vec3i(${dims})-vec3i(1)));`;
 if(readable){
  const body=kind==="vertex"?`let a=udVertex(p);if(a.w!=2){return textureLoad(${name},a.xyz${level});}
 let origin=vec3u(a.xyz)*4u;let t=vec3f(p-origin)/4.0;var values:array<f32,8>;
 for(var k=0u;k<${bound};k++){
  let corner=vec3u(k%2u,(k/2u)%2u,k/4u);let w=select(vec3f(1.0)-t,t,corner!=vec3u(0u));let weight=w.x*w.y*w.z;
  if(weight>0.0){values[k]=weight*textureLoad(${name},udVertex(origin+corner*4u).xyz${level}).x;}
 }
 return vec4f(((values[0]+values[5])+(values[1]+values[4]))+((values[2]+values[7])+(values[3]+values[6])),0.0,0.0,1.0);`
   :kind==="atlas"?`let a=udUnit(p);if(a.w<0){return vec4f(0.0);}return textureLoad(${name},a.xyz${level});`
   :`return textureLoad(${name},${ADDRESS[kind]}(p).xyz${level});`;
  out+=`fn udLoad_${name}(q:vec3i)->vec4f{
 ${clamp}
 ${body}
}
`;
 }
 if(writable){
  const fail=(bits:number)=>deposit?"return;":`udViolation(${bits}u);return;`;
  const check=kind==="cell"?`if(a.w==0&&!udCellCanonical(p)){${fail(UNIFORM_DETAIL_VIOLATION.cell)}}`
   :kind==="face"?`if(a.w==0&&!udFaceCanonical(p)){${fail(UNIFORM_DETAIL_VIOLATION.face)}}`
   :kind==="vertex"?`if(a.w==2){${fail(UNIFORM_DETAIL_VIOLATION.vertex)}}`
   :`if(a.w<0){${fail(UNIFORM_DETAIL_VIOLATION.atlas)}}`;
  out+=`fn udStore_${name}(q:vec3i,value:vec4f){
 if(any(q<vec3i(0))||any(q>=vec3i(${dims}))){return;}
 let p=vec3u(q);let a=${ADDRESS[kind]}(p);
 ${check}
 textureStore(${name},a.xyz,value);
}
`;
 }
 return out;
}

/** QA spike: the patch-local accessors (UniformDetailOptions.local). The home
 * entry is resolved at the head of every compute entry point into one
 * origin per class; a load is its logical clamp plus that origin. */
function detailLocalShader(code:string,l:UniformDetailLayout,where:UniformDetailLocal,pf:string,kinds:ReadonlyMap<string,{kind:UniformDetailClass;type:string}>):string{
 const shift=Math.log2(l.patchEdge),g=l.patchGrid,[lx,ly]=l.layer;
 const origin=(k:"cell"|"face"|"vertex")=>`udO_${k}=slot*${l.classes[k].spacing}+vec3i(0,0,${l.classes[k].atlasZ})-home;`;
 const patch=`vec3u(i%${g[0]}u,(i/${g[0]}u)%${g[1]}u,i/${g[0]*g[1]}u)`,read=`${pf}umTopology[${pf}UM_DETAIL+${HEADER}u+i]`;
 // i: the home patch index, a run-time zero (the unused counts word) the compiler cannot fold.
 const home=where==="constant"?`return vec4u(0u,0u,0u,0x80000000u);`
  :where==="record"?`let i=${pf}umCounts.z;return vec4u(${patch},0x80000000u|(i%${lx}u)|(((i/${lx}u)%${ly}u)<<10u)|((i/${lx*ly}u)<<20u));`
  :`let i=${pf}umCounts.z;return vec4u(${patch},${read});`;
 let helpers=/* wgsl */`
var<private> udO_cell:vec3i;var<private> udO_face:vec3i;var<private> udO_vertex:vec3i;
${where==="workgroup"?"var<workgroup> udShared:u32;\n":""}fn udHome()->vec4u{${home}}
fn udResolve(e:vec4u){
 let slot=vec3i(vec3u(e.w&1023u,(e.w>>10u)&1023u,(e.w>>20u)&1023u));let home=vec3i(e.xyz<<vec3u(${shift}u));
 ${origin("cell")}${origin("face")}${origin("vertex")}
}
`;
 for(const [name,{kind,type}] of kinds){
  const storage=type.startsWith("texture_storage"),o=`udO_${kind==="atlas"?"face":kind}`,dims=kind==="vertex"?`${pf}UM_D+vec3u(1u)`:`${pf}UM_D`;
  if(!storage||type.includes("read_write"))helpers+=`fn udLoad_${name}(q:vec3i)->vec4f{return textureLoad(${name},${o}+clamp(q,vec3i(0),vec3i(${dims})-vec3i(1))${storage?"":",0"});}\n`;
  if(storage)helpers+=`fn udStore_${name}(q:vec3i,value:vec4f){if(any(q<vec3i(0))||any(q>=vec3i(${dims}))){return;}textureStore(${name},${o}+q,value);}\n`;
 }
 let heads=0;
 const out=code.replace(/(@compute\s+@workgroup_size\([^)]*\)\s*fn\s+\w+\s*\()([^{]*)\)(\s*\{)/g,(_m,head:string,params:string,open:string)=>{
  heads++;
  if(where!=="workgroup")return `${head}${params})${open}udResolve(udHome());`;
  const lane=/@builtin\(local_invocation_index\)\s*(\w+)/.exec(params)?.[1];
  const list=lane?params:`${params.trim()?`${params},`:""}@builtin(local_invocation_index) udLane:u32`;
  return `${head}${list})${open}if(${lane??"udLane"}==0u){udShared=udHome().w;}udResolve(vec4u(0u,0u,0u,workgroupUniformLoad(&udShared)));`;
 });
 if(heads!==(code.match(/@compute\b/g)?.length??0))throw new Error("Uniform detail patch-local rewrite: a compute entry point has no resolve");
 return out+helpers;
}

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
 /** The module declares udWindow (uniformDetailShader): both store forms. */
 fold=false):string{
 const j="(((q&vec3i(3))+vec3i(1))>>vec3u(2u))";
 // ringed: a canonical store reaches the h texture only inside the ring.
 // watch (checked): every h texture load is tested against the ring (and the poison).
 const ringed=!!ring,watch=ringed&&domain.checked,sites=watch&&ring!.sites;
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
 var bits=${R.ran}u;if(!udRingLoad(clamp(q,vec3i(0),top))){bits|=${R.outside}u;}${domain.poison?`if(v==${UNIFORM_DETAIL_POISON_WGSL}){bits|=${R.poison}u;}`:""}
 ${mark}${domain.survey?"":`if((bits&${R.outside}u)!=0u){udViolation(${UNIFORM_DETAIL_RING_VIOLATION}u);}if((bits&${R.poison}u)!=0u){udViolation(${UNIFORM_DETAIL_POISON_VIOLATION}u);}`}
}
fn udWatchStore(${sites?"site:u32":""}){
 let bits=${R.store}u;${mark}${domain.survey?"":`udViolation(${UNIFORM_DETAIL_RING_STORE_VIOLATION}u);`}
}
fn udWatchHomes(q:vec3i,a:vec4f,h:vec4f${site}){
 var bits=${R.ran}u;if(udRingLoad(q)&&any(bitcast<vec4u>(a)!=bitcast<vec4u>(h))){bits|=${R.homes}u;}
 ${mark}${domain.survey?"":`if((bits&${R.homes}u)!=0u){udViolation(${UNIFORM_DETAIL_HOME_VIOLATION}u);}`}
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
${domain.compactFaces?`fn udFace(q:vec3i,b:u32)->vec3i{let c=q>>vec3u(2u*b);let j=${j};return vec3i(c.xy,c.z*i32(1u+3u*b)+(j.x+2*j.y+3*j.z)*i32(b));}\n`:""}${ringCode}`;
 // fold: each accessor is written in both store forms and picks one by the
 // pipeline constant udWindow (UniformDetailStorage.pipeline). _w is the
 // window's (above). _d is the lattice-sized store's: the raw address with
 // no clamp and no mask, a base told by its x extent, so a pipeline built
 // with udWindow false runs the kernel a store that never wraps needs.
 const one=(name:string,kind:UniformDetailClass,type:string,direct:boolean,sfx:string):string=>{
  let out="";
  const storage=type.startsWith("texture_storage"),level=storage?"":",0";
  const lattice=`(${dims})`,D=kind==="vertex"?`(${dims}+vec3u(1u))`:lattice,top=`vec3i(${D})-vec3i(1)`;
  // The field's base block (an atlas-only name binds a face field).
  const block=kind==="cell"?`(${dims}>>vec3u(2u))`:kind==="vertex"?`((${dims}>>vec3u(2u))+vec3u(1u))`:domain.compactFaces?`((${dims}>>vec3u(2u))*vec3u(1u,1u,4u))`:`(${dims}>>vec3u(1u))`;
  const based_=(t:string)=>direct?`u32(textureDimensions(${t}).x<${D}.x)`:`u32(all(textureDimensions(${t})==${block}))`,extent=based_(name);
  const mode=direct?`let b=${extent};`:`let d=textureDimensions(${name});let b=u32(all(d==${block}));let m=udMask(d,b,${lattice});`;
  const canonical=(v:string)=>kind==="face"?`udFaceCanonical(${v})`:`udAligned(${v})`;
  const clamped=(v:string)=>`clamp(${v},vec3i(0),${top})`;
  // The bound texture's texel of logical v, and an h store's (name: bound or twin).
  // An out-of-lattice load as the device clamps it (unsigned: a negative coordinate reads the top edge), made here where a mask follows.
  const fit=(v:string)=>direct?v:`vec3i(min(vec3u(${v}),vec3u(${top})))`,mask=direct?"":"&m";
  const at=(v:string)=>(kind!=="face"?`(${fit(v)}>>vec3u(2u*b))`:domain.compactFaces?`udFace(${clamped(v)},b)`:`(${fit(v)}>>vec3u(b))`)+mask;
  const stored=(t:string,v:string)=>direct?v:`${fit(v)}&udMask(textureDimensions(${t}),0u,${lattice})`;
  // A base block's texel of v, by b (the raw load: no window, the device's clamp).
  const raw=(v:string)=>kind!=="face"?`${v}>>vec3u(2u*b)`:domain.compactFaces?`udFace(${v},b)`:`${v}>>vec3u(b)`;
  // The bound texture's texel of a store at v.
  const put=(v:string)=>direct?kind==="atlas"?v:raw(v):`udPut(${kind==="atlas"?v:raw(v)},${v},${top},m)`;
  const home=(v:string)=>kind!=="face"?`${v}>>vec3u(2u)`:domain.compactFaces?`udFace(${v},1u)`:`${v}>>vec3u(1u)`;
  // A stored field's twin is its other home (the base beside an h texture,
  // the h texture beside a base, a 1³ stand-in while there is one home):
  // a canonical v's texel there, by the bound texture's b.
  const twin=twins?.get(name),other=(v:string)=>`(${kind!=="face"?`(${v}>>vec3u(2u*(1u-b)))`:domain.compactFaces?`udFace(${v},1u-b)`:`(${v}>>vec3u(1u-b))`}${direct?"":`&udMask(textureDimensions(${twin}),1u-b,${lattice})`})`;
  if(!storage||type.includes("read_write")){
   const check=domain.checked&&!deposit&&kind!=="atlas"?`if(b!=0u&&!${canonical(`clamp(q,vec3i(0),${top})`)}){udViolation(${UNIFORM_DETAIL_LOAD_VIOLATION[kind]}u);}\n `:"";
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
 ${domain.checked?`if(!${canonical(c)}){udViolation(${kind==="atlas"?UNIFORM_DETAIL_VIOLATION.atlas:UNIFORM_DETAIL_LOAD_VIOLATION[kind as "cell"]}u);}\n `:""}${both?`let v=textureLoad(${twin},${raw(kind==="face"&&domain.compactFaces?c:"q")},0);if(b!=0u&&${extent}==0u){udWatchHomes(${c},v,textureLoad(${name},${stored(name,"q")},0)${pass});}return v;`:`return textureLoad(${twin},${raw(kind==="face"&&domain.compactFaces?c:"q")},0);`}
}
`;
  }
  if(hOnly?.has(name))out+=`fn udLoadH_${name}${sfx}(q:vec3i${site})->vec4f{
 ${domain.checked?`if(${extent}!=0u){udViolation(${UNIFORM_DETAIL_H_VIOLATION}u);}\n `:""}${watch&&kind!=="atlas"?`let v=textureLoad(${name},${stored(name,"q")}${level});udWatch(q,${top},v.x${pass});return v;`:`return textureLoad(${name},${stored(name,"q")}${level});`}
}
`;
  if(storage){
   const fail=(bits:number)=>deposit?"return;":`udViolation(${bits}u);return;`;
   const inside=`all(q>=vec3i(0))&&all(q<=${top})`;
   // The ring. On a base: the h twin takes the texel inside the
   // ring. On an h texture: a canonical texel goes to the base twin, and to
   // the h texture only inside the ring; any other texel is the h texture's.
   if(twin&&kind!=="atlas"&&(ringed||domain.untwinned)){
    // The lattice-sized store's form stores an h texture's canonical texel
    // to both homes untested: the test drops a store only outside the
    // ring, where the h texel exists and is never loaded, and costs a
    // topology read at every canonical store inside it. (A base's h twin
    // keeps the test: there it saves the store.) Not under the QA fills.
    const gate=ringed&&!(direct&&!watch&&!domain.poison);
    out+=`fn udStore_${name}${sfx}(q:vec3i,value:vec4f${site}){
 ${mode}
 if(b!=0u){
  if(any(q<vec3i(0))||any(q>${top})){return;}if(!${canonical("q")}){${fail(UNIFORM_DETAIL_VIOLATION[kind])}}
  textureStore(${name},${put("q")},value);
  if(textureDimensions(${twin}).x>1u${ringed?"&&udRingStore(q)":""}){textureStore(${twin},${other("q")},value);}
  return;
 }
 ${domain.untwinned?"":watch&&ringed
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
 for(const [name,{kind,type}] of kinds){
  if(!fold){out+=one(name,kind,type,false,"");continue;}
  const wide=one(name,kind,type,false,"_w");
  out+=wide+one(name,kind,type,true,"_d");
  for(const [,fn,store] of wide.matchAll(/^fn (ud(?:Load[GC4H]?|(Store))_\w+)_w\(/gm)){
   out+=store?`fn ${fn}(q:vec3i,value:vec4f${site}){if(udWindow){${fn}_w(q,value${pass});}else{${fn}_d(q,value${pass});}}\n`
    :`fn ${fn}(q:vec3i${fn.startsWith("udLoadG_")?"":site})->vec4f{if(udWindow){return ${fn}_w(q${fn.startsWith("udLoadG_")?"":pass});}return ${fn}_d(q${fn.startsWith("udLoadG_")?"":pass});}\n`;
  }
 }
 return out;
}
type DetailKinds=ReadonlyMap<string,{kind:UniformDetailClass;type:string}>;
/** Haloed patches: the other resident boxes that can hold texel `p` of home
 * patch n: one direction per axis, so the non-empty submasks of the near
 * axes (one for a texel near a face, three near an edge, seven at a corner).
 * A loop, not blocks: Metal flattens every call, and this text is paid at
 * every site. */
function detailPatchNearWGSL(l:UniformDetailLayout,k:UniformDetailClass,p:string,visit:string):string{
 const P=l.patchEdge,r=haloOf(l,k);if(!r&&k!=="vertex")return "";
 return ` let l=${p}-(n<<vec3u(${Math.log2(P)}u));let s=select(vec3i(0),vec3i(-1),l<${k==="vertex"?"=":""}vec3i(${r}))+select(vec3i(0),vec3i(1),l>=vec3i(${P-r}));
 let near=u32(s.x!=0)|(u32(s.y!=0)<<1u)|(u32(s.z!=0)<<2u);
 for(var i=near;i!=0u;i=(i-1u)&near){
  let m=n+s*vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let f=udWord(m);if(f!=0u){${visit}}
 }
`;
}
/** Haloed patches: address helpers. The directory is the only table read.
 * udFind_<class>(p): the texel of logical p with no home in hand; w = 0 the
 * base (canonical: current by write-through), 1 a resident box (its home
 * patch's, else for a vertex or atlas texel a neighbour's halo), -1 no texel
 * (xyz: the base address). */
function detailPatchAddressWGSL(l:UniformDetailLayout,table:string|((index:string)=>string),start:string,tiles:string,support:string|undefined,atomic:boolean):string{
 const word=typeof table==="string"?(index:string)=>`${table}[${index}]`:table;
 const P=l.patchEdge,shift=Math.log2(P),g=l.patchGrid;
 const inBox=(k:UniformDetailClass)=>{const c=l.classes[k],r=haloOf(l,k);return `fn udIn_${k}(n:vec3i,e:u32,p:vec3i)->vec3i{return udSlot(e)*${c.spacing}+vec3i(${r},${r},${c.atlasZ+r})+p-(n<<vec3u(${shift}u));}\n`;};
 // udOut: p's texel outside the canonical set, from the directory: its home patch's box, else a neighbour's halo
 // (the seam ring for vertex and atlas; for cell and face the texels an unhomed h item reads across its patch's low side).
 const find=(k:UniformDetailClass)=>`fn udOut_${k}(p:vec3i)->vec4i{
 let n=udPatchOf(p);let e=udHomeWord(n);
 if(e!=0u){return vec4i(udIn_${k}(n,e,p),1);}
${detailPatchNearWGSL(l,k,"p",`return vec4i(udIn_${k}(m,f,p),1);`)} return vec4i(udBase_${k}(p),-1);
}
fn udFind_${k}(p:vec3i)->vec4i{${k==="atlas"?"":`if(udCanon_${k}(p)){return vec4i(udBase_${k}(p),0);}`}return udOut_${k}(p);}
`;
 return /* wgsl */`
fn udViolation(bits:u32){${!support?"":atomic?`atomicOr(&${support},bits);`:`${support}=${support}|bits;`}}
fn udWord(n:vec3i)->u32{
 if(any(n<vec3i(0))||any(n>=vec3i(${g.join(",")}))){return 0u;}
 let m=vec3u(n);return ${word(`${start}+m.x+${g[0]}u*(m.y+${g[1]}u*m.z)`)};
}
fn udHomeWord(n:vec3i)->u32{let m=vec3u(n);return ${word(`${start}+m.x+${g[0]}u*(m.y+${g[1]}u*m.z)`)};}
fn udSlot(e:u32)->vec3i{return vec3i(vec3u(e&1023u,(e>>10u)&1023u,(e>>20u)&1023u));}
fn udPatchOf(p:vec3i)->vec3i{return min(p>>vec3u(2u),vec3i(${tiles})-vec3i(1))>>vec3u(${shift-2}u);}
fn udCanon_cell(p:vec3i)->bool{return all((p&vec3i(3))==vec3i(0));}
fn udCanon_vertex(p:vec3i)->bool{return all((p&vec3i(3))==vec3i(0));}
fn udCanon_face(p:vec3i)->bool{let l=p&vec3i(3);let three=vec3i(l==vec3i(3));return all((l==vec3i(0))|(l==vec3i(3)))&&three.x+three.y+three.z<=1;}
fn udCanon_atlas(p:vec3i)->bool{return false;}
fn udBase_cell(p:vec3i)->vec3i{return p>>vec3u(2u);}
fn udBase_vertex(p:vec3i)->vec3i{return p>>vec3u(2u);}
fn udBase_face(p:vec3i)->vec3i{return p>>vec3u(1u);}
fn udBase_atlas(p:vec3i)->vec3i{return vec3i(0);}
${CLASSES.map(inBox).join("")}${CLASSES.map(find).join("")}`;
}
/** Haloed patches: the write-through store of one field. Every resident box
 * that holds q, then the base when canonical; returns whether any did. */
function detailPatchStoreAllWGSL(l:UniformDetailLayout,name:string,kind:UniformDetailClass):string{
 return `fn udStoreAll_${name}(q:vec3i,value:vec4f)->bool{
 let n=udPatchOf(q);var stored=false;
 let e=udHomeWord(n);if(e!=0u){textureStore(${name},udIn_${kind}(n,e,q),value);stored=true;}
${detailPatchNearWGSL(l,kind,"q",`textureStore(${name},udIn_${kind}(m,f,q),value);stored=true;`)} ${kind==="atlas"?"":`if(udCanon_${kind}(q)){textureStore(${name},udBase_${kind}(q),value);stored=true;}\n `}return stored;
}
`;
}
/** Haloed patches: the dense kernels' accessors (no home): a load finds its
 * texel from the directory and, where a tile has none, takes what packed
 * storage loads there (the tile's value, its parity face, the corners'
 * interpolant); a store that lands nowhere is dropped. */
function detailPatchDenseWGSL(l:UniformDetailLayout,kinds:DetailKinds,dims:string):string{
 let out="";
 for(const [name,{kind,type}] of kinds){
  const storage=type.startsWith("texture_storage"),level=storage?"":",0",D=kind==="vertex"?`(${dims}+vec3u(1u))`:`(${dims})`;
  if(!storage||type.includes("read_write")){
   const none=kind==="atlas"?"return vec4f(0.0);"
    :kind==="cell"?`return textureLoad(${name},udBase_cell(p)${level});`
    :kind==="face"?`return textureLoad(${name},2*(p>>vec3u(2u))+vec3i((p&vec3i(3))==vec3i(3))${level});`
    :`let home=min(p>>vec3u(2u),vec3i(${dims})/4-vec3i(1));let t=vec3f(p-4*home)/4.0;var values:array<f32,8>;
 for(var k=0;k<8;k++){
  let corner=vec3i(k%2,(k/2)%2,k/4);let w=select(vec3f(1.0)-t,t,corner!=vec3i(0));let weight=w.x*w.y*w.z;
  if(weight>0.0){values[k]=weight*textureLoad(${name},udFind_vertex(4*(home+corner)).xyz${level}).x;}
 }
 return vec4f(((values[0]+values[5])+(values[1]+values[4]))+((values[2]+values[7])+(values[3]+values[6])),0.0,0.0,1.0);`;
   out+=`fn udLoad_${name}(q:vec3i)->vec4f{
 let p=clamp(q,vec3i(0),vec3i(${D})-vec3i(1));let a=udFind_${kind}(p);
 if(a.w>=0){return textureLoad(${name},a.xyz${level});}
 ${none}
}
`;
  }
  if(storage)out+=detailPatchStoreAllWGSL(l,name,kind)+`fn udStore_${name}(q:vec3i,value:vec4f){
 if(any(q<vec3i(0))||any(q>=vec3i(${D}))){return;}
 udStoreAll_${name}(q,value);
}
`;
 }
 return out;
}
function fnv(text:string):string{let h=0x811c9dc5;for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,0x01000193);}return `${(h>>>0).toString(16)}.${text.length}`;}
/** A work item's owner comes from one of these (by job, lane or tile); an
 * owner looked up by position is a neighbour, never the home. */
const PATCH_OWNER=/fn (\w+)\(([^)]*)\)\s*->\s*(\w*)UMOwner\s*\{/g,PATCH_NEIGHBOUR=/(OwnerAt|VertexAuthority|StageOwner)$/;
/** Haloed patches: the mixed kernels' rewrite (UniformDetailPatch). The
 * work item's home is resolved once, where it takes its owner; with none
 * resolved the box is empty and every load is a base or directory load. */
function detailPatchShader(code:string,l:UniformDetailLayout,pf:string,kinds:DetailKinds,atomic:boolean):string{
 const patch=l.patch!,hash=fnv(code),module=[...code.matchAll(/@compute[^{]*?fn\s+(\w+)/g)].map(m=>m[1]!).slice(0,4).join(","),variants=new Set<string>();
 let ordinal=0;
 const calls=rewriteCalls(code,new Set(kinds.keys()),undefined,(field,store)=>{
  const key=`${hash}:${ordinal++}`,kind=kinds.get(field)!.kind;
  let index=patchSiteIndex.get(key);
  const form=patch.mode==="profile"?"P":store?"":patch.mode==="trust"&&!!patchProfile&&uniformDetailSiteClean(patchProfile.words.get(key),kind,patchProfile.halo,patch.halo)?"G":"F";
  if(index===undefined){
   index=patchSites.length;if(patch.mode==="profile"&&index>=PATCH_SITE_WORDS)throw new Error("Uniform detail patch profile: more sites than profile words");
   patchSites.push({key,module,field,kind,store,form});patchSiteIndex.set(key,index);
  }
  variants.add(`${store?"S":"L"}${form}_${field}`);
  return {suffix:form,argument:patch.mode==="profile"?`,${index}u`:""};
 });
 // Each owner source becomes <name>_udRaw and a wrapper that resolves the home; an owner source calling another calls it raw.
 const owners=[...calls.matchAll(PATCH_OWNER)].filter(m=>!PATCH_NEIGHBOUR.test(m[1]!)).map(m=>({name:m[1]!,args:m[2]!,p:m[3]!,at:m.index!,open:m.index!+m[0].length}));
 const inner=new RegExp(`\\b(${owners.map(o=>o.name).join("|")})\\(`,"g");
 let resolved="",cursor=0;
 for(const o of owners){
  let end=o.open,depth=1;for(;end<calls.length&&depth;end++){const c=calls[end];if(c==="{")depth++;else if(c==="}")depth--;}
  if(depth)throw new Error(`Uniform detail patch rewrite cannot parse ${o.name}`);
  resolved+=calls.slice(cursor,o.at)+`fn ${o.name}_udRaw(${o.args})->${o.p}UMOwner{`+calls.slice(o.open,end).replace(inner,"$1_udRaw(")
   +`\nfn ${o.name}(${o.args})->${o.p}UMOwner{let home=${o.name}_udRaw(${o.args.split(",").map(a=>a.split(":")[0]!.trim()).join(",")});if(home.width!=0u){udHome(${o.p}umTileCoord(home.tile));}return home;}`;
  cursor=end;
 }
 resolved+=calls.slice(cursor);
 const P=l.patchEdge,shift=Math.log2(P),support=`${pf}umSupport[${pf}UM_DETAIL_VIOLATION]`,S=UNIFORM_DETAIL_SITE;
 const K=CLASSES.filter(k=>[...kinds.values()].some(v=>v.kind===k));
 const site=`${pf}umSupport[${pf}UM_DETAIL_VIOLATION+4u+site]`;
 let out=resolved+detailPatchAddressWGSL(l,`${pf}umTopology`,`${pf}UM_DETAIL+${HEADER}u`,`${pf}UM_T`,support,atomic)+/* wgsl */`
${K.map(k=>`var<private> udL_${k}:vec3i;var<private> udH_${k}:vec3i;var<private> udO_${k}:vec3i;`).join("\n")}
fn udHome(tile:vec3u){
 let n=vec3i(tile>>vec3u(${shift-2}u));let e=udHomeWord(n);
 if(e==0u){${K.map(k=>`udH_${k}=udL_${k};`).join("")}return;}
 let x=n<<vec3u(${shift}u);let slot=udSlot(e);
${K.map(k=>{const c=l.classes[k],r=haloOf(l,k);return ` udL_${k}=x-vec3i(${r});udH_${k}=x+vec3i(${P+r+(k==="vertex"?1:0)});udO_${k}=slot*${c.spacing}+vec3i(${r},${r},${c.atlasZ+r})-x;`;}).join("\n")}
}
${patch.mode!=="profile"?"":`fn udMark(site:u32,bits:u32){${atomic?`if((atomicLoad(&${site})&bits)!=bits){atomicOr(&${site},bits);}`:`if((${site}&bits)!=bits){${site}=${site}|bits;}`}}\n`}`;
 for(const [name,{kind,type}] of kinds){
  const storage=type.startsWith("texture_storage"),level=storage?"":",0",h=kind,r=haloOf(l,kind),v=kind==="vertex"?1:0;
  const D=kind==="vertex"?`(${pf}UM_D+vec3u(1u))`:`(${pf}UM_D)`,clamp=`let p=clamp(q,vec3i(0),vec3i(${D})-vec3i(1));`;
  const inside=`all((p>=udL_${h})&(p<udH_${h}))`,home=`textureLoad(${name},udO_${h}+p${level})`;
  const fault=patch.checked?`udViolation(${kind==="atlas"?UNIFORM_DETAIL_VIOLATION.atlas:UNIFORM_DETAIL_LOAD_VIOLATION[kind]}u);`:"";
  // G: one load, the address a select; valid in the box or on a canonical texel.
  if(variants.has(`LG_${name}`))out+=kind==="atlas"?`fn udLoadG_${name}(q:vec3i)->vec4f{${clamp}return select(vec4f(0.0),${home},${inside});}\n`
   :`fn udLoadG_${name}(q:vec3i)->vec4f{${clamp}let n=${inside};${fault?`if(!n&&!udCanon_${kind}(p)){${fault}}`:""}return textureLoad(${name},select(udBase_${kind}(p),udO_${h}+p,n)${level});}\n`;
  // F: G's address, then the directory for a texel neither in the box nor canonical; one load.
  const none=kind==="atlas"?"if(a.w<0){return vec4f(0.0);}":"";
  if(variants.has(`LF_${name}`))out+=`fn udLoadF_${name}(q:vec3i)->vec4f{
 ${clamp}let n=${inside};var at=select(udBase_${kind}(p),udO_${h}+p,n);
 if(!n${kind==="atlas"?"":`&&!udCanon_${kind}(p)`}){let a=udOut_${kind}(p);${fault?`if(a.w<0){${fault}}`:""}${none}at=a.xyz;}
 return textureLoad(${name},at${level});
}
`;
  if(variants.has(`LP_${name}`))out+=`fn udLoadP_${name}(q:vec3i,site:u32)->vec4f{
 ${clamp}let patched=udH_${h}.x>udL_${h}.x;var a:vec4i;var bits:u32;
 if(${inside}){let d=max(udL_${h}+vec3i(${r})-p,p-udH_${h}+vec3i(${r+1}));let deep=max(d.x,max(d.y,d.z));
  bits=select(select(select(select(${S.halo|S.halo4}u,${S.halo|S.halo3}u,deep==3),${S.halo|S.halo2}u,deep==2),${S.halo}u,deep==1),${S.interior}u,deep<=0);a=vec4i(udO_${h}+p,1);}
 ${kind==="atlas"?"":`else if(udCanon_${kind}(p)){bits=select(${S.baseBaseHome}u,${S.basePatchHome}u,patched);a=vec4i(udBase_${kind}(p),0);}\n `}else{a=udOut_${kind}(p);
  if(patched){let d=max(max(udL_${h}-p,p-udH_${h}+vec3i(1)),vec3i(0));bits=select(${S.nowhere}u,${S.beyond}u,a.w>0)<<(u32(clamp(max(d.x,max(d.y,d.z)),1,8))-1u);}
  else{bits=select(${S.seamNowhere}u,${S.seam}u,a.w>0);}}
 udMark(site,bits);${none}return textureLoad(${name},a.xyz${level});
}
`;
  if(!storage)continue;
  const deep=`all((q>=udL_${h}+vec3i(${2*r+v}))&(q<udH_${h}-vec3i(${2*r+v})))`,range=`if(any(q<vec3i(0))||any(q>=vec3i(${D}))){return;}`;
  const single=`textureStore(${name},udO_${h}+q,value);${kind==="atlas"?"":`if(udCanon_${kind}(q)){textureStore(${name},udBase_${kind}(q),value);}`}`;
  const lost=`udViolation(${UNIFORM_DETAIL_VIOLATION[kind]}u);`;
  out+=detailPatchStoreAllWGSL(l,name,kind);
  if(variants.has(`S_${name}`))out+=`fn udStore_${name}(q:vec3i,value:vec4f){
 ${range}
 if(${deep}){${single}return;}
 if(!udStoreAll_${name}(q,value)){${lost}}
}
`;
  if(variants.has(`SP_${name}`))out+=`fn udStoreP_${name}(q:vec3i,value:vec4f,site:u32){
 ${range}
 if(${deep}){${single}udMark(site,${S.storeFast}u);return;}
 if(udStoreAll_${name}(q,value)){udMark(site,${S.storeSlow}u);}else{udMark(site,${S.storeNowhere}u);${patch.checked?lost:""}}
}
`;
 }
 return out;
}
/** Rewrite textureLoad/textureStore calls of `names` into accessor calls;
 * coordinates become vec3i, a load's level must be 0. hOnly (the domain
 * placement): a load marked UNIFORM_DETAIL_H_LOAD becomes udLoadH_<name>
 * and its name joins the set, one marked UNIFORM_DETAIL_4H_LOAD
 * udLoad4_<name> and joins `based`; elsewhere the marks are dropped.
 * site (haloed patches): the accessor variant and trailing argument of each
 * call, asked in source order of the calls' heads. */
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
 if(!layout||layout.placement==="identity")return code;
 if(layout.domain){
  // Survey and poison runs number every load and store but the canonical (base) loads.
  const domain=layout.domain;
  code=code.replaceAll(UNIFORM_DETAIL_RING_4H_LOAD,UNIFORM_DETAIL_4H_LOAD);
  const sites=!!(domain.checked&&(domain.survey||domain.poison)),hash=sites?fnv(code):"",module=sites?[...code.matchAll(/@compute[^{]*?fn\s+(\w+)/g)].map(m=>m[1]!).slice(0,4).join(","):"";
  let ordinal=0;
  const hOnly=new Set<string>(),based=new Set<string>(),either=new Set<string>(),calls=rewriteCalls(code,new Set(kinds.keys()),hOnly,!sites?undefined:(field,store,mark,text,scope)=>{
   const key=`${hash}:${ordinal++}`;let index=patchSiteIndex.get(key);
   if(index===undefined){
    index=patchSites.length;if(index>=PATCH_SITE_WORDS)throw new Error("Uniform detail survey: more sites than site words");
    patchSites.push({key,module,field,kind:kinds.get(field)!.kind,store,form:`${mark?`${mark} `:""}${scope}: ${text}`});patchSiteIndex.set(key,index);
   }
   return {suffix:"",argument:`,${index}u`};
  },based,undefined,either);
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
  // udWindow: whether the h store may be a wrapped window. A pipeline
  // constant, so each store form is the kernel it needs with no test left in
  // it (false: the direct accessors). Every entry point names it: a
  // pipeline may set it whatever its entry loads.
  const entries=calls.replace(/@compute\b[^{]*\{/g,head=>`${head}_=udWindow;`);
  return entries+"\noverride udWindow:bool=true;\n"+declarations+detailDomainWGSL(kinds,layout.domain,`${pf}UM_D`,`${pf}umSupport[${pf}UM_DETAIL_VIOLATION]`,atomic,false,hOnly,twins,based,{pf,target,sites},either,new Set([...kinds.keys()].filter(name=>calls.includes(`udLoadG_${name}(`))),true);
 }
 if(layout.patch)return detailPatchShader(code,layout,pf,kinds,atomic);
 if(layout.local)return detailLocalShader(rewriteCalls(code,new Set(kinds.keys())),layout,layout.local,pf,kinds);
 let helpers=detailAddressWGSL(layout,`${pf}umTopology`,`${pf}UM_DETAIL+${HEADER}u`,`${pf}UM_T`,`${pf}umSupport[${pf}UM_DETAIL_VIOLATION]`,atomic);
 for(const [name,{kind,type}] of kinds)helpers+=detailAccessorWGSL(name,kind,type,kind==="vertex"?`${pf}UM_D+vec3u(1u)`:`${pf}UM_D`,`${pf}umCounts.w`);
 return rewriteCalls(code,new Set(kinds.keys()))+helpers;
}
/** A dense reference shader (the t=0 authority and publication) through the
 * same accessors: `classes` names its field bindings, the directory is the
 * head of the residency state, a uniform at `group` (those kernels' own
 * layout has no storage binding to spare). A store outside the canonical texels is
 * dropped, so a lattice launch deposits exactly the texels that are state. */
export function uniformDetailDenseShader(code:string,storage:UniformDetailStorage,classes:Readonly<Record<string,UniformDetailClass>>,group:number):string{
 const l=storage.layout;if(l.placement==="identity")return code;
 const kinds=new Map<string,{kind:UniformDetailClass;type:string}>();
 for(const [,name,type] of code.matchAll(/var\s+(\w+)\s*:\s*(texture(?:_storage)?_3d<[^>]+>)/g)){const kind=classes[name!];if(kind)kinds.set(name!,{kind,type:type!});}
 for(const name of Object.keys(classes))if(!kinds.has(name))throw new Error(`Uniform detail dense rewrite: no 3D texture ${name}`);
 if(/fn\s+\w+\s*\([^)]*:\s*texture/.test(code))throw new Error("Uniform detail dense rewrite: a texture-typed function parameter bypasses the field accessors");
 for(const name of kinds.keys())if(new RegExp(`texture(?:Dimensions|SampleLevel)\\(\\s*${name}\\b`).test(code))throw new Error(`Uniform detail dense rewrite: ${name} is read past the accessors`);
 const vec=(d:readonly number[])=>`vec3u(${d.map(n=>`${n}u`).join(",")})`,dims=vec(l.dims);
 // Domain: the bound texture says base or h, so the kernels read no directory (the group stays bound, unused).
 if(l.domain)return rewriteCalls(code,new Set(kinds.keys()))+detailDomainWGSL(kinds,l.domain,dims,undefined,false,true);
 if(l.patch)return rewriteCalls(code,new Set(kinds.keys()))+`\n@group(${group}) @binding(0) var<uniform> udState:array<vec4u,${Math.ceil((D.words+l.patches)/4)}>;\n`
  +detailPatchAddressWGSL(l,i=>`udState[(${i})>>2u][(${i})&3u]`,`${D.words}u`,vec(l.tiles),undefined,false)+detailPatchDenseWGSL(l,kinds,dims);
 let helpers=`\n@group(${group}) @binding(0) var<uniform> udState:array<vec4u,${Math.ceil((D.words+l.patches)/4)}>;\n`+detailAddressWGSL(l,i=>`udState[(${i})>>2u][(${i})&3u]`,`${D.words}u`,vec(l.tiles),undefined,false);
 for(const [name,{kind,type}] of kinds)helpers+=detailAccessorWGSL(name,kind,type,kind==="vertex"?`${dims}+vec3u(1u)`:dims,"8u",true);
 return rewriteCalls(code,new Set(kinds.keys()))+helpers;
}
/** createShaderModule through the central rewrite. */
export function uniformDetailModule(device:GPUDevice,descriptor:GPUShaderModuleDescriptor):GPUShaderModule{
 const code=uniformDetailShader(descriptor.code,device),module=device.createShaderModule({...descriptor,code});
 if(code!==descriptor.code&&code.includes("\noverride udWindow:bool=true;\n"))folded.add(module);
 return module;
}
/** Modules whose accessors pick a store form by udWindow. */
const folded=new WeakSet<GPUShaderModule>();
/** A pipeline of such a module in the h store's forms: the key a stage
 * holds, and the variants built so far. general (udWindow true) is exact for
 * any store; direct (false) only while the store spans the lattice. */
class StoreTwin{
 general?:GPUComputePipeline;direct?:GPUComputePipeline;
 constructor(readonly storage:UniformDetailStorage,readonly descriptor:GPUComputePipelineDescriptor){}
 /** What a stage holds in a pipeline's place; a pass refuses to bind it. */
 get key():GPUComputePipeline{return this as unknown as GPUComputePipeline;}
}
const storageOf=new WeakMap<UniformMixedOwnership,UniformDetailStorage>();
/** createComputePipelineAsync for a stage of `ownership`. A module made by
 * uniformDetailModule under the domain placement compiles per store form
 * (UniformDetailStorage.pipeline): what resolves is then a key for
 * uniformDetailPick, never a pipeline to bind. Any other module compiles as
 * it is. */
export function uniformDetailPipeline(device:GPUDevice,ownership:UniformMixedOwnership,descriptor:GPUComputePipelineDescriptor):Promise<GPUComputePipeline>{
 if(!folded.has(descriptor.compute.module))return device.createComputePipelineAsync(descriptor);
 const storage=storageOf.get(ownership);
 if(!storage)throw new Error(`Uniform detail pipeline ${descriptor.compute.entryPoint}: its ownership is attached to no detail storage`);
 return storage.pipeline(descriptor);
}
/** At a dispatch site: the pipeline to bind, for the form the h store holds
 * this frame. A pipeline that is not a key is returned as it is. */
export function uniformDetailPick(pipeline:GPUComputePipeline):GPUComputePipeline{
 return pipeline instanceof StoreTwin?pipeline.storage.pick(pipeline):pipeline;
}
