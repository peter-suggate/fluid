import type { UniformClothGridFields, UniformClothStep } from './uniform-cloth-grid-coupling';

export interface ClothPorousCapacityCell {readonly p:readonly [number,number,number];readonly solidFraction:number;readonly boundFraction:number}
export interface ClothPorousCapacity {readonly cells:readonly ClothPorousCapacityCell[];readonly cellVolume_m3:number;readonly solidVolume_m3:number;readonly boundVolume_m3:number;readonly nodePositions_m:Float64Array;readonly nodeWaterVolume_m3:Float64Array;readonly nodeSolidVolume_m3:Float64Array}
/** Sparse cell capacity. The hard solid aperture remains separate: native
 * free water is fill(phi)*hardOpen*(1-fiberFraction-boundWaterFraction).
 * Header count is uint bits; each sorted z,y,x record has coordinate bits
 * and (fiber,bound,0,0). Empty/missing coordinates have unit free capacity. */
export function uniformPorousCapacityWGSL(group:number,binding:number):string{return /* wgsl */`
@group(${group}) @binding(${binding}) var umPorousCapacity:texture_2d<f32>;
fn umPorousWord(i:u32)->vec4f{let width=textureDimensions(umPorousCapacity).x;return textureLoad(umPorousCapacity,vec2i(i32(i%width),i32(i/width)),0);}
fn umPorousLess(a:vec3i,b:vec3i)->bool{return a.z<b.z||(a.z==b.z&&(a.y<b.y||(a.y==b.y&&a.x<b.x)));}
fn umPorousFractions(local:vec3i)->vec2f{
 let p=local+32*umFramePage();
 let count=bitcast<u32>(umPorousWord(0u).x);var low=0u;var high=count;
 loop{if(low>=high){break;}let mid=(low+high)/2u;if(umPorousLess(bitcast<vec4i>(umPorousWord(1u+2u*mid)).xyz,p)){low=mid+1u;}else{high=mid;}}
 if(low<count&&all(bitcast<vec4i>(umPorousWord(1u+2u*low)).xyz==p)){return umPorousWord(2u+2u*low).xy;}return vec2f(0.0);
}
fn umPorousFreeCapacity(p:vec3i)->f32{return max(0.0,1.0-dot(umPorousFractions(p),vec2f(1.0)));}
fn umPorousTileCapacity(origin:vec3i)->f32{
 if(bitcast<u32>(umPorousWord(0u).x)==0u){return 1.0;}var sum=0.0;
 for(var i=0u;i<64u;i++){sum+=umPorousFreeCapacity(origin+vec3i(i32(i&3u),i32((i>>2u)&3u),i32(i>>4u)));}return sum/64.0;
}
`;}

type CapacityLattice=Pick<UniformClothGridFields,'origin_m'|'cellSize_m'>;
const apertureOf=new WeakMap<object,ReadonlyMap<string,number>>();
export function setClothHardApertures(fields:CapacityLattice,values:ReadonlyMap<string,number>):void{apertureOf.set(fields,values);}
export function clothHardAperture(fields:CapacityLattice,p:readonly number[]):number{
 const aperture=apertureOf.get(fields);if(!aperture)return 1;
 const value=aperture.get(p.join(','));if(value===undefined)throw new Error(`Cloth hard aperture was not sampled at ${p}`);return value;
}
export interface ClothCellWeight {p:[number,number,number];weight:number;porosityWeight:number;hardOpen:number}
/** One normalized physical basis for storage, momentum and adjoint forces.
 * porosityWeight is relative to the cell's open volume; weight is relative
 * to full cell volume, so hardOpen*porosityWeight === weight. */
export function clothCellWeights(fields:CapacityLattice,position:ArrayLike<number>,raw=false):ClothCellWeight[]{
 const q=fields.cellSize_m.map((h,k)=>(position[k]!-fields.origin_m[k]!)/h-.5),base=q.map(Math.floor),fraction=q.map((v,k)=>v-base[k]!);
 if(q.some(x=>!Number.isFinite(x)))throw new Error('Invalid cloth capacity position');
 const cells:ClothCellWeight[]=[];let normalization=0;
 for(let c=0;c<8;c++){let w=1;const p=base.map((a,k)=>{const bit=(c>>k)&1;w*=bit?fraction[k]!:1-fraction[k]!;return a+bit;}) as [number,number,number];if(w<=0)continue;
  const hard=raw?1:clothHardAperture(fields,p);normalization+=w*hard;cells.push({p,weight:w*hard,porosityWeight:w,hardOpen:hard});
 }
 if(normalization<=1e-12)throw new Error('Cloth has no open local cell support; collision/admission must be resolved');
 return cells.filter(c=>c.hardOpen>0).map(c=>({...c,weight:c.weight/normalization,porosityWeight:c.porosityWeight/normalization}));
}

/** The source capture reservoir is the sum of the three staggered MAC
 * masses: each axis uses one half of each adjacent face's cell-average mass.
 * Weights sum to three, and are shared by available-volume measurement and
 * the local level-set receiver footprint (normalized there only for shape). */
export function clothMacCaptureReservoir(fields:CapacityLattice,position:ArrayLike<number>):readonly {p:[number,number,number];weight:number}[]{
 const cells=new Map<string,{p:[number,number,number];weight:number}>();
 for(const cell of clothCellWeights(fields,position))for(let axis=0;axis<3;axis++)for(const offset of [-1,0,1]){
  const p=[...cell.p] as [number,number,number];p[axis]!+=offset;const key=p.join(','),entry=cells.get(key)??{p,weight:0};entry.weight+=.25*cell.weight*(offset===0?2:1);cells.set(key,entry);
 }return [...cells.values()];
}

export function prepareClothPorousCapacity(fields:Pick<UniformClothGridFields,'origin_m'|'cellSize_m'>,step:Pick<UniformClothStep,'state'>):ClothPorousCapacity{
 const s=step.state,h=fields.cellSize_m,V=h.reduce((a,b)=>a*b,1),map=new Map<string,{p:[number,number,number];solidFraction:number;boundFraction:number}>();
 if(!(V>0))throw new Error('Invalid cloth capacity lattice');
 for(let v=0;v<s.solidVolume_m3.length;v++){
  const solid=s.solidVolume_m3[v]!,bound=s.waterVolume_m3[v]!;if(!Number.isFinite(solid)||!Number.isFinite(bound)||solid<0||bound<0)throw new Error('Invalid material capacity volume');
  if(solid+bound===0)continue;
  for(const {p,porosityWeight:weight} of clothCellWeights(fields,s.positions_m.subarray(3*v,3*v+3))){
   const key=p.join(','),cell=map.get(key)??{p,solidFraction:0,boundFraction:0};cell.solidFraction+=solid*weight/V;cell.boundFraction+=bound*weight/V;map.set(key,cell);
  }
 }
 const cells=[...map.values()].sort((a,b)=>a.p[2]-b.p[2]||a.p[1]-b.p[1]||a.p[0]-b.p[0]);
 for(const cell of cells)if(cell.solidFraction+cell.boundFraction>=1)throw new Error(`Cloth material fills cell ${cell.p}: refine the actual porous support`);
 return {cells,cellVolume_m3:V,nodePositions_m:s.positions_m.slice(),nodeWaterVolume_m3:s.waterVolume_m3.slice(),nodeSolidVolume_m3:s.solidVolume_m3.slice(),solidVolume_m3:s.solidVolume_m3.reduce((a,b)=>a+b,0),boundVolume_m3:s.waterVolume_m3.reduce((a,b)=>a+b,0)};
}

const tableOf=new WeakMap<GPUTexture,ClothPorousCapacity>();
export function clothPorousCapacityOf(texture:GPUTexture):ClothPorousCapacity|undefined{return tableOf.get(texture);}
export function uploadClothPorousCapacity(device:GPUDevice,table:ClothPorousCapacity):GPUTexture{
 const width=table.cells.length?Math.min(1024,device.limits.maxTextureDimension2D):1,height=Math.ceil((1+2*table.cells.length)/width);
 if(height>device.limits.maxTextureDimension2D)throw new Error('Porous material capacity exceeds the device texture limit');
 const data=new Float32Array(4*width*height),bits=new Int32Array(data.buffer);bits[0]=table.cells.length;
 table.cells.forEach((cell,i)=>{bits.set([...cell.p,0],4*(1+2*i));data.set([cell.solidFraction,cell.boundFraction,0,0],4*(2+2*i));});
 const texture=device.createTexture({label:'Cloth physical free capacity',size:[width,height],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
 device.queue.writeTexture({texture},data,{bytesPerRow:16*width,rowsPerImage:height},[width,height]);tableOf.set(texture,table);return texture;
}

/** Install an accepted material partition. Call before the first native volume
 * census; subsequent changes must pass the conservative capacity transaction. */
export function publishClothPorousCapacity(fields:UniformClothGridFields,step:Pick<UniformClothStep,'state'>):ClothPorousCapacity{
 const table=prepareClothPorousCapacity(fields,step),previous=fields.porousCapacity;
 fields.setPorousCapacity(uploadClothPorousCapacity(fields.device,table));
 fields.invalidateLocalWater();
 if(tableOf.has(previous))void fields.device.queue.onSubmittedWorkDone().then(()=>previous.destroy(),()=>previous.destroy());
 return table;
}

/** Detach the plugin-owned table. WebGPU retirement occurs after its last
 * submitted use; the solid module's identity texture is never destroyed. */
export function releaseClothPorousCapacity(fields:UniformClothGridFields):void{
 const previous=fields.porousCapacity;fields.setPorousCapacity();
 if(tableOf.has(previous))void fields.device.queue.onSubmittedWorkDone().then(()=>previous.destroy(),()=>previous.destroy());
}
