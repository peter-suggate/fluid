/**
 * Consumer ABI of the power-liquids coarse level-set sample directory.
 *
 * The water pipeline's global-fine classify and tetra passes sample coarse phi
 * wherever the Section-5 fine band has no valid brick, so they must speak this
 * directory exactly: its 32-byte header, its 32-byte entries, and the
 * binary-searched containing-leaf lookup that turns a domain position into one
 * of them. That is a description of a published buffer, not solver machinery,
 * and keeping it here is what lets a render pass read the publication without
 * constructing the power backend that fills it.
 *
 * The lookup fails closed in every ambiguous case — invalid publication state,
 * out-of-domain query, a key absent from the active set — and returns the
 * caller's own fallback rather than zero. A missing coarse sample that read as
 * zero would place a free surface exactly where the data ran out.
 */
import { OCTREE_COARSE_PHI_FLAG } from "./levelset-consumer-abi";

export const OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES = 32;
export const OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES = 32;

/**
 * Injects a containing-leaf sampler into the fine topology shader. A valid
 * sparse coarse publication has an explicit positive-air complement: compact
 * rows cover liquid/interface leaves, while a domain point absent from every
 * containing-leaf key is outside that active set. Invalid publications and
 * out-of-domain queries still fail closed.
 * The directory is rebuilt in the pressure row directory's deterministic
 * `(log2(size), morton)` order and marked valid only after every row passes.
 * It therefore needs one storage binding and never scans unused capacity.
 */
export function makeOctreePowerCoarseLevelSetSampleWGSL(binding = 9): string {
  if (!Number.isSafeInteger(binding) || binding < 0) throw new RangeError("Coarse-phi sample binding must be non-negative");
  return /* wgsl */ `
struct PowerCoarseSampleEntry { cellPlusOne:u32, size:u32, phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32, row:u32, physicalVolume:f32 }
struct PowerCoarseSampleDirectory { state:u32, generation:u32, rowCount:u32, maximumLeafSize:u32, dimensions:vec3u, physicalCellSize:f32, entries:array<PowerCoarseSampleEntry> }
@group(0) @binding(${binding}) var<storage,read> powerCoarseSamples:PowerCoarseSampleDirectory;
const POWER_COARSE_CELL_CENTERED:u32=0x08000000u;
fn powerCoarseMortonPart(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn powerCoarseMorton(cell:u32)->u32{let d=powerCoarseSamples.dimensions;let q=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));return powerCoarseMortonPart(q.x)|(powerCoarseMortonPart(q.y)<<1u)|(powerCoarseMortonPart(q.z)<<2u);}
fn powerCoarseLevel(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn powerCoarseLess(aLevel:u32,aMorton:u32,bLevel:u32,bMorton:u32)->bool{return aLevel<bLevel||(aLevel==bLevel&&aMorton<bMorton);}
fn powerCoarseLookup(cell:u32,size:u32)->u32{let count=min(powerCoarseSamples.rowCount,arrayLength(&powerCoarseSamples.entries));let wantedLevel=powerCoarseLevel(size);let wantedMorton=powerCoarseMorton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let entry=powerCoarseSamples.entries[middle];let entryMorton=powerCoarseMorton(entry.cellPlusOne-1u);if(powerCoarseLess(powerCoarseLevel(entry.size),entryMorton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let entry=powerCoarseSamples.entries[low];if(entry.cellPlusOne==cell+1u&&entry.size==size){return low;}}return 0xffffffffu;}
fn powerCoarseAdaptiveLookup(cell:u32,size:u32)->u32{let count=min(powerCoarseSamples.rowCount,arrayLength(&powerCoarseSamples.entries));var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let entry=powerCoarseSamples.entries[middle];let entryCell=entry.cellPlusOne-1u;if(entryCell<cell||(entryCell==cell&&entry.size<size)){low=middle+1u;}else{high=middle;}}if(low<count){let entry=powerCoarseSamples.entries[low];if(entry.cellPlusOne==cell+1u&&entry.size==size&&(entry.flags&0x10000000u)!=0u){return low;}}return 0xffffffffu;}
fn powerCoarseDenseValue(cell:u32,volume:u32)->f32{let invalidPhi=3.402823e38;let capacity=arrayLength(&powerCoarseSamples.entries);
 if(capacity<volume){return invalidPhi;}let dense=powerCoarseSamples.entries[capacity-volume+cell];
 if(dense.cellPlusOne!=cell+1u||dense.size!=1u||(dense.flags&${OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite}u)!=${OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite}u||dense.phi!=dense.phi){return invalidPhi;}return dense.phi;}
fn powerCoarseDenseSample(grid:vec3f)->f32{let invalidPhi=3.402823e38;if((powerCoarseSamples.generation&0x40000000u)==0u){return invalidPhi;}let d=powerCoarseSamples.dimensions;let volume=d.x*d.y*d.z;
 let lattice=clamp(grid-vec3f(0.5),vec3f(0.0),vec3f(d)-vec3f(1.0));let base=vec3u(floor(lattice));let fraction=fract(lattice);var terms:array<f32,8>;
 for(var corner=0u;corner<8u;corner+=1u){let o=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);let q=min(base+o,d-vec3u(1u));
  let cell=q.x+d.x*(q.y+d.y*q.z);let sample=powerCoarseDenseValue(cell,volume);if(!(sample<invalidPhi)){return invalidPhi;}
  let wx=select(1.0-fraction.x,fraction.x,o.x!=0u);let wy=select(1.0-fraction.y,fraction.y,o.y!=0u);let wz=select(1.0-fraction.z,fraction.z,o.z!=0u);terms[corner]=(wx*wz)*wy*sample;}
 for(var i=1u;i<8u;i+=1u){let term=terms[i];var j=i;loop{if(j==0u||terms[j-1u]<=term){break;}terms[j]=terms[j-1u];j-=1u;}terms[j]=term;}var value=0.;for(var i=0u;i<8u;i+=1u){value+=terms[i];}return value;}
fn powerCoarseAdaptiveCorner(aux:PowerCoarseSampleEntry,corner:u32)->f32{if(corner==0u){return bitcast<f32>(aux.cellPlusOne);}if(corner==1u){return bitcast<f32>(aux.size);}if(corner==2u){return aux.phi;}if(corner==3u){return aux.minimumPhi;}if(corner==4u){return aux.maximumPhi;}if(corner==5u){return bitcast<f32>(aux.flags);}if(corner==6u){return bitcast<f32>(aux.row);}return aux.physicalVolume;}
fn powerCoarseAdaptiveSample(grid:vec3f,slot:u32)->f32{let invalidPhi=3.402823e38;let entry=powerCoarseSamples.entries[slot];let auxSlot=powerCoarseSamples.rowCount+entry.row;if(entry.row>=powerCoarseSamples.rowCount||auxSlot>=arrayLength(&powerCoarseSamples.entries)){return invalidPhi;}let d=powerCoarseSamples.dimensions;let cell=entry.cellPlusOne-1u;let origin=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));let fraction=clamp((grid-vec3f(origin))/f32(entry.size),vec3f(0.),vec3f(1.));let aux=powerCoarseSamples.entries[auxSlot];var terms:array<f32,8>;for(var corner=0u;corner<8u;corner+=1u){let value=powerCoarseAdaptiveCorner(aux,corner);if(value!=value||abs(value)>=invalidPhi){return invalidPhi;}let o=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);let wx=select(1.-fraction.x,fraction.x,o.x!=0u);let wy=select(1.-fraction.y,fraction.y,o.y!=0u);let wz=select(1.-fraction.z,fraction.z,o.z!=0u);terms[corner]=(wx*wz)*wy*value;}for(var i=1u;i<8u;i+=1u){let term=terms[i];var j=i;loop{if(j==0u||terms[j-1u]<=term){break;}terms[j]=terms[j-1u];j-=1u;}terms[j]=term;}var result=0.;for(var i=0u;i<8u;i+=1u){result+=terms[i];}return result;}
fn powerCoarseContainingLookup(grid:vec3f)->u32{let onPlane=abs(grid-round(grid))<=vec3f(1e-4);for(var mask=0u;mask<8u;mask+=1u){var probe=grid;var eligible=true;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1u<<axis))!=0u){if(!onPlane[axis]||grid[axis]<=0.){eligible=false;}else{probe[axis]-=1e-5;}}}if(!eligible){continue;}let q=vec3u(floor(probe));if(any(q>=powerCoarseSamples.dimensions)){continue;}var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+powerCoarseSamples.dimensions.x*(origin.y+powerCoarseSamples.dimensions.y*origin.z);let adaptiveSlot=powerCoarseAdaptiveLookup(cell,size);if(adaptiveSlot!=0xffffffffu){return adaptiveSlot;}let legacySlot=powerCoarseLookup(cell,size);if(legacySlot!=0xffffffffu){return legacySlot;}if(size>=powerCoarseSamples.maximumLeafSize){break;}size*=2u;}}return 0xffffffffu;}
fn sampleAdaptiveCoarseOctreePhiAtGrid(grid:vec3f)->f32{let invalidPhi=3.402823e38;if(powerCoarseSamples.state!=0x80000000u||any(grid<vec3f(0.))||any(grid>vec3f(powerCoarseSamples.dimensions))){return invalidPhi;}let slot=powerCoarseContainingLookup(grid);if(slot==0xffffffffu){return invalidPhi;}let entry=powerCoarseSamples.entries[slot];if((entry.flags&0x10000001u)!=0x10000001u){return invalidPhi;}if((entry.flags&POWER_COARSE_CELL_CENTERED)!=0u){return entry.phi;}return powerCoarseAdaptiveSample(grid,slot);}
fn sampleCoarseOctreePhi(position:vec3f)->f32{let invalidPhi=3.402823e38;if(powerCoarseSamples.state!=0x80000000u||!(powerCoarseSamples.physicalCellSize>0.0)){return invalidPhi;}let rawGrid=position/powerCoarseSamples.physicalCellSize;let halfGrid=.5*round(2.*rawGrid);let grid=select(rawGrid,halfGrid,abs(rawGrid-halfGrid)<=vec3f(1e-4));if(any(grid<vec3f(0.0))||any(grid>=vec3f(powerCoarseSamples.dimensions))){return invalidPhi;}
 let dense=powerCoarseDenseSample(grid);if(dense<invalidPhi){return dense;}let slot=powerCoarseContainingLookup(grid);if(slot!=0xffffffffu){let entry=powerCoarseSamples.entries[slot];if((entry.flags&${OCTREE_COARSE_PHI_FLAG.valid}u)!=0u){if((entry.flags&POWER_COARSE_CELL_CENTERED)!=0u){return entry.phi;}if((entry.flags&0x10000000u)!=0u){return powerCoarseAdaptiveSample(grid,slot);}return entry.phi;}return invalidPhi;}return 0.5*powerCoarseSamples.physicalCellSize;}
`;
}
