import { particleSlabSpill } from "../lib/methods/uniform/uniform-particle-slab.wgsl";

/** Receipt words through the current spill count, followed by the particle bank. */
export const PARTICLE_MASS_HEADER_WORDS = 22;
export const particleMassReadbackBytes = (capacity: number, strideFloats: number) =>
  4 * (PARTICLE_MASS_HEADER_WORDS + capacity * strideFloats);

/** Decode only the current slab extent and spill; older tail records can remain live in memory. */
export function* particleMassRecords(raw: ArrayBuffer, capacity: number, strideFloats: number, merged: boolean) {
  const f = new Float32Array(raw), u = new Uint32Array(raw);
  const extent = Math.min(u[0]!, capacity - particleSlabSpill(capacity));
  const count = Math.min(capacity, extent + u[21]!);
  for (let i = 0; i < count; i++) {
    const at = PARTICLE_MASS_HEADER_WORDS + i * strideFloats;
    if (!(f[at + 3]! > 0)) continue;
    yield { x: f[at]!, y: f[at + 1]!, z: f[at + 2]!, depth: f[at + 7]!,
      vx: f[at + 4]!, vy: f[at + 5]!, vz: f[at + 6]!,
      bx: f[at + 8]!, by: f[at + 9]!, bz: f[at + 10]!,
      mass: !merged ? 1 : strideFloats === 28 ? f[at + 25]! + 1 : (u[at + 11]! & 7) + 1 };
  }
}

/** A diagnostic census of physical mass, including mass hidden above a
 * cell's reconstruction bound. The temporary readback is freed each time. */
export async function readParticleMassCensus(device:GPUDevice,flip:{state:GPUBuffer;activeParticles:GPUBuffer;capacity:number;particleStride:number;crowdedCells:string}){
 const stride=flip.particleStride,chunkRecords=Math.floor(4*1024*1024/(4*stride));
 const buffer=device.createBuffer({label:"Particle physical mass census",size:4*(PARTICLE_MASS_HEADER_WORDS+chunkRecords*stride),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const header=4*44,cells=new Map<string,number>();let mass=0,records=0,deep=0;
 try{
  let e=device.createCommandEncoder();e.copyBufferToBuffer(flip.state,0,buffer,0,header);device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);
  const status=new Uint32Array(buffer.getMappedRange(0,header));
  const count=Math.min(flip.capacity,Math.min(status[0]!,flip.capacity-particleSlabSpill(flip.capacity))+status[21]!);const retired=Array.from(status.subarray(32,44));buffer.unmap();
  for(let first=0;first<count;first+=chunkRecords){
   const n=Math.min(chunkRecords,count-first);e=device.createCommandEncoder();e.copyBufferToBuffer(flip.activeParticles,4*stride*first,buffer,0,4*stride*n);device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);
   const raw=buffer.getMappedRange(),f=new Float32Array(raw),u=new Uint32Array(raw);
   for(let i=0;i<n;i++){
    const at=i*stride;if(!(f[at+3]!>0))continue;
    const m=flip.crowdedCells!=="merge"?1:stride===28?f[at+25]!+1:(u[at+11]!&7)+1;
    records++;mass+=m;if(f[at+7]!< -4)deep+=m;
    const key=`${Math.floor(f[at]!)},${Math.floor(f[at+1]!)},${Math.floor(f[at+2]!)}`;cells.set(key,(cells.get(key)??0)+m);
   }
   buffer.unmap();
  }
  let excess=0,maximum=0;for(const m of cells.values()){excess+=Math.max(0,m-8);maximum=Math.max(maximum,m);}
  return {records,mass_cells:mass/8,excess_cells:excess/8,deep_cells:deep/8,maxCellMass:maximum,retired};
 }finally{buffer.destroy();}
}
