import { UniformNarrowBandOrder } from "../../lib/methods/uniform/uniform-narrow-band-order";
import { narrowBandParticleStride } from "../../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { uniformLatticeAddress, uniformTileKey } from "../../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayout } from "../../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../../lib/methods/uniform/uniform-mixed-ownership";

/** Isolate production bin scan, packing and canonical ordering, with a CPU
 * oracle independent of the GPU implementation. */
export async function particleOrderFixture(device:GPUDevice,capacity:number,apic=false,sparse?:boolean){
 const dimensions:[number,number,number]=[32,32,32],address=uniformLatticeAddress(dimensions);
 const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,undefined,address);
 const ownership=new UniformMixedOwnership(device,layout,true,0);
 const stride=narrowBandParticleStride(apic),cells=address.cellKeys,tiles=address.tileKeys,occupied=2*cells+tiles;
 const buffer=(size:number)=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
 const particles:[GPUBuffer,GPUBuffer]=[buffer(4*stride*capacity),buffer(4*stride*capacity)];
 const bins=buffer(4*(occupied+4+tiles)),links=buffer(4*(cells+4*capacity)),state=buffer(64);
 const order=new UniformNarrowBandOrder(device,address,ownership.presentation.buffer,particles,bins,links,state,occupied,sparse,apic);
 await order.initialize();
 const cellKey=(data:Float32Array,at:number)=>{const [x,y,z]=[data[at],data[at+1],data[at+2]].map(Math.floor);return 64*uniformTileKey(address,x>>2,y>>2,z>>2)+(x&3)+4*(y&3)+16*(z&3);};
 return {order,stride,particles,bins,cells,
  load(data:Float32Array,parity=0){
   const count=data.length/stride;if(count>capacity)throw new Error("Particle order fixture capacity exceeded");
   const counts=new Uint32Array(bins.size/4),occupiedTiles=new Set<number>();
   for(let i=0;i<count;i++){if(data[stride*i]<0)continue;const key=cellKey(data,stride*i);counts[key]++;counts[2*cells+(key>>6)]++;occupiedTiles.add(key>>6);}
   counts[occupied]=occupiedTiles.size;counts.set([...occupiedTiles],occupied+4);
   if(data.length)device.queue.writeBuffer(particles[1-parity],0,data.buffer as ArrayBuffer,data.byteOffset,data.byteLength);
   device.queue.writeBuffer(bins,0,counts);device.queue.writeBuffer(state,0,new Uint32Array([count,0]));
  },
  encode(encoder:GPUCommandEncoder,parity=0){encoder.clearBuffer(bins,4*cells,4*cells);order.encode(encoder,parity);},
  oracle(data:Float32Array){
   const bits=new Uint32Array(data.buffer,data.byteOffset,data.length),ids=Array.from({length:data.length/stride},(_,i)=>i).filter(i=>data[stride*i]>=0);
   const word=(i:number,w:number)=>!apic&&w>=11?(w===11?bits[stride*i+11]>>>31:bits[stride*i+11]&0x7fffffff):bits[stride*i+w];
   ids.sort((a,b)=>{const cell=cellKey(data,stride*a)-cellKey(data,stride*b);if(cell)return cell;
    for(let w=0;w<(apic?28:13);w++){const diff=word(a,w)-word(b,w);if(diff)return diff;}return a-b;
   });
   const sorted=new Uint32Array(ids.length*stride);ids.forEach((id,i)=>sorted.set(bits.subarray(id*stride,(id+1)*stride),i*stride));return sorted;
  },
  async read(count:number,parity=0){
   if(!count)return new Uint32Array();
   const read=device.createBuffer({size:4*count*stride,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   try{const e=device.createCommandEncoder();e.copyBufferToBuffer(particles[parity],0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);return new Uint32Array(read.getMappedRange()).slice();}
   finally{if(read.mapState==="mapped")read.unmap();read.destroy();}
  },
  destroy(){order.destroy();ownership.destroy();for(const b of [...particles,bins,links,state])b.destroy();}
 };
}

export type ParticleOrderPattern="positions"|"ties"|"identical"|"late";
/** Independent total count and cell occupancy, deterministic shuffled keys. */
export function particleOrderInput(count:number,occupancy:number,apic=false,pattern:ParticleOrderPattern="positions"){
 const stride=narrowBandParticleStride(apic),data=new Float32Array(count*stride);
 for(let i=0;i<count;i++){
  const cell=Math.floor(i/occupancy),x=cell%32,y=Math.floor(cell/32)%32,z=Math.floor(cell/1024)%32;
  const key=(Math.imul(i+1,1664525)+1013904223)>>>0,at=i*stride;
  data.set([x+(pattern==="positions"?0.125+0.75*((key>>>8)+.5)/16777216:.5),y+.5,z+.5,1],at);
  if(pattern!=="identical"){
   const field=pattern==="late"?stride-1:4;
   data[at+field]=((key>>>3)%65521)-32760;
   if(pattern!=="late"){data[at+stride-1]=(i%2?-1:1)*(i%11);}
  }
 }
 // Shuffle records across cells, not just inside each run.
 const temp=new Float32Array(stride);let random=19;
 for(let i=count-1;i>0;i--){random=(Math.imul(random,1664525)+1013904223)>>>0;const j=random%(i+1);temp.set(data.subarray(stride*i,stride*(i+1)));data.copyWithin(stride*i,stride*j,stride*(j+1));data.set(temp,stride*j);}
 return data;
}
