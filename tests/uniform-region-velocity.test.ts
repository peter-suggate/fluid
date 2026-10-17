import assert from "node:assert/strict";
import test from "node:test";
import {UniformMixedFramePlan} from "../lib/methods/uniform/uniform-mixed-frame-plan";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {uniformLatticeAddress,uniformPageBlock} from "../lib/methods/uniform/uniform-lattice-address";
import {UniformDetailStorage} from "../lib/methods/uniform/uniform-detail-fields";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {UNIFORM_SURFACE_VELOCITY_REACH_H} from "../lib/methods/uniform/uniform-surface-transport.wgsl";

test("surface velocity support encloses nested fine-tap coarse sources across 16h boundaries",()=>{
 // Enumerate positive-weight canonical source addresses, rather than just
 // the intermediate interpolated points. Coarse cache restriction patches
 // stay in the same 4h tile's transverse 16h regions. Their normal face is
 // owned by the tile below it, including a face on a region boundary.
 let checked=0;
 for(const footReach of [0,2,8])for(let vertex=1024;vertex<=1056;vertex++){
  const reach=footReach+UNIFORM_SURFACE_VELOCITY_REACH_H;
  const low=Math.floor((vertex-reach)/16),high=Math.floor((vertex+reach)/16);
  const check=(origin:number)=>{const region=Math.floor(origin/16);assert.ok(region>=low&&region<=high,`source ${origin} outside support of ${vertex}, foot reach ${footReach}`);checked++;};
  // One RK step of at most .5 cells can place its third stage 1.5 cells
  // from the foot; cover the full allowed foot displacement as well.
  const extent=64*(footReach+1.5);
  for(let shift=-extent;shift<=extent;shift++)for(const offset of [.5,1]){
   const point=vertex+shift/64;
   const coarse=point/4-offset,base=Math.floor(coarse),fraction=coarse-base;
   for(let bit=0;bit<2;bit++)if((bit?fraction:1-fraction)>0)check(4*(base+bit));
   const fine=point-offset,fineBase=Math.floor(fine),fineFraction=fine-fineBase;
   for(let bit=0;bit<2;bit++){
    if((bit?fineFraction:1-fineFraction)===0)continue;
    const anchor=fineBase+bit;check(anchor);
    // A fine tap at a coarse seam is itself a coarse interpolation at
    // that tap's MAC point, not a direct canonical face read.
    const nested=(anchor+offset)/4-offset,nestedBase=Math.floor(nested),f=nested-nestedBase;
    for(let tap=0;tap<2;tap++)if((tap?f:1-f)>0)check(4*(nestedBase+tap));
   }
  }
 }
 assert.ok(checked>500000,"exercise both MAC offsets and complete region-boundary periods");
});

test("production certificate publishes current signed 16h velocity intervals without quantizing calm motion",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("16h velocity intervals",async device=>{
  const dimensions:[number,number,number]=[64,64,64],address=uniformLatticeAddress(dimensions);
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[.05,.05,.05],origin_m:{x:0,y:0,z:0}},[],4,undefined,address);
  const ownership=new UniformMixedOwnership(device,layout,true,0),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const volume=storage.createField("region volume","cell","r32float"),phi=storage.createField("region phi","vertex","r32float"),velocity=storage.createField("region velocity","face","rgba32float");
  const negative=device.createBuffer({size:4*address.lowFaceKeys,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const plan=new UniformMixedFramePlan(device,ownership,volume,phi,velocity,negative,velocity,negative,true);
  const read=device.createBuffer({size:ownership.regionVelocity.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const at=6*(8*uniformPageBlock(address,0,0,0)+7);
  const decode=(word:number)=>new Float32Array(Uint32Array.of((word&0x80000000)?word^0x80000000:~word).buffer)[0]!;
  const fill=(v:readonly number[])=>{
   const texture=storage.physical(velocity),values=new Float32Array(4*texture.width*texture.height*texture.depthOrArrayLayers);
   for(let i=0;i<values.length;i+=4)values.set(v,i);
   device.queue.writeTexture({texture},values,{bytesPerRow:16*texture.width,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
  };
  const capture=async()=>{const e=device.createCommandEncoder();plan.encodeCertificate(e,1/60);e.copyBufferToBuffer(ownership.regionVelocity,0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const w=new Uint32Array(read.getMappedRange()).slice();read.unmap();return w;};
  try{
   await plan.initialize();
   // The upper octant of the first stored block is [16,32)^3, away from exterior low-face zeros.
   fill([.00003,-.125,.25,0]);let w=await capture();
   for(let a=0;a<3;a++){const expected=Math.fround([.00003,-.125,.25][a]!);assert.equal(decode(w[at+a]!),expected);assert.equal(-decode(w[at+3+a]!),expected);}
   // A new certificate must replace earlier extrema, including sign changes.
   fill([-.00001,.0625,-.5,0]);w=await capture();
   for(let a=0;a<3;a++){const expected=Math.fround([-.00001,.0625,-.5][a]!);assert.equal(decode(w[at+a]!),expected);assert.equal(-decode(w[at+3+a]!),expected);}
   fill([Infinity,0,0,0]);w=await capture();assert.equal(w[at],0xffffffff);assert.equal(w[at+3],0xffffffff,"nonfinite component cannot authorize an approximation");
  }finally{read.destroy();plan.destroy();negative.destroy();storage.destroy();ownership.destroy();}
 });
});
