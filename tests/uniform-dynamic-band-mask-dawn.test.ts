import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedDynamicClassifier,UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT} from "../lib/methods/uniform/uniform-mixed-dynamic";

type Point=readonly [number,number,number];
const N=16,T=N/4;
const key=([x,y,z]:Point,n:number)=>x+n*(y+n*z);
const modulePath=process.env.WEBGPU_NODE_MODULE;

// Exercise the production classifier, including both mask words and the
// prefix fast path. The oracle enumerates crossing owners and intersects
// their closed physical boxes; it does not use packed masks or face gaps.
(modulePath?test:test.skip)("dynamic band intersects actual crossing cells and preserves closed support",{timeout:120000},async t=>{
 let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const width of [1,4] as const){
   const lattice={dimensions:[N,N,N] as Point,cellSize_m:[1,1,1] as Point,origin_m:{x:0,y:0,z:0}};
   const ownership=new UniformMixedOwnership(d,createUniformMixedLayout(lattice,[],width),false);
   const texture=(n:number,format:GPUTextureFormat)=>d.createTexture({size:[n,n,n],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
   const phi=texture(N+1,"r32float"),volume=texture(N,"r32float"),velocity=texture(N+1,"rgba32float");
   const classifier=new UniformMixedDynamicClassifier(d,ownership,volume,phi,velocity);
   try{
    await classifier.initialize();
    const run=async(name:string,field:(p:Point)=>number,flow:Point=[0,0,0],dt=0)=>{
     const values=new Float32Array((N+1)**3),speeds=new Float32Array(4*values.length);
     for(let z=0;z<=N;z++)for(let y=0;y<=N;y++)for(let x=0;x<=N;x++){
      const i=key([x,y,z],N+1);values[i]=field([x,y,z]);speeds.set([...flow,0],4*i);
     }
     d.queue.writeTexture({texture:phi},values,{bytesPerRow:4*(N+1),rowsPerImage:N+1},[N+1,N+1,N+1]);
     d.queue.writeTexture({texture:velocity},speeds,{bytesPerRow:16*(N+1),rowsPerImage:N+1},[N+1,N+1,N+1]);
     const crossing:Point[]=[];
     for(let z=0;z<N;z+=width)for(let y=0;y<N;y+=width)for(let x=0;x<N;x+=width){
      let inside=0;
      for(let k=0;k<8;k++)if(values[key([x+width*(k&1),y+width*((k>>1)&1),z+width*((k>>2)&1)],N+1)]!<0)inside++;
      if(inside>0&&inside<8)crossing.push([x,y,z]);
     }
     const expected=new Uint8Array(T**3),margin=UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT;
     for(let z=0;z<T;z++)for(let y=0;y<T;y++)for(let x=0;x<T;x++){
      const q=[x,y,z];
      const lo=q.map((v,a)=>Math.max(0,4*v-Math.max(flow[a]!,0)*dt)-margin);
      const hi=q.map((v,a)=>Math.min(N,4*v+4-Math.min(flow[a]!,0)*dt)+margin);
      expected[key([x,y,z],T)]=crossing.some(p=>p.every((v,a)=>v<=hi[a]!+1e-3&&v+width>=lo[a]!-1e-3))?1:0;
     }
     const e=d.createCommandEncoder();
     classifier.encode(e,{dt,steps:1,gravity:[0,0,0],reach:0,hysteresis:0,fullTolerance:.25,emptyTolerance:1e-6,
      surfaceTolerance:0,fastTravel:0,boundaryTravel:0,closedWalls:0,up:0});
     d.queue.submit([e.finish()]);const census=await classifier.read();
     assert.deepEqual(census.fine,expected,`${name}, width ${width}: band differs from closed crossing-owner union`);
     return census.fine;
    };
    if(width===1){
     // Two separate surface pieces in one tile: one reaches +x, the other
     // +y. Their independent face minima used to spuriously retain +x,+y.
     const points:Point[]=[[7,5,6],[5,7,6]];
     const field=(p:Point)=>points.some(q=>q.every((v,a)=>v===p[a]))?-1:1;
     const band=await run("disconnected crossings",field);
     assert.equal(band[key([2,2,1],T)],0,"diagonal air must coarsen");
     assert.equal(band[key([2,1,1],T)],1,"shared x face stays fine");
     assert.equal(band[key([1,2,1],T)],1,"shared y face stays fine");
     await run("reflected crossings",p=>field([N-p[0],N-p[1],N-p[2]]));
     const corner=await run("shared corner",p=>p.every(v=>v===7)?-1:1);
     assert.equal(corner[key([2,2,2],T)],1,"touching only a corner still needs support");
     await run("signed translation",field,[3,-2,1],.5);
     await run("large departure spanning whole tiles",field,[10,-6,2],1);
    }
    // Coarse crossings represent the full 4h owner, never guessed h cells.
    await run("plane",p=>p[1]-6.75);
    await run("moving plane",p=>p[1]-6.75,[0,-2,0],.5);
    const empty=await run("cleared crossings",()=>1);
    assert.equal(empty.reduce((n,v)=>n+v,0),0,"no stale mask bits survive the next census");
    t.diagnostic(`width ${width}: exact agreement with crossing-owner oracle`);
   }finally{classifier.destroy();ownership.destroy();phi.destroy();volume.destroy();velocity.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();}
});
