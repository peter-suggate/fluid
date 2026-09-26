import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSurfaceVolume} from "../lib/methods/uniform/uniform-mixed-surface-volume";
import {geometricSeamRows,seamLayout,gradedSeamLayout} from "./helpers/uniform-geometric-seam";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed global surface correction matches physical volume without materializing hanging vertices",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed global surface volume");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),gradedSeamLayout(0)]){
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,size=d.map(n=>n+1),cells=geometricSeamRows(layout,()=>[0,0,0]).cells;
   const n=d.reduce((a,b)=>a*b),nv=size.reduce((a,b)=>a*b),ownership=new UniformMixedOwnership(device,layout),owned:(GPUTexture|GPUBuffer)[]=[];
   const texture=(extent:readonly number[])=>{const t=device!.createTexture({size:[...extent],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});owned.push(t);return t;};
   const phi=texture(size),output=texture(size),volume=texture(d);
   try{
    const stage:UniformMixedSurfaceVolume=new UniformMixedSurfaceVolume(device,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const scratch=device.createBuffer({size:stage.scratchBytes,usage:GPUBufferUsage.STORAGE});owned.push(scratch);const group=stage.bind(phi,volume,output,{buffer:scratch});
    const index=(p:readonly number[])=>p[0]!+size[0]!*(p[1]!+size[1]!*p[2]!);
    const authority=(p:readonly number[])=>cells.filter(c=>c.min.every((v,a)=>p[a]!>=v&&p[a]!<=v+c.width)).sort((a,b)=>b.width-a.width)[0]!;
    const canonical=(p:readonly number[])=>{const c=authority(p);return c.min.every((v,a)=>(p[a]!-v)%c.width===0);};
    const points:number[][]=[];for(let z=0;z<=d[2];z++)for(let y=0;y<=d[1];y++)for(let x=0;x<=d[0];x++)points.push([x,y,z]);
    const isCanonical=points.map(canonical);
    const upload=(t:GPUTexture,values:Float32Array<ArrayBuffer>)=>device!.queue.writeTexture({texture:t},values,{bytesPerRow:t.width*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
    const initial=Float32Array.from(points,(p,i)=>isCanonical[i]?(p[1]!-d[1]/2)*h[1]:NaN);
    for(const sign of [-1,1]){
     const height=d[1]/2+sign*.2*Math.min(...h)/h[1],values=new Float32Array(n).fill(NaN);let desired=0;
     for(const c of cells){const value=Math.max(0,Math.min(1,(height-c.min[1])/c.width));values[c.min[0]+d[0]*(c.min[1]+d[1]*c.min[2])]=value;desired+=value*c.width**3;}
     upload(phi,initial);upload(volume,values);upload(output,new Float32Array(nv).fill(NaN));
     const e=device.createCommandEncoder();stage.encode(e,group);device.queue.submit([e.finish()]);const result=await readMixedTexture(device,output);assert.deepEqual(errors,[]);
     const cache=new Map<number,number>();
     const vertex=(p:readonly number[]):number=>{const at=index(p);if(isCanonical[at])return result[at]!;const saved=cache.get(at);if(saved!==undefined)return saved;
      const c=authority(p);let value=0;for(let k=0;k<8;k++){const corner=c.min.map((v,a)=>v+((k>>a)&1)*c.width);const w=c.min.reduce((w,v,a)=>w*(((k>>a)&1)?(p[a]!-v)/c.width:1-(p[a]!-v)/c.width),1);if(w>0)value+=w*vertex(corner);}cache.set(at,value);return value;};
     // A horizontal contour has one linear crossing per owner column. Sample
     // its four vertical edges and integrate the bilinear crossing height.
     let actual=0;
     for(const c of cells){let fraction=0;for(let x=0;x<2;x++)for(let z=0;z<2;z++){
      const a=[c.min[0]+x*c.width,c.min[1],c.min[2]+z*c.width],b=[a[0]!,a[1]!+c.width,a[2]!];const lo=vertex(a),hi=vertex(b);
      fraction+=lo>=0?0:hi<=0?1:-lo/(hi-lo);
     }actual+=fraction*.25*c.width**3;}
     assert.ok(Math.abs(actual-desired)/desired<2e-4,`physical surface volume ${actual} != ${desired}`);
     for(let i=0;i<nv;i++)if(isCanonical[i])assert.ok(Number.isFinite(result[i]),"invalid canonical phi");else assert.ok(Number.isNaN(result[i]),"inactive vertex was materialized");
     assert.deepEqual(await readMixedTexture(device,volume),values,"surface correction changed conservative V");
    }
   }finally{ownership.destroy();owned.forEach(r=>r.destroy());}
  }
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
