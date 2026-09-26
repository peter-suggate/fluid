import { UniformMixedCellProlongation } from "../lib/methods/uniform/uniform-mixed-cell-prolongation";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedVelocityRestriction } from "../lib/methods/uniform/uniform-mixed-velocity-restriction";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { readMixedTexture, readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { seamLayout } from "./helpers/uniform-geometric-seam";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed MAC restriction packs intersecting patches without races and round-trips cell mass and face flux without field allocation",{timeout:180000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed MAC restriction");
  let device:GPUDevice|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),...mixedPressureLayouts()]){
      const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,halo=d.map(v=>v+2),fixture=mixedPressureFixture(layout);
      const ownership=new UniformMixedOwnership(device,layout),usage=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC;
      const input=device.createTexture({size:halo,dimension:"3d",format:"rgba32float",usage});
      const output=device.createTexture({size:[...d],dimension:"3d",format:"rgba32float",usage});
      const boundary=device.createBuffer({size:4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
      const extraTextures:GPUTexture[]=[],extraBuffers:GPUBuffer[]=[];
      try{
        const fine=new Float32Array(halo[0]!*halo[1]!*halo[2]!*4);
        for(let z=0;z<halo[2]!;z++)for(let y=0;y<halo[1]!;y++)for(let x=0;x<halo[0]!;x++)for(let a=0;a<3;a++)fine[4*(x+halo[0]!*(y+halo[1]!*z))+a]=Math.sin(x*3+y*5+z*7+a);
        device.queue.writeTexture({texture:input},fine,{bytesPerRow:halo[0]!*16,rowsPerImage:halo[1]!},halo);
        device.queue.writeTexture({texture:output},new Float32Array(d[0]*d[1]*d[2]*4).fill(NaN),{bytesPerRow:d[0]*16,rowsPerImage:d[1]},[...d]);
        device.queue.writeBuffer(boundary,0,new Float32Array(boundary.size/4).fill(NaN));
        const borrowed:GPUDevice=new Proxy(device,{get(target,key){
          if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("Mixed velocity restriction allocated a field");};
          const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
        }});
        const stage:UniformMixedVelocityRestriction=new UniformMixedVelocityRestriction(borrowed,ownership,input,output,boundary);await stage.initialize();assert.equal(stage.allocatedBytes,0);
        const encoder=device.createCommandEncoder();stage.encode(encoder);device.queue.submit([encoder.finish()]);
        const values=await readMixedTexture(device,output),negative=await readMixedBuffer(device,boundary),written=new Set<number>();
        const patches:{center:readonly number[];width:number;axis:number}[]=[];
        const check=(center:readonly number[],width:number,axis:number)=>{
          patches.push({center:[...center],width,axis});
          const anchor=center.map((v,a)=>Math.round(v-(axis===a?1:width/2))),u=(axis+1)%3,v=(axis+2)%3;let expected=0;
          for(let y=0;y<width;y++)for(let x=0;x<width;x++){
            const q=anchor.map(v=>v+1);q[u]!+=x;q[v]!+=y;expected+=fine[4*(q[0]!+halo[0]!*(q[1]!+halo[1]!*q[2]!))+axis]!;
          }
          expected/=width**2;let actual:number;
          if(anchor[axis]!<0){
            const [x,y,z]=anchor,at=axis===0?y!+d[1]*z!:axis===1?d[1]*d[2]+x!+d[0]*z!:d[1]*d[2]+d[0]*d[2]+x!+d[0]*y!;
            actual=negative[at]!;
          }else{const at=anchor[0]!+d[0]*(anchor[1]!+d[1]*anchor[2]!);written.add(at);actual=values[4*at+axis]!;}
          assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<1e-6,`face ${anchor}/${axis}: ${actual} != ${expected}`);
        };
        fixture.faces.forEach(f=>check(f.center.map((v,a)=>v/h[a]!),Math.min(fixture.cells[f.left]!.width,fixture.cells[f.right]!.width),f.axis));
        fixture.cells.forEach(c=>{
          const center=c.center.map((v,a)=>v/h[a]!);
          for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
            const plane=center[axis]!+sign*c.width/2;if(plane!==0&&plane!==d[axis])continue;
            const p=[...center];p[axis]=plane;check(p,c.width,axis);
          }
        });
        for(let i=0;i<values.length/4;i++)if(!written.has(i))for(let a=0;a<4;a++)assert.ok(Number.isNaN(values[i*4+a]),"restriction wrote a non-canonical texel");
        const scalar=()=>{const t=device!.createTexture({size:[...d],dimension:"3d",format:"r32float",usage});extraTextures.push(t);return t;};
        const compactVolume=scalar(),fineVolume=scalar();
        const fineVelocity:GPUTexture=device.createTexture({size:[...d],dimension:"3d",format:"rgba32float",usage});extraTextures.push(fineVelocity);
        const fineBoundary:GPUBuffer=device.createBuffer({size:boundary.size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});extraBuffers.push(fineBoundary);
        const density=new Float32Array(d[0]*d[1]*d[2]).fill(NaN);
        const cellValue=(i:number)=>Math.fround(.5+.25*Math.sin(i));
        const cellIndex=(q:readonly number[])=>q[0]!+d[0]*(q[1]!+d[1]*q[2]!);
        fixture.cells.forEach((c,i)=>{
          const origin=c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2));density[cellIndex(origin)]=cellValue(i);
        });
        device.queue.writeTexture({texture:compactVolume},density,{bytesPerRow:d[0]*4,rowsPerImage:d[1]},[...d]);
        const handoff:UniformMixedCellProlongation=new UniformMixedCellProlongation(borrowed,ownership,
          {volume:compactVolume,velocity:output,negativeFaces:boundary},
          {volume:fineVolume,velocity:fineVelocity,negativeFaces:fineBoundary});
        await handoff.initialize();assert.equal(handoff.allocatedBytes,0);
        const expand=device.createCommandEncoder();handoff.encode(expand);device.queue.submit([expand.finish()]);
        const expanded=await readMixedTexture(device,fineVelocity),expandedV=await readMixedTexture(device,fineVolume),expandedBoundary=await readMixedBuffer(device,fineBoundary);
        fixture.cells.forEach((c,i)=>{
          const origin=c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2));
          for(let z=0;z<c.width;z++)for(let y=0;y<c.width;y++)for(let x=0;x<c.width;x++)assert.equal(expandedV[cellIndex([origin[0]!+x,origin[1]!+y,origin[2]!+z])],cellValue(i));
        });
        const boundaryIndex=(q:readonly number[],axis:number)=>axis===0?q[1]!+d[1]*q[2]!:axis===1?d[1]*d[2]+q[0]!+d[0]*q[2]!:d[1]*d[2]+d[0]*d[2]+q[0]!+d[0]*q[1]!;
        for(const {center,width,axis} of patches){
          const anchor=center.map((v,a)=>Math.round(v-(a===axis?1:width/2))),u=(axis+1)%3,v=(axis+2)%3;
          const source=anchor[axis]!<0?negative[boundaryIndex(anchor,axis)]!:values[4*cellIndex(anchor)+axis]!;
          let sum=0;
          for(let y=0;y<width;y++)for(let x=0;x<width;x++){
            const q=[...anchor];q[u]!+=x;q[v]!+=y;
            sum+=anchor[axis]!<0?expandedBoundary[boundaryIndex(q,axis)]!:expanded[4*cellIndex(q)+axis]!;
          }
          assert.ok(Math.abs(sum/width**2-source)<1e-6,`handoff changed face flux at ${anchor}/${axis}`);
        }
        assert.ok(expanded.every(Number.isFinite)&&expandedV.every(Number.isFinite)&&expandedBoundary.every(Number.isFinite));

      }finally{extraTextures.forEach(t=>t.destroy());extraBuffers.forEach(b=>b.destroy());boundary.destroy();output.destroy();input.destroy();ownership.destroy();}
    }
    assert.deepEqual(errors,[]);
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
