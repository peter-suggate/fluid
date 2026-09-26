import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingWGSL } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";

const modulePath=process.env.WEBGPU_NODE_MODULE;
const affine=(p:readonly number[],axis:number)=>1+axis+.2*p[0]!-.3*p[1]!+.4*p[2]!;
const arbitrary=(p:readonly number[],axis:number)=>Math.sin(p[0]!*3+p[1]!*5+p[2]!*7+axis);
const fraction=(i:number)=>{const v=Math.sin(i*13)*43758.5453;return v-Math.floor(v);};

(modulePath?test:test.skip)("mixed MAC sampling reads only canonical faces and retains uniform interpolation",{timeout:180000},async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed MAC sampling");
  let device:GPUDevice|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const layouts=[seamLayout(0,"fine"),seamLayout(0,"coarse"),...mixedPressureLayouts().slice(0,5)];
    for(const [index,layout] of layouts.entries())await t.test(`layout ${index}`,async()=>{
      const ownership=new UniformMixedOwnership(device!,layout),fixture=mixedPressureFixture(layout),d=layout.lattice.dimensions,h=layout.lattice.cellSize_m;
      const n=d[0]*d[1]*d[2],count=256;
      // Deliberately poison all non-authoritative entries. A fine-grid read in
      // a coarse region must fail instead of passing on zero-filled storage.
      const data=new Float32Array(n*8);
      const field=device!.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const queries=device!.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const output=device!.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const read=device!.createBuffer({size:count*16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      try{
        const module=device!.createShaderModule({code:uniformMixedTopologyWGSL(layout,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read> velocity:array<vec4f>;
@group(1) @binding(1) var<storage,read> queries:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> result:array<vec4f>;
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 var q=anchor;var base=0u;if(q[axis]<0){base=${n}u;q[axis]=0;}
 return velocity[base+u32(q.x)+UM_D.x*(u32(q.y)+UM_D.y*u32(q.z))][axis];
}
${uniformMixedVelocitySamplingWGSL}
@compute @workgroup_size(64) fn sample(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=arrayLength(&queries)){return;}result[gid.x]=vec4f(umSampleVelocity(queries[gid.x].xyz),0);
}`});
        const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}: ${m.message}`),[]);
        const resources=device!.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?"storage" as const:"read-only-storage" as const}}))});
        const pipeline=await device!.createComputePipelineAsync({layout:device!.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"sample"}});
        const group=device!.createBindGroup({layout:resources,entries:[field,queries,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
        for(const mode of ["constant","affine","arbitrary","continuity"] as const){
          const value=mode==="constant"?(_:readonly number[],axis:number)=>axis+1:mode==="affine"?affine:arbitrary;
          data.fill(NaN);
          const put=(center:readonly number[],width:number,axis:number)=>{
            const anchor=center.map((v,a)=>Math.round(v-(a===axis?1:width/2)));
            let base=0;if(anchor[axis]!<0){base=n;anchor[axis]=0;}
            data[4*(base+anchor[0]!+d[0]*(anchor[1]!+d[1]*anchor[2]!))+axis]=value(center,axis);
          };
          fixture.faces.forEach(f=>put(f.center.map((v,a)=>v/h[a]!),Math.min(fixture.cells[f.left]!.width,fixture.cells[f.right]!.width),f.axis));
          fixture.cells.forEach(c=>{
            const center=c.center.map((v,a)=>v/h[a]!);
            for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
              const plane=center[axis]!+sign*c.width/2;
              if(plane!==0&&plane!==d[axis])continue;
              const p=[...center];p[axis]=plane;put(p,c.width,axis);
            }
          });
          let points=Array.from({length:count},(_,i)=>d.map((v,a)=>mode==="affine"&&index>=2
            ? 4+Math.max(0,v-8)*fraction(i*3+a+1):v*fraction(i*3+a+1)));
          if(mode==="continuity")points=Array.from({length:count},(_,i)=>{
            const pair=Math.floor(i/2),axis=pair%3;
            const p=d.map((v,a)=>v*fraction(pair*3+a+1));
            p[axis]=4*(1+Math.floor(fraction(pair+3)*(d[axis]!/4-1)))+(i%2?1:-1)*1e-5;return p;
          });
          const packed=Float32Array.from(points.flatMap(p=>[...p,0]));device!.queue.writeBuffer(field,0,data);device!.queue.writeBuffer(queries,0,packed);
          const encoder=device!.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);pass.dispatchWorkgroups(count/64);pass.end();
          encoder.copyBufferToBuffer(output,0,read,0,count*16);device!.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
          const actual=new Float32Array(read.getMappedRange());
          for(let i=0;i<count;i++)for(let axis=0;axis<3;axis++){
            assert.ok(Number.isFinite(actual[4*i+axis]),`${mode}, point ${i}, component ${axis}: non-authoritative read`);
            if(index<2){
              const width=index===0?1:4,p=Array.from(packed.subarray(i*4,i*4+3));
              const q=p.map((v,a)=>Math.min(d[a]!/width-1,Math.max(a===axis?-1:0,v/width-(a===axis?1:.5))));
              const base=q.map(Math.floor),f=q.map((v,a)=>v-base[a]!);let expected=0;
              for(let k=0;k<8;k++){
                const bits=[k&1,(k>>1)&1,k>>2],weight=bits.reduce((w,b,a)=>w*(b?f[a]!:1-f[a]!),1);if(!weight)continue;
                const center=base.map((v,a)=>(v+bits[a]!+(a===axis?1:.5))*width);expected+=weight*value(center,axis);
              }
              assert.ok(Math.abs(actual[4*i+axis]!-expected)<3e-5,`${mode}: native endpoint mismatch`);
            }else if(mode==="constant"||mode==="affine"){
              const expected=value(Array.from(packed.subarray(i*4,i*4+3)),axis);
              assert.ok(Math.abs(actual[4*i+axis]!-expected)<3e-5,`${mode}, point ${points[i]}, component ${axis}: ${actual[4*i+axis]} != ${expected}`);
            }
          }
          if(mode==="continuity"){
            let jump=0;for(let i=0;i<count;i+=2)for(let a=0;a<3;a++)jump=Math.max(jump,Math.abs(actual[4*i+a]!-actual[4*(i+1)+a]!));
            assert.ok(jump<1e-3,`sampling seam jump ${jump}`);
          }
          read.unmap();
        }
      }finally{if(read.mapState==="mapped")read.unmap();read.destroy();output.destroy();queries.destroy();field.destroy();ownership.destroy();}
    });
    assert.deepEqual(errors,[]);
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
