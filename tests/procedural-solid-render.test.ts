import assert from "node:assert/strict";
import test from "node:test";
import { bakeProceduralSolid, ProceduralSolidField } from "../lib/core/procedural-solid";
import { proceduralSolidRenderWGSL } from "../lib/core/procedural-solid-render.wgsl";
import { svoDualMarchingCubesFitWGSL } from "../lib/svo/features/meshing/dual-marching-cubes";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("render boundary samples and fitted normals follow the solver field across bricks and edits", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const h = [.02,.03,.04] as const, origin = [-.16,0,-.32] as const;
  const geometry = bakeProceduralSolid([16,16,16],h,([x,y,z]) => Math.max(Math.hypot(x-8,z-8)-5.25,y-12.25,2.25-y));
  const edited = new ProceduralSolidField(geometry.solidBoundary,[{operation:"clear",minimum:[12,6,8],maximumExclusive:[13,7,9]}]);
  const points: [number,number,number][] = [[12.5,6.5,8.5],[-1,5,8],[8,15,8]];
  for (const angle of [.1,.5,.9,1.5,2.1,2.8,3.2,4.5,5.7]) {
    points.push([8+5.25*Math.cos(angle),7.4,8+5.25*Math.sin(angle)]);
  }
  for (const x of [7.999,8,8.001]) points.push([x,12.25,8.25]);
  for (const z of [7.999,8,8.001]) points.push([13.25,7.4,z]);
  await withUniformDevice("procedural render boundary", async device => {
    // A nonzero base exercises the production arena offsets.
    const base = 32, packed = edited.pack(), words = new Uint32Array(base+packed.length);words.set(packed,base);
    const arena=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const queries=device.createBuffer({size:points.length*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:points.length*48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try {
      device.queue.writeBuffer(arena,0,words);
      device.queue.writeBuffer(queries,0,new Float32Array(points.flatMap(p=>[...p.map((v,a)=>origin[a]!+v*h[a]!),0])));
      const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> queries:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
fn swCell()->vec3f{return vec3f(${h.join(',')});}
fn swOrigin()->vec3f{return vec3f(${origin.join(',')});}
${proceduralSolidRenderWGSL(base)}
fn dcField(p:vec3f,dirty:u32,count:u32)->f32{return swBoundarySample(p).w;}
${svoDualMarchingCubesFitWGSL}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=arrayLength(&queries)){return;}
 let p=queries[id.x].xyz;let sample=swBoundarySample(p);output[3u*id.x]=sample;
 output[3u*id.x+2u]=vec4f(swBoundaryNormal(p,sample),0);
 let base=swOrigin()+floor((p-swOrigin())/swCell())*swCell();
 let fit=dmcFit(base,swCell(),0u,0u);output[3u*id.x+1u]=vec4f(fit.normal,fit.value);
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const layout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
        {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ]});
      const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout,entries:[arena,queries,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      const result=await readMixedBuffer(device,output);
      points.forEach((p,i)=>{
        const expected=edited.sample(p);
        if (!expected) {assert.ok(result[12*i+3]!>1e19,"absent and edited cells use the voxel fallback");return;}
        expected.forEach((v,a)=>assert.ok(Math.abs(result[12*i+a]!-v*.02/(a<3?h[a]!:1))<2e-5,`sample ${i} component ${a}`));
        for (let axis=0;axis<3;axis++) {
          const a=[...p] as [number,number,number],b=[...p] as [number,number,number];a[axis]!-=.5;b[axis]!+=.5;
          const lo=edited.sample(a),hi=edited.sample(b);
          const expectedGradient=lo&&hi ? (hi[3]-lo[3])*.02/h[axis]! : expected[axis]!*.02/h[axis]!;
          assert.ok(Math.abs(result[12*i+8+axis]!-expectedGradient)<2e-5,"surface normal matches the continuous stencil");
        }
        if (i>=3) {
          const gradient=expected.slice(0,3).map((v,a)=>v/h[a]!);const length=Math.hypot(...gradient);
          const alignment=gradient.reduce((sum,v,a)=>sum+v/length*result[12*i+4+a]!,0);
          assert.ok(alignment>.99,`fitted normal follows the continuous boundary: ${alignment}`);
        }
      });
      const left=12*(points.length-3)+8,right=12*(points.length-1)+8;
      assert.ok(Math.hypot(...[0,1,2].map(a=>result[left+a]!-result[right+a]!))<.002,"curved normal is continuous across a boundary-node seam");
    } finally {arena.destroy();queries.destroy();output.destroy();}
  });
});

test("boundary meshing uses the voxel plateau without inventing an air-side surface", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  // A convex positive distance patch in a boundary brick. A function-graph
  // QEF can extrapolate its unbounded tangent planes below zero even though
  // every sample is air. This is the failure exposed by the long slide's
  // sparse boundary/voxel-field handoff; the voxel arm saturates at 2/3 cell.
  const values=Array.from({length:729},(_,i)=>.05+(i%9-4)**2);
  const field=new ProceduralSolidField({version:1,dimensions:[8,8,8],cellSize_m:[1,1,1],patchCount:0,patchesKey:"",
    bricks:[{origin:[0,0,0],values}]},[]);
  await withUniformDevice("bounded procedural meshing field",async device=>{
    const packed=field.pack(),words=new Uint32Array(packed.length+32);words.set(packed,32);
    const arena=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try {
      device.queue.writeBuffer(arena,0,words);
      const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn swCell()->vec3f{return vec3f(1);}
fn swOrigin()->vec3f{return vec3f(0);}
${proceduralSolidRenderWGSL(32)}
fn dcField(p:vec3f,arm:u32,count:u32)->f32{
  if(arm==0u){return swBoundarySample(p).w;}
  return swBoundaryField(p,2./3.);
}
${svoDualMarchingCubesFitWGSL}
@compute @workgroup_size(1) fn main(){
  let raw=dmcFit(vec3f(3),vec3f(2,1,1),0u,0u);
  let bounded=dmcFit(vec3f(3),vec3f(2,1,1),1u,0u);
  output[0]=vec4f(raw.value,bounded.value,swBoundarySample(vec3f(4,3.5,3.5)).w,swBoundaryField(vec3f(9),2./3.));
  output[1]=swBoundarySample(vec3f(3.25,3.5,3.5));
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[arena,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      const result=await readMixedBuffer(device,output);
      assert.ok(result[0]!<0,"unbounded control reproduces a false solid fit in air");
      assert.ok(result[1]!>0,"the bounded production field keeps the fit in air");
      assert.ok(Math.abs(result[2]!-.05)<1e-5,"the original distance remains available to shading");
      assert.ok(result[3]!>1e19,"an absent brick still selects the voxel fallback");
      assert.ok(Math.abs(result[4]!+1)<1e-5,"clamping the meshing field does not flatten the shading gradient");
    }finally{arena.destroy();output.destroy();}
  });
});
