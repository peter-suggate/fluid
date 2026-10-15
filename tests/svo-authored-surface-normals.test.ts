import assert from "node:assert/strict";
import test from "node:test";
import { svoTerrainSurfaceNormalsWGSL } from "../lib/svo/features/primary-visibility/svo-authored-surface-normals";
import { signedSpatialCoordinateHash } from "../lib/core/signed-spatial-hash";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { buildSvoScenePrimitives } from "../lib/svo/features/scene-publication/svo-scene-primitives";

test("garden hose has direct material lookups into continuous sweeps; ferrule and leaves stay separate", () => {
  const built=buildSvoScenePrimitives(sceneDocument(getSceneDefinition("hero-garden-hose-x10")));
  const base=built.descriptors[0]!.materialId;
  const hose=built.metadata.filter(m=>m.tags.includes("hose")&&!m.tags.includes("ferrule"));
  assert.ok(hose.length>0);
  for(const m of hose){
    const shape=built.descriptors[m.primitiveIndex]!;
    assert.equal(shape.materialId-base,m.primitiveIndex,"dense material lookup reaches the hose's own record");
    assert.equal(shape.kind,"smooth-union-cluster");
    if(shape.kind==="smooth-union-cluster") assert.equal(shape.packing?.field,"tapered-sweep");
  }
  const ferrule=built.metadata.find(m=>m.tags.includes("ferrule"))!;
  assert.equal(built.descriptors[ferrule.primitiveIndex]!.kind,"capsule");
  assert.ok(built.metadata.some(m=>m.tags.includes("rosette")),"the detailed plants remain present");
});

test("GPU terrain normals follow a curved surface continuously and respect materials, boundaries and edits", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice("authored garden surface normals",async device=>{
    const base=32,nx=32,nz=32,hx=.02,hz=.03,ox=-.32,oz=-.48;
    const words=new Uint32Array(1200),floats=new Float32Array(words.buffer);
    floats.set([ox,oz,hx,hz],base);words.set([nx,nz,9,0],base+4);
    const height=(x:number,z:number)=>.3+2*x*x+z*z+.2*x-.1*z;
    for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)floats[base+8+x+nx*z]=height(ox+(x+.5)*hx,oz+(z+.5)*hz);
    const points=[[-.01001,.015],[-.01,.015],[-.00999,.015],[.15,-.12],[-.15,.12]];
    const queries=new Float32Array([...points.flatMap(([x,z])=>[x!,height(x!,z!),z!,9]),
      [.1,height(.1,.1),.1,55],[ox,height(ox,0),0,9],[.1,2,.1,9]].flat());
    const header=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const arena=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const input=device.createBuffer({size:queries.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:queries.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try{
      device.queue.writeBuffer(header,0,new Uint32Array([base,0,0,0]));device.queue.writeBuffer(arena,0,words);device.queue.writeBuffer(input,0,queries);
      const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> meshHeader:array<u32>;
@group(0) @binding(1) var<storage,read> meshMaintenance:array<u32>;
@group(0) @binding(2) var<storage,read> queries:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> output:array<vec4f>;
struct Mapping{cellSize:vec3f} struct Dry{mapping:Mapping} const dry=Dry(Mapping(vec3f(.003125)));
${svoTerrainSurfaceNormalsWGSL(0,1)}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=arrayLength(&queries)){return;}let q=queries[id.x];
 output[id.x]=vec4f(meshAuthoredTerrainNormal(q.xyz,u32(q.w),vec3f(0,1,0)),0);
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[header,arena,input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const run=async()=>{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);return readMixedBuffer(device,output);};
      const result=await run();
      points.forEach(([x,z],i)=>{const g=[-(4*x!+.2),1,-(2*z!-.1)],length=Math.hypot(...g);
        g.forEach((v,a)=>assert.ok(Math.abs(result[4*i+a]!-v/length)<.001,`analytic curved normal ${i}/${a}`));});
      for(let i=points.length;i<queries.length/4;i++)assert.deepEqual(Array.from(result.slice(4*i,4*i+3)),[0,0,0]);
      assert.ok(Math.hypot(...[0,1,2].map(a=>result[a]!-result[8+a]!))<.0001,"no normal jump across a height-sample seam");
      // A live edited page spanning the negative-X query. Signed coordinates
      // exercise the same hash and physical origin as the production page ABI.
      const edit=1100,coordinate=[-1,1,0] as const,hash=signedSpatialCoordinateHash(coordinate),slot=hash&3;
      words.set([0,0,4,3,24,0,1,1],edit);floats.set([0,0,0],edit+16);floats.set([.02,.02,.02],edit+20);
      words.set([2,hash,...coordinate.map(v=>v>>>0),0],edit+24+6*slot);
      device.queue.writeBuffer(arena,0,words);device.queue.writeBuffer(header,4,new Uint32Array([edit]));
      const edited=await run();assert.deepEqual(Array.from(edited.slice(0,3)),[0,0,0],"edited terrain retains fitted normals");
      assert.ok(edited[13]!>.5,"untouched terrain keeps smooth normals");
    }finally{header.destroy();arena.destroy();input.destroy();output.destroy();}
  });
});
