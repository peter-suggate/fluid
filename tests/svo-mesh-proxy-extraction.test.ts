import assert from "node:assert/strict";
import test from "node:test";
import { svoVoxelMeshWGSL } from "../lib/svo/features/meshing/voxels";
import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { withUniformDevice } from "./helpers/uniform-geometric";

// Exercise the production greedy extractor against small deterministic brick
// fields. The reference rejects sampled faces only at append, as before. Both
// arms use identical occupancy, mixed-resolution neighbours and output packing.
function extractionShader(early: boolean, retainCasters: boolean): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read> voxels:array<u32>;
struct Result { count:atomic<u32>, queries:atomic<u32>, errors:atomic<u32>, pad:u32, quads:array<vec4u> }
@group(0) @binding(1) var<storage,read_write> result:Result;
@group(0) @binding(2) var<storage,read_write> meshState:array<atomic<u32>>;
struct Mapping { brickSize:u32, maximumDepth:u32 }
struct Dry { mapping:Mapping }
const dry=Dry(Mapping(8u,1u));
struct Material { identity:vec4u }
fn dryMaterial(id:u32)->Material{return Material(vec4u(0u,0u,0u,select(0u,4u,id==2u)));}
fn sceneIdentityMaterial(id:u32)->u32{return id&65535u;}
fn sceneIdentitySolid(id:u32)->bool{return sceneIdentityMaterial(id)!=0u;}
fn sceneIdentityAt(i:u32)->u32{return voxels[i];}
fn meshMergeIdentity(id:u32)->u32{return sceneIdentityMaterial(id);}
fn dryVoxelCapacity()->u32{return 512u;}
fn svoBrickVoxelIndex(payload:u32,c:vec3u,n:u32)->u32{return c.x+n*(c.y+n*c.z);}
struct Leaf {topology:vec4u}
struct Node {address:vec4u}
fn svoLeafLoad(i:u32)->Leaf{return Leaf(vec4u(0u));}
fn svoNodeLoad(i:u32)->Node{return Node(vec4u(0u));}
fn svoDecodeMorton(x:u32,y:u32,z:u32)->vec3u{return vec3u(0u);}
fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return face|(level<<3u)|(depth<<6u);}
fn meshContourCode(i:u32)->u32{return 0u;}
struct MeshRegion {origin:vec3f,size:f32,identity:u32,contour:u32}
fn meshRegionAt(p:vec3f)->MeshRegion{
 atomicAdd(&result.queries,1u);
 // Alternating fine external cells force boundary subdivision; ordinary and
 // sampled solids must both continue hiding the neighbour's faces.
 let origin=floor(p);let parity=u32(abs(origin.x+origin.y+origin.z))%3u;
 return MeshRegion(origin,1.0,parity,0u);
}
fn meshNeighbourCovered(p:vec3f,cell:u32)->bool{return sceneIdentitySolid(meshRegionAt(p).identity);}
var<private> meshMaskAgreement:array<u32,64>;
var<private> meshFaceAgreement:u32;
fn meshCellMaterial(payload:u32,cell:vec3u,level:u32,n:u32)->u32{
 let span=1u<<level;
 for(var z=0u;z<span;z++){for(var y=0u;y<span;y++){for(var x=0u;x<span;x++){
  let id=sceneIdentityAt(svoBrickVoxelIndex(payload,cell*span+vec3u(x,y,z),n));
  if(sceneIdentitySolid(id)){return sceneIdentityMaterial(id);}
 }}}
 return 0u;
}
fn meshFaceIdentity(payload:u32,c:vec3u,level:u32,n:u32,face:u32)->u32{
 atomicAdd(&result.queries,1u);meshFaceAgreement=face+13u;
 return meshCellMaterial(payload,c,level,n)|(face<<16u);
}
fn meshAppend(origin:vec3u,extent:vec3u,face:u32,id:u32){
 ${retainCasters ? "" : "if(sceneIdentityMaterial(id)==2u){return;}"}
 let at=atomicAdd(&result.count,1u);
 if(at>=8192u){atomicAdd(&result.errors,1u);return;}
 result.quads[2u*at]=vec4u(origin,id);result.quads[2u*at+1u]=vec4u(extent,face);
}
${svoVoxelMeshWGSL(0, false, !early || retainCasters)}
@compute @workgroup_size(1) fn main(){for(var job=0u;job<51u;job++){meshVoxelExtractJob(0u,job);}}
`;
}

test("sampled proxy early rejection precedes boundary and contour work without changing occupancy", () => {
  const source = svoSurfaceMeshWGSL(3, 128, true, true, false);
  const exact = source.split("fn meshVoxelExtractJob(")[1]!;
  assert.ok(exact.indexOf("dryMaterial(sceneIdentityMaterial(identity))") < exact.indexOf("let code=meshContourCode(voxel)"));
  const coarse = source.split("fn meshBuildLevel(")[1]!.split("fn meshVoxelExtractJob(")[0]!;
  assert.ok(coarse.indexOf("dryMaterial(cells[") < coarse.indexOf("exposed=!meshNeighbourCovered"));
  assert.match(coarse, /cells\[i\]=meshCellMaterial/);
  assert.doesNotMatch(svoVoxelMeshWGSL(0, true, true), /dryMaterial/);
});

test("GPU early proxy rejection preserves exact and coarse quads across mixed material boundaries", {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
  await withUniformDevice("mesh proxy extraction", async device => {
    const inputs = [
      new Uint32Array(512).fill(2),
      new Uint32Array(512).fill(1),
      Uint32Array.from({length:512}, (_, i) => (i % 8 + Math.floor(i / 8) % 8 + Math.floor(i / 64)) % 3),
      Uint32Array.from({length:512}, (_, i) => i % 8 < 4 ? 2 : 1),
      Uint32Array.from({length:512}, (_, i) => i % 11 === 0 ? 1 : i % 7 === 0 ? 0 : 2),
    ];
    const bytes = 16 + 8192 * 32;
    const input = device.createBuffer({size:2048,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output = device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const state = device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    const read = device.createBuffer({size:bytes+4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try {
      for (const retainCasters of [false,true]) {
        const pipelines = await Promise.all([false,true].map(async early => {
          const module=device.createShaderModule({code:extractionShader(early,retainCasters)});
          assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
          return device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
        }));
        const groups=pipelines.map(p=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[input,output,state].map((buffer,binding)=>({binding,resource:{buffer}}))}));
        for (const [fixture, values] of inputs.entries()) {
          device.queue.writeBuffer(input,0,values);
          const results:Uint32Array[]=[];
          for (let arm=0;arm<2;arm++) {
            const encoder=device.createCommandEncoder();encoder.clearBuffer(output);encoder.clearBuffer(state);
            const pass=encoder.beginComputePass();pass.setPipeline(pipelines[arm]!);pass.setBindGroup(0,groups[arm]!);pass.dispatchWorkgroups(1);pass.end();
            encoder.copyBufferToBuffer(output,0,read,0,bytes);encoder.copyBufferToBuffer(state,0,read,bytes,4);
            device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
            results.push(new Uint32Array(read.getMappedRange()).slice());read.unmap();
          }
          const [reference,candidate]=results as [Uint32Array,Uint32Array];
          assert.equal(reference[2],0);assert.equal(reference[bytes/4],0);
          assert.equal(candidate[2],0);assert.equal(candidate[bytes/4],0);
          assert.equal(candidate[0],reference[0],`quad count, fixture ${fixture}, casters ${retainCasters}`);
          assert.deepEqual(candidate.slice(4,4+candidate[0]!*8),reference.slice(4,4+reference[0]!*8),`quad payload, fixture ${fixture}, casters ${retainCasters}`);
          if (retainCasters || fixture===1) assert.equal(candidate[1],reference[1]);
          else assert.ok(candidate[1]!<reference[1]!,`fewer neighbour/normal queries for fixture ${fixture}`);
          if (fixture===0&&!retainCasters) assert.equal(candidate[1],0,"all sampled proxies skip neighbour traversal and normal reduction");
        }
      }
    } finally {input.destroy();output.destroy();state.destroy();read.destroy();}
  });
});

test("GPU job-local counts match per-face atomics under shared-brick contention", {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
  await withUniformDevice("mesh job counts", async device => {
    for (const retainCasters of [false,true]) {
      const source=svoSurfaceMeshWGSL(3,128,true,true,retainCasters);
      const append=`fn meshAppend(${source.split("fn meshAppend(")[1]!.split("\n}")[0]!}\n}`;
      const header=/* wgsl */ `
struct SurfaceQuad { origin:vec3u, identity:u32, extent:vec3u, face:u32 }
@group(0) @binding(0) var<storage,read_write> meshWork:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> meshState:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> meshArena:array<SurfaceQuad>;
var<private> meshPhase:u32;var<private> meshJobCount:u32;var<private> meshEntry:u32;
var<private> meshEmitBase:u32;var<private> meshEmitCount:u32;
fn meshScratchAt(i:u32)->u32{return i;}
fn meshListAt(i:u32)->u32{return 0u;}
fn meshTableAt(i:u32)->u32{return 0u;}
fn sceneIdentityMaterial(i:u32)->u32{return i&65535u;}
struct Material {identity:vec4u}
fn dryMaterial(i:u32)->Material{return Material(vec4u(0u,0u,0u,select(0u,4u,i==2u)));}
`;
      const pipelines=await Promise.all([false,true].map(async batched=>{
        const body=batched?append:append.replace("meshJobCount+=1u;","atomicAdd(&meshWork[meshScratchAt(meshEntry)],1u);");
        const module=device.createShaderModule({code:`${header}${body}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 meshPhase=0u;meshEntry=id.x%32u;meshJobCount=0u;
 let counts=array<u32,5>(0u,1u,4u,63u,1000u);
 for(var i=0u;i<counts[id.x%5u];i++){
  meshAppend(vec3u(i),vec3u(1u),0u,1u+(i%3u));
 }
 ${batched?"if(meshJobCount!=0u){atomicAdd(&meshWork[meshScratchAt(meshEntry)],meshJobCount);}":""}
}`});
        assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
        return device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      }));
      const work=device.createBuffer({size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      const state=device.createBuffer({size:1024,usage:GPUBufferUsage.STORAGE});
      const arena=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE});
      const read=device.createBuffer({size:128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      try {
        const results:Uint32Array[]=[];
        for(const pipeline of pipelines){
          const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[work,state,arena].map((buffer,binding)=>({binding,resource:{buffer}}))});
          const encoder=device.createCommandEncoder();encoder.clearBuffer(work);const pass=encoder.beginComputePass();
          pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(16);pass.end();
          encoder.copyBufferToBuffer(work,0,read,0,128);device.queue.submit([encoder.finish()]);
          await read.mapAsync(GPUMapMode.READ);results.push(new Uint32Array(read.getMappedRange()).slice());read.unmap();
        }
        assert.deepEqual(results[1],results[0]);
        assert.ok(results[0]!.every(count=>count>0));
      } finally {work.destroy();state.destroy();arena.destroy();read.destroy();}
    }
  });
});
