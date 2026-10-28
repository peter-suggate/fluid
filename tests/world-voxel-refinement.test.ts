import assert from "node:assert/strict";
import test from "node:test";
import { worldVoxelRefinementWGSL } from "../tools/world-voxel-refinement.wgsl";

test("GPU structural voxel refinement recovers a sloped plane and preserves unknown, seam and discontinuity fallbacks",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 await withUniformDevice("world voxel scalar refinement",async device=>{
  const code=`
   @group(0) @binding(0) var field:texture_3d<f32>;
   @group(0) @binding(1) var<storage,read_write> output:array<vec4f,7>;
   var<private> mode:u32;
   fn worldVoxelCellSize()->vec3f{return vec3f(.5);}
   fn worldVoxelOrigin()->vec3f{return vec3f(-2,0,-2);}
   fn worldVoxelSample(p:vec3f)->WorldVoxelSample{
    let cell=vec3i(floor((p-worldVoxelOrigin())/.5));if(any(cell<vec3i(0))||any(cell>=vec3i(8))){return WorldVoxelSample();}
    if(mode==1u&&cell.x>=4){return WorldVoxelSample();}
    var value=textureLoad(field,cell,0).r;
    if(mode==6u){let q=(p-vec3f(-.25,1.25,-.25))/.5;value=-(q.x-.501)*(q.y-.51)*(q.z-.9);}
    if(mode==3u&&cell.x>=4){value+=3.0;}
    if(mode==4u){value=1.0;}
    let material=select(0u,select(7u,8u,mode==2u&&cell.x>=4),value<=0.0);
    return WorldVoxelSample(value,material,1u,select(1.0,2.0,mode==5u&&cell.x>=4));
   }
   ${worldVoxelRefinementWGSL}
   @compute @workgroup_size(1) fn main(){
    for(var i=0u;i<6u;i+=1u){mode=i;let hit=worldVoxelRefine(vec3f(.1,3,.1),vec3f(0,-1,0),1.5,7u);output[i]=vec4f(hit.normal,select(-hit.t,hit.t,hit.valid!=0u));}
    mode=6u;let narrow=worldVoxelRefine(vec3f(-.25,1.25,-.25),normalize(vec3f(1)),.45,7u);output[6]=vec4f(narrow.normal,select(-narrow.t,narrow.t,narrow.valid!=0u));
   }`;
  const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const field=device.createTexture({size:[8,8,8],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const values=new Float32Array(512);const magnitude=Math.sqrt(1+.2**2+.15**2);
  for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){const px=-2+(x+.5)*.5,py=(y+.5)*.5,pz=-2+(z+.5)*.5;values[x+8*(y+8*z)]=(py-(1.45+.2*px-.15*pz))/magnitude;}
  device.queue.writeTexture({texture:field},values,{bytesPerRow:32,rowsPerImage:8},[8,8,8]);
  const output=device.createBuffer({size:112,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:112,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:field.createView()},{binding:1,resource:{buffer:output}}]});const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,112);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange()).slice();read.unmap();
   const expected=[-.2/magnitude,1/magnitude,.15/magnitude,3-(1.45+.2*.1-.15*.1)];expected.forEach((v,i)=>assert.ok(Math.abs(result[i]!-v)<.0001,`${i}: ${result[i]} vs ${v}`));
   assert.ok(Math.abs(result[27]!-.501*Math.sqrt(.75))<.0001,"derivative brackets preserve the narrow first crossing, not the later .9 root");
   for(let i=1;i<6;i++)assert.deepEqual([...result.slice(i*4,i*4+4)],[0,0,0,-1.5],`mode ${i} retains original receiver`);
  }finally{field.destroy();output.destroy();read.destroy();}
 });
});
