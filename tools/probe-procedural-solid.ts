/** Real-GPU contract probe for boundary value, gradient, capacity and contact.
 * Run under the shared GPU lock; no fluid tuning or frame evolution involved. */
import assert from "node:assert/strict";
import { bakeProceduralSolid, ProceduralSolidField, PROCEDURAL_SOLID_MAGIC } from "../lib/core/procedural-solid";
import { uniformSolidBoundaryWGSL } from "../lib/methods/uniform/uniform-solid-boundary.wgsl";
import { narrowBandTerrainContactWGSL } from "../lib/methods/uniform/uniform-particle-contact.wgsl";
import { geometricPlaneBoxWGSL } from "../lib/core/geometric-plane-box.wgsl";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";

const geometry=bakeProceduralSolid([16,16,16],[.02,.02,.02],([x,y,z])=>y-8+x/16+z/32);
const field=new ProceduralSolidField(geometry.solidBoundary,[]), packed=field.pack();
const image=new Uint32Array(16+packed.length+2);image.set(packed,16);image.set([16,PROCEDURAL_SOLID_MAGIC],image.length-2);
const points=[[8.5,7.5,.5],[7.99,7.4,7.99],[8.01,7.4,8.01],[15.75,6.8,15.75]] as const;
const input=new Float32Array(points.flatMap(p=>[...p,0]));
await withUniformDevice("procedural solid contract",async device=>{
  const buffers:GPUBuffer[]=[];
  const buffer=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});buffers.push(b);return b;};
  try {
    const mask=buffer("boundary",image.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const queries=buffer("queries",input.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const output=buffer("results",points.length*48,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const read=buffer("read",output.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(mask,0,image);device.queue.writeBuffer(queries,0,input);
    const code=`
@group(0) @binding(0) var<storage,read> umSolidScratch:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> result:array<vec4f>;
struct Params{hDt:vec4f} const params=Params(vec4f(.02,.02,.02,1.0/60.0));
fn umFramePage()->vec3i{return vec3i(0);}
fn umSolidValid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(16));}
fn umSolidVoxel(p:vec3i)->bool{return !umSolidValid(p);}
fn umSolidHasTerrain()->bool{return false;}
fn umSolidDimensions()->vec3u{return vec3u(16);}
fn umSolidTerrainHeight(x:i32,z:i32)->f32{return 0.0;}
fn umTerrainSurface(p:vec3f)->vec3f{return vec3f(0);}
fn umBodyAt(p:vec3f)->i32{return -1;}
fn umSolidTraceWorld(p:vec3f)->vec3f{return p;}
fn umCellOpen(p:vec3i)->f32{return umBoundaryBoxOpen(vec3f(p)+.5,vec3f(1));}
${geometricPlaneBoxWGSL.replaceAll("geometric","umTerrain")}
${uniformSolidBoundaryWGSL}
${narrowBandTerrainContactWGSL}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;let q=points[i].xyz;
 result[3u*i]=umBoundarySample(q);
 result[3u*i+1u]=vec4f(umBoundaryBoxOpen(floor(q)+.5,vec3f(1)),umBoundaryBoxOpen(floor(q)+vec3f(1,.5,.5),vec3f(0,1,1)),0,0);
 let end=vec3f(8.5,7.3,.5);let start=end+vec3f(0,.5,0);
 let hit=nbBoundaryHit(start,end);let contact=nbProceduralContact(start,end,vec3f(0,-1,0));
 // Positive endpoints with a negative interior: (t-.5)^2-.01.
 let ridge=nbBoundaryCubicHit(vec4f(.24,-.0933333333,-.0933333333,.24));
 result[3u*i+2u]=vec4f(hit,umBoundarySample(contact.position).w,ridge,dot(contact.velocity,normalize(vec3f(.0625,1,.03125))));
}`;
    const module=device.createShaderModule({code});
    const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}:${m.message}`),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[mask,queries,output].map((b,binding)=>({binding,resource:{buffer:b}}))});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(points.length);pass.end();
    encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=new Float32Array(read.getMappedRange().slice(0));read.unmap();let maxError=0;
    points.forEach((p,i)=>{
      const expected=field.sample(p)!;
      expected.forEach((v,a)=>{const error=Math.abs(v-data[12*i+a]!);maxError=Math.max(maxError,error);assert.ok(error<3e-6,`sample ${i}.${a}: ${error}`);});
      assert.ok(Math.abs(field.open(p.map(Math.floor) as [number,number,number])!-data[12*i+4]!)<3e-6);
      assert.ok(Math.abs(data[12*i+8]!-.69375)<2e-4,`contact time ${data[12*i+8]}`);
      assert.ok(data[12*i+9]!>=0,"contact must leave the plane");
      assert.ok(Math.abs(data[12*i+10]!-.4)<2e-4,"interior crossing with clear endpoints");
      assert.ok(Math.abs(data[12*i+11]!)<2e-6,"no inward normal velocity");
    });
    console.log(JSON.stringify({passed:true,points:points.length,maxSampleError:maxError,contact:[...data.slice(8,12)]}));
  } finally { for(const b of buffers)b.destroy(); }
});
