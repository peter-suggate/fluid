import assert from "node:assert/strict";
import test from "node:test";
import {surfaceExtractionShader} from "../lib/core/webgpu-water-pipeline";

// The extraction's wall contact vertices, from the production WGSL, on a
// dense 16^3 lattice (x/z/upper halo closed) holding a tilted free surface.
// A contact vertex lies on the wall at the height the free surface beside it
// meets the wall, not at its open cell's centre: the contact line followed
// the cell centres and stepped a cell at a time along a wall in coarse tanks.
function contactFunctions(){
 const start=surfaceExtractionShader.indexOf("fn surfaceHaloClearance(");
 const last=surfaceExtractionShader.indexOf("fn surfaceContactLattice(");
 assert.ok(start>=0&&last>start);
 return surfaceExtractionShader.slice(start,surfaceExtractionShader.indexOf("\n}\n",last)+3);
}
// The free surface in cell coordinates: the liquid lies below y = H(z).
const H=(z:number)=>6.3+.07*z;

test("wall contact vertices meet the free surface on the wall and close wall corners",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import("./helpers/uniform-geometric");
 // onSurface: the vertex lies on the free surface, within .1 cell of expected
 // along the wall (one step down the in-plane gradient may slide it a little).
 type Case={a:number[],b:number[],expected:number[],label:string,onSurface?:boolean};
 const cases:Case[]=[];
 // Lattice p reads cell p - 1; the topmost liquid lattice point of a column.
 const top=(z:number)=>Math.floor(H(z-1)+.5);
 for(let z=2;z<=13;z++){
  const y=top(z),wall=H(z-1)+.5;
  cases.push({a:[1,y,z],b:[0,y,z],expected:[.5,wall,z],label:`low x wall z=${z}`,onSurface:true});
  cases.push({a:[16,y,z],b:[17,y,z],expected:[16.5,wall,z],label:`high x wall z=${z}`,onSurface:true});
  cases.push({a:[0,y-1,z],b:[1,y-1,z],expected:[.5,y-1,z],label:`submerged z=${z}`});
  cases.push({a:[5,y,z],b:[5,y+1,z],expected:[5,wall,z],label:`open edge z=${z}`});
 }
 // A vertical tank corner: the contact slides into it along both walls.
 cases.push({a:[1,top(1),1],b:[0,top(1),1],expected:[.5,H(0)+.5,.5],label:"corner"});
 cases.push({a:[1,top(1)-2,1],b:[1,top(1)-2,0],expected:[.5,top(1)-2,.5],label:"submerged corner"});
 await withUniformDevice("water wall contact",async device=>{
  const code=`
struct View{gridInfo:vec4f,container:vec4f}
struct Case{a:vec4f,b:vec4f}
@group(0) @binding(0) var<storage,read> inputs:array<Case>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
@group(0) @binding(2) var<storage,read> umTopology:array<u32>;
const UM_SOLID=1u;const UM_BULK16=2u;
var<private> u:View=View(vec4f(16,16,16,0),vec4f(16,16,16,0));
fn umPresentationEnabled()->bool{return arrayLength(&umTopology)>1u;}
fn uniformSurfaceCell(p:vec3i)->vec3i{return p-vec3i(1);}
fn umCellKind(p:vec3i)->u32{return 0u;}
fn surfaceHasSolids()->bool{return false;}
fn surfaceSolidClearance(q:vec3f)->f32{return 1e6;}
fn freeFieldCell(cell:vec3i)->f32{return 0.5-((f32(cell.y)+0.5)-(6.3+0.07*f32(cell.z)));}
${contactFunctions()}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let index=id.x;if(index>=arrayLength(&inputs)){return;}
 let c=inputs[index];let a=c.a.xyz;let b=c.b.xyz;
 let ends=vec2f(surfaceHaloClearance(vec3i(a)),surfaceHaloClearance(vec3i(b)));
 let la=freeFieldCell(vec3i(a)-vec3i(1));let lb=freeFieldCell(vec3i(b)-vec3i(1));
 let t=clamp((0.5-la)/(lb-la),0.0,1.0);
 output[index]=vec4f(surfaceContactLattice(a,b,t,1.0,ends),0);
}`;
  const module=device.createShaderModule({code});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const input=device.createBuffer({size:cases.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:cases.length*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const topology=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(input,0,new Float32Array(cases.flatMap(c=>[...c.a,0,...c.b,0])));
  try{
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);
   pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}},{binding:2,resource:{buffer:topology}}]}));
   pass.dispatchWorkgroups(Math.ceil(cases.length/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange()).slice();read.unmap();
   cases.forEach((c,i)=>{
    const got=Array.from(values.slice(4*i,4*i+3));
    if(c.onSurface){
     assert.ok(Math.abs(got[0]!-c.expected[0]!)<1e-6,`${c.label}: ${got} on the wall`);
     assert.ok(Math.abs(got[1]!-.5-H(got[2]!-1))<1e-4,`${c.label}: ${got} on the free surface`);
     assert.ok(Math.abs(got[2]!-c.expected[2]!)<.1&&Math.abs(got[1]!-c.expected[1]!)<.1,`${c.label}: ${got} expected ${c.expected}`);
    }else for(let k=0;k<3;k++)assert.ok(Math.abs(got[k]!-c.expected[k]!)<1e-4,`${c.label}: ${got} expected ${c.expected}`);
   });
  }finally{input.destroy();output.destroy();topology.destroy();read.destroy();}
 });
});
