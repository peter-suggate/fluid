import assert from "node:assert/strict";
import test from "node:test";
import { liveSvoRadianceFeedbackWGSLFor } from "../lib/svo/features/radiance/webgpu-svo-live-derived-builder";
import { radianceLightSamplingWGSL } from "../lib/svo/features/radiance/radiance-light-sampling";
import { svoLightWGSL } from "../lib/svo/contracts/svo-light-abi";
import { drySceneShader } from "../lib/svo/features/shading/program";
import { withUniformDevice } from "./helpers/uniform-geometric";

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};
async function run(device:GPUDevice,code:string,size:number,lightData:Uint32Array):Promise<Float32Array>{
  const module=device.createShaderModule({code});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const lights=device.createBuffer({size:lightData.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(lights,0,lightData as Uint32Array<ArrayBuffer>);
  const output=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:lights}},{binding:1,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange()).slice();read.unmap();return result;
  }finally{lights.destroy();output.destroy();read.destroy();}
}
function lights(kind:number):Uint32Array{
  const words=new Uint32Array(32*28),f=new Float32Array(words.buffer);
  for(let i=0;i<32;i++){
    const o=i*28;f.set([0,2,0,0],o);f.set([0,kind===1?1:-1,0,.5],o+4);f.set([1,1,1,i+1],o+8);f.set([1,0,0,1],o+12);f.set([0,0,1,1],o+16);f.set([.2,.8,0,0],o+20);words.set([kind,i,0,1],o+24);
  }return words;
}
const bindings=`@group(0) @binding(0) var<storage,read> lights:array<SvoLightRecord>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;`;

test("GPU radiance injection covers32 lights, preserves fractional visibility, and averages both area samples",gpu,async()=>{
  const source=liveSvoRadianceFeedbackWGSLFor();
  const incident=source.slice(source.indexOf("fn directIncident("),source.indexOf("/**\n * Outward normal",source.indexOf("fn directIncident(")));
  const visibility=source.slice(source.indexOf("fn directVisibility("),source.indexOf("fn directIncident("));
  await withUniformDevice("radiance injection light contract",async device=>{
    const code=`${svoLightWGSL}${radianceLightSamplingWGSL}${bindings}
const PI=3.141592653589793;
struct Params{direct:vec4u,mappingCellSize:vec4f} var<private> params:Params;
var<private> coverage:f32;var<private> sampleCalls:u32;
struct SceneSample{coverage:f32}
fn radianceFloorLevel()->u32{return 0u;}
fn sceneSample(p:vec3f,l:u32)->SceneSample{sampleCalls++;return SceneSample(coverage);}
${visibility}${incident}
@compute @workgroup_size(1) fn main(){
 params.direct=vec4u(4,0,0,32);params.mappingCellSize=vec4f(.1);
 coverage=0.0;sampleCalls=0u;output[0]=vec4f(directIncident(vec3f(0),vec3f(0),vec3f(0,1,0)),f32(sampleCalls));
 coverage=.5;sampleCalls=0u;output[1]=vec4f(directIncident(vec3f(0),vec3f(0),vec3f(0,1,0)),f32(sampleCalls));
 coverage=1.0;sampleCalls=0u;output[2]=vec4f(directIncident(vec3f(0),vec3f(0),vec3f(0,1,0)),f32(sampleCalls));
 params.direct.w=4u;coverage=0.0;sampleCalls=0u;output[3]=vec4f(directIncident(vec3f(0),vec3f(0),vec3f(0,1,0)),f32(sampleCalls));
}`;
    for(const kind of [1,4]){
      const result=await run(device,code,64,lights(kind));
      const factor=kind===1?1:16/(4+.45**2+.2**2)**2;
      assert.ok(Math.abs(result[0]!-528*factor/Math.PI)<1e-4);
      assert.ok(Math.abs(result[4]!-result[0]!/16)<1e-5,"four half-opacity samples transmit1/16, not all incident light");
      assert.equal(result[8],0);
      assert.ok(Math.abs(result[12]!-10*factor/Math.PI)<1e-5);
      assert.equal(result[3],kind===1?128:256,"every source sample has its own visibility");
      assert.equal(result[15],kind===1?16:32);
    }
  });
});

test("GPU radiance finite-source samples match the production direct surface sampler",gpu,async()=>{
  const direct=drySceneShader.slice(drySceneShader.indexOf("struct DryLightSample{"),drySceneShader.indexOf("// The static scene is traceStatic alone"));
  await withUniformDevice("radiance source quadrature parity",async device=>{
    const code=`const UNIFIED_PI=3.141592653589793;${svoLightWGSL}${radianceLightSamplingWGSL}${direct}${bindings}
@compute @workgroup_size(1) fn main(){for(var i=0u;i<5u;i++){var light=lights[i];light.identity.x=i+1u;for(var s=0u;s<2u;s++){let a=radianceLightSample(light,s,vec3f(.3,0,.1));let b=dryLightSample(light,s,vec3f(.3,0,.1));output[i*4u+s*2u]=vec4f(a.towardLight-b.towardLight,a.finiteDistance_m-b.finiteDistance_m);output[i*4u+s*2u+1u]=vec4f(a.radiance-b.radiance,f32(a.valid)-f32(b.valid));}}}`;
    const result=await run(device,code,320,lights(2));
    assert.ok([...result].every(v=>v===0),"all five source kinds, both quadrature points and finite endpoints agree");
  });
});
