/** A 2:1 neighbour is displaced tangentially as well as normally. An affine
 * density patch must not turn that tangential offset into a normal derivative.
 * The production stencil is tested independently of the sharpening weight,
 * density transport, pressure, and time integration.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
const source=readFileSync(new URL('../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts',import.meta.url),'utf8');
const stats=source.match(/fn sharpeningStats\([\s\S]*?\n}/)?.[0];assert.ok(stats);
const struct=source.match(/struct SharpeningStats \{[\s\S]*?\n}/)?.[0];assert.ok(struct);
const live=new Set<GPU>();Object.assign(globalThis,{sharpeningSeamAffineGPUs:live});
(process.env.WEBGPU_NODE_MODULE?test:test.skip)('sharpening reproduces affine directional derivatives at a 2:1 surface seam',async()=>{
 await acquireWebGPUExclusiveLock('dawn-test','sharpening-seam-affine');let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);live.add(gpu);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const gradients=[[0,.1,0],[0,0,.1],[0,-.1,0],[.1,.1,0],[0,0,0],[.1,0,0]];
  const caseCount=6*gradients.length;
  const code=`
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
const centres=array<vec3f,8>(vec3f(0),vec3f(-1.5,-.5,-.5),vec3f(1,0,0),
  vec3f(0,-1,0),vec3f(0,1,0),vec3f(0,0,-1),vec3f(0,0,1),vec3f(0,-1,-1));
const cells=array<u32,15>(1u,7u,5u,3u,0u,0u,2u,3u,0u,0u,4u,5u,0u,0u,6u);
const offsets=array<u32,6>(0u,5u,7u,9u,11u,13u);
const ownTerms=array<u32,6>(4u,5u,8u,9u,12u,13u);
const gradients=array<vec3f,${gradients.length}>(${gradients.map(g=>`vec3f(${g.map(x=>x.toFixed(2)).join(',')})`).join(',')});
var<private>gradient:vec3f;var<private>mirrored:bool;var<private>rotation:u32;
fn cellCenter(c:u32)->vec3f{
 var v=centres[c];if(mirrored){v.x=-v.x;}
 if(rotation==1u){return v.yzx;}if(rotation==2u){return v.zxy;}return v;
}
fn conditionedDensity(c:u32)->f32{return .3+dot(gradient,cellCenter(c));}
// Analytic interpolation oracle: isolates the production neighbour geometry.
fn sampleSharpeningDensity(p:vec3f)->f32{return .3+dot(gradient,p);}
fn incidenceBegin(c:u32)->u32{_=c;return 0u;}
fn incidenceEnd(c:u32)->u32{_=c;return 6u;}
fn incidenceRow(at:u32)->u32{return at;}
fn incidenceTerm(at:u32)->u32{return ownTerms[at];}
fn rowAccepted(r:u32)->bool{_=r;return true;}
fn rowKind(r:u32)->u32{return select(0u,2u,r==0u);}
fn rowAxis(r:u32)->u32{return (r/2u+3u-rotation)%3u;}
fn rowArea(r:u32)->f32{return select(1.0,4.0,r==0u);}
fn rowDistance(r:u32)->f32{return select(1.0,1.5,r==0u);}
fn rowTermOffset(r:u32)->u32{return offsets[r];}
fn rowTermCount(r:u32)->u32{return select(2u,5u,r==0u);}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(t:u32)->u32{return cells[t];}
fn termCoefficient(t:u32)->f32{
 let sign=select(1.0,-1.0,mirrored&&t<7u);
 if(t<5u){return sign*select(1.0/6.0,-2.0/3.0,t==0u);}
 return sign*select(-1.0,1.0,((t-5u)%2u)==1u);
}
fn cellActive(c:u32)->bool{_=c;return true;}
${struct}
${stats}
@compute @workgroup_size(1)fn main(@builtin(global_invocation_id)id:vec3u){
 rotation=id.x/12u;mirrored=((id.x/6u)%2u)==1u;gradient=gradients[id.x%6u];let s=sharpeningStats(0u);let rho=conditionedDensity(0u);
 result[2u*id.x]=vec4f((vec3f(rho)-s.negativeDensity/s.negativeArea)/(s.negativeDistance/s.negativeArea),0);
 result[2u*id.x+1u]=vec4f((s.positiveDensity/s.positiveArea-vec3f(rho))/(s.positiveDistance/s.positiveArea),0);
}`;
  const shader=device.createShaderModule({code});assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
  const output=device.createBuffer({size:caseCount*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(caseCount);pass.end();
  encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
  const values=[...new Float32Array(read.getMappedRange())];read.unmap();read.destroy();output.destroy();
  for(let i=0;i<caseCount;i++)for(let side=0;side<2;side++)for(let axis=0;axis<3;axis++){
    const g=gradients[i%gradients.length]!;
    const actual=values[8*i+4*side+axis]!;
    assert.ok(Math.abs(actual-g[axis]!)<1e-6,`case ${i}, gradient ${g}, side ${side}, axis ${axis}: ${actual}, expected ${g[axis]}`);
  }
 }finally{device?.destroy();live.clear();await releaseWebGPUExclusiveLock();}
});
