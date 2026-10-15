import assert from 'node:assert/strict';
import test from 'node:test';
import {narrowBandRedistanceWGSL} from '../lib/methods/uniform/uniform-narrow-band-redistance.wgsl';
import {withUniformDevice} from './helpers/uniform-geometric';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';

// Compile the production functions against a tiny address/sampling fixture.
// The crossing sampler and canonical-vertex sampler deliberately disagree,
// as they can at hanging vertices on an h/4h boundary.
function shaderFunction(name:string){
 const start=narrowBandRedistanceWGSL.indexOf(`fn ${name}(`);assert.ok(start>=0);
 let end=narrowBandRedistanceWGSL.indexOf('{',start)+1,depth=1;
 for(;depth;end++){if(narrowBandRedistanceWGSL[end]==='{')depth++;if(narrowBandRedistanceWGSL[end]==='}')depth--;}
 return narrowBandRedistanceWGSL.slice(start,end);
}
test('redistance uses crossing geometry in cached and uncached paths and never stores a failed distance', {skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('redistance crossing contract',async device=>{
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> fixture:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<vec4f>;
const UM_CELL_KEYS=1024u;const NB_BAND_SEARCH=2100u;const NB_BAND_CANDIDATES=2101u;const NB_BAND_MASK=2102u;
const NB_BAND_REACH=2104u;const NB_DEPTH_A=2200u;const NB_DEPTH_B=2300u;const NB_NO_SURFACE=0xffffffffu;
override nbCachedGeometry:bool=true;
struct Params{hDt:vec4f} var<private> params=Params(vec4f(0.00625));
var<workgroup> nbGeometryBase:u32;var<workgroup> nbGeometryMask:vec2u;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(0);}
fn umCellCoord(c:u32)->vec3u{return vec3u(0);}
fn nbBandTile(c:vec3u)->u32{return 0u;}
fn umCellAt(c:vec3u)->u32{return 0u;}
fn umVertexAt(p:vec3u)->u32{return 0u;}
fn nbCrossingVertex(tile:u32,origin:vec3u,p:vec3u)->f32{return fixture[p.x+2u*(p.y+2u*p.z)]*params.hDt.x;}
fn umVertexValue(p:vec3u)->f32{return 5.0*params.hDt.x;}
fn umSampleVertex(p:vec3f)->f32{return fixture[8]*params.hDt.x;}
fn umCellOpen(c:vec3i)->f32{return 1.0;}
fn nbPhiGradient(c:vec3u,q:vec3f)->vec4f{return vec4f(0,0,0,1);}
fn nbPointNear(c:vec3i,p:vec3f)->vec3f{return p;}
fn nbCellNear(c:vec3i,p:vec3i)->vec3i{return p;}
${['nbGeometryCapacity','nbGeometryRank'].map(shaderFunction).join('\n')}
@compute @workgroup_size(64) ${shaderFunction('depthGeometry')}
${['nbTrilinear','nbSurfaceDistance','nbBuildDistance'].map(shaderFunction).join('\n')}
@compute @workgroup_size(1) fn measure(){
 let p=vec3u(2,0,0);let distance=nbSurfaceDistance(vec3f(p),0u);
 nbBuildDistance(p,vec3u(0),0u);
 out[select(1u,0u,nbCachedGeometry)]=vec4f(distance,bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A])),0,0);
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.deepEqual(errors.map(m=>m.message),[]);
  const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===1?'read-only-storage' as const:'storage' as const}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const cache=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'depthGeometry'}});
  const pipelines=await Promise.all([true,false].map(cached=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'measure',constants:{nbCachedGeometry:Number(cached)}}})));
  const bins=device.createBuffer({size:4*2400,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const fixture=device.createBuffer({size:4*9,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const group=device.createBindGroup({layout,entries:[bins,fixture,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  try{
   const cases=[
    {name:'plane across hanging seam',corners:[-.25,.75,-.25,.75,-.25,.75,-.25,.75],initial:-2,distance:1.75,metric:-1.75},
    {name:'zero face',corners:[0,1,0,1,0,1,0,1],initial:2,distance:2,metric:2},
    {name:'thin corner crossing',corners:[-1e-6,1,1,1,1,1,1,1],initial:2,distance:1.999999,metric:1.999999},
    {name:'liquid without a zero',corners:Array(8).fill(-.1),initial:-2,distance:-1,metric:-5},
    {name:'air without a zero',corners:Array(8).fill(.1),initial:2,distance:-1,metric:5},
    {name:'small positive plateau',corners:Array(8).fill(1e-6),initial:2,distance:-1,metric:5},
    {name:'zero plateau',corners:Array(8).fill(0),initial:2,distance:-1,metric:5},
   ];
   for(const sample of cases){
    const words=new Uint32Array(2400);words[2102]=1;words[2104]=1;device.queue.writeBuffer(bins,0,words);
    device.queue.writeBuffer(fixture,0,new Float32Array([...sample.corners,sample.initial]));
    const e=device.createCommandEncoder();for(const pipeline of [cache,...pipelines]){const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();}device.queue.submit([e.finish()]);
    const result=await readMixedBuffer(device,output);
    for(const offset of [0,4]){
     assert.ok(Math.abs(result[offset]!-sample.distance)<1e-5,`${sample.name}: distance ${result[offset]}`);
     assert.ok(Math.abs(result[offset+1]!-sample.metric)<1e-5,`${sample.name}: metric ${result[offset+1]}`);
    }
    assert.deepEqual(result.subarray(0,4),result.subarray(4,8),`${sample.name}: cache and direct query agree`);
   }
  }finally{bins.destroy();fixture.destroy();output.destroy();}
 });
});
