import assert from 'node:assert/strict';
import test from 'node:test';
import {uniformDetailModule,uniformDetailPipeline} from '../lib/methods/uniform/uniform-detail-fields';
import {uniformWindowAddressWGSL} from '../lib/methods/uniform/uniform-window-address';
import {uniformLatticeAddress} from '../lib/methods/uniform/uniform-lattice-address';
import type {UniformMixedOwnership} from '../lib/methods/uniform/uniform-mixed-ownership';

function harness(){
 const descriptors:GPUComputePipelineDescriptor[]=[];
 const device={
  createShaderModule:(d:GPUShaderModuleDescriptor)=>({code:d.code}),
  createComputePipelineAsync:async(d:GPUComputePipelineDescriptor)=>{descriptors.push(d);return {};},
 } as unknown as GPUDevice;
 const module=uniformDetailModule(device,{code:uniformWindowAddressWGSL('pages','0u','3u')+uniformWindowAddressWGSL('oldPages','0u','3u','oldum')});
 const descriptor:GPUComputePipelineDescriptor={layout:'auto',compute:{module,entryPoint:'unused',constants:{existing:7}}};
 return {device,module,descriptor,descriptors};
}

test('standalone ownership keeps runtime header interpretation through reseeding',async()=>{
 const h=harness(),a=uniformLatticeAddress([32,32,32]);
 const ownership={fixedWorldGeometry:false,capacity:{address:a}} as UniformMixedOwnership;
 await uniformDetailPipeline(h.device,ownership,h.descriptor);
 const at=a.seed.layout.tail.window+24;
 a.seed.tail.set([1,2,3,4,5,6,63,0xffffffff],at);
 await uniformDetailPipeline(h.device,ownership,h.descriptor);
 for(const d of h.descriptors)assert.deepEqual(d.compute.constants,{existing:7});
 const code=(h.module as unknown as {code:string}).code;
 assert.match(code,/override umFixedWorldGeometry:bool=false/);
 assert.match(code,/select\(pages\[0u\+262168u\],umWindowWord24,umFixedWorldGeometry\)/);
});

test('fixed world specializes both ownership prefixes and preserves explicit caller constants',async()=>{
 const h=harness(),a=uniformLatticeAddress([64,32,32],true,{}, {boxPage:[30,0,30]});
 const ownership={fixedWorldGeometry:true,capacity:{address:a}} as UniformMixedOwnership;
 await uniformDetailPipeline(h.device,ownership,h.descriptor);
 const c=h.descriptors[0]!.compute.constants!;
 for(const prefix of ['um','oldum']){
  assert.equal(c[prefix+'FixedWorldGeometry'],1);
  for(let i=24;i<32;i++)assert.equal(c[prefix+'WindowWord'+i],a.seed.tail[a.seed.layout.tail.window+i]);
 }
 assert.equal(c.existing,7);
 assert.equal(Object.keys(c).some(k=>/WindowWord[0-9]$/.test(k)),false,'frame words never specialize');
 await uniformDetailPipeline(h.device,ownership,{...h.descriptor,compute:{...h.descriptor.compute,constants:{umFixedWorldGeometry:0}}});
 assert.equal(h.descriptors[1]!.compute.constants!.umFixedWorldGeometry,0);
});

test('paged scenes share modules and pipelines while bounds and capacity remain runtime data', async () => {
 const {managedGPUDevice,gpuCompilationManagerFor}=await import('../lib/core/gpu-compilation-manager');
 const {uniformMixedTopologyWGSL}=await import('../lib/methods/uniform/uniform-mixed-topology.wgsl');
 let modules=0,pipelines=0;
 const device=managedGPUDevice({
  createShaderModule:(d:GPUShaderModuleDescriptor)=>{modules++;return {code:d.code};},
  createComputePipelineAsync:async(d:GPUComputePipelineDescriptor)=>{pipelines++;return {descriptor:d};},
 } as unknown as GPUDevice,{requireWorkerRealm:false});
 const build=async(cells:number[],openTop:boolean)=>{
  const address=uniformLatticeAddress(cells,openTop);
  const lattice={dimensions:cells,cellSize_m:[0.1,0.1,0.1]} as unknown as UniformMixedOwnership['capacity']['lattice'];
  const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL({lattice,tiles:address.tileKeys},0)+
   '@compute @workgroup_size(1) fn main(){umSupport[0]=UM_TILES+u32(umWorldHigh().x);}'});
  const ownership={fixedWorldGeometry:true,capacity:{address}} as UniformMixedOwnership;
  return {module,pipeline:await uniformDetailPipeline(device,ownership,{layout:'auto',compute:{module,entryPoint:'main'}}),address};
 };
 const a=await build([32,32,32],false),b=await build([96,64,32],true);
 assert.notEqual(a.address.tileKeys,b.address.tileKeys);
 assert.equal(a.module,b.module);
 assert.equal(a.pipeline,b.pipeline);
 assert.equal(modules,1);assert.equal(pipelines,1);
 const code=(a.module as unknown as {code:string}).code;
 assert.match(code,/fn UM_TILES\(\)->u32\{return umCounts.dimensions.w;\}/);
 assert.match(code,/umCounts.window\[6u\]\[0u\]/);
 assert.doesNotMatch(code,/WindowWord|override umFixedFrame/);
 const descriptor=(a.pipeline as unknown as {descriptor:GPUComputePipelineDescriptor}).descriptor;
 assert.deepEqual(descriptor.compute.constants,{umFixedWorldGeometry:1});
 assert.equal(gpuCompilationManagerFor(device).snapshot().cacheStats.pipelineHits,1);
});

test('hot modules specialize immutable bounds at construction while movable frames stay runtime',async()=>{
 const {uniformMixedTopologyWGSL}=await import('../lib/methods/uniform/uniform-mixed-topology.wgsl');
 const descriptors:GPUComputePipelineDescriptor[]=[];
 const device={createShaderModule:(d:GPUShaderModuleDescriptor)=>({code:d.code}),createComputePipelineAsync:async(d:GPUComputePipelineDescriptor)=>{descriptors.push(d);return {};}} as unknown as GPUDevice;
 const address=uniformLatticeAddress([64,32,32],true,{}, {boxPage:[30,0,30]});
 const lattice={dimensions:address.cells,cellSize_m:[0.1,0.1,0.1]} as unknown as UniformMixedOwnership['capacity']['lattice'];
 const module=uniformDetailModule(device,{specializeWorldGeometry:true,code:uniformMixedTopologyWGSL({lattice,tiles:address.tileKeys},0)+'@compute @workgroup_size(1) fn main(){umSupport[0]=UM_TILES+u32(umWorldHigh().x);}'});
 const code=(module as unknown as {code:string}).code;
 assert.match(code,/override umWindowWord24:u32=0u/);
 assert.match(code,/fn UM_TILES\(\)->u32\{return umCounts.dimensions.w;\}/,'capacity remains runtime');
 const descriptor:GPUComputePipelineDescriptor={layout:'auto',compute:{module,entryPoint:'main'}};
 for(const fixedWorldGeometry of [true,false])await uniformDetailPipeline(device,{fixedWorldGeometry,capacity:{address}} as UniformMixedOwnership,descriptor);
 const fixed=descriptors[0]!.compute.constants!;
 assert.equal(fixed.umFixedWorldGeometry,1);
 for(let i=24;i<32;i++)assert.equal(fixed['umWindowWord'+i],address.seed.tail[address.seed.layout.tail.window+i]);
 assert.equal(fixed.umFixedFrame,undefined,'movable frame is never frozen by world specialization');
 assert.equal(descriptors[1]!.compute.constants,undefined,'mutable ownership reads the uniform even in a hot module');
});
