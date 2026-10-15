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
