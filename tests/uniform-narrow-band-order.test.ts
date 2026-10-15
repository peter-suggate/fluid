import assert from 'node:assert/strict';
import test from 'node:test';
import { UniformNarrowBandOrder } from '../lib/methods/uniform/uniform-narrow-band-order';
import { uniformLatticeAddress, uniformCellKey } from '../lib/methods/uniform/uniform-lattice-address';
import { narrowBandParticleStride } from '../lib/methods/uniform/uniform-narrow-band-apic.wgsl';
import { withUniformDevice } from './helpers/uniform-geometric';
import { readMixedBuffer } from './helpers/uniform-mixed-native-fields';

test('particle cell order is independent of input permutation, including crowded cells and equal positions', {skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('deterministic particle ordering',async device=>{
  for(const sparse of [false,true])for(const apic of [false,true]){
   const address=uniformLatticeAddress([32,32,32],false,{blocks:3}),stride=narrowBandParticleStride(apic),n=521;
   const make=(words:number)=>device.createBuffer({size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const topology=make(5*address.tileKeys+address.seed.tail.length),bins=make(2*address.cellKeys+address.tileKeys),links=make(address.cellKeys+4*n),state=make(16),particles=[make(stride*n),make(stride*n)] as const;
   device.queue.writeBuffer(topology,20*address.tileKeys,address.seed.tail);
   const order=new UniformNarrowBandOrder(device,address,topology,particles,bins,links,state,sparse,apic);
   try{
    await order.initialize();const original=new Float32Array(n*stride),counts=new Uint32Array(2*address.cellKeys+address.tileKeys);
    for(let i=0;i<n;i++){
     // Most records collide in one cell; 257 share a position but differ in
     // velocity/history. The second run spans workgroups and another tile.
     const x=i<400?1.25:5.25,y=1.25,z=1.25;
     original.set([x,y,z,1],i*stride);for(let k=4;k<stride;k++)original[i*stride+k]=((i%13)*17+k)%1031;
     const age=(i%11)*0.125,mode=i%2;
     original[i*stride+11]=apic?mode:mode?-age:age;
     if(apic)original[i*stride+stride-4]=age;
     const cell=uniformCellKey(address,Math.floor(x),Math.floor(y),Math.floor(z));counts[cell]++;counts[2*address.cellKeys+Math.floor(cell/64)]++;
    }
    const bits=new Uint32Array(original.buffer);
    const logical=(i:number)=>{
     const record=Array.from(bits.subarray(i*stride,(i+1)*stride));
     if(!apic){const packed=record[11]!;record[11]=packed>>>31?0x3f800000:0;record.push(packed&0x7fffffff,0,0,0);}
     return record;
    };
    const canonical=Array.from({length:n},(_,i)=>i).sort((a,b)=>{
     const pa=original.subarray(a*stride,a*stride+3),pb=original.subarray(b*stride,b*stride+3);
     const cell=(p:Float32Array)=>uniformCellKey(address,Math.floor(p[0]!),Math.floor(p[1]!),Math.floor(p[2]!));
     const delta=cell(pa)-cell(pb);if(delta)return delta;
     const x=logical(a),y=logical(b);for(let k=0;k<x.length;k++)if(x[k]!==y[k])return x[k]!-y[k]!;return a-b;
    });
    const expectedOrder=new Uint32Array(n*stride);canonical.forEach((i,k)=>expectedOrder.set(bits.subarray(i*stride,(i+1)*stride),k*stride));
    let baseline:Uint32Array|undefined;
    for(const reverse of [false,true]){
     const data=original.slice();if(reverse)for(let i=0;i<n;i++)data.set(original.subarray((n-1-i)*stride,(n-i)*stride),i*stride);
     device.queue.writeBuffer(particles[0],0,data);device.queue.writeBuffer(bins,0,counts);device.queue.writeBuffer(state,0,new Uint32Array([n,0,0,0]));
     const e=device.createCommandEncoder();order.encode(e,1);device.queue.submit([e.finish()]);
     const actual=new Uint32Array((await readMixedBuffer(device,particles[1])).buffer);
     if(baseline)assert.deepEqual(actual,baseline,`sparse=${sparse}, apic=${apic}`);else baseline=actual;
     assert.deepEqual(actual,expectedOrder,'ordering retains unpacked mode-before-age tie breaks');
     const records=Array.from({length:n},(_,i)=>Array.from(actual.subarray(i*stride,(i+1)*stride)).join(','));
     const expected=new Uint32Array(original.buffer);
     assert.deepEqual(records.sort(),Array.from({length:n},(_,i)=>Array.from(expected.subarray(i*stride,(i+1)*stride)).join(',')).sort(),'every particle survives exactly once');
     const positions=new Uint32Array((await readMixedBuffer(device,links)).buffer);
     for(let i=0;i<n;i++)assert.deepEqual(positions.subarray(address.cellKeys+4*i,address.cellKeys+4*i+4),actual.subarray(stride*i,stride*i+4),'compact position cache follows sorted particles');
     assert.equal(new Uint32Array((await readMixedBuffer(device,state)).buffer)[1],n);
    }
   }finally{order.destroy();for(const b of [topology,bins,links,state,...particles])b.destroy();}
  }
 });
});
