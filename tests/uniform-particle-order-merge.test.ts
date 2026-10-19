import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { particleOrderFixture, particleOrderInput, type ParticleOrderPattern } from "./helpers/uniform-particle-order";

test(`unified particle merge preserves exact canonical records across occupancy, ties, parity and sparse scans`,{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("particle order merge",async device=>{
  for(const apic of [false,true])for(const sparse of [false,true]){
   const f=await particleOrderFixture(device,16384,apic,sparse);
   try{
    let round=0;
    for(const pattern of ["positions","ties","identical","late"] as ParticleOrderPattern[]){
     for(const occupancy of [1,3,9,17,33,63,64,65,127,128,129,255,256,257,4096]){
      const count=occupancy<=257?Math.min(4096,occupancy*19+3):8193;
      const data=particleOrderInput(count,occupancy,apic,pattern),parity=round++%2;
      // A dropped particle must neither join a run nor shift its offsets.
      data[strideIndex(data,f.stride)]=-1;
      const expected=f.oracle(data);f.load(data,parity);f.order.workgroups=round%2?1:64;
      const e=device.createCommandEncoder();f.encode(e,parity);device.queue.submit([e.finish()]);
      const actual=await f.read(expected.length/f.stride,parity);
      assert.deepEqual(actual,expected,JSON.stringify({apic,sparse,pattern,occupancy,parity,grid:f.order.workgroups}));
      if(occupancy===65){f.load(data,parity);f.order.legacyCanonical=true;const old=device.createCommandEncoder();f.encode(old,parity);device.queue.submit([old.finish()]);assert.deepEqual(await f.read(expected.length/f.stride,parity),actual);f.order.legacyCanonical=false;}
     }
    }
    for(let field=3;field<f.stride;field++){
     const data=particleOrderInput(257,257,apic,"identical");
     for(let i=0;i<257;i++)data[i*f.stride+field]=(i%3===0?-1:1)*((i*73)%257);
     f.load(data);const e=device.createCommandEncoder();f.encode(e);device.queue.submit([e.finish()]);
     assert.deepEqual(await f.read(257),f.oracle(data),`payload field ${field}, apic=${apic}, sparse=${sparse}`);
    }
    for(const count of [0,1,64,65]){const data=particleOrderInput(count,65,apic);f.load(data);const e=device.createCommandEncoder();f.encode(e);device.queue.submit([e.finish()]);assert.deepEqual(await f.read(count),f.oracle(data));}
   }finally{f.destroy();}
  }
 });
});
function strideIndex(data:Float32Array,stride:number){return Math.floor(data.length/stride/2)*stride;}

test("unified particle merge covers maximal work lists and repeated cold-to-dense activation",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("particle merge stress",async device=>{
  for(const capacity of [1,64,65,128,129,131072]){
   const f=await particleOrderFixture(device,capacity);
   try{
    for(const occupancy of [1,65,129,capacity,1]){
     const count=Math.min(capacity,131072),data=particleOrderInput(count,Math.max(occupancy,Math.ceil(count/32768)));
     f.load(data);f.order.workgroups=1;
     const e=device.createCommandEncoder();f.encode(e);device.queue.submit([e.finish()]);
     assert.deepEqual(await f.read(count),f.oracle(data),`capacity=${capacity}, occupancy=${occupancy}`);
    }
   }finally{f.destroy();}
  }
 });
});
