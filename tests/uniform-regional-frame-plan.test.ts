import assert from "node:assert/strict";
import test from "node:test";
import { UniformRegionalFramePlan } from "../lib/methods/uniform/uniform-regional-frame-plan";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformLatticeAddress, uniformTileKey, uniformTileCoord, UNIFORM_PAGE_TABLE_WORDS } from "../lib/methods/uniform/uniform-lattice-address";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("shared direct frame plan scales with changed regions and includes activation and retired-page neighbours",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("shared regional frame plan",async device=>{
  for(const length of [128,512]){
   const dimensions:[number,number,number]=[length,64,64],address=uniformLatticeAddress(dimensions);
   const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,undefined,address);
   const o=new UniformMixedOwnership(device,layout,true,0),r=o.capacity.tiles/64;
   const make=(label:string,size:number)=>device.createBuffer({label,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const revisions=make("test revisions",8*r),tiers=make("test tiers",4*r);
   device.queue.writeBuffer(revisions,0,new Uint32Array(2*r).fill(1));
   let generation=0;
   const plan=new UniformRegionalFramePlan(device,o,{buffer:revisions,liveOffset:0,topologyOffset:4*r,regions:r},tiers,()=>generation);
   const read=device.createBuffer({size:plan.buffer.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const snapshot=async()=>{const e=device.createCommandEncoder();e.copyBufferToBuffer(plan.buffer,0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const a=new Uint32Array(read.getMappedRange()).slice();read.unmap();return a;};
   const frame=async()=>{plan.beginFrame();const e=device.createCommandEncoder();plan.encode(e);device.queue.submit([e.finish()]);return snapshot();};
   const list=(a:Uint32Array,ch:number)=>[...a.slice(plan.layout.lists+ch*r,plan.layout.lists+ch*r+a[ch]!)].sort((a,b)=>a-b);
   const tile=uniformTileKey(address,8,8,8),local=tile%512,region=8*Math.floor(tile/512)+((local&7)>>2)+2*(((local>>3)&7)>>2)+4*((local>>6)>>2);
   const neighbours=Array.from({length:r},(_,i)=>i).filter(i=>{
    const key=512*(i>>3)+4*(i&1)+32*((i>>1)&1)+256*((i>>2)&1);
    if(address.seed.blocks[i>>3]===0xffffffff)return false;
    const p=uniformTileCoord(address,key,[0,0,0]);return p.every(x=>Math.abs(x/4-2)<=1);
   });
   try{
    await plan.initialize();
    const first=await frame();assert.equal(first[0],length*64*64/4096);assert.equal(first[8],first[0]!*64);
    assert.deepEqual([...(await frame()).slice(0,9)],new Array(9).fill(0),"quiet frame has no work");
    const held=await snapshot();plan.beginFrame();let heldEncoder=device.createCommandEncoder();plan.encode(heldEncoder,true);device.queue.submit([heldEncoder.finish()]);
    assert.equal(plan.retainsGeometry,true);assert.equal(plan.active,false);assert.deepEqual(await snapshot(),held,"held frame submits no list rebuild");
    plan.enabled=false;plan.beginFrame();heldEncoder=device.createCommandEncoder();plan.encode(heldEncoder,true);device.queue.submit([heldEncoder.finish()]);assert.equal(plan.retainsGeometry,false);plan.enabled=true;
    device.queue.writeBuffer(revisions,4*region,new Uint32Array([2]));
    const changed=await frame();assert.deepEqual(list(changed,0),[region]);assert.deepEqual(list(changed,3),neighbours);assert.equal(changed[3],27);assert.equal(changed[8],27*64);
    device.queue.writeBuffer(revisions,4*region,new Uint32Array([3]));
    assert.deepEqual(await snapshot(),changed,"producer writes cannot mutate the sealed frame schedule");
    await frame();await frame();
    device.queue.writeBuffer(o.regionVelocity,24*region,new Uint32Array([123]));
    const moving=await frame();assert.equal(moving[0],0);assert.deepEqual(list(moving,2),[region]);assert.deepEqual(list(moving,5),neighbours);assert.deepEqual(list(moving,6),neighbours);
    device.queue.writeBuffer(tiers,4*region,new Uint32Array([2]));
    const hot=await frame();assert.deepEqual(list(hot,0),[region]);assert.deepEqual(list(hot,2),[region]);assert.equal(hot[3],27,"new HOT work is admitted immediately");
    device.queue.writeBuffer(tiers,4*region,new Uint32Array([0]));await frame();
    device.queue.writeBuffer(o.presentation.buffer,4*tile,new Uint32Array([layout.tiles[tile]!^1]));
    device.queue.writeBuffer(revisions,4*(r+region),new Uint32Array([2]));
    const topology=await frame();assert.deepEqual(list(topology,1),[region]);assert.deepEqual(list(topology,4),neighbours);
    // Release the page and its table entry: the old world neighbourhood
    // must be invalidated even though no current key resolves to that page.
    const block=tile>>9,page=address.seed.blocks[block]!;
    device.queue.writeBuffer(o.presentation.buffer,4*(5*o.capacity.tiles+UNIFORM_PAGE_TABLE_WORDS+block),new Uint32Array([0xffffffff]));
    device.queue.writeBuffer(o.presentation.buffer,4*(5*o.capacity.tiles+page),new Uint32Array([0]));
    const retired=await frame();const retiredList=list(retired,4);
    assert.ok(retiredList.length>0);assert.ok(retiredList.every(i=>(i>>3)!==block));
    assert.ok(neighbours.filter(i=>(i>>3)!==block).every(i=>retiredList.includes(i)),"release invalidates surviving neighbours");
    generation++;const invalidated=await frame();assert.equal(invalidated[0],first[0]!-8);
   }finally{read.destroy();plan.destroy();revisions.destroy();tiers.destroy();o.destroy();}
  }
 });
});
