import assert from "node:assert/strict";
import test from "node:test";
import { UniformFrameBricks, uniformFrameBrickCapacity } from "../lib/methods/uniform/uniform-frame-bricks";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("frame brick capacity: the h tile ceiling, within the floor and the lattice; an authored one is exact",()=>{
 assert.equal(uniformFrameBrickCapacity(10,1000),640);
 assert.equal(uniformFrameBrickCapacity(100,10),256);
 assert.equal(uniformFrameBrickCapacity(100,300),300);
 assert.equal(uniformFrameBrickCapacity(100,300,40),40);
 assert.throws(()=>uniformFrameBrickCapacity(100,300,0));
});

const COUNTS=18,RECORD=32;
// A brick of each h tile, the h tiles' mask in it (bit x|y<<1|z<<2 of the brick's 2³ tiles).
function expected(fineTiles:ArrayLike<number>):Map<number,number>{
 const bricks=new Map<number,number>();
 for(const key of Array.from(fineTiles)){
  // A tile key is 512·block + its tile in the page; a brick is two tiles an axis.
  const l=key&511,t=[l&7,(l>>3)&7,l>>6],id=64*(key>>9)+((t[0]!>>1)|((t[1]!>>1)<<2)|((t[2]!>>1)<<4)),bit=(t[0]!&1)|((t[1]!&1)<<1)|((t[2]!&1)<<2);
  bricks.set(id,(bricks.get(id)??0)|(1<<bit));
 }
 return bricks;
}

test("frame bricks: the h tiles' bricks are live with their tile masks, and keep their slots while marked",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("frame bricks",async device=>{
  const dimensions:[number,number,number]=[64,64,64],address=uniformLatticeAddress(dimensions),lattice={dimensions,cellSize_m:[1,1,1] as [number,number,number],origin_m:{x:0,y:0,z:0}};
  // h tiles in a box with odd corners (partial bricks), and a second box that leaves on frame 3.
  const inBox=(x:number,y:number,z:number,lo:number[],hi:number[])=>x>=lo[0]!&&y>=lo[1]!&&z>=lo[2]!&&x<hi[0]!&&y<hi[1]!&&z<hi[2]!;
  const both=createUniformMixedLayoutFromWidths(lattice,(x,y,z)=>inBox(x,y,z,[1,3,2],[6,8,5])||inBox(x,y,z,[10,10,10],[14,13,16])?1:4,[],address);
  const one=createUniformMixedLayoutFromWidths(lattice,(x,y,z)=>inBox(x,y,z,[1,3,2],[6,8,5])?1:4,[],address);
  const ownership=new UniformMixedOwnership(device,both,false,both.fineTiles.length);
  const bricks=new UniformFrameBricks(device,ownership);await bricks.initialize();
  const l=bricks.store.arena.tableLayout,read=device.createBuffer({size:bricks.table.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const frame=async(release:boolean)=>{
   const e=device.createCommandEncoder();bricks.encodeUpdate(e);e.copyBufferToBuffer(bricks.table,0,read,0,read.size);if(release)bricks.encodeRelease(e);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const w=new Uint32Array(read.getMappedRange().slice(0));read.unmap();
   assert.equal(w[bricks.fatalOffset/4],0,"fatal");
   const live=new Map<number,number>();for(let j=0;j<w[COUNTS+2]!;j++){const slot=w[l.live+j]!;live.set(w[l.brick+slot]!,slot);assert.equal(w[l.record+RECORD*slot+29],w[l.brick+slot]);}
   const mask=(id:number)=>w[l.record+RECORD*live.get(id)!+28]!;
   return {live,mask,enter:w[COUNTS]!,leave:w[COUNTS+1]!};
  };
  const check=(f:Awaited<ReturnType<typeof frame>>,tiles:ArrayLike<number>)=>{
   const want=expected(tiles);
   const got=[...f.live.keys()].sort((a,b)=>a-b),need=[...want.keys()].sort((a,b)=>a-b);
   assert.deepEqual(got,need,`live ${got.length} [${got.join(",")}], want ${need.length} [${need.join(",")}]`);
   for(const [id,bits] of want)assert.equal(f.mask(id),bits,`brick ${id} tiles`);
  };
  const first=await frame(true);check(first,both.fineTiles);assert.equal(first.enter,first.live.size);assert.equal(first.leave,0);
  const second=await frame(true);check(second,both.fineTiles);assert.equal(second.enter,0);assert.equal(second.leave,0);
  for(const [id,slot] of first.live)assert.equal(second.live.get(id),slot,`brick ${id} kept its slot`);
  ownership.update(one);
  const third=await frame(true);check(third,one.fineTiles);assert.equal(third.enter,0);assert.equal(third.leave,first.live.size-third.live.size);
  for(const [id,slot] of third.live)assert.equal(first.live.get(id),slot,`brick ${id} kept its slot through the churn`);
  ownership.update(both);
  const fourth=await frame(true);check(fourth,both.fineTiles);assert.equal(fourth.enter,first.live.size-third.live.size);assert.equal(fourth.leave,0);
  // Slots taken back by the release are reused: none past the first frame's high water mark.
  for(const slot of fourth.live.values())assert.ok(slot<first.live.size,`slot ${slot} reused`);
  read.destroy();bricks.destroy();
 });
});
