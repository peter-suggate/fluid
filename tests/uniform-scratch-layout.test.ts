import assert from "node:assert/strict";
import test from "node:test";
import { UniformScratchArena, UniformScratchLayout } from "../lib/methods/uniform/uniform-scratch-arena";

test("scratch layout preserves fine storage sizes and stage aliases", () => {
  for (const [dims, bytes, donorOffset, donorBytes] of [
    [[4,4,4],14720,2560,1792],
    [[32,32,32],2515456,1310720,917504],
    [[128,64,32],18670080,10485760,7340032],
  ] as const) {
    const p=new UniformScratchLayout(dims,dims.reduce((a,b)=>a*b,1)*40,false,false);
    assert.equal(p.byteLength,bytes);
    assert.equal(p.donorOffset,donorOffset);
    assert.equal(p.donorBytes,donorBytes);
    assert.equal(p.offset("Uniform Sec. 3.3 resolved FIM values"),p.offset("Uniform Sec. 3.3 FIM values A"));
    assert.equal(p.offset("Uniform CM11a Full-Cycle p_tmp"),p.offset("Uniform CM11a L0 phi A"));
    assert.equal(p.offset("Uniform CM11a accepted pressure"),p.offset("Uniform CM11a L0 phi B"));
    assert.ok(p.byteLength>=p.donorOffset+p.donorBytes);
    const far=new UniformScratchLayout(dims,p.edgeBytes,false,true);
    assert.ok(far.farTileOffset!*4>=p.byteLength);
    assert.equal(far.farTileOffset!%4,0);
    const diagnostic=new UniformScratchLayout(dims,p.edgeBytes,true,false);
    assert.equal(diagnostic.offset("Uniform Sec. 3.3 resolved FIM distances"),undefined);
    assert.equal(diagnostic.byteLength,p.byteLength);
  }
});

test("fine and coarse arena views share one allocation with one owner", () => {
  const previous=Object.getOwnPropertyDescriptor(globalThis,"GPUBufferUsage");
  Object.defineProperty(globalThis,"GPUBufferUsage",{configurable:true,value:{STORAGE:128,COPY_SRC:4,COPY_DST:8}});
  let allocations=0, destroys=0;
  const device={createBuffer:({size,usage}:GPUBufferDescriptor)=>{
    allocations++;
    return {size,usage,destroy:()=>{destroys++;}} as GPUBuffer;
  }} as GPUDevice;
  try {
    const fine=new UniformScratchArena(device,[128,64,32],128*64*32*40);
    const coarse=new UniformScratchArena(device,[32,16,8],32*16*8*40,false,fine.buffer);
    assert.equal(allocations,1);
    assert.equal(coarse.buffer,fine.buffer);
    assert.ok(coarse.byteLength<fine.byteLength);
    assert.notEqual(coarse.offset("Uniform CM11a L0 rhs B"),fine.offset("Uniform CM11a L0 rhs B"));
    assert.throws(()=>new UniformScratchArena(device,[256,128,64],256*128*64*40,false,fine.buffer),/capacity or usage/);
    assert.throws(()=>new UniformScratchArena(device,[4,4,4],2560,false,{size:fine.buffer.size,usage:128} as GPUBuffer),/capacity or usage/);
    assert.equal(allocations,1);
    coarse.destroy();
    assert.equal(destroys,0);
    fine.destroy();
    assert.equal(destroys,1);
  } finally {
    if(previous)Object.defineProperty(globalThis,"GPUBufferUsage",previous);
    else Reflect.deleteProperty(globalThis,"GPUBufferUsage");
  }
});

test("scratch planning rejects malformed dimensions and byte ranges before allocation", () => {
  assert.throws(()=>new UniformScratchLayout([0,4,4],0),RangeError);
  assert.throws(()=>new UniformScratchLayout([4,1.5,4],0),RangeError);
  assert.throws(()=>new UniformScratchLayout([4,4,4],-4),RangeError);
  assert.throws(()=>new UniformScratchLayout([4,4,4],3),RangeError);
});
