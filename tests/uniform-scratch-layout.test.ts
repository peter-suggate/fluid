import assert from "node:assert/strict";
import test from "node:test";
import { UniformScratchArena, UniformScratchLayout } from "../lib/methods/uniform/uniform-scratch-arena";

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
