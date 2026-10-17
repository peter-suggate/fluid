import assert from "node:assert/strict";
import test from "node:test";
import { uploadChangedWords } from "./svo-table-upload";
test("large asset uploads use bounded staging writes and preserve offsets, gaps and tail clears",()=>{
  const offset=64,next=new Uint32Array(800_003).fill(7),held=new Uint32Array(next.length+20),gpu=new Uint32Array(held.length+offset/4);
  let writes=0;
  const queue={writeBuffer(_buffer:GPUBuffer,at:number,data:Uint32Array,from=0,count=data.length){
    assert.ok(count*4<=1024*1024);writes++;gpu.set(data.subarray(from,from+count),at/4);
  }} as Pick<GPUQueue,"writeBuffer">;
  const bytes=uploadChangedWords(queue,{} as GPUBuffer,offset,held,next);
  assert.equal(bytes,next.byteLength);assert.equal(writes,4);
  assert.deepEqual(gpu.subarray(offset/4,offset/4+next.length),next);
  assert.equal(uploadChangedWords(queue,{} as GPUBuffer,offset,held,next),0);
  const shorter=next.slice(0,next.length-5);shorter[100]=9;
  uploadChangedWords(queue,{} as GPUBuffer,offset,held,shorter);
  assert.equal(gpu[offset/4+100],9);
  assert.ok(gpu.subarray(offset/4+shorter.length).every(v=>v===0));
});
