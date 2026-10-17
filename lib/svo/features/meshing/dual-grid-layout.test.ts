import assert from "node:assert/strict";
import test from "node:test";
import { packDualGridPoint, unpackDualGridPoint, DUAL_GRID_BYTES_PER_VOXEL } from "./dual-grid-layout";
test("compact fitted positions retain sub-output-grid precision, including cell boundaries",()=>{
  assert.equal(DUAL_GRID_BYTES_PER_VOXEL,8);
  for(let i=0;i<=1024;i++){
    const p=[i/1024,1-i/1024,(i*137%1025)/1024];
    const q=unpackDualGridPoint(packDualGridPoint(p));
    for(let a=0;a<3;a++)assert.ok(Math.abs(p[a]!-q[a]!)<=1/2046+1e-12);
  }
  assert.deepEqual(unpackDualGridPoint(packDualGridPoint([0,1,0])),[0,1,0]);
  assert.throws(()=>packDualGridPoint([NaN,0,0]));
  assert.throws(()=>packDualGridPoint([-.1,0,0]));
});
