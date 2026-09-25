import assert from "node:assert/strict";
import test from "node:test";
import { uniformVolumeWorkLayout } from "../lib/methods/uniform/uniform-volume-pages.wgsl";

test("receiver and donor topology lists remain disjoint at full occupancy",()=>{
 for(const [base,pages,tiles] of [[0,1,1],[256,12,1331],[65536,512,262144]]){
  const layout=uniformVolumeWorkLayout(base!,pages!,tiles!);
  assert.equal(layout.receiver,base!+8+2*pages!);
  assert.equal(layout.donor,layout.receiver+1+tiles!);
  assert.equal(layout.donorDispatch,layout.donor+1);
  assert.equal(layout.donorDispatch+3+tiles!,base!+layout.words);
  assert.ok((layout.words*4)%4===0);
 }
});
