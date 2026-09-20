import assert from "node:assert/strict";
import test from "node:test";
import { uniformPhiBandLimit, uniformPhiReadHalo, UNIFORM_PHI_BAND_CELLS } from "../lib/methods/uniform/uniform-phi-band";

test("finite phi cap contains exact-SDF redistance queries on anisotropic grids",()=>{
 for(const spacing of [[1,1,1],[.1,.04,.2],[.001,2,.3]] as const){
  const cap=uniformPhiBandLimit(spacing),max=Math.max(...spacing);
  // A 4hMax distance seed plus the diagonal query/interpolation reach.
  const furthest=4*max+Math.hypot(...spacing.map(h=>5.25*h));
  assert.ok(cap>furthest);
 }
 assert.equal(UNIFORM_PHI_BAND_CELLS,16);
 assert.deepEqual(uniformPhiReadHalo([0,2.5,40]),[6,8,46]);
 assert.throws(()=>uniformPhiBandLimit([1,0,1]));
 assert.throws(()=>uniformPhiReadHalo([0,NaN,1]));
});
