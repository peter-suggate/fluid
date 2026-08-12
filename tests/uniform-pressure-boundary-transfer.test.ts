import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL("../lib/webgpu-uniform-pressure-multigrid.wgsl.ts", import.meta.url), "utf8");
const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");

test("CM11a coarsening preserves half-volume dual cells at closed domain walls", () => {
  assert.match(shader, /if\(id\.x==i32\(mg\.coarseDims\.x\)-2\)\{topology\.y=0\.5;\}/);
  assert.match(shader, /topology\.z=select\(0\.5,1\.0,params\.boundary\.w>0\.5\)/);
  assert.match(shader, /if\(id\.z==i32\(mg\.coarseDims\.z\)-2\)\{topology\.w=0\.5;\}/);
});

test("CM11 prolongation excludes bookkeeping-halo pressure and renormalizes", () => {
  assert.match(shader, /if\(!mgInterior\(p,mg\.levelDims\.xyz\)\)\{continue;\}/);
  assert.match(shader, /return select\(0\.0,value\/total,total>0\.0\)/);
});

test("the pressure stage displays the fine residual rather than the coarsest residual", () => {
  assert.match(host, /uniformCM11aFineResidualInfinity\?: number;[\s\S]*?\.uniformCM11aFineResidualInfinity/);
  assert.doesNotMatch(host, /Fine-level residual[\s\S]{0,500}\.uniformCM11aResidualInfinity/);
});
