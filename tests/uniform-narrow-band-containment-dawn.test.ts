import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const gpuTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;

gpuTest("adaptive NB-FLIP contains a 20-second trough hose fill",{timeout:240_000},()=>{
 const run=spawnSync(process.execPath,["--import","tsx","tools/probe-uniform-trough-leak-dawn.ts","--method=nb-flip","--frames=1200"],{
  cwd:fileURLToPath(new URL("..",import.meta.url)),env:process.env,encoding:"utf8",timeout:220_000,maxBuffer:8*1024*1024,
 });
 assert.equal(run.status,0,`${run.error??""}\n${run.stdout}\n${run.stderr}`);
 const samples=run.stdout.split("\n").filter(line=>line.startsWith("{")).map(line=>JSON.parse(line) as {
  frame:number;total:number;exterior:number;exteriorWetCells:number;exteriorPhi:number;escapedParticles:number;particleCount:number;adaptiveSurface:boolean;
 });
 assert.equal(samples.at(-1)?.frame,1200,"complete the authored 20-second run");
 assert.ok(samples.at(-1)!.total>1000,"the hose actually fills the trough");
 assert.ok(samples.at(-1)!.particleCount>0,"the NB-FLIP particle layer is active");
 assert.equal(samples.at(-1)!.adaptiveSurface,true,"exercise the app's adaptive-surface profile");
 for(const sample of samples){
  assert.equal(sample.exterior,0,`frame ${sample.frame}: no liquid outside the vessel`);
  assert.equal(sample.exteriorWetCells,0,`frame ${sample.frame}: no exterior wet cells`);
  assert.equal(sample.exteriorPhi,0,`frame ${sample.frame}: no exterior liquid vertices`);
  assert.equal(sample.escapedParticles,0,`frame ${sample.frame}: particles stay inside`);
 }
});
