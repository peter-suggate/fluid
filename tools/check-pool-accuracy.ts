/** Compare full-scene probes; tolerances describe physical observables, not
 * particle ordering or bitwise trajectories. Usage: node --import tsx
 * tools/check-pool-accuracy.ts reference.json candidate.json */
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const [referencePath,candidatePath]=process.argv.slice(2);
assert.ok(referencePath&&candidatePath,"provide reference and candidate benchmark JSON paths");
type Row={frame:number;surface?:{crest:number};audit?:{unexplained:number[];invalidBudget:number[]}};
type Run={steps:number;from:number;runtime:{version:string};summary:{deepest:number;impulseY:number;maxMomentumError:number;bodyMeanY:number;volumeDrift:number;surfaceRms:number;surfaceMax:number};rows:Row[]};
const read=(path:string)=>JSON.parse(readFileSync(path,"utf8")) as Run;
const reference=read(referencePath),candidate=read(candidatePath),a=reference.summary,b=candidate.summary,h=.05;
assert.equal(candidate.steps,reference.steps);assert.equal(candidate.from,reference.from);assert.equal(candidate.runtime.version,reference.runtime.version);
const crest=(run:Run)=>Math.max(...run.rows.filter(r=>r.frame<=300&&r.surface).map(r=>r.surface!.crest));
// Half a native cell for impact and buoyancy; settled surface RMS below a
// fifth of a cell. The volume limit is four times tighter than the solver's
// 1% controller dead band, and corresponds to about 4 mm of pool depth.
const checks={
 volume:Math.abs(b.volumeDrift)<=.0025,
 surfaceRms:b.surfaceRms<=.2*h,
 surfaceMaximum:b.surfaceMax<=h,
 penetration:Math.abs(b.deepest-a.deepest)<=.5*h,
 buoyancy:Math.abs(b.bodyMeanY-a.bodyMeanY)<=.5*h,
 splash:Math.abs(crest(candidate)-crest(reference))<=.5*h,
 impulse:Math.abs(b.impulseY-a.impulseY)<=.001*Math.abs(a.impulseY),
 momentumBalance:b.maxMomentumError<=.002,
};
const audits=candidate.rows.filter(r=>r.audit);
assert.ok(audits.length>=5,"candidate must include settled-region audits");
for(const row of audits){assert.deepEqual(row.audit!.unexplained,[],`frame ${row.frame}: unexplained activity`);assert.deepEqual(row.audit!.invalidBudget,[],`frame ${row.frame}: freeze trigger below its budget`);}
console.log(JSON.stringify({referencePath,candidatePath,checks,referenceCrest:crest(reference),candidateCrest:crest(candidate),auditedFrames:audits.length},null,2));
assert.ok(Object.values(checks).every(Boolean),"physical accuracy comparison failed");
