import assert from "node:assert/strict";
import {readdirSync,readFileSync,statSync} from "node:fs";
import {join} from "node:path";
import {test} from "node:test";
import {uniformSolidSlotRowWGSL} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

// The solid record's slot rows have one definition (uniformSolidSlotRowWGSL).
// af94e6fa inserted the candidate-mask rows ahead of the slots; two readers
// that re-derived the slot base by hand then read other tiles' slots, which
// changed seep's coarsening at random (its readback volume went from 688k to
// 787k-884k, and a closure overran the residency cap at frame 3).
const SOLID="lib/methods/uniform/uniform-mixed-solid.wgsl.ts";
function sources(dir:string):string[]{
 return readdirSync(dir).flatMap(name=>{const path=join(dir,name);return statSync(path).isDirectory()?sources(path):path.endsWith(".ts")?[path]:[];});
}

test("no reader outside the solid record derives its 77-row slots by hand",()=>{
 const offenders=sources("lib/methods/uniform").filter(path=>path!==SOLID&&/\b77u\s*\*/.test(readFileSync(path,"utf8")));
 assert.deepEqual(offenders,[]);
});

test("slot s starts 77 s rows after the per-key rows, the candidate masks, the region mirror, its header, the epoch and the static rows",()=>{
 for(const tiles of [512,4096,512*37]){
  const count=tiles+1234;
  const rows=count+8*tiles+(tiles/4+tiles/512+1)+(tiles/64+1)+1+tiles;
  for(const slot of [0,5]){const value=Function(`return ${uniformSolidSlotRowWGSL(String(count),String(tiles),String(slot)).replace(/(\d+)u/g,"$1")};`)() as number;
  assert.equal(value,rows+77*slot);}
 }
});
