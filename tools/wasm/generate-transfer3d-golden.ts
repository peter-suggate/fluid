import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {resolve} from "node:path";
type Box={id:number;lower:[number,number,number];widths:[number,number,number];span:number};
const parent:Box={id:0,lower:[0,0,0],widths:[2,2,2],span:2};
const children:Box[]=[];for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)children.push({id:children.length,lower:[x,y,z],widths:[1,1,1],span:1});
function source(cells:Box[]){return{dimensions:[2,2,2],cells,faces:[]}}
function target(cells:Box[]){return cells;}
/** Independent box-overlap oracle for the Rust transfer implementation. */
function compileTransfer(input:ReturnType<typeof source>, output:Box[], newAir:unknown[]=[]){
 const cellOffsets=[0],cellSources:number[]=[],cellVolumes:number[]=[];
 for(const box of output){
  let covered=0;
  for(const old of input.cells){
   const overlap=box.lower.reduce((volume,low,axis)=>volume*Math.max(0,Math.min(low+box.widths[axis]!,old.lower[axis]!+old.widths[axis]!)-Math.max(low,old.lower[axis]!)),1);
   if(overlap){cellSources.push(old.id);cellVolumes.push(overlap);covered+=overlap;}
  }
  const volume=box.widths.reduce((a,b)=>a*b,1);
  if(covered<volume){if(!newAir.length)throw new Error("Uncovered target without explicit new-air coverage");cellSources.push(0xffffffff);cellVolumes.push(volume-covered);}
  cellOffsets.push(cellSources.length);
 }
 return {cellOffsets,cellSources,cellVolumes,faceOffsets:[0],faceSources:[],faceAreas:[]};
}
const encode=(p:ReturnType<typeof compileTransfer>)=>p;
const cases=[{id:"split-eight",source:source([parent]),target:children,newAir:[],expected:encode(compileTransfer(source([parent]),target(children)))},{id:"merge-eight",source:source(children),target:[parent],newAir:[],expected:encode(compileTransfer(source(children),target([parent])))},{id:"explicit-new-air",source:source([{id:0,lower:[0,0,0],widths:[1,2,2],span:2}]),target:[parent],newAir:[{minimumFine:[0,0,0],maximumExclusiveFine:[2,2,2]}],expected:encode(compileTransfer(source([{id:0,lower:[0,0,0],widths:[1,2,2],span:2}]),target([parent]),[{minimumFine:[0,0,0],maximumExclusiveFine:[2,2,2]}]))}];
const artifact=JSON.stringify({schemaVersion:1,source:"tools/wasm/generate-transfer3d-golden.ts#compileTransfer",cases},null,2)+"\n",output=resolve(import.meta.dirname,"../../rust/core/testdata/transfer3d-golden.json");if(process.argv.includes("--check")){if(readFileSync(output,"utf8")!==artifact)throw new Error(`${output} is stale`)}else{mkdirSync(resolve(output,".."),{recursive:true});writeFileSync(output,artifact);console.log(output)}
