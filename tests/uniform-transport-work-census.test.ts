import assert from "node:assert/strict";
import test from "node:test";
import {createUniformMixedLayoutFromWidths} from "../lib/methods/uniform/uniform-mixed-layout";
import {uniformTransportWorkCensus} from "../tools/uniform-transport-work-census";

test("frozen transport census exposes inactive fine owners without expanding coarse owners",()=>{
 const d=8,lattice={dimensions:[d,d,d] as const,origin_m:{x:0,y:0,z:0},cellSize_m:[1,1,1] as const};
 for(const widths of [new Uint8Array(8).fill(1),Uint8Array.from([4,1,1,1,1,1,1,1])]){
  const layout=createUniformMixedLayoutFromWidths(lattice,widths,[]),volume=new Float32Array(d**3),departures=new Float32Array(4*d**3);
  volume[0]=1;
  for(let z=0;z<d;z++)for(let y=0;y<d;y++)for(let x=0;x<d;x++){
   const width=widths[(x>>2)+2*((y>>2)+2*(z>>2))]!,at=4*(x+d*(y+d*z));departures.set([x+width/2,y+width/2,z+width/2,0],at);
  }
  const census=uniformTransportWorkCensus(lattice.dimensions,layout.tiles,volume,departures);
  for(const stage of Object.values(census.counts)){
   assert.equal(stage.ownerClosureOwners,1);
   assert.equal(stage.tileClosureOwners,widths[0]===1?64:1);
  }
 }
});
