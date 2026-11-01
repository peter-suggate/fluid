import assert from "node:assert/strict";
import test from "node:test";
import {UNIFORM_FINE_SLOT_FLOOR,uniformFineBlocks,uniformTierStart,uniformFineSlotCapacity,uniformFineSlotFootprintColumns,uniformFineSlotRule,uniformFineSlotSourcePages,uniformFineSlotInitWGSL,uniformFineSlotLayers,uniformFineSlotWords} from "../lib/methods/uniform/uniform-fine-slots";

test("h slot words: slots, need, free stack and claimed list of a capacity",()=>{
 const w=uniformFineSlotWords(10,4);
 assert.deepEqual(w,{slot:0,need:10,free:20,stack:21,claimed:25,list:26,words:30});
});

test("h slot layers: the block grid's width and height a layer, the sentinel included",()=>{
 assert.equal(uniformFineSlotLayers([8,8,4],63),1);
 assert.equal(uniformFineSlotLayers([8,8,4],64),2);
 assert.equal(uniformFineSlotLayers([2,2,5],3),1);
 assert.equal(uniformFineSlotLayers([2,2,5],4),2);
});

test("h slot capacity: the request, else the need x 1.5 over the floor, to a whole layer, within the blocks",()=>{
 // 64 slots a layer: the floor and 1.5 x 40 both fit one layer (63 past the sentinel).
 assert.equal(uniformFineSlotCapacity(226,[8,8,4],0),63);
 assert.equal(uniformFineSlotCapacity(226,[8,8,4],40),63);
 assert.equal(uniformFineSlotCapacity(226,[8,8,4],43),127);
 // No more than every block but the two sentinels.
 assert.equal(uniformFineSlotCapacity(20,[2,2,5],12),18);
 assert.equal(uniformFineSlotCapacity(226,[8,8,4],10,100),127);
 assert.ok(UNIFORM_FINE_SLOT_FLOOR>=1);
 assert.throws(()=>uniformFineSlotCapacity(226,[8,8,4],50,20),/fewer blocks than the t=0 detail ring needs/);
 assert.throws(()=>uniformFineSlotCapacity(226,[8,8,4],1,0),/positive whole number/);
});

test("h slot compaction: fineBlocks as set, else \"auto\" under a coarse start, else none",()=>{
 assert.equal(uniformFineBlocks(undefined),undefined);
 assert.equal(uniformFineBlocks({start:"h"}),undefined);
 assert.equal(uniformFineBlocks({start:"4h"}),"auto");
 assert.equal(uniformFineBlocks({start:"4h",fineBlocks:40}),40);
 assert.equal(uniformFineBlocks({fineBlocks:"auto"}),"auto");
});

test("tier start: the scene's, else the census start under Dynamic and h under any other policy",()=>{
 assert.equal(uniformTierStart(undefined,true),"census");
 assert.equal(uniformTierStart({},true),"census");
 assert.equal(uniformTierStart({start:"h"},true),"h");
 assert.equal(uniformTierStart({start:"4h"},false),"4h");
 assert.equal(uniformTierStart(undefined,false),"h");
 // The resolved start compacts as an explicit one does.
 assert.equal(uniformFineBlocks({start:uniformTierStart(undefined,true)}),"auto");
});

test("h slot init: every texel of a claimed slot, 32³ a cell or face block and 33³ a vertex block",()=>{
 const w=uniformFineSlotWords(10,4);
 const code=uniformFineSlotInitWGSL([{format:"r32float",vertex:false,empty:0},{format:"r32float",vertex:true,empty:2.5}],w,1);
 assert.match(code,/t<32768u/);assert.match(code,/t<35937u/);
 assert.match(code,/textureStore\(fsDst1,vec3i\(p\),vec4f\(2\.5,0\.0,0\.0,0\.0\)\)/);
 assert.match(code,new RegExp(`fsState\\[${w.claimed}u\\]`));
});

test("h slot rule: 1.5 x the census footprint's page columns x three band layers, one rule for every scene",()=>{
 assert.equal(uniformFineSlotRule(20),90);
 assert.equal(uniformFineSlotRule(1),5);
 // No footprint (an empty box): the one slot a store holds.
 assert.equal(uniformFineSlotRule(0),1);
 assert.throws(()=>uniformFineSlotRule(-1),/whole columns/);
 // A 4x2x3 box with one column wholly AIR and another resident only at its top.
 const air:[number,number,number][]=[[0,0,0],[0,1,0],[1,0,2]];
 assert.equal(uniformFineSlotFootprintColumns([4,2,3],air),11);
 const all:[number,number,number][]=[];for(let y=0;y<2;y++)all.push([2,y,1]);
 assert.equal(uniformFineSlotFootprintColumns([4,2,3],all),11);
 // On a seep-sized grid the rule's 90 rounds up to its layers of 16: 95 slots.
 assert.equal(uniformFineSlotCapacity(226,[4,4,15],0,uniformFineSlotRule(20)),95);
});

test("h slot rule: every t=0 source of h demand, a body's ball and an h region, adds the pages its tiles' rings meet",()=>{
 // The rule's sources on top of the band: 1.5 x (20 columns x 3 + 8 pages).
 assert.equal(uniformFineSlotRule(20,8),102);
 assert.equal(uniformFineSlotRule(20,0),uniformFineSlotRule(20));
 assert.throws(()=>uniformFineSlotRule(20,-1),/source pages/);
 // One tile at (8, 8, 8) (cells 32..35): the closure's neighbours, tiles 7..9, ring cells 15..51 (fsRingLow 4*7-13, fsRingHigh 4*9+15): pages 0..1 an axis.
 assert.equal(uniformFineSlotSourcePages([4,4,4],[{min:[8,8,8],max:[8,8,8]}]),8);
 // Tile (12, 12, 12) (cells 48..51): tiles 11..13, cells 31..67, pages 0..2 an axis.
 assert.equal(uniformFineSlotSourcePages([4,4,4],[{min:[12,12,12],max:[12,12,12]}]),27);
 // Clipped to the box's pages, and each source counted on its own.
 assert.equal(uniformFineSlotSourcePages([2,2,2],[{min:[0,0,0],max:[3,3,3]},{min:[12,12,12],max:[12,12,12]}]),1+8);
 assert.equal(uniformFineSlotSourcePages([4,4,4],[]),0);
});
