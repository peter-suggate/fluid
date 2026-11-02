import assert from "node:assert/strict";
import test from "node:test";
import {UNIFORM_FINE_SLOT_FLOOR,uniformFineBlocks,uniformTierStart,uniformFineSlotCapacity,uniformFineSlotFootprintColumns,uniformFineSlotRingPages,uniformFineSlotRule,uniformFineSlotSourcePages,uniformFineSlotInitWGSL,uniformFineSlotLayers,uniformFineSlotWords,uniformFineTileCeiling} from "../lib/methods/uniform/uniform-fine-slots";

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

test("h slot rule: 1.25 x the pages the t=0 seed's detail rings meet, one rule for every scene",()=>{
 assert.equal(uniformFineSlotRule(64),80);
 assert.equal(uniformFineSlotRule(1),2);
 // No seed (an empty box): the one slot a store holds.
 assert.equal(uniformFineSlotRule(0),1);
 assert.throws(()=>uniformFineSlotRule(-1),/ring pages/);
 // The footprint's columns still bound a 4h start (no census seed).
 const stored=(air:readonly (readonly number[])[])=>{const out:[number,number,number][]=[];for(let z=0;z<3;z++)for(let y=0;y<2;y++)for(let x=0;x<4;x++)if(!air.some(p=>p[0]===x&&p[1]===y&&p[2]===z))out.push([x,y,z]);return out;};
 assert.equal(uniformFineSlotFootprintColumns([4,2,3],stored([[0,0,0],[0,1,0],[1,0,2]])),11);
 // On a seep-sized grid 80 rounds up to its layers of 16: 95 slots.
 assert.equal(uniformFineSlotCapacity(226,[4,4,15],0,uniformFineSlotRule(64)),95);
});

test("h slot ring pages: a seed tile's ring covers cells [4p - 13, 4p + 15], clipped to the box",()=>{
 const tiles:[number,number,number]=[64,72,32],pages:[number,number,number]=[8,9,4],key=(x:number,y:number,z:number)=>x+64*(y+72*z);
 // The pure tank's surface tile row (y 61: cells 244..247): ring cells 231..259, pages 7 and 8 on y.
 assert.equal(uniformFineSlotRingPages(pages,tiles,[key(20,61,12)]),2);
 // A tile in mid-page (cells 48..51, page 1 on each axis): cells 35..63, one page an axis.
 assert.equal(uniformFineSlotRingPages(pages,tiles,[key(12,12,12)]),1);
 // A box corner tile: cells -13..15 clip to page 0.
 assert.equal(uniformFineSlotRingPages(pages,tiles,[key(0,0,0)]),1);
 // A whole surface row of tiles over the 8x4 columns meets two pages a column, 64 in all (the pure tank's measured first claim).
 const row:number[]=[];for(let z=0;z<32;z++)for(let x=0;x<64;x++)row.push(key(x,61,z));
 assert.equal(uniformFineSlotRingPages(pages,tiles,row),64);
 // Duplicates count once; tile 13 (cells 52..55) reaches cell 67, page 2 on x.
 assert.equal(uniformFineSlotRingPages(pages,tiles,[key(12,12,12),key(12,12,12),key(13,12,12)]),2);
});

test("h slot rule: every t=0 source of h demand, a body's ball and an h region, adds the pages its tiles' rings meet",()=>{
 // The rule's sources on top of the band, at their wider margin: 1.25 x 60 + 1.5 x 8.
 assert.equal(uniformFineSlotRule(60,8),87);
 assert.equal(uniformFineSlotRule(60,0),uniformFineSlotRule(60));
 assert.throws(()=>uniformFineSlotRule(60,-1),/source pages/);
 // One tile at (8, 8, 8) (cells 32..35): the closure's neighbours, tiles 7..9, ring cells 15..51 (fsRingLow 4*7-13, fsRingHigh 4*9+15): pages 0..1 an axis.
 assert.equal(uniformFineSlotSourcePages([4,4,4],[{min:[8,8,8],max:[8,8,8]}]),8);
 // Tile (12, 12, 12) (cells 48..51): tiles 11..13, cells 31..67, pages 0..2 an axis.
 assert.equal(uniformFineSlotSourcePages([4,4,4],[{min:[12,12,12],max:[12,12,12]}]),27);
 // Clipped to the box's pages, and each source counted on its own.
 assert.equal(uniformFineSlotSourcePages([2,2,2],[{min:[0,0,0],max:[3,3,3]},{min:[12,12,12],max:[12,12,12]}]),1+8);
 assert.equal(uniformFineSlotSourcePages([4,4,4],[]),0);
});

test("h ceiling: 128 a slot with a whole slot's floor, bounded by the ring growth over the seed, the sources and the fronts, never below the seed's",()=>{
  assert.equal(uniformFineTileCeiling(207),26496);
  assert.equal(uniformFineTileCeiling(1),512);
  // The vast seep's seed (12257 h tiles) bounds its 207 slots' 26496; its hole's one free-face tile adds a slot's sheet.
  assert.equal(uniformFineTileCeiling(207,12257),15322);
  assert.equal(uniformFineTileCeiling(207,12257,0,1),15482);
  // Letters: its 6144-tile seed and its drops' 2230 band tiles, twice over for the splash.
  assert.equal(uniformFineTileCeiling(192,6144,2230),13255);
  // The garden dam break: a 400-tile seed and 8 free-face tiles hold its measured 1034-tile peak.
  assert.ok(uniformFineTileCeiling(24,400,0,8)>=1.25*1034);
  // A front folds its surface: past the sheet's per-slot figure up to the slots' whole tiles (dam-break boxes: 143 free-face tiles, 8 slots).
  assert.equal(uniformFineTileCeiling(8,1078,0,143),4096);
  // The start builds its seed, so the ceiling never falls below the seed's ring growth.
  assert.equal(uniformFineTileCeiling(10,8000),10000);
  assert.equal(uniformFineTileCeiling(95,0),512);
  assert.throws(()=>uniformFineTileCeiling(0),/whole slots/);
  assert.throws(()=>uniformFineTileCeiling(4,1.5),/seed, source and front tiles/);
  assert.throws(()=>uniformFineTileCeiling(4,10,-1),/seed, source and front tiles/);
  assert.throws(()=>uniformFineTileCeiling(4,10,0,0.5),/seed, source and front tiles/);
});
