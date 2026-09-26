import assert from "node:assert/strict";
import test from "node:test";
import type { FluidRefinementRegion } from "../lib/core/model";
import { createUniformMixedLayout, mixedCellWidth } from "../lib/methods/uniform/uniform-mixed-layout";
const lattice = { dimensions: [8, 8, 8] as const, cellSize_m: [1, 2, .5] as const, origin_m: { x: -4, y: 0, z: -2 } };
function box(axis: number, fine = true): FluidRefinementRegion {
  const min = [-4, 0, -2], max = [4, 16, 2];
  max[axis] = min[axis]! + 4 * lattice.cellSize_m[axis]!;
  return { id: `region-${axis}`, min_m: { x: min[0]!, y: min[1]!, z: min[2]! }, max_m: { x: max[0]!, y: max[1]!, z: max[2]! },
    rule: "minimum-cell-size", minimumCellSize_cells: fine ? 1 : 4, maximumCellSize_cells: fine ? 1 : 4 };
}
test("manual regions produce compact, deterministic single-owner tiles", () => {
  const background = createUniformMixedLayout(lattice, [], true, 4);
  assert.equal(background.cellCount, 8); assert.equal(background.metadataBytes, 128);
  const a = box(0), b = box(1);
  const mixed = createUniformMixedLayout(lattice, [a], true, 4);
  assert.equal(mixed.cellCount, 288); assert.equal(mixed.fineTiles.length, 4);
  assert.equal(mixed.metadataBytes, mixed.stencils.byteLength + mixed.tiles.byteLength + mixed.fineTiles.byteLength + mixed.coarseTiles.byteLength + mixed.transitionTiles.byteLength);
  assert.deepEqual(createUniformMixedLayout(lattice, [a, b], true, 4).tiles, createUniformMixedLayout(lattice, [b, a], true, 4).tiles);
  assert.throws(() => createUniformMixedLayout(lattice, [a, { ...box(0, false), id: "conflict" }], true, 4), /Conflicting/);
  assert.throws(() => createUniformMixedLayout(lattice, [{ ...a, minimumCellSize_cells: 2, maximumCellSize_cells: 2 }], true, 4), /neither/);
  assert.equal(createUniformMixedLayout(lattice, [{ ...a, maximumCellSize_cells: undefined }], true, 4).cellCount, 8, "minimum 1 alone does not enforce fine");
  assert.throws(() => createUniformMixedLayout(lattice, [a, a], true, 4), /Duplicate/);
  assert.throws(() => createUniformMixedLayout(lattice, [{ ...a, maximumCellSize_cells: Infinity }], true, 4), /Invalid/);
  assert.throws(() => createUniformMixedLayout(lattice, [{ ...a, min_m: { x: NaN, y: 0, z: 0 } }], true, 4), /Invalid/);
  assert.throws(() => createUniformMixedLayout({ ...lattice, dimensions: [7, 8, 8] }, [], true, 4), /divisible/);
  assert.throws(() => createUniformMixedLayout(lattice, [{ ...a, min_m: { x: 20, y: 0, z: 0 }, max_m: { x: 21, y: 1, z: 1 } }], true, 4), /overlap/);
  const narrow = { ...a, min_m: { x: -3.8, y: 1, z: -1.9 }, max_m: { x: -3.7, y: 2, z: -1.8 } };
  assert.deepEqual(createUniformMixedLayout(lattice, [narrow], true, 4).regions[0], { id: a.id, min: [0, 0, 0], max: [1, 1, 1] });
});
test("decimal world bounds on a tile edge do not refine an extra tile",()=>{
 const l={dimensions:[64,64,64] as const,cellSize_m:[.0125,.0125,.0125] as const,origin_m:{x:-.4,y:0,z:-.4}};
 const r:FluidRefinementRegion={id:"decimal",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,min_m:{x:-.4,y:0,z:-.4},max_m:{x:.2,y:.4,z:.4}};
 assert.equal(createUniformMixedLayout(l,[r], true, 4).fineTiles.length,12*8*16);
});

test("strong 2:1 grading inserts internal 2h tiles at faces, edges and corners", async () => {
  const { mixedCellWidth, MIXED_CELL_MASK } = await import('../lib/methods/uniform/uniform-mixed-layout');
  const lattice = { dimensions: [20, 20, 20] as const, cellSize_m: [1, 1, 1] as const, origin_m: { x: 0, y: 0, z: 0 } };
  const fine: FluidRefinementRegion = { id: 'fine', rule: 'minimum-cell-size', minimumCellSize_cells: 1, maximumCellSize_cells: 1, min_m: { x: 8, y: 8, z: 8 }, max_m: { x: 12, y: 12, z: 12 } };
  const layout = createUniformMixedLayout(lattice, [fine], true, 4);
  const widths = Array.from(layout.tiles, mixedCellWidth);
  assert.equal(widths.filter(w => w === 1).length, 1);
  assert.equal(widths.filter(w => w === 2).length, 26);
  assert.equal(widths.filter(w => w === 4).length, 98);
  assert.equal(layout.cellCount, 64 + 26 * 8 + 98);
  const occupied = new Set<number>();
  layout.tiles.forEach(word => {
    const base = word & MIXED_CELL_MASK;
    for (let i = 0; i < (4 / mixedCellWidth(word)) ** 3; i++) { assert.ok(!occupied.has(base + i)); occupied.add(base + i); }
  });
  assert.equal(occupied.size, layout.cellCount);
  assert.equal(Math.max(...occupied), layout.cellCount - 1);
  const forcedCoarse = { ...fine, id: 'coarse', minimumCellSize_cells: 4, maximumCellSize_cells: 4, min_m: { x: 12, y: 12, z: 12 }, max_m: { x: 16, y: 16, z: 16 } };
  assert.throws(() => createUniformMixedLayout(lattice, [fine, forcedCoarse], true, 4), /conflicts with 2:1/);
  assert.deepEqual(createUniformMixedLayout(lattice, [], true, 4).tiles, createUniformMixedLayout(lattice, [], true, 4).tiles);
});

test("production ownership defaults fine and live coarse enforcement grades outside the region", () => {
  const lattice={dimensions:[32,16,16] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const baseline=createUniformMixedLayout(lattice,[]);
  assert.equal(baseline.fineTiles.length,8*4*4);
  const region={id:"half",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,min_m:{x:16,y:0,z:0},max_m:{x:32,y:16,z:16}};
  const mixed=createUniformMixedLayout(lattice,[region]);
  assert.equal(mixed.coarseTiles.length,4*4*4);
  assert.equal(mixed.transitionTiles.length,4*4);
  assert.equal(mixed.fineTiles.length,3*4*4);
  assert.deepEqual(createUniformMixedLayout(lattice,[]).tiles,baseline.tiles,"removing the region restores fine ownership");
});


test("frozen stencil masks describe clipped face, edge and corner neighborhoods",()=>{
 const lattice={dimensions:[24,20,16] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
 const region={id:"air",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,min_m:{x:12,y:8,z:4},max_m:{x:16,y:12,z:8}};
 for(const l of [createUniformMixedLayout(lattice,[]),createUniformMixedLayout(lattice,[region]),createUniformMixedLayout(lattice,[],true,4)]){
  const d=l.tileDimensions;
  l.tiles.forEach((_,tile)=>{
   const t=[tile%d[0],Math.floor(tile/d[0])%d[1],Math.floor(tile/(d[0]*d[1]))];
   let minimum=4,maximum=1;
   for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){
    const q=[t[0]!+x,t[1]!+y,t[2]!+z],bit=1<<((x+1)+3*((y+1)+3*(z+1)));
    let w=0;
    if(q.every((v,a)=>v>=0&&v<d[a]!)){w=mixedCellWidth(l.tiles[q[0]!+d[0]*(q[1]!+d[1]*q[2]!)]!);minimum=Math.min(minimum,w);maximum=Math.max(maximum,w);}
    assert.equal(!!(l.stencils[2*tile]!&bit),w===1);
    assert.equal(!!(l.stencils[2*tile+1]!&bit),w===2);
   }
   assert.equal(l.stencils[2*tile]!>>>27,maximum);assert.equal(l.stencils[2*tile+1]!>>>27,minimum);
  });
 }
});
