import assert from "node:assert/strict";
import test from "node:test";
import { bakeProceduralSolid, ProceduralSolidField, proceduralSolidField, proceduralSolidKey, rebaseProceduralSolid, sampleSolidNodes, solidPlaneOpen, validateProceduralSolid } from "../lib/core/procedural-solid";
import { nbflipDomain } from "../lib/core/nbflip-paper-scenes";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { initialLiquidCellVolumes } from "../lib/core/initial-liquid-lattice";

const dimensions = [16, 16, 16] as const, h = 0.02;
const plane = bakeProceduralSolid(dimensions, [h,h,h], ([x,y,z]) => y - 8 + x/16 + z/32);
const close = (a:number,b:number,epsilon=1e-6) => assert.ok(Math.abs(a-b)<epsilon, `${a} != ${b}`);
test("packed sampled boundaries bound every brick without increasing the image",()=>{
  const source={version:1 as const,dimensions:[64,64,64] as const,cellSize_m:[1,1,1] as const,patchCount:0,patchesKey:"",
    bricks:[{origin:[16,24,32] as const,values:new Array(729).fill(0)},{origin:[40,32,48] as const,values:new Array(729).fill(1)}]};
  const field=new ProceduralSolidField(source,[]),packed=field.pack();
  assert.deepEqual(Array.from(packed.slice(1,4)),[2|(6<<16),3|(5<<16),4|(7<<16)]);
  assert.equal(packed.length,4+(packed[0]!+1)*4+field.bricks.size*(16+729));
  const large=new ProceduralSolidField({...source,bricks:[{...source.bricks[0]!,origin:[65535*8,0,0]}]},[]).pack();
  assert.deepEqual(Array.from(large.slice(1,4)),[0,0,0],"unrepresentable bounds keep the legacy hash lookup");
});
function scene() {
  const result = nbflipDomain("procedural-test", dimensions,h,60,0,dimensions,1/60);
  Object.assign(result,plane); return result;
}

test("procedural surface retains the unquantized plane, gradient and cut-cell volume across brick seams", () => {
  const field = new ProceduralSolidField(plane.solidBoundary, []);
  for (const x of [0.25,7.999,8.001,15.75]) for (const z of [0.25,7.999,8.001,15.75]) {
    const y=8-x/16-z/32, sample=field.sample([x,y,z])!;
    close(sample[3],0); close(sample[0],1/16); close(sample[1],1); close(sample[2],1/32);
  }
  // Cell x=8, y=7, z=0 has plane mean y=7.453125: 0.546875 air.
  close(field.open([8,7,0])!,0.546875);
  const roundTrip=JSON.parse(JSON.stringify(plane.solidBoundary)); validateProceduralSolid(roundTrip);
  close(new ProceduralSolidField(roundTrip,[]).sample([8.5,7.5,0.5])![3],0.046875);
});

test("one spatial field preserves separate floor, ceiling, wall and stacked surfaces", () => {
  const slab = (y:number,lo:number,hi:number) => Math.max(lo-y,y-hi);
  const geometry = bakeProceduralSolid(dimensions,[h,h,h],([x,y]) => Math.min(slab(y,3.25,4.75),slab(y,10.25,11.75),x-1.25));
  const field = new ProceduralSolidField(geometry.solidBoundary,[]);
  assert.ok(field.sample([8.5,3.25,8.5])![1]<0,"underside normal faces down");
  assert.ok(field.sample([8.5,4.75,8.5])![1]>0,"top faces up");
  assert.ok(field.sample([8.5,10.25,8.5])![1]<0,"second underside remains separate");
  assert.ok(field.sample([1.25,7.5,8.5])![0]>0,"vertical wall retains horizontal normal");
});

test("voxel edits invalidate their boundary cells, paint preserves geometry, and undo restores it", () => {
  const base=scene(), p=[8.5,7.5,0.5] as const;
  assert.ok(proceduralSolidField(base)!.sample(p));
  const painted=sceneWithSolidStroke(base,[{operation:"paint",minimum:[8,7,0],maximumExclusive:[9,8,1],materialId:2}]);
  assert.ok(proceduralSolidField(painted)!.sample(p));
  const edited=sceneWithSolidStroke(base,[{operation:"clear",minimum:[8,7,0],maximumExclusive:[9,8,1]}]);
  assert.equal(proceduralSolidField(edited)!.sample(p),undefined);
  assert.ok(proceduralSolidField(edited)!.sample([9.5,7.5,0.5]),"unmodified neighbour retains its boundary");
  const undo={...edited,solidVoxels:base.solidVoxels};
  assert.ok(proceduralSolidField(undo)!.sample(p));
  assert.equal(proceduralSolidField({...base,solidVoxels:[]}),undefined,"replaced generated prefix cannot keep stale geometry");
});

test("initial liquid capacity uses the same surface as pressure instead of closing cut cells", () => {
  const s=scene(); s.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-.16,y:0,z:-.16},max_m:{x:.16,y:.32,z:.16}}];
  const volume=initialLiquidCellVolumes(s,dimensions).volume;
  close(volume[8+16*7]!,0.546875);
});

test("plane integration handles anisotropic boxes, face areas and opposite sides", () => {
  for(const g of [[1,0,0],[1,2,3],[1e-5,1,0]] as const) for(const d of [-.3,0,.3]) {
    close(solidPlaneOpen(g,d,[.2,1,2])+solidPlaneOpen(g.map(v=>-v) as [number,number,number],-d,[.2,1,2]),1);
  }
  close(solidPlaneOpen([0,1,0],.125,[1,.5,0]),.75);
  const nodes=Array.from({length:729},(_,i)=>i%9+2*(Math.floor(i/9)%9)-3*Math.floor(i/81));
  assert.deepEqual(sampleSolidNodes(nodes,[.5,1.5,2.5]),[1,2,-3,-4]);
});

test("boundary provenance survives a prepended shell and content keys survive JSON", () => {
  const s=scene(), before=s.solidVoxels;
  s.solidVoxels=[{operation:"fill",minimum:[-1,0,0],maximumExclusive:[0,16,16]},...before];
  rebaseProceduralSolid(s,before);
  assert.ok(proceduralSolidField(s)!.sample([8.5,7.5,.5]));
  const copy=JSON.parse(JSON.stringify(s.solidBoundary));
  assert.equal(proceduralSolidKey(copy),proceduralSolidKey(s.solidBoundary));
  copy.bricks[0].values[0]+=1;
  // Keys assume immutable documents, as the scene's other content caches do.
  assert.notEqual(proceduralSolidKey(JSON.parse(JSON.stringify(copy))),proceduralSolidKey(s.solidBoundary));
});

test("malformed samples and inconsistent brick seams are rejected", () => {
  const b=JSON.parse(JSON.stringify(plane.solidBoundary));
  b.bricks[0].values[0]=Infinity; assert.throws(()=>validateProceduralSolid(b),/Invalid/);
  const seam=JSON.parse(JSON.stringify(plane.solidBoundary));
  const left=seam.bricks.find((v:{origin:number[]})=>v.origin.join()==="0,0,0");
  left.values[8]+=1; assert.throws(()=>validateProceduralSolid(seam),/shared nodes/);
});
