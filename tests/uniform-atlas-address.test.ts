import {test} from "node:test";
import assert from "node:assert/strict";
import {UniformAtlasAddressExperiment} from "../tools/uniform-atlas-address-experiment";

test("atlas page relocation is bijective including shared outer vertex planes",()=>{
 for(const edge of [16,32] as const){
  const atlas=new UniformAtlasAddressExperiment([64,32,32],edge,"table");
  for(const d of [[64,32,32],[65,33,33]] as const){
   const seen=new Set<number>();
   for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++){
    const p=[x,y,z] as const,a=atlas.address(p,d);
    assert.ok(a.every((n,i)=>n>=0&&n<d[i]!));
    assert.deepEqual(atlas.address(a,d),p);
    seen.add(a[0]+d[0]*(a[1]+d[1]*a[2]));
    if(x===64||y===32||z===32)assert.deepEqual(a,p);
   }
   assert.equal(seen.size,d[0]*d[1]*d[2]);
  }
 }
});
test("atlas upload and canonical readback roundtrip all components",()=>{
 const atlas=new UniformAtlasAddressExperiment([32,16,16],16,"table"),d=[33,17,17] as const;
 const src=Float32Array.from({length:d[0]*d[1]*d[2]*4},(_,i)=>i);
 const packed=atlas.reorder(src,d,4);assert.notDeepEqual(packed,src);
 assert.deepEqual(atlas.reorder(packed,d,4),src);
 assert.deepEqual(atlas.address([-1,0,0],[32,16,16]),[-1,0,0]);
 assert.deepEqual(atlas.address([4,4,4],[8,4,4]),[4,4,4]);
});
test("dense specialization preserves source and values without translation",()=>{
 const atlas=new UniformAtlasAddressExperiment([32,32,32],16,"dense");
 const source="@group(0) @binding(0) var phi:texture_3d<f32>;";
 assert.equal(atlas.shader(source),source);
 const field=new Float32Array(32**3);assert.equal(atlas.reorder(field,[32,32,32],1),field);
});
