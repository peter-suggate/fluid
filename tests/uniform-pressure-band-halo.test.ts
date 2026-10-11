import assert from "node:assert/strict";
import test from "node:test";

test("packed wall and Neumann guards agree with all face kinds and ignore row flags",()=>{
 // Every possible 18-bit face classification, including reserved kind
 // values, crossed with all high flag bits set or clear. The guard must
 // never skip a wall, or interpret liquid/unit-coefficient flags as walls.
 for(let faces=0;faces<1<<18;faces++){
  let expectedWall=false,expectedNeumann=false;
  for(let face=0;face<6;face++){
   const kind=(faces>>>(3*face))&7;
   expectedWall||=kind===2;expectedNeumann||=kind===4;
  }
  for(const flags of [0,0xfffc0000]){
   const kinds=(faces|flags)>>>0;
   const anyWall=((kinds>>>1)&~kinds&~(kinds>>>2)&0x9249)!==0;
   const anyNeumann=((kinds>>>2)&~kinds&~(kinds>>>1)&0x9249)!==0;
   if(anyWall!==expectedWall||anyNeumann!==expectedNeumann)assert.fail(`face kinds ${faces.toString(8)}, flags ${flags.toString(16)}`);
  }
 }
});

test("fixed colour-row offsets address every interior and boundary neighbour",()=>{
 const row=([x,y,z]:number[])=>((x!+y!+z!)&1)*32+(x!>>1)+2*y!+8*z!;
 for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++){
  const local=[x,y,z],across=row(local)^32;
  for(let face=0;face<6;face++){
   const axis=face>>1,sign=face&1?1:-1,neighbor=[...local];neighbor[axis]!+=sign;
   const boundary=neighbor[axis]!<0||neighbor[axis]!>3;
   neighbor[axis]=(neighbor[axis]!+4)%4;
   const delta=boundary?-sign*[1,6,24][axis]!:
    axis===0?(sign>0?x&1:-1+(x&1)):sign*[0,2,8][axis]!;
   assert.equal(across+delta,row(neighbor),`cell ${local}, face ${face}`);
  }
 }
});
