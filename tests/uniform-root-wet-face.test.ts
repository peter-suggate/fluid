import assert from "node:assert/strict";
import test from "node:test";
import {cutFit,wallAxis,wetFace,type OpenCells,type WetTile} from "../tools/root-wet-face-reference";

const tile=(row:boolean):WetTile=>({open:[0xffffffff,0xffffffff],cut:true,fine:true,row,sides:1});
test("a submerged coarse row cannot create a pressure face through a dry fine shoreline",()=>{
 for(let axis=0;axis<3;axis++){
  assert.deepEqual(wetFace(axis,tile(true),tile(false),()=>1,()=>0.25),{v:0,mask:0});
  assert.deepEqual(wetFace(axis,tile(false),tile(true),()=>1,()=>0.25),{v:0,mask:0});
 }
});
test("a real fine liquid-air crossing retains its pressure aperture",()=>{
 for(let axis=0;axis<3;axis++){
  assert.deepEqual(wetFace(axis,tile(true),tile(false),()=>1,plane=>plane-1.25),{v:1,mask:65535});
 }
});

function halfOpen(axis:number):OpenCells{
 const open:[number,number]=[0,0];
 for(let i=0;i<64;i++)if(([i%4,(i>>2)%4,i>>4][axis]??0)>=2)open[i>>5]=(open[i>>5]!|(1<<(i&31)))>>>0;
 return open;
}
test("cached voxel normals constrain only the wall-normal translation",()=>{
 for(let normal=0;normal<3;normal++)for(let axis=0;axis<3;axis++){
  const open=halfOpen(normal);
  assert.equal(wallAxis(open,axis),axis===normal);
  const side={...tile(false),open};
  const face=wetFace(axis,side,side,()=>1,()=>1);
  assert.equal(face.mask&0x10000,axis===normal?0x10000:0);
  assert.equal(face.mask&0xffff,0,"wall metadata must not create a pressure aperture");
  assert.equal(face.v,0);
 }
});
test("a cut pressure plane excludes dry distance bending around a shoreline",()=>{
 const fit=cutFit(halfOpen(0),v=>{
  const x=v%5,y=Math.floor(v/5)%5,z=Math.floor(v/25),plane=y-2.4;
  return plane<0?plane:plane+0.7*x+0.3*z;
 },1);
 assert.ok(Math.abs(fit.phi+0.4)<1e-12,`hydrostatic plane at tile centre: ${fit.phi}`);
});
test("a cut liquid fit with no support at the centre leaves a finite air owner",()=>{
 const fit=cutFit(halfOpen(0),v=>Math.floor(v/5)%5-0.4,1);
 assert.equal(fit.phi,2);
});
test("liquid samples on the centre plane still determine a shallow contact row",()=>{
 // The actual failing shoreline tile: its only four negative vertices lie
 // on y=2, through the 4h centre. Its pressure is determined without dy.
 const open:OpenCells=[2147518464,1146881];
 const fit=cutFit(open,v=>{
  const x=v%5,y=Math.floor(v/5)%5,z=Math.floor(v/25);
  return x>=3&&y===2&&z<=1?-0.011622431688:0.025;
 },0.0125);
 assert.ok(Math.abs(fit.phi+0.011622431688)<1e-12);
});
test("a wholly dry cut tile retains the air distance used by surface pressure",()=>{
 const fit=cutFit(halfOpen(0),v=>Math.floor(v/5)%5+0.6,1);
 assert.ok(Math.abs(fit.phi-2.6)<1e-12);
});
