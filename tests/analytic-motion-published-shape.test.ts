import assert from "node:assert/strict";
import test from "node:test";
import { assessAnalyticSlabShape, semiImplicitFreeFallSlab, type AnalyticSurfaceColumn } from
  "../tools/analytic-motion-published-surface";

const h=.05;
function slab(bottom:number,top:number):AnalyticSurfaceColumn[]{
  return Array.from({length:8},(_,z)=>Array.from({length:16},(_,x)=>({
    x:.425+h*x,z:.025+h*z,bottom,top,wetLength:top-bottom,
  }))).flat();
}

test("analytic slab shape receipt resolves exact dimensions and seam continuity",()=>{
  const receipt=assessAnalyticSlabShape(slab(.8,1.2),h,.8,1.2);
  assert.ok(Math.abs(receipt.meanWidth_m!-.8)<1e-15);
  assert.ok(Math.abs(receipt.meanThickness_m!-.4)<1e-15);
  assert.equal(receipt.maximumTopSeamJump_m,0);
  assert.equal(receipt.maximumBottomSeamJump_m,0);
  assert.equal(receipt.minimumBottom_m,.8);
  assert.equal(receipt.maximumTop_m,1.2);
});

test("analytic slab shape receipt exposes deformation across the refinement seam",()=>{
  const columns=slab(.8,1.2).map(column=>column.x>.8
    ?{...column,bottom:.75,top:1.25,wetLength:.5}:column);
  const receipt=assessAnalyticSlabShape(columns,h,.8,1.2);
  assert.equal(receipt.maximumTopSeamJump_m,.050000000000000044);
  assert.equal(receipt.maximumBottomSeamJump_m,.050000000000000044);
  assert.ok(receipt.thicknessRmsError_m!>.07);
});

test("semi-implicit free fall first reaches the floor on frame 24",()=>{
  const frame23=semiImplicitFreeFallSlab(23,1/60);
  const frame24=semiImplicitFreeFallSlab(24,1/60);
  assert.equal(frame23.contactExpected,false);
  assert.equal(frame24.contactExpected,true);
  assert.ok(frame23.bottom_m>0&&frame24.bottom_m<0);
  assert.ok(Math.abs((frame23.top_m-frame23.bottom_m)-.4)<1e-15);
});
