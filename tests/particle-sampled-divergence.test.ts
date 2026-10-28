import {test} from "node:test";
import assert from "node:assert/strict";
import {sampledCellDivergence} from "../tools/particle-sampled-divergence";

test("sampled divergence recovers an affine field with cross derivatives at large coordinates",()=>{
  const points=Array.from({length:8},(_,i)=>{
    const x=1024+.2+.6*(i&1),y=896+.2+.6*((i>>1)&1),z=992+.2+.6*(i>>2);
    return {x,y,z,vx:2*x-3*y,vy:4*x-y+z,vz:2*y+3*z,mass:2,depth:-.8};
  });
  const [cell]=sampledCellDivergence(points,.005);
  assert.equal(cell!.mass,16);
  assert.ok(Math.abs(cell!.divergence_per_s!-800)<1e-6);
});

test("planar particle clouds cannot supply a three dimensional divergence",()=>{
  const points=Array.from({length:8},(_,i)=>({x:.1+.1*i,y:.25,z:.3,vx:0,vy:0,vz:0,mass:1,depth:-.5}));
  assert.equal(sampledCellDivergence(points,.005)[0]!.divergence_per_s,null);
});
