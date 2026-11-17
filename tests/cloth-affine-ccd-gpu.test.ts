import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createAffineCCDShadow} from '../tools/cloth-affine-ccd-gpu';
import {affineVertexTriangleCCD,affineEdgeEdgeCCD,type AffineVertex,type Vec3} from '../tools/cloth-affine-ccd';

test('M1 affine shadow supports agree conservatively with analytical and Float64 fixtures',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
  const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];
  device.addEventListener('uncapturederror',(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});
  try{
    for(const kind of ['vt','ee'] as const){
      const cases:{x:Vec3[];d:Vec3[];root?:number}[]=[];
      for(let i=0;i<64;i++){
        const a=.173*i,b=.317*i,rotate=([x,y,z]:Vec3):Vec3=>[Math.cos(b)*x+Math.sin(b)*(Math.sin(a)*y+Math.cos(a)*z),Math.cos(a)*y-Math.sin(a)*z,-Math.sin(b)*x+Math.cos(b)*(Math.sin(a)*y+Math.cos(a)*z)];
        const x:Vec3[]=kind==='vt'?[[0,0,1],[-2,-2,0],[2,-2,0],[0,2,0]]:[[-1,0,0],[1,0,0],[0,-1,1],[0,1,1]];
        const d:Vec3[]=kind==='vt'?[[0,0,-2],[0,0,0],[0,0,0],[0,0,0]]:[[0,0,0],[0,0,0],[0,0,-2],[0,0,-2]];
        cases.push({x:x.map(rotate),d:d.map(rotate),root:.45});
      }
      // Degenerate points, tangential contact, strict initial boundary and overlap.
      for(const offset of [.1,.1001,0])cases.push({x:[[0,0,0],[0,0,0],[-1,offset,0],[-1,offset,0]],d:[[0,0,0],[0,0,0],[2,0,0],[2,0,0]]});
      let seed=1729;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32-.5;};
      for(let i=0;i<96;i++)cases.push({x:Array.from({length:4},()=>[rnd()*4,rnd()*4,rnd()*4] as Vec3),d:Array.from({length:4},()=>[rnd()*2,rnd()*2,rnd()*2] as Vec3)});
      if(kind==='vt')cases.push({x:[[0,0,.4],[-2,-2,0],[2,-2,0],[0,2,0]],d:[[0,0,-1],[4,4,0],[-4,4,0],[0,-4,0]],root:.3});
      const x=new Float32Array(cases.length*16),d=new Float32Array(x.length),ids=new Uint32Array(cases.length*4);
      cases.forEach((c,i)=>{for(let j=0;j<4;j++){x.set(c.x[j]!,16*i+4*j);d.set(c.d[j]!,16*i+4*j);ids[4*i+j]=4*i+j;}});
      for(const projectedSupports of [false,true])for(const sameTimeEndpoints of [false,true])for(const advancement of (projectedSupports&&sameTimeEndpoints?[false,true]:[false]))for(const currentBounds of (projectedSupports&&sameTimeEndpoints&&!advancement?[false,true]:[false]))for(const wg of [32,64,128] as const){
        const runner=await createAffineCCDShadow(device,x,d,ids,kind,.1,wg,256,sameTimeEndpoints,undefined,projectedSupports,advancement,currentBounds);
        try{const enc=device.createCommandEncoder();runner.encode(enc);device.queue.submit([enc.finish()]);const result=await runner.read();
          cases.forEach((c,i)=>{
            const p=Array.from({length:4},(_,j):AffineVertex=>({x:Array.from(x.subarray(16*i+4*j,16*i+4*j+3)) as unknown as Vec3,d:Array.from(d.subarray(16*i+4*j,16*i+4*j+3)) as unknown as Vec3})) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];
            const cpu=(kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(.1),sameTimeEndpoints}),alpha=result[4*i]!;
            assert.ok(alpha>=0&&alpha<=1&&Number.isFinite(alpha));assert.ok(alpha<=cpu.alpha+2e-6,JSON.stringify({kind,wg,i,alpha,cpu}));
            if(c.root!==undefined)assert.ok(alpha<=c.root+2e-6&&alpha>c.root-.001,JSON.stringify({kind,wg,i,alpha}));
            if(advancement&&alpha>0){const proof=(kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(.1),maxAlpha:alpha,sameTimeEndpoints:true,exactEndpoints:true});assert.equal(proof.reason,'separated',JSON.stringify({kind,wg,i,alpha,proof}));}
          });
        }finally{runner.destroy();}
      }
    }
    assert.deepEqual(errors,[]);
  }finally{device.destroy();}
});
