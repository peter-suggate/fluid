/** Diagnostic-only rewrites. Keep the product arithmetic and external layouts. */
import {fnSpan, replaceFn, replaceOnce} from './probe-nbflip-variants';

export function surfacePacket(code:string, lanes:number):string {
 const [a,b]=fnSpan(code,'arenaDistanceCanonical');
 const body=code.slice(a,b).replaceAll('256u',`${lanes}u`);
 code=code.slice(0,a)+body+code.slice(b);
 code=replaceOnce(code,'@workgroup_size(256) fn arenaDistanceCanonical',`@workgroup_size(${lanes}) fn arenaDistanceCanonical`);
 code=replaceOnce(code,'nbArenaQueue:array<u32,2048>',`nbArenaQueue:array<u32,${8*lanes}>`);
 return replaceOnce(code,'nbArenaFound:array<atomic<u32>,256>',`nbArenaFound:array<atomic<u32>,${lanes}>`);
}

export function pressureTilePacket(code:string):string {
 code=replaceOnce(code,'var<workgroup> rN:u32;','var<workgroup> rN:u32;\nvar<workgroup> rLocalX:array<f32,128>;\nvar<private> rLocalBase:u32;');
 code=replaceOnce(code,'return pF(pPlane(select(pSrc,1u-pSrc,black&&own)),pNear0(at,slot,lb,f));',
  'if(own){var q=vec3i(lb&vec3u(3u));q[axis]+=bSign(f);return rLocalX[rLocalBase+u32(q.x+4*q.y+16*q.z)];}\n return pF(pPlane(pSrc),pNear0(at,slot,lb,f));');
 const [a,b]=fnSpan(code,'pRelax');let body=code.slice(a,b);
 body=replaceOnce(body,'let j=j0+(lane>>5u);',`let j=j0+(lane>>5u);rLocalBase=64u*(lane>>5u);
  for(var k=lane;k<128u;k+=64u){
   var x=0.0;let tile=j0+k/64u;
   if(tile<n){let w=pIndex(TLIST+tile);let q=vec3u(k&3u,(k>>2u)&3u,(k>>4u)&3u);x=pF(pPlane(pSrc),pAddr(pTileSlot(w),pTileCell(w,q)));}
   rLocalX[k]=x;
  }
  workgroupBarrier();`);
 body=replaceOnce(body,'pSet(pPlane(1u-pSrc),at,x);','pSet(pPlane(1u-pSrc),at,x);rLocalX[rLocalBase+l.x+4u*(l.y+4u*l.z)]=x;');
 body=replaceOnce(body,'storageBarrier();','workgroupBarrier();');
 return replaceFn(code,'pRelax',body);
}

export function depositLanes(code:string,lanes:number):string {
 code=replaceOnce(code,'const PB_LANES:u32=192u;',`const PB_LANES:u32=${lanes}u;`);
 return replaceOnce(code,'@workgroup_size(192) fn transferDeposit',`@workgroup_size(${lanes}) fn transferDeposit`);
}

export function depositGather(code:string):string {
 code=depositLanes(code,96);
 const helper=`
// A destination lane accumulates the same rounded integer contributions.
// Dense tiles fall back to the original scatter; setup is inside the kernel.
var<workgroup> pgPosition:array<vec4f,512>;
var<workgroup> pgVelocity:array<vec4f,512>;
var<workgroup> pgStarts:array<u32,65>;
fn pgGather(slot:u32,start:u32){
 let axis=slot/PB_WINDOW;let node=pbNode(slot);
 var offset=vec3f(0.5);offset[axis]=1.0;
 let low=max(node-vec3i(1),vec3i(0));var high=node+vec3i(1);high[axis]+=1;high=min(high,vec3i(3));
 var momentum=0;var weight=0u;
 for(var z=low.z;z<=high.z;z++){for(var y=low.y;y<=high.y;y++){
  let a=u32(low.x+4*y+16*z);let b=u32(high.x+1+4*y+16*z);
  for(var i=pgStarts[a];i<pgStarts[b];i++){
   let p=pgPosition[i-start];if(p.w==0.0){continue;}
   let base=vec3i(floor(p.xyz-offset-0.5));let r=node-base;
   if(any(r<vec3i(0))||any(r>vec3i(2))){continue;}
   let f=p.xyz-offset-vec3f(base);let wx=pbQuadratic(f.x);let wy=pbQuadratic(f.y);let wz=pbQuadratic(f.z);
   let wyz=wy[r.y]*wz[r.z];let w=wx[r.x]*wyz;
   momentum+=i32(round(w*pgVelocity[i-start][axis]*PB_MOMENTUM));weight+=u32(round(w*PB_WEIGHT));
  }
 }}
 atomicStore(&pbMomentum[slot],momentum);atomicStore(&pbWeight[slot],weight);
}
`;
 code=replaceOnce(code,'// Fixed direct groups stride the occupied tile list',helper+'\n// Fixed direct groups stride the occupied tile list');
 code=replaceOnce(code,'if(fits&&supported){pbScatter(clamp(q-vec3f(origin),vec3f(0.0),vec3f(3.9999998)),s.velocity.xyz);}',
 `if(end-start<=512u){pgPosition[i-start]=vec4f(clamp(q-vec3f(origin),vec3f(0.0),vec3f(3.9999998)),f32(supported));pgVelocity[i-start]=s.velocity;}
   if(fits&&supported&&end-start>512u){pbScatter(clamp(q-vec3f(origin),vec3f(0.0),vec3f(3.9999998)),s.velocity.xyz);}`);
 const [a,b]=fnSpan(code,'transferDeposit');let body=code.slice(a,b);
 body=replaceOnce(body,'if(!fits){continue;}\n  workgroupBarrier();',`if(!fits){continue;}
  if(lane<64u){pgStarts[lane]=links[64u*tile+lane];}if(lane==0u){pgStarts[64]=end;}
  workgroupBarrier();
  if(end-start<=512u){for(var slot=lane;slot<PB_NODES;slot+=PB_LANES){pgGather(slot,start);}}
  workgroupBarrier();`);
 return replaceFn(code,'transferDeposit',body);
}
