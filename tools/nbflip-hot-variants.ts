/** Same-arithmetic candidates for real-scene hot-kernel replay. */
import {fnSpan,replaceFn,replaceOnce,probeVariants} from './probe-nbflip-variants';
import {depositLanes,pressureTilePacket,surfacePacket} from './nbflip-stage-replay-variants';
export function separationCompact(code:string,streamOnly=false){
 return replaceOnce(code,'let other=particles[j].position;if(other.w<=0.0){continue;}',streamOnly?
  'let other=vec4f(nbPosition(j),1.0);':
  'let other=vec4f(nbPosition(j),particles[j].position.w);if(other.w<=0.0){continue;}');
}
export function transferGuard(code:string){
 return replaceOnce(code,'atomicMax(&state[5],bitcast<u32>(outside));','if(bitcast<u32>(outside)!=0u){atomicMax(&state[5],bitcast<u32>(outside));}');
}
export function transferLean(code:string){
 return replaceOnce(code,'let census=params.settings.w!=0.0;','let census=false;');
}
export function transferAggregate(code:string){
 code=replaceOnce(code,'var<workgroup> pbJobs:u32;','var<workgroup> pbJobs:u32;\nvar<workgroup> pbStats:array<atomic<u32>,4>;');
 const [a,b]=fnSpan(code,'transferDeposit');let body=code.slice(a,b);
 body=replaceOnce(body,'let fits=nbTransferFits();','if(lane<4u){atomicStore(&pbStats[lane],0u);}\n let fits=nbTransferFits();');
 body=replaceOnce(body,'let count=workgroupUniformLoad(&pbJobs);','let count=workgroupUniformLoad(&pbJobs);if(group.x>=count){return;}');
 body=replaceOnce(body,`atomicMax(&state[5],bitcast<u32>(outside));
 for(var k=0u;k<3u;k++){if(counts[k]!=0u){atomicAdd(&state[6u+k],counts[k]);}}`,
 `if(bitcast<u32>(outside)!=0u){atomicMax(&pbStats[0],bitcast<u32>(outside));}
 for(var k=0u;k<3u;k++){if(counts[k]!=0u){atomicAdd(&pbStats[k+1u],counts[k]);}}
 workgroupBarrier();
 if(lane<4u){let value=atomicLoad(&pbStats[lane]);if(value!=0u){if(lane==0u){atomicMax(&state[5],value);}else{atomicAdd(&state[5u+lane],value);}}}`);
 return replaceFn(code,'transferDeposit',body);
}
export function pressureStableWalls(code:string){
 const [a,b]=fnSpan(code,'pRelaxRow');let body=code.slice(a,b);
 body=replaceOnce(body,'for(var it=0u;it<4u;it++){','var previous=0xffffffffu;\n   for(var it=0u;it<4u;it++){var wallMask=0u;');
 body=replaceOnce(body,'if(v+g>=-base){d-=w;o+=w*g;}','if(v+g>=-base){wallMask|=1u<<f;d-=w;o+=w*g;}');
 body=replaceOnce(body,'if(d>0.0){v=o/d;}','if(wallMask==previous){break;}previous=wallMask;\n    if(d>0.0){v=o/d;}');
 return replaceFn(code,'pRelaxRow',body);
}
export function pressureUnroll(code:string){
 const [a,b]=fnSpan(code,'pRelaxRow');let body=code.slice(a,b);
 const loop=/for\(var f=0u;f<6u;f\+\+\)\{([\s\S]*?)\n \}/;const m=body.match(loop);if(!m)throw new Error('pRelaxRow face loop changed');
 body=replaceOnce(body,m[0],Array.from({length:6},(_,f)=>`{const f=${f}u;${m[1]}\n }`).join('\n'));
 return replaceFn(code,'pRelaxRow',body);
}
export function surfaceShape(code:string,entry:string,lanes:number){
 if(lanes===256)return code;
 const [a,b]=fnSpan(code,entry);code=replaceFn(code,entry,code.slice(a,b).replaceAll('256u',`${lanes}u`));
 code=replaceOnce(code,`@workgroup_size(256) fn ${entry}(`,`@workgroup_size(${lanes}) fn ${entry}(`);
 if(entry==='arenaDistance'){
  code=replaceOnce(code,'nbArenaQueue:array<u32,2048>',`nbArenaQueue:array<u32,${8*lanes}>`);
  code=replaceOnce(code,'nbArenaFound:array<atomic<u32>,256>',`nbArenaFound:array<atomic<u32>,${lanes}>`);
 }
 return code;
}
export function moveLanes(code:string,lanes:number){
 const [a,b]=fnSpan(code,'nbMove');let body=code.slice(a,b);
 body=replaceOnce(body,'if(nbWinOn){nbWinStage(lane);}',`if(nbWinOn){for(var w=lane;w<64u;w+=${lanes}u){nbWinStage(w);}}`);
 body=replaceOnce(body,'off+=64u',`off+=${lanes}u`);
 code=replaceFn(code,'nbMove',body);
 return replaceOnce(code,'@workgroup_size(64) fn advectParticlesDeferred',`@workgroup_size(${lanes}) fn advectParticlesDeferred`);
}
export function moveZeroShift(code:string){
 return replaceOnce(code,'fn nbSurfaceShift(q:vec3f,d:vec3f)->vec3f{','fn nbSurfaceShift(q:vec3f,d:vec3f)->vec3f{\n if(all(d==vec3f(0.0))){return d;}');
}
export function moveHeatCache(code:string){
 code=replaceOnce(code,'p.velocity=vec4f(v,depth);p.position.w=max(p.position.w,nbMoveHeat(q));','let heatAt=q;let homeHeat=nbMoveHeat(q);\n   p.velocity=vec4f(v,depth);p.position.w=max(p.position.w,homeHeat);');
 return replaceOnce(code,'let tileHeat=nbMoveHeat(q);','var tileHeat=homeHeat;if(any(q!=heatAt)){tileHeat=nbMoveHeat(q);}');
}
export function coarseOffsets(code:string){
 return replaceFn(code,'cOff',`fn cOff(c:vec3i,cell:CCell)->f32{
 let at=cBox(c);var off=0.0;
 ${['-1u','+1u','-12u','+12u','-144u','+144u'].map((offset,f)=>`off+=select(0.0,cell.k[${f}]*cX[at${offset}],cell.k[${f}]!=0.0);`).join('\n ')}
 return off;
}`);
}
export function coarseHalves(code:string){
 return replaceOnce(code,`for(var half=0u;half<2u;half++){
   var c=red;var cell=cR0;if(half==1u){c=black;cell=cB0;}
   if(cell.d>0.0){cX[cBox(c)]=(cell.r+cOff(c,cell))/cell.d;}
   workgroupBarrier();
  }`,`if(cR0.d>0.0){cX[cBox(red)]=(cR0.r+cOff(red,cR0))/cR0.d;}
  workgroupBarrier();
  if(cB0.d>0.0){cX[cBox(black)]=(cB0.r+cOff(black,cB0))/cB0.d;}
  workgroupBarrier();`);
}
const coarseEntry=(e:string)=>['pDownA','pDownB','pUpA','pUpB'].includes(e);
export const hotVariants:Record<string,(code:string,entry:string)=>string>={
 transfer128:(c,e)=>e==='transferDeposit'?depositLanes(c,128):c,
 moveZero:(c,e)=>e==='advectParticlesDeferred'?moveZeroShift(c):c,
 moveHeat:(c,e)=>e==='advectParticlesDeferred'?moveHeatCache(c):c,
 move128Zero:(c,e)=>e==='advectParticlesDeferred'?moveZeroShift(moveLanes(c,128)):c,
 lean192:(c,e)=>e==='transferDeposit'?transferGuard(transferLean(c)):c,
 lean128:(c,e)=>e==='transferDeposit'?transferGuard(transferLean(depositLanes(c,128))):c,
 aggregate128:(c,e)=>e==='transferDeposit'?transferAggregate(depositLanes(c,128)):c,
 leanAggregate128:(c,e)=>e==='transferDeposit'?transferAggregate(transferLean(depositLanes(c,128))):c,
 hotA:(c,e)=>coarseEntry(e)?coarseOffsets(c):e==='advectParticlesDeferred'?moveLanes(c,128):e==='separationPrepass'?separationCompact(c):e==='transferDeposit'?transferGuard(c):e==='pRelax'?pressureStableWalls(c):e==='arenaDistanceCanonical'?surfacePacket(c,128):e.startsWith('arena')?surfaceShape(c,e,128):c,
 hotB:(c,e)=>coarseEntry(e)?coarseHalves(c):e==='advectParticlesDeferred'?moveLanes(c,32):e==='separationPrepass'?separationCompact(c,true):e==='transferDeposit'?depositLanes(transferGuard(c),96):e==='pRelax'?pressureUnroll(c):e==='arenaDistanceCanonical'?surfacePacket(c,64):e.startsWith('arena')?surfaceShape(c,e,64):c,
 hotC:(c,e)=>coarseEntry(e)?coarseHalves(coarseOffsets(c)):e==='advectParticlesDeferred'?moveLanes(c,256):e==='separationPrepass'?probeVariants.separationselect!(c,e):e==='transferDeposit'?depositLanes(transferGuard(c),128):e==='pRelax'?pressureTilePacket(c):c,
};
