/** Text variants of generated NB-FLIP shader modules for tools/probe-nbflip-variant-replay.ts.
 * "ko" variants are knock-outs: they change values and only bound a cost. */
import assert from 'node:assert/strict';
import {fineSampleContext} from './qa-fine-sample-context';
/** The text of `fn name(` through its matching closing brace. */
export function fnSpan(code:string,name:string):[number,number]{
 const start=code.search(new RegExp(`\\bfn ${name}\\s*\\(`));assert.ok(start>=0,`fn ${name}`);
 let at=code.indexOf('{',start),depth=0;
 for(;at<code.length;at++){const c=code[at];if(c==='{')depth++;else if(c==='}'){depth--;if(depth===0)break;}}
 return [start,at+1];
}
export function replaceFn(code:string,name:string,text:string):string{const [a,b]=fnSpan(code,name);return code.slice(0,a)+text+code.slice(b);}
export function replaceOnce(code:string,a:string,b:string):string{assert.equal(code.split(a).length,2,`one occurrence: ${a.slice(0,80)}`);return code.replace(a,b);}
export const probeVariants:Record<string,(code:string,entry:string)=>string>={
 ctx:code=>fineSampleContext(code),
 // Sample-scoped atlas home with no page-crossing fallback: what a haloed unit cache would cost (values near seams wrong).
 koctx:code=>{
  const out=fineSampleContext(code);
  return replaceOnce(out," if(any((lo>>vec3u(5u))!=page)||any((hi>>vec3u(5u))!=page)){return vec4i(0,0,0,-1);}\n","");
 },
 // Counters in the unused state[15]: fallbacks of ctx's page check, and all fine-weighted samples.
 cntfb:code=>replaceOnce(fineSampleContext(code)," if(any((lo>>vec3u(5u))!=page)||any((hi>>vec3u(5u))!=page)){return vec4i(0,0,0,-1);}"," if(any((lo>>vec3u(5u))!=page)||any((hi>>vec3u(5u))!=page)){atomicAdd(&state[15],1u);return vec4i(0,0,0,-1);}"),
 cntall:code=>replaceOnce(fineSampleContext(code)," let block=s.key>>9u;if(!umBlockStored(block))"," atomicAdd(&state[15],1u);let block=s.key>>9u;if(!umBlockStored(block))"),
 // koctx keeping production's exact low-wall-face path (one page lookup per axis sample, as production pays).
 koflow:code=>replaceOnce(probeVariants.koctx!(code,'')," if(atlas.w>=0){\n"," if(atlas.w>=0&&!umFaceLow(base,axis)){\n"),
 cntflow:code=>replaceOnce(probeVariants.koctx!(code,'')," if(atlas.w>=0){\n"," if(atlas.w>=0&&umFaceLow(base,axis)){atomicAdd(&state[15],1u);}\n if(atlas.w>=0){\n"),
 // The unit halo's fast path off: the original unit sampler (the A/B of the halo).
 nohalo:code=>code.includes(" if(umUnitHome.w>=0){\n")?replaceOnce(code," if(umUnitHome.w>=0){\n"," if(false){\n"):code,
 // Halo vs original unit sampler per axis sample: state[15] counts mismatches, +1 not at a low wall face, +65536 at one.
 halochk:code=>{
  if(!code.includes('umUnitHome.w>=0'))return code;
  const [a,b]=fnSpan(code,'umSampleVelocity1');const body=code.slice(a,b);
  const orig=replaceOnce(body.replace('fn umSampleVelocity1(','fn umSampleVelocity1Orig('),' if(umUnitHome.w>=0){\n',' if(false){\n');
  const wrap=`fn umSampleVelocity1(p:vec3f,axis:u32)->f32 {
 let o=umSampleVelocity1Orig(p,axis);
 if(umUnitHome.w>=0){
  var offset=vec3f(0.5);offset[axis]=1.0;let b=umHold;var lower=b.lo/1.0;lower[axis]-=1.0;
  let q=clamp(p/1.0-offset,lower,b.hi/1.0-vec3f(1.0));let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
  for(var k=0u;k<8u;k++){terms[k]=textureLoad(unitHalo,base+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u))+umUnitHome.xyz,0)[axis];}
  let h=umVelocityLerp8(terms,fraction);
  if(h!=o){atomicAdd(&state[15],select(1u,65536u,umFaceLow(base,axis)));}
 }
 return o;
}`;
  return code.slice(0,a)+orig+'\n'+wrap+code.slice(b);
 },
 // halochk recording the largest |halo - original| (f32 bits) in state[15]; halorel: relative to max(|original|,1e-3).
 haloabs:code=>probeVariants.halochk!(code,'').replace('if(h!=o){atomicAdd(&state[15],select(1u,65536u,umFaceLow(base,axis)));}','atomicMax(&state[15],bitcast<u32>(abs(h-o)));'),
 // APIC taps knocked out (cost bound of nbApicTap), and the whole APIC sample.
 koapictap:code=>replaceFn(code,'nbApicTap','fn nbApicTap(index:vec3i,axis:u32)->f32{return f32(index.x+index.y+index.z)*1e-3;}'),
 kopic:code=>replaceOnce(code,'pic=sampleVelocity(particles[i].position.xyz);','pic=particles[i].velocity.xyz;'),
 // The APIC halo path off.
 noapichalo:code=>code.includes('if(nbApicHome.w>=0){')?replaceOnce(code,'if(nbApicHome.w>=0){','if(false){'):code,
 // Count APIC halo fallbacks (state[15] += 1) and halo attempts (+65536).
 cntapic:code=>replaceOnce(replaceOnce(code,'  nbApicHome=vec4i(-1);\n }','  atomicAdd(&state[15],1u);nbApicHome=vec4i(-1);\n }'),' var rows:array<vec4f,3>;\n',' var rows:array<vec4f,3>;if(nbApicHome.w>=0){atomicAdd(&state[15],65536u);}\n'),
 // Largest |halo - umVelocityTap1| of an APIC row's velocity (w) or affine terms (xyz), f32 bits in state[15].
 apicabsv:code=>replaceOnce(code,' let x=rows[0];let y=rows[1];let z=rows[2];',` if(nbApicHome.w>=0){nbApicHome=vec4i(-1);var r2:array<vec4f,3>;for(var a=0u;a<3u;a++){r2[a]=nbApicRow(p.position.xyz,a);}
  for(var a=0u;a<3u;a++){atomicMax(&state[15],bitcast<u32>(abs(r2[a].w-rows[a].w)));}}
 let x=rows[0];let y=rows[1];let z=rows[2];`),
 apicabsc:code=>replaceOnce(code,' let x=rows[0];let y=rows[1];let z=rows[2];',` if(nbApicHome.w>=0){nbApicHome=vec4i(-1);var r2:array<vec4f,3>;for(var a=0u;a<3u;a++){r2[a]=nbApicRow(p.position.xyz,a);}
  for(var a=0u;a<3u;a++){let d=abs(r2[a].xyz-rows[a].xyz);atomicMax(&state[15],bitcast<u32>(max(d.x,max(d.y,d.z))));}}
 let x=rows[0];let y=rows[1];let z=rows[2];`),
 // halochk/haloabs in the surface module, whose flip state is nbState.
 halochks:code=>probeVariants.halochk!(code,'').replace('&state[15]','&nbState[15]'),
 haloabss:code=>probeVariants.haloabs!(code,'').replace('&state[15]','&nbState[15]'),
 haloabsfl:code=>probeVariants.haloabss!(code,'').replace('atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));','if(umFaceLow(base,axis)){atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));}'),
 haloabsnf:code=>probeVariants.haloabss!(code,'').replace('atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));','if(!umFaceLow(base,axis)){atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));}'),
 // Flags of halo/original mismatches above 1e-3 in nbState[15]: 1 low face, 2 a tap off the home page, 4 home block free,
 // 8 a tap's halo texel differs from its unit texel, 16 a tap outside the halo reach, 32 a tap in an unstored page, 64 a tap's tile neither unit nor slotted.
 haloflags:code=>probeVariants.haloabss!(code,'').replace('atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));',`if(abs(h-o)>1e-3){
  let s=umSampleTileHeld(p);let page=s.tile>>vec3u(3u);let block=s.key>>9u;var f=select(0u,1u,umFaceLow(base,axis));
  if(umBlockWord(block)==UM_FREE){f|=4u;}if(any(umBlockPage(block)!=vec3u(page))){f|=4096u;}if(any(umTileCoord(s.key)!=vec3u(s.tile))){f|=8192u;}
  for(var k=0u;k<8u;k++){let t=base+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let l=t-(page<<vec3u(5u));
   if(any((t>>vec3u(5u))!=page)){f|=2u;}if(any(l< vec3i(-2))||any(l>vec3i(32))){f|=16u;}
   if(!umCellPresent(t)){f|=32u;}
   let key=umTileAt(vec3u(t)/4u);if(umTileWidth(key)!=1u&&umHanging[key]==UM_NO_SLOT){f|=64u;}
   if(any(textureLoad(unitHalo,t+umUnitHome.xyz,0).xyz!=textureLoad(unitVelocity,t,0).xyz)){f|=select(select(128u,256u,umCellKind(t)==UM_SOLID),select(2048u,1024u,any((t>>vec3u(5u))!=page)),umCellPresent(t));}}
  atomicOr(&nbState[15],f);}`),
 // At the entry's start: count halo texels (w=1, not SOLID) of stored blocks whose xyz differ from their unitVelocity texel (nbState[15]).
 halostale:(code,entry)=>{
  const id=entry==='wallReach'?'group.x*64u+lane':'gid.x';
  const m=code.match(new RegExp(`fn ${entry}\\(.*?\\)\\{`));assert.ok(m,entry);
  return code.replace(m[0],`${m[0]}
 {let E=textureDimensions(unitHalo);let n=E.x*E.y*E.z;let stride=groups.x*64u;
  for(var i=${id};i<n;i+=stride){let g=vec3u(i%E.x,(i/E.x)%E.y,i/(E.x*E.y));let cell=g/35u;if(any(cell>=UM_BLOCK_GRID())){continue;}
   let k=UM_BLOCK_BITS();let b=cell.x|(cell.y<<k)|(cell.z<<(2u*k));if(!umBlockStored(b)||umBlockWord(b)==UM_FREE){continue;}
   let tap=(vec3i(umBlockPage(b))<<vec3u(5u))+vec3i(g%35u)-vec3i(2);let h=textureLoad(unitHalo,vec3i(g),0);
   if(h.w==0.0||umCellKind(tap)==UM_SOLID){continue;}
   if(any(h.xyz!=textureLoad(unitVelocity,umCellTexel(tap),0).xyz)){atomicAdd(&nbState[15],1u);}}}
`);
 },
 // Per weighted tap of a non-low-face halo/original mismatch above 1e-3, flags in nbState[15]: 1 SOLID tap, 2 halo w 0,
 // 4 unstored non-SOLID tap, 8 stored w-1 non-SOLID tap, 16 tap off the home page, 32 the fraction leaves the tap 0 weight,
 // 64 umUnitHome differs from the held sample's home, 128 no differing tap.
 halotaps:code=>probeVariants.haloabss!(code,'').replace('atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));',`if(abs(h-o)>1e-3&&!umFaceLow(base,axis)){
  let s=umSampleTileHeld(p);let page=s.tile>>vec3u(3u);var f=select(0u,64u,any(umUnitHaloHome(s)!=umUnitHome));var seen=false;
  for(var k=0u;k<8u;k++){let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let t=base+bit;
   let hv=textureLoad(unitHalo,t+umUnitHome.xyz,0);if(hv[axis]==textureLoad(unitVelocity,umCellTexel(t),0)[axis]){continue;}seen=true;
   let ws=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));if(ws.x*ws.y*ws.z==0.0){f|=32u;}
   if(umCellKind(t)==UM_SOLID){f|=1u;}else if(!umCellPresent(t)){f|=4u;}else if(hv.w==0.0){f|=2u;}else{f|=8u;}
   if(any((t>>vec3u(5u))!=page)){f|=16u;}}
  if(!seen){f|=128u;}atomicOr(&nbState[15],f);}`),
 // halotaps for low-face samples, a tap's reference being umVelocityTap1 (256: the differing tap is a low face on axis).
 halotapsfl:code=>probeVariants.haloabss!(code,'').replace('atomicMax(&nbState[15],bitcast<u32>(abs(h-o)));',`if(abs(h-o)>1e-3&&umFaceLow(base,axis)){
  let s=umSampleTileHeld(p);let page=s.tile>>vec3u(3u);var f=select(0u,64u,any(umUnitHaloHome(s)!=umUnitHome));var seen=false;
  for(var k=0u;k<8u;k++){let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let t=base+bit;
   let ws=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));if(ws.x*ws.y*ws.z==0.0){continue;}
   let hv=textureLoad(unitHalo,t+umUnitHome.xyz,0);if(hv[axis]==umVelocityTap1(t,axis)){continue;}seen=true;
   if(umFaceLow(t,axis)){f|=256u;}
   if(umCellKind(t)==UM_SOLID){f|=1u;}else if(!umCellPresent(t)){f|=4u;}else if(hv.w==0.0){f|=2u;}else{f|=8u;}
   if(any((t>>vec3u(5u))!=page)){f|=16u;}}
  if(!seen){f|=128u;}atomicOr(&nbState[15],f);}`),
 // At advectDeferred's start: the largest |xyz| of the unit taps' last (SOLID stand-in) and first (AIR stand-in) blocks, f32 bits in nbState[15].
 sentinels:code=>{
  const m=code.match(/fn advectDeferred\(.*?\)\{/);assert.ok(m);
  return code.replace(m[0],`${m[0]}
 for(var i=gid.x;i<65536u;i+=groups.x*64u){let l=vec3i(vec3u(i&31u,(i>>5u)&31u,i>>10u));
  for(var s=0u;s<2u;s++){let b=select(0u,UM_BLOCKS()-1u,s==1u);let v=abs(textureLoad(unitVelocity,(umBlockCoord(b)<<vec3u(5u))+l,0).xyz);
   atomicMax(&nbState[15],bitcast<u32>(max(v.x,max(v.y,v.z))));}}
`);
 },
 // markParticleTiles knock-outs: every group returns at once (launch cost), or after its population test (empty-tile cost).
 komarkall:code=>replaceOnce(code,' let tile=group.x+65535u*group.y;if(!umTileLive(tile)){return;}',' let tile=group.x+65535u*group.y;if(tile<0xffffffffu){return;}'),
 komarkpop:code=>replaceOnce(code,' if(workgroupUniformLoad(&nbMarkedPopulation)==0u){return;}',' if(workgroupUniformLoad(&nbMarkedPopulation)<0xffffffffu){return;}'),
 komarkslow:code=>replaceOnce(code,'{markSurfaceTiles(q,h);continue;}','{continue;}'),
 komarkloop:code=>replaceOnce(code,' for(var i=start;i<stop;i++){\n  let a=UM_CELL_KEYS',' live=stop>start;for(var i=start;i<select(start,stop,heat<0.0);i++){\n  let a=UM_CELL_KEYS'),
 komarkglobal:code=>code.split('   atomicStore(&bins[NB_SURFACE_TILES+markedTile],1u);\n   atomicMax(&bins[NB_ACTIVITY_THETA+markedTile],atomicLoad(&nbMarkedHeat[lane]));\n').join(''),
 // markParticleTiles with its stray-sample marks back inside the loop (the shape before the split).
 markinline:code=>replaceOnce(replaceOnce(code,'{stray=true;continue;}','{markSurfaceTiles(q,h);continue;}'),' if(stray){',' if(false){'),
 kostray:code=>replaceOnce(code,' if(stray){',' if(false){'),
 // The particle loop reads only the count (cost of the loop's loads and tests).
 komarkload:code=>replaceOnce(code,'  if(any(lo!=reachLow)||any(hi!=reachHigh)){stray=true;continue;}\n  live=true;heat=max(heat,h);','  live=true;heat=max(heat,h+f32(lo.x+hi.x)*0.0);'),
 // markParticleTiles with the tile's samples strided over its lanes (each finds its cell among the 64 sorted runs).
 markstride:code=>{
  const a=code.indexOf(' if(lane==0u){atomicStore(&nbMarkedMask,0u);atomicStore(&nbOwnHeat,0u);}');const b=code.indexOf(' if(live){\n  var mask=0u;');
  assert.ok(a>0&&b>a);
  const body=` if(lane==0u){atomicStore(&nbMarkedMask,0u);atomicStore(&nbOwnHeat,0u);atomicStore(&nbStray,0u);atomicStore(&nbCellLive[0],0u);atomicStore(&nbCellLive[1],0u);}
 if(lane<27u){atomicStore(&nbMarkedHeat[lane],0u);}
 let begin=links[first];nbCellStart[lane]=links[first+lane]-begin;atomicStore(&nbCellHeat[lane],0u);
 let population=nbMarkedPopulation;workgroupBarrier();
 let local=umCorner(lane,4u);let coord=vec3i(umTileCoord(tile));
 let reachLow=coord+vec3i(local/2u)-1;let reachHigh=coord+vec3i(local/2u);
 let low=umTileNearFromPage(tile>>9u,coord>>vec3u(3u),reachLow);let high=umTileNearFromPage(tile>>9u,coord>>vec3u(3u),reachHigh);
 for(var j=lane;j<population;j+=64u){
  // The sample's cell: the last run starting at or before it.
  var cell=0u;for(var step=32u;step>0u;step>>=1u){if(nbCellStart[cell+step]<=j){cell+=step;}}
  let a=UM_CELL_KEYS()+4u*(begin+j);let q=bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));let h=max(0.0,bitcast<f32>(links[a+3u]));
  let l=umCorner(cell,4u);let rl=coord+vec3i(l/2u)-1;let rh=coord+vec3i(l/2u);
  let lo=vec3i(floor((q-2.0)/4.0));let hi=vec3i(floor((q+2.0)/4.0));
  if(any(lo!=rl)||any(hi!=rh)){atomicStore(&nbStray,1u);continue;}
  atomicMax(&nbCellHeat[cell],select(bitcast<u32>(h),0u,h==0.0));atomicOr(&nbCellLive[cell>>5u],1u<<(cell&31u));
 }
 if(workgroupUniformLoad(&nbStray)!=0u){
  for(var j=lane;j<population;j+=64u){
   var cell=0u;for(var step=32u;step>0u;step>>=1u){if(nbCellStart[cell+step]<=j){cell+=step;}}
   let a=UM_CELL_KEYS()+4u*(begin+j);let q=bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));
   let l=umCorner(cell,4u);let rl=coord+vec3i(l/2u)-1;let rh=coord+vec3i(l/2u);
   let lo=vec3i(floor((q-2.0)/4.0));let hi=vec3i(floor((q+2.0)/4.0));
   if(any(lo!=rl)||any(hi!=rh)){markSurfaceTiles(q,max(0.0,bitcast<f32>(links[a+3u])));}
  }
 }
 let live=((atomicLoad(&nbCellLive[lane>>5u])>>(lane&31u))&1u)!=0u;let heat=bitcast<f32>(atomicLoad(&nbCellHeat[lane]));
`;
  const out=code.slice(0,a)+body+code.slice(b);
  return replaceOnce(out,'var<workgroup> nbMarkedPopulation:u32;\n','var<workgroup> nbMarkedPopulation:u32;\nvar<workgroup> nbCellStart:array<u32,64>;\nvar<workgroup> nbCellHeat:array<atomic<u32>,64>;\nvar<workgroup> nbCellLive:array<atomic<u32>,2>;\nvar<workgroup> nbStray:atomic<u32>;\n');
 },
 // canonical knock-outs: no rank loop (scatter cost), and the rank loop's compares by position only.
 kocanonloop:code=>replaceOnce(code,'  for(var k=start;k<end;k++){if(k!=s&&orderBefore(orderKey(k),own)){rank++;}}\n','  rank=s-start;\n'),
 canonpos:code=>replaceOnce(code,'  for(var k=start;k<end;k++){if(k!=s&&orderBefore(orderKey(k),own)){rank++;}}\n','  for(var k=start;k<end;k++){let o=orderKey(k);if(k!=s&&(o.x<own.x||(o.x==own.x&&(o.y<own.y||(o.y==own.y&&o.z<own.z))))){rank++;}}\n'),
 // canonical with exact position ties ranked in a second loop (the tie-breaking compare out of the hot loop).
 canontie:code=>replaceOnce(code,'  for(var k=start;k<end;k++){if(k!=s&&orderBefore(orderKey(k),own)){rank++;}}\n',`  var tie=false;
  for(var k=start;k<end;k++){let o=orderKey(k).xyz;
   let less=o.x<own.x||(o.x==own.x&&(o.y<own.y||(o.y==own.y&&o.z<own.z)));
   rank+=select(0u,1u,less);tie=tie||(all(o==own.xyz)&&k!=s);}
  if(tie){for(var k=start;k<end;k++){let o=orderKey(k);if(k!=s&&all(o.xyz==own.xyz)&&orderBefore(o,own)){rank++;}}}
`),
 // canonical's tie-breaking compare loading one field of each record at a time (no whole Particle copies).
 canonlazy:code=>{
  const fields=['position','velocity','before','cx','cy','cz','lifetime'];
  const body=fields.map(f=>` {let x=bitcast<vec4u>(input[i].${f});let y=bitcast<vec4u>(input[j].${f});for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return select(1,-1,x[k]<y[k]);}}}`).join('\n');
  let out=code.replace(/fn orderCompare\(a:Particle,b:Particle\)->i32\{[\s\S]*?\n return 0;\n\}/,`fn orderCompare(i:u32,j:u32)->i32{\n${body}\n return 0;\n}`);
  assert.ok(out!==code);
  return replaceOnce(out,'let order=orderCompare(input[a.w],input[b.w]);','let order=orderCompare(a.w,b.w);');
 },
 canonlazytie:(code,entry)=>probeVariants.canontie!(probeVariants.canonlazy!(code,entry),entry),
 canontienone:(code,entry)=>replaceOnce(probeVariants.canontie!(code,entry),'  if(tie){','  if(false){'),
 // Count canonical lanes with an exact position tie (+1 in bins[0]) and their cells' populations (+65536*n... no: bins[1] += n).
 canontiecount:(code,entry)=>replaceOnce(probeVariants.canontie!(code,entry),'  if(tie){','  if(tie){atomicAdd(&bins[0],1u);}\n  if(tie){'),
 // canonical ranking against the workgroup's cells' keys staged 64 at a time in workgroup memory.
 canonshared:code=>replaceFn(replaceOnce(code,'@compute @workgroup_size(64) fn canonical(','var<workgroup> canonKeys:array<vec4u,64>;\nvar<workgroup> canonRange:array<atomic<u32>,2>;\nvar<workgroup> canonBounds:vec3u;\n@compute @workgroup_size(64) fn canonical('),'canonical',`fn canonical(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){canonBounds.z=min(state[1],arrayLength(&output));}
 let live=workgroupUniformLoad(&canonBounds).z;
 for(var base=64u*(group.x+groups.x*group.y);base<live;base+=64u*groups.x*groups.y){
  let s=base+lane;let inside=s<live;
  if(lane==0u){atomicStore(&canonRange[0],0xffffffffu);atomicStore(&canonRange[1],0u);}
  workgroupBarrier();
  var own=vec4u(0u);var start=0u;var end=0u;
  if(inside){own=orderKey(s);let cell=orderCellKey(vec3u(bitcast<vec3f>(own.xyz)));start=links[cell];end=start+atomicLoad(&bins[cell]);atomicMin(&canonRange[0],start);atomicMax(&canonRange[1],end);}
  workgroupBarrier();
  if(lane==0u){canonBounds.x=atomicLoad(&canonRange[0]);canonBounds.y=atomicLoad(&canonRange[1]);}
  let bounds=workgroupUniformLoad(&canonBounds);
  var rank=0u;
  for(var chunk=bounds.x;chunk<bounds.y;chunk+=64u){
   if(chunk+lane<bounds.y){canonKeys[lane]=orderKey(chunk+lane);}
   workgroupBarrier();
   if(inside){
    for(var k=max(start,chunk);k<min(end,chunk+64u);k++){if(k!=s&&orderBefore(canonKeys[k-chunk],own)){rank++;}}
   }
   workgroupBarrier();
  }
  if(inside){output[start+rank]=input[own.w];}
 }
}`),
 // Sum over samples of their cell's population (the rank loop's iterations) in bins[0], and samples with >256 in bins[2].
 canonsumsq:code=>replaceOnce(code,'  var rank=0u;\n  for(var k=start;k<end;k++)','  var rank=0u;atomicAdd(&bins[0],end-start);if(end-start>256u){atomicAdd(&bins[2],1u);}\n  for(var k=start;k<end;k++)'),
 canonbig:code=>replaceOnce(code,'  var rank=0u;\n  for(var k=start;k<end;k++)','  var rank=0u;if(end-start>256u){atomicAdd(&bins[0],1u);}if(s==start){atomicMax(&bins[2],end-start);}\n  for(var k=start;k<end;k++)'),
 // Surface advect knock-outs (cost bounds; values change) and counters (nbState[15]).
 skosmooth:code=>replaceFn(code,'umVertexSmoothing','fn umVertexSmoothing(vertex:vec3u,width:u32)->f32{return 0.0;}'),
 skocubic:code=>replaceFn(code,'umCubicPhi','fn umCubicPhi(q:vec3f,cell:UMVertexCell)->f32{return cell.value;}'),
 skofoot:code=>replaceFn(code,'umSurfaceGradient','fn umSurfaceGradient(p:vec3f,width:f32)->vec3f{return vec3f(0.0);}'),
 skowc:code=>replaceFn(code,'umWallContact','fn umWallContact(p:vec3f,value:f32,width:u32)->f32{return value;}'),
 skorel:code=>replaceFn(code,'umReleasedWalls','fn umReleasedWalls(p:vec3f,value:f32)->f32{return value;}'),
 skoemb:code=>replaceFn(code,'umEmbeddedAir','fn umEmbeddedAir(p:vec3f,end:vec3f,advected:f32)->f32{return advected;}'),
 skotrace:code=>replaceFn(code,'umSurfaceTrace','fn umSurfaceTrace(p:vec3f)->vec3f{return p;}'),
 scnttraced:code=>replaceOnce(code,' let p=vec3f(vertex);var defer=umDeferTraced;var value=0.0;',' atomicAdd(&nbState[15],1u);let p=vec3f(vertex);var defer=umDeferTraced;var value=0.0;'),
 scntall:code=>replaceOnce(code,'fn umAdvectStore(vertex:vec3u,width:u32){','fn umAdvectStore(vertex:vec3u,width:u32){atomicAdd(&nbState[15],1u);'),
 scntstore:code=>replaceOnce(code,'fn umAdvectStore(vertex:vec3u,width:u32){','fn umAdvectStore(vertex:vec3u,width:u32){atomicAdd(&nbState[15],1u);'),
 scntcubic:code=>{const [a]=fnSpan(code,'umCubicPhi');const at=code.indexOf('{',a)+1;return code.slice(0,at)+'atomicAdd(&nbState[15],1u);'+code.slice(at);},
 scntdef:code=>replaceOnce(code,'  let width=select(1u<<code,1u,code==3u);\n','  let width=select(1u<<code,1u,code==3u);atomicAdd(&nbState[15],1u);\n'),
 scntdefw1:code=>replaceOnce(code,'  let width=select(1u<<code,1u,code==3u);\n','  let width=select(1u<<code,1u,code==3u);if(width==1u){atomicAdd(&nbState[15],1u);}\n'),
 scntdefdeep:code=>replaceOnce(code,'  let width=select(1u<<code,1u,code==3u);\n','  let width=select(1u<<code,1u,code==3u);if(umDeepAir(vertex,width)>=0.0){atomicAdd(&nbState[15],1u);}\n'),
 scntdefwall:code=>replaceOnce(code,'  let width=select(1u<<code,1u,code==3u);\n','  let width=select(1u<<code,1u,code==3u);if(umWallPlane(vec3i(vertex))<6u){atomicAdd(&nbState[15],1u);}\n'),
 scntwall:code=>replaceOnce(code,'slot=atomicAdd(&wallListed,1u);','slot=atomicAdd(&wallListed,1u);atomicAdd(&nbState[15],1u);'),
 scntsmooth:code=>{const a='if(!(abs(c)<2.0*f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z)))){return 0.0;}';return replaceOnce(code,a,a+'atomicAdd(&nbState[15],1u);');},
 // Lattice-point phi samples: a sample at a cell's own origin is that corner's load (exact).
 slat:code=>replaceOnce(code,'  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;\n','  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;\n  if(all(t==vec3f(0.0))){if(width==4u){return umLoadCorner(origin);}return umLoadVertex(origin);}\n'),
 // The cubic's 64 taps by one run where its 4^3 box lies in one page's closure (exact).
 scubrun:code=>{
  const [a,b]=fnSpan(code,'umCubicPhi');let body=code.slice(a,b);
  const loop=(load:string,fast:string)=>{
   const generic=`    var s=0.0;if(inner){s=cell.corners[x-1u+2u*(y-1u)+4u*(z-1u)];}else{s=${load}(vec3u(vertex));}row+=wx[x]*s;`;
   assert.ok(body.includes(generic),load);
   const start=body.lastIndexOf(' for(var z=0u;z<4u;z++){var plane=0.0;',body.indexOf(generic));
   const end=body.indexOf('  }value+=wz[z]*plane;\n }',start)+'  }value+=wz[z]*plane;\n }'.length;
   const original=body.slice(start,end);
   const quick=original.replace('let vertex=umVertexNearFromPage(home,page,base+offset*width);','let vertex=base+offset*width;').replace(generic,`    var s=0.0;if(inner){s=cell.corners[x-1u+2u*(y-1u)+4u*(z-1u)];}else{s=textureLoad(${fast},run.xyz+offset+vec3i(1),0).x;}row+=wx[x]*s;`);
   const rb=fast==='phi'?'u32(textureDimensions(phi).x<UM_VERTEX_TEXTURE().x)':'u32(textureDimensions(udb_phi).x<UM_VERTEX_TEXTURE().x)';
   const runw=fast==='phi'?'0':'2';
   body=body.replace(original,` {let run=udVertexRun(base-vec3i(width),3u*u32(width),${rb});\n if(run.w==${runw}){\n${quick}\n }else{\n${original}\n }}`);
  };
  loop('umLoadCorner','udb_phi');loop('umLoadVertex','phi');
  return code.slice(0,a)+body+code.slice(b);
 },
 // Lattice samples on at most one non-integral axis: one or two loads, the 8-tap sum's nonzero terms in its order (exact).
 sline:code=>replaceOnce(code,'  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;\n','  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;\n  {let zt=t==vec3f(0.0);if(u32(zt.x)+u32(zt.y)+u32(zt.z)>=2u){let a=select(select(2u,1u,!zt.y),0u,!zt.x);var e=vec3u(0);e[a]=width;var line=array<f32,2>(0.0,0.0);\n   for(var s=0u;s<select(1u,2u,t[a]!=0.0);s++){let at=origin+e*s;if(width==4u){line[s]=umLoadCorner(at);}else{line[s]=umLoadVertex(at);}}\n   if(t[a]==0.0){return line[0];}return line[0]*(1.0-t[a])+line[1]*t[a];}}\n'),
 // sline through the 8-tap sum itself: zero terms, same products and order.
 sline2:code=>replaceOnce(code,'  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;\n','  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;\n  {let zt=t==vec3f(0.0);if(u32(zt.x)+u32(zt.y)+u32(zt.z)>=2u){let a=select(select(2u,1u,!zt.y),0u,!zt.x);var e=vec3u(0);e[a]=width;\n   for(var s=0u;s<select(1u,2u,t[a]!=0.0);s++){let at=origin+e*s;var v=0.0;if(width==4u){v=umLoadCorner(at);}else{v=umLoadVertex(at);}let corner=e*s/width;let w=select(vec3f(1)-t,t,corner!=vec3u(0));values[s<<a]=v*w.x*w.y*w.z;}\n   return umVertexSum8(values);}}\n'),
 // The cubic's per-tap burial test out of its unrolled 64 taps: one runtime-bounded loop before them (exact: the return does not depend on the tap).
 cubbury:code=>{
  const [a,b]=fnSpan(code,'umCubicPhi');let body=code.slice(a,b);
  const test='    if(!tapsClear&&wx[x]*wy[y]*wz[z]!=0.0&&umBuried(vec3f(vertex))){return cell.value;}\n';
  assert.equal(body.split(test).length,3);body=body.replaceAll(test,'');
  const at=body.indexOf('\n',body.indexOf('let wz=umHeldWeights('))+1;
  body=body.slice(0,at)+' for(var k=0u;k<select(64u,0u,tapsClear);k++){let x=k&3u;let y=(k>>2u)&3u;let z=k>>4u;if(wx[x]*wy[y]*wz[z]==0.0){continue;}\n  let vertex=umVertexNearFromPage(home,page,base+(vec3i(vec3u(x,y,z))-vec3i(1))*width);if(umBuried(vec3f(vertex))){return cell.value;}\n }\n'+body.slice(at);
  return code.slice(0,a)+body+code.slice(b);
 },
 slatcub:(code,entry)=>probeVariants.scubrun!(probeVariants.slat!(code,entry),entry),
 // Runtime skips that keep the code (arrayLength is never zero): work vs code-size bounds.
 dsmoothoff:code=>{const [a]=fnSpan(code,'umVertexSmoothing');const at=code.indexOf('{',a)+1;return code.slice(0,a)+code.slice(a,at)+'if(arrayLength(&umSupport)>0u){return 0.0;}'+code.slice(at);},
 dcuboff:code=>{const [a]=fnSpan(code,'umCubicPhi');const at=code.indexOf('{',a)+1;return code.slice(0,a)+code.slice(a,at)+'if(arrayLength(&umSupport)>0u){return cell.value;}'+code.slice(at);},
 dfootoff:code=>{const [a]=fnSpan(code,'umSurfaceGradient');const at=code.indexOf('{',a)+1;return code.slice(0,a)+code.slice(a,at)+'if(arrayLength(&umSupport)>0u){return vec3f(0.0);}'+code.slice(at);},
 dtraceoff:code=>{const [a]=fnSpan(code,'umSurfaceTrace');const at=code.indexOf('{',a)+1;return code.slice(0,a)+code.slice(a,at)+'if(arrayLength(&umSupport)>0u){return p;}'+code.slice(at);},
 // Any entry returning at its first statement: the launch's fixed cost.
 koentry:(code,entry)=>{const m=code.match(new RegExp(`fn ${entry}\\(.*?\\)\\{`));assert.ok(m,entry);return code.replace(m[0],`${m[0]}\n if(arrayLength(&umSupport)>0u){return;}`);},
 kobounds:code=>replaceFn(code,'umPointBounds','fn umPointBounds(p:vec3f,margin:f32)->umBounds{return umBounds(vec3f(-1e9),vec3f(1e9));}'),
 kosolid:code=>code.includes('fn umSolidEnabled')?replaceFn(code,'umSolidEnabled','fn umSolidEnabled()->bool{return false;}'):code,
};
// One function's solid gate decided false (its solid-free path): which solid code costs where solids are off.
for(const name of ['umTileOpen','umTileOpenAt','umTileSupported','umBedSlope','umBodyCount','umCellOpenGeometry','umCellOpen','umCellIsOpen','umCellInsideSolid','umSolidAtWorldCell','umSolidFaceVelocity','umFaceOpen','umPressureFaceV','umWalk','umSolidClear','umBuried','umCubicPhi','umSampleClosed'])
 probeVariants[`ks${name.slice(2)}`]=code=>{if(!code.includes(`fn ${name}(`))return code;const [a,b]=fnSpan(code,name);return code.slice(0,a)+code.slice(a,b).replaceAll('umSolidEnabled()','false')+code.slice(b);};
probeVariants.kTOC=code=>replaceFn(code,'umTileOpenCells','fn umTileOpenCells(p:vec3i)->f32{return 1.0;}');
probeVariants.kTOCrt=code=>replaceOnce(code,'var open=0.0;for(var i=0u;i<64u;i++){open+=umCellOpen(','var open=0.0;for(var i=0u;i<min(64u,64u*arrayLength(&umSupport));i++){open+=umCellOpen(');
probeVariants.kCOG1=code=>replaceOnce(code,' return umCellOpenGeometry(p);\n}',' if(arrayLength(&umSupport)>0u){return 1.0;}return umCellOpenGeometry(p);\n}');
// umCellOpenGeometry with the body and terrain fractions evaluated at one site each (exact).
probeVariants.cogmerge=code=>replaceFn(code,'umCellOpenGeometry',`fn umCellOpenGeometry(p:vec3i)->f32{
 if(!umSolidEnabled()){return select(0.0,1.0,umSolidValid(p));}
 if(umSolidTileClear(p)){return 1.0;}
 let boundary=umBoundaryRecord(p);
 if(boundary==0u&&umSolidVoxel(p)){return 0.0;}
 let body=umCellBodyFraction(p);let terrain=umCellTerrainFraction(p);
 if(boundary!=0u){return min(umBoundaryBoxOpen(vec3f(p)+0.5,vec3f(1)),min(1.0-body,1.0-terrain));}
 return clamp((1.0-body)*(1.0-terrain),0.0,1.0);
}`);
// The geometry's fixed sample loops runtime-bounded (exact; arrayLength is never zero): unrolled copies vs one.
probeVariants.cogrt=code=>{
 const rt=(n:number)=>`min(${n}u,${n}u*arrayLength(&umSupport))`;
 let c=code;
 for(const [fn,from,n] of [['umBoundaryBoxOpen','for(var k=0u;k<8u;k++){',8],['umBoundarySampleAt','for(var k=0u;k<8u;k++){',8],['umBodyFraction','for(var corner=0u;corner<8u;corner++){',8],['umTerrainBoxOpen','for(var k=0u;k<4u;k++){',4]] as const){
  const [a,b]=fnSpan(c,fn);const body=c.slice(a,b);assert.ok(body.includes(from),fn);
  c=c.slice(0,a)+body.replace(from,from.replace(`<${n}u;`,`<${rt(n)};`))+c.slice(b);
 }
 return c;
};
probeVariants.cogboth=(code,entry)=>probeVariants.cogrt!(probeVariants.cogmerge!(code,entry),entry);
probeVariants.cmNoBody=(code,entry)=>replaceFn(probeVariants.cogmerge!(code,entry),'umCellBodyFraction','fn umCellBodyFraction(p:vec3i)->f32{return 0.0;}');
probeVariants.cmNoTerrain=(code,entry)=>replaceFn(probeVariants.cogmerge!(code,entry),'umCellTerrainFraction','fn umCellTerrainFraction(p:vec3i)->f32{return 0.0;}');
probeVariants.cmNoBox=(code,entry)=>replaceFn(probeVariants.cogmerge!(code,entry),'umBoundaryBoxOpen','fn umBoundaryBoxOpen(q:vec3f,width:vec3f)->f32{return 1.0;}');
probeVariants.cmNoOw=(code,entry)=>replaceFn(probeVariants.cogmerge!(code,entry),'owHeight','fn owHeight(x:f32,z:f32)->f32{return x+z;}');
probeVariants.cmNoPlane=(code,entry)=>{const c=probeVariants.cogmerge!(code,entry);const [a,b]=fnSpan(c,'umTerrainBoxOpen');return c.slice(0,a)+c.slice(a,b).replace('umTerrainPlaneBoxFraction(vec3f(s.y,-1,s.z),at.y-s.x,part)','clamp(at.y-s.x+s.y,0.0,1.0)')+c.slice(b);};
probeVariants.cmNoSurf=(code,entry)=>replaceFn(probeVariants.cogmerge!(code,entry),'umTerrainSurface','fn umTerrainSurface(q:vec3f)->vec3f{return vec3f(textureLoad(umSolidTerrain,vec2i(q.xz),0).x,0.0,0.0);}');
probeVariants.cmrt2=(code,entry)=>{
 let c=probeVariants.cogmerge!(code,entry);const rt=(n:number)=>`min(${n}u,${n}u*arrayLength(&umSupport))`;
 for(const [fn,from,n] of [['umTerrainBoxOpen','for(var k=0u;k<4u;k++){',4],['umTerrainSurface','for(var k=0u;k<4u;k++){',4]] as const){
  const [a,b]=fnSpan(c,fn);const body=c.slice(a,b);assert.ok(body.includes(from),fn);
  c=c.slice(0,a)+body.replace(from,from.replace(`<${n}u;`,`<${rt(n)};`))+c.slice(b);
 }
 return c;
};
// umPressureFaceV with one terrain site and one umCellOpen site (exact).
probeVariants.pfv=(code,entry)=>{
 const c=probeVariants.cogmerge!(code,entry);if(!c.includes('fn umPressureFaceV('))return c;
 const [a,b]=fnSpan(c,'umPressureFaceV');let body=c.slice(a,b);
 const head=` if(umBodyCount()==0u&&umBoundaryTouchesBox(q,vec3f(1))){return min(umBoundaryBoxOpen(q,vec3f(1)),umTerrainBoxOpen(q,vec3f(1)));}
 var neighbor=id;neighbor[axis]+=1;
 if(umSolidFaceClear(id,neighbor,axis)){return 1.0;}
 if((umSolidVoxel(id)&&umBoundaryRecord(id)==0u)||(umSolidVoxel(neighbor)&&umBoundaryRecord(neighbor)==0u)){return 0.5*(umCellOpen(id)+umCellOpen(neighbor));}
 if(umBodyCount()==0u){var q=vec3f(id)+0.5;q[axis]+=0.5;return umTerrainBoxOpen(q,vec3f(1));}
`;
 assert.ok(body.includes(head));
 body=body.replace(head,` let still=umBodyCount()==0u;let touches=still&&umBoundaryTouchesBox(q,vec3f(1));
 if(!touches){
  var neighbor=id;neighbor[axis]+=1;
  if(umSolidFaceClear(id,neighbor,axis)){return 1.0;}
  if((umSolidVoxel(id)&&umBoundaryRecord(id)==0u)||(umSolidVoxel(neighbor)&&umBoundaryRecord(neighbor)==0u)){var open=0.0;for(var k=0u;k<2u;k++){open+=umCellOpen(select(id,neighbor,k==1u));}return 0.5*open;}
 }
 if(still){let terrain=umTerrainBoxOpen(q,vec3f(1));if(touches){return min(umBoundaryBoxOpen(q,vec3f(1)),terrain);}return terrain;}
`);
 return c.slice(0,a)+body+c.slice(b);
};
probeVariants.bdNoTrace=code=>replaceOnce(code,'   if(abs(value)<4.5){\n','   if(abs(value)<4.5&&arrayLength(&umSupport)==0u){\n');
probeVariants.bdNoSurf=code=>replaceOnce(code,'let candidate=nbSurfaceDistance(vec3f(p),nearest);','let candidate=-1.0;');
probeVariants.bdNoOpen=code=>replaceOnce(code,'if(umCellOpen(cell)<=1e-5){break;}','');
probeVariants.bdNoInit=code=>replaceOnce(code,'  let initial=umSampleVertex(vec3f(p));','  let initial=umLoadVertex(p);');
probeVariants.bdcntv=code=>replaceOnce(code,'  let initial=umSampleVertex(vec3f(p));','  atomicAdd(&state[15],1u);let initial=umSampleVertex(vec3f(p));');
probeVariants.bdcnt1=code=>replaceOnce(code,'    var q=vec3f(p);var cell=vec3i(c);\n','    atomicAdd(&state[15],1u);var q=vec3f(p);var cell=vec3i(c);\n');
probeVariants.bdcnt2=code=>replaceOnce(code,'     let sample=nbPhiGradient(vec3u(cell),q);','     atomicAdd(&state[15],1u);let sample=nbPhiGradient(vec3u(cell),q);');
probeVariants.bdcnt3=code=>replaceOnce(code,'cell=nbCellNear(cell,vec3i(floor(q)));','{let next=nbCellNear(cell,vec3i(floor(q)));if(any(next!=cell)){atomicAdd(&state[15],1u);}cell=next;}');
// buildDistance's gradient trace reloads an owner's eight corners only when its owner changes (exact).
probeVariants.bdcache=code=>{
 let c=replaceFn(code,'nbPhiGradient',`fn nbPhiGradient(v:array<f32,8>,width:u32,origin:vec3u,q:vec3f)->vec4f{
 let f=(q-vec3f(origin))/f32(width);var terms:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);
  let w=select(vec3f(1)-f,f,corner!=vec3u(0));terms[k]=v[k]*w.x*w.y*w.z;
 }
 let value=umVertexSum8(terms);
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 return vec4f(nbTrilinear(v,f).xyz/(h*f32(width)),value/h);
}`);
 c=replaceOnce(c,'    var q=vec3f(p);var cell=vec3i(c);\n','    var q=vec3f(p);var cell=vec3i(c);var corners:array<f32,8>;var loadedWidth=0u;var loadedOrigin=vec3u(0);\n');
 c=replaceOnce(c,'     let sample=nbPhiGradient(vec3u(cell),q);','     let at=vec3u(cell);let width=umTileWidth(umTileAt(at/4u));let origin=(at/width)*width;\n     if(width!=loadedWidth||any(origin!=loadedOrigin)){if(width==4u){corners=umLoadCorner8(origin,4u);}else{corners=umLoadVertex8(origin,width);}loadedWidth=width;loadedOrigin=origin;}\n     let sample=nbPhiGradient(corners,width,origin,q);');
 return c;
};
// Momentum's unit trace (uniformCachedMomentumSamplingWGSL): no departure, or no final sample.
probeVariants.mkoDep=code=>replaceOnce(code,'  at=umClampMomentum(position-select(0.5,1.0,stage==1u)*v*dt/h);\n','  at=position;\n');
probeVariants.mkoFinal=code=>replaceOnce(code,'  let last=stage==2u;\n','  let last=stage==2u;if(last){return at.x;}\n');
// No unit halo home: every unit interpolant through its original loads (the halo's A/B).
probeVariants.nohalo=code=>replaceFn(code,'umUnitHaloHome','fn umUnitHaloHome(s:UMSampleTile)->vec4i {return vec4i(-1);}');
