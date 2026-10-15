/** Experimental frozen-phi page halos, not wired into production. Fine
 * blocks hold logical vertices -1..33 (35³); coarse blocks -4..36 (11³).
 * Their values use each HOME page's wall mapping and the actual canonical
 * loaders, never the potentially stale duplicate skirts of the source.
 *
 * A production refresh belongs immediately before surface.encode, after
 * relayout phi resolve, detail retirement/settle and any authority writes.
 * Regular/deferred/wall surface entries read that same frozen phi epoch.
 * Rebuild after any phi write, ownership change, page reuse or frame shift.
 * Departure cell.origin determines home; an AIR home uses the original
 * sampler. Width-one base is 0..31, width-four base is 0..28, hence the
 * exact cubic halos above. They do not cover smoothing's wider +/-2w taps.
 *
 * This prototype duplicates resident values and includes all refresh reads
 * in its measurements; it is not a claim that a free halo already exists.
 * A 2-D r32float texture avoids adding to the surface's full buffer budget.
 */
export const SURFACE_HALO_FINE = 35 ** 3;
export const SURFACE_HALO_COARSE = 11 ** 3;
export const SURFACE_HALO_STRIDE = SURFACE_HALO_FINE + SURFACE_HALO_COARSE;
export function surfaceHaloWGSL(source:string,auditTaps=false):string {
 const match=source.match(/fn umCubicPhi\(q:vec3f,cell:UMVertexCell\)->f32\{[\s\S]*?\n\}/);
 if(!match)throw new Error('Missing surface cubic sampler');
 const original=match[0];
 let candidate=original.replace('let home=umBlockOf(page);','let home=umBlockOf(page);if(!umBlockStored(home)){return qaUncachedCubicPhi(q,cell);}');
 candidate=candidate.replaceAll('let vertex=umVertexNearFromPage(home,page,base+offset*width);','let vertex=base+offset*width;');
 candidate=candidate.replaceAll('s=umLoadCorner(vec3u(vertex));','s=qaHaloCorner(home,page,vertex);').replaceAll('s=umLoadVertex(vec3u(vertex));','s=qaHaloVertex(home,page,vertex);');
 candidate=candidate.replaceAll('umBuried(vec3f(vertex))','umBuried(vec3f(umVertexNearFromPage(home,page,vertex)))');
 // Diagnostic only: every consumed outer halo tap is compared bitwise with
 // its original home-relative canonical load in the same invocation. A
 // mismatch marks the final deferred output AFTER all wall/source rules;
 // never use this variant for timings.
 if(auditTaps){
  for(const [halo,load] of [['qaHaloCorner','umLoadCorner'],['qaHaloVertex','umLoadVertex']]){
   const tap=`s=${halo}(home,page,vertex);`;
   candidate=candidate.replaceAll(tap,`${tap}qaBadTap=qaBadTap||(bitcast<u32>(s)!=bitcast<u32>(${load}(vec3u(umVertexNearFromPage(home,page,vertex)))));`);
  }

 }

 if(candidate===original||candidate.includes('s=umLoadCorner(vec3u(vertex));')||candidate.includes('s=umLoadVertex(vec3u(vertex));'))throw new Error('Cubic halo rewrite incomplete');
 source=source.replace(original,candidate);
 if(auditTaps){
  const value='vec4f(umAdvected(vec3f(vertex),width)+umVertexSmoothing(vertex,width))';
  if(!source.includes(value))throw new Error('Missing deferred audit store');
  source=source.replaceAll(value,'qaAuditedAdvected(vertex,width)');
  source+='\nvar<private> qaBadTap:bool=false;\nfn qaAuditedAdvected(vertex:vec3u,width:u32)->vec4f{qaBadTap=false;let value=umAdvected(vec3f(vertex),width)+umVertexSmoothing(vertex,width);return vec4f(select(value,-1e20,qaBadTap));}\n';
 }
 const blocks=/fn\s+UM_BLOCKS\s*\(/.test(source)?'UM_BLOCKS()':'UM_BLOCKS';
 const fineCount=/struct UMSceneCounts/.test(source)?'umCounts.work.x':'umCounts.x';
 return source+`\n${original.replace('fn umCubicPhi(', 'fn qaUncachedCubicPhi(')}
@group(1) @binding(14) var qaSurfaceHalo:texture_2d<f32>;
@group(1) @binding(15) var qaSurfaceHaloOut:texture_storage_2d<r32float,write>;
fn qaHaloLoad(i:u32)->f32{return textureLoad(qaSurfaceHalo,vec2i(i32(i%4096u),i32(i/4096u)),0).x;}
fn qaHaloStore(i:u32,value:f32){textureStore(qaSurfaceHaloOut,vec2i(i32(i%4096u),i32(i/4096u)),vec4f(value));}
fn qaHaloVertex(home:u32,page:vec3i,p:vec3i)->f32{
 let l=vec3u(p-(page<<vec3u(5u))+vec3i(1));
 return qaHaloLoad(${SURFACE_HALO_STRIDE}u*home+l.x+35u*(l.y+35u*l.z));
}
fn qaHaloCorner(home:u32,page:vec3i,p:vec3i)->f32{
 let l=vec3u((p-(page<<vec3u(5u)))/4+vec3i(1));
 return qaHaloLoad(${SURFACE_HALO_STRIDE}u*home+${SURFACE_HALO_FINE}u+l.x+11u*(l.y+11u*l.z));
}
@compute @workgroup_size(64) fn qaFillSurfaceHalo(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let i=gid.x+64u*groups.x*gid.y;let home=i/${SURFACE_HALO_STRIDE}u;
 if(home>=${blocks}||umBlockWord(home)==0xffffffffu){return;}
 let page=vec3i(umBlockPage(home));let at=i%${SURFACE_HALO_STRIDE}u;
 if(at<${SURFACE_HALO_FINE}u){
  if(${fineCount}==0u){return;}
  let l=vec3i(i32(at%35u),i32((at/35u)%35u),i32(at/1225u))-vec3i(1);
  let p=umVertexNearFromPage(home,page,(page<<vec3u(5u))+l);
  qaHaloStore(i,umLoadVertex(vec3u(p)));
 }else{
  let a=at-${SURFACE_HALO_FINE}u;let l=4*(vec3i(i32(a%11u),i32((a/11u)%11u),i32(a/121u))-vec3i(1));
  let p=umVertexNearFromPage(home,page,(page<<vec3u(5u))+l);
  qaHaloStore(i,umLoadCorner(vec3u(p)));
 }
}
`;
}
