/** Tool-only experiment: carry one sample's known atlas page across all three
 * MAC components. The original sampler remains the exact boundary fallback. */
import assert from 'node:assert/strict';
export function fineSampleContext(code:string):string{
 const replace=(a:string,b:string)=>{assert.equal(code.split(a).length,2,`one occurrence: ${a}`);code=code.replace(a,b);};
 const extent=code.includes('fn UM_CELL_TEXTURE(')?'UM_CELL_TEXTURE()':'UM_CELL_TEXTURE';
 const helper=`
// lo..hi encloses all three unheld MAC tap footprints. Staying strictly in
// the known resident page proves no axis clamp changes them and no negative
// wall face is read. A page seam, wall, AIR block, or atlas stand-in falls
// back to the original sampler. There is no new page-directory lookup.
fn qaFineAtlasHome(p:vec3f,s:UMSampleTile)->vec4i{
 if(umRegularFine||textureDimensions(unitVelocity).x<${extent}.x){return vec4i(0,0,0,-1);}
 let block=s.key>>9u;if(!umBlockStored(block)){return vec4i(0,0,0,-1);}
 let page=s.tile>>vec3u(3u);let lo=vec3i(floor(p-1.0));let hi=vec3i(floor(p-0.5))+1;
 if(any((lo>>vec3u(5u))!=page)||any((hi>>vec3u(5u))!=page)){return vec4i(0,0,0,-1);}
 return vec4i((umBlockCoord(block)<<vec3u(5u))-(page<<vec3u(5u)),0);
}
`;
 replace('fn umSampleVelocity1(p:vec3f,axis:u32)->f32 {','fn umSampleVelocity1(p:vec3f,axis:u32,atlas:vec4i)->f32 {');
 const start=code.indexOf('fn umSampleVelocity1('),end=code.indexOf('\n}',start)+2;let scalar=code.slice(start,end);
 const marker=' let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;';assert.ok(scalar.includes(marker));
 scalar=scalar.replace(marker,marker+`
 if(atlas.w>=0){
  for(var k=0u;k<8u;k++){
   let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
   terms[k]=textureLoad(unitVelocity,base+bit+atlas.xyz,0)[axis];
  }
  return umVelocityLerp8(terms,fraction);
 }`);
 code=code.slice(0,start)+scalar+code.slice(end);
 replace('fn umSampleVelocityWeightedIn(p:vec3f,home:UMCoarseHome,axis:u32,fine:f32)->f32 {','fn umSampleVelocityWeightedIn(p:vec3f,home:UMCoarseHome,axis:u32,fine:f32,atlas:vec4i)->f32 {');
 replace('umSampleVelocity1(p,axis)','umSampleVelocity1(p,axis,atlas)');
 replace('fn umSampleVelocityAxis(p:vec3f,s:UMSampleTile,fineTile:bool,weights:f32,axis:u32)->f32 {','fn umSampleVelocityAxis(p:vec3f,s:UMSampleTile,fineTile:bool,weights:f32,axis:u32,atlas:vec4i)->f32 {');
 replace('umSampleVelocityWeightedIn(p,umCoarseHome(p,s),axis,weights)','umSampleVelocityWeightedIn(p,umCoarseHome(p,s),axis,weights,atlas)');
 replace('return umSampleVelocityAxis(p,s,fineTile,weights,axis);','return umSampleVelocityAxis(p,s,fineTile,weights,axis,vec4i(0,0,0,-1));');
 replace(' var v=vec3f(0.0);\n for(var a=0u;a<3u;a++){v[a]=umSampleVelocityAxis(p,s,fineTile,weights,a);}',
 ` var atlas=vec4i(0,0,0,-1);if(weights>0.0){atlas=qaFineAtlasHome(p,s);}
 var v=vec3f(0.0);
 for(var a=0u;a<3u;a++){v[a]=umSampleVelocityAxis(p,s,fineTile,weights,a,atlas);}`);
 return code+helper;
}
