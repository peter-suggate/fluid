/** Canonical cell order without changing the particle multiset. A lane owns
 * one record and counts the records of its cell's run that order before it;
 * that rank is its place in the run. The scatter leaves each slot's source
 * index and position bits in its compact-position words (UM_CELL_KEYS), so
 * the comparisons read those 16 bytes and load whole records only when two
 * positions tie. Compare every record bit, including age and affine state,
 * so coincident particles cannot inherit atomic slot order; records equal in
 * every bit are interchangeable and take their slot order. */
export function particleCanonicalWGSL(apic:boolean):string{
 const fields=['position','velocity','before',...(apic?['cx','cy','cz','lifetime']:[])];
 return /* wgsl */`
fn nbParticleLess(a:Particle,b:Particle)->bool{
 ${fields.map(field=>`{let x=bitcast<vec4u>(a.${field});let y=bitcast<vec4u>(b.${field});for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return x[k]<y[k];}}}`).join('\n ')}
 return false;
}
// The rank of slot at (source record index, position bits key) in the run
// of count slots from start.
fn nbParticleRank(at:u32,start:u32,count:u32,index:u32,key:vec3u)->u32{
 var rank=0u;
 for(var o=start;o<start+count;o++){
  if(o==at){continue;}
  let c=NB_LINKS+4u*o;let other=vec3u(links[c+1u],links[c+2u],links[c+3u]);
  var less=false;
  if(other.x!=key.x){less=other.x<key.x;}
  else if(other.y!=key.y){less=other.y<key.y;}
  else if(other.z!=key.z){less=other.z<key.z;}
  else{let a=source[links[c]];let b=source[index];less=nbParticleLess(a,b)||(o<at&&!nbParticleLess(b,a));}
  rank+=u32(less);
 }
 return rank;
}
`;
}
