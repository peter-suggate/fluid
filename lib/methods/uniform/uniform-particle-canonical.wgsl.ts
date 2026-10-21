/** Canonical cell order without changing the particle multiset. A lane owns
 * one cell's disjoint run; heapsort bounds crowded-cell work to O(n log n)
 * and uses constant scratch space. Compare every record bit, including age
 * and affine state, so coincident particles cannot inherit atomic slot order. */
export function particleCanonicalWGSL(apic:boolean):string{
 const fields=['position','velocity','before',...(apic?['cx','cy','cz','lifetime']:[])];
 return /* wgsl */`
fn nbParticleLess(a:Particle,b:Particle)->bool{
 ${fields.map(field=>`{let x=bitcast<vec4u>(a.${field});let y=bitcast<vec4u>(b.${field});for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return x[k]<y[k];}}}`).join('\n ')}
 return false;
}
fn nbParticleSift(start:u32,count:u32,root:u32){
 var parent=root;let item=particles[start+parent];
 loop{
  var child=2u*parent+1u;if(child>=count){break;}
  if(child+1u<count&&nbParticleLess(particles[start+child],particles[start+child+1u])){child++;}
  if(!nbParticleLess(item,particles[start+child])){break;}
  particles[start+parent]=particles[start+child];parent=child;
 }
 particles[start+parent]=item;
}
fn nbCanonicalCell(start:u32,count:u32){
 if(count<2u){return;}
 for(var root=count/2u;root>0u;root--){nbParticleSift(start,count,root-1u);}
 for(var end=count-1u;end>0u;end--){
  let item=particles[start];particles[start]=particles[start+end];particles[start+end]=item;
  nbParticleSift(start,end,0u);
 }
}
`;
}
