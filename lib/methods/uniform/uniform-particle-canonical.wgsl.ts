/** Canonical segmented parallel merge sort, without changing any record.
 * Two of each slot's four compact-position words ping-pong source indices;
 * source records stay immutable until the sorted records have been copied.
 * In each round a lane binary-searches its partner run and writes its unique
 * merged rank. O(n log² n) comparisons, O(log n) barrier rounds, no extra
 * allocation and no population cap. The caller supplies storage/workgroup
 * barriers between local rounds, or dispatch boundaries for dense cells.
 * Full-record uint order matches the old canonical rank. Source index breaks
 * only bit-identical ties (interchangeable payloads), making destinations unique. */
export function particleCanonicalWGSL(apic:boolean):string{
 const fields=['position','velocity','before',...(apic?['cx','cy','cz','lifetime']:[])];
 return /* wgsl */`
fn nbCanonicalIndexLess(a:u32,b:u32)->bool{
 ${fields.map(field=>`{let x=bitcast<vec4u>(source[a].${field});let y=bitcast<vec4u>(source[b].${field});for(var k=0u;k<4u;k++){if(x[k]!=y[k]){return x[k]<y[k];}}}`).join('\n ')}
 return a<b;
}
fn nbCanonicalMerge(at:u32,start:u32,count:u32,width:u32,bank:u32){
 let index=links[NB_LINKS+4u*at+bank];let end=start+count;
 let run=(at-start)/width;let other=min(end,start+(run^1u)*width);
 var low=other;var high=min(end,other+width);
 while(low<high){
  let mid=low+(high-low)/2u;
  if(nbCanonicalIndexLess(links[NB_LINKS+4u*mid+bank],index)){low=mid+1u;}else{high=mid;}
 }
 let destination=start+(run&~1u)*width+(at-start)%width+low-other;
 links[NB_LINKS+4u*destination+(bank^1u)]=index;
}
`;
}
