/** Cell coverage is the union of solid samples, including compound seams. */
export const uniformRigidCoverageWGSL = /* wgsl */ `
// Individual coverage is still needed for that body's pressure feedback.
fn umBodyFraction(i:u32,p:vec3i)->f32{
 let h=umSolidParams.cellGravity.xyz;let centre=umSolidWorldCell(p);
 if(!umBodyNear(i,centre,length(0.4*h))){return 0.0;}
 var inside=0.0;
 for(var corner=0u;corner<8u;corner++){
  let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
  if(umInsideBody(i,centre+offset*h)){inside+=1.0;}
 }
 return inside/8.0;
}
fn umCellBodyFraction(p:vec3i)->f32{
 let h=umSolidParams.cellGravity.xyz;let centre=umSolidWorldCell(p);
 var mask=0u;
 for(var i=0u;i<umBodyCount();i++){
  if(!umBodyNear(i,centre,length(0.4*h))){continue;}
  for(var corner=0u;corner<8u;corner++){
   let bit=1u<<corner;if((mask&bit)!=0u){continue;}
   let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
   if(umInsideBody(i,centre+offset*h)){mask|=bit;}
  }
  if(mask==255u){return 1.0;}
 }
 return f32(countOneBits(mask))/8.0;
}
`;
