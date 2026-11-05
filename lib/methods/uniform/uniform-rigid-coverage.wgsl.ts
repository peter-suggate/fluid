import {uniformRigidIntervalsWGSL,uniformRigidQuadratureWGSL} from './uniform-rigid-intervals.wgsl';
/** Cell coverage is the union of solid samples, including compound seams. */
export function uniformRigidCoverageSource(bodyMask?:string):string{return /* wgsl */ `
var<private> umRigidCoverageFailure:vec2u;
${uniformRigidIntervalsWGSL}
${uniformRigidQuadratureWGSL}
// Individual coverage is still needed for that body's pressure feedback.
fn umBodyFraction(i:u32,p:vec3i)->f32{
 let h=umSolidParams.cellGravity.xyz;let centre=umSolidWorldCell(p);
 if(umRigidHasPlate(1u<<i)){let axis=umRigidIntegrationAxis(1u<<i,3u);return umRigidIntegratedFraction(1u<<i,centre,h,axis,vec2u((axis+1u)%3u,(axis+2u)%3u),2u);}
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
 let rayCandidates=umBodyCandidates(centre,length(.5*h));
 if(umRigidHasPlate(rayCandidates)){let axis=umRigidIntegrationAxis(rayCandidates,3u);return umRigidIntegratedFraction(rayCandidates,centre,h,axis,vec2u((axis+1u)%3u,(axis+2u)%3u),2u);}
 var mask=0u;
 ${bodyMask?`var bodyCandidates=${bodyMask}&((1u<<umBodyCount())-1u);while(bodyCandidates!=0u){let i=firstTrailingBit(bodyCandidates);bodyCandidates&=bodyCandidates-1u;`:"for(var i=0u;i<umBodyCount();i++){"}
  if(!umBodyNear(i,centre,length(0.4*h))){continue;}
  // Exact clearance certificate: corners lie length(0.4h) from the centre
  // and shape distances are 1-Lipschitz (1% slack for rounding).
  let d=umBodyDistance(i,centre);let reach=1.01*length(0.4*h);
  if(d>reach){continue;}
  if(d< -reach&&umBodyNear(i,centre,-reach)){return 1.0;}
  for(var corner=0u;corner<8u;corner++){
   let bit=1u<<corner;if((mask&bit)!=0u){continue;}
   let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
   if(umInsideBody(i,centre+offset*h)){mask|=bit;}
  }
  if(mask==255u){return 1.0;}
 }
 return f32(countOneBits(mask))/8.0;
}
`;}
export const uniformRigidCoverageWGSL=uniformRigidCoverageSource();
