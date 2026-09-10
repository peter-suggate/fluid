import { boundedQefWGSL } from "./bounded-qef";

/** Schaefer/Warren function-graph fitting, specialized to a uniform sparse grid.
 * Eliminating the scalar coordinate of the 4D QEF leaves a bounded 3D solve.
 * Every resident cell is sampled, including cells with no corner sign change.
 */
export const svoDualMarchingCubesFitWGSL = /* wgsl */ `
${boundedQefWGSL}
struct DmcFit { point:vec3f, value:f32, normal:vec3f }
fn dmcGradient(p:vec3f,cell:vec3f,dirty:u32,count:u32)->vec3f{
  let e=max(cell*.001,vec3f(1e-6));
  return vec3f(dcField(p+vec3f(e.x,0,0),dirty,count)-dcField(p-vec3f(e.x,0,0),dirty,count),
    dcField(p+vec3f(0,e.y,0),dirty,count)-dcField(p-vec3f(0,e.y,0),dirty,count),
    dcField(p+vec3f(0,0,e.z),dirty,count)-dcField(p-vec3f(0,0,e.z),dirty,count))/(2.*e);
}
fn dmcFit(base:vec3f,cell:vec3f,dirty:u32,count:u32)->DmcFit{
  let unit=max(cell.x,max(cell.y,cell.z));
  var planes:array<vec4f,27>;var weights:array<f32,27>;
  var mean=vec4f(0);var total=0.;
  for(var i=0u;i<27u;i+=1u){
    let p=vec3f(f32(i%3u),f32((i/3u)%3u),f32(i/9u))*.5;
    let f=dcField(base+p*cell,dirty,count)/unit;
    // The shared field uses a positive sentinel where no source is available.
    if(abs(f)>1e12){continue;}
    let g=dmcGradient(base+p*cell,cell,dirty,count)*cell/unit;
    if(any(abs(g)>vec3f(1e12))){continue;}
    let plane=vec4f(g,dot(g,p)-f);let weight=1./(1.+dot(g,g));
    planes[i]=plane;weights[i]=weight;mean+=weight*plane;total+=weight;
  }
  if(total==0.){return DmcFit(vec3f(.5),1e20,vec3f(0,1,0));}
  mean/=total;
  // Centred covariance avoids subtracting nearly equal matrices on planes.
  var A=mat3x3f(vec3f(0),vec3f(0),vec3f(0));var B=vec3f(0);
  for(var i=0u;i<27u;i+=1u){let d=planes[i]-mean;let w=weights[i];
    A+=w*mat3x3f(d.xyz*d.x,d.xyz*d.y,d.xyz*d.z);B+=w*d.xyz*d.w;
  }
  var point=dcSolve(A*(1./total),B/total,vec3f(.5));
  // Paper's sliver elimination: prefer the restricted graph QEF at value=0
  // when its residual is negligible. This also removes numerical sign noise
  // at exactly planar faces and sharp edges, before triangulation/quantization.
  var zeroA=mat3x3f(vec3f(0),vec3f(0),vec3f(0));var zeroB=vec3f(0);
  for(var i=0u;i<27u;i+=1u){let p=planes[i];let w=weights[i];
    zeroA+=w*mat3x3f(p.xyz*p.x,p.xyz*p.y,p.xyz*p.z);zeroB+=w*p.xyz*p.w;
  }
  let zeroPoint=dcSolve(zeroA*(1./total),zeroB/total,vec3f(.5));var residual=0.;
  for(var i=0u;i<27u;i+=1u){let d=dot(planes[i].xyz,zeroPoint)-planes[i].w;residual+=weights[i]*d*d;}
  let snap=residual/total<1e-6;
  if(snap){point=zeroPoint;}
  point=round(point*256.)/256.;
  let value=select((dot(mean.xyz,point)-mean.w)*unit,0.,snap);
  let g=dmcGradient(base+point*cell,cell,dirty,count);var normal=vec3f(0,1,0);
  if(dot(g,g)>1e-20&&all(abs(g)<vec3f(1e12))){normal=normalize(g);}
  return DmcFit(point,value,normal);
}
`;
