// Frozen pre-optimization GPU fitter (670556ac), used as a numerical and timing oracle.
/** Shared GPU box-constrained quadratic minimizer. */
const boundedQefWGSL = /* wgsl */ `
// Box-constrained regularized QEF: enumerate interior, faces, edges and corners.
// Solve in local coordinates around the Hermite centroid, avoiding large-world
// cancellation and the unstable unconstrained inverse of a planar system.
fn dcSolve(a:mat3x3f,b:vec3f,mass:vec3f)->vec3f{
  let lambda=1e-5;let A=a+mat3x3f(vec3f(lambda,0,0),vec3f(0,lambda,0),vec3f(0,0,lambda));
  let B=b+lambda*mass;var best=mass;var error=dot(best,A*best)-2.*dot(B,best);
  for(var code=0u;code<27u;code+=1u){
    let state=vec3u(code%3u,(code/3u)%3u,code/9u);
    let free=vec3f(select(0.,1.,state.x==0u),select(0.,1.,state.y==0u),select(0.,1.,state.z==0u));
    let fixed=vec3f(select(0.,1.,state.x==2u),select(0.,1.,state.y==2u),select(0.,1.,state.z==2u));
    var M=A;for(var j=0u;j<3u;j+=1u){M[j]=A[j]*free*free[j];M[j][j]+=1.-free[j];}
    let rhs=(B-A*fixed)*free+fixed;
    let r0=cross(M[1],M[2]);let r1=cross(M[2],M[0]);let r2=cross(M[0],M[1]);
    let det=dot(M[0],r0);if(abs(det)<1e-20){continue;}
    let v=vec3f(dot(r0,rhs),dot(r1,rhs),dot(r2,rhs))/det;
    if(any(v<vec3f(-1e-5))||any(v>vec3f(1.00001))){continue;}
    let p=clamp(v,vec3f(0),vec3f(1));let e=dot(p,A*p)-2.*dot(B,p);
    if(e<error){error=e;best=p;}
  }
  return best;
}
`;


/** Schaefer/Warren function-graph fitting, specialized to a uniform sparse grid.
 * Eliminating the scalar coordinate of the 4D QEF leaves a bounded 3D solve.
 * Every resident cell is sampled, including cells with no corner sign change.
 */
export const referenceDmcFitWGSL = /* wgsl */ `
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
