/** Shared GPU box-constrained quadratic minimizer. */
export const boundedQefWGSL = /* wgsl */ `
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
