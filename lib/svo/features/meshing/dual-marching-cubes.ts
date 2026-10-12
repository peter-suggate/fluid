import { boundedQefWGSL } from "./bounded-qef";

/** Schaefer/Warren function-graph fitting, specialized to a uniform sparse grid.
 * Eliminating the scalar coordinate of the 4D QEF leaves a bounded 3D solve.
 * Every resident cell is sampled, including cells with no corner sign change.
 */
function createDmcFitWGSL(cached: boolean): string { return /* wgsl */ `
${boundedQefWGSL}
struct DmcFit { point:vec3f, value:f32, normal:vec3f }
// The field at p in w and its gradient in xyz. One call site for the seven
// evaluations: the field is inlined wherever it is called, and a copy of it a
// probe dominated the pipeline's compile. The point itself goes first, so a
// field that keeps its last stencil answers the six probes about it from that.
fn dmcProbe(p:vec3f,cell:vec3f,dirty:u32,count:u32)->vec4f{
  let e=max(cell*.001,vec3f(1e-6));
  var d:array<f32,7>;
  for(var i=0u;i<7u;i+=1u){
    var q=p;
    if(i>0u){let axis=(i-1u)>>1u;if((i&1u)==1u){q[axis]+=e[axis];}else{q[axis]-=e[axis];}}
    d[i]=dcField(q,dirty,count);
    // The shared field uses a positive sentinel where no source is available.
    if(i==0u&&abs(d[0]/max(cell.x,max(cell.y,cell.z)))>1e12){break;}
  }
  return vec4f(vec3f(d[1]-d[2],d[3]-d[4],d[5]-d[6])/(2.*e),d[0]);
}
fn dmcFit(base:vec3f,cell:vec3f,dirty:u32,count:u32)->DmcFit{
  let unit=max(cell.x,max(cell.y,cell.z));
  var planes:array<vec4f,27>;var weights:array<f32,27>;
  var mean=vec4f(0);var total=0.;
  for(var i=0u;i<27u;i+=1u){
    let p=vec3f(f32(i%3u),f32((i/3u)%3u),f32(i/9u))*.5;
    let sample=${cached ? "dmcSample(base,cell,dirty,count,i)" : "dmcProbe(base+p*cell,cell,dirty,count)"};let f=sample.w/unit;
    if(abs(f)>1e12){continue;}
    let g=sample.xyz*cell/unit;
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
  var point=zeroPoint;
  if(!snap){point=dcSolve(A*(1./total),B/total,vec3f(.5));}
  point=round(point*256.)/256.;
  let value=select((dot(mean.xyz,point)-mean.w)*unit,0.,snap);
  let g=dmcProbe(base+point*cell,cell,dirty,count).xyz;var normal=vec3f(0,1,0);
  if(dot(g,g)>1e-20&&all(abs(g)<vec3f(1e12))){normal=normalize(g);}
  return DmcFit(point,value,normal);
}
`; }

export const svoDualMarchingCubesFitWGSL = createDmcFitWGSL(false);

/** A 4³-cell workgroup shares the 9³ half-cell sample lattice. 11,664 bytes
 * of workgroup storage replace 1,728 independent sample/gradient evaluations
 * with 729. Final fitted-point gradients still evaluate the original field.
 */
export const svoDualMarchingCubesCachedFitWGSL = /* wgsl */ `
var<workgroup> dmcSamples:array<vec4f,729>;
var<private> dmcLocal:vec3u;
var<private> dmcCached:bool;
${createDmcFitWGSL(true)}
fn dmcSample(base:vec3f,cell:vec3f,dirty:u32,count:u32,i:u32)->vec4f{
  if(dmcCached){let q=dmcLocal*2u+vec3u(i%3u,(i/3u)%3u,i/9u);return dmcSamples[q.x+9u*q.y+81u*q.z];}
  return dmcProbe(base+vec3f(f32(i%3u),f32((i/3u)%3u),f32(i/9u))*.5*cell,cell,dirty,count);
}
// All 64 lanes must call this, including inactive tail workgroups. The barrier
// is unconditional; callers may return for out-of-range cells afterwards.
fn dmcCacheSamples(base:vec3f,cell:vec3f,dirty:u32,count:u32,lane:u32,enabled:bool){
  dmcCached=enabled;dmcLocal=vec3u(lane%4u,(lane/4u)%4u,lane/16u);
  if(enabled){for(var i=lane;i<729u;i+=64u){
    dmcSamples[i]=dmcProbe(base+vec3f(f32(i%9u),f32((i/9u)%9u),f32(i/81u))*.5*cell,cell,dirty,count);
  }}
  workgroupBarrier();
}
`;
