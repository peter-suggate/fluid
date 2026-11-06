/** Analytic ray geometry for native rigid cut-cell integration. Depends only
 * on the authoritative body records, body-local cut texture, and quaternion
 * helper. Ray parameters retain the caller's units; directions need not be
 * normalized. No march or thickness-dependent sample count is involved. */
export const uniformRigidIntervalsWGSL=/* wgsl */`
const UM_RAY_INF:f32=3.402823e38;
fn umRayEmpty()->vec2f{return vec2f(UM_RAY_INF,-UM_RAY_INF);}
fn umRayIntersect(a:vec2f,b:vec2f)->vec2f{return vec2f(max(a.x,b.x),min(a.y,b.y));}
fn umRayHull(a:vec2f,b:vec2f)->vec2f{if(b.y<=b.x){return a;}if(a.y<=a.x){return b;}return vec2f(min(a.x,b.x),max(a.y,b.y));}
fn umRaySlab(o:f32,d:f32,lo:f32,hi:f32)->vec2f{
 if(d==0.0){if(o<lo||o>hi){return umRayEmpty();}return vec2f(-UM_RAY_INF,UM_RAY_INF);}
 let a=(lo-o)/d;let b=(hi-o)/d;return vec2f(min(a,b),max(a,b));
}
fn umRayBox(o:vec3f,d:vec3f,h:vec3f)->vec2f{
 return umRayIntersect(umRayIntersect(umRaySlab(o.x,d.x,-h.x,h.x),umRaySlab(o.y,d.y,-h.y,h.y)),umRaySlab(o.z,d.z,-h.z,h.z));
}
fn umRayBall(o:vec3f,d:vec3f,r:f32)->vec2f{
 let a=dot(d,d);if(a==0.0){if(dot(o,o)<=r*r){return vec2f(-UM_RAY_INF,UM_RAY_INF);}return umRayEmpty();}
 let m=-dot(o,d)/a;let nearest=o+m*d;let q=(r*r-dot(nearest,nearest))/a;
 if(q<=0.0){return umRayEmpty();}let delta=sqrt(q);return vec2f(m-delta,m+delta);
}
fn umRayCylinder(o:vec3f,d:vec3f,r:f32,halfLength:f32,axis:u32)->vec2f{
 var radialO=o;var radialD=d;radialO[axis]=0.0;radialD[axis]=0.0;
 return umRayIntersect(umRayBall(radialO,radialD,r),umRaySlab(o[axis],d[axis],-halfLength,halfLength));
}
// Every outer shape here is convex. The rounded plate is the exact union of
// a central slab, four edge cylinders and four corner balls. Their union is
// convex, so the interval hull is exact, not a bounding geometry substitute.
fn umRayShapeOuter(shape:u32,size:vec3f,o:vec3f,d:vec3f)->vec2f{
 if(shape==0u){return umRayBall(o,d,size.x);}
 if(shape==1u){return umRayBox(o,d,0.5*size);}
 if(shape==3u||shape==4u){return umRayCylinder(o,d,size.x,0.5*size.y,1u);}
 if(shape==2u){
  var result=umRayCylinder(o,d,size.x,0.5*size.y,1u);
  result=umRayHull(result,umRayBall(o-vec3f(0.0,0.5*size.y,0.0),d,size.x));
  return umRayHull(result,umRayBall(o+vec3f(0.0,0.5*size.y,0.0),d,size.x));
 }
 if(shape==5u){
  var result=umRayBox(o,d,size);
  for(var s=0u;s<2u;s++){
   let sign=select(-1.0,1.0,s==1u);
   result=umRayHull(result,umRayCylinder(o-vec3f(0.0,0.0,sign*size.z),d,size.y,size.x,0u));
   result=umRayHull(result,umRayCylinder(o-vec3f(sign*size.x,0.0,0.0),d,size.y,size.z,2u));
   for(var z=0u;z<2u;z++){result=umRayHull(result,umRayBall(o-vec3f(sign*size.x,0.0,select(-size.z,size.z,z==1u)),d,size.y));}
  }
  return result;
 }
 return umRayEmpty();
}
fn umRayCupCavity(size:vec3f,o:vec3f,d:vec3f)->vec2f{
 let wall=clamp(size.z,1e-4,min(0.95*size.x,0.95*size.y));
 return umRayIntersect(umRayBall(vec3f(o.x,0.0,o.z),vec3f(d.x,0.0,d.z),size.x-wall),umRaySlab(o.y,d.y,-0.5*size.y+wall,UM_RAY_INF));
}
fn umRayCut(header:vec4f,k:u32,o:vec3f,d:vec3f)->vec2f{
 let at=i32(header.x)+3*i32(k);let center=textureLoad(umSolidBodyCuts,vec2i(at,0),0).xyz;
 let extent=textureLoad(umSolidBodyCuts,vec2i(at+1,0),0).xyz;let q=textureLoad(umSolidBodyCuts,vec2i(at+2,0),0);let inverse=vec4f(q.w,-q.xyz);
 return umRayBox(umQuatRotate(inverse,o-center),umQuatRotate(inverse,d),extent);
}
fn umRayNextRoot(interval:vec2f,after:f32,limit:f32)->f32{
 if(interval.y<=interval.x){return limit;}var result=limit;
 if(interval.x>after){result=min(result,interval.x);}if(interval.y>after){result=min(result,interval.y);}return result;
}
// Next boundary across all bodies, cavities and local cuts. This event sweep
// has no fixed interval-array limit: a scene with many cuts remains exact.
fn umRayNextBoundary(candidates:u32,origin:vec3f,direction:vec3f,after:f32,limit:f32)->f32{
 var next=limit;var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;let body=umSolidBodies[i];let inverse=vec4f(body.orientation.x,-body.orientation.yzw);
  let o=umQuatRotate(inverse,origin-body.positionShape.xyz);let d=umQuatRotate(inverse,direction);
  let code=i32(round(body.positionShape.w));let shape=u32(select(code,-code-1,code<0));
  next=umRayNextRoot(umRayShapeOuter(shape,body.dimensions.xyz,o,d),after,next);
  if(shape==4u){next=umRayNextRoot(umRayCupCavity(body.dimensions.xyz,o,d),after,next);}
  let header=textureLoad(umSolidBodyCuts,vec2i(i32(i),0),0);
  for(var k=0u;k<u32(header.y);k++){next=umRayNextRoot(umRayCut(header,k,o,d),after,next);}
 }
 return next;
}
// Membership between consecutive event roots uses the SAME analytic
// intervals, avoiding inconsistent SDF tolerances at very thin surfaces.
fn umRayBodyContains(i:u32,origin:vec3f,direction:vec3f,t:f32)->bool{
 let body=umSolidBodies[i];let inverse=vec4f(body.orientation.x,-body.orientation.yzw);
 let o=umQuatRotate(inverse,origin-body.positionShape.xyz);let d=umQuatRotate(inverse,direction);
 let code=i32(round(body.positionShape.w));let shape=u32(select(code,-code-1,code<0));let outer=umRayShapeOuter(shape,body.dimensions.xyz,o,d);
 var inside=t>=outer.x&&t<=outer.y;
 if(shape==4u){let cavity=umRayCupCavity(body.dimensions.xyz,o,d);inside=inside&&!(t>=cavity.x&&t<=cavity.y);}
 if(code<0){inside=!inside;}if(!inside){return false;}
 let header=textureLoad(umSolidBodyCuts,vec2i(i32(i),0),0);
 for(var k=0u;k<u32(header.y);k++){let cut=umRayCut(header,k,o,d);if(t>=cut.x&&t<=cut.y){return false;}}
 return true;
}
fn umRayUnionLength(candidates:u32,origin:vec3f,direction:vec3f,domain:vec2f)->f32{
 var at=domain.x;var result=0.0;
 loop{
  if(at>=domain.y){break;}let next=umRayNextBoundary(candidates,origin,direction,at,domain.y);
  let midpoint=at+0.5*(next-at);var remaining=candidates;var occupied=false;
  while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;if(umRayBodyContains(i,origin,direction,midpoint)){occupied=true;break;}}
  if(occupied){result+=next-at;}at=next;
 }
 return result;
}
`;

/** Embedded Gauss integration over the transverse directions. Exact ray
 * intervals handle thickness; geometric feature breakpoints prevent a tiny
 * axis-aligned body or cut from disappearing between quadrature points.
 * The returned error is an estimator, not a theorem about arbitrary CSG.
 * Unresolved work/depth/error raises the sticky native geometry-health flag. */
export const uniformRigidQuadratureWGSL=/* wgsl */`
override umRigidQuadratureWork:u32=2048u;
override umRigidQuadratureTolerance:f32=2e-4;
struct UMRayBounds {lo:vec3f,hi:vec3f}
fn umRigidFeatureBounds(i:u32,cut:i32)->UMRayBounds{
 let body=umSolidBodies[i];var center=body.positionShape.xyz;var extent=vec3f(0.0);
 if(cut>=0){
  let header=textureLoad(umSolidBodyCuts,vec2i(i32(i),0),0);let at=i32(header.x)+3*cut;
  let local=textureLoad(umSolidBodyCuts,vec2i(at,0),0).xyz;let h=textureLoad(umSolidBodyCuts,vec2i(at+1,0),0).xyz;let packed=textureLoad(umSolidBodyCuts,vec2i(at+2,0),0);let q=vec4f(packed.w,packed.xyz);
  center+=umQuatRotate(body.orientation,local);
  for(var k=0u;k<3u;k++){var basis=vec3f(0.0);basis[k]=1.0;extent+=h[k]*abs(umQuatRotate(body.orientation,umQuatRotate(q,basis)));}
 }else{
  let code=i32(round(body.positionShape.w));let shape=u32(select(code,-code-1,code<0));let d=body.dimensions.xyz;
  for(var k=0u;k<3u;k++){var basis=vec3f(0.0);basis[k]=1.0;let n=abs(umQuatRotate(vec4f(body.orientation.x,-body.orientation.yzw),basis));
   if(shape==0u){extent[k]=d.x;}
   else if(shape==1u){extent[k]=dot(.5*d,n);}
   else if(shape==2u){extent[k]=d.x+.5*d.y*n.y;}
   else if(shape==3u||shape==4u){extent[k]=d.x*length(n.xz)+.5*d.y*n.y;}
   else{extent[k]=d.x*n.x+d.z*n.z+d.y;}
  }
 }
 return UMRayBounds(center-extent,center+extent);
}
fn umRigidHasPlate(candidates:u32)->bool{
 var remaining=candidates;while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;let code=i32(round(umSolidBodies[i].positionShape.w));if(code==5||code== -6){return true;}}return false;
}
fn umRigidIntegrationAxis(candidates:u32,excluded:u32)->u32{
 var axis=(excluded+1u)%3u;var best=0.0;var remaining=candidates;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;let body=umSolidBodies[i];let code=i32(round(body.positionShape.w));if(code!=5&&code!= -6){continue;}let normal=abs(umQuatRotate(body.orientation,vec3f(0.0,1.0,0.0)));
  for(var k=0u;k<3u;k++){if(k!=excluded&&normal[k]>best){best=normal[k];axis=k;}}
 }return axis;
}
fn umRigidRayAt(candidates:u32,center:vec3f,size:vec3f,rayAxis:u32,axes:vec2u,uv:vec2f,dimensions:u32)->f32{
 var origin=center;origin[axes.x]+=size[axes.x]*uv.x;if(dimensions==2u){origin[axes.y]+=size[axes.y]*uv.y;}var direction=vec3f(0.0);direction[rayAxis]=1.0;
 return umRayUnionLength(candidates,origin,direction,vec2f(-.5,.5)*size[rayAxis])/size[rayAxis];
}
// Split at projected geometry bounds before applying a smooth quadrature.
// Body-local edits are first transformed by the current accepted body pose.
fn umRigidFeatureSplit(candidates:u32,center:vec3f,size:vec3f,rayAxis:u32,axes:vec2u,span:vec4f,dimensions:u32)->vec2f{
 var remaining=candidates;var best=vec2f(-1.0,0.0);var balance=0.0;
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;let count=i32(textureLoad(umSolidBodyCuts,vec2i(i32(i),0),0).y);
  for(var feature= -1;feature<count;feature++){
   let b=umRigidFeatureBounds(i,feature);if(b.lo[rayAxis]>=center[rayAxis]+.5*size[rayAxis]||b.hi[rayAxis]<=center[rayAxis]-.5*size[rayAxis]){continue;}
   for(var k=0u;k<dimensions;k++){let axis=axes[k];let lo=span[k];let hi=span[k+2u];
    for(var side=0u;side<2u;side++){let split=(select(b.lo[axis],b.hi[axis],side==1u)-center[axis])/size[axis];let weight=min(split-lo,hi-split);
     if(weight>max(1e-6*(hi-lo),1e-7)&&weight>balance){balance=weight;best=vec2f(f32(k),split);}
    }
   }
  }
 }
 return best;
}
fn umRigidGauss(candidates:u32,center:vec3f,size:vec3f,rayAxis:u32,axes:vec2u,span:vec4f,dimensions:u32,order:u32)->f32{
 let middle=.5*(span.xy+span.zw);let half=.5*(span.zw-span.xy);var result=0.0;
 for(var a=0u;a<order;a++){
  var x=select(-.57735026919,.57735026919,a==1u);var wx=.5;
  if(order==3u){x=(f32(a)-1.0)*.77459666924;wx=select(.27777777778,.44444444444,a==1u);}
  for(var b=0u;b<select(1u,order,dimensions==2u);b++){
   var y=0.0;var wy=1.0;if(dimensions==2u){y=select(-.57735026919,.57735026919,b==1u);wy=.5;if(order==3u){y=(f32(b)-1.0)*.77459666924;wy=select(.27777777778,.44444444444,b==1u);}}
   result+=wx*wy*umRigidRayAt(candidates,center,size,rayAxis,axes,middle+half*vec2f(x,y),dimensions);
  }
 }
 return result;
}
// A feature whose conservative projected support intersects this patch but
// no high-order ray intersects it cannot be declared absent just because
// both quadratures agree. Charge its entire projected support as unresolved.
fn umRigidUnseenFeatureBound(candidates:u32,center:vec3f,size:vec3f,rayAxis:u32,axes:vec2u,span:vec4f,dimensions:u32)->f32{
 let middle=.5*(span.xy+span.zw);let half=.5*(span.zw-span.xy);var unresolved=0.0;var remaining=candidates;var direction=vec3f(0.0);direction[rayAxis]=1.0;let domain=vec2f(-.5,.5)*size[rayAxis];
 while(remaining!=0u){let i=firstTrailingBit(remaining);remaining&=remaining-1u;let body=umSolidBodies[i];let inverse=vec4f(body.orientation.x,-body.orientation.yzw);let localD=umQuatRotate(inverse,direction);let header=textureLoad(umSolidBodyCuts,vec2i(i32(i),0),0);let count=i32(header.y);let code=i32(round(body.positionShape.w));let shape=u32(select(code,-code-1,code<0));
  for(var feature= -1;feature<count;feature++){
   let bounds=umRigidFeatureBounds(i,feature);if(bounds.lo[rayAxis]>=center[rayAxis]+domain.y||bounds.hi[rayAxis]<=center[rayAxis]+domain.x){continue;}
   if(dimensions==1u){let fixed=3u-rayAxis-axes.x;if(center[fixed]<bounds.lo[fixed]||center[fixed]>bounds.hi[fixed]){continue;}}
   var area=1.0;for(var k=0u;k<dimensions;k++){let axis=axes[k];area*=max(0.0,min(span[k+2u],(bounds.hi[axis]-center[axis])/size[axis])-max(span[k],(bounds.lo[axis]-center[axis])/size[axis]));}
   if(area==0.0){continue;}var seen=false;
   for(var a=0u;a<3u;a++){
    for(var b=0u;b<select(1u,3u,dimensions==2u);b++){
     var origin=center;origin[axes.x]+=size[axes.x]*(middle.x+half.x*(f32(a)-1.0)*.77459666924);if(dimensions==2u){origin[axes.y]+=size[axes.y]*(middle.y+half.y*(f32(b)-1.0)*.77459666924);}
     let localO=umQuatRotate(inverse,origin-body.positionShape.xyz);var interval=umRayShapeOuter(shape,body.dimensions.xyz,localO,localD);if(feature>=0){interval=umRayCut(header,u32(feature),localO,localD);}
     interval=umRayIntersect(interval,domain);if(interval.y>interval.x){seen=true;break;}
    }if(seen){break;}
   }
   if(!seen){unresolved+=area;}
  }
 }
 return min(unresolved,(span.z-span.x)*select(1.0,span.w-span.y,dimensions==2u));
}
// Exact rounded-plate coverage. In its frame the plate is the set
// dist((x,z),Q)^2+y^2<=r^2 around the rectangle Q=[-sx,sx]x[-sz,sz]
// (dimensions = sx,r,sz). Its intersection with a frame-aligned box has a
// closed form: the transverse plane splits into Q, four edge strips and four
// corner quadrants, which contribute constants, disk segments and spheres
// over polar triangles. Gauss refinement cannot certify the rim's
// square-root silhouette, so a plate it can reach exactly never uses it.
fn umRimSegment(t:f32,r:f32)->f32{let s=clamp(t,-r,r);return .5*(s*sqrt(max(0.0,r*r-s*s))+r*r*asin(s/r));}
// int over t in [t0,t1]∩[0,R] of sqrt(r^2-t^2) (root) or 1.
fn umRimLine(t0:f32,t1:f32,R:f32,r:f32,root:bool)->f32{
 let a=max(t0,0.0);let b=min(t1,R);if(b<=a){return 0.0;}if(root){return umRimSegment(b,r)-umRimSegment(a,r);}return b-a;
}
// int_0^rho f(s) s ds.
fn umRimDisk(rho:f32,r:f32,root:bool)->f32{if(root){let q=max(0.0,r*r-rho*rho);return (r*r*r-q*sqrt(q))/3.0;}return .5*rho*rho;}
// int_0^atan(t) (r^2-U^2 sec^2 th)^(3/2) dth, valid while U sec th <= r.
fn umRimSphere(U:f32,t:f32,r:f32)->f32{
 let A=max(r*r-U*U,0.0);let S=sqrt(max(0.0,A-U*U*t*t));let s=asin(clamp(U*t/max(sqrt(A),1e-30),-1.0,1.0));
 return -.5*U*U*t*S-(.5*A+r*r)*U*s+r*r*r*atan2(r*t,S);
}
// Polar triangle 0<=th<=Theta, s<=min(R,U sec th) around a corner, R<=r.
fn umRimTriangle(U:f32,Theta:f32,R:f32,r:f32,root:bool)->f32{
 if(U<=0.0||Theta<=0.0){return 0.0;}
 let thetaR=select(acos(min(1.0,U/R)),0.0,U>=R);let theta=min(Theta,thetaR);var v=0.0;
 if(theta>0.0){let t=tan(theta);if(root){v=(r*r*r*theta-umRimSphere(U,t,r))/3.0;}else{v=.5*U*U*t;}}
 return v+max(0.0,Theta-thetaR)*umRimDisk(R,r,root);
}
fn umRimQuadrant(U:f32,V:f32,R:f32,r:f32,root:bool)->f32{
 if(U<=0.0||V<=0.0){return 0.0;}let theta=atan2(V,U);
 return umRimTriangle(U,theta,R,r,root)+umRimTriangle(V,1.5707963268-theta,R,r,root);
}
fn umRimCorner(u:vec2f,v:vec2f,R:f32,r:f32,root:bool)->f32{
 let u0=max(u.x,0.0);let v0=max(v.x,0.0);if(u.y<=u0||v.y<=v0||u0*u0+v0*v0>=R*R){return 0.0;}
 return umRimQuadrant(u.y,v.y,R,r,root)-umRimQuadrant(u0,v.y,R,r,root)-umRimQuadrant(u.y,v0,R,r,root)+umRimQuadrant(u0,v0,R,r,root);
}
// int over the patch x*z of f(d), for distance d<=R from the rectangle of
// half-extents e; f=sqrt(r^2-d^2) (root) or 1.
fn umRimRegion(x:vec2f,z:vec2f,e:vec2f,R:f32,r:f32,root:bool)->f32{
 var total=0.0;
 for(var i=0u;i<3u;i++){
  let a=max(x.x,select(select(e.x,-e.x,i==1u),-UM_RAY_INF,i==0u));let b=min(x.y,select(select(UM_RAY_INF,e.x,i==1u),-e.x,i==0u));if(b<=a){continue;}
  let ux=select(vec2f(a-e.x,b-e.x),vec2f(-b-e.x,-a-e.x),i==0u);
  for(var j=0u;j<3u;j++){
   let c=max(z.x,select(select(e.y,-e.y,j==1u),-UM_RAY_INF,j==0u));let d=min(z.y,select(select(UM_RAY_INF,e.y,j==1u),-e.y,j==0u));if(d<=c){continue;}
   let uz=select(vec2f(c-e.y,d-e.y),vec2f(-d-e.y,-c-e.y),j==0u);
   if(i==1u&&j==1u){total+=(b-a)*(d-c)*select(1.0,r,root);}
   else if(j==1u){total+=(d-c)*umRimLine(ux.x,ux.y,R,r,root);}
   else if(i==1u){total+=(b-a)*umRimLine(uz.x,uz.y,R,r,root);}
   else{total+=umRimCorner(ux,uz,R,r,root);}
  }
 }
 return total;
}
// int over the patch of min(c, sqrt(r^2-d^2)), c>=0: the chord half above 0
// clipped at height c.
fn umRimClippedChord(c:f32,x:vec2f,z:vec2f,e:vec2f,r:f32)->f32{
 let full=umRimRegion(x,z,e,r,r,true);if(c>=r){return full;}
 let R=sqrt(max(0.0,r*r-c*c));return full-umRimRegion(x,z,e,R,r,true)+c*umRimRegion(x,z,e,R,r,false);
}
// A negative result means the exact form does not apply here: one candidate
// that is a rounded plate aligned with the lattice axes and no cut reaching
// this box. Integrands (dimensions 2: the cell; 1: its face plane) match
// umRigidRayAt exactly.
fn umRigidPlateExactFraction(candidates:u32,center:vec3f,size:vec3f,rayAxis:u32,axes:vec2u,dimensions:u32)->f32{
 if(countOneBits(candidates)!=1u){return -1.0;}
 let i=firstTrailingBit(candidates);let body=umSolidBodies[i];let code=i32(round(body.positionShape.w));if(code!=5&&code!= -6){return -1.0;}
 let fixed=3u-rayAxis-axes.x;var lo=center-.5*size;var hi=center+.5*size;if(dimensions==1u){lo[fixed]=center[fixed];hi[fixed]=center[fixed];}
 let header=textureLoad(umSolidBodyCuts,vec2i(i32(i),0),0);
 for(var k=0;k<i32(header.y);k++){let b=umRigidFeatureBounds(i,k);if(all(b.lo<=hi)&&all(b.hi>=lo)){return -1.0;}}
 let inverse=vec4f(body.orientation.x,-body.orientation.yzw);var llo=vec3f(0.0);var lhi=vec3f(0.0);var mapped=0u;var fixedLocal=0u;
 for(var k=0u;k<3u;k++){
  var basis=vec3f(0.0);basis[k]=1.0;let v=umQuatRotate(inverse,basis);let m=abs(v);
  let j=select(select(2u,1u,m.y>=m.z),0u,m.x>=m.y&&m.x>=m.z);if(m.x+m.y+m.z-m[j]>4e-6){return -1.0;}
  mapped|=1u<<j;if(k==fixed){fixedLocal=j;}
  let a=lo[k]-body.positionShape[k];let b=hi[k]-body.positionShape[k];
  if(v[j]>0.0){llo[j]=a;lhi[j]=b;}else{llo[j]=-b;lhi[j]=-a;}
 }
 if(mapped!=7u){return -1.0;}
 let e=body.dimensions.xz;let r=body.dimensions.y;var fraction=0.0;
 let x=vec2f(llo.x,lhi.x);let y=vec2f(llo.y,lhi.y);let z=vec2f(llo.z,lhi.z);
 // The plate is convex: corners inside certify a closed box exactly, where
 // the closed forms' rounding would leave a sliver of aperture.
 var inside=true;
 for(var k=0u;k<8u;k++){let q=select(llo,lhi,vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u));let dx=max(abs(q.x)-e.x,0.0);let dz=max(abs(q.z)-e.y,0.0);if(dx*dx+dz*dz+q.y*q.y>r*r){inside=false;break;}}
 if(inside){fraction=1.0;}
 else if(dimensions==2u){
  fraction=(sign(y.y)*umRimClippedChord(abs(y.y),x,z,e,r)-sign(y.x)*umRimClippedChord(abs(y.x),x,z,e,r))/(size.x*size.y*size.z);
 }else{
  // The face plane's section: a rounded rectangle (normal along the plate's
  // y) or a stadium of radius sqrt(r^2-w^2) (normal along x or z). The
  // plate is a closed set (as the corner certificate and rigidShapeInside):
  // a face plane lying on its flat top, |c|=r, sections the full rectangle.
  let c=llo[fixedLocal];var area=0.0;
  if(fixedLocal==1u){if(abs(c)<=r){area=umRimRegion(x,z,e,sqrt(r*r-c*c),r,false);}}
  else if(fixedLocal==0u){let w=max(0.0,abs(c)-e.x);if(w<r){let rw=sqrt(r*r-w*w);area=umRimRegion(y,z,vec2f(0.0,e.y),rw,rw,false);}}
  else{let w=max(0.0,abs(c)-e.y);if(w<r){let rw=sqrt(r*r-w*w);area=umRimRegion(x,y,vec2f(e.x,0.0),rw,rw,false);}}
  fraction=area/(size[rayAxis]*size[axes.x]);
 }
 if(code<0){fraction=1.0-fraction;}
 return clamp(fraction,0.0,1.0);
}
fn umRigidIntegratedFraction(candidates:u32,center:vec3f,size:vec3f,rayAxis:u32,axes:vec2u,dimensions:u32)->f32{
 let exact=umRigidPlateExactFraction(candidates,center,size,rayAxis,axes,dimensions);if(exact>=0.0){return exact;}
 // Binary depth-first refinement needs one sibling per level, not a queue
 // proportional to all leaves. A resource limit raises cause11, never a
 // silent change to the cell geometry accepted by the frame.
 var spans:array<vec4f,32>;var depths:array<u32,32>;var count=1u;spans[0]=vec4f(-.5,-.5,.5,.5);depths[0]=0u;
 var value=0.0;var error=0.0;var work=0u;
 loop{
  if(count==0u){break;}count--;let span=spans[count];let depth=depths[count];work++;
  let area=(span.z-span.x)*select(1.0,span.w-span.y,dimensions==2u);
  var split=umRigidFeatureSplit(candidates,center,size,rayAxis,axes,span,dimensions);
  var high=0.0;var estimate=area;
  if(split.x<0.0){
   high=umRigidGauss(candidates,center,size,rayAxis,axes,span,dimensions,3u);
   let low=umRigidGauss(candidates,center,size,rayAxis,axes,span,dimensions,2u);estimate=area*abs(high-low);let unseen=umRigidUnseenFeatureBound(candidates,center,size,rayAxis,axes,span,dimensions);estimate=max(estimate,unseen);
   if(estimate<=umRigidQuadratureTolerance*area||(unseen>0.0&&estimate<=umRigidQuadratureTolerance*.005)){value+=area*high;error+=estimate;continue;}
   let axis=select(0u,1u,dimensions==2u&&(span.w-span.y)>(span.z-span.x));split=vec2f(f32(axis),.5*(span[axis]+span[axis+2u]));
  }
  if(depth>=30u||work>=umRigidQuadratureWork||count>=30u){
   if(high==0.0){high=umRigidGauss(candidates,center,size,rayAxis,axes,span,dimensions,3u);}
   value+=area*high;error+=estimate;umRigidCoverageFailure=vec2u(1u,bitcast<u32>(max(error,estimate)));continue;
  }
  let axis=u32(split.x);var lower=span;var upper=span;lower[axis+2u]=split.y;upper[axis]=split.y;
  spans[count]=upper;depths[count]=depth+1u;spans[count+1u]=lower;depths[count+1u]=depth+1u;count+=2u;
 }
 // Keep a resource-limit cause (1) visible; 2 is a converged-budget overrun.
 if(error>umRigidQuadratureTolerance&&umRigidCoverageFailure.x==0u){umRigidCoverageFailure=vec2u(2u,bitcast<u32>(error));}
 return clamp(value,0.0,1.0);
}
`;
