import { finishClothMechanicalStep, type ClothStepOptions, type ClothStepReport } from './dynamics';
import { rodSegmentInertia_kg_m2 } from './material';
import { solveClothIncrement } from './nonlinear-solve';
import type { PreparedClothTopology } from './topology';
import type { ClothState } from './types';

/** GPU Newton-PCG for the implicit-Euler incremental potential that
 * lib/cloth/nonlinear-solve.ts minimizes in Float64: inertia, finite
 * attachments, cohesion, Green-strain membrane and dihedral hinge energies,
 * two-vertex rod stretch, each with its Kelvin-Voigt term.
 *
 * Unknowns are increments from the accepted state, so Float32 resolves motion
 * far below a world-coordinate ULP. Old edges come from a hi/lo split of the
 * accepted positions; Green strain and its Kelvin-Voigt increment are formed
 * from increments, so strain changes carry no constitutive cancellation.
 *
 * Each Newton iteration, entirely on the GPU:
 *   element kernels  -> per-element strain, stress, dihedral derivatives
 *   gradient         -> per-vertex gradient and the coordinate solver's own
 *                       convergence measure max |g_k / max(m/dt^2, d2E/dx_k^2)|
 *   assemble         -> 3x3 block CSR of the same positive-semidefinite
 *                       Hessian model as the CPU solve (pattern built once)
 *   block-Jacobi PCG -> Eisenstat-Walker forcing, GPU dot products
 *   line search      -> Armijo on Float32 energy differences at 8 step sizes,
 *                       each formed from increments, with a Float32 roundoff
 *                       bound as the acceptance slack
 * One small status readback per two Newton iterations; positions are read
 * back once. Cohesion's Hessian enters on its vertices' diagonal blocks only
 * (its pairs change every step; the pattern does not). A GPU solve that does
 * not reach the tolerance is refined by the Float64 CPU Newton solve.
 *
 * Unsupported (callers keep the coordinate path): rods with bending/twist
 * joints (three or more vertices). */

const WG = 64;
const ALPHAS = 8;
// Control words (atomic u32).
const C_MEASURE = 0, C_ACTIVE = 1, C_CGACTIVE = 2, C_CGCOUNT = 3, C_NEWTON = 4, C_STATUS = 5, C_STOPAFTER = 6, C_USEZ0 = 7, C_UNIT = 8, C_LAST = 9, C_LOG = 16, LOG_UNITS = 16, C_WORDS = C_LOG + 4 * LOG_UNITS;
const S_CONVERGED = 1, S_LINE = 2, S_NONFINITE = 4, S_CAP = 8, S_SINGULAR = 16;
const KIND_TRI = 0, KIND_HINGE = 1, KIND_SEG = 2;

export function gpuClothNewtonSupported(topology: PreparedClothTopology): boolean {
  return topology.rods.every(rod => rod.vertices.length <= 2);
}

const PARAM_FIELDS = [
  'n', 'triangles', 'hinges', 'segments', 'pairs', 'blocks', 'cgCap', 'newtonCap',
  'dVertex', 'dMaterial', 'dTri', 'dHinge', 'dSeg', 'partTotal', 'pTri', 'pHinge',
  'pSeg', 'pPair', 'wgV', 'qW', 'iTri', 'iHinge', 'iSeg', 'iRow',
  'iCol', 'iBlockRow', 'iContribStart', 'iContrib', 'iVStart', 'iVList', 'iDiag', 'qiPairStart',
  'wX', 'wG', 'wR', 'wZ', 'wZ0', 'wP', 'wQ', 'wS',
  'wPre', 'wPart', 'wScal', 'qiVStart', 'eTri', 'eHinge', 'eSeg', 'eTriOld',
  'eHingeOld', 'eSegOld', 'eA', 'qiVList', 'qVW', 'qiPairVertex', 'pad0', 'pad1',
] as const;
const FLOAT_FIELDS = ['dt', 'tolerance', 'pad2', 'pad3'] as const;
type ParamName = typeof PARAM_FIELDS[number];

export const clothNewtonWGSL = /* wgsl */ `
struct Params { ${PARAM_FIELDS.map(f => `${f}:u32`).join(', ')}, ${FLOAT_FIELDS.map(f => `${f}:f32`).join(', ')} }
@group(0) @binding(0) var<storage,read> D:array<vec4f>;
@group(0) @binding(1) var<storage,read> I:array<u32>;
@group(0) @binding(2) var<storage,read_write> W:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> E:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> C:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> Q:array<f32>;
@group(0) @binding(6) var<storage,read> QI:array<u32>;
@group(0) @binding(7) var<uniform> p:Params;
const EPS:f32=1.1920929e-7;
var<workgroup> red:array<vec4f,${4 * WG}>;
var<workgroup> flagWord:u32;
// A control word every thread of the workgroup agrees on (barrier-safe).
fn uniformWord(l:u32,word:u32)->u32{if(l==0u){flagWord=atomicLoad(&C[word]);}return workgroupUniformLoad(&flagWord);}
fn uniformActive(l:u32)->bool{return uniformWord(l,${C_ACTIVE}u)!=0u;}

fn isActive()->bool{return atomicLoad(&C[${C_ACTIVE}])!=0u;}
fn cgActive()->bool{return atomicLoad(&C[${C_CGACTIVE}])!=0u;}
fn done()->bool{return atomicLoad(&C[${C_STATUS}])!=0u;}
fn pred(i:u32)->vec4f{return D[p.dVertex+4u*i];}
fn attach(i:u32)->vec4f{return D[p.dVertex+4u*i+1u];}
fn hi(i:u32)->vec4f{return D[p.dVertex+4u*i+2u];}
fn lo(i:u32)->vec4f{return D[p.dVertex+4u*i+3u];}
fn isFixed(i:u32)->bool{return hi(i).w>.5;}
fn xv(i:u32)->vec3f{return W[p.wX+i].xyz;}
fn oldEdge(a:u32,b:u32)->vec3f{return (hi(b).xyz-hi(a).xyz)+(lo(b).xyz-lo(a).xyz);}
fn mat(m:u32,k:u32)->vec4f{return D[p.dMaterial+3u*m+k];}
fn fractionScale(m:u32,f:f32)->f32{let e=mat(m,0u).w;if(e==0.){return 1.;}if(f<=0.){return 0.;}return pow(f,e);}
fn S(k:u32)->vec4f{return W[p.wScal+k];}
fn dir(i:u32)->vec3f{if(atomicLoad(&C[${C_USEZ0}])!=0u){return W[p.wZ0+i].xyz;}return W[p.wS+i].xyz;}
fn outer(a:vec3f,b:vec3f)->mat3x3f{return mat3x3f(a*b.x,a*b.y,a*b.z);}

// ---- workgroup reduction of up to four vec4 lanes; thread 0 gets the sum.
fn reduce4(l:u32,count:u32){
 for(var s=${WG / 2}u;s>0u;s>>=1u){workgroupBarrier();if(l<s){for(var k=0u;k<count;k++){red[k*${WG}u+l]+=red[k*${WG}u+l+s];}}}
 workgroupBarrier();
}
fn writePartial(base:u32,wg:u32,l:u32,count:u32){reduce4(l,count);if(l==0u){for(var k=0u;k<count;k++){W[p.wPart+4u*(base+wg)+k]=red[k*${WG}u];}}}
// Single-workgroup sum of partials [first,to); every thread returns the total.
fn sumPartials(l:u32,first:u32,to:u32,count:u32)->array<vec4f,4>{
 for(var k=0u;k<4u;k++){red[k*${WG}u+l]=vec4f(0);}
 for(var j=first+l;j<to;j+=${WG}u){for(var k=0u;k<count;k++){red[k*${WG}u+l]+=W[p.wPart+4u*j+k];}}
 reduce4(l,count);
 var out:array<vec4f,4>;for(var k=0u;k<4u;k++){out[k]=red[k*${WG}u];}
 workgroupBarrier();
 return out;
}

// ---- triangles: static D[dTri+2t]=(ru,restArea), D[dTri+2t+1]=(rv,material)
fn triIds(t:u32)->vec3u{return vec3u(I[p.iTri+4u*t],I[p.iTri+4u*t+1u],I[p.iTri+4u*t+2u]);}
struct TriMaterial{factor:f32,gamma:f32,nu:f32}
fn triMaterial(t:u32,id:vec3u)->TriMaterial{
 let s0=D[p.dTri+2u*t];let m=u32(D[p.dTri+2u*t+1u].w);let f=(lo(id.x).w+lo(id.y).w+lo(id.z).w)/3.;
 return TriMaterial(s0.w*mat(m,0u).x*fractionScale(m,f),mat(m,1u).y/p.dt,mat(m,1u).x);
}
struct TriStrain{u:vec3f,w:vec3f,e:vec3f,de:vec3f}
// Green strain e and its increment de=e-e_old from the increments; e_old
// from the hi/lo old edges.
fn triStrain(t:u32,id:vec3u,xa:vec3f,xb:vec3f,xc:vec3f)->TriStrain{
 let ru=D[p.dTri+2u*t].xyz;let rv=D[p.dTri+2u*t+1u].xyz;
 let du=ru.x*xa+ru.y*xb+ru.z*xc;let dw=rv.x*xa+rv.y*xb+rv.z*xc;
 let o0=E[p.eTriOld+3u*t];let o1=E[p.eTriOld+3u*t+1u];let o2=E[p.eTriOld+3u*t+2u];
 let de=vec3f(dot(o0.xyz,du)+.5*dot(du,du),dot(o1.xyz,dw)+.5*dot(dw,dw),dot(o0.xyz,dw)+dot(du,o1.xyz)+dot(du,dw));
 return TriStrain(o0.xyz+du,o1.xyz+dw,vec3f(o0.w,o1.w,o2.x)+de,de);
}
fn bilinear(a:vec3f,b:vec3f,nu:f32)->f32{return a.x*b.x+a.y*b.y+nu*(a.x*b.y+a.y*b.x)+.5*(1.-nu)*a.z*b.z;}
@compute @workgroup_size(${WG}) fn prepTri(@builtin(global_invocation_id) g:vec3u){
 let t=g.x;if(t>=p.triangles){return;}let id=triIds(t);let ru=D[p.dTri+2u*t].xyz;let rv=D[p.dTri+2u*t+1u].xyz;
 let e1=oldEdge(id.x,id.y);let e2=oldEdge(id.x,id.z);let u=ru.y*e1+ru.z*e2;let w=rv.y*e1+rv.z*e2;
 E[p.eTriOld+3u*t]=vec4f(u,.5*(dot(u,u)-1.));E[p.eTriOld+3u*t+1u]=vec4f(w,.5*(dot(w,w)-1.));E[p.eTriOld+3u*t+2u]=vec4f(dot(u,w),0,0,0);
}
@compute @workgroup_size(${WG}) fn elemTri(@builtin(global_invocation_id) g:vec3u){
 let t=g.x;if(t>=p.triangles||done()){return;}let id=triIds(t);
 let s=triStrain(t,id,xv(id.x),xv(id.y),xv(id.z));let m=triMaterial(t,id);let nu=m.nu;
 let sg=s.e+m.gamma*s.de;let sx=sg.x+nu*sg.y;let sy=sg.y+nu*sg.x;let ss=.5*(1.-nu)*sg.z;
 // Stress tensor (sx ss; ss sy) projected onto its nonnegative eigenvalues.
 var px=sx;var py=sy;var ps=ss;let mean=.5*(sx+sy);let radius=length(vec2f(.5*(sx-sy),ss));let l1=mean+radius;let l2=mean-radius;
 if(l2<0.){if(l1<=0.){px=0.;py=0.;ps=0.;}else{var cx=1.;var cs=0.;if(radius>0.){cx=.5*(sx-sy)/radius;cs=ss/radius;}px=l1*.5*(1.+cx);py=l1*.5*(1.-cx);ps=l1*.5*cs;}}
 E[p.eTri+4u*t]=vec4f(s.u,m.factor*sx);E[p.eTri+4u*t+1u]=vec4f(s.w,m.factor*sy);E[p.eTri+4u*t+2u]=m.factor*vec4f(px,py,ps,ss);E[p.eTri+4u*t+3u]=vec4f(m.factor,m.gamma,nu,0);
}

// ---- hinges: static D[dHinge+h]=(restAngle,shapeFactor,material,0)
fn hingeIds(h:u32)->vec4u{return vec4u(I[p.iHinge+4u*h],I[p.iHinge+4u*h+1u],I[p.iHinge+4u*h+2u],I[p.iHinge+4u*h+3u]);}
fn bendingPhi(angle:f32,mode:f32)->vec3f{if(mode==1.){let t=tan(angle*.5);return vec3f(2.*t,1.+t*t,t*(1.+t*t));}if(mode==2.){return vec3f(sin(angle*.5),.5*cos(angle*.5),-.25*sin(angle*.5));}return vec3f(angle,1,0);}
fn dihedral(e:vec3f,u:vec3f,w:vec3f)->f32{let n=normalize(cross(e,u));let m=normalize(cross(w,e));return atan2(dot(normalize(e),cross(n,m)),dot(n,m));}
@compute @workgroup_size(${WG}) fn prepHinge(@builtin(global_invocation_id) g:vec3u){
 let h=g.x;if(h>=p.hinges){return;}let id=hingeIds(h);let s=D[p.dHinge+h];let mode=mat(u32(s.z),2u).x;
 let angle=dihedral(oldEdge(id.x,id.y),oldEdge(id.x,id.z),oldEdge(id.x,id.w));
 E[p.eHingeOld+h]=vec4f(bendingPhi(angle,mode).x,bendingPhi(s.x,mode).x,angle,0);
}
struct HingeAt{e:vec3f,u:vec3f,w:vec3f}
fn hingeEdges(id:vec4u,xa:vec3f,xb:vec3f,xc:vec3f,xd:vec3f)->HingeAt{return HingeAt(oldEdge(id.x,id.y)+(xb-xa),oldEdge(id.x,id.z)+(xc-xa),oldEdge(id.x,id.w)+(xd-xa));}
struct HingeMaterial{k:f32,gamma:f32,mode:f32}
fn hingeMaterial(h:u32,id:vec4u)->HingeMaterial{let s=D[p.dHinge+h];let m=u32(s.z);return HingeMaterial(mat(m,0u).y*fractionScale(m,.5*(lo(id.x).w+lo(id.y).w))*s.y,mat(m,1u).z/p.dt,mat(m,2u).x);}
@compute @workgroup_size(${WG}) fn elemHinge(@builtin(global_invocation_id) g:vec3u){
 let h=g.x;if(h>=p.hinges||done()){return;}let id=hingeIds(h);let at=hingeEdges(id,xv(id.x),xv(id.y),xv(id.z),xv(id.w));let mm=hingeMaterial(h,id);
 let e=at.e;let u=at.u;let w=at.w;let rawN=cross(e,u);let rawM=cross(w,e);let sn=length(rawN);let sm=length(rawM);let n=rawN/sn;let m=rawM/sm;
 let l=length(e);let t=e/l;let product=cross(n,m);let x=dot(n,m);let y=dot(t,product);let den=x*x+y*y;let angle=atan2(y,x);
 let phi=bendingPhi(angle,mm.mode);let old=E[p.eHingeOld+h];let strain=phi.x-old.y+mm.gamma*(phi.x-old.x);
 let A=mm.k*((1.+mm.gamma)*phi.y*phi.y+strain*phi.z);let B=mm.k*strain*phi.y;
 for(var local=0u;local<4u;local++){
  var grad=vec3f(0);var second=vec3f(0);
  for(var axis=0u;axis<3u;axis++){
   var unit=vec3f(0);unit[axis]=1.;
   let de=unit*(select(0.,1.,local==1u)-select(0.,1.,local==0u));let du=unit*(select(0.,1.,local==2u)-select(0.,1.,local==0u));let dv=unit*(select(0.,1.,local==3u)-select(0.,1.,local==0u));
   let dn=(cross(de,u)+cross(e,du))/sn;let dm=(cross(dv,e)+cross(w,de))/sm;
   let dl=dot(e,de)/l;let ddl=(dot(de,de)-dl*dl)/l;let td=de/l-e*(dl/(l*l));let tdd=de*(-2.*dl/(l*l))+e*(2.*dl*dl/(l*l*l)-ddl/(l*l));
   let productD=cross(dn,m)+cross(n,dm);let productDD=2.*cross(dn,dm);
   let dx=dot(dn,m)+dot(n,dm);let ddx=2.*dot(dn,dm);let dy=dot(td,product)+dot(t,productD);let ddy=dot(tdd,product)+2.*dot(td,productD)+dot(t,productDD);
   let num=x*dy-y*dx;grad[axis]=num/den;second[axis]=((x*ddy-y*ddx)*den-num*2.*(x*dx+y*dy))/(den*den);
  }
  E[p.eHinge+9u*h+2u*local]=vec4f(grad,0);E[p.eHinge+9u*h+2u*local+1u]=vec4f(second,0);
 }
 E[p.eHinge+9u*h+8u]=vec4f(A,B,max(0.,A),0);
}

// ---- two-vertex rod segments: static D[dSeg+s]=(restLength,material,0,0)
fn segIds(s:u32)->vec2u{return vec2u(I[p.iSeg+2u*s],I[p.iSeg+2u*s+1u]);}
struct SegMaterial{factor:f32,gamma:f32,rest:f32}
fn segMaterial(s:u32,id:vec2u,oldLength:f32)->SegMaterial{
 let d=D[p.dSeg+s];let m=u32(d.y);let multiplier=mat(m,2u).y>.5;var f=.5*(lo(id.x).w+lo(id.y).w);if(multiplier){f=lo(id.x).w;}
 var gamma=mat(m,1u).w/p.dt;if(multiplier){gamma*=d.x/oldLength;}
 return SegMaterial(mat(m,0u).z*fractionScale(m,f)/d.x,gamma,d.x);
}
@compute @workgroup_size(${WG}) fn prepSeg(@builtin(global_invocation_id) g:vec3u){let s=g.x;if(s>=p.segments){return;}let id=segIds(s);E[p.eSegOld+s]=vec4f(length(oldEdge(id.x,id.y)),0,0,0);}
// Length l and its increment l-l_old from the increment.
fn segLength(id:vec2u,delta:vec3f,oldLength:f32)->vec4f{let e0=oldEdge(id.x,id.y);let e=e0+delta;let l=length(e);return vec4f(e,(2.*dot(e0,delta)+dot(delta,delta))/(l+oldLength));}
@compute @workgroup_size(${WG}) fn elemSeg(@builtin(global_invocation_id) g:vec3u){
 let s=g.x;if(s>=p.segments||done()){return;}let id=segIds(s);let oldLength=E[p.eSegOld+s].x;let mm=segMaterial(s,id,oldLength);
 let q=segLength(id,xv(id.y)-xv(id.x),oldLength);let l=oldLength+q.w;let strain=(oldLength-mm.rest)+q.w+mm.gamma*q.w;
 E[p.eSeg+2u*s]=vec4f(q.xyz,l);E[p.eSeg+2u*s+1u]=vec4f(mm.factor,mm.gamma,strain,0);
}

// ---- cohesion pairs: Q[4k..]=(delta0.xyz,F); entries QI[qiPairVertex+j], Q[qW+j]
fn pairDelta(k:u32,useDir:bool)->vec3f{var d=vec3f(0);if(!useDir){d=vec3f(Q[4u*k],Q[4u*k+1u],Q[4u*k+2u]);}for(var j=QI[p.qiPairStart+k];j<QI[p.qiPairStart+k+1u];j++){let v=QI[p.qiPairVertex+j];if(useDir){d+=Q[p.qW+j]*dir(v);}else{d+=Q[p.qW+j]*xv(v);}}return d;}

@compute @workgroup_size(${WG}) fn initX(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i>=p.n){return;}if(isFixed(i)){W[p.wX+i]=vec4f(attach(i).xyz,0);}else{W[p.wX+i]=vec4f(pred(i).xyz,0);}}

// ---- gradient and convergence measure
@compute @workgroup_size(${WG}) fn gradient(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${C_STATUS}u)!=0u){return;}
 let i=g.x;var gg=0.;
 if(i<p.n){
  if(isFixed(i)){W[p.wG+i]=vec4f(0);}else{
   let x=xv(i);let pr=pred(i);let at=attach(i);var gr=pr.w*(x-pr.xyz)+at.w*(x-at.xyz);var diag=vec3f(pr.w+at.w);
   for(var k=I[p.iVStart+i];k<I[p.iVStart+i+1u];k++){
    let code=I[p.iVList+k];let el=code&0xffffffu;let local=(code>>24u)&3u;let kind=code>>28u;
    if(kind==${KIND_TRI}u){
     let s0=E[p.eTri+4u*el];let s1=E[p.eTri+4u*el+1u];let s2=E[p.eTri+4u*el+2u];let s3=E[p.eTri+4u*el+3u];
     let ru=D[p.dTri+2u*el][local];let rv=D[p.dTri+2u*el+1u][local];let a=ru*s0.xyz;let b=rv*s1.xyz;let c=ru*s1.xyz+rv*s0.xyz;let nu=s3.z;
     gr+=s0.w*a+s1.w*b+s2.w*c;diag+=s3.x*(1.+s3.y)*(a*a+b*b+2.*nu*a*b+.5*(1.-nu)*c*c)+vec3f(s0.w*ru*ru+s1.w*rv*rv+2.*s2.w*ru*rv);
    }else if(kind==${KIND_HINGE}u){
     let gp=E[p.eHinge+9u*el+2u*local].xyz;let kk=E[p.eHinge+9u*el+2u*local+1u].xyz;let ab=E[p.eHinge+9u*el+8u];
     gr+=ab.y*gp;diag+=ab.x*gp*gp+ab.y*kk;
    }else{
     let s0=E[p.eSeg+2u*el];let s1=E[p.eSeg+2u*el+1u];let d=s0.xyz/s0.w;let sign=select(-1.,1.,local==1u);
     gr+=s1.x*s1.z*d*sign;diag+=s1.x*((1.+s1.y)*d*d+s1.z/s0.w*(1.-d*d));
    }
   }
   for(var k=QI[p.qiVStart+i];k<QI[p.qiVStart+i+1u];k++){
    let pair=QI[p.qiVList+k];let w=Q[p.qVW+k];let delta=pairDelta(pair,false);let distance=max(1e-14,length(delta));let nr=delta/distance;let F=Q[4u*pair+3u];
    gr+=F*w*nr;diag+=F*w*w*(1.-nr*nr)/distance;
   }
   W[p.wG+i]=vec4f(gr,0);
   let r=abs(gr)/max(vec3f(pr.w),diag);let measure=max(r.x,max(r.y,r.z));
   if(measure>=0.){atomicMax(&C[${C_MEASURE}],bitcast<u32>(measure));}else{atomicOr(&C[${C_STATUS}],${S_NONFINITE}u);}
   if(!(abs(measure)<3e38)){atomicOr(&C[${C_STATUS}],${S_NONFINITE}u);}
   gg=dot(gr,gr);
  }
 }
 red[l]=vec4f(gg,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishMeasure(@builtin(local_invocation_index) l:u32){
 let sums=sumPartials(l,0u,p.wgV,1u);
 if(l!=0u){return;}
 let unit=atomicAdd(&C[${C_UNIT}],1u);let status=atomicLoad(&C[${C_STATUS}]);
 if(status!=0u){atomicStore(&C[${C_ACTIVE}],0u);atomicStore(&C[${C_CGACTIVE}],0u);return;}
 let measure=bitcast<f32>(atomicLoad(&C[${C_MEASURE}]));atomicStore(&C[${C_MEASURE}],0u);
 atomicStore(&C[${C_LAST}],bitcast<u32>(measure));if(unit<${LOG_UNITS}u){atomicStore(&C[${C_LOG}u+4u*unit],bitcast<u32>(measure));}
 var next=0u;
 if(measure<p.tolerance){next=${S_CONVERGED}u;}else if(atomicLoad(&C[${C_NEWTON}])>=p.newtonCap){next=${S_CAP}u;}
 if(next!=0u){atomicOr(&C[${C_STATUS}],next);atomicStore(&C[${C_ACTIVE}],0u);atomicStore(&C[${C_CGACTIVE}],0u);return;}
 atomicAdd(&C[${C_NEWTON}],1u);atomicStore(&C[${C_ACTIVE}],1u);
 let gn2=sums[0].x;let previous=S(1u).y;var forcing=.1;if(previous>0.){forcing=min(.1,max(1e-6,sqrt(gn2/previous)));}
 W[p.wScal+1u]=vec4f(gn2,gn2,0,0);W[p.wScal+2u]=vec4f(0,forcing,0,0);
}

// ---- Hessian model assembly, one 3x3 block per thread
@compute @workgroup_size(${WG}) fn assemble(@builtin(global_invocation_id) g:vec3u){
 let b=g.x;if(b>=p.blocks||!isActive()){return;}let row=I[p.iBlockRow+b];if(isFixed(row)){return;}let col=I[p.iCol+b];
 var M=mat3x3f(vec3f(0),vec3f(0),vec3f(0));
 for(var k=I[p.iContribStart+b];k<I[p.iContribStart+b+1u];k++){
  let code=I[p.iContrib+k];let el=code&0xffffffu;let a=(code>>24u)&3u;let c=(code>>26u)&3u;let kind=code>>28u;
  if(kind==${KIND_TRI}u){
   let s0=E[p.eTri+4u*el];let s1=E[p.eTri+4u*el+1u];let s2=E[p.eTri+4u*el+2u];let s3=E[p.eTri+4u*el+3u];let nu=s3.z;
   let ru=D[p.dTri+2u*el].xyz;let rv=D[p.dTri+2u*el+1u].xyz;
   let p0=ru[a]*s0.xyz;let p1=rv[a]*s1.xyz;let p2=rv[a]*s0.xyz+ru[a]*s1.xyz;let q0=ru[c]*s0.xyz;let q1=rv[c]*s1.xyz;let q2=rv[c]*s0.xyz+ru[c]*s1.xyz;
   let material=s3.x*(1.+s3.y);let geometric=s2.x*ru[a]*ru[c]+s2.y*rv[a]*rv[c]+s2.z*(ru[a]*rv[c]+rv[a]*ru[c]);
   M+=material*(outer(p0,q0)+outer(p1,q1)+nu*(outer(p0,q1)+outer(p1,q0))+.5*(1.-nu)*outer(p2,q2))+mat3x3f(vec3f(geometric,0,0),vec3f(0,geometric,0),vec3f(0,0,geometric));
  }else if(kind==${KIND_HINGE}u){
   let coefficient=E[p.eHinge+9u*el+8u].z;if(coefficient>0.){M+=coefficient*outer(E[p.eHinge+9u*el+2u*a].xyz,E[p.eHinge+9u*el+2u*c].xyz);}
  }else{
   let s0=E[p.eSeg+2u*el];let s1=E[p.eSeg+2u*el+1u];let t=s0.xyz/s0.w;let along=s1.x*(1.+s1.y);let across=s1.x*max(0.,s1.z)/s0.w;let tt=outer(t,t);
   M+=select(-1.,1.,a==c)*(along*tt+across*(mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1))-tt));
  }
 }
 if(row==col){
  let d=pred(row).w+attach(row).w;M+=mat3x3f(vec3f(d,0,0),vec3f(0,d,0),vec3f(0,0,d));
  for(var k=QI[p.qiVStart+row];k<QI[p.qiVStart+row+1u];k++){
   let pair=QI[p.qiVList+k];let w=Q[p.qVW+k];let delta=pairDelta(pair,false);let distance=max(1e-14,length(delta));let nr=delta/distance;
   M+=Q[4u*pair+3u]*w*w/distance*(mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1))-outer(nr,nr));
  }
 }
 let T=transpose(M);E[p.eA+3u*b]=vec4f(T[0],0);E[p.eA+3u*b+1u]=vec4f(T[1],0);E[p.eA+3u*b+2u]=vec4f(T[2],0);
}

// ---- block-Jacobi preconditioned CG on H s = -g
fn applyPre(i:u32,v:vec3f)->vec3f{return vec3f(dot(W[p.wPre+3u*i].xyz,v),dot(W[p.wPre+3u*i+1u].xyz,v),dot(W[p.wPre+3u*i+2u].xyz,v));}
@compute @workgroup_size(${WG}) fn cgInit(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let i=g.x;var rz=0.;
 if(i<p.n){
  let z0=vec4f(0);
  if(isFixed(i)){W[p.wPre+3u*i]=z0;W[p.wPre+3u*i+1u]=z0;W[p.wPre+3u*i+2u]=z0;W[p.wR+i]=z0;W[p.wZ+i]=z0;W[p.wZ0+i]=z0;W[p.wP+i]=z0;W[p.wS+i]=z0;}
  else{
   let b=I[p.iDiag+i];let r0=E[p.eA+3u*b].xyz;let r1=E[p.eA+3u*b+1u].xyz;let r2=E[p.eA+3u*b+2u].xyz;
   let c0=cross(r1,r2);let c1=cross(r2,r0);let c2=cross(r0,r1);let det=dot(r0,c0);
   if(!(abs(det)>0.&&abs(det)<3e38)){atomicOr(&C[${C_STATUS}],${S_SINGULAR}u);}
   // The inverse has columns c0,c1,c2 over det; store its rows.
   let it=transpose(mat3x3f(c0,c1,c2)*(1./det));
   W[p.wPre+3u*i]=vec4f(it[0],0);W[p.wPre+3u*i+1u]=vec4f(it[1],0);W[p.wPre+3u*i+2u]=vec4f(it[2],0);
   let r=-W[p.wG+i].xyz;let z=applyPre(i,r);W[p.wR+i]=vec4f(r,0);W[p.wZ+i]=vec4f(z,0);W[p.wZ0+i]=vec4f(z,0);W[p.wP+i]=vec4f(z,0);W[p.wS+i]=vec4f(0);rz=dot(r,z);
  }
 }
 red[l]=vec4f(rz,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishInit(@builtin(local_invocation_index) l:u32){
 if(!uniformActive(l)){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 if(atomicLoad(&C[${C_STATUS}])!=0u){atomicStore(&C[${C_ACTIVE}],0u);atomicStore(&C[${C_CGACTIVE}],0u);return;}
 let rz=sums[0].x;let forcing=S(2u).y;W[p.wScal]=vec4f(rz,forcing*forcing*rz,0,0);W[p.wScal+2u]=vec4f(rz,forcing,0,0);
 atomicStore(&C[${C_CGCOUNT}],0u);atomicStore(&C[${C_STOPAFTER}],0u);atomicStore(&C[${C_USEZ0}],0u);atomicStore(&C[${C_CGACTIVE}],select(0u,1u,rz>0.));
}
@compute @workgroup_size(${WG}) fn cgMatvec(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let i=g.x;var pq=0.;
 if(i<p.n){
  var y=vec3f(0);
  if(!isFixed(i)){for(var b=I[p.iRow+i];b<I[p.iRow+i+1u];b++){let v=W[p.wP+I[p.iCol+b]].xyz;y+=vec3f(dot(E[p.eA+3u*b].xyz,v),dot(E[p.eA+3u*b+1u].xyz,v),dot(E[p.eA+3u*b+2u].xyz,v));}}
  W[p.wQ+i]=vec4f(y,0);pq=dot(W[p.wP+i].xyz,y);
 }
 red[l]=vec4f(pq,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishAlpha(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 let curvature=sums[0].x;var s0=S(0u);
 if(!(curvature>0.)){if(atomicLoad(&C[${C_CGCOUNT}])==0u){s0.z=1.;atomicStore(&C[${C_STOPAFTER}],1u);}else{atomicStore(&C[${C_CGACTIVE}],0u);}}
 else{s0.z=s0.x/curvature;}
 W[p.wScal]=s0;
}
@compute @workgroup_size(${WG}) fn cgUpdate(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let i=g.x;var rz=0.;
 if(i<p.n){let alpha=S(0u).z;W[p.wS+i]=vec4f(W[p.wS+i].xyz+alpha*W[p.wP+i].xyz,0);let r=W[p.wR+i].xyz-alpha*W[p.wQ+i].xyz;let z=applyPre(i,r);W[p.wR+i]=vec4f(r,0);W[p.wZ+i]=vec4f(z,0);rz=dot(r,z);}
 red[l]=vec4f(rz,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishBeta(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 var s0=S(0u);let next=sums[0].x;s0.w=next/s0.x;s0.x=next;W[p.wScal]=s0;
 let count=atomicAdd(&C[${C_CGCOUNT}],1u)+1u;
 if(atomicLoad(&C[${C_STOPAFTER}])!=0u||!(next>s0.y)||!(next>0.)||count>=p.cgCap){atomicStore(&C[${C_CGACTIVE}],0u);}
}
@compute @workgroup_size(${WG}) fn cgDirection(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i>=p.n||!cgActive()){return;}W[p.wP+i]=vec4f(W[p.wZ+i].xyz+S(0u).w*W[p.wP+i].xyz,0);}

// ---- descent check and line search
@compute @workgroup_size(${WG}) fn slope(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let i=g.x;var gs=0.;if(i<p.n){gs=dot(W[p.wG+i].xyz,W[p.wS+i].xyz);}
 red[l]=vec4f(gs,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishSlope(@builtin(local_invocation_index) l:u32){
 if(!uniformActive(l)){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 var s1=S(1u);s1.z=sums[0].x;
 // Roundoff in CG: the preconditioned gradient is always a descent direction.
 if(!(s1.z<0.)){atomicStore(&C[${C_USEZ0}],1u);s1.z=-S(2u).x;}
 W[p.wScal+1u]=s1;
 let unit=atomicLoad(&C[${C_UNIT}])-1u;if(unit<${LOG_UNITS}u){atomicStore(&C[${C_LOG}u+4u*unit+2u],atomicLoad(&C[${C_CGCOUNT}])|(atomicLoad(&C[${C_USEZ0}])<<31u));}
}
fn alphaOf(k:u32)->f32{return exp2(-f32(k));}
fn storeLine(l:u32,de:array<f32,${ALPHAS}>,noise:array<f32,${ALPHAS}>){red[l]=vec4f(de[0],de[1],de[2],de[3]);red[${WG}u+l]=vec4f(de[4],de[5],de[6],de[7]);red[${2 * WG}u+l]=vec4f(noise[0],noise[1],noise[2],noise[3]);red[${3 * WG}u+l]=vec4f(noise[4],noise[5],noise[6],noise[7]);}
@compute @workgroup_size(${WG}) fn lineTri(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let t=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(t<p.triangles){
  let id=triIds(t);let s=triStrain(t,id,xv(id.x),xv(id.y),xv(id.z));let m=triMaterial(t,id);
  let ru=D[p.dTri+2u*t].xyz;let rv=D[p.dTri+2u*t+1u].xyz;let da=dir(id.x);let db=dir(id.y);let dc=dir(id.z);
  let du=ru.x*da+ru.y*db+ru.z*dc;let dw=rv.x*da+rv.y*db+rv.z*dc;
  let lin=vec3f(dot(s.u,du),dot(s.w,dw),dot(s.u,dw)+dot(du,s.w));let quad=vec3f(.5*dot(du,du),.5*dot(dw,dw),dot(du,dw));
  for(var k=0u;k<${ALPHAS}u;k++){
   let a=alphaOf(k);let change=a*lin+a*a*quad;
   let elastic=bilinear(change,2.*s.e+change,m.nu);let viscous=m.gamma*bilinear(change,2.*s.de+change,m.nu);
   de[k]=.5*m.factor*(elastic+viscous);noise[k]=32.*EPS*.5*m.factor*(abs(elastic)+abs(viscous));
  }
 }
 storeLine(l,de,noise);writePartial(p.pTri,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn lineHinge(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let h=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(h<p.hinges){
  let id=hingeIds(h);let mm=hingeMaterial(h,id);let old=E[p.eHingeOld+h];
  let xa=xv(id.x);let xb=xv(id.y);let xc=xv(id.z);let xd=xv(id.w);let at=hingeEdges(id,xa,xb,xc,xd);let angle=dihedral(at.e,at.u,at.w);let phi=bendingPhi(angle,mm.mode);
  let elastic=phi.x-old.y;let viscous=phi.x-old.x;
  let da=dir(id.x);let db=dir(id.y);let dc=dir(id.z);let dd=dir(id.w);
  for(var k=0u;k<${ALPHAS}u;k++){
   let a=alphaOf(k);let next=bendingPhi(dihedral(at.e+a*(db-da),at.u+a*(dc-da),at.w+a*(dd-da)),mm.mode);let change=next.x-phi.x;
   let ee=change*(2.*elastic+change);let vv=mm.gamma*change*(2.*viscous+change);de[k]=.5*mm.k*(ee+vv);
   // Each Float32 angle is good to a few ULP of pi.
   let angleError=8.*EPS*(3.2+abs(angle))*max(abs(phi.y),abs(next.y));
   noise[k]=32.*EPS*.5*mm.k*(abs(ee)+abs(vv))+.5*mm.k*angleError*(abs(2.*elastic+change)+mm.gamma*abs(2.*viscous+change)+2.*(1.+mm.gamma)*angleError);
  }
 }
 storeLine(l,de,noise);writePartial(p.pHinge,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn lineSeg(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let s=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(s<p.segments){
  let id=segIds(s);let oldLength=E[p.eSegOld+s].x;let mm=segMaterial(s,id,oldLength);let delta=xv(id.y)-xv(id.x);let q=segLength(id,delta,oldLength);let l0=oldLength+q.w;
  let dd=dir(id.y)-dir(id.x);let elastic=(oldLength-mm.rest)+q.w;
  for(var k=0u;k<${ALPHAS}u;k++){
   let a=alphaOf(k);let e1=q.xyz+a*dd;let change=(2.*a*dot(q.xyz,dd)+a*a*dot(dd,dd))/(length(e1)+l0);
   let ee=change*(2.*elastic+change);let vv=mm.gamma*change*(2.*q.w+change);de[k]=.5*mm.factor*(ee+vv);
   noise[k]=32.*EPS*.5*mm.factor*(abs(ee)+abs(vv))+mm.factor*abs(change)*(1.+mm.gamma)*4.*EPS*l0;
  }
 }
 storeLine(l,de,noise);writePartial(p.pSeg,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn lineVertex(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let i=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(i<p.n&&!isFixed(i)){
  let x=xv(i);let pr=pred(i);let at=attach(i);let d=dir(i);let dd=dot(d,d);let inertial=dot(x-pr.xyz,d);let spring=dot(x-at.xyz,d);
  for(var k=0u;k<${ALPHAS}u;k++){let a=alphaOf(k);let ii=pr.w*(a*inertial+.5*a*a*dd);let ss=at.w*(a*spring+.5*a*a*dd);de[k]=ii+ss;noise[k]=32.*EPS*(abs(ii)+abs(ss));}
 }
 storeLine(l,de,noise);writePartial(0u,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn linePair(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let k0=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(k0<p.pairs){
  let delta=pairDelta(k0,false);let d=pairDelta(k0,true);let length0=length(delta);let F=Q[4u*k0+3u];
  for(var k=0u;k<${ALPHAS}u;k++){let a=alphaOf(k);let lengthA=length(delta+a*d);var change=0.;if(lengthA+length0>0.){change=(2.*a*dot(delta,d)+a*a*dot(d,d))/(lengthA+length0);}de[k]=F*change;noise[k]=32.*EPS*abs(F*change);}
 }
 storeLine(l,de,noise);writePartial(p.pPair,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn finishLine(@builtin(local_invocation_index) l:u32){
 if(!uniformActive(l)){return;}
 var sums=sumPartials(l,0u,p.partTotal,4u);if(l!=0u){return;}
 let slope=S(1u).z;var chosen=0.;
 for(var k=0u;k<${ALPHAS}u;k++){
  let a=alphaOf(k);let change=sums[k/4u][k%4u];let noise=sums[2u+k/4u][k%4u];
  if(abs(change)<3e38&&change<=1e-4*a*slope+noise){chosen=a;break;}
 }
 var s1=S(1u);s1.w=chosen;W[p.wScal+1u]=s1;
 let unit=atomicLoad(&C[${C_UNIT}])-1u;if(unit<${LOG_UNITS}u){atomicStore(&C[${C_LOG}u+4u*unit+1u],bitcast<u32>(chosen));}
 if(chosen==0.){atomicOr(&C[${C_STATUS}],${S_LINE}u);atomicStore(&C[${C_ACTIVE}],0u);}
}
@compute @workgroup_size(${WG}) fn apply(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i>=p.n||!isActive()){return;}W[p.wX+i]=vec4f(xv(i)+S(1u).w*dir(i),0);}
`;

const ENTRY_POINTS = ['prepTri', 'prepHinge', 'prepSeg', 'initX', 'elemTri', 'elemHinge', 'elemSeg', 'gradient', 'finishMeasure', 'assemble', 'cgInit', 'finishInit',
  'cgMatvec', 'finishAlpha', 'cgUpdate', 'finishBeta', 'cgDirection', 'slope', 'finishSlope', 'lineTri', 'lineHinge', 'lineSeg', 'lineVertex', 'linePair', 'finishLine', 'apply'] as const;
type EntryPoint = typeof ENTRY_POINTS[number];

export interface GPUClothNewtonReport extends ClothStepReport {
  readonly newtonIterations: number;
  readonly linearIterations: number;
  readonly submissions: number;
  /** Wall time of the GPU solve (upload, submissions, readbacks) and of the
   * step's Float64 finish (contact, CPU refinement when needed). */
  readonly solve_ms: number;
  readonly finish_ms: number;
}

export class GPUClothNewton {
  private readonly buffers: GPUBuffer[] = [];
  private readonly n: number;
  private readonly blocks: number;
  private readonly offsets: Record<ParamName, number>;
  private readonly D: GPUBuffer;
  private readonly I: GPUBuffer;
  private W: GPUBuffer;
  private readonly E: GPUBuffer;
  private readonly C: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly statusRead: GPUBuffer;
  private readonly positionRead: GPUBuffer;
  private Q: GPUBuffer;
  private QI: GPUBuffer;
  private qCapacity = 0;
  private qiCapacity = 0;
  private pairGroupCapacity = 0;
  private uploadedPairs = -1;
  private pipelines?: Record<EntryPoint, GPUComputePipeline>;
  private layout?: GPUBindGroupLayout;
  private bindings?: GPUBindGroup;
  private cgCap = 64;
  private readonly segmentRods: Int32Array;
  private readonly storageLimit: number;

  constructor(private readonly device: GPUDevice, readonly topology: PreparedClothTopology) {
    if (!gpuClothNewtonSupported(topology)) throw new Error('GPU cloth Newton supports shells and two-vertex rods only; rod joints need the coordinate solver');
    const n = this.n = topology.vertexCount, T = topology.triangles.length, H = topology.hinges.length, S = topology.segmentCount;
    if (Math.max(n, T, H, S) >= 1 << 24) throw new Error('GPU cloth Newton element index exceeds 24 bits');
    // Static float data after the per-step vertex block: materials, triangles, hinges, segments.
    const materials = topology.materials, M = materials.length;
    const dMaterial = 4 * n, dTri = dMaterial + 3 * M, dHinge = dTri + 2 * T, dSeg = dHinge + H, dTotal = dSeg + S;
    const staticD = new Float32Array(4 * dTotal);
    materials.forEach((m, k) => {
      const a = m.rodRadius_m, b = m.rodRadiusB_m ?? a;
      staticD.set([m.youngModulus_Pa * m.thickness_m / (1 - m.poissonRatio ** 2), m.youngModulus_Pa * m.thickness_m ** 3 / (12 * (1 - m.poissonRatio ** 2)), m.youngModulus_Pa * Math.PI * a * b, m.solidFractionExponent,
        m.poissonRatio, m.membraneDamping_s ?? m.mechanicalDamping_s, m.bendingDamping_s ?? m.mechanicalDamping_s, m.rodStretchDamping_s ?? m.mechanicalDamping_s,
        m.bendingMode === 'tan-half' ? 1 : m.bendingMode === 'sin-half' ? 2 : 0, m.attachmentMultiplier !== undefined ? 1 : 0, 0, 0], 4 * (dMaterial + 3 * k));
    });
    topology.triangles.forEach((face, t) => {
      const scale = Math.max(...face.ru.map(Math.abs), ...face.rv.map(Math.abs));
      if (Math.abs(face.ru[0] + face.ru[1] + face.ru[2]) > 1e-9 * scale || Math.abs(face.rv[0] + face.rv[1] + face.rv[2]) > 1e-9 * scale) throw new Error(`Cloth triangle ${t} strain weights are not translation invariant`);
      staticD.set([face.ru[0], face.ru[1], face.ru[2], face.restArea_m2, face.rv[0], face.rv[1], face.rv[2], face.material], 4 * (dTri + 2 * t));
    });
    topology.hinges.forEach((hinge, h) => staticD.set([hinge.restAngle_rad, hinge.shapeFactor, topology.triangles[hinge.faces[0]]!.material, 0], 4 * (dHinge + h)));
    const segA = new Uint32Array(S), segB = new Uint32Array(S);
    this.segmentRods = new Int32Array(S);
    topology.rods.forEach((rod, r) => {for (let j = 0; j < rod.restLengths_m.length; j++) {const s = rod.segmentOffset + j; segA[s] = rod.vertices[j]!; segB[s] = rod.vertices[j + 1]!; this.segmentRods[s] = r; staticD.set([rod.restLengths_m[j]!, rod.material, 0, 0], 4 * (dSeg + s));}});
    // Elements as (kind, index, vertices).
    const elementVertices = (kind: number, e: number): readonly number[] => kind === KIND_TRI ? topology.triangles[e]!.vertices : kind === KIND_HINGE ? topology.hinges[e]!.vertices : [segA[e]!, segB[e]!];
    const counts = [T, H, S], forEachElement = (visit: (kind: number, e: number, vertices: readonly number[]) => void) => {for (let kind = 0; kind < 3; kind++) for (let e = 0; e < counts[kind]!; e++) visit(kind, e, elementVertices(kind, e));};
    // Block pattern: every vertex pair sharing an element.
    const neighbors: Set<number>[] = Array.from({length: n}, (_, i) => new Set([i]));
    forEachElement((_, __, vertices) => {for (const a of vertices) for (const b of vertices) neighbors[a]!.add(b);});
    const rowStart = new Uint32Array(n + 1), sorted = neighbors.map(set => Uint32Array.from(set).sort());
    for (let i = 0; i < n; i++) rowStart[i + 1] = rowStart[i]! + sorted[i]!.length;
    const blocks = this.blocks = rowStart[n]!, columns = new Uint32Array(blocks), blockRow = new Uint32Array(blocks), diagonal = new Uint32Array(n);
    if (Math.ceil(blocks / WG) > 65535) throw new Error('GPU cloth Newton block count exceeds one dispatch');
    for (let i = 0; i < n; i++) {columns.set(sorted[i]!, rowStart[i]!); blockRow.fill(i, rowStart[i]!, rowStart[i + 1]!); diagonal[i] = rowStart[i]! + sorted[i]!.indexOf(i);}
    const blockOf = (row: number, column: number) => {for (let b = rowStart[row]!; b < rowStart[row + 1]!; b++) if (columns[b] === column) return b; throw new Error('Cloth Newton block pattern is incomplete');};
    const contribStart = new Uint32Array(blocks + 1), vertexStart = new Uint32Array(n + 1);
    forEachElement((_, __, vertices) => {for (const a of vertices) {vertexStart[a + 1]!++; for (const b of vertices) contribStart[blockOf(a, b) + 1]!++;}});
    for (let b = 0; b < blocks; b++) contribStart[b + 1]! += contribStart[b]!;
    for (let i = 0; i < n; i++) vertexStart[i + 1]! += vertexStart[i]!;
    const contrib = new Uint32Array(contribStart[blocks]!), vertexList = new Uint32Array(vertexStart[n]!), contribFill = contribStart.slice(0, blocks), vertexFill = vertexStart.slice(0, n);
    forEachElement((kind, e, vertices) => vertices.forEach((a, pa) => {
      vertexList[vertexFill[a]!++] = e | pa << 24 | kind << 28;
      vertices.forEach((b, pb) => {contrib[contribFill[blockOf(a, b)]!++] = e | pa << 24 | pb << 26 | kind << 28;});
    }));
    const segIds = new Uint32Array(2 * S); for (let s = 0; s < S; s++) {segIds[2 * s] = segA[s]!; segIds[2 * s + 1] = segB[s]!;}
    const triIds = new Uint32Array(4 * T); topology.triangles.forEach((face, t) => triIds.set(face.vertices, 4 * t));
    const hingeIds = new Uint32Array(4 * H); topology.hinges.forEach((hinge, h) => hingeIds.set(hinge.vertices, 4 * h));
    const parts = [triIds, hingeIds, segIds, rowStart, columns, blockRow, contribStart, contrib, vertexStart, vertexList, diagonal], at: number[] = [];
    let total = 0; for (const part of parts) {at.push(total); total += part.length;}
    const indices = new Uint32Array(Math.max(4, total)); parts.forEach((part, k) => indices.set(part, at[k]!));
    const [iTri, iHinge, iSeg, iRow, iCol, iBlockRow, iContribStart, iContrib, iVStart, iVList, iDiag] = at as [number, number, number, number, number, number, number, number, number, number, number];
    // Read-write vectors (vec4): x g r z z0 p q s pre(3) scalars(4) partials.
    const wgV = Math.ceil(n / WG), pTri = wgV, pHinge = pTri + Math.ceil(T / WG), pSeg = pHinge + Math.ceil(H / WG), pPair = pSeg + Math.ceil(S / WG);
    const wX = 0, wG = n, wR = 2 * n, wZ = 3 * n, wZ0 = 4 * n, wP = 5 * n, wQ = 6 * n, wS = 7 * n, wPre = 8 * n, wScal = 11 * n, wPart = wScal + 4;
    const eTri = 0, eHinge = 4 * T, eSeg = eHinge + 9 * H, eTriOld = eSeg + 2 * S, eHingeOld = eTriOld + 3 * T, eSegOld = eHingeOld + H, eA = eSegOld + S, eTotal = eA + 3 * blocks;
    this.offsets = {
      n, triangles: T, hinges: H, segments: S, pairs: 0, blocks, cgCap: 0, newtonCap: 0,
      dVertex: 0, dMaterial, dTri, dHinge, dSeg, partTotal: 0, pTri, pHinge, pSeg, pPair, wgV, qW: 0,
      iTri, iHinge, iSeg, iRow, iCol, iBlockRow, iContribStart, iContrib, iVStart, iVList, iDiag, qiPairStart: 0,
      wX, wG, wR, wZ, wZ0, wP, wQ, wS, wPre, wPart, wScal, qiVStart: 0, eTri, eHinge, eSeg, eTriOld,
      eHingeOld, eSegOld, eA, qiVList: 0, qVW: 0, qiPairVertex: 0, pad0: 0, pad1: 0,
    };
    this.storageLimit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.D = this.make('Cloth Newton per-step and static data', staticD.byteLength, storage);
    this.I = this.make('Cloth Newton topology and block pattern', indices.byteLength, storage);
    this.W = this.make('Cloth Newton vectors and reductions', this.wBytes(0), storage);
    this.E = this.make('Cloth Newton element states and Hessian blocks', 16 * eTotal, storage);
    this.C = this.make('Cloth Newton control', 4 * C_WORDS, storage);
    this.params = this.make('Cloth Newton parameters', 4 * (PARAM_FIELDS.length + FLOAT_FIELDS.length), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.statusRead = this.make('Cloth Newton status readback', 4 * C_WORDS, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.positionRead = this.make('Cloth Newton increments readback', 16 * n, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.Q = this.make('Cloth Newton cohesion pairs', 16, storage);
    this.QI = this.make('Cloth Newton cohesion indices', 16, storage);
    this.writeWords(this.I, indices);
    this.writeWords(this.D, staticD);
  }
  private make(label: string, bytes: number, usage: number): GPUBuffer {
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    if (size > this.storageLimit) throw new RangeError(`${label} (${size} bytes) exceeds the GPU storage binding limit`);
    const buffer = this.device.createBuffer({label, size, usage}); this.buffers.push(buffer); return buffer;
  }
  private replace(old: GPUBuffer, label: string, bytes: number, usage: number): GPUBuffer {
    old.destroy(); this.buffers.splice(this.buffers.indexOf(old), 1);
    return this.make(label, bytes, usage);
  }
  /** Vectors, scalars and one 4-lane partial per workgroup of every reduction. */
  private wBytes(pairGroups: number): number {return 16 * (this.offsets.wPart + 4 * (this.offsets.pPair + pairGroups));}
  private writeWords(buffer: GPUBuffer, values: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, offsetWords = 0, from = 0, count = values.length - from): void {
    const chunk = 256 * 1024;
    for (let k = 0; k < count; k += chunk) this.device.queue.writeBuffer(buffer, 4 * (offsetWords + k), values, from + k, Math.min(chunk, count - k));
  }

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({label: 'Cloth Newton-PCG', code: clothNewtonWGSL});
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    this.layout = this.device.createBindGroupLayout({label: 'Cloth Newton', entries: Array.from({length: 8}, (_, binding) => ({binding, visibility: GPUShaderStage.COMPUTE,
      buffer: {type: binding === 7 ? 'uniform' as const : [0, 1, 5, 6].includes(binding) ? 'read-only-storage' as const : 'storage' as const}}))});
    const layout = this.device.createPipelineLayout({bindGroupLayouts: [this.layout]});
    const pipelines = await Promise.all(ENTRY_POINTS.map(entryPoint => this.device.createComputePipelineAsync({label: `Cloth Newton ${entryPoint}`, layout, compute: {module, entryPoint}})));
    this.pipelines = Object.fromEntries(ENTRY_POINTS.map((e, k) => [e, pipelines[k]!])) as Record<EntryPoint, GPUComputePipeline>;
    this.bind();
  }
  private bind(): void {
    this.bindings = this.device.createBindGroup({label: 'Cloth Newton', layout: this.layout!, entries: [this.D, this.I, this.W, this.E, this.C, this.Q, this.QI, this.params].map((buffer, binding) => ({binding, resource: {buffer}}))});
  }

  private uploadPairs(pairs: NonNullable<ClothStepOptions['cohesionPairs']>, old: Float64Array): {qW: number; qVW: number; qiPairStart: number; qiPairVertex: number; qiVStart: number; qiVList: number} {
    const n = this.n, P = pairs.length, entries = pairs.reduce((s, pair) => s + pair.vertices.length, 0);
    // Each pair's distinct vertices with merged weights, then a vertex-major CSR.
    const pairVertex = new Uint32Array(entries), pairWeight = new Float64Array(entries), distinctStart = new Uint32Array(P + 1), perVertex = new Uint32Array(n + 1);
    let distinct = 0;
    pairs.forEach((pair, k) => {
      if (pair.vertices.length !== pair.weights.length || !Number.isFinite(pair.force_N) || pair.force_N < 0 || !pair.weights.every(Number.isFinite)) throw new Error('Invalid cloth cohesion pair');
      const weight = new Map<number, number>();
      pair.vertices.forEach((v, j) => {if (!Number.isSafeInteger(v) || v < 0 || v >= n) throw new Error('Invalid cohesion vertex'); weight.set(v, (weight.get(v) ?? 0) + pair.weights[j]!);});
      for (const [v, w] of weight) {pairVertex[distinct] = v; pairWeight[distinct] = w; perVertex[v + 1]!++; distinct++;}
      distinctStart[k + 1] = distinct;
    });
    for (let i = 0; i < n; i++) perVertex[i + 1]! += perVertex[i]!;
    const incidentCount = distinct;
    const qW = 4 * P, qVW = qW + entries, qFloats = qVW + incidentCount;
    const qiPairStart = 0, qiPairVertex = P + 1, qiVStart = qiPairVertex + entries, qiVList = qiVStart + n + 1, qiWords = qiVList + incidentCount;
    const offsets = {qW, qVW, qiPairStart, qiPairVertex, qiVStart, qiVList};
    // An empty pair set over an already empty upload changes nothing on the GPU.
    if (!P && this.uploadedPairs === 0) return offsets;
    const q = new Float32Array(Math.max(4, qFloats)), qi = new Uint32Array(Math.max(4, qiWords));
    let at = 0;
    pairs.forEach((pair, k) => {
      // delta0 = sum w old - target, in Float64.
      const delta = [0, 0, 0];
      pair.vertices.forEach((v, j) => {for (let a = 0; a < 3; a++) delta[a]! += pair.weights[j]! * old[3 * v + a]!;});
      if (pair.targetPosition_m) for (let a = 0; a < 3; a++) delta[a]! -= pair.targetPosition_m[a]!;
      q.set([delta[0]!, delta[1]!, delta[2]!, pair.force_N], 4 * k);
      qi[qiPairStart + k] = at;
      pair.vertices.forEach((v, j) => {qi[qiPairVertex + at] = v; q[qW + at] = pair.weights[j]!; at++;});
    });
    qi[qiPairStart + P] = at;
    qi.set(perVertex, qiVStart);
    const fill = perVertex.slice(0, n);
    for (let k = 0; k < P; k++) for (let d = distinctStart[k]!; d < distinctStart[k + 1]!; d++) {const v = pairVertex[d]!, slot = fill[v]!++; qi[qiVList + slot] = k; q[qVW + slot] = pairWeight[d]!;}
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    // Pairs vary per step; their lists and partials grow geometrically.
    let rebind = false;
    if (q.byteLength > this.qCapacity || qi.byteLength > this.qiCapacity) {
      this.qCapacity = Math.max(2 * this.qCapacity, Math.ceil(q.byteLength / 16) * 16, 4096);
      this.qiCapacity = Math.max(2 * this.qiCapacity, Math.ceil(qi.byteLength / 16) * 16, 4096);
      this.Q = this.replace(this.Q, 'Cloth Newton cohesion pairs', this.qCapacity, storage);
      this.QI = this.replace(this.QI, 'Cloth Newton cohesion indices', this.qiCapacity, storage);
      rebind = true;
    }
    const pairGroups = Math.ceil(P / WG);
    if (pairGroups > this.pairGroupCapacity) {
      this.pairGroupCapacity = Math.max(2 * this.pairGroupCapacity, pairGroups);
      this.W = this.replace(this.W, 'Cloth Newton vectors and reductions', this.wBytes(this.pairGroupCapacity), storage | GPUBufferUsage.COPY_SRC);
      rebind = true;
    }
    if (rebind) this.bind();
    this.writeWords(this.Q, q);
    this.writeWords(this.QI, qi);
    this.uploadedPairs = P;
    return offsets;
  }

  private encodeUnit(pass: GPUComputePassEncoder, cgCap: number, pairs: number): void {
    const o = this.offsets, n = this.n, groups = (count: number) => Math.ceil(count / WG);
    const run = (entry: EntryPoint, count: number) => {if (count > 0) {pass.setPipeline(this.pipelines![entry]); pass.dispatchWorkgroups(count);}};
    run('elemTri', groups(o.triangles)); run('elemHinge', groups(o.hinges)); run('elemSeg', groups(o.segments));
    run('gradient', groups(n)); run('finishMeasure', 1);
    run('assemble', groups(this.blocks)); run('cgInit', groups(n)); run('finishInit', 1);
    for (let k = 0; k < cgCap; k++) {run('cgMatvec', groups(n)); run('finishAlpha', 1); run('cgUpdate', groups(n)); run('finishBeta', 1); run('cgDirection', groups(n));}
    run('slope', groups(n)); run('finishSlope', 1);
    run('lineTri', groups(o.triangles)); run('lineHinge', groups(o.hinges)); run('lineSeg', groups(o.segments)); run('lineVertex', groups(n)); run('linePair', groups(pairs));
    run('finishLine', 1); run('apply', groups(n));
  }

  async step(state: ClothState, dt: number, options: ClothStepOptions = {}): Promise<GPUClothNewtonReport> {
    if (!this.pipelines || !this.bindings) throw new Error('Initialize GPU cloth Newton before stepping');
    if (!(dt > 0 && Number.isFinite(dt))) throw new Error('Invalid cloth GPU timestep');
    const started = performance.now(), topology = this.topology, n = this.n, old = state.positions_m.slice(), oldTwist = state.twist_rad.slice();
    const gravity = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.restPositions_m;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < old.length; i++) {const a = i % 3; if (old[i]! < lo[a]!) lo[a] = old[i]!; if (old[i]! > hi[a]!) hi[a] = old[i]!;}
    const origin = lo.map((v, a) => (v + hi[a]!) / 2), vertex = new Float32Array(16 * n), predicted = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const mass = state.dryMass_kg[i]! + state.waterVolume_m3[i]! * topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3, fixed = (topology.kinematicFixed[i] ?? 0) & 1;
      if (!(mass > 0 && Number.isFinite(mass))) throw new Error(`Invalid cloth vertex mass ${i}`);
      const o = 16 * i;
      for (let a = 0; a < 3; a++) {
        const k = 3 * i + a, increment = dt * state.velocities_m_s[k]! + dt * dt * (gravity[a]! + (options.externalForces_N?.[k] ?? 0) / mass), shifted = old[k]! - origin[a]!, high = Math.fround(shifted);
        predicted[k] = old[k]! + increment;
        vertex[o + a] = increment; vertex[o + 4 + a] = target[k]! - old[k]!; vertex[o + 8 + a] = high; vertex[o + 12 + a] = shifted - high;
      }
      vertex[o + 3] = mass / (dt * dt); vertex[o + 7] = topology.attachmentTranslation_N_m[i]!; vertex[o + 11] = fixed; vertex[o + 15] = state.solidFraction[i]!;
    }
    this.writeWords(this.D, vertex);
    const pairs = options.cohesionPairs ?? [], pairOffsets = this.uploadPairs(pairs, old);
    const pairGroups = Math.ceil(pairs.length / WG), newtonCap = options.iterations ?? 64, tolerance = options.tolerance_m ?? 1e-8;
    if (!Number.isInteger(newtonCap) || newtonCap < 1) throw new Error('Cloth iterations must be a positive integer');
    const values: Record<string, number> = {...this.offsets, ...pairOffsets, pairs: pairs.length, partTotal: this.offsets.pPair + pairGroups, cgCap: this.cgCap, newtonCap, dt, tolerance};
    const words = new ArrayBuffer(4 * (PARAM_FIELDS.length + FLOAT_FIELDS.length)), u = new Uint32Array(words), f = new Float32Array(words);
    PARAM_FIELDS.forEach((name, k) => {u[k] = values[name] ?? 0;});
    FLOAT_FIELDS.forEach((name, k) => {f[PARAM_FIELDS.length + k] = values[name] ?? 0;});
    this.device.queue.writeBuffer(this.params, 0, words);
    this.device.queue.writeBuffer(this.W, 16 * this.offsets.wScal, new Float32Array(16));
    const o = this.offsets, groups = (count: number) => Math.ceil(count / WG), unitsPerSubmission = 2;
    let submissions = 0, status = 0, control = new Uint32Array(C_WORDS);
    while (true) {
      const encoder = this.device.createCommandEncoder({label: 'Cloth Newton-PCG'});
      if (!submissions) encoder.clearBuffer(this.C);
      const pass = encoder.beginComputePass({label: 'Cloth Newton iterations'});
      pass.setBindGroup(0, this.bindings);
      if (!submissions) for (const [entry, count] of [['initX', n], ['prepTri', o.triangles], ['prepHinge', o.hinges], ['prepSeg', o.segments]] as const) if (count) {pass.setPipeline(this.pipelines[entry]); pass.dispatchWorkgroups(groups(count));}
      for (let u = 0; u < unitsPerSubmission; u++) this.encodeUnit(pass, this.cgCap, pairs.length);
      pass.end();
      encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
      encoder.copyBufferToBuffer(this.W, 16 * o.wX, this.positionRead, 0, 16 * n);
      this.device.queue.submit([encoder.finish()]);
      submissions++;
      await this.statusRead.mapAsync(GPUMapMode.READ);
      control = new Uint32Array(this.statusRead.getMappedRange().slice(0));
      this.statusRead.unmap();
      status = control[C_STATUS]!;
      if (status) break;
    }
    const units = Math.min(control[C_UNIT]!, LOG_UNITS), newton = control[C_NEWTON]!;
    const measure = new Float32Array(Uint32Array.of(control[C_LAST]!).buffer)[0]!;
    let linear = 0, cgUsed = 0;
    for (let k = 0; k < units; k++) {const cg = control[C_LOG + 4 * k + 2]! & 0x7fffffff; linear += cg; cgUsed = Math.max(cgUsed, cg);}
    // Adapt the encoded CG depth to what the last step used (inexact Newton
    // remains a descent method when the depth truncates CG).
    this.cgCap = cgUsed >= this.cgCap ? Math.min(2048, 2 * this.cgCap) : Math.max(16, Math.min(this.cgCap, 8 * Math.ceil(1.5 * cgUsed / 8)));
    await this.positionRead.mapAsync(GPUMapMode.READ);
    const increments = new Float32Array(this.positionRead.getMappedRange().slice(0));
    this.positionRead.unmap();
    const converged = (status & S_CONVERGED) !== 0, solved = performance.now(), solve_ms = solved - started;
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) state.positions_m[3 * i + a] = old[3 * i + a]! + increments[4 * i + a]!;
    // Jointless rod twist couples to nothing but its inertia and attachment.
    for (let s = 0; s < state.twist_rad.length; s++) {
      const r = this.segmentRods[s]!, rod = topology.rods[r]!, j = s - rod.segmentOffset, a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
      if (((topology.kinematicFixed[a] ?? 0) | (topology.kinematicFixed[b] ?? 0)) & 2) {state.twist_rad[s] = oldTwist[s]!; continue;}
      const inertia = rodSegmentInertia_kg_m2(topology, state, r, j) / (dt * dt), spring = topology.attachmentTwist_N_m[s]!;
      state.twist_rad[s] = (inertia * (oldTwist[s]! + dt * state.angularVelocity_rad_s[s]!) + spring * state.attachmentTwistTarget_rad[s]!) / (inertia + spring);
    }
    if (!converged) {
      // Float32 stalled (line search, cap) or broke: the Float64 Newton solve
      // finishes from the GPU iterate when it is finite, else from the prediction.
      const finite = !(status & (S_NONFINITE | S_SINGULAR)) && state.positions_m.every(Number.isFinite);
      const initialGuess = {positions_m: finite ? state.positions_m.slice() : predicted, twist_rad: state.twist_rad.slice()};
      state.positions_m.set(old); state.twist_rad.set(oldTwist);
      const refined = solveClothIncrement(topology, state, dt, options, initialGuess);
      return {...refined, iterations: newton + refined.iterations, gpuIterations: newton, refinementIterations: refined.iterations, newtonIterations: newton + refined.newtonIterations, linearIterations: linear + refined.linearIterations, submissions, solve_ms, finish_ms: performance.now() - solved};
    }
    const finished = finishClothMechanicalStep(topology, state, old, oldTwist, dt, options, {iterations: newton, converged, maximumUpdate_m: measure});
    return {...finished, gpuIterations: newton, refinementIterations: 0, newtonIterations: newton, linearIterations: linear, submissions, solve_ms, finish_ms: performance.now() - solved};
  }

  destroy(): void {for (const buffer of this.buffers) buffer.destroy();}
}
