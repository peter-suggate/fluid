import { beginClothMechanicalFinish, completeClothMechanicalFinish, extendClothStepReport, finishClothMechanicalStep, type ClothStepOptions, type ClothStepReport } from './dynamics';
import { ClothContactSet, clothContactWarmStart, FRICTION_SLIP_SPEED_M_S, inSolveContactUnsupported, MAXIMUM_CONTACT_ITERATIONS, rememberClothContactMultipliers, type ClothContactConstraint } from './contact-constraints';
import { rodSegmentInertia_kg_m2 } from './material';
import { solveClothIncrement } from './nonlinear-solve';
import { CONTACT_CONTROL, CONTACT_CONTROL_WORDS, E_CAPACITY, E_CELLS, E_CROSSED, E_GRID, E_NONFINITE, E_STIFFNESS, E_TABLE, SCAN_WG, clothContactStatic, clothContactWGSL, gpuContactUnsupported, packClothColliders, type ClothContactStatic } from './gpu-contact';
import type { ClothCollider, ClothContactReport } from './contact';
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
 * Contact is inside the minimization, as in the CPU solve
 * (lib/cloth/contact-constraints.ts): the Float64 contact set detects on the
 * CPU and its augmented-Lagrangian terms (smoothed normal penalty, lagged
 * friction) enter the GPU gradient, the diagonal blocks, a matrix-free gather
 * in the matvec (contacts couple vertices outside the block pattern) and the
 * line search. Between GPU solves the multipliers, collider tangent planes and
 * detection update on the CPU; each such outer iteration reads the positions
 * back once more. No projection follows the solve.
 *
 * Unsupported (callers keep the coordinate path): rods with bending/twist
 * joints (three or more vertices). */

const WG = 64;
const ALPHAS = 8;
// Armijo backtracking: LINE_PASSES gated passes of ALPHAS halvings each, the
// Float64 solve's 40 (a pass runs only while no earlier step size was accepted).
const LINE_PASSES = 5;
// Control words (atomic u32).
const C_MEASURE = 0, C_ACTIVE = 1, C_CGACTIVE = 2, C_CGCOUNT = 3, C_NEWTON = 4, C_STATUS = 5, C_STOPAFTER = 6, C_USEZ0 = 7, C_UNIT = 8, C_LAST = 9,
  C_NEWTONMIN = 10, C_NEWTON_TOTAL = 11, C_LINEAR = 12, C_CGMAX = 13, C_SEARCH = 14, C_LOG = 16, LOG_UNITS = 16, C_WORDS = CONTACT_CONTROL_WORDS;
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
  'eHingeOld', 'eSegOld', 'eA', 'qiVList', 'qVW', 'qiPairVertex', 'pContact', 'wK',
  // Contacts (lib/cloth/gpu-contact.ts): mode 0 none, 1 GPU detection, 2 uploaded from the CPU set.
  'contactMode', 'uploadedContacts', 'contactGroups', 'kMax', 'kuCV', 'kuMeta', 'kvStart', 'kvList',
  'uBlocks', 'uEntry', 'uTabVal', 'aTable', 'aHeads', 'aCount', 'tableSlots', 'tableCur',
  'prevValid', 'stiffnessValid', 'gridBuckets', 'gridEntries', 'kfFaceBox', 'kfEdgeBox', 'sVInfo', 'sRingStart',
  'sRingList', 'sFrStart', 'sFrList', 'sLvStart', 'sLvList', 'sEdge', 'sE3Start', 'sE3List',
  'sFaceR', 'sCollider', 'colliders', 'selfCollision', 'edges',
] as const;
const FLOAT_FIELDS = ['dt', 'tolerance', 'contactDelta', 'frictionEpsilon', 'cellSize'] as const;
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
@group(0) @binding(8) var<storage,read_write> KF:array<vec4f>;
@group(0) @binding(9) var<storage,read_write> KU:array<u32>;
@group(0) @binding(10) var<storage,read_write> KA:array<atomic<u32>>;
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

// ---- contacts of lib/cloth/contact-constraints.ts: g = g0 + n.sum w u, slip =
// P_t(sum w u) - P_t(obstacle), smoothed augmented-Lagrangian penalty of
// m = lambda/rho - g and lagged smoothed friction mu lambda_f f0(|slip|).
// KF[4c..]=(n,g0),(P_t obstacle,lambda),(rho,lambda_f,mu,radius),(w0..w3);
// KU[kuCV+4c..] vertices; free vertices' lists KU[kvStart+i..] -> KU[kvList+k]=c<<2|j.
struct Contact{n:vec3f,g0:f32,t:vec3f,lambda:f32,rho:f32,mu:f32,w:vec4f,v:vec4u}
fn contactAt(c:u32)->Contact{let a=KF[4u*c];let t=KF[4u*c+1u];let r=KF[4u*c+2u];let j=p.kuCV+4u*c;return Contact(a.xyz,a.w,t.xyz,t.w,r.x,r.z*r.y,KF[4u*c+3u],vec4u(KU[j],KU[j+1u],KU[j+2u],KU[j+3u]));}
// sum_j w_j of increments (0), search direction (1) or CG direction (2).
fn contactSum(c:Contact,which:u32)->vec3f{var s=vec3f(0);for(var j=0u;j<4u;j++){let w=c.w[j];if(w!=0.){let v=c.v[j];if(which==0u){s+=w*xv(v);}else if(which==1u){s+=w*dir(v);}else{s+=w*W[p.wP+v].xyz;}}}return s;}
fn contactM(c:Contact,sx:vec3f)->f32{return c.lambda/c.rho-(c.g0+dot(c.n,sx));}
fn tangent(n:vec3f,v:vec3f)->vec3f{return v-dot(n,v)*n;}
fn cForce(m:f32,rho:f32)->f32{let d=p.contactDelta;if(m<=-d){return 0.;}if(m<d){return rho*(m+d)*(m+d)/(4.*d);}return rho*m;}
fn cCurv(m:f32,rho:f32)->f32{let d=p.contactDelta;if(m<=-d){return 0.;}if(m<d){return rho*(m+d)/(2.*d);}return rho;}
fn cPen(m:f32,rho:f32)->f32{let d=p.contactDelta;if(m<=-d){return 0.;}if(m<d){return rho*(m+d)*(m+d)*(m+d)/(12.*d);}return .5*rho*m*m+rho*d*d/6.;}
// Penalty change from m to m+dm, factored within a branch; (change, magnitude for the roundoff bound).
fn cPenChange(m:f32,dm:f32,rho:f32)->vec2f{
 let d=p.contactDelta;let m1=m+dm;
 if(m>=d&&m1>=d){let c=rho*dm*(m+.5*dm);return vec2f(c,abs(c));}
 if(m<=-d&&m1<=-d){return vec2f(0);}
 if(abs(m)<d&&abs(m1)<d){let a=m1+d;let b=m+d;let c=rho*dm*(a*a+a*b+b*b)/(12.*d);return vec2f(c,abs(c));}
 let a=cPen(m1,rho);let b=cPen(m,rho);return vec2f(a-b,abs(a)+abs(b));
}
fn fF0(y:f32)->f32{let e=p.frictionEpsilon;if(y<e){return -y*y*y/(3.*e*e)+y*y/e+e/3.;}return y;}
fn fF1(y:f32)->f32{let e=p.frictionEpsilon;if(y<e){return 2./e-y/(e*e);}return 1./y;}
// f0(y1) - f0(y0) given squared change q = y1^2 - y0^2.
fn fF0Change(y0:f32,y1:f32,q:f32)->vec2f{
 let e=p.frictionEpsilon;let sum=y0+y1;
 if(sum>0.&&y0>=e&&y1>=e){let c=q/sum;return vec2f(c,abs(c));}
 if(sum>0.&&y0<e&&y1<e){let c=q/sum*((y0+y1)/e-(y0*y0+y0*y1+y1*y1)/(3.*e*e));return vec2f(c,abs(c));}
 let a=fF0(y1);let b=fF0(y0);return vec2f(a-b,abs(a)+abs(b));
}
// Normal and tangential Hessian coefficients at the current iterate.
fn contactCoefficients(c:Contact)->vec2f{let sx=contactSum(c,0u);var f=0.;if(c.mu>0.){f=c.mu*fF1(length(tangent(c.n,sx)-c.t));}return vec2f(cCurv(contactM(c,sx),c.rho),f);}

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
   // Coordinate stiffness without contact: sizes the penalties of new contacts.
   W[p.wK+i]=vec4f(diag,0);
   if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
    let e=KU[p.kvList+k];let c=contactAt(e>>2u);let w=c.w[e&3u];let sx=contactSum(c,0u);let m=contactM(c,sx);
    gr-=w*cForce(m,c.rho)*c.n;diag+=w*w*cCurv(m,c.rho)*c.n*c.n;
    if(c.mu>0.){let slip=tangent(c.n,sx)-c.t;let f=c.mu*fF1(length(slip));gr+=w*f*slip;diag+=w*w*f*(vec3f(1)-c.n*c.n);}
   }}
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
 if(measure<p.tolerance&&atomicLoad(&C[${C_NEWTON}])>=atomicLoad(&C[${C_NEWTONMIN}])){next=${S_CONVERGED}u;}else if(atomicLoad(&C[${C_NEWTON}])>=p.newtonCap){next=${S_CAP}u;}
 if(next!=0u){atomicOr(&C[${C_STATUS}],next);atomicStore(&C[${C_ACTIVE}],0u);atomicStore(&C[${C_CGACTIVE}],0u);return;}
 atomicAdd(&C[${C_NEWTON}],1u);atomicAdd(&C[${C_NEWTON_TOTAL}],1u);atomicStore(&C[${C_ACTIVE}],1u);
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
// The contacts' diagonal 3x3 block at vertex i (symmetric). It enters the
// preconditioner only: the matvec gathers the whole contact term itself.
fn contactBlock(i:u32)->mat3x3f{
 var M=mat3x3f(vec3f(0),vec3f(0),vec3f(0));
 if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
  let e=KU[p.kvList+k];let c=contactAt(e>>2u);let w=c.w[e&3u];let cf=contactCoefficients(c);let nn=outer(c.n,c.n);
  M+=w*w*(cf.x*nn+cf.y*(mat3x3f(vec3f(1,0,0),vec3f(0,1,0),vec3f(0,0,1))-nn));
 }}
 return M;
}
fn applyPre(i:u32,v:vec3f)->vec3f{return vec3f(dot(W[p.wPre+3u*i].xyz,v),dot(W[p.wPre+3u*i+1u].xyz,v),dot(W[p.wPre+3u*i+2u].xyz,v));}
@compute @workgroup_size(${WG}) fn cgInit(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let i=g.x;var rz=0.;
 if(i<p.n){
  let z0=vec4f(0);
  if(isFixed(i)){W[p.wPre+3u*i]=z0;W[p.wPre+3u*i+1u]=z0;W[p.wPre+3u*i+2u]=z0;W[p.wR+i]=z0;W[p.wZ+i]=z0;W[p.wZ0+i]=z0;W[p.wP+i]=z0;W[p.wS+i]=z0;}
  else{
   let b=I[p.iDiag+i];let K=contactBlock(i);let r0=E[p.eA+3u*b].xyz+K[0];let r1=E[p.eA+3u*b+1u].xyz+K[1];let r2=E[p.eA+3u*b+2u].xyz+K[2];
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
  if(!isFixed(i)){for(var b=I[p.iRow+i];b<I[p.iRow+i+1u];b++){let v=W[p.wP+I[p.iCol+b]].xyz;y+=vec3f(dot(E[p.eA+3u*b].xyz,v),dot(E[p.eA+3u*b+1u].xyz,v),dot(E[p.eA+3u*b+2u].xyz,v));}
   // Contacts couple vertices outside the block pattern: gathered matrix-free.
   if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
    let e=KU[p.kvList+k];let c=contactAt(e>>2u);let w=c.w[e&3u];let cf=contactCoefficients(c);let s=contactSum(c,2u);let sn=dot(c.n,s);
    y+=w*(cf.x*sn*c.n+cf.y*(s-sn*c.n));
   }}
  }
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
 W[p.wScal+1u]=s1;atomicStore(&C[${C_SEARCH}],1u);
 let cg=atomicLoad(&C[${C_CGCOUNT}]);atomicAdd(&C[${C_LINEAR}],cg);atomicMax(&C[${C_CGMAX}],cg);
 let unit=atomicLoad(&C[${C_UNIT}])-1u;if(unit<${LOG_UNITS}u){atomicStore(&C[${C_LOG}u+4u*unit+2u],cg|(atomicLoad(&C[${C_USEZ0}])<<31u));}
}
var<private> alphaBase:u32;
// The search pass every thread agrees on (0: no search in progress); sets the pass's first halving.
fn uniformSearch(l:u32)->bool{let stage=uniformWord(l,${C_SEARCH}u);alphaBase=${ALPHAS}u*(max(stage,1u)-1u);return stage!=0u;}
fn alphaOf(k:u32)->f32{return exp2(-f32(k+alphaBase));}
fn storeLine(l:u32,de:array<f32,${ALPHAS}>,noise:array<f32,${ALPHAS}>){red[l]=vec4f(de[0],de[1],de[2],de[3]);red[${WG}u+l]=vec4f(de[4],de[5],de[6],de[7]);red[${2 * WG}u+l]=vec4f(noise[0],noise[1],noise[2],noise[3]);red[${3 * WG}u+l]=vec4f(noise[4],noise[5],noise[6],noise[7]);}
@compute @workgroup_size(${WG}) fn lineTri(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformSearch(l)){return;}
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
 if(!uniformSearch(l)){return;}
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
 if(!uniformSearch(l)){return;}
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
 if(!uniformSearch(l)){return;}
 let i=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(i<p.n&&!isFixed(i)){
  let x=xv(i);let pr=pred(i);let at=attach(i);let d=dir(i);let dd=dot(d,d);let inertial=dot(x-pr.xyz,d);let spring=dot(x-at.xyz,d);
  for(var k=0u;k<${ALPHAS}u;k++){let a=alphaOf(k);let ii=pr.w*(a*inertial+.5*a*a*dd);let ss=at.w*(a*spring+.5*a*a*dd);de[k]=ii+ss;noise[k]=32.*EPS*(abs(ii)+abs(ss));}
 }
 storeLine(l,de,noise);writePartial(0u,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn linePair(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformSearch(l)){return;}
 let k0=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(k0<p.pairs){
  let delta=pairDelta(k0,false);let d=pairDelta(k0,true);let length0=length(delta);let F=Q[4u*k0+3u];
  for(var k=0u;k<${ALPHAS}u;k++){let a=alphaOf(k);let lengthA=length(delta+a*d);var change=0.;if(lengthA+length0>0.){change=(2.*a*dot(delta,d)+a*a*dot(d,d))/(lengthA+length0);}de[k]=F*change;noise[k]=32.*EPS*abs(F*change);}
 }
 storeLine(l,de,noise);writePartial(p.pPair,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn lineContact(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformSearch(l)){return;}
 var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 let count=min(contactCount(),p.kMax);
 for(var k0=g.x;k0<count;k0+=p.contactGroups*${WG}u){
  let c=contactAt(k0);let sx=contactSum(c,0u);let sd=contactSum(c,1u);let m=contactM(c,sx);let dn=dot(c.n,sd);
  let slip=tangent(c.n,sx)-c.t;let td=tangent(c.n,sd);let y0=length(slip);
  for(var k=0u;k<${ALPHAS}u;k++){
   let a=alphaOf(k);var change=cPenChange(m,-a*dn,c.rho);
   if(c.mu>0.){let f=fF0Change(y0,length(slip+a*td),2.*a*dot(slip,td)+a*a*dot(td,td));change+=c.mu*f;}
   de[k]+=change.x;noise[k]+=32.*EPS*change.y;
  }
 }
 storeLine(l,de,noise);writePartial(p.pContact,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn finishLine(@builtin(local_invocation_index) l:u32){
 if(!uniformSearch(l)){return;}
 var sums=sumPartials(l,0u,p.partTotal,4u);if(l!=0u){return;}
 let slope=S(1u).z;var chosen=0.;
 for(var k=0u;k<${ALPHAS}u;k++){
  let a=alphaOf(k);let change=sums[k/4u][k%4u];let noise=sums[2u+k/4u][k%4u];
  if(abs(change)<3e38&&change<=1e-4*a*slope+noise){chosen=a;break;}
 }
 let stage=atomicLoad(&C[${C_SEARCH}]);
 if(chosen==0.&&stage<${LINE_PASSES}u){atomicStore(&C[${C_SEARCH}],stage+1u);return;}
 atomicStore(&C[${C_SEARCH}],0u);
 var s1=S(1u);s1.w=chosen;W[p.wScal+1u]=s1;
 let unit=atomicLoad(&C[${C_UNIT}])-1u;if(unit<${LOG_UNITS}u){atomicStore(&C[${C_LOG}u+4u*unit+1u],bitcast<u32>(chosen));}
 if(chosen==0.){atomicOr(&C[${C_STATUS}],${S_LINE}u);atomicStore(&C[${C_ACTIVE}],0u);}
}
@compute @workgroup_size(${WG}) fn apply(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i>=p.n||!isActive()){return;}W[p.wX+i]=vec4f(xv(i)+S(1u).w*dir(i),0);}
${clothContactWGSL(WG, S_CONVERGED, {C_STATUS, C_NEWTON, C_ACTIVE, C_CGACTIVE, C_NEWTONMIN}, MAXIMUM_CONTACT_ITERATIONS)}
`;

const ENTRY_POINTS = ['prepTri', 'prepHinge', 'prepSeg', 'initX', 'elemTri', 'elemHinge', 'elemSeg', 'gradient', 'finishMeasure', 'assemble', 'cgInit', 'finishInit',
  'cgMatvec', 'finishAlpha', 'cgUpdate', 'finishBeta', 'cgDirection', 'slope', 'finishSlope', 'lineTri', 'lineHinge', 'lineSeg', 'lineVertex', 'linePair', 'lineContact', 'finishLine', 'apply',
  'tableClear', 'rememberMultipliers', 'roundBegin', 'detectBegin', 'updateMultipliers', 'relinearize', 'roundDecide', 'penalties',
  'listCount', 'listScan1', 'listScan2', 'listScan3', 'listFill', 'listReset', 'gridClear', 'gridFaces', 'gridEdges', 'detectFaces', 'detectEdges', 'detectColliders'] as const;
type EntryPoint = typeof ENTRY_POINTS[number];

export interface GPUClothNewtonReport extends ClothStepReport {
  readonly newtonIterations: number;
  readonly linearIterations: number;
  readonly submissions: number;
  /** Wall time of the GPU solve (upload, submissions, readbacks; with GPU
   * detection this includes every contact round) and of the step's Float64
   * finish (CPU refinement when needed). */
  readonly solve_ms: number;
  readonly finish_ms: number;
  /** CPU contact work between GPU solves (CPU detection only). */
  readonly contact_ms: number;
  /** GPU solves this step: one, plus one per contact (outer) iteration. */
  readonly gpuSolves: number;
  readonly contactModel: 'in-solve' | 'post-projection';
  readonly contactIterations?: number;
  /** Where in-solve contacts were detected, and why not on the GPU. */
  readonly contactDetection: 'gpu' | 'cpu' | 'none';
  readonly contactFallback?: string;
  /** The last GPU solve's status bits (converged 1, line search 2, nonfinite 4, Newton cap 8, singular 16). */
  readonly gpuStatus: number;
}

/** One constraint of the GPU contact set (tests and diagnostics). */
export interface GPUClothContact {
  readonly key: string; readonly vertices: readonly number[]; readonly weights: readonly number[];
  readonly normal: readonly [number, number, number]; readonly lambda: number; readonly frictionLambda: number; readonly rho: number;
}

interface ContactLayout {
  kMax: number; tableSlots: number; gridBuckets: number; gridEntries: number;
  kuCV: number; kuMeta: number; kvStart: number; kvList: number; uBlocks: number; uEntry: number; uTabVal: number; kuWords: number;
  aTable: number; aHeads: number; aCount: number; kaWords: number; kfFaceBox: number; kfEdgeBox: number; kfWords: number;
}
const pow2 = (x: number) => 2 ** Math.ceil(Math.log2(Math.max(1, x)));
const errorNames: [number, string][] = [[E_CAPACITY, 'contact capacity'], [E_GRID, 'grid entries'], [E_CROSSED, 'a vertex crossed a collider within the step'], [E_STIFFNESS, 'a contact without free stiffness'], [E_TABLE, 'key table'], [E_CELLS, 'a swept primitive covers more than the grid cell cap'], [E_NONFINITE, 'a nonfinite contact residual']];
const describeErrors = (bits: number) => errorNames.filter(([b]) => bits & b).map(([, name]) => name).join(', ');
const GROWABLE = E_CAPACITY | E_GRID | E_TABLE;
type Pairs = NonNullable<ClothStepOptions['cohesionPairs']>;
interface SolveResult {converged: boolean; status: number; measure: number; newton: number; linear: number; submissions: number; increments: Float32Array; control: Uint32Array<ArrayBuffer>}

export class GPUClothNewton {
  /** 'auto' detects on the GPU whenever the colliders allow it; 'cpu' keeps
   * the Float64 ClothContactSet (uploaded each outer iteration). */
  contactDetection: 'auto' | 'cpu' = 'auto';
  /** Diagnostics: the last solve's control words. */
  lastControl?: Uint32Array;
  /** Per Newton iteration of the last solve (first 16): convergence measure, chosen step, CG iterations. */
  lastIterations(): {measure: number; alpha: number; cg: number; preconditionedGradient: boolean}[] {
    const c = this.lastControl; if (!c) return [];
    const f = (w: number) => new Float32Array(Uint32Array.of(w).buffer)[0]!;
    return Array.from({length: Math.min(c[C_UNIT]!, LOG_UNITS)}, (_, k) => ({measure: f(c[C_LOG + 4*k]!), alpha: f(c[C_LOG + 4*k + 1]!), cg: c[C_LOG + 4*k + 2]! & 0x7fffffff, preconditionedGradient: (c[C_LOG + 4*k + 2]! >>> 31) === 1}));
  }
  private readonly buffers: GPUBuffer[] = [];
  private readonly n: number;
  private readonly blocks: number;
  private readonly offsets: Record<ParamName, number>;
  private readonly D: GPUBuffer;
  private I: GPUBuffer;
  private W: GPUBuffer;
  private readonly E: GPUBuffer;
  private readonly C: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly statusRead: GPUBuffer;
  private readonly positionRead: GPUBuffer;
  private readonly stiffnessRead: GPUBuffer;
  /** Coordinate stiffness without contact (3n) at the last solve's iterate. */
  private stiffness?: Float64Array;
  private Q: GPUBuffer;
  private QI: GPUBuffer;
  private KF: GPUBuffer;
  private KU: GPUBuffer;
  private KA: GPUBuffer;
  /** Topology words; contact statics and packed colliders follow them in I. */
  private readonly indexWords: Uint32Array<ArrayBuffer>;
  private indexCapacity: number;
  private qCapacity = 0;
  private qiCapacity = 0;
  private groupCapacity: number;
  private uploaded = -1;
  private pipelines?: Record<EntryPoint, GPUComputePipeline>;
  private layout?: GPUBindGroupLayout;
  private bindings?: GPUBindGroup;
  private cgCap = 64;
  private readonly segmentRods: Int32Array;
  private readonly storageLimit: number;
  /** Line-search partials and grid-stride width of the contact kernels. */
  private readonly contactGroups = 256;
  private contactLayout?: ContactLayout;
  private contactStatic?: ClothContactStatic;
  /** Fixed frame of the hi/lo split, so contact geometry and the key table
   * persist across steps (set at the first step). */
  private frame?: [number, number, number];
  private staticWritten = false;
  private colliderSignature?: string;
  private colliderCount = 0;
  private tableCur = 0;
  private prevValid = false;
  /** W's coordinate stiffness holds a gradient evaluation of this buffer. */
  private stiffnessValid = false;

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
    const wX = 0, wG = n, wR = 2 * n, wZ = 3 * n, wZ0 = 4 * n, wP = 5 * n, wQ = 6 * n, wS = 7 * n, wPre = 8 * n, wK = 11 * n, wScal = 12 * n, wPart = wScal + 4;
    const eTri = 0, eHinge = 4 * T, eSeg = eHinge + 9 * H, eTriOld = eSeg + 2 * S, eHingeOld = eTriOld + 3 * T, eSegOld = eHingeOld + H, eA = eSegOld + S, eTotal = eA + 3 * blocks;
    this.offsets = Object.fromEntries(PARAM_FIELDS.map(name => [name, 0])) as Record<ParamName, number>;
    Object.assign(this.offsets, {
      n, triangles: T, hinges: H, segments: S, blocks, dMaterial, dTri, dHinge, dSeg, pTri, pHinge, pSeg, pPair, wgV,
      iTri, iHinge, iSeg, iRow, iCol, iBlockRow, iContribStart, iContrib, iVStart, iVList, iDiag,
      wX, wG, wR, wZ, wZ0, wP, wQ, wS, wPre, wPart, wScal, wK, eTri, eHinge, eSeg, eTriOld, eHingeOld, eSegOld, eA,
    });
    if (device.limits.maxStorageBuffersPerShaderStage < 10) throw new Error(`GPU cloth Newton needs 10 storage buffers per stage; the device allows ${device.limits.maxStorageBuffersPerShaderStage}`);
    this.storageLimit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.groupCapacity = this.contactGroups;
    this.D = this.make('Cloth Newton per-step and static data', staticD.byteLength, storage);
    this.indexWords = indices; this.indexCapacity = indices.length;
    this.I = this.make('Cloth Newton topology and block pattern', indices.byteLength, storage);
    this.W = this.make('Cloth Newton vectors and reductions', this.wBytes(this.groupCapacity), storage);
    this.E = this.make('Cloth Newton element states and Hessian blocks', 16 * eTotal, storage);
    this.C = this.make('Cloth Newton control', 4 * C_WORDS, storage);
    this.params = this.make('Cloth Newton parameters', 4 * (PARAM_FIELDS.length + FLOAT_FIELDS.length), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.statusRead = this.make('Cloth Newton status readback', 4 * C_WORDS, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.positionRead = this.make('Cloth Newton increments readback', 16 * n, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.stiffnessRead = this.make('Cloth Newton coordinate stiffness readback', 16 * n, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.Q = this.make('Cloth Newton cohesion pairs', 16, storage);
    this.QI = this.make('Cloth Newton cohesion indices', 16, storage);
    this.KF = this.make('Cloth contact floats', 16, storage);
    this.KU = this.make('Cloth contact words', 16, storage);
    this.KA = this.make('Cloth contact atomics', 16, storage);
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
  /** Vectors, scalars and one 4-lane partial per workgroup of every reduction
   * (pair and contact line-search groups follow the fixed ones). */
  private wBytes(varyingGroups: number): number {return 16 * (this.offsets.wPart + 4 * (this.offsets.pPair + varyingGroups));}
  private writeWords(buffer: GPUBuffer, values: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, offsetWords = 0, from = 0, count = values.length - from): void {
    const chunk = 256 * 1024;
    for (let k = 0; k < count; k += chunk) this.device.queue.writeBuffer(buffer, 4 * (offsetWords + k), values, from + k, Math.min(chunk, count - k));
  }

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({label: 'Cloth Newton-PCG', code: clothNewtonWGSL});
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    this.layout = this.device.createBindGroupLayout({label: 'Cloth Newton', entries: Array.from({length: 11}, (_, binding) => ({binding, visibility: GPUShaderStage.COMPUTE,
      buffer: {type: binding === 7 ? 'uniform' as const : [0, 1, 5, 6].includes(binding) ? 'read-only-storage' as const : 'storage' as const}}))});
    const layout = this.device.createPipelineLayout({bindGroupLayouts: [this.layout]});
    const pipelines = await Promise.all(ENTRY_POINTS.map(entryPoint => this.device.createComputePipelineAsync({label: `Cloth Newton ${entryPoint}`, layout, compute: {module, entryPoint}})));
    this.pipelines = Object.fromEntries(ENTRY_POINTS.map((e, k) => [e, pipelines[k]!])) as Record<EntryPoint, GPUComputePipeline>;
    this.bind();
  }
  private bind(): void {
    this.bindings = this.device.createBindGroup({label: 'Cloth Newton', layout: this.layout!, entries: [this.D, this.I, this.W, this.E, this.C, this.Q, this.QI, this.params, this.KF, this.KU, this.KA].map((buffer, binding) => ({binding, resource: {buffer}}))});
  }

  /** Cohesion pairs of this step, packed in Q/QI. */
  private uploadPairs(pairs: Pairs, old: Float64Array): {qW: number; qVW: number; qiPairStart: number; qiPairVertex: number; qiVStart: number; qiVList: number} {
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
    // An empty set over an already empty upload changes nothing on the GPU.
    if (!P && this.uploaded === 0) return offsets;
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
    const varyingGroups = Math.ceil(P / WG) + this.contactGroups;
    if (varyingGroups > this.groupCapacity) {
      this.groupCapacity = Math.max(2 * this.groupCapacity, varyingGroups);
      this.W = this.replace(this.W, 'Cloth Newton vectors and reductions', this.wBytes(this.groupCapacity), storage | GPUBufferUsage.COPY_SRC);
      this.stiffnessValid = false;
      rebind = true;
    }
    if (rebind) this.bind();
    this.writeWords(this.Q, q);
    this.writeWords(this.QI, qi);
    this.uploaded = P;
    return offsets;
  }

  /** Contact buffers for kMax constraints (and the detection grid and key table). */
  private ensureContactCapacity(kMax: number, gridEntries = 0, tableFactor = 2): ContactLayout {
    const held = this.contactLayout, T = this.offsets.triangles, edges = this.contactStatic?.edges ?? 0, n = this.n;
    if (held && held.kMax >= kMax && held.gridEntries >= gridEntries && held.tableSlots >= tableFactor * held.kMax && held.kfEdgeBox + 2 * edges <= held.kfWords) return held;
    kMax = Math.max(kMax, held?.kMax ?? 0, 4096);
    gridEntries = Math.max(gridEntries, held?.gridEntries ?? 0, 4 * (T + edges), 1024);
    const tableSlots = Math.max(pow2(tableFactor * kMax), held?.tableSlots ?? 0), gridBuckets = pow2(Math.max(1024, T + edges));
    const kuCV = 0, kuMeta = 4 * kMax, kvStart = 8 * kMax, kvList = kvStart + n + 1, uBlocks = kvList + 4 * kMax, uEntry = uBlocks + Math.ceil(n / SCAN_WG) + 1, uTabVal = uEntry + 2 * gridEntries, kuWords = uTabVal + 2 * tableSlots;
    const aTable = 0, aHeads = 4 * tableSlots, aCount = aHeads + 2 * gridBuckets, kaWords = aCount + n;
    const kfFaceBox = 4 * kMax, kfEdgeBox = kfFaceBox + 2 * T, kfWords = kfEdgeBox + 2 * edges;
    const layout = {kMax, tableSlots, gridBuckets, gridEntries, kuCV, kuMeta, kvStart, kvList, uBlocks, uEntry, uTabVal, kuWords, aTable, aHeads, aCount, kaWords, kfFaceBox, kfEdgeBox, kfWords};
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.KF = this.replace(this.KF, 'Cloth contact floats', 16 * kfWords, storage);
    this.KU = this.replace(this.KU, 'Cloth contact words', 4 * kuWords, storage);
    this.KA = this.replace(this.KA, 'Cloth contact atomics', 4 * kaWords, storage);
    // New buffers are zero: an empty key table and grid, zero list counts.
    this.prevValid = false;
    this.contactLayout = layout;
    this.bind();
    return layout;
  }

  /** Rest-mesh exclusions and the colliders in the fixed frame, after the
   * topology words in I (the device allows ten storage buffers per stage). */
  private uploadContactStatic(colliders: readonly ClothCollider[]): void {
    const statics = this.contactStatic ??= clothContactStatic(this.topology);
    const signature = colliders.map(c => c.kind === 'sampled' ? `sampled#${this.colliderId(c)}` : JSON.stringify(c)).join('|');
    if (signature === this.colliderSignature) return;
    const base = this.indexWords.length, packed = packClothColliders(colliders, this.frame!, base + statics.words.length), total = base + statics.words.length + packed.length;
    if (total > this.indexCapacity) {
      this.indexCapacity = Math.max(total, base + 2 * (this.indexCapacity - base));
      this.I = this.replace(this.I, 'Cloth Newton topology, block pattern and contact statics', 4 * this.indexCapacity, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      this.writeWords(this.I, this.indexWords);
      this.staticWritten = false;
      this.bind();
    }
    if (!this.staticWritten) {this.writeWords(this.I, statics.words, base); this.staticWritten = true;}
    if (packed.length) this.writeWords(this.I, packed, base + statics.words.length);
    this.colliderSignature = signature;
    this.colliderCount = colliders.length;
  }
  private readonly colliderIds = new WeakMap<object, number>();
  private nextColliderId = 0;
  private colliderId(c: object): number {let id = this.colliderIds.get(c); if (id === undefined) {id = this.nextColliderId++; this.colliderIds.set(c, id);} return id;}

  /** The Float64 contact set of this solve in the KF/KU layout (CPU detection). */
  private uploadContacts(contacts: readonly ClothContactConstraint[], old: Float64Array): void {
    const n = this.n, K = contacts.length, layout = this.ensureContactCapacity(K);
    const kf = new Float32Array(Math.max(4, 16 * K)), cv = new Uint32Array(Math.max(4, 4 * K)), start = new Uint32Array(n + 1);
    for (const c of contacts) {
      if (c.vertices.length > 4 || c.vertices.length !== c.weights.length) throw new Error(`Cloth contact ${c.key} has more than four vertices`);
      for (const v of c.vertices) if (!((this.topology.kinematicFixed[v] ?? 0) & 1)) start[v + 1]!++;
    }
    for (let i = 0; i < n; i++) start[i + 1]! += start[i]!;
    const list = new Uint32Array(Math.max(4, start[n]!)), fill = start.slice(0, n);
    // g0 = n . sum w old - offset and P_t(obstacle) in Float64; increments in Float32 on the GPU.
    contacts.forEach((c, k) => {
      const nn = c.normal, o = c.obstacle, on = o[0]*nn[0] + o[1]*nn[1] + o[2]*nn[2];
      let g0 = -c.offset;
      c.vertices.forEach((v, j) => {
        g0 += c.weights[j]! * (nn[0]*old[3*v]! + nn[1]*old[3*v + 1]! + nn[2]*old[3*v + 2]!);
        cv[4*k + j] = v; kf[16*k + 12 + j] = c.weights[j]!;
        if (!((this.topology.kinematicFixed[v] ?? 0) & 1)) list[fill[v]!++] = k << 2 | j;
      });
      kf.set([nn[0], nn[1], nn[2], g0, o[0] - on*nn[0], o[1] - on*nn[1], o[2] - on*nn[2], c.lambda, c.rho, c.frictionLambda, c.friction, c.radius], 16*k);
    });
    this.writeWords(this.KF, kf, 0, 0, 16 * K);
    this.writeWords(this.KU, cv, layout.kuCV, 0, 4 * K);
    this.writeWords(this.KU, start, layout.kvStart);
    this.writeWords(this.KU, list, layout.kvList, 0, start[n]!);
  }

  private encodeUnit(pass: GPUComputePassEncoder, cgCap: number, pairs: number): void {
    const o = this.offsets, n = this.n, groups = (count: number) => Math.ceil(count / WG);
    const run = (entry: EntryPoint, count: number) => {if (count > 0) {pass.setPipeline(this.pipelines![entry]); pass.dispatchWorkgroups(count);}};
    run('elemTri', groups(o.triangles)); run('elemHinge', groups(o.hinges)); run('elemSeg', groups(o.segments));
    run('gradient', groups(n)); run('finishMeasure', 1);
    run('assemble', groups(this.blocks)); run('cgInit', groups(n)); run('finishInit', 1);
    for (let k = 0; k < cgCap; k++) {run('cgMatvec', groups(n)); run('finishAlpha', 1); run('cgUpdate', groups(n)); run('finishBeta', 1); run('cgDirection', groups(n));}
    run('slope', groups(n)); run('finishSlope', 1);
    for (let pass = 0; pass < LINE_PASSES; pass++) {
      run('lineTri', groups(o.triangles)); run('lineHinge', groups(o.hinges)); run('lineSeg', groups(o.segments)); run('lineVertex', groups(n)); run('linePair', groups(pairs)); run('lineContact', this.contactGroups);
      run('finishLine', 1);
    }
    run('apply', groups(n));
  }
  private run(pass: GPUComputePassEncoder, entry: EntryPoint, count: number): void {if (count > 0) {pass.setPipeline(this.pipelines![entry]); pass.dispatchWorkgroups(count);}}
  private strided(count: number): number {return Math.max(1, Math.min(4096, Math.ceil(count / WG)));}
  /** Swept-primitive grid and narrowphase (each kernel gates on K_DETECT). */
  private encodeDetection(pass: GPUComputePassEncoder): void {
    const o = this.offsets, layout = this.contactLayout!, edges = this.contactStatic!.edges, groups = (count: number) => Math.ceil(count / WG), self = this.selfCollision;
    if (self) {
      this.run(pass, 'gridClear', this.strided(2 * layout.gridBuckets)); this.run(pass, 'gridFaces', groups(o.triangles)); this.run(pass, 'gridEdges', groups(edges));
      this.run(pass, 'detectFaces', groups(this.n)); this.run(pass, 'detectEdges', groups(edges));
    }
    if (this.colliderCount) this.run(pass, 'detectColliders', this.strided(this.n * this.colliderCount));
  }
  /** Penalties of new pairs, friction lag, and the per-vertex lists when pairs were added. */
  private encodeAfterDetection(pass: GPUComputePassEncoder): void {
    const scanGroups = Math.ceil(this.n / SCAN_WG);
    this.run(pass, 'penalties', this.contactGroups);
    this.run(pass, 'listCount', this.contactGroups); this.run(pass, 'listScan1', scanGroups); this.run(pass, 'listScan2', 1); this.run(pass, 'listScan3', scanGroups);
    this.run(pass, 'listFill', this.contactGroups); this.run(pass, 'listReset', Math.ceil(this.n / WG));
  }
  private encodeRound(pass: GPUComputePassEncoder): void {
    this.run(pass, 'roundBegin', 1); this.run(pass, 'updateMultipliers', this.contactGroups); this.run(pass, 'relinearize', this.contactGroups);
    this.encodeDetection(pass);
    this.run(pass, 'roundDecide', 1);
    this.encodeAfterDetection(pass);
  }
  private selfCollision = false;

  /** Predicted increments, attachment targets and the hi/lo old positions in the fixed frame. */
  private uploadVertexBlock(state: ClothState, dt: number, options: ClothStepOptions): {old: Float64Array; predicted: Float64Array; vertex: Float32Array} {
    const topology = this.topology, n = this.n, old = state.positions_m.slice();
    const gravity = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.restPositions_m;
    if (!this.frame) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < old.length; i++) {const a = i % 3; if (old[i]! < lo[a]!) lo[a] = old[i]!; if (old[i]! > hi[a]!) hi[a] = old[i]!;}
      this.frame = lo.map((v, a) => Math.fround((v + hi[a]!) / 2)) as [number, number, number];
    }
    const origin = this.frame, vertex = new Float32Array(16 * n), predicted = new Float64Array(3 * n);
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
    return {old, predicted, vertex};
  }
  private writeParams(values: Record<string, number>): void {
    const words = new ArrayBuffer(4 * (PARAM_FIELDS.length + FLOAT_FIELDS.length)), u = new Uint32Array(words), f = new Float32Array(words);
    PARAM_FIELDS.forEach((name, k) => {u[k] = values[name] ?? 0;});
    FLOAT_FIELDS.forEach((name, k) => {f[PARAM_FIELDS.length + k] = values[name] ?? 0;});
    this.device.queue.writeBuffer(this.params, 0, words);
  }
  private contactParams(): Record<string, number> {
    const layout = this.contactLayout, statics = this.contactStatic;
    if (!layout) return {};
    const base = this.indexWords.length, shifted = statics ? Object.fromEntries(Object.entries(statics.offsets).map(([k, v]) => [k, base + v])) : {};
    return {...layout, ...(statics ? {...shifted, edges: statics.edges, cellSize: statics.cellSize, sCollider: base + statics.words.length} : {}), tableCur: this.tableCur, prevValid: this.prevValid ? 1 : 0, stiffnessValid: this.stiffnessValid ? 1 : 0};
  }
  private async readControl(): Promise<Uint32Array<ArrayBuffer>> {
    await this.statusRead.mapAsync(GPUMapMode.READ);
    const control = new Uint32Array(this.statusRead.getMappedRange().slice(0));
    this.statusRead.unmap();
    return control;
  }
  private adaptCG(control: Uint32Array): void {
    // Adapt the encoded CG depth to what the last solve used (inexact Newton
    // remains a descent method when the depth truncates CG).
    const cgUsed = control[C_CGMAX]!;
    this.cgCap = cgUsed >= this.cgCap ? Math.min(2048, 2 * this.cgCap) : Math.max(16, Math.min(this.cgCap, 8 * Math.ceil(1.5 * cgUsed / 8)));
  }
  private async readIncrements(): Promise<Float32Array> {
    await this.positionRead.mapAsync(GPUMapMode.READ);
    const increments = new Float32Array(this.positionRead.getMappedRange().slice(0));
    this.positionRead.unmap();
    return increments;
  }

  async step(state: ClothState, dt: number, options: ClothStepOptions = {}): Promise<GPUClothNewtonReport> {
    if (!this.pipelines || !this.bindings) throw new Error('Initialize GPU cloth Newton before stepping');
    if (!(dt > 0 && Number.isFinite(dt))) throw new Error('Invalid cloth GPU timestep');
    const started = performance.now(), topology = this.topology, n = this.n, oldTwist = state.twist_rad.slice();
    const {old, predicted, vertex} = this.uploadVertexBlock(state, dt, options);
    const target = options.supportPositions_m ?? topology.data.restPositions_m;
    const pairs = options.cohesionPairs ?? [], newtonCap = options.iterations ?? 64, tolerance = options.tolerance_m ?? 1e-8;
    if (!Number.isInteger(newtonCap) || newtonCap < 1) throw new Error('Cloth iterations must be a positive integer');
    const colliders = options.colliders ?? [], selfCollision = options.selfCollision ?? false;
    // Contact inside the minimization, as the CPU solve. GPU detection keeps
    // the outer iterations on the GPU; otherwise the Float64 set detects, and
    // its multipliers, tangent planes and detection update between GPU solves.
    const inSolve = (colliders.length > 0 || selfCollision) && !inSolveContactUnsupported(topology, colliders);
    let fallback = inSolve ? (this.contactDetection === 'cpu' ? 'CPU detection requested' : gpuContactUnsupported(topology, colliders)) : undefined;
    const x = predicted.slice();
    const positions = (increments: Float32Array) => {for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) x[3 * i + a] = old[3 * i + a]! + increments[4 * i + a]!;};
    let solve: SolveResult | undefined, contact_ms = 0, solves = 1, outer = 0, contactResidual = 0, contactReport: ClothContactReport | undefined;
    let newton = 0, linear = 0, submissions = 0;
    if (inSolve && !fallback) {
      const frictionEpsilon = FRICTION_SLIP_SPEED_M_S * dt;
      for (let attempt = 0; ; attempt++) {
        solve = await this.solveWithGPUContact(pairs, old, {newtonCap, tolerance, dt, frictionEpsilon, colliders, selfCollision});
        newton += solve.newton; linear += solve.linear; submissions += solve.submissions;
        const errors = solve.control[CONTACT_CONTROL.K_ERROR]!;
        if (!errors) break;
        if (errors & ~GROWABLE || attempt >= 3) {fallback = `GPU detection: ${describeErrors(errors)}`; solve = undefined; break;}
        // Too small for this contact set: grow from what the GPU counted, and redo the step.
        const c = solve.control, layout = this.contactLayout!;
        this.ensureContactCapacity(errors & E_CAPACITY ? pow2(1.5 * Math.max(c[CONTACT_CONTROL.K_COUNT]!, 2 * layout.kMax)) : layout.kMax,
          errors & E_GRID ? pow2(1.5 * Math.max(c[CONTACT_CONTROL.K_ENTRIES]!, 2 * layout.gridEntries)) : layout.gridEntries, errors & E_TABLE ? 4 : 2);
      }
      if (solve) {
        positions(solve.increments);
        const c = solve.control, f = (word: number) => new Float32Array(Uint32Array.of(c[word]!).buffer)[0]!;
        outer = c[CONTACT_CONTROL.K_OUTER]!; solves = 1 + c[CONTACT_CONTROL.K_SOLVES]!; contactResidual = f(CONTACT_CONTROL.K_RESID);
        contactReport = {iterations: outer + 1, residual_m: contactResidual, contacts: c[CONTACT_CONTROL.K_CONTACTS]!, selfContacts: c[CONTACT_CONTROL.K_SELF]!, maximumPenetration_m: f(CONTACT_CONTROL.K_REPPEN), reactions: []};
        solve = {...solve, converged: solve.converged && c[CONTACT_CONTROL.K_DONE]! !== 0 && contactResidual < tolerance};
      }
    }
    const contactSet = inSolve && fallback ? new ClothContactSet(topology, old, dt, colliders, selfCollision, tolerance, clothContactWarmStart(state)) : undefined;
    if (!solve) {
      if (contactSet) {
        const begun = performance.now();
        for (let i = 0; i < n; i++) if ((topology.kinematicFixed[i] ?? 0) & 1) for (let a = 0; a < 3; a++) x[3 * i + a] = target[3 * i + a]!;
        if (contactSet.detect(x)) contactSet.assignPenalties(this.stiffness ?? this.inertialStiffness(vertex));
        contact_ms += performance.now() - begun;
      }
      solve = await this.solveOnce(pairs, old, contactSet?.constraints, {first: true, newtonMin: 0, newtonCap, tolerance, dt, readStiffness: !!contactSet, frictionEpsilon: contactSet?.epsilon ?? 0});
      newton += solve.newton; linear += solve.linear; submissions += solve.submissions;
      positions(solve.increments);
      let previous = Infinity;
      if (contactSet && solve.converged) for (;; outer++) {
        const begun = performance.now();
        const {kkt_m, friction_m} = contactSet.updateMultipliers(x), penetration = contactSet.relinearize(x), added = contactSet.detect(x);
        contactResidual = Math.max(kkt_m, friction_m, penetration);
        const done = (!added && contactResidual < tolerance) || outer >= MAXIMUM_CONTACT_ITERATIONS;
        if (!done) {
          contactSet.assignPenalties(this.stiffness!);
          if (!added && contactResidual > .25 * previous) contactSet.stiffen(x, tolerance);
          previous = contactResidual;
          contactSet.lagFriction();
        }
        contact_ms += performance.now() - begun;
        if (done) break;
        solve = await this.solveOnce(pairs, old, contactSet.constraints, {first: false, newtonMin: 1, newtonCap, tolerance, dt, readStiffness: true, frictionEpsilon: contactSet.epsilon});
        newton += solve.newton; linear += solve.linear; submissions += solve.submissions; solves++;
        positions(solve.increments);
        if (!solve.converged) break;
      }
      if (contactSet) {
        solve = {...solve, converged: solve.converged && contactResidual < tolerance};
        if (solve.converged) contactReport = contactSet.report(x, outer + 1, contactResidual);
      }
    }
    this.stiffnessValid = true; this.lastControl = solve.control;
    const converged = solve.converged, solved = performance.now(), solve_ms = solved - started - contact_ms;
    state.positions_m.set(x);
    // Jointless rod twist couples to nothing but its inertia and attachment.
    for (let s = 0; s < state.twist_rad.length; s++) {
      const r = this.segmentRods[s]!, rod = topology.rods[r]!, j = s - rod.segmentOffset, a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
      if (((topology.kinematicFixed[a] ?? 0) | (topology.kinematicFixed[b] ?? 0)) & 2) {state.twist_rad[s] = oldTwist[s]!; continue;}
      const inertia = rodSegmentInertia_kg_m2(topology, state, r, j) / (dt * dt), spring = topology.attachmentTwist_N_m[s]!;
      state.twist_rad[s] = (inertia * (oldTwist[s]! + dt * state.angularVelocity_rad_s[s]!) + spring * state.attachmentTwistTarget_rad[s]!) / (inertia + spring);
    }
    const contactDetection = !inSolve ? 'none' as const : fallback ? 'cpu' as const : 'gpu' as const;
    const extra = {gpuSolves: solves, contact_ms, submissions, solve_ms, contactDetection, gpuStatus: solve.status, ...(fallback && inSolve ? {contactFallback: fallback} : {})};
    if (!converged) {
      // Float32 stalled (line search, cap), broke, or the contact iterations
      // did not settle: the Float64 Newton solve (with its own in-solve
      // contact) finishes from the GPU iterate when it is finite, else from the prediction.
      const finite = !(solve.status & (S_NONFINITE | S_SINGULAR)) && state.positions_m.every(Number.isFinite);
      const initialGuess = {positions_m: finite ? state.positions_m.slice() : predicted, twist_rad: state.twist_rad.slice()};
      state.positions_m.set(old); state.twist_rad.set(oldTwist);
      const refined = solveClothIncrement(topology, state, dt, options, initialGuess);
      return extendClothStepReport(refined, {...extra, iterations: newton + refined.iterations, gpuIterations: newton, refinementIterations: refined.iterations, newtonIterations: newton + refined.newtonIterations, linearIterations: linear + refined.linearIterations, finish_ms: performance.now() - solved});
    }
    if (!inSolve) {
      const finished = finishClothMechanicalStep(topology, state, old, oldTwist, dt, options, {iterations: newton, converged, maximumUpdate_m: solve.measure});
      return extendClothStepReport(finished, {...extra, gpuIterations: newton, refinementIterations: 0, newtonIterations: newton, linearIterations: linear, contactModel: 'post-projection' as const, finish_ms: performance.now() - solved});
    }
    if (contactSet) rememberClothContactMultipliers(state, contactSet);
    const cohesionReactions = beginClothMechanicalFinish(state, old, oldTwist, dt, options);
    const finished = completeClothMechanicalFinish(topology, state, old, oldTwist, dt, options, {iterations: newton, converged, maximumUpdate_m: Math.max(solve.measure, contactResidual)}, cohesionReactions, contactReport!);
    return extendClothStepReport(finished, {...extra, gpuIterations: newton, refinementIterations: 0, newtonIterations: newton, linearIterations: linear, contactModel: 'in-solve' as const, contactIterations: outer + 1, finish_ms: performance.now() - solved});
  }

  /** Mass/dt^2 + attachment per coordinate: the stiffness floor that sizes
   * contact penalties before any solve has read back the elastic one. */
  private inertialStiffness(vertex: Float32Array): Float64Array {
    const k = new Float64Array(3 * this.n);
    for (let i = 0; i < this.n; i++) k.fill(vertex[16 * i + 3]! + vertex[16 * i + 7]!, 3 * i, 3 * i + 3);
    return k;
  }

  private prepareGPUContact(colliders: readonly ClothCollider[], selfCollision: boolean): void {
    this.contactStatic ??= clothContactStatic(this.topology);
    this.ensureContactCapacity(Math.max(4096, 2 * this.n));
    this.uploadContactStatic(colliders);
    this.selfCollision = selfCollision;
  }

  /** The whole step with GPU detection: the first detection at the
   * prediction, then inner solves and outer rounds (multipliers, tangent
   * planes, re-detection, penalties) in the same submissions until the
   * contact residual settles. Reads back the control words only, then the
   * increments once. */
  private async solveWithGPUContact(pairs: Pairs, old: Float64Array, settings: {newtonCap: number; tolerance: number; dt: number; frictionEpsilon: number; colliders: readonly ClothCollider[]; selfCollision: boolean}): Promise<SolveResult> {
    const n = this.n, o = this.offsets, pairOffsets = this.uploadPairs(pairs, old);
    this.prepareGPUContact(settings.colliders, settings.selfCollision);
    const layout = this.contactLayout!, pairGroups = Math.ceil(pairs.length / WG);
    this.writeParams({...this.offsets, ...pairOffsets, ...this.contactParams(), pairs: pairs.length, pContact: o.pPair + pairGroups, partTotal: o.pPair + pairGroups + this.contactGroups,
      cgCap: this.cgCap, newtonCap: settings.newtonCap, dt: settings.dt, tolerance: settings.tolerance, contactDelta: settings.tolerance, frictionEpsilon: settings.frictionEpsilon,
      contactMode: 1, contactGroups: this.contactGroups, colliders: this.colliderCount, selfCollision: settings.selfCollision ? 1 : 0});
    this.device.queue.writeBuffer(this.W, 16 * o.wScal, new Float32Array(16));
    const groups = (count: number) => Math.ceil(count / WG), unitsPerSubmission = 2;
    let submissions = 0, control = new Uint32Array(C_WORDS);
    const K = CONTACT_CONTROL;
    for (;;) {
      const encoder = this.device.createCommandEncoder({label: 'Cloth Newton-PCG with GPU contact'});
      if (!submissions) {encoder.clearBuffer(this.C); encoder.clearBuffer(this.KU, 4 * layout.kvStart, 4 * (n + 1));}
      const pass = encoder.beginComputePass({label: 'Cloth Newton iterations and contact rounds'});
      pass.setBindGroup(0, this.bindings!);
      for (let k = 0; k < unitsPerSubmission; k++) {
        if (!submissions && !k) {
          for (const [entry, count] of [['initX', n], ['prepTri', o.triangles], ['prepHinge', o.hinges], ['prepSeg', o.segments]] as const) this.run(pass, entry, groups(count));
          this.run(pass, 'detectBegin', 1); this.run(pass, 'tableClear', this.strided(2 * layout.tableSlots));
          this.encodeDetection(pass); this.encodeAfterDetection(pass);
        } else this.encodeRound(pass);
        this.encodeUnit(pass, this.cgCap, pairs.length);
      }
      pass.end();
      encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
      this.device.queue.submit([encoder.finish()]);
      submissions++;
      control = await this.readControl();
      const status = control[C_STATUS]!;
      if (control[K.K_ERROR] || control[K.K_DONE] || (status && status !== S_CONVERGED)) break;
    }
    const success = control[K.K_DONE] !== 0 && !control[K.K_ERROR] && control[C_STATUS] === S_CONVERGED;
    const encoder = this.device.createCommandEncoder({label: 'Cloth Newton result'});
    if (success) {const pass = encoder.beginComputePass(); pass.setBindGroup(0, this.bindings!); this.run(pass, 'rememberMultipliers', this.contactGroups); pass.end();}
    encoder.copyBufferToBuffer(this.W, 16 * o.wX, this.positionRead, 0, 16 * n);
    this.device.queue.submit([encoder.finish()]);
    const increments = await this.readIncrements();
    // The table now holds this step's multipliers; the next step reads it as the previous one.
    if (success) {this.tableCur = 1 - this.tableCur; this.prevValid = true;} else this.prevValid = false;
    this.adaptCG(control);
    const measure = new Float32Array(Uint32Array.of(control[C_LAST]!).buffer)[0]!;
    return {converged: success, status: control[C_STATUS]!, measure, newton: control[C_NEWTON_TOTAL]!, linear: control[C_LINEAR]!, submissions: submissions + 1, increments, control};
  }

  /** One GPU Newton-PCG solve of the current potential (no contact, or the
   * CPU-detected set). The first solve of a step starts from the prediction;
   * later ones resume from the GPU iterate after a contact update and take at
   * least newtonMin steps. */
  private async solveOnce(pairs: Pairs, old: Float64Array, contacts: readonly ClothContactConstraint[] | undefined,
    settings: {first: boolean; newtonMin: number; newtonCap: number; tolerance: number; dt: number; readStiffness: boolean; frictionEpsilon: number}): Promise<SolveResult> {
    const n = this.n, o = this.offsets, pairOffsets = this.uploadPairs(pairs, old);
    if (contacts) this.uploadContacts(contacts, old);
    const pairGroups = Math.ceil(pairs.length / WG);
    this.writeParams({...this.offsets, ...pairOffsets, ...(contacts ? this.contactParams() : {}), pairs: pairs.length, pContact: o.pPair + pairGroups, partTotal: o.pPair + pairGroups + this.contactGroups,
      cgCap: this.cgCap, newtonCap: settings.newtonCap, dt: settings.dt, tolerance: settings.tolerance, contactDelta: settings.tolerance, frictionEpsilon: settings.frictionEpsilon,
      contactMode: contacts ? 2 : 0, uploadedContacts: contacts?.length ?? 0, contactGroups: this.contactGroups});
    this.device.queue.writeBuffer(this.W, 16 * o.wScal, new Float32Array(16));
    const groups = (count: number) => Math.ceil(count / WG), unitsPerSubmission = 2;
    let submissions = 0, status = 0, control = new Uint32Array(C_WORDS);
    while (true) {
      const encoder = this.device.createCommandEncoder({label: 'Cloth Newton-PCG'});
      // Fresh control words, with the Newton minimum of a resumed solve.
      if (!submissions) {const words = new Uint32Array(C_WORDS); words[C_NEWTONMIN] = settings.newtonMin; this.device.queue.writeBuffer(this.C, 0, words);}
      const pass = encoder.beginComputePass({label: 'Cloth Newton iterations'});
      pass.setBindGroup(0, this.bindings!);
      if (!submissions && settings.first) for (const [entry, count] of [['initX', n], ['prepTri', o.triangles], ['prepHinge', o.hinges], ['prepSeg', o.segments]] as const) this.run(pass, entry, groups(count));
      for (let k = 0; k < unitsPerSubmission; k++) this.encodeUnit(pass, this.cgCap, pairs.length);
      pass.end();
      encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
      encoder.copyBufferToBuffer(this.W, 16 * o.wX, this.positionRead, 0, 16 * n);
      if (settings.readStiffness) encoder.copyBufferToBuffer(this.W, 16 * o.wK, this.stiffnessRead, 0, 16 * n);
      this.device.queue.submit([encoder.finish()]);
      submissions++;
      control = await this.readControl();
      status = control[C_STATUS]!;
      if (status) break;
    }
    this.adaptCG(control);
    const measure = new Float32Array(Uint32Array.of(control[C_LAST]!).buffer)[0]!;
    const increments = await this.readIncrements();
    if (settings.readStiffness) {
      await this.stiffnessRead.mapAsync(GPUMapMode.READ);
      const k = new Float32Array(this.stiffnessRead.getMappedRange().slice(0));
      this.stiffnessRead.unmap();
      const stiffness = this.stiffness ??= new Float64Array(3 * n);
      for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) stiffness[3 * i + a] = k[4 * i + a]!;
    }
    return {converged: (status & S_CONVERGED) !== 0, status, measure, newton: control[C_NEWTON]!, linear: control[C_LINEAR]!, submissions, increments, control};
  }

  /** Diagnostics: the GPU's first detection of a step from `state` (at the
   * prediction), without solving. The table and multipliers of the previous
   * step are left intact. */
  async detectContacts(state: ClothState, dt: number, options: ClothStepOptions = {}): Promise<{contacts: GPUClothContact[]; detect_ms: number; errors: number}> {
    if (!this.pipelines || !this.bindings) throw new Error('Initialize GPU cloth Newton before detecting');
    const colliders = options.colliders ?? [], unsupported = gpuContactUnsupported(this.topology, colliders);
    if (unsupported) throw new Error(`GPU cloth contact: ${unsupported}`);
    this.uploadVertexBlock(state, dt, options);
    this.prepareGPUContact(colliders, options.selfCollision ?? false);
    const n = this.n, layout = this.contactLayout!;
    this.writeParams({...this.offsets, ...this.contactParams(), contactMode: 1, contactGroups: this.contactGroups, colliders: this.colliderCount, selfCollision: this.selfCollision ? 1 : 0,
      dt, tolerance: options.tolerance_m ?? 1e-8, contactDelta: options.tolerance_m ?? 1e-8, frictionEpsilon: FRICTION_SLIP_SPEED_M_S * dt, newtonCap: 1, cgCap: 1});
    const begun = performance.now(), encoder = this.device.createCommandEncoder({label: 'Cloth contact detection'});
    encoder.clearBuffer(this.C); encoder.clearBuffer(this.KU, 4 * layout.kvStart, 4 * (n + 1));
    const pass = encoder.beginComputePass(); pass.setBindGroup(0, this.bindings!);
    this.run(pass, 'initX', Math.ceil(n / WG)); this.run(pass, 'detectBegin', 1); this.run(pass, 'tableClear', this.strided(2 * layout.tableSlots));
    this.encodeDetection(pass); this.encodeAfterDetection(pass);
    pass.end();
    encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
    this.device.queue.submit([encoder.finish()]);
    const control = await this.readControl(), detect_ms = performance.now() - begun;
    return {contacts: await this.readContactSet(Math.min(control[CONTACT_CONTROL.K_COUNT]!, layout.kMax)), detect_ms, errors: control[CONTACT_CONTROL.K_ERROR]!};
  }
  /** Diagnostics: the GPU contact set after the last step (or detection). */
  async contacts(): Promise<GPUClothContact[]> {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
    this.device.queue.submit([encoder.finish()]);
    const control = await this.readControl();
    return this.readContactSet(Math.min(control[CONTACT_CONTROL.K_COUNT]!, this.contactLayout?.kMax ?? 0));
  }
  private async readContactSet(count: number): Promise<GPUClothContact[]> {
    if (!count) return [];
    const layout = this.contactLayout!, floats = this.device.createBuffer({size: 64 * count, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}), words = this.device.createBuffer({size: 32 * count, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.KF, 0, floats, 0, 64 * count);
      encoder.copyBufferToBuffer(this.KU, 4 * layout.kuCV, words, 0, 16 * count);
      encoder.copyBufferToBuffer(this.KU, 4 * layout.kuMeta, words, 16 * count, 16 * count);
      this.device.queue.submit([encoder.finish()]);
      await Promise.all([floats.mapAsync(GPUMapMode.READ), words.mapAsync(GPUMapMode.READ)]);
      const f = new Float32Array(floats.getMappedRange().slice(0)), u = new Uint32Array(words.getMappedRange().slice(0));
      return Array.from({length: count}, (_, c) => {
        const k0 = u[4 * count + 4 * c]!, k1 = u[4 * count + 4 * c + 1]!, kind = k0 >>> 30, a = k0 & 0x3fffffff;
        const key = kind === 0 ? `c:${a}:${k1}` : kind === 1 ? `v:${a}:${k1}` : `e:${a}:${k1}`, size = kind === 0 ? 1 : 4;
        return {key, vertices: Array.from(u.subarray(4 * c, 4 * c + size)), weights: Array.from(f.subarray(16 * c + 12, 16 * c + 12 + size)), normal: [f[16 * c]!, f[16 * c + 1]!, f[16 * c + 2]!] as const,
          lambda: f[16 * c + 7]!, rho: f[16 * c + 8]!, frictionLambda: f[16 * c + 9]!};
      });
    } finally {floats.destroy(); words.destroy();}
  }

  destroy(): void {for (const buffer of this.buffers) buffer.destroy();}
}
