import { beginClothMechanicalFinish, completeClothMechanicalFinish, extendClothStepReport, finishClothMechanicalStep, rodJoint, type ClothStepOptions, type ClothStepReport } from './dynamics';
import { ClothContactSet, clothContactWarmStart, FRICTION_SLIP_SPEED_M_S, inSolveContactUnsupported, MAXIMUM_CONTACT_ITERATIONS, rememberClothContactMultipliers, type ClothContactConstraint } from './contact-constraints';
import { rodRigidity, rodSegmentInertia_kg_m2 } from './material';
import { CONTACT_CONTROL, CONTACT_CONTROL_WORDS, CONTACT_GRID_CELL_BOUND, E_CAPACITY, GRID_CELL_EPS, E_CELLS, E_CROSSED, E_GRID, E_BOX_NONFINITE, E_NONFINITE, E_STIFFNESS, E_TABLE, E_UNSUPPORTED, SCAN_WG, clothContactCPURemnant, clothContactStatic, clothContactWGSL, gpuContactUnsupported, packClothColliders, type ClothContactStatic } from './gpu-contact';
import type { ClothCollider, ClothContactReport } from './contact';
import type { PreparedClothTopology } from './topology';
import type { ClothState } from './types';

/** GPU Newton-PCG for the implicit-Euler incremental potential that
 * lib/cloth/nonlinear-solve.ts minimizes in Float64: inertia, finite
 * attachments, cohesion, Green-strain membrane and dihedral hinge energies,
 * rod stretch, and discrete-elastic-rod joints (material-frame bending and
 * twist, with each segment's twist an unknown), each with its Kelvin-Voigt term.
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
 * (its pairs change every step; the pattern does not). A solve that reaches
 * the Newton cap resumes on the GPU from its iterate (NEWTON_RESUMES more
 * caps); one that still does not converge fails the step with its GPU status.
 * Nothing falls back to the Float64 CPU solve, which stays the test oracle.
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
 * DER joints follow the Float64 Hessian model: rod bending enters the
 * diagonal (max(0, d2E/dx_k^2) per coordinate, from exact forward-mode second
 * derivatives along each coordinate), and twist couples only along its rod
 * (a tridiagonal solve per rod after the PCG); the line search is over
 * positions and twist together. */

const WG = 64;
const ALPHAS = 8;
// Armijo backtracking: LINE_PASSES gated passes of ALPHAS halvings each, the
// Float64 solve's 40 (a pass runs only while no earlier step size was accepted).
const LINE_PASSES = 5;
const MAX_UNITS = 48;
/** One workgroup runs the whole CG loop up to this size (launches dominate below it). */
const FUSED_CG_MAX_VERTICES = 4096;
/** Workgroups of the per-contact A p pass (grid-strided over the live contacts). */
const CONTACT_FORCE_GROUPS = 1024;
/** Up to this size the direction lives in workgroup memory (16 B a vertex of the 32 KiB). */
const SHARED_CG_MAX_VERTICES = 1024, SHARED_CG_PER_THREAD = SHARED_CG_MAX_VERTICES / WG;
// A solve at the Newton cap continues from its iterate for this many more caps.
const NEWTON_RESUMES = 3;
// Control words (atomic u32).
const C_MEASURE = 0, C_ACTIVE = 1, C_CGACTIVE = 2, C_CGCOUNT = 3, C_NEWTON = 4, C_STATUS = 5, C_STOPAFTER = 6, C_USEZ0 = 7, C_UNIT = 8, C_LAST = 9,
  C_NEWTONMIN = 10, C_NEWTON_TOTAL = 11, C_LINEAR = 12, C_CGMAX = 13, C_SEARCH = 14, C_WORKED = 15, C_LOG = 16, LOG_UNITS = 16, C_WORDS = CONTACT_CONTROL_WORDS;
const S_CONVERGED = 1, S_LINE = 2, S_NONFINITE = 4, S_CAP = 8, S_SINGULAR = 16;
const KIND_TRI = 0, KIND_HINGE = 1, KIND_SEG = 2, KIND_JOINT = 3, NONE = 0xffffffff;

/** Every prepared topology: shells, rod stretch, and DER rod joints (bending, twist). */
export function gpuClothNewtonSupported(topology: PreparedClothTopology): boolean {
  return topology.rods.every(rod => rod.vertices.length >= 2 && rod.restLengths_m.length === rod.vertices.length - 1);
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
  // DER rod joints and segment twist.
  'joints', 'rods', 'iJoint', 'iRod', 'iSegJoint', 'dTwist', 'dJoint', 'wT', 'wT2', 'eJoint', 'pJoint', 'pTwist',
  // Old strains from rest: per-step displacements, static Float64 rest frames.
  'dDisp', 'dTriRest', 'dSegRest',
  // Workgroups (and partial slots) of the contact line search: one for a small mesh's fused unit.
  'lineGroups',
  // Per-contact Hessian coefficients (one Newton linearization) and the contact term of A p (one CG iteration).
  'kfCoef', 'kfForce',
  // Hinge old edges: static Float64 rest edges, and per step rest + displacement (3 per hinge).
  'dHingeRest', 'eHingeEdge',
  // Rod contact (round, uncut rods; lib/cloth/gpu-contact.ts): per-contact records, list rows n + segments, rod statics.
  'rodContact', 'kuRod', 'kfRod', 'kfRodForce', 'kfRodArm', 'kfSegBox', 'listRows', 'sSegInfo', 'sNbStart', 'sNbList', 'sRpXStart', 'sRpXList', 'sRsXStart', 'sRsXList',
] as const;
const FLOAT_FIELDS = ['dt', 'tolerance', 'contactDelta', 'frictionEpsilon', 'cellSize'] as const;
type ParamName = typeof PARAM_FIELDS[number];

const clothNewtonKernels = /* wgsl */ `
// Read at run time: a NaN or infinite literal is a const-expression error.
var<private> NAN_BITS:u32=0x7fc00000u;
var<private> INF_BITS:u32=0x7f800000u;
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
// The old frame's displacement from rest (rounded once from Float64).
fn restDisplacement(v:u32)->vec3f{return D[p.dDisp+v].xyz;}
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
 // Old Green strain from the rest displacement: the rest frame ur,wr is
 // orthonormal (Float64), so e = sym(ur.du) + du.du/2 has no rounded rest term.
 let ur=D[p.dTriRest+2u*t].xyz;let wr=D[p.dTriRest+2u*t+1u].xyz;let da=restDisplacement(id.x);let d1=restDisplacement(id.y)-da;let d2=restDisplacement(id.z)-da;
 let du=ru.y*d1+ru.z*d2;let dw=rv.y*d1+rv.z*d2;let u=ur+du;let w=wr+dw;
 E[p.eTriOld+3u*t]=vec4f(u,dot(ur,du)+.5*dot(du,du));E[p.eTriOld+3u*t+1u]=vec4f(w,dot(wr,dw)+.5*dot(dw,dw));E[p.eTriOld+3u*t+2u]=vec4f(dot(ur,dw)+dot(du,wr)+dot(du,dw),0,0,0);
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
// Old edges as the Float64 rest edge plus the displacement since rest, stored once a
// step: an edge evaluated in a Newton iterate or a line search is then a loaded old
// edge plus an increment difference. Built from the hi/lo frame positions, the compiler
// reassociates (hi_b-hi_a)+(lo_b-lo_a)+(x_b-x_a) and every evaluation rounds at the
// frame-relative position scale (1.5e-8 m at 0.13 m), a residual floor above 1e-8 m.
fn hingeOldEdge(h:u32,k:u32)->vec3f{return E[p.eHingeEdge+3u*h+k].xyz;}
@compute @workgroup_size(${WG}) fn prepHinge(@builtin(global_invocation_id) g:vec3u){
 let h=g.x;if(h>=p.hinges){return;}let id=hingeIds(h);let s=D[p.dHinge+h];let mode=mat(u32(s.z),2u).x;
 let da=restDisplacement(id.x);
 let e=D[p.dHingeRest+3u*h].xyz+(restDisplacement(id.y)-da);let u=D[p.dHingeRest+3u*h+1u].xyz+(restDisplacement(id.z)-da);let w=D[p.dHingeRest+3u*h+2u].xyz+(restDisplacement(id.w)-da);
 E[p.eHingeEdge+3u*h]=vec4f(e,0);E[p.eHingeEdge+3u*h+1u]=vec4f(u,0);E[p.eHingeEdge+3u*h+2u]=vec4f(w,0);
 let angle=dihedral(e,u,w);
 E[p.eHingeOld+h]=vec4f(bendingPhi(angle,mode).x,bendingPhi(s.x,mode).x,angle,0);
}
struct HingeAt{e:vec3f,u:vec3f,w:vec3f}
fn hingeEdges(h:u32,xa:vec3f,xb:vec3f,xc:vec3f,xd:vec3f)->HingeAt{return HingeAt(hingeOldEdge(h,0u)+(xb-xa),hingeOldEdge(h,1u)+(xc-xa),hingeOldEdge(h,2u)+(xd-xa));}
struct HingeMaterial{k:f32,gamma:f32,mode:f32}
fn hingeMaterial(h:u32,id:vec4u)->HingeMaterial{let s=D[p.dHinge+h];let m=u32(s.z);return HingeMaterial(mat(m,0u).y*fractionScale(m,.5*(lo(id.x).w+lo(id.y).w))*s.y,mat(m,1u).z/p.dt,mat(m,2u).x);}
@compute @workgroup_size(${WG}) fn elemHinge(@builtin(global_invocation_id) g:vec3u){
 let h=g.x;if(h>=p.hinges||done()){return;}let id=hingeIds(h);let at=hingeEdges(h,xv(id.x),xv(id.y),xv(id.z),xv(id.w));let mm=hingeMaterial(h,id);
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
@compute @workgroup_size(${WG}) fn prepSeg(@builtin(global_invocation_id) g:vec3u){
 let s=g.x;if(s>=p.segments){return;}let id=segIds(s);
 // l-L=(|r+d|^2-L^2)/(l+L) with |r|=L in Float64: the stretch keeps the precision of d.
 // The old edge r+d is stored: evaluations add increments to it (see hingeOldEdge).
 let r=D[p.dSegRest+s].xyz;let d=restDisplacement(id.y)-restDisplacement(id.x);let rest=D[p.dSeg+s].x;let e=r+d;let l=length(e);
 E[p.eSegOld+2u*s]=vec4f(l,(2.*dot(r,d)+dot(d,d))/(l+rest),0,0);E[p.eSegOld+2u*s+1u]=vec4f(e,0);
}
fn segOldEdge(s:u32)->vec3f{return E[p.eSegOld+2u*s+1u].xyz;}
// Length l and its increment l-l_old from the increment.
fn segLength(s:u32,delta:vec3f,oldLength:f32)->vec4f{let e0=segOldEdge(s);let e=e0+delta;let l=length(e);return vec4f(e,(2.*dot(e0,delta)+dot(delta,delta))/(l+oldLength));}
@compute @workgroup_size(${WG}) fn elemSeg(@builtin(global_invocation_id) g:vec3u){
 let s=g.x;if(s>=p.segments||done()){return;}let id=segIds(s);let oldLength=E[p.eSegOld+2u*s].x;let mm=segMaterial(s,id,oldLength);
 let q=segLength(s,xv(id.y)-xv(id.x),oldLength);let l=oldLength+q.w;let strain=E[p.eSegOld+2u*s].y+q.w+mm.gamma*q.w;
 E[p.eSeg+2u*s]=vec4f(q.xyz,l);E[p.eSeg+2u*s+1u]=vec4f(mm.factor,mm.gamma,strain,0);
}

// ---- cohesion pairs: Q[4k..]=(delta0.xyz,F); entries QI[qiPairVertex+j], Q[qW+j]
fn pairDelta(k:u32,useDir:bool)->vec3f{var d=vec3f(0);if(!useDir){d=vec3f(Q[4u*k],Q[4u*k+1u],Q[4u*k+2u]);}for(var j=QI[p.qiPairStart+k];j<QI[p.qiPairStart+k+1u];j++){let v=QI[p.qiPairVertex+j];if(useDir){d+=Q[p.qW+j]*dir(v);}else{d+=Q[p.qW+j]*xv(v);}}return d;}

// ---- contacts of lib/cloth/contact-constraints.ts: g = g0 + n.sum w u, slip =
// P_t(sum w u) - P_t(obstacle), smoothed augmented-Lagrangian penalty of
// m = lambda/rho - g and lagged smoothed friction mu lambda_f f0(|slip|).
// KF[4c..]=(n,g0),(P_t obstacle,lambda),(rho,lambda_f,mu,radius),(w0..w3);
// KU[kuCV+4c..] vertices; free vertices' lists KU[kvStart+i..] -> KU[kvList+k]=c<<3|j.
// Rod pairs (p.rodContact, lib/cloth/gpu-contact.ts) add a fifth slot (KU[kuRod+4c], KF[kfRod+3c+2].x)
// and rodm = KU[kuRod+4c+3] (0 off rods): slots, the sides' endpoint slots, bit 15 a second side.
struct Contact{n:vec3f,g0:f32,t:vec3f,lambda:f32,rho:f32,mu:f32,w:vec4f,v:vec4u,v4:u32,w4:f32,rodm:u32}
fn contactAt(c:u32)->Contact{
 let a=KF[4u*c];let t=KF[4u*c+1u];let r=KF[4u*c+2u];let j=p.kuCV+4u*c;var rodm=0u;var v4=0u;var w4=0.;
 if(p.rodContact!=0u){rodm=KU[p.kuRod+4u*c+3u];if(rodm!=0u){v4=KU[p.kuRod+4u*c];w4=KF[p.kfRod+3u*c+2u].x;}}
 return Contact(a.xyz,a.w,t.xyz,t.w,r.x,r.z*r.y,KF[4u*c+3u],vec4u(KU[j],KU[j+1u],KU[j+2u],KU[j+3u]),v4,w4,rodm);
}
fn slotV(c:Contact,j:u32)->u32{if(j<4u){return c.v[j];}return c.v4;}
fn slotW(c:Contact,j:u32)->f32{if(j<4u){return c.w[j];}return c.w4;}
// sum_j w_j of increments (0), search direction (1) or CG direction (2).
fn contactSum(c:Contact,which:u32)->vec3f{var s=vec3f(0);for(var j=0u;j<5u;j++){let w=slotW(c,j);if(w!=0.){let v=slotV(c,j);if(which==0u){s+=w*xv(v);}else if(which==1u){s+=w*dir(v);}else{s+=w*W[p.wP+v].xyz;}}}return s;}
// A rod pair's material-point Jacobian (linearizeRolling of ClothContactSet): J_j = w_j I + sum
// over its sides of (+C at the side's b slot, -C at its a slot), C = (r/L)(t m^T - (t.m) I) with m
// the side's arm over r (the section's support in n over r: n itself for a round section), and
// each side's twist rolls the material point by -r theta (t x m). The gap's linear part is
// n.(sum J u + roll): rows J_j^T n (w_j n for a round section, n^T C = 0) and the twist rows
// -r n.(t x m) (zero for a round section).
// KF[kfRod+3c..] = (t0, r0/L0), (t1, r1/L1), (w4, r0, r1, thickness); KF[kfRodArm+2c..] = (m0, twist
// row 0), (m1, twist row 1); segments KU[kuRod+4c+1..2].
struct RodGeom{t0:vec4f,t1:vec4f,r:vec2f,two:bool,a:vec2u,b:vec2u,m0:vec4f,m1:vec4f}
fn rodAt(c:u32,k:Contact)->RodGeom{
 let m=k.rodm;
 return RodGeom(KF[p.kfRod+3u*c],KF[p.kfRod+3u*c+1u],KF[p.kfRod+3u*c+2u].yz,((m>>15u)&1u)!=0u,vec2u((m>>3u)&7u,(m>>9u)&7u),vec2u((m>>6u)&7u,(m>>12u)&7u),KF[p.kfRodArm+2u*c],KF[p.kfRodArm+2u*c+1u]);
}
fn rodC(m:vec3f,t:vec4f)->mat3x3f{let tm=dot(t.xyz,m);return t.w*(outer(t.xyz,m)-mat3x3f(vec3f(tm,0,0),vec3f(0,tm,0),vec3f(0,0,tm)));}
fn rodCv(m:vec3f,t:vec4f,v:vec3f)->vec3f{return t.w*(t.xyz*dot(m,v)-dot(t.xyz,m)*v);}
// C^T v: a side's share of J^T at its b slot (minus at a).
fn rodCt(m:vec3f,t:vec4f,v:vec3f)->vec3f{return t.w*(m*dot(t.xyz,v)-dot(t.xyz,m)*v);}
fn rodJ(c:Contact,g:RodGeom,j:u32)->mat3x3f{
 let w=slotW(c,j);var M=mat3x3f(vec3f(w,0,0),vec3f(0,w,0),vec3f(0,0,w));
 if(j==g.b.x){M+=rodC(g.m0.xyz,g.t0);}if(j==g.a.x){M-=rodC(g.m0.xyz,g.t0);}
 if(g.two){if(j==g.b.y){M+=rodC(g.m1.xyz,g.t1);}if(j==g.a.y){M-=rodC(g.m1.xyz,g.t1);}}
 return M;
}
// A side's roll per unit twist, -r (t x m): its twist row is n.roll.
fn rodRoll(g:RodGeom,side:u32)->vec3f{if(side==1u){return -g.r.y*cross(g.t1.xyz,g.m1.xyz);}return -g.r.x*cross(g.t0.xyz,g.m0.xyz);}
fn rodRole(rodm:u32,j:u32,side:u32)->f32{
 if(side==1u&&((rodm>>15u)&1u)==0u){return 0.;}
 let a=(rodm>>(3u+6u*side))&7u;let b=(rodm>>(6u+6u*side))&7u;return select(0.,1.,j==b)-select(0.,1.,j==a);
}
// Slot motions: increments (0), search direction (1), CG direction (2).
fn rodSlotU(c:Contact,which:u32)->array<vec3f,5>{
 var u:array<vec3f,5>;
 for(var j=0u;j<(c.rodm&7u);j++){let v=slotV(c,j);if(which==0u){u[j]=xv(v);}else if(which==1u){u[j]=dir(v);}else{u[j]=W[p.wP+v].xyz;}}
 return u;
}
// The sides' twist: the increment (0), the twist step (1) or the CG direction's twist rows (2).
fn rodTwist(ci:u32,which:u32)->vec2f{
 var t=vec2f(0);if(which==2u&&!twistCG()){return t;}
 for(var side=0u;side<2u;side++){let s=KU[p.kuRod+4u*ci+1u+side];if(s!=NONE){if(which==2u){t[side]=W[p.wP+p.n+s].x;}else{t[side]=select(W[p.wT+s].x,W[p.wT+s].z,which==1u);}}}
 return t;
}
// With rod contact the twist joins the CG (rows n+s after the vertices): friction
// rolls the rod, coupling twist and positions too stiffly for separate block steps.
fn twistCG()->bool{return p.contactMode!=0u&&p.rodContact!=0u;}
// sum_j J_j u_j plus the twist roll (before the tangent projection).
fn rodMove(c:Contact,g:RodGeom,uIn:array<vec3f,5>,twist:vec2f)->vec3f{
 var u=uIn;var s=vec3f(0);for(var j=0u;j<(c.rodm&7u);j++){s+=slotW(c,j)*u[j];}
 s+=rodCv(g.m0.xyz,g.t0,u[g.b.x]-u[g.a.x])+twist.x*rodRoll(g,0u);
 if(g.two){s+=rodCv(g.m1.xyz,g.t1,u[g.b.y]-u[g.a.y])+twist.y*rodRoll(g,1u);}
 return s;
}
// The slip at the iterate: P_t(sum J u + roll) - P_t(obstacle) for a rod pair, else P_t(sum w u) - P_t(obstacle).
fn contactSlip(ci:u32,c:Contact,sx:vec3f)->vec3f{
 if(c.rodm!=0u){return tangent(c.n,rodMove(c,rodAt(ci,c),rodSlotU(c,0u),rodTwist(ci,0u)))-c.t;}
 return tangent(c.n,sx)-c.t;
}
fn contactM(c:Contact,sx:vec3f)->f32{return c.lambda/c.rho-(c.g0+dot(c.n,sx));}
// The gap's motion sum at the iterate: sum w u, or for a rod pair sum J u plus the roll (n.this is the gap's linear part).
fn contactMotion(ci:u32,c:Contact,sx:vec3f)->vec3f{if(c.rodm!=0u){return rodMove(c,rodAt(ci,c),rodSlotU(c,0u),rodTwist(ci,0u));}return sx;}
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
fn contactCoefficients(ci:u32,c:Contact)->vec2f{let sx=contactSum(c,0u);var f=0.;if(c.mu>0.){f=c.mu*fF1(length(contactSlip(ci,c,sx)));}return vec2f(cCurv(contactM(c,contactMotion(ci,c,sx)),c.rho),f);}

@compute @workgroup_size(${WG}) fn initX(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i<p.segments){W[p.wT+i]=vec4f(select(0.,segT(i,2u).w,twistFree(i)),0,0,1);W[p.wT2+i]=vec4f(1,1,0,0);}if(i>=p.n){return;}if(isFixed(i)){W[p.wX+i]=vec4f(attach(i).xyz,0);}else{W[p.wX+i]=vec4f(pred(i).xyz,0);}}

// ---- explicit predictor (nonFluidClothPredictor of lib/cloth/dynamics.ts):
// v + dt (g + (f - dE/dx)/m) at the accepted positions (zero increments), with
// the predicted increment dt v + dt^2 (g + f/m) and m/dt^2 of the vertex block.
@compute @workgroup_size(${WG}) fn zeroX(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i<p.segments){W[p.wT+i]=vec4f(0,0,0,1);W[p.wT2+i]=vec4f(1,1,0,0);}if(i<p.n){W[p.wX+i]=vec4f(0);}}
@compute @workgroup_size(${WG}) fn predictVelocity(@builtin(global_invocation_id) g:vec3u){
 let i=g.x;if(i>=p.n){return;}let pr=pred(i);let at=attach(i);
 if(isFixed(i)){W[p.wS+i]=vec4f(at.xyz/p.dt,0);return;}
 let gr=elasticGradient(i)[0]-at.w*at.xyz;
 W[p.wS+i]=vec4f((pr.xyz-gr/pr.w)/p.dt,0);
}

// ---- gradient and convergence measure
// Elastic and cohesion gradient of vertex i at the iterate, with the
// diagonal of the Hessian model (inertia, attachment and contact excluded).
fn elasticGradient(i:u32)->array<vec3f,2>{
 var gr=vec3f(0);var diag=vec3f(0);
   for(var k=I[p.iVStart+i];k<I[p.iVStart+i+1u];k++){
    let code=I[p.iVList+k];let el=code&0xffffffu;let local=(code>>24u)&3u;let kind=code>>28u;
    if(kind==${KIND_TRI}u){
     let s0=E[p.eTri+4u*el];let s1=E[p.eTri+4u*el+1u];let s2=E[p.eTri+4u*el+2u];let s3=E[p.eTri+4u*el+3u];
     let ru=D[p.dTri+2u*el][local];let rv=D[p.dTri+2u*el+1u][local];let a=ru*s0.xyz;let b=rv*s1.xyz;let c=ru*s1.xyz+rv*s0.xyz;let nu=s3.z;
     gr+=s0.w*a+s1.w*b+s2.w*c;diag+=s3.x*(1.+s3.y)*(a*a+b*b+2.*nu*a*b+.5*(1.-nu)*c*c)+vec3f(s0.w*ru*ru+s1.w*rv*rv+2.*s2.w*ru*rv);
    }else if(kind==${KIND_HINGE}u){
     let gp=E[p.eHinge+9u*el+2u*local].xyz;let kk=E[p.eHinge+9u*el+2u*local+1u].xyz;let ab=E[p.eHinge+9u*el+8u];
     gr+=ab.y*gp;diag+=ab.x*gp*gp+ab.y*kk;
    }else if(kind==${KIND_SEG}u){
     let s0=E[p.eSeg+2u*el];let s1=E[p.eSeg+2u*el+1u];let d=s0.xyz/s0.w;let sign=select(-1.,1.,local==1u);
     gr+=s1.x*s1.z*d*sign;diag+=s1.x*((1.+s1.y)*d*d+s1.z/s0.w*(1.-d*d));
    }else{
     gr+=E[p.eJoint+9u*el+2u*local].xyz;diag+=E[p.eJoint+9u*el+2u*local+1u].xyz;
    }
   }
   for(var k=QI[p.qiVStart+i];k<QI[p.qiVStart+i+1u];k++){
    let pair=QI[p.qiVList+k];let w=Q[p.qVW+k];let delta=pairDelta(pair,false);let distance=max(1e-14,length(delta));let nr=delta/distance;let F=Q[4u*pair+3u];
    gr+=F*w*nr;diag+=F*w*w*(1.-nr*nr)/distance;
   }
 return array<vec3f,2>(gr,diag);
}
@compute @workgroup_size(${WG}) fn gradient(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${C_STATUS}u)!=0u){return;}
 let i=g.x;var gg=0.;
 if(i<p.n){
  if(isFixed(i)){W[p.wG+i]=vec4f(0);}else{
   let x=xv(i);let pr=pred(i);let at=attach(i);let elastic=elasticGradient(i);var gr=pr.w*(x-pr.xyz)+at.w*(x-at.xyz)+elastic[0];var diag=vec3f(pr.w+at.w)+elastic[1];
   // Coordinate stiffness without contact: sizes the penalties of new contacts.
   W[p.wK+i]=vec4f(diag,0);
   if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
    let e=KU[p.kvList+k];let ci=e>>3u;let j=e&7u;let c=contactAt(ci);let w=slotW(c,j);let sx=contactSum(c,0u);let m=contactM(c,contactMotion(ci,c,sx));
    // The gap row: w n, or J_j^T n for a rod pair.
    var M=mat3x3f(vec3f(w,0,0),vec3f(0,w,0),vec3f(0,0,w));if(c.rodm!=0u){M=rodJ(c,rodAt(ci,c),j);}
    let row=transpose(M)*c.n;
    gr-=cForce(m,c.rho)*row;diag+=cCurv(m,c.rho)*row*row;
    if(c.mu>0.){
     let slip=contactSlip(ci,c,sx);let f=c.mu*fF1(length(slip));
     // A rod pair: J_j^T slip, and the diagonal of J_j^T P_t J_j = J_j^T J_j - (J_j^T n)(J_j^T n)^T.
     if(c.rodm!=0u){let MM=transpose(M)*M;gr+=f*(transpose(M)*slip);diag+=f*(vec3f(MM[0].x,MM[1].y,MM[2].z)-row*row);}
     else{gr+=w*f*slip;diag+=w*w*f*(vec3f(1)-c.n*c.n);}
    }
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
 // Units of this solve through the last one that had work (sizes the next submissions).
 atomicStore(&C[${C_WORKED}],unit+1u);
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
  // Rod bending enters on the diagonal only, as max(0, d2E/dx_k^2) (the Float64 model).
  var rod=vec3f(0);var bending=false;
  for(var k=I[p.iVStart+row];k<I[p.iVStart+row+1u];k++){let code=I[p.iVList+k];if((code>>28u)==${KIND_JOINT}u){rod+=E[p.eJoint+9u*(code&0xffffffu)+2u*((code>>24u)&3u)+1u].xyz;bending=true;}}
  if(bending){let r=max(rod,vec3f(0));M+=mat3x3f(vec3f(r.x,0,0),vec3f(0,r.y,0),vec3f(0,0,r.z));}
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
// The contacts' Hessian coefficients at this Newton iterate: x is fixed through
// the CG and the line search, so they are computed once per contact, not at every
// incidence of every matvec.
@compute @workgroup_size(${WG}) fn contactCoef(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(p.contactMode==0u||!isActive()){return;}
 let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  let k=contactAt(c);KF[p.kfCoef+c]=vec4f(contactCoefficients(c,k),0,0);
 }
}
// The contact term of A p, once per contact: f = a n (n.s) + b (s - n (n.s)) with
// s = sum_j w_j p_j; the matvec gathers w_i f at each of the contact's vertices.
// A rod pair: s = sum_j J_j p_j plus the roll, F = a n (n.s) + b P_t s gathered by
// weight, and each side's C^T F (KF[kfRodForce+2c+side]) gathered by its endpoint roles.
fn contactForceAt(c:u32){
 if(p.rodContact!=0u){let k=contactAt(c);if(k.rodm!=0u){
  let g=rodAt(c,k);let cf=KF[p.kfCoef+c].xy;let s=rodMove(k,g,rodSlotU(k,2u),rodTwist(c,2u));let sn=dot(k.n,s);let F=cf.x*sn*k.n+cf.y*(s-sn*k.n);
  KF[p.kfForce+c]=vec4f(F,0);KF[p.kfRodForce+2u*c]=vec4f(rodCt(g.m0.xyz,g.t0,F),0);
  var second=vec3f(0);if(g.two){second=rodCt(g.m1.xyz,g.t1,F);}KF[p.kfRodForce+2u*c+1u]=vec4f(second,0);
  return;
 }}
 let n=KF[4u*c].xyz;let w=KF[4u*c+3u];let j=p.kuCV+4u*c;let cf=KF[p.kfCoef+c].xy;var s=vec3f(0);
 for(var q=0u;q<4u;q++){if(w[q]!=0.){s+=w[q]*W[p.wP+KU[j+q]].xyz;}}
 let sn=dot(n,s);KF[p.kfForce+c]=vec4f(cf.x*sn*n+cf.y*(s-sn*n),0);
}
@compute @workgroup_size(${WG}) fn cgContact(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(num_workgroups) ng:vec3u){
 if(uniformWord(l,${C_CGACTIVE}u)==0u||p.contactMode==0u){return;}
 let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){contactForceAt(c);}
}
fn contactBlock(i:u32)->mat3x3f{
 var M=mat3x3f(vec3f(0),vec3f(0),vec3f(0));
 if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
  let e=KU[p.kvList+k];let ci=e>>3u;let c=contactAt(ci);let w=slotW(c,e&7u);let cf=KF[p.kfCoef+ci].xy;let nn=outer(c.n,c.n);
  if(c.rodm!=0u){let J=rodJ(c,rodAt(ci,c),e&7u);let row=transpose(J)*c.n;M+=(cf.x-cf.y)*outer(row,row)+cf.y*(transpose(J)*J);continue;}
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
 if(i<p.segments&&twistCG()){rz+=twistInitAt(i);}
 red[l]=vec4f(rz,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishInit(@builtin(local_invocation_index) l:u32){
 if(!uniformActive(l)){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 if(atomicLoad(&C[${C_STATUS}])!=0u){atomicStore(&C[${C_ACTIVE}],0u);atomicStore(&C[${C_CGACTIVE}],0u);return;}
 let rz=sums[0].x;let forcing=S(2u).y;W[p.wScal]=vec4f(rz,forcing*forcing*rz,0,0);W[p.wScal+2u]=vec4f(rz,forcing,0,0);
 atomicStore(&C[${C_CGCOUNT}],0u);atomicStore(&C[${C_STOPAFTER}],0u);atomicStore(&C[${C_USEZ0}],0u);atomicStore(&C[${C_CGACTIVE}],select(0u,1u,rz>0.));
}
// Twist row s (CG unknown n+s): Jacobi on the model diagonal; held twist rows stay zero.
fn twistPre(s:u32)->f32{if(!twistFree(s)){return 0.;}return 1./W[p.wT2+s].y;}
fn twistInitAt(s:u32)->f32{
 let k=p.n+s;let r=select(0.,-W[p.wT+s].y,twistFree(s));let z=twistPre(s)*r;
 W[p.wR+k]=vec4f(r,0,0,0);W[p.wZ+k]=vec4f(z,0,0,0);W[p.wZ0+k]=vec4f(z,0,0,0);W[p.wP+k]=vec4f(z,0,0,0);W[p.wS+k]=vec4f(0);return r*z;
}
// The twist row of A p: the rod's tridiagonal model plus each pair side's roll
// against cgContact's force, -r (t x m).F (for a round section its friction part).
fn twistAp(s:u32,pt:f32,below:f32,above:f32)->f32{
 if(!twistFree(s)){return 0.;}
 var y=W[p.wT2+s].z*pt;if(s>0u){y+=twistCoupling(s-1u)*below;}if(s+1u<p.segments){y+=twistCoupling(s)*above;}
 return y;
}
fn twistSide(ci:u32,c:Contact,side:u32,F:vec3f)->f32{return dot(rodRoll(rodAt(ci,c),side),F);}
fn twistMatvecAt(s:u32)->f32{
 let k=p.n+s;if(!twistFree(s)){W[p.wQ+k]=vec4f(0);return 0.;}
 var below=0.;var above=0.;if(s>0u){below=W[p.wP+k-1u].x;}if(s+1u<p.segments){above=W[p.wP+k+1u].x;}
 var y=twistAp(s,W[p.wP+k].x,below,above);
 for(var e=KU[p.kvStart+k];e<KU[p.kvStart+k+1u];e++){
  let entry=KU[p.kvList+e];let ci=entry>>3u;let c=contactAt(ci);let F=KF[p.kfForce+ci].xyz;
  y+=twistSide(ci,c,(entry&7u)-5u,F);
 }
 W[p.wQ+k]=vec4f(y,0,0,0);return W[p.wP+k].x*y;
}
fn twistUpdateAt(s:u32)->f32{
 let k=p.n+s;let alpha=S(0u).z;W[p.wS+k]=vec4f(W[p.wS+k].x+alpha*W[p.wP+k].x,0,0,0);
 let r=W[p.wR+k].x-alpha*W[p.wQ+k].x;let z=twistPre(s)*r;W[p.wR+k]=vec4f(r,0,0,0);W[p.wZ+k]=vec4f(z,0,0,0);return r*z;
}
fn twistDirectionAt(s:u32){let k=p.n+s;W[p.wP+k]=vec4f(W[p.wZ+k].x+S(0u).w*W[p.wP+k].x,0,0,0);}
// q = A p at vertex i; returns p.q there.
fn matvecAt(i:u32)->f32{
  var y=vec3f(0);
  if(!isFixed(i)){for(var b=I[p.iRow+i];b<I[p.iRow+i+1u];b++){let v=W[p.wP+I[p.iCol+b]].xyz;y+=vec3f(dot(E[p.eA+3u*b].xyz,v),dot(E[p.eA+3u*b+1u].xyz,v),dot(E[p.eA+3u*b+2u].xyz,v));}
   // Contacts couple vertices outside the block pattern: matrix-free, each contact's
   // term of A p computed once (cgContact) and gathered here by weight.
   if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
    let e=KU[p.kvList+k];let c=e>>3u;let j=e&7u;var w=0.;if(j<4u){w=KF[4u*c+3u][j];}else{w=KF[p.kfRod+3u*c+2u].x;}
    y+=w*KF[p.kfForce+c].xyz;
    if(p.rodContact!=0u){let m=KU[p.kuRod+4u*c+3u];if(m!=0u){y+=rodRole(m,j,0u)*KF[p.kfRodForce+2u*c].xyz+rodRole(m,j,1u)*KF[p.kfRodForce+2u*c+1u].xyz;}}
   }}
  }
  W[p.wQ+i]=vec4f(y,0);return dot(W[p.wP+i].xyz,y);
}
fn alphaFinish(curvature:f32){
 var s0=S(0u);
 if(!(curvature>0.)){if(atomicLoad(&C[${C_CGCOUNT}])==0u){s0.z=1.;atomicStore(&C[${C_STOPAFTER}],1u);}else{atomicStore(&C[${C_CGACTIVE}],0u);}}
 else{s0.z=s0.x/curvature;}
 W[p.wScal]=s0;
}
fn updateAt(i:u32)->f32{let alpha=S(0u).z;W[p.wS+i]=vec4f(W[p.wS+i].xyz+alpha*W[p.wP+i].xyz,0);let r=W[p.wR+i].xyz-alpha*W[p.wQ+i].xyz;let z=applyPre(i,r);W[p.wR+i]=vec4f(r,0);W[p.wZ+i]=vec4f(z,0);return dot(r,z);}
fn betaFinish(next:f32){
 var s0=S(0u);s0.w=next/s0.x;s0.x=next;W[p.wScal]=s0;
 let count=atomicAdd(&C[${C_CGCOUNT}],1u)+1u;
 if(atomicLoad(&C[${C_STOPAFTER}])!=0u||!(next>s0.y)||!(next>0.)||count>=p.cgCap){atomicStore(&C[${C_CGACTIVE}],0u);}
}
fn directionAt(i:u32){W[p.wP+i]=vec4f(W[p.wZ+i].xyz+S(0u).w*W[p.wP+i].xyz,0);}
@compute @workgroup_size(${WG}) fn cgMatvec(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let i=g.x;var pq=0.;if(i<p.n){pq=matvecAt(i);}if(i<p.segments&&twistCG()){pq+=twistMatvecAt(i);}
 red[l]=vec4f(pq,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishAlpha(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 alphaFinish(sums[0].x);
}
@compute @workgroup_size(${WG}) fn cgUpdate(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let i=g.x;var rz=0.;if(i<p.n){rz=updateAt(i);}if(i<p.segments&&twistCG()){rz+=twistUpdateAt(i);}
 red[l]=vec4f(rz,0,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishBeta(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 betaFinish(sums[0].x);
}
@compute @workgroup_size(${WG}) fn cgDirection(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i>=p.n||!cgActive()){return;}directionAt(i);if(i<p.segments&&twistCG()){twistDirectionAt(i);}}
// The whole CG loop in one workgroup (small meshes): the five dispatches of an
// iteration become barrier-separated phases; the same control words gate it.
@compute @workgroup_size(${WG}) fn cgFused(@builtin(local_invocation_index) l:u32){
 for(var k=0u;k<p.cgCap;k++){
  if(uniformWord(l,${C_CGACTIVE}u)==0u){break;}
  if(p.contactMode!=0u){let count=min(contactCount(),p.kMax);for(var c=l;c<count;c+=${WG}u){contactForceAt(c);}storageBarrier();}
  var pq=0.;for(var i=l;i<p.n;i+=${WG}u){pq+=matvecAt(i);if(i<p.segments&&twistCG()){pq+=twistMatvecAt(i);}}
  red[l]=vec4f(pq,0,0,0);reduce4(l,1u);
  if(l==0u){alphaFinish(red[0].x);}
  storageBarrier();
  if(uniformWord(l,${C_CGACTIVE}u)==0u){break;}
  var rz=0.;for(var i=l;i<p.n;i+=${WG}u){rz+=updateAt(i);if(i<p.segments&&twistCG()){rz+=twistUpdateAt(i);}}
  red[l]=vec4f(rz,0,0,0);reduce4(l,1u);
  if(l==0u){betaFinish(red[0].x);}
  storageBarrier();
  if(uniformWord(l,${C_CGACTIVE}u)==0u){break;}
  for(var i=l;i<p.n;i+=${WG}u){directionAt(i);if(i<p.segments&&twistCG()){twistDirectionAt(i);}}
  storageBarrier();
 }
}
// The same loop with the CG direction and scalars resident in the workgroup and
// each thread's residual, preconditioned residual and A p in registers: no device
// atomics or storage barriers inside the loop (n <= SHARED_CG_MAX_VERTICES). The
// loop is latency-bound (one workgroup), so each reduction is two barriers, a
// two-level sum (eight threads of eight, then thread 0 of the eight: WG = 64), and one
// uniform load of the new scalars, not a six-level tree.
var<workgroup> pw:array<vec4f,${SHARED_CG_MAX_VERTICES}>;
struct SharedCG{s:vec4f,count:u32,stop:u32,on:u32}
var<workgroup> cg:SharedCG;
fn sharedContactP(c:Contact)->vec3f{var s=vec3f(0);for(var j=0u;j<5u;j++){let w=slotW(c,j);if(w!=0.){s+=w*pw[slotV(c,j)].xyz;}}return s;}
fn sharedMatvecAt(i:u32)->vec3f{
  var y=vec3f(0);
  if(!isFixed(i)){for(var b=I[p.iRow+i];b<I[p.iRow+i+1u];b++){let v=pw[I[p.iCol+b]].xyz;y+=vec3f(dot(E[p.eA+3u*b].xyz,v),dot(E[p.eA+3u*b+1u].xyz,v),dot(E[p.eA+3u*b+2u].xyz,v));}
   if(p.contactMode!=0u){for(var k=KU[p.kvStart+i];k<KU[p.kvStart+i+1u];k++){
    let e=KU[p.kvList+k];let ci=e>>3u;let c=contactAt(ci);let w=slotW(c,e&7u);let cf=KF[p.kfCoef+ci].xy;
    if(c.rodm!=0u){
     var u:array<vec3f,5>;for(var q=0u;q<(c.rodm&7u);q++){u[q]=pw[slotV(c,q)].xyz;}
     let g=rodAt(ci,c);let s=rodMove(c,g,u,sharedTwistP(ci));let sn=dot(c.n,s);
     y+=transpose(rodJ(c,g,e&7u))*(cf.x*sn*c.n+cf.y*(s-sn*c.n));continue;
    }
    let s=sharedContactP(c);let sn=dot(c.n,s);
    y+=w*(cf.x*sn*c.n+cf.y*(s-sn*c.n));
   }}
  }
  return y;
}
// The CG direction's twist rows of a rod pair's sides (workgroup copy: .w of the segment's index).
fn sharedTwistP(ci:u32)->vec2f{
 var t=vec2f(0);if(!twistCG()){return t;}
 for(var side=0u;side<2u;side++){let s=KU[p.kuRod+4u*ci+1u+side];if(s!=NONE){t[side]=pw[s].w;}}
 return t;
}
fn sharedTwistMatvecAt(s:u32)->f32{
 let k=p.n+s;if(!twistFree(s)){return 0.;}
 var below=0.;var above=0.;if(s>0u){below=pw[s-1u].w;}if(s+1u<p.segments){above=pw[s+1u].w;}
 var y=twistAp(s,pw[s].w,below,above);
 for(var e=KU[p.kvStart+k];e<KU[p.kvStart+k+1u];e++){
  let entry=KU[p.kvList+e];let ci=entry>>3u;let c=contactAt(ci);let cf=KF[p.kfCoef+ci].xy;
  var u:array<vec3f,5>;for(var q=0u;q<(c.rodm&7u);q++){u[q]=pw[slotV(c,q)].xyz;}
  let sv=rodMove(c,rodAt(ci,c),u,sharedTwistP(ci));let sn=dot(sv,c.n);
  y+=twistSide(ci,c,(entry&7u)-5u,cf.x*sn*c.n+cf.y*(sv-sn*c.n));
 }
 return y;
}
fn partial8(l:u32){if(l<8u){var a=0.;var b=0.;for(var j=0u;j<8u;j+=2u){a+=red[8u*l+j].x;b+=red[8u*l+j+1u].x;}red[${WG}u+l].x=a+b;}workgroupBarrier();}
fn sum8()->f32{var a=0.;var b=0.;for(var j=0u;j<8u;j+=2u){a+=red[${WG}u+j].x;b+=red[${WG}u+j+1u].x;}return a+b;}
@compute @workgroup_size(${WG}) fn cgShared(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${C_CGACTIVE}u)==0u){return;}
 // Twist row n+i (i < segments < n) rides in .w of vertex i's p, r, z and q: no extra registers.
 var r:array<vec4f,${SHARED_CG_PER_THREAD}>;var z:array<vec4f,${SHARED_CG_PER_THREAD}>;var q:array<vec4f,${SHARED_CG_PER_THREAD}>;let tw=twistCG();
 for(var j=0u;j<${SHARED_CG_PER_THREAD}u;j++){let i=l+j*${WG}u;if(i>=p.n){break;}
  var pi=vec4f(W[p.wP+i].xyz,0);var ri=vec4f(W[p.wR+i].xyz,0);var zi=vec4f(W[p.wZ+i].xyz,0);
  if(tw&&i<p.segments){let k=p.n+i;pi.w=W[p.wP+k].x;ri.w=W[p.wR+k].x;zi.w=W[p.wZ+k].x;}
  pw[i]=pi;r[j]=ri;z[j]=zi;}
 if(l==0u){cg=SharedCG(S(0u),atomicLoad(&C[${C_CGCOUNT}]),atomicLoad(&C[${C_STOPAFTER}]),1u);}
 var state=workgroupUniformLoad(&cg);
 for(var k=0u;k<p.cgCap;k++){
  if(state.on==0u){break;}
  var pq=0.;
  for(var j=0u;j<${SHARED_CG_PER_THREAD}u;j++){let i=l+j*${WG}u;if(i>=p.n){break;}var y=vec4f(sharedMatvecAt(i),0);
   if(tw&&i<p.segments){y.w=sharedTwistMatvecAt(i);}q[j]=y;pq+=dot(pw[i],y);}
  red[l]=vec4f(pq,0,0,0);workgroupBarrier();partial8(l);
  // alphaFinish on the workgroup copies.
  if(l==0u){let curvature=sum8();if(!(curvature>0.)){if(cg.count==0u){cg.s.z=1.;cg.stop=1u;}else{cg.on=0u;}}else{cg.s.z=cg.s.x/curvature;}}
  state=workgroupUniformLoad(&cg);
  if(state.on==0u){break;}
  let alpha=state.s.z;var rz=0.;
  for(var j=0u;j<${SHARED_CG_PER_THREAD}u;j++){let i=l+j*${WG}u;if(i>=p.n){break;}
   W[p.wS+i]=vec4f(W[p.wS+i].xyz+alpha*pw[i].xyz,0);let ri=r[j]-alpha*q[j];var zi=vec4f(applyPre(i,ri.xyz),0);
   if(tw&&i<p.segments){let k=p.n+i;W[p.wS+k]=vec4f(W[p.wS+k].x+alpha*pw[i].w,0,0,0);zi.w=twistPre(i)*ri.w;}
   r[j]=ri;z[j]=zi;rz+=dot(ri,zi);}
  red[l]=vec4f(rz,0,0,0);workgroupBarrier();partial8(l);
  // betaFinish on the workgroup copies.
  if(l==0u){let next=sum8();cg.s.w=next/cg.s.x;cg.s.x=next;cg.count++;if(cg.stop!=0u||!(next>cg.s.y)||!(next>0.)||cg.count>=p.cgCap){cg.on=0u;}}
  state=workgroupUniformLoad(&cg);
  if(state.on==0u){break;}
  let beta=state.s.w;
  for(var j=0u;j<${SHARED_CG_PER_THREAD}u;j++){let i=l+j*${WG}u;if(i>=p.n){break;}pw[i]=z[j]+beta*pw[i];}
  workgroupBarrier();
 }
 for(var j=0u;j<${SHARED_CG_PER_THREAD}u;j++){let i=l+j*${WG}u;if(i>=p.n){break;}
  W[p.wP+i]=vec4f(pw[i].xyz,0);W[p.wR+i]=vec4f(r[j].xyz,0);W[p.wZ+i]=vec4f(z[j].xyz,0);W[p.wQ+i]=vec4f(q[j].xyz,0);
  if(tw&&i<p.segments){let k=p.n+i;W[p.wP+k]=vec4f(pw[i].w,0,0,0);W[p.wR+k]=vec4f(r[j].w,0,0,0);W[p.wZ+k]=vec4f(z[j].w,0,0,0);W[p.wQ+k]=vec4f(q[j].w,0,0,0);}}
 if(l==0u){W[p.wScal]=state.s;atomicStore(&C[${C_CGCOUNT}],state.count);atomicStore(&C[${C_STOPAFTER}],state.stop);atomicStore(&C[${C_CGACTIVE}],state.on);}
}

// ---- descent check and line search
@compute @workgroup_size(${WG}) fn slope(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformActive(l)){return;}
 let i=g.x;var gs=0.;var gt=0.;if(i<p.n){gs=dot(W[p.wG+i].xyz,W[p.wS+i].xyz);}if(i<p.segments){gt=W[p.wT+i].y*W[p.wT+i].z;}
 red[l]=vec4f(gs,gt,0,0);writePartial(0u,wg.x,l,1u);
}
@compute @workgroup_size(${WG}) fn finishSlope(@builtin(local_invocation_index) l:u32){
 if(!uniformActive(l)){return;}
 let sums=sumPartials(l,0u,p.wgV,1u);if(l!=0u){return;}
 var s1=S(1u);s1.z=sums[0].x+sums[0].y;
 // Roundoff in CG: the preconditioned gradient is always a descent direction (the twist step is kept).
 if(!(s1.z<0.)){atomicStore(&C[${C_USEZ0}],1u);s1.z=-S(2u).x+sums[0].y;}
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
  let xa=xv(id.x);let xb=xv(id.y);let xc=xv(id.z);let xd=xv(id.w);let at=hingeEdges(h,xa,xb,xc,xd);let angle=dihedral(at.e,at.u,at.w);let phi=bendingPhi(angle,mm.mode);
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
  let id=segIds(s);let oldLength=E[p.eSegOld+2u*s].x;let mm=segMaterial(s,id,oldLength);let delta=xv(id.y)-xv(id.x);let q=segLength(s,delta,oldLength);let l0=oldLength+q.w;
  let dd=dir(id.y)-dir(id.x);let elastic=E[p.eSegOld+2u*s].y+q.w;
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
 // Over vertices and their key-ordered lists, each contact at its first listed vertex:
 // a fixed summation order (slot order follows thread scheduling). A contact with no
 // free vertex does not move along the direction, so it adds no change.
 if(p.contactMode==0u){storeLine(l,de,noise);writePartial(p.pContact,wg.x,l,4u);return;}
 for(var i=g.x;i<p.n;i+=p.lineGroups*${WG}u){for(var q=KU[p.kvStart+i];q<KU[p.kvStart+i+1u];q++){
  let e=KU[p.kvList+q];let k0=e>>3u;let c=contactAt(k0);var owner=true;for(var j=0u;j<(e&7u);j++){if(!isFixed(slotV(c,j))){owner=false;}}
  if(!owner){continue;}
  let sx=contactSum(c,0u);var sd=contactSum(c,1u);let m=contactM(c,contactMotion(k0,c,sx));
  if(c.rodm!=0u){sd=rodMove(c,rodAt(k0,c),rodSlotU(c,1u),rodTwist(k0,1u));}
  let dn=dot(c.n,sd);let slip=contactSlip(k0,c,sx);let td=tangent(c.n,sd);let y0=length(slip);
  for(var k=0u;k<${ALPHAS}u;k++){
   let a=alphaOf(k);var change=cPenChange(m,-a*dn,c.rho);
   if(c.mu>0.){let f=fF0Change(y0,length(slip+a*td),2.*a*dot(slip,td)+a*a*dot(td,td));change+=c.mu*f;}
   de[k]+=change.x;noise[k]+=32.*EPS*change.y;
  }
 }}
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
@compute @workgroup_size(${WG}) fn apply(@builtin(global_invocation_id) g:vec3u){let i=g.x;if(i>=p.n||!isActive()){return;}W[p.wX+i]=vec4f(xv(i)+S(1u).w*dir(i),0);if(i<p.segments){W[p.wT+i].x+=S(1u).w*W[p.wT+i].z;}}
// ---- DER rod joints (rodEnergy of lib/cloth/dynamics.ts) and segment twist.
// Joint j: I[iJoint+4j..]=(a,b,c,first segment); per step D[dJoint+4j..]=
// (rest curvature),(old curvature),(bend1/len,bend2/len,twist/len,gamma_b),
// (gamma_t, tau0 offset, tau_old offset, wrapped prior reference twist).
// Segment s: per step D[dTwist+4s..]=(director,cos theta_old),(tangent,sin theta_old),
// (I/dt^2, attachment, theta_old-target, predicted increment),(free, radius, 0, 0);
// W[wT+s]=(increment, gradient, step, d2E/dtheta^2), W[wT2+s]=(pivot, diagonal, 0, 0).
// Position derivatives are exact second-order forward derivatives (jets)
// along one coordinate; twist derivatives are analytic.
struct J1{v:f32,d:f32,dd:f32}
struct J3{v:vec3f,d:vec3f,dd:vec3f}
fn j3c(v:vec3f)->J3{return J3(v,vec3f(0),vec3f(0));}
fn jAdd3(a:J3,b:J3)->J3{return J3(a.v+b.v,a.d+b.d,a.dd+b.dd);}
fn jSub3(a:J3,b:J3)->J3{return J3(a.v-b.v,a.d-b.d,a.dd-b.dd);}
fn jScaleC(a:J3,s:f32)->J3{return J3(a.v*s,a.d*s,a.dd*s);}
fn jScale3(a:J3,s:J1)->J3{return J3(a.v*s.v,a.d*s.v+a.v*s.d,a.dd*s.v+2.*a.d*s.d+a.v*s.dd);}
fn jDot(a:J3,b:J3)->J1{return J1(dot(a.v,b.v),dot(a.d,b.v)+dot(a.v,b.d),dot(a.dd,b.v)+2.*dot(a.d,b.d)+dot(a.v,b.dd));}
fn jCross(a:J3,b:J3)->J3{return J3(cross(a.v,b.v),cross(a.d,b.v)+cross(a.v,b.d),cross(a.dd,b.v)+2.*cross(a.d,b.d)+cross(a.v,b.dd));}
fn jRecip(a:J1)->J1{let r=1./a.v;return J1(r,-a.d*r*r,(2.*a.d*a.d*r-a.dd)*r*r);}
fn jSqrt(a:J1)->J1{let r=sqrt(a.v);let d=a.d/(2.*r);return J1(r,d,(a.dd-2.*d*d)/(2.*r));}
fn jNormalize(a:J3)->J3{return jScale3(a,jRecip(jSqrt(jDot(a,a))));}
fn jAtan2(y:J1,x:J1)->J1{let den=x.v*x.v+y.v*y.v;let num=x.v*y.d-y.v*x.d;return J1(atan2(y.v,x.v),num/den,((x.v*y.dd-y.v*x.dd)*den-num*2.*(x.v*x.d+y.v*y.d))/(den*den));}
// transportDirector of lib/cloth/topology.ts.
fn jTransport(d:J3,source:J3,to:J3)->J3{
 let axis=jCross(source,to);let c=jDot(source,to);var rotated=d;
 if(c.v>-1.+1e-6){let a=jCross(axis,d);let b=jCross(axis,a);rotated=jAdd3(jAdd3(d,a),jScale3(b,jRecip(J1(1.+min(c.v,1.),c.d,c.dd))));}
 let normal=jSub3(rotated,jScale3(to,jDot(rotated,to)));
 if(length(normal.v)<1e-8){return jNormalize(jCross(to,j3c(select(vec3f(0,1,0),vec3f(1,0,0),abs(to.v.x)<.8))));}
 return jNormalize(normal);
}
fn segT(s:u32,k:u32)->vec4f{return D[p.dTwist+4u*s+k];}
fn jd(j:u32,k:u32)->vec4f{return D[p.dJoint+4u*j+k];}
fn jointIds(j:u32)->vec4u{return vec4u(I[p.iJoint+4u*j],I[p.iJoint+4u*j+1u],I[p.iJoint+4u*j+2u],I[p.iJoint+4u*j+3u]);}
// (cos, sin) of theta_old + increment, by the angle sum (theta_old may be large).
fn twistCS(s:u32,increment:f32)->vec2f{let c=segT(s,0u).w;let sn=segT(s,1u).w;let ci=cos(increment);let si=sin(increment);return vec2f(c*ci-sn*si,sn*ci+c*si);}
struct JFrame{t:J3,d:J3,m1:J3,m2:J3}
fn jFrame(s:u32,e:J3,cs:vec2f)->JFrame{
 let t=jNormalize(e);let d=jTransport(j3c(segT(s,0u).xyz),j3c(segT(s,1u).xyz),t);let o=jCross(t,d);let m1=jAdd3(jScaleC(d,cs.x),jScaleC(o,cs.y));
 return JFrame(t,d,m1,jCross(t,m1));
}
// Curvature (kb.a.m2, -kb.a.m1, kb.b.m2, -kb.b.m1) and the reference twist
// change wrap(raw - prior) along the seeded coordinate; ok=false where the
// Float64 energy is infinite (degenerate edge or reversed tangents).
struct JointEval{k0:J1,k1:J1,k2:J1,k3:J1,r:J1,ok:bool}
fn jointEval(j:u32,id:vec4u,xa:vec3f,xb:vec3f,xc:vec3f,da:vec3f,db:vec3f,dc:vec3f,ca:vec2f,cb:vec2f)->JointEval{
 let ea=J3(segOldEdge(id.w)+(xb-xa),db-da,vec3f(0));let eb=J3(segOldEdge(id.w+1u)+(xc-xb),dc-db,vec3f(0));
 let A=jFrame(id.w,ea,ca);let B=jFrame(id.w+1u,eb,cb);let c=jDot(A.t,B.t);
 let kb=jScaleC(jScale3(jCross(A.t,B.t),jRecip(J1(1.+c.v,c.d,c.dd))),2.);
 let moved=jTransport(A.d,A.t,B.t);let raw=jAtan2(jDot(B.t,jCross(moved,B.d)),jDot(moved,B.d));
 var w=raw.v-jd(j,3u).w;if(w>3.14159265){w-=6.28318531;}else if(w< -3.14159265){w+=6.28318531;}
 let m0=jDot(kb,A.m1);let m1=jDot(kb,B.m1);
 let ok=1.+c.v>=1e-6&&length(ea.v)>=1e-14&&length(eb.v)>=1e-14;
 return JointEval(jDot(kb,A.m2),J1(-m0.v,-m0.d,-m0.dd),jDot(kb,B.m2),J1(-m1.v,-m1.d,-m1.dd),J1(w,raw.d,raw.dd),ok);
}
fn jointCurvature(e:JointEval)->vec4f{return vec4f(e.k0.v,e.k1.v,e.k2.v,e.k3.v);}
fn jointTwistIncrement(j:u32,id:vec4u)->f32{return W[p.wT+id.w+1u].x-W[p.wT+id.w].x;}
@compute @workgroup_size(${WG}) fn elemJoint(@builtin(global_invocation_id) g:vec3u){
 let j=g.x/3u;let local=g.x%3u;if(j>=p.joints||done()){return;}
 let id=jointIds(j);let xa=xv(id.x);let xb=xv(id.y);let xc=xv(id.z);let v=id[local];
 let ca=twistCS(id.w,W[p.wT+id.w].x);let cb=twistCS(id.w+1u,W[p.wT+id.w+1u].x);
 let rest=jd(j,0u);let old=jd(j,1u);let k=jd(j,2u);let q=jd(j,3u);let w=vec4f(k.x,k.y,k.x,k.y);
 var grad=vec3f(0);var second=vec3f(0);var value:JointEval;var bad=false;
 for(var axis=0u;axis<3u;axis++){
  var unit=vec3f(0);unit[axis]=1.;
  let e=jointEval(j,id,xa,xb,xc,select(vec3f(0),unit,id.x==v),select(vec3f(0),unit,id.y==v),select(vec3f(0),unit,id.z==v),ca,cb);value=e;bad=bad||!e.ok;
  let kv=jointCurvature(e);let P=(kv-rest)+k.w*(kv-old);let kd=vec4f(e.k0.d,e.k1.d,e.k2.d,e.k3.d);let kdd=vec4f(e.k0.dd,e.k1.dd,e.k2.dd,e.k3.dd);
  let tw=jointTwistIncrement(j,id)+e.r.v;let Q=(q.y+tw)+q.x*(q.z+tw);
  grad[axis]=.5*dot(w,P*kd)+k.z*Q*e.r.d;second[axis]=.5*dot(w,(1.+k.w)*kd*kd+P*kdd)+k.z*((1.+q.x)*e.r.d*e.r.d+Q*e.r.dd);
 }
 if(bad){grad=vec3f(bitcast<f32>(NAN_BITS));}
 E[p.eJoint+9u*j+2u*local]=vec4f(grad,0);E[p.eJoint+9u*j+2u*local+1u]=vec4f(second,0);
 if(local==0u){
  // d kappa/d theta: (k1,-k0) on the first segment's frame, (k3,-k2) on the second's.
  let kv=jointCurvature(value);let P=(kv-rest)+k.w*(kv-old);let tw=jointTwistIncrement(j,id)+value.r.v;let Q=(q.y+tw)+q.x*(q.z+tw);let tt=k.z*(1.+q.x);
  let ga=.5*(w.x*P.x*kv.y-w.y*P.y*kv.x)-k.z*Q;let gb=.5*(w.z*P.z*kv.w-w.w*P.w*kv.z)+k.z*Q;
  let ha=.5*(w.x*((1.+k.w)*kv.y*kv.y-P.x*kv.x)+w.y*((1.+k.w)*kv.x*kv.x-P.y*kv.y))+tt;
  let hb=.5*(w.z*((1.+k.w)*kv.w*kv.w-P.z*kv.z)+w.w*((1.+k.w)*kv.z*kv.z-P.w*kv.w))+tt;
  var nan=1.;if(bad){nan=bitcast<f32>(NAN_BITS);}
  E[p.eJoint+9u*j+6u]=vec4f(ga,gb,ha,hb)*nan;E[p.eJoint+9u*j+7u]=kv;E[p.eJoint+9u*j+8u]=vec4f(q.y+tw,q.z+tw,value.r.v,0);
 }
}
// Twist gradient, its second derivative and the convergence measure |g/max(I/dt^2,d2E)|*radius.
@compute @workgroup_size(${WG}) fn twistGradient(@builtin(global_invocation_id) g:vec3u){
 let s=g.x;if(s>=p.segments||done()){return;}
 let f=segT(s,3u);if(f.x<.5){W[p.wT+s]=vec4f(0,0,0,1);W[p.wT2+s]=vec4f(1,1,0,0);return;}
 let a=segT(s,2u);let t=W[p.wT+s];var gr=a.x*(t.x-a.w)+a.y*(a.z+t.x);var h=a.x+a.y;
 let asSecond=I[p.iSegJoint+2u*s];if(asSecond!=0xffffffffu){let e=E[p.eJoint+9u*asSecond+6u];gr+=e.y;h+=e.w;}
 let asFirst=I[p.iSegJoint+2u*s+1u];if(asFirst!=0xffffffffu){let e=E[p.eJoint+9u*asFirst+6u];gr+=e.x;h+=e.z;}
 // Rod contact on the twist, per pair side of roll v = -r (t x m): the normal force's torque
 // -F_n (n.v) with curvature a (n.v)^2 (an elliptical section's, zero for a round one), and the
 // rolling friction f v.slip with f |P_t v|^2.
 var hc=0.;
 if(p.contactMode!=0u&&p.rodContact!=0u){for(var k=KU[p.kvStart+p.n+s];k<KU[p.kvStart+p.n+s+1u];k++){
  let e=KU[p.kvList+k];let ci=e>>3u;let side=(e&7u)-5u;let c=contactAt(ci);let g=rodAt(ci,c);
  let v=rodRoll(g,side);let vn=dot(v,c.n);let sx=contactSum(c,0u);
  if(vn!=0.){let m=contactM(c,contactMotion(ci,c,sx));gr-=cForce(m,c.rho)*vn;hc+=cCurv(m,c.rho)*vn*vn;}
  if(!(c.mu>0.)){continue;}
  let slip=contactSlip(ci,c,sx);let f=c.mu*fF1(length(slip));
  gr+=f*dot(v,slip);hc+=f*(dot(v,v)-vn*vn);
 }}
 h+=hc;
 let diagonal=max(a.x,h);let measure=abs(gr/diagonal)*f.y;
 if(measure>=0.&&abs(measure)<3e38){atomicMax(&C[${C_MEASURE}],bitcast<u32>(measure));}else{atomicOr(&C[${C_STATUS}],${S_NONFINITE}u);}
 // The CG's twist row: the model diagonal less the contact curvature (the contact operator adds that).
 W[p.wT+s]=vec4f(t.x,gr,t.z,h);W[p.wT2+s]=vec4f(W[p.wT2+s].x,diagonal,diagonal-hc,0);
}
// Twist couples to its rod neighbours through their shared joint: a
// tridiagonal Hessian per rod (exact second derivatives, coupling
// -twist/len (1+gamma_t)), solved by Thomas elimination; a nonpositive pivot
// leaves that rod on the diagonal step.
fn twistFree(s:u32)->bool{return segT(s,3u).x>.5;}
fn twistCoupling(s:u32)->f32{let j=I[p.iSegJoint+2u*s+1u];if(j==0xffffffffu||!twistFree(s)||!twistFree(s+1u)){return 0.;}return -jd(j,2u).z*(1.+jd(j,3u).x);}
@compute @workgroup_size(${WG}) fn twistSolve(@builtin(global_invocation_id) g:vec3u){
 let r=g.x;if(r>=p.rods||!isActive()){return;}
 let base=I[p.iRod+2u*r];let count=I[p.iRod+2u*r+1u];var positive=true;
 // With rod contact the CG solved the twist with the positions.
 if(twistCG()){for(var j=0u;j<count;j++){let s=base+j;W[p.wT+s].z=select(0.,W[p.wS+p.n+s].x,twistFree(s));}return;}
 for(var j=0u;j<count&&positive;j++){
  let s=base+j;if(!twistFree(s)){W[p.wT2+s].x=1.;W[p.wT+s].z=0.;continue;}
  var coupling=0.;var previous=1.;var carried=0.;
  if(j>0u){coupling=twistCoupling(s-1u);if(twistFree(s-1u)){previous=W[p.wT2+s-1u].x;}carried=coupling*W[p.wT+s-1u].z/previous;}
  let t=W[p.wT+s];let pivot=t.w-coupling*coupling/previous;W[p.wT2+s].x=pivot;W[p.wT+s].z=-t.y-carried;
  positive=pivot>0.&&abs(pivot)<3e38;
 }
 if(positive){
  for(var j=i32(count)-1;j>=0;j--){
   let s=base+u32(j);if(!twistFree(s)){continue;}
   var next=0.;if(u32(j)+1u<count){next=twistCoupling(s)*W[p.wT+s+1u].z;}
   W[p.wT+s].z=(W[p.wT+s].z-next)/W[p.wT2+s].x;
  }
 }else{for(var j=0u;j<count;j++){let s=base+j;if(twistFree(s)){W[p.wT+s].z=-W[p.wT+s].y/W[p.wT2+s].y;}else{W[p.wT+s].z=0.;}}}
}
@compute @workgroup_size(${WG}) fn lineJoint(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformSearch(l)){return;}
 let j=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(j<p.joints){
  let id=jointIds(j);let rest=jd(j,0u);let old=jd(j,1u);let k=jd(j,2u);let q=jd(j,3u);let w=vec4f(k.x,k.y,k.x,k.y);
  let kv=E[p.eJoint+9u*j+7u];let tv=E[p.eJoint+9u*j+8u];let elastic=kv-rest;let viscous=kv-old;
  let xa=xv(id.x);let xb=xv(id.y);let xc=xv(id.z);let da=dir(id.x);let db=dir(id.y);let dc=dir(id.z);
  let ta=W[p.wT+id.w];let tb=W[p.wT+id.w+1u];let z=vec3f(0);
  // Each Float32 curvature and reference twist is good to a few ULP of their scale.
  let kErr=16.*EPS*(1.+abs(kv));let rErr=16.*EPS*(3.2+abs(tv.z));
  for(var a=0u;a<${ALPHAS}u;a++){
   let al=alphaOf(a);let e=jointEval(j,id,xa+al*da,xb+al*db,xc+al*dc,z,z,z,twistCS(id.w,ta.x+al*ta.z),twistCS(id.w+1u,tb.x+al*tb.z));
   let dk=jointCurvature(e)-kv;let dtw=al*(tb.z-ta.z)+(e.r.v-tv.z);
   let ee=dk*(2.*elastic+dk);let vv=k.w*dk*(2.*viscous+dk);let te=dtw*(2.*tv.x+dtw);let tvv=q.x*dtw*(2.*tv.y+dtw);
   de[a]=.25*dot(w,ee+vv)+.5*k.z*(te+tvv);
   noise[a]=32.*EPS*(.25*dot(w,abs(ee)+abs(vv))+.5*k.z*(abs(te)+abs(tvv)))
    +.25*dot(w,kErr*(abs(2.*elastic+dk)+k.w*abs(2.*viscous+dk)+2.*(1.+k.w)*kErr))+.5*k.z*rErr*(abs(2.*tv.x+dtw)+q.x*abs(2.*tv.y+dtw)+2.*(1.+q.x)*rErr);
   if(!e.ok){de[a]=bitcast<f32>(INF_BITS);}
  }
 }
 storeLine(l,de,noise);writePartial(p.pJoint,wg.x,l,4u);
}
@compute @workgroup_size(${WG}) fn lineTwist(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(!uniformSearch(l)){return;}
 let s=g.x;var de:array<f32,${ALPHAS}>;var noise:array<f32,${ALPHAS}>;
 if(s<p.segments&&twistFree(s)){
  let a=segT(s,2u);let t=W[p.wT+s];let inertial=t.z*(t.x-a.w);let spring=t.z*(a.z+t.x);let ss=t.z*t.z;
  for(var k=0u;k<${ALPHAS}u;k++){let al=alphaOf(k);let ii=a.x*(al*inertial+.5*al*al*ss);let sp=a.y*(al*spring+.5*al*al*ss);de[k]=ii+sp;noise[k]=32.*EPS*(abs(ii)+abs(sp));}
 }
 storeLine(l,de,noise);writePartial(p.pTwist,wg.x,l,4u);
}
${clothContactWGSL(WG, S_CONVERGED, {C_STATUS, C_NEWTON, C_ACTIVE, C_CGACTIVE, C_NEWTONMIN}, MAXIMUM_CONTACT_ITERATIONS)}
`;

// Small meshes (n <= SHARED_CG_MAX_VERTICES) run in one workgroup: each kernel
// below becomes a body run over its virtual workgroups in sequence (the same
// partial slots, so the same reductions), barrier-separated. A launch here is
// latency-bound, so a contact round, the step's first detection, and the unit
// before and after its CG are one launch each rather than ~75 per unit. The
// virtual workgroup counts are encodeUnit's, encodeRound's and the first submission's.
const SMALL_BODIES = ['elemTri', 'elemHinge', 'elemSeg', 'elemJoint', 'gradient', 'twistGradient', 'finishMeasure', 'assemble', 'contactCoef', 'cgInit', 'finishInit',
  'twistSolve', 'slope', 'finishSlope', 'lineTri', 'lineHinge', 'lineSeg', 'lineJoint', 'lineTwist', 'lineVertex', 'linePair', 'lineContact', 'finishLine', 'apply',
  'initX', 'prepTri', 'prepHinge', 'prepSeg', 'detectBegin', 'tableClear', 'roundBegin', 'updateMultipliers', 'relinearize', 'roundDecide', 'penalties',
  'listCount', 'listFill', 'listReset', 'gridClear', 'boxFaces', 'boxEdges', 'gridFaces', 'gridEdges', 'detectFaces', 'detectEdges', 'detectColliders',
  'boxSegments', 'gridSegments', 'detectRodPoints', 'detectRodSegments', 'detectRodColliders'] as const;
function withSmallMeshKernels(source: string): string {
  // The fused kernels' value of each builtin a body takes (v: the virtual workgroup;
  // a strided kernel runs as one workgroup that strides over everything).
  const value: Record<string, string> = {global_invocation_id: `vec3u(v*${WG}u+l,0u,0u)`, local_invocation_index: 'l', workgroup_id: 'vec3u(v,0u,0u)', num_workgroups: 'vec3u(1u,1u,1u)'};
  let out = source, wrappers = '';
  const calls: Record<string, string> = {};
  for (const name of SMALL_BODIES) {
    const sizes = [`${WG}`, '1'].filter(size => out.includes(`@compute @workgroup_size(${size}) fn ${name}(`));
    if (sizes.length !== 1) throw new Error(`Cloth Newton kernel ${name} is not a single ${WG}-wide or serial entry point`);
    const head = `@compute @workgroup_size(${sizes[0]}) fn ${name}(`, at = out.indexOf(head);
    if (out.indexOf(head, at + 1) >= 0) throw new Error(`Cloth Newton kernel ${name} is defined twice`);
    const close = out.indexOf('){', at + head.length);
    if (close < 0) throw new Error(`Cloth Newton kernel ${name} has no body`);
    const list = out.slice(at + head.length, close).trim(), params = (list ? list.split(',') : []).map(entry => {
      const match = /^@builtin\((\w+)\) (\w+):(\w+)$/.exec(entry.trim());
      if (!match || !value[match[1]!]) throw new Error(`Cloth Newton kernel ${name}: unsupported parameter ${entry}`);
      return {builtin: match[1]!, name: match[2]!, type: match[3]!};
    });
    out = out.slice(0, at) + `fn ${name}Body(${params.map(q => `${q.name}:${q.type}`).join(',')}){` + out.slice(close + 2);
    wrappers += `${head}${params.map(q => `@builtin(${q.builtin}) ${q.name}:${q.type}`).join(',')}){${name}Body(${params.map(q => q.name).join(',')});}\n`;
    const call = `${name}Body(${params.map(q => value[q.builtin]).join(',')})`;
    calls[name] = sizes[0] === '1' ? `if(l==0u){${call};}` : `${call};`;
  }
  const sync = 'workgroupBarrier();storageBarrier();';
  const phase = (name: string, count: string) => ` for(var v=0u;v<(${count}+${WG - 1}u)/${WG}u;v++){${calls[name]}}${sync}\n`;
  const once = (name: string) => ` {let v=0u;${calls[name]}}${sync}\n`;
  // Every free segment's twist is an unknown (jointless rods included, as the Float64 solve).
  const twists = 'p.segments';
  const line = ['lineTri:p.triangles', 'lineHinge:p.hinges', 'lineSeg:p.segments', 'lineJoint:p.joints', `lineTwist:${twists}`, 'lineVertex:p.n', 'linePair:p.pairs'].map(e => {const [k, c] = e.split(':'); return phase(k!, c!);}).join('')
    + ` for(var v=0u;v<p.lineGroups;v++){${calls.lineContact}}${sync}\n` + once('finishLine');
  // Every box pass before any insert or query: they size this detection's grid cell (gridCell).
  const detection = ` if(p.selfCollision!=0u){\n${once('gridClear')}${phase('boxFaces', 'p.triangles')}${phase('boxEdges', 'p.edges')}`
    + ` if(p.rodContact!=0u){\n${phase('boxSegments', 'p.segments')} }\n${phase('gridFaces', 'p.triangles')}${phase('gridEdges', 'p.edges')}`
    + ` if(p.rodContact!=0u){\n${phase('gridSegments', 'p.segments')} }\n${phase('detectFaces', 'p.n')}${phase('detectEdges', 'p.edges')}`
    + ` if(p.rodContact!=0u){\n${phase('detectRodPoints', 'p.n')}${phase('detectRodSegments', 'p.segments')} }\n }\n`
    + ` if(p.colliders>0u){\n${once('detectColliders')} if(p.rodContact!=0u){\n${once('detectRodColliders')} }\n }\n`;
  const afterDetection = once('penalties') + once('listCount') + ` listScanSmall(l);${sync}\n` + once('listFill') + phase('listReset', 'p.listRows');
  const kernel = (name: string, body: string) => `@compute @workgroup_size(${WG}) fn ${name}(@builtin(local_invocation_index) l:u32){\n${body}}\n`;
  return out + wrappers + `
// The per-row list offsets of listScan1-3 in one workgroup: each thread a run of rows
// (vertices, then rod segments' twist).
var<workgroup> scanSmall:array<u32,${WG}>;
fn listScanSmall(l:u32){
 if(uniformWord(l,${CONTACT_CONTROL.K_ADDED}u)==0u){return;}
 let rows=p.listRows;let per=(rows+${WG - 1}u)/${WG}u;var total=0u;
 for(var j=0u;j<per;j++){let i=l*per+j;if(i<rows){let value=atomicExchange(&KA[p.aCount+i],0u);KU[p.kvStart+i]=total;total+=value;}}
 scanSmall[l]=total;workgroupBarrier();
 if(l==0u){var sum=0u;for(var j=0u;j<${WG}u;j++){let t=scanSmall[j];scanSmall[j]=sum;sum+=t;}KU[p.kvStart+rows]=sum;}
 workgroupBarrier();
 let offset=scanSmall[l];for(var j=0u;j<per;j++){let i=l*per+j;if(i<rows){KU[p.kvStart+i]+=offset;}}
}
` + kernel('unitSmall', phase('elemTri', 'p.triangles') + phase('elemHinge', 'p.hinges') + phase('elemSeg', 'p.segments') + phase('elemJoint', '3u*p.joints')
    + phase('gradient', 'p.n') + phase('twistGradient', twists) + once('finishMeasure') + phase('assemble', 'p.blocks') + once('contactCoef') + phase('cgInit', 'p.n') + once('finishInit'))
    + kernel('unitSmallB', phase('twistSolve', 'p.rods') + phase('slope', 'p.n') + once('finishSlope')
    + Array.from({length: LINE_PASSES}, () => ` if(uniformWord(l,${C_SEARCH}u)!=0u){\n${line} }\n`).join('') + phase('apply', 'p.n'))
    + kernel('roundSmall', once('roundBegin') + once('updateMultipliers') + once('relinearize') + detection + once('roundDecide') + afterDetection)
    + kernel('firstSmall', phase('initX', 'p.n') + phase('prepTri', 'p.triangles') + phase('prepHinge', 'p.hinges') + phase('prepSeg', 'p.segments')
    + once('detectBegin') + once('tableClear') + detection + afterDetection);
}
export const clothNewtonWGSL = withSmallMeshKernels(clothNewtonKernels);

const ENTRY_POINTS = ['prepTri', 'prepHinge', 'prepSeg', 'initX', 'zeroX', 'predictVelocity', 'elemTri', 'elemHinge', 'elemSeg', 'elemJoint', 'twistGradient', 'twistSolve', 'lineJoint', 'lineTwist', 'gradient', 'finishMeasure', 'assemble', 'cgInit', 'finishInit',
  'unitSmall', 'unitSmallB', 'roundSmall', 'firstSmall', 'cgShared', 'cgFused', 'contactCoef', 'cgContact', 'cgMatvec', 'finishAlpha', 'cgUpdate', 'finishBeta', 'cgDirection', 'slope', 'finishSlope', 'lineTri', 'lineHinge', 'lineSeg', 'lineVertex', 'linePair', 'lineContact', 'finishLine', 'apply',
  'tableClear', 'rememberMultipliers', 'roundBegin', 'detectBegin', 'updateMultipliers', 'relinearize', 'roundDecide', 'penalties',
  'listCount', 'listScan1', 'listScan2', 'listScan3', 'listFill', 'listReset', 'gridClear', 'boxFaces', 'boxEdges', 'gridFaces', 'gridEdges', 'detectFaces', 'detectEdges', 'detectColliders',
  'boxSegments', 'gridSegments', 'detectRodPoints', 'detectRodSegments', 'detectRodColliders'] as const;
type EntryPoint = typeof ENTRY_POINTS[number];

export interface GPUClothNewtonReport extends ClothStepReport {
  readonly newtonIterations: number;
  readonly linearIterations: number;
  readonly submissions: number;
  /** Wall time of the GPU solve (upload, submissions, readbacks; with GPU
   * detection this includes every contact round) and of the step's Float64
   * finish. */
  readonly solve_ms: number;
  readonly finish_ms: number;
  /** CPU contact work between GPU solves (CPU detection only). */
  readonly contact_ms: number;
  /** GPU solves this step: one, plus one per contact (outer) iteration. */
  readonly gpuSolves: number;
  readonly contactModel: 'in-solve' | 'post-projection';
  readonly contactIterations?: number;
  /** Where in-solve contacts were detected, and (contactFallback) the CPU
   * remnant's reason when not on the GPU (clothContactCPURemnant). */
  readonly contactDetection: 'gpu' | 'cpu' | 'none';
  readonly contactFallback?: string;
  /** GPU detection: pairs held in the contact pool at the end of the step (its peak authors physics.contactCapacity). */
  readonly contactPool?: number;
  /** Swept-primitive grid entries of the step's last detection (its peak authors physics.contactGridEntries). */
  readonly contactGridFill?: number;
  /** That detection's grid cell: the rest cell, or wider than its widest swept, padded primitive box. */
  readonly contactGridCell_m?: number;
  /** The last GPU solve's status bits (converged 1, line search 2, nonfinite 4, Newton cap 8, singular 16). */
  readonly gpuStatus: number;
  /** Times a solve this step reached the Newton cap and resumed on the GPU from its iterate. */
  readonly gpuResumes: number;
}

/** One constraint of the GPU contact set (tests and diagnostics). */
export interface GPUClothContact {
  readonly key: string; readonly vertices: readonly number[]; readonly weights: readonly number[];
  readonly normal: readonly [number, number, number]; readonly lambda: number; readonly frictionLambda: number; readonly rho: number;
}

interface ContactLayout {
  kMax: number; tableSlots: number; gridBuckets: number; gridEntries: number;
  kuCV: number; kuMeta: number; kvStart: number; kvList: number; uBlocks: number; uEntry: number; uTabVal: number; kuWords: number;
  aTable: number; aHeads: number; aCount: number; kaWords: number; kfCoef: number; kfForce: number; kfFaceBox: number; kfEdgeBox: number; kfWords: number;
  kuRod: number; kfRod: number; kfRodForce: number; kfRodArm: number; kfSegBox: number; listRows: number;
}
const pow2 = (x: number) => 2 ** Math.ceil(Math.log2(Math.max(1, x)));
const errorNames: [number, string][] = [[E_CAPACITY, 'contact capacity'], [E_GRID, 'grid entries'], [E_CROSSED, 'a vertex crossed a collider within the step'], [E_STIFFNESS, 'a contact without free stiffness'], [E_TABLE, 'key table'], [E_CELLS, 'a swept primitive covers more than the grid cell cap'], [E_BOX_NONFINITE, 'a nonfinite swept primitive box'], [E_NONFINITE, 'a nonfinite contact residual'], [E_UNSUPPORTED, 'a rod segment against a collider kind without GPU rod contact']];
const describeErrors = (bits: number) => errorNames.filter(([b]) => bits & b).map(([, name]) => name).join(', ');
const statusNames: [number, string][] = [[S_CONVERGED, 'inner solve converged'], [S_LINE, 'line search found no decrease'], [S_NONFINITE, 'nonfinite gradient'], [S_CAP, 'Newton cap'], [S_SINGULAR, 'singular block']];
const describeStatus = (bits: number) => statusNames.filter(([b]) => bits & b).map(([, name]) => name).join(', ') || 'no status';
type Pairs = NonNullable<ClothStepOptions['cohesionPairs']>;
interface SolveResult {converged: boolean; status: number; measure: number; newton: number; linear: number; submissions: number; resumes: number; increments: Float32Array; twist?: Float32Array; control: Uint32Array<ArrayBuffer>}

export class GPUClothNewton {
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
  /** Newton units per submission: one more than the last solve needed, so a
   * step usually submits once and reads back once; units after convergence
   * early-exit on the device. */
  private units = 2;
  private readonly segmentRods: Int32Array;
  /** DER joints (rod, local joint index); every rod segment's twist is solved on the GPU. */
  private readonly joints: {rod: number; local: number}[] = [];
  private readonly twistRead: GPUBuffer;
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
  /** The grid holds 8 (T + E + S) entries, which each detection's cell (sized to its widest box) guarantees (CONTACT_GRID_CELL_BOUND): the default unless a scene authors a measured grid size. */
  private readonly gridCellBound: boolean;

  /** contactCapacity (pairs) and contactGridEntries (swept-primitive grid
   * entries): the contact pool. Pairs default to max(4096, 2 vertices) and the
   * grid to the two-cells-per-axis bound unless the scene authors a
   * measured peak. It never grows at run time: a step that fills it fails with
   * the measured counts. */
  constructor(private readonly device: GPUDevice, readonly topology: PreparedClothTopology, private readonly authored: {contactCapacity?: number; contactGridEntries?: number | typeof CONTACT_GRID_CELL_BOUND} = {}) {
    for (const [name, value] of Object.entries(authored)) if (value !== undefined && !(name === 'contactGridEntries' && value === CONTACT_GRID_CELL_BOUND) && !(Number.isSafeInteger(value) && (value as number) > 0)) throw new Error(`Invalid authored cloth ${name} ${value}`);
    this.gridCellBound = authored.contactGridEntries === undefined || authored.contactGridEntries === CONTACT_GRID_CELL_BOUND;
    if (!gpuClothNewtonSupported(topology)) throw new Error('GPU cloth Newton: a rod has fewer than two vertices or mismatched rest lengths');
    const n = this.n = topology.vertexCount, T = topology.triangles.length, H = topology.hinges.length, S = topology.segmentCount;
    // DER joints of rods with three or more vertices: (a, b, c, first segment);
    // each segment's joint as the second segment, then as the first.
    const jointWords: number[] = [], rodWords: number[] = [], segJoint = new Uint32Array(Math.max(4, 2 * S)).fill(NONE);
    topology.rods.forEach((rod, r) => {
      // Every rod's twist (a jointless rod's couples only to its inertia, attachment and contact).
      rodWords.push(rod.segmentOffset, rod.restLengths_m.length);
      for (let j = 1; j < rod.restLengths_m.length; j++) {
        const k = this.joints.length, s0 = rod.segmentOffset + j - 1;
        jointWords.push(rod.vertices[j - 1]!, rod.vertices[j]!, rod.vertices[j + 1]!, s0);
        segJoint[2 * (s0 + 1)] = k; segJoint[2 * s0 + 1] = k;
        this.joints.push({rod: r, local: j});
      }
    });
    const J = this.joints.length;
    if (Math.max(n, T, H, S, J) >= 1 << 24) throw new Error('GPU cloth Newton element index exceeds 24 bits');
    // Static float data after the per-step vertex block: materials, triangles, hinges, segments;
    // then the per-step segment twist and joint blocks.
    const materials = topology.materials, M = materials.length;
    const dMaterial = 4 * n, dTri = dMaterial + 3 * M, dHinge = dTri + 2 * T, dSeg = dHinge + H, dTwist = dSeg + S, dJoint = dTwist + 4 * S, dDisp = dJoint + 4 * J, dTriRest = dDisp + n, dSegRest = dTriRest + 2 * T, dHingeRest = dSegRest + S, dTotal = dHingeRest + 3 * H;
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
      // The rest frame u_r = ru.x, w_r = rv.x in Float64 (orthonormal up to roundoff).
      const rest = topology.data.restPositions_m, [a, b, c] = face.vertices;
      for (let k = 0; k < 3; k++) {
        staticD[4 * (dTriRest + 2 * t) + k] = face.ru[0] * rest[3 * a + k]! + face.ru[1] * rest[3 * b + k]! + face.ru[2] * rest[3 * c + k]!;
        staticD[4 * (dTriRest + 2 * t + 1) + k] = face.rv[0] * rest[3 * a + k]! + face.rv[1] * rest[3 * b + k]! + face.rv[2] * rest[3 * c + k]!;
      }
    });
    topology.hinges.forEach((hinge, h) => {
      staticD.set([hinge.restAngle_rad, hinge.shapeFactor, topology.triangles[hinge.faces[0]]!.material, 0], 4 * (dHinge + h));
      // Rest edges a->b, a->c, a->d differenced in Float64 (see hingeOldEdge).
      const rest = topology.data.restPositions_m, [a, b, c, d] = hinge.vertices;
      [b, c, d].forEach((v, k) => {for (let q = 0; q < 3; q++) staticD[4 * (dHingeRest + 3 * h + k) + q] = rest[3 * v! + q]! - rest[3 * a! + q]!;});
    });
    const segA = new Uint32Array(S), segB = new Uint32Array(S);
    this.segmentRods = new Int32Array(S);
    topology.rods.forEach((rod, r) => {for (let j = 0; j < rod.restLengths_m.length; j++) {const s = rod.segmentOffset + j; segA[s] = rod.vertices[j]!; segB[s] = rod.vertices[j + 1]!; this.segmentRods[s] = r; staticD.set([rod.restLengths_m[j]!, rod.material, 0, 0], 4 * (dSeg + s));
      const rest = topology.data.restPositions_m, a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
      staticD.set([rest[3 * b]! - rest[3 * a]!, rest[3 * b + 1]! - rest[3 * a + 1]!, rest[3 * b + 2]! - rest[3 * a + 2]!, 0], 4 * (dSegRest + s));}});
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
    // Joints enter the vertex lists only (their bending Hessian is diagonal, no blocks).
    const jointLocals = (k: number) => [0, 1, 2].filter(l => jointWords.slice(4 * k, 4 * k + l).indexOf(jointWords[4 * k + l]!) < 0);
    for (let k = 0; k < J; k++) for (const l of jointLocals(k)) vertexStart[jointWords[4 * k + l]! + 1]!++;
    for (let b = 0; b < blocks; b++) contribStart[b + 1]! += contribStart[b]!;
    for (let i = 0; i < n; i++) vertexStart[i + 1]! += vertexStart[i]!;
    const contrib = new Uint32Array(contribStart[blocks]!), vertexList = new Uint32Array(vertexStart[n]!), contribFill = contribStart.slice(0, blocks), vertexFill = vertexStart.slice(0, n);
    forEachElement((kind, e, vertices) => vertices.forEach((a, pa) => {
      vertexList[vertexFill[a]!++] = e | pa << 24 | kind << 28;
      vertices.forEach((b, pb) => {contrib[contribFill[blockOf(a, b)]!++] = e | pa << 24 | pb << 26 | kind << 28;});
    }));
    for (let k = 0; k < J; k++) for (const l of jointLocals(k)) {const a = jointWords[4 * k + l]!; vertexList[vertexFill[a]!++] = (k | l << 24 | KIND_JOINT << 28) >>> 0;}
    const segIds = new Uint32Array(2 * S); for (let s = 0; s < S; s++) {segIds[2 * s] = segA[s]!; segIds[2 * s + 1] = segB[s]!;}
    const triIds = new Uint32Array(4 * T); topology.triangles.forEach((face, t) => triIds.set(face.vertices, 4 * t));
    const hingeIds = new Uint32Array(4 * H); topology.hinges.forEach((hinge, h) => hingeIds.set(hinge.vertices, 4 * h));
    const parts = [triIds, hingeIds, segIds, rowStart, columns, blockRow, contribStart, contrib, vertexStart, vertexList, diagonal, Uint32Array.from(jointWords), Uint32Array.from(rodWords), segJoint], at: number[] = [];
    let total = 0; for (const part of parts) {at.push(total); total += part.length;}
    const indices = new Uint32Array(Math.max(4, total)); parts.forEach((part, k) => indices.set(part, at[k]!));
    const [iTri, iHinge, iSeg, iRow, iCol, iBlockRow, iContribStart, iContrib, iVStart, iVList, iDiag, iJoint, iRod, iSegJoint] = at as [number, number, number, number, number, number, number, number, number, number, number, number, number, number];
    // Read-write vectors (vec4): x g r z z0 p q s pre(3) scalars(4) partials.
    const wgV = Math.ceil(n / WG), pTri = wgV, pHinge = pTri + Math.ceil(T / WG), pSeg = pHinge + Math.ceil(H / WG), pJoint = pSeg + Math.ceil(S / WG), pTwist = pJoint + Math.ceil(J / WG), pPair = pTwist + Math.ceil(S / WG);
    // Read-write twist (per segment): (increment, gradient, step, d2E), (pivot, diagonal).
    // The CG vectors r z z0 p q s carry n vertex rows then S twist rows (rod contact couples them).
    const N = n + S, wX = 0, wG = n, wR = 2 * n, wZ = wR + N, wZ0 = wZ + N, wP = wZ0 + N, wQ = wP + N, wS = wQ + N, wPre = wS + N, wK = wPre + 3 * n, wT = wK + n, wT2 = wT + S, wScal = wT2 + S, wPart = wScal + 4;
    const eTri = 0, eHinge = 4 * T, eSeg = eHinge + 9 * H, eTriOld = eSeg + 2 * S, eHingeOld = eTriOld + 3 * T, eSegOld = eHingeOld + H, eHingeEdge = eSegOld + 2 * S, eJoint = eHingeEdge + 3 * H, eA = eJoint + 9 * J, eTotal = eA + 3 * blocks;
    this.offsets = Object.fromEntries(PARAM_FIELDS.map(name => [name, 0])) as Record<ParamName, number>;
    Object.assign(this.offsets, {
      n, triangles: T, hinges: H, segments: S, blocks, dMaterial, dTri, dHinge, dSeg, pTri, pHinge, pSeg, pPair, wgV,
      iTri, iHinge, iSeg, iRow, iCol, iBlockRow, iContribStart, iContrib, iVStart, iVList, iDiag,
      wX, wG, wR, wZ, wZ0, wP, wQ, wS, wPre, wPart, wScal, wK, eTri, eHinge, eSeg, eTriOld, eHingeOld, eSegOld, eA,
      joints: J, rods: rodWords.length / 2, iJoint, iRod, iSegJoint, dTwist, dJoint, wT, wT2, eJoint, pJoint, pTwist, dDisp, dTriRest, dSegRest, dHingeRest, eHingeEdge,
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
    this.twistRead = this.make('Cloth Newton twist readback', 16 * S, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
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
    // Rods: a rod record per pair (5 slots and 2 twist rows a pair), list rows n + segments, a segment grid.
    const S = this.topology.rods.length ? this.offsets.segments : 0, rod = S > 0, rows = n + S;
    if (held && held.kMax >= kMax && held.gridEntries >= gridEntries && held.tableSlots >= tableFactor * held.kMax && held.kfSegBox + 2 * S <= held.kfWords) return held;
    kMax = Math.max(kMax, held?.kMax ?? 0, 4096);
    gridEntries = Math.max(gridEntries, held?.gridEntries ?? 0, (this.gridCellBound ? 8 : 4) * (T + edges + S), 1024);
    const tableSlots = Math.max(pow2(tableFactor * kMax), held?.tableSlots ?? 0), gridBuckets = pow2(Math.max(1024, T + edges + S));
    const kuCV = 0, kuMeta = 4 * kMax, kuRod = 8 * kMax, kvStart = (rod ? 12 : 8) * kMax, kvList = kvStart + rows + 1, uBlocks = kvList + (rod ? 7 : 4) * kMax, uEntry = uBlocks + Math.ceil(rows / SCAN_WG) + 1, uTabVal = uEntry + 2 * gridEntries, kuWords = uTabVal + 2 * tableSlots;
    const aTable = 0, aHeads = 4 * tableSlots, aCount = aHeads + (rod ? 3 : 2) * gridBuckets, kaWords = aCount + rows;
    const kfCoef = 4 * kMax, kfForce = 5 * kMax, kfRod = 6 * kMax, kfRodForce = 9 * kMax, kfRodArm = 11 * kMax, kfFaceBox = (rod ? 13 : 6) * kMax, kfEdgeBox = kfFaceBox + 2 * T, kfSegBox = kfEdgeBox + 2 * edges, kfWords = kfSegBox + 2 * S;
    const layout = {kMax, tableSlots, gridBuckets, gridEntries, kuCV, kuMeta, kvStart, kvList, uBlocks, uEntry, uTabVal, kuWords, aTable, aHeads, aCount, kaWords, kfCoef, kfForce, kfFaceBox, kfEdgeBox, kfWords,
      kuRod, kfRod, kfRodForce, kfRodArm, kfSegBox, listRows: rows};
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
      if (c.rod) throw new Error(`Cloth rod contact ${c.key} is detected on the GPU only (the uploaded layout has no rolling Jacobian)`);
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
        if (!((this.topology.kinematicFixed[v] ?? 0) & 1)) list[fill[v]!++] = k << 3 | j;
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
    // Small meshes: the whole unit in one workgroup (one launch).
    if (this.small) {run('unitSmall', 1); run('cgShared', 1); run('unitSmallB', 1); return;}
    const J = o.joints, twists = o.segments;
    run('elemTri', groups(o.triangles)); run('elemHinge', groups(o.hinges)); run('elemSeg', groups(o.segments)); run('elemJoint', groups(3 * J));
    run('gradient', groups(n)); run('twistGradient', groups(twists)); run('finishMeasure', 1);
    run('assemble', groups(this.blocks)); run('contactCoef', this.contactGroups); run('cgInit', groups(n)); run('finishInit', 1);
    // Small meshes run the CG loop in one workgroup: 1 dispatch, not 5 per iteration.
    if (n <= FUSED_CG_MAX_VERTICES) run('cgFused', 1);
    else for (let k = 0; k < cgCap; k++) {run('cgContact', CONTACT_FORCE_GROUPS); run('cgMatvec', groups(n)); run('finishAlpha', 1); run('cgUpdate', groups(n)); run('finishBeta', 1); run('cgDirection', groups(n));}
    run('twistSolve', groups(o.rods));
    run('slope', groups(n)); run('finishSlope', 1);
    for (let pass = 0; pass < LINE_PASSES; pass++) {
      run('lineTri', groups(o.triangles)); run('lineHinge', groups(o.hinges)); run('lineSeg', groups(o.segments)); run('lineJoint', groups(J)); run('lineTwist', groups(twists)); run('lineVertex', groups(n)); run('linePair', groups(pairs)); run('lineContact', this.lineGroups);
      run('finishLine', 1);
    }
    run('apply', groups(n));
  }
  /** One workgroup runs each unit, contact round and first detection (see withSmallMeshKernels). */
  private get small(): boolean {return this.n <= SHARED_CG_MAX_VERTICES;}
  private get lineGroups(): number {return this.small ? 1 : this.contactGroups;}
  private run(pass: GPUComputePassEncoder, entry: EntryPoint, count: number): void {if (count > 0) {pass.setPipeline(this.pipelines![entry]); pass.dispatchWorkgroups(count);}}
  private strided(count: number): number {return Math.max(1, Math.min(4096, Math.ceil(count / WG)));}
  /** Swept-primitive grid and narrowphase (each kernel gates on K_DETECT). */
  private encodeDetection(pass: GPUComputePassEncoder): void {
    const o = this.offsets, layout = this.contactLayout!, edges = this.contactStatic!.edges, groups = (count: number) => Math.ceil(count / WG), self = this.selfCollision;
    const rods = this.topology.rods.length ? o.segments : 0;
    if (self) {
      // Every box pass before any insert or query: they size this detection's grid cell (gridCell).
      this.run(pass, 'gridClear', this.strided((rods ? 3 : 2) * layout.gridBuckets)); this.run(pass, 'boxFaces', groups(o.triangles)); this.run(pass, 'boxEdges', groups(edges)); this.run(pass, 'boxSegments', groups(rods));
      this.run(pass, 'gridFaces', groups(o.triangles)); this.run(pass, 'gridEdges', groups(edges)); this.run(pass, 'gridSegments', groups(rods));
      this.run(pass, 'detectFaces', groups(this.n)); this.run(pass, 'detectEdges', groups(edges));
      if (rods) {this.run(pass, 'detectRodPoints', groups(this.n)); this.run(pass, 'detectRodSegments', groups(rods));}
    }
    if (this.colliderCount) this.run(pass, 'detectColliders', this.strided(this.n * this.colliderCount));
    if (this.colliderCount && rods) this.run(pass, 'detectRodColliders', this.strided(rods * this.colliderCount));
  }
  /** Penalties of new pairs, friction lag, and the per-vertex lists when pairs were added. */
  private encodeAfterDetection(pass: GPUComputePassEncoder): void {
    const rows = this.contactLayout!.listRows, scanGroups = Math.ceil(rows / SCAN_WG);
    this.run(pass, 'penalties', this.contactGroups);
    this.run(pass, 'listCount', this.contactGroups); this.run(pass, 'listScan1', scanGroups); this.run(pass, 'listScan2', 1); this.run(pass, 'listScan3', scanGroups);
    this.run(pass, 'listFill', this.contactGroups); this.run(pass, 'listReset', Math.ceil(rows / WG));
  }
  private encodeRound(pass: GPUComputePassEncoder): void {
    if (this.small) {this.run(pass, 'roundSmall', 1); return;}
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
    // The old frame's displacement from rest, rounded once from Float64: the
    // old strains are measured from it. The shader compiler reassociates the
    // hi/lo sums, so old edges carry an absolute rounding floor of the
    // frame-relative positions (strain ~1e-8 at rest, stressing a sheet at rest).
    const origin = this.frame, vertex = new Float32Array(16 * n), predicted = new Float64Array(3 * n), displacement = new Float32Array(4 * n), rest = topology.data.restPositions_m;
    for (let i = 0; i < n; i++) {
      const mass = state.dryMass_kg[i]! + state.waterVolume_m3[i]! * topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3, fixed = (topology.kinematicFixed[i] ?? 0) & 1;
      if (!(mass > 0 && Number.isFinite(mass))) throw new Error(`Invalid cloth vertex mass ${i}`);
      const o = 16 * i;
      for (let a = 0; a < 3; a++) {
        const k = 3 * i + a, increment = dt * state.velocities_m_s[k]! + dt * dt * (gravity[a]! + (options.externalForces_N?.[k] ?? 0) / mass), shifted = old[k]! - origin[a]!, high = Math.fround(shifted);
        predicted[k] = old[k]! + increment;
        vertex[o + a] = increment; vertex[o + 4 + a] = target[k]! - old[k]!; vertex[o + 8 + a] = high; vertex[o + 12 + a] = shifted - high; displacement[4 * i + a] = old[k]! - rest[k]!;
      }
      vertex[o + 3] = mass / (dt * dt); vertex[o + 7] = topology.attachmentTranslation_N_m[i]!; vertex[o + 11] = fixed; vertex[o + 15] = state.solidFraction[i]!;
    }
    this.writeWords(this.D, vertex);
    this.writeWords(this.D, displacement, 4 * this.offsets.dDisp);
    return {old, predicted, vertex};
  }
  /** Rod twist (every segment) and DER joint data of this step, in Float64 then
   * rounded: frames, cos/sin of the accepted twist (increments are solved),
   * old curvature and twist, and the twist offsets relative to the wrapped
   * prior reference twist. */
  private uploadRodBlock(state: ClothState, dt: number): void {
    const J = this.joints.length, topology = this.topology, S = topology.segmentCount;
    if (!S) return;
    const data = new Float32Array(16 * (S + J)), wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
    for (let s = 0; s < S; s++) {
      const r = this.segmentRods[s]!, rod = topology.rods[r]!, j = s - rod.segmentOffset, m = topology.materials[rod.material]!, a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
      const free = ((topology.kinematicFixed[a] ?? 0) | (topology.kinematicFixed[b] ?? 0)) & 2 ? 0 : 1, theta = state.twist_rad[s]!, inertia = rodSegmentInertia_kg_m2(topology, state, r, j) / (dt * dt);
      data.set([state.rodDirectors[3 * s]!, state.rodDirectors[3 * s + 1]!, state.rodDirectors[3 * s + 2]!, Math.cos(theta), state.rodTangents[3 * s]!, state.rodTangents[3 * s + 1]!, state.rodTangents[3 * s + 2]!, Math.sin(theta),
        inertia, topology.attachmentTwist_N_m[s]!, theta - state.attachmentTwistTarget_rad[s]!, dt * state.angularVelocity_rad_s[s]!, free, Math.max(m.rodRadius_m, m.rodRadiusB_m ?? m.rodRadius_m), 0, 0], 16 * s);
    }
    this.joints.forEach(({rod: r, local: j}, k) => {
      const rod = topology.rods[r]!, m = topology.materials[rod.material]!, v = rod.vertices, f = state.solidFraction;
      const rigidity = rodRigidity(m, m.attachmentMultiplier !== undefined ? f[v[j]!]! : (f[v[j - 1]!]! + f[v[j]!]! + f[v[j + 1]!]!) / 3), length = (rod.restLengths_m[j - 1]! + rod.restLengths_m[j]!) / 2;
      const old = rodJoint(rod, j, state);
      if (!old) throw new Error(`Cloth rod ${r} joint ${j} is degenerate`);
      const s0 = rod.segmentOffset + j - 1, prior = state.rodReferenceTwist_rad[s0 + 1]!, base = state.twist_rad[s0 + 1]! - state.twist_rad[s0]! + prior;
      data.set([...rod.restCurvature.subarray(4 * (j - 1), 4 * j), ...old.curvature, rigidity.bend1_N_m2 / length, rigidity.bend2_N_m2 / length, rigidity.twist_N_m2 / length, (m.rodBendingDamping_s ?? m.mechanicalDamping_s) / dt,
        (m.rodTwistDamping_s ?? m.mechanicalDamping_s) / dt, base - rod.restTwist[j - 1]!, base - old.twist, wrap(prior)], 16 * (S + k));
    });
    this.writeWords(this.D, data, 4 * this.offsets.dTwist);
  }
  private async readTwist(): Promise<Float32Array | undefined> {
    if (!this.offsets.segments) return undefined;
    await this.twistRead.mapAsync(GPUMapMode.READ);
    const twist = new Float32Array(this.twistRead.getMappedRange().slice(0));
    this.twistRead.unmap();
    return twist;
  }
  private writeParams(values: Record<string, number>): void {
    const words = new ArrayBuffer(4 * (PARAM_FIELDS.length + FLOAT_FIELDS.length)), u = new Uint32Array(words), f = new Float32Array(words);
    PARAM_FIELDS.forEach((name, k) => {u[k] = values[name] ?? 0;});
    FLOAT_FIELDS.forEach((name, k) => {f[PARAM_FIELDS.length + k] = values[name] ?? 0;});
    this.device.queue.writeBuffer(this.params, 0, words);
  }
  /** A detection's grid cell from its control words, as gridCell computes it. */
  private gridCell(c: Uint32Array): number {
    const width = new Float32Array(Uint32Array.of(c[CONTACT_CONTROL.K_BOX_WIDTH]!).buffer)[0]!;
    return Math.max(Math.fround(this.contactStatic!.cellSize), Math.fround(width * (1 + GRID_CELL_EPS)));
  }
  /** The first primitive whose swept box is not finite. */
  private describeNonfiniteBox(c: Uint32Array): string {
    const word = c[CONTACT_CONTROL.K_BOX_NONFINITE]!;
    if (!word) return '; the offending primitive was not recorded';
    return `; ${['triangle', 'edge', 'rod segment', 'primitive'][word >>> 30]} ${(word & 0x3fffffff) - 1} has a nonfinite swept box`;
  }
  private contactParams(): Record<string, number> {
    const layout = this.contactLayout, statics = this.contactStatic;
    if (!layout) return {};
    const base = this.indexWords.length, shifted = statics ? Object.fromEntries(Object.entries(statics.offsets).map(([k, v]) => [k, base + v])) : {};
    return {...layout, ...(statics ? {...shifted, edges: statics.edges, cellSize: statics.cellSize, sCollider: base + statics.words.length} : {}), tableCur: this.tableCur, prevValid: this.prevValid ? 1 : 0, stiffnessValid: this.stiffnessValid ? 1 : 0};
  }
  /** Continue a solve stopped at the Newton cap: the next unit measures the
   * GPU iterate and steps on with a fresh cap (the totals keep counting). */
  private resume(): void {
    const words = new Uint32Array(2); // C_NEWTON, C_STATUS
    this.device.queue.writeBuffer(this.C, 4 * C_NEWTON, words);
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
  /** The control words and the iterate of a submission, mapped together
   * (one wait): the iterate is discarded unless the solve stopped. */
  private async readSolve(stiffness: boolean): Promise<{control: Uint32Array<ArrayBuffer>; increments: Float32Array; twist?: Float32Array; stiffness?: Float32Array}> {
    const twist = this.offsets.segments > 0, maps = [this.statusRead, this.positionRead, ...(twist ? [this.twistRead] : []), ...(stiffness ? [this.stiffnessRead] : [])];
    await Promise.all(maps.map(b => b.mapAsync(GPUMapMode.READ)));
    const copy = (b: GPUBuffer) => b.getMappedRange().slice(0);
    const result = {control: new Uint32Array(copy(this.statusRead)), increments: new Float32Array(copy(this.positionRead)),
      ...(twist ? {twist: new Float32Array(copy(this.twistRead))} : {}), ...(stiffness ? {stiffness: new Float32Array(copy(this.stiffnessRead))} : {})};
    for (const b of maps) b.unmap();
    return result;
  }
  /** Units for the next solve's submissions: this solve's working units (the
   * contact path's round decision runs in one more), with one to spare. An
   * idle unit costs early-exit launches; a short submission costs a readback.
   * That holds for the one-workgroup small-mesh unit (three launches); a large
   * mesh's unit encodes six launches per CG iteration up to cgCap (up to 2048),
   * so its idle units cost more than a readback and it keeps two per submission. */
  private nextUnits(control: Uint32Array, contact: boolean): void {
    this.units = this.small ? Math.max(2, Math.min(MAX_UNITS, control[C_WORKED]! + (contact ? 1 : 0) + 1)) : 2;
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
    this.uploadRodBlock(state, dt);
    const target = options.supportPositions_m ?? topology.data.restPositions_m;
    const pairs = options.cohesionPairs ?? [], newtonCap = options.iterations ?? 64, tolerance = options.tolerance_m ?? 1e-8;
    if (!Number.isInteger(newtonCap) || newtonCap < 1) throw new Error('Cloth iterations must be a positive integer');
    const colliders = options.colliders ?? [], selfCollision = options.selfCollision ?? false;
    // Contact inside the minimization, detected and solved on the GPU: the
    // outer iterations stay on the device, and any contact error fails the
    // step, a full pool included (the pool is authored per scene from its
    // measured peak and never grows at run time). Only the CPU remnant
    // (clothContactCPURemnant) leaves the GPU: there rods keep the
    // post-projection and other cloth the Float64 set, which detects and
    // updates multipliers between GPU solves.
    const rods = topology.rods.length > 0, contact = colliders.length > 0 || selfCollision;
    const remnant = contact ? clothContactCPURemnant(topology, colliders) : undefined;
    const inSolve = contact && !inSolveContactUnsupported(topology, colliders) && !(rods && remnant);
    const fallback = inSolve ? remnant : undefined;
    const x = predicted.slice();
    const positions = (increments: Float32Array) => {for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) x[3 * i + a] = old[3 * i + a]! + increments[4 * i + a]!;};
    let solve: SolveResult | undefined, contact_ms = 0, solves = 1, outer = 0, contactResidual = 0, contactReport: ClothContactReport | undefined;
    let newton = 0, linear = 0, submissions = 0, resumes = 0, contactPool = 0, contactGridFill = 0, contactGridCell_m = 0;
    if (inSolve && !fallback) {
      const frictionEpsilon = FRICTION_SLIP_SPEED_M_S * dt;
      solve = await this.solveWithGPUContact(pairs, old, {newtonCap, tolerance, dt, frictionEpsilon, colliders, selfCollision});
      newton += solve.newton; linear += solve.linear; submissions += solve.submissions; resumes += solve.resumes;
      // Any contact error fails the step: a crossing, a contact the solve cannot
      // stiffen, a nonfinite iterate, or a pool the step filled (no growth, no
      // CPU re-solve).
      const c = solve.control, errors = c[CONTACT_CONTROL.K_ERROR]!, layout = this.contactLayout!;
      if (errors) throw new Error(`Cloth GPU contact: ${describeErrors(errors)} (${c[CONTACT_CONTROL.K_COUNT]!} pairs against the authored ${layout.kMax}, ${c[CONTACT_CONTROL.K_ENTRIES]!} grid entries against ${layout.gridEntries}, a ${layout.tableSlots}-slot key table)${errors & E_BOX_NONFINITE ? this.describeNonfiniteBox(c) : ''}`);
      if (solve) {
        positions(solve.increments);
        const c = solve.control, f = (word: number) => new Float32Array(Uint32Array.of(c[word]!).buffer)[0]!;
        outer = c[CONTACT_CONTROL.K_OUTER]!; contactPool = c[CONTACT_CONTROL.K_COUNT]!; contactGridFill = c[CONTACT_CONTROL.K_ENTRIES]!; contactGridCell_m = this.gridCell(c); solves = 1 + c[CONTACT_CONTROL.K_SOLVES]!; contactResidual = f(CONTACT_CONTROL.K_RESID);
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
      newton += solve.newton; linear += solve.linear; submissions += solve.submissions; resumes += solve.resumes;
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
        newton += solve.newton; linear += solve.linear; submissions += solve.submissions; resumes += solve.resumes; solves++;
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
    // The GPU's twist increments (zero where the twist is held).
    for (let s = 0; s < state.twist_rad.length; s++) state.twist_rad[s] = oldTwist[s]! + solve.twist![4 * s]!;
    const contactDetection = !inSolve ? 'none' as const : fallback ? 'cpu' as const : 'gpu' as const;
    const extra = {gpuSolves: solves, contact_ms, submissions, solve_ms, contactDetection, gpuStatus: solve.status, gpuResumes: resumes, ...(inSolve && !fallback ? {contactPool, contactGridFill, contactGridCell_m} : {}), ...(fallback && inSolve ? {contactFallback: fallback} : {})};
    if (!converged) {
      // The GPU solve stalled (line search, or the cap after its resumes),
      // broke, or the contact iterations did not settle. The step fails with
      // what the GPU reported; nothing finishes it on the CPU.
      state.positions_m.set(old); state.twist_rad.set(oldTwist);
      throw new Error(`GPU cloth Newton did not converge: ${describeStatus(solve.status)} (measure ${solve.measure} m, tolerance ${tolerance} m, ${newton} Newton iterations, ${resumes} resumes`
        + `${inSolve ? `, contact residual ${contactResidual} m after ${outer + 1} contact iterations` : ''})`);
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

  /** The explicit predictor velocity of nonFluidClothPredictor
   * (lib/cloth/dynamics.ts, the Float64 oracle) on the GPU: one submission
   * evaluates the elastic, attachment and cohesion gradient at the accepted
   * positions; one readback returns v + dt (g + (f - dE/dx)/m). The state is
   * not changed. */
  async predictVelocity(state: ClothState, dt: number, options: ClothStepOptions = {}): Promise<Float64Array> {
    if (!this.pipelines || !this.bindings) throw new Error('Initialize GPU cloth Newton before predicting');
    if (!(dt > 0 && Number.isFinite(dt))) throw new Error('Invalid cloth GPU predictor timestep');
    const n = this.n, o = this.offsets, {old} = this.uploadVertexBlock(state, dt, options);
    this.uploadRodBlock(state, dt);
    const pairs = options.cohesionPairs ?? [], pairOffsets = this.uploadPairs(pairs, old);
    this.writeParams({...this.offsets, ...pairOffsets, pairs: pairs.length, dt, contactMode: 0});
    const groups = (count: number) => Math.ceil(count / WG), encoder = this.device.createCommandEncoder({label: 'Cloth explicit predictor'});
    encoder.clearBuffer(this.C);
    const pass = encoder.beginComputePass({label: 'Cloth predictor gradient'});
    pass.setBindGroup(0, this.bindings);
    for (const [entry, count] of [['prepTri', o.triangles], ['prepHinge', o.hinges], ['prepSeg', o.segments], ['zeroX', Math.max(n, o.segments)],
      ['elemTri', o.triangles], ['elemHinge', o.hinges], ['elemSeg', o.segments], ['elemJoint', 3 * o.joints], ['predictVelocity', n]] as const) this.run(pass, entry, groups(count));
    pass.end();
    encoder.copyBufferToBuffer(this.W, 16 * o.wS, this.positionRead, 0, 16 * n);
    this.device.queue.submit([encoder.finish()]);
    const velocity = await this.readIncrements(), result = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) result[3 * i + a] = velocity[4 * i + a]!;
    if (!result.every(Number.isFinite)) throw new Error('Nonfinite GPU cloth predictor');
    return result;
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
    this.ensureContactCapacity(Math.max(4096, 2 * this.n, this.authored.contactCapacity ?? 0), this.gridCellBound ? 0 : this.authored.contactGridEntries as number | undefined ?? 0);
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
    this.writeParams({...this.offsets, ...pairOffsets, ...this.contactParams(), pairs: pairs.length, pContact: o.pPair + pairGroups, partTotal: o.pPair + pairGroups + this.lineGroups,
      cgCap: this.cgCap, newtonCap: settings.newtonCap, dt: settings.dt, tolerance: settings.tolerance, contactDelta: settings.tolerance, frictionEpsilon: settings.frictionEpsilon,
      contactMode: 1, contactGroups: this.contactGroups, lineGroups: this.lineGroups, colliders: this.colliderCount, selfCollision: settings.selfCollision ? 1 : 0, rodContact: this.topology.rods.length ? 1 : 0});
    this.device.queue.writeBuffer(this.W, 16 * o.wScal, new Float32Array(16));
    const groups = (count: number) => Math.ceil(count / WG), unitsPerSubmission = this.units;
    let submissions = 0, resumes = 0, control = new Uint32Array(C_WORDS), read: Awaited<ReturnType<GPUClothNewton['readSolve']>> | undefined;
    const K = CONTACT_CONTROL;
    for (;;) {
      const encoder = this.device.createCommandEncoder({label: 'Cloth Newton-PCG with GPU contact'});
      if (!submissions) {encoder.clearBuffer(this.C); encoder.clearBuffer(this.KU, 4 * layout.kvStart, 4 * (layout.listRows + 1));}
      const pass = encoder.beginComputePass({label: 'Cloth Newton iterations and contact rounds'});
      pass.setBindGroup(0, this.bindings!);
      for (let k = 0; k < unitsPerSubmission; k++) {
        if (!submissions && !k && this.small) this.run(pass, 'firstSmall', 1);
        else if (!submissions && !k) {
          for (const [entry, count] of [['initX', n], ['prepTri', o.triangles], ['prepHinge', o.hinges], ['prepSeg', o.segments]] as const) this.run(pass, entry, groups(count));
          this.run(pass, 'detectBegin', 1); this.run(pass, 'tableClear', this.strided(2 * layout.tableSlots));
          this.encodeDetection(pass); this.encodeAfterDetection(pass);
        } else this.encodeRound(pass);
        this.encodeUnit(pass, this.cgCap, pairs.length);
      }
      // Gated on the device: it runs only once the contact rounds succeeded.
      this.run(pass, 'rememberMultipliers', this.contactGroups);
      pass.end();
      encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
      encoder.copyBufferToBuffer(this.W, 16 * o.wX, this.positionRead, 0, 16 * n);
      if (o.segments) encoder.copyBufferToBuffer(this.W, 16 * o.wT, this.twistRead, 0, 16 * o.segments);
      this.device.queue.submit([encoder.finish()]);
      submissions++;
      read = await this.readSolve(false);
      control = read.control;
      const status = control[C_STATUS]!;
      if (status === S_CAP && !control[K.K_ERROR] && !control[K.K_DONE] && resumes < NEWTON_RESUMES) {resumes++; this.resume(); continue;}
      if (control[K.K_ERROR] || control[K.K_DONE] || (status && status !== S_CONVERGED)) break;
    }
    const success = control[K.K_DONE] !== 0 && !control[K.K_ERROR] && control[C_STATUS] === S_CONVERGED;
    const {increments, twist} = read!;
    this.nextUnits(control, true);
    // The table now holds this step's multipliers; the next step reads it as the previous one.
    if (success) {this.tableCur = 1 - this.tableCur; this.prevValid = true;} else this.prevValid = false;
    this.adaptCG(control);
    const measure = new Float32Array(Uint32Array.of(control[C_LAST]!).buffer)[0]!;
    return {converged: success, status: control[C_STATUS]!, measure, newton: control[C_NEWTON_TOTAL]!, linear: control[C_LINEAR]!, submissions, resumes, increments, twist, control};
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
    this.writeParams({...this.offsets, ...pairOffsets, ...(contacts ? this.contactParams() : {}), pairs: pairs.length, pContact: o.pPair + pairGroups, partTotal: o.pPair + pairGroups + this.lineGroups,
      cgCap: this.cgCap, newtonCap: settings.newtonCap, dt: settings.dt, tolerance: settings.tolerance, contactDelta: settings.tolerance, frictionEpsilon: settings.frictionEpsilon,
      contactMode: contacts ? 2 : 0, uploadedContacts: contacts?.length ?? 0, contactGroups: this.contactGroups, lineGroups: this.lineGroups});
    this.device.queue.writeBuffer(this.W, 16 * o.wScal, new Float32Array(16));
    const groups = (count: number) => Math.ceil(count / WG), unitsPerSubmission = this.units;
    let submissions = 0, status = 0, resumes = 0, control = new Uint32Array(C_WORDS), read: Awaited<ReturnType<GPUClothNewton['readSolve']>> | undefined;
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
      if (o.segments) encoder.copyBufferToBuffer(this.W, 16 * o.wT, this.twistRead, 0, 16 * o.segments);
      if (settings.readStiffness) encoder.copyBufferToBuffer(this.W, 16 * o.wK, this.stiffnessRead, 0, 16 * n);
      this.device.queue.submit([encoder.finish()]);
      submissions++;
      read = await this.readSolve(settings.readStiffness);
      control = read.control;
      status = control[C_STATUS]!;
      if (status === S_CAP && resumes < NEWTON_RESUMES) {resumes++; this.resume(); continue;}
      if (status) break;
    }
    this.adaptCG(control);
    this.nextUnits(control, false);
    const measure = new Float32Array(Uint32Array.of(control[C_LAST]!).buffer)[0]!;
    const {increments, twist} = read!;
    if (settings.readStiffness) {
      const k = read!.stiffness!;
      const stiffness = this.stiffness ??= new Float64Array(3 * n);
      for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) stiffness[3 * i + a] = k[4 * i + a]!;
    }
    return {converged: (status & S_CONVERGED) !== 0, status, measure, newton: control[C_NEWTON_TOTAL]!, linear: control[C_LINEAR]!, submissions, resumes, increments, twist, control};
  }

  /** Diagnostics: the GPU's first detection of a step from `state` (at the
   * prediction), without solving. The table and multipliers of the previous
   * step are left intact. */
  async detectContacts(state: ClothState, dt: number, options: ClothStepOptions = {}): Promise<{contacts: GPUClothContact[]; detect_ms: number; errors: number; gridEntries: number; gridCell_m: number}> {
    if (!this.pipelines || !this.bindings) throw new Error('Initialize GPU cloth Newton before detecting');
    const colliders = options.colliders ?? [], unsupported = gpuContactUnsupported(this.topology, colliders);
    if (unsupported) throw new Error(`GPU cloth contact: ${unsupported}`);
    this.uploadVertexBlock(state, dt, options);
    this.uploadRodBlock(state, dt);
    this.prepareGPUContact(colliders, options.selfCollision ?? false);
    const n = this.n, layout = this.contactLayout!;
    this.writeParams({...this.offsets, ...this.contactParams(), contactMode: 1, contactGroups: this.contactGroups, colliders: this.colliderCount, selfCollision: this.selfCollision ? 1 : 0, rodContact: this.topology.rods.length ? 1 : 0,
      dt, tolerance: options.tolerance_m ?? 1e-8, contactDelta: options.tolerance_m ?? 1e-8, frictionEpsilon: FRICTION_SLIP_SPEED_M_S * dt, newtonCap: 1, cgCap: 1});
    const begun = performance.now(), encoder = this.device.createCommandEncoder({label: 'Cloth contact detection'});
    encoder.clearBuffer(this.C); encoder.clearBuffer(this.KU, 4 * layout.kvStart, 4 * (layout.listRows + 1));
    const pass = encoder.beginComputePass(); pass.setBindGroup(0, this.bindings!);
    this.run(pass, 'initX', Math.ceil(Math.max(n, this.offsets.segments) / WG)); this.run(pass, 'prepSeg', Math.ceil(this.offsets.segments / WG)); this.run(pass, 'detectBegin', 1); this.run(pass, 'tableClear', this.strided(2 * layout.tableSlots));
    this.encodeDetection(pass); this.encodeAfterDetection(pass);
    pass.end();
    encoder.copyBufferToBuffer(this.C, 0, this.statusRead, 0, 4 * C_WORDS);
    this.device.queue.submit([encoder.finish()]);
    const control = await this.readControl(), detect_ms = performance.now() - begun;
    return {contacts: await this.readContactSet(Math.min(control[CONTACT_CONTROL.K_COUNT]!, layout.kMax)), detect_ms, errors: control[CONTACT_CONTROL.K_ERROR]!, gridEntries: control[CONTACT_CONTROL.K_ENTRIES]!, gridCell_m: this.gridCell(control)};
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
    const layout = this.contactLayout!, rod = this.topology.rods.length > 0;
    const floats = this.device.createBuffer({size: 64 * count + (rod ? 48 * count : 0), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}), words = this.device.createBuffer({size: (rod ? 48 : 32) * count, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.KF, 0, floats, 0, 64 * count);
      encoder.copyBufferToBuffer(this.KU, 4 * layout.kuCV, words, 0, 16 * count);
      encoder.copyBufferToBuffer(this.KU, 4 * layout.kuMeta, words, 16 * count, 16 * count);
      if (rod) {encoder.copyBufferToBuffer(this.KF, 16 * layout.kfRod, floats, 64 * count, 48 * count); encoder.copyBufferToBuffer(this.KU, 4 * layout.kuRod, words, 32 * count, 16 * count);}
      this.device.queue.submit([encoder.finish()]);
      await Promise.all([floats.mapAsync(GPUMapMode.READ), words.mapAsync(GPUMapMode.READ)]);
      const f = new Float32Array(floats.getMappedRange().slice(0)), u = new Uint32Array(words.getMappedRange().slice(0));
      // Keys of ClothContactSet: kind << 28 | first feature (lib/cloth/gpu-contact.ts).
      const prefixes = ['c', 'v', 'e', 'rc', 'rv', 'rq', 'rp', 'rf', 'rx', 'rs'];
      return Array.from({length: count}, (_, c) => {
        const k0 = u[4 * count + 4 * c]!, k1 = u[4 * count + 4 * c + 1]!, kind = k0 >>> 28, a = k0 & 0x0fffffff, prefix = prefixes[kind];
        if (!prefix) throw new Error(`GPU cloth contact ${c} has unknown kind ${kind}`);
        const rodm = rod ? u[8 * count + 4 * c + 3]! : 0, size = rodm ? rodm & 7 : kind === 0 ? 1 : 4;
        const vertices = Array.from(u.subarray(4 * c, 4 * c + Math.min(4, size))), weights = Array.from(f.subarray(16 * c + 12, 16 * c + 12 + Math.min(4, size)));
        if (size === 5) {vertices.push(u[8 * count + 4 * c]!); weights.push(f[16 * count + 12 * c + 8]!);}
        return {key: `${prefix}:${a}:${k1}`, vertices, weights, normal: [f[16 * c]!, f[16 * c + 1]!, f[16 * c + 2]!] as const,
          lambda: f[16 * c + 7]!, rho: f[16 * c + 8]!, frictionLambda: f[16 * c + 9]!};
      });
    } finally {floats.destroy(); words.destroy();}
  }

  destroy(): void {for (const buffer of this.buffers) buffer.destroy();}
}
