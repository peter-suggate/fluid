import type { ClothMaterial, ClothPhysicsOptions, ClothState, ClothTopologyData, ClothVector } from './types';
import { porousDragCoefficients, porousSuctionPressure } from './porous-material';

interface FlowEdge {a:number;b:number;ab:number;ba:number;source?:number}
interface SolveResult {x:Float64Array;iterations:number;residual:number}

export interface ManifoldVolumeFlux {readonly a:number;readonly b:number;readonly volume_m3:number}
/** Positive flux travels a->b. A nonmonotone FEM candidate may undershoot
 * at a wet front. Limit the paired donor flux, never clamp one node and lose
 * its opposite transfer. Smooth positive candidates are unchanged. */
export function acceptPositiveManifoldFlux(old:Float64Array,proposed:readonly ManifoldVolumeFlux[]):{water:Float64Array;fluxes:ManifoldVolumeFlux[];limited:boolean}{
 if(old.some(v=>!Number.isFinite(v)||v<0))throw new Error('Invalid manifold water inventory');
 const water=old.slice(),outgoing=new Float64Array(old.length);
 for(const f of proposed){if(!Number.isFinite(f.volume_m3)||!Number.isInteger(f.a)||!Number.isInteger(f.b)||f.a<0||f.b<0||f.a>=old.length||f.b>=old.length)throw new Error('Invalid manifold volume flux');water[f.a]!-=f.volume_m3;water[f.b]!+=f.volume_m3;outgoing[f.volume_m3>=0?f.a:f.b]!+=Math.abs(f.volume_m3);}
 const tolerance=1e-13*Math.max(1e-30,old.reduce((a,b)=>a+b,0));
 if(water.every(v=>v>= -tolerance))return {water,fluxes:[...proposed],limited:false};
 const fraction=Float64Array.from(old,(v,i)=>outgoing[i]!>v?Math.max(0,v/outgoing[i]!):1);
 const fluxes=proposed.map(f=>({...f,volume_m3:f.volume_m3*fraction[f.volume_m3>=0?f.a:f.b]!}));water.set(old);
 for(const f of fluxes){water[f.a]!-=f.volume_m3;water[f.b]!+=f.volume_m3;}
 if(water.some(v=>v< -tolerance))throw new Error('Paired manifold positivity limiter failed');
 return {water,fluxes,limited:true};
}

/** Jacobi-preconditioned BiCGSTAB on the conservative FEM mass matrix.
 * Monotone metrics give an M-matrix. A deformed non-Delaunay metric retains
 * its signed coefficients and passes its fluxes through the paired limiter. */
interface EdgeArrays {readonly a:Int32Array;readonly b:Int32Array;readonly ab:Float64Array;readonly ba:Float64Array;readonly source?:Float64Array}
function solveMass(capacity:Float64Array,rhs:Float64Array,edges:EdgeArrays,initial:Float64Array,fixed?:ArrayLike<number>):SolveResult{
 const n=capacity.length,m=edges.a.length,ea=edges.a,eb=edges.b,ab=edges.ab,ba=edges.ba,source=edges.source,x=initial.slice(),diag=capacity.slice(),fixedMask=new Uint8Array(n);
 if(fixed)for(let i=0;i<n;i++)fixedMask[i]=((fixed[i]??0)&1)!==0?1:0;
 const adjusted=rhs.slice();if(source)for(let k=0;k<m;k++){const q=source[k]!;if(q){const a=ea[k]!,b=eb[k]!;if(!fixedMask[a])adjusted[a]!-=q;if(!fixedMask[b])adjusted[b]!+=q;}}
 for(let k=0;k<m;k++){diag[ea[k]!]!+=ab[k]!;diag[eb[k]!]!+=ba[k]!;}for(let i=0;i<n;i++)if(fixedMask[i])diag[i]=1;
 const apply=(v:Float64Array,out:Float64Array)=>{
  for(let i=0;i<n;i++)out[i]=capacity[i]!*v[i]!;
  for(let k=0;k<m;k++){const a=ea[k]!,b=eb[k]!,flux=ab[k]!*v[a]!-ba[k]!*v[b]!;out[a]!+=flux;out[b]!-=flux;}
  for(let i=0;i<n;i++)if(fixedMask[i])out[i]=v[i]!;
 };
 const dot=(a:Float64Array,b:Float64Array)=>{let s=0;for(let i=0;i<n;i++)s+=a[i]!*b[i]!;return s;};
 const norm=(a:Float64Array)=>{let m=0;for(let i=0;i<n;i++)m=Math.max(m,Math.abs(a[i]!));return m;};
 const r=new Float64Array(n),hat=new Float64Array(n),p=new Float64Array(n),v=new Float64Array(n),s=new Float64Array(n),y=new Float64Array(n),z=new Float64Array(n),t=new Float64Array(n);
 apply(x,v);for(let i=0;i<n;i++)r[i]=hat[i]=adjusted[i]!-v[i]!;v.fill(0);
 const tolerance=1e-11*Math.max(norm(adjusted),1e-30);let residual=norm(r),rhoOld=1,alpha=1,omega=1,iterations=0;
 while(residual>tolerance&&iterations<2000){
  const rho=dot(hat,r);if(!Number.isFinite(rho)||rho===0)throw new Error('Pore transport Krylov breakdown');
  const beta=(rho/rhoOld)*(alpha/omega);
  for(let i=0;i<n;i++){p[i]=r[i]!+beta*(p[i]!-omega*v[i]!);y[i]=p[i]!/diag[i]!;}
  apply(y,v);const divisor=dot(hat,v);if(divisor===0)throw new Error('Pore transport singular direction');alpha=rho/divisor;
  for(let i=0;i<n;i++)s[i]=r[i]!-alpha*v[i]!;
  if(norm(s)<=tolerance){for(let i=0;i<n;i++)x[i]!+=alpha*y[i]!;residual=norm(s);iterations++;break;}
  for(let i=0;i<n;i++)z[i]=s[i]!/diag[i]!;
  apply(z,t);const tt=dot(t,t);if(tt===0)throw new Error('Pore transport null drift direction');omega=dot(t,s)/tt;
  if(!Number.isFinite(omega)||omega===0)throw new Error('Pore transport stagnated');
  for(let i=0;i<n;i++){x[i]!+=alpha*y[i]!+omega*z[i]!;r[i]=s[i]!-omega*t[i]!;}
  residual=norm(r);rhoOld=rho;iterations++;
 }
 if(residual>tolerance)throw new Error(`Pore transport failed to converge (${residual})`);
 return {x,iterations,residual};
}

export interface PoreFlowStep {state:ClothState;data:ClothTopologyData;kinematicFixed?:ArrayLike<number>;materials:readonly ClothMaterial[];dt_s:number;gravity_m_s2:ClothVector;/** Outer fluid timestep used by the released source nonlinear drag predictor. */manifoldPredictorDt_s?:number;physics?:ClothPhysicsOptions;pressureAcceleration_m_s2?:ArrayLike<number>;solidAcceleration_m_s2?:ArrayLike<number>;/** Parity output: accepted volume a->b of each edge (key a*n+b, a<b), summed over substeps. */edgeFlux_m3?:Map<number,number>}
export interface PoreFlowResult {readonly iterations:number;readonly movedVolume_m3:number;readonly limitedVolume_m3:number;readonly volumeError_m3:number;readonly supportReaction_Ns:ClothVector}

/** Source LinearizedImplicitEuler.cpp2668: full vector force / fluid mass,
 * multiplied by void fraction and OUTER dt, even inside manifold substeps. */
export function manifoldForcePredictorSpeed(solidFraction:number,dt_s:number,forcePerFluidMass_m_s2:readonly number[]):number{
 if(!(solidFraction>=0&&solidFraction<=1)||!(dt_s>=0)||!Number.isFinite(dt_s)||forcePerFluidMass_m_s2.length!==3||forcePerFluidMass_m_s2.some(v=>!Number.isFinite(v)))throw new Error('Invalid manifold Gauss force predictor');
 return (1-solidFraction)*dt_s*Math.hypot(...forcePerFluidMass_m_s2);
}

/** Deformation-aware finite-volume Darcy flow on the shell/rod manifold.
 * Shells use the actual barycentric finite-element metric. Rods use the
 * current gradient and accepted control-volume convention.
 * Gravity is upwinded implicitly; momentum uses the SAME accepted edge fluxes
 * in a second conservative mass solve, so stiff flow cannot overdraw a node's
 * old momentum. Compressed overflow remains stored until free-water release. */
export function advanceClothPoreTransport(step:PoreFlowStep):PoreFlowResult{
 const {dt_s:dt}=step;
 const substeps=step.physics?.manifoldSubsteps??1,predictorDt=step.manifoldPredictorDt_s??dt;
 if(!(predictorDt>=0)||!Number.isFinite(predictorDt))throw new Error('Invalid manifold force predictor interval');
 if(!Number.isInteger(substeps)||substeps<1)throw new Error('Manifold substeps must be a positive integer');
 // Positions, packing, pressure and frame accelerations are fixed over the
 // outer step, so the manifold metric and constitutive edges are built once
 // and every substep reuses them; only saturation-dependent terms change.
 const cache:ManifoldPlanCache={};
 if(substeps>1){let iterations=0,movedVolume_m3=0,limitedVolume_m3=0,volumeError_m3=0;const reaction=[0,0,0];
  const sub:PoreFlowStep={...step,dt_s:dt/substeps,manifoldPredictorDt_s:predictorDt,physics:{...step.physics,manifoldSubsteps:1}};
  for(let i=0;i<substeps;i++){const result=advanceManifoldSubstep(sub,cache);iterations+=result.iterations;movedVolume_m3+=result.movedVolume_m3;limitedVolume_m3+=result.limitedVolume_m3;volumeError_m3+=result.volumeError_m3;for(let a=0;a<3;a++)reaction[a]!+=result.supportReaction_Ns[a]!;}
  return {iterations,movedVolume_m3,limitedVolume_m3,volumeError_m3,supportReaction_Ns:reaction as unknown as ClothVector};
 }
 return advanceManifoldSubstep(step,cache);
}

interface GaussPredictor {vertices:readonly number[];weights:readonly number[];gradients:readonly (readonly number[])[];material:ClothMaterial;index:number;suction:Float64Array;fraction:number;constant:Float64Array}
interface ManifoldConstitutive {gauss:GaussPredictor;a:number;b:number;slot:number;areaOverLength:number;length:number;voidFraction:number;suctionA:number;suctionB:number;drive:number;linear:number;nonlinear:number;exponent:number}
interface ManifoldPlan {capacity:Float64Array;root:Uint32Array;edgeA:Int32Array;edgeB:Int32Array;
 /** Per constitutive (element-edge) term, in assembly order. */
 cA:Int32Array;cB:Int32Array;cSlot:Int32Array;cGauss:Int32Array;cAreaOverLength:Float64Array;cVoid:Float64Array;cSuctionA:Float64Array;cSuctionB:Float64Array;cDrive:Float64Array;cLinear:Float64Array;cNonlinear:Float64Array;cExponent:Float64Array;
 /** Per Gauss predictor: vertex offsets into the flat vertex arrays. */
 gOffset:Int32Array;gVertex:Int32Array;gWeight:Float64Array;gGradient:Float64Array;gSuction:Float64Array;gConstant:Float64Array;gFraction:Float64Array;
 /** Exact reuse across Picard iterations and substeps of one outer step:
  * a Gauss predictor or term is re-evaluated only when one of its input
  * saturations changed; slots are re-summed in the same term order. */
 terms?:ManifoldTermCache}
interface ManifoldTermCache {saturation:Float64Array;changed:Uint8Array;dirty:Uint8Array;speedPower:Float64Array;floorMass:Float64Array;ab:Float64Array;ba:Float64Array;source:Float64Array;valid:boolean}
interface ManifoldPlanCache {plan?:ManifoldPlan|null}
const EMPTY=new Float64Array(0);

function advanceManifoldSubstep(step:PoreFlowStep,cache:ManifoldPlanCache):PoreFlowResult{
 const {state:s,data,materials,dt_s:dt,gravity_m_s2:g}=step,n=s.waterVolume_m3.length,predictorDt=step.manifoldPredictorDt_s??dt;
 if(!Number.isFinite(dt)||dt<0||g.some(v=>!Number.isFinite(v)))throw new Error('Invalid pore transport step');
 const empty:PoreFlowResult={iterations:0,movedVolume_m3:0,limitedVolume_m3:0,volumeError_m3:0,supportReaction_Ns:[0,0,0]};
 if(cache.plan===undefined)for(const field of [step.pressureAcceleration_m_s2,step.solidAcceleration_m_s2])if(field&&(field.length!==3*n||Array.from(field).some(v=>!Number.isFinite(v))))throw new Error('Invalid pore-frame acceleration');
 if(!dt||step.physics?.manifoldTransport===false||!s.waterVolume_m3.some(v=>v>0))return empty;
 const rho=materials[0]?.waterDensity_kg_m3;if(!(rho!>0)||materials.some(m=>m.waterDensity_kg_m3!==rho))throw new Error('Pore transport requires a common water density');
 if(cache.plan===undefined)cache.plan=buildManifoldPlan(step,rho!);
 const plan=cache.plan;if(!plan)return empty;
 return solveManifoldSubstep(step,plan,rho!,predictorDt);
}

function buildManifoldPlan(step:PoreFlowStep,rho:number):ManifoldPlan|null{
 const {state:s,data,materials,gravity_m_s2:g}=step,n=s.waterVolume_m3.length;
 const capacity=Float64Array.from(s.controlVolume_m3,(v,i)=>Math.max(0,v-s.solidVolume_m3[i]!));
 const constitutive:ManifoldConstitutive[]=[],gaussList:GaussPredictor[]=[],slots=new Map<number,number>(),edges:FlowEdge[]=[];
 const prepareGauss=(gauss:GaussPredictor)=>{
  if(gauss.index>=0)return;gauss.index=gaussList.length;gaussList.push(gauss);
  const k=gauss.vertices.length;gauss.suction=new Float64Array(k);gauss.constant=new Float64Array(3*k);let fraction=0;
  gauss.vertices.forEach((v,j)=>{const weight=gauss.weights[j]!;gauss.suction[j]=porousSuctionPressure(s.solidFraction[v]!,gauss.material).alongFiber_Pa;fraction+=weight*s.solidFraction[v]!;
   for(let axis=0;axis<3;axis++)gauss.constant[3*j+axis]=weight*(g[axis]!+(step.physics?.pressureOnManifold===false?0:(step.pressureAcceleration_m_s2?.[3*v+axis]??0))-(step.solidAcceleration_m_s2?.[3*v+axis]??0));
  });gauss.fraction=fraction;
 };
 const edge=(a:number,b:number,material:ClothMaterial,areaOverLength:number,gauss:GaussPredictor)=>{
  if(capacity[a]===0||capacity[b]===0||areaOverLength===0)return;
  if(a>b)[a,b]=[b,a];const fa=s.solidFraction[a]!,fb=s.solidFraction[b]!;
  if(!(fa>0&&fa<1&&fb>0&&fb<1))return;
  const exponent=step.physics?.dragExponent??1.6,fraction=gauss.vertices.reduce((sum,v,j)=>sum+gauss.weights[j]!*s.solidFraction[v]!,0),c0=porousDragCoefficients(fraction,0,material,false,exponent),c1=porousDragCoefficients(fraction,1,material,step.physics?.nonlinearDrag!==false,exponent);
  const suctionA=porousSuctionPressure(fa,material).alongFiber_Pa,suctionB=porousSuctionPressure(fb,material).alongFiber_Pa;
  const drive=rho!*g.reduce((sum,value,axis)=>{
   const pressure=step.physics?.pressureOnManifold===false?0:.5*((step.pressureAcceleration_m_s2?.[3*a+axis]??0)+(step.pressureAcceleration_m_s2?.[3*b+axis]??0));
   const acceleration=.5*((step.solidAcceleration_m_s2?.[3*a+axis]??0)+(step.solidAcceleration_m_s2?.[3*b+axis]??0));
   return sum+(value+pressure-acceleration)*(s.positions_m[3*b+axis]!-s.positions_m[3*a+axis]!);
  },0);
  const length=Math.hypot(...[0,1,2].map(k=>s.positions_m[3*b+k]!-s.positions_m[3*a+k]!));
  const key=a*n+b;let slot=slots.get(key);if(slot===undefined){slot=edges.length;slots.set(key,slot);edges.push({a,b,ab:0,ba:0,source:0});}
  prepareGauss(gauss);
  constitutive.push({gauss,a,b,slot,areaOverLength,length,voidFraction:1-fraction,suctionA,suctionB,drive,linear:c0.alongFiber_kg_m3_s,nonlinear:c1.alongFiber_kg_m3_s-c0.alongFiber_kg_m3_s,exponent});
 };
 const distance=(a:number,b:number)=>Math.hypot(...[0,1,2].map(k=>s.positions_m[3*b+k]!-s.positions_m[3*a+k]!));
 const shellEdges:{a:number;b:number;material:number;coefficient:number;scale:number;gauss:GaussPredictor}[]=[];
 for(let t=0;t<data.triangles.length/3;t++){
  const ids=data.triangles.slice(3*t,3*t+3),m=materials[data.triangleMaterials?.[t]??0]!;
  const u=[0,1,2].map(k=>s.positions_m[3*ids[1]!+k]!-s.positions_m[3*ids[0]!+k]!),v=[0,1,2].map(k=>s.positions_m[3*ids[2]!+k]!-s.positions_m[3*ids[0]!+k]!);
  const area=.5*Math.hypot(u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!);
  if(!(area>0))throw new Error('Collapsed pore-flow shell face');
  let effectiveThickness=m.thickness_m;
  if(step.physics?.varyingSolidFraction===false){
   const ru=[0,1,2].map(k=>data.restPositions_m[3*ids[1]!+k]!-data.restPositions_m[3*ids[0]!+k]!),rv=[0,1,2].map(k=>data.restPositions_m[3*ids[2]!+k]!-data.restPositions_m[3*ids[0]!+k]!);
   const restArea=.5*Math.hypot(ru[1]!*rv[2]!-ru[2]!*rv[1]!,ru[2]!*rv[0]!-ru[0]!*rv[2]!,ru[0]!*rv[1]!-ru[1]!*rv[0]!);effectiveThickness*=restArea/area;
  }
  const normal=[u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!],normal2=4*area*area;
  const gradients=ids.map((_,i)=>{const a=ids[(i+1)%3]!,b=ids[(i+2)%3]!,edge=[0,1,2].map(k=>s.positions_m[3*b+k]!-s.positions_m[3*a+k]!);return [(normal[1]!*edge[2]!-normal[2]!*edge[1]!)/normal2,(normal[2]!*edge[0]!-normal[0]!*edge[2]!)/normal2,(normal[0]!*edge[1]!-normal[1]!*edge[0]!)/normal2];});
  const weights=ids.map((a,i)=>{const b=ids[(i+1)%3]!,c=ids[(i+2)%3]!,e=[0,1,2].map(k=>data.restPositions_m[3*b+k]!-data.restPositions_m[3*a+k]!),f=[0,1,2].map(k=>data.restPositions_m[3*c+k]!-data.restPositions_m[3*a+k]!);return Math.atan2(Math.hypot(e[1]!*f[2]!-e[2]!*f[1]!,e[2]!*f[0]!-e[0]!*f[2]!,e[0]!*f[1]!-e[1]!*f[0]!),e.reduce((sum,v,k)=>sum+v*f[k]!,0))/Math.PI;});
  const gauss:GaussPredictor={vertices:ids,weights,gradients,material:m,index:-1,suction:EMPTY,fraction:0,constant:EMPTY};
  for(let j=0;j<3;j++){
   let a=ids[j]!,b=ids[(j+1)%3]!;const opposite=ids[(j+2)%3]!;if(a>b)[a,b]=[b,a];
   const dot=[0,1,2].reduce((sum,k)=>sum+(s.positions_m[3*a+k]!-s.positions_m[3*opposite+k]!)*(s.positions_m[3*b+k]!-s.positions_m[3*opposite+k]!),0);
   // -V * grad(N_a).grad(N_b). Keep element contributions separate until
   // each Gauss-wide nonlinear drag is evaluated; then assemble edges.
   const coefficient=effectiveThickness*dot/(4*area),material=data.triangleMaterials?.[t]??0;
   shellEdges.push({a,b,material,coefficient,scale:Math.abs(effectiveThickness),gauss});
  }
 }
 for(const e of shellEdges){
  edge(e.a,e.b,materials[e.material]!,Math.abs(e.coefficient)<1e-12*e.scale?0:e.coefficient,e.gauss);
 }
 for(let r=0;r+1<data.rodOffsets.length;r++)for(let j=data.rodOffsets[r]!;j+1<data.rodOffsets[r+1]!;j++){
  const a=data.rodIndices[j]!,b=data.rodIndices[j+1]!,m=materials[data.rodMaterials?.[r]??0]!,length=distance(a,b);if(!(length>0))throw new Error('Collapsed pore-flow rod edge');
  const restLength=Math.hypot(...[0,1,2].map(k=>data.restPositions_m[3*b+k]!-data.restPositions_m[3*a+k]!));
  const gradient=[0,1,2].map(k=>(s.positions_m[3*b+k]!-s.positions_m[3*a+k]!)/(length*length));
  edge(a,b,m,Math.PI*m.rodRadius_m*(m.rodRadiusB_m??m.rodRadius_m)*(step.physics?.varyingSolidFraction===false?restLength/length:1)/length,{vertices:[a,b],weights:[.5,.5],gradients:[gradient.map(v=>-v),gradient],material:m,index:-1,suction:EMPTY,fraction:0,constant:EMPTY});
 }
 if(!constitutive.length)return null;
 const parent=Uint32Array.from({length:n},(_,i)=>i),find=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
 for(const e of edges)parent[find(e.b)]=find(e.a);
 const root=Uint32Array.from({length:n},(_,i)=>find(i));
 const C=constitutive.length,G=gaussList.length,E=edges.length;
 const plan:ManifoldPlan={capacity,root,edgeA:Int32Array.from(edges,e=>e.a),edgeB:Int32Array.from(edges,e=>e.b),
  cA:new Int32Array(C),cB:new Int32Array(C),cSlot:new Int32Array(C),cGauss:new Int32Array(C),cAreaOverLength:new Float64Array(C),cVoid:new Float64Array(C),cSuctionA:new Float64Array(C),cSuctionB:new Float64Array(C),cDrive:new Float64Array(C),cLinear:new Float64Array(C),cNonlinear:new Float64Array(C),cExponent:new Float64Array(C),
  gOffset:new Int32Array(G+1),gVertex:new Int32Array(0),gWeight:new Float64Array(0),gGradient:new Float64Array(0),gSuction:new Float64Array(0),gConstant:new Float64Array(0),gFraction:new Float64Array(G)};
 if(E!==plan.edgeA.length)throw new Error('Manifold edge plan mismatch');
 constitutive.forEach((c,k)=>{plan.cA[k]=c.a;plan.cB[k]=c.b;plan.cSlot[k]=c.slot;plan.cGauss[k]=c.gauss.index;plan.cAreaOverLength[k]=c.areaOverLength;plan.cVoid[k]=c.voidFraction;plan.cSuctionA[k]=c.suctionA;plan.cSuctionB[k]=c.suctionB;plan.cDrive[k]=c.drive;plan.cLinear[k]=c.linear;plan.cNonlinear[k]=c.nonlinear;plan.cExponent[k]=c.exponent;});
 let total=0;gaussList.forEach((gauss,k)=>{plan.gOffset[k]=total;total+=gauss.vertices.length;});plan.gOffset[G]=total;
 plan.gVertex=new Int32Array(total);plan.gWeight=new Float64Array(total);plan.gGradient=new Float64Array(3*total);plan.gSuction=new Float64Array(total);plan.gConstant=new Float64Array(3*total);
 gaussList.forEach((gauss,k)=>{plan.gFraction[k]=gauss.fraction;for(let j=0;j<gauss.vertices.length;j++){const o=plan.gOffset[k]!+j;plan.gVertex[o]=gauss.vertices[j]!;plan.gWeight[o]=gauss.weights[j]!;plan.gSuction[o]=gauss.suction[j]!;for(let axis=0;axis<3;axis++){plan.gGradient[3*o+axis]=gauss.gradients[j]![axis]!;plan.gConstant[3*o+axis]=gauss.constant[3*j+axis]!;}}});
 return plan;
}

function solveManifoldSubstep(step:PoreFlowStep,plan:ManifoldPlan,rho:number,predictorDt:number):PoreFlowResult{
 const {state:s,data,dt_s:dt}=step,n=s.waterVolume_m3.length,{capacity}=plan,C=plan.cA.length,G=plan.gFraction.length,E=plan.edgeA.length;
 const referenceVolume=step.physics?.manifoldDragFloorReferenceVolume_m3;
 if(referenceVolume!==undefined&&(!(referenceVolume>0)||!Number.isFinite(referenceVolume)))throw new Error('Invalid source manifold regularizer volume');
 if(!(predictorDt>=0)||!Number.isFinite(predictorDt))throw new Error('Invalid manifold Gauss force predictor');
 if(s.waterVolume_m3.some(v=>!Number.isFinite(v)||v<0))throw new Error('Invalid manifold water inventory');
 const old=s.waterVolume_m3.slice(),activeCapacity=new Float64Array(n),initial=new Float64Array(n);
 for(let i=0;i<n;i++){const c=capacity[i]!;activeCapacity[i]=c>0?c:1;initial[i]=old[i]!/activeCapacity[i]!;}
 const terms=plan.terms??={saturation:new Float64Array(n),changed:new Uint8Array(n),dirty:new Uint8Array(G),speedPower:new Float64Array(G),floorMass:new Float64Array(G),ab:new Float64Array(C),ba:new Float64Array(C),source:new Float64Array(C),valid:false};
 const {speedPower,floorMass,changed,dirty}=terms,ab=new Float64Array(E),ba=new Float64Array(E),source=new Float64Array(E),edges:EdgeArrays={a:plan.edgeA,b:plan.edgeB,ab,ba,source};
 const {cA,cB,cSlot,cGauss,cAreaOverLength,cVoid,cSuctionA,cSuctionB,cDrive,cLinear,cNonlinear,cExponent,gOffset,gVertex,gWeight,gGradient,gSuction,gConstant,gFraction}=plan;
 const exponent=C?cExponent[0]!:1;for(let k=1;k<C;k++)if(cExponent[k]!==exponent)throw new Error('Manifold drag exponent must be uniform');
 const buildEdges=(saturation:Float64Array):EdgeArrays=>{
  const first=!terms.valid;
  for(let v=0;v<n;v++){changed[v]=first||saturation[v]!==terms.saturation[v]?1:0;}
  terms.saturation.set(saturation);
  // Each Gauss point's predictor speed (source Gauss-wide nonlinear drag).
  for(let k=0;k<G;k++){
   if(!first){let any=0;for(let o=gOffset[k]!;o<gOffset[k+1]!;o++)if(changed[gVertex[o]!]){any=1;break;}dirty[k]=any;if(!any)continue;}else dirty[k]=1;
   let a0=0,a1=0,a2=0,fluid=0;
   for(let o=gOffset[k]!;o<gOffset[k+1]!;o++){const v=gVertex[o]!,sv=saturation[v]!,pressure=gSuction[o]!*(1-Math.max(0,Math.min(1,sv)));
    a0+=gGradient[3*o]!*pressure/rho+gConstant[3*o]!;a1+=gGradient[3*o+1]!*pressure/rho+gConstant[3*o+1]!;a2+=gGradient[3*o+2]!*pressure/rho+gConstant[3*o+2]!;
    if(referenceVolume!==undefined)fluid+=gWeight[o]!*capacity[v]!*Math.max(0,sv);
   }
   if(!Number.isFinite(a0)||!Number.isFinite(a1)||!Number.isFinite(a2))throw new Error('Invalid manifold Gauss force predictor');
   speedPower[k]=Math.abs((1-gFraction[k]!)*predictorDt*Math.hypot(a0,a1,a2))**exponent;floorMass[k]=rho*fluid;
  }
  const termAB=terms.ab,termBA=terms.ba,termSource=terms.source;
  for(let k=0;k<C;k++){const a=cA[k]!,b=cB[k]!,gauss=cGauss[k]!;
   if(!first&&!changed[a]&&!changed[b]&&!dirty[gauss])continue;
   const Sa=saturation[a]!,Sb=saturation[b]!,sa=Math.max(0,Math.min(1,Sa)),sb=Math.max(0,Math.min(1,Sb));
   let drag=cLinear[k]!+cNonlinear[k]!*speedPower[gauss]!;
   if(referenceVolume!==undefined)drag=Math.max(drag,.001*floorMass[gauss]!/(predictorDt*referenceVolume));
   // Eq.18: epsilon_L=(1-phi)S and mobility (1-phi)*epsilon_L/C_alpha.
   const base=cAreaOverLength[k]!*cVoid[k]!*cVoid[k]!/drag,mobility=base*.5*(sa+sb),drift=base*cDrive[k]!;
   const clampA=Sa>0?sa/Sa:1,clampB=Sb>0?sb/Sb:1;
   // Keep the full difference Pc_b(1-S_b)-Pc_a(1-S_a), including the
   // gradient of packing-dependent Pc. Equal saturation is not equilibrium
   // in a nonuniform material; equal actual pore pressure is.
   termAB[k]=dt*(mobility*cSuctionA[k]!*clampA+Math.max(drift,0));termBA[k]=dt*(mobility*cSuctionB[k]!*clampB+Math.max(-drift,0));
   termSource[k]=dt*mobility*(cSuctionB[k]!-cSuctionA[k]!);
  }
  terms.valid=true;
  ab.fill(0);ba.fill(0);source.fill(0);
  for(let k=0;k<C;k++){const e=cSlot[k]!;ab[e]!+=termAB[k]!;ba[e]!+=termBA[k]!;source[e]!+=termSource[k]!;}
  return edges;
 };
 let guess=initial,solve=solveMass(activeCapacity,old,buildEdges(guess),guess),nonlinearIterations=solve.iterations;
 let relaxation=1,previousError=Infinity;
 for(let iteration=0;iteration<96;iteration++){
  let error=0,scale=1;for(let i=0;i<n;i++){const x=solve.x[i]!;error=Math.max(error,Math.abs(x-guess[i]!));scale=Math.max(scale,x);}
  if(error<=1e-10*scale)break;
  if(error>=.5*previousError)relaxation=Math.max(.125,.5*relaxation);previousError=error;
  const next=new Float64Array(n);for(let i=0;i<n;i++)next[i]=relaxation*solve.x[i]!+(1-relaxation)*guess[i]!;guess=next;
  solve=solveMass(activeCapacity,old,buildEdges(guess),solve.x);nonlinearIterations+=solve.iterations;
  if(iteration===95)throw new Error('Nonlinear manifold mobility did not converge');
 }
 // The same accepted signed edge fluxes update water and transported
 // momentum. On an obtuse/deformed mesh a negative FEM coefficient is kept;
 // only a candidate that actually violates positivity is donor-limited.
 const flux=new Float64Array(E),water=old.slice(),outgoing=new Float64Array(n);
 for(let k=0;k<E;k++){const a=plan.edgeA[k]!,b=plan.edgeB[k]!,f=ab[k]!*solve.x[a]!-ba[k]!*solve.x[b]!+source[k]!;if(!Number.isFinite(f))throw new Error('Invalid manifold volume flux');flux[k]=f;water[a]!-=f;water[b]!+=f;outgoing[f>=0?a:b]!+=Math.abs(f);}
 let total=0;for(let i=0;i<n;i++)total+=old[i]!;
 let limitedVolume_m3=0;
 if(water.some(v=>v< -1e-13*Math.max(1e-30,total))){
  // Paired donor limiter: scale each donor's outgoing fluxes together.
  water.set(old);
  for(let k=0;k<E;k++){const a=plan.edgeA[k]!,b=plan.edgeB[k]!,donor=flux[k]!>=0?a:b,limited=flux[k]!*(outgoing[donor]!>old[donor]!?Math.max(0,old[donor]!/outgoing[donor]!):1);limitedVolume_m3+=Math.abs(flux[k]!-limited);flux[k]=limited;water[a]!-=limited;water[b]!+=limited;}
  if(water.some(v=>v< -1e-13*Math.max(1e-30,total)))throw new Error('Paired manifold positivity limiter failed');
 }
 if(step.edgeFlux_m3)for(let k=0;k<E;k++){const key=plan.edgeA[k]!*n+plan.edgeB[k]!;step.edgeFlux_m3.set(key,(step.edgeFlux_m3.get(key)??0)+flux[k]!);}
 const tolerance=1e-9*Math.max(total,1e-30);
 if(water.some(v=>!Number.isFinite(v)||v< -tolerance))throw new Error('Pore flow produced invalid water');
 // Roundoff repair is component-local and bounded by the solved residual.
 const root=plan.root,before=new Float64Array(n),after=new Float64Array(n),largest=new Int32Array(n).fill(-1);
 for(let i=0;i<n;i++){const r=root[i]!;if(largest[r]!<0)largest[r]=i;before[r]!+=old[i]!;water[i]=Math.max(0,water[i]!);after[r]!+=water[i]!;if(water[i]!>water[largest[r]!]!)largest[r]=i;}
 for(let r=0;r<n;r++){if(largest[r]!<0)continue;if(Math.abs(before[r]!-after[r]!)>tolerance)throw new Error('Pore flow lost component volume');water[largest[r]!]!+=before[r]!-after[r]!;}
 const mass=new Float64Array(n),oldMass=new Float64Array(n);for(let i=0;i<n;i++){mass[i]=s.dryMass_kg[i]!+rho*water[i]!;oldMass[i]=s.dryMass_kg[i]!+rho*old[i]!;}
 const massAB=new Float64Array(E),massBA=new Float64Array(E);let moved=0;
 for(let k=0;k<E;k++){const f=flux[k]!;moved+=Math.abs(f);massAB[k]=rho*Math.max(f,0);massBA[k]=rho*Math.max(-f,0);}
 const massEdges:EdgeArrays={a:plan.edgeA,b:plan.edgeB,ab:massAB,ba:massBA};
 let iterations=nonlinearIterations;const reaction=[0,0,0];
 const allFixed=step.physics?.solveSolid===false,fixed=allFixed?undefined:(step.kinematicFixed??data.fixed);
 for(let a=0;a<3;a++){
  // Every vertex prescribed: velocity is retained, the support absorbs the carried momentum.
  if(allFixed){for(let i=0;i<n;i++){const v=s.velocities_m_s[3*i+a]!;reaction[a]!+=mass[i]!*v-oldMass[i]!*v;}continue;}
  const velocity=new Float64Array(n),rhs=new Float64Array(n);for(let i=0;i<n;i++){const v=s.velocities_m_s[3*i+a]!;velocity[i]=v;rhs[i]=((fixed?.[i]??0)&1)?v:oldMass[i]!*v;}
  const result=solveMass(mass,rhs,massEdges,velocity,fixed);iterations+=result.iterations;
  for(let i=0;i<n;i++){reaction[a]!+=mass[i]!*result.x[i]!-oldMass[i]!*velocity[i]!;s.velocities_m_s[3*i+a]=result.x[i]!;}
 }
 s.waterVolume_m3.set(water);
 return {iterations,movedVolume_m3:moved,limitedVolume_m3,volumeError_m3:water.reduce((a,b)=>a+b,0)-total,supportReaction_Ns:reaction as unknown as ClothVector};
}
