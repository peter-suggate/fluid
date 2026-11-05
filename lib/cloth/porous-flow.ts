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
function solveMass(capacity:Float64Array,rhs:Float64Array,edges:readonly FlowEdge[],initial:Float64Array,fixed?:ArrayLike<number>):SolveResult{
 const n=capacity.length,x=initial.slice(),diag=capacity.slice(),isFixed=(i:number)=>((fixed?.[i]??0)&1)!==0;
 const adjusted=rhs.slice();for(const e of edges)if(e.source){if(!isFixed(e.a))adjusted[e.a]!-=e.source;if(!isFixed(e.b))adjusted[e.b]!+=e.source;}
 for(const e of edges){diag[e.a]!+=e.ab;diag[e.b]!+=e.ba;}for(let i=0;i<n;i++)if(isFixed(i))diag[i]=1;
 const apply=(v:Float64Array,out:Float64Array)=>{
  for(let i=0;i<n;i++)out[i]=capacity[i]!*v[i]!;
  for(const e of edges){const flux=e.ab*v[e.a]!-e.ba*v[e.b]!;out[e.a]!+=flux;out[e.b]!-=flux;}
  for(let i=0;i<n;i++)if(isFixed(i))out[i]=v[i]!;
 };
 const dot=(a:Float64Array,b:Float64Array)=>{let s=0;for(let i=0;i<n;i++)s+=a[i]!*b[i]!;return s;};
 const norm=(a:Float64Array)=>a.reduce((m,v)=>Math.max(m,Math.abs(v)),0);
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

export interface PoreFlowStep {state:ClothState;data:ClothTopologyData;kinematicFixed?:ArrayLike<number>;materials:readonly ClothMaterial[];dt_s:number;gravity_m_s2:ClothVector;/** Outer fluid timestep used by the released source nonlinear drag predictor. */manifoldPredictorDt_s?:number;physics?:ClothPhysicsOptions;pressureAcceleration_m_s2?:ArrayLike<number>;solidAcceleration_m_s2?:ArrayLike<number>}
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
 const {state:s,data,materials,dt_s:dt,gravity_m_s2:g}=step,n=s.waterVolume_m3.length;
 const substeps=step.physics?.manifoldSubsteps??1,predictorDt=step.manifoldPredictorDt_s??dt;
 if(!(predictorDt>=0)||!Number.isFinite(predictorDt))throw new Error('Invalid manifold force predictor interval');
 if(!Number.isInteger(substeps)||substeps<1)throw new Error('Manifold substeps must be a positive integer');
 if(substeps>1){let iterations=0,movedVolume_m3=0,limitedVolume_m3=0,volumeError_m3=0;const reaction=[0,0,0];
  for(let i=0;i<substeps;i++){const result=advanceClothPoreTransport({...step,dt_s:dt/substeps,manifoldPredictorDt_s:predictorDt,physics:{...step.physics,manifoldSubsteps:1}});iterations+=result.iterations;movedVolume_m3+=result.movedVolume_m3;limitedVolume_m3+=result.limitedVolume_m3;volumeError_m3+=result.volumeError_m3;for(let a=0;a<3;a++)reaction[a]!+=result.supportReaction_Ns[a]!;}
  return {iterations,movedVolume_m3,limitedVolume_m3,volumeError_m3,supportReaction_Ns:reaction as unknown as ClothVector};
 }
 if(!Number.isFinite(dt)||dt<0||g.some(v=>!Number.isFinite(v)))throw new Error('Invalid pore transport step');
 const empty:PoreFlowResult={iterations:0,movedVolume_m3:0,limitedVolume_m3:0,volumeError_m3:0,supportReaction_Ns:[0,0,0]};
 for(const field of [step.pressureAcceleration_m_s2,step.solidAcceleration_m_s2])if(field&&(field.length!==3*n||Array.from(field).some(v=>!Number.isFinite(v))))throw new Error('Invalid pore-frame acceleration');
 if(!dt||step.physics?.manifoldTransport===false||!s.waterVolume_m3.some(v=>v>0))return empty;
 const rho=materials[0]?.waterDensity_kg_m3;if(!(rho!>0)||materials.some(m=>m.waterDensity_kg_m3!==rho))throw new Error('Pore transport requires a common water density');
 const capacity=Float64Array.from(s.controlVolume_m3,(v,i)=>Math.max(0,v-s.solidVolume_m3[i]!));
 interface GaussPredictor {vertices:readonly number[];weights:readonly number[];gradients:readonly (readonly number[])[];material:ClothMaterial}
 const constitutive:{gauss:GaussPredictor;a:number;b:number;areaOverLength:number;length:number;voidFraction:number;suctionA:number;suctionB:number;drive:number;linear:number;nonlinear:number;exponent:number}[]=[];
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
  constitutive.push({gauss,a,b,areaOverLength,length,voidFraction:1-fraction,suctionA,suctionB,drive,linear:c0.alongFiber_kg_m3_s,nonlinear:c1.alongFiber_kg_m3_s-c0.alongFiber_kg_m3_s,exponent});
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
  const gauss:GaussPredictor={vertices:ids,weights,gradients,material:m};
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
  edge(a,b,m,Math.PI*m.rodRadius_m*(m.rodRadiusB_m??m.rodRadius_m)*(step.physics?.varyingSolidFraction===false?restLength/length:1)/length,{vertices:[a,b],weights:[.5,.5],gradients:[gradient.map(v=>-v),gradient],material:m});
 }
 if(!constitutive.length)return empty;
 const activeCapacity=Float64Array.from(capacity,v=>v>0?v:1),old=s.waterVolume_m3.slice(),initial=Float64Array.from(old,(v,i)=>v/activeCapacity[i]!);
 const buildEdges=(saturation:Float64Array):FlowEdge[]=>{
  const map=new Map<string,FlowEdge>(),predictors=new Map<GaussPredictor,number>();
  for(const c of constitutive){const sa=Math.max(0,Math.min(1,saturation[c.a]!)),sb=Math.max(0,Math.min(1,saturation[c.b]!));
   let speed=predictors.get(c.gauss);
   if(speed===undefined){
    const gauss=c.gauss,acceleration=[0,0,0];let fraction=0;
    gauss.vertices.forEach((v,j)=>{const weight=gauss.weights[j]!,pressure=porousSuctionPressure(s.solidFraction[v]!,gauss.material).alongFiber_Pa*(1-Math.max(0,Math.min(1,saturation[v]!)));fraction+=weight*s.solidFraction[v]!;
     for(let axis=0;axis<3;axis++)acceleration[axis]!+=gauss.gradients[j]![axis]!*pressure/rho!+weight*(g[axis]!+(step.physics?.pressureOnManifold===false?0:(step.pressureAcceleration_m_s2?.[3*v+axis]??0))-(step.solidAcceleration_m_s2?.[3*v+axis]??0));
    });
    speed=manifoldForcePredictorSpeed(fraction,predictorDt,acceleration);predictors.set(gauss,speed);
   }
   let drag=c.linear+c.nonlinear*Math.abs(speed)**c.exponent;
   const referenceVolume=step.physics?.manifoldDragFloorReferenceVolume_m3;
   if(referenceVolume!==undefined){if(!(referenceVolume>0)||!Number.isFinite(referenceVolume))throw new Error('Invalid source manifold regularizer volume');const fluidMass=rho!*c.gauss.vertices.reduce((sum,v,j)=>sum+c.gauss.weights[j]!*capacity[v]!*Math.max(0,saturation[v]!),0);drag=Math.max(drag,.001*fluidMass/(predictorDt*referenceVolume));}
   // Eq.18: epsilon_L=(1-phi)S and mobility (1-phi)*epsilon_L/C_alpha.
   const base=c.areaOverLength*c.voidFraction*c.voidFraction/drag,mobility=base*.5*(sa+sb),drift=base*c.drive;
   const clampA=saturation[c.a]!>0?sa/saturation[c.a]!:1,clampB=saturation[c.b]!>0?sb/saturation[c.b]!:1;
   // Keep the full difference Pc_b(1-S_b)-Pc_a(1-S_a), including the
   // gradient of packing-dependent Pc. Equal saturation is not equilibrium
   // in a nonuniform material; equal actual pore pressure is.
   const key=`${c.a},${c.b}`,e=map.get(key)??{a:c.a,b:c.b,ab:0,ba:0,source:0};
   e.ab+=dt*(mobility*c.suctionA*clampA+Math.max(drift,0));e.ba+=dt*(mobility*c.suctionB*clampB+Math.max(-drift,0));
   e.source!+=dt*mobility*(c.suctionB-c.suctionA);map.set(key,e);
  }return [...map.values()];
 };
 let guess=initial,edges=buildEdges(guess),solve=solveMass(activeCapacity,old,edges,guess),nonlinearIterations=solve.iterations;
 for(let iteration=0;iteration<96;iteration++){
  const error=solve.x.reduce((m,x,i)=>Math.max(m,Math.abs(x-guess[i]!)),0),scale=solve.x.reduce((m,x)=>Math.max(m,x),1);
  if(error<=1e-10*scale)break;
  guess=Float64Array.from(solve.x,(x,i)=>.5*(x+guess[i]!));edges=buildEdges(guess);solve=solveMass(activeCapacity,old,edges,solve.x);nonlinearIterations+=solve.iterations;
  if(iteration===95)throw new Error('Nonlinear manifold mobility did not converge');
 }
 // The same accepted signed edge fluxes update water and transported
 // momentum. On an obtuse/deformed mesh a negative FEM coefficient is kept;
 // only a candidate that actually violates positivity is donor-limited.
 const proposed=edges.map(e=>({a:e.a,b:e.b,volume_m3:e.ab*solve.x[e.a]!-e.ba*solve.x[e.b]!+(e.source??0)})),accepted=acceptPositiveManifoldFlux(old,proposed),water=accepted.water;
 const limitedVolume_m3=proposed.reduce((sum,f,i)=>sum+Math.abs(f.volume_m3-accepted.fluxes[i]!.volume_m3),0);
 const total=old.reduce((a,b)=>a+b,0),tolerance=1e-9*Math.max(total,1e-30);
 if(water.some(v=>!Number.isFinite(v)||v< -tolerance))throw new Error('Pore flow produced invalid water');
 // Roundoff repair is component-local and bounded by the solved residual.
 const parent=Uint32Array.from({length:n},(_,i)=>i),root=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
 for(const e of edges)parent[root(e.b)]=root(e.a);
 const sums=new Map<number,{before:number;after:number;largest:number}>();
 for(let i=0;i<n;i++){const r=root(i),sum=sums.get(r)??{before:0,after:0,largest:i};sum.before+=old[i]!;water[i]=Math.max(0,water[i]!);sum.after+=water[i]!;if(water[i]!>water[sum.largest]!)sum.largest=i;sums.set(r,sum);}
 for(const sum of sums.values()){if(Math.abs(sum.before-sum.after)>tolerance)throw new Error('Pore flow lost component volume');water[sum.largest]!+=sum.before-sum.after;}
 const mass=Float64Array.from(water,(v,i)=>s.dryMass_kg[i]!+rho!*v),oldMass=Float64Array.from(old,(v,i)=>s.dryMass_kg[i]!+rho!*v),massEdges:FlowEdge[]=[];let moved=0;
 for(const e of accepted.fluxes){const flux=e.volume_m3;moved+=Math.abs(flux);massEdges.push({a:e.a,b:e.b,ab:rho!*Math.max(flux,0),ba:rho!*Math.max(-flux,0)});}
 let iterations=nonlinearIterations;const reaction=[0,0,0];
 for(let a=0;a<3;a++){
  const fixed=step.physics?.solveSolid===false?new Uint8Array(old.length).fill(1):(step.kinematicFixed??data.fixed),velocity=Float64Array.from(old,(_,i)=>s.velocities_m_s[3*i+a]!),rhs=Float64Array.from(velocity,(v,i)=>((fixed?.[i]??0)&1)?v:oldMass[i]!*v),result=solveMass(mass,rhs,massEdges,velocity,fixed);iterations+=result.iterations;
  for(let i=0;i<n;i++){reaction[a]!+=mass[i]!*result.x[i]!-oldMass[i]!*velocity[i]!;s.velocities_m_s[3*i+a]=result.x[i]!;}
 }
 s.waterVolume_m3.set(water);
 return {iterations,movedVolume_m3:moved,limitedVolume_m3,volumeError_m3:water.reduce((a,b)=>a+b,0)-total,supportReaction_Ns:reaction as unknown as ClothVector};
}
