import type { ClothFrameIdentity, ClothState, ClothVector } from './types';

/** One component of the backward-Euler two-mass drag system. Coefficient is
 * kg/s (constitutive drag per volume times the actual shared wet volume).
 * The two returned impulses are exactly opposite; pressure is separate. */
export function implicitDragExchange(fluidMass_kg: number, solidMass_kg: number,
  fluidVelocity_m_s: number, solidVelocity_m_s: number, drag_kg_s: number, dt_s: number,
  fixedSolid = false): { fluidImpulse_Ns: number; solidImpulse_Ns: number; fluidVelocity_m_s: number; solidVelocity_m_s: number } {
  for (const [name, value] of Object.entries({ fluidMass_kg, solidMass_kg, drag_kg_s, dt_s })) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${name}`);
  }
  if (!Number.isFinite(fluidVelocity_m_s) || !Number.isFinite(solidVelocity_m_s)) throw new Error('Invalid exchange velocity');
  if (solidMass_kg === 0 && !fixedSolid) throw new Error('A free solid needs positive mass');
  if (fluidMass_kg === 0 || dt_s === 0 || drag_kg_s === 0) {
    return { fluidImpulse_Ns: 0, solidImpulse_Ns: 0, fluidVelocity_m_s, solidVelocity_m_s };
  }
  const inverseSolid = fixedSolid ? 0 : 1 / solidMass_kg;
  // Reciprocal form avoids overflow for very stiff drag.
  const solidImpulse_Ns = (fluidVelocity_m_s - solidVelocity_m_s)
    / (1 / fluidMass_kg + inverseSolid + (1 / drag_kg_s) / dt_s);
  return { fluidImpulse_Ns: -solidImpulse_Ns, solidImpulse_Ns,
    fluidVelocity_m_s: fluidVelocity_m_s - solidImpulse_Ns / fluidMass_kg,
    solidVelocity_m_s: fixedSolid ? solidVelocity_m_s : solidVelocity_m_s + solidImpulse_Ns / solidMass_kg };
}

/** Full 3-D two-mass backward Euler oracle. The Cholesky solve preserves
 * cross-component terms of any rotated positive-definite drag tensor. */
export function implicitTensorDragExchange(fluidMass_kg:number,solidMass_kg:number,
 fluidVelocity:ClothVector,solidVelocity:ClothVector,tensor_kg_s:ArrayLike<number>,dt_s:number):ClothVector{
 if(!(fluidMass_kg>0&&solidMass_kg>0&&dt_s>0)||tensor_kg_s.length!==9)throw new Error('Invalid tensor exchange');
 const solve=(matrix:number[],rhs:number[])=>{
  const L=new Float64Array(9);
  for(let i=0;i<3;i++)for(let j=0;j<=i;j++){
   let value=matrix[3*i+j]!;for(let k=0;k<j;k++)value-=L[3*i+k]!*L[3*j+k]!;
   if(i===j){if(!(value>0))throw new Error('Drag tensor is not positive definite');L[3*i+j]=Math.sqrt(value);}else L[3*i+j]=value/L[3*j+j]!;
  }
  const x=[0,0,0],y=[0,0,0];for(let i=0;i<3;i++){let v=rhs[i]!;for(let j=0;j<i;j++)v-=L[3*i+j]!*y[j]!;y[i]=v/L[3*i+i]!;}
  for(let i=2;i>=0;i--){let v=y[i]!;for(let j=i+1;j<3;j++)v-=L[3*j+i]!*x[j]!;x[i]=v/L[3*i+i]!;}return x;
 };
 const C=Array.from(tensor_kg_s),inverse=new Array<number>(9).fill(0);
 for(let col=0;col<3;col++){const basis=[0,0,0];basis[col]=1;const x=solve(C,basis);for(let row=0;row<3;row++)inverse[3*row+col]=x[row]!/dt_s;}
 for(let a=0;a<3;a++)inverse[4*a]!+=1/fluidMass_kg+1/solidMass_kg;
 return solve(inverse,fluidVelocity.map((v,a)=>v-solidVelocity[a]!)) as unknown as ClothVector;
}

export interface ClothImpulsePublication extends ClothFrameIdentity {
  /** Actual projected free-fluid pressure acceleration, independent of the solid buoyancy ablation. */
  readonly fluidPressureAcceleration_m_s2?:Float64Array;
  readonly impulse_Ns: Float32Array | Float64Array;
  readonly dragImpulse_Ns?: Float32Array;
  readonly pressureImpulse_Ns?: Float32Array;
  /** Pressure/porosity projection must be distinguished from drag evidence. */
  readonly mechanism: 'drag' | 'pressure' | 'fluid' | 'contact';
}

/** Consume a generation/tick publication once. Fixed vertices retain their
 * prescribed velocity; their reaction is returned to the support owner. */
export class ClothImpulseConsumer {
  private readonly consumed = new Map<string, number>();
  apply(state: ClothState, publication: ClothImpulsePublication, waterDensity_kg_m3: number,
    fixed?: ArrayLike<number>): ClothVector {
    if (publication.generation !== state.generation || publication.topologyRevision !== state.topologyRevision) throw new Error('Stale cloth impulse publication');
    const key = `${publication.generation}:${publication.topologyRevision}:${publication.mechanism}`;
    if (publication.tick <= (this.consumed.get(key) ?? -1)) throw new Error('Cloth impulses must be applied once in increasing tick order');
    if (publication.impulse_Ns.length !== state.velocities_m_s.length || !(waterDensity_kg_m3 > 0)) throw new Error('Invalid cloth impulse payload');
    const reactions = [0, 0, 0];
    // Validate the whole exchange before changing any velocity.
    for (let v = 0; v < state.dryMass_kg.length; v++) {
      const mass = state.dryMass_kg[v]! + waterDensity_kg_m3 * state.waterVolume_m3[v]!;
      if (!((fixed?.[v] ?? 0) & 1) && !(mass > 0)) throw new Error('Free cloth vertex has no mass');
      for (let a = 0; a < 3; a++) if (!Number.isFinite(publication.impulse_Ns[3 * v + a])) throw new Error('Non-finite cloth impulse');
    }
    for (let v = 0; v < state.dryMass_kg.length; v++) {
      const mass = state.dryMass_kg[v]! + waterDensity_kg_m3 * state.waterVolume_m3[v]!;
      for (let a = 0; a < 3; a++) {
        const impulse = publication.impulse_Ns[3 * v + a]!;
        if ((fixed?.[v] ?? 0) & 1) reactions[a]! += impulse;
        else state.velocities_m_s[3 * v + a]! += impulse / mass;
      }
    }
    this.consumed.set(key, publication.tick);
    return reactions as unknown as ClothVector;
  }
}

