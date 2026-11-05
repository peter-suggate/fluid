import type { ClothMaterial, ClothVector } from './types';

/** SI constitutive model from Fei et al. 2018, equations 9, 13 and 15.
 * Fiber diameter describes the pore microstructure, not the rendered rod. */
export type PorousWaterMaterial = Pick<ClothMaterial, 'solidFraction' | 'fiberDiameter_m' | 'contactAngle_rad'
  | 'surfaceTension_N_m' | 'waterDensity_kg_m3' | 'waterViscosity_Pa_s'>;

export interface PorousPermeability {
  readonly alongFiber_m2: number;
  readonly acrossFiber_m2: number;
  readonly capillaryRadius_m: number;
}

function finitePositive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and positive`);
}

export function porousPermeability(solidFraction: number, fiberDiameter_m: number): PorousPermeability {
  finitePositive(fiberDiameter_m, 'Fiber diameter');
  if (!Number.isFinite(solidFraction) || solidFraction <= 0 || solidFraction >= 1) {
    throw new Error('Porous solid fraction must lie strictly between zero and one');
  }
  const f = solidFraction, d2 = fiberDiameter_m ** 2;
  const common = -Math.log(f) - 1.476 + 2 * f;
  const alongFiber_m2 = (common - .5 * f * f) * d2 / (16 * f);
  const acrossFiber_m2 = (common - 1.774 * f * f + 4.078 * f ** 3) * d2 / (32 * f);
  if (!(alongFiber_m2 > 0 && acrossFiber_m2 > 0)) throw new Error('Fiber permeability fit is not positive at this solid fraction');
  return { alongFiber_m2, acrossFiber_m2,
    capillaryRadius_m: .5 * fiberDiameter_m * Math.sqrt((1 - f) / f) };
}

/** The two principal values of equation 9; suction is zero at 90 degrees.
 * These are material pressures; multiply by (1 - saturation) for pore stress. */
export function porousSuctionPressure(solidFraction: number, material: PorousWaterMaterial) {
  porousPermeability(solidFraction, material.fiberDiameter_m);
  // The microscopic pore radius is a material/rest parameter. Packing can
  // vary spatially or under deformation without silently redefining it.
  const permeability = porousPermeability(material.solidFraction, material.fiberDiameter_m);
  if (!Number.isFinite(material.surfaceTension_N_m) || material.surfaceTension_N_m < 0
    || !Number.isFinite(material.contactAngle_rad) || material.contactAngle_rad < 0 || material.contactAngle_rad > Math.PI) {
    throw new Error('Invalid water surface tension or contact angle');
  }
  const cosine = Math.abs(material.contactAngle_rad - Math.PI / 2) < 1e-14 ? 0 : Math.cos(material.contactAngle_rad);
  const alongFiber_Pa = 2 * solidFraction * material.surfaceTension_N_m * cosine
    / ((1 - solidFraction) * permeability.capillaryRadius_m);
  return { alongFiber_Pa, acrossFiber_Pa: alongFiber_Pa / 2 };
}

/** Drag per mixture volume, in kg/(m^3 s). Nonlinearity is evaluated from
 * relative speed, then integrated implicitly by the exchange solver. */
export function porousDragCoefficients(solidFraction: number, relativeSpeed_m_s: number,
  material: PorousWaterMaterial, nonlinear = true, exponent = 1.6) {
  const k = porousPermeability(solidFraction, material.fiberDiameter_m);
  finitePositive(material.waterViscosity_Pa_s, 'Water viscosity');
  finitePositive(material.waterDensity_kg_m3, 'Water density');
  if (!Number.isFinite(relativeSpeed_m_s) || relativeSpeed_m_s < 0) throw new Error('Relative speed must be finite and non-negative');
  if(!Number.isFinite(exponent)||exponent<=0)throw new Error('Drag exponent must be positive');
  const mu = material.waterViscosity_Pa_s, rho = material.waterDensity_kg_m3;
  const nonlinearScale = nonlinear ? (1.75 / Math.sqrt(150)) * material.fiberDiameter_m ** (exponent - 1)
    * mu ** (1 - exponent) * (rho * relativeSpeed_m_s) ** exponent / (1 - solidFraction) ** 1.5 : 0;
  const evaluate = (permeability: number) => mu / permeability + nonlinearScale / Math.sqrt(permeability);
  return { alongFiber_kg_m3_s: evaluate(k.alongFiber_m2), acrossFiber_kg_m3_s: evaluate(k.acrossFiber_m2) };
}

/** Axisymmetric tensor in world coordinates. For a shell, axis is its normal
 * and shapeFraction=1; for a rod it is its tangent and shapeFraction=0. */
export function porousTensorDiagonal(alongFiber: number, acrossFiber: number, axis: ClothVector, shapeFraction: number): ClothVector {
  if (!Number.isFinite(shapeFraction) || shapeFraction < 0 || shapeFraction > 1) throw new Error('Shape fraction must be in [0, 1]');
  const length = Math.hypot(...axis);
  finitePositive(length, 'Material direction length');
  const tangent = shapeFraction * alongFiber + (1 - shapeFraction) * acrossFiber;
  const axial = shapeFraction * acrossFiber + (1 - shapeFraction) * alongFiber;
  return axis.map(component => tangent + (axial - tangent) * (component / length) ** 2) as unknown as ClothVector;
}

/** Full symmetric tensor, column-major. Keeping the cross terms is required
 * when material fibers are oblique to the simulation grid. */
export function porousTensor(alongFiber:number,acrossFiber:number,axis:ClothVector,shapeFraction:number):Float64Array{
 const diagonal=porousTensorDiagonal(alongFiber,acrossFiber,axis,shapeFraction),length=Math.hypot(...axis);
 const tangent=shapeFraction*alongFiber+(1-shapeFraction)*acrossFiber,axial=shapeFraction*acrossFiber+(1-shapeFraction)*alongFiber;
 return Float64Array.from({length:9},(_,i)=>{const row=i%3,column=Math.floor(i/3);return row===column?diagonal[row]!:(axial-tangent)*axis[row]!*axis[column]!/(length*length);});
}

export function poreCapacity_m3(controlVolume_m3: number, solidFraction: number): number {
  if (!Number.isFinite(controlVolume_m3) || controlVolume_m3 < 0 || !Number.isFinite(solidFraction) || solidFraction < 0 || solidFraction > 1) {
    throw new Error('Invalid porous control volume or solid fraction');
  }
  return controlVolume_m3 * (1 - solidFraction);
}


/** Eq.15 drag uses the opposite canonical ordering from Eq.10 suction:
 * cloth diag(C_beta,C_beta,C_alpha), rod diag(C_alpha,C_alpha,C_beta). */
export function porousDragTensor(alpha:number,beta:number,axis:ClothVector,shapeFraction:number):Float64Array{
 return porousTensor(beta,alpha,axis,shapeFraction);
}

/** Monotone scalar Eq.17 closure C(|u|)u = drivingForce. */
export function quasiStaticPorousSpeed(drivingForce_N_m3:number,linear_kg_m3_s:number,nonlinearCoefficient:number,exponent:number):number{
 if(!(linear_kg_m3_s>0)||nonlinearCoefficient<0||exponent<=0||!Number.isFinite(drivingForce_N_m3))throw new Error('Invalid quasi-static porous closure');
 const force=Math.abs(drivingForce_N_m3);if(!force)return 0;
 let lo=0,hi=force/linear_kg_m3_s;
 for(let i=0;i<64;i++){const mid=(lo+hi)/2;if((linear_kg_m3_s+nonlinearCoefficient*mid**exponent)*mid<force)lo=mid;else hi=mid;}
 return Math.sign(drivingForce_N_m3)*.5*(lo+hi);
}

/** Upstream getVerticalDiffusivity/captureLiquid: grid h, not sheet thickness. */
export function porousCaptureFraction(solidFraction:number,material:PorousWaterMaterial,cellSize_m:number,dt_s:number,captureRate=1):number{
 if(!(cellSize_m>0)||!Number.isFinite(dt_s)||dt_s<0||!Number.isFinite(captureRate)||captureRate<0)throw new Error('Invalid pore capture timescale');
 const permeability=porousPermeability(solidFraction,material.fiberDiameter_m),pressure=porousSuctionPressure(solidFraction,material).acrossFiber_Pa;
 return Math.min(1,captureRate*dt_s*permeability.acrossFiber_m2*Math.max(0,pressure)/(material.waterViscosity_Pa_s*cellSize_m*cellSize_m));
}
