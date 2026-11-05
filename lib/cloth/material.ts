import type { ClothMaterial } from "./types";

/** Cotton-like source material; SI conversion of the paper's common preset. */
export const DEFAULT_CLOTH_MATERIAL: ClothMaterial = Object.freeze({
  id: "cotton", density_kg_m3: 1320, youngModulus_Pa: 1.83e6,
  poissonRatio: .202, thickness_m: .00036, rodRadius_m: .00018,
  solidFraction: .4, solidFractionExponent: 2, fiberDiameter_m: .0001,
  contactAngle_rad: 40.8 * Math.PI / 180, surfaceTension_N_m: .072,
  waterDensity_kg_m3: 1000, waterViscosity_Pa_s: .00089,
  friction: Math.tan(10 * Math.PI / 180), mechanicalDamping_s: 2.78e4 / 1.83e7,
});

export function clothMaterialErrors(m: ClothMaterial): string[] {
  const errors: string[] = [];
  if (!m || typeof m !== "object") return ["Cloth material must be an object"];
  if(m.attachmentMultiplier!==undefined&&(!Number.isFinite(m.attachmentMultiplier)||m.attachmentMultiplier<0))errors.push(`${m.id}: attachment multiplier must be finite and nonnegative`);
  for(const key of ['membraneDamping_s','bendingDamping_s','rodStretchDamping_s','rodBendingDamping_s','rodTwistDamping_s'] as const)if(m[key]!==undefined&&(!Number.isFinite(m[key])||m[key]!<0))errors.push(`${m.id}: ${key} must be finite and nonnegative`);
  if (!m.id) errors.push("Cloth material requires an id");
  if(m.rodRadiusB_m!==undefined&&!(Number.isFinite(m.rodRadiusB_m)&&m.rodRadiusB_m>0))errors.push(`${m.id}: rodRadiusB_m must be positive and finite`);
  for (const key of ["density_kg_m3", "youngModulus_Pa", "thickness_m", "rodRadius_m", "fiberDiameter_m", "waterDensity_kg_m3", "waterViscosity_Pa_s"] as const)
    if (!(Number.isFinite(m[key]) && m[key] > 0)) errors.push(`${m.id}: ${key} must be positive and finite`);
  for (const key of ["surfaceTension_N_m", "friction", "mechanicalDamping_s", "solidFractionExponent"] as const)
    if (!(Number.isFinite(m[key]) && m[key] >= 0)) errors.push(`${m.id}: ${key} must be nonnegative and finite`);
  if (!(m.poissonRatio > -1 && m.poissonRatio < .5)) errors.push(`${m.id}: Poisson ratio must be in (-1, 0.5)`);
  if (!(m.solidFraction > 0 && m.solidFraction <= 1)) errors.push(`${m.id}: solid fraction must be in (0, 1]`);
  if (!(m.contactAngle_rad >= 0 && m.contactAngle_rad <= Math.PI)) errors.push(`${m.id}: contact angle must be in [0, pi]`);
  if (m.bendingMode !== undefined && !["theta", "tan-half", "sin-half"].includes(m.bendingMode)) errors.push(`${m.id}: unsupported bending mode`);
  return errors;
}

/** Plane-stress modulus including source effective-stress solid-fraction law. */
export function membraneModulus(m: ClothMaterial, fraction = m.solidFraction): number {
  return m.youngModulus_Pa * m.thickness_m / (1 - m.poissonRatio ** 2) * fraction ** m.solidFractionExponent;
}

export function bendingRigidity(m: ClothMaterial, fraction = m.solidFraction): number {
  return m.youngModulus_Pa * m.thickness_m ** 3 / (12 * (1 - m.poissonRatio ** 2)) * fraction ** m.solidFractionExponent;
}

export function rodRigidity(m: ClothMaterial, fraction = m.solidFraction) {
  const fractionScale = fraction ** m.solidFractionExponent;
  const a=m.rodRadius_m,b=m.rodRadiusB_m??a,area=Math.PI*a*b;
  const secondMoment1=Math.PI*a**3*b/4,secondMoment2=Math.PI*a*b**3/4;
  return { stretch_N: m.youngModulus_Pa * area * fractionScale,
    bend_N_m2: m.youngModulus_Pa * secondMoment1 * fractionScale,
    bend1_N_m2:m.youngModulus_Pa*secondMoment1*fractionScale,bend2_N_m2:m.youngModulus_Pa*secondMoment2*fractionScale,
    twist_N_m2: m.youngModulus_Pa / (2 * (1 + m.poissonRatio)) * (secondMoment1+secondMoment2) * fractionScale };
}

/** Released source stores angular inertia on each outgoing segment's start
 * particle (the terminal tip is an uncoupled extra scalar). Generic rods retain
 * their edge-centered polar mass. Bound water has no source angular inertia. */
export function rodSegmentInertia_kg_m2(topology:import('./topology').PreparedClothTopology,state:import('./types').ClothState,rodIndex:number,local:number):number{
 const rod=topology.rods[rodIndex]!,m=topology.materials[rod.material]!,a=rod.vertices[local]!,b=rod.vertices[local+1]!;
 if(m.attachmentMultiplier!==undefined){
  // loadHairs overwrites the particle angular inertia with its newly loaded
  // rod mass, excluding any previously loaded shell mass at a fuzzy root.
  const incident=topology.incidentRods[a]!,last=topology.rods[incident[incident.length-1]!]!,source=topology.materials[last.material]!;
  let length=0;for(const index of incident){const adjacent=topology.rods[index]!;for(let j=0;j<adjacent.restLengths_m.length;j++)if(adjacent.vertices[j]===a||adjacent.vertices[j+1]===a)length+=adjacent.restLengths_m[j]!/2;}
  const b=m.rodRadiusB_m??m.rodRadius_m;
  return .25*source.density_kg_m3*source.solidFraction*Math.PI*m.rodRadius_m*b*(m.rodRadius_m**2+b*b)*length;
 }
 const radiusB=m.rodRadiusB_m??m.rodRadius_m;
 return .25*m.density_kg_m3*Math.PI*m.rodRadius_m*radiusB*(m.rodRadius_m**2+radiusB**2)*rod.restLengths_m[local]!*(state.solidFraction[a]!+state.solidFraction[b]!)/2;
}
