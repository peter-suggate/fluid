import type { Vec3 } from "./model";
import { SVO_MATERIAL_FUNCTION_IDS } from "./svo-material-abi";
import type { LinearRgb } from "./webgpu-lighting";

/** No resources are needed: these stable policies are selected by the material record. */
export interface SvoProceduralMaterialPolicy {
  readonly functionId: number;
  readonly key: string;
  readonly seed: number;
  readonly frequency_mInv: readonly [number, number, number];
  readonly colorAmplitude: number;
  readonly roughnessAmplitude: number;
}

export const SVO_PROCEDURAL_VARIATION_ACTIVE = 0x8000_0000;

export const SVO_PROCEDURAL_MATERIAL_POLICIES = Object.freeze([
  { functionId: SVO_MATERIAL_FUNCTION_IDS.architecturalSurface, key: "architectural-surface", seed: 0x243f_6a88, frequency_mInv: [2.25, 2.25, 2.25], colorAmplitude: 0.10, roughnessAmplitude: 0.055 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.wood, key: "wood", seed: 0x85a3_08d3, frequency_mInv: [1.7, 7.5, 1.7], colorAmplitude: 0.12, roughnessAmplitude: 0.065 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.stone, key: "stone", seed: 0x1319_8a2e, frequency_mInv: [5.0, 5.0, 5.0], colorAmplitude: 0.08, roughnessAmplitude: 0.075 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.foliage, key: "foliage", seed: 0x0370_7344, frequency_mInv: [9.0, 6.0, 9.0], colorAmplitude: 0.09, roughnessAmplitude: 0.045 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.ceramic, key: "ceramic", seed: 0xa409_3822, frequency_mInv: [3.5, 3.5, 3.5], colorAmplitude: 0.045, roughnessAmplitude: 0.035 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.brushedMetal, key: "brushed-metal", seed: 0x299f_31d0, frequency_mInv: [12.0, 2.0, 12.0], colorAmplitude: 0.035, roughnessAmplitude: 0.055 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.organic, key: "organic", seed: 0x082e_fa98, frequency_mInv: [6.0, 4.0, 6.0], colorAmplitude: 0.07, roughnessAmplitude: 0.06 },
] as const satisfies readonly SvoProceduralMaterialPolicy[]);

const policyByFunctionId = new Map<number, SvoProceduralMaterialPolicy>(
  SVO_PROCEDURAL_MATERIAL_POLICIES.map((policy) => [policy.functionId, policy]),
);

export interface SvoProceduralMaterialSample {
  baseColorLinear: [number, number, number];
  roughness: number;
  variationFlags: number;
}

const f32 = Math.fround;

function hashMix(value: number): number {
  let result = value >>> 0;
  result = (result ^ (result >>> 16)) >>> 0;
  result = Math.imul(result, 0x7feb_352d) >>> 0;
  result = (result ^ (result >>> 15)) >>> 0;
  result = Math.imul(result, 0x846c_a68b) >>> 0;
  return (result ^ (result >>> 16)) >>> 0;
}

/** Exact uint32 CPU mirror of `svoProceduralHashCell`. */
export function svoProceduralHashCell(cell: readonly [number, number, number], seed: number): number {
  let hash = seed >>> 0;
  hash = (hash ^ Math.imul(cell[0] | 0, 0x9e37_79b1)) >>> 0;
  hash = (hash ^ Math.imul(cell[1] | 0, 0x85eb_ca77)) >>> 0;
  hash = (hash ^ Math.imul(cell[2] | 0, 0xc2b2_ae3d)) >>> 0;
  return f32((hashMix(hash) & 0x00ff_ffff) / 0x00ff_ffff);
}

function interpolate(left: number, right: number, amount: number): number {
  return f32(f32(left * f32(1 - amount)) + f32(right * amount));
}

/** Continuous seeded value noise; identical world points are independent of primitive tessellation. */
export function sampleSvoProceduralNoise(position_m: Vec3, frequency_mInv: readonly [number, number, number], seed: number): number {
  const point = [
    f32(position_m.x * frequency_mInv[0]),
    f32(position_m.y * frequency_mInv[1]),
    f32(position_m.z * frequency_mInv[2]),
  ] as const;
  const cell = point.map(Math.floor) as [number, number, number];
  const fraction = point.map((coordinate, axis) => {
    const linear = f32(coordinate - cell[axis]);
    return f32(f32(linear * linear) * f32(3 - f32(2 * linear)));
  }) as [number, number, number];
  const corner = (x: number, y: number, z: number) => svoProceduralHashCell(
    [cell[0] + x, cell[1] + y, cell[2] + z], seed,
  );
  const x00 = interpolate(corner(0, 0, 0), corner(1, 0, 0), fraction[0]);
  const x10 = interpolate(corner(0, 1, 0), corner(1, 1, 0), fraction[0]);
  const x01 = interpolate(corner(0, 0, 1), corner(1, 0, 1), fraction[0]);
  const x11 = interpolate(corner(0, 1, 1), corner(1, 1, 1), fraction[0]);
  return interpolate(interpolate(x00, x10, fraction[1]), interpolate(x01, x11, fraction[1]), fraction[2]);
}

/** CPU authoring/test mirror for the binding-free direct-SVO material function. */
export function evaluateSvoProceduralMaterial(
  functionId: number,
  baseColorLinear: LinearRgb,
  roughness: number,
  position_m: Vec3,
): SvoProceduralMaterialSample {
  const policy = policyByFunctionId.get(functionId);
  if (!policy) return {
    baseColorLinear: [...baseColorLinear],
    roughness: Math.min(1, Math.max(0.04, roughness)),
    variationFlags: 0,
  };
  const tone = sampleSvoProceduralNoise(position_m, policy.frequency_mInv, policy.seed);
  const roughnessNoise = sampleSvoProceduralNoise(position_m, policy.frequency_mInv, policy.seed ^ 0x68bc_21eb);
  const colorScale = 1 + policy.colorAmplitude * (2 * tone - 1);
  return {
    baseColorLinear: baseColorLinear.map((channel) => Math.min(1, Math.max(0, channel * colorScale))) as [number, number, number],
    roughness: Math.min(1, Math.max(0.04, roughness + policy.roughnessAmplitude * (2 * roughnessNoise - 1))),
    variationFlags: (SVO_PROCEDURAL_VARIATION_ACTIVE | functionId) >>> 0,
  };
}

const wgslPolicies = SVO_PROCEDURAL_MATERIAL_POLICIES.map((policy, index) => `${index === 0 ? "if" : "else if"}(functionId==${policy.functionId}u){
    frequency=vec3f(${policy.frequency_mInv.map((value) => value.toFixed(8)).join(",")});seed=${policy.seed}u;colorAmplitude=${policy.colorAmplitude.toFixed(8)};roughnessAmplitude=${policy.roughnessAmplitude.toFixed(8)};
  }`).join("\n  ");

/** Binding-free WGSL generated from the same stable policy table as the CPU mirror. */
export const svoProceduralMaterialWGSL = /* wgsl */ `
const SVO_PROCEDURAL_VARIATION_ACTIVE:u32=${SVO_PROCEDURAL_VARIATION_ACTIVE}u;
struct SvoProceduralMaterialSample{baseColorLinear:vec3f,roughness:f32,variationFlags:u32}
fn svoProceduralHashCell(cell:vec3i,seed:u32)->f32{
  var hash=seed;
  hash=hash^(bitcast<u32>(cell.x)*0x9e3779b1u);
  hash=hash^(bitcast<u32>(cell.y)*0x85ebca77u);
  hash=hash^(bitcast<u32>(cell.z)*0xc2b2ae3du);
  hash=hash^(hash>>16u);hash=hash*0x7feb352du;hash=hash^(hash>>15u);hash=hash*0x846ca68bu;hash=hash^(hash>>16u);
  return f32(hash&0x00ffffffu)/16777215.0;
}
fn svoProceduralNoise(position_m:vec3f,frequency_mInv:vec3f,seed:u32)->f32{
  let point=position_m*frequency_mInv;let cell=vec3i(floor(point));let linear=fract(point);let blend=linear*linear*(vec3f(3.0)-2.0*linear);
  let x00=mix(svoProceduralHashCell(cell+vec3i(0,0,0),seed),svoProceduralHashCell(cell+vec3i(1,0,0),seed),blend.x);
  let x10=mix(svoProceduralHashCell(cell+vec3i(0,1,0),seed),svoProceduralHashCell(cell+vec3i(1,1,0),seed),blend.x);
  let x01=mix(svoProceduralHashCell(cell+vec3i(0,0,1),seed),svoProceduralHashCell(cell+vec3i(1,0,1),seed),blend.x);
  let x11=mix(svoProceduralHashCell(cell+vec3i(0,1,1),seed),svoProceduralHashCell(cell+vec3i(1,1,1),seed),blend.x);
  return mix(mix(x00,x10,blend.y),mix(x01,x11,blend.y),blend.z);
}
fn svoProceduralMaterial(functionId:u32,baseColorLinear:vec3f,roughness:f32,position_m:vec3f)->SvoProceduralMaterialSample{
  var frequency=vec3f(0.0);var seed=0u;var colorAmplitude=0.0;var roughnessAmplitude=0.0;
  ${wgslPolicies}
  else{return SvoProceduralMaterialSample(baseColorLinear,clamp(roughness,0.04,1.0),0u);}
  let tone=svoProceduralNoise(position_m,frequency,seed);let roughnessNoise=svoProceduralNoise(position_m,frequency,seed^0x68bc21ebu);
  let variedColor=clamp(baseColorLinear*(1.0+colorAmplitude*(2.0*tone-1.0)),vec3f(0.0),vec3f(1.0));
  let variedRoughness=clamp(roughness+roughnessAmplitude*(2.0*roughnessNoise-1.0),0.04,1.0);
  return SvoProceduralMaterialSample(variedColor,variedRoughness,SVO_PROCEDURAL_VARIATION_ACTIVE|functionId);
}
`;
