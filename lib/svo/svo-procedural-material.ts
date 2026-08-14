import type { Vec3 } from "../core/model";
import { SVO_MATERIAL_FUNCTION_IDS } from "./svo-material-abi";
import type { LinearRgb } from "../core/webgpu-lighting";

/**
 * No resources are needed: these stable policies are selected by the material
 * record.
 *
 * A policy is up to three scales of the same value noise, because that is what
 * separates *mottle* from *grain*. One scale can say "this surface varies"; it
 * cannot say what the surface is made of. Granite is a fine speckle a few
 * millimetres across, sparse crystals an order of magnitude larger, and a
 * roughness that moves with both — three statements about three scales, and a
 * single frequency can carry only one of them.
 *
 * All three read the same world position, so the grain does not know where one
 * primitive ends and the next begins: a coping spelled as a chain of a hundred
 * cones is one continuous stone, not a hundred stones.
 */
export interface SvoProceduralMaterialPolicy {
  readonly functionId: number;
  readonly key: string;
  readonly seed: number;
  /** The coarse scale: the mottle, and the field the flecks are cut out of. */
  readonly frequency_mInv: readonly [number, number, number];
  readonly colorAmplitude: number;
  readonly roughnessAmplitude: number;
  /**
   * The fine scale, as a multiple of `frequency_mInv`. Together with
   * `detailWeight` this is the grain.
   *
   * It is bounded from below by the pixel, and nothing here can fix that: the
   * material has no filtering, so a grain finer than a pixel aliases rather than
   * blurs. At the hero pond's own framing a pixel is about 2.9 mm on the near
   * coping, so a 4.5 mm speckle is one and a half pixels and shimmers slightly;
   * on any close view it is eight or more and is the point of the surface. The
   * real fix is a distance-driven octave count, which is a shading-path change
   * and not a number in this table.
   */
  readonly detailOctave: number;
  /** Share of both amplitudes the fine scale carries, in [0, 1]. Zero is one scale. */
  readonly detailWeight: number;
  /**
   * Quantile of the coarse field above which a crystal shows, in [0, 1).
   *
   * Value noise concentrates near a half, so this is well up the tail: three
   * quarters leaves flecks scattered rather than tiled, which is what a crystal
   * in a matrix looks like and what an amplitude alone cannot produce.
   */
  readonly fleckThreshold: number;
  /**
   * How far a crystal lifts albedo and drops roughness, in their own units.
   * Zero is no flecks and costs nothing. One number for both because a crystal
   * face is lighter *and* smoother than the matrix around it; splitting them
   * would invite a fleck that is bright without being shiny, which reads as dirt.
   */
  readonly fleckAmplitude: number;
}

export const SVO_PROCEDURAL_VARIATION_ACTIVE = 0x8000_0000;

/** One scale, no flecks: the shape every policy had before granite needed three. */
const SINGLE_SCALE = { detailOctave: 1, detailWeight: 0, fleckThreshold: 0.75, fleckAmplitude: 0 } as const;

export const SVO_PROCEDURAL_MATERIAL_POLICIES = Object.freeze([
  { functionId: SVO_MATERIAL_FUNCTION_IDS.architecturalSurface, key: "architectural-surface", seed: 0x243f_6a88, frequency_mInv: [2.25, 2.25, 2.25], colorAmplitude: 0.10, roughnessAmplitude: 0.055, ...SINGLE_SCALE },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.wood, key: "wood", seed: 0x85a3_08d3, frequency_mInv: [1.7, 7.5, 1.7], colorAmplitude: 0.12, roughnessAmplitude: 0.065, ...SINGLE_SCALE },
  /**
   * Granite, and the one policy in this table that is a *material* rather than a
   * variation.
   *
   * It was 5/m at 0.08 — a 200 mm mottle, which on a 20 mm pebble is one flat
   * tone per pebble and on a coping is a cloud. Stone at arm's length does not
   * look like that. What it looks like is 14 mm crystals in a 4.5 mm speckle,
   * which is what these three scales are: the coarse field carries the crystals
   * and a slow value drift, the fine field carries the speckle at not quite half
   * the amplitude, and roughness follows both so the crest catches the light
   * differently from the flanks a centimetre away.
   *
   * The amplitudes look large and are not, and this is worth stating because the
   * first attempt at them was three times too small. `colorAmplitude` bounds the
   * excursion, and value noise concentrates hard around its mean: a field of
   * trilinearly interpolated uniform hashes has a standard deviation near 0.15,
   * not 0.29, so a stated amplitude of `a` delivers about `0.3 a` of contrast in
   * practice. Measured on the rendered rim, 0.13 produced six sRGB levels of
   * total variation across a 55 mm patch — form shading included — which is a
   * clean surface with a rumour of grain on it. These numbers are what read as
   * stone at arm's length; the bound is never reached.
   *
   * This is shared by every stone in every set — pebbles, plinths, kerbs — and
   * that is a deliberate choice rather than an oversight: they are all the same
   * rock. If a set ever wants polished marble beside rough granite, the answer is
   * a second function ID, not a compromise between them in this row.
   */
  { functionId: SVO_MATERIAL_FUNCTION_IDS.stone, key: "stone", seed: 0x1319_8a2e, frequency_mInv: [72.0, 72.0, 72.0], colorAmplitude: 0.30, roughnessAmplitude: 0.22, detailOctave: 3.1, detailWeight: 0.45, fleckThreshold: 0.74, fleckAmplitude: 0.24 },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.foliage, key: "foliage", seed: 0x0370_7344, frequency_mInv: [9.0, 6.0, 9.0], colorAmplitude: 0.09, roughnessAmplitude: 0.045, ...SINGLE_SCALE },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.ceramic, key: "ceramic", seed: 0xa409_3822, frequency_mInv: [3.5, 3.5, 3.5], colorAmplitude: 0.045, roughnessAmplitude: 0.035, ...SINGLE_SCALE },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.brushedMetal, key: "brushed-metal", seed: 0x299f_31d0, frequency_mInv: [12.0, 2.0, 12.0], colorAmplitude: 0.035, roughnessAmplitude: 0.055, ...SINGLE_SCALE },
  { functionId: SVO_MATERIAL_FUNCTION_IDS.organic, key: "organic", seed: 0x082e_fa98, frequency_mInv: [6.0, 4.0, 6.0], colorAmplitude: 0.07, roughnessAmplitude: 0.06, ...SINGLE_SCALE },
  /**
   * Fired plaster, and the only policy in this table that touches no albedo at
   * all. It is the closure a *porcelain* scene wears — the ground, the coping,
   * the boulders, the pebbles, the tree, all of it — and the whole of its job is
   * to keep white surfaces white while still giving them a sheen that travels.
   *
   * `colorAmplitude` is zero on purpose, and it is the point of the row rather
   * than an omission. It was 0.07 with a 6.2 mm speckle riding on it, argued for
   * as "a cloud you can see and not one you can name" — and on a set that is
   * white *everywhere*, a cloud you can see is the thing that reads as dirt. The
   * failure mode on white was never flatness; it is grubbiness, and every gram
   * of albedo modulation on fired porcelain spends itself on that. So the albedo
   * is exactly the authored value at every point on every surface, which is what
   * "white everywhere" has to mean if it means anything: two adjacent stones
   * differ because they were authored at different values, never because a noise
   * field passed between them. Zero is also *cheap* — `evaluateSvoProceduralMaterial`
   * and its WGSL mirror both skip the colour field's noise evaluation outright
   * when the amplitude is zero, so the closure that now shades an entire scene
   * costs one value-noise fetch per hit rather than two.
   *
   * Roughness is where the form goes, and it is all that is left here. 0.12 on
   * the porcelain's 0.62 wanders the sheen through roughly 0.58 to 0.66 in
   * normal use and 0.50 to 0.74 at the extremes — never near enough to a mirror
   * or to chalk to change what the surface is, but enough that a curved white
   * wall has a highlight that moves across it instead of one flat tone. A single
   * scale at 4.5 /m is a 222 mm drift: about the width of the near coping, so it
   * crosses a form rather than tiling it. The 162.5 /m speckle this row used to
   * carry is gone with the colour it was there to support — at the hero framing
   * a pixel is roughly 2.9 mm on the near coping, so a 6.2 mm cell was two
   * pixels of shimmer, which is grain by any name and grain is what was asked to
   * leave.
   *
   * No flecks, for the same reason as before: a crystal in white plaster does
   * not read as a crystal, it reads as a speck of dirt.
   */
  { functionId: SVO_MATERIAL_FUNCTION_IDS.plaster, key: "plaster", seed: 0xec4e_6c89, frequency_mInv: [4.5, 4.5, 4.5], colorAmplitude: 0, roughnessAmplitude: 0.12, ...SINGLE_SCALE },
] as const satisfies readonly SvoProceduralMaterialPolicy[]);

/** Salts for the second and third fields. Distinct, and stable ABI values. */
const DETAIL_SALT = 0x9e37_79b9;
const ROUGHNESS_SALT = 0x68bc_21eb;
const FLECK_SALT = 0x2545_f491;

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
  // Guarded rather than weighted by zero, in both directions: this is up to two
  // noise evaluations per shaded hit on a path already measured as
  // primary-visibility bound. Most policies do not want the fine scales, and
  // `plaster` — the closure a whole porcelain scene wears — does not want the
  // colour field at all. A half is the field's own mean, so the guarded arm is
  // exactly what the unguarded one computes when the amplitude is zero.
  const colorVaries = policy.colorAmplitude > 0;
  let tone = colorVaries ? sampleSvoProceduralNoise(position_m, policy.frequency_mInv, policy.seed) : 0.5;
  let roughnessNoise = sampleSvoProceduralNoise(position_m, policy.frequency_mInv, policy.seed ^ ROUGHNESS_SALT);
  if (policy.detailWeight > 0) {
    const fine = policy.frequency_mInv.map((value) => f32(value * policy.detailOctave)) as [number, number, number];
    if (colorVaries) {
      tone = interpolate(tone, sampleSvoProceduralNoise(position_m, fine, policy.seed ^ DETAIL_SALT), policy.detailWeight);
    }
    roughnessNoise = interpolate(
      roughnessNoise,
      sampleSvoProceduralNoise(position_m, fine, policy.seed ^ ROUGHNESS_SALT ^ DETAIL_SALT),
      policy.detailWeight,
    );
  }
  let fleck = 0;
  if (policy.fleckAmplitude > 0) {
    const crystal = sampleSvoProceduralNoise(position_m, policy.frequency_mInv, policy.seed ^ FLECK_SALT);
    fleck = Math.max(0, crystal - policy.fleckThreshold) / Math.max(1e-4, 1 - policy.fleckThreshold);
  }
  const colorScale = 1 + policy.colorAmplitude * (2 * tone - 1) + policy.fleckAmplitude * fleck;
  return {
    baseColorLinear: baseColorLinear.map((channel) => Math.min(1, Math.max(0, channel * colorScale))) as [number, number, number],
    roughness: Math.min(1, Math.max(0.04,
      roughness + policy.roughnessAmplitude * (2 * roughnessNoise - 1) - policy.fleckAmplitude * fleck)),
    variationFlags: (SVO_PROCEDURAL_VARIATION_ACTIVE | functionId) >>> 0,
  };
}

const wgslPolicies = SVO_PROCEDURAL_MATERIAL_POLICIES.map((policy, index) => `${index === 0 ? "if" : "else if"}(functionId==${policy.functionId}u){
    frequency=vec3f(${policy.frequency_mInv.map((value) => value.toFixed(8)).join(",")});seed=${policy.seed}u;colorAmplitude=${policy.colorAmplitude.toFixed(8)};roughnessAmplitude=${policy.roughnessAmplitude.toFixed(8)};detailOctave=${policy.detailOctave.toFixed(8)};detailWeight=${policy.detailWeight.toFixed(8)};fleckThreshold=${policy.fleckThreshold.toFixed(8)};fleckAmplitude=${policy.fleckAmplitude.toFixed(8)};
  }`).join("\n  ");

/**
 * The hash and the value noise on their own, without the material table.
 *
 * Split out because the shape language needs the *noise* and nothing else:
 * `svoFieldProgramWGSL` generates a domain warp over `svoProceduralNoise` and
 * deliberately does not carry its own copy, so that the CPU evaluator and the
 * shader agree by construction rather than by transcription. A consumer that
 * warps geometry has no use for a colour policy table, and the live voxelizer in
 * particular has no material shading at all — pulling the whole library in for
 * two functions would put a dozen unused policy branches in a maintenance pass
 * that holds itself to four storage bindings.
 *
 * It is the whole of this module's WGSL now — the per-pixel colour policy that
 * used to wrap it is gone — so there is still one noise in the tree and one place
 * it is written down.
 */
export const svoProceduralNoiseWGSL = /* wgsl */ `
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
`;

/**
 * The per-pixel WGSL variation policy that used to live here is deleted, for the
 * same reason the terrain one was: it coloured a surface from its world position,
 * which is a question the voxel cannot answer and the render path no longer asks.
 *
 * `svoProceduralNoiseWGSL` above is retained and is still emitted by both the
 * voxeliser and the dry pass — the field-program tape and the cluster fields are
 * *geometry*, and they are built on it.
 */
