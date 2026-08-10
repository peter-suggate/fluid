/** Linear RGB tuple used by the CPU side of the shared GPU material contract. */
export type LinearRgb = readonly [red: number, green: number, blue: number];

/**
 * Renderer-independent material inputs for the unified WebGPU lighting model.
 *
 * The current scene stores these values in several different GPU records. The
 * WGSL contract deliberately operates on a local value instead of prescribing
 * a bind-group layout, so raster bodies, optical interfaces, and sparse voxel
 * materials can all construct the same closure from their existing buffers.
 */
export interface UnifiedLightingMaterial {
  baseColorLinear: LinearRgb;
  emissiveLinear: LinearRgb;
  roughness: number;
  /** Zero is a dielectric; one uses base color as the conductor F0. */
  metallic?: number;
  specularF0Linear: LinearRgb;
  specularWeight: number;
  ambientDiffuse: number;
  rimColorLinear: LinearRgb;
  rimWeight: number;
}

export interface UnifiedLightingSample {
  shadingNormal: LinearRgb;
  /** Defaults to shadingNormal for legacy analytic and raster surfaces. */
  geometricNormal?: LinearRgb;
  towardViewer: LinearRgb;
  towardLight: LinearRgb;
  lightColorLinear: LinearRgb;
}

/**
 * Clean water, as the default every scene starts from.
 *
 * These are the pure-water coefficients: red is absorbed an order of magnitude
 * harder than green and blue, which is why a swimming pool is blue and a glass
 * of water is not. They are the right numbers for a tank a metre across and the
 * wrong ones for a pond: at the hero garden's 0.115 m of still depth
 * Beer-Lambert leaves transmission (0.950, 0.990, 0.993), so red is 4.6 % down
 * on green and the water carries essentially no hue. See `WaterOpticsAuthoring`
 * for why that is a scene decision rather than a bug in this table.
 */
export const WATER_OPTICS = Object.freeze({
  indexOfRefraction: 1.333,
  fresnelF0: 0.02037,
  absorption: [0.45, 0.09, 0.06] as LinearRgb,
  scatter: [0.012, 0.055, 0.049] as LinearRgb
});

/**
 * What a scene may say about the water it is filled with.
 *
 * Every field is optional and falls back to `WATER_OPTICS`, so a document that
 * says nothing is byte-identical to the frozen table and no existing scene
 * moves. The reason this exists at all is depth: absorption is a *rate*, and a
 * rate that reads as clean water over a metre reads as nothing at all over a
 * hand's breadth. A vessel whose water is meant to carry colour at pond depth
 * needs coefficients close to an order of magnitude larger — and which pond it
 * is deciding that is precisely why the table cannot be the only authority.
 *
 * Units are per metre, scene-linear RGB, matching `unifiedBeerLambert`.
 */
export interface WaterOpticsAuthoring {
  /** Beer-Lambert extinction per metre, per channel. */
  readonly absorption_mInv?: LinearRgb;
  /** Radiance the medium adds back as the beam is extinguished. */
  readonly scatter?: LinearRgb;
  readonly indexOfRefraction?: number;
  readonly fresnelF0?: number;
  /**
   * How much of the refracted caustic modulation reaches the receiver, in
   * [0, 1]. Zero skips the caustic pass entirely for this scene.
   */
  readonly causticStrength?: number;
}

export interface ResolvedWaterOptics {
  readonly absorption_mInv: [number, number, number];
  readonly scatter: [number, number, number];
  readonly indexOfRefraction: number;
  readonly fresnelF0: number;
  readonly causticStrength: number;
}

/** Default caustic strength: the map is consumed at full modulation. */
export const WATER_CAUSTIC_DEFAULT_STRENGTH = 1;

function opticalChannels(value: LinearRgb | undefined, fallback: LinearRgb): [number, number, number] {
  if (!value || value.length !== 3) return [...fallback] as [number, number, number];
  return value.map((channel, index) =>
    Number.isFinite(channel) && channel >= 0 ? channel : fallback[index]) as [number, number, number];
}

function opticalScalar(value: number | undefined, fallback: number, minimum: number, maximum: number): number {
  if (value === undefined || !Number.isFinite(value)) return fallback;
  return Math.min(maximum, Math.max(minimum, value));
}

/**
 * Fold an authored block onto the frozen defaults.
 *
 * Out-of-range values fall back per field rather than throwing: this runs on
 * the render path for a document a user may be editing live, and a negative
 * absorption typed into an inspector should leave the water looking like water
 * rather than take the frame down.
 */
export function resolveWaterOptics(authored?: WaterOpticsAuthoring): ResolvedWaterOptics {
  return {
    absorption_mInv: opticalChannels(authored?.absorption_mInv, WATER_OPTICS.absorption),
    scatter: opticalChannels(authored?.scatter, WATER_OPTICS.scatter),
    // Below one there is no interface to refract at, and above two the
    // composite's exit refraction total-internal-reflects on nearly every ray.
    indexOfRefraction: opticalScalar(authored?.indexOfRefraction, WATER_OPTICS.indexOfRefraction, 1, 2),
    fresnelF0: opticalScalar(authored?.fresnelF0, WATER_OPTICS.fresnelF0, 0, 1),
    causticStrength: opticalScalar(authored?.causticStrength, WATER_CAUSTIC_DEFAULT_STRENGTH, 0, 1),
  };
}

/**
 * The directional key as the water sees it.
 *
 * The renderer hands this the same key `buildSvoSceneLights` resolves through
 * `svoSceneLighting`, so dry geometry, water highlights, and caustics cannot
 * disagree about the rig. `authored` distinguishes that resolved packet from
 * the legacy fallback retained for callers which pass no scene lighting.
 */
export interface ResolvedWaterKeyLight {
  readonly direction: [number, number, number];
  /** Colour times intensity: the radiance the caustic pass and highlight use. */
  readonly radianceLinear: [number, number, number];
  readonly authored: boolean;
}

/** Mirrors `buildSvoSceneLights`' directional defaults exactly. */
const WATER_KEY_DEFAULT_DIRECTION: LinearRgb = [-0.45, 0.86, 0.28];
const WATER_KEY_DEFAULT_COLOR: LinearRgb = [1.04, 1, 0.91];

export function resolveWaterKeyLight(directional?: {
  readonly direction?: LinearRgb;
  readonly colorLinear?: LinearRgb;
  readonly intensity?: number;
}): ResolvedWaterKeyLight {
  const authored = Boolean(directional?.direction
    && directional.direction.length === 3
    && directional.direction.every(Number.isFinite)
    && Math.hypot(...directional.direction) > 1e-12);
  const direction = normalize3(
    authored ? directional!.direction! : WATER_KEY_DEFAULT_DIRECTION, WATER_KEY_DEFAULT_DIRECTION,
  );
  const intensity = opticalScalar(directional?.intensity, 1, 0, 64);
  const color = opticalChannels(directional?.colorLinear, WATER_KEY_DEFAULT_COLOR);
  return { direction, radianceLinear: color.map((channel) => channel * intensity) as [number, number, number], authored };
}

/**
 * Seven vec4 lanes shared by the water composite and the caustic projection.
 *
 * Lanes 0-3 are the scene's medium and key; lanes 4-5 describe the caustic
 * receiver, which the renderer resamples from `scene.terrain`; lane 6 is the
 * final display grade. They live in one buffer because all are per-scene facts
 * consumed by the same compositor, and a second uniform binding buys nothing.
 */
export const WATER_SCENE_OPTICS_LANES = 7;
export const WATER_SCENE_OPTICS_FLOATS = 4 * WATER_SCENE_OPTICS_LANES;
export const WATER_SCENE_OPTICS_BYTES = 4 * WATER_SCENE_OPTICS_FLOATS;
/** Where the pipeline's own receiver description starts, in floats. */
export const WATER_SCENE_OPTICS_RECEIVER_FLOAT_OFFSET = 16;

/** Complete document packet. The pipeline preserves and overwrites receiver lanes 4-5. */
export function packWaterSceneOptics(
  optics: ResolvedWaterOptics,
  key: ResolvedWaterKeyLight,
  grade: ResolvedDisplayGrade = DISPLAY_GRADE_NEUTRAL,
): Float32Array {
  return new Float32Array([
    ...optics.absorption_mInv, optics.indexOfRefraction,
    ...optics.scatter, optics.fresnelF0,
    ...key.direction, key.authored ? 1 : 0,
    ...key.radianceLinear, optics.causticStrength,
    // Two receiver lanes are filled by RasterWaterPipeline after it resamples
    // the scene terrain. Keeping their slots in this complete document packet
    // lets setSceneOptics update every other lane without stale values.
    0, 0, 0, 0,
    0, 0, 0, 0,
    // Lane 6 had two free floats and the balance needs three channels, so the
    // green gain is *not* transported: it is reconstructed in the shader from
    // the luminance-normalisation constraint the resolver already enforces,
    // `dot(gain, REC709_LUMINANCE) == 1`. That is exact rather than a rounding,
    // and it is why `resolveDisplayGrade` normalises rather than leaving the
    // gains as authored. Widening the block would have moved every lane offset
    // in two shaders for one float.
    grade.exposure, SCENE_TONE_CURVE_CODES[grade.toneCurve], grade.whiteBalance[0], grade.whiteBalance[2],
  ]);
}

/**
 * WGSL mirror of the lanes above, plus the receiver's bilinear fetch.
 *
 * The bindings are emitted by the library rather than by its callers, so the
 * two shaders that consume this uniform cannot bind it to different slots or
 * declare two divergent copies of the struct. `waterReceiver` holds receiver
 * heights in metres on a container-aligned lattice, and `waterReceiverHeight`
 * is the exact WGSL mirror of `sampleTerrainGrid` in lib/terrain.ts — clamped
 * bilinear on that lattice — so a caustic lands on the surface the renderer
 * actually draws rather than on a plane chosen to make the maths easy.
 */
export function waterSceneOpticsShaderLibrary(group: number, opticsBinding: number, receiverBinding: number): string {
  return /* wgsl */ `
struct WaterSceneOptics {
  absorptionIor: vec4f,
  scatterFresnel: vec4f,
  keyDirectionAuthored: vec4f,
  keyRadianceCaustic: vec4f,
  receiverOriginSpacing: vec4f,
  receiverSize: vec4f,
  displayGrade: vec4f,
}
@group(${group}) @binding(${opticsBinding}) var<uniform> waterScene: WaterSceneOptics;
@group(${group}) @binding(${receiverBinding}) var waterReceiver: texture_2d<f32>;
fn waterAbsorption() -> vec3f { return max(waterScene.absorptionIor.xyz, vec3f(0.0)); }
fn waterIndexOfRefraction() -> f32 { return max(waterScene.absorptionIor.w, 1.0); }
fn waterScatter() -> vec3f { return max(waterScene.scatterFresnel.xyz, vec3f(0.0)); }
fn waterFresnelF0() -> f32 { return clamp(waterScene.scatterFresnel.w, 0.0, 1.0); }
fn waterKeyAuthored() -> bool { return waterScene.keyDirectionAuthored.w > 0.5; }
fn waterAuthoredKeyDirection() -> vec3f { return normalize(waterScene.keyDirectionAuthored.xyz); }
fn waterKeyRadiance() -> vec3f { return max(waterScene.keyRadianceCaustic.xyz, vec3f(0.0)); }
fn waterCausticStrength() -> f32 { return clamp(waterScene.keyRadianceCaustic.w, 0.0, 1.0); }
fn waterDisplayExposure() -> f32 { return max(waterScene.displayGrade.x, 0.0); }
fn waterDisplayToneCurve() -> u32 { return u32(round(max(waterScene.displayGrade.y, 0.0))); }
// Green is reconstructed from dot(gain, REC709_LUMINANCE) == 1, the constraint
// resolveDisplayGrade enforces; see the note in packWaterSceneOptics.
fn waterDisplayWhiteBalance() -> vec3f {
  let red = max(waterScene.displayGrade.z, 0.0);
  let blue = max(waterScene.displayGrade.w, 0.0);
  if (red <= 0.0 && blue <= 0.0) { return vec3f(1.0); }
  let green = (1.0 - ${REC709_LUMINANCE[0]} * red - ${REC709_LUMINANCE[2]} * blue) / ${REC709_LUMINANCE[1]};
  return vec3f(red, max(green, 0.0), blue);
}
fn waterReceiverPresent() -> bool { return waterScene.receiverSize.x >= 2.0 && waterScene.receiverSize.y >= 2.0; }
fn waterReceiverHeight(x: f32, z: f32) -> f32 {
  if (!waterReceiverPresent()) { return 0.0; }
  let size = waterScene.receiverSize.xy;
  let spacing = max(waterScene.receiverOriginSpacing.z, 1e-6);
  let fx = clamp((x - waterScene.receiverOriginSpacing.x) / spacing, 0.0, size.x - 1.0);
  let fz = clamp((z - waterScene.receiverOriginSpacing.y) / spacing, 0.0, size.y - 1.0);
  let i0 = vec2i(i32(floor(fx)), i32(floor(fz)));
  let i1 = min(i0 + vec2i(1), vec2i(i32(size.x) - 1, i32(size.y) - 1));
  let t = vec2f(fx - f32(i0.x), fz - f32(i0.y));
  let h00 = textureLoad(waterReceiver, vec2i(i0.x, i0.y), 0).x;
  let h10 = textureLoad(waterReceiver, vec2i(i1.x, i0.y), 0).x;
  let h01 = textureLoad(waterReceiver, vec2i(i0.x, i1.y), 0).x;
  let h11 = textureLoad(waterReceiver, vec2i(i1.x, i1.y), 0).x;
  return mix(mix(h00, h10, t.x), mix(h01, h11, t.x), t.y);
}
`;
}

export const GLASS_OPTICS = Object.freeze({
  indexOfRefraction: 1.5,
  fresnelF0: 0.04,
  tint: [0.30, 0.58, 0.54] as LinearRgb
});

/** CPU mirror used in tests and tooling that author optical materials. */
export function dielectricFresnel(cosine: number, f0: number) {
  const c = Math.min(1, Math.max(0, cosine));
  const base = Math.min(1, Math.max(0, f0));
  return base + (1 - base) * (1 - c) ** 5;
}

/** CPU mirror of the shared absorption closure. */
export function beerLambert(absorption: LinearRgb, distance: number): [number, number, number] {
  const d = Math.max(0, distance);
  return [Math.exp(-Math.max(0, absorption[0]) * d), Math.exp(-Math.max(0, absorption[1]) * d), Math.exp(-Math.max(0, absorption[2]) * d)];
}

/**
 * The tone curves a document may ask for by name.
 *
 * `reinhard` is `x / (x + 1)`, what this renderer has always applied. It has no
 * shoulder in the photographic sense: it approaches white asymptotically and
 * spends most of its range compressing, so a surface at unit radiance lands at
 * 0.5 linear — 0.73 after gamma — and reads as a light grey. The only way to
 * get a white out of it is to over-light, which is what a scene that wants
 * near-white plaster has had to do.
 *
 * `aces` is the Narkowicz 2015 fit of the ACES RRT+ODT, the standard cheap
 * filmic curve. It is chosen over Hable's because it is one rational with no
 * free parameters to mis-tune, and over Khronos PBR Neutral because the
 * saturation-preserving behaviour PBR Neutral exists for is the opposite of
 * what is wanted here: the reference is a photograph, and its highlights do
 * desaturate. It puts unit radiance at 0.80 linear (0.91 after gamma) with a
 * real roll-off above it, so plaster can run near-white on a key of 1.0.
 *
 * The trade is contrast, not brightness: ACES also pulls the toe down, so the
 * shadowed underside of a rock goes darker than Reinhard left it. That is the
 * point — the flatness of the current frame is the curve's, and no amount of
 * relighting fixes a transfer function.
 */
export type SceneToneCurve = "reinhard" | "aces";

/** WGSL selector values. Ordered so zero is the historical behaviour. */
export const SCENE_TONE_CURVE_CODES = Object.freeze({ reinhard: 0, aces: 1 } as const);

/**
 * What a scene may say about how its radiance becomes pixels.
 *
 * Exposure is a plain linear scale applied before the curve, in stops-free
 * units: 2 is one stop up. It exists so a scene can be graded without touching
 * its lights, which are physical facts about the set and should not be doing
 * two jobs.
 */
export interface DisplayGradeAuthoring {
  readonly exposure?: number;
  readonly toneCurve?: SceneToneCurve;
  /**
   * Per-channel scene-linear gains, before exposure and the curve.
   *
   * H1 of `docs/hero-fidelity-1000x-handoff.md` asks for white balance, and the
   * measurement that motivates it is blunt: on the hero garden the frame's
   * neutral set sits at b* ≈ 0.5 while the plate's sits at b* ≈ 5. The set is
   * *neutral* and the plate is *warm*, and nothing in the light rig can fix
   * that without also changing the lit-to-shadow ratio, which is a physical
   * fact about the set rather than a grading choice.
   *
   * Authored freely; `resolveDisplayGrade` normalises for luminance, so this
   * knob changes only chromaticity and never brightness. That orthogonality is
   * what lets exposure and balance be solved independently — see
   * `tools/solve-hero-grade.ts`, which does exactly that.
   */
  readonly whiteBalance?: readonly [number, number, number];
}

export interface ResolvedDisplayGrade {
  readonly exposure: number;
  readonly toneCurve: SceneToneCurve;
  /** Luminance-normalised, so `dot(whiteBalance, REC709_LUMINANCE) === 1`. */
  readonly whiteBalance: readonly [number, number, number];
}

/**
 * Rec.709 luminance weights, shared by the resolver and both shader mirrors.
 *
 * The white balance is normalised against these, and the WGSL reconstructs the
 * green gain from them (see `packWaterSceneOptics`), so all three must be the
 * same three numbers or a warm grade quietly becomes a warm *and darker* one.
 */
export const REC709_LUMINANCE = Object.freeze([0.2126, 0.7152, 0.0722] as const);

/**
 * The grade a document gets when it authors nothing.
 *
 * Unit exposure and Reinhard, which together are the identity against every
 * frame this renderer has produced. Every default in this block exists to make
 * "scene says nothing" and "scene did not have the field" the same frame.
 */
export const DISPLAY_GRADE_NEUTRAL: ResolvedDisplayGrade = Object.freeze({
  exposure: 1,
  toneCurve: "reinhard",
  whiteBalance: Object.freeze([1, 1, 1] as const),
});

/** A channel gain outside this is a colour cast, not a balance; both clamp. */
export const DISPLAY_WHITE_BALANCE_RANGE = Object.freeze({ minimum: 0.25, maximum: 4 });

/** Exposure below this is a black frame and above it a white one; both clamp. */
export const DISPLAY_EXPOSURE_RANGE = Object.freeze({ minimum: 0.05, maximum: 20 });

export function resolveDisplayGrade(authored?: DisplayGradeAuthoring): ResolvedDisplayGrade {
  const clamped = (authored?.whiteBalance ?? DISPLAY_GRADE_NEUTRAL.whiteBalance).map((gain) =>
    opticalScalar(gain, 1, DISPLAY_WHITE_BALANCE_RANGE.minimum, DISPLAY_WHITE_BALANCE_RANGE.maximum),
  ) as [number, number, number];
  // Normalised so a neutral input keeps its luminance: balance is a hue
  // decision and exposure is a brightness decision, and a knob that did both
  // could not be solved for either.
  const luminance =
    REC709_LUMINANCE[0] * clamped[0] + REC709_LUMINANCE[1] * clamped[1] + REC709_LUMINANCE[2] * clamped[2];
  const scale = luminance > 1e-6 ? 1 / luminance : 1;
  return {
    exposure: opticalScalar(authored?.exposure, DISPLAY_GRADE_NEUTRAL.exposure,
      DISPLAY_EXPOSURE_RANGE.minimum, DISPLAY_EXPOSURE_RANGE.maximum),
    toneCurve: authored?.toneCurve === "aces" ? "aces" : DISPLAY_GRADE_NEUTRAL.toneCurve,
    whiteBalance: [clamped[0] * scale, clamped[1] * scale, clamped[2] * scale],
  };
}

function toneCurve(value: number, curve: SceneToneCurve): number {
  if (curve !== "aces") return value / (value + 1);
  return Math.min(1, Math.max(0,
    (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14)));
}

/**
 * The single final-output transform shared by raster and sparse-voxel frames.
 * Lighting and media code must keep values scene-linear and call this only
 * when writing the presentation target.
 *
 * The grade defaults to {@link DISPLAY_GRADE_NEUTRAL}, so the one-argument call
 * every existing caller makes is byte-identical to what it was.
 */
export function sceneLinearToDisplay(
  sceneLinear: LinearRgb,
  grade: ResolvedDisplayGrade = DISPLAY_GRADE_NEUTRAL,
): [number, number, number] {
  return sceneLinear.map((channel, index) => {
    const balanced = Math.max(0, Number.isFinite(channel) ? channel : 0) * grade.whiteBalance[index];
    return toneCurve(balanced * grade.exposure, grade.toneCurve) ** (1 / 2.2);
  }) as [number, number, number];
}

/**
 * Binding-free WGSL mirror of `sceneLinearToDisplay`.
 *
 * Two entry points rather than one parameterised entry point: `unifiedDisplayTransfer`
 * is called by presentation shaders that have no uniform to read a grade from
 * and must not grow a binding to acquire one, and it stays the exact expression
 * it has always been so a diff of the composite's output is a diff of the
 * scene. `unifiedDisplayGrade` is what a shader calls once it can supply the
 * scene's exposure and curve code.
 *
 * The frame that reaches the canvas is finished in exactly one place — the
 * `finish` helper of `compositeShader` in `lib/webgpu-water-pipeline.ts` — so
 * that is the single call site an authored grade has to travel through, and
 * until it does, `SceneDescription["lighting"].grade` is resolved and validated
 * but not yet applied.
 */
export const unifiedDisplayTransferShaderLibrary = /* wgsl */ `
fn unifiedToneCurve(exposed: vec3f, toneCurve: u32) -> vec3f {
  if (toneCurve == ${SCENE_TONE_CURVE_CODES.aces}u) {
    // Narkowicz 2015 ACES fit. One rational per channel, so it costs the same
    // as the Reinhard it replaces and has no transcendental to schedule.
    return clamp((exposed * (2.51 * exposed + vec3f(0.03)))
      / (exposed * (2.43 * exposed + vec3f(0.59)) + vec3f(0.14)), vec3f(0.0), vec3f(1.0));
  }
  return exposed / (exposed + vec3f(1.0));
}
fn unifiedDisplayGrade(sceneLinear: vec3f, exposure: f32, toneCurve: u32) -> vec3f {
  return unifiedDisplayGradeBalanced(sceneLinear, exposure, toneCurve, vec3f(1.0));
}
fn unifiedDisplayGradeBalanced(sceneLinear: vec3f, exposure: f32, toneCurve: u32, whiteBalance: vec3f) -> vec3f {
  let balanced = max(sceneLinear, vec3f(0.0)) * max(whiteBalance, vec3f(0.0));
  return pow(unifiedToneCurve(balanced * max(exposure, 0.0), toneCurve), vec3f(1.0 / 2.2));
}
fn unifiedDisplayTransfer(sceneLinear: vec3f) -> vec3f {
  let nonNegative = max(sceneLinear, vec3f(0.0));
  let toneMapped = nonNegative / (nonNegative + vec3f(1.0));
  return pow(toneMapped, vec3f(1.0 / 2.2));
}
`;

const PBR_PI = Math.PI;
const PBR_EPSILON = 1e-8;

function saturate(value: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

function canonicalColor(value: LinearRgb, maximum = Number.POSITIVE_INFINITY): [number, number, number] {
  return value.map((channel) => Math.min(maximum, Math.max(0, Number.isFinite(channel) ? channel : 0))) as [number, number, number];
}

function dot3(left: LinearRgb, right: LinearRgb): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function normalize3(value: LinearRgb, fallback: LinearRgb): [number, number, number] {
  const magnitude = Math.hypot(value[0], value[1], value[2]);
  if (Number.isFinite(magnitude) && magnitude > PBR_EPSILON) return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
  const fallbackMagnitude = Math.hypot(fallback[0], fallback[1], fallback[2]);
  if (Number.isFinite(fallbackMagnitude) && fallbackMagnitude > PBR_EPSILON) return [fallback[0] / fallbackMagnitude, fallback[1] / fallbackMagnitude, fallback[2] / fallbackMagnitude];
  return [0, 1, 0];
}

/** RGB Schlick Fresnel mirror used by the quantitative lighting tests. */
export function schlickFresnel(cosine: number, f0: LinearRgb): [number, number, number] {
  const grazing = (1 - saturate(cosine)) ** 5;
  const base = canonicalColor(f0, 1);
  return base.map((channel) => channel + (1 - channel) * grazing) as [number, number, number];
}

/**
 * Integrated GGX environment response without a sampled BRDF lookup table.
 *
 * A prefiltered environment already integrates the light lobe, but it still
 * needs the view/roughness half of the split-sum approximation. Multiplying it
 * by raw Schlick Fresnel is only correct for an ideal mirror: at a silhouette
 * Schlick reaches one regardless of roughness, turning a bright sky into a
 * one-pixel white contour around matte objects. This fit includes GGX masking
 * and shadowing, so rough surfaces retain a broad sheen without acquiring a
 * mirror-bright grazing rim.
 */
export function integratedEnvironmentBrdf(
  nDotV: number,
  roughness: number,
  f0: LinearRgb,
): [number, number, number] {
  const view = saturate(nDotV);
  const perceptualRoughness = saturate(roughness);
  const rX = 1 - perceptualRoughness;
  const rY = 0.0425 - 0.0275 * perceptualRoughness;
  const rZ = 1.04 - 0.572 * perceptualRoughness;
  const rW = -0.04 + 0.022 * perceptualRoughness;
  const a004 = Math.min(rX * rX, 2 ** (-9.28 * view)) * rX + rY;
  const scale = -1.04 * a004 + rZ;
  const bias = 1.04 * a004 + rW;
  return canonicalColor(f0, 1).map((channel) =>
    Math.min(1, Math.max(0, channel * scale + bias))) as [number, number, number];
}

/** Trowbridge-Reitz/GGX normal-distribution function. */
export function ggxNormalDistribution(nDotH: number, roughness: number): number {
  const perceptualRoughness = Math.min(1, Math.max(0.04, Number.isFinite(roughness) ? roughness : 1));
  const alpha = perceptualRoughness * perceptualRoughness;
  const alphaSquared = alpha * alpha;
  const cosine = saturate(nDotH);
  const denominator = cosine * cosine * (alphaSquared - 1) + 1;
  return alphaSquared / Math.max(PBR_PI * denominator * denominator, PBR_EPSILON);
}

/** Height-correlated Smith GGX visibility, equal to G/(4 NdotL NdotV). */
export function smithGgxVisibility(nDotV: number, nDotL: number, roughness: number): number {
  const view = saturate(nDotV);
  const light = saturate(nDotL);
  if (view <= 0 || light <= 0) return 0;
  const perceptualRoughness = Math.min(1, Math.max(0.04, Number.isFinite(roughness) ? roughness : 1));
  const alpha = perceptualRoughness * perceptualRoughness;
  const alphaSquared = alpha * alpha;
  const viewTerm = light * Math.sqrt(view * view * (1 - alphaSquared) + alphaSquared);
  const lightTerm = view * Math.sqrt(light * light * (1 - alphaSquared) + alphaSquared);
  return 0.5 / Math.max(viewTerm + lightTerm, PBR_EPSILON);
}

/**
 * CPU mirror of `shadeUnifiedSurface`. The direct-light result is a BRDF times
 * incident radiance and the geometric projected cosine. Ambient, rim, and
 * emissive are retained for compatibility with the existing authored scenes.
 */
export function evaluateUnifiedLighting(
  materialInput: UnifiedLightingMaterial,
  input: UnifiedLightingSample,
): [number, number, number] {
  const baseColor = canonicalColor(materialInput.baseColorLinear, 1);
  const emissive = canonicalColor(materialInput.emissiveLinear);
  const f0Dielectric = canonicalColor(materialInput.specularF0Linear, 1);
  const lightColor = canonicalColor(input.lightColorLinear);
  const roughness = Math.min(1, Math.max(0.04, Number.isFinite(materialInput.roughness) ? materialInput.roughness : 1));
  const metallic = saturate(materialInput.metallic ?? 0);
  const specularWeight = saturate(materialInput.specularWeight);
  const ambientDiffuse = saturate(materialInput.ambientDiffuse);
  const rimColor = canonicalColor(materialInput.rimColorLinear);
  const rimWeight = saturate(materialInput.rimWeight);

  const towardViewer = normalize3(input.towardViewer, [0, 0, 1]);
  let geometricNormal = normalize3(input.geometricNormal ?? input.shadingNormal, [0, 1, 0]);
  if (dot3(geometricNormal, towardViewer) < 0) geometricNormal = geometricNormal.map((value) => -value) as [number, number, number];
  let shadingNormal = normalize3(input.shadingNormal, geometricNormal);
  if (dot3(shadingNormal, geometricNormal) < 0) shadingNormal = shadingNormal.map((value) => -value) as [number, number, number];
  const towardLight = normalize3(input.towardLight, geometricNormal);
  const halfVector = normalize3([
    towardViewer[0] + towardLight[0],
    towardViewer[1] + towardLight[1],
    towardViewer[2] + towardLight[2],
  ], shadingNormal);

  const gDotV = Math.max(0, dot3(geometricNormal, towardViewer));
  const gDotL = Math.max(0, dot3(geometricNormal, towardLight));
  const nDotV = Math.max(0, dot3(shadingNormal, towardViewer));
  const nDotL = Math.max(0, dot3(shadingNormal, towardLight));
  const nDotH = Math.max(0, dot3(shadingNormal, halfVector));
  const lDotH = Math.max(0, dot3(towardLight, halfVector));
  const f0 = baseColor.map((channel, index) =>
    f0Dielectric[index] * specularWeight * (1 - metallic) + channel * metallic) as [number, number, number];
  const fresnel = schlickFresnel(lDotH, f0);
  const distribution = ggxNormalDistribution(nDotH, roughness);
  const visibility = smithGgxVisibility(nDotV, nDotL, roughness);
  const shadingCorrection = nDotV > PBR_EPSILON && nDotL > PBR_EPSILON && gDotV > PBR_EPSILON
    ? Math.min(1, Math.max(0, (gDotL * nDotV) / Math.max(nDotL * gDotV, PBR_EPSILON)))
    : 0;
  const directEnabled = gDotL > 0 && gDotV > 0 ? 1 : 0;
  const rimFresnel = schlickFresnel(nDotV, f0);

  return baseColor.map((channel, index) => {
    const diffuse = channel * (1 - metallic) * (1 - fresnel[index]) / PBR_PI;
    const specular = distribution * visibility * fresnel[index];
    const direct = (diffuse + specular) * lightColor[index] * gDotL * shadingCorrection * directEnabled;
    const ambient = channel * (1 - metallic) * ambientDiffuse;
    const rim = rimColor[index] * rimWeight * (1 - rimFresnel[index]) * (1 - nDotV) ** 3;
    return Math.max(0, direct + ambient + rim + emissive[index]);
  }) as [number, number, number];
}

/**
 * Canonical WGSL lighting/material library.
 *
 * Required convention:
 * - all colors are scene-linear RGB;
 * - normal, towardViewer, and towardLight point away from the surface;
 * - lightColor is un-tonemapped radiance;
 * - display transfer remains the responsibility of the final output pass.
 *
 * The library has no resource bindings and no dependency on the environment
 * shader. Consumers supply the active environment's light direction/color.
 */
export const unifiedLightingShaderLibrary = /* wgsl */ `
struct UnifiedLightingMaterial {
  baseColor: vec3f,
  roughness: f32,
  emissive: vec3f,
  ambientDiffuse: f32,
  specularF0: vec3f,
  specularWeight: f32,
  metallic: f32,
  rimColor: vec3f,
  rimWeight: f32,
}

struct UnifiedLightingInput {
  normal: vec3f,
  geometricNormal: vec3f,
  towardViewer: vec3f,
  towardLight: vec3f,
  lightColor: vec3f,
}

const UNIFIED_PI: f32 = 3.141592653589793;
const UNIFIED_EPSILON: f32 = 1e-6;

fn unifiedSafeNormal(value: vec3f, fallback: vec3f) -> vec3f {
  let magnitudeSquared = dot(value, value);
  if (magnitudeSquared > UNIFIED_EPSILON * UNIFIED_EPSILON) { return value * inverseSqrt(magnitudeSquared); }
  let fallbackSquared = dot(fallback, fallback);
  if (fallbackSquared > UNIFIED_EPSILON * UNIFIED_EPSILON) { return fallback * inverseSqrt(fallbackSquared); }
  return vec3f(0.0, 1.0, 0.0);
}

fn unifiedPbrMaterial(
  baseColor: vec3f,
  metallic: f32,
  roughness: f32,
  emissive: vec3f,
  ambientDiffuse: f32,
  specularF0: vec3f,
  specularWeight: f32,
  rimColor: vec3f,
  rimWeight: f32
) -> UnifiedLightingMaterial {
  return UnifiedLightingMaterial(
    clamp(baseColor, vec3f(0.0), vec3f(1.0)),
    clamp(roughness, 0.04, 1.0),
    max(emissive, vec3f(0.0)),
    clamp(ambientDiffuse, 0.0, 1.0),
    clamp(specularF0, vec3f(0.0), vec3f(1.0)),
    clamp(specularWeight, 0.0, 1.0),
    clamp(metallic, 0.0, 1.0),
    max(rimColor, vec3f(0.0)),
    clamp(rimWeight, 0.0, 1.0)
  );
}

fn unifiedMaterial(
  baseColor: vec3f,
  roughness: f32,
  emissive: vec3f,
  ambientDiffuse: f32,
  specularF0: vec3f,
  specularWeight: f32,
  rimColor: vec3f,
  rimWeight: f32
) -> UnifiedLightingMaterial {
  return unifiedPbrMaterial(baseColor, 0.0, roughness, emissive, ambientDiffuse, specularF0, specularWeight, rimColor, rimWeight);
}

fn unifiedLightingInput(normal: vec3f, towardViewer: vec3f, towardLight: vec3f, lightColor: vec3f) -> UnifiedLightingInput {
  return unifiedLightingInputWithGeometry(normal, normal, towardViewer, towardLight, lightColor);
}

fn unifiedLightingInputWithGeometry(shadingNormal: vec3f, geometricNormal: vec3f, towardViewer: vec3f, towardLight: vec3f, lightColor: vec3f) -> UnifiedLightingInput {
  let view = unifiedSafeNormal(towardViewer, vec3f(0.0, 0.0, 1.0));
  var geometry = unifiedSafeNormal(geometricNormal, vec3f(0.0, 1.0, 0.0));
  if (dot(geometry, view) < 0.0) { geometry = -geometry; }
  var shading = unifiedSafeNormal(shadingNormal, geometry);
  if (dot(shading, geometry) < 0.0) { shading = -shading; }
  return UnifiedLightingInput(shading, geometry, view, unifiedSafeNormal(towardLight, geometry), max(lightColor, vec3f(0.0)));
}

fn unifiedSchlick(cosine: f32, f0: vec3f) -> vec3f {
  let grazing = pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * grazing;
}

// Analytic split-sum DFG fit for prefiltered environment lighting. Raw Schlick
// alone reaches unit reflectance at every silhouette, even for rough stone.
fn unifiedEnvironmentBrdf(nDotVIn:f32,roughnessIn:f32,f0In:vec3f)->vec3f {
  let nDotV=clamp(nDotVIn,0.0,1.0);let roughness=clamp(roughnessIn,0.0,1.0);let f0=clamp(f0In,vec3f(0.0),vec3f(1.0));
  let c0=vec4f(-1.0,-.0275,-.572,.022);let c1=vec4f(1.0,.0425,1.04,-.04);let r=roughness*c0+c1;
  let a004=min(r.x*r.x,exp2(-9.28*nDotV))*r.x+r.y;let response=f0*(-1.04*a004+r.z)+(1.04*a004+r.w);
  return clamp(response,vec3f(0.0),vec3f(1.0));
}

fn unifiedDielectricFresnel(cosine: f32, f0: f32) -> f32 {
  return unifiedSchlick(cosine, vec3f(clamp(f0, 0.0, 1.0))).x;
}

fn unifiedGgxDistribution(nDotH: f32, roughness: f32) -> f32 {
  let alpha = roughness * roughness;
  let alphaSquared = alpha * alpha;
  let cosine = clamp(nDotH, 0.0, 1.0);
  let denominator = cosine * cosine * (alphaSquared - 1.0) + 1.0;
  return alphaSquared / max(UNIFIED_PI * denominator * denominator, UNIFIED_EPSILON);
}

fn unifiedSmithGgxVisibility(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
  let view = clamp(nDotV, 0.0, 1.0);
  let light = clamp(nDotL, 0.0, 1.0);
  if (view <= 0.0 || light <= 0.0) { return 0.0; }
  let alpha = roughness * roughness;
  let alphaSquared = alpha * alpha;
  let viewTerm = light * sqrt(view * view * (1.0 - alphaSquared) + alphaSquared);
  let lightTerm = view * sqrt(light * light * (1.0 - alphaSquared) + alphaSquared);
  return 0.5 / max(viewTerm + lightTerm, UNIFIED_EPSILON);
}

fn unifiedSpecularLobe(normal: vec3f, towardViewer: vec3f, towardLight: vec3f, exponent: f32) -> f32 {
  let reflected = reflect(-normalize(towardViewer), normalize(normal));
  return pow(max(dot(reflected, normalize(towardLight)), 0.0), max(exponent, 1.0));
}

fn shadeUnifiedSurface(material: UnifiedLightingMaterial, input: UnifiedLightingInput) -> vec3f {
  let geometricDotV = max(dot(input.geometricNormal, input.towardViewer), 0.0);
  let geometricDotL = max(dot(input.geometricNormal, input.towardLight), 0.0);
  let nDotV = max(dot(input.normal, input.towardViewer), 0.0);
  let nDotL = max(dot(input.normal, input.towardLight), 0.0);
  let halfVector = unifiedSafeNormal(input.towardViewer + input.towardLight, input.normal);
  let nDotH = max(dot(input.normal, halfVector), 0.0);
  let lDotH = max(dot(input.towardLight, halfVector), 0.0);
  let dielectricF0 = material.specularF0 * material.specularWeight;
  let f0 = mix(dielectricF0, material.baseColor, material.metallic);
  let fresnel = unifiedSchlick(lDotH, f0);
  let diffuseColor = material.baseColor * (1.0 - material.metallic);
  let diffuse = diffuseColor * (vec3f(1.0) - fresnel) / UNIFIED_PI;
  let distribution = unifiedGgxDistribution(nDotH, material.roughness);
  let visibility = unifiedSmithGgxVisibility(nDotV, nDotL, material.roughness);
  let specular = distribution * visibility * fresnel;
  var correction = 0.0;
  if (nDotL > UNIFIED_EPSILON && nDotV > UNIFIED_EPSILON && geometricDotV > UNIFIED_EPSILON) {
    correction = clamp((geometricDotL * nDotV) / max(nDotL * geometricDotV, UNIFIED_EPSILON), 0.0, 1.0);
  }
  let direct = (diffuse + specular) * input.lightColor * geometricDotL * correction;
  let ambient = diffuseColor * material.ambientDiffuse;
  let rimRemaining = vec3f(1.0) - unifiedSchlick(nDotV, f0);
  let rim = material.rimColor * material.rimWeight * rimRemaining * pow(1.0 - nDotV, 3.0);
  return max(direct + ambient + rim + material.emissive, vec3f(0.0));
}

fn unifiedBeerLambert(absorption: vec3f, distance: f32) -> vec3f {
  return exp(-max(absorption, vec3f(0.0)) * max(distance, 0.0));
}

fn unifiedAbsorbingTransmission(
  incident: vec3f,
  absorption: vec3f,
  scatterColor: vec3f,
  distance: f32
) -> vec3f {
  let transmission = unifiedBeerLambert(absorption, distance);
  return incident * transmission + max(scatterColor, vec3f(0.0)) * (vec3f(1.0) - transmission);
}
`;
