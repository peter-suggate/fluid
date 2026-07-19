import type { SvoVec3 } from "./webgpu-svo-traversal";

/** The field which exclusively owns one world-space fluid sample. */
export type SvoFluidFieldOwner = "none" | "coarse" | "fine";

export interface SvoFluidFieldValue {
  /** Negative in liquid, zero on the boundary, positive in air; world metres. */
  phi_m: number;
  /** Validity is explicit: a finite value, including zero, is not sufficient. */
  valid: boolean;
}

export interface SvoFluidFieldPair {
  coarse: SvoFluidFieldValue;
  fine?: SvoFluidFieldValue;
}

export interface SvoFluidOwnedSample {
  phi_m: number;
  valid: boolean;
  owner: SvoFluidFieldOwner;
}

export interface SvoFluidRaySample extends SvoFluidOwnedSample {
  t_m: number;
}

export interface SvoFluidSignChangeInterval {
  lower: SvoFluidRaySample;
  upper: SvoFluidRaySample;
}

export interface SvoFluidRefinementOptions {
  maximumIterations?: number;
  tTolerance_m?: number;
  phiTolerance_m?: number;
}

export type SvoFluidRootResult =
  | {
    status: "hit";
    sample: SvoFluidRaySample;
    iterations: number;
    converged: boolean;
  }
  | { status: "invalid-interval"; iterations: 0 }
  | { status: "invalid-field"; iterations: number };

export interface SvoFluidGradientOptions {
  /** Used when the finite-difference gradient is invalid or too small. */
  fallbackNormal?: SvoVec3;
  gradientEpsilon?: number;
}

export interface SvoFluidGradient {
  normal: SvoVec3;
  gradient: SvoVec3;
  valid: boolean;
  scheme: "central" | "mixed" | "fallback";
}

export interface SvoFluidVisibilityRay {
  origin_m: SvoVec3;
  /** Need not be normalized. Trace distances are always returned in metres. */
  direction: SvoVec3;
  tMin_m?: number;
  tMax_m: number;
}

export interface SvoFluidTraceOptions extends SvoFluidRefinementOptions, SvoFluidGradientOptions {
  cellSize_m: SvoVec3;
  step_m?: number;
  maximumSteps?: number;
}

export type SvoFluidTraceResult =
  | {
    status: "hit";
    t_m: number;
    position_m: SvoVec3;
    normal: SvoVec3;
    gradient: SvoFluidGradient;
    fieldOwner: Exclude<SvoFluidFieldOwner, "none">;
    insideFluidAtStart: boolean;
    steps: number;
    refinementIterations: number;
    refinementConverged: boolean;
  }
  | { status: "miss"; steps: number; insideFluidAtStart: boolean }
  | { status: "work-exhausted"; steps: number; insideFluidAtStart: boolean }
  | { status: "invalid-field"; steps: number; insideFluidAtStart: boolean };

export type SvoFluidFieldSampler = (position_m: SvoVec3) => SvoFluidFieldPair;

const DEFAULT_MAXIMUM_STEPS = 256;
const DEFAULT_REFINEMENT_ITERATIONS = 8;

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function positiveFinite(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be finite and positive`);
  return value;
}

function boundedPositiveInteger(value: number, maximum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new RangeError(`${label} must be an integer from 1 to ${maximum}`);
  }
  return value;
}

function isUsable(value: SvoFluidFieldValue | undefined): value is SvoFluidFieldValue {
  return value?.valid === true && Number.isFinite(value.phi_m);
}

/**
 * Choose exactly one fluid field at a point. A valid fine sample owns the
 * point; otherwise the valid coarse sample owns it. No solid value participates
 * in this decision, and an invalid fine value can never mask valid coarse phi.
 */
export function resolveSvoFluidPhi(fields: SvoFluidFieldPair): SvoFluidOwnedSample {
  if (isUsable(fields.fine)) return { phi_m: fields.fine.phi_m, valid: true, owner: "fine" };
  if (isUsable(fields.coarse)) return { phi_m: fields.coarse.phi_m, valid: true, owner: "coarse" };
  return { phi_m: Number.POSITIVE_INFINITY, valid: false, owner: "none" };
}

/** Boundary-inclusive sign test. Invalid samples never form an interval. */
export function svoFluidSamplesCrossZero(a: SvoFluidOwnedSample, b: SvoFluidOwnedSample): boolean {
  if (!a.valid || !b.valid || !Number.isFinite(a.phi_m) || !Number.isFinite(b.phi_m)) return false;
  return a.phi_m === 0 || b.phi_m === 0 || (a.phi_m < 0) !== (b.phi_m < 0);
}

/** Find the nearest adjacent valid pair which brackets phi=0. */
export function findNearestSvoFluidSignChange(
  samples: readonly SvoFluidRaySample[],
): SvoFluidSignChangeInterval | undefined {
  let previous: SvoFluidRaySample | undefined;
  let previousT = Number.NEGATIVE_INFINITY;
  for (const sample of samples) {
    if (!Number.isFinite(sample.t_m) || sample.t_m < previousT) {
      throw new RangeError("Fluid ray samples must have finite, non-decreasing distances");
    }
    previousT = sample.t_m;
    if (!sample.valid || !Number.isFinite(sample.phi_m)) {
      previous = undefined;
      continue;
    }
    if (sample.phi_m === 0) return { lower: sample, upper: sample };
    if (previous && svoFluidSamplesCrossZero(previous, sample)) return { lower: previous, upper: sample };
    previous = sample;
  }
  return undefined;
}

function refinementSettings(options: SvoFluidRefinementOptions) {
  return {
    maximumIterations: boundedPositiveInteger(
      options.maximumIterations ?? DEFAULT_REFINEMENT_ITERATIONS,
      64,
      "Fluid root-refinement iteration count",
    ),
    tTolerance_m: positiveFinite(options.tTolerance_m ?? 1e-5, "Fluid root distance tolerance"),
    phiTolerance_m: positiveFinite(options.phiTolerance_m ?? 1e-5, "Fluid root phi tolerance"),
  };
}

function closerEndpoint(a: SvoFluidRaySample, b: SvoFluidRaySample): SvoFluidRaySample {
  return Math.abs(a.phi_m) <= Math.abs(b.phi_m) ? a : b;
}

/**
 * Refine a bracket with safeguarded secant updates. Secant candidates outside
 * the central 80% of the current bracket fall back to bisection, preventing
 * false-position stagnation while retaining fast convergence on smooth phi.
 */
export function refineSvoFluidZero(
  interval: SvoFluidSignChangeInterval,
  sampleAtDistance: (t_m: number) => SvoFluidOwnedSample,
  options: SvoFluidRefinementOptions = {},
): SvoFluidRootResult {
  const settings = refinementSettings(options);
  let a = interval.lower;
  let b = interval.upper;
  if (!Number.isFinite(a.t_m) || !Number.isFinite(b.t_m) || b.t_m < a.t_m
      || !svoFluidSamplesCrossZero(a, b)) {
    return { status: "invalid-interval", iterations: 0 };
  }
  if (a.phi_m === 0 || a.t_m === b.t_m) return { status: "hit", sample: a, iterations: 0, converged: true };
  if (b.phi_m === 0) return { status: "hit", sample: b, iterations: 0, converged: true };

  for (let iteration = 1; iteration <= settings.maximumIterations; iteration += 1) {
    const width = b.t_m - a.t_m;
    if (width <= settings.tTolerance_m) {
      return { status: "hit", sample: closerEndpoint(a, b), iterations: iteration - 1, converged: true };
    }
    const denominator = b.phi_m - a.phi_m;
    let candidateT = a.t_m - a.phi_m * width / denominator;
    const guard = 0.1 * width;
    if (!Number.isFinite(candidateT) || candidateT <= a.t_m + guard || candidateT >= b.t_m - guard) {
      candidateT = 0.5 * (a.t_m + b.t_m);
    }
    const owned = sampleAtDistance(candidateT);
    if (!owned.valid || owned.owner === "none" || !Number.isFinite(owned.phi_m)) {
      return { status: "invalid-field", iterations: iteration };
    }
    const candidate: SvoFluidRaySample = { ...owned, t_m: candidateT };
    if (Math.abs(candidate.phi_m) <= settings.phiTolerance_m) {
      return { status: "hit", sample: candidate, iterations: iteration, converged: true };
    }
    if ((a.phi_m < 0) === (candidate.phi_m < 0)) a = candidate;
    else b = candidate;
  }
  return {
    status: "hit",
    sample: closerEndpoint(a, b),
    iterations: settings.maximumIterations,
    converged: b.t_m - a.t_m <= settings.tTolerance_m,
  };
}

function length3(value: SvoVec3): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalizeOrUp(value: SvoVec3): SvoVec3 {
  const magnitude = length3(value);
  if (!Number.isFinite(magnitude) || magnitude <= 1e-12) return [0, 1, 0];
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

/** World-space finite differences; per-axis cell sizes are never collapsed. */
export function estimateSvoFluidGradient(
  position_m: SvoVec3,
  cellSize_m: SvoVec3,
  sampleFields: SvoFluidFieldSampler,
  options: SvoFluidGradientOptions = {},
): SvoFluidGradient {
  finiteVec3(position_m, "Fluid gradient position");
  finiteVec3(cellSize_m, "Fluid gradient cell size");
  cellSize_m.forEach((size, axis) => positiveFinite(size, `Fluid gradient cell size axis ${axis}`));
  const epsilon = positiveFinite(options.gradientEpsilon ?? 1e-8, "Fluid gradient epsilon");
  const center = resolveSvoFluidPhi(sampleFields(position_m));
  const gradient = [0, 0, 0];
  let centralAxes = 0;
  let usableAxes = 0;

  for (let axis = 0; axis < 3; axis += 1) {
    const minusPosition = [...position_m] as [number, number, number];
    const plusPosition = [...position_m] as [number, number, number];
    minusPosition[axis] -= cellSize_m[axis];
    plusPosition[axis] += cellSize_m[axis];
    const minus = resolveSvoFluidPhi(sampleFields(minusPosition));
    const plus = resolveSvoFluidPhi(sampleFields(plusPosition));
    if (minus.valid && plus.valid) {
      gradient[axis] = (plus.phi_m - minus.phi_m) / (2 * cellSize_m[axis]);
      centralAxes += 1;
      usableAxes += 1;
    } else if (center.valid && plus.valid) {
      gradient[axis] = (plus.phi_m - center.phi_m) / cellSize_m[axis];
      usableAxes += 1;
    } else if (center.valid && minus.valid) {
      gradient[axis] = (center.phi_m - minus.phi_m) / cellSize_m[axis];
      usableAxes += 1;
    }
  }

  const gradientTuple = gradient as [number, number, number];
  const magnitude = length3(gradientTuple);
  if (usableAxes > 0 && gradient.every(Number.isFinite) && Number.isFinite(magnitude) && magnitude > epsilon) {
    return {
      normal: [gradient[0] / magnitude, gradient[1] / magnitude, gradient[2] / magnitude],
      gradient: gradientTuple,
      valid: true,
      scheme: centralAxes === 3 ? "central" : "mixed",
    };
  }

  return {
    normal: normalizeOrUp(options.fallbackNormal ?? [0, 1, 0]),
    gradient: gradientTuple,
    valid: false,
    scheme: "fallback",
  };
}

/** CPU visibility oracle for a bounded, fixed-step leaf/voxel interval walk. */
export function traceSvoFluidLevelSet(
  ray: SvoFluidVisibilityRay,
  sampleFields: SvoFluidFieldSampler,
  options: SvoFluidTraceOptions,
): SvoFluidTraceResult {
  finiteVec3(ray.origin_m, "Fluid ray origin");
  finiteVec3(ray.direction, "Fluid ray direction");
  finiteVec3(options.cellSize_m, "Fluid trace cell size");
  options.cellSize_m.forEach((size, axis) => positiveFinite(size, `Fluid trace cell size axis ${axis}`));
  const directionLength = length3(ray.direction);
  if (directionLength <= 1e-12) throw new RangeError("Fluid ray direction must be non-zero");
  const direction: SvoVec3 = ray.direction.map((value) => value / directionLength) as [number, number, number];
  const tMinimum = ray.tMin_m ?? 0;
  if (!Number.isFinite(tMinimum) || !Number.isFinite(ray.tMax_m) || ray.tMax_m < tMinimum) {
    throw new RangeError("Fluid ray distances must define a finite non-decreasing interval");
  }
  const step = positiveFinite(options.step_m ?? 0.5 * Math.min(...options.cellSize_m), "Fluid trace step");
  const maximumSteps = boundedPositiveInteger(options.maximumSteps ?? DEFAULT_MAXIMUM_STEPS, 65_536, "Fluid trace step count");
  const pointAt = (t_m: number): SvoVec3 => [
    ray.origin_m[0] + direction[0] * t_m,
    ray.origin_m[1] + direction[1] * t_m,
    ray.origin_m[2] + direction[2] * t_m,
  ];
  const ownedAt = (t_m: number) => resolveSvoFluidPhi(sampleFields(pointAt(t_m)));

  let previous: SvoFluidRaySample = { ...ownedAt(tMinimum), t_m: tMinimum };
  const insideFluidAtStart = previous.valid && previous.phi_m < 0;
  if (!previous.valid) return { status: "invalid-field", steps: 0, insideFluidAtStart };

  for (let steps = 0; steps < maximumSteps; steps += 1) {
    if (previous.phi_m === 0) {
      const gradient = estimateSvoFluidGradient(pointAt(previous.t_m), options.cellSize_m, sampleFields, {
        fallbackNormal: options.fallbackNormal ?? [-direction[0], -direction[1], -direction[2]],
        gradientEpsilon: options.gradientEpsilon,
      });
      return {
        status: "hit", t_m: previous.t_m, position_m: pointAt(previous.t_m), normal: gradient.normal,
        gradient, fieldOwner: previous.owner as Exclude<SvoFluidFieldOwner, "none">, insideFluidAtStart,
        steps, refinementIterations: 0, refinementConverged: true,
      };
    }
    if (previous.t_m >= ray.tMax_m) return { status: "miss", steps, insideFluidAtStart };
    const nextT = Math.min(ray.tMax_m, previous.t_m + step);
    const next: SvoFluidRaySample = { ...ownedAt(nextT), t_m: nextT };
    if (!next.valid) return { status: "invalid-field", steps: steps + 1, insideFluidAtStart };
    if (svoFluidSamplesCrossZero(previous, next)) {
      const root = refineSvoFluidZero({ lower: previous, upper: next }, ownedAt, options);
      if (root.status !== "hit") return { status: "invalid-field", steps: steps + 1, insideFluidAtStart };
      const position_m = pointAt(root.sample.t_m);
      const gradient = estimateSvoFluidGradient(position_m, options.cellSize_m, sampleFields, {
        fallbackNormal: options.fallbackNormal ?? [-direction[0], -direction[1], -direction[2]],
        gradientEpsilon: options.gradientEpsilon,
      });
      return {
        status: "hit",
        t_m: root.sample.t_m,
        position_m,
        normal: gradient.normal,
        gradient,
        fieldOwner: root.sample.owner as Exclude<SvoFluidFieldOwner, "none">,
        insideFluidAtStart,
        steps: steps + 1,
        refinementIterations: root.iterations,
        refinementConverged: root.converged,
      };
    }
    previous = next;
  }
  return previous.t_m >= ray.tMax_m
    ? { status: "miss", steps: maximumSteps, insideFluidAtStart }
    : { status: "work-exhausted", steps: maximumSteps, insideFluidAtStart };
}

/**
 * Binding-free WGSL helpers. The including shader supplies
 * `fn svoFluidSampleAt(worldPosition_m: vec3f) -> SvoFluidOwnedSample` from its
 * coarse payload and optional fine page table. These helpers never read or
 * combine the separate solid field.
 */
export const svoFluidVisibilityWGSL = /* wgsl */ `
struct SvoFluidFieldValue { phi_m:f32, valid:u32 }
struct SvoFluidOwnedSample { phi_m:f32, owner:u32, valid:u32 }
struct SvoFluidRoot { t_m:f32, phi_m:f32, owner:u32, valid:u32, iterations:u32, converged:u32 }
struct SvoFluidNormal { normal:vec3f, gradient:vec3f, valid:u32 }

const SVO_FLUID_OWNER_NONE:u32 = 0u;
const SVO_FLUID_OWNER_COARSE:u32 = 1u;
const SVO_FLUID_OWNER_FINE:u32 = 2u;
const SVO_FLUID_REFINE_ITERATIONS:u32 = 8u;

fn svoFluidFinite(value:f32)->bool { return value==value && abs(value)<=3.402823e38; }

fn svoResolveFluidPhi(coarse:SvoFluidFieldValue,fine:SvoFluidFieldValue)->SvoFluidOwnedSample {
  if(fine.valid!=0u && svoFluidFinite(fine.phi_m)){return SvoFluidOwnedSample(fine.phi_m,SVO_FLUID_OWNER_FINE,1u);}
  if(coarse.valid!=0u && svoFluidFinite(coarse.phi_m)){return SvoFluidOwnedSample(coarse.phi_m,SVO_FLUID_OWNER_COARSE,1u);}
  return SvoFluidOwnedSample(3.402823e38,SVO_FLUID_OWNER_NONE,0u);
}

fn svoFluidCrossesZero(a:SvoFluidOwnedSample,b:SvoFluidOwnedSample)->bool {
  return a.valid!=0u && b.valid!=0u && svoFluidFinite(a.phi_m) && svoFluidFinite(b.phi_m)
    && (a.phi_m==0.0 || b.phi_m==0.0 || (a.phi_m<0.0)!=(b.phi_m<0.0));
}

fn svoFluidSecantOrBisect(aT:f32,aPhi:f32,bT:f32,bPhi:f32)->f32 {
  let width=bT-aT;let denominator=bPhi-aPhi;var candidate=aT-aPhi*width/denominator;let guard=.1*width;
  if(!svoFluidFinite(candidate)||candidate<=aT+guard||candidate>=bT-guard){candidate=.5*(aT+bT);}
  return candidate;
}

// Requires the including shader's svoFluidSampleAt adapter described above.
fn svoFluidRefineZero(ro:vec3f,rd:vec3f,aTIn:f32,bTIn:f32,aIn:SvoFluidOwnedSample,bIn:SvoFluidOwnedSample,tTolerance_m:f32,phiTolerance_m:f32)->SvoFluidRoot {
  if(!svoFluidCrossesZero(aIn,bIn)){return SvoFluidRoot(0.0,3.402823e38,SVO_FLUID_OWNER_NONE,0u,0u,0u);}
  var aT=aTIn;var bT=bTIn;var a=aIn;var b=bIn;
  if(a.phi_m==0.0){return SvoFluidRoot(aT,a.phi_m,a.owner,1u,0u,1u);}
  if(b.phi_m==0.0){return SvoFluidRoot(bT,b.phi_m,b.owner,1u,0u,1u);}
  for(var iteration=0u;iteration<SVO_FLUID_REFINE_ITERATIONS;iteration+=1u){
    if(bT-aT<=tTolerance_m){let useA=abs(a.phi_m)<=abs(b.phi_m);return SvoFluidRoot(select(bT,aT,useA),select(b.phi_m,a.phi_m,useA),select(b.owner,a.owner,useA),1u,iteration,1u);}
    let candidateT=svoFluidSecantOrBisect(aT,a.phi_m,bT,b.phi_m);let candidate=svoFluidSampleAt(ro+rd*candidateT);
    if(candidate.valid==0u||!svoFluidFinite(candidate.phi_m)){return SvoFluidRoot(candidateT,3.402823e38,SVO_FLUID_OWNER_NONE,0u,iteration+1u,0u);}
    if(abs(candidate.phi_m)<=phiTolerance_m){return SvoFluidRoot(candidateT,candidate.phi_m,candidate.owner,1u,iteration+1u,1u);}
    if((a.phi_m<0.0)==(candidate.phi_m<0.0)){aT=candidateT;a=candidate;}else{bT=candidateT;b=candidate;}
  }
  let useA=abs(a.phi_m)<=abs(b.phi_m);return SvoFluidRoot(select(bT,aT,useA),select(b.phi_m,a.phi_m,useA),select(b.owner,a.owner,useA),1u,SVO_FLUID_REFINE_ITERATIONS,select(0u,1u,bT-aT<=tTolerance_m));
}

fn svoFluidGradientNormal(position_m:vec3f,cellSize_m:vec3f,rayDirection:vec3f)->SvoFluidNormal {
  let xm=svoFluidSampleAt(position_m-vec3f(cellSize_m.x,0.0,0.0));let xp=svoFluidSampleAt(position_m+vec3f(cellSize_m.x,0.0,0.0));
  let ym=svoFluidSampleAt(position_m-vec3f(0.0,cellSize_m.y,0.0));let yp=svoFluidSampleAt(position_m+vec3f(0.0,cellSize_m.y,0.0));
  let zm=svoFluidSampleAt(position_m-vec3f(0.0,0.0,cellSize_m.z));let zp=svoFluidSampleAt(position_m+vec3f(0.0,0.0,cellSize_m.z));
  let valid=xm.valid!=0u&&xp.valid!=0u&&ym.valid!=0u&&yp.valid!=0u&&zm.valid!=0u&&zp.valid!=0u;
  if(valid){let gradient=vec3f((xp.phi_m-xm.phi_m)/(2.0*cellSize_m.x),(yp.phi_m-ym.phi_m)/(2.0*cellSize_m.y),(zp.phi_m-zm.phi_m)/(2.0*cellSize_m.z));let magnitude2=dot(gradient,gradient);if(all(gradient==gradient)&&magnitude2>1e-16&&svoFluidFinite(magnitude2)){return SvoFluidNormal(gradient*inverseSqrt(magnitude2),gradient,1u);}}
  var fallback=-rayDirection;let fallback2=dot(fallback,fallback);if(!svoFluidFinite(fallback2)||fallback2<=1e-16){fallback=vec3f(0.0,1.0,0.0);}else{fallback*=inverseSqrt(fallback2);}
  return SvoFluidNormal(fallback,vec3f(0.0),0u);
}
`;
