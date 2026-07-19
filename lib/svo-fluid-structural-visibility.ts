import {
  estimateSvoFluidGradient,
  refineSvoFluidZero,
  svoFluidSamplesCrossZero,
  svoFluidVisibilityWGSL,
  type SvoFluidFieldPair,
  type SvoFluidOwnedSample,
  type SvoFluidRaySample,
  type SvoFluidVisibilityRay,
} from "./svo-fluid-visibility";
import {
  SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS,
  lookupSvoStructuralCoarseFluidCell,
  svoStructuralCoarseFluidSamplingWGSL,
  type SvoStructuralFluidInvalidReason,
  type SvoStructuralFluidMissReason,
  type SvoStructuralFluidPackedFixture,
} from "./svo-fluid-structural-sampling";
import type { SvoVec3 } from "./webgpu-svo-traversal";
import { SPARSE_VOXEL_PUBLICATION_STATE } from "./webgpu-voxel-debug";

export const SVO_STRUCTURAL_INTERPOLATION_MAX_CORNERS = 8;
export const SVO_STRUCTURAL_VISIBILITY_DEFAULT_MAX_STEPS = 512;
export const SVO_STRUCTURAL_VISIBILITY_MAX_STEPS = 65_536;
export const SVO_STRUCTURAL_VISIBILITY_DEFAULT_NODE_BUDGET = 65_536;
export const SVO_STRUCTURAL_VISIBILITY_MAX_NODE_BUDGET = 1_048_576;

export type SvoStructuralInterpolationFallback = "none" | "domain-clamp" | "zero-weight";

export type SvoStructuralTrilinearSample =
  | {
    status: "valid";
    phi_m: number;
    nodeVisits: number;
    leafIndices: readonly number[];
    anchorNodeIndex: number;
    anchorLeafIndex: number;
    fallback: SvoStructuralInterpolationFallback;
  }
  | { status: "miss"; reason: SvoStructuralFluidMissReason; nodeVisits: number }
  | { status: "invalid"; reason: SvoStructuralFluidInvalidReason | "node-budget-exhausted"; nodeVisits: number };

export interface SvoStructuralVisibilityOptions {
  step_m?: number;
  maximumSteps?: number;
  maximumNodeVisits?: number;
  maximumRefinementIterations?: number;
  tTolerance_m?: number;
  phiTolerance_m?: number;
  gradientEpsilon?: number;
  fallbackNormal?: SvoVec3;
}

export interface SvoStructuralVisibilityDiagnostics {
  source: "structural-coarse";
  completeGeneration: number;
  coarseFluidRevision: number;
  interpolationSamples: number;
  topologyNodeVisits: number;
  crossLeafSamples: number;
  boundaryFallbackSamples: number;
  maximumSteps: number;
  maximumNodeVisits: number;
  failureReason?: SvoStructuralFluidInvalidReason | SvoStructuralFluidMissReason | "node-budget-exhausted";
}

export type SvoStructuralFluidVisibilityResult =
  | {
    status: "hit";
    t_m: number;
    position_m: SvoVec3;
    normal: SvoVec3;
    gradient: SvoVec3;
    gradientValid: boolean;
    insideFluidAtStart: boolean;
    steps: number;
    refinementIterations: number;
    refinementConverged: boolean;
    diagnostics: SvoStructuralVisibilityDiagnostics;
  }
  | { status: "miss"; steps: number; insideFluidAtStart: boolean; diagnostics: SvoStructuralVisibilityDiagnostics }
  | { status: "invalid-field"; steps: number; insideFluidAtStart: boolean; diagnostics: SvoStructuralVisibilityDiagnostics }
  | { status: "work-exhausted"; steps: number; insideFluidAtStart: boolean; diagnostics: SvoStructuralVisibilityDiagnostics };

interface SamplingBudget {
  remainingNodeVisits: number;
}

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function positive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be finite and positive`);
  return value;
}

function boundedInteger(value: number, maximum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) throw new RangeError(`${label} must be an integer from 1 to ${maximum}`);
  return value;
}

function normalized(value: SvoVec3): SvoVec3 {
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError("Structural fluid ray direction must be non-zero");
  return [value[0] / length, value[1] / length, value[2] / length];
}

function boundaryFallback(a: SvoStructuralInterpolationFallback, b: SvoStructuralInterpolationFallback): SvoStructuralInterpolationFallback {
  if (a === "domain-clamp" || b === "domain-clamp") return "domain-clamp";
  if (a === "zero-weight" || b === "zero-weight") return "zero-weight";
  return "none";
}

/**
 * Cell-centred trilinear interpolation over the structural coarse field.
 * Physical-domain clamping duplicates the nearest boundary cell and is
 * zero-set safe. Required interior corners never fall back to nearest data.
 */
export function sampleSvoStructuralCoarseFluidTrilinear(
  source: SvoStructuralFluidPackedFixture,
  position_m: SvoVec3,
  budget?: SamplingBudget,
): SvoStructuralTrilinearSample {
  finiteVec3(position_m, "Structural interpolation position");
  const domainMaximum = source.domain.worldOrigin_m.map((origin, axis) => origin
    + source.domain.dimensionsCells[axis] * source.domain.cellSize_m[axis]);
  if (position_m.some((component, axis) => component < source.domain.worldOrigin_m[axis] || component >= domainMaximum[axis])) {
    return { status: "miss", reason: "outside-domain", nodeVisits: 0 };
  }
  const grid = position_m.map((component, axis) => (
    (component - source.domain.worldOrigin_m[axis]) / source.domain.cellSize_m[axis] - 0.5
  ));
  const base = grid.map(Math.floor);
  const fraction = grid.map((component, axis) => component - base[axis]);
  const weightedCells = new Map<string, { cell: [number, number, number]; weight: number }>();
  let fallback: SvoStructuralInterpolationFallback = "none";
  for (let octant = 0; octant < SVO_STRUCTURAL_INTERPOLATION_MAX_CORNERS; octant += 1) {
    const high = [octant & 1, (octant >>> 1) & 1, (octant >>> 2) & 1];
    const weight = high.reduce((product, bit, axis) => product * (bit ? fraction[axis] : 1 - fraction[axis]), 1);
    if (weight <= 1e-15) { fallback = boundaryFallback(fallback, "zero-weight"); continue; }
    const raw = base.map((component, axis) => component + high[axis]);
    const cell = raw.map((component, axis) => Math.max(0, Math.min(source.domain.dimensionsCells[axis] - 1, component))) as [number, number, number];
    if (cell.some((component, axis) => component !== raw[axis])) fallback = "domain-clamp";
    const key = cell.join(",");
    const existing = weightedCells.get(key);
    if (existing) existing.weight += weight;
    else weightedCells.set(key, { cell, weight });
  }

  let phi_m = 0;
  let nodeVisits = 0;
  let anchorNodeIndex = -1;
  let anchorLeafIndex = -1;
  const leafIndices = new Set<number>();
  for (const corner of weightedCells.values()) {
    if (budget && budget.remainingNodeVisits < SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS) {
      return { status: "invalid", reason: "node-budget-exhausted", nodeVisits };
    }
    const sample = lookupSvoStructuralCoarseFluidCell(source, corner.cell);
    nodeVisits += sample.nodeVisits;
    if (budget) budget.remainingNodeVisits -= sample.nodeVisits;
    if (sample.status === "miss") return { status: "miss", reason: sample.reason, nodeVisits };
    if (sample.status === "invalid") return { status: "invalid", reason: sample.reason, nodeVisits };
    phi_m += corner.weight * sample.phi_m;
    leafIndices.add(sample.leafIndex);
    if (anchorNodeIndex < 0) { anchorNodeIndex = sample.nodeIndex; anchorLeafIndex = sample.leafIndex; }
  }
  if (!Number.isFinite(phi_m) || anchorNodeIndex < 0) return { status: "invalid", reason: "invalid-payload", nodeVisits };
  return { status: "valid", phi_m, nodeVisits, leafIndices: [...leafIndices].sort((a, b) => a - b), anchorNodeIndex, anchorLeafIndex, fallback };
}

function decodeMorton(low: number, high: number, level: number): readonly [number, number, number] {
  const result = [0, 0, 0];
  for (let bit = 0; bit < level; bit += 1) {
    const scale = 2 ** bit;
    for (let axis = 0; axis < 3; axis += 1) {
      const address = bit * 3 + axis;
      const word = address < 32 ? low : high;
      result[axis] += ((word >>> (address < 32 ? address : address - 32)) & 1) * scale;
    }
  }
  return result as [number, number, number];
}

function distanceToLeafExit(
  source: SvoStructuralFluidPackedFixture,
  sample: Extract<SvoStructuralTrilinearSample, { status: "valid" }>,
  position_m: SvoVec3,
  direction: SvoVec3,
): number {
  const base = sample.anchorNodeIndex * 8;
  const level = source.nodes[base + 2];
  const coordinate = decodeMorton(source.nodes[base], source.nodes[base + 1], level);
  const scale = 2 ** (source.domain.maximumDepth - level) * source.domain.brickSize;
  let exit = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(direction[axis]) <= 1e-12) continue;
    const minimum = source.domain.worldOrigin_m[axis] + coordinate[axis] * scale * source.domain.cellSize_m[axis];
    const maximum = minimum + scale * source.domain.cellSize_m[axis];
    const boundary = direction[axis] > 0 ? maximum : minimum;
    const distance = (boundary - position_m[axis]) / direction[axis];
    if (distance > 1e-9) exit = Math.min(exit, distance);
  }
  return exit;
}

/**
 * Trace the continuous structural coarse field. Sequential leaf/cell-bounded
 * samples guarantee the first observed sign-changing interval is refined.
 */
export function traceSvoStructuralCoarseFluid(
  source: SvoStructuralFluidPackedFixture,
  ray: SvoFluidVisibilityRay,
  options: SvoStructuralVisibilityOptions = {},
): SvoStructuralFluidVisibilityResult {
  finiteVec3(ray.origin_m, "Structural fluid ray origin");
  finiteVec3(ray.direction, "Structural fluid ray direction");
  const direction = normalized(ray.direction);
  const tMinimum = ray.tMin_m ?? 0;
  if (!Number.isFinite(tMinimum) || !Number.isFinite(ray.tMax_m) || ray.tMax_m < tMinimum) {
    throw new RangeError("Structural fluid ray interval must be finite and ordered");
  }
  const maximumSteps = boundedInteger(
    options.maximumSteps ?? SVO_STRUCTURAL_VISIBILITY_DEFAULT_MAX_STEPS,
    SVO_STRUCTURAL_VISIBILITY_MAX_STEPS,
    "Structural fluid trace steps",
  );
  const maximumNodeVisits = boundedInteger(
    options.maximumNodeVisits ?? SVO_STRUCTURAL_VISIBILITY_DEFAULT_NODE_BUDGET,
    SVO_STRUCTURAL_VISIBILITY_MAX_NODE_BUDGET,
    "Structural fluid topology-node budget",
  );
  const halfCellStep = 0.5 * Math.min(...source.domain.cellSize_m);
  // `step_m` is an upper bound only: never jump more than half of the finest
  // anisotropic cell even when a caller requests a larger march step.
  const baseStep = Math.min(positive(options.step_m ?? halfCellStep, "Structural fluid trace step"), halfCellStep);
  const budget: SamplingBudget = { remainingNodeVisits: maximumNodeVisits };
  const diagnostics: SvoStructuralVisibilityDiagnostics = {
    source: "structural-coarse",
    completeGeneration: source.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] ?? 0,
    coarseFluidRevision: source.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision] ?? 0,
    interpolationSamples: 0,
    topologyNodeVisits: 0,
    crossLeafSamples: 0,
    boundaryFallbackSamples: 0,
    maximumSteps,
    maximumNodeVisits,
  };
  const pointAt = (t_m: number): SvoVec3 => [
    ray.origin_m[0] + direction[0] * t_m,
    ray.origin_m[1] + direction[1] * t_m,
    ray.origin_m[2] + direction[2] * t_m,
  ];
  let lastStructural: SvoStructuralTrilinearSample | undefined;
  const sampleAtPosition = (position_m: SvoVec3): SvoFluidFieldPair => {
    const sample = sampleSvoStructuralCoarseFluidTrilinear(source, position_m, budget);
    lastStructural = sample;
    diagnostics.interpolationSamples += 1;
    diagnostics.topologyNodeVisits += sample.nodeVisits;
    if (sample.status !== "valid") {
      diagnostics.failureReason = sample.reason;
      return { coarse: { phi_m: Number.POSITIVE_INFINITY, valid: false } };
    }
    if (sample.leafIndices.length > 1) diagnostics.crossLeafSamples += 1;
    if (sample.fallback !== "none") diagnostics.boundaryFallbackSamples += 1;
    return { coarse: { phi_m: sample.phi_m, valid: true } };
  };
  const ownedAt = (t_m: number): SvoFluidOwnedSample => {
    const coarse = sampleAtPosition(pointAt(t_m)).coarse;
    return coarse.valid && Number.isFinite(coarse.phi_m)
      ? { phi_m: coarse.phi_m, valid: true, owner: "coarse" }
      : { phi_m: Number.POSITIVE_INFINITY, valid: false, owner: "none" };
  };

  let previous: SvoFluidRaySample = { ...ownedAt(tMinimum), t_m: tMinimum };
  const insideFluidAtStart = previous.valid && previous.phi_m < 0;
  if (!previous.valid) {
    return diagnostics.failureReason === "node-budget-exhausted"
      ? { status: "work-exhausted", steps: 0, insideFluidAtStart, diagnostics }
      : { status: "invalid-field", steps: 0, insideFluidAtStart, diagnostics };
  }

  for (let steps = 0; steps < maximumSteps; steps += 1) {
    if (previous.phi_m === 0) {
      const position_m = pointAt(previous.t_m);
      const gradient = estimateSvoFluidGradient(position_m, source.domain.cellSize_m, sampleAtPosition, {
        fallbackNormal: options.fallbackNormal ?? [-direction[0], -direction[1], -direction[2]],
        gradientEpsilon: options.gradientEpsilon,
      });
      return { status: "hit", t_m: previous.t_m, position_m, normal: gradient.normal, gradient: gradient.gradient,
        gradientValid: gradient.valid, insideFluidAtStart, steps, refinementIterations: 0, refinementConverged: true, diagnostics };
    }
    if (previous.t_m >= ray.tMax_m) return { status: "miss", steps, insideFluidAtStart, diagnostics };
    const previousPosition = pointAt(previous.t_m);
    const leafExit = lastStructural?.status === "valid"
      ? distanceToLeafExit(source, lastStructural, previousPosition, direction) : Number.POSITIVE_INFINITY;
    const minimumProgress = 1e-6 * Math.min(...source.domain.cellSize_m);
    const step = Math.max(minimumProgress, Math.min(baseStep, leafExit));
    const nextT = Math.min(ray.tMax_m, previous.t_m + step);
    const next: SvoFluidRaySample = { ...ownedAt(nextT), t_m: nextT };
    if (!next.valid) {
      return diagnostics.failureReason === "node-budget-exhausted"
        ? { status: "work-exhausted", steps: steps + 1, insideFluidAtStart, diagnostics }
        : { status: "invalid-field", steps: steps + 1, insideFluidAtStart, diagnostics };
    }
    if (svoFluidSamplesCrossZero(previous, next)) {
      const root = refineSvoFluidZero({ lower: previous, upper: next }, ownedAt, {
        maximumIterations: options.maximumRefinementIterations,
        tTolerance_m: options.tTolerance_m,
        phiTolerance_m: options.phiTolerance_m,
      });
      if (root.status !== "hit") {
        return diagnostics.failureReason === "node-budget-exhausted"
          ? { status: "work-exhausted", steps: steps + 1, insideFluidAtStart, diagnostics }
          : { status: "invalid-field", steps: steps + 1, insideFluidAtStart, diagnostics };
      }
      const position_m = pointAt(root.sample.t_m);
      const gradient = estimateSvoFluidGradient(position_m, source.domain.cellSize_m, sampleAtPosition, {
        fallbackNormal: options.fallbackNormal ?? [-direction[0], -direction[1], -direction[2]],
        gradientEpsilon: options.gradientEpsilon,
      });
      return { status: "hit", t_m: root.sample.t_m, position_m, normal: gradient.normal, gradient: gradient.gradient,
        gradientValid: gradient.valid, insideFluidAtStart, steps: steps + 1, refinementIterations: root.iterations,
        refinementConverged: root.converged, diagnostics };
    }
    previous = next;
  }
  return previous.t_m >= ray.tMax_m
    ? { status: "miss", steps: maximumSteps, insideFluidAtStart, diagnostics }
    : { status: "work-exhausted", steps: maximumSteps, insideFluidAtStart, diagnostics };
}

const structuralTrilinearWGSL = /* wgsl */ `
struct SvoStructuralTrilinearSample { phi_m:f32, status:u32, nodeVisits:u32, fallback:u32 }
fn svoStructuralCoarseFluidTrilinear(domain:SvoStructuralSamplingDomain,position_m:vec3f)->SvoStructuralTrilinearSample{
  let maximum=domain.worldOrigin_m.xyz+vec3f(domain.dimensionsBrick.xyz)*domain.cellSize_m.xyz;
  if(any(position_m<domain.worldOrigin_m.xyz)||any(position_m>=maximum)){return SvoStructuralTrilinearSample(0.0,SVO_STRUCTURAL_SAMPLE_MISS,0u,0u);}
  let grid=(position_m-domain.worldOrigin_m.xyz)/domain.cellSize_m.xyz-vec3f(0.5);let base=vec3i(floor(grid));let fraction=fract(grid);
  var phi=0.0;var visits=0u;var fallback=0u;
  for(var octant=0u;octant<8u;octant+=1u){
    let high=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);let highf=vec3f(high);
    let weights=mix(vec3f(1.0)-fraction,fraction,highf);let weight=weights.x*weights.y*weights.z;if(weight<=1e-15){fallback=max(fallback,1u);continue;}
    let raw=base+vec3i(high);let cell=clamp(raw,vec3i(0),vec3i(domain.dimensionsBrick.xyz)-vec3i(1));if(any(cell!=raw)){fallback=2u;}
    let sample=svoStructuralCoarseFluidCell(domain,cell);visits+=u32(sample.value.y);if(sample.identity.x!=SVO_STRUCTURAL_SAMPLE_VALID){return SvoStructuralTrilinearSample(0.0,sample.identity.x,visits,fallback);}
    phi+=weight*sample.value.x;
  }
  return SvoStructuralTrilinearSample(phi,SVO_STRUCTURAL_SAMPLE_VALID,visits,fallback);
}
fn svoFluidSampleAt(position_m:vec3f)->SvoFluidOwnedSample{
  let coarse=svoStructuralCoarseFluidTrilinear(svoStructuralFluidDomain,position_m);
  if(coarse.status==SVO_STRUCTURAL_SAMPLE_VALID){return SvoFluidOwnedSample(coarse.phi_m,SVO_FLUID_OWNER_COARSE,1u);}
  return SvoFluidOwnedSample(3.402823e38,SVO_FLUID_OWNER_NONE,0u);
}
`;

/** Binding-free composition; the consumer declares structural arrays and `svoStructuralFluidDomain`. */
export const svoStructuralFluidVisibilityWGSL = `${svoStructuralCoarseFluidSamplingWGSL}\n${svoFluidVisibilityWGSL}\n${structuralTrilinearWGSL}`;
