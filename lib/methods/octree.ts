import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type MethodParamValues, type SimulationMethod } from "./types";
import type { GPUQuality } from "../tall-cell-grid";
import type { SceneDescription } from "../model";

const params: MethodParamSpec[] = [
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "4", tier: "coarse", options: [{ value: "4", label: "4× fine band · paper default" }, { value: "8", label: "8× fine band · experimental" }], hint: "The paper tracks the interface on a separate 4× or 8× sparse narrow band." },
  { kind: "select", key: "maximumLeafSize", label: "Largest pressure cell", default: "16", tier: "fine", options: [{ value: "2", label: "2³ finest cells" }, { value: "4", label: "4³ finest cells" }, { value: "8", label: "8³ finest cells" }, { value: "16", label: "16³ finest cells" }, { value: "32", label: "32³ finest cells" }], hint: "Largest dyadic octree cell away from interfaces. The topology remains strictly 2:1 graded for valid power-diagram stencils." },
  { kind: "number", key: "interfaceRefinementBandCells", label: "Pressure refinement band", unit: "finest cells", min: 0, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Keeps the octree pressure grid fine around liquid and solid interfaces. This is distinct from the separate high-resolution surface-tracking band." },
  // The floor is relational, not absolute: the fine band has to reach at least
  // as far as the pressure cell centres that read it, or the fine-to-coarse
  // restriction finds no valid phi and the publication is rejected
  // (`fineRestrictionValid: 0`). Measured on the Dawn mini dam break at factor
  // 4, band 1 fails against a pressure band of 3 while band 2 runs clean --
  // that is a mismatch with the pressure band, not a property of the value 1,
  // so the control opens at 1 and an incompatible pair fails closed and
  // visibly rather than being clamped away.
  { kind: "number", key: "fineLevelSetBandCells", label: "Surface tracking band", unit: "finest cells", min: 1, max: 32, step: 1, digits: 0, default: 4, tier: "fine", hint: "Half-width of the Section 5 high-resolution narrow band, independent of the pressure refinement band. The paper only requires it to stay wide enough that the surface remains inside after advection; extra width widens coarse-phi correction coverage rather than surface accuracy, and costs band volume plus redistance passes. The usable maximum is domain-dependent: on a small tank a wide band's dilation exceeds the fine brick capacity and the publication is rejected rather than degraded." }
];

const maximumLeafSize = (value: unknown): 2 | 4 | 8 | 16 | 32 => {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 4;
  return numeric >= 32 ? 32 : numeric >= 16 ? 16 : numeric >= 8 ? 8 : numeric <= 2 ? 2 : 4;
};

const globalFineLevelSetFactor = (value: unknown): 4 | 8 => Number(value) >= 8 ? 8 : 4;

/**
 * The surface band as authored, or the resolved pressure band when it is not.
 *
 * Returning `undefined` for an unauthored band is deliberate: the projection
 * then couples it to `interfaceRefinementBandCells` itself, which is the width
 * every lane was measured at before the two bands separated. Substituting this
 * spec's default instead diverges from any scene profile that pins the
 * pressure band alone.
 */
const authoredFineBand = (values: MethodParamValues): number | undefined =>
  typeof values.fineLevelSetBandCells === "number"
    && Number.isFinite(values.fineLevelSetBandCells)
    ? numberValue(values, params, "fineLevelSetBandCells") : undefined;

/** Canonical browser/native construction options for the power-octree method. */
export const octreeSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) => ({
  densitySharpening: false,
  velocityTransport: "maccormack" as const,
  octree: {
    maximumLeafSize: maximumLeafSize(values.maximumLeafSize ?? 16),
    // Pressure topology is a method setting, not a solver setting. Keep the
    // same adaptive octree when comparing the two pressure implementations.
    adaptivity: 1,
    interfaceRefinementBandCells: numberValue(values, params, "interfaceRefinementBandCells"),
    // Section 5's surface band is independent of the pressure band, so a
    // thickness sweep moves only the fine level set's cost -- but only once it
    // is authored. Unauthored it must FOLLOW the resolved pressure band rather
    // than fall to this spec's own default: a scene profile that pins the
    // pressure band (POWER_VALIDATION_METHOD_PROFILE pins 3) overrides only
    // that key, so reading a bare default here silently widened the band to 4
    // and overflowed the fine brick capacity a few generations in.
    fineLevelSetBandCells: authoredFineBand(values),
    globalFineLevelSetFactor: globalFineLevelSetFactor(values.globalFineLevelSetFactor),
    // Diagnostic-only capacity override. Not an authored UI parameter: the
    // production capacity comes from `planOctreePressureCapacity`, and this
    // exists so a harness can bisect the arena demand of a large domain
    // against the on-GPU overflow report.
    pressureRowCapacity: typeof values.pressureRowCapacity === "number" && values.pressureRowCapacity > 0
      ? values.pressureRowCapacity : undefined,
  }
});

export const octreeMethod: SimulationMethod = {
  id: "octree",
  label: "Power-diagram octree",
  shortLabel: "Power octree",
  badge: "POWER OCTREE",
  description: "GPU-resident adaptive power cells with the sparse-pyramid hybrid MGPCG pressure solve and a high-resolution interface band.",
  detail: "2:1-graded adaptive pressure cells and six structured velocity families, one production Section 4.3 hybrid MGPCG authority, factor-4 signed-distance narrow band, frame-lagged variational rigid-body coupling, and no topology readbacks or alternate solver path",
  backend: "webgpu",
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  params,
  pressureMapping: "The production lane uses the matrix-free Section 4.3 hybrid MGPCG authority over the current adaptive liquid frontier. Failure is terminal for the publication; no alternate pressure solver is retained.",
  presetFor: () => ({
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    // Deliberately absent. Seeding it here makes every consumer's band
    // "authored", which defeats the coupling in `authoredFineBand` for any
    // scene profile that overrides only the pressure band.
    globalFineLevelSetFactor: "4",
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values)),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUUniformEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, octreeSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
