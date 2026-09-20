import type { FluidPipelineContext, FluidPipelineGraph, FluidPipelineStage } from "../../core/fluid-pipeline";
import { UNIFORM_ADVANCE_PHASE } from "./uniform-stages";
import { uniformDensityPostProcessingEnabled } from "./uniform-options";
import { UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS } from "./parameters";
import { UNIFORM_CM11A_FULL_CYCLES, UNIFORM_CM11A_V_CYCLES, UNIFORM_CM11A_PRE_SWEEPS, UNIFORM_CM11A_POST_SWEEPS } from "./pressure-policy";

/**
 * The uniform method's pipeline graph, shared with the encoder whose seams
 * it names. Every `phaseLabels` entry below is a `UNIFORM_ADVANCE_PHASE` label
 * — the invariant tests hold the two to an exact one-to-one partition, so a
 * stage card's figure is always the hardware time of its own passes.
 */

const uniformFacts = (context: FluidPipelineContext) => context.info?.uniformPipelineFacts;

const uniformExtensionChip = (context: FluidPipelineContext): string => {
  const facts = uniformFacts(context);
  return facts
    ? `${facts.extrapolationPassesPerInvocation} passes · ${facts.extrapolationHierarchyLevels} MIP levels`
    : "narrow-band FIM + hierarchy fill";
};

const uniformExtensionTip = () => ({
  summary: "Sec. 3.3: builds the interface authority field, marches extended face velocities across the narrow band with a fast-iterative-method front, then fills the far field through a coarse hierarchy so every advection lookup lands on defined velocity.",
  reads: "MAC velocity, surface density",
  writes: "extended MAC velocity, FIM active state",
  feeds: "conservative density and velocity advection",
} as const);

const UNIFORM_FLUID_STAGES: readonly FluidPipelineStage[] = [
  {
    id: "velocity-extension",
    band: "extension",
    side: "left",
    label: "Velocity extension",
    phaseLabels: [
      UNIFORM_ADVANCE_PHASE.extensionAuthority.label,
      UNIFORM_ADVANCE_PHASE.extensionFront.label,
      UNIFORM_ADVANCE_PHASE.extensionHierarchy.label,
    ],
    tip: uniformExtensionTip(),
    controls: [
      {
        kind: "param-choice",
        param: "activeRegion",
        label: "Dispatch",
        options: [
          { value: "on", label: "Active region" },
          { value: "off", label: "Dense control" },
        ],
      },
      {
        kind: "readout",
        label: "Work box",
        hint: "Latest GPU-measured dispatch volume as a share of the uniform lattice.",
        value: (context) => context.values.activeRegion === "off"
          ? "100% dense"
          : context.info?.uniformActiveRegionFraction !== undefined
            ? `${(100 * context.info.uniformActiveRegionFraction).toFixed(1)}%`
            : "—",
      },
      {
        kind: "param-range", param: "extensionFrontSweeps", label: "Front sweeps", unit: "sweeps",
        min: 1, max: 16, step: 1, digits: 0,
        hint: "FIM sweep budget for the two-cell accurate band; every sweep costs an update and a dispatch-gate pass even once converged. Below the sweeps the front needs, unreached band faces fall to the hierarchy fill. Changes apply live.",
      },
      {
        kind: "readout",
        label: "Sweeps with work",
        hint: "Sweeps that still had active faces in the latest diagnostics sample, against the budget. Faces left active when the budget ran out were resolved unconverged.",
        value: (context) => {
          const facts = uniformFacts(context);
          const executed = context.info?.uniformFIMExecutedPasses;
          if (!facts || executed === undefined) return "—";
          const left = context.info?.uniformFIMTerminalActiveFaces ?? 0;
          return `${executed} of ${facts.extrapolationFrontSweeps} · ${left === 0 ? "converged" : `${left} faces unconverged`}`;
        },
      },
    ],
    state: () => "on",
    chip: uniformExtensionChip,
  },
  {
    id: "density-advection",
    band: "surface",
    side: "left",
    label: "Density advection",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.densityAdvection.label],
    tip: {
      summary: "Sec. 3.4: the modified conservative semi-Lagrangian operator — trace gamma and beta backward along the extended velocity, scatter density deficits, gather the conserved result. Mass is redistributed, never created.",
      reads: "surface density, extended MAC velocity",
      writes: "surface density, gamma",
      feeds: "gamma diffusion",
    },
    state: () => "on",
    chip: () => "3 passes · mass-conserving",
  },
  {
    id: "gamma-diffusion",
    band: "surface",
    side: "right",
    label: "Gamma diffusion",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.gammaDiffusion.label],
    tip: {
      summary: "Sec. 3.4: each axis gathers neighbouring half-fluxes from one Jacobi snapshot while transferring the matching donor density; completed axes feed the next axis.",
      reads: "surface density, gamma",
      writes: "surface density, gamma",
      feeds: "interface sharpening",
    },
    toggle: {
      param: "gammaDiffusion", on: "on", off: "off",
      hint: "Toggle Sec. 3.4 axis-Jacobi gamma diffusion. Density transport remains complete when it is off.",
    },
    controls: [{
      kind: "param-range",
      param: "gammaDiffusionIterations",
      label: "Iterations",
      min: 1, max: 7, step: 1,
      hint: "One iteration is three snapshot axis passes; the paper permits one through seven.",
      enabled: (context) => context.values.gammaDiffusion !== "off",
    }],
    state: (context) => context.values.gammaDiffusion === "off" ? "off" : "on",
    chip: (context) => context.values.gammaDiffusion === "off"
      ? "off · identity handoff"
      : `${3 * Number(context.values.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS)} passes · ${context.values.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS} iterations`,
  },
  {
    id: "interface-sharpening",
    band: "surface",
    side: "left",
    label: "Density correction",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.interfaceSharpening.label],
    tip: {
      summary: "Sec. 3.5: computes and applies the local density correction that steepens the smeared liquid-air interface. Its strength scales the paper's pseudo-time dose; local mass return is exposed as the following stage.",
      reads: "surface density, gamma",
      writes: "surface density",
      feeds: "local mass return, then velocity prediction as the liquid mask",
    },
    toggle: {
      param: "densitySharpening", on: "on", off: "off",
      hint: "Toggle the Sec. 3.5 interface-sharpening correction and its dependent mass-return stage.",
    },
    controls: [{
      kind: "param-range",
      param: "sharpeningStrength",
      label: "Strength",
      unit: "×",
      min: 0.25, max: 2, step: 0.05, digits: 2,
      hint: "Multiplier over the paper's 3dt sharpening pseudo-time dose.",
      enabled: (context) => context.values.densitySharpening !== "off",
    }],
    state: (context) => context.values.densitySharpening === "off" ? "off" : "on",
    chip: (context) => context.values.densitySharpening === "off"
      ? "off · advected density"
      : `1 pass · ${Number(context.values.sharpeningStrength ?? 1).toFixed(2)}× dose`,
  },
  {
    id: "sharpening-mass-correction",
    band: "surface",
    side: "right",
    label: "Local mass return",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.sharpeningMassCorrection.label],
    tip: {
      summary: "Sec. 3.5 Algorithm 2: traces each removed density parcel toward the 0.5 iso-contour, scatters it locally, and resolves the deposits. Disabling it intentionally exposes the raw, non-conservative density correction while preserving a valid downstream field.",
      reads: "corrected density, per-cell removed mass",
      writes: "mass-returned surface density",
      feeds: "partial-solid excess and pressure classification",
      gate: "interface sharpening and local mass return are both on",
    },
    toggle: {
      param: "sharpeningMassCorrection", on: "on", off: "off",
      hint: "Toggle only Algorithm 2's local conservation step; density correction remains active.",
    },
    controls: [{
      kind: "param-range",
      param: "sharpeningDistance",
      label: "Trace distance",
      unit: "cells",
      min: 0.1, max: 3.1, step: 0.1, digits: 1,
      hint: "Maximum gradient-trace distance D; the paper explores 1.1–3.1 cells, and values below that keep returned mass local.",
      enabled: (context) => context.values.densitySharpening !== "off"
        && context.values.sharpeningMassCorrection !== "off",
    }],
    state: (context) => context.values.densitySharpening === "off"
      ? "unavailable"
      : context.values.sharpeningMassCorrection === "off" ? "off" : "on",
    chip: (context) => context.values.densitySharpening === "off"
      ? "requires density correction"
      : context.values.sharpeningMassCorrection === "off"
        ? "off · non-conservative ablation"
        : `2 passes · D ${Number(context.values.sharpeningDistance ?? 2.1).toFixed(1)} cells`,
  },
  {
    id: "solid-excess",
    band: "surface",
    side: "right",
    label: "Partial-solid excess",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.solidExcess.label],
    tip: {
      summary: "Sec. 3.6: current density is reconciled before transport can mask newly covered donors, then checked again after sharpening. Excess beyond a cut cell's open fraction is scattered to open neighbours and resolved conservatively; genuinely enclosed excess is published as telemetry.",
      reads: "surface density, solid fractions",
      writes: "surface density",
      gate: "the scene has rigid bodies or terrain",
    },
    toggle: {
      param: "solidExcessCorrection", on: "on", off: "off",
      hint: "Toggle Sec. 3.6 cut-cell excess redistribution in scenes that contain solids.",
    },
    state: (context) => context.bodyCount > 0 || context.hasTerrain
      ? context.values.solidExcessCorrection === "off" ? "off" : "on"
      : "unavailable",
    chip: (context) => context.bodyCount > 0 || context.hasTerrain
      ? context.values.solidExcessCorrection === "off" ? "off · excess retained" : "4 passes · entry + post-sharpening"
      : "no solids in scene",
  },
  {
    id: "velocity-advection",
    band: "momentum",
    side: "left",
    label: "Velocity advection",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.advectionCorrection.label],
    tip: {
      summary: "Algorithm 1 step 3: configurable semi-Lagrangian or CM11b bounded MacCormack velocity transport, followed by gravity, viscosity, and surface tension.",
      reads: "extended MAC velocity; predicted and reverse fields in MacCormack mode",
      writes: "advected MAC velocity",
      feeds: "pressure solve",
    },
    controls: [{
      kind: "param-choice",
      param: "velocityTransport",
      label: "Transport",
      hint: "Choose the one-pass semi-Lagrangian update or the higher-order bounded MacCormack sequence.",
      options: [
        { value: "semi-lagrangian", label: "Semi-Lagrangian" },
        { value: "maccormack", label: "Bounded MacCormack" },
      ],
    }],
    state: () => "on",
    chip: (context) => context.values.velocityTransport === "maccormack"
      ? "forward + reverse + bounded correction"
      : "one backward-trace pass",
  },
  {
    id: "pressure-system",
    band: "pressure",
    side: "left",
    label: "System build",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.pressureSetup.label],
    tip: {
      summary: "CM11a setup: classify cell topology, build the divergence right-hand side from the advected velocity, and restrict both down the multigrid pyramid before any cycle runs.",
      reads: "advected MAC velocity, surface density, solid fractions",
      writes: "per-level topology + RHS",
      feeds: "multigrid cycles",
    },
    state: () => "on",
    chip: (context) => {
      const facts = uniformFacts(context);
      return facts
        ? `${facts.multigridPasses.setup} passes · ${facts.multigridLevels} levels`
        : "topology + RHS pyramid";
    },
  },
  {
    id: "pressure-cycles",
    band: "pressure",
    side: "right",
    label: "Multigrid cycles",
    phaseLabels: [
      UNIFORM_ADVANCE_PHASE.pressureFullCycles.label,
      UNIFORM_ADVANCE_PHASE.pressureVCycles.label,
    ],
    tip: {
      summary: "The CM11a LCP multigrid solve: full cycles first (coarsest-up, seeding every level), then V-cycles to polish. The counts are caps: remaining cycles exit once the fine projected residual meets tolerance. Each level runs projected red-black Gauss-Seidel sweeps with the liquid-air complementarity condition enforced per sweep. Under the lagged cycle budget the cap is also applied on the host, so cycles the last observed step did not need are never encoded and never pay their launch floor.",
      reads: "per-level topology + RHS",
      writes: "pressure",
      feeds: "parity copy + fine residual",
    },
    controls: [
      {
        kind: "param-range", param: "pressureResidualTolerance", label: "Residual tolerance", unit: "s⁻¹",
        min: 0, max: 100, step: 0.0001, digits: 4, editable: true,
        hint: "Stop remaining Full-Cycles and V-Cycles once the projected residual ∞-norm is at or below tolerance. Zero disables early exit; changes apply live.",
      },
      {
        kind: "param-choice", param: "pressureCycleBudget", label: "Cycle budget",
        options: [
          { value: "lagged", label: "Lagged", hint: "Encode only as many cycles as the latest diagnostics sample says the solve needed, plus the headroom. A cycle that is never encoded costs neither its GPU launch floor (~6-13 µs a pass, whether or not the body runs) nor its ~11 µs of CPU encode. The residual gate still stops a converged solve inside the encoded prefix." },
          { value: "fixed", label: "Fixed", hint: "Always encode the configured schedule and let the GPU-side gate skip the remainder. This is the command stream the solver encoded before the budget existed." },
        ],
      },
      {
        kind: "param-range", param: "pressureBudgetHeadroom", label: "Budget headroom",
        unit: "cycles", min: 0, max: 4, step: 1, digits: 0,
        hint: "Cycles encoded above the last observed demand. The signal lags the encoded step by one or more frames, so the rule is asymmetric: it shrinks by this headroom and grows by doubling whenever a step used every encoded cycle without meeting tolerance.",
        enabled: (context) => context.values.pressureCycleBudget !== "fixed",
      },
      {
        kind: "readout", label: "Cycles encoded",
        hint: "Cycles in the command stream this step, against the configured schedule. Under Fixed the two are always equal.",
        value: (context) => {
          const info = context.info as unknown as {
            uniformPressureCyclesEncoded?: number; uniformPressureCyclesConfigured?: number } | null;
          const encoded = info?.uniformPressureCyclesEncoded;
          const configured = info?.uniformPressureCyclesConfigured;
          return encoded === undefined || configured === undefined
            ? "—" : `${encoded} of ${configured}`;
        },
      },
      {
        kind: "readout", label: "Passes encoded",
        hint: "Compute passes the whole pressure solve encoded this step — setup pyramid, cycles and finish — against the configured schedule's count.",
        value: (context) => {
          const info = context.info as unknown as {
            uniformPressurePassesEncoded?: number; uniformPressurePassesConfigured?: number } | null;
          const encoded = info?.uniformPressurePassesEncoded;
          const configured = info?.uniformPressurePassesConfigured;
          if (encoded === undefined || configured === undefined) return "—";
          const share = configured > 0 ? Math.round(100 * encoded / configured) : 100;
          return `${encoded} / ${configured} (${share}%)`;
        },
      },
      {
        kind: "param-range",
        param: "pressureFullCycles",
        label: "Max Full-Cycles",
        min: 0, max: 5, step: 1,
        hint: "Coarsest-up CM11a Full-Cycles; the paper schedule uses three. Changing it rebuilds the precomputed pressure plan and resets to t=0.",
      },
      {
        kind: "param-range",
        param: "pressureVCycles",
        label: "Max V-Cycles",
        min: 0, max: 8, step: 1,
        hint: "Refinement V-Cycles after the Full-Cycles; the paper schedule uses four. Changing it rebuilds the precomputed pressure plan and resets to t=0.",
      },
      {
        kind: "param-range",
        param: "pressureSweeps",
        label: "Pre/post sweeps",
        min: 1, max: 8, step: 1,
        hint: "Projected red-black Gauss-Seidel sweeps before and after each coarse correction. Changing it rebuilds the precomputed pressure plan and resets to t=0.",
      },
      {
        kind: "readout", label: "Completed cycles",
        hint: "Cycles actually executed in the latest diagnostics sample. Remaining encoded cycles fast-exit after convergence.",
        value: context => {
          const info = context.info as unknown as { uniformCM11aFullCyclesExecuted?: number; uniformCM11aVCyclesExecuted?: number; uniformCM11aCycleConverged?: boolean } | null;
          if (info?.uniformCM11aFullCyclesExecuted === undefined) return "—";
          return `${info.uniformCM11aFullCyclesExecuted} full + ${info.uniformCM11aVCyclesExecuted} V · ${info.uniformCM11aCycleConverged ? "tolerance met" : "cycle limit"}`;
        },
      },
      {
        kind: "readout",
        label: "Residual ∞-norm",
        hint: "Fine-level residual after the configured schedule, from diagnostics readback.",
        value: (context) => {
          const residual = (context.info as unknown as {
            uniformCM11aFineResidualInfinity?: number;
          } | null)?.uniformCM11aFineResidualInfinity;
          return residual === undefined || !Number.isFinite(residual)
            ? "—"
            : residual.toExponential(2);
        },
      },
    ],
    state: () => "on",
    chip: (context) => {
      const facts = uniformFacts(context);
      const configured = facts?.pressureSchedule;
      const fullCycles = configured?.fullCycles ?? Number(context.values.pressureFullCycles ?? UNIFORM_CM11A_FULL_CYCLES);
      const vCycles = configured?.vCycles ?? Number(context.values.pressureVCycles ?? UNIFORM_CM11A_V_CYCLES);
      const preSweeps = configured?.preSweeps ?? Number(context.values.pressureSweeps ?? UNIFORM_CM11A_PRE_SWEEPS);
      const postSweeps = configured?.postSweeps ?? Number(context.values.pressureSweeps ?? UNIFORM_CM11A_POST_SWEEPS);
      const schedule = `${fullCycles} full + ${vCycles} V · ${preSweeps}+${postSweeps} sweeps`;
      const cyclePasses = facts
        ? facts.multigridPasses["full-cycle"] + facts.multigridPasses["v-cycle"] : undefined;
      const info = context.info as unknown as {
        uniformPressureCycleBudget?: "lagged" | "fixed";
        uniformPressureCyclesEncoded?: number; uniformPressureCyclesConfigured?: number;
        uniformPressurePassesEncoded?: number } | null;
      const lagged = (info?.uniformPressureCycleBudget
        ?? (context.values.pressureCycleBudget === "fixed" ? "fixed" : "lagged")) === "lagged";
      const encodedCycles = info?.uniformPressureCyclesEncoded;
      const configuredCycles = info?.uniformPressureCyclesConfigured;
      if (!lagged) {
        return cyclePasses === undefined
          ? `fixed · ${schedule}` : `fixed · ${cyclePasses} passes · ${schedule}`;
      }
      if (encodedCycles === undefined || configuredCycles === undefined) return `lagged · ${schedule}`;
      // Cycle passes only, so the number means the same thing on both arms:
      // the encoded stream minus the setup pyramid and the finish section.
      const encodedCyclePasses = facts && info?.uniformPressurePassesEncoded !== undefined
        ? info.uniformPressurePassesEncoded - facts.multigridPasses.setup - facts.multigridPasses.finish
        : undefined;
      return `lagged · ${encodedCycles} of ${configuredCycles} cycles`
        + (encodedCyclePasses !== undefined && cyclePasses !== undefined
          ? ` · ${encodedCyclePasses} of ${cyclePasses} passes` : "");
    },
  },
  {
    id: "pressure-finish",
    band: "pressure",
    side: "left",
    label: "Solve finish",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.pressureFinish.label],
    tip: {
      summary: "Copies the converged pressure to the parity texture the projection reads and computes the fine-level residual the diagnostics report.",
      reads: "pressure",
      writes: "pressure (parity), residual diagnostics",
      feeds: "pressure projection",
    },
    state: () => "on",
    chip: (context) => {
      const facts = uniformFacts(context);
      return facts ? `${facts.multigridPasses.finish} passes` : "parity copy + residual";
    },
  },
  {
    id: "pressure-projection",
    band: "pressure",
    side: "right",
    label: "Pressure projection",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.pressureProjection.label],
    tip: {
      summary: "Subtracts the pressure gradient from the advected velocity, restoring a divergence-free field at every liquid face.",
      reads: "advected MAC velocity, pressure",
      writes: "divergence-free MAC velocity",
      feeds: "rigid coupling, next advance",
    },
    state: () => "on",
    chip: () => "1 pass",
  },
  {
    id: "rigid-coupling",
    band: "coupling",
    side: "left",
    label: "Rigid coupling",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.rigidCoupling.label],
    tip: {
      summary: "Two-way momentum exchange: the fluid pushes on each body through sampled pressure and drag, bodies push back on the face velocities, then the rigid system integrates poses for the next advance.",
      reads: "divergence-free MAC velocity, surface density, body poses",
      writes: "MAC velocity, body poses + momenta",
      gate: "the scene has rigid bodies",
    },
    toggle: {
      param: "rigidCoupling", on: "on", off: "off",
      hint: "Toggle two-way fluid/body momentum exchange and rigid integration; bodies remain solid pressure boundaries.",
    },
    state: (context) => context.bodyCount > 0
      ? context.values.rigidCoupling === "off" ? "off" : "on"
      : "unavailable",
    chip: (context) => context.bodyCount > 0
      ? context.values.rigidCoupling === "off"
        ? `off · ${context.bodyCount} ${context.bodyCount === 1 ? "body" : "bodies"} held`
        : `${context.bodyCount} ${context.bodyCount === 1 ? "body" : "bodies"}`
      : "no rigid bodies",
  },
  {
    id: "density-post-process",
    band: "output",
    side: "left",
    label: "Render density",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.densityPostProcess.label],
    tip: {
      summary: "Always reconstructs sub-half-cell liquid supported by tank walls and embedded solids at a mass-proportional thickness. Sec. 3.8 optionally adds its global gamma-blur reconstruction. The result never feeds simulation state.",
      reads: "surface density, gamma",
      writes: "render surface texture",
      gate: "wall films always; global Sec. 3.8 reconstruction when enabled",
    },
    controls: [{
      kind: "param-choice",
      param: "densityPostProcessing",
      label: "Sec. 3.8 reconstruction",
      hint: "Render-only surface smoothing; simulation state is identical either way. Changing it rebuilds the solver.",
      options: [
        { value: "scene", label: "Scene", hint: "On for symmetry and mini-dam scenes where sub-grid sheets are presentation-critical." },
        { value: "off", label: "Wall films", hint: "Mass-proportional solid-supported sheets only." },
        { value: "on", label: "Wall films + Sec. 3.8", hint: "Adds gamma blur and global sub-grid resolve." },
      ],
    }],
    toggle: {
      param: "densityPostProcessing", on: "on", off: "off",
      hint: "Toggle the render-only Sec. 3.8 density reconstruction without changing simulation physics.",
    },
    state: () => "on",
    chip: (context) => uniformDensityPostProcessingEnabled(
      context.values.densityPostProcessing, context.sceneId)
      ? "4 passes · Sec. 3.8 + wall films"
      : context.values.densityPostProcessing === "scene"
        ? "1 pass · wall films"
        : "1 pass · wall films",
  },
  {
    id: "diagnostics-reduction",
    band: "output",
    side: "right",
    label: "Diagnostics reduction",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.diagnosticsReduction.label],
    tip: {
      summary: "Reduces liquid volume, front position, and maximum speed into the telemetry buffer the stats readback maps; the panel's drift and speed figures come from here.",
      reads: "surface density, MAC velocity",
      writes: "reductions buffer",
      feeds: "readStats() telemetry",
    },
    state: () => "on",
    chip: () => "1 pass",
  },
];

export const UNIFORM_FLUID_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "uniform",
  bands: [
    { id: "extension", label: "Velocity extension · Sec. 3.3" },
    { id: "surface", label: "Surface density · Secs. 3.4–3.6" },
    { id: "momentum", label: "Momentum transport" },
    { id: "pressure", label: "Pressure · CM11a multigrid" },
    { id: "coupling", label: "Rigid coupling" },
    { id: "output", label: "Output + diagnostics" },
  ],
  stages: UNIFORM_FLUID_STAGES,
});
