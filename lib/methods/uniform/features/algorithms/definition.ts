import type { MethodParamSpec } from "../../../../core/method-contract";
import { parameterVariantFeature } from "../../../../core/method-parameter-variants";
const runtimeUpdate = {update:"runtime" as const};

export const ALGORITHM_PARAMS: MethodParamSpec[] = [
  {
    kind: "select",
    key: "activeRegion",
    label: "Active-region dispatch",
    default: "off",
    tier: "coarse",
    options: [
      { value: "on", label: "On · sparse GPU work box" },
      { value: "off", label: "Off · dense control" },
    ],
    hint: "Dense full-lattice dispatch is the reference default. Enable the GPU-resident sparse work box only as an explicit optimization A/B.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "gammaDiffusion",
    label: "Gamma diffusion",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · axis-Jacobi diffusion" },
      { value: "off", label: "Off · retain transported gamma" },
    ],
    hint: "Ablates Sec. 3.4's snapshot axis diffusion. The conservative density transport still publishes a complete rho/gamma state for downstream stages.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "densitySharpening",
    label: "Interface sharpening",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · Sec. 3.5" },
      { value: "off", label: "Off · advected density" },
    ],
    hint: "Ablates the local Sec. 3.5 density correction. Turning it off bypasses both sharpening and its dependent mass-return stage.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "sharpeningMassCorrection",
    label: "Sharpening mass return",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · local conservative return" },
      { value: "off", label: "Off · raw density correction" },
    ],
    hint: "Controls Algorithm 2 separately from the density correction. Off keeps the sharpened field but omits the scatter/resolve that returns removed mass locally.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "solidExcessCorrection",
    label: "Partial-solid excess",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · Sec. 3.6" },
      { value: "off", label: "Off · retain cut-cell excess" },
    ],
    hint: "Ablates the conservative redistribution of density that exceeds a cut cell's open fraction.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "rigidCoupling",
    label: "Rigid coupling",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · two-way coupling" },
      { value: "off", label: "Off · fluid-only motion" },
    ],
    hint: "Disables fluid/body momentum exchange and rigid integration while retaining the bodies as solid boundaries.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "velocityTransport",
    label: "Velocity advection",
    default: "semi-lagrangian",
    tier: "coarse",
    options: [
      { value: "semi-lagrangian", label: "Semi-Lagrangian · one pass" },
      { value: "maccormack", label: "Bounded MacCormack · three passes" },
    ],
    hint: "Semi-Lagrangian uses the original single backward-trace update. Bounded MacCormack adds a forward prediction, predicted-field extension, reverse trace, and local-extrema-limited correction.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "liquidOnlyVelocityAdvection",
    label: "Liquid-only velocity advection",
    default: "off",
    tier: "coarse",
    options: [
      { value: "off", label: "Off · paper feedback" },
      { value: "on", label: "On · phase-masked liquid" },
    ],
    hint: "On gathers liquid momentum only from prior liquid faces in the authoritative velocity field. The extension still defines the SL characteristic but supplies no momentum. Off restores unrestricted CM11b transport-field sampling for comparison. Scenes do not override this live control.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "timeStep",
    label: "Time step",
    default: "paper",
    tier: "coarse",
    options: [
      { value: "paper", label: "Paper · 1/30 s large steps" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    hint: "Chentanez-Müller run dt=1/30 s (CFL 8-25) in every example; Sec. 3.5 sharpening only balances transport diffusion at that per-step dose. Scene-step mode exists for matched-dt comparison lanes and dilutes the interface at small dt.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "densityPostProcessing",
    label: "Sub-grid rendering",
    default: "off",
    tier: "fine",
    options: [
      { value: "scene", label: "Scene · Sec. 3.8 where needed" },
      { value: "off", label: "Wall films only" },
      { value: "on", label: "Wall films + Sec. 3.8" },
    ],
    hint: "Render-only: mass-proportional sheets against walls and solids are always reconstructed. Scene/On additionally enable the paper's Sec. 3.8 global reconstruction. Neither feeds simulation physics.",
  },
];

export const algorithmFeature = parameterVariantFeature("simulation.uniform.algorithms", ALGORITHM_PARAMS, ["simulation.dense-grid"]);
