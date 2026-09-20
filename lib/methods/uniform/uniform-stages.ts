import type { GPUTimestampPhase } from "../../core/performance-trace";

/**
 * The exact partition of one uniform advance, in encode order.
 *
 * These are the trace seams `advanceTo` emits and the labels the fluid
 * pipeline panel's stage graph owns — one table serving both, shared with
 * the encoder, so a stage cannot drift from its measurement. A
 * seam closes everything since the previous seam, clears and copies included:
 * every phase is charged for the buffer state its passes depend on.
 */
export const UNIFORM_ADVANCE_PHASE = Object.freeze({
  extensionAuthority: { id: "velocity-extrapolation", label: "Sec. 3.3 interface authority" },
  extensionFront: { id: "velocity-extrapolation", label: "Sec. 3.3 narrow-band FIM front" },
  extensionHierarchy: { id: "velocity-extrapolation", label: "Sec. 3.3 hierarchy fill + transport shell" },
  densityAdvection: { id: "fine-sdf-advection", label: "Sec. 3.4 conservative density advection" },
  gammaDiffusion: { id: "fine-sdf-advection", label: "Sec. 3.4 axis-Jacobi gamma diffusion" },
  interfaceSharpening: { id: "fine-sdf-redistance", label: "Sec. 3.5 interface density correction" },
  sharpeningMassCorrection: { id: "fine-sdf-redistance", label: "Sec. 3.5 local mass return" },
  solidExcess: { id: "fine-sdf-redistance", label: "Sec. 3.6 partial-solid excess" },
  advectionCorrection: { id: "velocity-advection", label: "Velocity advection + body forces" },
  pressureSetup: { id: "pressure-system", label: "CM11a topology + RHS pyramid" },
  pressureFullCycles: { id: "pressure-solve", label: "CM11a Full-Cycles" },
  pressureVCycles: { id: "pressure-solve", label: "CM11a V-Cycles" },
  pressureFinish: { id: "pressure-solve", label: "CM11a parity copy + fine residual" },
  pressureProjection: { id: "velocity-projection", label: "Pressure projection" },
  rigidCoupling: { id: "other", label: "Rigid two-way coupling + integration" },
  densityPostProcess: { id: "surface-extraction", label: "Render-only wall-film / Sec. 3.8 reconstruction" },
  diagnosticsReduction: { id: "other", label: "Diagnostics reduction" },
} satisfies Record<string, GPUTimestampPhase>);
