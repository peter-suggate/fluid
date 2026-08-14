import type { Vec3 } from "./model";

export interface CouplingDiagnostics {
  displacedVolume_m3: number;
  bodyImpulse_N_s: Vec3;
  fluidReactionImpulse_N_s: Vec3;
  momentumClosureError_N_s: number;
  coupledBodyCount: number;
}
