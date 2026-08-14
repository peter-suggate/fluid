/**
 * Numerically shared Chentanez--Muller 2012 formulas.
 *
 * Keep pure paper algebra here. Grid addressing, topology, traces, storage,
 * and dispatch policy belong to each method. That split lets the dense
 * reference and an adaptive control-volume implementation share one source
 * for the parts that must agree at a same-resolution seam.
 */

export const CM12_PAPER_DT_S = 1 / 30;
export const CM12_LIQUID_ISOVALUE = 0.5;
export const CM12_GHOST_FLUID_THETA_MIN = 0.05;
export const CM12_TRANSPORT_FIXED_SCALE = 1_048_576;
export const CM12_GAMMA_INTERIOR_MIN = 0.5;
export const CM12_GAMMA_MAX = 2.5;
export const CM12_VOLUME_CORRECTION_LAMBDA = 0.5;
export const CM12_VOLUME_CORRECTION_ETA = 1;
export const CM12_SHARPENING_TAU = 0.4;

export function cm12GhostFluidTheta(
  liquidPhi: number,
  airPhi: number,
  denominatorEpsilon: number,
): number {
  return Math.min(1, Math.max(CM12_GHOST_FLUID_THETA_MIN,
    Math.abs(liquidPhi) / Math.max(Math.abs(liquidPhi) + Math.abs(airPhi), denominatorEpsilon)));
}

/** CM12 Sec. 3.4's bounded persistent-gamma sample. */
export function cm12ConditionedGamma(sampledGamma: number, visibleStencilTotal: number): number {
  return visibleStencilTotal >= 1 - 1e-6
    ? Math.min(CM12_GAMMA_MAX, Math.max(CM12_GAMMA_INTERIOR_MIN, sampledGamma))
    : Math.min(sampledGamma, CM12_GAMMA_MAX);
}

/** Equal-volume A_ij after the paper's max(1,beta_j) column clamp. */
export function cm12ConditionedRowCoefficient(
  advectedGamma: number,
  normalizedBackwardWeight: number,
  donorBeta: number,
): number {
  return advectedGamma * normalizedBackwardWeight / Math.max(1, donorBeta);
}

/**
 * Contribution of A_ij to beta_j on unequal control volumes.
 *
 * The conservative column condition is sum_i(V_i A_ij) = V_j, hence
 * beta_j = sum_i((V_i / V_j) A_ij). It collapses to A_ij on a uniform grid.
 */
export function cm12VolumeWeightedBetaContribution(
  receiverVolume: number,
  donorVolume: number,
  rowCoefficient: number,
): number {
  return receiverVolume * rowCoefficient / donorVolume;
}

/**
 * A receiver-row coefficient that fills a weighted beta deficit at a donor.
 * Its beta contribution is exactly deficit * normalizedForwardWeight.
 */
export function cm12VolumeScaledDeficitCoefficient(
  donorVolume: number,
  receiverVolume: number,
  deficit: number,
  normalizedForwardWeight: number,
): number {
  return donorVolume * deficit * normalizedForwardWeight / receiverVolume;
}

/** Transfer a donor intensive quantity through the volume-scaled deficit row. */
export function cm12VolumeScaledDeficitTransfer(
  donorQuantity: number,
  donorVolume: number,
  receiverVolume: number,
  deficit: number,
  normalizedForwardWeight: number,
): number {
  return donorQuantity * donorVolume * deficit * normalizedForwardWeight / receiverVolume;
}

export interface Cm12GammaDiffusionFlux {
  rho: number;
  gamma: number;
}

/** LAF11/CM12 snapshot-Jacobi flux into the first endpoint of a face. */
export function cm12GammaDiffusionFluxInto(
  ownRho: number,
  ownGamma: number,
  neighborRho: number,
  neighborGamma: number,
  open: number,
): Cm12GammaDiffusionFlux {
  const gamma = 0.5 * open * (neighborGamma - ownGamma);
  let rho = 0;
  if (neighborGamma > ownGamma) {
    rho = open * neighborRho * (neighborGamma - ownGamma) / (2 * Math.max(neighborGamma, 1e-9));
  } else if (ownGamma > neighborGamma) {
    rho = -open * ownRho * (ownGamma - neighborGamma) / (2 * Math.max(ownGamma, 1e-9));
  }
  return { rho, gamma };
}

/** CM12 Sec. 3.7 excess-volume divergence, with the existing one-cell/dt cap. */
export function cm12VolumeCorrectionDivergence(
  normalizedDensity: number,
  cellSize: number,
  dt: number,
): number {
  const excess = Math.max(0, normalizedDensity - 1);
  if (excess <= 0) return 0;
  const paperRate = Math.min(CM12_VOLUME_CORRECTION_LAMBDA * excess,
    CM12_VOLUME_CORRECTION_ETA) / cellSize;
  return Math.min(paperRate, 1 / dt);
}

/** Cubic Sec. 3.5 density-sharpening weight before the Godunov gradient. */
export function cm12SharpeningWeight(rho: number, maximumNeighborDifference: number): number {
  const displacement = rho - CM12_LIQUID_ISOVALUE;
  return displacement * displacement * displacement
    * (1 - Math.min(1, maximumNeighborDifference / CM12_SHARPENING_TAU));
}

function wgslF32(value: number): string {
  const literal = String(value);
  return literal.includes(".") || literal.includes("e") ? literal : `${literal}.0`;
}

/**
 * Emit the matching device-side formulas. Callers may request constants only
 * when their shader needs CM12 classification but none of the pure formulas.
 */
export function createCm12NumericsWGSL(constantsOnly = false): string {
  const constants = /* wgsl */ `
const CM12_LIQUID_ISOVALUE:f32=${wgslF32(CM12_LIQUID_ISOVALUE)};
const CM12_GHOST_FLUID_THETA_MIN:f32=${wgslF32(CM12_GHOST_FLUID_THETA_MIN)};
const CM12_TRANSPORT_FIXED:f32=${wgslF32(CM12_TRANSPORT_FIXED_SCALE)};
const CM12_GAMMA_INTERIOR_MIN:f32=${wgslF32(CM12_GAMMA_INTERIOR_MIN)};
const CM12_GAMMA_MAX:f32=${wgslF32(CM12_GAMMA_MAX)};
const CM12_VOLUME_CORRECTION_LAMBDA:f32=${wgslF32(CM12_VOLUME_CORRECTION_LAMBDA)};
const CM12_VOLUME_CORRECTION_ETA:f32=${wgslF32(CM12_VOLUME_CORRECTION_ETA)};
const CM12_SHARPENING_TAU:f32=${wgslF32(CM12_SHARPENING_TAU)};
`;
  if (constantsOnly) return constants;
  return constants + /* wgsl */ `
fn cm12GhostFluidTheta(liquidPhi:f32,airPhi:f32,denominatorEpsilon:f32)->f32{
  return clamp(abs(liquidPhi)/max(abs(liquidPhi)+abs(airPhi),denominatorEpsilon),CM12_GHOST_FLUID_THETA_MIN,1.0);
}
fn cm12ConditionedGamma(sampledGamma:f32,visibleStencilTotal:f32)->f32{
  return select(min(sampledGamma,CM12_GAMMA_MAX),clamp(sampledGamma,CM12_GAMMA_INTERIOR_MIN,CM12_GAMMA_MAX),visibleStencilTotal>=1.0-1e-6);
}
fn cm12ConditionedRowCoefficient(advectedGamma:f32,normalizedBackwardWeight:f32,donorBeta:f32)->f32{
  return advectedGamma*normalizedBackwardWeight/max(1.0,donorBeta);
}
fn cm12VolumeWeightedBetaContribution(receiverVolume:f32,donorVolume:f32,rowCoefficient:f32)->f32{
  return receiverVolume*rowCoefficient/donorVolume;
}
fn cm12VolumeScaledDeficitCoefficient(donorVolume:f32,receiverVolume:f32,deficit:f32,normalizedForwardWeight:f32)->f32{
  return donorVolume*deficit*normalizedForwardWeight/receiverVolume;
}
fn cm12VolumeScaledDeficitTransfer(donorQuantity:f32,donorVolume:f32,receiverVolume:f32,deficit:f32,normalizedForwardWeight:f32)->f32{
  return donorQuantity*donorVolume*deficit*normalizedForwardWeight/receiverVolume;
}
fn cm12GammaDiffusionFluxInto(ownRho:f32,ownGamma:f32,neighborRho:f32,neighborGamma:f32,open:f32)->vec2f{
  let gammaFlux=0.5*open*(neighborGamma-ownGamma);
  var rhoFlux=0.0;
  if(neighborGamma>ownGamma){
    rhoFlux=open*neighborRho*(neighborGamma-ownGamma)/(2.0*max(neighborGamma,1e-9));
  }else if(ownGamma>neighborGamma){
    rhoFlux=-open*ownRho*(ownGamma-neighborGamma)/(2.0*max(ownGamma,1e-9));
  }
  return vec2f(rhoFlux,gammaFlux);
}
fn cm12VolumeCorrectionDivergence(normalizedDensity:f32,cellSize:f32,dt:f32)->f32{
  let excess=max(0.0,normalizedDensity-1.0);
  if(excess<=0.0){return 0.0;}
  let paperRate=min(CM12_VOLUME_CORRECTION_LAMBDA*excess,CM12_VOLUME_CORRECTION_ETA)/cellSize;
  return min(paperRate,1.0/dt);
}
fn cm12SharpeningWeight(rho:f32,maximumNeighborDifference:f32)->f32{
  let displacement=rho-CM12_LIQUID_ISOVALUE;
  return displacement*displacement*displacement*(1.0-min(1.0,maximumNeighborDifference/CM12_SHARPENING_TAU));
}
`;
}
