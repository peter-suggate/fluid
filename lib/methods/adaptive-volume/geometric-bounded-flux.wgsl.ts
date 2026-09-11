/**
 * Binding-free conservative flux-corrected volume transport primitives.
 *
 * All amounts, capacities, sweeps and fluxes are physical volumes. A shared
 * physical subface has one canonical orientation: positive goes from its
 * negative cell to its positive cell. Freeze its low/high flux before any cell
 * reads it. Coarse faces must be sums of these same physical subfaces.
 *
 * Dispatch ordering:
 *  1. Freeze face sweeps and low/high fluxes from the accepted old state.
 *  2. Each cell gathers low fluxes and the positive/negative anti-flux budgets.
 *  3. Freeze each cell's increase/decrease limits.
 *  4. Each face computes ONE limited flux using BOTH endpoint limits.
 *  5. Each cell gathers those identical signed face fluxes into its new amount.
 * Separate dispatches establish these dependencies; in-place updates do not.
 *
 * Preconditions: 0 <= old volume <= old open capacity, nonnegative capacity,
 * and a bounded low-order update. For fixed capacities the upwind low scheme
 * needs outgoing swept-open-volume <= capacity AND a physical face divergence
 * compatible with pressure (zero for full incompressible interior cells).
 * Moving capacities require the corresponding wall/geometric-conservation
 * flux balance. Donor CFL alone cannot prevent receiver overfill. Pressure
 * row velocities or collocated velocities must not be assumed to provide this
 * physical subface flux identity without verification.
 *
 * Invalid low states are reported explicitly; the caller must reject/reduce
 * the substep or repair its compatible flux construction. Limiting anti-flux
 * cannot fix an already overfilled or negative low state. No authority amount
 * is clamped by these functions. Existing scalar terrain capacity may be used
 * provisionally with its existing wall semantics; this is not exact clipped
 * solid geometry. Algebraic conservation uses the shared face value; f32 cell
 * accumulation still incurs rounding, so global receipts need roundoff bounds.
 */
export const geometricBoundedFluxWGSL = /* wgsl */ `
struct GeometricFCTCellLimits {
  increase:f32,
  decrease:f32,
  valid:u32,
}
struct GeometricFCTFaceFlux {
  low:f32,
  high:f32,
}
struct GeometricFCTLimitedFace {
  flux:f32,
  factor:f32,
  valid:u32,
}

// Exact comparisons intentionally expose an invalid authority/low-order state.
// A separate diagnostic policy can classify roundoff; it must not silently
// convert this failure into a clamped volume. NaNs fail the comparisons.
fn geometricFctVolumeValid(volume:f32,capacity:f32)->bool{
  return capacity>=0.0&&capacity<=3.402823466e38
    &&volume>=0.0&&volume<=capacity;
}

// Full donor faces carry the entire signed open sweep. A dry/closed donor
// carries zero. Callers must separately validate sweeps and old-state bounds.
fn geometricFctUpwindFlux(signedOpenSweep:f32,negativeVolume:f32,
 negativeCapacity:f32,positiveVolume:f32,positiveCapacity:f32)->f32{
  if(signedOpenSweep>0.0){
    if(negativeCapacity<=0.0){return 0.0;}
    return signedOpenSweep*(negativeVolume/negativeCapacity);
  }
  if(signedOpenSweep<0.0){
    if(positiveCapacity<=0.0){return 0.0;}
    return signedOpenSweep*(positiveVolume/positiveCapacity);
  }
  return 0.0;
}

// Contribution to this cell's extensive amount. The negative endpoint loses
// positive oriented flux; the positive endpoint gains it.
fn geometricFctCellDelta(orientedFlux:f32,isNegativeEndpoint:bool)->f32{
  return select(orientedFlux,-orientedFlux,isNegativeEndpoint);
}

// x = potential increase, y = potential decrease. Gather across all incident
// shared subfaces, using the same frozen high-minus-low value at both ends.
fn geometricFctAntidiffusionBudget(face:GeometricFCTFaceFlux,
 isNegativeEndpoint:bool)->vec2f{
  let delta=geometricFctCellDelta(face.high-face.low,isNegativeEndpoint);
  return vec2f(max(delta,0.0),max(-delta,0.0));
}

fn geometricFctCellLimits(lowVolume:f32,capacity:f32,
 positiveBudget:f32,negativeBudget:f32)->GeometricFCTCellLimits{
  if(!geometricFctVolumeValid(lowVolume,capacity)
    ||!(positiveBudget>=0.0&&positiveBudget<=3.402823466e38)
    ||!(negativeBudget>=0.0&&negativeBudget<=3.402823466e38)){
    return GeometricFCTCellLimits(0.0,0.0,0u);
  }
  var increase=1.0;var decrease=1.0;
  if(positiveBudget>0.0){increase=min(1.0,(capacity-lowVolume)/positiveBudget);}
  if(negativeBudget>0.0){decrease=min(1.0,lowVolume/negativeBudget);}
  return GeometricFCTCellLimits(increase,decrease,1u);
}

// Limits refer to the sign of ANTI-flux, which can oppose the low-order flux.
// Do not choose donor/receiver from velocity or low-flux sign here.
fn geometricFctLimitFace(face:GeometricFCTFaceFlux,
 negative:GeometricFCTCellLimits,positive:GeometricFCTCellLimits)->GeometricFCTLimitedFace{
  if(negative.valid==0u||positive.valid==0u){
    return GeometricFCTLimitedFace(0.0,0.0,0u);
  }
  let anti=face.high-face.low;
  if(!(abs(face.low)<=3.402823466e38&&abs(face.high)<=3.402823466e38
    &&abs(anti)<=3.402823466e38)){
    return GeometricFCTLimitedFace(0.0,0.0,0u);
  }
  let factor=select(min(negative.increase,positive.decrease),
    min(negative.decrease,positive.increase),anti>=0.0);
  return GeometricFCTLimitedFace(face.low+factor*anti,factor,1u);
}
`;
