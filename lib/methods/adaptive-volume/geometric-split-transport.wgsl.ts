/**
 * Binding-free directional geometric transport building blocks.
 * Requires geometricInterfaceWGSL and createGeometricSubfacesWGSL().
 *
 * These helpers do not dispatch or mutate state. The integrating resident must
 * audit accepted subfaces, transport one axis at a time, reconstruct between
 * sweeps, freeze its compression coefficient over the complete axis cycle,
 * store each paired flux once, and reject every nonzero error before publication.
 * Scalar solid capacities do not supply the clipped geometry this needs.
 */
export const geometricSplitTransportWGSL = /* wgsl */ `
const GEOMETRIC_SPLIT_GEOMETRY_ERROR:u32=1u;
const GEOMETRIC_SPLIT_AMOUNT_ERROR:u32=2u;
const GEOMETRIC_SPLIT_CFL_ERROR:u32=4u;
const GEOMETRIC_SPLIT_PLANE_ERROR:u32=8u;
const GEOMETRIC_SPLIT_CUT_CELL_ERROR:u32=16u;
const GEOMETRIC_SPLIT_BOUNDS_ERROR:u32=32u;
const GEOMETRIC_SPLIT_BULK_ERROR:u32=64u;
const GEOMETRIC_SPLIT_DIVERGENCE_ERROR:u32=128u;

struct GeometricSplitFlux {
  errors:u32,
  negativeCell:u32,
  positiveCell:u32,
  // Positive values transport from negativeCell to positiveCell.
  signedLiquidFine3:f32,
  signedBulkFine3:f32,
}

// A sufficient no-overlap condition for an entire cell's same-axis donor
// prisms. The maxima are over ALL outward subfaces on each of its two sides.
// Same-side tangential rectangles must already be an audited disjoint partition.
// Checking each face independently is insufficient when both sides flow out.
fn geometricSplitDonorCFL(widthFine:f32,negativeOutwardSpeed:f32,
  positiveOutwardSpeed:f32,dtSeconds:f32)->u32{
  if(!(widthFine>0.0)||!(negativeOutwardSpeed>=0.0)
    ||!(positiveOutwardSpeed>=0.0)||!(dtSeconds>=0.0)){
    return GEOMETRIC_SPLIT_CFL_ERROR;
  }
  return select(GEOMETRIC_SPLIT_CFL_ERROR,0u,
    dtSeconds*(negativeOutwardSpeed+positiveOutwardSpeed)<=widthFine);
}

// A partial-cell plane must have been reconstructed from the current directional
// sweep's amount, not from the amount before the first axis or a CM12 density.
// donorCapacityFine3 must equal the full rectangular box volume exactly.
fn geometricSplitSubfaceFlux(face:GeometricSubface,rowVelocityFinePerSecond:f32,
  dtSeconds:f32,donorCenterFine:vec3f,donorWidthsFine:vec3f,
  donorAmountFine3:f32,donorCapacityFine3:f32,
  donorPlane:GeometricInterfacePlane,amountToleranceFine3:f32)->GeometricSplitFlux{
  var flux:GeometricSplitFlux;
  flux.negativeCell=face.negativeCell;flux.positiveCell=face.positiveCell;
  if(face.status!=1u||face.axis>2u||!(face.areaFine2>0.0)
    ||!all(donorWidthsFine>vec3f(0.0))||!(dtSeconds>=0.0)
    ||!(amountToleranceFine3>=0.0)){
    flux.errors=GEOMETRIC_SPLIT_GEOMETRY_ERROR;return flux;
  }
  let boxVolume=donorWidthsFine.x*donorWidthsFine.y*donorWidthsFine.z;
  if(!(donorCapacityFine3==boxVolume)){
    flux.errors=GEOMETRIC_SPLIT_CUT_CELL_ERROR;return flux;
  }
  if(!(donorAmountFine3>=0.0&&donorAmountFine3<=donorCapacityFine3)){
    flux.errors=GEOMETRIC_SPLIT_AMOUNT_ERROR;return flux;
  }
  let travel=abs(rowVelocityFinePerSecond)*dtSeconds;
  if(!(travel<=donorWidthsFine[face.axis])){
    flux.errors=GEOMETRIC_SPLIT_CFL_ERROR;return flux;
  }
  if(travel==0.0){return flux;}
  // Work in donor-local coordinates so a small swept distance is not added
  // to a large world position and rounded away before geometric integration.
  var prismMinimum=face.minimumFine-donorCenterFine;
  var prismMaximum=face.maximumFine-donorCenterFine;
  if(rowVelocityFinePerSecond>0.0){prismMinimum[face.axis]-=travel;}
  else{prismMaximum[face.axis]+=travel;}
  let donorMinimum=-0.5*donorWidthsFine;
  let donorMaximum=0.5*donorWidthsFine;
  // The caller passes the upwind cell. Do not clip a bad prism into its box:
  // that would hide a donor-selection/geometry/CFL error by reducing the flux.
  if(!all(prismMinimum>=donorMinimum)||!all(prismMaximum<=donorMaximum)){
    flux.errors=GEOMETRIC_SPLIT_GEOMETRY_ERROR;return flux;
  }
  var prismWidths=prismMaximum-prismMinimum;
  prismWidths[face.axis]=travel;
  let bulk=face.areaFine2*travel;
  var liquid=0.0;
  if(donorAmountFine3==donorCapacityFine3){liquid=bulk;}
  else if(donorAmountFine3>0.0){
    if(!geometricInterfacePlaneValid(donorPlane)){
      flux.errors=GEOMETRIC_SPLIT_PLANE_ERROR;return flux;
    }
    let reconstructedAmount=boxVolume*geometricPlaneBoxFraction(
      donorPlane.normal,donorPlane.offset,donorWidthsFine);
    if(!(abs(reconstructedAmount-donorAmountFine3)<=amountToleranceFine3)){
      flux.errors=GEOMETRIC_SPLIT_PLANE_ERROR;return flux;
    }
    var relativeCenter=0.5*(prismMinimum+prismMaximum);
    relativeCenter[face.axis]=face.minimumFine[face.axis]-donorCenterFine[face.axis]
      +select(0.5*travel,-0.5*travel,rowVelocityFinePerSecond>0.0);
    let prismOffset=donorPlane.offset-dot(donorPlane.normal,relativeCenter);
    liquid=bulk*geometricPlaneBoxFraction(donorPlane.normal,prismOffset,prismWidths);
  }
  let orientation=select(-1.0,1.0,rowVelocityFinePerSecond>0.0);
  flux.signedLiquidFine3=orientation*liquid;
  flux.signedBulkFine3=orientation*bulk;
  return flux;
}

// Frozen once per complete three-axis cycle, including each synchronized
// subcycle. A tie uses the liquid side consistently. No input clamping.
fn geometricSplitFrozenCompression(amountFine3:f32,capacityFine3:f32)->f32{
  return select(0.0,1.0,amountFine3>=0.5*capacityFine3);
}

struct GeometricSplitGather {
  errors:u32,
  conservativeAmountFine3:f32,
  compressionFine3:f32,
  amountFine3:f32,
  lowerViolationFine3:f32,
  upperViolationFine3:f32,
}

// Gather shared face records with their stored sign; do not independently
// recompute the liquid flux from the receiving cell's reconstruction.
// All incoming/outgoing quantities below are nonnegative magnitudes.
fn geometricSplitGatherCell(amountFine3:f32,capacityFine3:f32,
  incomingLiquidFine3:f32,outgoingLiquidFine3:f32,
  incomingBulkFine3:f32,outgoingBulkFine3:f32,
  frozenCompression:f32,amountToleranceFine3:f32)->GeometricSplitGather{
  var result:GeometricSplitGather;
  if(!(capacityFine3>0.0)||!(amountFine3>=0.0&&amountFine3<=capacityFine3)
    ||!(amountToleranceFine3>=0.0)
    ||!(frozenCompression==0.0||frozenCompression==1.0)){
    result.errors=GEOMETRIC_SPLIT_AMOUNT_ERROR;return result;
  }
  if(!(incomingLiquidFine3>=0.0&&outgoingLiquidFine3>=0.0
    &&incomingBulkFine3>=0.0&&outgoingBulkFine3>=0.0)
    ||incomingLiquidFine3>incomingBulkFine3+amountToleranceFine3
    ||outgoingLiquidFine3>outgoingBulkFine3+amountToleranceFine3){
    result.errors=GEOMETRIC_SPLIT_BULK_ERROR;return result;
  }
  result.conservativeAmountFine3=amountFine3+incomingLiquidFine3-outgoingLiquidFine3;
  result.compressionFine3=frozenCompression*(outgoingBulkFine3-incomingBulkFine3);
  result.amountFine3=result.conservativeAmountFine3+result.compressionFine3;
  result.lowerViolationFine3=max(0.0,-result.amountFine3);
  result.upperViolationFine3=max(0.0,result.amountFine3-capacityFine3);
  if(!(result.amountFine3>=-amountToleranceFine3
    &&result.amountFine3<=capacityFine3+amountToleranceFine3)){
    result.errors=GEOMETRIC_SPLIT_BOUNDS_ERROR;
  }
  // The reported amount remains unchanged, including roundoff excursions.
  // Tolerances are receipts, never permission to clamp conserved state.
  return result;
}

struct GeometricSplitCycleAudit {
  errors:u32,
  compressionDefectFine3:f32,
  divergenceResidualFine3:f32,
}

// Supply the sum of outward-minus-inward BULK swept volume for the cell across
// all three axes. Frozen c makes the cycle's compression defect exactly c*sum.
// This must be audited per cell; a global cancellation can conceal local loss.
fn geometricSplitAuditCycle(frozenCompression:f32,
  sumDirectionalBulkDivergenceFine3:f32,residualToleranceFine3:f32)
  ->GeometricSplitCycleAudit{
  var result:GeometricSplitCycleAudit;
  result.divergenceResidualFine3=sumDirectionalBulkDivergenceFine3;
  result.compressionDefectFine3=frozenCompression*sumDirectionalBulkDivergenceFine3;
  if(!(residualToleranceFine3>=0.0)
    ||!(frozenCompression==0.0||frozenCompression==1.0)
    ||!(abs(sumDirectionalBulkDivergenceFine3)<=residualToleranceFine3)){
    result.errors=GEOMETRIC_SPLIT_DIVERGENCE_ERROR;
  }
  return result;
}
`;
