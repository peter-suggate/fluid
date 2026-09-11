/**
 * Rectangular intersections of the accepted composite row's opposing cells.
 *
 * Bindings are supplied by the resident: rowTermRange, termCell,
 * termCoefficient, rowAxis, rowArea, rowStaticArea, rowDistance, cellCenter,
 * and cellWidths. All geometry and velocities use finest-lattice units.
 * This module does not allocate resources or introduce an execution stage.
 *
 * Enumerate each row exactly once, nesting negativeTerm and positiveTerm over
 * rowTermRange(row). Only status == 1 descriptors may contribute. A row must
 * pass geometricSubfaceRowAudit before geometric transport may consume it;
 * malformed or unsupported rows require an explicit failure, never CM12 fallback.
 * The audit is preparation/diagnostic work, not a per-flux hot-loop operation.
 */
export function createGeometricSubfacesWGSL(): string {
  return /* wgsl */ `
struct GeometricSubface {
  // 0: no opposite-side intersection, 1: geometric rectangle, 2: malformed.
  status:u32,
  row:u32,
  negativeTerm:u32,
  positiveTerm:u32,
  negativeCell:u32,
  positiveCell:u32,
  axis:u32,
  areaFine2:f32,
  minimumFine:vec3f,
  maximumFine:vec3f,
  distanceFine:f32,
}

// Ordering is by coefficient sign. No coefficient products determine area.
// Accepted dyadic cells have exactly representable integer/half-integer bounds;
// exact normal contact rejects gaps and overlaps rather than fabricating a face.
fn geometricSubface(row:u32,negativeTerm:u32,positiveTerm:u32)->GeometricSubface{
  var face:GeometricSubface;
  face.row=row;face.negativeTerm=negativeTerm;face.positiveTerm=positiveTerm;
  let range=rowTermRange(row);
  if(negativeTerm<range.x||negativeTerm>=range.y
    ||positiveTerm<range.x||positiveTerm>=range.y){face.status=2u;return face;}
  let negativeCoefficient=termCoefficient(negativeTerm);
  let positiveCoefficient=termCoefficient(positiveTerm);
  if(!(negativeCoefficient<0.0)||!(positiveCoefficient>0.0)){return face;}
  let negative=termCell(negativeTerm);let positive=termCell(positiveTerm);
  let axis=rowAxis(row);
  if(axis>2u||negative==positive){face.status=2u;return face;}
  // Duplicated cells would duplicate physical intersections. Reject them even
  // if each duplicate's coefficient was divided to preserve an algebraic sum.
  for(var term=range.x;term<range.y;term+=1u){
    let cell=termCell(term);
    if((term!=negativeTerm&&cell==negative)
      ||(term!=positiveTerm&&cell==positive)){face.status=2u;return face;}
  }
  let negativeCenter=cellCenter(negative);let positiveCenter=cellCenter(positive);
  let negativeWidths=cellWidths(negative);let positiveWidths=cellWidths(positive);
  if(!all(negativeWidths>vec3f(0.0))||!all(positiveWidths>vec3f(0.0))){
    face.status=2u;return face;
  }
  let negativeMinimum=negativeCenter-0.5*negativeWidths;
  let negativeMaximum=negativeCenter+0.5*negativeWidths;
  let positiveMinimum=positiveCenter-0.5*positiveWidths;
  let positiveMaximum=positiveCenter+0.5*positiveWidths;
  if(negativeMaximum[axis]!=positiveMinimum[axis]){face.status=2u;return face;}
  let tangent0=(axis+1u)%3u;let tangent1=(axis+2u)%3u;
  var minimum=max(negativeMinimum,positiveMinimum);
  var maximum=min(negativeMaximum,positiveMaximum);
  let width0=maximum[tangent0]-minimum[tangent0];
  let width1=maximum[tangent1]-minimum[tangent1];
  if(!(width0>0.0)||!(width1>0.0)){return face;}
  let distance=positiveCenter[axis]-negativeCenter[axis];
  if(!(distance>0.0)){face.status=2u;return face;}
  minimum[axis]=negativeMaximum[axis];maximum[axis]=negativeMaximum[axis];
  face.status=1u;face.negativeCell=negative;face.positiveCell=positive;face.axis=axis;
  face.minimumFine=minimum;face.maximumFine=maximum;
  face.areaFine2=width0*width1;face.distanceFine=distance;
  return face;
}

struct GeometricSubfaceRowAudit {
  // valid == 1 certifies a two-sided, uncut geometric row only.
  valid:u32,
  subfaceCount:u32,
  malformedPairs:u32,
  geometryAreaFine2:f32,
  staticAreaFine2:f32,
  effectiveAreaFine2:f32,
  maximumMarginalErrorFine2:f32,
}

// In an uncut row, sum_j A_ij = abs(c_i)*A_row*d_row.
// Therefore one row velocity produces exactly the existing pressure divergence
// incidence (apart from floating-point summation order). Comparing *marginals*
// catches geometrically false Cartesian products even when their total agrees.
fn geometricSubfaceRowAudit(row:u32)->GeometricSubfaceRowAudit{
  var audit:GeometricSubfaceRowAudit;
  let range=rowTermRange(row);let area=rowStaticArea(row);let distance=rowDistance(row);
  audit.staticAreaFine2=area;audit.effectiveAreaFine2=rowArea(row);
  if(!(area>0.0)||!(distance>0.0)||rowAxis(row)>2u){return audit;}
  var negativeCount=0u;var positiveCount=0u;var malformedTerms=0u;
  for(var own=range.x;own<range.y;own+=1u){
    let coefficient=termCoefficient(own);
    if(coefficient<0.0){negativeCount+=1u;}
    else if(coefficient>0.0){positiveCount+=1u;}
    else{malformedTerms+=1u;continue;}
    var marginal=0.0;
    for(var other=range.x;other<range.y;other+=1u){
      let negative=select(other,own,coefficient<0.0);
      let positive=select(own,other,coefficient<0.0);
      let face=geometricSubface(row,negative,positive);
      if(face.status==2u){audit.malformedPairs+=1u;}
      if(face.status!=1u){continue;}
      marginal+=face.areaFine2;
      if(coefficient<0.0){
        audit.subfaceCount+=1u;audit.geometryAreaFine2+=face.areaFine2;
      }
    }
    let expected=abs(coefficient)*area*distance;
    audit.maximumMarginalErrorFine2=max(audit.maximumMarginalErrorFine2,
      abs(marginal-expected));
  }
  // This tolerance only audits f32 coefficient reconstruction; geometry is
  // never rescaled or repaired to make the pressure coefficients fit.
  let tolerance=8.0*1.1920928955078125e-7*area;
  audit.valid=select(0u,1u,negativeCount>0u&&positiveCount>0u
    &&malformedTerms==0u&&audit.malformedPairs==0u
    &&abs(audit.geometryAreaFine2-area)<=tolerance
    &&audit.maximumMarginalErrorFine2<=tolerance
    &&audit.effectiveAreaFine2==area);
  return audit;
}

// Sum this contribution over the row's unique geometric subfaces to recover
// sum_i c_i*p_i for an audited uncut row. The *row* distance is intentional:
// replacing it with each pair's center distance would change the pressure DOF.
fn geometricSubfacePressureGradient(face:GeometricSubface,
  negativePressure:f32,positivePressure:f32)->f32{
  if(face.status!=1u){return 0.0;}
  return face.areaFine2*(positivePressure-negativePressure)
    /(rowStaticArea(face.row)*rowDistance(face.row));
}

// Positive row velocity carries volume from negativeCell to positiveCell.
// Scatter -q to the negative cell and +q to the positive cell exactly once.
// Multiply q by finestCellSize_m^3 only when converting to physical volume.
fn geometricSubfaceSweptVolumeFine3(face:GeometricSubface,
  rowVelocityFinePerSecond:f32,dtSeconds:f32)->f32{
  if(face.status!=1u){return 0.0;}
  return face.areaFine2*rowVelocityFinePerSecond*dtSeconds;
}
`;
}
