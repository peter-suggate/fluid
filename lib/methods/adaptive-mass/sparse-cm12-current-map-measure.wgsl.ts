/** Fine-support amounts of the same spatial density used by point queries.
 *
 * The two banks each contain x-fastest vec4(mean density, mean momentum).
 * Momentum is in fine-cell units per second. Native cells only sum these
 * amounts; neither old native means nor an interface reconstruction enters
 * the quadrature.
 */
export interface SparseCM12CurrentMapMeasureLayout {
  readonly baseWords: number;
  readonly dimensions: readonly [number, number, number];
  readonly absoluteTolerance?: number;
  readonly relativeTolerance?: number;
  /** Default 3. Explicit overrides through 5 retain the same local error
   * budgets while increasing the bounded adaptive work and stack capacity. */
  readonly maximumRefinementDepth?: number;
  /** Optional proof hook (q:vec3i)->bool. It must establish that both density
   * and transported momentum equal the accepted measure throughout this
   * support. Equality at a few point samples does not establish that fact. */
  readonly unchangedSupportHook?: string;
  /** Six words per fine support, produced after candidate filtering and before
   * quadrature. Optional so standalone scalar/proof probes remain unchanged. */
  readonly rangeCacheBaseWords?: number;
}

/** Gauss rules on [0,1]. Exported for independent numerical QA. */
export const SPARSE_CM12_CURRENT_MAP_GAUSS_2 = Object.freeze({
  nodes: [0.21132486540518713, 0.7886751345948129] as const,
  weights: [0.5, 0.5] as const,
});
export const SPARSE_CM12_CURRENT_MAP_GAUSS_4 = Object.freeze({
  nodes: [0.06943184420297371, 0.33000947820757187,
    0.6699905217924281, 0.9305681557970262] as const,
  weights: [0.17392742256872693, 0.32607257743127307,
    0.32607257743127307, 0.17392742256872693] as const,
});

export function createSparseCM12CurrentMapMeasureWGSL(
  layout: SparseCM12CurrentMapMeasureLayout,
): string {
  const absoluteTolerance = layout.absoluteTolerance ?? 1e-4;
  const relativeTolerance = layout.relativeTolerance ?? 1e-4;
  const maximumDepth = layout.maximumRefinementDepth ?? 3;
  const count = layout.dimensions.reduce((a, b) => a * b, 1);
  if (!Number.isInteger(layout.baseWords) || layout.baseWords < 0
    || layout.dimensions.length !== 3
    || layout.dimensions.some(n => !Number.isInteger(n) || n < 1)
    || !Number.isSafeInteger(count) || layout.baseWords + 8 * count > 0xffffffff) {
    throw new RangeError("Invalid current-map fine measure storage");
  }
  if (!Number.isFinite(absoluteTolerance) || absoluteTolerance <= 0
    || !Number.isFinite(relativeTolerance) || relativeTolerance < 0
    || !Number.isInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > 5) {
    throw new RangeError("Invalid current-map measure quadrature controls");
  }
  const unchangedHook = layout.unchangedSupportHook;
  const rangeCache = layout.rangeCacheBaseWords;
  if (rangeCache !== undefined && (!Number.isSafeInteger(rangeCache)
    || rangeCache < 0 || rangeCache + 6 * count > 0xffffffff)) {
    throw new RangeError("Invalid current-map measure range cache");
  }
  if (unchangedHook !== undefined && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(unchangedHook)) {
    throw new TypeError("unchangedSupportHook must be a WGSL identifier");
  }
  const floats = (values: readonly number[]) => values.map(n => `${n}`).join(",");
  return /* wgsl */ `
const CM12_CURRENT_MAP_MEASURE_BASE:u32=${layout.baseWords}u;
const CM12_CURRENT_MAP_MEASURE_DIMS:vec3i=vec3i(${layout.dimensions.join(",")});
const CM12_CURRENT_MAP_MEASURE_COUNT:u32=${count}u;
const CM12_CURRENT_MAP_MEASURE_ABS_TOL:f32=${absoluteTolerance};
const CM12_CURRENT_MAP_MEASURE_REL_TOL:f32=${relativeTolerance};
const CM12_CURRENT_MAP_MEASURE_MAX_DEPTH:u32=${maximumDepth}u;
${rangeCache === undefined ? "" : `const CM12_CURRENT_MAP_MEASURE_RANGE_BASE:u32=${rangeCache}u;
@compute @workgroup_size(64)
fn compileCurrentMapFineMeasureRanges(@builtin(global_invocation_id)gid:vec3u){
  let ordinal=gid.x;if(ordinal>=CM12_CURRENT_MAP_MEASURE_COUNT||cm12CurrentMapFailed()){return;}
  let nx=u32(CM12_CURRENT_MAP_MEASURE_DIMS.x);let ny=u32(CM12_CURRENT_MAP_MEASURE_DIMS.y);
  let q=vec3i(i32(ordinal%nx),i32((ordinal/nx)%ny),i32(ordinal/(nx*ny)));
  let range=cm12CurrentMapRangeOnFineSupport(q,cm12CurrentMapCandidateBank());
  let at=CM12_CURRENT_MAP_MEASURE_RANGE_BASE+6u*ordinal;
  for(var axis=0u;axis<3u;axis++){
    state[at+axis]=range[0][axis];state[at+3u+axis]=range[1][axis];
  }
}
fn cm12CurrentMapCachedFineMeasureRange(ordinal:u32)->mat2x3f{
  let at=CM12_CURRENT_MAP_MEASURE_RANGE_BASE+6u*ordinal;
  return mat2x3f(vec3f(state[at],state[at+1u],state[at+2u]),
    vec3f(state[at+3u],state[at+4u],state[at+5u]));
}`}
const CM12_CURRENT_MAP_GAUSS_2_X:array<f32,2>=array<f32,2>(${floats(SPARSE_CM12_CURRENT_MAP_GAUSS_2.nodes)});
const CM12_CURRENT_MAP_GAUSS_2_W:array<f32,2>=array<f32,2>(${floats(SPARSE_CM12_CURRENT_MAP_GAUSS_2.weights)});
const CM12_CURRENT_MAP_GAUSS_4_X:array<f32,4>=array<f32,4>(${floats(SPARSE_CM12_CURRENT_MAP_GAUSS_4.nodes)});
const CM12_CURRENT_MAP_GAUSS_4_W:array<f32,4>=array<f32,4>(${floats(SPARSE_CM12_CURRENT_MAP_GAUSS_4.weights)});

struct CurrentMapPointDensity{
  density:f32,
  seedPhi:f32,
  jacobian:f32,
}
struct CurrentMapMeasureRuleResult{
  value:vec4f,
  evaluations:u32,
}
struct CurrentMapMeasureEstimate{
  value:vec4f,
  error:vec4f,
  evaluations:u32,
  unresolved:u32,
}

fn cm12CurrentMapMeasureInside(q:vec3i)->bool{
  return all(q>=vec3i(0))&&all(q<CM12_CURRENT_MAP_MEASURE_DIMS);
}
fn cm12CurrentMapMeasureOrdinal(q:vec3i)->u32{
  return u32(q.x+CM12_CURRENT_MAP_MEASURE_DIMS.x*
    (q.y+CM12_CURRENT_MAP_MEASURE_DIMS.y*q.z));
}
fn cm12CurrentMapMeasureAddress(ordinal:u32,bank:u32)->u32{
  return CM12_CURRENT_MAP_MEASURE_BASE+4u*(ordinal+bank*CM12_CURRENT_MAP_MEASURE_COUNT);
}
fn cm12CurrentMapFineMeasure(q:vec3i,bank:u32)->vec4f{
  if(!cm12CurrentMapMeasureInside(q)){return vec4f(0.0);}
  let at=cm12CurrentMapMeasureAddress(cm12CurrentMapMeasureOrdinal(q),bank);
  return vec4f(state[at],state[at+1u],state[at+2u],state[at+3u]);
}
fn cm12CurrentMapWriteFineMeasure(ordinal:u32,bank:u32,value:vec4f){
  let at=cm12CurrentMapMeasureAddress(ordinal,bank);
  state[at]=value.x;state[at+1u]=value.y;
  state[at+2u]=value.z;state[at+3u]=value.w;
}
fn cm12CurrentMapPointOpen(point:vec3f)->bool{
  if(any(point<vec3f(0.0))||any(point>vec3f(CM12_CURRENT_MAP_MEASURE_DIMS))){return false;}
  let solid=cm12SolidVoxelFractionQ8(vec3i(floor(point)));
  return solid<255u&&fract(point.y)>=f32(solid)/255.0
    &&cm12RetainedDensityRigidPointOpen(point);
}
fn cm12CurrentMapSeedRangeIsDry(lowerFine:vec3f,upperFine:vec3f)->bool{
  let origin=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+12u);
  let lower=origin+p.frame.y*lowerFine;let upper=origin+p.frame.y*upperFine;
  let domainLower=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+4u);
  let domainUpper=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+8u);
  let supportDimensions=vec3f(state[CM12_RETAINED_FIELD_BASE+7u],
    state[CM12_RETAINED_FIELD_BASE+11u],state[CM12_RETAINED_FIELD_BASE+15u]);
  let supportUpper=select(domainUpper,domainLower+supportDimensions*p.frame.y,
    all(supportDimensions>vec3f(0.0)));
  let width=state[CM12_RETAINED_FIELD_BASE+3u];
  let magnitude=max(vec3f(1.0),max(abs(lower),abs(upper)));
  let margin=1e-5*max(magnitude.x,max(magnitude.y,magnitude.z));
  if(any(upper<domainLower-vec3f(margin))||any(lower>supportUpper+vec3f(margin))){return true;}
  let count=u32(state[CM12_RETAINED_FIELD_BASE+1u]);
  for(var primitive=0u;primitive<count;primitive+=1u){
    let at=CM12_RETAINED_FIELD_BASE+16u+16u*primitive;
    let kind=u32(state[at]);let a=cm12RetainedDensityVector(at+4u);
    let b=cm12RetainedDensityVector(at+8u);var minimum=-1e6;
    if(kind==1u){
      for(var axis=0u;axis<3u;axis+=1u){
        if(a[axis]>domainLower[axis]){minimum=max(minimum,a[axis]-upper[axis]);}
        if(b[axis]<domainUpper[axis]){minimum=max(minimum,lower[axis]-b[axis]);}
      }
    }else if(kind==2u){
      let closest=clamp(a,lower,upper);let delta=(closest-a)/b;
      minimum=0.5*min(b.x,min(b.y,b.z))*(dot(delta,delta)-1.0);
    }else if(kind==3u){
      let lo=lower-a;let hi=upper-a;
      let squareMaximum=max(lo*lo,hi*hi);
      let nearest=max(max(lo,-hi),vec3f(0.0));let squareMinimum=nearest*nearest;
      let quadraticMaximum=max(b*squareMinimum,b*squareMaximum);
      minimum=lo.y-quadraticMaximum.x-quadraticMaximum.z;
    }else{return false;}
    // Every authored branch must be dry over the entire mapped AABB. A
    // zero quadrature sample alone cannot justify skipping a support.
    if(minimum<0.5*width+margin){return false;}
  }
  return true;
}
fn cm12CurrentMapSeedRangeIsSaturated(lowerFine:vec3f,upperFine:vec3f)->bool{
  let origin=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+12u);
  let lower=origin+p.frame.y*lowerFine;let upper=origin+p.frame.y*upperFine;
  let domainLower=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+4u);
  let domainUpper=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+8u);
  let supportDimensions=vec3f(state[CM12_RETAINED_FIELD_BASE+7u],
    state[CM12_RETAINED_FIELD_BASE+11u],state[CM12_RETAINED_FIELD_BASE+15u]);
  let supportUpper=select(domainUpper,domainLower+supportDimensions*p.frame.y,
    all(supportDimensions>vec3f(0.0)));
  let magnitude=max(vec3f(1.0),max(abs(lower),abs(upper)));
  let margin=1e-5*max(magnitude.x,max(magnitude.y,magnitude.z));
  if(any(lower<domainLower+vec3f(margin))||any(upper>supportUpper-vec3f(margin))){return false;}
  let width=state[CM12_RETAINED_FIELD_BASE+3u];
  let count=u32(state[CM12_RETAINED_FIELD_BASE+1u]);
  for(var primitive=0u;primitive<count;primitive+=1u){
    let at=CM12_RETAINED_FIELD_BASE+16u+16u*primitive;
    let kind=u32(state[at]);let a=cm12RetainedDensityVector(at+4u);
    let b=cm12RetainedDensityVector(at+8u);var maximum=1e6;
    if(kind==1u){
      maximum=-1e6;
      for(var axis=0u;axis<3u;axis+=1u){
        if(a[axis]>domainLower[axis]){maximum=max(maximum,a[axis]-lower[axis]);}
        if(b[axis]<domainUpper[axis]){maximum=max(maximum,upper[axis]-b[axis]);}
      }
    }else if(kind==2u){
      let lo=(lower-a)/b;let hi=(upper-a)/b;let squareMaximum=max(lo*lo,hi*hi);
      maximum=0.5*min(b.x,min(b.y,b.z))*(squareMaximum.x+squareMaximum.y+squareMaximum.z-1.0);
    }else if(kind==3u){
      let lo=lower-a;let hi=upper-a;let squareMaximum=max(lo*lo,hi*hi);
      let nearest=max(max(lo,-hi),vec3f(0.0));let squareMinimum=nearest*nearest;
      let quadraticMinimum=min(b*squareMinimum,b*squareMaximum);
      maximum=hi.y-quadraticMinimum.x-quadraticMinimum.z;
    }
    if(maximum<=-0.5*width-margin){return true;}
  }
  return false;
}
fn cm12CurrentMapSeedPhiAtFine(point:vec3f,bank:u32)->f32{
  let width=state[CM12_RETAINED_FIELD_BASE+3u];
  if(!cm12CurrentMapPointOpen(point)){return width;}
  let mapped=cm12CurrentMapEvaluatePoint(point,bank);
  let origin=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+12u);
  return cm12RetainedDensityPhiMetres(origin+mapped*p.frame.y);
}
fn cm12CurrentMapPointDensity(point:vec3f,bank:u32)->CurrentMapPointDensity{
  let width=state[CM12_RETAINED_FIELD_BASE+3u];
  if(!cm12CurrentMapPointOpen(point)){
    return CurrentMapPointDensity(0.0,width,1.0);
  }
  let mapped=cm12CurrentMapEvaluate(point,bank);
  let jacobian=determinant(mapped.jacobian);
  let origin=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+12u);
  let phi=cm12RetainedDensityPhiMetres(origin+mapped.point*p.frame.y);
  let seed=clamp(0.5-phi/width,0.0,1.0);
  // The determinant multiplies the measure. Clamping this product to one
  // would discard compression and break the change-of-variables identity.
  return CurrentMapPointDensity(seed*jacobian,phi,jacobian);
}
fn cm12CurrentMapDensityAtFine(point:vec3f,bank:u32)->f32{
  return cm12CurrentMapPointDensity(point,bank).density;
}
fn cm12CurrentMapPhiAtFine(point:vec3f,bank:u32)->f32{
  let sample=cm12CurrentMapPointDensity(point,bank);
  let width=state[CM12_RETAINED_FIELD_BASE+3u];
  // For a threshold strictly inside the seed ramp, this analytic companion
  // has exactly the density's zero and sign. It retains the seed curvature
  // without subtracting nearly equal clamped densities at the interface.
  // Outside that range use the actual density residual: in particular J<.5
  // has no half-density surface, even in fully saturated seed material.
  if(sample.jacobian>0.5){
    return sample.seedPhi-width*(0.5-0.5/sample.jacobian);
  }
  return width*(0.5-sample.density);
}
fn cm12CurrentMapMeasureSample(point:vec3f,bank:u32)->vec4f{
  let density=cm12CurrentMapDensityAtFine(point,bank);
  if(density==0.0){return vec4f(0.0);}
  // Density composes the accepted map with this certified current increment.
  // Momentum must use that same increment, including its spatial numerical
  // approximation, rather than independently retracing the raw RK2 path.
  let departure=cm12CurrentMapEvaluateIncrement(point,bank).point;
  let velocity=cm12CurrentMapVelocity(departure);
  return vec4f(density,density*velocity);
}
fn cm12CurrentMapMeasureOrder()->u32{
  // Cell size is a positive runtime uniform. Keeping this loop limit
  // runtime-valued prevents Metal from expanding every nested quadrature,
  // map interpolation, and velocity sample into one enormous expression.
  return select(0u,4u,p.frame.y>0.0);
}
fn cm12CurrentMapMeasureIntegrationAxis(lower:vec3f,bank:u32)->u32{
  let center=lower+vec3f(0.5);var selected=1u;var greatestVariation=-1.0;
  for(var axis=0u;axis<3u;axis+=1u){
    var lo=center;var hi=center;lo[axis]-=0.5;hi[axis]+=0.5;
    let variation=abs(cm12CurrentMapSeedPhiAtFine(hi,bank)
      -cm12CurrentMapSeedPhiAtFine(lo,bank));
    if(variation>greatestVariation){greatestVariation=variation;selected=axis;}
  }
  return selected;
}
fn cm12CurrentMapMeasureLine(lower:vec3f,span:f32,bank:u32,splitSaturation:bool,
 integrationAxis:u32)->CurrentMapMeasureRuleResult{
  // Isolate both seed saturation boundaries along the direction of greatest
  // mapped seed variation. A fixed y axis leaves x/z-normal caps as sharp
  // outer-integral transitions even when their physical motion is uniform.
  // This is quadrature of q_seed(X)*det(DX), not a substitute geometry.
  // A bounded bracket scan can miss a tangency: the outer error estimator
  // and independent global mass checks are therefore still required.
  let width=state[CM12_RETAINED_FIELD_BASE+3u];let order=cm12CurrentMapMeasureOrder();
  var direction=vec3f(0.0);direction[integrationAxis]=span;
  if(!splitSaturation){
    var value=vec4f(0.0);
    for(var y=0u;y<order;y+=1u){
      value+=CM12_CURRENT_MAP_GAUSS_4_W[y]*cm12CurrentMapMeasureSample(
        lower+direction*CM12_CURRENT_MAP_GAUSS_4_X[y],bank);
    }
    return CurrentMapMeasureRuleResult(value,order);
  }
  var roots:array<f32,18>;roots[0]=0.0;var count=1u;
  var evaluations=1u;
  var previous=cm12CurrentMapSeedPhiAtFine(lower,bank);
  for(var interval=0u;interval<2u*order;interval+=1u){
    let end=f32(interval+1u)*0.125;
    let current=cm12CurrentMapSeedPhiAtFine(lower+direction*end,bank);
    evaluations+=1u;
    var found:array<f32,2>;var foundCount=0u;
    for(var side=0u;side<2u;side+=1u){
      let threshold=select(-0.5*width,0.5*width,side==1u);
      if((previous<threshold)==(current<threshold)){continue;}
      var left=f32(interval)*0.125;var right=end;
      for(var iteration=0u;iteration<3u*order;iteration+=1u){
        let middle=0.5*(left+right);
        let phi=cm12CurrentMapSeedPhiAtFine(lower+direction*middle,bank);
        evaluations+=1u;
        if((phi<threshold)==(previous<threshold)){left=middle;}else{right=middle;}
      }
      found[foundCount]=0.5*(left+right);foundCount+=1u;
    }
    if(foundCount==2u&&found[0]>found[1]){let swap=found[0];found[0]=found[1];found[1]=swap;}
    for(var i=0u;i<foundCount;i+=1u){roots[count]=found[i];count+=1u;}
    previous=current;
  }
  roots[count]=1.0;count+=1u;
  var value=vec4f(0.0);
  for(var segment=0u;segment+1u<count;segment+=1u){
    let length=roots[segment+1u]-roots[segment];var integral=vec4f(0.0);
    for(var y=0u;y<order;y+=1u){
      let t=roots[segment]+length*CM12_CURRENT_MAP_GAUSS_4_X[y];
      integral+=CM12_CURRENT_MAP_GAUSS_4_W[y]*
        cm12CurrentMapMeasureSample(lower+direction*t,bank);
      evaluations+=1u;
    }
    value+=length*integral;
  }
  return CurrentMapMeasureRuleResult(value,evaluations);
}
fn cm12CurrentMapMeasureRule(lower:vec3f,span:f32,bank:u32,splitSaturation:bool,
 integrationAxis:u32)->CurrentMapMeasureRuleResult{
  var total=vec4f(0.0);var evaluations=0u;let order=cm12CurrentMapMeasureOrder();
  for(var z=0u;z<order;z+=1u){var row=vec4f(0.0);
    for(var x=0u;x<order;x+=1u){
      var lineLower=lower;
      lineLower[(integrationAxis+1u)%3u]+=span*CM12_CURRENT_MAP_GAUSS_4_X[x];
      lineLower[(integrationAxis+2u)%3u]+=span*CM12_CURRENT_MAP_GAUSS_4_X[z];
      let line=cm12CurrentMapMeasureLine(lineLower,span,bank,splitSaturation,integrationAxis);
      row+=CM12_CURRENT_MAP_GAUSS_4_W[x]*line.value;evaluations+=line.evaluations;
    }total+=CM12_CURRENT_MAP_GAUSS_4_W[z]*row;
  }
  return CurrentMapMeasureRuleResult(total*(span*span*span),evaluations);
}
fn cm12CurrentMapMeasureSupportAxis(lower:vec3f,bank:u32,splitSaturation:bool,
 integrationAxis:u32)->CurrentMapMeasureEstimate{
  // An explicit bounded depth-first stack avoids recursion and allocates
  // only 1+7*depth boxes. Each estimate compares a 4^3 rule with eight
  // child 4^3 rules. Child values are reused if refinement is necessary.
  // The difference is an error ESTIMATE, not an interval enclosure:
  // independent global mass and coverage QA remain necessary.
  var stack:array<vec4f,${1 + 7 * maximumDepth}>;
  var coarse:array<vec4f,${1 + 7 * maximumDepth}>;
  var depths:array<u32,${1 + 7 * maximumDepth}>;
  stack[0]=vec4f(lower,1.0);depths[0]=0u;
  let initial=cm12CurrentMapMeasureRule(lower,1.0,bank,splitSaturation,integrationAxis);coarse[0]=initial.value;
  var pending=1u;
  var result=CurrentMapMeasureEstimate(vec4f(0.0),vec4f(0.0),initial.evaluations,0u);
  for(var visited=0u;visited<${(8 ** (maximumDepth + 1) - 1) / 7}u;visited+=1u){
    if(pending==0u){break;}
    pending-=1u;let box=stack[pending];let depth=depths[pending];
    let low=coarse[pending];let half=0.5*box.w;
    var children:array<vec4f,8>;var high=vec4f(0.0);
    for(var child=0u;child<8u;child+=1u){
      let offset=vec3f(f32(child&1u),f32((child>>1u)&1u),f32(child>>2u));
      let childIntegral=cm12CurrentMapMeasureRule(box.xyz+half*offset,half,bank,splitSaturation,integrationAxis);
      children[child]=childIntegral.value;high+=childIntegral.value;
      result.evaluations+=childIntegral.evaluations;
    }
    let error=abs(high-low);
    let tolerance=vec4f(CM12_CURRENT_MAP_MEASURE_ABS_TOL*box.w*box.w*box.w)
      +CM12_CURRENT_MAP_MEASURE_REL_TOL*abs(high);
    if(any(error>tolerance)&&depth<CM12_CURRENT_MAP_MEASURE_MAX_DEPTH){
      for(var child=0u;child<8u;child+=1u){
        let offset=vec3f(f32(child&1u),f32((child>>1u)&1u),f32(child>>2u));
        stack[pending]=vec4f(box.xyz+half*offset,half);
        coarse[pending]=children[child];depths[pending]=depth+1u;pending+=1u;
      }
    }else{
      result.value+=high;result.error+=error;
    }
  }
  // Local subdivision distributes the support's absolute error budget by
  // volume. At the work limit judge the sum of leaf estimates against the
  // original support budget, rather than requiring every leaf to spend an
  // identical fraction of it. No unresolved support is published.
  let supportTolerance=vec4f(CM12_CURRENT_MAP_MEASURE_ABS_TOL)
    +CM12_CURRENT_MAP_MEASURE_REL_TOL*abs(result.value);
  result.unresolved=select(0u,1u,any(result.error>supportTolerance));
  return result;
}
fn cm12CurrentMapMeasureSupport(lower:vec3f,bank:u32)->CurrentMapMeasureEstimate{
  let range=cm12CurrentMapRangeOnFineSupport(vec3i(lower),bank);
  let splitSaturation=!cm12CurrentMapSeedRangeIsSaturated(range[0],range[1]);
  var integrationAxis=1u;
  if(splitSaturation){integrationAxis=cm12CurrentMapMeasureIntegrationAxis(lower,bank);}
  var result=cm12CurrentMapMeasureSupportAxis(lower,bank,splitSaturation,integrationAxis);
  var evaluations=result.evaluations+select(0u,6u,splitSaturation);
  // A single support may contain competing branches or a curved sliver.
  // Its strongest overall variation need not be the best slicing direction
  // on the material actually inside it. Retry only unresolved supports in
  // deterministic cyclic axis order, retaining exactly the same density,
  // momentum quadrature, depth limit, and numerical acceptance budget.
  for(var offset=1u;offset<3u;offset+=1u){
    if(result.unresolved==0u){break;}
    let next=cm12CurrentMapMeasureSupportAxis(lower,bank,splitSaturation,
      (integrationAxis+offset)%3u);
    evaluations+=next.evaluations;result=next;
  }
  result.evaluations=evaluations;
  return result;
}
@compute @workgroup_size(64)
fn integrateCurrentMapFineMeasure(@builtin(global_invocation_id)gid:vec3u){
  let ordinal=gid.x;
  if(ordinal>=CM12_CURRENT_MAP_MEASURE_COUNT||cm12CurrentMapFailed()){return;}
  let nx=u32(CM12_CURRENT_MAP_MEASURE_DIMS.x);
  let ny=u32(CM12_CURRENT_MAP_MEASURE_DIMS.y);
  let q=vec3i(i32(ordinal%nx),i32((ordinal/nx)%ny),i32(ordinal/(nx*ny)));
  let bank=cm12CurrentMapCandidateBank();
  let range=cm12CurrentMapRangeOnFineSupport(q,bank);
  if(cm12CurrentMapSeedRangeIsDry(range[0],range[1])){
    cm12CurrentMapWriteFineMeasure(ordinal,bank,vec4f(0.0));return;
  }
${unchangedHook ? `  if(${unchangedHook}(q)){
    cm12CurrentMapWriteFineMeasure(ordinal,bank,cm12CurrentMapFineMeasure(q,1u-bank));return;
  }` : ""}
  let estimate=cm12CurrentMapMeasureSupport(vec3f(q),bank);
  let finite=all(abs(estimate.value)<=vec4f(3.402823e38))
    &&all(abs(estimate.error)<=vec4f(3.402823e38));
  if(!finite||estimate.value.x<0.0){
    cm12CurrentMapFail(20u,ordinal);
    return;
  }
  if(estimate.unresolved!=0u){
    let tolerance=vec4f(CM12_CURRENT_MAP_MEASURE_ABS_TOL)
      +CM12_CURRENT_MAP_MEASURE_REL_TOL*abs(estimate.value);
    cm12CurrentMapMeasureFailure(ordinal,estimate.error,tolerance,estimate.value);
    return;
  }
  cm12CurrentMapWriteFineMeasure(ordinal,bank,estimate.value);
}

// One workgroup owns a support and its adaptive tree. The root-cut and
// inner Gauss rule are exactly the scalar helpers above. Sharing the outer
// lines avoids putting an entire adaptive integral in one GPU invocation.
// Individual line values retain the scalar x-row/z-row summation order.
var<workgroup> cm12CurrentMapMeasureLines:array<vec4f,128>;
var<workgroup> cm12CurrentMapMeasureLineEvaluations:array<u32,128>;
var<workgroup> cm12CurrentMapMeasureRules:array<vec4f,8>;
var<workgroup> cm12CurrentMapMeasureRuleEvaluations:array<u32,8>;
var<workgroup> cm12CurrentMapMeasureStack:array<vec4f,${1 + 7 * maximumDepth}>;
var<workgroup> cm12CurrentMapMeasureCoarse:array<vec4f,${1 + 7 * maximumDepth}>;
var<workgroup> cm12CurrentMapMeasureDepths:array<u32,${1 + 7 * maximumDepth}>;
var<workgroup> cm12CurrentMapMeasurePending:u32;
var<workgroup> cm12CurrentMapMeasureBox:vec4f;
var<workgroup> cm12CurrentMapMeasureBoxCoarse:vec4f;
var<workgroup> cm12CurrentMapMeasureBoxDepth:u32;
var<workgroup> cm12CurrentMapMeasureSum:vec4f;
var<workgroup> cm12CurrentMapMeasureError:vec4f;
var<workgroup> cm12CurrentMapMeasureEvaluations:u32;
var<workgroup> cm12CurrentMapMeasureStatus:u32;
var<workgroup> cm12CurrentMapMeasureBank:u32;
var<workgroup> cm12CurrentMapMeasureSplit:u32;
var<workgroup> cm12CurrentMapMeasureFirstAxis:u32;

fn cm12CurrentMapMeasureCooperativeRules(lower:vec3f,span:f32,bank:u32,
 splitSaturation:bool,integrationAxis:u32,children:bool,lane:u32){
  let ruleCount=select(1u,8u,children);
  let ruleSpan=select(span,0.5*span,children);
  // Eight lanes per child evaluate its sixteen outer lines. On the initial
  // unsplit support only lanes 0..15 work; every lane reaches both barriers.
  let child=select(0u,lane/8u,children);
  let firstLine=select(lane,lane%8u,children);
  let stride=select(64u,8u,children);
  var offset=vec3f(0.0);
  if(children){offset=vec3f(f32(child&1u),f32((child>>1u)&1u),f32(child>>2u));}
  for(var line=firstLine;line<16u;line+=stride){
    let x=line%4u;let z=line/4u;var lineLower=lower+ruleSpan*offset;
    lineLower[(integrationAxis+1u)%3u]+=ruleSpan*CM12_CURRENT_MAP_GAUSS_4_X[x];
    lineLower[(integrationAxis+2u)%3u]+=ruleSpan*CM12_CURRENT_MAP_GAUSS_4_X[z];
    let value=cm12CurrentMapMeasureLine(lineLower,ruleSpan,bank,splitSaturation,integrationAxis);
    let at=16u*child+line;
    cm12CurrentMapMeasureLines[at]=value.value;
    cm12CurrentMapMeasureLineEvaluations[at]=value.evaluations;
  }
  workgroupBarrier();
  if(lane==0u){
    for(var rule=0u;rule<ruleCount;rule++){
      var total=vec4f(0.0);var evaluations=0u;
      for(var z=0u;z<4u;z++){
        var row=vec4f(0.0);
        for(var x=0u;x<4u;x++){
          let at=16u*rule+4u*z+x;
          row+=CM12_CURRENT_MAP_GAUSS_4_W[x]*cm12CurrentMapMeasureLines[at];
          evaluations+=cm12CurrentMapMeasureLineEvaluations[at];
        }
        total+=CM12_CURRENT_MAP_GAUSS_4_W[z]*row;
      }
      cm12CurrentMapMeasureRules[rule]=total*(ruleSpan*ruleSpan*ruleSpan);
      cm12CurrentMapMeasureRuleEvaluations[rule]=evaluations;
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(64)
fn integrateCurrentMapFineMeasureCooperative(@builtin(workgroup_id)group:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let ordinal=group.x;
  let nx=u32(CM12_CURRENT_MAP_MEASURE_DIMS.x);
  let ny=u32(CM12_CURRENT_MAP_MEASURE_DIMS.y);
  let q=vec3i(i32(ordinal%nx),i32((ordinal/nx)%ny),i32(ordinal/(nx*ny)));
  if(lane==0u){
    cm12CurrentMapMeasureStatus=0u;
    if(ordinal<CM12_CURRENT_MAP_MEASURE_COUNT&&!cm12CurrentMapFailed()){
      let bank=cm12CurrentMapCandidateBank();cm12CurrentMapMeasureBank=bank;
      let range=${rangeCache === undefined ? "cm12CurrentMapRangeOnFineSupport(q,bank)" : "cm12CurrentMapCachedFineMeasureRange(ordinal)"};
      if(cm12CurrentMapSeedRangeIsDry(range[0],range[1])){
        cm12CurrentMapWriteFineMeasure(ordinal,bank,vec4f(0.0));
        cm12CurrentMapMeasureStatus=1u;
      }else{
${unchangedHook ? `        if(${unchangedHook}(q)){
          cm12CurrentMapWriteFineMeasure(ordinal,bank,cm12CurrentMapFineMeasure(q,1u-bank));
          cm12CurrentMapMeasureStatus=1u;
        }else{
` : ""}        let split=!cm12CurrentMapSeedRangeIsSaturated(range[0],range[1]);
        cm12CurrentMapMeasureSplit=select(0u,1u,split);
        cm12CurrentMapMeasureFirstAxis=1u;
        if(split){cm12CurrentMapMeasureFirstAxis=cm12CurrentMapMeasureIntegrationAxis(vec3f(q),bank);}
        cm12CurrentMapMeasureEvaluations=select(0u,6u,split);
        cm12CurrentMapMeasureStatus=2u;
${unchangedHook ? "        }\n" : ""}      }
    }
  }
  // Only workgroup-uniform state controls paths containing barriers. A dry
  // support still completes exactly once, after its zero measure is written.
  let status=workgroupUniformLoad(&cm12CurrentMapMeasureStatus);
  if(status!=2u){
    if(lane==0u&&status==1u){cm12CurrentMapMeasureCompleted(ordinal);}
    return;
  }
  let bank=workgroupUniformLoad(&cm12CurrentMapMeasureBank);
  let split=workgroupUniformLoad(&cm12CurrentMapMeasureSplit)!=0u;
  let firstAxis=workgroupUniformLoad(&cm12CurrentMapMeasureFirstAxis);
  var finalValue=vec4f(0.0);var finalError=vec4f(0.0);var unresolved=true;
  for(var attempt=0u;attempt<3u;attempt++){
    let axis=(firstAxis+attempt)%3u;
    cm12CurrentMapMeasureCooperativeRules(vec3f(q),1.0,bank,split,axis,false,lane);
    if(lane==0u){
      cm12CurrentMapMeasureStack[0]=vec4f(vec3f(q),1.0);
      cm12CurrentMapMeasureCoarse[0]=cm12CurrentMapMeasureRules[0];
      cm12CurrentMapMeasureDepths[0]=0u;cm12CurrentMapMeasurePending=1u;
      cm12CurrentMapMeasureSum=vec4f(0.0);cm12CurrentMapMeasureError=vec4f(0.0);
      cm12CurrentMapMeasureEvaluations+=cm12CurrentMapMeasureRuleEvaluations[0];
    }
    for(var visited=0u;visited<${(8 ** (maximumDepth + 1) - 1) / 7}u;visited++){
      if(workgroupUniformLoad(&cm12CurrentMapMeasurePending)==0u){break;}
      if(lane==0u){
        cm12CurrentMapMeasurePending--;
        let at=cm12CurrentMapMeasurePending;
        cm12CurrentMapMeasureBox=cm12CurrentMapMeasureStack[at];
        cm12CurrentMapMeasureBoxCoarse=cm12CurrentMapMeasureCoarse[at];
        cm12CurrentMapMeasureBoxDepth=cm12CurrentMapMeasureDepths[at];
      }
      let box=workgroupUniformLoad(&cm12CurrentMapMeasureBox);
      cm12CurrentMapMeasureCooperativeRules(box.xyz,box.w,bank,split,axis,true,lane);
      if(lane==0u){
        var high=vec4f(0.0);
        for(var child=0u;child<8u;child++){
          high+=cm12CurrentMapMeasureRules[child];
          cm12CurrentMapMeasureEvaluations+=cm12CurrentMapMeasureRuleEvaluations[child];
        }
        let error=abs(high-cm12CurrentMapMeasureBoxCoarse);
        let tolerance=vec4f(CM12_CURRENT_MAP_MEASURE_ABS_TOL*box.w*box.w*box.w)
          +CM12_CURRENT_MAP_MEASURE_REL_TOL*abs(high);
        let depth=cm12CurrentMapMeasureBoxDepth;let half=0.5*box.w;
        if(any(error>tolerance)&&depth<CM12_CURRENT_MAP_MEASURE_MAX_DEPTH){
          // With d pending sibling levels, popping before eight pushes
          // bounds the stack by 1+7*maximumDepth, exactly as the scalar DFS.
          for(var child=0u;child<8u;child++){
            let offset=vec3f(f32(child&1u),f32((child>>1u)&1u),f32(child>>2u));
            let at=cm12CurrentMapMeasurePending;
            cm12CurrentMapMeasureStack[at]=vec4f(box.xyz+half*offset,half);
            cm12CurrentMapMeasureCoarse[at]=cm12CurrentMapMeasureRules[child];
            cm12CurrentMapMeasureDepths[at]=depth+1u;cm12CurrentMapMeasurePending++;
          }
        }else{
          cm12CurrentMapMeasureSum+=high;cm12CurrentMapMeasureError+=error;
        }
      }
    }
    finalValue=workgroupUniformLoad(&cm12CurrentMapMeasureSum);
    finalError=workgroupUniformLoad(&cm12CurrentMapMeasureError);
    let tolerance=vec4f(CM12_CURRENT_MAP_MEASURE_ABS_TOL)
      +CM12_CURRENT_MAP_MEASURE_REL_TOL*abs(finalValue);
    unresolved=any(finalError>tolerance);
    if(!unresolved){break;}
  }
  // All collective work is complete before the only publishing lane exits.
  if(lane!=0u){return;}
  if(cm12CurrentMapFailed()){return;}
  let finite=all(abs(finalValue)<=vec4f(3.402823e38))
    &&all(abs(finalError)<=vec4f(3.402823e38));
  if(!finite||finalValue.x<0.0){cm12CurrentMapFail(20u,ordinal);return;}
  if(unresolved){
    let tolerance=vec4f(CM12_CURRENT_MAP_MEASURE_ABS_TOL)
      +CM12_CURRENT_MAP_MEASURE_REL_TOL*abs(finalValue);
    cm12CurrentMapMeasureFailure(ordinal,finalError,tolerance,finalValue);return;
  }
  cm12CurrentMapWriteFineMeasure(ordinal,bank,finalValue);
  cm12CurrentMapMeasureCompleted(ordinal);
}
`;
}
