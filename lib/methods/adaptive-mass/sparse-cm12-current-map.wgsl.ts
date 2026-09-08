/** Current inverse deformation map, independent of the native partition.
 * Coordinates and displacement are in finest-cell units. Each accepted step
 * archives its certified C2 departure increment; the current map is their
 * ordered composition. The cumulative map is never spatially resampled.
 * Storage is explicitly bounded and a full archive rejects the next step
 * before publication. */
export interface SparseCM12CurrentMapLayout {
  readonly baseWords: number;
  readonly dimensions: readonly [number, number, number];
  readonly padding: number;
  readonly spacingFine: number;
  readonly originFine: readonly [number, number, number];
  readonly nodeDimensions: readonly [number, number, number];
  readonly cellDimensions: readonly [number, number, number];
  readonly nodeCount: number;
  readonly cellCount: number;
  readonly coefficientBaseWords: readonly [number, number];
  readonly nodalBaseWords: number;
  readonly scratchBaseWords: number;
  readonly immutableVelocityBaseWords: number;
  readonly boundaryBoundsBaseWords: number;
  readonly boundarySampleCount: number;
  readonly traceSubstepCountBaseWords: number;
  readonly chainCapacity: number;
  readonly chainBaseWords: number;
  readonly chainCountBaseWords: number;
  readonly lineCounts: readonly [number, number, number];
  readonly maximumLineLength: number;
  readonly totalWords: number;
  readonly endWords: number;
}

export function createSparseCM12CurrentMapLayout(baseWords: number,
  dimensions: readonly [number, number, number], padding = 16, spacingFine = 0.5,
  chainCapacity = 32): SparseCM12CurrentMapLayout {
  if (!Number.isSafeInteger(baseWords) || baseWords < 0
    || !Number.isSafeInteger(padding) || padding < 4
    || (spacingFine !== 1 && spacingFine !== 0.5)
    || !Number.isSafeInteger(chainCapacity) || chainCapacity < 1 || chainCapacity > 1024
    || dimensions.some(n => !Number.isSafeInteger(n) || n < 1)) {
    throw new RangeError("Invalid current-map geometry or arena offset");
  }
  const nodeDimensions = dimensions.map(n => (n + 2 * padding) / spacingFine + 1) as [number, number, number];
  const cellDimensions = nodeDimensions.map(n => n - 1) as [number, number, number];
  const nodeCount = nodeDimensions.reduce((a, b) => a * b, 1);
  const cellCount = cellDimensions.reduce((a, b) => a * b, 1);
  const totalWords = (15 + 3 * chainCapacity) * nodeCount + 8;
  if (!Number.isSafeInteger(totalWords) || baseWords + totalWords > 0xffff_ffff
    || Math.max(...nodeDimensions) > 1024) throw new RangeError("Current-map storage exceeds bounded indexing");
  return Object.freeze({ baseWords, dimensions: Object.freeze([...dimensions]) as unknown as readonly [number, number, number],
    padding, spacingFine, originFine: Object.freeze([-padding, -padding, -padding]) as readonly [number, number, number],
    nodeDimensions: Object.freeze(nodeDimensions), cellDimensions: Object.freeze(cellDimensions), nodeCount, cellCount,
    coefficientBaseWords: Object.freeze([baseWords, baseWords + 3 * nodeCount]) as readonly [number, number],
    nodalBaseWords: baseWords + 6 * nodeCount, scratchBaseWords: baseWords + 9 * nodeCount,
    immutableVelocityBaseWords: baseWords + 12 * nodeCount,
    boundaryBoundsBaseWords: baseWords + 15 * nodeCount,
    boundarySampleCount: 2 * (dimensions[0] * dimensions[1] + dimensions[0] * dimensions[2] + dimensions[1] * dimensions[2]),
    traceSubstepCountBaseWords: baseWords + 15 * nodeCount + 6,
    chainCapacity, chainBaseWords: baseWords + 15 * nodeCount + 7,
    chainCountBaseWords: baseWords + (15 + 3 * chainCapacity) * nodeCount + 7,
    lineCounts: Object.freeze([nodeDimensions[1] * nodeDimensions[2], nodeDimensions[0] * nodeDimensions[2],
      nodeDimensions[0] * nodeDimensions[1]]) as readonly [number, number, number],
    maximumLineLength: Math.max(...nodeDimensions), totalWords, endWords: baseWords + totalWords });
}

/** Requires state:array<f32>, p.frame.x (seconds),
 * cm12RetainedDensityAcceptedBank(), cm12CurrentMapNativeVelocity(pointFine)
 * returning vec4f(velocity,validity), cm12CurrentMapInitializeVelocity(id,sample),
 * cm12CurrentMapVelocityBoundary(pointFine,velocity),
 * cm12CurrentMapCoefficientBoundaryPoint(pointFine),
 * cm12CurrentMapCoefficientBoundaryValue(pointFine,value), cm12CurrentMapFail(code,id).
 * A boundary-value hook that zeros a wall's normal component must pair it
 * with odd normal/even tangential ghost controls in the point/value hooks.
 * Every archived increment is filtered and certified under that contract.
 * compileCurrentMapVelocity freezes the native transport-VEX generation on
 * its finest-centre lattice before any gather can overwrite it. Failure
 * codes: 1 nonfinite, 2 excessive travel,
 * 3 unresolved orientation, 4 nonzero boundary collar, 5 chain capacity. */
export function createSparseCM12CurrentMapWGSL(layout: SparseCM12CurrentMapLayout): string {
  const uintVector = (values: readonly number[]) => values.map(n => `${n}u`).join(",");
  const [nx, ny, nz] = layout.dimensions;
  const boundaryFaceOffsets = [0];
  for (const count of [ny * nz, ny * nz, nx * nz, nx * nz, nx * ny, nx * ny]) {
    boundaryFaceOffsets.push(boundaryFaceOffsets[boundaryFaceOffsets.length - 1]! + count);
  }
  return /* wgsl */ `
const CM12_CURRENT_MAP_NODES:vec3u=vec3u(${uintVector(layout.nodeDimensions)});
const CM12_CURRENT_MAP_CELLS:vec3u=vec3u(${uintVector(layout.cellDimensions)});
const CM12_CURRENT_MAP_ORIGIN:vec3f=vec3f(-${layout.padding}.0);
const CM12_CURRENT_MAP_SPACING:f32=${layout.spacingFine.toFixed(1)};
const CM12_CURRENT_MAP_INVERSE_SPACING:f32=${(1 / layout.spacingFine).toFixed(1)};
const CM12_CURRENT_MAP_CELLS_PER_FINE:u32=${1 / layout.spacingFine}u;
const CM12_CURRENT_MAP_NODE_COUNT:u32=${layout.nodeCount}u;
const CM12_CURRENT_MAP_CELL_COUNT:u32=${layout.cellCount}u;
const CM12_CURRENT_MAP_NODAL_BASE:u32=${layout.nodalBaseWords}u;
const CM12_CURRENT_MAP_SCRATCH_BASE:u32=${layout.scratchBaseWords}u;
const CM12_CURRENT_MAP_VELOCITY_BASE:u32=${layout.immutableVelocityBaseWords}u;
const CM12_CURRENT_MAP_BOUNDARY_BOUNDS_BASE:u32=${layout.boundaryBoundsBaseWords}u;
const CM12_CURRENT_MAP_BOUNDARY_SAMPLE_COUNT:u32=${layout.boundarySampleCount}u;
const CM12_CURRENT_MAP_BOUNDARY_FACE_OFFSETS:array<u32,7>=array<u32,7>(${uintVector(boundaryFaceOffsets)});
const CM12_CURRENT_MAP_TRACE_SUBSTEP_COUNT_BASE:u32=${layout.traceSubstepCountBaseWords}u;
const CM12_CURRENT_MAP_CHAIN_BASE:u32=${layout.chainBaseWords}u;
const CM12_CURRENT_MAP_CHAIN_COUNT_BASE:u32=${layout.chainCountBaseWords}u;
const CM12_CURRENT_MAP_CHAIN_CAPACITY:u32=${layout.chainCapacity}u;
const CM12_CURRENT_MAP_ROUNDOFF:f32=0.00000762939453125;
struct CurrentMapEvaluation {point:vec3f,jacobian:mat3x3f}
fn cm12CurrentMapCandidateBank()->u32{return 1u-cm12RetainedDensityAcceptedBank();}
fn cm12CurrentMapChainCount()->u32{return u32(state[CM12_CURRENT_MAP_CHAIN_COUNT_BASE]);}
fn cm12CurrentMapChainBase(slot:u32)->u32{return CM12_CURRENT_MAP_CHAIN_BASE+3u*CM12_CURRENT_MAP_NODE_COUNT*slot;}
fn cm12CurrentMapCommitIncrement(){
  state[CM12_CURRENT_MAP_CHAIN_COUNT_BASE]=f32(cm12CurrentMapChainCount()+1u);
}
fn cm12CurrentMapCoefficientBase(bank:u32)->u32{
  return select(${layout.coefficientBaseWords[0]}u,${layout.coefficientBaseWords[1]}u,bank!=0u);
}
fn cm12CurrentMapNodeId(q:vec3u)->u32{
  return q.x+CM12_CURRENT_MAP_NODES.x*(q.y+CM12_CURRENT_MAP_NODES.y*q.z);
}
fn cm12CurrentMapNodeCoordinate(id:u32)->vec3u{
  return vec3u(id%CM12_CURRENT_MAP_NODES.x,(id/CM12_CURRENT_MAP_NODES.x)%CM12_CURRENT_MAP_NODES.y,
    id/(CM12_CURRENT_MAP_NODES.x*CM12_CURRENT_MAP_NODES.y));
}
fn cm12CurrentMapRead(base:u32,id:u32)->vec3f{
  let at=base+3u*id;return vec3f(state[at],state[at+1u],state[at+2u]);
}
fn cm12CurrentMapWrite(base:u32,id:u32,value:vec3f){
  let at=base+3u*id;state[at]=value.x;state[at+1u]=value.y;state[at+2u]=value.z;
}
fn cm12CurrentMapFinite(value:vec3f)->bool{return all(abs(value)<vec3f(3.4e38));}
@compute @workgroup_size(64)
fn compileCurrentMapVelocity(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_NODE_COUNT){return;}
  // Sample on native finest-cell centres. A second node-centred interpolation
  // would unnecessarily convolve the all-fine trilinear VEX field again.
  let point=CM12_CURRENT_MAP_ORIGIN+CM12_CURRENT_MAP_SPACING*vec3f(cm12CurrentMapNodeCoordinate(id))+vec3f(0.5);
  let sample=cm12CurrentMapNativeVelocity(point);
  if(!cm12CurrentMapFinite(sample.xyz)){cm12CurrentMapFail(1u,id);return;}
  cm12CurrentMapInitializeVelocity(id,sample);
}
fn cm12CurrentMapVelocity(point:vec3f)->vec3f{
  let local=(point-CM12_CURRENT_MAP_ORIGIN-vec3f(0.5))*CM12_CURRENT_MAP_INVERSE_SPACING;
  let first=vec3i(floor(local));let t=local-vec3f(first);var value=vec3f(0.0);
  for(var corner=0u;corner<8u;corner++){
    let offset=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);let q=first+vec3i(offset);
    if(any(q<vec3i(0))||any(q>=vec3i(CM12_CURRENT_MAP_NODES))){continue;}
    let weight=select(vec3f(1.0)-t,t,offset!=vec3u(0u));
    value+=weight.x*weight.y*weight.z*cm12CurrentMapRead(CM12_CURRENT_MAP_VELOCITY_BASE,cm12CurrentMapNodeId(vec3u(q)));
  }
  return cm12CurrentMapVelocityBoundary(point,value);
}
@compute @workgroup_size(64)
fn compileCurrentMapTraceSchedule(@builtin(global_invocation_id)gid:vec3u){
  let line=gid.x;let lineCount=CM12_CURRENT_MAP_NODES.y*CM12_CURRENT_MAP_NODES.z;
  if(line>=lineCount){return;}
  var maximum=0.0;
  for(var x=0u;x<CM12_CURRENT_MAP_NODES.x;x++){
    let id=x+CM12_CURRENT_MAP_NODES.x*line;
    let velocity=cm12CurrentMapRead(CM12_CURRENT_MAP_VELOCITY_BASE,id);
    if(!cm12CurrentMapFinite(velocity)){cm12CurrentMapFail(1u,id);maximum=3.4e38;break;}
    maximum=max(maximum,length(velocity));
  }
  // Velocity extension has finished, so its original-validity nodal scratch
  // is reusable. All lines have disjoint reduction slots.
  state[CM12_CURRENT_MAP_NODAL_BASE+3u*line]=maximum;
}
@compute @workgroup_size(1)
fn sealCurrentMapTraceSchedule(){
  let lineCount=CM12_CURRENT_MAP_NODES.y*CM12_CURRENT_MAP_NODES.z;
  var maximum=0.0;
  for(var line=0u;line<lineCount;line++){
    maximum=max(maximum,state[CM12_CURRENT_MAP_NODAL_BASE+3u*line]);
  }
  let travel=maximum*p.frame.x;
  if(!(travel>=0.0)||travel>16.0){
    cm12CurrentMapFail(2u,0u);state[CM12_CURRENT_MAP_TRACE_SUBSTEP_COUNT_BASE]=1.0;return;
  }
  // Trilinear velocity is a convex combination of cached samples; the
  // boundary hook may only damp components. One count therefore bounds
  // every characteristic and keeps the numerical departure continuous.
  state[CM12_CURRENT_MAP_TRACE_SUBSTEP_COUNT_BASE]=max(1.0,ceil(travel));
}
fn cm12CurrentMapCollar(q:vec3u)->bool{
  return any(q<vec3u(2u))||any(q+vec3u(2u)>=CM12_CURRENT_MAP_NODES);
}
fn cm12CurrentMapControlAtBase(q:vec3i,base:u32)->vec3f{
  // The endpoint extension has linear ghost controls 2*c0-c1. Both controls
  // are zero in the explicitly imposed exterior collar, so the ghost is
  // exactly zero too. The resulting displacement joins the identity in C2.
  if(any(q<vec3i(0))||any(q>=vec3i(CM12_CURRENT_MAP_NODES))){return vec3f(0.0);}
  return cm12CurrentMapRead(base,cm12CurrentMapNodeId(vec3u(q)));
}
fn cm12CurrentMapControl(q:vec3i,bank:u32)->vec3f{
  return cm12CurrentMapControlAtBase(q,cm12CurrentMapCoefficientBase(bank));
}
fn cm12CurrentMapBasis(t:f32)->vec4f{
  let u=1.0-t;let t2=t*t;let t3=t2*t;
  return vec4f(u*u*u,4.0-6.0*t2+3.0*t3,1.0+3.0*t+3.0*t2-3.0*t3,t3)/6.0;
}
fn cm12CurrentMapBasisDerivative(t:f32)->vec4f{
  let u=1.0-t;
  return vec4f(-0.5*u*u,1.5*t*t-2.0*t,-1.5*t*t+t+0.5,0.5*t*t);
}
fn cm12CurrentMapEvaluateIncrementAtBase(point:vec3f,base:u32)->CurrentMapEvaluation{
  var result:CurrentMapEvaluation;result.point=point;
  result.jacobian=mat3x3f(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),vec3f(0.0,0.0,1.0));
  let local=(point-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING;
  if(any(local<vec3f(0.0))||any(local>vec3f(CM12_CURRENT_MAP_CELLS))){return result;}
  let cell=min(vec3i(floor(local)),vec3i(CM12_CURRENT_MAP_CELLS)-vec3i(1));
  let t=local-vec3f(cell);
  let bx=cm12CurrentMapBasis(t.x);let by=cm12CurrentMapBasis(t.y);let bz=cm12CurrentMapBasis(t.z);
  let dx=cm12CurrentMapBasisDerivative(t.x);let dy=cm12CurrentMapBasisDerivative(t.y);let dz=cm12CurrentMapBasisDerivative(t.z);
  var displacement=vec3f(0.0);var gradientX=vec3f(0.0);var gradientY=vec3f(0.0);var gradientZ=vec3f(0.0);
  for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
    let value=cm12CurrentMapControlAtBase(cell+vec3i(i32(x)-1,i32(y)-1,i32(z)-1),base);
    displacement+=value*(bx[x]*by[y]*bz[z]);
    gradientX+=value*(dx[x]*by[y]*bz[z]);gradientY+=value*(bx[x]*dy[y]*bz[z]);
    gradientZ+=value*(bx[x]*by[y]*dz[z]);
  }}}
  result.point+=displacement;
  result.jacobian[0]+=CM12_CURRENT_MAP_INVERSE_SPACING*gradientX;
  result.jacobian[1]+=CM12_CURRENT_MAP_INVERSE_SPACING*gradientY;
  result.jacobian[2]+=CM12_CURRENT_MAP_INVERSE_SPACING*gradientZ;
  return result;
}
fn cm12CurrentMapEvaluateIncrement(point:vec3f,bank:u32)->CurrentMapEvaluation{
  return cm12CurrentMapEvaluateIncrementAtBase(point,cm12CurrentMapCoefficientBase(bank));
}
fn cm12CurrentMapEvaluate(point:vec3f,bank:u32)->CurrentMapEvaluation{
  var result:CurrentMapEvaluation;result.point=point;
  result.jacobian=mat3x3f(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),vec3f(0.0,0.0,1.0));
  if(bank!=cm12RetainedDensityAcceptedBank()){
    result=cm12CurrentMapEvaluateIncrementAtBase(point,cm12CurrentMapCoefficientBase(bank));
  }
  var count=cm12CurrentMapChainCount();
  loop{
    if(count==0u){break;}count--;
    let previous=cm12CurrentMapEvaluateIncrementAtBase(result.point,cm12CurrentMapChainBase(count));
    result.point=previous.point;result.jacobian=previous.jacobian*result.jacobian;
  }
  return result;
}
// Seed-phi cuts need only the mapped point. Keep the point arithmetic and
// chain order identical to the full evaluator while omitting derivatives.
fn cm12CurrentMapEvaluatePointIncrementAtBase(point:vec3f,base:u32)->vec3f{
  var result=point;
  let local=(point-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING;
  if(any(local<vec3f(0.0))||any(local>vec3f(CM12_CURRENT_MAP_CELLS))){return result;}
  let cell=min(vec3i(floor(local)),vec3i(CM12_CURRENT_MAP_CELLS)-vec3i(1));
  let t=local-vec3f(cell);
  let bx=cm12CurrentMapBasis(t.x);let by=cm12CurrentMapBasis(t.y);let bz=cm12CurrentMapBasis(t.z);
  var displacement=vec3f(0.0);
  for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
    let value=cm12CurrentMapControlAtBase(cell+vec3i(i32(x)-1,i32(y)-1,i32(z)-1),base);
    displacement+=value*(bx[x]*by[y]*bz[z]);
  }}}
  result+=displacement;
  return result;
}
fn cm12CurrentMapEvaluatePoint(point:vec3f,bank:u32)->vec3f{
  var result=point;
  if(bank!=cm12RetainedDensityAcceptedBank()){
    result=cm12CurrentMapEvaluatePointIncrementAtBase(point,cm12CurrentMapCoefficientBase(bank));
  }
  var count=cm12CurrentMapChainCount();
  loop{
    if(count==0u){break;}count--;
    let previous=cm12CurrentMapEvaluatePointIncrementAtBase(result,cm12CurrentMapChainBase(count));
    result=previous;
  }
  return result;
}
fn cm12CurrentMapUnchangedOnFineSupport(q:vec3i)->bool{
  let cell=vec3i((vec3f(q)-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING);
  let candidate=cm12CurrentMapCandidateBank();let controls=i32(CM12_CURRENT_MAP_CELLS_PER_FINE)+3;
  for(var z=0;z<controls;z++){for(var y=0;y<controls;y++){for(var x=0;x<controls;x++){
    let at=cell+vec3i(x-1,y-1,z-1);
    if(any(cm12CurrentMapControl(at,candidate)!=vec3f(0.0))){return false;}
  }}}
  return true;
}
fn cm12CurrentMapDeparture(point:vec3f)->vec3f{
  let schedule=state[CM12_CURRENT_MAP_TRACE_SUBSTEP_COUNT_BASE];
  if(!(schedule>=1.0)||schedule>16.0){cm12CurrentMapFail(2u,0u);return point;}
  let count=u32(schedule);let dt=p.frame.x/f32(count);var departure=point;
  for(var step=0u;step<count;step++){
    let first=cm12CurrentMapVelocity(departure);
    let midpoint=departure-0.5*dt*first;
    let velocity=cm12CurrentMapVelocity(midpoint);
    if(!cm12CurrentMapFinite(first)||!cm12CurrentMapFinite(velocity)){
      cm12CurrentMapFail(1u,0u);return point;
    }
    departure-=dt*velocity;
  }
  return departure;
}
@compute @workgroup_size(64)
fn advanceCurrentMapNodes(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_NODE_COUNT){return;}
  if(cm12CurrentMapChainCount()>=CM12_CURRENT_MAP_CHAIN_CAPACITY){cm12CurrentMapFail(5u,id);return;}
  let coordinate=cm12CurrentMapNodeCoordinate(id);var displacement=vec3f(0.0);
  if(!cm12CurrentMapCollar(coordinate)){
    let point=CM12_CURRENT_MAP_ORIGIN+CM12_CURRENT_MAP_SPACING*vec3f(coordinate);
    let departure=cm12CurrentMapDeparture(point);
    displacement=departure-point;
    if(!cm12CurrentMapFinite(displacement)){cm12CurrentMapFail(1u,id);return;}
  }
  cm12CurrentMapWrite(CM12_CURRENT_MAP_NODAL_BASE,id,displacement);
}
@compute @workgroup_size(64)
fn publishCurrentMapIncrement(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_NODE_COUNT||cm12CurrentMapFailed()){return;}
  let count=cm12CurrentMapChainCount();
  if(count>=CM12_CURRENT_MAP_CHAIN_CAPACITY){cm12CurrentMapFail(5u,id);return;}
  cm12CurrentMapWrite(cm12CurrentMapChainBase(count),id,
    cm12CurrentMapRead(cm12CurrentMapCoefficientBase(cm12CurrentMapCandidateBank()),id));
}
fn cm12CurrentMapLineCoordinate(line:u32,axis:u32)->vec3u{
  if(axis==0u){return vec3u(0u,line%CM12_CURRENT_MAP_NODES.y,line/CM12_CURRENT_MAP_NODES.y);}
  if(axis==1u){return vec3u(line%CM12_CURRENT_MAP_NODES.x,0u,line/CM12_CURRENT_MAP_NODES.x);}
  return vec3u(line%CM12_CURRENT_MAP_NODES.x,line/CM12_CURRENT_MAP_NODES.x,0u);
}
fn cm12CurrentMapFilterLine(line:u32,axis:u32,source:u32,destination:u32){
  let count=CM12_CURRENT_MAP_NODES[axis];let lineCount=CM12_CURRENT_MAP_NODE_COUNT/count;
  if(line>=lineCount){return;}
  var coordinate=cm12CurrentMapLineCoordinate(line,axis);
  // Local fourth-order cubic B-spline quasi-interpolation:
  // c_i=d_i-(d_(i-1)-2*d_i+d_(i+1))/6. Together with cubic evaluation this
  // reproduces every cubic and has compact dependence on six source nodes.
  // The previous global interpolating solve propagated air-edge errors into
  // distant fluid and could fold between monotone transported node values.
  // This is a spatial approximation to the transported map, with the same
  // unchanged orientation and mass/geometry checks as every candidate.
  for(var i=0u;i<count;i++){
    coordinate=cm12CurrentMapLineCoordinate(line,axis);coordinate[axis]=i;
    let destinationCoordinate=coordinate;
    let point=CM12_CURRENT_MAP_ORIGIN+CM12_CURRENT_MAP_SPACING*vec3f(coordinate);
    if(axis==2u){
      // Read reflected controls from immutable Y-filtered data. This applies
      // physical wall parity without a race against another final Z write.
      coordinate=vec3u(round((cm12CurrentMapCoefficientBoundaryPoint(point)-CM12_CURRENT_MAP_ORIGIN)
        *CM12_CURRENT_MAP_INVERSE_SPACING));
    }
    let centerIndex=coordinate[axis];let center=cm12CurrentMapRead(source,cm12CurrentMapNodeId(coordinate));
    coordinate[axis]=select(centerIndex-1u,0u,centerIndex==0u);
    var left=cm12CurrentMapRead(source,cm12CurrentMapNodeId(coordinate));
    coordinate[axis]=min(centerIndex+1u,count-1u);var right=cm12CurrentMapRead(source,cm12CurrentMapNodeId(coordinate));
    if(centerIndex==0u){left=2.0*center-right;}if(centerIndex==count-1u){right=2.0*center-left;}
    var value=center-((left-center)+(right-center))/6.0;
    if(axis==2u){
      value=cm12CurrentMapCoefficientBoundaryValue(point,value);
      if(cm12CurrentMapCollar(destinationCoordinate)){value=vec3f(0.0);}
    }
    if(!cm12CurrentMapFinite(value)){cm12CurrentMapFail(1u,line);return;}
    cm12CurrentMapWrite(destination,cm12CurrentMapNodeId(destinationCoordinate),value);
  }
}
@compute @workgroup_size(64)
fn filterCurrentMapX(@builtin(global_invocation_id)gid:vec3u){
  cm12CurrentMapFilterLine(gid.x,0u,CM12_CURRENT_MAP_NODAL_BASE,CM12_CURRENT_MAP_SCRATCH_BASE);
}
@compute @workgroup_size(64)
fn filterCurrentMapY(@builtin(global_invocation_id)gid:vec3u){
  cm12CurrentMapFilterLine(gid.x,1u,CM12_CURRENT_MAP_SCRATCH_BASE,CM12_CURRENT_MAP_NODAL_BASE);
}
@compute @workgroup_size(64)
fn filterCurrentMapZ(@builtin(global_invocation_id)gid:vec3u){
  cm12CurrentMapFilterLine(gid.x,2u,CM12_CURRENT_MAP_NODAL_BASE,cm12CurrentMapCoefficientBase(cm12CurrentMapCandidateBank()));
}
fn cm12CurrentMapBernsteinAtBase(cell:vec3i,coefficientBase:u32)->array<vec3f,64>{
  var values:array<vec3f,64>;var next:array<vec3f,64>;
  for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
    values[x+4u*(y+4u*z)]=cm12CurrentMapControlAtBase(cell+vec3i(i32(x)-1,i32(y)-1,i32(z)-1),coefficientBase);
  }}}
  var stride=1u;
  for(var axis=0u;axis<3u;axis++){
    for(var base=0u;base<64u;base+=4u*stride){for(var offset=0u;offset<stride;offset++){
      let i=base+offset;let a=values[i];let b=values[i+stride];let c=values[i+2u*stride];let d=values[i+3u*stride];
      next[i]=(a+4.0*b+c)/6.0;next[i+stride]=(2.0*b+c)/3.0;
      next[i+2u*stride]=(b+2.0*c)/3.0;next[i+3u*stride]=(b+4.0*c+d)/6.0;
    }}
    values=next;stride*=4u;
  }
  return values;
}
fn cm12CurrentMapBernstein(cell:vec3i,bank:u32)->array<vec3f,64>{
  return cm12CurrentMapBernsteinAtBase(cell,cm12CurrentMapCoefficientBase(bank));
}
fn cm12CurrentMapRestrictBernstein(values:array<vec3f,64>,lower:vec3f,upper:vec3f)->array<vec3f,64>{
  var current=values;var next:array<vec3f,64>;var stride=1u;
  for(var axis=0u;axis<3u;axis++){
    let end=upper[axis];let start=select(0.0,lower[axis]/max(end,1e-30),end>0.0);
    for(var base=0u;base<64u;base+=4u*stride){for(var offset=0u;offset<stride;offset++){
      let i=base+offset;let a=current[i];let b=current[i+stride];let c=current[i+2u*stride];let d=current[i+3u*stride];
      // Keep the left polynomial after splitting at upper, then the right
      // polynomial after splitting that at lower/upper (de Casteljau).
      let ab=mix(a,b,end);let bc=mix(b,c,end);let cd=mix(c,d,end);
      let abc=mix(ab,bc,end);let bcd=mix(bc,cd,end);let atEnd=mix(abc,bcd,end);
      let leftAB=mix(a,ab,start);let leftBC=mix(ab,abc,start);let leftCD=mix(abc,atEnd,start);
      let leftABC=mix(leftAB,leftBC,start);let leftBCD=mix(leftBC,leftCD,start);
      next[i]=mix(leftABC,leftBCD,start);next[i+stride]=leftBCD;
      next[i+2u*stride]=leftCD;next[i+3u*stride]=atEnd;
    }}
    current=next;stride*=4u;
  }
  return current;
}
fn cm12CurrentMapBoxAfterIncrement(lower:vec3f,upper:vec3f,base:u32)->mat2x3f{
  let localLower=(lower-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING;
  let localUpper=(upper-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING;
  if(any(localUpper<vec3f(0.0))||any(localLower>vec3f(CM12_CURRENT_MAP_CELLS))){return mat2x3f(lower,upper);}
  let first=clamp(vec3i(floor(localLower)),vec3i(0),vec3i(CM12_CURRENT_MAP_CELLS)-vec3i(1));
  let last=max(first,clamp(vec3i(ceil(localUpper))-vec3i(1),vec3i(0),vec3i(CM12_CURRENT_MAP_CELLS)-vec3i(1)));
  var displacementLower=vec3f(3.4e38);var displacementUpper=vec3f(-3.4e38);var magnitude=0.0;
  if(any(localLower<vec3f(0.0))||any(localUpper>vec3f(CM12_CURRENT_MAP_CELLS))){
    displacementLower=vec3f(0.0);displacementUpper=vec3f(0.0);
  }
  for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
    let cell=vec3i(x,y,z);
    let coefficients=cm12CurrentMapBernsteinAtBase(cell,base);
    let restricted=cm12CurrentMapRestrictBernstein(coefficients,
      clamp(localLower-vec3f(cell),vec3f(0.0),vec3f(1.0)),clamp(localUpper-vec3f(cell),vec3f(0.0),vec3f(1.0)));
    for(var i=0u;i<64u;i++){
      let q=cell+vec3i(i32(i%4u)-1,i32((i/4u)%4u)-1,i32(i/16u)-1);
      let original=abs(cm12CurrentMapControlAtBase(q,base));
      magnitude=max(magnitude,max(original.x,max(original.y,original.z)));
      displacementLower=min(displacementLower,restricted[i]);displacementUpper=max(displacementUpper,restricted[i]);
    }
  }}}
  // An identically zero control stencil is an exact identity restriction;
  // no arithmetic enclosure error is introduced by skipping that link.
  if(magnitude==0.0){return mat2x3f(lower,upper);}
  let resultLower=lower+displacementLower;let resultUpper=upper+displacementUpper;
  magnitude=max(max(1.0,magnitude),max(max(abs(resultLower.x),max(abs(resultLower.y),abs(resultLower.z))),
    max(abs(resultUpper.x),max(abs(resultUpper.y),abs(resultUpper.z)))));
  let error=vec3f(CM12_CURRENT_MAP_ROUNDOFF*magnitude);
  return mat2x3f(resultLower-error,resultUpper+error);
}
fn cm12CurrentMapRangeOnBox(lower:vec3f,upper:vec3f,bank:u32)->mat2x3f{
  var bounds=mat2x3f(lower,upper);
  if(bank!=cm12RetainedDensityAcceptedBank()){
    bounds=cm12CurrentMapBoxAfterIncrement(bounds[0],bounds[1],cm12CurrentMapCoefficientBase(bank));
  }
  var count=cm12CurrentMapChainCount();
  loop{
    if(count==0u){break;}count--;
    bounds=cm12CurrentMapBoxAfterIncrement(bounds[0],bounds[1],cm12CurrentMapChainBase(count));
  }
  return bounds;
}
fn cm12CurrentMapRangeOnMapCell(cell:vec3i,bank:u32)->mat2x3f{
  let origin=CM12_CURRENT_MAP_ORIGIN+CM12_CURRENT_MAP_SPACING*vec3f(cell);
  if(any(cell<vec3i(0))||any(cell>=vec3i(CM12_CURRENT_MAP_CELLS))){
    return mat2x3f(origin,origin+vec3f(CM12_CURRENT_MAP_SPACING));
  }
  let displacement=cm12CurrentMapBernstein(cell,bank);
  var lower=vec3f(3.4e38);var upper=vec3f(-3.4e38);var magnitude=1.0;
  for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
    let control=cm12CurrentMapControl(cell+vec3i(i32(x)-1,i32(y)-1,i32(z)-1),bank);
    magnitude=max(magnitude,max(abs(control.x),max(abs(control.y),abs(control.z))));
    let value=origin+CM12_CURRENT_MAP_SPACING*vec3f(f32(x),f32(y),f32(z))/3.0+displacement[x+4u*(y+4u*z)];
    lower=min(lower,value);upper=max(upper,value);
  }}}
  magnitude=max(magnitude,max(max(abs(lower.x),abs(lower.y)),max(abs(lower.z),
    max(max(abs(upper.x),abs(upper.y)),abs(upper.z)))));
  let error=vec3f(CM12_CURRENT_MAP_ROUNDOFF*magnitude);
  return mat2x3f(lower-error,upper+error);
}
fn cm12CurrentMapRangeOnFineSupport(q:vec3i,bank:u32)->mat2x3f{
  return cm12CurrentMapRangeOnBox(vec3f(q),vec3f(q)+vec3f(1.0),bank);
}
fn cm12CurrentMapBoundarySubfaceNormalRange(cell:vec3i,axis:u32,side:u32,bank:u32)->vec2f{
  let values=cm12CurrentMapBernstein(cell,bank);
  let stride=select(select(1u,4u,axis==1u),16u,axis==2u);let face=select(0u,3u,side!=0u);
  var lower=3.4e38;var upper=-3.4e38;var sourceMagnitude=0.0;
  for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
    let control=cm12CurrentMapControl(cell+vec3i(x-1,y-1,z-1),bank);
    sourceMagnitude=max(sourceMagnitude,abs(control[axis]));
  }}}
  for(var i=0u;i<64u;i++){if((i/stride)%4u==face){lower=min(lower,values[i][axis]);upper=max(upper,values[i][axis]);}}
  if(sourceMagnitude==0.0){return vec2f(0.0);}
  let error=CM12_CURRENT_MAP_ROUNDOFF*sourceMagnitude;
  return vec2f(lower-error,upper+error);
}
fn cm12CurrentMapIncrementBoundaryNormalRange(q:vec3i,axis:u32,side:u32,bank:u32)->vec2f{
  // Cover every refined subface of the physical fine-cell face. The normal
  // cell index is the map cell immediately inside the domain boundary.
  var first=vec3i((vec3f(q)-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING);
  let physicalFace=select(0.0,f32(vec3u(${uintVector(layout.dimensions)})[axis]),side!=0u);
  first[axis]=i32((physicalFace-CM12_CURRENT_MAP_ORIGIN[axis])*CM12_CURRENT_MAP_INVERSE_SPACING)-i32(side);
  let tangentU=select(0u,1u,axis==0u);let tangentV=select(2u,1u,axis==2u);
  var lower=3.4e38;var upper=-3.4e38;
  for(var v=0u;v<CM12_CURRENT_MAP_CELLS_PER_FINE;v++){
    for(var u=0u;u<CM12_CURRENT_MAP_CELLS_PER_FINE;u++){
      var cell=first;cell[tangentU]+=i32(u);cell[tangentV]+=i32(v);
      let range=cm12CurrentMapBoundarySubfaceNormalRange(cell,axis,side,bank);
      lower=min(lower,range.x);upper=max(upper,range.y);
    }
  }
  return vec2f(lower,upper);
}
fn cm12CurrentMapBoundaryNormalRange(q:vec3i,axis:u32,side:u32,bank:u32)->vec2f{
  let face=select(0.0,f32(vec3u(${uintVector(layout.dimensions)})[axis]),side!=0u);
  var lower=vec3f(q);lower[axis]=face;
  // The boundary hook's explicit parity contract makes this a polynomial
  // identity: c0=0 and c(-1)=-c(1) set normal displacement to zero on the
  // entire plane. Every archived increment preserves that same plane, so
  // their composition does too. This proof does not accumulate unrelated
  // generic AABB rounding margins into a fictitious missing-material slab.
  if(cm12CurrentMapCoefficientBoundaryValue(lower,vec3f(1.0))[axis]==0.0){return vec2f(0.0);}
  var upper=lower+vec3f(1.0);upper[axis]=face;
  let bounds=cm12CurrentMapRangeOnBox(lower,upper,bank);
  return vec2f(bounds[0][axis]-face,bounds[1][axis]-face);
}
fn cm12CurrentMapBoundaryNormalBound(q:vec3i,axis:u32,side:u32,bank:u32)->f32{
  let range=cm12CurrentMapBoundaryNormalRange(q,axis,side,bank);
  return max(abs(range.x),abs(range.y));
}
@compute @workgroup_size(64)
fn compileCurrentMapPhysicalBoundary(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_BOUNDARY_SAMPLE_COUNT){return;}
  var face=0u;
  for(var next=1u;next<6u;next++){
    if(id<CM12_CURRENT_MAP_BOUNDARY_FACE_OFFSETS[next]){break;}face=next;
  }
  let axis=face/2u;let side=face%2u;let dimensions=vec3u(${uintVector(layout.dimensions)});
  // Map filtering has consumed its nodal temporary. Reuse that disjoint
  // scratch for one scalar bound per physical finest-cell boundary face.
  // Composed Bernstein restrictions run in parallel, not in a single
  // invocation containing every face and every archived map link.
  var planePoint=vec3f(0.0);planePoint[axis]=select(0.0,f32(dimensions[axis]),side!=0u);
  if(cm12CurrentMapCoefficientBoundaryValue(planePoint,vec3f(1.0))[axis]==0.0){
    state[CM12_CURRENT_MAP_NODAL_BASE+id]=0.0;return;
  }
  let tangentU=select(0u,1u,axis==0u);let tangentV=select(2u,1u,axis==2u);
  let local=id-CM12_CURRENT_MAP_BOUNDARY_FACE_OFFSETS[face];
  var q=vec3i(0);q[tangentU]=i32(local%dimensions[tangentU]);q[tangentV]=i32(local/dimensions[tangentU]);
  let range=cm12CurrentMapBoundaryNormalRange(q,axis,side,cm12CurrentMapCandidateBank());
  let bound=max(0.0,select(range.y,-range.x,side!=0u));
  if(!(bound>=0.0&&bound<3.4e38)){cm12CurrentMapFail(1u,face);return;}
  state[CM12_CURRENT_MAP_NODAL_BASE+id]=bound;
}
@compute @workgroup_size(1)
fn boundCurrentMapPhysicalBoundary(@builtin(global_invocation_id)gid:vec3u){
  let face=gid.x;if(face>=6u){return;}
  var maximum=0.0;
  for(var id=CM12_CURRENT_MAP_BOUNDARY_FACE_OFFSETS[face];id<CM12_CURRENT_MAP_BOUNDARY_FACE_OFFSETS[face+1u];id++){
    maximum=max(maximum,state[CM12_CURRENT_MAP_NODAL_BASE+id]);
  }
  // Inward-only bounds, ordered x-low,x-high,y-low,y-high,z-low,z-high,
  // in fine-cell units. Outward motion expands the map image and cannot omit
  // source density. The omitted source-volume bound is each inward bound times
  // its original tangential domain area (q_seed <= 1); this is a numerical
  // coverage residual, not a declaration that the physical boundary is exact.
  state[CM12_CURRENT_MAP_BOUNDARY_BOUNDS_BASE+face]=maximum;
}
fn cm12CurrentMapIntervalProduct(a:vec2f,b:vec2f)->vec2f{
  let products=vec4f(a.x*b.x,a.x*b.y,a.y*b.x,a.y*b.y);
  let error=CM12_CURRENT_MAP_ROUNDOFF*max(1.0,max(max(abs(products.x),abs(products.y)),max(abs(products.z),abs(products.w))));
  return vec2f(min(min(products.x,products.y),min(products.z,products.w))-error,
    max(max(products.x,products.y),max(products.z,products.w))+error);
}
fn cm12CurrentMapIntervalDifference(a:vec2f,b:vec2f)->vec2f{
  let lower=a.x-b.y;let upper=a.y-b.x;
  let error=CM12_CURRENT_MAP_ROUNDOFF*max(1.0,max(abs(lower),abs(upper)));
  return vec2f(lower-error,upper+error);
}
fn cm12CurrentMapPositiveJacobian(values:array<vec3f,64>,scale:f32,sourceMagnitude:f32)->bool{
  var low=mat3x3f(vec3f(3.4e38),vec3f(3.4e38),vec3f(3.4e38));
  var high=mat3x3f(vec3f(-3.4e38),vec3f(-3.4e38),vec3f(-3.4e38));
  var magnitude=max(1.0,sourceMagnitude);var stride=1u;
  for(var i=0u;i<64u;i++){magnitude=max(magnitude,max(abs(values[i].x),max(abs(values[i].y),abs(values[i].z))));}
  for(var axis=0u;axis<3u;axis++){
    for(var i=0u;i<64u;i++){
      if((i/stride)%4u>=3u){continue;}
      let derivative=3.0*scale*(values[i+stride]-values[i]);
      low[axis]=min(low[axis],derivative);high[axis]=max(high[axis],derivative);
    }
    low[axis][axis]+=1.0;high[axis][axis]+=1.0;stride*=4u;
  }
  let error=CM12_CURRENT_MAP_ROUNDOFF*magnitude*scale;
  for(var axis=0u;axis<3u;axis++){low[axis]-=vec3f(error);high[axis]+=vec3f(error);}
  // The symmetric Jacobian is a cheap sufficient certificate where available.
  var monotone=true;
  for(var row=0u;row<3u;row++){
    var radius=0.0;
    for(var column=0u;column<3u;column++){
      if(column==row){continue;}
      radius+=0.5*max(abs(low[column][row]+low[row][column]),abs(high[column][row]+high[row][column]));
    }
    if(!(low[row][row]>radius+0.0001)){monotone=false;}
  }
  if(monotone){return true;}
  let a=vec2f(low[0][0],high[0][0]);let b=vec2f(low[1][0],high[1][0]);let c=vec2f(low[2][0],high[2][0]);
  let d=vec2f(low[0][1],high[0][1]);let e=vec2f(low[1][1],high[1][1]);let f=vec2f(low[2][1],high[2][1]);
  let g=vec2f(low[0][2],high[0][2]);let h=vec2f(low[1][2],high[1][2]);let j=vec2f(low[2][2],high[2][2]);
  let first=cm12CurrentMapIntervalProduct(a,cm12CurrentMapIntervalDifference(
    cm12CurrentMapIntervalProduct(e,j),cm12CurrentMapIntervalProduct(f,h)));
  let second=cm12CurrentMapIntervalProduct(b,cm12CurrentMapIntervalDifference(
    cm12CurrentMapIntervalProduct(d,j),cm12CurrentMapIntervalProduct(f,g)));
  let third=cm12CurrentMapIntervalProduct(c,cm12CurrentMapIntervalDifference(
    cm12CurrentMapIntervalProduct(d,h),cm12CurrentMapIntervalProduct(e,g)));
  let partial=cm12CurrentMapIntervalDifference(first,second);
  return partial.x+third.x>0.0001+CM12_CURRENT_MAP_ROUNDOFF*max(1.0,abs(partial.x)+abs(third.x));
}
fn cm12CurrentMapHalf(values:array<vec3f,64>,child:u32)->array<vec3f,64>{
  var current=values;var next:array<vec3f,64>;var stride=1u;
  for(var axis=0u;axis<3u;axis++){
    for(var base=0u;base<64u;base+=4u*stride){for(var offset=0u;offset<stride;offset++){
      let i=base+offset;let a=current[i];let b=current[i+stride];let c=current[i+2u*stride];let d=current[i+3u*stride];
      let ab=0.5*(a+b);let bc=0.5*(b+c);let cd=0.5*(c+d);
      let abc=0.5*(ab+bc);let bcd=0.5*(bc+cd);let middle=0.5*(abc+bcd);
      if((child&(1u<<axis))==0u){next[i]=a;next[i+stride]=ab;next[i+2u*stride]=abc;next[i+3u*stride]=middle;}
      else{next[i]=middle;next[i+stride]=bcd;next[i+2u*stride]=cd;next[i+3u*stride]=d;}
    }}
    current=next;stride*=4u;
  }
  return current;
}
fn cm12CurrentMapPatchCertified(values:array<vec3f,64>,sourceMagnitude:f32)->bool{
  if(cm12CurrentMapPositiveJacobian(values,CM12_CURRENT_MAP_INVERSE_SPACING,sourceMagnitude)){return true;}
  // Bounded subdivision tightens a sufficient interval certificate. Failure
  // means unresolved orientation and rejects publication; no sampled positive
  // determinant, coefficient clipping, or tolerance relaxation admits it.
  for(var child=0u;child<8u;child++){
    let half=cm12CurrentMapHalf(values,child);
    if(cm12CurrentMapPositiveJacobian(half,2.0*CM12_CURRENT_MAP_INVERSE_SPACING,sourceMagnitude)){continue;}
    for(var grandchild=0u;grandchild<8u;grandchild++){
      let quarter=cm12CurrentMapHalf(half,grandchild);
      if(cm12CurrentMapPositiveJacobian(quarter,4.0*CM12_CURRENT_MAP_INVERSE_SPACING,sourceMagnitude)){continue;}
      for(var greatGrandchild=0u;greatGrandchild<8u;greatGrandchild++){
        if(!cm12CurrentMapPositiveJacobian(cm12CurrentMapHalf(quarter,greatGrandchild),
          8.0*CM12_CURRENT_MAP_INVERSE_SPACING,sourceMagnitude)){return false;}
      }
    }
  }
  return true;
}
@compute @workgroup_size(64)
fn certifyCurrentMap(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=CM12_CURRENT_MAP_CELL_COUNT){return;}
  let coordinate=vec3u(id%CM12_CURRENT_MAP_CELLS.x,(id/CM12_CURRENT_MAP_CELLS.x)%CM12_CURRENT_MAP_CELLS.y,
    id/(CM12_CURRENT_MAP_CELLS.x*CM12_CURRENT_MAP_CELLS.y));
  let bank=cm12CurrentMapCandidateBank();
  if(any(coordinate<vec3u(2u))||any(coordinate+vec3u(3u)>=CM12_CURRENT_MAP_NODES)){
    for(var corner=0u;corner<8u;corner++){
      let node=coordinate+vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);
      if(cm12CurrentMapCollar(node)&&any(cm12CurrentMapRead(cm12CurrentMapCoefficientBase(bank),
        cm12CurrentMapNodeId(node))!=vec3f(0.0))){cm12CurrentMapFail(4u,id);return;}
    }
  }
  // Bound arithmetic from the original spline controls, before signed
  // cancellation in the Bernstein conversion or its subdivisions.
  var sourceMagnitude=1.0;
  for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
    let control=cm12CurrentMapControl(vec3i(coordinate)+vec3i(x-1,y-1,z-1),bank);
    sourceMagnitude=max(sourceMagnitude,max(abs(control.x),max(abs(control.y),abs(control.z))));
  }}}
  let values=cm12CurrentMapBernstein(vec3i(coordinate),bank);
  for(var i=0u;i<64u;i++){if(!cm12CurrentMapFinite(values[i])){cm12CurrentMapFail(1u,id);return;}}
  if(!cm12CurrentMapPatchCertified(values,sourceMagnitude)){cm12CurrentMapFail(3u,id);}
}
`;
}
