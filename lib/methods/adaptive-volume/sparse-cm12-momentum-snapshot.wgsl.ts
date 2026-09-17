import { momentumSnapshotLayout } from "./sparse-cm12-momentum-snapshot";

/** `scalars` is rebound to the snapshot only during capture and momentum
 * preparation. Hash construction borrows dead conditioning scratch. */
export function createMomentumSnapshotWGSL(cells: number, rows: number, incidences: number): string {
  const l=momentumSnapshotLayout(cells,rows,incidences);
  return /* wgsl */ `
const MOM_C:u32=${l.cellBase}u;
const MOM_F:u32=${l.rowBase}u;
const MOM_I:u32=${l.incidenceBase}u;
const MOM_HASH:u32=${l.hashBase}u;
const MOM_HASH_SIZE:u32=${l.hashCapacity}u;
fn momentumWord(at:u32)->u32{return bitcast<u32>(scalars[at]);}
fn momentumVector(at:u32)->vec3f{return vec3f(scalars[at],scalars[at+1u],scalars[at+2u]);}
fn momentumHash(lower:vec3i,level:u32)->u32{
  let q=bitcast<vec3u>(lower);var h=(q.x*0x8da6b343u)^(q.y*0xd8163841u)^(q.z*0xcb1ab31fu)^(level*0x9e3779b9u);
  h^=h>>16u;return h&(MOM_HASH_SIZE-1u);
}
fn momentumSnapshotReady()->bool{return p.frame.w>0.5&&momentumWord(0u)==0x4d4f4d31u;}
fn momentumSnapshotClamp(q:vec3f,margin:vec3f)->vec3f{
  let lower=momentumVector(4u)+margin;return clamp(q,lower,max(lower,momentumVector(8u)-margin));}
fn momentumSnapshotOwner(qInput:vec3f)->u32{
  let q=momentumSnapshotClamp(qInput,vec3f(1e-4));
  for(var level=0u;level<=momentumWord(3u);level+=1u){
    let scale=f32(1u<<level);let lower=vec3i(floor(q/scale))*i32(1u<<level);
    let hash=momentumHash(lower,level);
    for(var probe=0u;probe<MOM_HASH_SIZE;probe+=1u){
      let word=momentumWord(MOM_HASH+((hash+probe)&(MOM_HASH_SIZE-1u)));if(word==0u){break;}
      let cell=word-1u;let at=MOM_C+8u*cell;
      if((momentumWord(at+3u)&31u)==level&&all(vec3i(floor(momentumVector(at)/scale))*i32(1u<<level)==lower)
        &&all(q>=momentumVector(at))&&all(q<momentumVector(at)+momentumVector(at+4u))){return cell;}
    }
  }
  return INVALID;
}
fn momentumCellIncidences(cell:u32)->vec2u{
  let at=MOM_C+8u*cell;let begin=momentumWord(at+7u);return vec2u(begin,begin+(momentumWord(at+3u)>>5u));}
@compute @workgroup_size(1)fn momentumSnapshotBegin(){
  scalars[0]=0.0;scalars[1]=bitcast<f32>(cnxAcceptedCellCount());scalars[2]=bitcast<f32>(cnxAcceptedRowCount());
  atomicStore(&conditioning[MOM_HASH_SIZE],0);
  let lower=cm12WorldFineLower();let upper=cm12WorldFineUpper();
  for(var axis=0u;axis<3u;axis+=1u){scalars[4u+axis]=lower[axis];scalars[8u+axis]=upper[axis];}
}
@compute @workgroup_size(64)fn momentumSnapshotCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=cnxAcceptedCellInvocation(gid.x);if(cell==INVALID){return;}
  let widths=cellWidths(cell);let lower=cellCenter(cell)-.5*widths;
  let level=u32(ceil(log2(max(1.0,max(widths.x,max(widths.y,widths.z))))));
  let range=cnxCellIncidenceRangeUnchecked(cell);let at=MOM_C+8u*cell;
  for(var axis=0u;axis<3u;axis+=1u){scalars[at+axis]=lower[axis];scalars[at+4u+axis]=widths[axis];}
  scalars[at+3u]=bitcast<f32>(level|((range.y-range.x)<<5u));scalars[at+7u]=bitcast<f32>(range.x);
  for(var entry=range.x;entry<range.y;entry+=1u){scalars[MOM_I+entry]=bitcast<f32>(cnxIncidenceRowOrdinalUnchecked(entry));}
  atomicMax(&conditioning[MOM_HASH_SIZE],i32(level));
  let scale=f32(1u<<level);
  let hash=momentumHash(vec3i(floor(lower/scale))*i32(1u<<level),level);
  for(var probe=0u;probe<MOM_HASH_SIZE;probe+=1u){
    let result=atomicCompareExchangeWeak(&conditioning[(hash+probe)&(MOM_HASH_SIZE-1u)],0,i32(cell+1u));
    if(result.exchanged){return;}
    // A weak CAS can spuriously fail on an empty slot: retry it rather than
    // leaving a hole that would terminate the immutable lookup early.
    if(result.old_value==0){probe-=1u;}
  }
}
@compute @workgroup_size(64)fn momentumSnapshotFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=cnxAcceptedRowInvocation(gid.x);if(row==INVALID){return;}
  let center=rowCenter(row);let terms=cnxRowTermRangeByOrdinalUnchecked(gid.x);var widths=vec3f(1.0);
  for(var t=terms.x;t<terms.y;t+=1u){widths=max(widths,cellWidths(cnxRowTermCellUnchecked(t)));}
  let face=partials[AIR_F+row];let open=rowOpenFraction(row);var value=face.x;
  if(!rowSeparatingFromClosedWorld(row)){
    value=select(rowSolidVelocity(row),(value-(1.0-open)*rowSolidVelocity(row))/max(open,1e-8),open>1e-8);}
  let at=MOM_F+8u*gid.x;
  for(var axis=0u;axis<3u;axis+=1u){scalars[at+axis]=center[axis];scalars[at+4u+axis]=widths[axis];}
  scalars[at+3u]=bitcast<f32>((cnxRowPackedMetadataByOrdinal(gid.x)&3u)|select(0u,4u,face.w>0.0)
    |select(0u,8u,open<=1e-8&&!rowSeparatingFromClosedWorld(row)));
  scalars[at+7u]=value;
}
@compute @workgroup_size(1)fn momentumSnapshotSeal(){
  scalars[3]=bitcast<f32>(u32(atomicLoad(&conditioning[MOM_HASH_SIZE])));
  scalars[0]=bitcast<f32>(select(0u,0x4d4f4d31u,airTransportReady()));
}
fn momentumSnapshotSample(position:vec3f)->vec3f{
  let point=momentumSnapshotClamp(position,vec3f(0.0));let home=momentumSnapshotOwner(point);
  if(home==INVALID){return vec3f(0.0);}
  let spans=max(momentumVector(MOM_C+8u*home+4u),vec3f(1.0));
  let stencilPoint=momentumSnapshotClamp(point,spans);let boundaryAxes=abs(stencilPoint-point)>vec3f(1e-6);
  let lower=floor(stencilPoint/spans-vec3f(.5));var owners:array<u32,8>;
  for(var k=0u;k<8u;k+=1u){let offset=vec3f(f32(k&1u),f32((k>>1u)&1u),f32((k>>2u)&1u));
    owners[k]=momentumSnapshotOwner(spans*(lower+offset+vec3f(.5)));}
  var adaptive=false;
  for(var k=0u;k<8u;k+=1u){if(owners[k]!=INVALID){adaptive=adaptive||any(momentumVector(MOM_C+8u*owners[k]+4u)!=spans);}}
  var moments:array<mat4x4f,3>;var rhs:array<vec4f,3>;
  var wallWeights=vec3f(0.0);var wallValues=vec3f(0.0);
  for(var k=0u;k<8u;k+=1u){let cell=owners[k];if(cell==INVALID){continue;}
    var duplicate=false;for(var previous=0u;previous<k;previous+=1u){duplicate=duplicate||owners[previous]==cell;}
    if(duplicate){continue;}let incidence=momentumCellIncidences(cell);
    for(var entry=incidence.x;entry<incidence.y;entry+=1u){let ordinal=momentumWord(MOM_I+entry);var seen=false;
      for(var previous=0u;previous<k;previous+=1u){if(owners[previous]==INVALID){continue;}
        let prior=momentumCellIncidences(owners[previous]);
        for(var i=prior.x;i<prior.y;i+=1u){seen=seen||momentumWord(MOM_I+i)==ordinal;}}
      if(seen){continue;}let at=MOM_F+8u*ordinal;let descriptor=momentumWord(at+3u);if((descriptor&4u)==0u){continue;}
      let axis=descriptor&3u;let center=momentumVector(at);let widths=momentumVector(at+4u);let delta=(center-point)/spans;
      let support=select(max(widths,spans),max(widths,2.0*spans),boundaryAxes|vec3<bool>(adaptive));
      let tent=max(vec3f(0.0),vec3f(1.0)-abs(center-point)/support);let weight=tent.x*tent.y*tent.z;if(weight<=0.0){continue;}
      if((descriptor&8u)!=0u&&abs(center[axis]-point[axis])<=1e-6){
        wallWeights[axis]+=weight;wallValues[axis]+=weight*scalars[at+7u];}
      let basis=vec4f(1.0,delta);rhs[axis]+=weight*scalars[at+7u]*basis;
      for(var column=0u;column<4u;column+=1u){moments[axis][column]+=weight*basis[column]*basis;}
    }
  }
  var result=airFitVelocity(moments,rhs);
  for(var axis=0u;axis<3u;axis+=1u){if(wallWeights[axis]>0.0){result[axis]=wallValues[axis]/wallWeights[axis];}}
  return result;
}
`;
}
