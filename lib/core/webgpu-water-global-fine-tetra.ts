import { makeOctreePowerCoarseLevelSetSampleWGSL } from "./octree-power-coarse-levelset-sample-abi";
import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";

export const FLUID_FINE_FILTERED_NORMALS_ENV = "FLUID_FINE_FILTERED_NORMALS";

/** Factor one keeps its established compensator; oversampled bands opt in so
 * existing image baselines remain unchanged until explicitly re-blessed. */
export function fineSurfaceFilteredNormalsEnabled(
  fineFactor: number,
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return fineFactor === 1 || ((fineFactor === 4 || fineFactor === 8)
    && environment?.[FLUID_FINE_FILTERED_NORMALS_ENV] === "1");
}

const filteredFineFactorsWGSL = fineSurfaceFilteredNormalsEnabled(4)
  ? "(factor==1u||factor==4u||factor==8u)"
  : "factor==1u";

const TETS = [
  [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
  [0, 7, 4, 6], [0, 5, 6, 4], [0, 5, 1, 6],
] as const;

/**
 * Reflection-equivariant cube contouring.  The twelve Cartesian cube edges
 * are the only scalar interpolation sites.  Per-face links therefore agree
 * on both sides of a shared face, unlike a fixed tetrahedral diagonal.  The
 * asymptotic sign test selects the two links on a four-crossing face; its
 * exact tie treats the face centre as air, a scalar convention which is
 * unchanged by every D4 transform.
 */
export const globalFineCubeContourWGSL = /* wgsl */ `
const CONTOUR_INVALID:u32=0xffffffffu;
const CONTOUR_QUANTIZER:f32=65536.;
const CONTOUR_EDGE_A=array<u32,12>(0u,1u,3u,0u,4u,5u,7u,4u,0u,1u,2u,3u);
const CONTOUR_EDGE_B=array<u32,12>(1u,2u,2u,3u,5u,6u,6u,7u,4u,5u,6u,7u);
const CONTOUR_CORNERS=array<vec3f,8>(vec3f(0,0,0),vec3f(1,0,0),vec3f(1,1,0),vec3f(0,1,0),vec3f(0,0,1),vec3f(1,0,1),vec3f(1,1,1),vec3f(0,1,1));
fn contourInside(v:f32)->bool{return v>=.5;}
fn contourCrosses(values:array<f32,8>,edgeId:u32)->bool{return contourInside(values[CONTOUR_EDGE_A[edgeId]])!=contourInside(values[CONTOUR_EDGE_B[edgeId]]);}
fn contourAddOne(a:ptr<function,array<u32,12>>,b:ptr<function,array<u32,12>>,x:u32,y:u32)->bool{if((*a)[x]==CONTOUR_INVALID){(*a)[x]=y;return true;}if((*a)[x]==y){return false;}if((*b)[x]==CONTOUR_INVALID){(*b)[x]=y;return true;}return false;}
fn contourLink(a:ptr<function,array<u32,12>>,b:ptr<function,array<u32,12>>,x:u32,y:u32)->bool{return x!=y&&contourAddOne(a,b,x,y)&&contourAddOne(a,b,y,x);}
fn contourFace(values:array<f32,8>,corners:vec4u,edges:vec4u,a:ptr<function,array<u32,12>>,b:ptr<function,array<u32,12>>)->bool{
  var crossing=array<u32,4>();var count=0u;
  for(var i=0u;i<4u;i+=1u){if(contourCrosses(values,edges[i])){crossing[count]=edges[i];count+=1u;}}
  if(count==0u){return true;}if(count==2u){return contourLink(a,b,crossing[0],crossing[1]);}if(count!=4u){return false;}
  let s0=values[corners.x]-.5;let s1=values[corners.y]-.5;let s2=values[corners.z]-.5;let s3=values[corners.w]-.5;
  let determinant=s0*s2-s1*s3;var centreInside=true;
  if(determinant>0.){centreInside=contourInside(values[corners.x]);}else if(determinant<0.){centreInside=contourInside(values[corners.y]);}
  var linked=0u;
  for(var i=0u;i<4u;i+=1u){if(contourInside(values[corners[i]])!=centreInside){if(!contourLink(a,b,edges[(i+3u)&3u],edges[i])){return false;}linked+=1u;}}
  return linked==2u;
}
fn buildContour(values:array<f32,8>,a:ptr<function,array<u32,12>>,b:ptr<function,array<u32,12>>)->u32{
  for(var i=0u;i<12u;i+=1u){(*a)[i]=CONTOUR_INVALID;(*b)[i]=CONTOUR_INVALID;}
  var mask=0u;for(var edgeId=0u;edgeId<12u;edgeId+=1u){if(contourCrosses(values,edgeId)){mask|=1u<<edgeId;}}
  if(mask==0u){return 0u;}
  var valid=true;
  valid=valid&&contourFace(values,vec4u(0,1,2,3),vec4u(0,1,2,3),a,b);
  valid=valid&&contourFace(values,vec4u(4,5,6,7),vec4u(4,5,6,7),a,b);
  valid=valid&&contourFace(values,vec4u(0,1,5,4),vec4u(0,9,4,8),a,b);
  valid=valid&&contourFace(values,vec4u(3,2,6,7),vec4u(2,10,6,11),a,b);
  valid=valid&&contourFace(values,vec4u(0,4,7,3),vec4u(8,7,11,3),a,b);
  valid=valid&&contourFace(values,vec4u(1,2,6,5),vec4u(1,10,5,9),a,b);
  if(!valid){return CONTOUR_INVALID;}return mask;
}
fn contourTriangleCount(values:array<f32,8>)->u32{
  var a:array<u32,12>;var b:array<u32,12>;let mask=buildContour(values,&a,&b);if(mask==0u||mask==CONTOUR_INVALID){return 0u;}
  var visited=0u;var total=0u;
  for(var start=0u;start<12u;start+=1u){let bit=1u<<start;if((mask&bit)==0u||(visited&bit)!=0u){continue;}var current=start;var previous=CONTOUR_INVALID;var closed=false;var rawEdges:array<u32,12>;var rawCount=0u;
    for(var step=0u;step<12u;step+=1u){let currentBit=1u<<current;if((visited&currentBit)!=0u){break;}visited|=currentBit;rawEdges[rawCount]=current;rawCount+=1u;let next=select(a[current],b[current],a[current]==previous);if(next==CONTOUR_INVALID){break;}previous=current;current=next;if(current==start){closed=true;break;}}
    if(!closed){return 0u;}var uniqueCount=0u;var firstPoint=vec3f(0);var previousPoint=vec3f(0);
    for(var local=0u;local<rawCount;local+=1u){let q=contourPoint(rawEdges[local],values);if(uniqueCount==0u||any(q!=previousPoint)){if(uniqueCount==0u){firstPoint=q;}previousPoint=q;uniqueCount+=1u;}}
    if(uniqueCount>1u&&all(previousPoint==firstPoint)){uniqueCount-=1u;}if(uniqueCount<3u){continue;}total+=uniqueCount;
  }
  return select(0u,total,visited==mask);
}
fn contourSnap(v:f32)->f32{return round(v*CONTOUR_QUANTIZER)/CONTOUR_QUANTIZER;}
fn contourPoint(edgeId:u32,values:array<f32,8>)->vec3f{let ia=CONTOUR_EDGE_A[edgeId];let ib=CONTOUR_EDGE_B[edgeId];let sa=values[ia]-.5;let sb=values[ib]-.5;let denominator=abs(sa)+abs(sb);if(denominator<=0.){return .5*(CONTOUR_CORNERS[ia]+CONTOUR_CORNERS[ib]);}let centred=contourSnap(.5*(abs(sa)-abs(sb))/denominator);return CONTOUR_CORNERS[ia]+(CONTOUR_CORNERS[ib]-CONTOUR_CORNERS[ia])*(.5+centred);}
fn wallFaceCorners(axis:u32,side:u32)->vec4u{if(axis==0u){return select(vec4u(0,3,7,4),vec4u(1,2,6,5),side!=0u);}if(axis==1u){return vec4u(0,1,5,4);}return select(vec4u(0,1,2,3),vec4u(4,5,6,7),side!=0u);}
fn wallFaceValues(values:array<f32,8>,descriptor:u32)->vec4f{let c=wallFaceCorners((descriptor>>14u)&3u,(descriptor&255u)-252u);return vec4f(values[c.x],values[c.y],values[c.z],values[c.w]);}
fn wallMask(face:vec4f)->u32{return select(0u,1u,face.x>=.5)|select(0u,2u,face.y>=.5)|select(0u,4u,face.z>=.5)|select(0u,8u,face.w>=.5);}
fn wallCorner(index:u32)->vec2f{return array<vec2f,4>(vec2f(0,0),vec2f(1,0),vec2f(1,1),vec2f(0,1))[index&3u];}
fn wallValue(face:vec4f,index:u32)->f32{return face[index&3u];}
fn wallEdgePoint(face:vec4f,index:u32)->vec2f{let next=(index+1u)&3u;let a=wallValue(face,index)-.5;let b=wallValue(face,next)-.5;let denominator=abs(a)+abs(b);let t=select(.5,contourSnap(abs(a)/denominator),denominator>0.);return mix(wallCorner(index),wallCorner(next),t);}
fn wallBoundaryCount(face:vec4f)->u32{var count=0u;for(var i=0u;i<4u;i+=1u){let next=(i+1u)&3u;let inside=wallValue(face,i)>=.5;let nextInside=wallValue(face,next)>=.5;count+=select(0u,1u,inside)+select(0u,1u,inside!=nextInside);}return count;}
fn wallBoundaryPoint(face:vec4f,wanted:u32)->vec2f{var cursor=0u;for(var i=0u;i<4u;i+=1u){let next=(i+1u)&3u;let inside=wallValue(face,i)>=.5;let nextInside=wallValue(face,next)>=.5;if(inside){if(cursor==wanted){return wallCorner(i);}cursor+=1u;}if(inside!=nextInside){if(cursor==wanted){return wallEdgePoint(face,i);}cursor+=1u;}}return vec2f(0);}
fn wallAmbiguousConnected(face:vec4f)->bool{let determinant=(face.x-.5)*(face.z-.5)-(face.y-.5)*(face.w-.5);if(determinant>0.){return face.x>=.5;}if(determinant<0.){return face.y>=.5;}return true;}
fn wallTriangleCount(values:array<f32,8>,descriptor:u32)->u32{let face=wallFaceValues(values,descriptor);let mask=wallMask(face);if(mask==0u){return 0u;}let ambiguous=mask==5u||mask==10u;if(ambiguous&&!wallAmbiguousConnected(face)){return 2u;}let count=wallBoundaryCount(face);if(ambiguous){return count;}return select(0u,count-2u,count>=3u);}
`;

const CONTOUR_EDGE_A = [0, 1, 3, 0, 4, 5, 7, 4, 0, 1, 2, 3] as const;
const CONTOUR_EDGE_B = [1, 2, 2, 3, 5, 6, 6, 7, 4, 5, 6, 7] as const;
const CONTOUR_FACES = [
  [[0, 1, 2, 3], [0, 1, 2, 3]], [[4, 5, 6, 7], [4, 5, 6, 7]],
  [[0, 1, 5, 4], [0, 9, 4, 8]], [[3, 2, 6, 7], [2, 10, 6, 11]],
  [[0, 4, 7, 3], [8, 7, 11, 3]], [[1, 2, 6, 5], [1, 10, 5, 9]],
] as const;

/** CPU mirror of the production edge interpolation, used to pin that native
 * adaptive phi retains sub-cell zero crossings rather than snapping to nodes. */
export function cubeContourEdgePoint(edge: number, values: readonly number[]): readonly [number, number, number] {
  if (!Number.isInteger(edge) || edge < 0 || edge >= 12 || values.length !== 8
    || values.some(value => !Number.isFinite(value))) throw new RangeError("invalid cube contour edge sample");
  const a = CONTOUR_EDGE_A[edge]!, b = CONTOUR_EDGE_B[edge]!;
  const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]] as const;
  const sa = values[a]! - .5, sb = values[b]! - .5;
  const denominator = Math.abs(sa) + Math.abs(sb);
  const centred = denominator <= 0 ? 0
    : Math.round((.5 * (Math.abs(sa) - Math.abs(sb)) / denominator) * 65_536) / 65_536;
  const t = .5 + centred;
  const point = (axis: number) => corners[a]![axis]! + (corners[b]![axis]! - corners[a]![axis]!) * t;
  return [point(0), point(1), point(2)];
}

export type WallFacePoint = readonly [number, number];
export type WallFaceTriangle = readonly [WallFacePoint, WallFacePoint, WallFacePoint];

/** CPU mirror of reserved codes 252/253: a scalar-clipped closed-tank face. Values are in
 * cyclic square order `(0,0),(1,0),(1,1),(0,1)` and liquid is `>= 0.5`. */
export function wallFaceContourTriangles(values: readonly number[]): readonly WallFaceTriangle[] {
  if (values.length !== 4 || values.some(value => !Number.isFinite(value))) {
    throw new RangeError("wall face contour needs four finite samples");
  }
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1]] as const;
  const inside = (index: number) => values[index]! >= 0.5;
  const edgePoint = (index: number): WallFacePoint => {
    const next = (index + 1) & 3;
    const a = values[index]! - 0.5, b = values[next]! - 0.5;
    const denominator = Math.abs(a) + Math.abs(b);
    const t = denominator > 0 ? Math.round(Math.abs(a) / denominator * 65_536) / 65_536 : 0.5;
    return [corners[index]![0] + t * (corners[next]![0] - corners[index]![0]),
      corners[index]![1] + t * (corners[next]![1] - corners[index]![1])];
  };
  const boundary: WallFacePoint[] = [];
  for (let index = 0; index < 4; index += 1) {
    const next = (index + 1) & 3;
    if (inside(index)) boundary.push(corners[index]);
    if (inside(index) !== inside(next)) boundary.push(edgePoint(index));
  }
  const mask = values.reduce((bits, _, index) => bits | (inside(index) ? 1 << index : 0), 0);
  if (mask === 0) return [];
  const ambiguous = mask === 5 || mask === 10;
  const determinant = (values[0]! - 0.5) * (values[2]! - 0.5)
    - (values[1]! - 0.5) * (values[3]! - 0.5);
  const connected = determinant > 0 ? inside(0) : determinant < 0 ? inside(1) : true;
  if (ambiguous && !connected) {
    const selected = mask === 5 ? [0, 2] : [1, 3];
    return selected.map(corner => [corners[corner], edgePoint(corner), edgePoint((corner + 3) & 3)]);
  }
  if (ambiguous) {
    const centre = [0, 1].map(axis => Math.round(boundary.reduce((sum, point) =>
      sum + point[axis]!, 0) / boundary.length * 65_536) / 65_536) as [number, number];
    return boundary.map((point, index) => [centre, point, boundary[(index + 1) % boundary.length]!]);
  }
  return boundary.slice(1, -1).map((point, index) => [boundary[0]!, point,
    boundary[index + 2]!]);
}

/** CPU mirror used by structural/D4 tests. Invalid face connectivity fails closed. */
export function cubeContourEdgeLoops(values: readonly number[]): readonly (readonly number[])[] {
  if (values.length !== 8 || values.some(value => !Number.isFinite(value))) return [];
  const inside = (corner: number) => values[corner]! >= 0.5;
  const crosses = (edge: number) => inside(CONTOUR_EDGE_A[edge]!) !== inside(CONTOUR_EDGE_B[edge]!);
  const adjacent = Array.from({ length: 12 }, () => [] as number[]);
  const link = (x: number, y: number) => {
    if (x === y || adjacent[x]!.includes(y) || adjacent[y]!.includes(x)
      || adjacent[x]!.length >= 2 || adjacent[y]!.length >= 2) return false;
    adjacent[x]!.push(y); adjacent[y]!.push(x); return true;
  };
  for (const [corners, edges] of CONTOUR_FACES) {
    const crossing = edges.filter(crosses);
    if (crossing.length === 0) continue;
    if (crossing.length === 2) { if (!link(crossing[0]!, crossing[1]!)) return []; continue; }
    if (crossing.length !== 4) return [];
    const signed = corners.map(corner => values[corner]! - 0.5);
    const determinant = signed[0]! * signed[2]! - signed[1]! * signed[3]!;
    const centreInside = determinant > 0 ? inside(corners[0]!)
      : determinant < 0 ? inside(corners[1]!) : true;
    let linked = 0;
    for (let index = 0; index < 4; index += 1) if (inside(corners[index]!) !== centreInside) {
      if (!link(edges[(index + 3) & 3]!, edges[index]!)) return [];
      linked += 1;
    }
    if (linked !== 2) return [];
  }
  const crossing = CONTOUR_EDGE_A.map((_, edge) => edge).filter(crosses);
  if (crossing.some(edge => adjacent[edge]!.length !== 2)) return [];
  const visited = new Set<number>(); const loops: number[][] = [];
  for (const start of crossing) {
    if (visited.has(start)) continue;
    const loop: number[] = []; let current = start; let previous = -1;
    while (!visited.has(current) && loop.length < 12) {
      visited.add(current); loop.push(current);
      const next = adjacent[current]![0] === previous ? adjacent[current]![1]! : adjacent[current]![0]!;
      previous = current; current = next;
    }
    if (current !== start || loop.length < 3) return [];
    loops.push(loop);
  }
  return visited.size === crossing.length ? loops : [];
}

export const globalFineClassifiedScanShader = /* wgsl */ `
struct V{position:vec4f,normal:vec4f}struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>,globalFineAuthorityLatch:atomic<u32>,meshPublicationGeneration:atomic<u32>}struct P{sample:vec4u,bricks:vec4u,table:vec4u,settings:vec4f,cell:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(3)var<storage,read_write>out:array<V>;@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;@group(0)@binding(10)var<uniform>p:P;
${globalFineCubeContourWGSL}
var<workgroup>laneOffsets:array<u32,256>;
@compute @workgroup_size(256)fn scanGlobalFineTriangles(@builtin(local_invocation_index)lid:u32){
  let published=atomicLoad(&args.vertexAllocator)!=0xffffffffu;
  let count=select(0u,min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),min(arrayLength(&offsets)/6u,arrayLength(&values)/2u))),published);
  let base=count/256u;let extra=count%256u;let begin=lid*base+min(lid,extra);
  let end=begin+base+select(0u,1u,lid<extra);var localTotal=0u;
  for(var i=begin;i<end;i+=1u){let descriptor=cubes[i].y>>16u;let code=descriptor&255u;let wallFace=code==252u||code==253u;let clipMask=(descriptor>>8u)&7u;let lo=values[i*2u];let hi=values[i*2u+1u];let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);if(wallFace){localTotal+=3u*wallTriangleCount(samples,descriptor);}else if(clipMask!=0u){localTotal+=6u;}else{localTotal+=3u*contourTriangleCount(samples);}}
  laneOffsets[lid]=localTotal;workgroupBarrier();
  if(lid==0u){var total=0u;for(var lane=0u;lane<256u;lane+=1u){let subtotal=laneOffsets[lane];laneOffsets[lane]=total;total+=subtotal;}if(published){let capacity=arrayLength(&out)-arrayLength(&out)%3u;atomicStore(&args.vertexCount,min(total,capacity));atomicStore(&args.vertexAllocator,total);if(p.table.y!=6u){atomicStore(&args.meshPublicationGeneration,p.table.w);}}}
  workgroupBarrier();var cursor=laneOffsets[lid];
  for(var i=begin;i<end;i+=1u){let descriptor=cubes[i].y>>16u;let code=descriptor&255u;let wallFace=code==252u||code==253u;let clipMask=(descriptor>>8u)&7u;let lo=values[i*2u];let hi=values[i*2u+1u];let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);var triangleCount=select(contourTriangleCount(samples),2u,clipMask!=0u);if(wallFace){triangleCount=wallTriangleCount(samples,descriptor);}for(var lane=0u;lane<6u;lane+=1u){offsets[i*6u+lane]=cursor;if(wallFace){cursor+=select(0u,3u,lane<triangleCount);}else{cursor+=3u*min(2u,triangleCount-min(triangleCount,2u*lane));}}}
}
`;

/** Production scan variant that publishes the exact tetrahedron emission grid. */
export const globalFineClassifiedIndirectScanShader = globalFineClassifiedScanShader
  .replace(
    "@group(0)@binding(10)var<uniform>p:P;",
    "@group(0)@binding(10)var<uniform>p:P;@group(0)@binding(11)var<storage,read_write>emitDispatch:array<u32>;",
  )
  .replace(
    "if(published){let capacity=arrayLength(&out)-arrayLength(&out)%3u;",
    "emitDispatch[0]=(count+63u)/64u;emitDispatch[1]=6u;emitDispatch[2]=1u;if(published){let capacity=arrayLength(&out)-arrayLength(&out)%3u;",
  );

const cornerPosition = [
  "vec3f(0,0,0)", "vec3f(1,0,0)", "vec3f(1,1,0)", "vec3f(0,1,0)",
  "vec3f(0,0,1)", "vec3f(1,0,1)", "vec3f(1,1,1)", "vec3f(0,1,1)",
] as const;

const factorOneFilteredNormalShader = /* wgsl */ `
@group(0)@binding(8)var<storage,read>fineWorklist:array<u32>;
@group(0)@binding(9)var<storage,read>fineSamples:array<u32>;
@group(0)@binding(12)var<storage,read>metadata:array<u32>;
${makeOctreePowerCoarseLevelSetSampleWGSL(16)}
const INVALID:u32=0xffffffffu;
${fineLevelSetPackedSampleWGSL("fineSamples", false)}
fn finitePhi(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn finePage(key:u32)->u32{
  if(p.table.y!=7u||arrayLength(&fineWorklist)<7u||fineWorklist[0]!=p.table.w
    ||fineWorklist[2]!=p.table.z||(fineWorklist[3]&3u)!=3u
    ||fineWorklist[5]!=1u||fineWorklist[6]!=1u){return INVALID;}
  let logicalCount=p.bricks.x*p.bricks.y*p.bricks.z;
  if(key>=logicalCount){return INVALID;}
  if((fineWorklist[3]&0x80000000u)!=0u){
    let count=min(fineWorklist[1],p.table.z);var low=0u;var high=count;
    loop{if(low>=high){break;}let middle=low+(high-low)/2u;let base=middle*4u;
      if(base+2u>=arrayLength(&metadata)){return INVALID;}let candidate=metadata[base+1u];
      if(candidate<key){low=middle+1u;}else{high=middle;}}
    let base=low*4u;return select(INVALID,low,low<count&&base+2u<arrayLength(&metadata)
      &&metadata[base]==low&&metadata[base+1u]==key&&metadata[base+2u]==p.table.w);
  }
  let directoryBase=7u+p.table.z;
  if(directoryBase+key>=arrayLength(&fineWorklist)){return INVALID;}
  let id=fineWorklist[directoryBase+key];let base=id*4u;
  return select(INVALID,id,id<p.table.z&&base+2u<arrayLength(&metadata)
    &&metadata[base]==id&&metadata[base+1u]==key&&metadata[base+2u]==p.table.w);
}
fn compactBrickFineResolution()->u32{
  let encoded=(fineWorklist[3]>>16u)&31u;return select(8u,encoded,encoded!=0u);
}
fn compactPagesPerSolverAxis()->u32{
  return max(1u,compactBrickFineResolution()/max(1u,p.sample.w));
}
fn compactSourceSpanLog(source:u32)->u32{
  if(compactPagesPerSolverAxis()==4u){
    return select(0u,(source>>24u)&31u,(source&0x80000000u)!=0u);
  }
  return source>>27u;
}
fn compactSampleAddress(q:vec3u)->vec2u{
  let r=max(1u,p.sample.w);let pageCoordinate=q/r;
  let exactKey=pageCoordinate.x+p.bricks.x*(pageCoordinate.y+p.bricks.y*pageCoordinate.z);
  let exact=finePage(exactKey);
  if(exact!=INVALID){
    if((fineWorklist[3]&0x80000000u)==0u){let local=q-pageCoordinate*r;
      return vec2u(exact,local.x+r*(local.y+r*local.z));}
    let spanLog=compactSourceSpanLog(metadata[4u*exact+3u]);
    if(spanLog==0u){let local=q-pageCoordinate*r;
      return vec2u(exact,local.x+r*(local.y+r*local.z));}
    let scale=compactPagesPerSolverAxis()*(1u<<spanLog);let origin=pageCoordinate*r;
    let local=min((q-origin)/scale,vec3u(r-1u));
    return vec2u(exact,local.x+r*(local.y+r*local.z));
  }
  let maximumSpanLog=(fineWorklist[3]>>8u)&31u;
  for(var spanLog=1u;spanLog<=maximumSpanLog;spanLog+=1u){
    let span=1u<<spanLog;let pageSpan=compactPagesPerSolverAxis()*span;
    let originPage=(pageCoordinate/pageSpan)*pageSpan;
    let key=originPage.x+p.bricks.x*(originPage.y+p.bricks.y*originPage.z);
    let id=finePage(key);if(id==INVALID){continue;}
    if(compactSourceSpanLog(metadata[4u*id+3u])!=spanLog){continue;}
    let origin=originPage*r;let local=min((q-origin)/pageSpan,vec3u(r-1u));
    return vec2u(id,local.x+r*(local.y+r*local.z));
  }
  return vec2u(INVALID);
}
fn signedPhi(qi:vec3i)->f32{
  // The optical wall cubes keep their existing normals. For the adjoining free
  // surface, repeat the nearest in-domain signed sample so the smoothing kernel
  // has a stable one-sided extension instead of reading the artificial wall cap.
  let dims=vec3i(p.sample.xyz);
  // Factor-one adaptive publication is nodal. Sampling it through the legacy
  // cell-centred frame shifts the reconstruction by half a cell and is why the
  // native contour path originally disabled filtered normals altogether. Keep
  // the emitted vertices in their established frame, but reconstruct normals
  // directly from the accepted nodal SDF at integer grid coordinates.
  if(p.table.y==6u){
    let q=clamp(qi,vec3i(0),dims);
    return sampleAdaptiveCoarseOctreePhiAtGrid(vec3f(q));
  }
  let q=vec3u(clamp(qi,vec3i(0),dims-vec3i(1)));
  let address=compactSampleAddress(q);
  if(address.x!=INVALID){
    let index=address.x*p.bricks.w+address.y;
    if(index<arrayLength(&fineSamples)
      &&(finePackedFlags(index)&1u)!=0u&&finitePhi(finePackedPhi(index))){return finePackedPhi(index);}
  }
  if((fineWorklist[3]&0x80000000u)!=0u){return 4.0*p.settings.w;}
  return sampleCoarseOctreePhi(p.settings.xyz+(vec3f(q)+vec3f(.5))*p.settings.w);
}
fn filteredNormalAt(lattice:vec3f,fallback:vec3f,filterEnabled:bool)->vec3f{
  // Factor-4/8 filtering is compiled in only under the fidelity flag, keeping
  // the established baseline byte-for-byte when the flag is absent.
  let factor=u32(round(p.cell.x));
  if(!(${filteredFineFactorsWGSL})||!filterEnabled){return fallback;}
  // Native nodal vertices carry a +0.5 render-lattice adapter in clipped().
  // Removing that adapter recovers their integer SDF lattice. Legacy optical
  // samples retain the established one-cell reconstruction frame.
  let x=lattice-select(vec3f(1.0),vec3f(0.5),p.table.y==6u);
  let center=vec3i(round(x));
  var weightSum=0.0;var phiSum=0.0;
  var derivativeWeightSum=vec3f(0.0);var phiDerivativeSum=vec3f(0.0);
  var valid=true;
  // Gradient of a normalized 3x3x3 Gaussian reconstruction evaluated at the
  // emitted vertex. Subtracting the normalization derivative makes constants
  // reproduce exactly even when the vertex is off-centre in the sample stencil.
  for(var oz=0;oz<3;oz+=1){for(var oy=0;oy<3;oy+=1){for(var ox=0;ox<3;ox+=1){
    let q=center+vec3i(ox-1,oy-1,oz-1);
    let delta=vec3f(q)-x;
    let weight=exp(-.5*dot(delta,delta)/(.85*.85));
    let value=signedPhi(q);
    if(!finitePhi(value)){valid=false;}
    weightSum+=weight;phiSum+=weight*value;
    derivativeWeightSum+=weight*delta;
    phiDerivativeSum+=weight*value*delta;
  }}}
  if(!valid||weightSum<=1e-8){return fallback;}
  let derivative=phiDerivativeSum*weightSum-phiSum*derivativeWeightSum;
  let size=vec3f(p.sample.xyz);
  let gradient=vec3f(derivative.x*size.x/u.container.x,
    derivative.y*size.y/u.container.y,derivative.z*size.z/u.container.z);
  return select(fallback,normalize(gradient),length(gradient)>1e-8);
}
`;

/** Shared verbatim by production and the focused Dawn rectangle reproducer. */
export const globalFineDirectSharpPatchWGSL = /* wgsl */ `
fn directPatch(cursor:ptr<function,u32>,base:vec3f,scale:f32,descriptor:u32,n:vec3f){let mask=(descriptor>>8u)&7u;let axis=(descriptor>>14u)&3u;let code=descriptor&255u;let nativeFace=code==2u||code==3u;let plane=select(.5,select(0.,1.,code==3u),nativeFace);var lx=0.0;var ly=0.0;var lz=0.0;var hx=1.0;var hy=1.0;var hz=1.0;if((mask&1u)!=0u&&!nativeFace){let high=((descriptor>>11u)&1u)!=0u;lx=select(0.0,0.5,high);hx=select(0.5,1.0,high);}if((mask&2u)!=0u&&!nativeFace){let high=((descriptor>>12u)&1u)!=0u;ly=select(0.0,0.5,high);hy=select(0.5,1.0,high);}if((mask&4u)!=0u&&!nativeFace){let high=((descriptor>>13u)&1u)!=0u;lz=select(0.0,0.5,high);hz=select(0.5,1.0,high);}var a=vec3f(0);var b=vec3f(0);var c=vec3f(0);var d=vec3f(0);if(axis==0u){a=vec3f(plane,ly,lz);b=vec3f(plane,hy,lz);c=vec3f(plane,hy,hz);d=vec3f(plane,ly,hz);}else if(axis==1u){a=vec3f(lx,plane,lz);b=vec3f(lx,plane,hz);c=vec3f(hx,plane,hz);d=vec3f(hx,plane,lz);}else{a=vec3f(lx,ly,plane);b=vec3f(hx,ly,plane);c=vec3f(hx,hy,plane);d=vec3f(lx,hy,plane);}let shift=select(vec3f(0),vec3f(.5),nativeFace);a=base+scale*a+shift;b=base+scale*b+shift;c=base+scale*c+shift;d=base+scale*d+shift;tri(cursor,a,b,c,n,false);tri(cursor,a,c,d,n,false);}
fn wallPoint3(q:vec2f,axis:u32,plane:f32)->vec3f{if(axis==0u){return vec3f(plane,q.x,q.y);}if(axis==1u){return vec3f(q.x,plane,q.y);}return vec3f(q.x,q.y,plane);}
fn emitWallLane(cursor:ptr<function,u32>,base:vec3f,descriptor:u32,values:array<f32,8>,lane:u32){
  let face=wallFaceValues(values,descriptor);let mask=wallMask(face);let axis=(descriptor>>14u)&3u;let side=(descriptor&255u)-252u;let plane=f32(side);let scaleCode=max(1u,(descriptor>>8u)&63u);let wallScale=f32(1u<<(scaleCode-1u));let ambiguous=mask==5u||mask==10u;let connected=wallAmbiguousConnected(face);let count=wallBoundaryCount(face);let triangles=wallTriangleCount(values,descriptor);if(lane>=triangles){return;}
  var a=vec2f(0);var b=vec2f(0);var c=vec2f(0);
  if(ambiguous&&!connected){let corner=select(select(0u,2u,lane!=0u),select(1u,3u,lane!=0u),mask==10u);a=wallCorner(corner);b=wallEdgePoint(face,corner);c=wallEdgePoint(face,(corner+3u)&3u);}
  else if(ambiguous){var centre=vec2f(0);for(var i=0u;i<count;i+=1u){centre+=wallBoundaryPoint(face,i);}centre=vec2f(contourSnap(centre.x/f32(count)),contourSnap(centre.y/f32(count)));a=centre;b=wallBoundaryPoint(face,lane);c=wallBoundaryPoint(face,(lane+1u)%count);}
  else{a=wallBoundaryPoint(face,0u);b=wallBoundaryPoint(face,lane+1u);c=wallBoundaryPoint(face,lane+2u);}
  var normal=vec3f(0);normal[axis]=select(-1.,1.,side!=0u);let shift=vec3f(.5);tri(cursor,base+wallPoint3(wallScale*a,axis,plane)+shift,base+wallPoint3(wallScale*b,axis,plane)+shift,base+wallPoint3(wallScale*c,axis,plane)+shift,normal,false);
}
`;

const globalFineContourEmitWGSL = /* wgsl */ `
fn contourOrdered4(a:f32,b:f32,c:f32,d:f32)->f32{var terms=array<f32,4>(a,b,c,d);for(var i=1u;i<4u;i+=1u){let term=terms[i];var j=i;loop{if(j==0u||terms[j-1u]<=term){break;}terms[j]=terms[j-1u];j-=1u;}terms[j]=term;}return(terms[0]+terms[1])+(terms[2]+terms[3]);}
fn contourOrdered3(a:f32,b:f32,c:f32)->f32{var terms=array<f32,3>(a,b,c);if(terms[1]<terms[0]){let swap=terms[0];terms[0]=terms[1];terms[1]=swap;}if(terms[2]<terms[1]){let swap=terms[1];terms[1]=terms[2];terms[2]=swap;}if(terms[1]<terms[0]){let swap=terms[0];terms[0]=terms[1];terms[1]=swap;}return(terms[0]+terms[1])+terms[2];}
fn contourNormal(values:array<f32,8>,size:vec3f,container:vec3f)->vec3f{let gx=.25*(contourOrdered4(values[1],values[2],values[5],values[6])-contourOrdered4(values[0],values[3],values[4],values[7]));let gy=.25*(contourOrdered4(values[2],values[3],values[6],values[7])-contourOrdered4(values[0],values[1],values[4],values[5]));let gz=.25*(contourOrdered4(values[4],values[5],values[6],values[7])-contourOrdered4(values[0],values[1],values[2],values[3]));let gradient=vec3f(gx*size.x/container.x,gy*size.y/container.y,gz*size.z/container.z);let magnitude=sqrt(contourOrdered3(gradient.x*gradient.x,gradient.y*gradient.y,gradient.z*gradient.z));if(!(magnitude>1e-5)){return vec3f(0,1,0);}let raw=-gradient/magnitude;return vec3f(select(0.,contourSnap(raw.x),gradient.x!=0.),select(0.,contourSnap(raw.y),gradient.y!=0.),select(0.,contourSnap(raw.z),gradient.z!=0.));}
fn emitContourLane(cursor:ptr<function,u32>,base:vec3f,scale:f32,descriptor:u32,values:array<f32,8>,n:vec3f,filterEnabled:bool,lane:u32){
  var adjacentA:array<u32,12>;var adjacentB:array<u32,12>;let mask=buildContour(values,&adjacentA,&adjacentB);if(mask==0u||mask==CONTOUR_INVALID){return;}
  let firstTriangle=2u*lane;let lastTriangle=firstTriangle+2u;var triangleIndex=0u;var visited=0u;
  for(var start=0u;start<12u;start+=1u){let startBit=1u<<start;if((mask&startBit)==0u||(visited&startBit)!=0u){continue;}
    var rawEdges:array<u32,12>;var rawCount=0u;var current=start;var previous=CONTOUR_INVALID;
    for(var step=0u;step<12u;step+=1u){let bit=1u<<current;if((visited&bit)!=0u){break;}visited|=bit;rawEdges[rawCount]=current;rawCount+=1u;let next=select(adjacentA[current],adjacentB[current],adjacentA[current]==previous);previous=current;current=next;if(current==start){break;}}
    var loopEdges:array<u32,12>;var loopCount=0u;var centredSum=vec3f(0);var firstPoint=vec3f(0);var previousPoint=vec3f(0);
    for(var raw=0u;raw<rawCount;raw+=1u){let q=contourPoint(rawEdges[raw],values);if(loopCount==0u||any(q!=previousPoint)){if(loopCount==0u){firstPoint=q;}loopEdges[loopCount]=rawEdges[raw];loopCount+=1u;previousPoint=q;centredSum+=q-vec3f(.5);}}
    if(loopCount>1u&&all(previousPoint==firstPoint)){loopCount-=1u;centredSum-=previousPoint-vec3f(.5);}if(loopCount<3u){continue;}let centre=vec3f(.5)+vec3f(contourSnap(centredSum.x/f32(loopCount)),contourSnap(centredSum.y/f32(loopCount)),contourSnap(centredSum.z/f32(loopCount)));
    for(var local=0u;local<loopCount;local+=1u){if(triangleIndex>=firstTriangle&&triangleIndex<lastTriangle){let qa=contourPoint(loopEdges[local],values);let qb=contourPoint(loopEdges[(local+1u)%loopCount],values);tri(cursor,clipped(base,scale,qa,descriptor),clipped(base,scale,qb,descriptor),clipped(base,scale,centre,descriptor),n,filterEnabled);}triangleIndex+=1u;}
  }
}
`;

function emitShader(tetraIndex: number): string {
  const [a, b, c, d] = TETS[tetraIndex];
  return /* wgsl */ `
struct U{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f}struct V{position:vec4f,normal:vec4f}struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>,globalFineAuthorityLatch:atomic<u32>,meshPublicationGeneration:atomic<u32>}struct P{sample:vec4u,bricks:vec4u,table:vec4u,settings:vec4f,cell:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(3)var<storage,read_write>out:array<V>;@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;@group(0)@binding(10)var<uniform>p:P;
${factorOneFilteredNormalShader}
${globalFineCubeContourWGSL}
fn world(q:vec3f)->vec3f{let dims=vec3f(p.sample.xyz);let center=.5*(dims+vec3f(1));let horizontal=clamp((q-center)/dims,vec3f(-.5),vec3f(.5));let vertical=clamp((q.y-.5)/dims.y,0.,1.);return vec3f(horizontal.x*u.container.x,vertical*u.container.y,horizontal.z*u.container.z);}fn edge(x:vec3f,y:vec3f,a:f32,b:f32)->vec3f{return mix(x,y,clamp((.5-a)/(b-a),0.,1.));}fn countTet(a:f32,b:f32,c:f32,d:f32)->u32{let n=select(0u,1u,a>=.5)+select(0u,1u,b>=.5)+select(0u,1u,c>=.5)+select(0u,1u,d>=.5);if(n==0u||n==4u){return 0u;}return select(3u,6u,n==2u);}
fn tri(cursor:ptr<function,u32>,a:vec3f,b:vec3f,c:vec3f,n:vec3f,filterEnabled:bool){let first=*cursor;*cursor=first+3u;let limit=min(atomicLoad(&args.vertexCount),arrayLength(&out));if(first+3u>limit){return;}let x=V(vec4f(world(a),1),vec4f(filteredNormalAt(a,n,filterEnabled),0));let y=V(vec4f(world(b),1),vec4f(filteredNormalAt(b,n,filterEnabled),0));let z=V(vec4f(world(c),1),vec4f(filteredNormalAt(c,n,filterEnabled),0));out[first]=x;if(dot(cross(y.position.xyz-x.position.xyz,z.position.xyz-x.position.xyz),n)>=0.){out[first+1u]=y;out[first+2u]=z;}else{out[first+1u]=z;out[first+2u]=y;}}
fn tet(cursor:ptr<function,u32>,pa:vec3f,pb:vec3f,pc:vec3f,pd:vec3f,va:f32,vb:f32,vc:f32,vd:f32,n:vec3f,filterEnabled:bool){let m=select(0u,1u,va>=.5)|select(0u,2u,vb>=.5)|select(0u,4u,vc>=.5)|select(0u,8u,vd>=.5);if(m==1u||m==14u){tri(cursor,edge(pa,pb,va,vb),edge(pa,pc,va,vc),edge(pa,pd,va,vd),n,filterEnabled);}else if(m==2u||m==13u){tri(cursor,edge(pb,pa,vb,va),edge(pb,pd,vb,vd),edge(pb,pc,vb,vc),n,filterEnabled);}else if(m==4u||m==11u){tri(cursor,edge(pc,pa,vc,va),edge(pc,pb,vc,vb),edge(pc,pd,vc,vd),n,filterEnabled);}else if(m==8u||m==7u){tri(cursor,edge(pd,pa,vd,va),edge(pd,pc,vd,vc),edge(pd,pb,vd,vb),n,filterEnabled);}else if(m==3u||m==12u){let ac=edge(pa,pc,va,vc);let ad=edge(pa,pd,va,vd);let bc=edge(pb,pc,vb,vc);let bd=edge(pb,pd,vb,vd);tri(cursor,ac,bc,bd,n,filterEnabled);tri(cursor,ac,bd,ad,n,filterEnabled);}else if(m==5u||m==10u){let ab=edge(pa,pb,va,vb);let ad=edge(pa,pd,va,vd);let cb=edge(pc,pb,vc,vb);let cd=edge(pc,pd,vc,vd);tri(cursor,ab,cb,cd,n,filterEnabled);tri(cursor,ab,cd,ad,n,filterEnabled);}else if(m==6u||m==9u){let ba=edge(pb,pa,vb,va);let bd=edge(pb,pd,vb,vd);let ca=edge(pc,pa,vc,va);let cd=edge(pc,pd,vc,vd);tri(cursor,ba,ca,cd,n,filterEnabled);tri(cursor,ba,cd,bd,n,filterEnabled);}}
fn clipped(base:vec3f,scale:f32,q:vec3f,descriptor:u32)->vec3f{let nodal=(descriptor&255u)==0u;var r=base+scale*q+select(vec3f(0),vec3f(.5),nodal);let mask=(descriptor>>8u)&7u;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1u<<axis))!=0u){let high=((descriptor>>(11u+axis))&1u)!=0u;r[axis]=base[axis]+scale*select(.5*q[axis],.5+.5*q[axis],high)+select(0.,.5,nodal);}}return r;}
${globalFineDirectSharpPatchWGSL}
${globalFineContourEmitWGSL}
@compute @workgroup_size(64)fn emitGlobalFineTetra${tetraIndex}(@builtin(global_invocation_id)g:vec3u){let i=g.x;let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)/6u));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let packed=cubes[i];let descriptor=packed.y>>16u;let code=descriptor&255u;let wallFace=code==252u||code==253u;let nativeFace=(code==2u||code==3u)&&((descriptor>>8u)&7u)!=0u;let scale=select(f32(max(1u,code)),1.,nativeFace||wallFace);let base=vec3f(f32(packed.x&0xffffu),f32(packed.y&0xffffu),f32(packed.x>>16u));let lo=values[i*2u];let hi=values[i*2u+1u];let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);let n=contourNormal(samples,vec3f(p.sample.xyz),u.container.xyz);let clipMask=(descriptor>>8u)&7u;let filterEnabled=!nativeFace&&!wallFace&&clipMask==0u&&base.x>0.0&&base.y>0.0&&base.z>0.0&&base.x<f32(p.sample.x)&&base.y<f32(p.sample.y)&&base.z<f32(p.sample.z);var cursor=offsets[i*6u+${tetraIndex}u];if(wallFace){emitWallLane(&cursor,base,descriptor,samples,${tetraIndex}u);return;}if(clipMask!=0u){if(${tetraIndex}u==0u){directPatch(&cursor,base,scale,descriptor,n);}return;}emitContourLane(&cursor,base,scale,descriptor,samples,n,filterEnabled,${tetraIndex}u);}
`;
}

export const globalFineClassifiedEmitShaders = TETS.map((_, index) => emitShader(index));

/** One bounded 2-D dispatch: x selects a classified cube, y one of six exact tetrahedra. */
export const globalFineClassifiedEmitShader = emitShader(0).replace(
  /@compute @workgroup_size\(64\)fn emitGlobalFineTetra0[\s\S]*$/,
  `@compute @workgroup_size(64)fn emitGlobalFineTetrahedra(@builtin(global_invocation_id)g:vec3u){let i=g.x;let lane=g.y;if(lane>=6u){return;}let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)/6u));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let packed=cubes[i];let descriptor=packed.y>>16u;let code=descriptor&255u;let wallFace=code==252u||code==253u;let nativeFace=(code==2u||code==3u)&&((descriptor>>8u)&7u)!=0u;let scale=select(f32(max(1u,code)),1.,nativeFace||wallFace);let base=vec3f(f32(packed.x&0xffffu),f32(packed.y&0xffffu),f32(packed.x>>16u));let lo=values[i*2u];let hi=values[i*2u+1u];let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);let n=contourNormal(samples,vec3f(p.sample.xyz),u.container.xyz);let clipMask=(descriptor>>8u)&7u;let filterEnabled=!nativeFace&&!wallFace&&clipMask==0u&&base.x>0.0&&base.y>0.0&&base.z>0.0&&base.x<f32(p.sample.x)&&base.y<f32(p.sample.y)&&base.z<f32(p.sample.z);var cursor=offsets[i*6u+lane];if(wallFace){emitWallLane(&cursor,base,descriptor,samples,lane);return;}if(clipMask!=0u){if(lane==0u){directPatch(&cursor,base,scale,descriptor,n);}return;}emitContourLane(&cursor,base,scale,descriptor,samples,n,filterEnabled,lane);}`,
);
