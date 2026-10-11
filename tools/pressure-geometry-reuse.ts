/** Diagnostic source transforms for two independent pressure-stage savings.
 * Neither changes production, ownership, or numerical parameters. */
import assert from "node:assert/strict";

/** Held diagnostic: this coarse-only predicate is insufficient. First-pass
 * particle classification and coarse-face transfer also read metric nodes.
 * Do not benchmark or promote it until those readers have a certificate. */
export function pressureMetricCullWGSL(code:string):string {
 void code;
 throw new Error("Metric culling is held pending classification/transfer reader coverage");
}

/** select evaluates both operands. The explicit branch retains the selected
 * result but avoids evaluating the pure plane-box integral for nonplanar
 * fields. No reductions or floating-point expressions are reordered. */
export function lazyPressureFractionWGSL(code:string):string {
 const original="let fraction=select(fill/8.0,geometricPlaneBoxFraction(gradient,-centre,vec3f(1)),residual<=1e-4*(1.0+magnitude));";
 assert.equal(code.split(original).length,2);
 return code.replace(original,"var fraction=fill/8.0;if(residual<=1e-4*(1.0+magnitude)){fraction=geometricPlaneBoxFraction(gradient,-centre,vec3f(1));}");
}

/** The solid record's tile mean is exactly zero only when all 64 open-cell
 * fractions are zero. Its sums are nonnegative; each nonzero input is at
 * least 2^-27 (body eighths times 1 minus a bounded f32 terrain fraction),
 * so division by 64 cannot underflow. The cut sum therefore has no terms.
 * Keep the original centre-phi arithmetic, including nonfinite corners. */
export function closedPressureTilesWGSL(code:string):string {
 const guard="if(owner.width!=1u||umTileOpen(owner.tile)>=0.99999){return;}";
 assert.equal(code.split(guard).length,2);
 code=code.replace(guard,"if(owner.width!=1u){return;}let tileOpen=umTileOpen(owner.tile);if(tileOpen==0.0||tileOpen>=0.99999){return;}");
 const centre="var centre:array<f32,8>;for(var j=0u;j<8u;j++){centre[j]=vertices[j]*0.125;}";
 assert.equal(code.split(centre).length,2);
 return code.replace(centre,centre+"\n if(open==0.0){return vec2f(0.0,d4Sum8(centre));}");
}

/** A closed h cell's finite fill is multiplied by zero. Preserve its exact
 * centre and the origin lane's separate pressure-tile evaluation. The f32
 * magnitude certificate (at most 2^120) leaves ample headroom for the finite
 * probe/gradient sums; nonfinite and extreme values keep the original path.
 * This arithmetic guard is independent of scene dimensions or parameters. */
export function closedPressureCellsWGSL(code:string):string {
 code=closedPressureTilesWGSL(code);
 const evaluate="var value=gcEvaluate(owner,vertices,open,false);";
 assert.equal(code.split(evaluate).length,2);
 code=code.replace(evaluate,/* wgsl */`
 var cellOpen=1.0;if(owner.width==1u){cellOpen=umCellOpen(vec3i(origin));}
 var value=vec2f(0.0);var skipped=false;
 if(owner.width==1u&&cellOpen==0.0){
  var centre:array<f32,8>;var magnitudeBits=0u;
  for(var j=0u;j<8u;j++){centre[j]=vertices[j]*0.125;magnitudeBits=max(magnitudeBits,bitcast<u32>(vertices[j])&0x7fffffffu);}
  let centrePhi=d4Sum8(centre);
  if(magnitudeBits<=0x7b800000u){value=vec2f(0.0,centrePhi);skipped=true;}
 }
 if(!skipped){value=gcEvaluate(owner,vertices,open,false);}
`);
 const scale="value.x*select(1.0,umCellOpen(vec3i(origin)),owner.width==1u)";
 assert.equal(code.split(scale).length,2);
 return code.replace(scale,"value.x*cellOpen");
}

/** Untimed workload census in the otherwise-unused full-launch changes
 * header. The probe clears/captures it separately and restores the snapshot
 * before any measured work; changed-tile launches must never use this form. */
export function pressurePlaneCoverageWGSL(code:string):string {
 const term=code.indexOf("fn umSurfaceTargetOpenTerm("),plane=code.indexOf("var samples:array<f32,8>",term),planeEnd=code.indexOf("  let f=0.25*",plane);
 assert.ok(term>=0&&plane>term&&planeEnd>plane);
 const begin=code.indexOf("fn gcShare(owner:UMOwner){"),end=code.indexOf("fn gcSharedFill()",begin);
 assert.ok(begin>=0&&end>begin);
 let share=code.slice(begin,end);
 share=share.replace("fn gcShare(owner:UMOwner){","fn gcShare(owner:UMOwner){\n if(owner.width==1u&&owner.lane==0u){atomicAdd(&changes[0],1u);}");
 share=share.replace(" let base=umTileCoord(owner.tile)*4u;"," if(owner.lane==0u){atomicAdd(&changes[1],1u);if(umTileOpen(owner.tile)==0.0){atomicAdd(&changes[7],1u);}}\n let base=umTileCoord(owner.tile)*4u;");
 const assignment=" gcTerms[owner.lane]=umSurfaceTargetOpenTerm";
 assert.equal(share.split(assignment).length,2);
 share=share.replace(assignment,/* wgsl */`
 if(owner.lane==0u){
  atomicAdd(&changes[2],1u);let vertices=corners;
  ${code.slice(plane,planeEnd)}
  if(residual<=1e-4*(1.0+magnitude)){atomicAdd(&changes[4],1u);}else{atomicAdd(&changes[5],1u);}
 }
 let cell=vec3u(owner.lane&3u,(owner.lane>>2u)&3u,owner.lane>>4u);
 if(umCellOpen(vec3i(base+cell))>0.0){atomicAdd(&changes[3],1u);}
`+assignment);
 code=code.slice(0,begin)+share+code.slice(end);
 const near="else if(!(positive&&4.0*(high-low)<=low)){";
 assert.equal(code.split(near).length,2);
 code=code.replace(near,near+"\n if(owner.width==4u&&open<0.99999&&!terms){atomicAdd(&changes[6],1u);}");
 const centre="var centre:array<f32,8>;for(var j=0u;j<8u;j++){centre[j]=vertices[j]*0.125;}";
 assert.equal(code.split(centre).length,2);
 code=code.replace(centre,centre+"\n if(owner.width==4u&&open==0.0&&!terms){atomicAdd(&changes[8],1u);}");
 const owner="fn gcOwner(owner:UMOwner,terms:bool){\n let origin=umOrigin(owner);";
 assert.equal(code.split(owner).length,2);
 code=code.replace(owner,owner+"\n if(owner.width==1u&&umCellOpen(vec3i(origin))==0.0){atomicAdd(&changes[9],1u);}");
 const fraction="let fraction=select(fill/8.0,geometricPlaneBoxFraction(gradient,-centre,vec3f(1)),residual<=1e-4*(1.0+magnitude));";
 assert.equal(code.split(fraction).length,2);
 return code.replace(fraction,"if(owner.width==1u){atomicAdd(&changes[10u+u32(residual<=1e-4*(1.0+magnitude))],1u);}\n "+fraction);
}

/** A subgroup leader computes the original pressure plane, then broadcasts
 * its unchanged values. There is no added workgroup storage or barrier; the
 * original ordered 64-term reduction remains in place. Fine jobs contain
 * one tile regardless of subgroup size. Coarse jobs do not consume terms.
 * Call only on a device with subgroups; retain the original pipeline otherwise. */
export function subgroupPressurePlaneWGSL(code:string,lazyCorners=true,cooperative=false):string {
 const term=code.indexOf("fn umSurfaceTargetOpenTerm("),plane=code.indexOf("var samples:array<f32,8>",term),planeEnd=code.indexOf("  let f=0.25*",plane);
 assert.ok(term>=0&&plane>term&&planeEnd>plane);
 const planeMath=code.slice(plane,planeEnd);
 const begin=code.indexOf("fn gcShare(owner:UMOwner){"),end=code.indexOf("fn gcSharedFill()",begin);
 assert.ok(begin>=0&&end>begin);
 const original=code.slice(begin,end),assignment=original.indexOf(" gcTerms[owner.lane]=umSurfaceTargetOpenTerm");
 assert.ok(assignment>0);
 let prepare=original.slice(0,assignment).replace("fn gcShare(owner:UMOwner){",/* wgsl */`struct GCSubgroupPlane {low:vec4f,high:vec4f,plane:vec4f,flags:u32}
fn gcPrepareSubgroupPlane(owner:UMOwner)->GCSubgroupPlane{
 var result:GCSubgroupPlane;`).replaceAll("return;","return result;");
 prepare+=/* wgsl */`
 ${cooperative?"":`let vertices=corners;\n${planeMath}`}
 result.low=vec4f(corners[0],corners[1],corners[2],corners[3]);
 result.high=vec4f(corners[4],corners[5],corners[6],corners[7]);
 ${cooperative?"result.flags=1u;":"result.plane=vec4f(gradient,centre);result.flags=1u|select(0u,2u,residual<=1e-4*(1.0+magnitude));"}
 return result;
}
${cooperative?original.replace("fn gcShare(","fn gcShareOriginal("):""}
// Broadcast flags are identical within their subgroup. Tint's analysis
// does not infer that subgroup-only uniformity through the leader branch;
// all broadcasts below are reached by either every lane or no lane of the
// subgroup. Participating subgroups reach the first flags broadcast whole.
@diagnostic(off,subgroup_uniformity)
fn gcShare(owner:UMOwner,subgroupLane:u32${cooperative?",subgroupSize:u32":""}){
 ${cooperative?"if(subgroupSize<8u){gcShareOriginal(owner);return;}":""}
 var local:GCSubgroupPlane;if(subgroupLane==0u){local=gcPrepareSubgroupPlane(owner);}
 var flags=subgroupBroadcastFirst(local.flags);
 if((flags&1u)==0u){return;}
 ${cooperative?/* wgsl */`
 let low=subgroupBroadcastFirst(local.low);let high=subgroupBroadcastFirst(local.high);
 let corners=array<f32,8>(low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w);
 let probe=umTargetSample(corners,vec3f(0.25)+0.5*vec3f(uvCorner(subgroupLane&7u)));
 // Lanes 0..7 each evaluate one original probe. Broadcast the exact values
 // into the original serial centre, gradient and residual loops.
 let samples=array<f32,8>(${Array.from({length:8},(_,k)=>`subgroupBroadcast(probe,${k}u)`).join(",")});
 var centre=0.0;var magnitude=0.0;
 for(var k=0u;k<8u;k++){centre+=0.125*samples[k];magnitude=max(magnitude,abs(samples[k]));}
 ${planeMath.slice(planeMath.indexOf("  var gradient="))}
 let plane=vec4f(gradient,centre);flags=1u|select(0u,2u,residual<=1e-4*(1.0+magnitude));
 `:/* wgsl */`
 let plane=subgroupBroadcastFirst(local.plane);
 ${lazyCorners?"var low:vec4f;var high:vec4f;if(flags==1u){low=subgroupBroadcastFirst(local.low);high=subgroupBroadcastFirst(local.high);}":"let low=subgroupBroadcastFirst(local.low);let high=subgroupBroadcastFirst(local.high);"}
 `}
 let base=umTileCoord(owner.tile)*4u;
 let cell=vec3u(owner.lane&3u,(owner.lane>>2u)&3u,owner.lane>>4u);
 let open=umCellOpen(vec3i(base+cell));
 if(open<=0.0){gcTerms[owner.lane]=0.0;return;}
 let f=0.25*(vec3f(cell)+vec3f(0.5));
 if((flags&2u)!=0u){gcTerms[owner.lane]=open*geometricPlaneBoxFraction(plane.xyz,-(plane.w+dot(plane.xyz,f-vec3f(0.5))),vec3f(0.25));}
 else{
  let vertices=array<f32,8>(low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w);
  let value=umTargetSample(vertices,f);gcTerms[owner.lane]=open*select(select(0.0,1.0,value<0.0),0.5,value==0.0);
 }
}
`;
 code=code.slice(0,begin)+prepare+code.slice(end);
 for(const [before,after] of [
  ["gcShare(owner);workgroupBarrier();",`gcShare(owner,subgroupLane${cooperative?",subgroupSize":""});workgroupBarrier();`],
  ["fn geometryJob(gid:vec3u)",`fn geometryJob(gid:vec3u,subgroupLane:u32${cooperative?",subgroupSize:u32":""})`],
  ["geometryJob(vec3u(umJob*64u+umLane,0u,0u));",`geometryJob(vec3u(umJob*64u+umLane,0u,0u),subgroupLane${cooperative?",subgroupSize":""});`],
 ] as const){assert.equal(code.split(before).length,2,before);code=code.replace(before,after);}
 const signature=/(@compute\s+@workgroup_size\(64\)\s+fn geometry\([^\n]+)(\)\s*\{)/;
 assert.ok(signature.test(code));code=code.replace(signature,`$1,@builtin(subgroup_invocation_id) subgroupLane:u32${cooperative?",@builtin(subgroup_size) subgroupSize:u32":""}$2`);
 return code.includes("enable subgroups;")?code:"enable subgroups;\n"+code;
}

/** Each h tile's 64 open-cell terms use one identical trilinear plane.
 * Prepare it once, retaining the original probe/gradient/residual order,
 * then leave the 64 distinct terms and their final ordered sum unchanged. */
export function sharedPressurePlaneWGSL(code:string):string {
 const term=code.indexOf("fn umSurfaceTargetOpenTerm("),plane=code.indexOf("var samples:array<f32,8>",term),planeEnd=code.indexOf("  let f=0.25*",plane);
 assert.ok(term>=0&&plane>term&&planeEnd>plane);
 const planeMath=code.slice(plane,planeEnd);
 const begin=code.indexOf("fn gcShare(owner:UMOwner){"),end=code.indexOf("fn gcSharedFill()",begin);
 assert.ok(begin>=0&&end>begin);
 const original=code.slice(begin,end),assignment=original.indexOf(" gcTerms[owner.lane]=umSurfaceTargetOpenTerm");
 assert.ok(assignment>0);
 let prepare=original.slice(0,assignment).replace("fn gcShare(owner:UMOwner){","fn gcPreparePlane(owner:UMOwner){\n if(owner.lane!=0u){return;}");
 prepare=prepare.replace(" if(owner.width!=1u||umTileOpen(owner.tile)>=0.99999){return;}"," if(owner.width!=1u){return;}gcPlaneActive=0u;\n if(umTileOpen(owner.tile)>=0.99999){return;}");
 assert.ok(prepare.includes("gcPlaneActive=0u;"));
 prepare+=/* wgsl */`
 for(var j=0u;j<8u;j++){gcPlaneCorners[j]=corners[j];}
 let vertices=corners;
 ${planeMath}
 gcPlane=vec4f(gradient,centre);gcPlanar=u32(residual<=1e-4*(1.0+magnitude));gcPlaneActive=1u;
}
fn gcShare(owner:UMOwner){
 if(owner.width!=1u||gcPlaneActive==0u){return;}
 let base=umTileCoord(owner.tile)*4u;
 let cell=vec3u(owner.lane&3u,(owner.lane>>2u)&3u,owner.lane>>4u);
 let open=umCellOpen(vec3i(base+cell));
 if(open<=0.0){gcTerms[owner.lane]=0.0;return;}
 let f=0.25*(vec3f(cell)+vec3f(0.5));
 if(gcPlanar!=0u){gcTerms[owner.lane]=open*geometricPlaneBoxFraction(gcPlane.xyz,-(gcPlane.w+dot(gcPlane.xyz,f-vec3f(0.5))),vec3f(0.25));}
 else{
  var vertices:array<f32,8>;for(var j=0u;j<8u;j++){vertices[j]=gcPlaneCorners[j];}
  let value=umTargetSample(vertices,f);gcTerms[owner.lane]=open*select(select(0.0,1.0,value<0.0),0.5,value==0.0);
 }
}
`;
 code=code.slice(0,begin)+/* wgsl */`var<workgroup> gcPlaneCorners:array<f32,8>;
var<workgroup> gcPlane:vec4f;
var<workgroup> gcPlanar:u32;
var<workgroup> gcPlaneActive:u32;
`+prepare+code.slice(end);
 const invoke="gcShare(owner);workgroupBarrier();";assert.equal(code.split(invoke).length,2);
 return code.replace(invoke,"gcPreparePlane(owner);workgroupBarrier();gcShare(owner);workgroupBarrier();");
}
