/** Diagnostic-only transfer setup: 192 cell-face reach predicates address at
 * most four tile words. Reuse those words without changing face arithmetic. */
import assert from "node:assert/strict";
import type {CompactMomentumCapture} from "./compact-momentum";

export function transferReachCacheSource(source:string,early=false):string{
 const dims=source.includes("fn UM_D(")?"UM_D()":"UM_D";
 const start=source.indexOf("fn transferPropertiesJob(");assert.ok(start>=0);
 const end=source.indexOf("\n}",start);assert.ok(end>start);
 let body=source.slice(start,end);
 const origin="let origin=4u*umTileCoord(owner.tile);";
 assert.ok(body.includes(origin));
 const loads=early?/* wgsl */`
 if(lane==0u){
  var reach=vec4u(nbBandReach(origin)&1u,0u,0u,0u);
  for(var axis=0u;axis<3u;axis++){
   var q=origin;q[axis]+=4u;
   if(q[axis]<${dims}[axis]){reach[axis+1u]=nbBandReach(q)&1u;}
  }
  nbTransferReach=reach;
 }
 let reach=workgroupUniformLoad(&nbTransferReach);
 if(all(reach==vec4u(0))){
  let cell=origin+umCorner(lane,4u);
  udStore_output(vec3i(cell),udLoad_velocity(vec3i(cell)));
  return;
 }
`:/* wgsl */`
 if(lane<4u){
  var q=origin;if(lane!=0u){q[lane-1u]+=4u;}
  var reach=0u;
  if(all(q<${dims})){reach=nbBandReach(q)&1u;}
  nbTransferReach[lane]=reach;
 }
`;
 body=body.replace(origin,origin+loads);
 const predicate="(nbBandReach(origin+low)&1u)!=0u";
 assert.ok(body.includes(predicate));
 body=body.replace(predicate,`${early?"reach":"nbTransferReach"}[select(0u,axis+1u,local[axis]==3u)]!=0u`);
 return source.slice(0,start)+`var<workgroup> nbTransferReach:${early?"vec4u":"array<u32,4>"};\n`+body+source.slice(end);
}

export async function createTransferReachReplay(device:GPUDevice,frame:any,capture:CompactMomentumCapture,early=false){
 const flip=frame.narrowBandFlip;
 const original=flip.pipelines.get("transferProperties") as GPUComputePipeline;
 assert.ok(original);
 const descriptor=capture.descriptors.get(original)!;assert.ok(descriptor);
 const source=capture.sources.get(descriptor.compute.module)!;assert.ok(source);
 const code=transferReachCacheSource(source,early);
 const module=device.createShaderModule({label:"Transfer cached four-tile reach",code});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
 const pipeline=await device.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module}});
 return {code,run(encode:()=>void){
  flip.pipelines.set("transferProperties",pipeline);
  try{encode();}finally{flip.pipelines.set("transferProperties",original);}
 }};
}
