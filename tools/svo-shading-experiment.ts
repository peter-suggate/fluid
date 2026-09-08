/** Test-only shader surgery. Never imported by the production renderer. */
export function shadingExperimentSource(code:string,variant:string):string {
 if(!code.includes('fn dryLightingMain'))return code;
 if(variant.includes('cone-only'))code=code.replaceAll('if((dry.materialPublication.w&4u)!=0u){','if(true){');
 if(variant.includes('opaque'))code=code.replace('if(dryHitThinDielectric(hit)){return shadeDryThinDielectric(hit,ro,rd);}','');
 if(variant.includes('visibility-guide'))code=code.replace('return materialMatches&&metadata==dryPrepassHitMetadata(hit)', 'return (materialMatches||dry.tuningCounts2.w==4u)&&metadata==dryPrepassHitMetadata(hit)');
 if(variant.includes('no-edge'))code=code.replace('shaded*=dryVoxelFaceEdgeFactor(position,hit.normal,hit.t,hit.fieldSource);','');
 if(variant.includes('cheap-guide')){
  const start=code.indexOf('fn dryPrepassUseExactReceiver('),end=code.indexOf('fn dryPrepassRecoverExactReceiver(',start);
  if(start>=0){let body=code.slice(start,end);
   body=body.replace('let depthWeight=exp(-24.0*abs(geometry.x-depth)/max(depth,1e-3));', 'if(abs(geometry.x-depth)>0.057762265*max(depth,1e-3)){return false;}');
   body=body.replace('let normalWeight=pow(max(dot(normal,dryPrepassDecodeNormal(geometry.yz)),0.0),8.0);', 'let normalDot=dot(normal,dryPrepassDecodeNormal(geometry.yz));');
   body=body.replace('if(depthWeight<0.25||normalWeight<0.25)', 'if(normalDot<0.840896415)');
   code=code.slice(0,start)+body+code.slice(end);
  }
 }
 if(variant.includes('one-light')){
  const channelStart=code.indexOf('fn dryPrepassChannel('),channelEnd=code.indexOf('fn dryPrepassReceiverCompatible(',channelStart);
  if(channelStart>=0)code=code.slice(0,channelStart)+'fn dryPrepassChannel(index:u32)->f32{return dryPrepassData0.y;}\n'+code.slice(channelEnd);
  const start=code.indexOf('fn shadeDryOpaque('),end=code.indexOf('fn dryVoxelExit_m(',start);
  let body=code.slice(start,end).replace(/lightIndex<\d+u/,'lightIndex<1u').replace(/let sampleCount=select\(select\(1u,max\(dry.tuningCounts1.x,dry.tuningCounts0.w\),area\),1u,globalIllumination\);/,'let sampleCount=1u;');
  body=body.replace('let light=dryLighting.lights[lightIndex];','var light=dryLighting.lights[lightIndex];light.identity.x=SVO_LIGHT_DIRECTIONAL;');
  code=code.slice(0,start)+body+code.slice(end);
 }
 if(variant.includes('diffuse')){
  code=code.replace('direct+=shadeUnifiedSurface(directClosure,lighting);','direct+=surface.baseColor*sample.radiance*visibility*max(dot(hit.normal,sample.towardLight),0.0)/(UNIFIED_PI*f32(sampleCount));');
  code=code.replace('let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);','let diffuseEnergy=vec3f(1.0);');
  code=code.replaceAll('let specularEnvironment=dryEnvironment(reflected,surface.roughness)*environmentBrdf;','let specularEnvironment=vec3f(0.0);');
 }
 if(variant.includes('lambert'))code=code.replaceAll('let diffuseEnergy=max(vec3f(0.0),vec3f(1.0)-environmentBrdf);','let diffuseEnergy=vec3f(1.0);');
 if(variant==='split'&&code.includes('fn dryPrepassResolve(')){
  code='override DRY_FAST:bool=false;override DRY_SLOW:bool=false;\n'+code;
  const start=code.indexOf('@fragment fn dryLightingMain'),end=code.indexOf('@fragment fn drySkyLightingMain',start);
  let entry=code.slice(start,end);
  entry=entry.replace('if(glassKey>0u){',`if(DRY_FAST&&(dryPrepassState!=1u||dryHitThinDielectric(opaque)||glassKey>0u)){discard;}
    if(DRY_SLOW&&dryPrepassState==1u&&!dryHitThinDielectric(opaque)&&glassKey==0u){discard;}
    if(!DRY_FAST&&glassKey>0u){`);
  entry=entry.replace('if(glassVisible){','if(!DRY_FAST&&glassVisible){');
  code=code.slice(0,start)+entry+code.slice(end);
  code=code.replace('if(dryHitThinDielectric(hit)){return shadeDryThinDielectric(hit,ro,rd);}','if(!DRY_FAST&&dryHitThinDielectric(hit)){return shadeDryThinDielectric(hit,ro,rd);}');
  code=code.replaceAll('if((dry.materialPublication.w&4u)!=0u){','if(DRY_FAST||(dry.materialPublication.w&4u)!=0u){');
  const light=code.indexOf('fn dryLightVisibilitySolid(');
  code=code.slice(0,light)+code.slice(light).replace('let coneCell_m=max(dry.mapping.cellSize.x,','if(DRY_FAST){return vec3f(0.0);}\n    let coneCell_m=max(dry.mapping.cellSize.x,');
  const contact=code.indexOf('fn dryContactVisibilitySolid('),tail=code.slice(contact);
  code=code.slice(0,contact)+tail.replace('let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(1.0);}var visibility=0.0;', 'if(DRY_FAST){return vec3f(0.0);}\n    let radius=dryContactVisibilityRadius();if(radius<=0.0){return vec3f(1.0);}var visibility=0.0;');
 }
 return code;
}

export function shadingExperimentEncoder(encoder:GPUCommandEncoder,pipelines:Map<GPURenderPipeline,readonly GPURenderPipeline[]>):GPUCommandEncoder {
 return new Proxy(encoder,{get(t,k){
  if(k==='beginRenderPass')return (d:GPURenderPassDescriptor)=>{
   const pass=t.beginRenderPass(d);let active:readonly GPURenderPipeline[]|undefined;
   return new Proxy(pass,{get(p,key){
    if(key==='setPipeline')return (pipeline:GPURenderPipeline)=>{active=pipelines.get(pipeline);p.setPipeline(pipeline);};
    if(key==='draw')return (...args:Parameters<GPURenderPassEncoder['draw']>)=>{if(active){for(const pipeline of active){p.setPipeline(pipeline);p.draw(...args);}}else p.draw(...args);};
    const v=Reflect.get(p,key,p);return typeof v==='function'?v.bind(p):v;
   }});
  };
  const v=Reflect.get(t,k,t);return typeof v==='function'?v.bind(t):v;
 }});
}
