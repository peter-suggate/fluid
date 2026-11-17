import {clothAffineCCDShadowWGSL} from './cloth-affine-ccd-wgsl';

/** Diagnostic harness, intentionally outside the application's imports. */
export async function createAffineCCDShadow(device:GPUDevice,x:Float32Array<ArrayBuffer>,d:Float32Array<ArrayBuffer>,ids:Uint32Array<ArrayBuffer>,kind:'vt'|'ee',thickness:number,workgroupSize:32|64|128,budget=256,sameTimeEndpoints=false,externalPairs?:{queue:GPUBuffer;stats:GPUBuffer;indirect:GPUBuffer;capacity:number},projectedSupports=false,advancement=false,currentBounds=false){
  if((!externalPairs&&!ids.length)||ids.length%4||x.length!==d.length||x.length%4||!Number.isFinite(thickness)||thickness<0||!Number.isInteger(budget)||budget<1)throw new Error('Invalid shadow CCD input');
  for(const i of ids)if(i>=x.length/4)throw new Error('Shadow CCD index out of bounds');
  const count=externalPairs?.capacity??ids.length/4;if(Math.ceil(count/workgroupSize)>device.limits.maxComputeWorkgroupsPerDimension)throw new Error('Shadow CCD dispatch too large');
  const module=device.createShaderModule({label:'Off-live affine CCD',code:clothAffineCCDShadowWGSL(workgroupSize,sameTimeEndpoints,!!externalPairs,projectedSupports,advancement,currentBounds)});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==='error');
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  const upload=(data:ArrayBufferView<ArrayBuffer>,usage:number)=>{const b=device.createBuffer({size:data.byteLength,usage:usage|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,data);return b;};
  const params=new Uint32Array([count,kind==='vt'?0:1,0,budget]);new Float32Array(params.buffer)[2]=thickness;
  const xb=upload(x,GPUBufferUsage.STORAGE),db=upload(d,GPUBufferUsage.STORAGE),ib=externalPairs?.queue??upload(ids,GPUBufferUsage.STORAGE),pb=upload(params,GPUBufferUsage.UNIFORM);
  const output=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[xb,db,ib,output,pb,...(externalPairs?[externalPairs.stats]:[])].map((buffer,binding)=>({binding,resource:{buffer}}))});
  return {
    encodeMotion(encoder:GPUCommandEncoder,nextX:GPUBuffer,nextD:GPUBuffer){encoder.copyBufferToBuffer(nextX,0,xb,0,x.byteLength);encoder.copyBufferToBuffer(nextD,0,db,0,d.byteLength);},
    updateMotion(nextX:Float32Array<ArrayBuffer>,nextD:Float32Array<ArrayBuffer>){if(nextX.length!==x.length||nextD.length!==d.length)throw new Error('Motion buffer size mismatch');device.queue.writeBuffer(xb,0,nextX);device.queue.writeBuffer(db,0,nextD);},
    encode(encoder:GPUCommandEncoder,timestampWrites?:GPUComputePassTimestampWrites){const pass=encoder.beginComputePass({timestampWrites});pass.setPipeline(pipeline);pass.setBindGroup(0,group);if(externalPairs)pass.dispatchWorkgroupsIndirect(externalPairs.indirect,0);else pass.dispatchWorkgroups(Math.ceil(count/workgroupSize));pass.end();},
    async read(actualCount=count){if(!Number.isInteger(actualCount)||actualCount<0||actualCount>count)throw new Error('Invalid CCD read count');if(!actualCount)return new Float32Array();const read=device.createBuffer({size:16*actualCount,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const enc=device.createCommandEncoder();enc.copyBufferToBuffer(output,0,read,0,16*actualCount);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);return new Float32Array(read.getMappedRange().slice(0));}finally{read.destroy();}},
    destroy(){for(const b of [xb,db,pb,output,...(externalPairs?[]:[ib])])b.destroy();},
  };
}
