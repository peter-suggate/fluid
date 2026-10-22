import { mixedPhysical, readMixedBuffer } from '../tests/helpers/uniform-mixed-native-fields';

/** Diagnostic only: measure physical velocity after each producer without changing it. */
export async function installChurningStageTrace(device: GPUDevice, solver: any) {
  const frame = solver.mixedFrame;
  const labels: string[] = [];
  const result = device.createBuffer({size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
  const module = device.createShaderModule({code: `
    enable subgroups;
    @group(0) @binding(0) var field:texture_3d<f32>;
    @group(0) @binding(1) var<storage,read_write> result:array<atomic<u32>>;
    override slot:u32=0u;
    @compute @workgroup_size(256) fn measure(@builtin(global_invocation_id) id:vec3u) {
      let d=textureDimensions(field); let i=min(id.x,d.x*d.y*d.z-1u);
      let v=abs(textureLoad(field,vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y))),0).xyz);
      let peak=subgroupMax(max(v.x,max(v.y,v.z)));
      if(subgroupElect()){atomicMax(&result[slot],bitcast<u32>(peak));}
    }`});
  const info=await module.getCompilationInfo();if(info.messages.length)console.log(info.messages.map(m=>({message:m.message,line:m.lineNum})));
  async function hook(object:any, name:string, label:string, field:GPUTexture) {
    const slot=labels.length;labels.push(label);
    const texture=mixedPhysical(field);
    const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'measure',constants:{slot}}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:1,resource:{buffer:result}}]});
    const original=object[name].bind(object);
    object[name]=(encoder:GPUCommandEncoder,...args:any[])=>{
      const returned=original(encoder,...args); if(!process.argv.includes("--shared-stages")) frame.blit.finishStage(encoder);
      const pass=encoder.beginComputePass({label:'Diagnostic '+label});pass.setPipeline(pipeline);pass.setBindGroup(0,group);
      pass.dispatchWorkgroups(Math.ceil(texture.width*texture.height*texture.depthOrArrayLayers/256));pass.end();return returned;
    };
  }
  await hook(frame.narrowBandFlip,'move','moveGrid',frame.fields.velocity);
  await hook(frame.narrowBandFlip,'move','moveCache',frame.coarseCache);
  await hook(frame.narrowBandFlip,'move','moveTaps',frame.hanging.unitVelocity);
  await hook(frame.narrowBandFlip,'move','moveHalo',frame.hanging.haloOf(frame.hanging.unitVelocity));
  await hook(frame,'encodeParticleMomentum','P2G',frame.fields.velocity);
  await hook(frame.forces,'encode','forces',frame.fields.velocityScratch);
  await hook(frame.band,'encodeProject','pressure',frame.fields.velocity);
  await hook(frame.extension,'finishPair','extension',frame.fields.velocity);
  return {
    reset(){device.queue.writeBuffer(result,0,new Uint32Array(64));},
    async read(){const values=await readMixedBuffer(device,result);return Object.fromEntries(labels.map((label,i)=>[label,values[i]]));},
    destroy(){result.destroy();}
  };
}

export async function dumpChurningPressure(device:GPUDevice,solver:any) {
  const {readUniformLiveAddress}=await import('../tests/helpers/uniform-geometric');
  const {readMixedTexture,readMixedTileWords}=await import('../tests/helpers/uniform-mixed-native-fields');
  const {uniformTileKey}=await import('../lib/methods/uniform/uniform-lattice-address');
  const address=await readUniformLiveAddress(device,solver),band=solver.mixedFrame.band;
  const [pool,index,velocity]=await Promise.all([readMixedBuffer(device,band.pool),readMixedBuffer(device,band.index),readMixedTexture(device,solver.velocityTexture,address)]);
  const tiles=await readMixedTileWords(device,solver);
  const bits=new Uint32Array(pool.buffer),words=new Uint32Array(index.buffer);
  const {nx,ny}=solver.info,base=(n:string)=>band.arena.base(n,0);
  const row=(q:number[])=>{
    const tile=uniformTileKey(address,q[0]>>2,q[1]>>2,q[2]>>2),j=words[band.slotMapOffset/4+tile]-1;
    if(j<0){const at=520*Math.floor(tile/512)+(tile%512),b=(n:string)=>band.arena.base(n,2);
      return {q,tile,tileWord:tiles[tile].toString(16),j,at,nm:bits[b('nm')+at].toString(16),...Object.fromEntries(['nx','nb','nr','nd',...Array.from({length:6},(_,f)=>['nk'+f,'nl'+f]).flat()].map(n=>[n,pool[b(n)+at]]))};}
    const w=words[band.tileListWord+j],slot=w>>3,t=w&7;
    const l=[(q[0]&3)+4*(t&1),(q[1]&3)+4*((t>>1)&1),(q[2]&3)+4*(t>>2)];
    const at=520*slot+l[0]+8*(l[1]+8*l[2]);
    return {q,tile,tileWord:tiles[tile].toString(16),j,at,kinds:bits[base('kinds')+at].toString(16),...Object.fromEntries(['x','base','rhs','diag',...Array.from({length:6},(_,f)=>['c'+f,'u'+f,'v'+f]).flat()].map(n=>[n,pool[base(n)+at]]))};
  };
  const top:number[]=[];
  for(let i=0;i<velocity.length;i++)if(i%4!==3&&Math.abs(velocity[i])>(process.argv.includes("--all-peaks")?0:10)){top.push(i);}
  top.sort((a,b)=>Math.abs(velocity[b])-Math.abs(velocity[a]));
  top.unshift(4*(75+nx*(35+ny*91))+2);
  return top.slice(0,12).map(i=>{const cell=Math.floor(i/4),q=[cell%nx,Math.floor(cell/nx)%ny,Math.floor(cell/(nx*ny))],n=q.slice();n[i%4]++;
    return {value:velocity[i],axis:i%4,low:row(q),high:row(n)};});
}

export async function installPressureTrace(device:GPUDevice,solver:any) {
  const band=solver.mixedFrame.band, a=band.arena;
  const channels=[['xa',1,'da'],['ya',1,'da'],['xb',2,'db'],['yb',2,'db'],['nx',2,'nd'],['xt',2,'dt'],['yt',2,'dt']] as const;
  const result=device.createBuffer({size:256*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const slots=device.createBuffer({size:256*256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const values=new Uint32Array(256*64);for(let i=0;i<256;i++)values[64*i]=i;device.queue.writeBuffer(slots,0,values);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform',hasDynamicOffset:true}}]});
  const module=device.createShaderModule({code:`enable subgroups;
    @group(0) @binding(0) var<storage,read> pool:array<f32>;
    @group(0) @binding(1) var<storage,read_write> out:array<atomic<u32>>;
    @group(0) @binding(2) var<uniform> slot:u32;
    @compute @workgroup_size(256) fn measure(@builtin(global_invocation_id) id:vec3u,@builtin(workgroup_id) group:vec3u){
      let i=id.x;
      ${channels.map(([name,l,d],j)=>`{var v=0.0;if(i<${520*a.capacity(l)}u&&pool[${a.base(d,l)}u+i]>0.0){v=abs(pool[${a.base(name,l)}u+i]);}let m=subgroupMax(v);if(subgroupElect()){atomicMax(&out[slot*8u+${j}u],bitcast<u32>(m));}}`).join('\n')}
    }`});
  const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'measure'}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:band.pool}},{binding:1,resource:{buffer:result}},{binding:2,resource:{buffer:slots,size:4}}]});
  let labels:string[]=[];let measuring=!process.argv.includes("--matrix-only");
  const snapshots=process.argv.includes('--snapshot-pressure')?[band.pool,band.index,band.fields.root.pressure.buffer].map((live:GPUBuffer)=>({live,saved:device.createBuffer({size:live.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})})):[];
  const solve=band.encodeSolve.bind(band);
  if(snapshots.length)band.encodeSolve=(encoder:GPUCommandEncoder)=>{solver.mixedFrame.blit.finishStage(encoder);for(const p of snapshots)encoder.copyBufferToBuffer(p.live,0,p.saved,0,p.live.size);solve(encoder);};
  const original=band.dispatch.bind(band);
  band.dispatch=(pass:GPUComputePassEncoder,name:string,launch:any)=>{
    original(pass,name,launch);
    if(!measuring)return;
    if(!/p(Rebase|Down|Up|Tail|Residual|Restrict|Prolong|Final)/.test(name))return;
    const slot=labels.length;labels.push(name);
    pass.setPipeline(pipeline);pass.setBindGroup(0,group,[slot*256]);
    pass.dispatchWorkgroups(Math.ceil(520*a.capacity(1)/256));
    band.boundPass=undefined;
  };
  return {reset(){labels=[];device.queue.writeBuffer(result,0,new Uint32Array(2048));},
    async read(){const data=await readMixedBuffer(device,result);return labels.map((label,i)=>({label,...Object.fromEntries(channels.map(([n],j)=>[n,data[8*i+j]]))}));},
    async replay(){if(!snapshots.length)return;measuring=true;labels=[];device.queue.writeBuffer(result,0,new Uint32Array(2048));const e=device.createCommandEncoder();for(const p of snapshots)e.copyBufferToBuffer(p.saved,0,p.live,0,p.live.size);solve(e);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();},
    async dump(){const data=await readMixedBuffer(device,band.pool);const levels:any[]=[];
      if(process.argv.includes('--save-matrix')){
        const {writeFile}=await import('node:fs/promises');const index=await readMixedBuffer(device,band.index),bits=new Uint32Array(index.buffer);
        const names=['xb','yb','rb','db',...Array.from({length:6},(_,i)=>'kb'+i),'xt','yt','rt','dt',...Array.from({length:6},(_,i)=>'kt'+i)];
        const channels=Object.fromEntries(names.map(n=>[n,Array.from(data.slice(a.base(n,2),a.base(n,2)+520*a.capacity(2)))]));
        await writeFile('/tmp/churn-pressure-matrix.json',JSON.stringify({channels,records:Array.from(bits.slice(band.tableWord+a.layout.record2,band.tableWord+a.layout.record2+32*a.capacity(2))),pages:Array.from(bits.slice(band.tableWord+a.layout.pages,band.tableWord+a.layout.pages+bits[band.tableWord+22]))}));
      }
      for(const [l,d,r,x,k,count] of [[1,'da','ra','xa','ka',512],[2,'db','rb','xb','kb',512],[2,'dt','rt','xt','kt',73]] as const){
        const rows:any[]=[];
        for(let slot=0;slot<a.capacity(l);slot++)for(let i=0;i<count;i++){
          const at=520*slot+i,diag=data[a.base(d,l)+at];if(!(diag>0))continue;
          const ks=Array.from({length:6},(_,f)=>data[a.base(k+f,l)+at]);
          const rhs=data[a.base(r,l)+at],v=data[a.base(x,l)+at];
          rows.push({at,diag,rhs,x:v,k:ks,balance:diag-ks.reduce((s,c)=>s+c,0),ratio:Math.abs(rhs/diag)});
        }
        levels.push({d,smallest:rows.toSorted((a,b)=>a.diag-b.diag).slice(0,8),largest:rows.toSorted((a,b)=>Math.abs(b.x)-Math.abs(a.x)).slice(0,8),negative:rows.filter(r=>r.balance<-.01).toSorted((a,b)=>a.balance-b.balance).slice(0,8)});
      }
      return levels;
    },destroy(){result.destroy();slots.destroy();for(const p of snapshots)p.saved.destroy();}};
}
