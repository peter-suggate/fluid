import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {uniformMixedDustMass} from "../lib/methods/uniform/uniform-mixed-dust-accounting.wgsl";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSharpening} from "../lib/methods/uniform/uniform-mixed-sharpening";
import {SHARPENING_REFERENCE_TOLERANCE,UniformSharpeningReference,sharpeningTopology} from "./helpers/uniform-sharpening-reference";
import {UniformMixedSolid} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

// The production sweeps against a dense reference that shares none of their
// text (tests/helpers/uniform-sharpening-reference.ts), within
// SHARPENING_REFERENCE_TOLERANCE, and against what sharpening must do
// whatever its launches look like: every patch moves one volume between its
// two owners (h/4h seams included), V stays in [0, open], closed owners are
// left alone, and the volume that leaves the field is the dust it reports.
// Nothing here reads the work list or knows how jobs are packed.
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed sharpening matches a dense reference and conserves volume across h/4h seams",{timeout:240_000},async()=>{
 let device:GPUDevice|undefined,ownership:UniformMixedOwnership|undefined,solid:UniformMixedSolid|undefined,reference:UniformSharpeningReference|undefined;
 const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  device=managedGPUDevice(raw,{requireWorkerRealm:false});const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const dims=[32,24,16] as const,vertices=dims.map(n=>n+1),h=[.04,.05,.06] as const,n=dims.reduce((a,b)=>a*b,1);
  const lattice={dimensions:dims,cellSize_m:h,origin_m:{x:0,y:0,z:0}};
  const fine=createUniformMixedLayout(lattice,[]),coarse=createUniformMixedLayout(lattice,[],4);
  // A single coarse island is one seam owner with h tiles on all six sides.
  // A larger island has seam owners with one, two and three h sides around
  // a regular coarse owner; the scattered layout has every other count.
  const island=(size:number)=>createUniformMixedLayout(lattice,[{id:"island",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,
   min_m:{x:.32,y:.2,z:.24},max_m:{x:.32+size*4*h[0],y:.2+size*4*h[1],z:.24+size*4*h[2]}}]);
  const odd=island(1),mixed=island(3);
  const scattered=createUniformMixedLayout(lattice,[],4,Uint8Array.from({length:fine.tiles.length},(_,i)=>((i*37)^(i>>1))%11<3?1:0));
  ownership=new UniformMixedOwnership(d,fine,false);
  const texture=(size:readonly number[])=>{const t=d.createTexture({size:[...size],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
  const buffer=(size:number,uniform=false)=>{const b=d.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.push(b);return b;};
  const write=(t:GPUTexture,a:Float32Array<ArrayBuffer>)=>d.queue.writeTexture({texture:t},a,{bytesPerRow:t.width*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  const read=async(b:GPUBuffer,size=b.size)=>{const staging=d.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=d.createCommandEncoder();e.copyBufferToBuffer(b,0,staging,0,size);d.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);const words=new Uint32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();return words;};
  const phi=texture(vertices),target=texture(dims),center=texture(dims),params=buffer(32,true);
  const solidParams=buffer(272,true),words=Math.ceil((dims[0]+2)*(dims[1]+2)*(dims[2]+2)/32),cutMapOffsetWords=Math.ceil((4+words)/64)*64;
  const solidScratch=buffer(4*(cutMapOffsetWords+ownership.capacity.tiles));
  const solidData=new Uint32Array(solidScratch.size/4);solidData.fill(1,cutMapOffsetWords);
  // An embedded block in the fine side of the mixed layout; coarse cubic
  // stencils can still reach it. Also exercise the solid-present fallback.
  for(let z=1;z<8;z++)for(let y=1;y<8;y++)for(let x=1;x<8;x++){
   const i=x+1+(dims[0]+2)*(y+1+(dims[1]+2)*(z+1));solidData[4+(i>>>5)]!|=1<<(i&31);
  }
  d.queue.writeBuffer(solidScratch,0,solidData);d.queue.writeBuffer(solidParams,0,new Float32Array([...dims,.037,...h,0,dims[0]*h[0],dims[1]*h[1],dims[2]*h[2],0]));
  const terrain=d.createTexture({size:[dims[0],dims[2]],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING});textures.push(terrain);
  solid=new UniformMixedSolid(d,{params:solidParams,scratch:solidScratch,terrain,bodies:buffer(12*128,true),coupledTiles:0,cutMapOffsetWords});

  const work=buffer(UniformMixedSharpening.workBytes(ownership.capacity.tiles));
  const stage=new UniformMixedSharpening(d,ownership,solid,{list:work});await stage.initialize();
  const a=texture(dims),b=texture(dims),scratch=buffer(stage.scratchBytes),reductions=buffer(64);
  const groups=[stage.bind(a,b,phi,target,center,{buffer:scratch},params,reductions),stage.bind(b,a,phi,target,center,{buffer:scratch},params,reductions)] as const;
  reference=new UniformSharpeningReference(d,ownership,solid);await reference.initialize();reference.bind(phi,target,center,params);

  const tolerance=SHARPENING_REFERENCE_TOLERANCE,dustThreshold=.001,unwritten=123;
  // What the lane must have exercised by its end.
  const seen={deviation:0,balance:0,conservation:0,moved:0,dust:0,seamFlux:0,closedOwners:0,singleSeam:false,solidMatters:false,sides:new Set<number>()};
  const clearResults=new Map<string,Float32Array>(),seamCarried=new Map<string,number>();
  for(const present of [false,true]){solid.present=present;
  for(const [name,layout] of [["coarse",coarse],["odd",odd],["mixed",mixed],["scattered",scattered],["fine",fine],["mixed-again",mixed]] as const){
   ownership.update(layout);
   // A stale launch receipt must still cover every job the sweeps hold.
   (ownership as any).work.sharpen=1;
   const info=await reference.describe(),{owners,patches}=sharpeningTopology(dims,info.width);
   const cubed=(o:{width:number})=>o.width**3;
   // Seam 4h owners by their number of h sides (sixteen patches each).
   const seamSides=new Map<number,number>();
   for(const o of owners.values())if(o.width===4){const sides=new Set(o.patches.filter(p=>p.patch.seam).map(p=>`${Math.floor(p.patch.patch%3)}${p.sign}`)).size;if(sides)seamSides.set(o.cell,sides);}
   for(const policy of [[0,0],[1,1],[0,2]]){
    const label=`${present?"solid":"clear"} ${name} policy ${policy}`;
    d.queue.writeBuffer(params,0,new Float32Array([.8,2.1,dustThreshold,0,...policy,0,0]));
    const values=new Float32Array(n),desired=new Float32Array(n),distances=new Float32Array(n);
    const field=(x:number,y:number,z:number)=>name==="scattered"?Math.hypot((x-16)*h[0],(y-12)*h[1],(z-8)*h[2])-.4:(x-16)*h[0]*.31+(y-10)*h[1]*.83+(z-8)*h[2]*.46;
    for(let z=0;z<dims[2];z++)for(let y=0;y<dims[1];y++)for(let x=0;x<dims[0];x++){
     const i=x+dims[0]*(y+dims[1]*z),distance=field(x+.5,y+.5,z+.5);
     distances[i]=distance;desired[i]=Math.max(0,Math.min(1,.5-distance/.12));
     values[i]=i%17===0?-.0003:i%13===0?.0004:i%7===0?desired[i]!:.15+.7*(.5+.5*Math.sin(x*.61+y*.79+z*.37));
     // An owner starts within what its open fraction holds.
     if(info.width[i]!>0)values[i]=Math.min(values[i]!,info.open[i]!);
    }
    const vertex=new Float32Array(vertices.reduce((a,b)=>a*b,1));
    for(let z=0;z<vertices[2]!;z++)for(let y=0;y<vertices[1]!;y++)for(let x=0;x<vertices[0]!;x++)vertex[x+vertices[0]!*(y+vertices[1]!*z)]=field(x,y,z);
    write(phi,vertex);write(target,desired);write(center,distances);
    write(a,values);write(b,new Float32Array(n).fill(unwritten));d.queue.writeBuffer(scratch,0,new Float32Array(scratch.size/4).fill(71));
    {const e=d.createCommandEncoder();e.clearBuffer(reductions);stage.encodeGeometry(e,groups[0]);d.queue.submit([e.finish()]);}
    reference.start(values);
    let result=values as Float32Array,dust=new Uint32Array(16);
    // Observe all four pairs of sweeps, including each scratch output, so
    // stale writes and errors that cancel by sweep eight cannot hide.
    for(let pair=0;pair<4;pair++){
     const at=`${label} sweep ${2*pair+2}`;
     {const e=d.createCommandEncoder();stage.encodeSweeps(e,groups,2);d.queue.submit([e.finish()]);}
     const even=await readMixedTexture(d,a),oddOutput=await readMixedTexture(d,b);dust=await read(reductions);
     const expected=await reference.sweep(2);
     for(let i=0;i<n;i++){
      if(info.width[i]===0){
       // Only owners are written: a 4h owner's other texels are not state.
       assert.ok(Object.is(even[i],values[i])&&oddOutput[i]===unwritten,`${at}: a texel that is not an owner's was written at ${i}`);continue;
      }
      assert.ok(Number.isFinite(even[i]!)&&Number.isFinite(oddOutput[i]!),`${at}: a sweep output is not finite at ${i}`);
      const deviation=Math.abs(even[i]!-expected.volume[i]!);seen.deviation=Math.max(seen.deviation,deviation);
      assert.ok(deviation<=tolerance,`${at}: V ${even[i]} at ${i} (width ${info.width[i]}), the reference holds ${expected.volume[i]}`);
      // An owner the sweeps do not visit is one the dense reference leaves alone.
      if(oddOutput[i]===unwritten)assert.ok(Object.is(expected.previous[i],values[i])&&Object.is(expected.volume[i],values[i]),`${at}: the sweeps skipped owner ${i}, which the reference changes`);
      else assert.ok(Math.abs(oddOutput[i]!-expected.previous[i]!)<=tolerance,`${at}: odd sweep V ${oddOutput[i]} at ${i}, the reference holds ${expected.previous[i]}`);
      assert.ok(even[i]!>=0&&even[i]!<=info.open[i]!+tolerance,`${at}: V ${even[i]} at ${i} is outside [0, open ${info.open[i]}]`);
     }
     result=even;
    }
    const accounts=await reference.accounts();
    // Face by face: each owner's change is the committed fluxes of its
    // patches, one value a patch whichever side reads it, less its dust.
    let initial=0,final=0,moved=0,removed=0,seamFlux=0;
    for(const o of owners.values()){
     const cells=cubed(o);let outflow=0;
     for(const {patch,sign} of o.patches)outflow+=sign*accounts.flux[patch.patch]!;
     const balance=Math.abs((result[o.cell]!-values[o.cell]!)*cells+outflow+accounts.dustMass[o.cell]!)/cells;seen.balance=Math.max(seen.balance,balance);
     assert.ok(balance<=2*tolerance,`${label}: owner ${o.cell} (width ${o.width}) changed by ${(result[o.cell]!-values[o.cell]!)*cells} cells against ${-outflow} through its patches and ${accounts.dustMass[o.cell]} of dust`);
     initial+=values[o.cell]!*cells;final+=result[o.cell]!*cells;moved+=Math.abs(result[o.cell]!-values[o.cell]!)*cells/2;removed+=Math.abs(accounts.dustMass[o.cell]!);
     // A closed or cut owner trades nothing: it keeps its volume, or loses it whole to the dust rule.
     if(info.open[o.cell]!<=.99999){seen.closedOwners++;assert.ok(Object.is(result[o.cell],values[o.cell])||(result[o.cell]===0&&Math.abs(values[o.cell]!)<dustThreshold),`${label}: closed owner ${o.cell} changed from ${values[o.cell]} to ${result[o.cell]}`);}
    }
    for(const p of patches)if(p.seam)seamFlux+=Math.abs(accounts.flux[p.patch]!);
    // The whole field: what left it is the dust the production stage reports.
    const reported=uniformMixedDustMass(dust,5,dustThreshold,n),conservation=Math.abs(initial-final-reported);
    seen.conservation=Math.max(seen.conservation,conservation);assert.ok(conservation<=2.5e-4,`${label}: ${initial-final} cells left the field and ${reported} were reported as dust`);
    seen.moved+=moved;seen.dust+=removed;seen.seamFlux+=seamFlux;
    for(const [cell,sides] of seamSides)if(Math.abs(result[cell]!-values[cell]!)>1e-3){seen.sides.add(sides);if(name==="odd"&&seamSides.size===1)seen.singleSeam=true;}
    if(seamSides.size)seamCarried.set(`${present} ${name}`,Math.max(seamCarried.get(`${present} ${name}`)??0,seamFlux));
    // The same case without and with solids: the solid block must matter.
    if(!present)clearResults.set(`${name} ${policy}`,result);
    else if(result.some((v,i)=>info.width[i]!>0&&!Object.is(v,clearResults.get(`${name} ${policy}`)![i])))seen.solidMatters=true;
   }
  }}
  console.log(`sharpening reference: max |V - reference| ${seen.deviation.toExponential(2)} (tolerance ${tolerance.toExponential(2)}), max owner imbalance ${seen.balance.toExponential(2)}, max field imbalance ${seen.conservation.toExponential(2)} cells, moved ${seen.moved.toFixed(1)} cells, dust ${seen.dust.toFixed(3)} cells, seam flux ${seen.seamFlux.toFixed(1)} cells, seam side counts ${[...seen.sides].sort()}`);
  assert.ok(seen.singleSeam,"a layout with a single seam owner must move volume through it");
  for(const [layout,carried] of seamCarried)assert.ok(carried>1,`the seam patches of ${layout} carried ${carried} cells under every policy`);
  assert.deepEqual([...seen.sides].sort(),[1,2,3,4,5,6],"seam owners with every number of h sides must move volume");
  assert.ok(seen.closedOwners>0&&seen.solidMatters,"the solid block must close owners and change the result");
  assert.ok(seen.moved>10*seen.dust,"sharpening must move more volume than dust cleanup alone");assert.deepEqual(errors,[]);
 }finally{reference?.destroy();ownership?.destroy();solid?.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());device?.destroy();}
});
