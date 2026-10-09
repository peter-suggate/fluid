import assert from "node:assert/strict";
import test from "node:test";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformPressureBand} from "../lib/methods/uniform/uniform-pressure-band";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("grouped pressure half sweeps preserve the one-tile solve across partial workgroups",async()=>{
 await withUniformDevice("pressure band dispatch parity",async device=>{
  const n=20,t=n/4,cells=n**3,vertices=(n+1)**3;
  const lattice={dimensions:[n,n,n] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const layout=createUniformMixedLayout(lattice,[]),ownership=new UniformMixedOwnership(device,layout,false);
  const pressure=new UniformMixedOwnership(device,createUniformMixedLayout(lattice,[],4),false);
  const buffers:GPUBuffer[]=[],textures:GPUTexture[]=[],bands:UniformPressureBand[]=[];
  const buffer=(size:number,uniform=false)=>{const b=device.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.push(b);return b;};
  const texture=(size:number,format:GPUTextureFormat)=>{const x=device.createTexture({size:[size,size,size],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});textures.push(x);return x;};
  const phi=buffer(4*cells),vertexPhi=texture(n+1,"r32float"),correction=texture(n,"r32float"),velocity=texture(n,"rgba32float"),copy=texture(n,"rgba32float");
  const negative=buffer(12*n*n),coarsePressure=buffer(4*t**3),params=buffer(64,true),presentation=buffer(4*cells);
  device.queue.writeBuffer(params,0,new Float32Array([1,1,1,1/60,1000,1,1/60000,1,5,1,1,1]));
  const create=device.createShaderModule.bind(device);let reference=false,replaced=0;
  const instrumented=new Proxy(device,{get:(target,key)=>{if(key!=="createShaderModule"){const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}return (d:GPUShaderModuleDescriptor)=>{
   let code=d.code;
   if(reference&&d.label==="Uniform pressure band sweep"){
    const size=Number(code.match(/@workgroup_size\((\d+)\) fn bandSweep\(/)?.[1]);const block=size/32;
    assert.ok(block>1&&Number.isInteger(block));
    code=code.replace(`@workgroup_size(${size}) fn bandSweep(`,"@workgroup_size(32) fn bandSweep(")
     .replace(`group.x*${block}u+(lane>>5u)`,"group.x")
     .replace(`groups.x*${block}u`,"groups.x");replaced++;
   }
   return create({...d,code});
  };}});
  try{
   for(const old of [true,false]){
    reference=old;
    const b=new UniformPressureBand(instrumented,ownership,pressure,{phi:{buffer:phi},vertexPhi,correction,forced:{velocity,negative},velocity,negative,copy,coarsePressure:{buffer:coarsePressure},params,presentation:{buffer:presentation,word:0}});
    bands.push(b);await b.initialize();
   }
   reference=false;assert.equal(replaced,1);
   const input=new Float32Array(cells*4);
   for(let z=0;z<n;z++)for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const at=4*(x+n*(y+n*z));input.set([.03*Math.sin(x*.7+y*.3),.02*Math.cos(y*.4+z*.6),.04*Math.sin(z*.5+x*.2),0],at);
   }
   for(const level of [7.3,17.3]){
    const v=new Float32Array(vertices),centres=new Float32Array(cells);
    for(let z=0;z<=n;z++)for(let y=0;y<=n;y++)for(let x=0;x<=n;x++)v[x+(n+1)*(y+(n+1)*z)]=y-level+.1*Math.sin(x*.4+z*.3);
    for(let tile=0;tile<t**3;tile++)for(let lane=0;lane<64;lane++){
     const x=4*(tile%t)+lane%4,y=4*(Math.floor(tile/t)%t)+Math.floor(lane/4)%4,z=4*Math.floor(tile/(t*t))+Math.floor(lane/16);
     let value=0;for(let k=0;k<8;k++)value+=v[x+(k&1)+(n+1)*(y+((k>>1)&1)+(n+1)*(z+((k>>2)&1)))]!*.125;
     centres[64*tile+lane]=value;
    }
    device.queue.writeTexture({texture:vertexPhi},v,{bytesPerRow:4*(n+1),rowsPerImage:n+1},[n+1,n+1,n+1]);device.queue.writeBuffer(phi,0,centres);
    const results:Float32Array[]=[];
    for(const band of bands){
     device.queue.writeBuffer(negative,0,new Float32Array(3*n*n));
     device.queue.writeTexture({texture:velocity},input,{bytesPerRow:16*n,rowsPerImage:n},[n,n,n]);
     const encoder=device.createCommandEncoder();band.encodePrepare(encoder);band.encodeSolve(encoder);device.queue.submit([encoder.finish()]);
     results.push(await readMixedTexture(device,velocity));
    }
    let error=0,changed=0;
    for(let i=0;i<input.length;i++){assert.ok(Number.isFinite(results[1]![i]!));error=Math.max(error,Math.abs(results[0]![i]!-results[1]![i]!));if(results[1]![i]!==input[i])changed++;}
    assert.ok(changed>100,"exercise a nonzero pressure correction");assert.equal(error,0,"workgroup grouping must not change a pressure update");
   }
  }finally{bands.forEach(b=>b.destroy());ownership.destroy();pressure.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());}
 });
});
