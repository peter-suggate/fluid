import assert from "node:assert/strict";
import test from "node:test";
import { HdrLensBloom, DEFAULT_LENS_BLOOM_OPTIONS, resolveLensBloomOptions, lensBloomShader } from "../lib/core/webgpu-lens-bloom";
import { compositeShader, RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import { resolveDisplayGrade, sceneLinearToDisplay, unifiedDisplayTransferShaderLibrary } from "../lib/core/webgpu-lighting";

test("lens scatter is opt-in and validates independent camera controls",()=>{
  assert.equal(DEFAULT_LENS_BLOOM_OPTIONS.enabled,false);
  assert.deepEqual(resolveLensBloomOptions({enabled:true,strength:2,threshold:-1,radius:Infinity}),{enabled:true,strength:1,threshold:0,radius:.03});
});

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};
function half(value:number):number{const sign=(value&32768)?-1:1,e=(value>>10)&31,m=value&1023;return sign*(e===0?m*2**-24:e===31?m?NaN:Infinity:(1+m/1024)*2**(e-15));}
async function pixels(device:GPUDevice,texture:GPUTexture):Promise<Float32Array|Uint8Array>{
  const halfFormat=texture.format==="rgba16float";const pixelBytes=halfFormat?8:4;
  const pitch=Math.ceil(texture.width*pixelBytes/256)*256;
  const read=device.createBuffer({size:pitch*texture.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:read,bytesPerRow:pitch},{width:texture.width,height:texture.height});device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=read.getMappedRange();const out=halfFormat?new Float32Array(texture.width*texture.height*4):new Uint8Array(texture.width*texture.height*4);
    for(let y=0;y<texture.height;y++){if(halfFormat){const row=new Uint16Array(data,y*pitch,texture.width*4);for(let x=0;x<row.length;x++)out[(y*texture.width*4)+x]=half(row[x]);}else out.set(new Uint8Array(data,y*pitch,texture.width*4),y*texture.width*4);}read.unmap();return out;
  }finally{read.destroy();}
}

// Production finish() is exercised independently of expensive scene traversal.
const finish=/fn finish\(color:vec3f,ndc:vec2f\)->vec4f\{[^\n]+\}/.exec(compositeShader)![0];
const vertex=`struct V{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertex(@builtin(vertex_index) i:u32)->V{let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3))[i];return V(vec4f(p,0,1),vec2f(p.x*.5+.5,.5-p.y*.5));}`;

test("GPU disabled composite is byte-identical; raw HDR contains the weighted water answer before grading",gpu,async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  await withUniformDevice("Camera HDR handoff",async device=>{
    const module=device.createShaderModule({code:`${unifiedDisplayTransferShaderLibrary}${vertex}
    override linearHdrOutput:f32=0;fn waterDisplayExposure()->f32{return 1.0;}fn waterDisplayToneCurve()->u32{return 0u;}fn waterDisplayWhiteBalance()->vec3f{return vec3f(1);}
    ${finish}
    fn oldFinish(color:vec3f,ndc:vec2f)->vec4f{let c=color*(1.0-.08*dot(ndc*.55,ndc*.55));return vec4f(unifiedDisplayGradeBalanced(c,waterDisplayExposure(),waterDisplayToneCurve(),waterDisplayWhiteBalance()),1);}
    fn answer()->vec3f{return mix(vec3f(.25,.5,1),vec3f(40,8,2),.125)+vec3f(.1);}
    @fragment fn current(v:V)->@location(0) vec4f{return finish(answer(),v.uv*2-1);}
    @fragment fn old(v:V)->@location(0) vec4f{return oldFinish(answer(),v.uv*2-1);}`});
    const targets:GPUTexture[]=[];
    try{
      for(const [entryPoint,raw] of [["old",false],["current",false],["current",true]] as const){
        const format=raw?"rgba16float":"rgba8unorm";const target=device.createTexture({size:[8,8],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});targets.push(target);
        const pipeline=await device.createRenderPipelineAsync({layout:"auto",vertex:{module,entryPoint:"vertex"},fragment:{module,entryPoint,constants:{linearHdrOutput:raw?1:0},targets:[{format}]}});
        const e=device.createCommandEncoder();const p=e.beginRenderPass({colorAttachments:[{view:target.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:0}}]});p.setPipeline(pipeline);p.draw(3);p.end();device.queue.submit([e.finish()]);
      }
      assert.deepEqual(await pixels(device,targets[0]),await pixels(device,targets[1]));
      const raw=await pixels(device,targets[2]);const expected=[.875*.25+.125*40+.1,.875*.5+.125*8+.1,.875+.125*2+.1];
      for(let c=0;c<3;c++)assert.ok(Math.abs(raw[c]-expected[c])<.005,`weighted HDR channel ${c}`);
      assert.ok(raw[0]>1,"HDR highlight survives handoff");
      // Even enabling strength zero never allocates the optional camera path.
      const stub=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM});const water=new RasterWaterPipeline(device,"rgba8unorm",stub,stub);
      water.setLensBloomOptions({enabled:true,strength:0});await water.ensureLensBloomReady();assert.equal(water.lensBloomReady,true);assert.equal(water.lensBloomLinearComposite,undefined);water.destroy();stub.destroy();
    }finally{for(const t of targets)t.destroy();}
  });
});

test("GPU HDR lens pyramid preserves uniform fields and spreads finite highlight energy before one display grade",gpu,async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  await withUniformDevice("Normalized camera lens scatter",async device=>{
    const camera=new HdrLensBloom(device,"rgba8unorm");await camera.ensureReady();
    const options={...DEFAULT_LENS_BLOOM_OPTIONS,enabled:true,strength:.3,threshold:1,radius:.12};
    const source=camera.ensureSize(64,64,options);const target=device.createTexture({size:[64,64],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
    try{
      const grade=resolveDisplayGrade({exposure:.7,toneCurve:"reinhard"});let e=device.createCommandEncoder();let p=e.beginRenderPass({colorAttachments:[{view:source.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:4,g:2,b:1,a:1}}]});p.end();camera.encode(e,target.createView(),options,grade);device.queue.submit([e.finish()]);
      const uniform=await pixels(device,target);const center=(32*64+32)*4;const ndc=(32.5/64)*2-1;const vignette=1-.08*2*(ndc*.55)**2;const expected=sceneLinearToDisplay([4*vignette,2*vignette,vignette],grade);
      for(let c=0;c<3;c++)assert.ok(Math.abs(uniform[center+c]-Math.round(expected[c]*255))<=1,"uniform field unchanged, grade applied once");
      const module=device.createShaderModule({code:`${vertex}@fragment fn fragment(v:V)->@location(0) vec4f{let bright=all(vec2i(v.position.xy)==vec2i(32));return vec4f(select(vec3f(0),vec3f(64,16,4),bright),1);}`});
      const paint=await device.createRenderPipelineAsync({layout:"auto",vertex:{module,entryPoint:"vertex"},fragment:{module,entryPoint:"fragment",targets:[{format:"rgba16float"}]}});
      e=device.createCommandEncoder();p=e.beginRenderPass({colorAttachments:[{view:source.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});p.setPipeline(paint);p.draw(3);p.end();camera.encode(e,target.createView(),options,grade);device.queue.submit([e.finish()]);
      const impulse=await pixels(device,target);assert.ok(impulse[center]>0);assert.ok(impulse[center-4]>0,"halo reaches neighbor");assert.ok(impulse[(2*64+2)*4]<2,"dark far field remains dark");
      const blur=await pixels(device,camera.blurredHighlights!);let energy=0;for(let i=0;i<blur.length;i+=4){assert.ok(Number.isFinite(blur[i]));energy+=blur[i];}
      const luminance=64*.2126+16*.7152+4*.0722;const sourceHighlight=64*(luminance-1)/luminance;
      // Half-resolution pixels cover four input pixels. The normalized pyramid
      // preserves integrated highlight energy away from the clamp boundary.
      assert.ok(Math.abs(energy*4-sourceHighlight)/sourceHighlight<.025,`highlight energy ${energy*4} versus ${sourceHighlight}`);
    }finally{await device.queue.onSubmittedWorkDone();camera.destroy();target.destroy();}
  });
});
