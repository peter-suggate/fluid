import { SCENE_TONE_CURVE_CODES, type ResolvedDisplayGrade, unifiedDisplayTransferShaderLibrary } from "./webgpu-lighting";

/** Camera-space highlight scatter. Radius is a fraction of the shorter image side. */
export interface LensBloomOptions {
  readonly enabled: boolean;
  readonly strength: number;
  readonly threshold: number;
  readonly radius: number;
}
export const DEFAULT_LENS_BLOOM_OPTIONS: LensBloomOptions = Object.freeze({ enabled:false, strength:.08, threshold:1, radius:.03 });
export function resolveLensBloomOptions(input: Partial<LensBloomOptions> = {}): LensBloomOptions {
  const finite=(v:number|undefined,fallback:number,min:number,max:number)=>Number.isFinite(v)?Math.min(max,Math.max(min,v!)):fallback;
  return {enabled:input.enabled===true,strength:finite(input.strength,.08,0,1),threshold:finite(input.threshold,1,0,64),radius:finite(input.radius,.03,.002,.15)};
}

/** The input is the complete linear optical composite, never dry-only or unweighted reflection radiance. */
export const lensBloomShader = /* wgsl */ `
${unifiedDisplayTransferShaderLibrary}
struct Camera{optics:vec4f,whiteBalance:vec4f}
struct Pass{fineWeight:f32,pad0:f32,pad1:f32,pad2:f32}
@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var coarse:texture_2d<f32>;
@group(0) @binding(2) var linearSampler:sampler;
@group(0) @binding(3) var<uniform> camera:Camera;
@group(0) @binding(4) var<uniform> pyramid:Pass;
struct Vertex{ @builtin(position) position:vec4f,@location(0) uv:vec2f }
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->Vertex{
  let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3))[index];
  return Vertex(vec4f(p,0,1),vec2f(p.x*.5+.5,.5-p.y*.5));
}
fn highlight(cIn:vec3f)->vec3f{
  let c=max(cIn,vec3f(0));let luminance=dot(c,vec3f(.2126,.7152,.0722));
  let threshold=camera.optics.x;let knee=max(.5*threshold,1e-5);
  var soft=clamp(luminance-threshold+knee,0.0,2.0*knee);soft=soft*soft/(4.0*knee);
  return c*clamp(max(luminance-threshold,soft)/max(luminance,1e-6),0.0,1.0);
}
@fragment fn extract(input:Vertex)->@location(0) vec4f{
  let size=vec2i(textureDimensions(source));let base=vec2i(input.position.xy)*2;var sum=vec3f(0);
  for(var y=0;y<2;y++){for(var x=0;x<2;x++){
    sum+=highlight(textureLoad(source,clamp(base+vec2i(x,y),vec2i(0),size-1),0).rgb);
  }}return vec4f(sum*.25,1);
}
@fragment fn downsample(input:Vertex)->@location(0) vec4f{
  let step=1.0/vec2f(textureDimensions(source));var sum=vec3f(0);
  for(var y=0;y<2;y++){for(var x=0;x<2;x++){
    sum+=textureSampleLevel(source,linearSampler,input.uv+(vec2f(f32(x),f32(y))-.5)*step,0).rgb;
  }}return vec4f(sum*.25,1);
}
fn tent(uv:vec2f)->vec3f{
  let step=1.0/vec2f(textureDimensions(coarse));var sum=vec3f(0);
  for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
    let weight=f32((2-abs(x))*(2-abs(y)));
    sum+=textureSampleLevel(coarse,linearSampler,uv+vec2f(f32(x),f32(y))*step,0).rgb*weight;
  }}return sum/16.0;
}
@fragment fn upsample(input:Vertex)->@location(0) vec4f{
  let fine=textureSampleLevel(source,linearSampler,input.uv,0).rgb;
  return vec4f(mix(tent(input.uv),fine,pyramid.fineWeight),1);
}
fn scattered(uv:vec2f)->vec3f{
  let raw=textureLoad(source,clamp(vec2i(uv*vec2f(textureDimensions(source))),vec2i(0),vec2i(textureDimensions(source))-1),0).rgb;
  // Normalized lens PSF redistributes highlight energy; it does not raise the
  // exposure of a uniform bright field. Strength<=1 keeps the local term positive.
  return max(raw+camera.optics.y*(tent(uv)-highlight(raw)),vec3f(0));
}
@fragment fn linearOutput(input:Vertex)->@location(0) vec4f{return vec4f(scattered(input.uv),1);}
@fragment fn display(input:Vertex)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let c=scattered(input.uv)*(1.0-.08*dot(ndc*.55,ndc*.55));
  return vec4f(unifiedDisplayGradeBalanced(c,camera.optics.z,u32(round(camera.optics.w)),camera.whiteBalance.xyz),1);
}
`;

/** Optional post-optical camera pass. All buffers/textures are allocated only on use. */
export class HdrLensBloom {
  private layout?: GPUBindGroupLayout;
  private sampler?: GPUSampler;
  private uniform?: GPUBuffer;
  private pipelines?: {extract:GPURenderPipeline;downsample:GPURenderPipeline;upsample:GPURenderPipeline;display:GPURenderPipeline};
  private pending?: Promise<void>;
  private disposed=false;
  private key="";
  private source?: GPUTexture;
  private down:GPUTexture[]=[];
  private up:GPUTexture[]=[];
  private weights:GPUBuffer[]=[];
  private retired:(GPUTexture|GPUBuffer)[]=[];
  constructor(private readonly device:GPUDevice,private readonly targetFormat:GPUTextureFormat){}
  get ready(){return !!this.pipelines;}
  get linearComposite(){return this.source;}
  get blurredHighlights(){return this.down.length>1?this.up[0]:this.down[0];}
  ensureReady():Promise<void>{
    if(this.pending)return this.pending;
    this.pending=(async()=>{
      const module=this.device.createShaderModule({label:"HDR camera lens scatter",code:lensBloomShader});
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
      if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
      this.layout=this.device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},
        {binding:3,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},
        {binding:4,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},
      ]});
      const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.layout]});
      const create=(entryPoint:string,format:GPUTextureFormat)=>this.device.createRenderPipelineAsync({label:`Camera ${entryPoint}`,layout,vertex:{module,entryPoint:"vertexMain"},fragment:{module,entryPoint,targets:[{format}]},primitive:{topology:"triangle-list"}});
      const [extract,downsample,upsample,display]=await Promise.all([create("extract","rgba16float"),create("downsample","rgba16float"),create("upsample","rgba16float"),create("display",this.targetFormat)]);
      this.sampler=this.device.createSampler({minFilter:"linear",magFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"});
      this.uniform=this.device.createBuffer({label:"HDR camera parameters",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      if(this.disposed){this.uniform.destroy();throw new Error("HDR camera destroyed during preparation");}
      this.pipelines={extract,downsample,upsample,display};
    })();return this.pending;
  }
  ensureSize(width:number,height:number,options:LensBloomOptions):GPUTexture{
    if(!this.ready)throw new Error("HDR camera is not compiled");
    const count=Math.max(1,Math.min(8,Math.ceil(Math.log2(Math.max(2,Math.min(width,height)*options.radius)))));
    const key=`${width}x${height}/${count}`;
    if(key!==this.key){
      this.retired.push(...this.down,...this.up,...this.weights);if(this.source)this.retired.push(this.source);
      this.down=[];this.up=[];this.weights=[];
      const texture=(label:string,w:number,h:number)=>this.device.createTexture({label,size:[w,h],format:"rgba16float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      this.source=texture("Complete optical HDR composite",width,height);
      let w=width,h=height;
      for(let i=0;i<count;i++){
        w=Math.max(1,Math.ceil(w/2));h=Math.max(1,Math.ceil(h/2));
        this.down.push(texture(`Lens bloom down ${i}`,w,h));this.up.push(texture(`Lens bloom up ${i}`,w,h));
        const weight=this.device.createBuffer({label:`Lens normalized scale weight ${i}`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
        this.device.queue.writeBuffer(weight,0,new Float32Array([1/(count-i),0,0,0]));this.weights.push(weight);
      }this.key=key;
    }return this.source!;
  }
  encode(encoder:GPUCommandEncoder,output:GPUTextureView,options:LensBloomOptions,grade:ResolvedDisplayGrade):void{
    if(!this.source||!this.pipelines||!this.layout||!this.uniform||!this.sampler)throw new Error("HDR camera target is not ready");
    this.device.queue.writeBuffer(this.uniform,0,new Float32Array([options.threshold,options.strength,grade.exposure,SCENE_TONE_CURVE_CODES[grade.toneCurve],...grade.whiteBalance,0]));
    const draw=(label:string,pipeline:GPURenderPipeline,source:GPUTexture,coarse:GPUTexture,target:GPUTextureView,weight:GPUBuffer)=>{
      const group=this.device.createBindGroup({layout:this.layout!,entries:[{binding:0,resource:source.createView()},{binding:1,resource:coarse.createView()},{binding:2,resource:this.sampler!},{binding:3,resource:{buffer:this.uniform!}},{binding:4,resource:{buffer:weight}}]});
      const pass=encoder.beginRenderPass({label,colorAttachments:[{view:target,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();
    };
    for(let i=0;i<this.down.length;i++){
      const source=i===0?this.source:this.down[i-1];draw(`Lens highlight ${i}`,i===0?this.pipelines.extract:this.pipelines.downsample,source,source,this.down[i].createView(),this.weights[i]);
    }
    let blurred=this.down[this.down.length-1];
    for(let i=this.down.length-2;i>=0;i--){draw(`Lens normalized upsample ${i}`,this.pipelines.upsample,this.down[i],blurred,this.up[i].createView(),this.weights[i]);blurred=this.up[i];}
    draw("Lens scatter and single display grade",this.pipelines.display,this.source,blurred,output,this.weights[0]);
  }
  /** Retire only after the submission containing the last use has completed. */
  completeFrame(completion:Promise<unknown>):void{
    const retired=this.retired.splice(0);if(retired.length)void completion.catch(()=>{}).then(()=>{for(const r of retired)r.destroy();});
  }
  destroy():void{this.disposed=true;for(const r of [this.source,this.uniform,...this.down,...this.up,...this.weights,...this.retired])r?.destroy();this.retired=[];}
}
