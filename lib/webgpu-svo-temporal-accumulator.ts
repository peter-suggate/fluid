import { svoGBufferWGSL } from "./svo-gbuffer";
import { svoTemporalHistoryWGSL } from "./svo-temporal-history";
import type { SparseVoxelGBufferTextures } from "./webgpu-svo-gbuffer-targets";

export const SVO_TEMPORAL_ACCUMULATION_LAYOUT = Object.freeze({
  paramsBytes: 160,
  historyColorFormat: "rgba16float" as GPUTextureFormat,
  momentsFormat: "rgba16float" as GPUTextureFormat,
  keyFormat: "rgba16uint" as GPUTextureFormat,
  pingPongBytesPerPixel: 64,
  maximumAccumulationSamples: 64,
  maximumStoredSamples: 255,
} as const);

export interface SparseVoxelTemporalCameraState {
  position_m: readonly [number, number, number];
  forward: readonly [number, number, number];
  right: readonly [number, number, number];
  up: readonly [number, number, number];
}

export interface SparseVoxelTemporalFrameState {
  camera: SparseVoxelTemporalCameraState;
  deltaTime_s: number;
  cellSize_m: number;
  paused: boolean;
  /** History is dry-only and must resolve before the legacy raster-water compositor samples it. */
  composition: "dry-before-legacy-water";
}

interface TemporalHistorySet {
  color: GPUTexture;
  moments: GPUTexture;
  keyA: GPUTexture;
  keyB: GPUTexture;
}

const shader = /* wgsl */ `
${svoGBufferWGSL}
${svoTemporalHistoryWGSL}
struct TemporalParams{
  viewport:vec4f,
  currentPosition:vec4f,
  currentForwardAspect:vec4f,
  currentRightTanHalfFov:vec4f,
  currentUp:vec4f,
  previousPosition:vec4f,
  previousForward:vec4f,
  previousRight:vec4f,
  previousUp:vec4f,
  control:vec4f,
}
@group(0) @binding(0) var<uniform> temporal:TemporalParams;
@group(0) @binding(1) var currentColor:texture_2d<f32>;
@group(0) @binding(2) var currentPackedSurface:texture_2d<u32>;
@group(0) @binding(3) var currentIdentityMedia:texture_2d<u32>;
@group(0) @binding(4) var previousColor:texture_2d<f32>;
@group(0) @binding(5) var previousMoments:texture_2d<f32>;
@group(0) @binding(6) var previousKeyA:texture_2d<u32>;
@group(0) @binding(7) var previousKeyB:texture_2d<u32>;
const TEMPORAL_REQUIRED_FLAGS:u32=SVO_GBUFFER_VALID_SURFACE|SVO_GBUFFER_DEPTH_VALID|SVO_GBUFFER_GEOMETRIC_NORMAL_VALID|SVO_GBUFFER_SHADING_NORMAL_VALID|SVO_GBUFFER_MOTION_VALID|SVO_GBUFFER_MEDIA_VALID;
const TEMPORAL_TAN_HALF_FOV:f32=.72;
struct TemporalVertexOut{@builtin(position) position:vec4f}
@vertex fn temporalVertex(@builtin(vertex_index) index:u32)->TemporalVertexOut{var points=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return TemporalVertexOut(vec4f(points[index],0,1));}
struct TemporalOut{
  @location(0) historyColor:vec4f,
  @location(1) moments:vec4f,
  @location(2) keyA:vec4u,
  @location(3) keyB:vec4u,
}
fn temporalKey(color:vec4f,packed:vec4u,identity:vec4u)->SvoTemporalHitKey{
  let targets=SvoGBufferTargets(color,packed,identity);let key=svoGBufferTemporalKey(targets);
  return SvoTemporalHitKey(key.depth_m,key.geometricNormalOct,key.shadingNormalOct,key.materialOwner,key.media,key.localTopologyGeneration);
}
fn temporalPublishedKey(packed:vec4u,identity:vec4u)->array<vec4u,2>{
  let normals=packed.x;let generation=packed.y;return array<vec4u,2>(vec4u(normals&0xffffu,normals>>16u,identity.x,identity.y),vec4u(identity.z,identity.w,generation&0xffffu,generation>>16u));
}
fn temporalPreviousKey(color:vec4f,keyA:vec4u,keyB:vec4u)->SvoTemporalHitKey{
  let normals=keyA.x|(keyA.y<<16u);let generation=keyB.z|(keyB.w<<16u);let packed=vec4u(normals,generation,0u,0u);let identity=vec4u(keyA.z,keyA.w,keyB.x,keyB.y);return temporalKey(color,packed,identity);
}
fn temporalCurrentRay(pixel:vec2f)->vec3f{
  let uv=pixel*temporal.viewport.zw;let ndc=uv*2.0-1.0;return normalize(temporal.currentForwardAspect.xyz+temporal.currentRightTanHalfFov.xyz*(ndc.x*temporal.currentForwardAspect.w*temporal.currentRightTanHalfFov.w)+temporal.currentUp.xyz*(ndc.y*temporal.currentRightTanHalfFov.w));
}
fn temporalNeighborhood(pixel:vec2i,dimensions:vec2i)->array<vec3f,2>{
  var minimum=vec3f(65504.0);var maximum=vec3f(0.0);for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){let coordinate=clamp(pixel+vec2i(x,y),vec2i(0),dimensions-vec2i(1));let sample=max(textureLoad(currentColor,coordinate,0).rgb,vec3f(0.0));minimum=min(minimum,sample);maximum=max(maximum,sample);}}return array<vec3f,2>(minimum,maximum);
}
fn temporalLuminance(color:vec3f)->f32{return dot(color,vec3f(.2126,.7152,.0722));}
fn temporalSignedVelocityLane(word:u32,shift:u32)->f32{let raw=i32((word>>shift)&0x3ffu);let signed=select(raw,raw-1024,raw>=512);return f32(signed)*(SVO_GBUFFER_MAX_VELOCITY_M_S/511.0);}
fn temporalVelocity(word:u32)->vec3f{return vec3f(temporalSignedVelocityLane(word,0u),temporalSignedVelocityLane(word,10u),temporalSignedVelocityLane(word,20u));}
fn temporalVarianceClamp(colorIn:vec3f,moments:vec4f)->vec3f{
  let luminance=temporalLuminance(colorIn);let variance=max(moments.y-moments.x*moments.x,0.0);let deviation=max(2.0*sqrt(variance),.01);let clipped=clamp(luminance,moments.x-deviation,moments.x+deviation);return colorIn*select(1.0,clipped/max(luminance,1e-6),luminance>1e-6);
}
@fragment fn temporalFragment(@builtin(position) position:vec4f)->TemporalOut{
  let dimensions=vec2i(textureDimensions(currentColor));let pixel=clamp(vec2i(position.xy),vec2i(0),dimensions-vec2i(1));let current=textureLoad(currentColor,pixel,0);let packed=textureLoad(currentPackedSurface,pixel,0);let identity=textureLoad(currentIdentityMedia,pixel,0);let published=temporalPublishedKey(packed,identity);let metadataFlags=(packed.w>>4u)&0xffffu;let motionKind=packed.z>>30u;let velocity=temporalVelocity(packed.z);let supportedMotion=motionKind==SVO_TEMPORAL_MOTION_STATIC||motionKind==SVO_TEMPORAL_MOTION_RIGID;let currentUsable=(metadataFlags&TEMPORAL_REQUIRED_FLAGS)==TEMPORAL_REQUIRED_FLAGS&&supportedMotion&&current.a>0.0;
  var accepted=false;var previous=vec4f(0.0);var oldMoments=vec4f(0.0);var previousPixel=vec2i(0);var expectedPreviousDistance=0.0;
  if(temporal.control.z>.5&&currentUsable){
    let world=temporal.currentPosition.xyz+temporalCurrentRay(position.xy)*current.a;let previousWorld=world-velocity*temporal.control.y;let relative=previousWorld-temporal.previousPosition.xyz;let previousForwardDepth=dot(relative,temporal.previousForward.xyz);let previousNdc=vec2f(dot(relative,temporal.previousRight.xyz)/(max(previousForwardDepth,1e-6)*temporal.currentForwardAspect.w*TEMPORAL_TAN_HALF_FOV),dot(relative,temporal.previousUp.xyz)/(max(previousForwardDepth,1e-6)*TEMPORAL_TAN_HALF_FOV));let previousUv=previousNdc*.5+.5;let reprojectValid=previousForwardDepth>0.0&&all(previousUv>=vec2f(0.0))&&all(previousUv<vec2f(1.0));
    if(reprojectValid){previousPixel=clamp(vec2i(floor(previousUv*temporal.viewport.xy)),vec2i(0),dimensions-vec2i(1));previous=textureLoad(previousColor,previousPixel,0);oldMoments=textureLoad(previousMoments,previousPixel,0);let keyA=textureLoad(previousKeyA,previousPixel,0);let keyB=textureLoad(previousKeyB,previousPixel,0);var currentKey=temporalKey(current,packed,identity);expectedPreviousDistance=length(relative);currentKey.depth_m=expectedPreviousDistance;let previousKey=temporalPreviousKey(previous,keyA,keyB);let error=abs(previous.a-expectedPreviousDistance);accepted=oldMoments.z>0.0&&svoTemporalHistoryReason(currentKey,previousKey,temporal.control.x,temporal.control.y,velocity,motionKind,true,true,error)==SVO_TEMPORAL_REASON_ACCEPTED;}
  }
  var result=current.rgb;var sampleCount=select(-1.0,1.0,currentUsable);var pausedStable=select(0.0,1.0,temporal.control.w>.5&&currentUsable);if(accepted){let neighborhood=temporalNeighborhood(pixel,dimensions);var history=clamp(previous.rgb,neighborhood[0],neighborhood[1]);history=temporalVarianceClamp(history,oldMoments);sampleCount=min(oldMoments.z+1.0,255.0);pausedStable=select(0.0,min(oldMoments.w+1.0,255.0),temporal.control.w>.5);let accumulationCount=min(sampleCount,${SVO_TEMPORAL_ACCUMULATION_LAYOUT.maximumAccumulationSamples}.0);result=mix(current.rgb,history,(accumulationCount-1.0)/accumulationCount);}
  let luminance=min(temporalLuminance(current.rgb),255.0);let momentWeight=1.0/max(sampleCount,1.0);let mean=select(luminance,mix(oldMoments.x,luminance,momentWeight),accepted);let second=select(luminance*luminance,mix(oldMoments.y,luminance*luminance,momentWeight),accepted);return TemporalOut(vec4f(max(result,vec3f(0.0)),current.a),vec4f(mean,second,sampleCount,pausedStable),published[0],published[1]);
}
`;

async function checkedModule(device: GPUDevice): Promise<GPUShaderModule> {
  const shaderModule = device.createShaderModule({ label: "Sparse voxel temporal accumulation", code: shader });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length) throw new Error(`Sparse voxel temporal accumulation:\n${errors.map(({ lineNum, linePos, message }) => `${lineNum}:${linePos} ${message}`).join("\n")}`);
  return shaderModule;
}

function finiteCamera(camera: SparseVoxelTemporalCameraState): boolean {
  return [...camera.position_m, ...camera.forward, ...camera.right, ...camera.up].every(Number.isFinite);
}

export class SparseVoxelTemporalAccumulator {
  private layout?: GPUBindGroupLayout;
  private pipeline?: GPURenderPipeline;
  private readonly paramsBuffer: GPUBuffer;
  private history?: readonly [TemporalHistorySet, TemporalHistorySet];
  private width = 0;
  private height = 0;
  private previousIndex = 0;
  private historyValid = false;
  private previousCamera?: SparseVoxelTemporalCameraState;

  constructor(private readonly device: GPUDevice) {
    this.paramsBuffer = device.createBuffer({ label: "Sparse voxel temporal parameters", size: SVO_TEMPORAL_ACCUMULATION_LAYOUT.paramsBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async initialize(): Promise<void> {
    const shaderModule = await checkedModule(this.device);
    this.layout = this.device.createBindGroupLayout({ label: "Sparse voxel temporal bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ...[1, 4, 5].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" as const } })),
      ...[2, 3, 6, 7].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" as const } })),
    ] });
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Sparse voxel temporal accumulation",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: shaderModule, entryPoint: "temporalVertex" },
      fragment: { module: shaderModule, entryPoint: "temporalFragment", targets: [
        { format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.historyColorFormat },
        { format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.momentsFormat },
        { format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.keyFormat },
        { format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.keyFormat },
      ] },
      primitive: { topology: "triangle-list" },
    });
  }

  ensureSize(width: number, height: number): boolean {
    if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) throw new RangeError("Sparse voxel temporal dimensions must be positive safe integers");
    if (this.history && width === this.width && height === this.height) return false;
    this.releaseHistory();
    const make = (index: number): TemporalHistorySet => ({
      color: this.device.createTexture({ label: `Sparse voxel temporal HDR ${index}`, size: [width, height], format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.historyColorFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC }),
      moments: this.device.createTexture({ label: `Sparse voxel temporal moments ${index}`, size: [width, height], format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.momentsFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
      keyA: this.device.createTexture({ label: `Sparse voxel temporal key A ${index}`, size: [width, height], format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.keyFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
      keyB: this.device.createTexture({ label: `Sparse voxel temporal key B ${index}`, size: [width, height], format: SVO_TEMPORAL_ACCUMULATION_LAYOUT.keyFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
    });
    this.history = [make(0), make(1)];
    this.width = width; this.height = height; this.invalidate();
    return true;
  }

  invalidate(): void {
    this.historyValid = false;
    this.previousCamera = undefined;
  }

  encode(
    encoder: GPUCommandEncoder,
    currentTarget: GPUTexture,
    gBuffer: SparseVoxelGBufferTextures,
    frame: SparseVoxelTemporalFrameState,
  ): boolean {
    if (!this.pipeline || !this.layout || !this.history || currentTarget.width !== this.width || currentTarget.height !== this.height
      || gBuffer.width !== this.width || gBuffer.height !== this.height || frame.composition !== "dry-before-legacy-water"
      || !finiteCamera(frame.camera) || !Number.isFinite(frame.cellSize_m) || !(frame.cellSize_m > 0)
      || !Number.isFinite(frame.deltaTime_s) || frame.deltaTime_s < 0) { this.invalidate(); return false; }
    const previous = this.history[this.previousIndex], nextIndex = 1 - this.previousIndex, next = this.history[nextIndex];
    const previousCamera = this.previousCamera ?? frame.camera;
    const buffer = new Float32Array(SVO_TEMPORAL_ACCUMULATION_LAYOUT.paramsBytes / 4);
    buffer.set([this.width, this.height, 1 / this.width, 1 / this.height], 0);
    buffer.set([...frame.camera.position_m, 0], 4);
    buffer.set([...frame.camera.forward, this.width / this.height], 8);
    buffer.set([...frame.camera.right, .72], 12);
    buffer.set([...frame.camera.up, 0], 16);
    buffer.set([...previousCamera.position_m, 0], 20);
    buffer.set([...previousCamera.forward, 0], 24);
    buffer.set([...previousCamera.right, 0], 28);
    buffer.set([...previousCamera.up, 0], 32);
    buffer.set([frame.cellSize_m, frame.deltaTime_s, this.historyValid ? 1 : 0, frame.paused ? 1 : 0], 36);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    const bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.paramsBuffer } },
      { binding: 1, resource: currentTarget.createView() },
      { binding: 2, resource: gBuffer.packedSurface.createView() },
      { binding: 3, resource: gBuffer.identityMedia.createView() },
      { binding: 4, resource: previous.color.createView() },
      { binding: 5, resource: previous.moments.createView() },
      { binding: 6, resource: previous.keyA.createView() },
      { binding: 7, resource: previous.keyB.createView() },
    ] });
    const pass = encoder.beginRenderPass({ label: "Sparse voxel temporal accumulation", colorAttachments: [
      { view: next.color.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      { view: next.moments.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      { view: next.keyA.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      { view: next.keyB.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
    ] });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, bindGroup); pass.draw(3); pass.end();
    encoder.copyTextureToTexture({ texture: next.color }, { texture: currentTarget }, [this.width, this.height, 1]);
    this.previousIndex = nextIndex; this.previousCamera = frame.camera; this.historyValid = true;
    return true;
  }

  destroy(): void {
    this.releaseHistory();
    this.paramsBuffer.destroy();
    this.layout = undefined; this.pipeline = undefined;
  }

  private releaseHistory(): void {
    for (const set of this.history ?? []) for (const texture of [set.color, set.moments, set.keyA, set.keyB]) texture.destroy();
    this.history = undefined; this.width = 0; this.height = 0; this.previousIndex = 0; this.invalidate();
  }
}

export const sparseVoxelTemporalAccumulatorShader = shader;
