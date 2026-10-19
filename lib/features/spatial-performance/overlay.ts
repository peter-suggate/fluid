import type { SpatialCostSnapshot } from "../../core/spatial-performance";
import { DEFAULT_PERFORMANCE_VIEW, performanceViewRecords, type PerformanceView } from "./view";
import { cameraApertureShaderLibrary } from "../../core/webgpu-camera";

/** Generic instanced boxes/quads: no solver imports, field reads, atomics, GPU
 * timestamp queries or CPU synchronisation in the presentation pass. */
export const spatialPerformanceWGSL = /* wgsl */`
struct View { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f }
@group(0) @binding(0) var<uniform> u:View;
struct Settings { control:vec4f, scale:vec4f, low:vec4f, high:vec4f }
@group(0) @binding(1) var<uniform> s:Settings;
struct Cell { origin:vec4f, shape:vec4f, cost:vec4f }
@group(0) @binding(2) var<storage,read> cells:array<Cell>;
@group(0) @binding(3) var<storage,read> instances:array<vec2u>;
${cameraApertureShaderLibrary("u")}
struct Vertex { @builtin(position) position:vec4f, @location(0) uv:vec2f, @location(1) @interpolate(flat) color:vec3f, @location(2) @interpolate(flat) tier:f32, @location(3) @interpolate(flat) valid:f32 }
fn costColor(r:f32)->vec3f {
 let p=array<vec3f,5>(vec3f(.094,.129,.220),vec3f(.275,.267,.553),vec3f(.129,.557,.667),vec3f(.467,.784,.643),vec3f(.996,.902,.541));
 let t=clamp(select(2.0+log2(max(r,1e-20))*.5,r*4.0,r<=.25),0.0,4.0);let k=min(3u,u32(t));return mix(p[k],p[k+1u],t-f32(k));
}
fn tierColor(t:f32)->vec3f {if(t<.5){return vec3f(.25,.58,.94);}if(t<1.5){return vec3f(.98,.72,.26);}return vec3f(.96,.28,.38);}
@vertex fn vertexMain(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->Vertex {
 var result:Vertex;result.position=vec4f(2,2,2,1);
 let address=instances[instance];let cell=cells[address.x];let sub=address.y;
 let owners=u32(cell.shape.w);if(sub>=owners){return result;}
 let side=select(1.0,4.0,owners==64u);let size=vec3f(cell.origin.w,cell.shape.xy)/side;
 let origin=cell.origin.xyz+vec3f(f32(sub%4u),f32((sub/4u)%4u),f32(sub/16u))*size;
 let corner=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(0,1),vec2f(1,0),vec2f(1,1))[vertex%6u];
 var p=origin;var uv=corner;
 if(s.control.y>.5){
  let face=vertex/6u;let axis=face/2u;let high=f32(face%2u);
  if(axis==0u){p+=vec3f(high,corner)*size;}else if(axis==1u){p+=vec3f(corner.x,high,corner.y)*size;}else{p+=vec3f(corner,high)*size;}
 }else{
  // Same X/Y/Z convention as the existing field view: 1 Z, 2 X, 3 Y.
  let axis=i32(round(u.debug.x));var a=2u;if(axis==2){a=0u;}if(axis==3){a=1u;}
  let coordinate=mix(s.low[a],s.high[a],clamp(u.debug.y,0.0,.999999));
  if(coordinate<origin[a]||coordinate>=origin[a]+size[a]){return result;}
  if(a==0u){p+=vec3f(0,corner)*size;}else if(a==1u){p+=vec3f(corner.x,0,corner.y)*size;}else{p+=vec3f(corner,0)*size;}p[a]=coordinate;
 }
 let eye=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-eye);var right=cross(forward,vec3f(0,1,0));if(length(right)<1e-5){right=vec3f(1,0,0);}right=normalize(right);let up=normalize(cross(right,forward));
 let delta=p-eye;let depth=dot(delta,forward);let fov=cameraTanHalfFov();
 result.position=vec4f(dot(delta,right)/(fov*u.viewport.x/max(1.0,u.viewport.y)),dot(delta,up)/fov,max(.0001,depth-.001),depth);
 let ratio=cell.cost.x/f32(owners)/max(s.scale.x,1e-20);
 result.color=select(costColor(ratio),tierColor(cell.shape.z),s.control.w>.5);
 result.uv=uv;result.tier=cell.shape.z;result.valid=cell.cost.y;return result;
}
@fragment fn fragmentMain(v:Vertex)->@location(0) vec4f {
 let edge=min(v.uv,1.0-v.uv)/max(fwidth(v.uv),vec2f(1e-5));let line=1.0-smoothstep(.5,1.5,min(edge.x,edge.y));
 var color=v.color;
 // Slices keep a categorical rim. In 3D the cost-coloured wire is the
 // primary signal, with a faint tier halo outside it.
 var alpha=s.control.x;
 if(s.control.y>.5){
  let core=1.0-smoothstep(.15,.8,min(edge.x,edge.y));
  let halo=max(0.0,line-core);
  let fill=.08/max(1.0,s.scale.y);
  alpha*=fill+.22*core+.06*halo;
  color=mix(color,tierColor(v.tier),.06*halo/max(fill+.22*core+.06*halo,1e-5));
 }else{color=mix(color,tierColor(v.tier),line*.8);}
 if(v.valid<.5&&s.control.w<.5&&(u32(v.position.x+v.position.y)%10u)<2u){color*=.72;}
 return vec4f(color,alpha);
}`;
export class SpatialPerformanceOverlay {
  private pipeline?: GPURenderPipeline;
  private volumePipeline?: GPURenderPipeline;
  private depth?: GPUTexture;
  private width = 0;
  private height = 0;
  private buffer?: GPUBuffer;
  private instances?: GPUBuffer;
  private instanceCount = 0;
  private group?: GPUBindGroup;
  private readonly settings: GPUBuffer;
  private snapshot?: SpatialCostSnapshot;
  private key = "";
  private count = 0;
  private view = DEFAULT_PERFORMANCE_VIEW;
  private shown = false;
  constructor(private readonly device: GPUDevice, private readonly format: GPUTextureFormat, private readonly camera: GPUBuffer) {
    this.settings = device.createBuffer({ label: "Spatial performance view", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  async initialize(): Promise<void> {
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ] })] });
    const module = this.device.createShaderModule({ label: "Spatial performance overlay", code: spatialPerformanceWGSL });
    this.pipeline = await this.device.createRenderPipelineAsync({ label: "Spatial performance overlay", layout,
      vertex: { module, entryPoint: "vertexMain" }, fragment: { module, entryPoint: "fragmentMain", targets: [{ format: this.format,
        blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] }, primitive: { topology: "triangle-list" }, depthStencil: { format: "depth24plus", depthCompare: "less", depthWriteEnabled: true } });
    // X-ray volume view: every depth contributes; no front box can hide the
    // interior. Share the slice bind layout so changing views needs no rebind.
    this.volumePipeline = await this.device.createRenderPipelineAsync({ label: "Spatial performance transparent volume",
      layout,
      vertex: { module, entryPoint: "vertexMain" }, fragment: { module, entryPoint: "fragmentMain", targets: [{ format: this.format,
        blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] },
      primitive: { topology: "triangle-list" }, depthStencil: { format: "depth24plus", depthCompare: "always", depthWriteEnabled: false } });
  }
  set(snapshot: SpatialCostSnapshot | undefined, view: PerformanceView, opacity: number, width = 1, height = 1): void {
    this.shown = opacity > 0 && !!snapshot;this.view = view;
    if (!snapshot || !this.pipeline || !this.shown) return;
    if (width !== this.width || height !== this.height) {
      this.width = width; this.height = height;this.depth?.destroy();
      this.depth = this.device.createTexture({ label: "Performance diagnostic depth", size: [Math.max(1,width), Math.max(1,height)], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const key = JSON.stringify([view, opacity]);
    if (snapshot === this.snapshot && key === this.key) return;
    this.snapshot = snapshot;this.key = key;
    const data = performanceViewRecords(snapshot, view);this.count = data.count;
    if (!data.count) return;
    if (!this.buffer || this.buffer.size < data.records.byteLength) {
      this.buffer?.destroy();this.buffer = this.device.createBuffer({ label: "Spatial performance cells", size: data.records.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    // Draw each actual owner once. A coarse tile must not launch 63 discarded
    // cell instances just because a neighbouring tile is fine.
    this.instanceCount = data.entities;
    const instances = new Uint32Array(2 * this.instanceCount);
    for (let i = 0, at = 0; i < data.count; i++) for (let sub = 0; sub < data.records[12 * i + 7]!; sub++) { instances[at++] = i; instances[at++] = sub; }
    if (!this.instances || this.instances.size < instances.byteLength) {
      this.instances?.destroy();this.instances = this.device.createBuffer({ label: "Spatial performance owners", size: instances.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    this.group = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.camera } }, { binding: 1, resource: { buffer: this.settings } },
      { binding: 2, resource: { buffer: this.buffer } }, { binding: 3, resource: { buffer: this.instances } },
    ] });
    this.device.queue.writeBuffer(this.instances, 0, instances);
    this.device.queue.writeBuffer(this.buffer, 0, data.records);
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    for (const c of snapshot.cells) c.origin.forEach((v, a) => { low[a] = Math.min(low[a]!, v); high[a] = Math.max(high[a]!, v + c.size[a]!); });
    this.device.queue.writeBuffer(this.settings, 0, new Float32Array([opacity, +(view.dimension === "3d"), +(view.grain === "cell"), +(view.paint === "tiers"), snapshot.report.wall_ms / Math.max(1, data.entities), Math.cbrt(Math.max(1, data.entities)), 0, 0, ...low, 0, ...high, 0]));
  }
  encode(encoder: GPUCommandEncoder, target: GPUTextureView): boolean {
    if (!this.shown || !this.count || !this.pipeline || !this.group) return false;
    const pass = encoder.beginRenderPass({ label: "Spatial performance overlay", colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }], depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" } });
    pass.setPipeline(this.view.dimension === "3d" ? this.volumePipeline! : this.pipeline);pass.setBindGroup(0, this.group);
    pass.draw(this.view.dimension === "3d" ? 36 : 6, this.instanceCount);pass.end();return true;
  }
  destroy(): void { this.depth?.destroy(); this.instances?.destroy(); this.buffer?.destroy();this.settings.destroy(); }
}
