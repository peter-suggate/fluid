import type { ClothDocument, ClothVector } from './types';
import { cameraApertureShaderLibrary } from '../core/webgpu-camera';
import { SVO_DRY_SCENE_REVERSED_Z_NEAR_M } from '../svo/features/shading/program';

/** GPU-resident cloth geometry the solver owns and the renderer reads in
 * place: no CPU copy of the pose sits between the step and the draw. */
export interface ClothRenderSource {
  readonly document: ClothDocument;
  /** array<vec4f>: cloth-local positions (xyz). */
  readonly positions: GPUBuffer;
  /** array<vec4f>: unit vertex normals (xyz), cloth-local. */
  readonly normals: GPUBuffer;
  /** array<vec4u>: vertex indices (xyz) and material (w). */
  readonly triangles: GPUBuffer;
  readonly triangleCount: number;
  /** World position of the cloth-local origin. */
  readonly origin_m: ClothVector;
  /** Changes whenever the buffers above are replaced (topology edits). */
  readonly revision: number;
  /** Accepted step count: the drawn pose changes with it. */
  readonly tick: number;
}

/** Draws the solver's surface into the opaque scene consumed by water optics.
 * Alpha is ray distance in metres, not opacity. */
export const clothSurfaceWGSL = /* wgsl */ `
struct View { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, cameraBasisPadding:array<vec4f,23>, cameraBasisRight:vec4f }
struct Cloth { origin:vec4f, colors:array<vec4f,15> }
@group(0) @binding(0) var<uniform> u:View;
@group(0) @binding(1) var<storage,read> positions:array<vec4f>;
@group(0) @binding(2) var<storage,read> normals:array<vec4f>;
@group(0) @binding(3) var<storage,read> triangles:array<vec4u>;
@group(0) @binding(4) var<uniform> cloth:Cloth;
${cameraApertureShaderLibrary('u')}
const NEAR:f32=${SVO_DRY_SCENE_REVERSED_Z_NEAR_M};
fn forward()->vec3f{return normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);}
fn project(p:vec3f)->vec4f{
 let f=forward();let r=cameraRightDirection(f);let up=cross(r,f);let q=p-u.cameraPosition.xyz;
 let d=dot(q,f);let aperture=cameraTanHalfFov();
 return vec4f(dot(q,r)/(u.viewport.x/u.viewport.y*aperture),dot(q,up)/aperture,NEAR,d);
}
struct Out { @builtin(position) clip:vec4f,@location(0) world:vec3f,@location(1) normal:vec3f,@location(2) @interpolate(flat) color:vec3f }
@vertex fn shellVertex(@builtin(vertex_index) i:u32)->Out {
 let t=triangles[i/3u];let v=t[i%3u];let p=positions[v].xyz+cloth.origin.xyz;
 var o:Out;o.clip=project(p);o.world=p;o.normal=normals[v].xyz;o.color=cloth.colors[min(t.w,14u)].xyz;return o;
}
struct Fragment { @location(0) color:vec4f,@builtin(frag_depth) depth:f32 }
@fragment fn shellFragment(v:Out)->Fragment{
 let view=normalize(u.cameraPosition.xyz-v.world);var n=normalize(v.normal);n=select(n,-n,dot(n,view)<0.0);
 let light=normalize(vec3f(-.4,1,.6));
 var o:Fragment;o.color=vec4f(v.color*(.3+.7*max(0.0,dot(n,light))),length(v.world-u.cameraPosition.xyz));
 o.depth=NEAR/dot(v.world-u.cameraPosition.xyz,forward());return o;
}
`;

/** Per-material colours, at most 15 materials per document (the uniform's size). */
export function packClothRenderUniform(source: Pick<ClothRenderSource, 'document' | 'origin_m'>): Float32Array<ArrayBuffer> {
  const materials = source.document.materials;
  if (materials.length > 15) throw new Error('Cloth renderer supports at most 15 materials per document');
  const values = new Float32Array(64);
  values.set([...source.origin_m, 0], 0);
  materials.forEach((m, k) => values.set([...(m.baseColor ?? [0.72, 0.66, 0.51]), 1], 4 + 4 * k));
  return values;
}

interface Bound { source: ClothRenderSource; group: GPUBindGroup; uniform: GPUBuffer }
export class ClothSurfaceRenderer {
  private layout?: GPUBindGroupLayout;
  private pipeline?: GPURenderPipeline;
  private readonly bound = new Map<string, Bound>();
  constructor(private readonly device: GPUDevice, private readonly view: GPUBuffer) {}
  async initialize(): Promise<void> {
    const storage = {type: 'read-only-storage' as const};
    this.layout = this.device.createBindGroupLayout({label: 'Cloth surface bindings', entries: [
      {binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {type: 'uniform'}},
      {binding: 1, visibility: GPUShaderStage.VERTEX, buffer: storage},
      {binding: 2, visibility: GPUShaderStage.VERTEX, buffer: storage},
      {binding: 3, visibility: GPUShaderStage.VERTEX, buffer: storage},
      {binding: 4, visibility: GPUShaderStage.VERTEX, buffer: {type: 'uniform'}}]});
    const module = this.device.createShaderModule({label: 'Cloth surface', code: clothSurfaceWGSL});
    this.pipeline = await this.device.createRenderPipelineAsync({label: 'Cloth shell triangles', layout: this.device.createPipelineLayout({bindGroupLayouts: [this.layout]}),
      vertex: {module, entryPoint: 'shellVertex'}, fragment: {module, entryPoint: 'shellFragment', targets: [{format: 'rgba16float'}]},
      primitive: {topology: 'triangle-list', cullMode: 'none'}, depthStencil: {format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater'}});
  }
  /** Bind the solver's current buffers; unchanged sources keep their bind groups. */
  setSources(sources: readonly ClothRenderSource[]): void {
    if (!this.layout) throw new Error('Cloth renderer is not initialized');
    const live = new Set(sources.map(s => s.document.id));
    for (const [id, b] of this.bound) if (!live.has(id)) { b.uniform.destroy(); this.bound.delete(id); }
    for (const source of sources) {
      const old = this.bound.get(source.document.id);
      if (old && old.source.revision === source.revision && old.source.positions === source.positions && old.source.document === source.document) continue;
      const uniform = old?.uniform ?? this.device.createBuffer({label: 'Cloth render uniform', size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
      this.device.queue.writeBuffer(uniform, 0, packClothRenderUniform(source));
      const group = this.device.createBindGroup({label: 'Cloth surface', layout: this.layout, entries: [
        {binding: 0, resource: {buffer: this.view}}, {binding: 1, resource: {buffer: source.positions}},
        {binding: 2, resource: {buffer: source.normals}}, {binding: 3, resource: {buffer: source.triangles}}, {binding: 4, resource: {buffer: uniform}}]});
      this.bound.set(source.document.id, {source, group, uniform});
    }
  }
  encode(encoder: GPUCommandEncoder, target: GPUTextureView, depth: GPUTextureView): void {
    if (!this.pipeline || !this.bound.size) return;
    const pass = encoder.beginRenderPass({label: 'Cloth physical surfaces', colorAttachments: [{view: target, loadOp: 'load', storeOp: 'store'}], depthStencilAttachment: {view: depth, depthLoadOp: 'load', depthStoreOp: 'store'}});
    pass.setPipeline(this.pipeline);
    for (const b of this.bound.values()) { pass.setBindGroup(0, b.group); pass.draw(3 * b.source.triangleCount); }
    pass.end();
  }
  destroy(): void { for (const b of this.bound.values()) b.uniform.destroy(); this.bound.clear(); }
}
