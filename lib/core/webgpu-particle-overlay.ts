import { fieldVisualization, type Visualization } from "./visualization-registry";
import { LAYER_PALETTE, LAYER_PARTICLE_SPEED_SCALE, PARTICLE_LEGEND } from "./visual-layers";

/**
 * Draws a particle method's own particles as shaded spheres over the frame.
 *
 * The surface render shows what the particles were reconstructed into; this
 * shows the samples themselves: where a narrow band keeps them and where it
 * has none, how evenly a cell is populated, where they bunch or thin out.
 *
 * Each sphere is an impostor: one camera-facing quad per particle whose
 * fragment stage rebuilds the sphere's normal and depth, so the spheres shade,
 * intersect and occlude one another through a depth buffer this pass owns.
 * Opaque scenery hides them through the dry scene's depth; the water does not,
 * because the particles are inside it and the liquid is what they are drawn
 * over.
 *
 * Records are read straight out of the solver's particle buffer. There is no
 * readback and no staging copy. A solver whose live count exists only on the
 * GPU publishes where that word is, and it is copied into this pass's indirect
 * draw arguments on the queue, so the draw covers exactly the live prefix.
 */

/** Where a solver keeps its particles, and how to read one record. */
export interface GPUFluidParticleSource {
  /**
   * Particle records, `strideFloats` f32 each: position in floats 0..2,
   * a positive float 3 while the record is live, velocity (m/s) in 4..6.
   */
  readonly buffer: GPUBuffer;
  readonly strideFloats: number;
  /** Records the buffer holds. The draw never reads past it. */
  readonly capacity: number;
  /** Metres per position unit. Positions are tank-local with a corner origin. */
  readonly positionScale_m: readonly [number, number, number];
  /** Sphere radius: a little under half the seed spacing. */
  readonly radius_m: number;
  /**
   * The live records are the prefix this u32 counts. Absent: every one of
   * `capacity` records may be live.
   */
  readonly liveCount?: { readonly buffer: GPUBuffer; readonly byteOffset: number };
}

export const PARTICLE_OVERLAY_UNIFORM_BYTES = 128;

export interface ParticleOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  /** tan(verticalFieldOfView / 2). */
  readonly tanHalfFov: number;
  readonly aspect: number;
}

export interface ParticleOverlayFrame {
  readonly camera: ParticleOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly container_m: readonly [number, number, number];
  /** Reversed-Z near plane the scene depth was written with. */
  readonly depthNear_m: number;
  /** 1 draws opaque spheres. */
  readonly opacity?: number;
}

const paletteWGSL = (name: string, rgb: readonly number[]) =>
  `const ${name}:vec3f=vec3f(${rgb.map(n => (n / 255).toFixed(8)).join(",")});`;

export const particleOverlayShader = /* wgsl */ `
struct ParticleOverlayUniforms {
  // xyz camera position, w tan(halfFov)
  cameraPosition:vec4f,
  // xyz camera forward, w aspect
  cameraForward:vec4f,
  // xyz camera right, w opacity
  cameraRight:vec4f,
  // xyz camera up, w sphere radius in metres
  cameraUp:vec4f,
  // xy viewport pixels, z reversed-Z near metres, w scene depth valid
  viewport:vec4f,
  // xyz metres per position unit, w full-scale speed in m/s
  scale:vec4f,
  // xyz world position of the tank's minimum corner
  origin:vec4f,
  // x record stride in floats, y record capacity
  records:vec4u,
}

@group(0) @binding(0) var<uniform> overlay:ParticleOverlayUniforms;
@group(0) @binding(1) var<storage, read> particles:array<f32>;
@group(0) @binding(2) var sceneDepth:texture_depth_2d;

struct VertexOut {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
  // Unit-disc coordinate across the quad.
  @location(1) disc:vec2f,
  // x view depth of the sphere centre, y drawn radius, both in metres.
  @location(2) sphere:vec2f,
}

struct FragmentOut {
  @location(0) color:vec4f,
  @builtin(frag_depth) depth:f32,
}

${paletteWGSL("PARTICLE_SLOW", LAYER_PALETTE.particleSlow)}
${paletteWGSL("PARTICLE_MID", LAYER_PALETTE.particleMid)}
${paletteWGSL("PARTICLE_FAST", LAYER_PALETTE.particleFast)}
// A sphere smaller than this would alias into crawling noise, so a distant
// cloud is drawn slightly fat instead.
const PARTICLE_MINIMUM_PIXELS:f32=0.75;

fn particleSpeedColor(speed:f32)->vec3f {
  let t=clamp(speed/max(overlay.scale.w,1e-6),0.0,1.0);
  return mix(mix(PARTICLE_SLOW,PARTICLE_MID,clamp(2.0*t,0.0,1.0)),PARTICLE_FAST,clamp(2.0*t-1.0,0.0,1.0));
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->VertexOut {
  var corners=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),
    vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  var output:VertexOut;
  output.color=vec3f(0.0);
  output.disc=corners[vertexIndex];
  output.sphere=vec2f(0.0);
  // A collapsed quad: a retired record, or one past the buffer.
  output.position=vec4f(0.0,0.0,0.0,1.0);
  if (instance>=overlay.records.y) { return output; }
  let at=overlay.records.x*instance;
  if (!(particles[at+3u]>0.0)) { return output; }
  let local=vec3f(particles[at],particles[at+1u],particles[at+2u]);
  let velocity=vec3f(particles[at+4u],particles[at+5u],particles[at+6u]);
  let world=local*overlay.scale.xyz+overlay.origin.xyz;
  let relative=world-overlay.cameraPosition.xyz;
  let view=vec3f(dot(relative,overlay.cameraRight.xyz),
    dot(relative,overlay.cameraUp.xyz),dot(relative,overlay.cameraForward.xyz));
  let near=overlay.viewport.z;
  let tangent=max(overlay.cameraPosition.w,1e-4);
  let viewport=max(overlay.viewport.xy,vec2f(1.0));
  let pixelsPerMetre=0.5*viewport.y/(max(view.z,near)*tangent);
  let radius=max(overlay.cameraUp.w,PARTICLE_MINIMUM_PIXELS/pixelsPerMetre);
  // The whole sphere stays in front of the near plane, so its depth is valid.
  if (!(view.z-radius>near)) { return output; }
  let corner=view.xy+corners[vertexIndex]*radius;
  output.position=vec4f(corner.x/(view.z*tangent*max(overlay.cameraForward.w,1e-4)),
    corner.y/(view.z*tangent),0.0,1.0);
  output.color=particleSpeedColor(length(velocity));
  output.sphere=vec2f(view.z,radius);
  return output;
}

@fragment fn fragmentMain(input:VertexOut)->FragmentOut {
  let radial=dot(input.disc,input.disc);
  if (radial>1.0 || input.sphere.y<=0.0) { discard; }
  let towardCamera=sqrt(1.0-radial);
  let viewDepth_m=input.sphere.x-input.sphere.y*towardCamera;
  let near=overlay.viewport.z;
  if (overlay.viewport.w>0.5) {
    // Reversed-Z: depth = near / viewDepth, and zero means nothing was drawn.
    let stored=textureLoad(sceneDepth,vec2u(input.position.xy),0);
    if (stored>0.0 && viewDepth_m>near/stored) { discard; }
  }
  let normal=normalize(overlay.cameraRight.xyz*input.disc.x+overlay.cameraUp.xyz*input.disc.y
    -overlay.cameraForward.xyz*towardCamera);
  // The key light rides over the viewer's left shoulder, so the spheres read
  // as round from every orbit rather than going flat on the side away from
  // a fixed sun.
  let light=normalize(0.65*overlay.cameraUp.xyz-0.40*overlay.cameraRight.xyz-0.65*overlay.cameraForward.xyz);
  let diffuse=max(dot(normal,light),0.0);
  // Sky above, bounce below: the unlit side still reads as round.
  let ambient=mix(0.28,0.46,0.5+0.5*normal.y);
  let highlight=pow(max(dot(normal,normalize(light-overlay.cameraForward.xyz)),0.0),32.0);
  var output:FragmentOut;
  output.color=vec4f(input.color*(ambient+0.58*diffuse)+vec3f(0.18*highlight),
    clamp(overlay.cameraRight.w,0.0,1.0));
  output.depth=clamp(near/max(viewDepth_m,near),0.0,1.0);
  return output;
}
`;

/** One instanced draw over the solver's resident particle records. */
export class ParticleOverlay {
  private pipeline?: GPURenderPipeline;
  private layout?: GPUBindGroupLayout;
  private uniforms?: GPUBuffer;
  private indirect?: GPUBuffer;
  private fallbackDepth?: GPUTexture;
  private fallbackDepthView?: GPUTextureView;
  private depth?: GPUTexture;
  private depthView?: GPUTextureView;
  /** One bind group per particle buffer: a solver may alternate two. */
  private readonly bindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private boundSceneDepth?: GPUTextureView;
  private source?: GPUFluidParticleSource;
  private destroyed = false;
  private readonly uniformData = new ArrayBuffer(PARTICLE_OVERLAY_UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
  ) {}

  async initialize(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Fluid particle overlay", code: particleOverlayShader,
    });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`Fluid particle overlay:\n${errors
        .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`)
        .join("\n")}`);
    }
    this.uniforms = this.device.createBuffer({
      label: "Fluid particle overlay uniforms",
      size: PARTICLE_OVERLAY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // vertexCount, instanceCount, firstVertex, firstInstance. The instance
    // count is the only word a frame rewrites.
    this.indirect = this.device.createBuffer({
      label: "Fluid particle overlay draw arguments",
      size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indirect, 0, new Uint32Array([6, 0, 0, 0]));
    this.fallbackDepth = this.device.createTexture({
      label: "Fluid particle overlay scene depth fallback",
      size: [1, 1], format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fallbackDepthView = this.fallbackDepth.createView();
    this.layout = this.device.createBindGroupLayout({
      label: "Fluid particle overlay bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth" } },
      ],
    });
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Fluid particle overlay",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: shaderModule, entryPoint: "vertexMain" },
      fragment: {
        module: shaderModule, entryPoint: "fragmentMain",
        targets: [{
          format: this.targetFormat,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
      // Reversed-Z, as the scene depth is: nearer is greater, and the clear is 0.
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" },
    });
  }

  get ready(): boolean {
    return Boolean(this.pipeline && this.uniforms && this.layout && this.indirect);
  }

  setSource(source: GPUFluidParticleSource | undefined): void {
    this.source = source;
  }

  private bindGroupFor(buffer: GPUBuffer, sceneDepth: GPUTextureView | undefined): GPUBindGroup {
    const view = sceneDepth ?? this.fallbackDepthView!;
    if (view !== this.boundSceneDepth) { this.bindGroups.clear(); this.boundSceneDepth = view; }
    let group = this.bindGroups.get(buffer);
    if (!group) {
      // A third buffer means a new solver: the cached two are its predecessor's.
      if (this.bindGroups.size >= 2) this.bindGroups.clear();
      group = this.device.createBindGroup({
        label: "Fluid particle overlay bind group",
        layout: this.layout!,
        entries: [
          { binding: 0, resource: { buffer: this.uniforms! } },
          { binding: 1, resource: { buffer } },
          { binding: 2, resource: view },
        ],
      });
      this.bindGroups.set(buffer, group);
    }
    return group;
  }

  private depthViewFor(width: number, height: number): GPUTextureView {
    if (!this.depth || this.depth.width !== width || this.depth.height !== height) {
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        label: "Fluid particle overlay depth",
        size: [width, height], format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.depthView = this.depth.createView();
    }
    return this.depthView!;
  }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sceneDepth: GPUTextureView | undefined,
    frame: ParticleOverlayFrame,
  ): boolean {
    const source = this.source;
    if (!this.ready || !source || source.capacity === 0) return false;
    const { camera } = frame;
    const width = Math.max(1, Math.round(frame.viewportWidth));
    const height = Math.max(1, Math.round(frame.viewportHeight));
    const f = this.uniformF32, u = this.uniformU32;
    f.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    f.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    f.set([...camera.right, Math.max(0, Math.min(1, frame.opacity ?? 1))], 8);
    f.set([...camera.up, source.radius_m], 12);
    f.set([width, height, Math.max(1e-6, frame.depthNear_m), sceneDepth ? 1 : 0], 16);
    f.set([...source.positionScale_m, LAYER_PARTICLE_SPEED_SCALE], 20);
    f.set([-0.5 * frame.container_m[0], 0, -0.5 * frame.container_m[2], 0], 24);
    u.set([source.strideFloats, source.capacity, 0, 0], 28);
    this.device.queue.writeBuffer(this.uniforms!, 0, this.uniformData);
    if (source.liveCount) {
      encoder.copyBufferToBuffer(source.liveCount.buffer, source.liveCount.byteOffset, this.indirect!, 4, 4);
    }
    const pass = encoder.beginRenderPass({
      label: "Fluid particle overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: {
        view: this.depthViewFor(width, height),
        depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "discard",
      },
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroupFor(source.buffer, sceneDepth));
    if (source.liveCount) pass.drawIndirect(this.indirect!, 0);
    else pass.draw(6, source.capacity);
    pass.end();
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniforms?.destroy();
    this.indirect?.destroy();
    this.fallbackDepth?.destroy();
    this.depth?.destroy();
    this.bindGroups.clear();
    this.uniforms = undefined;
    this.indirect = undefined;
    this.fallbackDepth = undefined;
    this.depth = undefined;
    this.pipeline = undefined;
    this.source = undefined;
  }
}

/** The same spheres as a single field view, for methods on the one-view row. */
export const particleOverlayVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "fluid-particles/spheres", pass: "Fluid particles",
    label: "Particles",
    description: "The method's own particles as shaded spheres, coloured by speed. Shows the samples the surface is reconstructed from: how evenly cells are populated and where particles bunch or thin out.",
    source: "Solver-resident particle records, drawn on the GPU without readback",
    mode: "particles", axis: "volume", planeless: true, icon: "tracers",
    swatch: "#8f8cdb",
    legend: PARTICLE_LEGEND.map(entry => ({ swatch: entry.color, mark: "point" as const, label: entry.label })),
  }),
]);
