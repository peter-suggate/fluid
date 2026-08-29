import { fieldVisualization, type Visualization } from "./visualization-registry";

/**
 * Draws the solver's fluid markers as a coloured cloud over the finished image.
 *
 * The question this view answers is the one a still frame cannot: *which* water
 * is this. A surface render shows where the liquid is; it cannot show that the
 * front of a collapsing column is made of what used to be its base, or that a
 * splash re-entering the pool never mixes below its own crater. Colouring each
 * parcel by where it started and following it is the standard way that gets
 * shown in the particle literature, and it reads the same way here.
 *
 * **Nothing about the colour is stored or advected.** A marker's seed position
 * is a pure function of its index (see `sparseCM12TracerLattice`), so the vertex
 * stage below recomputes it from `instance_index` and the solver only ever
 * carries four floats per marker: where it is now, and whether it is still
 * alive. Two consequences are worth being explicit about, because they are the
 * reason this is a marker cloud rather than a dyed field:
 *
 * - The spectrum never blurs. A passive scalar advected on the grid would be
 *   smeared by the transport scheme within a few hundred steps — which destroys
 *   exactly the information the view exists to show. A colour that never passes
 *   through an advection scheme cannot be numerically diffused.
 * - Cost is decoupled from resolution. The marker count is a budget, not a
 *   fraction of the grid, so refining the scene does not refine the cloud.
 *
 * Markers are read straight out of the solver's own state buffer at the offset
 * it reserved for them. There is no readback, no staging copy, and no host-side
 * geometry: the buffer the physics wrote this frame is the buffer this draw
 * indexes.
 */

/** Where a solver keeps its markers, and the arithmetic that colours them. */
export interface GPUFluidTracerSource {
  /** The solver's state buffer. Markers live at `floatOffset`, 4 floats each. */
  readonly buffer: GPUBuffer;
  readonly floatOffset: number;
  readonly count: number;
  /** Seed lattice extent; an index decomposes against this into its colour. */
  readonly latticeDimensions: readonly [number, number, number];
  readonly originFine: readonly [number, number, number];
  readonly spacingFine: number;
  /** Fine-lattice domain extent, for the lattice-to-metres mapping. */
  readonly domainFine: readonly [number, number, number];
}

export const TRACER_OVERLAY_UNIFORM_BYTES = 144;

export interface TracerOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  /** tan(verticalFieldOfView / 2). */
  readonly tanHalfFov: number;
  readonly aspect: number;
}

export interface TracerOverlayFrame {
  readonly camera: TracerOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly container_m: readonly [number, number, number];
  /** Reversed-Z near plane the scene depth was written with. */
  readonly depthNear_m: number;
  readonly globalAlpha?: number;
  /** Alpha multiplier for markers behind opaque scenery. */
  readonly occludedAlpha?: number;
}

/**
 * The faintest a marker cloud is ever drawn.
 *
 * The opacity comes from the field flyout's slice control, which a plane view
 * is free to park at zero. Carrying that across to a marker view would draw
 * nothing, and a view that draws nothing is indistinguishable from one that is
 * broken — so the floor is here rather than in the control, which has its own
 * reasons to reach zero.
 */
const TRACER_MINIMUM_ALPHA = 0.05;

export const tracerOverlayShader = /* wgsl */ `
struct TracerOverlayUniforms {
  // xyz camera position, w tan(halfFov)
  cameraPosition:vec4f,
  // xyz camera forward, w aspect
  cameraForward:vec4f,
  // xyz camera right, w global alpha
  cameraRight:vec4f,
  // xyz camera up, w occluded alpha
  cameraUp:vec4f,
  // xy viewport pixels, z reversed-Z near metres, w depth valid
  viewport:vec4f,
  // xyz container metres, w marker world radius in metres
  container:vec4f,
  // xyz fine-lattice domain extent, w unused
  domain:vec4f,
  // xyz seed lattice dimensions, w marker float offset into the state buffer
  lattice:vec4u,
}

@group(0) @binding(0) var<uniform> overlay:TracerOverlayUniforms;
@group(0) @binding(1) var<storage, read> markers:array<f32>;
@group(0) @binding(2) var sceneDepth:texture_depth_2d;

struct VertexOut {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
  @location(1) alpha:f32,
  // Unit-disc coordinate across the sprite, for the round falloff.
  @location(2) disc:vec2f,
  @location(3) viewDepth_m:f32,
}

const TRACER_NEAR_VIEW_M:f32=0.02;
const TRACER_MINIMUM_PIXELS:f32=1.0;
const TRACER_MAXIMUM_PIXELS:f32=22.0;

fn tracerViewSpace(point:vec3f)->vec3f {
  let relative=point-overlay.cameraPosition.xyz;
  return vec3f(dot(relative,overlay.cameraRight.xyz),
    dot(relative,overlay.cameraUp.xyz),dot(relative,overlay.cameraForward.xyz));
}

fn tracerNdc(view:vec3f)->vec2f {
  let tangent=max(overlay.cameraPosition.w,1e-4);
  let depth=max(view.z,TRACER_NEAR_VIEW_M);
  return vec2f(view.x/(depth*tangent*max(overlay.cameraForward.w,1e-4)),
    view.y/(depth*tangent));
}

// The seed lattice index this marker was born at. The compute pass derives the
// same triple from the same index, which is what lets the colour live nowhere.
fn tracerSeedLattice(instance:u32)->vec3u {
  let dimensions=max(overlay.lattice.xyz,vec3u(1u));
  return vec3u(instance%dimensions.x,(instance/dimensions.x)%dimensions.y,
    instance/(dimensions.x*dimensions.y));
}

fn tracerSeedColor(instance:u32)->vec3f {
  let dimensions=max(overlay.lattice.xyz,vec3u(1u));
  let normalized=(vec3f(tracerSeedLattice(instance))+vec3f(0.5))/vec3f(dimensions);
  // Lifted off black so a marker seeded at the domain corner is still visible,
  // and pushed away from grey so that two parcels from different corners stay
  // distinguishable after alpha compositing has averaged their neighbours.
  var color=vec3f(0.16)+vec3f(0.84)*smoothstep(vec3f(0.0),vec3f(1.0),normalized);
  let luminance=dot(color,vec3f(0.299,0.587,0.114));
  return clamp(luminance+(color-luminance)*1.45,vec3f(0.04),vec3f(1.0));
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->VertexOut {
  var corners=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),
    vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  var output:VertexOut;
  output.color=vec3f(0.0);
  output.alpha=0.0;
  output.disc=corners[vertexIndex];
  output.viewDepth_m=0.0;
  // A collapsed quad is how a retired marker costs nothing: the solver leaves
  // dead lanes in place rather than compacting a buffer the topology would
  // invalidate every epoch anyway.
  output.position=vec4f(0.0,0.0,0.0,1.0);

  let at=overlay.lattice.w+4u*instance;
  if (markers[at+3u]<0.5) { return output; }
  let fine=vec3f(markers[at],markers[at+1u],markers[at+2u]);
  // Fine-lattice coordinates are corner-origin and continuous: 0 is the domain
  // minimum and the domain extent is its maximum, on every axis.
  let half=0.5*overlay.container.xyz;
  let world=fine/max(overlay.domain.xyz,vec3f(1.0))*overlay.container.xyz
    +vec3f(-half.x,0.0,-half.z);
  let view=tracerViewSpace(world);
  if (view.z<TRACER_NEAR_VIEW_M) { return output; }

  let tangent=max(overlay.cameraPosition.w,1e-4);
  let viewport=max(overlay.viewport.xy,vec2f(1.0));
  // A world-space radius, projected. Constant-pixel dots would give the cloud no
  // depth cue at all; the clamp is what stops a distant cloud from dissolving
  // into sub-pixel noise or a near one from covering the frame.
  let radiusPixels=clamp(overlay.container.w/(view.z*tangent)*0.5*viewport.y,
    TRACER_MINIMUM_PIXELS,TRACER_MAXIMUM_PIXELS);
  let pixel=tracerNdc(view)*0.5*viewport+corners[vertexIndex]*radiusPixels;
  output.position=vec4f(pixel/(0.5*viewport),0.0,1.0);
  output.color=tracerSeedColor(instance);
  output.alpha=clamp(overlay.cameraRight.w,0.0,1.0);
  output.viewDepth_m=view.z;
  return output;
}

@fragment fn fragmentMain(input:VertexOut)->@location(0) vec4f {
  if (input.alpha<=0.0) { discard; }
  let radial=dot(input.disc,input.disc);
  if (radial>1.0) { discard; }
  // Soft edge, so a cloud of one-pixel dots reads as a body rather than as
  // aliasing that crawls when the camera moves.
  var alpha=input.alpha*(1.0-smoothstep(0.35,1.0,radial));
  if (overlay.viewport.w>0.5) {
    let coordinate=vec2u(input.position.xy);
    let stored=textureLoad(sceneDepth,coordinate,0);
    // Reversed-Z: depth = near / viewDepth, and zero means nothing was drawn.
    // The depth here is the dry scene's, not the water's, so this dims markers
    // behind walls and terrain while leaving submerged ones at full strength —
    // which is the point, since the interior is what the view is for.
    if (stored>0.0) {
      let sceneDepth_m=overlay.viewport.z/max(stored,1e-7);
      if (input.viewDepth_m>sceneDepth_m) { alpha*=max(overlay.cameraUp.w,0.0); }
    }
  }
  return vec4f(input.color,alpha);
}
`;

/** One instanced draw over the solver's resident marker range. */
export class TracerOverlay {
  private pipeline?: GPURenderPipeline;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private uniforms?: GPUBuffer;
  private fallbackDepth?: GPUTexture;
  private fallbackDepthView?: GPUTextureView;
  private fallbackMarkers?: GPUBuffer;
  private boundDepthView?: GPUTextureView;
  private boundBuffer?: GPUBuffer;
  private source?: GPUFluidTracerSource;
  private destroyed = false;
  private readonly uniformData =
    new ArrayBuffer(TRACER_OVERLAY_UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
  ) {}

  async initialize(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Fluid tracer overlay", code: tracerOverlayShader,
    });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`Fluid tracer overlay:\n${errors
        .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`)
        .join("\n")}`);
    }
    this.uniforms = this.device.createBuffer({
      label: "Fluid tracer overlay uniforms",
      size: TRACER_OVERLAY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.fallbackDepth = this.device.createTexture({
      label: "Fluid tracer overlay depth fallback",
      size: [1, 1], format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fallbackDepthView = this.fallbackDepth.createView();
    // A bind group needs a buffer even in the frame before a solver publishes
    // one; the draw is skipped by count, not by a missing binding.
    this.fallbackMarkers = this.device.createBuffer({
      label: "Fluid tracer overlay marker fallback",
      size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.layout = this.device.createBindGroupLayout({
      label: "Fluid tracer overlay bindings",
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
      label: "Fluid tracer overlay",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: shaderModule, entryPoint: "vertexMain" },
      fragment: {
        module: shaderModule, entryPoint: "fragmentMain",
        targets: [{
          format: this.targetFormat,
          // Ordinary alpha, not the line overlay's additive blend: a marker
          // cloud piles up hundreds deep along a view ray, and additive would
          // saturate every dense region to white — which is exactly where the
          // interesting structure is.
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.rebuildBindGroup(this.fallbackDepthView);
  }

  get ready(): boolean {
    return Boolean(this.pipeline && this.uniforms && this.layout);
  }

  /** Idempotent: re-binding the same buffer must not rebuild a bind group. */
  setSource(source: GPUFluidTracerSource | undefined): void {
    this.source = source;
    if (source && source.buffer !== this.boundBuffer) {
      this.boundDepthView = undefined;
      this.rebuildBindGroup(this.boundDepthView);
    }
  }

  private rebuildBindGroup(depthView: GPUTextureView | undefined): void {
    if (!this.layout || !this.uniforms || !this.fallbackMarkers) return;
    const view = depthView ?? this.fallbackDepthView;
    if (!view) return;
    const buffer = this.source?.buffer ?? this.fallbackMarkers;
    if (this.bindGroup && this.boundDepthView === view && this.boundBuffer === buffer) {
      return;
    }
    this.bindGroup = this.device.createBindGroup({
      label: "Fluid tracer overlay bind group",
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer } },
        { binding: 2, resource: view },
      ],
    });
    this.boundDepthView = view;
    this.boundBuffer = buffer;
  }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sceneDepth: GPUTextureView | undefined,
    frame: TracerOverlayFrame,
  ): boolean {
    const source = this.source;
    if (!this.ready || !source || source.count === 0) return false;
    this.rebuildBindGroup(sceneDepth);
    if (!this.bindGroup) return false;
    const { camera } = frame;
    const f = this.uniformF32, u = this.uniformU32;
    f.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    f.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    // Semi-transparent by default: the cloud is worth looking at because you
    // can see through its near face into the parcels behind it, and unordered
    // alpha over a dense lattice averages rather than showing whichever dot the
    // draw happened to reach last.
    f.set([...camera.right,
      Math.max(TRACER_MINIMUM_ALPHA, Math.min(1, frame.globalAlpha ?? 0.5))], 8);
    f.set([...camera.up, Math.max(0, Math.min(1, frame.occludedAlpha ?? 0.3))], 12);
    f.set([
      Math.max(1, frame.viewportWidth), Math.max(1, frame.viewportHeight),
      Math.max(1e-6, frame.depthNear_m), sceneDepth ? 1 : 0,
    ], 16);
    // A marker is drawn a little under a third of its seed spacing across, so
    // an undisturbed cloud reads as separated dots and a compressed one closes
    // into a solid colour — which is itself the reading: dots crowding together
    // is fluid piling up.
    const cellSize_m = Math.min(
      ...frame.container_m.map((metres, axis) =>
        metres / Math.max(1, source.domainFine[axis]!)));
    f.set([...frame.container_m, 0.3 * source.spacingFine * cellSize_m], 20);
    f.set([...source.domainFine, 0], 24);
    u.set([...source.latticeDimensions, source.floatOffset], 28);
    this.device.queue.writeBuffer(this.uniforms!, 0, this.uniformData);
    const pass = encoder.beginRenderPass({
      label: "Fluid tracer overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, source.count);
    pass.end();
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniforms?.destroy();
    this.fallbackDepth?.destroy();
    this.fallbackMarkers?.destroy();
    this.uniforms = undefined;
    this.fallbackDepth = undefined;
    this.fallbackMarkers = undefined;
    this.pipeline = undefined;
    this.bindGroup = undefined;
    this.source = undefined;
  }
}

export const tracerOverlayVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "fluid-tracers/seed-spectrum", pass: "Fluid tracers",
    label: "Marker spectrum",
    description: "Massless markers seeded through the liquid and coloured by where they started, then carried along the same characteristic the conservative transport integrates. Follows which water ends up where — mixing, overturning, and the parcels a surface render cannot tell apart. Enabling re-seeds from the current liquid, so the colours always date from the moment the view was switched on.",
    source: "Solver-resident marker positions, advected on the GPU without readback",
    mode: "tracers", axis: "volume", planeless: true, icon: "tracers",
    swatch: "#c86ed2",
    legend: [
      { swatch: "linear-gradient(90deg,#d43f3f,#d9d341,#48c46b)", mark: "point", label: "seeded left → right" },
      { swatch: "linear-gradient(90deg,#2f2f38,#8f8fa8,#f0f0f5)", mark: "point", label: "seeded low → high" },
      { swatch: "linear-gradient(90deg,#c25a2a,#5a7fd4)", mark: "point", label: "seeded near → far" },
    ],
  }),
]);
