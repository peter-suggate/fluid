import { DECORATION_SEGMENT_FLOATS } from "./visualization-decorations";

/**
 * Draws one frame's assembled decorations as 3D line work over the finished image.
 *
 * Every decoration any pass contributes is a world-space segment (see
 * `visualization-decorations.ts`), so a single instanced draw covers the whole
 * overlay however many passes fed it: each instance expands to a
 * screen-space-thickened quad whose width is constant in device pixels, which
 * is what keeps a 5 cm brick edge legible from across the scene without turning
 * a near box into a slab. An instance carrying a non-zero head width becomes an
 * arrowhead at its end point instead of a shaft, so directional work says which
 * way it went.
 *
 * The camera basis is the same one every dry-scene entry point builds, so the
 * projection here is the exact forward transform of the ray a probe traced and
 * of the `worldToFine` a solver shader inverted: what is drawn lands on the
 * pixels whose work it describes.
 *
 * Occluded line work is not hidden, it is ghosted. A wireframe that vanished
 * behind the surface it explains would hide the traversal, and the whole point
 * is to see the boxes the ray walked through — including the ones inside the
 * thing it eventually hit.
 */
export const DECORATION_OVERLAY_UNIFORM_BYTES = 96;

/**
 * The whole contract a built geometry has to satisfy to be drawn here.
 *
 * Nothing in this pipeline is specific to any one producer: a segment is two
 * world points with a colour, a width and a dash. `assembleDecorations` merges
 * every enabled contributor into one of these, and a producer that already
 * emits the instance layout joins by concatenation rather than by rewrite.
 */
export interface DecorationGeometry {
  /** Interleaved per-instance data; see DECORATION_SEGMENT_FLOATS. */
  readonly segments: Float32Array<ArrayBuffer>;
  readonly segmentCount: number;
}

export const DECORATION_OVERLAY_INSTANCE_BYTES = DECORATION_SEGMENT_FLOATS * 4;
/** Segments beyond this are dropped rather than silently growing GPU memory. */
export const DECORATION_OVERLAY_MAXIMUM_SEGMENTS = 65_536;

export interface DecorationOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  /** tan(verticalFieldOfView / 2); CAMERA_TAN_HALF_FOV for every path here. */
  readonly tanHalfFov: number;
  readonly aspect: number;
}

export interface DecorationOverlayFrame {
  readonly camera: DecorationOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  /** 0 replays the trace from the camera outward; 1 shows all of it. */
  readonly reveal: number;
  /** Width of the reveal's bright leading edge, as a fraction of the trace. */
  readonly revealSoftness?: number;
  readonly globalAlpha?: number;
  /** Alpha multiplier for line work behind the visible surface. */
  readonly occludedAlpha?: number;
  /** Reversed-Z near plane the scene depth was written with. */
  readonly depthNear_m: number;
}

export const decorationOverlayShader = /* wgsl */ `
struct TraceOverlayUniforms {
  // xyz camera position, w tan(halfFov)
  cameraPosition:vec4f,
  // xyz camera forward, w aspect
  cameraForward:vec4f,
  // xyz camera right, w reveal
  cameraRight:vec4f,
  // xyz camera up, w global alpha
  cameraUp:vec4f,
  // xy viewport pixels, z reversed-Z near metres, w occluded alpha
  viewport:vec4f,
  // x depth valid, y reveal softness, z reserved, w reserved
  controls:vec4f,
}

@group(0) @binding(0) var<uniform> overlay:TraceOverlayUniforms;
@group(0) @binding(1) var sceneDepth:texture_depth_2d;

struct VertexOut {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
  @location(1) alpha:f32,
  @location(2) alongSegment_m:f32,
  @location(3) dashPeriod_m:f32,
  @location(4) viewDepth_m:f32,
  @location(5) edgeCoordinate:f32,
}

const TRACE_NEAR_VIEW_M:f32=0.02;

fn traceViewSpace(point:vec3f)->vec3f {
  let relative=point-overlay.cameraPosition.xyz;
  return vec3f(dot(relative,overlay.cameraRight.xyz),dot(relative,overlay.cameraUp.xyz),dot(relative,overlay.cameraForward.xyz));
}

fn traceNdc(view:vec3f)->vec2f {
  let tangent=max(overlay.cameraPosition.w,1e-4);
  let depth=max(view.z,TRACE_NEAR_VIEW_M);
  return vec2f(view.x/(depth*tangent*max(overlay.cameraForward.w,1e-4)),view.y/(depth*tangent));
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @location(0) startWidth:vec4f,
  @location(1) endOrder:vec4f,
  @location(2) colorIntensity:vec4f,
  @location(3) kindDashHead:vec4f,
)->VertexOut {
  // Two triangles: x runs along the segment, y across it.
  var corners=array<vec2f,6>(vec2f(0.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(0.0,-1.0),vec2f(1.0,1.0),vec2f(0.0,1.0));
  let corner=corners[vertexIndex];
  var startView=traceViewSpace(startWidth.xyz);
  var endView=traceViewSpace(endOrder.xyz);
  var output:VertexOut;
  output.color=colorIntensity.xyz;
  output.alpha=0.0;
  output.alongSegment_m=0.0;
  output.dashPeriod_m=kindDashHead.y;
  output.viewDepth_m=0.0;
  output.edgeCoordinate=corner.y;
  // A segment entirely behind the camera collapses; one that straddles the near
  // plane is clipped to it so the visible part still draws.
  if (startView.z<TRACE_NEAR_VIEW_M&&endView.z<TRACE_NEAR_VIEW_M) {
    output.position=vec4f(0.0,0.0,0.0,1.0);
    return output;
  }
  let fullLength_m=length(endView-startView);
  if (startView.z<TRACE_NEAR_VIEW_M) {
    startView=mix(startView,endView,(TRACE_NEAR_VIEW_M-startView.z)/max(endView.z-startView.z,1e-6));
  } else if (endView.z<TRACE_NEAR_VIEW_M) {
    endView=mix(endView,startView,(TRACE_NEAR_VIEW_M-endView.z)/max(startView.z-endView.z,1e-6));
  }
  let startNdc=traceNdc(startView);
  let endNdc=traceNdc(endView);
  let viewport=max(overlay.viewport.xy,vec2f(1.0));
  let startPixel=startNdc*.5*viewport;
  let endPixel=endNdc*.5*viewport;
  var alongPixel=endPixel-startPixel;
  let alongLength=length(alongPixel);
  // A degenerate projection still needs a stable perpendicular so a segment
  // seen end-on renders as a dot rather than disappearing.
  alongPixel=select(vec2f(1.0,0.0),alongPixel/alongLength,alongLength>1e-5);
  let acrossPixel=vec2f(-alongPixel.y,alongPixel.x);
  let halfWidth=max(startWidth.w,0.35)*.5;
  // An arrowhead instance is built in screen space, not in the world: the tip
  // sits on the segment's end and the base is a fixed number of device pixels
  // back along the projected direction. A world-space head would foreshorten to
  // nothing exactly when the ray points at the camera, which is when knowing
  // which way the ray goes matters most.
  let head=clamp(kindDashHead.z,0.0,64.0);
  var pixel=vec2f(0.0);
  var alongFraction=corner.x;
  if (head>0.0) {
    // corners.x picks tip over base, so the quad's first triangle collapses and
    // exactly one triangle is rasterized. Two would double the additive alpha.
    let base=endPixel-alongPixel*head*1.8;
    pixel=select(base+acrossPixel*corner.y*head*.9,endPixel,corner.x>.5);
    alongFraction=1.0;
    // A head is a solid triangle: no dash to cut it up, no edge fade to hollow
    // it out. Both are shaft treatments and neither has a meaning here.
    output.dashPeriod_m=0.0;
    output.edgeCoordinate=0.0;
  } else {
    pixel=mix(startPixel,endPixel,corner.x)+acrossPixel*corner.y*halfWidth;
  }
  let ndc=pixel/(.5*viewport);
  output.position=vec4f(ndc,0.0,1.0);
  output.alongSegment_m=fullLength_m*alongFraction;
  output.viewDepth_m=mix(startView.z,endView.z,alongFraction);

  // Reveal: the trace draws itself in the order the shader did the work. The
  // leading edge is brightened so the eye can follow the sweep.
  let reveal=overlay.cameraRight.w;
  let softness=max(overlay.controls.y,1e-4);
  let order=endOrder.w;
  var alpha=colorIntensity.w*max(overlay.cameraUp.w,0.0);
  if (order>reveal) { alpha=0.0; }
  let lead=1.0-clamp((reveal-order)/softness,0.0,1.0);
  output.color=colorIntensity.xyz*(1.0+lead*2.2);
  output.alpha=alpha;
  return output;
}

@fragment fn fragmentMain(input:VertexOut)->@location(0) vec4f {
  if (input.alpha<=0.0) { discard; }
  var alpha=input.alpha;
  // Dashes mark work that was considered and put aside: deferred stack entries,
  // rejected boxes, unoccluded shadow rays.
  if (input.dashPeriod_m>0.0) {
    let phase=fract(input.alongSegment_m/input.dashPeriod_m);
    if (phase>.55) { discard; }
  }
  // Soft edge across the line, so thin work does not alias into dotted noise.
  alpha*=1.0-.85*pow(abs(input.edgeCoordinate),3.0);
  if (overlay.controls.x>.5) {
    let coordinate=vec2u(input.position.xy);
    let stored=textureLoad(sceneDepth,coordinate,0);
    // Reversed-Z: depth = near / viewDepth, and zero means nothing was drawn.
    if (stored>0.0) {
      let sceneDepth_m=overlay.viewport.z/max(stored,1e-7);
      if (input.viewDepth_m>sceneDepth_m) { alpha*=max(overlay.viewport.w,0.0); }
    }
  }
  return vec4f(input.color*alpha,alpha);
}
`;

/** Instanced segment overlay for one decoded pixel trace. */
export class DecorationOverlay {
  private pipeline?: GPURenderPipeline;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private instances?: GPUBuffer;
  private instanceCapacity = 0;
  private uniforms?: GPUBuffer;
  private fallbackDepth?: GPUTexture;
  private fallbackDepthView?: GPUTextureView;
  private boundDepthView?: GPUTextureView;
  private segmentCount = 0;
  private readonly uniformData = new Float32Array(new ArrayBuffer(DECORATION_OVERLAY_UNIFORM_BYTES));
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly targetFormat: GPUTextureFormat) {}

  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({ label: "Decoration overlay", code: decorationOverlayShader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`Decoration overlay:\n${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
    }
    this.uniforms = this.device.createBuffer({
      label: "Decoration overlay uniforms",
      size: DECORATION_OVERLAY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.fallbackDepth = this.device.createTexture({
      label: "Decoration overlay depth fallback",
      size: [1, 1],
      format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fallbackDepthView = this.fallbackDepth.createView();
    this.layout = this.device.createBindGroupLayout({
      label: "Decoration overlay bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
      ],
    });
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Decoration overlay",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: {
        module,
        entryPoint: "vertexMain",
        buffers: [{
          arrayStride: DECORATION_OVERLAY_INSTANCE_BYTES,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x4" },
            { shaderLocation: 1, offset: 16, format: "float32x4" },
            { shaderLocation: 2, offset: 32, format: "float32x4" },
            { shaderLocation: 3, offset: 48, format: "float32x4" },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: "fragmentMain",
        targets: [{
          format: this.targetFormat,
          // Alpha-weighted additive: line work glows over the frame instead of
          // punching a flat hole in it, and crossings read as brighter.
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one" },
            alpha: { srcFactor: "one", dstFactor: "one" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.rebuildBindGroup(this.fallbackDepthView);
  }

  get ready(): boolean { return Boolean(this.pipeline && this.bindGroup && this.uniforms); }

  private rebuildBindGroup(depthView: GPUTextureView | undefined): void {
    if (!this.layout || !this.uniforms) return;
    const view = depthView ?? this.fallbackDepthView;
    if (!view) return;
    if (this.bindGroup && this.boundDepthView === view) return;
    this.bindGroup = this.device.createBindGroup({
      label: "Decoration overlay bind group",
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: view },
      ],
    });
    this.boundDepthView = view;
  }

  /**
   * Upload one built geometry. Returns the number of segments that will draw;
   * an over-capacity trace is truncated and reported rather than dropped whole.
   */
  setGeometry(geometry: DecorationGeometry): number {
    if (this.destroyed) return 0;
    const requested = Math.min(geometry.segmentCount, DECORATION_OVERLAY_MAXIMUM_SEGMENTS);
    this.segmentCount = requested;
    if (requested === 0) return 0;
    if (!this.instances || this.instanceCapacity < requested) {
      this.instances?.destroy();
      // Grow in powers of two so a dense hover does not reallocate per frame.
      this.instanceCapacity = Math.min(DECORATION_OVERLAY_MAXIMUM_SEGMENTS, 2 ** Math.ceil(Math.log2(Math.max(256, requested))));
      this.instances = this.device.createBuffer({
        label: "Decoration overlay segments",
        size: this.instanceCapacity * DECORATION_OVERLAY_INSTANCE_BYTES,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    const floats = requested * DECORATION_SEGMENT_FLOATS;
    this.device.queue.writeBuffer(this.instances, 0, geometry.segments, 0, floats);
    return requested;
  }

  clear(): void { this.segmentCount = 0; }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sceneDepth: GPUTextureView | undefined,
    frame: DecorationOverlayFrame,
  ): boolean {
    if (!this.ready || !this.instances || this.segmentCount === 0) return false;
    this.rebuildBindGroup(sceneDepth);
    if (!this.bindGroup) return false;
    const { camera } = frame;
    const data = this.uniformData;
    data.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    data.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    data.set([...camera.right, Math.max(0, Math.min(1, frame.reveal))], 8);
    data.set([...camera.up, Math.max(0, Math.min(1, frame.globalAlpha ?? 1))], 12);
    data.set([
      Math.max(1, frame.viewportWidth), Math.max(1, frame.viewportHeight),
      Math.max(1e-6, frame.depthNear_m), Math.max(0, Math.min(1, frame.occludedAlpha ?? 0.22)),
    ], 16);
    data.set([sceneDepth ? 1 : 0, Math.max(1e-3, frame.revealSoftness ?? 0.08), 0, 0], 20);
    this.device.queue.writeBuffer(this.uniforms!, 0, data);
    const pass = encoder.beginRenderPass({
      label: "Decoration overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.instances);
    pass.draw(6, this.segmentCount);
    pass.end();
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.instances?.destroy();
    this.uniforms?.destroy();
    this.fallbackDepth?.destroy();
    this.instances = undefined;
    this.uniforms = undefined;
    this.fallbackDepth = undefined;
    this.bindGroup = undefined;
    this.segmentCount = 0;
  }
}
