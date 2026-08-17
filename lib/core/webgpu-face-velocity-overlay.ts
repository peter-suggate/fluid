import { fieldVisualization, type Visualization } from "./visualization-registry";

/**
 * Draws the solver's MAC face velocities as arrows over the finished image.
 *
 * A surface render shows where the water is and the marker cloud shows which
 * water it is; neither shows the field that moved it. This view draws the
 * velocity the solver actually solves for — one arrow per gradient row, which
 * is to say one arrow per cell face — pointing along that face's own axis, with
 * both its length and its colour set by the speed through it.
 *
 * **The arrows are the solver's faces, not a sampling of them.** There is no
 * seeding lattice and no interpolation: the instance index *is* a row id, its
 * centre and axis come from that row's arena record, and its speed comes from
 * the same face-velocity bank the projection just wrote. So the picture shows
 * the adaptive grid honestly — a coarse brick gets a few long arrows and a
 * refined one gets many short ones, because that is how many faces are there.
 *
 * **It costs no simulation time.** Unlike the marker cloud, nothing has to be
 * seeded, advected or stored for this view: every number it reads already
 * exists because the solve needs it. Switching it on adds a draw and nothing
 * else, which is why there is no enable flag on the solver and no stage on the
 * advance partition to price.
 *
 * Two things the view deliberately does not do:
 *
 * - **It does not reconstruct a cell-centred vector.** A MAC face carries one
 *   scalar, the component through its own normal. Collocating three of them
 *   into a arrow per cell would draw a direction the solver never computed at
 *   that point. Every arrow here points along ±x, ±y or ±z because that is the
 *   whole of what its face knows.
 * - **It does not auto-scale.** The reference speed is free fall over the
 *   container height, which is a property of the scene rather than of the
 *   frame, so an arrow means the same thing at t=0 and at t=10 s and two frames
 *   can be compared. A colour ramp normalised per frame would repaint the whole
 *   field every time the fastest arrow changed.
 */

/** Where a solver keeps its face velocities and the records that place them. */
export interface GPUFluidFaceVelocitySource {
  /** The solver's state buffer; face velocities are one float per row. */
  readonly state: GPUBuffer;
  /** The topology arena, holding the row records this addresses. */
  readonly topologyArena: GPUBuffer;
  /**
   * Float offset of the *accepted* face-velocity bank. The banks swap with
   * solver parity, so this is read fresh each frame rather than cached.
   */
  readonly faceFloatOffset: number;
  /** Optional GPU-owned accepted parity word in topologyArena. When present,
   * faceFloatOffset names bank A and bank B immediately follows rowCount. */
  readonly parityWordOffset?: number;
  readonly rowCount: number;
  /** Fine-lattice domain extent, for the lattice-to-metres mapping. */
  readonly domainFine: readonly [number, number, number];
  /** Finest cell width in metres — the unit face velocity is stored in. */
  readonly finestCell_m: number;
}

export const FACE_VELOCITY_OVERLAY_UNIFORM_BYTES = 144;

/**
 * Most arrows a single draw will issue.
 *
 * Row counts run to ~1.5 M on the shipped 128³ scenes, and a face plot that
 * dense is past the point where more arrows add information. Over the budget
 * the draw takes an even stride through the rows, which decimates the field
 * uniformly rather than dropping a contiguous region of it — and `arrowStride`
 * on the encode result reports what was applied, so a thinned plot is never
 * silently presented as a complete one.
 */
export const FACE_VELOCITY_ARROW_BUDGET = 1_048_576;

/** Vertices per arrow: a shaft quad and a head triangle. */
const FACE_VELOCITY_ARROW_VERTICES = 9;

/** Standard gravity, for the reference speed. */
const FACE_VELOCITY_GRAVITY_M_S2 = 9.80665;

/**
 * Full-scale speed, as a fall height in container heights.
 *
 * The *form* is physics: gravity-driven flow scales as the square root of the
 * length it falls through, so a reference of `sqrt(2gfH)` keeps one arrow
 * meaning the same thing in a bathtub and in a swimming pool. The fraction is
 * calibration, and it was measured rather than guessed — free fall over the
 * whole container (f = 1) puts a settled tank entirely in the first colour stop
 * and leaves the top two unused, because the projection removes almost all of
 * the free-fall acceleration in the step that adds it.
 *
 * A tenth of the height is where the measured face speeds of the shipped scenes
 * spread across the whole ramp: median around a quarter scale, ninety-ninth
 * percentile near full, and only a few percent of a violent front clipped.
 *
 * A per-frame auto-scale would fit every scene perfectly and be worth much
 * less: the ramp would repaint whenever the fastest arrow changed, and no two
 * frames of a run could be compared.
 */
const FACE_VELOCITY_REFERENCE_FALL = 0.1;

export interface FaceVelocityOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  /** tan(verticalFieldOfView / 2). */
  readonly tanHalfFov: number;
  readonly aspect: number;
}

export interface FaceVelocityOverlayFrame {
  readonly camera: FaceVelocityOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly container_m: readonly [number, number, number];
  /** Reversed-Z near plane the scene depth was written with. */
  readonly depthNear_m: number;
  readonly globalAlpha?: number;
  /** Alpha multiplier for arrows behind opaque scenery. */
  readonly occludedAlpha?: number;
  /**
   * Speed a full-length, full-heat arrow stands for. Defaults to a calibrated
   * fraction of free fall over the container height; see
   * `FACE_VELOCITY_REFERENCE_FALL` for why that fraction is not one.
   */
  readonly referenceSpeed_m_s?: number;
}

/**
 * The faintest an arrow field is ever drawn.
 *
 * The opacity comes from the field flyout's slice control, which a plane view
 * is free to park at zero; carrying that across would draw nothing, and a view
 * that draws nothing is indistinguishable from a broken one.
 */
const FACE_VELOCITY_MINIMUM_ALPHA = 0.08;

/**
 * Speed below which a face is not drawn, as a fraction of the reference.
 *
 * This does double duty. A row the current topology epoch does not accept is
 * written to exactly zero by the solver, so the same test that hides still
 * water also hides every inactive resolution level of every brick — which is
 * the majority of the row buffer, rejected by one load.
 */
const FACE_VELOCITY_MINIMUM_FRACTION = 0.004;

/**
 * Compression applied to the speed before it becomes a length and a colour.
 *
 * Linear length would leave almost every arrow in a settled body a sub-pixel
 * stub, because the interesting range of a liquid spans two decades and the
 * fast tail is a thin front. The same exponent drives both channels, so a
 * longer arrow is always a hotter arrow and the two never disagree.
 *
 * Measured against the shipped scenes at this reference, 0.45 is what spends
 * all five colour stops on a running dam break while leaving a settling tank in
 * the cool half — which is the reading, since a settling tank *is* slow.
 */
const FACE_VELOCITY_LENGTH_EXPONENT = 0.45;

/** Shaft half-width and head length, in pixels. */
const FACE_VELOCITY_SHAFT_HALF_PIXELS = 1.1;
const FACE_VELOCITY_HEAD_PIXELS = 5;

export const faceVelocityOverlayShader = /* wgsl */ `
struct FaceVelocityUniforms {
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
  // xyz container metres, w finest cell metres
  container:vec4f,
  // xyz fine-lattice domain extent, w reference speed in metres per second
  domain:vec4f,
  // shaft half pixels, head pixels, minimum speed fraction, length exponent
  style:vec4f,
  // face A float offset, row count, row stride, GPU parity word or INVALID
  addressing:vec4u,
}

@group(0) @binding(0) var<uniform> overlay:FaceVelocityUniforms;
@group(0) @binding(1) var<storage, read> state:array<f32>;
@group(0) @binding(2) var<storage, read> arena:array<u32>;
@group(0) @binding(3) var sceneDepth:texture_depth_2d;

struct VertexOut {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
  @location(1) alpha:f32,
  @location(2) viewDepth_m:f32,
}

const FACE_NEAR_VIEW_M:f32=0.02;
// Below this the arrow is aimed at the camera and has projected to a point.
const FACE_MINIMUM_SPAN_PIXELS:f32=1.5;

// Rows are nine field-wise planes based at arena word 7. The packed metadata
// plane contains axis in its high two bits; distance and centre occupy exact
// f32 planes 4 and 6..8. The compute module reads the same layout atomically;
// a plain read-only view is what lets a vertex stage index it at all.
fn rowWord(row:u32,plane:u32)->u32{return arena[7u]+plane*arena[3u]+row;}
fn rowFloat(index:u32)->f32{return bitcast<f32>(arena[index]);}

fn faceViewSpace(point:vec3f)->vec3f {
  let relative=point-overlay.cameraPosition.xyz;
  return vec3f(dot(relative,overlay.cameraRight.xyz),
    dot(relative,overlay.cameraUp.xyz),dot(relative,overlay.cameraForward.xyz));
}

fn faceNdc(view:vec3f)->vec2f {
  let tangent=max(overlay.cameraPosition.w,1e-4);
  let depth=max(view.z,FACE_NEAR_VIEW_M);
  return vec2f(view.x/(depth*tangent*max(overlay.cameraForward.w,1e-4)),
    view.y/(depth*tangent));
}

// A five-stop sequential ramp, evenly spaced so the legend can name the stops.
// Cool and receding for slow water, hot and advancing for fast, which is the
// convention every velocity figure in the literature already trains a reader on.
fn faceVelocityColor(t:f32)->vec3f {
  let x=clamp(t,0.0,1.0)*4.0;
  let stop=min(u32(x),3u);
  let f=x-f32(stop);
  var a=vec3f(0.15,0.31,0.74);
  var b=vec3f(0.16,0.72,0.84);
  if (stop==1u) { a=vec3f(0.16,0.72,0.84); b=vec3f(0.38,0.81,0.36); }
  else if (stop==2u) { a=vec3f(0.38,0.81,0.36); b=vec3f(0.96,0.83,0.24); }
  else if (stop==3u) { a=vec3f(0.96,0.83,0.24); b=vec3f(0.91,0.24,0.17); }
  return mix(a,b,f);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->VertexOut {
  var output:VertexOut;
  output.color=vec3f(0.0);
  output.alpha=0.0;
  output.viewDepth_m=0.0;
  // A collapsed triangle is how a rejected face costs nothing downstream.
  output.position=vec4f(0.0,0.0,0.0,1.0);

  let row=instance*max(overlay.addressing.z,1u);
  if (row>=overlay.addressing.y) { return output; }

  // The one load that rejects most of the buffer, taken before the arena is
  // touched at all: an unaccepted row and a still face are both exactly zero.
  var parity=0u;
  if(overlay.addressing.w!=0xffffffffu){parity=arena[overlay.addressing.w]&1u;}
  let faceOffset=overlay.addressing.x+parity*overlay.addressing.y;
  let speed_m_s=state[faceOffset+row]*overlay.container.w;
  let normalized=abs(speed_m_s)/max(overlay.domain.w,1e-6);
  if (normalized<overlay.style.z) { return output; }

  let axis=arena[rowWord(row,1u)]>>30u;
  let centerFine=vec3f(rowFloat(rowWord(row,6u)),rowFloat(rowWord(row,7u)),
    rowFloat(rowWord(row,8u)));
  // The centre-to-centre distance is the local cell scale, in fine cells. Using
  // it rather than a constant is what makes one arrow never overrun its own
  // cell on either side of a 2:1 seam.
  let spanFine=max(rowFloat(rowWord(row,4u)),1e-4);

  // Fine-lattice coordinates are corner-origin and continuous: 0 is the domain
  // minimum and the domain extent is its maximum, on every axis.
  let half=0.5*overlay.container.xyz;
  let world=centerFine/max(overlay.domain.xyz,vec3f(1.0))*overlay.container.xyz
    +vec3f(-half.x,0.0,-half.z);

  var unit=vec3f(0.0,0.0,1.0);
  if (axis==0u) { unit=vec3f(1.0,0.0,0.0); }
  else if (axis==1u) { unit=vec3f(0.0,1.0,0.0); }
  let direction=unit*sign(speed_m_s);

  let heat=pow(min(normalized,1.0),max(overlay.style.w,1e-3));
  // Centred on the face rather than trailing from it: the arrow stays inside
  // its own cell, so a dense field reads as a field instead of as streaks
  // reaching across their neighbours.
  let length_m=heat*spanFine*overlay.container.w;
  let tailView=faceViewSpace(world-0.5*length_m*direction);
  let headView=faceViewSpace(world+0.5*length_m*direction);
  if (tailView.z<FACE_NEAR_VIEW_M||headView.z<FACE_NEAR_VIEW_M) { return output; }

  let viewport=max(overlay.viewport.xy,vec2f(1.0));
  let tailPixel=faceNdc(tailView)*0.5*viewport;
  let headPixel=faceNdc(headView)*0.5*viewport;
  let delta=headPixel-tailPixel;
  let span=length(delta);
  // An arrow aimed at the camera has projected to a point. There is no
  // direction left to draw, and inventing one would assert a component this
  // face does not carry, so the honest degenerate case is to draw nothing.
  if (span<FACE_MINIMUM_SPAN_PIXELS) { return output; }
  let forward=delta/span;
  let side=vec2f(-forward.y,forward.x);

  let headLength=min(overlay.style.y,0.55*span);
  let shaftHalf=min(overlay.style.x,0.4*headLength);
  let neck=headPixel-forward*headLength;

  var pixel=headPixel;
  if (vertexIndex<6u) {
    var quad=array<vec2f,6>(vec2f(0.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),
      vec2f(0.0,-1.0),vec2f(1.0,1.0),vec2f(0.0,1.0));
    let corner=quad[vertexIndex];
    pixel=mix(tailPixel,neck,corner.x)+side*shaftHalf*corner.y;
  } else if (vertexIndex==7u) { pixel=neck+side*2.6*shaftHalf; }
  else if (vertexIndex==8u) { pixel=neck-side*2.6*shaftHalf; }

  output.position=vec4f(pixel/(0.5*viewport),0.0,1.0);
  output.color=faceVelocityColor(heat);
  output.alpha=clamp(overlay.cameraRight.w,0.0,1.0);
  output.viewDepth_m=0.5*(tailView.z+headView.z);
  return output;
}

@fragment fn fragmentMain(input:VertexOut)->@location(0) vec4f {
  if (input.alpha<=0.0) { discard; }
  var alpha=input.alpha;
  if (overlay.viewport.w>0.5) {
    let coordinate=vec2u(input.position.xy);
    let stored=textureLoad(sceneDepth,coordinate,0);
    // Reversed-Z: depth = near / viewDepth, and zero means nothing was drawn.
    // The depth is the dry scene's, so this dims arrows behind walls and
    // terrain while leaving submerged ones at full strength — the interior
    // field being the whole reason to look.
    if (stored>0.0) {
      let sceneDepth_m=overlay.viewport.z/max(stored,1e-7);
      if (input.viewDepth_m>sceneDepth_m) { alpha*=max(overlay.cameraUp.w,0.0); }
    }
  }
  return vec4f(input.color,alpha);
}
`;

/** One instanced draw over the solver's resident gradient rows. */
export class FaceVelocityOverlay {
  private pipeline?: GPURenderPipeline;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private uniforms?: GPUBuffer;
  private fallbackDepth?: GPUTexture;
  private fallbackDepthView?: GPUTextureView;
  private fallbackStorage?: GPUBuffer;
  private boundDepthView?: GPUTextureView;
  private boundState?: GPUBuffer;
  private boundArena?: GPUBuffer;
  private source?: GPUFluidFaceVelocitySource;
  private destroyed = false;
  private readonly uniformData =
    new ArrayBuffer(FACE_VELOCITY_OVERLAY_UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
  ) {}

  async initialize(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Fluid face velocity overlay", code: faceVelocityOverlayShader,
    });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`Fluid face velocity overlay:\n${errors
        .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`)
        .join("\n")}`);
    }
    this.uniforms = this.device.createBuffer({
      label: "Fluid face velocity overlay uniforms",
      size: FACE_VELOCITY_OVERLAY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.fallbackDepth = this.device.createTexture({
      label: "Fluid face velocity overlay depth fallback",
      size: [1, 1], format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fallbackDepthView = this.fallbackDepth.createView();
    // A bind group needs buffers even in the frame before a solver publishes
    // any; the draw is skipped by row count, not by a missing binding.
    this.fallbackStorage = this.device.createBuffer({
      label: "Fluid face velocity overlay fallback",
      size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.layout = this.device.createBindGroupLayout({
      label: "Fluid face velocity overlay bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth" } },
      ],
    });
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Fluid face velocity overlay",
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
      // No culling: the head triangle's winding flips with the arrow's
      // projected direction, and a back-facing arrowhead is still an arrowhead.
      primitive: { topology: "triangle-list" },
    });
    this.rebuildBindGroup(this.fallbackDepthView);
  }

  get ready(): boolean {
    return Boolean(this.pipeline && this.uniforms && this.layout);
  }

  /** Idempotent: re-binding the same buffers must not rebuild a bind group. */
  setSource(source: GPUFluidFaceVelocitySource | undefined): void {
    this.source = source;
    if (source && (source.state !== this.boundState
      || source.topologyArena !== this.boundArena)) {
      this.rebuildBindGroup(this.boundDepthView);
    }
  }

  private rebuildBindGroup(depthView: GPUTextureView | undefined): void {
    if (!this.layout || !this.uniforms || !this.fallbackStorage) return;
    const view = depthView ?? this.fallbackDepthView;
    if (!view) return;
    const state = this.source?.state ?? this.fallbackStorage;
    const arena = this.source?.topologyArena ?? this.fallbackStorage;
    if (this.bindGroup && this.boundDepthView === view
      && this.boundState === state && this.boundArena === arena) {
      return;
    }
    this.bindGroup = this.device.createBindGroup({
      label: "Fluid face velocity overlay bind group",
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: state } },
        { binding: 2, resource: { buffer: arena } },
        { binding: 3, resource: view },
      ],
    });
    this.boundDepthView = view;
    this.boundState = state;
    this.boundArena = arena;
  }

  /**
   * The stride a given row count draws at, and the arrows that survive it.
   *
   * Exposed so a caller can say what it drew rather than implying it drew
   * everything; `encode` applies exactly this.
   */
  static plan(rowCount: number, budget: number = FACE_VELOCITY_ARROW_BUDGET): {
    readonly stride: number; readonly arrows: number;
  } {
    if (!(rowCount > 0) || !(budget >= 1)) return { stride: 1, arrows: 0 };
    const stride = Math.max(1, Math.ceil(rowCount / budget));
    return { stride, arrows: Math.ceil(rowCount / stride) };
  }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sceneDepth: GPUTextureView | undefined,
    frame: FaceVelocityOverlayFrame,
  ): boolean {
    const source = this.source;
    if (!this.ready || !source || source.rowCount === 0) return false;
    this.rebuildBindGroup(sceneDepth);
    if (!this.bindGroup) return false;
    const { stride, arrows } = FaceVelocityOverlay.plan(source.rowCount);
    if (arrows === 0) return false;
    const { camera } = frame;
    const f = this.uniformF32, u = this.uniformU32;
    f.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    f.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    f.set([...camera.right,
      Math.max(FACE_VELOCITY_MINIMUM_ALPHA, Math.min(1, frame.globalAlpha ?? 0.85))], 8);
    f.set([...camera.up, Math.max(0, Math.min(1, frame.occludedAlpha ?? 0.28))], 12);
    f.set([
      Math.max(1, frame.viewportWidth), Math.max(1, frame.viewportHeight),
      Math.max(1e-6, frame.depthNear_m), sceneDepth ? 1 : 0,
    ], 16);
    f.set([...frame.container_m, Math.max(1e-9, source.finestCell_m)], 20);
    // A scene property, not a frame one, so an arrow means the same thing in
    // every frame of a run and in two runs of the same scene.
    const reference_m_s = frame.referenceSpeed_m_s ?? Math.sqrt(
      2 * FACE_VELOCITY_GRAVITY_M_S2 * FACE_VELOCITY_REFERENCE_FALL
      * Math.max(1e-3, frame.container_m[1]!));
    f.set([...source.domainFine, Math.max(1e-6, reference_m_s)], 24);
    f.set([FACE_VELOCITY_SHAFT_HALF_PIXELS, FACE_VELOCITY_HEAD_PIXELS,
      FACE_VELOCITY_MINIMUM_FRACTION, FACE_VELOCITY_LENGTH_EXPONENT], 28);
    u.set([source.faceFloatOffset, source.rowCount, stride,
      source.parityWordOffset ?? 0xffff_ffff], 32);
    this.device.queue.writeBuffer(this.uniforms!, 0, this.uniformData);
    const pass = encoder.beginRenderPass({
      label: "Fluid face velocity overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(FACE_VELOCITY_ARROW_VERTICES, arrows);
    pass.end();
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniforms?.destroy();
    this.fallbackDepth?.destroy();
    this.fallbackStorage?.destroy();
    this.uniforms = undefined;
    this.fallbackDepth = undefined;
    this.fallbackStorage = undefined;
    this.pipeline = undefined;
    this.bindGroup = undefined;
    this.source = undefined;
  }
}

export const faceVelocityOverlayVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "fluid-faces/velocity-arrows", pass: "Fluid faces",
    label: "Face velocity",
    description: "One arrow per cell face, drawn along that face's own axis with its length and colour set by the speed through it. This is the MAC field the solver solves for, not a reconstruction of it: a coarse brick shows a few long arrows and a refined one many short ones, because that is how many faces exist there. Scaled to a fixed speed derived from the container rather than to the current frame, so an arrow means the same thing at the first step and the thousandth.",
    source: "Solver-resident face velocities and gradient-row records, read at draw time",
    mode: "face-velocity", axis: "volume", planeless: true,
    swatch: "#4ec9f5",
    legend: [
      { swatch: "linear-gradient(90deg,#2650bd,#29b8d7,#61cf5c,#f5d43d,#e83e2b)", mark: "line", label: "slow → full scale" },
      { swatch: "#61cf5c", mark: "line", label: "length spans one cell at full scale" },
    ],
  }),
]);
