import {
  SPARSE_CM12_PRESSURE_JOURNAL_FIELDS,
  SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS,
  SPARSE_CM12_PRESSURE_JOURNAL_RECORD,
  sparseCM12PressureJournalSnapshotOffset,
  type SparseCM12PressureJournalField,
  type SparseCM12PressureJournalLayout,
} from "../methods/adaptive-mass/sparse-cm12-pressure-journal";
import { fieldVisualization, type Visualization } from "./visualization-registry";

/**
 * Draws one captured iteration of the Sparse CM12 pressure solve.
 *
 * The solve's own cells, at their own sizes. The instance index *is* a cell id,
 * its centre and width come from that cell's arena record, and its value comes
 * from the journal snapshot the scrubber is parked on. So a coarse brick draws
 * a few large squares and a refined one draws many small ones, and a 2:1 seam
 * is visible as exactly what it is — a size change in the unknowns — rather
 * than as a resampling artifact of a dense lattice the solver never had.
 *
 * **Nothing is read back to draw this.** The snapshots are the large part of a
 * capture, four fields over every accepted cell at every scheduled iteration,
 * and they stay in the solver's state buffer where the solve wrote them.
 * Scrubbing to another iteration writes a uniform. That is what makes the film
 * scrub at frame rate instead of stalling on a map for every step.
 *
 * Three choices that decide whether the picture teaches anything:
 *
 * - **It does not auto-scale.** The reference magnitude is fixed for the whole
 *   film, so watching the residual fall is watching the colour drain out of the
 *   domain. A per-iteration normalisation would repaint every frame to full
 *   contrast and hide the single thing the film exists to show.
 * - **Signed fields get a diverging ramp.** A residual is not a magnitude; a
 *   cell that is over-pressured and one that is under-pressured are opposite
 *   errors, and a sequential ramp would print them the same colour.
 * - **The residual is drawn on a log ramp.** It spans decades, and a linear
 *   ramp shows the first iteration and then black. The decade count is on the
 *   legend, because a log ramp without a stated base is a lie about contrast.
 */

/** Where a captured pressure solve keeps its journal and the records to place it. */
export interface GPUPressureJournalSource {
  /** The solver's state buffer; the journal is a tail range of it. */
  readonly state: GPUBuffer;
  /** The topology arena, holding the cell records this addresses. */
  readonly topologyArena: GPUBuffer;
  /** Float offset of the journal region within `state`. */
  readonly journalFloatOffset: number;
  readonly layout: SparseCM12PressureJournalLayout;
  /**
   * Snapshot slots the last armed encode actually filled.
   *
   * The reserved capacity is only an upper bound: the schedule is log-spaced
   * over the iteration ceiling, so a short solve leaves slots untouched and a
   * scrub that ran to capacity would replay a previous capture's frames as
   * though they belonged to this one. Zero means nothing has been captured.
   */
  readonly snapshotCount: number;
  /**
   * Float offset of the liquid mask.
   *
   * The one load that rejects most of the buffer before the arena is touched:
   * a cell with no pressure row in the captured frame is exactly zero there,
   * which is every dry cell and every inactive resolution level of every brick.
   */
  readonly liquidFloatOffset: number;
  readonly cellCount: number;
  /** Fine-lattice domain extent, for the lattice-to-metres mapping. */
  readonly domainFine: readonly [number, number, number];
  /** Finest cell width in metres. */
  readonly finestCell_m: number;
}

export const PRESSURE_JOURNAL_OVERLAY_UNIFORM_BYTES = 176;

/**
 * Most cells a single draw will issue.
 *
 * Past this the squares are sub-pixel and more of them add nothing but
 * overdraw. Over budget the draw takes an even stride through the cells, which
 * thins the field uniformly rather than dropping a contiguous region of it, and
 * `encode` reports the stride so a thinned plot is never presented as complete.
 */
export const PRESSURE_JOURNAL_CELL_BUDGET = 1_048_576;

/** Six vertices: one camera-facing quad per cell. */
const PRESSURE_JOURNAL_CELL_VERTICES = 6;

/** The faintest the field is ever drawn, for the same reason the arrows have one. */
const PRESSURE_JOURNAL_MINIMUM_ALPHA = 0.1;

/**
 * Value below which a cell is not drawn, as a fraction of the reference.
 *
 * Converged cells are the overwhelming majority by the end of a solve, and
 * drawing them as a wash of neutral grey hides the handful that still carry
 * residual — which is the whole question the last iterations raise.
 */
const PRESSURE_JOURNAL_MINIMUM_FRACTION = 0.002;

/**
 * Decades a log-ramped channel spans below its reference.
 *
 * Six, and it is a measurement rather than a preference. On the mini 32³ dam
 * break the residual's cell magnitudes run from 45 at the seed down to 3e-9 by
 * iteration 64 — ten decades across one film, which no colour ramp shows
 * usefully. Six is where the *informative* span lands: iterations 0 through 16,
 * over which the relative residual falls from 1 to 0.01, occupy four decades
 * below the seed maximum, and iteration 32 sits at the floor. So the ramp
 * covers the part of the solve that is still working and lets the converged
 * tail drain off the bottom, which is the correct picture rather than a lost
 * one — a scale that kept the tail visible would have to compress the early
 * iterations into nothing.
 *
 * Measured by `tools/probe-sparse-cm12-pressure-journal.ts`.
 */
export const PRESSURE_JOURNAL_DECADES = 6;

/**
 * Decades the pressure channel spans, which is far fewer.
 *
 * Pressure does not drain: it converges to a fixed field and stays there, and
 * on the same run its live cells span 1e-4 to 15.9 with a median of 0.64. Three
 * decades below the reference puts that median at mid-ramp and keeps the tail
 * of large values distinguishable, where the residual's six would flatten the
 * entire field into the top fifth of the scale.
 */
export const PRESSURE_JOURNAL_PRESSURE_DECADES = 3;

/** How a channel's value becomes a colour. */
export type PressureJournalRamp = "diverging-log" | "diverging-linear" | "sequential-log";

export interface PressureJournalChannel {
  readonly field: SparseCM12PressureJournalField;
  readonly ramp: PressureJournalRamp;
  /** Value that saturates the ramp. Fixed across the film; never per-frame. */
  readonly reference: number;
  /** Decades the ramp spans below the reference; ignored by a linear ramp. */
  readonly decades: number;
  /**
   * Subtract a depth-linear hydrostatic field before ramping.
   *
   * Off by default, and the default is measured rather than assumed. The
   * expectation was that raw pressure would be a vertical gradient swamping
   * everything dynamic in it — but on a real capture the solve's pressure is
   * non-negative with a median of 0.64 against a maximum of 15.9, which is a
   * bulk near zero with a heavy tail, not a ramp. Subtracting a gradient that
   * is not there would manufacture a sign the field does not have. The control
   * stays because a settled deep scene may yet look the way the expectation
   * says; it is no longer switched on for every scene on the strength of it.
   */
  readonly subtractHydrostatic?: boolean;
}

export interface PressureJournalOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  readonly tanHalfFov: number;
  readonly aspect: number;
}

/**
 * A slab of the domain to restrict the draw to.
 *
 * A pressure field is an interior quantity, and a solid cloud of billboards
 * shows only its outer shell. Cutting to a slab is how the inside becomes
 * visible; it is a slab rather than a plane so that coarse cells, which a
 * zero-thickness plane would miss entirely unless it happened to cross their
 * centre, are still drawn beside the fine ones they couple to.
 */
export interface PressureJournalSlab {
  readonly axis: 0 | 1 | 2;
  /** Slab centre along the axis, in fine-lattice units. */
  readonly centerFine: number;
  /** Half-thickness in fine-lattice units. Zero draws the whole volume. */
  readonly halfThicknessFine: number;
}

export interface PressureJournalOverlayFrame {
  readonly camera: PressureJournalOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly container_m: readonly [number, number, number];
  /** Reversed-Z near plane the scene depth was written with. */
  readonly depthNear_m: number;
  /** Which captured snapshot to draw: the scrubber's position. */
  readonly snapshot: number;
  readonly channel: PressureJournalChannel;
  readonly slab?: PressureJournalSlab;
  readonly globalAlpha?: number;
  readonly occludedAlpha?: number;
  /** Draw only cells at this brick resolution, to isolate one level. */
  readonly onlyResolution?: number;
}

export const pressureJournalOverlayShader = /* wgsl */ `
struct JournalUniforms {
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
  // xyz fine-lattice domain extent, w reference magnitude
  domain:vec4f,
  // decades, minimum fraction, ramp code, hydrostatic subtraction scale
  style:vec4f,
  // slab axis, slab enabled, slab centre fine, slab half thickness fine
  slab:vec4f,
  // snapshot field float offset, cell count, cell stride, liquid float offset
  addressing:vec4u,
  // resolution filter (0 = all), unused
  filters:vec4u,
}

@group(0) @binding(0) var<uniform> overlay:JournalUniforms;
@group(0) @binding(1) var<storage, read> state:array<f32>;
@group(0) @binding(2) var<storage, read> arena:array<u32>;
@group(0) @binding(3) var sceneDepth:texture_depth_2d;

struct VertexOut {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
  @location(1) alpha:f32,
  @location(2) viewDepth_m:f32,
}

const JOURNAL_NEAR_VIEW_M:f32=0.02;
// Below this the cell has projected to nothing worth a triangle.
const JOURNAL_MINIMUM_SPAN_PIXELS:f32=0.75;

// Cell records are eight words each, based at arena word 6. Words 0..2 are the
// centre in fine cells, 4..6 the widths, and word 7 packs brick/resolution —
// which is the coarse/fine level, and the only reason this view can draw the
// adaptive grid honestly instead of assuming one cell size.
fn cellBase(id:u32)->u32{return arena[6u]+id*8u;}

// The scale is the capture's own, read from the film rather than handed in.
//
// It has to be the same number for every iteration — that is the whole reason
// the view is worth looking at, since a per-frame renormalisation would repaint
// the domain each time the largest remaining residual moved — and the host
// cannot supply it without mapping a buffer every frame to ask what the solve
// started at. So the seed record's own initial residual max-norm is read here,
// out of the same buffer the values come from. A capture that never ran leaves
// that word at zero and the host's fallback covers it.
fn journalReference()->f32 {
  if (overlay.filters.y!=0u) {
    let recorded=abs(state[overlay.filters.y]);
    if (recorded>0.0) { return recorded; }
  }
  return max(overlay.domain.w,1e-30);
}
fn arenaFloat(index:u32)->f32{return bitcast<f32>(arena[index]);}

fn journalViewSpace(point:vec3f)->vec3f {
  let relative=point-overlay.cameraPosition.xyz;
  return vec3f(dot(relative,overlay.cameraRight.xyz),
    dot(relative,overlay.cameraUp.xyz),dot(relative,overlay.cameraForward.xyz));
}

fn journalNdc(view:vec3f)->vec2f {
  let tangent=max(overlay.cameraPosition.w,1e-4);
  let depth=max(view.z,JOURNAL_NEAR_VIEW_M);
  return vec2f(view.x/(depth*tangent*max(overlay.cameraForward.w,1e-4)),
    view.y/(depth*tangent));
}

// Blue for negative, red for positive, with a desaturated centre rather than a
// white one: pure white at zero reads as "no data" against a light viewport,
// and a converged interior is mostly zero.
fn divergingColor(signed:f32)->vec3f {
  let t=clamp(signed,-1.0,1.0);
  let neutral=vec3f(0.86,0.87,0.89);
  if (t>=0.0) {
    let hot=vec3f(0.83,0.24,0.16);
    let mid=vec3f(0.95,0.66,0.32);
    if (t<0.5) { return mix(neutral,mid,t*2.0); }
    return mix(mid,hot,(t-0.5)*2.0);
  }
  let cold=vec3f(0.11,0.31,0.68);
  let mid=vec3f(0.35,0.66,0.87);
  let m=-t;
  if (m<0.5) { return mix(neutral,mid,m*2.0); }
  return mix(mid,cold,(m-0.5)*2.0);
}

fn sequentialColor(t:f32)->vec3f {
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

// Signed log: the magnitude is compressed over a fixed number of decades below
// the reference and the sign is carried through untouched. Values under the
// floor land at zero rather than at negative infinity.
fn logMagnitude(magnitude:f32)->f32 {
  let decades=max(overlay.style.x,1.0);
  let normalized=magnitude/journalReference();
  if (normalized<=0.0) { return 0.0; }
  return clamp(1.0+log(normalized)/(2.302585093*decades),0.0,1.0);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->VertexOut {
  var output:VertexOut;
  output.color=vec3f(0.0);
  output.alpha=0.0;
  output.viewDepth_m=0.0;
  output.position=vec4f(0.0,0.0,0.0,1.0);

  let cell=instance*max(overlay.addressing.z,1u);
  if (cell>=overlay.addressing.y) { return output; }

  // One load rejects every cell the captured solve had no unknown for.
  if (state[overlay.addressing.w+cell]<0.5) { return output; }

  let base=cellBase(cell);
  if (overlay.filters.x!=0u&&(arena[base+7u]&15u)!=overlay.filters.x) { return output; }
  let centerFine=vec3f(arenaFloat(base),arenaFloat(base+1u),arenaFloat(base+2u));
  let widthsFine=vec3f(arenaFloat(base+4u),arenaFloat(base+5u),arenaFloat(base+6u));

  if (overlay.slab.y>0.5) {
    let axis=u32(overlay.slab.x);
    // A cell is in the slab when its own extent overlaps it, so a coarse cell
    // whose centre is outside still draws when the plane passes through it.
    let reach=overlay.slab.w+0.5*widthsFine[axis];
    if (abs(centerFine[axis]-overlay.slab.z)>reach) { return output; }
  }

  var value=state[overlay.addressing.x+cell];
  if (overlay.style.w!=0.0) {
    // Depth below the free surface is not known here, so the subtraction is a
    // fixed linear-in-height field: it removes the gradient, not the offset,
    // which is exactly what makes the dynamic part visible.
    value=value-overlay.style.w*(overlay.domain.y-centerFine.y);
  }

  let magnitude=abs(value);
  let rampCode=overlay.style.z;
  var heat=0.0;
  if (rampCode>1.5) { heat=logMagnitude(magnitude); }
  else { heat=clamp(magnitude/journalReference(),0.0,1.0); }
  if (heat<overlay.style.y) { return output; }

  let half=0.5*overlay.container.xyz;
  let world=centerFine/max(overlay.domain.xyz,vec3f(1.0))*overlay.container.xyz
    +vec3f(-half.x,0.0,-half.z);
  let view=journalViewSpace(world);
  if (view.z<JOURNAL_NEAR_VIEW_M) { return output; }

  // The square is the cell's own size, so the picture reports the adaptive grid
  // rather than a uniform stipple over it.
  let widest=max(widthsFine.x,max(widthsFine.y,widthsFine.z));
  let sizeView=0.5*widest*overlay.container.w;
  let viewport=max(overlay.viewport.xy,vec2f(1.0));
  let centerPixel=journalNdc(view)*0.5*viewport;
  let edgePixel=journalNdc(view+vec3f(sizeView,0.0,0.0))*0.5*viewport;
  let halfSpan=max(length(edgePixel-centerPixel),JOURNAL_MINIMUM_SPAN_PIXELS);

  var quad=array<vec2f,6>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),
    vec2f(-1.0,-1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0));
  let pixel=centerPixel+quad[vertexIndex]*halfSpan;

  output.position=vec4f(pixel/(0.5*viewport),0.0,1.0);
  if (rampCode>0.5) { output.color=divergingColor(sign(value)*heat); }
  else { output.color=sequentialColor(heat); }
  // Opacity carries the magnitude too, so a converged interior fades out of the
  // way instead of forming an opaque shell over the cells that still matter.
  output.alpha=clamp(overlay.cameraRight.w,0.0,1.0)*mix(0.25,1.0,heat);
  output.viewDepth_m=view.z;
  return output;
}

@fragment fn fragmentMain(input:VertexOut)->@location(0) vec4f {
  if (input.alpha<=0.0) { discard; }
  var alpha=input.alpha;
  if (overlay.viewport.w>0.5) {
    let coordinate=vec2u(input.position.xy);
    let stored=textureLoad(sceneDepth,coordinate,0);
    if (stored>0.0) {
      let sceneDepth_m=overlay.viewport.z/max(stored,1e-7);
      if (input.viewDepth_m>sceneDepth_m) { alpha*=max(overlay.cameraUp.w,0.0); }
    }
  }
  return vec4f(input.color,alpha);
}
`;

const RAMP_CODE: Readonly<Record<PressureJournalRamp, number>> = Object.freeze({
  "sequential-log": 2,
  "diverging-log": 3,
  "diverging-linear": 1,
});

/** One instanced draw over the captured cells of one journal snapshot. */
export class PressureJournalOverlay {
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
  private source?: GPUPressureJournalSource;
  private destroyed = false;
  private readonly uniformData =
    new ArrayBuffer(PRESSURE_JOURNAL_OVERLAY_UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
  ) {}

  async initialize(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Pressure journal overlay", code: pressureJournalOverlayShader,
    });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`Pressure journal overlay:\n${errors
        .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`)
        .join("\n")}`);
    }
    this.uniforms = this.device.createBuffer({
      label: "Pressure journal overlay uniforms",
      size: PRESSURE_JOURNAL_OVERLAY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.fallbackDepth = this.device.createTexture({
      label: "Pressure journal overlay depth fallback",
      size: [1, 1], format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fallbackDepthView = this.fallbackDepth.createView();
    this.fallbackStorage = this.device.createBuffer({
      label: "Pressure journal overlay fallback",
      size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.layout = this.device.createBindGroupLayout({
      label: "Pressure journal overlay bindings",
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
      label: "Pressure journal overlay",
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
    });
    this.rebuildBindGroup(this.fallbackDepthView);
  }

  get ready(): boolean {
    return Boolean(this.pipeline && this.uniforms && this.layout);
  }

  /** Idempotent: re-binding the same buffers must not rebuild a bind group. */
  setSource(source: GPUPressureJournalSource | undefined): void {
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
      label: "Pressure journal overlay bind group",
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

  /** The stride a given cell count draws at, and the squares that survive it. */
  static plan(cellCount: number, budget: number = PRESSURE_JOURNAL_CELL_BUDGET): {
    readonly stride: number; readonly cells: number;
  } {
    if (!(cellCount > 0) || !(budget >= 1)) return { stride: 1, cells: 0 };
    const stride = Math.max(1, Math.ceil(cellCount / budget));
    return { stride, cells: Math.ceil(cellCount / stride) };
  }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sceneDepth: GPUTextureView | undefined,
    frame: PressureJournalOverlayFrame,
  ): boolean {
    const source = this.source;
    if (!this.ready || !source || source.cellCount === 0) return false;
    const { layout } = source;
    if (layout.snapshotCapacity === 0) return false;
    // Clamp to what was filled, not to what was reserved.
    const filled = Math.min(layout.snapshotCapacity,
      source.snapshotCount > 0 ? source.snapshotCount : layout.snapshotCapacity);
    const snapshot = Math.max(0, Math.min(filled - 1, Math.round(frame.snapshot)));
    const field = SPARSE_CM12_PRESSURE_JOURNAL_FIELDS.indexOf(frame.channel.field);
    if (field < 0) return false;
    this.rebuildBindGroup(sceneDepth);
    if (!this.bindGroup) return false;
    const { stride, cells } = PressureJournalOverlay.plan(source.cellCount);
    if (cells === 0) return false;
    const { camera } = frame;
    const f = this.uniformF32, u = this.uniformU32;
    f.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    f.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    f.set([...camera.right,
      Math.max(PRESSURE_JOURNAL_MINIMUM_ALPHA, Math.min(1, frame.globalAlpha ?? 0.9))], 8);
    f.set([...camera.up, Math.max(0, Math.min(1, frame.occludedAlpha ?? 0.35))], 12);
    f.set([
      Math.max(1, frame.viewportWidth), Math.max(1, frame.viewportHeight),
      Math.max(1e-6, frame.depthNear_m), sceneDepth ? 1 : 0,
    ], 16);
    f.set([...frame.container_m, Math.max(1e-9, source.finestCell_m)], 20);
    f.set([...source.domainFine,
      Math.max(1e-30, Math.abs(frame.channel.reference))], 24);
    f.set([
      Math.max(1, frame.channel.decades),
      PRESSURE_JOURNAL_MINIMUM_FRACTION,
      RAMP_CODE[frame.channel.ramp],
      // Per fine cell of height. The solve's pressure is scaled, so the
      // gradient the film subtracts is expressed in the same reference the
      // ramp uses rather than in pascals.
      frame.channel.subtractHydrostatic
        ? Math.abs(frame.channel.reference) / Math.max(1, source.domainFine[1])
        : 0,
    ], 28);
    const slab = frame.slab;
    f.set(slab
      ? [slab.axis, slab.halfThicknessFine > 0 ? 1 : 0, slab.centerFine,
        slab.halfThicknessFine]
      : [0, 0, 0, 0], 32);
    u.set([
      source.journalFloatOffset
        + sparseCM12PressureJournalSnapshotOffset(layout, snapshot, field),
      source.cellCount, stride, source.liquidFloatOffset,
    ], 36);
    // Pressure is the one channel with no recorded scale of its own, so it
    // alone falls back to the host reference; the residual family points the
    // shader at the seed record and is normalised by the capture itself.
    const referenceWord = frame.channel.field === "pressure" ? 0
      : source.journalFloatOffset + SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS
        + SPARSE_CM12_PRESSURE_JOURNAL_RECORD.initialTrueMaximum;
    u.set([Math.max(0, Math.round(frame.onlyResolution ?? 0)), referenceWord, 0, 0], 40);
    this.device.queue.writeBuffer(this.uniforms!, 0, this.uniformData);
    const pass = encoder.beginRenderPass({
      label: "Pressure journal overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(PRESSURE_JOURNAL_CELL_VERTICES, cells);
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

/**
 * The overlay mode strings the four journal views select by.
 *
 * One mode per captured field rather than one mode with a channel control,
 * because the catalog is the field picker: a view that appeared once and then
 * needed a second control to say what it was showing would be invisible to
 * everything that lists what can be drawn.
 */
export type PressureJournalOverlayMode =
  | "pressure-journal-residual"
  | "pressure-journal-pressure"
  | "pressure-journal-preconditioned"
  | "pressure-journal-direction";

const PRESSURE_JOURNAL_MODE_FIELD:
Readonly<Record<PressureJournalOverlayMode, SparseCM12PressureJournalField>> =
  Object.freeze({
    "pressure-journal-residual": "residual",
    "pressure-journal-pressure": "pressure",
    "pressure-journal-preconditioned": "preconditioned",
    "pressure-journal-direction": "direction",
  });

export function isPressureJournalOverlayMode(
  mode: string | undefined,
): mode is PressureJournalOverlayMode {
  return mode !== undefined && mode in PRESSURE_JOURNAL_MODE_FIELD;
}

/**
 * Reference for the pressure channel, in the solve's own scaled units.
 *
 * The other three channels get their scale from the capture itself — the seed
 * record carries the initial residual max-norm, and z and d are residual-scaled
 * by construction, since z = M⁻¹r and d starts as z. Pressure has no such
 * recorded scalar, so this is the one number here that is a calibration rather
 * than a measurement of the frame being drawn: 10.7 was the 99th percentile of
 * live-cell pressure on the mini 32³ dam break, and the field had settled to it
 * by iteration 8 and did not move again.
 *
 * That makes it scene-dependent in a way the residual reference is not, and it
 * is the known soft spot in this view's calibration. It is overridable per
 * frame; a scene whose pressure lives at a different scale will wash out or
 * clip until either it is set or the capture starts recording a pressure norm.
 */
export const PRESSURE_JOURNAL_PRESSURE_REFERENCE = 10.7;

/**
 * The channel a mode draws, given the capture's own residual scale.
 *
 * `residualReference` is the seed record's initial true residual maximum. Pass
 * it from the capture rather than from the frame: the entire argument for the
 * film is that iteration 30 is drawn on the same scale as iteration 1, and a
 * reference taken per-frame would repaint the domain every time the largest
 * remaining residual moved.
 */
export function pressureJournalOverlayChannel(
  mode: PressureJournalOverlayMode,
  residualReference: number,
  pressureReference: number = PRESSURE_JOURNAL_PRESSURE_REFERENCE,
): PressureJournalChannel {
  const field = PRESSURE_JOURNAL_MODE_FIELD[mode];
  if (field === "pressure") {
    // Sequential, not diverging, and that is measured too: pressure came back
    // non-negative on every captured iteration past the first few (0.1% of
    // cells negative at iteration 16, none at 32). Half a diverging ramp would
    // go unused, and its neutral centre would put the bulk of the field at the
    // colour that reads as "no data".
    return {
      field, ramp: "sequential-log",
      reference: Math.max(1e-30, Math.abs(pressureReference)),
      decades: PRESSURE_JOURNAL_PRESSURE_DECADES,
    };
  }
  return {
    field, ramp: "diverging-log",
    reference: Math.max(1e-30, Math.abs(residualReference)),
    decades: PRESSURE_JOURNAL_DECADES,
  };
}

export const pressureJournalOverlayVisualizations: readonly Visualization[] =
  Object.freeze([
    fieldVisualization({
      kind: "field", id: "pressure-journal/residual", pass: "Pressure lab",
      label: "Residual, by iteration",
      description: "The captured pressure solve's residual b − Ap at one iteration, one square per pressure unknown at that unknown's own size. Signed — about half the live cells are negative at every iteration — on a diverging ramp over six decades below the seed residual, with the scale held fixed for the whole capture, so scrubbing forward is watching the colour drain out of the domain rather than watching a per-frame renormalisation repaint it. The drain is real and it is steep: the residual falls ten decades over sixty-four iterations, so a converged tail genuinely empties the view. Where colour survives longest is where the solve is still working.",
      source: "Captured journal snapshots, read on the device at draw time",
      mode: "pressure-journal-residual", axis: "volume", planeless: true,
      swatch: "#d63e29",
      legend: [
        { swatch: "linear-gradient(90deg,#1c4fad,#59a8de,#dbdee4,#f2a852,#d43e29)", mark: "box", label: "−r → 0 → +r, six decades" },
        { swatch: "#dbdee4", mark: "box", label: "square is the cell's own size" },
      ],
    }),
    fieldVisualization({
      kind: "field", id: "pressure-journal/pressure", pass: "Pressure lab",
      label: "Pressure, by iteration",
      description: "The pressure the captured solve had reached at one iteration. Non-negative and heavy-tailed rather than the vertical hydrostatic ramp one might expect — measured on a real capture the live cells run from near zero to about sixteen with a median under one — so it is drawn on a sequential ramp over three decades, which keeps the bulk near mid-scale instead of crushing it against the bottom. Unlike the residual it does not drain: scrub forward and it stops moving, and the iteration where it stops is the iteration the solve was finished at.",
      source: "Captured journal snapshots, read on the device at draw time",
      mode: "pressure-journal-pressure", axis: "volume", planeless: true,
      swatch: "#4f8fd0",
      legend: [
        { swatch: "linear-gradient(90deg,#2650bd,#29b8d7,#61cf5c,#f5d43d,#e83e2b)", mark: "box", label: "low → high pressure, three decades" },
      ],
    }),
    fieldVisualization({
      kind: "field", id: "pressure-journal/preconditioned", pass: "Pressure lab",
      label: "Preconditioned residual",
      description: "z = M⁻¹r at one captured iteration. Drawn beside the raw residual this is the clearest picture of what the brick-aggregate hierarchy actually does: r is local and high-frequency, z is smooth and reaches across bricks, and the difference between them is the whole argument for preconditioning at all. It shares the residual's scale, since z is residual-scaled by construction.",
      source: "Captured journal snapshots, read on the device at draw time",
      mode: "pressure-journal-preconditioned", axis: "volume", planeless: true,
      swatch: "#3aa17e",
      legend: [
        { swatch: "linear-gradient(90deg,#1c4fad,#59a8de,#dbdee4,#f2a852,#d43e29)", mark: "box", label: "−z → 0 → +z, six decades" },
      ],
    }),
    fieldVisualization({
      kind: "field", id: "pressure-journal/direction", pass: "Pressure lab",
      label: "Search direction",
      description: "The conjugate direction d the iteration stepped along, at one captured iteration. It is the field alpha multiplies into the pressure, so watching it is watching where the next iteration is about to spend its correction. At the seed it is exactly the preconditioned residual — d₀ = z₀ — and the two views are identical there by construction; the iteration they start to differ at is the first one conjugacy has done anything.",
      source: "Captured journal snapshots, read on the device at draw time",
      mode: "pressure-journal-direction", axis: "volume", planeless: true,
      swatch: "#9a6fd0",
      legend: [
        { swatch: "linear-gradient(90deg,#1c4fad,#59a8de,#dbdee4,#f2a852,#d43e29)", mark: "box", label: "−d → 0 → +d, six decades" },
      ],
    }),
  ]);
