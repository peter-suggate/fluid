/**
 * The renderer's one pipeline for every stage lens.
 *
 * A lens is a declaration, not a shader: it names publications, taps, a palette
 * and a classifier per layer kind. This is what turns that declaration into
 * draws. It owns the camera, the slice, the palette upload and the instancing;
 * the lens owns only "what colour is this row, in this phase".
 *
 * One overlay rather than one per stage because the cost model is *armed or
 * not*. Nothing here compiles a shader, allocates a buffer or touches the
 * solver until a lens is selected, and selecting a different lens drops the
 * previous one's pipelines. A method with eighteen stages therefore pays for at
 * most one of them, which is what makes it reasonable to declare a lens beside
 * every stage rather than only beside the interesting ones.
 *
 * ## Fail closed, then fail visibly
 *
 * A program whose resources the source cannot supply is not drawn — a partially
 * bound group would either fail validation or, worse, read a buffer left over
 * from a previous publication and paint a confident wrong picture. What *is*
 * drawn when something is missing is the classifier's own fault class, which
 * the palette paints magenta. Absence is a picture, never a silence.
 */
import {
  LENS_PALETTE_SIZE,
  STAGE_LENS_UNIFORM_RESOURCE,
  lensBindingProgram,
  parseLensTapResource,
  stageLensProblems,
  type AnyStageLens,
  type LensLayerKind,
  type StageLensSource,
} from "./stage-lens";
import {
  STAGE_LENS_UNIFORM_BYTES,
  stageLensHelpersWGSL,
  stageLensLayerVertexCount,
  stageLensLayerWGSL,
  stageLensStructsWGSL,
} from "./stage-lens-layers.wgsl";
import {
  visualizationBindGroupEntries,
  visualizationBindingPreambleWGSL,
  type VisualizationBindingKind,
} from "./visualization-bindings";

/**
 * Instances one layer will draw before it starts skipping.
 *
 * A slice rejects most instances in the vertex stage, but a rejected instance
 * still costs its vertices, so a scene with a million rows would spend tens of
 * millions of invocations to show ten thousand glyphs. Beyond this the layer
 * strides, and the stride is reported rather than hidden: a decimated picture
 * that says it is decimated is useful, one that does not is a lie.
 */
export const STAGE_LENS_INSTANCE_BUDGET = 262_144;

/** Half-width of a drawn segment, in pixels. */
const STAGE_LENS_LINE_HALF_PIXELS = 1.1;
const STAGE_LENS_ARROW_HEAD_PIXELS = 6.5;
const STAGE_LENS_GLYPH_HALF_PIXELS = 4.5;
/** Fraction of a cell a tile pulls back from its own edges, so cells read apart. */
const STAGE_LENS_TILE_INSET = 0.08;
/** Slab a slice accepts either side of its plane, in fine cells. */
const STAGE_LENS_SLAB_HALF_FINE = 0.5;

export interface StageLensOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  readonly tanHalfFov: number;
  readonly aspect: number;
}

export interface StageLensOverlayFrame {
  readonly camera: StageLensOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly container_m: readonly [number, number, number];
  /** The grid overlay's own axis vocabulary. `off` draws nothing. */
  readonly axis: "off" | "x" | "y" | "z" | "volume";
  /** Normalised slice position along that axis, as the slider reports it. */
  readonly position: number;
  /** Scrubber index into the lens's phases. */
  readonly phase: number;
  readonly globalAlpha?: number;
}

/** What one layer of one phase actually drew. */
export interface StageLensLayerReport {
  readonly program: string;
  readonly kind: LensLayerKind;
  readonly instances: number;
  readonly stride: number;
}

const AXIS_CODE: Readonly<Record<StageLensOverlayFrame["axis"], number>> = Object.freeze({
  off: 0, z: 1, x: 2, y: 3, volume: 4,
});

/** Which lattice axis a slice code walks along, or -1 for the whole volume. */
const AXIS_INDEX: readonly number[] = Object.freeze([-1, 2, 0, 1, -1]);

const BINDING_LAYOUT: Readonly<Record<VisualizationBindingKind, GPUBindGroupLayoutEntry["buffer"] | "texture">> =
  Object.freeze({
    "uniform": { type: "uniform" as const },
    "read-only-storage": { type: "read-only-storage" as const },
    "storage": { type: "read-only-storage" as const },
    "texture-3d-uint": "texture",
    "texture-depth-2d": "texture",
  });

interface CompiledProgram {
  readonly lensId: string;
  readonly program: string;
  readonly kind: LensLayerKind;
  readonly pipeline: GPURenderPipeline;
  readonly layout: GPUBindGroupLayout;
  readonly palette: Float32Array;
  readonly scale: number;
}

function paletteFloats(colors: readonly string[]): Float32Array {
  const floats = new Float32Array(4 * LENS_PALETTE_SIZE);
  for (let index = 0; index < LENS_PALETTE_SIZE; index += 1) {
    const [r, g, b, a] = parseColor(colors[Math.min(index, colors.length - 1)] ?? "#ff00ff");
    floats.set([r, g, b, a], 4 * index);
  }
  return floats;
}

/**
 * `#rgb`, `#rrggbb` and `#rrggbbaa`, and nothing else.
 *
 * A lens palette is authored in the same hex the design tokens use. Anything
 * else resolves to the fault colour rather than to black, so a mistyped swatch
 * looks like the fault it is.
 */
function parseColor(value: string): readonly [number, number, number, number] {
  const hex = value.trim().replace(/^#/, "");
  const expand = hex.length === 3 || hex.length === 4
    ? [...hex].map((digit) => digit + digit).join("")
    : hex;
  if (!/^[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/.test(expand)) return [0.882, 0.306, 0.800, 1];
  const byte = (at: number): number => Number.parseInt(expand.slice(at, at + 2), 16) / 255;
  return [byte(0), byte(2), byte(4), expand.length === 8 ? byte(6) : 1];
}

/**
 * The complete WGSL module for one program of one lens.
 *
 * Exported rather than inlined into `compile` because the offline gates — the
 * naga pass and the Dawn pass — have to compile the *same* text the renderer
 * does. A gate that reassembled the pieces itself would keep passing while the
 * composition order here changed underneath it, which is the one failure a
 * shader gate cannot be allowed to have.
 *
 * Returns empty for a program the lens does not declare; the caller has
 * already refused to draw one by then.
 */
export function stageLensProgramWGSL(lens: AnyStageLens, program: string): string {
  const declared = lens.programs[program];
  if (!declared) return "";
  const binding = lensBindingProgram(lens, program);
  // Struct declarations the bound publications carry, ahead of the preamble
  // that names them. De-duplicated because two publications may share a block,
  // and WGSL will not take it twice.
  const structs = new Set<string>();
  for (const declaredBinding of binding.bindings) {
    const tapped = parseLensTapResource(declaredBinding.resource);
    const publication = lens.publications[tapped?.publication ?? declaredBinding.resource];
    if (publication?.wgslStruct) structs.add(publication.wgslStruct);
  }
  return [
    stageLensStructsWGSL,
    ...structs,
    visualizationBindingPreambleWGSL(binding),
    stageLensHelpersWGSL,
    declared.wgsl(),
    stageLensLayerWGSL(declared.kind),
  ].join("\n");
}

/**
 * Whether two resolutions of the same binding are the same thing.
 *
 * A buffer binding is a fresh object every frame — the source rebuilds its
 * ranges from the current addressing rather than caching them — so identity on
 * the wrapper says nothing. What matters is the buffer and the window into it.
 */
function sameResource(a: GPUBindingResource, b: GPUBindingResource): boolean {
  if (a === b) return true;
  const left = a as GPUBufferBinding, right = b as GPUBufferBinding;
  if (!left?.buffer || !right?.buffer) return false;
  return left.buffer === right.buffer
    && (left.offset ?? 0) === (right.offset ?? 0)
    && (left.size ?? 0) === (right.size ?? 0);
}

export class StageLensOverlay {
  private readonly programs = new Map<string, CompiledProgram>();
  private readonly bindGroups = new Map<string, { group: GPUBindGroup;
    readonly resolved: readonly GPUBindingResource[] }>();
  private readonly compiling = new Set<string>();
  private uniforms?: GPUBuffer;
  private uniformStride = 512;
  private source?: StageLensSource;
  private selected?: string;
  private destroyed = false;
  private lastReport: readonly StageLensLayerReport[] = [];
  private readonly uniformData = new ArrayBuffer(STAGE_LENS_UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
  ) {}

  async initialize(): Promise<void> {
    const alignment = Math.max(256, this.device.limits.minUniformBufferOffsetAlignment ?? 256);
    this.uniformStride = Math.ceil(STAGE_LENS_UNIFORM_BYTES / alignment) * alignment;
    this.uniforms = this.device.createBuffer({
      label: "Stage lens uniforms",
      // One slot per layer a phase may draw; a phase with more layers than this
      // reuses slots, which is safe because every slot is written before the
      // pass that reads it begins.
      size: this.uniformStride * 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  get ready(): boolean {
    return Boolean(this.uniforms) && !this.destroyed;
  }

  /** What the last `encode` drew, for the receipt the flyout shows. */
  get report(): readonly StageLensLayerReport[] {
    return this.lastReport;
  }

  setSource(source: StageLensSource | undefined): void {
    if (source === this.source) return;
    this.source = source;
    this.bindGroups.clear();
    // Pipelines survive a source change — they are a property of the lens
    // declaration, not of the buffers it happens to be pointed at today.
    if (source) source.arm(this.selected);
  }

  /**
   * Choose the lens to draw, or none.
   *
   * Arming is the source's business and idempotent, so this is safe to call
   * every frame with whatever the overlay config says. Compilation starts here
   * and finishes whenever it finishes: `encode` draws the programs that are
   * ready and skips the rest, so the first frame or two of a newly opened lens
   * may be partial. That is preferable to blocking a frame on a shader compile.
   */
  select(lensId: string | undefined): void {
    if (lensId === this.selected) return;
    this.selected = lensId;
    this.bindGroups.clear();
    this.source?.arm(lensId);
    const lens = this.lens();
    if (!lens) return;
    for (const program of Object.keys(lens.programs)) void this.compile(lens, program);
  }

  private lens(): AnyStageLens | undefined {
    if (!this.selected) return undefined;
    return this.source?.lenses.find((candidate) => candidate.id === this.selected);
  }

  private async compile(lens: AnyStageLens, program: string): Promise<void> {
    const key = `${lens.id}#${program}`;
    if (this.programs.has(key) || this.compiling.has(key) || !this.ready) return;
    const declared = lens.programs[program];
    if (!declared) return;
    const problems = stageLensProblems(lens);
    if (problems.length > 0) {
      console.warn(`Stage lens ${lens.id} is not drawable:\n${problems.join("\n")}`);
      return;
    }
    this.compiling.add(key);
    try {
      const binding = lensBindingProgram(lens, program);
      const code = stageLensProgramWGSL(lens, program);
      const shaderModule = this.device.createShaderModule({ label: `Stage lens ${key}`, code });
      const info = await shaderModule.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length > 0) {
        throw new Error(`Stage lens ${key}:\n${errors
          .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
      }
      const layout = this.device.createBindGroupLayout({
        label: `Stage lens ${key} bindings`,
        entries: binding.bindings.map((declaredBinding, index) => {
          const shape = BINDING_LAYOUT[declaredBinding.kind];
          const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
          if (shape === "texture") {
            return declaredBinding.kind === "texture-depth-2d"
              ? { binding: index, visibility, texture: { sampleType: "depth" as const } }
              : { binding: index, visibility, texture: { sampleType: "uint" as const, viewDimension: "3d" as const } };
          }
          return {
            binding: index, visibility,
            buffer: { ...shape, hasDynamicOffset: declaredBinding.kind === "uniform"
              && declaredBinding.resource === STAGE_LENS_UNIFORM_RESOURCE },
          };
        }),
      });
      const pipeline = await this.device.createRenderPipelineAsync({
        label: `Stage lens ${key}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
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
        // A glyph's winding flips with its projected direction, and a
        // back-facing arrowhead is still an arrowhead.
        primitive: { topology: "triangle-list" },
      });
      if (this.destroyed) return;
      this.programs.set(key, {
        lensId: lens.id, program, kind: declared.kind, pipeline, layout,
        palette: paletteFloats(declared.palette),
        scale: declared.scale && declared.scale > 0 ? declared.scale : 1,
      });
    } catch (error) {
      console.warn(`Stage lens ${key} failed to build`, error);
    } finally {
      this.compiling.delete(key);
    }
  }

  /**
   * Bind group for one compiled program, or null when a resource is missing.
   *
   * Resolution runs every frame and the group is rebuilt only when what it
   * resolved to changes. Caching the *failure* would be the tempting shortcut
   * and it is wrong twice over: a tap has captured nothing on the frame a lens
   * is armed, so every tap-backed program would be refused once and never
   * reconsidered; and a snapshot that grows after a re-mesh is a new buffer
   * behind an old group.
   */
  private group(lens: AnyStageLens, compiled: CompiledProgram): GPUBindGroup | null {
    const key = `${compiled.lensId}#${compiled.program}`;
    const source = this.source;
    const binding = lensBindingProgram(lens, compiled.program);
    const entries = source && this.uniforms
      ? visualizationBindGroupEntries<GPUBindingResource>(binding, (resource) => {
        if (resource === STAGE_LENS_UNIFORM_RESOURCE) {
          return { buffer: this.uniforms!, offset: 0, size: STAGE_LENS_UNIFORM_BYTES };
        }
        const tapped = parseLensTapResource(resource);
        return tapped
          ? source.snapshot(lens.id, tapped.tap, tapped.publication)
          : source.publication(resource);
      })
      : undefined;
    if (!entries) { this.bindGroups.delete(key); return null; }
    const resolved = entries.map((entry) => entry.resource);
    const cached = this.bindGroups.get(key);
    if (cached && cached.resolved.length === resolved.length
      && cached.resolved.every((was, index) => sameResource(was, resolved[index]!))) {
      return cached.group;
    }
    const group = this.device.createBindGroup({
      label: `Stage lens ${key} bind group`,
      layout: compiled.layout,
      entries: entries.map((entry) => ({ binding: entry.binding, resource: entry.resource })),
    });
    this.bindGroups.set(key, { group, resolved });
    return group;
  }

  /**
   * Instances a layer draws and the stride it draws them at.
   *
   * Exposed so a caller can say what it drew rather than implying it drew
   * everything; `encode` applies exactly this.
   */
  static plan(count: number, budget: number = STAGE_LENS_INSTANCE_BUDGET): {
    readonly stride: number; readonly instances: number;
  } {
    if (!(count > 0) || !(budget >= 1)) return { stride: 1, instances: 0 };
    const stride = Math.max(1, Math.ceil(count / budget));
    return { stride, instances: Math.ceil(count / stride) };
  }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    frame: StageLensOverlayFrame,
  ): boolean {
    this.lastReport = [];
    const lens = this.lens();
    const source = this.source;
    if (!this.ready || !lens || !source || frame.axis === "off") return false;
    const phase = lens.phases[Math.max(0, Math.min(lens.phases.length - 1, Math.trunc(frame.phase)))];
    if (!phase || phase.layers.length === 0) return false;

    const axisCode = AXIS_CODE[frame.axis];
    const axis = AXIS_INDEX[axisCode] ?? -1;
    // Snapped to a cell layer rather than left continuous: a slab that lands
    // between two layers of face centres shows a stripe of each and neither
    // cleanly, which is the picture the whole slice discipline exists to avoid.
    const extentFine = axis >= 0 ? Math.max(1, source.space.dimensions[axis] ?? 1) : 1;
    const layerFine = axis >= 0
      ? Math.max(0, Math.min(extentFine - 1,
        Math.floor(Math.max(0, Math.min(1, frame.position)) * extentFine)))
      : 0;

    const draws: { compiled: CompiledProgram; group: GPUBindGroup; slot: number;
      instances: number; stride: number }[] = [];
    let slot = 0;
    for (const layer of phase.layers) {
      const compiled = this.programs.get(`${lens.id}#${layer.program}`);
      if (!compiled) { void this.compile(lens, layer.program); continue; }
      const group = this.group(lens, compiled);
      if (!group) continue;
      const count = compiled.kind === "row-glyph" ? source.capacities.rows
        : compiled.kind === "brick-frame" ? source.capacities.bricks
        : source.capacities.cells;
      const { stride, instances } = StageLensOverlay.plan(count);
      if (instances === 0) continue;
      const offset = (slot % 8) * this.uniformStride;
      this.writeUniform(source, compiled, frame, axisCode, layerFine, count, stride);
      this.device.queue.writeBuffer(this.uniforms!, offset, this.uniformData);
      draws.push({ compiled, group, slot: offset, instances, stride });
      slot += 1;
    }
    if (draws.length === 0) return false;

    const pass = encoder.beginRenderPass({
      label: `Stage lens ${lens.id}`,
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    for (const draw of draws) {
      pass.setPipeline(draw.compiled.pipeline);
      pass.setBindGroup(0, draw.group, [draw.slot]);
      pass.draw(stageLensLayerVertexCount(draw.compiled.kind), draw.instances);
    }
    pass.end();
    this.lastReport = draws.map((draw) => ({
      program: draw.compiled.program, kind: draw.compiled.kind,
      instances: draw.instances, stride: draw.stride,
    }));
    return true;
  }

  private writeUniform(
    source: StageLensSource,
    compiled: CompiledProgram,
    frame: StageLensOverlayFrame,
    axisCode: number,
    layerFine: number,
    count: number,
    stride: number,
  ): void {
    const { camera } = frame;
    const f = this.uniformF32, u = this.uniformU32;
    f.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    f.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    f.set([...camera.right, Math.max(0, Math.min(1, frame.globalAlpha ?? 0.92))], 8);
    f.set([...camera.up, 0], 12);
    f.set([Math.max(1, frame.viewportWidth), Math.max(1, frame.viewportHeight), 0, 0], 16);
    const finest = Math.max(1e-9, source.space.extent_m[0]! / Math.max(1, source.space.dimensions[0]!));
    f.set([...frame.container_m, finest], 20);
    f.set([...source.space.dimensions, compiled.scale], 24);
    f.set([axisCode, layerFine, STAGE_LENS_SLAB_HALF_FINE, 0], 28);
    f.set([STAGE_LENS_LINE_HALF_PIXELS, STAGE_LENS_ARROW_HEAD_PIXELS,
      STAGE_LENS_GLYPH_HALF_PIXELS, STAGE_LENS_TILE_INSET], 32);
    u.set([Math.max(0, Math.trunc(frame.phase)), stride, count, 0], 36);
    u.set([], 40);
    // The palette occupies the tail of the block, one vec4 per class.
    f.set(compiled.palette, 40);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniforms?.destroy();
    this.uniforms = undefined;
    this.programs.clear();
    this.bindGroups.clear();
    this.source?.arm(undefined);
    this.source = undefined;
  }
}
