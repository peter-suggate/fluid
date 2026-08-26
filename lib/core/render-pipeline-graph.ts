/**
 * The frame graph, as a table.
 *
 * One row per pass the presentation encoder actually emits, in encode order,
 * with the *stage ids* that pass closes its seam under. The render panel draws
 * the trunk from this and nothing else, so the diagram, the per-node timing,
 * and the stage-plane taps cannot disagree about what the frame did — they are
 * three projections of the same list.
 *
 * ## Why stage ids and not labels
 *
 * Every seam used to carry its own `{ id, label }` literal at the encode site,
 * and this table matched on the label string. Two hand-maintained lists, one
 * of them inside three ten-thousand-line encoders: a renamed pass detached its
 * row from its cost silently, and a detached row did not go blank — it became
 * the band's only unmeasured row and inherited the band's whole
 * fence-partitioned wall. That is how *world build* came to report 27.9 ms on
 * frames whose world maintenance encoded nothing at all.
 *
 * The seam now names a stage id from `render-frame-stages.ts` and the trace
 * label is looked up from it, so there is one list; `STAGE_NODE` below assigns
 * every stage to exactly one row and is exhaustive over the ABI, so adding a
 * stage without giving it a row is a type error rather than a pass charged to
 * whoever closes next.
 *
 * ## What a row's number is a number of
 *
 * A row is priced from the frame's own manifest first and its timestamps
 * second, because the manifest answers a question timestamps cannot: a stage
 * that encoded no pass is a true zero, and a stage whose passes are render
 * passes on a tile-based GPU is unpriceable — the two used to arrive here
 * identically as "no measurement". Only the second kind may absorb a band's
 * wall residual; the first kind reads zero, which is what it is.
 */

import type { SvoRenderStageView } from "../svo/svo-render-diagnostics";
import {
  RENDER_FRAME_STAGES,
  RENDER_FRAME_STAGE_TRACE,
  type RenderFrameManifest,
  type RenderFrameStageId,
} from "./render-frame-stages";
import type { PerformanceTrace } from "./performance-trace";
import type { SvoConeTracingMode } from "../svo/svo-render-options";
import type { SvoConeRadianceReconstruction, SvoRenderTuning } from "../svo/svo-render-tuning";
import type { DisabledRenderStages, RenderStageSwitchId } from "./render-stage-switches";

export type RenderPipelineBandId = "source" | "primary" | "lighting" | "shading" | "output";

export interface RenderPipelineBand {
  readonly id: RenderPipelineBandId;
  readonly label: string;
}

export const RENDER_PIPELINE_BANDS: readonly RenderPipelineBand[] = Object.freeze([
  { id: "source", label: "Source" },
  { id: "primary", label: "Primary visibility" },
  { id: "lighting", label: "Lighting visibility" },
  { id: "shading", label: "Shading" },
  { id: "output", label: "Output" },
]);

/**
 * What a node is doing this frame.
 *
 * `armed` is not a third kind of "on": it is a node the user has enabled whose
 * work the frame declined to do — for example, a cache that was already warm.
 * Drawing that as `on` would credit it with time it never spent,
 * and drawing it as `off` would say the user turned it off.
 */
export type RenderPipelineNodeState = "on" | "off" | "armed" | "unavailable";

/** Everything a node consults to describe itself. Read once per render. */
export interface RenderPipelineContext {
  /** Stages withheld from the encode. A node in here is `off`, whatever else it would say. */
  readonly disabledStages: DisabledRenderStages;
  readonly coneTracingMode: SvoConeTracingMode;
  readonly shadowsEnabled: boolean;
  readonly ambientOcclusionEnabled: boolean;
  readonly seamClosureEnabled: boolean;
  readonly globalIlluminationEnabled: boolean;
  readonly tuning: SvoRenderTuning;
  readonly sceneHasFluid: boolean;
  readonly refinementDepth: number;
  readonly leafVoxel_mm: number;
  readonly brickCount?: number;
  readonly presentationWidth?: number;
  readonly presentationHeight?: number;
  readonly rendererActive: boolean;
  /** The plane currently replacing the composite, or `off`. */
  readonly stageView: SvoRenderStageView;
  /**
   * Whether the frame is actually running the rasterized brick-proxy primary.
   *
   * False in production — the resolver answers `traced` unconditionally — which
   * is what makes three of the primary-band nodes `unavailable` rather than
   * `off`: their passes are not withheld, they are not reached. Their switches
   * still exist and take effect under `FLUID_SVO_PRIMARY_TRAVERSAL=raster`, and
   * saying "unavailable" is how the panel avoids claiming a saving it cannot
   * make on the arm that is running.
   */
  readonly rasterPrimaryActive: boolean;
}

export interface RenderPipelineTip {
  readonly summary: string;
  readonly reads?: string;
  readonly writes?: string;
  readonly feeds?: string;
  /** The condition that decides whether this pass runs at all. */
  readonly gate?: string;
}

/**
 * Every row of the trunk. Declared as a union so `STAGE_NODE` cannot file a
 * stage under a row that does not exist.
 */
export type RenderPipelineNodeId =
  | "sparse-world-build"
  | "derived-lighting"
  | "rigid-pose-mirror"
  | "fluid-coverage"
  | "primary-traversal"
  | "thin-glass"
  | "scene-primitive"
  | "rigid-impostor"
  | "seam-closure"
  | "cone-visibility"
  | "voxel-light-cache"
  | "world-gi-cache"
  | "reduced-shade"
  | "sky-lighting"
  | "deferred-lighting"
  | "gi-composition"
  | "surface-extraction"
  | "water-interfaces"
  | "caustics"
  | "optical-composite"
  | "inspection-overlays"
  | "present";

/**
 * Which row reports each stage of the frame.
 *
 * Exhaustive over the encoders' stage ABI: a stage added to `RENDER_FRAME_STAGES`
 * without an entry here is a type error, which is the whole point — an
 * unassigned stage is a pass chain nothing prices.
 */
const STAGE_NODE = {
  "world-topology-publish": "sparse-world-build",
  "world-proxy-voxelize": "sparse-world-build",
  "world-derived-lighting": "derived-lighting",
  "world-radiance-feedback": "derived-lighting",
  "rigid-pose-mirror": "rigid-pose-mirror",
  "fluid-coverage": "fluid-coverage",
  "surface-extraction": "surface-extraction",
  caustics: "caustics",
  "scene-primitive-visibility": "scene-primitive",
  "near-field-band": "scene-primitive",
  "brick-cull": "primary-traversal",
  "cone-prepass": "cone-visibility",
  "primary-traversal": "primary-traversal",
  "rigid-discovery": "rigid-impostor",
  "thin-glass-discovery": "thin-glass",
  "seam-closure": "seam-closure",
  "voxel-light-cache": "voxel-light-cache",
  "compact-cone-lighting": "cone-visibility",
  "cone-fanout": "cone-visibility",
  "world-gi-cache": "world-gi-cache",
  "reduced-shade": "reduced-shade",
  "sky-lighting": "sky-lighting",
  "deferred-lighting": "deferred-lighting",
  "inline-traversal-shading": "primary-traversal",
  "fluid-only-rigid-bodies": "deferred-lighting",
  "fluid-only-background": "deferred-lighting",
  "dry-scene-unavailable": "deferred-lighting",
  "water-front-interface": "water-interfaces",
  "water-back-interface": "water-interfaces",
  "water-rear-interfaces": "water-interfaces",
  "optical-composite": "optical-composite",
  "inspection-overlays": "inspection-overlays",
  present: "present",
} as const satisfies Record<RenderFrameStageId, RenderPipelineNodeId>;

/** The stages one row owns, in encode order. */
export const renderPipelineNodeStages = (node: RenderPipelineNodeId): readonly RenderFrameStageId[] =>
  RENDER_FRAME_STAGES.filter((stage) => STAGE_NODE[stage] === node);

export interface RenderPipelineNodeDefinition {
  readonly id: RenderPipelineNodeId;
  readonly band: RenderPipelineBandId;
  readonly side: "left" | "right";
  readonly label: string;
  /**
   * The ablation switch this node's lamp throws.
   *
   * Absent on the three nodes switched by a contract the shaders already compile
   * against — cone visibility, GI composition, and seam closure. Those keep
   * their own flag; a second way to turn one off would be two
   * sources of truth for one bit. {@link switchedBy} names the flag instead.
   */
  readonly stage?: RenderStageSwitchId;
  /** For a node with no `stage`: the store flag its lamp moves. */
  readonly switchedBy?: string;
  /**
   * Set when this node's work is a term inside another node's pass rather than
   * a dispatch of its own, naming the node that owns the pass. There is no
   * phase to sum, so the panel reports the host's cost, marked as shared,
   * instead of an em dash that would read as "not measured".
   */
  readonly costInsideNode?: string;
  /**
   * Set on a node that decides something rather than doing something. It spends
   * no frame time of its own however it is switched, so its cost is a true zero
   * and not a missing measurement.
   */
  readonly spendsNoFrameTime?: boolean;
  /**
   * Published planes this node wrote, in the order the stage catalogue lists
   * them. The ◨ tap presents the first; the node's expanded card offers the
   * rest. Every one of the eighteen views belongs to exactly one node, which is
   * what retired the flat grid without retiring any of the planes.
   */
  readonly taps: readonly SvoRenderStageView[];
  /** True when the node's lamp is a user control rather than a readout. */
  readonly toggleable: boolean;
  /**
   * Rows sharing a group render as one collapsed row while every member is
   * `unavailable` — a run of dashed placeholders is diagram space spent on a
   * path the frame cannot take. The group expands back into its member rows
   * the moment any member becomes reachable.
   */
  readonly collapseGroup?: RenderPipelineCollapseGroupId;
  readonly tip: RenderPipelineTip;
  readonly state: (context: RenderPipelineContext) => RenderPipelineNodeState;
  /** The short factual chip under the label. Never a description. */
  readonly chip: (context: RenderPipelineContext) => string;
}

const reduced = (context: RenderPipelineContext) =>
  context.coneTracingMode === "cones" && context.tuning.coneLightingScale !== 1;

/**
 * Relight modes consume the world-GI cache; reconstruction modes interpolate a
 * reduced-rate material result instead and the cache never runs. The two
 * families are the real switch, so the panel offers them as one.
 */
export const RELIGHT_RECONSTRUCTIONS: readonly SvoConeRadianceReconstruction[] =
  Object.freeze(["wide-relight", "full-res-relight"]);

export const isRelightReconstruction = (mode: SvoConeRadianceReconstruction) =>
  RELIGHT_RECONSTRUCTIONS.includes(mode);

const coneRateLabel = (scale: number) =>
  scale === 1 ? "full rate" : `${1 / scale}×${1 / scale}`;

/**
 * The three visibility tiers that exist only on the rasterized brick-proxy
 * primary.
 *
 * `unavailable` rather than `off` when the megakernel is the primary: the pass
 * is not being withheld, it is not reachable, and reporting a saving for
 * switching it would be a saving the frame never makes.
 */
const rasterTierState = (context: RenderPipelineContext, stage: RenderStageSwitchId): RenderPipelineNodeState =>
  (!context.rasterPrimaryActive ? "unavailable" : context.disabledStages.has(stage) ? "off" : "on");

const rasterTierChip = (context: RenderPipelineContext, label: string) =>
  (!context.rasterPrimaryActive ? label : "withheld · not encoded");

export type RenderPipelineCollapseGroupId = "raster-arm";

export interface RenderPipelineCollapseGroupDefinition {
  readonly label: string;
  readonly chip: string;
  readonly summary: string;
}

/** What a collapsed run of rows reads as while its arm is unreachable. */
export const RENDER_PIPELINE_COLLAPSE_GROUPS: Readonly<Record<RenderPipelineCollapseGroupId, RenderPipelineCollapseGroupDefinition>> = Object.freeze({
  "raster-arm": {
    label: "Raster arm",
    chip: "3 tiers · not reachable",
    summary: "The rasterized brick-proxy primary's own visibility tiers — thin-glass discovery, the scene-primitive tier, and rigid impostors. The megakernel primary resolves all three inline, so none of these passes can encode; the arm expands into its rows only under FLUID_SVO_PRIMARY_TRAVERSAL=raster.",
  },
});

// Declared before freezing so every `state`/`chip` closure is contextually
// typed by `RenderPipelineNode`; `Object.freeze` on a bare literal widens the
// array and takes the parameter types with it.
const NODES: readonly RenderPipelineNodeDefinition[] = [
  {
    id: "sparse-world-build",
    band: "source",
    side: "left",
    label: "Sparse world build",
    // Incremental voxelization, encoded before any pass that marches it.
    stage: "sparse-world-build",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Voxelizes the authored scene into the sparse octree. Everything below is priced in the leaf this sets — a finer leaf is more bricks for the primary to march and more levels for the cone hierarchy to carry. Off freezes the world at whatever is already built instead of emptying it, so a stationary scene keeps its image and loses only the incremental work.",
      writes: "octree topology · brick payloads · node-mip opacity pyramid",
      feeds: "every pass in the frame",
      gate: "only a scene the solver does not own can spend refinement depth; a solver brick pins its node",
    },
    state: (context) => (context.disabledStages.has("sparse-world-build") ? "off"
      : context.rendererActive ? "on" : "unavailable"),
    chip: (context) => (context.disabledStages.has("sparse-world-build") ? "frozen · no maintenance"
      : `depth ${context.refinementDepth} · leaf ${context.leafVoxel_mm.toFixed(3).replace(/\.?0+$/, "")} mm`),
  },
  {
    id: "derived-lighting",
    band: "source",
    side: "right",
    label: "Derived lighting publish",
    // Split out of the world-build row, which used to close one seam over the
    // whole maintenance subtree and then report whatever the source band's
    // wall residual happened to be. The planner and the builder are the
    // expensive half of that subtree and they run on a different schedule from
    // voxelization — a settled world still runs radiance feedback for a
    // bounded window after every source revision — so one row could never be
    // read as either.
    taps: [],
    toggleable: false,
    tip: {
      summary: "Rebuilds what the lighting passes march instead of the octree itself: the node-mip opacity pyramid's address plan, the tetrahedral radiance pages, and the bounded diffuse feedback window that follows every source revision. Runs off the world build's completion, not the camera, so it is silent on a settled scene and appears whenever the scene is touched.",
      writes: "node-mip opacity pyramid · tetrahedral radiance pages",
      feeds: "cone visibility · GI composition",
      gate: "a valid derived address plan, after a source revision or while the feedback window is open",
    },
    state: (context) => (context.disabledStages.has("sparse-world-build") ? "off"
      : context.rendererActive ? "on" : "unavailable"),
    chip: (context) => (context.disabledStages.has("sparse-world-build") ? "frozen · with the world"
      : "plan · build · feedback"),
  },
  {
    id: "rigid-pose-mirror",
    band: "source",
    side: "left",
    label: "Rigid pose mirror",
    // Copies, not passes — but they are encoded work between two seams, and a
    // seam that does not own them charges them to a neighbour. The row exists
    // so the partition is a partition.
    taps: [],
    toggleable: false,
    tip: {
      summary: "Mirrors the solver's resident rigid poses into the renderer's own body buffer and stages a copy for the host, so a gizmo tracks a body the solver is moving. Buffer copies rather than passes: this row is here so the frame's partition has no unowned commands in it, and it should read at or near zero.",
      reads: "solver-resident rigid pose buffer",
      writes: "renderer body buffer · pose staging",
      gate: "a resident rigid buffer with at least one body",
    },
    state: (context) => (context.rendererActive ? "on" : "unavailable"),
    chip: () => "copy · staged readback",
  },
  {
    id: "fluid-coverage",
    band: "source",
    side: "right",
    label: "Fluid coverage",
    stage: "fluid-coverage",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Fills the fluid-coverage volume and reduces it down its mip chain, one compute pass per level, so the dry passes know where water occludes the scene. Off freezes the volume at its last fill — consumers keep reading the retained texture — so the delta is the fill plus the whole mip chain and nothing downstream.",
      writes: "fluid coverage volume · one mip per level",
      feeds: "primary traversal · deferred lighting",
      gate: "a scene with fluid, under the sparse presentation",
    },
    state: (context) => (context.disabledStages.has("fluid-coverage") ? "off"
      : context.sceneHasFluid && context.rendererActive ? "on" : "unavailable"),
    chip: (context) => (context.disabledStages.has("fluid-coverage") ? "withheld · volume frozen"
      : context.sceneHasFluid ? "fill + mip chain" : "dry scene"),
  },
  {
    id: "primary-traversal",
    band: "primary",
    side: "left",
    label: "Primary traversal",
    stage: "primary-traversal",
    taps: [
      "pass-claimant", "primary-depth", "surface-normal", "primary-failure", "publication-generation",
      "owner-identity", "material-identity", "media-stack", "surface-motion",
    ],
    toggleable: true,
    tip: {
      summary: "The full-screen traversal megakernel: one ray per pixel, marching the octree for itself. This is the primary — the rasterized brick-proxy arm remains compiled but is reachable only through FLUID_SVO_PRIMARY_TRAVERSAL, because it measures 2.1–4.4× slower across the refinement ladder. Off keeps the G-buffer clears and drops the march, so every pixel misses and the frame resolves to sky: the delta is the whole cost of primary visibility.",
      writes: "packedSurface · identityMedia · hardwareDepth · splitGeometry",
      feeds: "every lighting and shading pass",
    },
    state: (context) => (context.disabledStages.has("primary-traversal") ? "off" : "on"),
    chip: (context) => (context.disabledStages.has("primary-traversal")
      ? "withheld · clears only" : "megakernel · canonical-parametric"),
  },
  {
    id: "thin-glass",
    band: "primary",
    side: "right",
    label: "Thin-glass discovery",
    stage: "thin-glass",
    taps: ["glass-discovery"],
    toggleable: true,
    collapseGroup: "raster-arm",
    tip: {
      summary: "Records the nearest glass pane per pixel. The megakernel resolves panes inline and packs the winning key into the opaque identity's spare bits, so the separate raster pass only runs on the raster primary.",
      writes: "splitGlassKey",
      gate: "raster primary with at least one authored pane",
    },
    state: (context) => rasterTierState(context, "thin-glass"),
    chip: (context) => rasterTierChip(context, "inline · raster arm only"),
  },
  {
    id: "scene-primitive",
    band: "primary",
    side: "left",
    label: "Scene-primitive tier",
    stage: "scene-primitive",
    taps: [],
    toggleable: true,
    collapseGroup: "raster-arm",
    tip: {
      summary: "Draws authored analytic records as their own visibility tier, ahead of the voxels. The megakernel returns at the first voxel instead, which is the whole of the bounded 0.087%-of-pixels difference between the two primary arms.",
      gate: "raster primary only",
    },
    state: (context) => rasterTierState(context, "scene-primitive"),
    chip: (context) => rasterTierChip(context, "raster arm only"),
  },
  {
    id: "rigid-impostor",
    band: "primary",
    side: "right",
    label: "Rigid impostor",
    stage: "rigid-impostor",
    taps: ["rigid-impostor"],
    toggleable: true,
    collapseGroup: "raster-arm",
    tip: {
      summary: "Rasterizes solver rigid bodies as analytic impostors, then bridges their certificate into primary geometry. The megakernel folds the analytic body loop into traceOpaqueScene instead.",
      gate: "raster primary only",
    },
    state: (context) => rasterTierState(context, "rigid-impostor"),
    chip: (context) => rasterTierChip(context, "inline · raster arm only"),
  },
  {
    id: "seam-closure",
    band: "primary",
    side: "left",
    label: "Primary seam closure",
    switchedBy: "silhouetteRefinementEnabled",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Closes one-pixel background seams, but only where opposing foreground surfaces bracket the pixel. Runs before sky and deferred lighting.",
      reads: "packedSurface · hardwareDepth",
      writes: "packedSurface · identityMedia · hardwareDepth",
    },
    state: (context) => (context.seamClosureEnabled ? "on" : "off"),
    chip: (context) => (context.seamClosureEnabled ? "bracketed pixels only" : "off"),
  },
  {
    id: "cone-visibility",
    band: "lighting",
    side: "right",
    label: "Cone prepass + visibility",
    switchedBy: "svoConeTracingMode",
    taps: ["cone-ambient-visibility", "cone-light-visibility", "cone-geometry"],
    toggleable: true,
    tip: {
      summary: "Cone-traced soft shadows, AO and GI, marched at the prepass rate against the node-mip opacity pyramid. Turning this off does not remove shadows — it hands them to EXACT rays, which are sharper and cost more per pixel. OFF removes visibility work entirely.",
      reads: "primary G-buffer · node-mip opacity pyramid",
      writes: "conePrepassVisibility · 8 packed light slots",
      feeds: "world GI cache → deferred lighting",
      gate: "cone hierarchy of at most twelve levels; past that derived lighting withdraws and the status line reads EXACT FALLBACK",
    },
    state: (context) => (context.coneTracingMode === "cones" ? "on" : context.coneTracingMode === "exact" ? "armed" : "off"),
    chip: (context) => context.coneTracingMode === "cones"
      ? `cones · ${coneRateLabel(context.tuning.coneLightingScale)} · 8 slots`
      : context.coneTracingMode === "exact" ? "exact rays · no cone stage" : "no visibility work",
  },
  {
    id: "voxel-light-cache",
    band: "lighting",
    side: "left",
    label: "Voxel light cache",
    stage: "voxel-light-cache",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Persistent level-0 voxel visibility for directional light slot zero, drained over several frames by a bounded queue. Off withholds the demand and population dispatches; the consumers keep their bindings and read whatever the cache last held, so this measures the drain and not the lookup.",
      writes: "rg32uint voxel visibility cache",
      gate: "cones with shadows on a dry scene, split shading, and a device exposing at least seventeen sampled textures per stage",
    },
    // Mirrors the encoder's own gate: fluid coverage displaces the cache, and
    // it only drains under cones with shadows enabled. Reporting `on` for a wet
    // scene lit the lamp over dispatches the frame never encoded.
    state: (context) => (context.disabledStages.has("voxel-light-cache") ? "off"
      : context.coneTracingMode !== "cones" || !context.shadowsEnabled || context.sceneHasFluid ? "off" : "on"),
    chip: (context) => (context.disabledStages.has("voxel-light-cache") ? "withheld · drain stopped"
      : context.sceneHasFluid ? "fluid coverage · cache idle"
      : context.coneTracingMode !== "cones" || !context.shadowsEnabled ? "needs cones + shadows"
      : "slot 0 · 16 384 voxels/frame"),
  },
  {
    id: "world-gi-cache",
    band: "lighting",
    side: "right",
    label: "World-space GI cache",
    stage: "world-gi-cache",
    taps: ["cone-radiance"],
    toggleable: true,
    tip: {
      summary: "A world-keyed cache of gathered indirect radiance and visibility. Camera motion changes which keys are queried but never invalidates an entry; only a source, scene or lighting change clears it. Off withholds the fill; the deferred pass keeps querying it and reads a cleared cache, so indirect light goes flat rather than stale.",
      writes: "262 144-entry GI closure cache",
      feeds: "deferred lighting",
      gate: "a relight reconstruction, at a reduced cone rate, with cones active",
    },
    state: (context) => (context.disabledStages.has("world-gi-cache") ? "off"
      : reduced(context) && isRelightReconstruction(context.tuning.coneRadianceReconstruction) ? "on" : "off"),
    chip: (context) => (context.disabledStages.has("world-gi-cache") ? "withheld · fill stopped"
      : isRelightReconstruction(context.tuning.coneRadianceReconstruction)
        ? "relight · 262 144 entries" : "upsample · cache idle"),
  },
  {
    id: "reduced-shade",
    band: "shading",
    side: "left",
    label: "Reduced-rate opaque shade",
    stage: "reduced-shade",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Evaluates the full material at the reduced cone rate so the deferred pass has something to reconstruct from. Encoded only under a reconstruction mode: the relight modes evaluate the material at full rate inside deferred lighting instead, and this pass never runs there. Off keeps the clear and drops the draw, so reconstruction guides off an empty result and the image loses its reduced-rate colour.",
      reads: "conePrepassVisibility · conePrepassGeometry",
      writes: "conePrepassRadiance",
      gate: "a reduced cone rate, under a reconstruction (non-relight) mode",
    },
    // On only when the encoder actually emits the pass: a relight reconstruction
    // never encodes it, and reporting `on` there left the lamp lit over a row
    // that could only ever read idle.
    state: (context) => (context.disabledStages.has("reduced-shade") ? "off"
      : reduced(context) && !isRelightReconstruction(context.tuning.coneRadianceReconstruction) ? "on" : "off"),
    chip: (context) => (context.disabledStages.has("reduced-shade") ? "withheld · clear only"
      : reduced(context)
        ? isRelightReconstruction(context.tuning.coneRadianceReconstruction) ? "relight · not encoded" : "upsample"
        : "full rate · not encoded"),
  },
  {
    id: "sky-lighting",
    band: "shading",
    side: "right",
    label: "Sky lighting",
    stage: "sky-lighting",
    taps: [],
    toggleable: true,
    tip: {
      summary: "The miss half of the depth partition: its own pass resolves sky and thin glass over open sky at every pixel primary visibility left empty. Off keeps the pass and its clear, so the miss pixels go black and the delta is exactly what the sky resolve was worth.",
      reads: "primary depth · environment · thin-glass key plane",
      writes: "dry scene HDR (miss pixels)",
      feeds: "deferred lighting · optical composite",
    },
    state: (context) => (context.disabledStages.has("sky-lighting") ? "off" : "on"),
    chip: (context) => (context.disabledStages.has("sky-lighting")
      ? "withheld · clear only" : "miss pixels · own pass"),
  },
  {
    id: "deferred-lighting",
    band: "shading",
    side: "right",
    label: "Deferred lighting",
    stage: "deferred-lighting",
    taps: ["dry-radiance", "lighting-partition"],
    toggleable: true,
    tip: {
      summary: "The surface half of the depth partition, in its own pass since the sky split out: the full deferred shader (and the reconstruction draw at reduced cone rates) shades every pixel the depth buffer resolved to a surface. Off keeps the pass, which loads the sky pass's result, so geometry goes black while the sky stays and the delta is the whole cost of deferred shading.",
      reads: "primary G-buffer · cone visibility · GI cache",
      writes: "dry scene HDR",
      feeds: "optical composite",
    },
    state: (context) => (context.disabledStages.has("deferred-lighting") ? "off" : "on"),
    chip: (context) => (context.disabledStages.has("deferred-lighting")
      ? "withheld · sky retained" : "surface pixels · own pass"),
  },
  {
    id: "gi-composition",
    band: "shading",
    side: "left",
    label: "GI composition",
    // A term inside the deferred draw, not a dispatch. It has a row because it
    // is the frame's energy balance and has to be steerable; it owns no stage
    // of its own, so it reports the pass it runs inside.
    costInsideNode: "deferred-lighting",
    switchedBy: "svoGlobalIlluminationEnabled",
    taps: [],
    toggleable: true,
    tip: {
      summary: "The indirect gather and its energy balance: how much bounced light, how much broad cone occlusion, and how much analytic sky fill survives beside them. Switching it off withholds the gather and the world-GI cache rather than scaling them to zero, so the frame gets that time back; cone shadows and AO are unaffected.",
      reads: "node-mip pyramid · tetrahedral radiance · cone visibility",
      feeds: "deferred lighting",
      gate: "cones, and a complete derived lighting publication",
    },
    state: (context) => (context.coneTracingMode !== "cones" ? "off"
      : context.globalIlluminationEnabled ? "on" : "off"),
    chip: (context) => (context.coneTracingMode !== "cones" ? "needs cones"
      : context.globalIlluminationEnabled
        ? `${context.tuning.giConeCount} cones · ${context.tuning.giConeAperture.toFixed(2)} rad`
        : "withheld · gather not encoded"),
  },
  {
    id: "surface-extraction",
    band: "output",
    side: "left",
    label: "Surface extraction",
    stage: "surface-extraction",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Classifies the level set, sizes the indirect dispatch and emits the water isosurface triangles — the largest compute block in a wet frame, throttled to the solver revision. Off freezes the mesh at its last extraction: the interfaces keep drawing the retained surface, so the delta is classify + scan + emit and nothing downstream. The t=0 startup capture overrides the withhold so a fresh scene can still admit its first frame.",
      writes: "surface vertex buffer · indirect draw args",
      feeds: "water interfaces · caustic map",
      gate: "a scene with fluid, on solver revision change under a 250 ms throttle",
    },
    state: (context) => (context.disabledStages.has("surface-extraction") ? "off"
      : context.sceneHasFluid ? "on" : "unavailable"),
    chip: (context) => (context.disabledStages.has("surface-extraction") ? "withheld · mesh frozen"
      : context.sceneHasFluid ? "classify · scan · emit" : "dry scene"),
  },
  {
    id: "water-interfaces",
    band: "output",
    side: "right",
    label: "Water + spray interfaces",
    stage: "water-interfaces",
    taps: ["water-depth"],
    toggleable: true,
    tip: {
      summary: "Draws four depth-peeled interface layers for water and spray from the extracted surface. A dry scene clears the attachments once and never draws them again. Off takes those same once-only clears on a wet scene, so the composite sees no interfaces rather than compositing the last ones drawn.",
      reads: "surface vertex buffer",
      writes: "front/back position · normal · depth",
      feeds: "optical composite",
      gate: "a scene with fluid",
    },
    state: (context) => (context.disabledStages.has("water-interfaces") ? "off"
      : context.sceneHasFluid ? "on" : "unavailable"),
    chip: (context) => (context.disabledStages.has("water-interfaces") ? "withheld · cleared once"
      : context.sceneHasFluid ? "4 peeled layers" : "dry scene"),
  },
  {
    id: "caustics",
    band: "output",
    side: "right",
    label: "Caustic map",
    stage: "caustics",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Projects the surface mesh onto the caustic receiver. A retained surface deposits the same bundles, so the pass is skipped unless the mesh moved, and a scene that authors zero caustic strength never encodes it at all. That retention is why this row is usually near zero already: it costs a full pass on the frames the mesh moves and nothing on the frames between.",
      gate: "a scene with fluid and non-zero authored caustic strength",
    },
    state: (context) => (context.disabledStages.has("caustics") ? "off"
      : context.sceneHasFluid ? "on" : "unavailable"),
    chip: (context) => (context.disabledStages.has("caustics") ? "withheld · map retained"
      : context.sceneHasFluid ? "on surface change" : "dry scene"),
  },
  {
    id: "optical-composite",
    band: "output",
    side: "left",
    label: "Layered optical composite",
    stage: "optical-composite",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Resolves the layered water optics over the dry scene and applies the scene's display grade — exposure, tone curve and white balance — as the single final transform shared by every render path. Off keeps the clear and drops the draw, so the presentation target is the background colour: everything above still runs and nothing of it reaches the screen.",
      reads: "dry scene HDR · interface layers · caustic map",
      writes: "presentation target",
    },
    state: (context) => (context.disabledStages.has("optical-composite") ? "off" : "on"),
    chip: (context) => (context.disabledStages.has("optical-composite")
      ? "withheld · clear only" : "1 draw · HDR → display"),
  },
  {
    id: "inspection-overlays",
    band: "output",
    side: "right",
    label: "Inspection overlays",
    stage: "inspection-overlays",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Stage-plane decodes, field overlays and the ray/cell probe decorations, drawn after the composite. Strictly read-only, so the frame an overlay explains is the frame that shipped. Off withholds all of them at once — including the ◨ taps and the probes below — which is the honest way to price a measurement against the frame it is measuring.",
      gate: "a stage tap, a grid overlay, or a probe",
    },
    state: (context) => (context.disabledStages.has("inspection-overlays") ? "off"
      : context.stageView !== "off" ? "on" : "armed"),
    chip: (context) => (context.disabledStages.has("inspection-overlays") ? "withheld · none drawn"
      : context.stageView !== "off" ? `plane · ${context.stageView}` : "stage · grid · probes"),
  },
  {
    id: "present",
    band: "output",
    side: "left",
    label: "Upscale + present",
    stage: "present",
    taps: [],
    toggleable: true,
    tip: {
      summary: "Blits the internal target to the swap chain. Render resolution lives here because it is the one dial that moves every pass above at once — the frame costs roughly 19–22 ms per megapixel with cones and 13–15 without. Off still acquires and clears the swap-chain texture and drops only the blit, so the canvas goes to the background colour instead of freezing on a frame that would make a withheld blit look free.",
      writes: "swap chain",
    },
    state: (context) => (context.disabledStages.has("present") ? "off" : "on"),
    chip: (context) => {
      if (context.disabledStages.has("present")) return "withheld · canvas cleared";
      const scale = Math.round(context.tuning.resolutionScale * 100);
      const width = context.presentationWidth;
      const height = context.presentationHeight;
      return width && height ? `${scale}% → ${width}×${height}` : `${scale}% of target`;
    },
  },
];

export interface RenderPipelineNode extends RenderPipelineNodeDefinition {
  /** The stages this node owns, in encode order. Summed for its measured cost. */
  readonly stages: readonly RenderFrameStageId[];
}

export const RENDER_PIPELINE_NODES: readonly RenderPipelineNode[] = Object.freeze(
  NODES.map((node) => Object.freeze({ ...node, stages: renderPipelineNodeStages(node.id) })));

/** A stage's trace label back to its stage id. Derived, so it cannot drift. */
const STAGE_BY_TRACE_LABEL: ReadonlyMap<string, RenderFrameStageId> = new Map(
  RENDER_FRAME_STAGES.map((stage) => [RENDER_FRAME_STAGE_TRACE[stage].phase.label, stage]));

/**
 * Measured milliseconds per *stage*, summed across a whole averaged trace.
 *
 * Summed rather than replaced: a stage can close its seam more than once in a
 * frame — the scene-primitive tier does — and the stage owns all of it. Phases
 * whose label names no stage are ignored here and reported separately by
 * {@link renderPipelineUnownedPhases}, because a phase nothing owns is a
 * measurement bug and must be visible as one rather than silently dropped.
 */
export function renderPipelineStageDurations(
  trace: PerformanceTrace | undefined,
): ReadonlyMap<RenderFrameStageId, number> {
  const durations = new Map<RenderFrameStageId, number>();
  if (!trace) return durations;
  for (const phase of trace.phases) {
    const stage = STAGE_BY_TRACE_LABEL.get(phase.label);
    if (stage === undefined) continue;
    durations.set(stage, (durations.get(stage) ?? 0) + phase.duration_ms);
  }
  return durations;
}

/** Trace labels in a captured partition that belong to no stage of the ABI. */
export function renderPipelineUnownedPhases(trace: PerformanceTrace | undefined): readonly string[] {
  if (!trace) return [];
  return [...new Set(trace.phases
    .filter((phase) => !STAGE_BY_TRACE_LABEL.has(phase.label))
    .map((phase) => phase.label))];
}

/**
 * Why a node shows the number it shows.
 *
 * The distinction that matters is between the three ways a row can have no
 * per-pass figure, which used to be one em dash:
 *
 * - `withheld`   the stage encoded no pass. A true zero, not a gap. This is
 *                what the manifest is for, and it is the one state that
 *                *disqualifies* a row from absorbing a band's wall residual.
 * - `unpriced`   the stage encoded render passes, whose timestamp pairs are
 *                tiler windows rather than costs on this hardware. The only
 *                honest figure for these is a fence-partitioned band wall.
 * - `unmeasured` no trace has arrived yet, or the partition is too coarse to
 *                name this node.
 *
 * and beside them:
 *
 * - `measured`   summed from the stages this node owns in the averaged trace.
 * - `shared`     a term inside another node's pass; the figure is the host's.
 * - `structural` a gate or a decision that never spends frame time either way.
 * - `wall`       derived from the band's fence-partitioned wall: this row is
 *                the band's only `unpriced` row, so the band wall minus the
 *                band's measured compute lands at its junction.
 */
export type RenderPipelineCostKind =
  | "measured" | "withheld" | "shared" | "unpriced" | "structural" | "unmeasured" | "wall";

export interface RenderPipelineMeasurement {
  readonly kind: RenderPipelineCostKind;
  /** Undefined only when `kind` is `unmeasured` or `unpriced`. */
  readonly duration_ms?: number;
  /** Fraction of the frame, for the node's bar. */
  readonly share: number;
  /** For `shared`, the node whose pass carries this one's work. */
  readonly insideNode?: string;
  /** For `measured` and `unpriced`: render passes this row owns that no timestamp can price. */
  readonly unpricedRenderPasses?: number;
}

const UNMEASURED: RenderPipelineMeasurement = Object.freeze({ kind: "unmeasured", share: 0 });

/**
 * What the frame encoded, as the panel asks it: per stage, and only the two
 * questions a row's cost depends on.
 */
export interface RenderPipelineEncoding {
  readonly encoded: boolean;
  readonly renderPasses: number;
}

export function renderPipelineEncodings(
  manifest: RenderFrameManifest | undefined,
): ReadonlyMap<RenderFrameStageId, RenderPipelineEncoding> | undefined {
  if (!manifest) return undefined;
  return new Map(manifest.stages.map((entry) => [entry.stage, {
    encoded: entry.computePasses + entry.renderPasses > 0,
    renderPasses: entry.renderPasses,
  }]));
}

function measureStages(
  stages: readonly RenderFrameStageId[],
  durations: ReadonlyMap<RenderFrameStageId, number>,
  encodings: ReadonlyMap<RenderFrameStageId, RenderPipelineEncoding> | undefined,
  total_ms: number,
): RenderPipelineMeasurement {
  if (stages.length === 0) return UNMEASURED;
  let measured = false;
  let duration_ms = 0;
  let renderPasses = 0;
  let encodedAnything = false;
  for (const stage of stages) {
    const value = durations.get(stage);
    if (value !== undefined) {
      measured = true;
      duration_ms += value;
    }
    const encoding = encodings?.get(stage);
    if (encoding?.encoded) encodedAnything = true;
    renderPasses += encoding?.renderPasses ?? 0;
  }
  // The manifest decides first. A stage the frame reached and encoded nothing
  // into is zero however loud the band around it was, and saying so is the
  // difference between this panel and the one that reported 27.9 ms of world
  // build on a world that did not move.
  if (encodings && !encodedAnything) return { kind: "withheld", duration_ms: 0, share: 0 };
  if (measured && renderPasses === 0) {
    return { kind: "measured", duration_ms, share: total_ms > 0 ? Math.min(1, duration_ms / total_ms) : 0 };
  }
  if (renderPasses > 0) {
    return measured
      // Part of this row is compute and priced, part is render and is not. The
      // figure shown is the part that is real; the count says how much of the
      // row it leaves out.
      ? { kind: "measured", duration_ms, share: total_ms > 0 ? Math.min(1, duration_ms / total_ms) : 0, unpricedRenderPasses: renderPasses }
      : { kind: "unpriced", share: 0, unpricedRenderPasses: renderPasses };
  }
  if (measured) {
    return { kind: "measured", duration_ms, share: total_ms > 0 ? Math.min(1, duration_ms / total_ms) : 0 };
  }
  return UNMEASURED;
}

/**
 * What this node cost the frame, and on what basis.
 *
 * `state` is an input rather than something re-derived here because it already
 * encodes the whole answer for a node the user switched off. Everything else
 * now comes from the frame's own manifest, which is why a stage that encoded
 * nothing no longer depends on the panel guessing from a predicate.
 */
export function measureRenderPipelineNode(
  node: RenderPipelineNode,
  durations: ReadonlyMap<RenderFrameStageId, number>,
  total_ms: number,
  state: RenderPipelineNodeState,
  encodings?: ReadonlyMap<RenderFrameStageId, RenderPipelineEncoding>,
): RenderPipelineMeasurement {
  if (node.spendsNoFrameTime) return { kind: "structural", duration_ms: 0, share: 0 };
  const own = measureStages(node.stages, durations, encodings, total_ms);
  if (own.kind !== "unmeasured") return own;
  // A node the frame did not encode is zero, and saying so is the point of
  // having a switch. Claiming a measured zero without a trace is not, so this
  // still requires evidence to have arrived — either a manifest or a partition.
  if ((state === "off" || state === "unavailable") && (encodings !== undefined || durations.size > 0)) {
    return { kind: "withheld", duration_ms: 0, share: 0 };
  }
  if (node.costInsideNode) {
    const host = RENDER_PIPELINE_NODES.find((candidate) => candidate.id === node.costInsideNode);
    const hostCost = host ? measureStages(host.stages, durations, encodings, total_ms) : UNMEASURED;
    if (hostCost.kind === "measured") {
      return { ...hostCost, kind: "shared", insideNode: node.costInsideNode };
    }
  }
  return UNMEASURED;
}

/** Band totals, so a collar can price the section it heads. */
export function measureRenderPipelineBand(
  band: RenderPipelineBandId,
  durations: ReadonlyMap<RenderFrameStageId, number>,
  total_ms: number,
  encodings?: ReadonlyMap<RenderFrameStageId, RenderPipelineEncoding>,
): RenderPipelineMeasurement {
  // Own stages only. A `costInsideNode` node's figure is its host's, so summing
  // it into the band would count one pass twice and make the bands overrun the
  // frame they are shares of.
  return measureStages(
    RENDER_PIPELINE_NODES.filter((node) => node.band === band).flatMap((node) => node.stages),
    durations,
    encodings,
    total_ms,
  );
}

/** The hover tip, as one string. Every word of prose in the panel comes through here. */
export function renderPipelineTipText(node: RenderPipelineNode, chip: string): string {
  const { tip } = node;
  const lines = [`${node.label} · ${chip}`, "", tip.summary];
  if (tip.reads) lines.push("", `READS   ${tip.reads}`);
  if (tip.writes) lines.push(tip.reads ? `WRITES  ${tip.writes}` : `\nWRITES  ${tip.writes}`);
  if (tip.feeds) lines.push(`FEEDS   ${tip.feeds}`);
  if (tip.gate) lines.push(`GATE    ${tip.gate}`);
  return lines.join("\n");
}
