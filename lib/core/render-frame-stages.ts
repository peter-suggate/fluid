/**
 * The presentation frame's stage ABI — one id per seam the encoders close.
 *
 * This is the render path's counterpart to `SPARSE_CM12_RESIDENT_STAGES`: the
 * encoders name a *stage id* at each seam and nothing else, and everything the
 * panel says about a stage — its label, its trace phase, its band, its prose,
 * its switch — is looked up from that id. Before this, each seam carried its
 * own hand-written `{ id, label }` literal and the panel matched on the label
 * string, so a renamed pass silently detached its row from its cost and the
 * row went on reporting *something* — whatever the band's residual happened to
 * be. Naming the stage instead makes that class of drift a type error.
 *
 * ## The fact a stage publishes
 *
 * Its seam closes the stage in both the hardware timestamp chain and a paired
 * CPU command-encoding chain. Adjacent boundaries define exclusive intervals,
 * so every row and band is a direct slice of whichever one exact partition the
 * panel displays. Metal cannot reliably order timestamps across mixed render
 * and compute work; in that case FRAME follows SIM and displays the matching
 * CPU chain rather than combining unrelated instruments.
 *
 * ## Ownership
 *
 * Four encoders close seams, and each may close only its own stages —
 * {@link RenderFrameSeam} is parameterised by owner, so the water pipeline
 * cannot close an SVO stage by accident any more than it can misspell one.
 *
 * | Owner      | Encoder                                            |
 * |------------|----------------------------------------------------|
 * | `world`    | `webgpu-svo-sparse-bricks.ts` scene maintenance     |
 * | `renderer` | `webgpu-renderer.ts` source band and frame tail     |
 * | `water`    | `webgpu-water-pipeline.ts`                         |
 * | `svo`      | `webgpu-svo-dry-scene.ts`                          |
 *
 * This module is a leaf: types plus two frozen tables and one recorder, no
 * imports from any encoder, so every encoder may import it and the panel may
 * import it beside them.
 */

import type { GPUTimestampPhase } from "./performance-trace";

/** Which encoder closes a stage's seam. */
export type RenderFrameStageOwner = "world" | "renderer" | "water" | "svo";

/**
 * Every stage of the presentation frame, in encode order.
 *
 * The order is the order the seams close on a full SVO frame; a frame that
 * takes another arm closes a subsequence of it, never a permutation, and
 * `tests/render-frame-stage-partition.test.ts` holds that.
 */
export const RENDER_FRAME_STAGES = Object.freeze([
  // SOURCE — the world the rest of the frame marches, maintained in place.
  "world-topology-publish",
  "world-proxy-voxelize",
  "world-derived-lighting",
  "world-radiance-feedback",
  "rigid-pose-mirror",
  "fluid-coverage",
  // WATER SURFACE — the isosurface and what is projected from it.
  "surface-extraction",
  "caustics",
  // PRIMARY VISIBILITY — who is in front of whom.
  "scene-primitive-visibility",
  "near-field-band",
  "brick-cull",
  "cone-prepass",
  "primary-entry-prepass",
  "primary-traversal",
  "rigid-discovery",
  "thin-glass-discovery",
  "seam-closure",
  // LIGHTING VISIBILITY — what can see which light.
  "voxel-light-cache",
  "compact-cone-lighting",
  "cone-fanout",
  "world-gi-cache",
  "reduced-shade",
  // SHADING — the depth partition's two halves, and the inline arm's fusion.
  "sky-lighting",
  "deferred-lighting",
  "inline-traversal-shading",
  // OUTPUT — the fluid-only fallbacks, the interfaces, the composite, present.
  "fluid-only-rigid-bodies",
  "fluid-only-background",
  "dry-scene-unavailable",
  "water-front-interface",
  "water-back-interface",
  "water-rear-interfaces",
  "optical-composite",
  "inspection-overlays",
  "present",
] as const);

export type RenderFrameStageId = (typeof RENDER_FRAME_STAGES)[number];

/** The frame-panel row a stage contributes its timing to. */
export type RenderPipelineNodeId =
  | "sparse-world-build"
  | "derived-lighting"
  | "rigid-pose-mirror"
  | "fluid-coverage"
  | "primary-entry-prepass"
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
 * One timing plugin shared by the encoder seam and the frame-panel row.
 * Keeping ownership, trace identity and display destination together is the
 * presentation equivalent of the SIM stage registry: adding a timed stage
 * cannot leave its measurement floating outside the graph.
 */
export interface RenderFrameStagePlugin {
  readonly owner: RenderFrameStageOwner;
  readonly node: RenderPipelineNodeId;
  /** The trace phase this stage's seam closes under. */
  readonly phase: GPUTimestampPhase;
}

/**
 * Each stage's encoder owner, panel row and trace phase.
 *
 * The label here is the only place a stage's trace label is written down. It
 * is what appears in a captured `PerformanceTrace`, so it stays close to the
 * strings the previous hand-written seams published — a stored capture and a
 * live frame still speak the same language — but nothing matches on it any
 * more.
 *
 * Several stages share a `PaperPhaseId`: five share `svo-primary` and four
 * share `svo-cone-lighting`, which is exactly why an id-keyed lookup was never
 * the finest grain available and why the stage id is a separate thing.
 */
export const RENDER_FRAME_STAGE_PLUGINS = Object.freeze({
  "world-topology-publish": {
    owner: "world",
    node: "sparse-world-build",
    phase: { id: "scene-upload", label: "Sparse world topology publish" },
  },
  "world-proxy-voxelize": {
    owner: "world",
    node: "sparse-world-build",
    phase: { id: "scene-upload", label: "Sparse world proxy voxelization" },
  },
  "world-derived-lighting": {
    owner: "world",
    node: "derived-lighting",
    phase: { id: "scene-upload", label: "Sparse world derived lighting" },
  },
  "world-radiance-feedback": {
    owner: "world",
    node: "derived-lighting",
    phase: { id: "scene-upload", label: "Sparse world radiance feedback" },
  },
  "rigid-pose-mirror": {
    owner: "renderer",
    node: "rigid-pose-mirror",
    phase: { id: "scene-upload", label: "Rigid pose mirror + readback" },
  },
  "fluid-coverage": {
    owner: "renderer",
    node: "fluid-coverage",
    phase: { id: "scene-upload", label: "SVO fluid coverage" },
  },
  "surface-extraction": {
    owner: "water",
    node: "surface-extraction",
    phase: { id: "surface-extraction", label: "Water surface extraction" },
  },
  caustics: {
    owner: "water",
    node: "caustics",
    phase: { id: "water-caustics", label: "Water caustic map" },
  },
  "scene-primitive-visibility": {
    owner: "svo",
    node: "scene-primitive",
    phase: { id: "svo-scene-primitive", label: "SVO exact live-scene primitive visibility" },
  },
  "near-field-band": {
    owner: "svo",
    node: "scene-primitive",
    phase: { id: "svo-band", label: "SVO near-field analytic band selection" },
  },
  "brick-cull": {
    owner: "svo",
    node: "primary-traversal",
    phase: { id: "svo-brick-cull", label: "SVO brick instance cull" },
  },
  "cone-prepass": {
    owner: "svo",
    node: "cone-visibility",
    phase: { id: "svo-cone-lighting", label: "SVO cone-lighting prepass" },
  },
  "primary-entry-prepass": {
    owner: "svo",
    node: "primary-entry-prepass",
    phase: { id: "svo-primary", label: "SVO primary entry-depth prepass" },
  },
  "primary-traversal": {
    owner: "svo",
    node: "primary-traversal",
    phase: { id: "svo-primary", label: "SVO primary visibility" },
  },
  "rigid-discovery": {
    owner: "svo",
    node: "rigid-impostor",
    phase: { id: "svo-rigid", label: "SVO analytic rigid discovery" },
  },
  "thin-glass-discovery": {
    owner: "svo",
    node: "thin-glass",
    phase: { id: "svo-glass", label: "SVO raster thin-glass discovery" },
  },
  "seam-closure": {
    owner: "svo",
    node: "seam-closure",
    phase: { id: "svo-primary", label: "SVO primary seam closure" },
  },
  "voxel-light-cache": {
    owner: "svo",
    node: "voxel-light-cache",
    phase: { id: "svo-voxel-light", label: "SVO voxel light cache" },
  },
  "compact-cone-lighting": {
    owner: "svo",
    node: "cone-visibility",
    phase: { id: "svo-cone-lighting", label: "SVO compacted cone lighting" },
  },
  "cone-fanout": {
    owner: "svo",
    node: "cone-visibility",
    phase: { id: "svo-cone-lighting", label: "SVO cone sample fan-out" },
  },
  "world-gi-cache": {
    owner: "svo",
    node: "world-gi-cache",
    phase: { id: "svo-environment-gi", label: "SVO persistent world-space environmental GI" },
  },
  "reduced-shade": {
    owner: "svo",
    node: "reduced-shade",
    phase: { id: "svo-cone-lighting", label: "SVO reduced-rate opaque shading" },
  },
  "sky-lighting": {
    owner: "svo",
    node: "sky-lighting",
    phase: { id: "dry-scene", label: "SVO deferred sky lighting" },
  },
  "deferred-lighting": {
    owner: "svo",
    node: "deferred-lighting",
    phase: { id: "dry-scene", label: "SVO deferred dry lighting" },
  },
  "inline-traversal-shading": {
    owner: "svo",
    node: "primary-traversal",
    phase: { id: "svo-primary", label: "SVO traversal + dry shading" },
  },
  "fluid-only-rigid-bodies": {
    owner: "water",
    node: "deferred-lighting",
    phase: { id: "dry-scene", label: "Fluid-only rigid bodies" },
  },
  "fluid-only-background": {
    owner: "water",
    node: "deferred-lighting",
    phase: { id: "dry-scene", label: "Fluid-only background clear" },
  },
  "dry-scene-unavailable": {
    owner: "water",
    node: "deferred-lighting",
    phase: { id: "dry-scene", label: "SVO dry-scene unavailable · fail closed" },
  },
  "water-front-interface": {
    owner: "water",
    node: "water-interfaces",
    phase: { id: "water-front-interface", label: "Water + spray front interface" },
  },
  "water-back-interface": {
    owner: "water",
    node: "water-interfaces",
    phase: { id: "water-back-interface", label: "Water + spray back interface" },
  },
  "water-rear-interfaces": {
    owner: "water",
    node: "water-interfaces",
    phase: { id: "water-interfaces", label: "Water rear interfaces" },
  },
  "optical-composite": {
    owner: "water",
    node: "optical-composite",
    phase: { id: "optical-composite", label: "Layered optical composite" },
  },
  "inspection-overlays": {
    owner: "renderer",
    node: "inspection-overlays",
    phase: { id: "inspection-overlay", label: "Inspection overlays" },
  },
  present: {
    owner: "renderer",
    node: "present",
    phase: { id: "present", label: "Final upscale + present" },
  },
} as const satisfies Record<RenderFrameStageId, RenderFrameStagePlugin>);

/** The stages one encoder owns. A seam typed to it can close no other. */
export type RenderFrameStageOf<Owner extends RenderFrameStageOwner> = {
  [Stage in RenderFrameStageId]:
  (typeof RENDER_FRAME_STAGE_PLUGINS)[Stage]["owner"] extends Owner ? Stage : never;
}[RenderFrameStageId];

/**
 * What an encoder is handed to close its own seams.
 *
 * Optional at every call site, exactly as the `tracePhase` callback it
 * replaces was: an untraced frame passes nothing and encodes the same command
 * graph.
 */
export type RenderFrameSeam<Owner extends RenderFrameStageOwner = RenderFrameStageOwner> =
  (stage: RenderFrameStageOf<Owner>) => void;

/** Where a stage's cost is displayed. Bands are the panel's collars. */
export type RenderFrameBandId = "source" | "primary" | "lighting" | "shading" | "output";

/** What one stage encoded in one frame. */
export interface RenderFrameStageEncoding {
  readonly stage: RenderFrameStageId;
  /** Compute passes closed under this seam. These are the ones with a trustworthy cost. */
  readonly computePasses: number;
  /** Render passes closed under this seam. Their timestamp pairs are tiler windows. */
  readonly renderPasses: number;
}

/**
 * What the frame encoded, stage by stage.
 *
 * `unclaimed` is the count of passes encoded after the final seam — passes no
 * stage owns. It is zero on a healthy frame and is the invariant the partition
 * test asserts; a non-zero value here means some pass chain is being charged
 * to nothing rather than to a row.
 */
export interface RenderFrameManifest {
  readonly stages: readonly RenderFrameStageEncoding[];
  readonly unclaimed: number;
}

/**
 * Counts the passes each seam closes over.
 *
 * The count is what separates "this stage did nothing" from "this stage's
 * passes cannot be timed", and only the encoder can answer it: by the time a
 * timestamp reading is decoded, an unsampled pass and an absent pass look the
 * same. Instrumenting is a `Proxy` over `beginComputePass`/`beginRenderPass`
 * and two integer increments per pass, and it is installed only while
 * measurement is on.
 *
 * Every encoder the frame writes into must be instrumented, including the ones
 * a fence-partitioned sampling frame creates after each submit — see
 * `FencePartitionedFrameSampler`, which takes this recorder's `instrument` so
 * its successors are counted too.
 */
export class RenderFrameSeamRecorder {
  private computePasses = 0;
  private renderPasses = 0;
  private readonly encodings: RenderFrameStageEncoding[] = [];

  /** Count every pass opened through this encoder against the open seam. */
  instrument(encoder: GPUCommandEncoder): GPUCommandEncoder {
    return new Proxy(encoder, {
      get: (target, property) => {
        if (property === "beginComputePass") {
          return (descriptor?: GPUComputePassDescriptor) => {
            this.computePasses += 1;
            return target.beginComputePass(descriptor);
          };
        }
        if (property === "beginRenderPass") {
          return (descriptor: GPURenderPassDescriptor) => {
            this.renderPasses += 1;
            return target.beginRenderPass(descriptor);
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUCommandEncoder;
  }

  /** Close the named stage over every pass encoded since the previous seam. */
  close(stage: RenderFrameStageId): void {
    this.encodings.push({
      stage,
      computePasses: this.computePasses,
      renderPasses: this.renderPasses,
    });
    this.computePasses = 0;
    this.renderPasses = 0;
  }

  /**
   * The frame's manifest, with a stage closed more than once summed into one
   * entry. Stages keep their first-closed order, which is encode order.
   */
  manifest(): RenderFrameManifest {
    const merged = new Map<RenderFrameStageId, RenderFrameStageEncoding>();
    for (const encoding of this.encodings) {
      const existing = merged.get(encoding.stage);
      merged.set(encoding.stage, existing
        ? {
          stage: encoding.stage,
          computePasses: existing.computePasses + encoding.computePasses,
          renderPasses: existing.renderPasses + encoding.renderPasses,
        }
        : encoding);
    }
    return {
      stages: [...merged.values()],
      unclaimed: this.computePasses + this.renderPasses,
    };
  }
}

/** Merge per-stage encodings from several frames, keeping the maximum seen. */
export function mergeRenderFrameManifests(
  manifests: readonly RenderFrameManifest[],
): RenderFrameManifest | undefined {
  if (manifests.length === 0) return undefined;
  const merged = new Map<RenderFrameStageId, RenderFrameStageEncoding>();
  let unclaimed = 0;
  for (const manifest of manifests) {
    unclaimed = Math.max(unclaimed, manifest.unclaimed);
    for (const encoding of manifest.stages) {
      const existing = merged.get(encoding.stage);
      // The maximum rather than the mean: the question a row asks of the
      // manifest is "does this stage ever encode a pass", and a stage that
      // encodes on one frame in twelve must not average its way to zero and
      // become eligible for a band's whole wall.
      merged.set(encoding.stage, existing
        ? {
          stage: encoding.stage,
          computePasses: Math.max(existing.computePasses, encoding.computePasses),
          renderPasses: Math.max(existing.renderPasses, encoding.renderPasses),
        }
        : encoding);
    }
  }
  return { stages: [...merged.values()], unclaimed };
}
