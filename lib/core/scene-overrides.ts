import {
  COMPARE_ABSENT,
  COMPARE_ACTIVE_KEY,
  COMPARE_KEY_PREFIX,
  COMPARE_LINK_KEY,
  isCompareQueryKey,
} from "./compare/compare-query";
import { withRefinementRegionsFromQuery, refinementRegionsToQuery } from "./editor-refinement-region";
import { sceneSeedsQuery, withSceneSeedsFromQuery } from "./initial-brick-seed-query";
import { defaultMethodId, getMethod, registeredSimulationMethods } from "./method-registry";
import type { MethodParamValue } from "./method-contract";
import type { SceneDescription } from "./model";
import { cameraForPreset, findSceneDefinition, getScenePreset } from "./scenes";
import { useUIStore } from "./stores/ui-store";
import { sceneStoneQuery, withSceneStoneQuery } from "./stone-look-controls";
import type { SvoRenderTuning } from "../svo/svo-render-tuning";
import type { GPUQuality } from "./gpu-quality";
import { sceneCanopyQuery, withSceneCanopyQuery } from "./tree-canopy-controls";
import { sceneQueryPathValue, withSceneQueryPathValue } from "./url-state";
import { sceneRimQuery, withSceneRimQuery } from "./vessel-rim-controls";

/**
 * What a scene is carrying that its author did not put there.
 *
 * The address bar is the studio's whole persistence layer, and
 * `serializeQueryState` already writes exactly one key per value that differs
 * from the scene's authored baseline — a diff, not a document. So the honest
 * definition of "this scene has been overridden" is "its canonical query has
 * keys beyond its identity", and that is what this module reads. Deriving the
 * list from the serializer rather than re-comparing the stores means the badge
 * cannot disagree with the link: anything shareable is listed, and anything
 * listed is clearable.
 *
 * Five groups, because they are cleared by five different mechanisms and answer
 * different questions:
 *  - `scene`   document values patched over the preset's own factory output,
 *  - `method`  the solver, its quality, and per-key parameter tuning,
 *  - `render`  presentation and debug arms that change what the frame shows,
 *  - `view`    the camera and the open scene instrument — in the link so a
 *              shared link reopens the view that was shared, not because
 *              anything is overridden, so they are listed but never counted,
 *  - `link`    keys this build does not manage, of which `gpu` is the one that
 *              matters: it is read once during module startup, so clearing it
 *              is a navigation rather than a store edit,
 *  - `compare` the `b.*` block: pane B's diff. Listed under its own heading and
 *              never counted, because it is not something done to *this* scene —
 *              it is the second pane, and the second pane is the whole point of
 *              the link it appears in.
 */
export type SceneOverrideGroup = "scene" | "method" | "render" | "view" | "link" | "compare";

export interface SceneOverride {
  /** Every query key this row owns; clearing the row clears all of them. */
  readonly keys: readonly string[];
  readonly group: SceneOverrideGroup;
  readonly label: string;
  /** The overriding value, already shortened for a chip-width readout. */
  readonly value: string;
  readonly hint?: string;
  /** False for rows that are in the link without being an override. */
  readonly counted: boolean;
  /** A startup flag cannot be un-read; clearing it has to reload the page. */
  readonly clearedBy: "stores" | "reload";
}

/** Keys that name *which* scene this is, rather than anything done to it. */
const IDENTITY_KEYS = new Set(["scene", "view"]);

/**
 * The scene layers that ride their own key instead of a `scene.*` path,
 * because they are container-relative or too large to patch — see the block
 * comments on `REGIONS_QUERY_KEY` and friends in `url-state`.
 */
const SCENE_LAYER_KEYS: Readonly<Record<string, { readonly label: string; readonly hint: string }>> = {
  regions: { label: "Refinement regions", hint: "Drawn refinement boxes, as percentages of the container" },
  canopy: { label: "Tree canopy", hint: "Canopy art-direction dials" },
  stones: { label: "Stone looks", hint: "Stone look dials and seeds" },
  rim: { label: "Coping rim", hint: "Vessel rim dials" },
  seeds: { label: "Painted water", hint: "Seeded brick occupancy, as a bitset over the bricks it wets" },
};

interface UIOverrideSpec {
  readonly label: string;
  readonly group: Exclude<SceneOverrideGroup, "scene" | "method" | "link">;
  readonly hint?: string;
  /** Rendered as on/off rather than as the 1/0 the key actually carries. */
  readonly boolean?: boolean;
  readonly counted?: boolean;
}

const UI_OVERRIDES: Readonly<Record<string, UIOverrideSpec>> = {
  grid: { label: "Grid overlay", group: "render", hint: "Slice axis of the field overlay" },
  gridSlice: { label: "Grid slice", group: "render" },
  gridMode: { label: "Overlay field", group: "render" },
  lensPhase: { label: "Lens phase", group: "render", hint: "Scrubber position within the open stage lens" },
  svoShadows: { label: "Shadows", group: "render", boolean: true },
  svoAO: { label: "Ambient occlusion", group: "render", boolean: true },
  svoPrimarySeamClosure: { label: "Silhouette refinement", group: "render", boolean: true },
  svoCones: { label: "Cone tracing", group: "render" },
  svoPrimary: { label: "Primary traversal", group: "render" },
  svoStage: { label: "Stage view", group: "render", hint: "Debug stage isolation, not a product frame" },
  svoFlatExempt: { label: "Planar refinement exemption", group: "render", boolean: true },
  svoLodPixels: { label: "LOD threshold", group: "render", hint: "Screen-space pixels before a finer leaf is descended into" },
  // Listed but never counted, on the camera's contract: an open instrument is
  // in the link so a shared link reopens the view that was shared, not because
  // anything about the scene has been overridden.
  overlay: { label: "Open instrument", group: "view", counted: false, hint: "The pipeline or diagnostics reading drawn over the scene" },
};

/** Startup flags read outside the store chassis, named so they read as English. */
const LINK_OVERRIDES: Readonly<Record<string, { readonly label: string; readonly hint: string }>> = {
  gpu: { label: "GPU startup mode", hint: "Read once when the page loads; clearing it reloads" },
  gpuRecovery: { label: "Automatic GPU recovery", hint: "Read once when the page loads; clearing it reloads" },
};

const GROUP_ORDER: readonly SceneOverrideGroup[] = ["scene", "method", "render", "view", "compare", "link"];

function shortValue(raw: string, boolish = false) {
  if (boolish) return raw === "1" ? "on" : raw === "0" ? "off" : raw;
  if (raw === "") return "none";
  return raw.length > 46 ? `${raw.slice(0, 45)}…` : raw;
}

function methodParamOverride(key: string, raw: string): SceneOverride | undefined {
  // `param.<methodId>.<paramKey>`; a method id never contains a dot, so the
  // remainder after the second segment is the parameter key.
  const [, methodId, ...rest] = key.split(".");
  const paramKey = rest.join(".");
  const method = registeredSimulationMethods().find((candidate) => candidate.id === methodId);
  const spec = method?.params.find((candidate) => candidate.key === paramKey);
  if (!method || !spec) return undefined;
  return {
    keys: [key],
    group: "method",
    label: `${method.shortLabel} · ${spec.label}`,
    value: shortValue(raw),
    hint: spec.hint,
    counted: true,
    clearedBy: "stores",
  };
}

export interface SceneOverrideContext {
  /**
   * False for a document with no authored baseline to diff against — a saved
   * scene from a build that no longer has its origin, or a blank starter.
   * `getScenePreset` falls back to the first catalog entry for those, so every
   * `scene.*` key in their query is a comparison against an unrelated scene and
   * would read as dozens of overrides nobody made.
   */
  readonly hasPresetBaseline: boolean;
}

/**
 * Classify one canonical studio query into the overrides it carries.
 *
 * Takes the serialized query rather than the stores so that the list is exactly
 * what the link contains, including keys this build does not manage.
 */
export function sceneOverridesInQuery(search: string, context: SceneOverrideContext): readonly SceneOverride[] {
  const query = new URLSearchParams(search);
  const overrides: SceneOverride[] = [];
  const cameraKeys: string[] = [];

  for (const [key, raw] of query) {
    if (IDENTITY_KEYS.has(key)) continue;
    if (key.startsWith("camera.")) { cameraKeys.push(key); continue; }

    // Pane B, before anything else claims one of its keys: `b.scene.…` and
    // `b.param.…` would otherwise be read as edits to pane A's own document.
    if (isCompareQueryKey(key)) {
      overrides.push({
        keys: [key], group: "compare",
        label: key === COMPARE_ACTIVE_KEY ? "Second pane"
          : key === COMPARE_LINK_KEY ? "Unlinked groups"
          : key.slice(COMPARE_KEY_PREFIX.length),
        value: shortValue(raw === COMPARE_ABSENT ? "not set in B" : raw),
        hint: key === COMPARE_ACTIVE_KEY
          ? "Compare mode is open with an empty diff — pane B is pane A"
          : key === COMPARE_LINK_KEY
            ? "Groups a reader unlinked, so the two panes may differ in them"
            : "Pane B's value for this key; pane A keeps its own",
        counted: false, clearedBy: "stores",
      });
      continue;
    }

    if (key.startsWith("scene.")) {
      if (!context.hasPresetBaseline) continue;
      overrides.push({
        keys: [key], group: "scene", label: key.slice("scene.".length),
        value: shortValue(raw === "~delete" ? "removed" : raw),
        hint: "Patched over the value this scene's factory produced",
        counted: true, clearedBy: "stores",
      });
      continue;
    }

    const layer = SCENE_LAYER_KEYS[key];
    if (layer) {
      if (!context.hasPresetBaseline) continue;
      overrides.push({
        keys: [key], group: "scene", label: layer.label, value: shortValue(raw),
        hint: layer.hint, counted: true, clearedBy: "stores",
      });
      continue;
    }

    if (key === "method" || key === "quality") {
      overrides.push({
        keys: [key], group: "method",
        label: key === "method" ? "Solver" : "Quality",
        value: key === "method" ? getMethod(raw).shortLabel : raw,
        counted: true, clearedBy: "stores",
      });
      continue;
    }

    if (key.startsWith("param.")) {
      // A key naming a parameter this build retired is link cruft, not tuning:
      // nothing consumes it, so it falls through to the `link` group below
      // where it can still be cleared.
      const param = methodParamOverride(key, raw);
      if (param) { overrides.push(param); continue; }
    }

    const ui = UI_OVERRIDES[key];
    if (ui) {
      overrides.push({
        keys: [key], group: ui.group, label: ui.label,
        value: shortValue(raw, ui.boolean), hint: ui.hint,
        counted: ui.counted ?? true, clearedBy: "stores",
      });
      continue;
    }

    const link = LINK_OVERRIDES[key];
    overrides.push({
      keys: [key], group: "link", label: link?.label ?? key,
      value: shortValue(raw), hint: link?.hint ?? "Not a setting this build writes; carried in from the link",
      counted: true, clearedBy: "reload",
    });
  }

  // One row, not six: the camera is a single thing the reader moved, and six
  // rows of radians would bury the values that were actually overridden.
  if (cameraKeys.length > 0 && context.hasPresetBaseline) {
    overrides.push({
      keys: cameraKeys, group: "view", label: "Camera framing",
      value: "orbited", hint: "Restores the framing this scene was authored with",
      counted: false, clearedBy: "stores",
    });
  }

  return overrides.sort((left, right) =>
    GROUP_ORDER.indexOf(left.group) - GROUP_ORDER.indexOf(right.group)
    || left.label.localeCompare(right.label));
}

/** How many of these a reader should be told about; see `counted`. */
export function countedSceneOverrides(overrides: readonly SceneOverride[]) {
  return overrides.filter((override) => override.counted);
}

export interface SceneOverrideClearPlan {
  /** Absent when no scene-document key was among the cleared keys. */
  readonly scene?: SceneDescription;
  readonly methodId?: string;
  readonly quality?: GPUQuality;
  /** `value` absent means the override is simply dropped to the preset chain. */
  readonly methodParams: readonly { readonly methodId: string; readonly key: string; readonly value?: MethodParamValue }[];
  readonly ui: Partial<ReturnType<typeof useUIStore.getState>>;
  /** Merged into the live tuning record rather than replacing it. */
  readonly svoRenderTuning: Partial<SvoRenderTuning>;
  /** Keys that only a reload can retire, in the order they should be removed. */
  readonly reload: readonly string[];
}

/**
 * The inverse of every key in `keys`, resolved against the scene's own preset.
 *
 * Each clear is a *targeted* restore of one value onto the live document, never
 * a re-parse of the reduced query. Re-parsing would be shorter and is wrong:
 * the query is deliberately not a complete document — a sculpted terrain grid
 * and the scenery graph are both too large to serialize — so rebuilding from it
 * would silently discard authoring work that has nothing to do with the
 * override being cleared.
 */
export function sceneOverrideClearPlan(
  keys: readonly string[],
  document: { readonly scene: SceneDescription; readonly presetId: string },
): SceneOverrideClearPlan {
  const cleared = new Set(keys);
  const preset = getScenePreset(document.presetId);
  // Session-invariant: the factory's authored defaults, not a pane's state,
  // so this plan is the same whichever session asked for it.
  const initialUI = useUIStore.getInitialState();
  const methodParams: { methodId: string; key: string; value?: MethodParamValue }[] = [];
  const ui: Partial<ReturnType<typeof useUIStore.getState>> = {};
  const svoRenderTuning: { -readonly [Key in keyof SvoRenderTuning]?: SvoRenderTuning[Key] } = {};
  const reload: string[] = [];

  const touchesScene = [...cleared].some((key) => key.startsWith("scene.") || key in SCENE_LAYER_KEYS);
  // Built once, and only when something actually needs it: a preset factory is
  // not cheap — the hero pond bakes a 289 x 193 heightfield to produce one.
  const baseScene = touchesScene ? preset.create() : undefined;
  let scene = document.scene;
  if (baseScene) {
    for (const key of cleared) {
      if (!key.startsWith("scene.")) continue;
      const path = key.slice("scene.".length);
      // A patched value goes back to the factory's own; note that for
      // `voxelDomain` this restores the number a re-authored document *carries*
      // and does not re-run the factory at that lattice. Re-authoring is its
      // own control in the scene configuration, for the reasons documented
      // there — this is the same edit the finest-cell field makes.
      scene = withSceneQueryPathValue(scene, path, sceneQueryPathValue(baseScene, path));
    }
    if (cleared.has("regions")) scene = withRefinementRegionsFromQuery(scene, refinementRegionsToQuery(baseScene));
    if (cleared.has("canopy")) scene = withSceneCanopyQuery(scene, sceneCanopyQuery(baseScene));
    if (cleared.has("stones")) scene = withSceneStoneQuery(scene, sceneStoneQuery(baseScene));
    if (cleared.has("rim")) scene = withSceneRimQuery(scene, sceneRimQuery(baseScene));
    if (cleared.has("seeds")) scene = withSceneSeedsFromQuery(scene, sceneSeedsQuery(baseScene));
  }

  for (const key of cleared) {
    if (key.startsWith("param.")) {
      const [, methodId, ...rest] = key.split(".");
      const paramKey = rest.join(".");
      if (!registeredSimulationMethods().some((method) => method.id === methodId)) { reload.push(key); continue; }
      // Restoring means the profile's value where the scene authors one, and
      // only otherwise the quality chain: a profiled scene's baseline is its
      // profile, which is what the URL diffed against in the first place.
      const authored = preset.methodProfile?.methodId === methodId
        ? preset.methodProfile.overrides[paramKey] : undefined;
      methodParams.push({ methodId, key: paramKey, value: authored });
      continue;
    }
    if (key.startsWith("camera.")) { ui.camera = cameraForPreset(preset); continue; }
    switch (key) {
      case "overlay": ui.sceneOverlay = initialUI.sceneOverlay; break;
      case "grid": ui.gridOverlayAxis = initialUI.gridOverlayAxis; break;
      case "gridSlice": ui.gridOverlaySlice = initialUI.gridOverlaySlice; break;
      case "gridMode": ui.gridOverlayMode = initialUI.gridOverlayMode; break;
      case "lensPhase": ui.gridOverlayLensPhase = initialUI.gridOverlayLensPhase; break;
      case "svoShadows": ui.svoShadowsEnabled = initialUI.svoShadowsEnabled; break;
      case "svoAO": ui.svoAmbientOcclusionEnabled = initialUI.svoAmbientOcclusionEnabled; break;
      case "svoPrimarySeamClosure": ui.silhouetteRefinementEnabled = initialUI.silhouetteRefinementEnabled; break;
      case "svoCones": ui.svoConeTracingMode = initialUI.svoConeTracingMode; break;
      case "svoPrimary": ui.svoPrimaryTraversal = initialUI.svoPrimaryTraversal; break;
      case "svoStage": ui.svoStageView = initialUI.svoStageView; break;
      case "svoFlatExempt":
        svoRenderTuning.environmentPlanarRefinementExemption =
          initialUI.svoRenderTuning.environmentPlanarRefinementExemption;
        break;
      case "svoLodPixels":
        svoRenderTuning.lodScreenSpacePixels = initialUI.svoRenderTuning.lodScreenSpacePixels;
        break;
      case "method": break;
      case "quality": break;
      default:
        if (!key.startsWith("scene.") && !(key in SCENE_LAYER_KEYS)) reload.push(key);
    }
  }

  return {
    scene: baseScene && scene !== document.scene ? scene : undefined,
    methodId: cleared.has("method") ? defaultMethodId() : undefined,
    quality: cleared.has("quality") ? "balanced" : undefined,
    methodParams,
    ui,
    svoRenderTuning,
    reload,
  };
}

/** Whether this document has an authored baseline any of this can mean anything against. */
export function sceneHasPresetBaseline(presetId: string) {
  return findSceneDefinition(presetId) !== undefined;
}

/**
 * The location for this scene with nothing on it.
 *
 * Not "the current link minus the overrides": a preset id is the whole of a
 * clean scene link, so this is what the reader would have got by opening the
 * card, which is the thing they are asking to share.
 */
export function cleanSceneLink(presetId: string) {
  return `/scene?scene=${encodeURIComponent(presetId)}`;
}
