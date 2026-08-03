import type { CameraState, SceneDescription } from "./model";
import { defaultCamera } from "./model";
import type { EnvironmentId } from "./environments";
import { environmentIds } from "./environments";
import type { MethodProfile } from "./methods";
import { sceneWithEnvironment } from "./scenery-presets";

/**
 * What a scene *is*, as data.
 *
 * Before this module a scene existed in seven places at once: the authored
 * preset list, eleven local factories in the WebGPU smoke catalog that rebuilt
 * the same documents differently, the saved-scene store, URL hydration, a
 * handful of one-off test and tool fixtures, and the bare scene-id strings in
 * npm scripts and lane tables. Two of those produced *different documents under
 * the same name* — the smoke locals set `environment` as a bare string and
 * shipped with no `scenery` graph at all, so `hose-tank` on the GPU lane was
 * not the `hose-tank` in the product.
 *
 * The fix is not another wrapper. It is one accessor:  `sceneDocument` is the
 * only supported way to turn an identity into a document, and everything that
 * needs a scene — the library, the preset list, a smoke lane, a benchmark —
 * goes through it. A definition therefore has to carry everything those callers
 * used to reimplement: identity, where it belongs in the library, its art
 * direction, its camera, the solver profile a numerical scene requires, and the
 * named deltas (`variants`) that a lane applies instead of forking the body.
 *
 * `build` stays an ordinary function. A closure was never the problem —
 * duplication was. Composition between scenes is an ordinary call to another
 * factory, which stays typed and greppable, rather than a bespoke layer engine.
 *
 * As with `EDITOR_ENTITIES` and `VISUALIZATION_CATALOG`, there is deliberately
 * no mutable `register()`: import order and hot reload cannot change what the
 * product offers.
 */

/**
 * Who a scene is for, which is not the same as what it is made of.
 *
 * `group` used to be a shelf label written for us — "Comparisons" held fourteen
 * numerical oracles, and they were a third of the first thing a visitor saw. A
 * scene's audience decides whether it is offered or disclosed; its `shelf`
 * decides where it sits once offered.
 */
export type SceneAudience = "explore" | "study" | "validation";

export const SCENE_AUDIENCES: ReadonlyArray<{
  readonly id: SceneAudience;
  readonly label: string;
  readonly blurb: string;
  /** Validation scenes are real work, not clutter — disclosed, never removed. */
  readonly disclosed: boolean;
}> = Object.freeze([
  { id: "explore", label: "Explore", blurb: "Scenes to watch, poke and edit.", disclosed: false },
  { id: "study", label: "Study", blurb: "Published figures and method comparisons.", disclosed: false },
  { id: "validation", label: "Research & validation", blurb: "Analytic oracles with known answers.", disclosed: true },
]);

/**
 * A named delta over a definition's document.
 *
 * This is what replaces a forked factory. A GPU lane that wants the mini dam
 * with surface tension zeroed and a pinned timestep declares that here, beside
 * the scene it is a variant *of*, instead of rebuilding the body somewhere the
 * product never looks.
 */
export interface SceneVariant {
  readonly id: string;
  /** Why this delta exists; shown in tooling, never in the library. */
  readonly description: string;
  /** Applied to the built body, before the environment is attached. */
  readonly apply: (scene: SceneDescription) => SceneDescription;
}

export interface SceneDefinition {
  /** Stable identity. Persisted in URLs, saved scenes, lane tables and scripts. */
  readonly id: string;
  readonly name: string;
  /** One or two sentences, written for the person choosing it. */
  readonly blurb: string;
  readonly audience: SceneAudience;
  /** Display grouping within an audience, e.g. "Garden", "Free-fall contact". */
  readonly shelf: string;
  /** Art-directed environment; part of this scene's presentation. */
  readonly environment: EnvironmentId;
  readonly camera?: Partial<CameraState>;
  /** Exact solver profile a numerical comparison requires. */
  readonly methodProfile?: MethodProfile;
  /** The document body, without environment or scenery. */
  readonly build: () => SceneDescription;
  readonly variants?: Readonly<Record<string, SceneVariant>>;
}

/** Recursively freeze declarative data while leaving factories callable. */
function deepFreeze<T>(value: T): T {
  if (value === null || (typeof value !== "object" && typeof value !== "function") || Object.isFrozen(value)) {
    return value;
  }
  Object.freeze(value);
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  return value;
}

const ENVIRONMENT_IDS: ReadonlySet<string> = new Set(environmentIds);

/**
 * Validate at definition time, the way `defineSceneWebGPUSmokeSuite` does.
 *
 * A malformed catalog entry is a startup error rather than a card that renders
 * blank in the library or a lane that silently runs a different scene.
 */
export function defineScene(definition: SceneDefinition): SceneDefinition {
  const { id } = definition;
  if (!id.trim()) throw new Error("A scene definition needs a non-empty id");
  if (!definition.name.trim()) throw new Error(`Scene ${id} needs a name`);
  if (!definition.blurb.trim()) throw new Error(`Scene ${id} needs a blurb`);
  if (!definition.shelf.trim()) throw new Error(`Scene ${id} needs a shelf`);
  if (!ENVIRONMENT_IDS.has(definition.environment)) {
    throw new Error(`Scene ${id} names an unknown environment ${definition.environment}`);
  }
  for (const [key, variant] of Object.entries(definition.variants ?? {})) {
    if (variant.id !== key) throw new Error(`Scene ${id} variant key ${key} differs from variant id ${variant.id}`);
    if (!variant.description.trim()) throw new Error(`Scene ${id} variant ${key} needs a description`);
  }
  return deepFreeze(definition);
}

/**
 * The one supported way to obtain a document.
 *
 * Attaching the environment here is what makes the product and the GPU lanes
 * agree: the smoke catalog's local factories used to assign `scene.environment`
 * directly, which names a background without copying its scenery, so those
 * documents rendered in an empty world. A scene that staged its own scenery
 * keeps it — re-seeding would throw away exactly the edits that make a study
 * scene a study — and from that point the document owns what it looks like.
 */
export function sceneDocument(definition: SceneDefinition, variantId?: string): SceneDescription {
  const variant = variantId === undefined ? undefined : definition.variants?.[variantId];
  if (variantId !== undefined && !variant) {
    throw new Error(`Scene ${definition.id} has no variant ${variantId}`);
  }
  const body = definition.build();
  const scene = variant ? variant.apply(body) : body;
  return scene.scenery
    ? { ...scene, environment: definition.environment }
    : sceneWithEnvironment(scene, definition.environment);
}

/** The camera a definition opens on, filled out from the shared default. */
export function sceneDefinitionCamera(definition: SceneDefinition): CameraState {
  return { ...defaultCamera, ...definition.camera };
}

export function duplicateSceneDefinitionIds(definitions: readonly SceneDefinition[]): string[] {
  const seen = new Set<string>(), duplicates = new Set<string>();
  for (const definition of definitions) {
    if (seen.has(definition.id)) duplicates.add(definition.id);
    seen.add(definition.id);
  }
  return [...duplicates];
}

// ---- library cards ---------------------------------------------------------

/**
 * What opening a card does to the product.
 *
 * Loading a preset and loading a saved scene used to be different code paths
 * with different behaviour — only one of them applied the camera or the solver
 * profile — which a grid that shows both in one row cannot afford.
 */
export interface SceneOpening {
  readonly scene: SceneDescription;
  /** Identity carried into the scene store and the URL. */
  readonly presetId: string;
  /**
   * The framing to adopt, already resolved against the shared default.
   *
   * Resolved rather than partial so that opening a scene which declares no
   * camera still restores the default view, as loading a preset always did.
   * Absent means "keep the camera where it is", which is only right for a saved
   * document whose origin scene this build no longer has.
   */
  readonly camera?: CameraState;
  readonly methodProfile?: MethodProfile;
}

export type SceneCardSource = "catalog" | "saved" | "starter";

export interface SceneCard {
  readonly id: string;
  readonly source: SceneCardSource;
  readonly name: string;
  readonly blurb: string;
  /** Authored scenes only. A saved scene or a starter belongs to the person, not to an audience. */
  readonly audience?: SceneAudience;
  readonly shelf: string;
  /** Saved scenes only; the library sorts its own shelf by recency. */
  readonly savedAt_ms?: number;
  readonly open: () => SceneOpening;
}

export function sceneCardForDefinition(definition: SceneDefinition): SceneCard {
  return {
    id: definition.id,
    source: "catalog",
    name: definition.name,
    blurb: definition.blurb,
    audience: definition.audience,
    shelf: definition.shelf,
    open: () => ({
      scene: sceneDocument(definition),
      presetId: definition.id,
      camera: sceneDefinitionCamera(definition),
      methodProfile: definition.methodProfile,
    }),
  };
}

export interface SceneShelf {
  readonly shelf: string;
  readonly cards: readonly SceneCard[];
}

/**
 * Group cards onto shelves without reordering them.
 *
 * Order is the catalog's, because the catalog's order is an editorial decision
 * about what to offer first; a shelf that re-sorted alphabetically would quietly
 * overrule it.
 */
export function sceneShelves(cards: readonly SceneCard[]): SceneShelf[] {
  const shelves: { shelf: string; cards: SceneCard[] }[] = [];
  for (const card of cards) {
    const existing = shelves.find((entry) => entry.shelf === card.shelf);
    if (existing) existing.cards.push(card);
    else shelves.push({ shelf: card.shelf, cards: [card] });
  }
  return shelves;
}
