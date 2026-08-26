import type { CameraState, SceneDescription } from "./model";
import { defaultCamera } from "./model";
import type { EnvironmentId } from "./environments";
import { environmentIds } from "./environments";
import type { MethodProfile } from "./method-contract";
import { sceneWithEnvironment } from "./scenery-presets";
import { studioStageFits } from "./studio-stage-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "./voxel-environments";
import { svoEnvironmentPayloadBytes } from "../svo/svo-environment-coarsening";
import {
  svoSceneryDetailCellSize_m,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM,
} from "../svo/svo-render-tuning";
import { solidVoxelEditsForScene, solidVoxelShellForScene } from "./scene-lattice";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

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
 * scene's audience decides where its section sits in the reading order; its
 * `shelf` decides where it sits within the section. The oracles used to be
 * collapsed behind a disclosure as well, but the map down the page's side now
 * names every shelf, and a shelf the map points into must be on the page.
 */
export type SceneAudience = "explore" | "study" | "validation";

export const SCENE_AUDIENCES: ReadonlyArray<{
  readonly id: SceneAudience;
  readonly label: string;
  readonly blurb: string;
}> = Object.freeze([
  { id: "explore", label: "Explore", blurb: "Scenes to watch, poke and edit." },
  { id: "study", label: "Study", blurb: "Published figures and method comparisons." },
  { id: "validation", label: "Research & validation", blurb: "Analytic oracles with known answers." },
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

/**
 * The two sizes a document is *built* at, as opposed to patched to.
 *
 * `cellSize_m` is the octree's finest cell. `detailCellSize_m` is the finest
 * voxel the authored set is drawn into, which is smaller whenever a dry scene
 * spends `SvoRenderTuning.environmentRefinementDepth` levels under the tree's
 * own lattice — see `svoSceneryDetailCellSize_m`, which is the function that
 * answers it.
 *
 * Both are inputs to construction and that is the whole point of the type. A
 * heightfield is baked once, and a generator's legibility floors are counts of
 * detail voxels resolved while it expands; a size written onto a finished
 * document moves the tree and nothing that feeds it. The smoke lane learned
 * this the expensive way — `FLUID_SVO_DRY_SMOKE_CELL_MM` overwrote
 * `voxelDomain.finestCellSize_m` on a built scene until
 * `createHeroGardenHoseSceneWithSet` took a lattice — and the browser had no
 * equivalent path at all.
 */
export interface SceneLattice {
  readonly cellSize_m: number;
  /** Defaults to `cellSize_m`, which is every scene that asks for no extra levels. */
  readonly detailCellSize_m?: number;
}

/**
 * How much of the authored world a scene asks the renderer to present.
 *
 * `fluid-only` is the measurement path: the raster water pipeline keeps its
 * complete surface and optics work, but consumes a retained clear background
 * instead of constructing or drawing the sparse dry world. It is definition
 * data rather than a dam-break special case so future simulation-focused scenes
 * can select the same path without changing the renderer.
 *
 * It used to be the *default*, on the reasoning that most of the catalog is a
 * container of water in the bare studio environment and the dry world there was
 * a floor and four walls that cost a construction and a draw to show. That was
 * true of the room the studio used to be, and it stopped being true when the
 * house set became a stage — a floor and one practical are what put the water
 * somewhere, and a scene presented against a retained clear is a subject in a
 * void. So `full-scene` is now what a definition gets by saying nothing, and
 * `fluid-only` is the opt-out; see `presentationModeForScene`.
 *
 * Any scene that opens dry (`systems.fluid === false`) must present full-scene,
 * since fluid-only would leave it with nothing at all.
 */
export type ScenePresentationMode = "full-scene" | "fluid-only";

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
  /** Defaults to `full-scene`; see {@link presentationModeForScene}. */
  readonly presentationMode?: ScenePresentationMode;
  readonly camera?: Partial<CameraState>;
  /** Exact solver profile a numerical comparison requires. */
  readonly methodProfile?: MethodProfile;
  /** The document body, without environment or scenery. */
  readonly build: () => SceneDescription;
  /**
   * The same body, re-authored at a lattice — the one route to a finer document.
   *
   * Optional because most scenes are a container of water whose only size is
   * `voxelDomain.finestCellSize_m`, and for those a patch and a rebuild are the
   * same document. It is present on the scenes that bake something at
   * construction: the hero garden bakes a heightfield, composes a set against
   * the surface that bake produced, and hands every generator a detail voxel to
   * resolve its legibility floors against.
   *
   * A definition that declares this promises `buildAt({ cellSize_m: c })` agrees
   * with `build()` when `c` is the scene's authored lattice, so the preset and
   * the rebuilt document are the same scene rather than two.
   */
  readonly buildAt?: (lattice: SceneLattice) => SceneDescription;
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
/**
 * The document the product opens, at the rung it opens scenes at.
 *
 * A definition's `build()` is its own default and answers to nobody; the
 * *product's* choice of refinement depth is a policy, and this is the one place
 * it is applied. A dry definition that declares `buildAt` is therefore built
 * twice: once plainly, to learn the lattice it chose for itself, and again at
 * `SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT` levels under it. Measured at 3-8 ms
 * a build on the hero garden — the terrain is a procedural description derived
 * on demand rather than a baked grid, so construction is cheap and the second
 * pass costs nothing worth avoiding.
 *
 * Twice rather than once because the depth is expressed as levels under a
 * lattice and only the factory knows which lattice that is. Asking for a *depth*
 * and letting the factory resolve the size is the alternative, and it would put
 * the same policy into every factory.
 *
 * Only through `buildAt`. Writing a finer detail cell onto a document whose
 * factory already ran would claim a leaf its terrain was never sampled for,
 * which is the exact class of disagreement this file's lattice type exists to
 * prevent — see {@link SceneLattice}. A definition with no `buildAt` opens at
 * its own lattice, and the render panel's depth control patches it from there
 * with the consequence stated.
 */
export function sceneDocument(definition: SceneDefinition, variantId?: string): SceneDescription {
  const plain = finishSceneDocument(definition, definition.build(), variantId);
  if (!definition.buildAt || plain.systems?.fluid !== false) return plain;
  const cellSize_m = plain.voxelDomain.finestCellSize_m;
  const requestedDepth = Math.max(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM,
    Math.trunc(SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT));
  if (requestedDepth < 0) {
    const coarseCellSize_m = cellSize_m * 2 ** -requestedDepth;
    const coarse = finishSceneDocument(definition, definition.buildAt({
      cellSize_m: coarseCellSize_m,
      detailCellSize_m: coarseCellSize_m,
    }), variantId);
    coarse.voxelDomain.environmentRefinementBaseCellSize_m = cellSize_m;
    return coarse;
  }
  const detailCellSize_m = svoSceneryDetailCellSize_m(cellSize_m, {
    environmentRefinementDepth: requestedDepth,
    fluid: false,
  });
  // A factory that already authored a finer set has said something this policy
  // has no better answer to, so it keeps it.
  if ((plain.voxelDomain.detailCellSize_m ?? cellSize_m) <= detailCellSize_m) return plain;
  return finishSceneDocument(definition, definition.buildAt({ cellSize_m, detailCellSize_m }), variantId);
}

/**
 * The same document, re-authored at a lattice.
 *
 * This is the browser's counterpart to what `tools/run-svo-dry-render-smoke.ts`
 * does with `FLUID_SVO_DRY_SMOKE_REFINEMENT`: regenerate the preset through its
 * own factory with the sizes as *inputs*, rather than write them onto a finished
 * document. A definition with no `buildAt` has nothing that a lattice would
 * change at construction, so it falls back to `build()` and the caller patches
 * the tree as it always did — announced by the returned `rebuilt` flag rather
 * than by silently producing the coarse document.
 */
export function sceneDocumentAtLattice(
  definition: SceneDefinition,
  lattice: SceneLattice,
  variantId?: string,
): { readonly scene: SceneDescription; readonly rebuilt: boolean } {
  assertSceneLattice(definition.id, lattice);
  if (!definition.buildAt) return { scene: sceneDocument(definition, variantId), rebuilt: false };
  return { scene: finishSceneDocument(definition, definition.buildAt(lattice), variantId), rebuilt: true };
}

/** Whether a lattice is an input this scene's factory takes, or only a patch. */
export function sceneDefinitionTakesLattice(definition: SceneDefinition): boolean {
  return definition.buildAt !== undefined;
}

function assertSceneLattice(id: string, lattice: SceneLattice): void {
  const { cellSize_m, detailCellSize_m } = lattice;
  if (!(cellSize_m > 0) || !Number.isFinite(cellSize_m)) {
    throw new RangeError(`Scene ${id} lattice needs a positive finite cell size`);
  }
  if (detailCellSize_m !== undefined
    && (!(detailCellSize_m > 0) || detailCellSize_m > cellSize_m + 1e-12)) {
    throw new RangeError(`Scene ${id} detail cell size must be positive and no coarser than the lattice`);
  }
}

function finishSceneDocument(
  definition: SceneDefinition,
  body: SceneDescription,
  variantId?: string,
): SceneDescription {
  const variant = variantId === undefined ? undefined : definition.variants?.[variantId];
  if (variantId !== undefined && !variant) {
    throw new Error(`Scene ${definition.id} has no variant ${variantId}`);
  }
  const scene = variant ? variant.apply(body) : body;
  const authoredEdits = solidVoxelEditsForScene(scene);
  if (definition.environment === "garden") {
    // The garden's generated terrain is the vessel. An additional rectangular
    // container is both visually wrong (a glass tank around the set) and, now
    // that SolidWorld is the renderer's geometry authority, catastrophically
    // expensive to look through. Keep ordinary opaque/clear edits, but remove
    // both a factory's legacy shell and any independently authored glass fill.
    scene.solidVoxels = authoredEdits.filter((patch) => patch.operation !== "fill"
      || (patch.materialId ?? VOXEL_MATERIAL_IDS.containerGlass)
        !== VOXEL_MATERIAL_IDS.containerGlass);
  } else {
    // Preset factories carry only generic extra voxel edits. Compile the
    // ordinary shell once on the final lattice.
    scene.solidVoxels = [...solidVoxelShellForScene(scene), ...authoredEdits];
  }
  return scene.scenery
    ? { ...scene, environment: definition.environment }
    : sceneWithEnvironment(scene, definition.environment);
}

/**
 * What a definition presents, and the one thing that can still say no.
 *
 * Every scene is presented full-scene unless it states otherwise. Audience is
 * deliberately *not* a term here: an earlier cut of this rule kept the
 * validation lanes bare on the reasoning that a scene which exists to be
 * measured should not pay for a set, and it excluded twenty scenes — the
 * hydrostatic oracles, the dam-break ladder, the free-fall and rigid-coupling
 * contacts — which is most of the catalog and most of what a person scrolling
 * the library actually sees. The measurement argument turned out to be thinner
 * than it looked: `presentationMode` is read by the viewport alone, so the Dawn
 * lanes and benchmarks that produce those scenes' numbers never consult it and
 * are unaffected either way. What it changes is the picture in the app, and
 * there the answer is the same for an oracle as for anything else.
 *
 * The set is the exception, and it is a hard one rather than a preference: a
 * set is paid for in voxels, and one too large to resolve does not degrade —
 * it fails to allocate, and the scene has no picture at all. Two things can
 * put a stage scene there, and they are different questions with different
 * answers.
 *
 * The first is the *house set* against the scene's lattice: `studioStageFits`,
 * calibrated on scenes that measurably fail to build. A container it says no to
 * is seeded with `bareStageSceneryGraph` instead, and presenting that scene
 * full-scene would construct and trace a dry world containing a shell with no
 * faces — the whole cost of the set and none of the picture.
 *
 * The second is the set the scene *actually carries*, which for a scene with
 * its own authored graph is not the house set at all and can be far larger than
 * one. The CM12 paper figures are the case: a plain room shell left to size
 * itself off a 12.8 m tank expands to seventy metres of 50 mm wall, and 50 mm
 * is this lattice's own cell, so every metre of it resolves at the finest level
 * there is. That set claims about 3.5 GiB of voxel payload — more than a
 * conformant device is obliged to give a single buffer, and eight times the
 * whole of `ocean-seiche`, house set included. Asking `studioStageFits` about
 * such a scene is asking about a set it will never receive; it happened to
 * answer no for the wrong reason and now answers yes, which is why the claim
 * is measured directly.
 *
 * Only the *stage* is subject to either. A scene with any other environment has
 * a room worth presenting whatever it costs, and a definition that states its
 * own mode is not second-guessed here either.
 */
export function presentationModeForScene(
  definition: Pick<SceneDefinition, "presentationMode" | "environment">,
  scene: SceneDescription,
): ScenePresentationMode {
  if (definition.presentationMode) return definition.presentationMode;
  if (definition.environment !== "stage") return "full-scene";
  if (!studioStageFits(scene)) return "fluid-only";
  return sceneEnvironmentPayloadBytes(scene) <= SCENE_ENVIRONMENT_PAYLOAD_BUDGET_BYTES
    ? "full-scene" : "fluid-only";
}

/**
 * The voxel payload a stage scene's set may claim and still be presented.
 *
 * 256 MiB, which is not a taste: it is WebGPU's guaranteed `maxBufferSize`, the
 * largest single allocation every conformant device must be able to make. A set
 * that cannot fit in that is one this gate has no business promising anywhere,
 * and the estimate it is compared against is a ceiling rather than a reading,
 * so the number wants to sit in open space rather than at a real limit.
 *
 * It does. Measured across the catalog with `svoEnvironmentPayloadBytes`, every
 * stage scene carrying the house set lands between 2.4 and 30.0 MiB — the whole
 * ladder, the rigid contacts, the droplet and fill sweeps, `ocean-seiche` at
 * 10.1 and `power-hybrid-deep-ocean`, the largest, at 30.0. Every scene
 * carrying its own set lands between 887 MiB and 3.5 GiB. The budget sits in a
 * thirtyfold gap with nothing in it, so no scene is near enough to the line for
 * the estimate's looseness to decide its answer.
 */
export const SCENE_ENVIRONMENT_PAYLOAD_BUDGET_BYTES = 256 * 1024 * 1024;

/**
 * What the set a scene carries will claim, at the rungs it will be drawn at.
 *
 * The catalog is built here rather than passed in because it is the same cached
 * catalog the voxeliser will ask for, keyed on the scene and its environment —
 * so on the path that goes on to build the dry world this is free, and on the
 * path that does not it is three to seven expanded solids.
 */
function sceneEnvironmentPayloadBytes(scene: SceneDescription): number {
  const environmentId = scene.environment ?? "default";
  return svoEnvironmentPayloadBytes(
    environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, environmentId)),
    {
      cellSize_m: scene.voxelDomain.finestCellSize_m,
      brickSize: scene.voxelDomain.brickSize_cells,
    });
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
