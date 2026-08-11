import type { SceneryGeneratorId, SceneryMaterial, SceneryNode } from "./scenery-graph";
import { bonsaiNodes, type BonsaiForm } from "./voxel-scenery/bonsai";
import { pondVesselPlanCurve, type PondVesselSpec } from "./voxel-scenery/pond-vessel";
import { rosetteNodes, type RosetteForm } from "./voxel-scenery/rosette";
import {
  cappedBoulderNodes,
  steppingStoneNodes,
  stoneSet,
  stoneSetBoulderNodes,
  stoneSetPebbleNodes,
  stoneSetSteppingNodes,
  type CappedBoulderForm,
  type PlanOutline,
  type SteppingStoneForm,
} from "./voxel-scenery/stone-set";
import { sweptCopingNodes, type SweptCopingForm } from "./voxel-scenery/swept-coping";

/**
 * The generators a scene document may name, as vocabulary rather than as code
 * the scene factory happens to call.
 *
 * Every one of these existed before this file and every one of them was invoked
 * the same way: a scene factory called `bonsaiNodes(...)` and splatted a few
 * hundred `SceneryNode`s into its own node list. That is a *bake*. The seed and
 * the parameters were consumed and thrown away, and what the document ended up
 * holding — and `cloneScene` copied on every edit, and `localStorage` would have
 * had to store — was the hero garden's 684 nodes and 884 kB of ellipsoid centres
 * where 1.7 kB of `{ generator, seed, params }` says the same thing. Two things
 * follow from a bake and both are bad: re-seeding is not an edit but a re-run of
 * a factory that discards whatever the user changed, and a saved scene stops
 * being a description of itself.
 *
 * So a generator is now a *node kind* — see `SceneryGeneratorNode` — and this is
 * the table that kind is keyed to. `lib/voxel-scenery/procedural-tree.ts` was
 * always the exception that proved it possible: `tree` has been a parameterized
 * node since the graph existed, and `emitTree` is the hard-coded arm this
 * catalog generalizes.
 *
 * ## Why the declarations are here and not beside each species
 *
 * `EDITOR_ENTITIES` composes declarations that live beside their own
 * implementations, and that is the better default. It is deliberately not what
 * happens here, for the reason `lib/voxel-scenery/README.md` states as its first
 * rule: *depend on geometry, never on a scene.* A species takes a plan curve, a
 * ground query and a level. What a document can hold is a *named vessel* and a
 * seed. Something has to turn one into the other — resolve the vessel a node
 * names into the rail a coping sweeps, and the scene's terrain into the ground
 * query a bonsai plants its toes against — and that adaptation is exactly the
 * scene knowledge the species are not allowed to have. It belongs on this side
 * of the line, so `stone-set.ts` and `bonsai.ts` stay reusable in any set.
 *
 * ## What a node may carry, and what it may not
 *
 * A generator node is stored, cloned (`cloneScene` is a JSON round-trip) and
 * eventually written to `localStorage`, so `params` must be plain JSON *and*
 * must be a description rather than an output. The rail is the case that decides
 * the shape of this whole file: `sweptCopingNodes` takes a closed polyline, and
 * the hero pond's is 336 points — 672 numbers, derived by
 * `pondVesselPlanCurve(vessel, 48)` from about thirty. Persisting the polyline
 * would be persisting baked geometry under a different name, and it would drift
 * from the pond the moment a control point moved. So a node that needs a run
 * *names a vessel* the graph declares, and the generator re-derives its own rail
 * at expansion time. Same argument, same answer, for the ground: the query is
 * not in the document at all, because a function cannot be, and it comes from
 * the expansion context — the same heightfield `place.anchor: "terrain"` already
 * resolves against.
 *
 * The rule that falls out: **a saved node is small and regenerates.** Re-seeding
 * is then an ordinary edit to one number, which is the property the whole
 * exercise is for.
 *
 * ## The five, and the one that is missing
 *
 * `capped-boulder`, `stepping-path`, `bonsai`, `rosette` and `swept-coping` are
 * the species; `pond-stone-set` is the arrangement — four boulders, two beds and
 * a wading path, all laid against a vessel's own plan curve, which is one node
 * and one seed for the whole bank. On the hero document the three that are used
 * publish 672, 187 and 1 338 primitives from about 1.7 kB of description.
 *
 * `pebbleBedNodes` is the species with no node kind, and the reason is worth
 * recording rather than fixing badly. A bed's band width is `(turn) => number` —
 * wide where a bank opens out, a single course where it is a wall — and no
 * document holds a function. The hero's beds get theirs from the vessel's own
 * inner-face run and from a table of clusters, which is what `pond-stone-set`
 * exists to supply. Giving `pebble-bed` a node kind means first making that
 * table declarative, which is a real change to the species' spec and not one to
 * make while converting the callers.
 *
 * ## No `register()`
 *
 * Static, frozen, and total over `SceneryGeneratorId` — a missing entry and an
 * unknown entry are both compile errors, and a node naming an id that is not
 * here does not typecheck. Import order and hot reload cannot change what a
 * document may say. `tests/scene-catalog.test.ts` pins the freeze.
 */

/**
 * Rail samples per plan lobe when a coping is swept along a named vessel.
 *
 * Three times the plan's own sixteen, and the multiplier is load-bearing. A cone
 * chain's *outline* is settled at sixteen — the chord sags 0.18 mm on a metre of
 * pond — but its shading normal steps by the whole turn angle at every joint, so
 * at sixteen the rim comes back banded like a caterpillar. Three degrees per
 * joint bands; one degree does not. The rail's resolution is set by the
 * highlight, not by the silhouette, which is also the honest argument for a
 * marched swept-arc primitive: one record would span an arc and there would be
 * no joints to shade across.
 */
export const SWEPT_COPING_RAIL_SAMPLES_PER_LOBE = 48;

/**
 * Everything a generator is given that the document cannot hold.
 *
 * Identity (`key`, `seed`) is here rather than in `params` for the reason
 * `lib/voxel-scenery/README.md` gives: the id prefix and the seed are not
 * parameters, they are which one this is. They come off the node itself, so
 * every generator spells them the same way and none of them can be given a key
 * that collides with the node it was grown from.
 */
export interface SceneryGeneratorRequest {
  /** Id prefix for every node published: the generator node's own id. */
  readonly key: string;
  /** The only entropy, from the node. */
  readonly seed: number;
  /**
   * Ground height at a world point, in metres above the container floor.
   *
   * The heightfield the renderer draws and the solver collides against, not a
   * re-derivation of it — a specimen seated against a surface subtly different
   * from the one the ray hits is the failure this closes.
   */
  readonly groundHeightAt: (x_m: number, z_m: number) => number;
  /**
   * The finest voxel this set will be drawn at, in metres.
   *
   * The third thing a document cannot hold, and it is here for exactly the
   * reason `groundHeightAt` is: a species has an opinion about proportion and no
   * opinion at all about what resolution a scene runs, so the leaf is
   * *placement* rather than form. `BonsaiSpec.leafSize_m` says so in as many
   * words and has said so since the canopy ladder was built — and the hero
   * document never passed one, so the ladder converged on nothing and every
   * specimen in the app drew at the 25 mm floor the species defaults to.
   *
   * Persisting it in `params` was the other option and is the one that goes
   * stale: a saved node carrying `leafSize_m: 0.025` is a bake of today's
   * lattice, and re-seeding would not clear it. Coming off the expansion context
   * means a document written at 25 mm and opened at 3.125 mm draws the finer
   * specimen with nothing re-authored, which is the property
   * `bonsaiCanopyLadder` was designed around.
   *
   * `EnvironmentSceneryContext.detailCellSize_m` is where it comes from; see
   * that field for why it is not `voxelDomain.finestCellSize_m`.
   */
  readonly detailCellSize_m: number;
  /**
   * The vessel this node named, resolved against the graph's own table.
   *
   * Throws when the node named none, or named one the graph does not declare.
   * `validateSceneryGraph` reports the unknown name; the missing one is caught
   * by `growSceneryGenerator` before any geometry is planned.
   */
  readonly vessel: () => PondVesselSpec;
}

/**
 * One entry. Narrow on purpose: an id, whether it needs a vessel, and a pure
 * function from a description to nodes. There is nothing here about lifecycle,
 * readiness or allocation, because a generator has none of those — which is why
 * this is a sibling of `RESOURCE_PLUGIN_CATALOG` and not an entry inside it.
 */
export interface SceneryGenerator<Params> {
  readonly id: string;
  /**
   * Whether `grow` resolves a named vessel.
   *
   * Declared rather than discovered so the failure lands on the document. A
   * species may reach for the rail well inside a plan, and an error thrown there
   * names the species; `growSceneryGenerator` checks this first and names the
   * node that forgot to say which pond it rings.
   */
  readonly needsVessel: boolean;
  readonly grow: (params: Params, request: SceneryGeneratorRequest) => SceneryNode[];
}

/** A rim swept along a named vessel's plan curve, set into the ground it meets. */
export interface SweptCopingGeneratorParams extends SweptCopingForm {
  /** Defaults to `SWEPT_COPING_RAIL_SAMPLES_PER_LOBE`. */
  readonly railSamplesPerLobe?: number;
  /**
   * The selectable object, and — because the material regex is load-bearing —
   * the surface closure. `svoMaterialFunctionIdForEnvironmentProxy` matches
   * `coping`, so renaming this restyles the rim.
   */
  readonly group?: string;
  readonly material: SceneryMaterial;
}

/**
 * The three things a pond stone set is made of, as the split a document may
 * author them along. `boulders` is the capped-boulder family, `beds` the cobble
 * heaps and shingle, `path` the wading stepping stones.
 */
export type PondStoneSetFamily = "boulders" | "beds" | "path";

/** A pond's whole stone arrangement, laid against the vessel the node names. */
export interface PondStoneSetGeneratorParams {
  /**
   * Still-water level in metres. Bedded stones stay above it and the wading path
   * holds its freeboard over it, so this is what decides where the shore is.
   */
  readonly waterline_m: number;
  /**
   * Which of the set's families this node emits. Absent is all three.
   *
   * This is what lets a scene split the arrangement into selectable pieces:
   * author each boulder as its own `capped-boulder` node (positions and seeds
   * from `stoneSetBoulderPlan`), keep the beds on one node and the path on
   * another. Emission order is always boulders → beds → path regardless of the
   * order given here, so a split document publishes in the unsplit order and
   * owner indices never move.
   */
  readonly families?: readonly PondStoneSetFamily[];
}

/**
 * One specimen, standing where it stands and leaning where it leans.
 *
 * Deliberately without a leaf size. `BonsaiSpec.leafSize_m` is placement, not
 * form, and it arrives from `SceneryGeneratorRequest.detailCellSize_m` — see the
 * note there for why a document that pinned it would be a bake.
 */
export interface BonsaiGeneratorParams extends BonsaiForm {
  /** Where the trunk meets the ground, in world metres on the plan. */
  readonly at_m: readonly [number, number];
  /** Direction the crown overhangs toward, in XZ. Need not be normalized. */
  readonly lean: readonly [number, number];
}

/** One air plant or grass tuft, wedged into whatever it stands in. */
export interface RosetteGeneratorParams extends RosetteForm {
  readonly at_m: readonly [number, number];
  /** How far the neck is pushed under the ground. Defaults inside the species. */
  readonly bed_m?: number;
}

/** One mushroom-cap boulder or rounded cobble. */
export interface CappedBoulderGeneratorParams extends CappedBoulderForm {
  readonly at_m: readonly [number, number];
}

/**
 * A chain of stepping stones along an authored line.
 *
 * The control points are a description and not a bake — six pairs for the hero's
 * shallow S — so unlike a rail they belong in the document. What is *not* here
 * is where each plate ends up: that is solved per stone against the bed and the
 * level, every rebuild.
 */
export interface SteppingPathGeneratorParams extends SteppingStoneForm {
  readonly path: PlanOutline;
  readonly count: number;
  /** Tread radius at the first stone and at the last. */
  readonly radiusStart_m: number;
  readonly radiusEnd_m: number;
  /** Clear water between consecutive treads. */
  readonly stride_m: number;
  /** The still-water level the path wades through. Omit for a dry path. */
  readonly level_m?: number;
}

/**
 * The params each id carries, as a map rather than as inference off the catalog.
 *
 * Written out because the alternative is circular: `SceneryNode` includes
 * `SceneryGeneratorNode`, which needs the params type, which would be inferred
 * from `grow`, whose return type is `SceneryNode[]`. Declaring the map breaks
 * the loop, and the catalog's own annotation below then makes an entry that
 * disagrees with it a compile error.
 *
 * Keyed to `SceneryGeneratorId`, which is `lib/scenery-graph.ts`'s frozen id
 * list — see the note there for why the ids live on the schema's side and the
 * geometry on this one. Totality is enforced by the catalog's annotation below:
 * an id with no entry here cannot be used to index this map, and an entry here
 * with no id is an excess property on the catalog literal.
 */
export interface SceneryGeneratorParamsByKind {
  readonly "swept-coping": SweptCopingGeneratorParams;
  readonly "pond-stone-set": PondStoneSetGeneratorParams;
  readonly bonsai: BonsaiGeneratorParams;
  readonly rosette: RosetteGeneratorParams;
  readonly "capped-boulder": CappedBoulderGeneratorParams;
  readonly "stepping-path": SteppingPathGeneratorParams;
}

export type SceneryGeneratorParams<Id extends SceneryGeneratorId> = SceneryGeneratorParamsByKind[Id];

export const SCENERY_GENERATORS: Readonly<{
  [Id in SceneryGeneratorId]: SceneryGenerator<SceneryGeneratorParams<Id>>;
}> = Object.freeze({
  "swept-coping": {
    id: "swept-coping",
    needsVessel: true,
    grow: ({ railSamplesPerLobe, ...form }, request) => sweptCopingNodes({
      ...form,
      key: request.key,
      seed: request.seed,
      rail: pondVesselPlanCurve(request.vessel(), railSamplesPerLobe ?? SWEPT_COPING_RAIL_SAMPLES_PER_LOBE),
      groundHeightAt: request.groundHeightAt,
      leafSize_m: request.detailCellSize_m,
    }),
  },
  "pond-stone-set": {
    id: "pond-stone-set",
    needsVessel: true,
    // Deliberately not handed `request.groundHeightAt`. The set plans against
    // the vessel's own analytic height function, which is what `stoneSet` builds
    // for itself: a bake and a bilinear fetch of that bake disagree by up to two
    // millimetres at a coping's foot, which is a fifth of a pebble, and the
    // decisions this generator makes — is this bank dry, how deep is the bed
    // here — want the surface that was designed rather than the one that was
    // sampled. Where a stone finally *sits* is still the terrain anchor's, and
    // therefore still the baked grid's.
    grow: (params, request) => {
      const spec = {
        vessel: request.vessel(),
        waterline_m: params.waterline_m,
        seed: request.seed,
        key: request.key,
        leafSize_m: request.detailCellSize_m,
      };
      if (params.families === undefined) return stoneSet(spec);
      const nodes: SceneryNode[] = [];
      if (params.families.includes("boulders")) nodes.push(...stoneSetBoulderNodes(spec));
      if (params.families.includes("beds")) nodes.push(...stoneSetPebbleNodes(spec));
      if (params.families.includes("path")) nodes.push(...stoneSetSteppingNodes(spec));
      return nodes;
    },
  },
  bonsai: {
    id: "bonsai",
    needsVessel: false,
    grow: ({ at_m, lean, ...form }, request) => bonsaiNodes({
      ...form,
      key: request.key,
      at_m,
      lean,
      groundHeightAt: request.groundHeightAt,
      // The scene's own lattice, so `bonsaiCanopyLadder` publishes the finest
      // floret the picture can actually resolve. At 25 mm the plate's 30 mm
      // floret is 1.2 leaves and the legibility floor raises it to 75 mm; by
      // 6.25 mm the floor has stopped binding and the specimen is drawing the
      // plate's own proportion at 4.8 leaves across. Nothing is re-authored
      // between those — the ladder was always a derivation, it was simply never
      // handed the number it derives from.
      leafSize_m: request.detailCellSize_m,
      seed: request.seed,
    }),
  },
  rosette: {
    id: "rosette",
    needsVessel: false,
    grow: ({ at_m, bed_m, ...form }, request) => rosetteNodes({
      ...form,
      key: request.key,
      at_m,
      bed_m,
      groundHeightAt: request.groundHeightAt,
      seed: request.seed,
      leafSize_m: request.detailCellSize_m,
    }),
  },
  "capped-boulder": {
    id: "capped-boulder",
    needsVessel: false,
    // No ground query: a boulder's shape does not depend on the ground, only its
    // datum does, and the datum is the terrain anchor's business.
    grow: ({ at_m, ...form }, request) => cappedBoulderNodes({
      ...form,
      key: request.key,
      at_m,
      seed: request.seed,
      leafSize_m: request.detailCellSize_m,
    }),
  },
  "stepping-path": {
    id: "stepping-path",
    needsVessel: false,
    grow: ({ path, count, radiusStart_m, radiusEnd_m, stride_m, level_m, ...form }, request) =>
      steppingStoneNodes({
        ...form,
        key: request.key,
        seed: request.seed,
        path,
        count,
        radiusStart_m,
        radiusEnd_m,
        stride_m,
        level_m,
        groundHeightAt: request.groundHeightAt,
        leafSize_m: request.detailCellSize_m,
      }),
  },
});

/**
 * Run the generator a node names.
 *
 * The one cast in this file, and it is the well-known correlated-union hole:
 * `generator` and `params` are correlated in `SceneryGeneratorNode`'s own type,
 * and TypeScript loses that correlation the instant the id is used as an index.
 * The cast is sound by construction — the node type is generated *from*
 * `SceneryGeneratorParamsByKind`, so an id can only ever appear beside its own
 * params — and the alternative is a hand-written switch in the expander, which
 * is precisely the registration surface this catalog exists to remove.
 */
export function growSceneryGenerator(
  id: SceneryGeneratorId,
  params: SceneryGeneratorParams<SceneryGeneratorId>,
  request: SceneryGeneratorRequest,
): SceneryNode[] {
  const generator = SCENERY_GENERATORS[id] as SceneryGenerator<typeof params>;
  if (generator.needsVessel) request.vessel();
  return generator.grow(params, request);
}
