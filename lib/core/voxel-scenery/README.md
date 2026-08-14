# Procedural objects

Everything in this directory generates **reusable scenery**, not props for one
scene. A generator here is a small parameterised species: give it numbers and a
seed and it grows one specimen, anywhere, in any set.

That is a stronger requirement than "the hero pond looks right", and it is worth
saying why. The sets in `lib/scenery-presets.ts` are authored as flat
`SceneryNode[]`, one per environment — every boulder, every mushroom, every
flagstone spelled out as coordinates. That is a fine way to place five things and
a terrible way to place five hundred, and it means a stone bed built for the
garden cannot ring the bathhouse pool without being retyped. A generator that
takes a curve and a ground query can do both, and re-seeding it is how the same
set becomes a *family* rather than a one-off.

## The shape of a generator

Three types, in this order. `lib/voxel-scenery/rosette.ts` is the reference
implementation.

```ts
// 1. The FORM: shape only. No position, no seed, no id. This is the species.
export interface RosetteForm {
  bladeCount: number; length_m: number; halfWidth_m: number; /* … */
}

// 2. Named forms: the specimens worth having, as data.
export const ROSETTE_AIR_PLANT: RosetteForm = { /* … */ };
export const ROSETTE_GRASS_TUFT: RosetteForm = { /* … */ };

// 3. The SPEC: a form, plus where this one goes and which one it is.
export interface RosetteSpec extends RosetteForm {
  key: string;                                        // id prefix — see below
  at_m: readonly [number, number];
  groundHeightAt: (x_m: number, z_m: number) => number;
  seed: number;                                       // the only entropy
}

export function rosetteNodes(spec: RosetteSpec): SceneryNode[];
```

Separating form from placement is what makes a species reusable: four air plants
in one scene are one `ROSETTE_AIR_PLANT` and four specs, and a new set gets the
species for free.

## The five rules

**1. Depend on geometry, never on a scene.** Take a plan curve, a ground query, a
level — not `PondVesselSpec`, not `HERO_GARDEN_VESSEL`, not a scene document.

```ts
function stoneBed(spec: { rail: readonly (readonly [number, number])[]; … })   // reusable
function stoneBed(spec: { vessel: PondVesselSpec; … })                          // hero-only
```

The pond hands out `pondVesselPlanCurve` precisely so that generators can follow
an outline without knowing what made it. A bed that takes a curve can hug a pond,
a path, a wall footing or a stream bank; a bed that takes the pond can do one.

**2. Ground comes in as a function.** `(x_m, z_m) => number`. Callers pass
`terrainHeightAt(scene.terrain, x, z)` — the baked heightfield the renderer draws
and the solver collides against, not a re-derivation of it. A generator that
computes its own ground will disagree with the ground on the day someone bakes
the terrain differently.

**3. Every number is a parameter with a stated default.** If you tuned it against
a render, it is a parameter and the comment says what moving it does. Constants
buried in the body are the difference between a species and a prop. The two
things that are *not* parameters are the id prefix and the seed, which are
identity.

**4. `key` prefixes every id.** Node ids must be unique across the whole scene, so
a generator that hard-codes `"boulder-1"` can be instantiated exactly once.
`${key}/cap`, `${key}/stem`. The `group` a node declares is separate and is what
the editor selects and what
`svoMaterialFunctionIdForEnvironmentProxy` reads to choose a surface closure —
name it for what the thing *is*, and know that the regex is load-bearing.

**5. Deterministic, and re-seedable into siblings.** One seed in, identical
geometry out, every rebuild — the sparse publication cache is keyed on it. And a
different seed must grow a recognisable sibling, not a different species: if
re-seeding can produce something degenerate, the form has a parameter that wants
clamping.

## Reaching a scene document

A species is a function, and a scene document is JSON. The bridge is
`lib/scenery-generators.ts`: a frozen `SCENERY_GENERATORS` catalog, keyed to the
`generator` node kind in `lib/scenery-graph.ts`, so a document holds
`{ generator, seed, params }` and the several hundred primitives it grows never
appear in it. Before that catalog existed, every caller *baked*: the hero
garden's three generators were called at build time and their 684 nodes — 884 kB
of ellipsoid centres — were splatted into the node list, which made re-seeding a
factory re-run that discarded the user's edits and made a saved scene a copy of
its output rather than a description of itself.

What that means for writing one:

- **`key` and `seed` come off the node.** The catalog passes the generator node's
  own id as `key`, so ids stay unique by construction, and the node's `seed` is
  the only entropy. Neither belongs in `params` — see rule 3.
- **`groundHeightAt` is supplied at expansion**, from the scene's baked
  heightfield. That is the whole reason a generator is a node kind rather than a
  serialized call: a document cannot hold a function.
- **A run is named, not enumerated.** A generator that follows an outline takes
  a `PondVesselSpec` from the graph's `vessels` table and derives its own rail.
  Keep taking a `PlanRail` or a `PlanOutline` in the species — rule 1 — and let
  the catalog entry do the conversion. A rail in `params` is baked geometry with
  a different name.
- **`params` must survive `JSON.stringify`.** No functions, no class instances.
  A species whose spec needs a callback (`pebbleBedNodes`' band width is
  `(turn) => number`) cannot have a node kind until that parameter is made
  declarative; today it is reached through the arrangement that supplies it.

Adding an entry is one line in `SCENERY_GENERATOR_IDS`, one in
`SceneryGeneratorParamsByKind` and one in `SCENERY_GENERATORS`; anything missing
is a compile error and there is no `register()`.

## Oracles

Each generator gets a CPU test pinning, at least: determinism; re-seeded
siblinghood; leaf count against its budget; extent; that ground-standing parts
are seated on the ground rather than floating or buried; and any material-closure
regex the object relies on, so a rename cannot silently restyle it.

The scene-wide primitive ceiling is
`SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 4096` (`lib/svo-primitive-candidates.ts`),
shared by every object in the frame. State your budget and assert it.

## Looking at one

`tools/preview/README.md`. Build on `heroPreviewScene`, never on a scene factory
directly — a factory returns a document body and loses the art-directed
environment the catalog attaches on the way out.
