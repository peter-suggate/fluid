import { sceneryMaximumSwayExcursion_m } from "../scenery-sway";
import { terrainHeightAt } from "../terrain";
import {
  C,
  cmul,
  V,
  type EnvironmentLinearColor,
  type EnvironmentSceneryModule,
} from "./builder";
import { emitProceduralTree, planProceduralTree } from "./procedural-tree";

/**
 * Porcelain garden: a bonsai landscape in glazed white clay, built around the
 * pond. The ground is a real heightfield rather than shell boxes, so the basin
 * the water settles into is the same surface the solver collides against — and
 * every prop here is sited on that surface rather than on a nominal lawn plane,
 * which is what lets reeds stand in the shallows and rocks bed into the knoll.
 *
 * Composition rule: the water is the subject. Nothing stands in the middle of
 * the basin; trees, mushrooms, rocks, reeds and the bridge all lean toward the
 * pond, ring it, or are reflected in it. Lily pads at the waterline are the one
 * deliberate exception, and they are kept to the shallow rim so the open water
 * stays open.
 *
 * One tree is the subject of the set rather than part of its backdrop: the
 * procedural specimen on the west bank is grown from a seed, is the only thing
 * here that moves, and is deliberately the most detailed object in frame. The
 * scattered small props were thinned to pay for it — a ring of nine mushrooms
 * and a field of eight pebbles were reading as clutter around a pond that is
 * supposed to be the subject, and they were competing with the one silhouette
 * that should carry the frame.
 *
 * Budget: the whole catalog stays at or under SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES
 * (128). Production tolerates more — it falls back to SVO payload traversal —
 * but the offline candidate audit in tests/svo-primitive-candidates.test.ts
 * only sweeps catalogs that fit, so going over silently drops this set's
 * no-false-negative coverage. Spend new detail by taking it from somewhere else.
 *
 * Palette: value only. `clay` and `stone` never leave a ~5% neutral band, so
 * form reads through light, shadow and ambient occlusion instead of hue. The
 * single exception is emission — the lamppost lantern and the stone lantern's
 * ember are deliberately warm, and they are the only saturated colour in the set.
 */

/** Glazed white clay. Faintly warm, and only ever varied in value. */
const clay = (value: number): EnvironmentLinearColor => C(value, value * .985, value * .955);
/** Unglazed stone: the same value range a touch cooler, so rock parts from porcelain. */
const stone = (value: number): EnvironmentLinearColor => C(value * .972, value * .984, value);

/**
 * Deterministic scatter. An integer avalanche hash of the loop index stands in
 * for the RNG a scattered clump would otherwise want: the same garden is built
 * on every rebuild, which the catalog's static revision depends on.
 */
const hash01 = (n: number): number => {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
};

export const gardenScenery: EnvironmentSceneryModule = {
  id: "garden",
  build: (b, context) => {
    const { s, floorY_m: floorY, roomHalf_m: roomHalf, scene } = context;
    const terrainTop = Math.max(scene.container.height_m, scene.terrain?.baseHeight_m ?? 0);
    const shell = {
      kind: "terrain-heightfield" as const,
      floorY_m: floorY,
      bounds_m: { min: V(-roomHalf.x, 0, -roomHalf.z), max: V(roomHalf.x, terrainTop, roomHalf.z) },
      primitives: b.shell,
      materialModel: "garden-terrain" as const,
    };
    const g = scene.terrain?.baseHeight_m ?? 0;
    /** Still-pond level: 3.5 cm of exposed bank below the lawn, in scene units. */
    const waterline = g - .0117 * s;

    /**
     * Anchor one object to the ground under its own root. Every part of that
     * object shares the datum, so a tree on the knoll rises with the knoll
     * instead of shearing its canopy off its trunk. Positions are fractions of
     * the environment scale and heights are rises above that datum; on flat
     * lawn the datum is exactly `g`.
     */
    const rootedAt = (rootX: number, rootZ: number) => {
      const base = scene.terrain ? terrainHeightAt(scene.terrain, rootX * s, rootZ * s) : g;
      return (x: number, rise: number, z: number) => V(x * s, base + rise * s, z * s);
    };

    const gill = clay(.45);
    // The lamppost was the last saturated surface in the set: a near-black
    // metal post. Glazed porcelain instead, so the only colour left anywhere in
    // this garden is the light itself, and the warm lantern rakes across a pale
    // post rather than being lost against a black one.
    const lampClay = stone(.60);
    const lampGlow = C(1.0, .48, .19);

    // ---- The specimen tree ------------------------------------------------
    // The one grown object in the set, on the west bank where the old hand-
    // built cloud tree stood. It keeps that tree's root and its lawn datum, so
    // the bank it stands on is unchanged; everything above the root is
    // generated from the seed below, and re-seeding re-grows the whole tree.
    //
    // It is also the only thing here that moves. The gust is bounded by the
    // sparse lattice rather than by taste: a swaying prop is re-posed in the
    // render ABI every frame and is never re-voxelized, so its surface has to
    // stay inside the cell ownership the proxy voxelizer wrote once at
    // bring-up. lib/scenery-sway.ts owns that budget.
    const heroRoot = rootedAt(-.45, -.117);
    const heroTree = planProceduralTree({
      key: "tree-hero",
      root_m: heroRoot(-.45, 0, -.117),
      height_m: .66 * s,
      rootRadius_m: .036 * s,
      spread_m: .30 * s,
      seed: 0x5eed_9a3,
      // Leans over the deep basin, so the crown is the thing reflected in the
      // open water and the trunk still clears the bank.
      leanXZ: [1, .28],
      bark: clay,
      leaf: clay,
    });
    emitProceduralTree(b, heroTree, {
      excursion_m: sceneryMaximumSwayExcursion_m(scene.voxelDomain.finestCellSize_m),
    });

    // ---- Cloud tree -------------------------------------------------------
    // The one hand-built tree left is backdrop: flattened pads at varied
    // heights, standing well back on the knoll so the specimen owns the middle
    // distance. The near-corner tree that used to frame the pond is gone — it
    // sat between the camera and the specimen and cropped it.
    const knollTree = rootedAt(-.373, .233);
    b.cylinder("tree-knoll/trunk", "tree-trunk", knollTree(-.373, .135, .233), .038 * s, .135 * s, clay(.49), 0, ["tree"]);
    b.ellipsoid("tree-knoll/canopy-low", "leaf-foliage", knollTree(-.405, .245, .200), V(.145 * s, .078 * s, .135 * s), clay(.82), 0, ["tree"]);
    b.ellipsoid("tree-knoll/canopy-main", "leaf-foliage", knollTree(-.355, .350, .245), V(.175 * s, .092 * s, .165 * s), clay(.89), 0, ["tree"]);
    b.ellipsoid("tree-knoll/canopy-top", "leaf-foliage", knollTree(-.320, .450, .215), V(.115 * s, .065 * s, .110 * s), clay(.95), 0, ["tree"]);

    // ---- Mushrooms --------------------------------------------------------
    // Three, not the ring of nine this once was. Oversized to toy scale and
    // kept to the far bank, where a lit cap still reads against the water
    // without the near lawn turning into a fairy circle.
    const grand = rootedAt(-.20, -.33);
    b.cylinder("mushroom-grand/stem", "mushroom-stem", grand(-.20, .10, -.33), .05 * s, .10 * s, clay(.72), 0, ["mushroom"]);
    b.cylinder("mushroom-grand/gill", "mushroom-gill", grand(-.20, .205, -.33), .115 * s, .012 * s, gill, 0, ["mushroom"]);
    b.ellipsoid("mushroom-grand/cap", "mushroom-cap", grand(-.20, .25, -.33), V(.14 * s, .088 * s, .14 * s), clay(.94), 0, ["mushroom"]);

    const tall = rootedAt(-.30, .25);
    b.cylinder("mushroom-tall/stem", "mushroom-stem", tall(-.30, .075, .25), .036 * s, .075 * s, clay(.70), 0, ["mushroom"]);
    b.cylinder("mushroom-tall/gill", "mushroom-gill", tall(-.30, .16, .25), .085 * s, .010 * s, gill, 0, ["mushroom"]);
    b.ellipsoid("mushroom-tall/cap", "mushroom-cap", tall(-.30, .20, .25), V(.105 * s, .066 * s, .105 * s), clay(.91), 0, ["mushroom"]);

    const crown = rootedAt(.413, .133);
    b.cylinder("mushroom-crown/stem", "mushroom-stem", crown(.413, .115, .133), .052 * s, .115 * s, clay(.73), 0, ["mushroom"]);
    b.cylinder("mushroom-crown/gill", "mushroom-gill", crown(.413, .235, .133), .12 * s, .013 * s, clay(.44), 0, ["mushroom"]);
    b.ellipsoid("mushroom-crown/cap", "mushroom-cap", crown(.413, .285, .133), V(.15 * s, .095 * s, .15 * s), clay(.95), 0, ["mushroom"]);

    // ---- Rock arrangement -------------------------------------------------
    // A two-stone group bedded into the west bank where the pond narrows, one
    // on the rockery promontory, and one low boulder in the near corner to stop
    // the foreground running out of the frame flat. The stones that used to
    // crowd the specimen tree's root are gone: its flare does that work now.
    const sentinel = rootedAt(-.393, -.04);
    b.ellipsoid("rock-sentinel/body", "stone-rock", sentinel(-.393, .175, -.04), V(.105 * s, .175 * s, .09 * s), stone(.60), 0, ["rock"]);
    b.ellipsoid("rock-sentinel/shoulder", "stone-rock", sentinel(-.420, .085, -.055), V(.085 * s, .085 * s, .075 * s), stone(.66), 0, ["rock"]);
    const recline = rootedAt(-.433, .02);
    b.ellipsoid("rock-recline/body", "stone-rock", recline(-.433, .075, .02), V(.170 * s, .075 * s, .115 * s), stone(.64), 0, ["rock"]);
    const promontoryA = rootedAt(.06, .267);
    b.ellipsoid("rock-promontory-a/body", "stone-rock", promontoryA(.06, .075, .267), V(.100 * s, .075 * s, .085 * s), stone(.63), 0, ["rock"]);
    const nearRock = rootedAt(.433, .183);
    b.ellipsoid("rock-near/body", "stone-rock", nearRock(.433, .090, .183), V(.115 * s, .090 * s, .100 * s), stone(.61), 0, ["rock"]);

    // ---- Stone lantern ----------------------------------------------------
    // A low yukimi on the rockery promontory, jutting into the back of the
    // pond. Its fire box is two walls with the ±z faces open, so the ember
    // spills toward the camera and lays a warm reflection across the water
    // rather than being sealed inside an opaque block.
    const lantern = rootedAt(.0167, .233);
    b.cylinder("lantern-stone/base", "stone-lantern", lantern(.0167, .022, .233), .105 * s, .022 * s, stone(.72), 0, ["lantern"]);
    b.cylinder("lantern-stone/post", "stone-lantern", lantern(.0167, .058, .233), .038 * s, .022 * s, stone(.78), 0, ["lantern"]);
    b.box("lantern-stone/wall-left", "stone-lantern", lantern(-.0363, .105, .233), V(.014 * s, .037 * s, .058 * s), stone(.84), 0, ["lantern"]);
    b.box("lantern-stone/wall-right", "stone-lantern", lantern(.0697, .105, .233), V(.014 * s, .037 * s, .058 * s), stone(.84), 0, ["lantern"]);
    b.ellipsoid("lantern-stone/ember", "emissive-fixture", lantern(.0167, .105, .233), V(.036 * s, .033 * s, .036 * s), C(1.0, .62, .30), 1.4, ["lantern", "fixture", "light", "point-light"]);
    b.cylinder("lantern-stone/roof", "stone-lantern", lantern(.0167, .160, .233), .105 * s, .022 * s, stone(.87), 0, ["lantern"]);
    b.ellipsoid("lantern-stone/finial", "stone-lantern", lantern(.0167, .200, .233), V(.028 * s, .032 * s, .028 * s), stone(.91), 0, ["lantern"]);

    // ---- Arched bridge ----------------------------------------------------
    // Over the narrow throat at the back of the pond, from the rockery
    // promontory to the east lawn. The deck is a stepped arc of blocks rather
    // than a curve — this is a voxel world, and the stepped silhouette reads.
    // Every segment clears the still waterline, so a settled pond passes
    // cleanly underneath it. Anchored to the lawn datum: a bridge is one rigid
    // object and must not follow the bank it spans.
    const bridgeZ = .22, deckHalf = V(.032 * s, .016 * s, .05 * s);
    const deck: readonly (readonly [string, number, number, number])[] = [
      ["deck-1", .0467, .020, .86],
      ["deck-2", .1000, .042, .88],
      ["deck-3", .1533, .050, .90],
      ["deck-4", .2067, .042, .88],
      ["deck-5", .2600, .020, .86],
    ];
    for (const [name, x, rise, value] of deck) {
      b.box(`bridge/${name}`, "bridge-deck", V(x * s, g + rise * s, bridgeZ * s), deckHalf, clay(value), 0, ["bridge"]);
    }
    b.box("bridge/footing-west", "stone-footing", V(.013 * s, g - .035 * s, bridgeZ * s), V(.045 * s, .06 * s, .055 * s), stone(.70), 0, ["bridge"]);
    b.box("bridge/footing-east", "stone-footing", V(.293 * s, g - .035 * s, bridgeZ * s), V(.045 * s, .06 * s, .055 * s), stone(.70), 0, ["bridge"]);
    const railZ = .0433;
    for (const [name, x, z] of [
      ["post-a", .100, -railZ], ["post-b", .100, railZ],
      ["post-c", .2067, -railZ], ["post-d", .2067, railZ],
    ] as const) {
      b.cylinder(`bridge/${name}`, "bridge-post", V(x * s, g + .088 * s, (bridgeZ + z) * s), .014 * s, .034 * s, clay(.84), 0, ["bridge"]);
    }

    // ---- Reeds and grass tufts -------------------------------------------
    // The direct replacement for the screen-space grass that used to be pinned
    // to the bottom of the frame. These are real tapered clumps standing on the
    // bank at the waterline: they parallax, occlude the far shore, and throw
    // shadow across the water. Blades are chunky on purpose — anything under
    // about two voxels vanishes in cell-centre voxelization.
    const reedClump = (
      key: string, rootX: number, rootZ: number, blades: number, seed: number,
      spread: number, low: number, high: number, thick: number, head = false,
    ) => {
      for (let i = 0; i < blades; i += 1) {
        const lead = head && i === 0;
        const angle = 2 * Math.PI * hash01(seed + 5 * i);
        const reach = lead ? 0 : spread * (.28 + .72 * hash01(seed + 5 * i + 1));
        const half = .5 * (lead ? high : low + (high - low) * hash01(seed + 5 * i + 2));
        const radius = thick * (.86 + .40 * hash01(seed + 5 * i + 3));
        const x = rootX + reach * Math.cos(angle), z = rootZ + reach * Math.sin(angle);
        // Each blade beds into the ground it actually stands on, not the clump
        // centre's: a clump straddling the bank then has some stalks on dry
        // soil and some rising out of the shallows, which is the whole point of
        // building these as geometry instead of painting them.
        const blade = rootedAt(x, z);
        b.cylinder(`${key}/blade-${i}`, "leaf-reed", blade(x, half - .012, z), radius * s, half * s,
          clay(.56 + .28 * hash01(seed + 5 * i + 4)), 0, ["reed", "bank"]);
        if (lead) {
          b.ellipsoid(`${key}/seed-head`, "leaf-seedhead", blade(x, 2 * half + .020, z),
            V(.019 * s, .038 * s, .019 * s), clay(.50), 0, ["reed", "bank"]);
        }
      }
    };
    const tussock = (key: string, rootX: number, rootZ: number, spread: number, value: number) => {
      b.ellipsoid(`${key}/tussock`, "leaf-tussock", rootedAt(rootX, rootZ)(rootX, .016, rootZ),
        V(spread * 1.55 * s, .026 * s, spread * 1.45 * s), clay(value), 0, ["reed", "bank"]);
    };

    // The near-east bank is the closest shoreline to the default camera, so it
    // carries the biggest clump and the one cattail that breaks the horizon.
    tussock("reed-east", .320, .0867, .042, .62);
    reedClump("reed-east", .320, .0867, 5, 1301, .042, .055, .118, .0110, true);
    reedClump("reed-shore", .2667, .2067, 3, 2203, .038, .050, .100, .0105);
    reedClump("reed-north", .0833, .2467, 3, 3307, .030, .048, .095, .0105);
    reedClump("reed-west", -.3467, -.0467, 3, 4409, .040, .052, .108, .0108, true);
    reedClump("reed-beach", .0733, -.3000, 3, 6607, .036, .046, .092, .0102);
    // One tuft on the immediate foreground lawn, where the painted blades used
    // to sit. Short, so it reads as grass rather than as more reeds.
    reedClump("tuft-near", .3500, .3667, 2, 7703, .030, .030, .060, .0098);

    // ---- Lily pads --------------------------------------------------------
    // Held to the shallow rim rather than the open basin: they trace the
    // waterline instead of covering the hero surface, and when a preset starts
    // the pond as a puddle they settle a few centimetres above the shelf
    // instead of hanging in open air over the deep end.
    const pads: readonly (readonly [string, number, number, number, number])[] = [
      ["pad-east", .3000, .0433, .045, .76],
      ["pad-lobe", .2833, .1567, .038, .72],
      ["pad-bridge", .1300, .2367, .042, .78],
      ["pad-west", -.3333, -.0733, .040, .74],
      ["pad-beach", .0400, -.2867, .036, .77],
    ];
    // Centred on the waterline so each disc is half-drowned the way a real pad
    // floats, and thick enough to survive cell-centre voxelization.
    for (const [name, x, z, radius, value] of pads) {
      b.cylinder(`lily/${name}`, "lilypad-wet", V(x * s, waterline, z * s), radius * s, .009 * s, clay(value), 0, ["lily", "waterline"]);
    }
    b.ellipsoid("lily/bud-east", "lilypad-wet", V(.3100 * s, waterline + .038 * s, .0300 * s), V(.024 * s, .036 * s, .024 * s), clay(.93), 0, ["lily", "waterline"]);
    b.ellipsoid("lily/bud-west", "lilypad-wet", V(-.3400 * s, waterline + .036 * s, -.0900 * s), V(.021 * s, .032 * s, .021 * s), clay(.91), 0, ["lily", "waterline"]);

    // ---- Pebbles and stepping stones -------------------------------------
    // Scattered along the beach shelf shore, continuing the line of the
    // authored stepping-stone bodies up onto both banks.
    const pebbles: readonly (readonly [string, number, number, number, number, number])[] = [
      ["pebble-1", .2100, -.2900, .050, .034, .68],
      ["pebble-2", .2600, -.3250, .036, .024, .74],
      ["pebble-3", -.0500, .3100, .045, .030, .70],
    ];
    for (const [name, x, z, radius, rise, value] of pebbles) {
      b.ellipsoid(`${name}/body`, "stone-pebble", rootedAt(x, z)(x, rise * .62, z),
        V(radius * s, rise * s, radius * .89 * s), stone(value), 0, ["pebble"]);
    }
    b.cylinder("stepping/bank-in", "stone-stepping", rootedAt(-.0633, -.2633)(-.0633, .011, -.2633), .045 * s, .011 * s, stone(.73), 0, ["stepping"]);
    b.cylinder("stepping/bank-out", "stone-stepping", rootedAt(.2833, -.0833)(.2833, .011, -.0833), .045 * s, .011 * s, stone(.71), 0, ["stepping"]);

    if (scene.sceneId !== "garden-svo-lighting-study") return shell;
    // The lighting-study composition needed an actual fixture: previously the
    // garden catalog contained trees, mushrooms and pebbles but no lamppost.
    // Keep the source just inside the right bank so its inverse-square pool
    // reaches the pond while the pole and cap produce a recognizable silhouette.
    const lampX = .32 * s, lampZ = .24 * s;
    b.cylinder("lamppost/base", "lamp-fixture", V(lampX, g + .02 * s, lampZ), .05 * s, .02 * s, cmul(lampClay, 1.18), 0, ["lamppost", "fixture"]);
    b.cylinder("lamppost/pole", "lamp-fixture", V(lampX, g + .17 * s, lampZ), .014 * s, .13 * s, lampClay, 0, ["lamppost", "fixture"]);
    b.ellipsoid("lamppost/lantern", "emissive-fixture", V(lampX, g + .35 * s, lampZ), V(.05 * s, .06 * s, .05 * s), lampGlow, 11.0, ["lamppost", "lantern", "fixture", "light", "point-light"]);
    b.cylinder("lamppost/cap", "lamp-fixture", V(lampX, g + .43 * s, lampZ), .06 * s, .012 * s, cmul(lampClay, 1.38), 0, ["lamppost", "fixture"]);
    return shell;
  },
};
