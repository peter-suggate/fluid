import { terrainHeightAt } from "../terrain";
import {
  C,
  cmul,
  V,
  type EnvironmentLinearColor,
  type EnvironmentSceneryModule,
} from "./builder";

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

    const trunk = clay(.50);
    const gill = clay(.45);
    // The lamppost was the last saturated surface in the set: a near-black
    // metal post. Glazed porcelain instead, so the only colour left anywhere in
    // this garden is the light itself, and the warm lantern rakes across a pale
    // post rather than being lost against a black one.
    const lampClay = stone(.60);
    const lampGlow = C(1.0, .48, .19);

    // ---- Cloud trees ------------------------------------------------------
    // Niwaki, not lollipops: every canopy is a flattened pad, stacked at
    // varied heights so the silhouette reads as cut cloud even when the top of
    // the frame crops it. The tallest tree backs the far bank; a small one sits
    // in the near corner to frame the pond.
    const bigTree = rootedAt(-.45, -.117);
    // Held: the trunk datum is the lawn, and downstream tests read its top at
    // g + 0.30 s. Everything else on this tree hangs off that.
    b.cylinder("tree-big/trunk", "tree-trunk", V(-.45 * s, g + .15 * s, -.117 * s), .055 * s, .15 * s, trunk, 0, ["tree"]);
    b.ellipsoid("tree-big/canopy-low", "leaf-foliage", bigTree(-.345, .255, -.215), V(.160 * s, .085 * s, .150 * s), clay(.80), 0, ["tree"]);
    b.ellipsoid("tree-big/canopy-main", "leaf-foliage", bigTree(-.40, .40, -.08), V(.30 * s, .155 * s, .28 * s), clay(.90), 0, ["tree"]);
    b.ellipsoid("tree-big/canopy-side", "leaf-foliage", bigTree(-.56, .325, -.20), V(.175 * s, .098 * s, .165 * s), clay(.83), 0, ["tree"]);
    b.ellipsoid("tree-big/canopy-top", "leaf-foliage", bigTree(-.325, .525, -.02), V(.155 * s, .090 * s, .145 * s), clay(.95), 0, ["tree"]);

    const roundTree = rootedAt(.45, -.25);
    b.cylinder("tree-round/trunk", "tree-trunk", roundTree(.45, .12, -.25), .045 * s, .12 * s, clay(.48), 0, ["tree"]);
    b.ellipsoid("tree-round/canopy-main", "leaf-foliage", roundTree(.45, .325, -.25), V(.225 * s, .125 * s, .215 * s), clay(.87), 0, ["tree"]);
    b.ellipsoid("tree-round/canopy-side", "leaf-foliage", roundTree(.56, .42, -.19), V(.125 * s, .072 * s, .115 * s), clay(.93), 0, ["tree"]);

    const knollTree = rootedAt(-.373, .233);
    b.cylinder("tree-knoll/trunk", "tree-trunk", knollTree(-.373, .135, .233), .038 * s, .135 * s, clay(.49), 0, ["tree"]);
    b.ellipsoid("tree-knoll/canopy-low", "leaf-foliage", knollTree(-.405, .245, .200), V(.145 * s, .078 * s, .135 * s), clay(.82), 0, ["tree"]);
    b.ellipsoid("tree-knoll/canopy-main", "leaf-foliage", knollTree(-.355, .350, .245), V(.175 * s, .092 * s, .165 * s), clay(.89), 0, ["tree"]);
    b.ellipsoid("tree-knoll/canopy-top", "leaf-foliage", knollTree(-.320, .450, .215), V(.115 * s, .065 * s, .110 * s), clay(.95), 0, ["tree"]);

    const nearTree = rootedAt(.54, .34);
    b.cylinder("tree-near/trunk", "tree-trunk", nearTree(.54, .115, .34), .034 * s, .115 * s, clay(.47), 0, ["tree"]);
    b.ellipsoid("tree-near/canopy-main", "leaf-foliage", nearTree(.545, .245, .335), V(.160 * s, .085 * s, .150 * s), clay(.86), 0, ["tree"]);
    b.ellipsoid("tree-near/canopy-top", "leaf-foliage", nearTree(.515, .335, .300), V(.105 * s, .060 * s, .100 * s), clay(.92), 0, ["tree"]);

    const farTree = rootedAt(-.117, -.45);
    b.cylinder("tree-far/trunk", "tree-trunk", farTree(-.117, .10, -.45), .030 * s, .10 * s, clay(.46), 0, ["tree"]);
    b.ellipsoid("tree-far/canopy", "leaf-foliage", farTree(-.117, .215, -.45), V(.145 * s, .078 * s, .135 * s), clay(.84), 0, ["tree"]);

    // ---- Mushroom ring ----------------------------------------------------
    // Oversized to toy scale and spaced right around the pond, so wherever the
    // camera lands there is a cap catching light on the far bank.
    const grand = rootedAt(-.20, -.33);
    b.cylinder("mushroom-grand/stem", "mushroom-stem", grand(-.20, .10, -.33), .05 * s, .10 * s, clay(.72), 0, ["mushroom"]);
    b.cylinder("mushroom-grand/gill", "mushroom-gill", grand(-.20, .205, -.33), .115 * s, .012 * s, gill, 0, ["mushroom"]);
    b.ellipsoid("mushroom-grand/cap", "mushroom-cap", grand(-.20, .25, -.33), V(.14 * s, .088 * s, .14 * s), clay(.94), 0, ["mushroom"]);

    const sprout = rootedAt(-.13, -.36);
    b.cylinder("mushroom-sprout/stem", "mushroom-stem", sprout(-.13, .055, -.36), .028 * s, .055 * s, clay(.68), 0, ["mushroom"]);
    b.ellipsoid("mushroom-sprout/cap", "mushroom-cap", sprout(-.13, .135, -.36), V(.072 * s, .046 * s, .072 * s), clay(.88), 0, ["mushroom"]);

    const tall = rootedAt(-.30, .25);
    b.cylinder("mushroom-tall/stem", "mushroom-stem", tall(-.30, .075, .25), .036 * s, .075 * s, clay(.70), 0, ["mushroom"]);
    b.cylinder("mushroom-tall/gill", "mushroom-gill", tall(-.30, .16, .25), .085 * s, .010 * s, gill, 0, ["mushroom"]);
    b.ellipsoid("mushroom-tall/cap", "mushroom-cap", tall(-.30, .20, .25), V(.105 * s, .066 * s, .105 * s), clay(.91), 0, ["mushroom"]);

    const button = rootedAt(.38, .07);
    b.cylinder("mushroom-button/stem", "mushroom-stem", button(.38, .06, .07), .032 * s, .06 * s, clay(.69), 0, ["mushroom"]);
    b.ellipsoid("mushroom-button/cap", "mushroom-cap", button(.38, .15, .07), V(.085 * s, .054 * s, .085 * s), clay(.89), 0, ["mushroom"]);

    const pip = rootedAt(.24, -.345);
    b.cylinder("mushroom-pip/stem", "mushroom-stem", pip(.24, .045, -.345), .024 * s, .045 * s, clay(.66), 0, ["mushroom"]);
    b.ellipsoid("mushroom-pip/cap", "mushroom-cap", pip(.24, .115, -.345), V(.06 * s, .04 * s, .06 * s), clay(.86), 0, ["mushroom"]);

    const crown = rootedAt(.413, .133);
    b.cylinder("mushroom-crown/stem", "mushroom-stem", crown(.413, .115, .133), .052 * s, .115 * s, clay(.73), 0, ["mushroom"]);
    b.cylinder("mushroom-crown/gill", "mushroom-gill", crown(.413, .235, .133), .12 * s, .013 * s, clay(.44), 0, ["mushroom"]);
    b.ellipsoid("mushroom-crown/cap", "mushroom-cap", crown(.413, .285, .133), V(.15 * s, .095 * s, .15 * s), clay(.95), 0, ["mushroom"]);

    const knollCap = rootedAt(-.413, .207);
    b.cylinder("mushroom-knoll/stem", "mushroom-stem", knollCap(-.413, .07, .207), .034 * s, .07 * s, clay(.70), 0, ["mushroom"]);
    b.ellipsoid("mushroom-knoll/cap", "mushroom-cap", knollCap(-.413, .155, .207), V(.095 * s, .06 * s, .095 * s), clay(.90), 0, ["mushroom"]);

    const twin = rootedAt(.207, .347);
    b.cylinder("mushroom-twin/stem", "mushroom-stem", twin(.207, .062, .347), .030 * s, .062 * s, clay(.69), 0, ["mushroom"]);
    b.ellipsoid("mushroom-twin/cap", "mushroom-cap", twin(.207, .148, .347), V(.082 * s, .052 * s, .082 * s), clay(.92), 0, ["mushroom"]);

    // ---- Rock arrangement -------------------------------------------------
    // A three-stone group bedded into the west bank where the pond narrows, a
    // pair flanking the rockery promontory, and one low boulder in the near
    // corner to stop the foreground running out of the frame flat.
    const sentinel = rootedAt(-.393, -.04);
    b.ellipsoid("rock-sentinel/body", "stone-rock", sentinel(-.393, .175, -.04), V(.105 * s, .175 * s, .09 * s), stone(.60), 0, ["rock"]);
    b.ellipsoid("rock-sentinel/shoulder", "stone-rock", sentinel(-.420, .085, -.055), V(.085 * s, .085 * s, .075 * s), stone(.66), 0, ["rock"]);
    const recline = rootedAt(-.433, .02);
    b.ellipsoid("rock-recline/body", "stone-rock", recline(-.433, .075, .02), V(.170 * s, .075 * s, .115 * s), stone(.64), 0, ["rock"]);
    const lowRock = rootedAt(-.36, -.12);
    b.ellipsoid("rock-low/body", "stone-rock", lowRock(-.36, .052, -.12), V(.105 * s, .052 * s, .09 * s), stone(.71), 0, ["rock"]);
    const promontoryA = rootedAt(.06, .267);
    b.ellipsoid("rock-promontory-a/body", "stone-rock", promontoryA(.06, .075, .267), V(.100 * s, .075 * s, .085 * s), stone(.63), 0, ["rock"]);
    const promontoryB = rootedAt(-.047, .247);
    b.ellipsoid("rock-promontory-b/body", "stone-rock", promontoryB(-.047, .058, .247), V(.085 * s, .058 * s, .075 * s), stone(.68), 0, ["rock"]);
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
    reedClump("reed-cove", -.1533, .1467, 2, 5501, .032, .045, .085, .0100);
    reedClump("reed-beach", .0733, -.3000, 3, 6607, .036, .046, .092, .0102);
    // Two tufts on the immediate foreground lawn, where the painted blades used
    // to sit. Short, so they read as grass rather than as more reeds.
    reedClump("tuft-near", .3500, .3667, 2, 7703, .030, .030, .060, .0098);
    reedClump("tuft-fore", -.1000, .3500, 2, 8809, .028, .028, .055, .0098);

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
      ["pebble-4", .0550, -.2900, .036, .024, .69],
      ["pebble-5", .0830, -.3090, .028, .019, .75],
      ["pebble-6", .1150, -.3230, .042, .027, .66],
      ["pebble-7", .1510, -.3310, .024, .017, .72],
      ["pebble-8", .1870, -.3330, .034, .023, .71],
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
