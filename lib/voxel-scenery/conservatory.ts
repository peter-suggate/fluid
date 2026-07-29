import {
  addRoomShell,
  C,
  cmul,
  V,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * Hose-filled tank (paper Figure 3), staged in a Victorian glasshouse.
 *
 * Everything here exists to explain one event: a hose fills a shallow tank.
 * The coil lies on the flagstones at the near left, uncoils toward the tank,
 * climbs a cane and stops dead at the left glass — exactly where the scene's
 * jet enters — so the eye reads hose, then jet, then rising level. The staging
 * bench sits directly behind the tank in the paper camera, which makes its row
 * of pots the backdrop the water is measured against, and the glazing bay
 * behind it is lit from outside so the mullions and rails throw hard bars of
 * sun forward across the free surface.
 *
 * The palette is a single near-neutral grey ramp: limewashed timber, chalk,
 * flagstone, zinc, and porcelain foliage glazed white. Only the sun panels and
 * the pendant globes carry colour, so every other read in the room comes from
 * value, shadow and falloff.
 */

/** Limewashed joinery: the brightest matte value in the room. */
const LIME = C(.84, .83, .805);
/** Chalky painted plaster and stoneware. */
const CHALK = C(.70, .695, .68);
/** "Pale terracotta": value and a trace of warmth, no hue. */
const CLAY = C(.655, .620, .585);
const STONE = C(.44, .435, .425);
const SLATE = C(.255, .26, .27);
/** The one cool-leaning family, so metal still reads as metal in a white room. */
const ZINC = C(.53, .545, .565);
const RUBBER = C(.075, .078, .082);
/** Glazed clay foliage. */
const PORCELAIN = C(.86, .865, .85);
/** Emissive only, and therefore the sole saturated colour in the environment. */
const SUN = C(1, .88, .62);

/** Deterministic per-index scatter: scenery is a pure function of its context. */
const scatter = (index: number) => ((Math.imul(index + 1, 2246822519) >>> 11) & 0xff) / 255;
/** Value-only variation, so scatter never introduces hue. */
const shade = (base: readonly [number, number, number], index: number, spread = .18) =>
  cmul(base, 1 - spread * .5 + spread * scatter(index));

/** Eight chunky beads per turn read as a coil; ellipsoids are stretched along the local tangent. */
const COIL_TURN: ReadonlyArray<{ readonly u: number; readonly v: number; readonly rx: number; readonly rz: number }> = [
  { u: 1, v: 0, rx: .075, rz: .155 },
  { u: .7071, v: .7071, rx: .125, rz: .125 },
  { u: 0, v: 1, rx: .155, rz: .075 },
  { u: -.7071, v: .7071, rx: .125, rz: .125 },
  { u: -1, v: 0, rx: .075, rz: .155 },
  { u: -.7071, v: -.7071, rx: .125, rz: .125 },
  { u: 0, v: -1, rx: .155, rz: .075 },
  { u: .7071, v: -.7071, rx: .125, rz: .125 },
];
const COIL_INNER: ReadonlyArray<{ readonly u: number; readonly v: number; readonly rx: number; readonly rz: number }> = [
  { u: 1, v: 0, rx: .07, rz: .135 },
  { u: .5, v: .866, rx: .125, rz: .10 },
  { u: -.5, v: .866, rx: .125, rz: .10 },
  { u: -1, v: 0, rx: .07, rz: .135 },
  { u: -.5, v: -.866, rx: .125, rz: .10 },
  { u: .5, v: -.866, rx: .125, rz: .10 },
];

/** Flagstone slabs, hand-authored so the paving ring never crosses the tank footprint. */
const FLAGSTONES: ReadonlyArray<readonly [number, number]> = [
  [-1.95, -.75], [-1.35, -.15], [-1.85, .45], [-1.25, 1.05], [-.85, .30],
  [-1.80, 1.30], [-2.25, .35], [.90, -.35], [1.50, -.95], [-.30, 1.15],
];

export const conservatoryScenery: EnvironmentSceneryModule = {
  id: "conservatory",
  build: (b, context) => {
    const shell = addRoomShell(b, context, {
      materialModel: "conservatory",
      colors: { floor: C(.40, .395, .385), wall: C(.70, .695, .68), ceiling: C(.78, .785, .79) },
    });
    const s = context.s;
    const g = context.floorY_m;
    const zGlaze = -1.48 * s;
    // The scene's jet crosses the left glass at roughly this height; the hose
    // is brought to the wall there so the two read as one continuous thing.
    const jetY = .46 * s;

    // --- Glazing bay -------------------------------------------------------
    // The three inner mullions and the two rails hold their authored positions:
    // the glass panes are derived from them, so moving them would unglaze the
    // wall. Everything else extends the same grid across the frame.
    const frame = LIME;
    for (let i = -1; i <= 1; i++) b.box(`glazing/frame-${i + 1}`, "glazing-frame", V(i * 1.12 * s, .92 * s, zGlaze), V(.027 * s, .92 * s, .027 * s), frame, 0, ["fixture", "frame"]);
    b.box("glazing/rail-low", "glazing-frame", V(0, .62 * s, zGlaze), V(1.18 * s, .025 * s, .027 * s), frame);
    b.box("glazing/rail-high", "glazing-frame", V(0, 1.26 * s, zGlaze), V(1.18 * s, .025 * s, .027 * s), frame);
    const outerMullions = [-2.30, -1.71, 1.71, 2.30];
    outerMullions.forEach((x, i) => b.box(`glazing/mullion-${i + 1}`, "glazing-frame", V(x * s, .95 * s, zGlaze), V(.034 * s, .95 * s, .030 * s), shade(frame, i, .10), 0, ["fixture", "frame"]));
    for (const [name, y] of [["low", .62], ["high", 1.26]] as const) {
      b.box(`glazing/rail-outer-${name}`, "glazing-frame", V(0, y * s, zGlaze - .045 * s), V(2.42 * s, .022 * s, .022 * s), cmul(frame, .94), 0, ["fixture", "frame"]);
    }
    b.box("glazing/sill", "glazing-frame", V(0, g + .07 * s, zGlaze), V(2.42 * s, .07 * s, .055 * s), cmul(frame, .84), 0, ["fixture", "frame"]);
    b.box("glazing/head", "glazing-frame", V(0, 1.90 * s, zGlaze), V(2.42 * s, .06 * s, .055 * s), cmul(frame, .90), 0, ["fixture", "frame"]);

    // Daylight outside the glass. Two rectangle-area emitters behind the bay,
    // asymmetric in size and level so the mullion shadows rake the tank from
    // the back left instead of arriving flat.
    b.box("sun/bay-left", "emissive-sky", V(-.58 * s, 1.02 * s, zGlaze - .13 * s), V(.50 * s, .60 * s, .022 * s), SUN, 2.4, ["light", "emits-positive-z"]);
    b.box("sun/bay-right", "emissive-sky", V(.60 * s, .86 * s, zGlaze - .13 * s), V(.46 * s, .50 * s, .022 * s), cmul(SUN, .94), 1.5, ["light", "emits-positive-z"]);

    // --- Staging bench -----------------------------------------------------
    // Directly behind the tank from the paper camera, and raised to working
    // height so its pots clear the near rim and frame the waterline.
    const benchX = -1.18 * s, benchTop = g + .635 * s;
    b.box("bench/seat", "wood-bench", V(benchX, g + .60 * s, -.70 * s), V(.62 * s, .035 * s, .24 * s), LIME, 0, ["bench"]);
    b.box("bench/back", "wood-bench", V(benchX, g + .80 * s, -.92 * s), V(.62 * s, .16 * s, .035 * s), cmul(LIME, .88), 0, ["bench"]);
    for (const i of [-1, 1]) b.box(`bench/leg-${i < 0 ? "left" : "right"}`, "wood-bench", V(benchX + .52 * i * s, g + .30 * s, -.80 * s), V(.045 * s, .30 * s, .045 * s), cmul(LIME, .72), 0, ["bench"]);
    for (const i of [-1, 1]) b.box(`bench/leg-front-${i < 0 ? "left" : "right"}`, "wood-bench", V(benchX + .52 * i * s, g + .30 * s, -.56 * s), V(.045 * s, .30 * s, .045 * s), cmul(LIME, .66), 0, ["bench"]);
    b.box("bench/lower-shelf", "wood-bench", V(benchX, g + .22 * s, -.70 * s), V(.56 * s, .022 * s, .19 * s), cmul(LIME, .60), 0, ["bench"]);
    b.box("bench/front-lip", "wood-bench", V(benchX, g + .665 * s, -.47 * s), V(.62 * s, .03 * s, .022 * s), cmul(LIME, .78), 0, ["bench"]);

    // Pale terracotta on the staging, largest at the ends so the silhouette
    // steps down toward the water.
    const pots: ReadonlyArray<readonly [string, number, number, number, number]> = [
      ["a", -1.66, -.78, .115, .125],
      ["b", -1.40, -.62, .085, .090],
      ["c", -.98, -.74, .125, .140],
      ["d", -.80, -.88, .070, .075],
    ];
    pots.forEach(([name, x, z, r, h], i) => b.cylinder(`bench/pot-${name}`, "terracotta-pot", V(x * s, benchTop + h * s, z * s), r * s, h * s, shade(CLAY, i, .16), 0, ["pot"]));
    b.cylinder("bench/pot-upturned", "terracotta-pot", V(-1.19 * s, benchTop + .08 * s, -.86 * s), .105 * s, .08 * s, cmul(CLAY, .82), 0, ["pot"]);
    b.box("bench/seed-tray", "wood-bench", V(-1.36 * s, benchTop + .032 * s, -.545 * s), V(.20 * s, .032 * s, .105 * s), cmul(CHALK, .92), 0, ["bench"]);
    b.ellipsoid("bench/seedling-a", "porcelain-foliage", V(-1.66 * s, benchTop + .30 * s, -.78 * s), V(.115 * s, .09 * s, .10 * s), PORCELAIN, 0, ["plant"]);
    b.ellipsoid("bench/seedling-b", "porcelain-foliage", V(-.98 * s, benchTop + .34 * s, -.74 * s), V(.135 * s, .11 * s, .12 * s), cmul(PORCELAIN, .90), 0, ["plant"]);

    // Zinc can waiting on the staging, spout turned to the water.
    b.cylinder("bench/can-body", "zinc-metal", V(-.86 * s, benchTop + .105 * s, -.80 * s), .105 * s, .105 * s, ZINC, 0, ["watering-can"]);
    b.box("bench/can-spout", "zinc-metal", V(-.68 * s, benchTop + .13 * s, -.80 * s), V(.10 * s, .028 * s, .028 * s), cmul(ZINC, .86), 0, ["watering-can"]);
    b.box("bench/can-handle", "zinc-metal", V(-.86 * s, benchTop + .245 * s, -.80 * s), V(.035 * s, .055 * s, .085 * s), cmul(ZINC, .74), 0, ["watering-can"]);

    // --- Porcelain planter, screen right ----------------------------------
    b.box("planter/stone", "stone-planter", V(1.12 * s, g + .26 * s, -.86 * s), V(.28 * s, .26 * s, .28 * s), CHALK, 0, ["planter"]);
    b.box("planter/plinth", "stone-planter", V(1.12 * s, g + .035 * s, -.86 * s), V(.315 * s, .035 * s, .315 * s), cmul(CHALK, .78), 0, ["planter"]);
    b.box("planter/rim", "stone-planter", V(1.12 * s, g + .545 * s, -.86 * s), V(.315 * s, .035 * s, .315 * s), cmul(CHALK, 1.08), 0, ["planter"]);
    b.cylinder("planter/stem", "porcelain-foliage", V(1.12 * s, g + .72 * s, -.86 * s), .048 * s, .18 * s, cmul(PORCELAIN, .74), 0, ["plant"]);
    b.ellipsoid("planter/foliage-main", "porcelain-foliage", V(1.12 * s, g + .90 * s, -.86 * s), V(.40 * s, .34 * s, .30 * s), PORCELAIN, 0, ["plant"]);
    b.ellipsoid("planter/foliage-left", "porcelain-foliage", V(.88 * s, g + 1.08 * s, -.84 * s), V(.24 * s, .34 * s, .19 * s), cmul(PORCELAIN, .84), 0, ["plant"]);
    b.ellipsoid("planter/foliage-right", "porcelain-foliage", V(1.36 * s, g + 1.14 * s, -.90 * s), V(.22 * s, .38 * s, .18 * s), cmul(PORCELAIN, .70), 0, ["plant"]);
    b.ellipsoid("planter/foliage-crown", "porcelain-foliage", V(1.10 * s, g + 1.34 * s, -.88 * s), V(.16 * s, .22 * s, .15 * s), cmul(PORCELAIN, .96), 0, ["plant"]);

    // --- Zinc trough, screen right, running back toward the glazing --------
    const troughX = 1.52 * s, troughZ = -.06 * s;
    b.box("trough/body", "zinc-metal", V(troughX, g + .30 * s, troughZ), V(.30 * s, .16 * s, .52 * s), ZINC, 0, ["trough"]);
    for (const i of [-1, 1]) b.box(`trough/rim-${i < 0 ? "left" : "right"}`, "zinc-metal", V(troughX + .30 * i * s, g + .47 * s, troughZ), V(.035 * s, .03 * s, .55 * s), cmul(ZINC, 1.14), 0, ["trough"]);
    for (const i of [-1, 1]) b.box(`trough/rim-${i < 0 ? "front" : "back"}`, "zinc-metal", V(troughX, g + .47 * s, troughZ + .52 * i * s), V(.33 * s, .03 * s, .035 * s), cmul(ZINC, 1.06), 0, ["trough"]);
    for (const i of [-1, 1]) b.box(`trough/trestle-${i < 0 ? "front" : "back"}`, "wood-bench", V(troughX, g + .07 * s, troughZ + .38 * i * s), V(.24 * s, .07 * s, .05 * s), cmul(LIME, .62), 0, ["trough"]);

    // --- The hose ----------------------------------------------------------
    // Two turns of coil on the paving at the near left, then a run that
    // straightens, climbs a cane and terminates at the tank's left glass.
    const coil = { x: -1.02 * s, z: .82 * s };
    COIL_TURN.forEach((node, i) => b.ellipsoid(`hose/coil-outer-${i + 1}`, "rubber-hose", V(coil.x + node.u * .32 * s, g + .055 * s, coil.z + node.v * .32 * s), V(node.rx * s, .055 * s, node.rz * s), shade(RUBBER, i, .35), 0, ["hose"]));
    COIL_INNER.forEach((node, i) => b.ellipsoid(`hose/coil-inner-${i + 1}`, "rubber-hose", V(coil.x + node.u * .19 * s, g + .145 * s, coil.z + node.v * .19 * s), V(node.rx * s, .05 * s, node.rz * s), shade(cmul(RUBBER, 1.35), i + 8, .35), 0, ["hose"]));
    b.box("hose/run-back", "rubber-hose", V(-.86 * s, g + .05 * s, .42 * s), V(.055 * s, .05 * s, .34 * s), cmul(RUBBER, 1.2), 0, ["hose"]);
    b.box("hose/run-in", "rubber-hose", V(-.745 * s, g + .05 * s, .06 * s), V(.17 * s, .05 * s, .055 * s), cmul(RUBBER, 1.3), 0, ["hose"]);
    b.cylinder("hose/riser", "rubber-hose", V(-.635 * s, g + .25 * s, .04 * s), .05 * s, .25 * s, cmul(RUBBER, 1.5), 0, ["hose"]);
    b.cylinder("hose/cane", "wood-bench", V(-.68 * s, g + .36 * s, .10 * s), .022 * s, .36 * s, cmul(LIME, .70), 0, ["hose"]);
    b.box("hose/nozzle-arm", "zinc-metal", V(-.585 * s, jetY, .04 * s), V(.07 * s, .045 * s, .045 * s), cmul(ZINC, .92), 0, ["hose", "fixture"]);
    b.box("hose/nozzle-collar", "zinc-metal", V(-.545 * s, jetY, .04 * s), V(.033 * s, .030 * s, .030 * s), cmul(ZINC, 1.20), 0, ["hose", "fixture"]);

    // --- Foreground can, bottom right --------------------------------------
    const canX = 1.08 * s, canZ = .42 * s;
    b.cylinder("can/body", "zinc-metal", V(canX, g + .17 * s, canZ), .17 * s, .17 * s, cmul(ZINC, .96), 0, ["watering-can"]);
    b.cylinder("can/shoulder", "zinc-metal", V(canX, g + .40 * s, canZ), .115 * s, .06 * s, cmul(ZINC, 1.12), 0, ["watering-can"]);
    b.box("can/spout", "zinc-metal", V(.82 * s, g + .30 * s, canZ), V(.16 * s, .035 * s, .035 * s), cmul(ZINC, .84), 0, ["watering-can"]);
    b.ellipsoid("can/rose", "zinc-metal", V(.665 * s, g + .30 * s, canZ), V(.06 * s, .06 * s, .06 * s), cmul(ZINC, 1.22), 0, ["watering-can"]);
    b.box("can/handle", "zinc-metal", V(canX, g + .48 * s, canZ), V(.045 * s, .09 * s, .13 * s), cmul(ZINC, .78), 0, ["watering-can"]);

    // --- Pots on the flagstones, near left ---------------------------------
    const floorPots: ReadonlyArray<readonly [string, number, number, number, number]> = [
      ["a", -1.72, -.30, .16, .19],
      ["b", -1.44, -.14, .12, .145],
      ["c", -1.62, .12, .13, .10],
      ["d", -1.34, .34, .105, .16],
    ];
    floorPots.forEach(([name, x, z, r, h], i) => b.cylinder(`pots/floor-${name}`, "terracotta-pot", V(x * s, g + h * s, z * s), r * s, h * s, shade(CLAY, i + 4, .20), 0, ["pot"]));

    // A stack of upturned pots holds the near-left edge of the paper camera,
    // so the frame closes on that side without anything crowding the tank.
    const stack: ReadonlyArray<readonly [string, number, number, number]> = [
      ["base", .085, .180, .085], ["mid", .235, .155, .075], ["top", .375, .130, .065],
    ];
    stack.forEach(([name, y, r, h], i) => b.cylinder(`pots/stack-${name}`, "terracotta-pot", V(-1.85 * s, g + y * s, 1.25 * s), r * s, h * s, shade(CLAY, i + 9, .22), 0, ["pot"]));

    // --- Paving ------------------------------------------------------------
    FLAGSTONES.forEach(([x, z], i) => b.box(`paving/slab-${i + 1}`, "flagstone-paving", V(x * s, g + .03 * s, z * s), V(.30 * s, .03 * s, .30 * s), shade(STONE, i + 16, .26), 0, ["paving"]));

    // --- Porcelain topiary, upper left -------------------------------------
    b.box("topiary/plinth", "stone-planter", V(-2.15 * s, g + .30 * s, -1.05 * s), V(.20 * s, .30 * s, .20 * s), cmul(CHALK, .86), 0, ["planter"]);
    b.ellipsoid("topiary/ball", "porcelain-foliage", V(-2.15 * s, g + .78 * s, -1.05 * s), V(.26 * s, .24 * s, .26 * s), cmul(PORCELAIN, .92), 0, ["plant"]);
    b.ellipsoid("topiary/finial", "porcelain-foliage", V(-2.15 * s, g + 1.02 * s, -1.05 * s), V(.15 * s, .14 * s, .15 * s), PORCELAIN, 0, ["plant"]);

    // --- Pendants ----------------------------------------------------------
    for (let i = -1; i <= 1; i++) {
      b.cylinder(`pendant-${i + 1}/cord`, "fixture-cord", V(i * .56 * s, 1.50 * s, -1.04 * s), .012 * s, .34 * s, SLATE, 0, ["fixture"]);
      b.ellipsoid(`pendant-${i + 1}/globe`, "emissive-glass", V(i * .56 * s, 1.18 * s, -1.04 * s), V(.095 * s, .095 * s, .095 * s), C(.85, .68, .38), .48, ["fixture", "light"]);
    }
    return shell;
  },
};
