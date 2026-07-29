import {
  addRoomShell,
  C,
  cmul,
  V,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * Mediterranean courtyard at noon. Everything here is chalky white lime plaster
 * and cut stone: the scene is authored entirely in value, so form arrives as
 * hard-edged shadow and bounce rather than as hue. The only colour in the set is
 * emissive — a warm sun-washed wall behind the colonnade, a cool sky panel
 * overhead, a hot glare patch on the east paving — because that is what makes
 * the light legible against the neutral stone.
 *
 * The composition funnels the eye to the tank. A four-column colonnade on a
 * three-step stylobate closes the back with its widest bay centred on the water;
 * a slatted pergola crosses overhead so the noon light lands on the free surface
 * as hard parallel stripes; a rill runs in from the west parapet and stops one
 * hand's width short of the coping, pointing at the tank; and a stepped kerb of
 * the brightest stone in the scene rings the water so the highest value in the
 * frame is the rim the eye should land on.
 */
export const courtyardScenery: EnvironmentSceneryModule = {
  id: "courtyard",
  build: (b, context) => {
    const shell = addRoomShell(b, context, {
      materialModel: "courtyard",
      colors: { floor: C(.60, .585, .555), wall: C(.83, .815, .785), ceiling: C(.72, .735, .775) },
    });
    const { s, floorY_m: y0, scene } = context;
    // Ground-relative height, in scale units, so every prop moves with `s`.
    const Y = (h: number) => y0 + h * s;
    // The water's keep-out: nothing that rises above the container floor may
    // overlap this footprint, so near-tank props are placed off these faces
    // rather than off hand-tuned constants that only hold for one preset.
    const cw = scene.container.width_m * .5;
    const cd = scene.container.depth_m * .5;
    const plaster = C(.80, .785, .755);

    // --- Colonnade: stylobate, four columns, architrave, cornice -------------
    const colZ = -1.34 * s;
    b.box("stylobate/step-1", "plinth-step", V(0, Y(.045), colZ), V(2.62 * s, .045 * s, .70 * s), cmul(plaster, .80), 0, ["plinth"]);
    b.box("stylobate/step-2", "plinth-step", V(0, Y(.135), colZ), V(2.52 * s, .045 * s, .62 * s), cmul(plaster, .88), 0, ["plinth"]);
    b.box("stylobate/step-3", "plinth-step", V(0, Y(.215), colZ), V(2.42 * s, .035 * s, .55 * s), cmul(plaster, .96), 0, ["plinth"]);
    // The centre bay is left twice as wide as the outer two: the gap between the
    // inner pair is the aperture the tank is read through.
    const columns = [[-2.10, "outer-left"], [-1.16, "left"], [1.16, "right"], [2.10, "outer-right"]] as const;
    for (const [offset, name] of columns) {
      const x = offset * s;
      b.box(`column-${name}/base`, "stone-column", V(x, Y(.31), colZ), V(.20 * s, .06 * s, .20 * s), cmul(plaster, .82), 0, ["column"]);
      b.cylinder(`column-${name}/shaft`, "stone-column", V(x, Y(.845), colZ), .135 * s, .475 * s, plaster, 0, ["column"]);
      b.box(`column-${name}/capital`, "stone-column", V(x, Y(1.375), colZ), V(.195 * s, .055 * s, .195 * s), cmul(plaster, .94), 0, ["column"]);
      b.box(`column-${name}/bracket`, "stone-column", V(x, Y(1.465), colZ), V(.30 * s, .035 * s, .185 * s), cmul(plaster, .88), 0, ["column"]);
    }
    b.box("colonnade/architrave", "stone-lintel", V(0, Y(1.545), colZ), V(2.55 * s, .045 * s, .165 * s), cmul(plaster, .90), 0, ["colonnade"]);
    b.box("colonnade/cornice", "stone-lintel", V(0, Y(1.635), colZ), V(2.70 * s, .045 * s, .225 * s), C(.87, .855, .825), 0, ["colonnade"]);
    // The sunlit wall beyond the colonnade. Bright enough to read as blown-out
    // noon, and it rakes forward between the shafts onto the free surface.
    b.box("sun-wall/glare", "sun-wash", V(0, Y(.86), colZ - .40 * s), V(2.45 * s, .74 * s, .04 * s), C(1, .93, .79), .55, ["fixture", "light", "emits-positive-z"]);

    // --- Stepped kerb around the tank ---------------------------------------
    // Two concentric rings, held off the container faces by a deliberate gap so
    // the water volume stays the water's. The coping is the brightest stone in
    // the courtyard: it is the value the eye lands on.
    const gap = .06 * s, cope = .17 * s, copeH = .075 * s;
    const copeInX = cw + gap, copeInZ = cd + gap;
    const copeColor = C(.87, .86, .83);
    for (const i of [-1, 1]) {
      b.box(`tank-coping/${i < 0 ? "west" : "east"}`, "stone-coping", V(i * (copeInX + cope * .5), y0 + copeH, 0), V(cope * .5, copeH, copeInZ + cope), copeColor, 0, ["plinth"]);
      b.box(`tank-coping/${i < 0 ? "north" : "south"}`, "stone-coping", V(0, y0 + copeH, i * (copeInZ + cope * .5)), V(copeInX, copeH, cope * .5), cmul(copeColor, .94), 0, ["plinth"]);
    }
    const stepW = .28 * s, stepH = .045 * s;
    const stepInX = copeInX + cope, stepInZ = copeInZ + cope;
    for (const i of [-1, 1]) {
      b.box(`tank-step/${i < 0 ? "west" : "east"}`, "plinth-step", V(i * (stepInX + stepW * .5), y0 + stepH, 0), V(stepW * .5, stepH, stepInZ + stepW), cmul(plaster, .90), 0, ["plinth"]);
      b.box(`tank-step/${i < 0 ? "north" : "south"}`, "plinth-step", V(0, y0 + stepH, i * (stepInZ + stepW * .5)), V(stepInX, stepH, stepW * .5), cmul(plaster, .84), 0, ["plinth"]);
    }

    // --- Rill: a shallow runnel entering from the west, aimed at the tank ----
    const rillZ = -.10 * s, rillHalf = .13 * s;
    const rillWest = -2.30 * s, rillEast = -(cw + .16 * s);
    const rillX = (rillWest + rillEast) * .5, rillLen = (rillEast - rillWest) * .5;
    // A near-mirror bed: the one glossy surface outside the tank, so the runnel
    // reads as standing water even where none is simulated.
    b.box("rill/channel", "rill-glass", V(rillX, Y(.018), rillZ), V(rillLen, .018 * s, rillHalf), C(.107, .112, .118), 0, ["rill"]);
    for (const i of [-1, 1]) b.box(`rill/kerb-${i < 0 ? "north" : "south"}`, "stone-coping", V(rillX, Y(.038), rillZ + i * (rillHalf + .065 * s)), V(rillLen, .038 * s, .065 * s), cmul(plaster, .96), 0, ["rill"]);
    for (let i = 0; i < 3; i++) b.box(`rill/weir-${i + 1}`, "stone-coping", V(rillWest + (.42 + .52 * i) * s, Y(.032), rillZ), V(.035 * s, .032 * s, rillHalf), cmul(plaster, .86), 0, ["rill"]);
    b.box("rill/spout-block", "stone-plinth", V(-2.52 * s, Y(.28), rillZ), V(.20 * s, .28 * s, .30 * s), cmul(plaster, .94), 0, ["rill"]);
    b.box("rill/spout-mouth", "rill-glass", V(-2.34 * s, Y(.36), rillZ), V(.05 * s, .075 * s, .095 * s), C(.070, .073, .077), 0, ["rill"]);
    b.box("rill/tank-mouth", "stone-coping", V(rillEast - .06 * s, Y(.055), rillZ), V(.08 * s, .055 * s, rillHalf + .06 * s), cmul(plaster, .90), 0, ["rill"]);

    // --- Low parapet along the west edge, split for the rill to pass ---------
    const parX = -2.20 * s;
    const parapet = [[-1.195, .755, "north"], [1.095, .855, "south"]] as const;
    for (const [offset, half, name] of parapet) {
      b.box(`parapet/${name}`, "stone-parapet", V(parX, Y(.23), offset * s), V(.115 * s, .23 * s, half * s), cmul(plaster, .98), 0, ["parapet"]);
      b.box(`parapet/${name}-coping`, "stone-coping", V(parX, Y(.495), offset * s), V(.155 * s, .035 * s, (half + .02) * s), C(.88, .87, .84), 0, ["parapet"]);
    }
    for (const i of [-1, 1]) b.box(`parapet/pier-${i < 0 ? "north" : "south"}`, "stone-parapet", V(parX, Y(.31), rillZ + i * .40 * s), V(.16 * s, .31 * s, .10 * s), cmul(plaster, .92), 0, ["parapet"]);

    // --- Stone bench, backed against the bottom step ------------------------
    const benchLen = 1.05 * s, benchX = -(cw + .22 * s) - benchLen * .5, benchZ = -.44 * s;
    b.box("bench/seat", "stone-seat", V(benchX, Y(.285), benchZ), V(benchLen * .5, .055 * s, .19 * s), cmul(plaster, .99), 0, ["bench"]);
    for (const i of [-1, 1]) b.box(`bench/leg-${i < 0 ? "left" : "right"}`, "stone-seat", V(benchX + i * .40 * s, Y(.115), benchZ), V(.055 * s, .115 * s, .155 * s), cmul(plaster, .74), 0, ["bench"]);
    b.box("bench/back", "stone-seat", V(benchX, Y(.50), benchZ - .155 * s), V(benchLen * .5, .16 * s, .04 * s), cmul(plaster, .90), 0, ["bench"]);

    // --- Citrus in big chalk-white pots -------------------------------------
    // Foliage is the darkest mass in the courtyard, so the trees read as
    // silhouette against the sun wall instead of as colour.
    const leaf = C(.345, .36, .325);
    const potWhite = C(.845, .83, .80);
    const citX = 1.32 * s, citZ = -.62 * s;
    b.cylinder("citrus/pot", "plaster-pot", V(citX, Y(.30), citZ), .34 * s, .30 * s, potWhite, 0, ["pot", "plant"]);
    b.cylinder("citrus/pot-rim", "plaster-pot", V(citX, Y(.565), citZ), .37 * s, .045 * s, C(.89, .88, .85), 0, ["pot", "plant"]);
    b.cylinder("citrus/pot-foot", "plaster-pot", V(citX, Y(.045), citZ), .30 * s, .045 * s, C(.66, .65, .63), 0, ["pot", "plant"]);
    b.cylinder("citrus/trunk", "tree-trunk", V(citX, Y(.83), citZ), .05 * s, .26 * s, C(.30, .29, .27), 0, ["tree", "plant"]);
    b.ellipsoid("citrus/canopy-main", "leaf-foliage", V(citX, Y(1.42), citZ), V(.52 * s, .40 * s, .44 * s), leaf, 0, ["tree", "plant"]);
    b.ellipsoid("citrus/canopy-left", "leaf-foliage", V(citX - .34 * s, Y(1.27), citZ - .06 * s), V(.28 * s, .26 * s, .27 * s), cmul(leaf, .86), 0, ["tree", "plant"]);
    b.ellipsoid("citrus/canopy-right", "leaf-foliage", V(citX + .31 * s, Y(1.52), citZ + .05 * s), V(.26 * s, .24 * s, .25 * s), cmul(leaf, .94), 0, ["tree", "plant"]);
    // Hand-authored scatter: the one warm accent allowed, and only because it
    // is emissive rather than pigment.
    const fruit = [[-.22, 1.30, .34, "left"], [.18, 1.55, .28, "right"], [-.38, 1.50, .10, "high"], [.34, 1.20, .22, "low"]] as const;
    for (const [dx, y, dz, name] of fruit) {
      b.ellipsoid(`citrus/fruit-${name}`, "fruit", V(citX + dx * s, Y(y), citZ + dz * s), V(.05 * s, .05 * s, .05 * s), C(.95, .74, .42), .10, ["fruit", "plant", "emissive-surface-only"]);
    }
    const cit2X = -1.70 * s, cit2Z = .62 * s;
    b.cylinder("citrus-west/pot", "plaster-pot", V(cit2X, Y(.25), cit2Z), .27 * s, .25 * s, cmul(potWhite, .95), 0, ["pot", "plant"]);
    b.cylinder("citrus-west/pot-rim", "plaster-pot", V(cit2X, Y(.475), cit2Z), .30 * s, .04 * s, C(.88, .87, .84), 0, ["pot", "plant"]);
    b.cylinder("citrus-west/pot-foot", "plaster-pot", V(cit2X, Y(.04), cit2Z), .24 * s, .04 * s, C(.64, .63, .61), 0, ["pot", "plant"]);
    b.cylinder("citrus-west/trunk", "tree-trunk", V(cit2X, Y(.68), cit2Z), .042 * s, .21 * s, C(.29, .28, .265), 0, ["tree", "plant"]);
    b.ellipsoid("citrus-west/canopy", "leaf-foliage", V(cit2X, Y(1.12), cit2Z), V(.34 * s, .29 * s, .32 * s), cmul(leaf, .92), 0, ["tree", "plant"]);
    b.ellipsoid("citrus-west/canopy-top", "leaf-foliage", V(cit2X + .12 * s, Y(1.34), cit2Z - .05 * s), V(.20 * s, .18 * s, .19 * s), leaf, 0, ["tree", "plant"]);
    b.ellipsoid("citrus-west/fruit", "fruit", V(cit2X + .16 * s, Y(1.06), cit2Z + .22 * s), V(.045 * s, .045 * s, .045 * s), C(.95, .74, .42), .10, ["fruit", "plant", "emissive-surface-only"]);

    // --- Water jars, near-right, standing in for the deleted lens overlay ----
    const jarWhite = C(.825, .81, .785);
    b.ellipsoid("jar/body", "plaster-pot", V(1.98 * s, Y(.36), .26 * s), V(.31 * s, .35 * s, .31 * s), jarWhite, 0, ["pot"]);
    b.cylinder("jar/neck", "plaster-pot", V(1.98 * s, Y(.72), .26 * s), .115 * s, .085 * s, cmul(jarWhite, .90), 0, ["pot"]);
    b.cylinder("jar/lip", "plaster-pot", V(1.98 * s, Y(.815), .26 * s), .155 * s, .035 * s, C(.88, .87, .84), 0, ["pot"]);
    b.ellipsoid("jar-small/body", "plaster-pot", V(2.40 * s, Y(.24), -.30 * s), V(.22 * s, .24 * s, .22 * s), cmul(jarWhite, .94), 0, ["pot"]);
    b.cylinder("jar-small/lip", "plaster-pot", V(2.40 * s, Y(.50), -.30 * s), .12 * s, .035 * s, C(.86, .85, .82), 0, ["pot"]);

    // --- Slatted pergola ----------------------------------------------------
    // Held a clear margin above the container so the water volume stays empty,
    // and run in Z so the noon stripes converge on the tank. The far girder
    // lands on the colonnade cornice; the near pair of posts stands off the
    // camera side for depth.
    const deckY = Math.max(y0 + 1.72 * s, scene.container.height_m + .55 * s);
    const postTop = deckY - .075 * s;
    const timber = C(.76, .745, .715);
    for (const i of [-1, 1]) {
      b.box(`pergola/post-${i < 0 ? "west" : "east"}`, "wood-pergola", V(i * 1.98 * s, (y0 + postTop) * .5, 1.24 * s), V(.075 * s, (postTop - y0) * .5, .075 * s), cmul(timber, .86), 0, ["pergola"]);
    }
    b.box("pergola/girder-north", "wood-pergola", V(0, deckY, -1.30 * s), V(2.15 * s, .075 * s, .085 * s), cmul(timber, .92), 0, ["pergola"]);
    b.box("pergola/girder-south", "wood-pergola", V(0, deckY, 1.24 * s), V(2.15 * s, .075 * s, .085 * s), cmul(timber, .92), 0, ["pergola"]);
    b.box("pergola/purlin", "wood-pergola", V(0, deckY, -.10 * s), V(2.15 * s, .075 * s, .075 * s), cmul(timber, .88), 0, ["pergola"]);
    for (let i = 0; i < 13; i++) {
      b.box(`pergola/slat-${i + 1}`, "wood-pergola", V((-1.80 + .30 * i) * s, deckY + .13 * s, -.03 * s), V(.05 * s, .055 * s, 1.31 * s), timber, 0, ["pergola"]);
    }

    // --- Remaining light: cool sky over the slats, hot bounce off the stone --
    b.box("sky/wash", "sun-wash", V(0, deckY + .95 * s, -.10 * s), V(2.20 * s, .03 * s, 1.75 * s), C(.79, .855, 1), .30, ["fixture", "light", "emits-negative-y"]);
    b.box("sun-patch/east", "sun-wash", V(1.62 * s, Y(.022), .10 * s), V(.80 * s, .012 * s, 1.15 * s), C(1, .90, .74), .24, ["fixture", "light", "emits-positive-y"]);
    b.box("sun-wall/west-bounce", "sun-wash", V(-2.62 * s, Y(.90), -.20 * s), V(.04 * s, .62 * s, 1.55 * s), C(.99, .95, .87), .26, ["fixture", "light", "emits-positive-x"]);
    return shell;
  },
};
