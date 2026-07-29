import {
  addRoomShell,
  C,
  cmul,
  V,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * Submerged research station: a white painted pressure hull built to hold an
 * enormous mass of water back, so the engineering itself is what tells you how
 * deep the tank is. Everything is monochrome — value, not hue — and the only
 * saturated colour in the room is instrument emission.
 *
 * Two presets share this set and they disagree violently about proportion:
 * "Ocean · rolling wave" is a broad, shallow 8 m basin (s = 8) and
 * "Deep-water A/B" is a 20 m column (s = 20). So the composition is authored as
 * two overlapping reads: a wide low band of deck furniture, coaming and gantry
 * that frames the free surface, and a tall vertical repetition — ribs,
 * stringers, depth ticks, pipe collars — that keeps climbing well past the
 * water line. Nothing is in absolute metres; everything is a multiple of `s`.
 */
export const researchStationScenery: EnvironmentSceneryModule = {
  id: "research-station",
  build: (b, context) => {
    // Bright hull, dark shadow. The old scene painted this room at ~0.03
    // linear and every form in it died; the walls now carry the light and the
    // deck stays dark so the water reads against it.
    const shell = addRoomShell(b, context, {
      materialModel: "station",
      colors: { floor: C(.215, .223, .236), wall: C(.600, .616, .636), ceiling: C(.395, .407, .424) },
    });
    const { s, floorY_m: y0, roomHalf_m: roomHalf } = context;
    const xWall = roomHalf.x, zBack = -roomHalf.z;

    // Near-neutral greys, cool by a hair. Value alone separates them.
    const ribWhite = C(.760, .776, .796);
    const hullPaint = C(.655, .670, .690);
    const plate = C(.495, .508, .527);
    const steel = C(.315, .328, .345);
    const steelDark = C(.140, .149, .163);
    const paint = C(.860, .872, .884);
    const voidBlack = C(.028, .034, .042);
    // Emission is the one licensed colour: a cold instrument cyan, plus the
    // near-white work floods that actually rake the surface.
    const glow = C(.06, .48, .58);
    const beacon = C(.10, .65, .72);
    const flood = C(.70, .86, .95);

    // Deterministic value scatter: an integer hash of the loop index, never a
    // clock and never Math.random, so every rebuild is byte-identical.
    const jitter = (n: number) => (((n * 2654435761) >>> 0) % 1000) / 1000;

    // ---------------------------------------------------------------- hull
    // Heavy pressure ribs marching away down both side walls. This is the
    // repetition that says "engineered against pressure"; it also gives the
    // 20 m preset something to measure height against, so each rib runs from
    // the deck to 3.1 s.
    for (const side of [-1, 1]) {
      const tag = side < 0 ? "port" : "stbd";
      for (let i = 0; i < 6; i++) {
        const z = (-1.78 + i * .63) * s;
        b.box(`hull/rib-${tag}-${i + 1}`, "hull-rib", V(side * (xWall - .07 * s), y0 + 1.55 * s, z),
          V(.07 * s, 1.55 * s, .105 * s), cmul(ribWhite, .90 + .16 * jitter(i + (side < 0 ? 0 : 7))), 0, ["rib", "hull"]);
      }
      b.box(`hull/rib-foot-${tag}`, "hull-rib", V(side * (xWall - .16 * s), y0 + .19 * s, 0),
        V(.16 * s, .19 * s, roomHalf.z * .92), cmul(hullPaint, .74), 0, ["rib", "hull"]);
      b.box(`hull/rib-cap-${tag}`, "hull-stringer", V(side * (xWall - .12 * s), y0 + 3.16 * s, 0),
        V(.12 * s, .075 * s, roomHalf.z * .92), hullPaint, 0, ["rib", "hull"]);
    }

    // The back wall is the backdrop the water is read against: two flanking
    // ribs, three stringer bands climbing past the port, one kick plate.
    for (const side of [-1, 1]) {
      b.box(`hull/back-rib-${side < 0 ? "port" : "stbd"}`, "hull-rib", V(side * 1.36 * s, y0 + 1.58 * s, zBack + .095 * s),
        V(.095 * s, 1.58 * s, .095 * s), ribWhite, 0, ["rib", "hull"]);
    }
    const stringerY = [.40, 2.72, 3.12];
    stringerY.forEach((y, i) => b.box(`hull/back-stringer-${i + 1}`, "hull-stringer", V(0, y0 + y * s, zBack + .07 * s),
      V(1.90 * s, .07 * s, .06 * s), cmul(hullPaint, .96 - .08 * i), 0, ["hull"]));
    b.box("hull/back-kick", "hull-stringer", V(0, y0 + .16 * s, zBack + .09 * s), V(1.98 * s, .16 * s, .08 * s), cmul(hullPaint, .70), 0, ["hull"]);

    // ------------------------------------------------- observation port
    // The engine authors an elliptical thick-glass lens at exactly
    // (0, floorY + 1.55 s, zBack + .018 s) with radii (.66 s, .39 s), so the
    // whole assembly is composed around that fixed opening: a bolted flange
    // plate, a bright elliptical rim, a black void read through the lens, four
    // heavy flange bars and a ring of chunky bolt heads. It is deliberately the
    // largest single object in the room.
    const portY = y0 + 1.55 * s, portZ = zBack;
    b.box("observation-port/backing", "metal-frame", V(0, portY, portZ + .004 * s), V(1.10 * s, .76 * s, .003 * s), plate, 0, ["porthole", "frame"]);
    b.ellipsoid("observation-port/rim", "metal-frame", V(0, portY, portZ + .012 * s), V(.86 * s, .56 * s, .012 * s), ribWhite, 0, ["porthole", "frame", "fixture"]);
    b.ellipsoid("observation-port/void", "port-void", V(0, portY, portZ + .020 * s), V(.620 * s, .365 * s, .010 * s), voidBlack, 0, ["porthole"]);
    b.box("observation-port/frame-left", "metal-frame", V(-1.14 * s, portY, portZ + .028 * s), V(.10 * s, .94 * s, .028 * s), plate, 0, ["porthole", "frame", "fixture"]);
    b.box("observation-port/frame-right", "metal-frame", V(1.14 * s, portY, portZ + .028 * s), V(.10 * s, .94 * s, .028 * s), plate, 0, ["porthole", "frame", "fixture"]);
    b.box("observation-port/frame-bottom", "metal-frame", V(0, portY - .84 * s, portZ + .028 * s), V(1.24 * s, .10 * s, .028 * s), plate, 0, ["porthole", "frame", "fixture"]);
    b.box("observation-port/frame-top", "metal-frame", V(0, portY + .84 * s, portZ + .028 * s), V(1.24 * s, .10 * s, .028 * s), plate, 0, ["porthole", "frame", "fixture"]);
    for (let i = 0; i < 10; i++) {
      const a = i * Math.PI / 5;
      b.ellipsoid(`observation-port/bolt-${i + 1}`, "metal-frame",
        V(Math.cos(a) * .760 * s, portY + Math.sin(a) * .470 * s, portZ + .046 * s),
        V(.042 * s, .042 * s, .042 * s), cmul(ribWhite, .88 + .18 * jitter(i)), 0, ["porthole", "frame", "fixture"]);
    }

    // --------------------------------------------------- depth gauge
    // Painted depth ticks climbing the back wall past the port. Nine stations
    // over three metres of scale height: the cheapest possible way to tell the
    // viewer how far down the water goes, and it keeps working when s = 20.
    b.box("depth-gauge/spine", "gauge-mark", V(-1.79 * s, y0 + 1.60 * s, zBack + .05 * s), V(.022 * s, 1.55 * s, .05 * s), paint, 0, ["gauge"]);
    for (let i = 0; i < 9; i++) {
      const major = i % 2 === 0, len = (major ? .155 : .085) * s;
      b.box(`depth-gauge/tick-${i + 1}`, "gauge-mark", V(-1.768 * s + len, y0 + (.22 + .355 * i) * s, zBack + .05 * s),
        V(len, (major ? .030 : .020) * s, .045 * s), major ? paint : cmul(paint, .78), 0, ["gauge"]);
    }

    // ------------------------------------------------------ gantry walkway
    // A catwalk crossing behind the tank with a handrail, hung floods and an
    // access ladder. It gives the composition a hard horizontal above the free
    // surface and — because its two floods point straight down — it is also the
    // light that rakes the water.
    const gantryY = y0 + .62 * s, railZ = -.50 * s;
    b.box("gantry/deck", "gantry-steel", V(0, gantryY, -.78 * s), V(1.95 * s, .035 * s, .30 * s), steel, 0, ["gantry"]);
    b.box("gantry/edge-beam", "gantry-steel", V(0, gantryY - .055 * s, railZ - .03 * s), V(1.95 * s, .055 * s, .045 * s), cmul(steel, .82), 0, ["gantry"]);
    b.box("gantry/mid-rail", "gantry-steel", V(0, gantryY + .18 * s, railZ), V(1.95 * s, .022 * s, .022 * s), cmul(ribWhite, .82), 0, ["gantry", "rail"]);
    b.box("gantry/top-rail", "gantry-steel", V(0, gantryY + .34 * s, railZ), V(1.95 * s, .030 * s, .030 * s), ribWhite, 0, ["gantry", "rail"]);
    for (let i = 0; i < 6; i++) {
      b.cylinder(`gantry/post-${i + 1}`, "gantry-steel", V((-1.75 + i * .70) * s, gantryY + .17 * s, railZ), .030 * s, .17 * s, cmul(ribWhite, .74), 0, ["gantry", "rail"]);
    }
    for (const side of [-1, 1]) {
      b.cylinder(`gantry/leg-${side < 0 ? "port" : "stbd"}`, "gantry-steel", V(side * 1.80 * s, y0 + .29 * s, -.78 * s), .075 * s, .29 * s, steel, 0, ["gantry"]);
    }
    for (const [i, x] of [-1.62, -1.44].entries()) {
      b.cylinder(`gantry/ladder-stile-${i + 1}`, "ladder-steel", V(x * s, y0 + .34 * s, -1.22 * s), .020 * s, .34 * s, cmul(ribWhite, .70), 0, ["gantry", "ladder"]);
    }
    for (let i = 0; i < 4; i++) {
      b.box(`gantry/ladder-rung-${i + 1}`, "ladder-steel", V(-1.53 * s, y0 + (.16 + .17 * i) * s, -1.22 * s), V(.11 * s, .014 * s, .014 * s), cmul(ribWhite, .70), 0, ["gantry", "ladder"]);
    }
    for (const side of [-1, 1]) {
      b.box(`gantry/flood-${side < 0 ? "port" : "stbd"}`, "emissive-fixture", V(side * 1.00 * s, gantryY - .064 * s, -.70 * s),
        V(.34 * s, .026 * s, .13 * s), flood, 2.4, ["gantry", "fixture", "light", "emits-negative-y"]);
    }

    // ------------------------------------------------------- ballast plant
    // Two trunk risers with bolted collars at three heights, a header across
    // the top and hand wheels. The collars are the vertical tick marks that
    // make the 20 m preset legible from any camera height.
    for (const side of [-1, 1]) {
      const tag = side < 0 ? "port" : "stbd", x = side * 1.72 * s;
      b.cylinder(`ballast/riser-${tag}`, "metal-pipe", V(x, y0 + 1.55 * s, -1.72 * s), .115 * s, 1.55 * s, cmul(hullPaint, .88), 0, ["pipe"]);
      b.box(`ballast/riser-base-${tag}`, "metal-frame", V(x, y0 + .055 * s, -1.72 * s), V(.24 * s, .055 * s, .24 * s), plate, 0, ["pipe", "flange"]);
      for (const [i, y] of [.36, 1.46, 2.56].entries()) {
        b.cylinder(`ballast/collar-${tag}-${i + 1}`, "metal-frame", V(x, y0 + y * s, -1.72 * s), .175 * s, .050 * s, plate, 0, ["pipe", "flange", "fixture"]);
      }
      b.cylinder(`ballast/wheel-${tag}`, "metal-frame", V(x, y0 + 2.88 * s, -1.72 * s), .235 * s, .028 * s, cmul(ribWhite, .84), 0, ["pipe", "fixture"]);
    }
    b.box("ballast/header", "metal-pipe", V(0, y0 + 3.06 * s, -1.72 * s), V(1.72 * s, .085 * s, .085 * s), cmul(hullPaint, .88), 0, ["pipe"]);

    // ------------------------------------------------------- deck coaming
    // A heavy bolted curb standing off the tank on all four sides. It reads as
    // the lip that the water is retained behind and, more importantly, it draws
    // a rectangle on the deck that points the eye straight at the surface.
    for (const side of [-1, 1]) {
      b.box(`coaming/${side < 0 ? "near" : "far"}`, "deck-coaming", V(0, y0 + .07 * s, side * .60 * s), V(1.05 * s, .07 * s, .17 * s), cmul(plate, 1.02), 0, ["deck"]);
      b.box(`coaming/${side < 0 ? "port" : "stbd"}`, "deck-coaming", V(side * .82 * s, y0 + .07 * s, 0), V(.17 * s, .07 * s, .77 * s), cmul(plate, .92), 0, ["deck"]);
    }

    // ---------------------------------------------------- control station
    // The two consoles flank the port and face the water, so their cyan glow is
    // thrown forward across the surface rather than into the wall.
    for (const i of [-1, 1]) {
      const side = i < 0 ? "left" : "right", x = i * 1.05 * s;
      b.box(`console-${side}/cabinet`, "metal-console", V(x, y0 + .42 * s, -1.80 * s), V(.38 * s, .42 * s, .30 * s), steelDark, 0, ["console"]);
      b.box(`console-${side}/monitor`, "monitor-glass", V(x, y0 + .78 * s, -1.48 * s), V(.30 * s, .17 * s, .020 * s), glow, .30, ["console", "monitor", "light", "emits-positive-z"]);
      b.cylinder(`console-${side}/pipe`, "metal-pipe", V(x + i * .46 * s, y0 + 1.30 * s, -1.90 * s), .058 * s, 1.30 * s, cmul(hullPaint, .80), 0, ["console", "pipe"]);
    }
    b.box("equipment-case/body", "equipment-case", V(-1.55 * s, y0 + .27 * s, .58 * s), V(.34 * s, .27 * s, .26 * s), cmul(steel, .78), 0, ["equipment"]);
    b.box("equipment-case/lid", "equipment-case", V(-1.55 * s, y0 + .555 * s, .58 * s), V(.29 * s, .020 * s, .21 * s), paint, 0, ["equipment"]);

    // Beacon domes high on the back wall, above the port: three cold points
    // that keep the top of the hull from going flat black.
    for (let i = -1; i <= 1; i++) {
      b.ellipsoid(`indicator-${i + 2}`, "emissive-fixture", V(i * .98 * s, y0 + 2.68 * s, zBack + .11 * s), V(.075 * s, .075 * s, .075 * s), beacon, .40, ["fixture", "light"]);
    }

    // ------------------------------------------------------ wave probe
    // A slim instrument mast standing on the deck beside the tank, with its
    // sensor arm reaching out over the surface. It is the nearest thing to the
    // camera in both presets, so it does the parallax work the deleted
    // screen-space overlay used to fake.
    const probe = { x: .82 * s, z: .62 * s };
    b.box("wave-probe/base", "metal-frame", V(probe.x, y0 + .06 * s, probe.z), V(.22 * s, .06 * s, .22 * s), plate, 0, ["instrument"]);
    b.cylinder("wave-probe/mast", "instrument-mast", V(probe.x, y0 + 1.16 * s, probe.z), .045 * s, 1.10 * s, cmul(ribWhite, .86), 0, ["instrument"]);
    for (const [i, y] of [.62, 1.32, 1.94].entries()) {
      b.cylinder(`wave-probe/collar-${i + 1}`, "metal-frame", V(probe.x, y0 + y * s, probe.z), .098 * s, .028 * s, plate, 0, ["instrument", "fixture"]);
    }
    b.box("wave-probe/arm", "instrument-mast", V(probe.x - .21 * s, y0 + 2.18 * s, probe.z), V(.21 * s, .022 * s, .022 * s), cmul(ribWhite, .86), 0, ["instrument"]);
    b.ellipsoid("wave-probe/head", "emissive-fixture", V(probe.x, y0 + 2.30 * s, probe.z), V(.062 * s, .062 * s, .062 * s), beacon, .80, ["instrument", "fixture", "light"]);
    return shell;
  },
};
