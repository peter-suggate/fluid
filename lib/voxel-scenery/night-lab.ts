import {
  addRoomShell,
  C,
  cmul,
  V,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * Night research laboratory: the tank is a specimen under observation. Every
 * prop is measuring kit aimed at the water — a camera and ring light on a
 * tripod square to the glass, a checkerboard calibration target standing behind
 * it, a laser-sheet head raking the free surface, instrument racks and cable
 * runs off to the side, and a city window spilling cool light in from behind.
 *
 * The palette is deliberately monochrome: only the emitters carry hue, so the
 * one warm task lamp reads against the cool pendants, monitor, laser and city.
 */
export const nightLabScenery: EnvironmentSceneryModule = {
  id: "night-lab",
  build: (b, context) => {
    const { s, floorY_m: floorY, roomHalf_m: roomHalf, scene } = context;
    const shell = addRoomShell(b, context, {
      materialModel: "night-lab",
      colors: { floor: C(.172, .178, .186), wall: C(.395, .400, .410), ceiling: C(.455, .458, .462) },
      // The raster room shader has a city window in its back wall. A single
      // union-only wall box concealed the authored thin-glass pane and forced
      // the whole night-lab SVO path to raster. Four ordinary analytic boxes
      // preserve the same wall while leaving an exact rectangular opening.
      backWall: (builder, spec) => {
        const { colors: c, roomHalf_m: half, floorY_m: y0, t, centerY_m, shellThickness_m: thickness } = spec;
        const openingHalfWidth = Math.min(1.62 * spec.s, half.x - thickness);
        const openingHalfHeight = Math.min(.55 * spec.s, half.y - thickness);
        const openingCenterY = y0 + 1.60 * spec.s;
        const openingBottom = Math.max(y0, openingCenterY - openingHalfHeight);
        const openingTop = Math.min(y0 + 2 * half.y, openingCenterY + openingHalfHeight);
        const sideHalfWidth = .5 * (half.x - openingHalfWidth);
        const sideCenterX = openingHalfWidth + sideHalfWidth;
        const bottomHalfHeight = .5 * (openingBottom - y0);
        const topHalfHeight = .5 * (y0 + 2 * half.y - openingTop);
        builder.box("shell/wall-back-left", "shell-wall", V(-sideCenterX, centerY_m, -half.z - t), V(sideHalfWidth, half.y, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
        builder.box("shell/wall-back-right", "shell-wall", V(sideCenterX, centerY_m, -half.z - t), V(sideHalfWidth, half.y, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
        builder.box("shell/wall-back-bottom", "shell-wall", V(0, y0 + bottomHalfHeight, -half.z - t), V(openingHalfWidth, bottomHalfHeight, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
        builder.box("shell/wall-back-top", "shell-wall", V(0, openingTop + topHalfHeight, -half.z - t), V(openingHalfWidth, topHalfHeight, t), c.wall, 0, ["shell", "wall", "window-cutout"], true);
      },
    });

    // ---- palette -----------------------------------------------------------
    // Near-neutral greys separated by value alone; hue is reserved for emitters.
    const ink = C(.042, .044, .048);
    const graphite = C(.085, .089, .095);
    const slate = C(.150, .156, .163);
    const steel = C(.250, .258, .268);
    const pewter = C(.355, .360, .366);
    const ash = C(.470, .472, .474);
    const bone = C(.640, .640, .632);
    const chalk = C(.815, .813, .802);
    const paper = C(.915, .910, .895);
    // The one warm source in the room, and the cool family it plays against.
    const tungsten = C(1.00, .585, .235);
    const coolBar = C(.520, .690, 1.00);
    const displayCool = C(.230, .455, .720);
    const ringCool = C(.760, .855, 1.00);
    const sheetCool = C(.560, .800, 1.00);

    // ---- frame -------------------------------------------------------------
    const th = { x: scene.container.width_m * .5 + .30 * s, y: scene.container.depth_m * .5 + .26 * s };
    const zb = -roomHalf.z + .36 * s, ceilY = floorY + 2 * roomHalf.y;
    // Everything on the bench keeps clear of the water: the tank interior is
    // |x| < w/2, |z| < d/2, 0 < y < h, so bench props live behind z = -.45 s,
    // in front of z = +.37 s, or outboard of the tank walls.
    const zFront = .50 * s;

    // ---- bench -------------------------------------------------------------
    b.box("desk/top", "lab-bench", V(0, -.021 * s, 0), V(th.x, .019 * s, th.y), C(.600, .600, .592), 0, ["desk", "bench"]);
    b.box("desk/apron", "lab-bench", V(0, -.074 * s, 0), V(th.x - .07 * s, .034 * s, th.y - .07 * s), graphite, 0, ["desk", "bench"]);
    for (const i of [-1, 1]) for (const j of [-1, 1]) b.box(`desk/leg-${i < 0 ? "l" : "r"}${j < 0 ? "f" : "b"}`, "steel-frame", V(i * (th.x - .10 * s), floorY + .34 * s, j * (th.y - .10 * s)), V(.027 * s, .34 * s, .027 * s), steel, 0, ["desk", "bench"]);
    b.box("desk/lower-shelf", "steel-frame", V(0, floorY + .16 * s, 0), V(th.x - .13 * s, .013 * s, th.y - .13 * s), slate, 0, ["desk", "bench"]);
    b.box("desk/controller", "instrument", V(-.35 * th.x, floorY + .235 * s, .15 * th.y), V(.15 * s, .062 * s, .115 * s), pewter, 0, ["desk", "instrument"]);
    b.box("desk/mat", "floor-mat", V(0, floorY + .006 * s, .18 * s), V(1.15 * s, .006 * s, .95 * s), C(.115, .119, .124), 0, ["desk", "floor"]);

    // Warm task lamp, moved behind the tank so it back-rakes the free surface
    // and rims the calibration target. It is the only warm light in the room.
    const lamp = V(-(th.x - .17 * s), 0, -(th.y - .24 * s));
    b.cylinder("desk-lamp/base", "metal-fixture", V(lamp.x, .010 * s, lamp.z), .085 * s, .013 * s, ink, 0, ["desk", "fixture"]);
    b.cylinder("desk-lamp/stem", "metal-fixture", V(lamp.x, .31 * s, lamp.z), .012 * s, .29 * s, graphite, 0, ["desk", "fixture"]);
    b.ellipsoid("desk-lamp/shade", "metal-fixture", V(lamp.x, .63 * s, lamp.z), V(.104 * s, .076 * s, .104 * s), C(.050, .051, .054), 0, ["desk", "fixture"]);
    b.ellipsoid("desk-lamp/bulb", "emissive-glass", V(lamp.x, .580 * s, lamp.z), V(.048 * s, .048 * s, .048 * s), tungsten, 3.4, ["desk", "fixture", "light"]);

    const stoolX = th.x + .45 * s;
    b.cylinder("stool/seat", "stool-seat", V(stoolX, floorY + .47 * s, .10 * s), .17 * s, .024 * s, bone, 0, ["stool", "chair"]);
    b.cylinder("stool/post", "steel-frame", V(stoolX, floorY + .235 * s, .10 * s), .024 * s, .215 * s, cmul(steel, .8), 0, ["stool", "chair"]);
    b.cylinder("stool/base", "steel-frame", V(stoolX, floorY + .02 * s, .10 * s), .145 * s, .014 * s, cmul(steel, .6), 0, ["stool", "chair"]);

    // ---- back counter ------------------------------------------------------
    b.box("counter/cabinet", "lab-counter", V(0, floorY + .42 * s, zb), V(1.72 * s, .42 * s, .30 * s), slate, 0, ["counter"]);
    b.box("counter/worktop", "lab-counter", V(0, floorY + .862 * s, zb), V(1.80 * s, .022 * s, .34 * s), ash, 0, ["counter"]);
    b.box("counter/monitor-stand", "metal-fixture", V(.95 * s, floorY + .93 * s, zb), V(.05 * s, .055 * s, .05 * s), ink, 0, ["counter", "instrument"]);
    b.box("counter/monitor", "monitor-frame", V(.95 * s, floorY + 1.17 * s, zb + .05 * s), V(.30 * s, .19 * s, .014 * s), C(.030, .034, .040), 0, ["counter", "instrument", "monitor"]);
    b.box("counter/monitor-screen", "monitor-glass", V(.95 * s, floorY + 1.17 * s, zb + .068 * s), V(.265 * s, .155 * s, .004 * s), displayCool, 1.35, ["counter", "instrument", "monitor", "light", "emits-positive-z"]);
    b.box("counter/keyboard", "instrument", V(.95 * s, floorY + .892 * s, zb + .22 * s), V(.19 * s, .007 * s, .07 * s), graphite, 0, ["counter", "instrument"]);
    b.cylinder("counter/instrument-a", "instrument", V(-.58 * s, floorY + .966 * s, zb), .070 * s, .082 * s, pewter, .02, ["counter", "instrument", "emissive-surface-only"]);
    b.cylinder("counter/instrument-b", "instrument", V(-.86 * s, floorY + 1.014 * s, zb + .04 * s), .046 * s, .13 * s, cmul(pewter, .86), .02, ["counter", "instrument", "emissive-surface-only"]);
    b.ellipsoid("counter/instrument-c", "instrument", V(-1.12 * s, floorY + .980 * s, zb - .02 * s), V(.088 * s, .096 * s, .088 * s), cmul(pewter, .94), .02, ["counter", "instrument", "emissive-surface-only"]);
    b.box("counter/shelf", "metal-fixture", V(-.20 * s, floorY + 1.62 * s, zb - .05 * s), V(1.22 * s, .016 * s, .17 * s), steel, 0, ["counter", "shelf"]);
    const bottleValues = [chalk, C(.300, .305, .310), bone];
    for (let i = 0; i < 3; i++) b.box(`counter/shelf-bottle-${i + 1}`, "instrument", V((-.92 + .13 * i) * s, floorY + 1.751 * s, zb - .05 * s), V(.050 * s, .115 * s, .135 * s), bottleValues[i], 0, ["counter", "shelf", "instrument"]);

    // ---- cool overhead -----------------------------------------------------
    // Two of the four troffers are hung low over the bench so the cool bars are
    // actually in frame and rake the free surface; the other two stay flush to
    // the ceiling as ambient fill.
    const pendantY = floorY + 2.14 * s;
    const troffers = [
      { key: "fixtures/troffer-left-1", c: V(-.30 * s, pendantY, .14 * s), h: V(.78 * s, .013 * s, .130 * s) },
      { key: "fixtures/troffer-left-2", c: V(-1.35 * s, ceilY - .035 * s, -.40 * s), h: V(.55 * s, .012 * s, .20 * s) },
      { key: "fixtures/troffer-right-1", c: V(.34 * s, pendantY, -.32 * s), h: V(.72 * s, .013 * s, .130 * s) },
      { key: "fixtures/troffer-right-2", c: V(1.35 * s, ceilY - .035 * s, .70 * s), h: V(.55 * s, .012 * s, .20 * s) },
    ] as const;
    for (const t of troffers) b.box(t.key, "emissive-fixture", t.c, t.h, coolBar, 2.4, ["fixture", "light", "emits-negative-y"]);
    const rodHalf = .5 * (ceilY - pendantY);
    const rods = [[-.88, .14], [.28, .14], [-.20, -.32], [.88, -.32]] as const;
    rods.forEach(([x, z], i) => b.cylinder(`rig/pendant-rod-${i + 1}`, "metal-fixture", V(x * s, pendantY + rodHalf, z * s), .010 * s, rodHalf, steel, 0, ["fixture"]));

    // ---- observation camera on its tripod, square to the tank --------------
    const camX = -.02 * s, camZ = 1.26 * s, camY = .555 * s;
    b.cylinder("rig/tripod-leg-a", "tripod-steel", V(camX - .20 * s, -.31 * s, camZ - .09 * s), .020 * s, .41 * s, steel, 0, ["rig", "instrument"]);
    b.cylinder("rig/tripod-leg-b", "tripod-steel", V(camX + .20 * s, -.31 * s, camZ - .09 * s), .020 * s, .41 * s, steel, 0, ["rig", "instrument"]);
    b.cylinder("rig/tripod-leg-c", "tripod-steel", V(camX, -.31 * s, camZ + .22 * s), .020 * s, .41 * s, steel, 0, ["rig", "instrument"]);
    b.box("rig/tripod-spreader", "tripod-steel", V(camX, -.40 * s, camZ + .015 * s), V(.215 * s, .012 * s, .185 * s), graphite, 0, ["rig", "instrument"]);
    b.cylinder("rig/tripod-column", "tripod-steel", V(camX, .24 * s, camZ), .036 * s, .16 * s, pewter, 0, ["rig", "instrument"]);
    b.box("rig/tripod-head", "tripod-steel", V(camX, .445 * s, camZ), V(.050 * s, .045 * s, .050 * s), ink, 0, ["rig", "instrument"]);
    b.box("rig/camera-body", "camera-instrument", V(camX, camY, camZ + .01 * s), V(.100 * s, .078 * s, .130 * s), C(.070, .073, .078), 0, ["rig", "instrument"]);
    b.ellipsoid("rig/camera-lens", "camera-instrument", V(camX, camY, camZ - .16 * s), V(.058 * s, .058 * s, .100 * s), ink, 0, ["rig", "instrument"]);
    b.box("rig/ring-light", "emissive-fixture", V(camX, camY, camZ - .205 * s), V(.185 * s, .185 * s, .010 * s), ringCool, 1.7, ["rig", "fixture", "light", "emits-negative-z"]);
    b.box("rig/camera-cable-a", "cable-run", V(camX + .06 * s, .40 * s, camZ + .16 * s), V(.014 * s, .100 * s, .014 * s), ink, 0, ["rig", "cable"]);
    b.box("rig/camera-cable-b", "cable-run", V(camX + .06 * s, .30 * s, camZ + .30 * s), V(.014 * s, .014 * s, .145 * s), ink, 0, ["rig", "cable"]);

    // ---- calibration target standing behind the glass ----------------------
    const boardX = -.10 * s, boardZ = -.47 * s;
    b.box("calib/board", "calibration-target", V(boardX, .55 * s, boardZ), V(.42 * s, .55 * s, .014 * s), chalk, 0, ["calibration"]);
    for (const i of [-1, 1]) b.box(`calib/foot-${i < 0 ? "left" : "right"}`, "steel-frame", V(boardX + i * .34 * s, .022 * s, boardZ - .035 * s), V(.09 * s, .022 * s, .052 * s), steel, 0, ["calibration"]);
    const tileColumns = [-.315, -.105, .105, .315] as const;
    const tileRows = [.19, .55, .91] as const;
    for (let r = 0; r < tileRows.length; r++) for (let c = 0; c < tileColumns.length; c++) {
      if ((r + c) % 2 !== 0) continue;
      b.box(`calib/tile-r${r}c${c}`, "calibration-target", V(boardX + tileColumns[c] * s, tileRows[r] * s, boardZ + .0235 * s), V(.100 * s, .170 * s, .010 * s), C(.048, .049, .051), 0, ["calibration"]);
    }

    // ---- laser-sheet head, aimed across the tank ---------------------------
    const laserX = .68 * s, laserZ = -.02 * s;
    b.cylinder("laser/post", "laser-instrument", V(laserX, .15 * s, laserZ), .045 * s, .15 * s, steel, 0, ["laser", "instrument"]);
    b.box("laser/housing", "laser-instrument", V(laserX, .40 * s, laserZ), V(.105 * s, .100 * s, .145 * s), C(.062, .065, .069), 0, ["laser", "instrument"]);
    b.box("laser/aperture", "emissive-fixture", V(laserX - .1185 * s, .38 * s, laserZ), V(.0065 * s, .085 * s, .012 * s), sheetCool, 2.4, ["laser", "fixture", "light", "emits-negative-x"]);
    for (let i = 0; i < 3; i++) b.box(`laser/fin-${i + 1}`, "steel-frame", V(laserX, .512 * s, laserZ + (i - 1) * .055 * s), V(.085 * s, .014 * s, .012 * s), pewter, 0, ["laser", "instrument"]);

    // ---- instrument racks against the left wall ----------------------------
    const racks = [
      { id: "a", x: -2.40 * s, z: -.14 * s, half: .60 * s },
      { id: "b", x: -2.40 * s, z: .66 * s, half: .52 * s },
    ] as const;
    for (const rack of racks) {
      const midY = floorY + .07 * s + rack.half;
      b.box(`rack-${rack.id}/plinth`, "steel-frame", V(rack.x, floorY + .035 * s, rack.z), V(.24 * s, .035 * s, .28 * s), ink, 0, ["rack"]);
      b.box(`rack-${rack.id}/case`, "rack-console", V(rack.x, midY, rack.z), V(.26 * s, rack.half, .30 * s), slate, 0, ["rack", "instrument"]);
      for (const i of [-1, 1]) b.box(`rack-${rack.id}/ear-${i < 0 ? "far" : "near"}`, "rack-steel-panel", V(rack.x + .272 * s, midY, rack.z + i * .245 * s), V(.016 * s, rack.half - .04 * s, .045 * s), pewter, 0, ["rack", "instrument"]);
      b.box(`rack-${rack.id}/unit-1`, "rack-steel-panel", V(rack.x + .268 * s, midY + .40 * s, rack.z), V(.012 * s, .100 * s, .235 * s), ash, 0, ["rack", "instrument"]);
      b.box(`rack-${rack.id}/unit-2`, "rack-steel-panel", V(rack.x + .268 * s, midY - .02 * s, rack.z), V(.012 * s, .130 * s, .235 * s), C(.300, .306, .312), 0, ["rack", "instrument"]);
      b.box(`rack-${rack.id}/led`, "rack-steel-panel", V(rack.x + .284 * s, midY + .40 * s, rack.z + .160 * s), V(.006 * s, .012 * s, .045 * s), C(.360, .620, 1.00), .07, ["rack", "instrument", "emissive-surface-only"]);
    }

    // ---- cable runs draping off the bench toward the racks -----------------
    const drapeZ = [-.30, .36] as const;
    drapeZ.forEach((z, k) => {
      const tag = k === 0 ? "back" : "front";
      b.box(`cable/${tag}-hang`, "cable-run", V(-.80 * s, -.16 * s, z * s), V(.020 * s, .100 * s, .020 * s), C(.055, .057, .060), 0, ["cable"]);
      b.box(`cable/${tag}-sag`, "cable-run", V(-.90 * s, -.32 * s, z * s), V(.110 * s, .020 * s, .020 * s), C(.055, .057, .060), 0, ["cable"]);
      b.box(`cable/${tag}-drop`, "cable-run", V(-1.02 * s, -.50 * s, z * s), V(.020 * s, .180 * s, .020 * s), C(.055, .057, .060), 0, ["cable"]);
      b.box(`cable/${tag}-floor`, "cable-run", V(-1.55 * s, floorY + .022 * s, z * s), V(.550 * s, .020 * s, .022 * s), C(.062, .064, .068), 0, ["cable"]);
    });
    b.cylinder("cable/coil", "cable-run", V(-1.62 * s, floorY + .035 * s, .96 * s), .19 * s, .035 * s, C(.072, .074, .078), 0, ["cable"]);

    // ---- notebooks, trays and a scale bar on the bench ---------------------
    b.box("bench/notebook-a", "paper-stock", V(.42 * s, .012 * s, zFront), V(.125 * s, .012 * s, .140 * s), paper, 0, ["bench", "paper"]);
    b.box("bench/notebook-b", "paper-stock", V(.40 * s, .036 * s, zFront + .01 * s), V(.115 * s, .012 * s, .125 * s), bone, 0, ["bench", "paper"]);
    b.box("bench/pen", "steel-frame", V(.20 * s, .026 * s, zFront + .03 * s), V(.070 * s, .007 * s, .010 * s), ink, 0, ["bench", "instrument"]);
    b.box("bench/scale-bar", "paper-stock", V(-.06 * s, .008 * s, .555 * s), V(.400 * s, .008 * s, .016 * s), paper, 0, ["bench", "instrument"]);
    const trayValues = [ash, bone, chalk] as const;
    for (let i = 0; i < 3; i++) b.box(`bench/tray-${i + 1}`, "tray-steel", V((-.58 + .005 * i) * s, (.014 + .028 * i) * s, zFront + .005 * i * s), V((.175 - .005 * i) * s, .014 * s, (.115 - .005 * i) * s), trayValues[i], 0, ["bench", "tray"]);

    // Benchtop scope, parked on top of the taller rack so it reads as a second
    // cool display without crowding the laser head beside the tank.
    b.box("scope/body", "scope-instrument", V(-2.35 * s, floorY + 1.365 * s, -.10 * s), V(.150 * s, .095 * s, .105 * s), graphite, 0, ["rack", "instrument"]);
    b.box("scope/screen", "scope-glass", V(-2.195 * s, floorY + 1.375 * s, -.10 * s), V(.005 * s, .062 * s, .085 * s), displayCool, .10, ["rack", "instrument", "emissive-surface-only"]);

    // ---- left wall: schematic board, cable tray, conduit -------------------
    const wallX = -roomHalf.x;
    b.box("wall/panel", "wall-panel-board", V(wallX + .028 * s, floorY + 1.62 * s, .20 * s), V(.028 * s, .58 * s, 1.05 * s), bone, 0, ["wall", "board"]);
    const marks = [
      { y: 1.90, z: -.30, hy: .020, hz: .38 },
      { y: 1.70, z: -.10, hy: .018, hz: .26 },
      { y: 1.32, z: .30, hy: .022, hz: .50 },
    ] as const;
    marks.forEach((m, i) => b.box(`wall/panel-mark-${i + 1}`, "wall-panel-board", V(wallX + .058 * s, floorY + m.y * s, m.z * s), V(.006 * s, m.hy * s, m.hz * s), C(.075, .078, .082), 0, ["wall", "board"]));
    b.box("wall/cable-tray", "conduit-pipe", V(wallX + .090 * s, floorY + 2.35 * s, .10 * s), V(.090 * s, .035 * s, 1.50 * s), C(.190, .196, .204), 0, ["wall", "conduit"]);
    for (const rack of racks) b.box(`wall/conduit-${rack.id}`, "conduit-pipe", V(wallX + .060 * s, floorY + 1.175 * s, rack.z + .10 * s), V(.030 * s, 1.175 * s, .030 * s), C(.200, .206, .214), 0, ["wall", "conduit"]);

    // ---- gas dewar, balancing the rack bay across the frame ----------------
    const dewar = V(1.67 * s, 0, -1.125 * s);
    b.cylinder("dewar/body", "dewar-steel", V(dewar.x, floorY + .62 * s, dewar.z), .17 * s, .62 * s, ash, 0, ["dewar", "instrument"]);
    b.cylinder("dewar/collar", "dewar-steel", V(dewar.x, floorY + 1.25 * s, dewar.z), .185 * s, .035 * s, pewter, 0, ["dewar", "instrument"]);
    b.cylinder("dewar/cap", "dewar-steel", V(dewar.x, floorY + 1.38 * s, dewar.z), .115 * s, .100 * s, steel, 0, ["dewar", "instrument"]);
    return shell;
  },
};
