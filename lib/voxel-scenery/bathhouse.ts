import {
  addRoomShell,
  C,
  cmul,
  V,
  type EnvironmentLinearColor,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * Japanese bathhouse, for the settled tank: the stillest room in the app, and
 * the moment before a body breaks the surface.
 *
 * The composition is entirely horizontal. A slatted cedar duckboard runs out to
 * the walls and stops short of the tank, so the water reads as a sunken bath set
 * into the deck behind a chunky stone lip. Behind it a long batten screen is
 * backlit by three shoji panels: the screen is a silhouette, and its slats cut
 * the glow into level stripes that travel forward across the free surface. Two
 * hanging shoji wings rake the same warm light in from the back corners at
 * surface height, which is what makes ripples legible.
 *
 * Everything else is neutral by design. Surfaces are authored as near-grey
 * values from wet-stone black to bleached-linen white, so form arrives through
 * shading, occlusion and falloff; the only saturated colour in the room is the
 * warm paper of the emitters, which is what tells the eye where the light is.
 */
export const bathhouseScenery: EnvironmentSceneryModule = {
  id: "bathhouse",
  build: (b, context) => {
    const shell = addRoomShell(b, context, {
      materialModel: "bathhouse",
      // A dark, low-value room. The shoji are the brightest thing in it.
      colors: { floor: C(.135, .138, .134), wall: C(.470, .462, .446), ceiling: C(.196, .198, .206) },
    });

    const { s, floorY_m: floorY, scene } = context;
    const halfW = scene.container.width_m / 2;
    const halfD = scene.container.depth_m / 2;

    // Bleached, near-neutral palette: value carries the read, not hue.
    const cedar = C(.615, .600, .566);
    const cedarPale = C(.706, .694, .658);
    const cedarDeep = C(.284, .280, .266);
    const stonePale = C(.548, .548, .536);
    const stoneMid = C(.322, .326, .322);
    const stoneDeep = C(.146, .148, .152);
    const linen = C(.884, .876, .856);
    const paper = C(1, .845, .60);
    const paperWarm = C(1, .755, .465);

    /** Deterministic per-index wobble in [-1, 1], so slats vary without a PRNG. */
    const wob = (i: number): number => ((((i * 2654435761) >>> 17) % 1001) / 500) - 1;

    // ---------------------------------------------------------------- duckboard
    // A broad slatted deck. Its slats run across the frame, and it deliberately
    // stops outside the tank so the water sits in a clean rectangular void.
    const deckTopY = floorY + .078 * s;
    const slatHy = .028 * s;
    const slatCy = deckTopY - slatHy;
    const slatHz = .058 * s;
    const pitch = .162 * s;
    const deckHalfX = 2.35 * s;
    const curbT = .075 * s;
    const curbOffX = halfW + .09 * s;
    const curbOffZ = halfD + .09 * s;
    const curbOuterX = curbOffX + curbT;
    const curbOuterZ = curbOffZ + curbT;
    const deckZ0 = curbOuterZ + .045 * s + slatHz;

    for (let k = 0; k < 6; k++) {
      b.box(`deck/front-slat-${k + 1}`, "cedar-duckboard", V(0, slatCy, deckZ0 + k * pitch), V(deckHalfX, slatHy, slatHz), cmul(cedar, .94 + .07 * wob(k + 11)), 0, ["deck", "duckboard"]);
    }
    for (let k = 0; k < 3; k++) {
      b.box(`deck/back-slat-${k + 1}`, "cedar-duckboard", V(0, slatCy, -(deckZ0 + k * pitch)), V(deckHalfX, slatHy, slatHz), cmul(cedar, .80 + .06 * wob(k + 29)), 0, ["deck", "duckboard"]);
    }
    // The flanking decks are wide boards run the other way, the way a real
    // duckboard borders itself, and they carry the bench, stools and basin.
    const sideInnerX = curbOuterX + .04 * s;
    const sideSpan = (deckHalfX - sideInnerX) / 3;
    for (const sign of [-1, 1]) for (let k = 0; k < 3; k++) {
      const cx = sign * (sideInnerX + sideSpan * (k + .5));
      b.box(`deck/${sign < 0 ? "left" : "right"}-board-${k + 1}`, "cedar-duckboard", V(cx, slatCy, 0), V(sideSpan * .47, slatHy, halfD + .245 * s), cmul(cedar, .88 + .07 * wob(k + 3 + (sign < 0 ? 40 : 60))), 0, ["deck", "duckboard"]);
    }

    // -------------------------------------------------------------- stone curb
    // The bath lip. Chunky, unbroken and just proud of the deck, so the eye is
    // walked around the void and dropped onto the water.
    const curbTopY = .27 * s;
    const curbCy = .5 * (floorY + curbTopY);
    const curbHy = .5 * (curbTopY - floorY);
    b.box("curb/front", "stone-curb", V(0, curbCy, curbOffZ), V(curbOuterX, curbHy, curbT), stonePale, 0, ["curb"]);
    b.box("curb/back", "stone-curb", V(0, curbCy, -curbOffZ), V(curbOuterX, curbHy, curbT), cmul(stonePale, .74), 0, ["curb"]);
    b.box("curb/left", "stone-curb", V(-curbOffX, curbCy, 0), V(curbT, curbHy, curbOuterZ), cmul(stonePale, .86), 0, ["curb"]);
    b.box("curb/right", "stone-curb", V(curbOffX, curbCy, 0), V(curbT, curbHy, curbOuterZ), cmul(stonePale, .92), 0, ["curb"]);
    // A single stone block set into the duckboard: the step down into the bath.
    b.box("curb/step", "stone-step", V(0, deckTopY + .055 * s, deckZ0), V(.58 * s, .055 * s, .105 * s), cmul(stonePale, .80), 0, ["curb"]);
    // Folded linen left on the lip: the brightest non-emissive value in the
    // room, sitting right at the waterline where the eye should end up.
    b.box("curb/towel-a", "stone-washed-linen", V(curbOffX, curbTopY + .042 * s, .17 * s), V(curbT * .92, .042 * s, .165 * s), linen, 0, ["linen"]);
    b.box("curb/towel-b", "stone-washed-linen", V(curbOffX, curbTopY + .118 * s, .19 * s), V(curbT * .84, .036 * s, .142 * s), cmul(linen, .93), 0, ["linen"]);

    // ------------------------------------------------------------ batten screen
    // The hero. Ten level slats plus three heavier rails, read as a silhouette
    // against the shoji and striping their light forward over the tank.
    const screenZ = -1.30 * s;
    const battenZ = -1.245 * s;
    const screenTop = 1.79 * s;
    for (let i = -2; i <= 2; i++) {
      b.box(`screen/post-${i + 3}`, "cedar-screen", V(i * .62 * s, .5 * (floorY + screenTop), screenZ), V(.030 * s, .5 * (screenTop - floorY), .048 * s), cmul(cedarDeep, 1.05), 0, ["screen", "fixture"]);
    }
    const rails: readonly (readonly [number, number, EnvironmentLinearColor])[] = [
      [.155 * s, .062 * s, cedar], [1.685 * s, .058 * s, cedarPale], [1.905 * s, .085 * s, cmul(cedar, .70)],
    ];
    rails.forEach(([y, hy, tint], i) => {
      b.box(`screen/rail-${i + 1}`, "cedar-screen", V(0, y, battenZ), V(2.42 * s, hy, .052 * s), tint, 0, ["screen", "fixture"]);
    });
    for (let k = 0; k < 10; k++) {
      b.box(`screen/batten-${k + 1}`, "cedar-screen", V(0, (.28 + .142 * k) * s, battenZ), V(2.38 * s, .034 * s, .038 * s), cmul(cedar, .90 + .10 * wob(k + 5)), 0, ["screen", "fixture"]);
    }

    // --------------------------------------------------------- shoji back wall
    // Three broad warm panels behind the screen. These are the room: everything
    // in front of them is edge-lit, and the tank sits in their reflection.
    const shojiZ = -1.46 * s;
    for (let i = 0; i < 3; i++) {
      b.box(`shoji/panel-${i + 1}`, "shoji-paper", V((i - 1) * 1.22 * s, .82 * s, shojiZ), V(.58 * s, .70 * s, .028 * s), paper, 1.6, ["shoji", "fixture", "light", "emits-positive-z"]);
    }
    b.box("shoji/frame-top", "cedar-shoji-frame", V(0, 1.565 * s, shojiZ + .022 * s), V(1.86 * s, .042 * s, .044 * s), cedarDeep, 0, ["shoji", "fixture"]);
    b.box("shoji/frame-bottom", "cedar-shoji-frame", V(0, .095 * s, shojiZ + .022 * s), V(1.86 * s, .052 * s, .044 * s), cedarDeep, 0, ["shoji", "fixture"]);
    for (const sign of [-1, 1]) {
      b.box(`shoji/frame-${sign < 0 ? "left" : "right"}`, "cedar-shoji-frame", V(sign * 1.818 * s, .82 * s, shojiZ + .022 * s), V(.046 * s, .74 * s, .044 * s), cedarDeep, 0, ["shoji", "fixture"]);
    }

    // ------------------------------------------------------------ hanging wings
    // Two suspended shoji over the back corners of the deck, hung low and turned
    // inward so their light travels along the surface rather than down onto it.
    const wingZ = -.68 * s;
    for (const i of [-1, 1]) {
      const side = i < 0 ? "left" : "right", x = i * 1.55 * s;
      b.box(`lantern-${side}/shade`, "shoji-paper", V(x, .78 * s, wingZ), V(.032 * s, .52 * s, .46 * s), paper, 1.15, ["lantern", "fixture", "light", i < 0 ? "emits-positive-x" : "emits-negative-x"]);
      b.box(`lantern-${side}/hanger`, "cedar-shoji-frame", V(x, 1.335 * s, wingZ), V(.055 * s, .038 * s, .48 * s), cedarDeep, 0, ["lantern", "fixture"]);
      b.cylinder(`lantern-${side}/cord`, "metal-fixture", V(x, 1.86 * s, wingZ), .009 * s, .49 * s, cmul(stoneDeep, 1.4), 0, ["lantern", "fixture"]);
    }

    // ------------------------------------------------------------------- andon
    // A small floor lantern on the back deck, low enough to graze the curb.
    const andonX = -.92 * s, andonZ = -.56 * s;
    b.box("andon/base", "cedar-lantern", V(andonX, deckTopY + .032 * s, andonZ), V(.175 * s, .032 * s, .175 * s), cedarDeep, 0, ["lantern", "fixture"]);
    b.box("andon/paper", "shoji-paper", V(andonX, deckTopY + .275 * s, andonZ), V(.148 * s, .215 * s, .148 * s), paperWarm, 1.5, ["lantern", "fixture", "light", "point-light"]);
    b.box("andon/cap", "cedar-lantern", V(andonX, deckTopY + .518 * s, andonZ), V(.19 * s, .030 * s, .19 * s), cmul(cedarDeep, .8), 0, ["lantern", "fixture"]);

    // -------------------------------------------------------------- back ledge
    // A long low shelf across the back: one more level line behind the water.
    const ledgeX = -.30 * s, ledgeZ = -.94 * s, ledgeY = deckTopY + .35 * s;
    b.box("ledge/top", "stone-ledge", V(ledgeX, ledgeY, ledgeZ), V(.88 * s, .048 * s, .235 * s), cmul(stonePale, .70), 0, ["ledge"]);
    for (const i of [-1, 1]) {
      b.box(`ledge/leg-${i < 0 ? "left" : "right"}`, "stone-ledge", V(ledgeX + i * .70 * s, .5 * (deckTopY + ledgeY), ledgeZ), V(.10 * s, .5 * (ledgeY - deckTopY), .175 * s), cmul(stoneDeep, 1.6), 0, ["ledge"]);
    }

    // -------------------------------------------------------------- wash basin
    // Low stone tsukubai with a bamboo spout. The trickle is the only thing in
    // the room that moves, so it is a thin bright line and nothing more.
    const bx = 1.05 * s, bz = -.80 * s;
    b.cylinder("basin/plinth", "stone-basin", V(bx, deckTopY + .085 * s, bz), .335 * s, .085 * s, cmul(stoneMid, .82), 0, ["basin"]);
    b.cylinder("basin/bowl", "stone-basin", V(bx, deckTopY + .275 * s, bz), .375 * s, .155 * s, stoneMid, 0, ["basin"]);
    b.cylinder("basin/rim", "stone-basin", V(bx, deckTopY + .448 * s, bz), .395 * s, .034 * s, cmul(stonePale, .92), 0, ["basin"]);
    b.cylinder("basin/water", "basin-glass", V(bx, deckTopY + .418 * s, bz), .335 * s, .014 * s, C(.086, .092, .098), 0, ["basin"]);
    b.cylinder("bamboo/post", "cedar-bamboo", V(bx - .40 * s, deckTopY + .40 * s, bz - .06 * s), .048 * s, .40 * s, cedarPale, 0, ["basin", "fixture"]);
    b.box("bamboo/spout", "cedar-bamboo", V(bx - .20 * s, deckTopY + .755 * s, bz - .06 * s), V(.245 * s, .038 * s, .038 * s), cmul(cedarPale, .95), 0, ["basin", "fixture"]);
    b.box("bamboo/trickle", "water-glass", V(bx, deckTopY + .59 * s, bz - .06 * s), V(.011 * s, .155 * s, .011 * s), C(.78, .80, .82), .05, ["basin", "fixture", "emissive-surface-only"]);
    b.ellipsoid("basin/stone-a", "river-stone", V(bx - .10 * s, deckTopY + .066 * s, bz + .62 * s), V(.20 * s, .085 * s, .17 * s), cmul(stoneDeep, 1.5), 0, ["stone"]);
    b.ellipsoid("basin/stone-b", "river-stone", V(bx + .58 * s, deckTopY + .055 * s, bz + .34 * s), V(.155 * s, .07 * s, .19 * s), cmul(stoneDeep, 1.9), 0, ["stone"]);

    // ------------------------------------------------------- bench and towels
    // A long bench parallel to the tank, stacked with folded linen: the last
    // repeat of the horizontal motif, and the scene's brightest accent.
    const benchX = -2.00 * s, benchTopY = deckTopY + .33 * s;
    b.box("bench/top", "cedar-bench", V(benchX, benchTopY, 0), V(.245 * s, .036 * s, .82 * s), cedarPale, 0, ["bench"]);
    b.box("bench/rail", "cedar-bench", V(benchX, deckTopY + .135 * s, 0), V(.055 * s, .030 * s, .72 * s), cmul(cedar, .62), 0, ["bench"]);
    for (const i of [-1, 1]) {
      b.box(`bench/leg-${i < 0 ? "back" : "front"}`, "cedar-bench", V(benchX, .5 * (deckTopY + benchTopY), i * .64 * s), V(.205 * s, .5 * (benchTopY - deckTopY), .046 * s), cmul(cedar, .78), 0, ["bench"]);
    }
    const towels: readonly (readonly [number, number, number, number])[] = [
      [-.46, 0, .152, .94], [-.44, 1, .146, 1.0], [-.48, 2, .138, .88],
      [.12, 0, .156, .97], [.10, 1, .148, .90], [.60, 0, .150, 1.0],
    ];
    towels.forEach(([z, tier, hz, value], i) => {
      b.box(`bench/towel-${i + 1}`, "stone-washed-linen", V(benchX + .02 * s * wob(i + 7), benchTopY + (.075 + .078 * tier) * s, z * s), V(.185 * s, .039 * s, hz * s), cmul(linen, value), 0, ["linen"]);
    });

    // ------------------------------------------------------------ river stones
    // Big, smooth and low. Scattered from a hand table so the layout is a pure
    // function of the module, and sized to read at silhouette.
    const stones: readonly (readonly [number, number, number, number, number, number])[] = [
      [-1.08, 1.16, .26, .115, .22, 1.00],
      [-.52, 1.40, .19, .080, .165, 1.55],
      [.30, 1.12, .225, .095, .20, 1.30],
      [1.30, 1.15, .175, .075, .155, 1.80],
      [-1.46, -.52, .245, .105, .215, 1.15],
      [2.05, .12, .21, .090, .185, 1.40],
      [1.98, 1.30, .28, .120, .235, .92],
    ];
    stones.forEach(([x, z, rx, ry, rz, value], i) => {
      b.ellipsoid(`stone/river-${i + 1}`, "river-stone", V(x * s, deckTopY + ry * s * .78, z * s), V(rx * s, ry * s, rz * s), cmul(stoneDeep, value), 0, ["stone"]);
    });

    // ---------------------------------------------------------- stools, bucket
    b.cylinder("stool/left", "stone-plinth", V(-1.10 * s, deckTopY + .12 * s, .58 * s), .25 * s, .12 * s, cmul(stonePale, .78), 0, ["stool"]);
    b.cylinder("stool/right", "stone-plinth", V(1.20 * s, deckTopY + .105 * s, .46 * s), .245 * s, .105 * s, cmul(stonePale, .66), 0, ["stool"]);
    b.cylinder("bucket/body", "cedar-bucket", V(.76 * s, deckTopY + .175 * s, 1.16 * s), .215 * s, .175 * s, cmul(cedarPale, .90), 0, ["bucket"]);
    b.cylinder("bucket/rim", "cedar-bucket", V(.76 * s, deckTopY + .368 * s, 1.16 * s), .228 * s, .028 * s, cedarPale, 0, ["bucket"]);
    b.box("bucket/ladle", "cedar-bucket", V(.76 * s, deckTopY + .40 * s, 1.30 * s), V(.038 * s, .020 * s, .26 * s), cmul(cedar, .85), 0, ["bucket"]);

    return shell;
  },
};
