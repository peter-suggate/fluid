import {
  addRoomShell,
  C,
  cmul,
  V,
  type EnvironmentLinearColor,
  type EnvironmentSceneryModule,
} from "./builder";

/**
 * Brutalist water gallery for the dam-break figure: board-formed concrete, a
 * luminous portal punched through a freestanding blade wall, and a deep
 * recessed light slot at water level directly behind the tank.
 *
 * The room is a plinth for a violent event, so it is composed entirely for
 * back- and side-light. Nothing here is coloured: every surface is a neutral
 * grey and only the three emitters carry a tint, which is what makes the
 * raking light legible. The slot rakes forward across the free surface and
 * throws long hard shadows from the white monoliths towards the viewer; the
 * portal silhouettes the water and the box stack it is about to hit; the low
 * cool wall-washer on the left counter-rakes so the shadows cross.
 */
export const concreteGalleryScenery: EnvironmentSceneryModule = {
  id: "concrete-gallery",
  build: (b, context) => {
    /** Near-neutral grey; `tint` leans warm at +1 and cool at -1, never far. */
    const tone = (value: number, tint = 0): EnvironmentLinearColor =>
      C(value * (1 + .055 * tint), value, value * (1 - .05 * tint));

    const shell = addRoomShell(b, context, {
      materialModel: "gallery",
      colors: { floor: tone(.055, .3), wall: tone(.135), ceiling: tone(.075, -.3) },
    });

    const { s, floorY_m: floorY, roomHalf_m: room, scene } = context;
    const tankX = scene.container.width_m * .5;
    const tankZ = scene.container.depth_m * .5;

    /** Floor-standing box: `half.y` is measured up from the floor plane. */
    const stand = (
      key: string, group: string, x: number, z: number,
      hx: number, hy: number, hz: number,
      color: EnvironmentLinearColor, tags: readonly string[],
    ) => b.box(key, group, V(x, floorY + hy, z), V(hx, hy, hz), color, 0, tags);

    /** Wall panel rising from the floor to `top`. */
    const panel = (
      key: string, group: string, x: number, hx: number, top: number, z: number, hz: number,
      color: EnvironmentLinearColor, tags: readonly string[],
    ) => b.box(key, group, V(x, (floorY + top) * .5, z), V(hx, (top - floorY) * .5, hz), color, 0, tags);

    /** Deterministic board-to-board value variation; no randomness anywhere. */
    const jitter = (i: number) => (((Math.imul(i + 1, 0x9e3779b1) >>> 17) & 0xff) / 255 - .5) * .09;

    // ---------------------------------------------------------------- shell relief
    // Board-formed relief on the left wall: six proud lifts starting above the
    // washer's own lip, the ladder of hard shadow that lip exists to cast.
    const wallX = -room.x;
    for (let i = 0; i < 6; i++) {
      const y = (.52 + .46 * i) * s;
      b.box(`relief/board-${i + 1}`, "board-formed-stone", V(wallX + .056 * s, floorY + y, 0), V(.055 * s, .085 * s, room.z), tone(.20 + jitter(i)), 0, ["relief", "board-formed"]);
    }

    // ------------------------------------------------------------- the blade wall
    // A freestanding concrete blade behind the tank with one rectangular hole
    // in it. The hole is the portal; the wall is thick enough that the glowing
    // panel sits well back inside it and throws a hard-edged shaft forward.
    // The leaves are deliberately unequal: the long one runs off frame-left, the
    // short one terminates in view so the blade reads as an object, not a wall.
    const bladeZ = -1.45 * s, bladeHalfZ = .17 * s, bladeTop = 2.05 * s;
    const bladeEnd = { left: -2.30 * s, right: 1.34 * s };
    const apertureX = .84 * s, apertureTop = 1.62 * s, apertureSill = .05 * s;
    const blade = tone(.30);
    const leaf = { left: [(bladeEnd.left - apertureX) * .5, (apertureX + bladeEnd.left) * -.5] as const, right: [(bladeEnd.right + apertureX) * .5, (bladeEnd.right - apertureX) * .5] as const };
    panel("wall/screen-left", "screen-stone", leaf.left[0], leaf.left[1], bladeTop, bladeZ, bladeHalfZ, blade, ["blade-wall"]);
    panel("wall/screen-right", "screen-stone", leaf.right[0], leaf.right[1], bladeTop, bladeZ, bladeHalfZ, blade, ["blade-wall"]);
    b.box("wall/screen-header", "screen-stone", V(0, (apertureTop + bladeTop) * .5, bladeZ), V(apertureX, (bladeTop - apertureTop) * .5, bladeHalfZ), cmul(blade, 1.08), 0, ["blade-wall"]);
    panel("wall/screen-sill", "screen-stone", 0, apertureX, apertureSill, bladeZ, bladeHalfZ, cmul(blade, .8), ["blade-wall"]);
    // Three shadow-gap reveals per leaf, so the blade reads as poured lifts.
    for (let i = 0; i < 3; i++) {
      const y = (.46 + .62 * i) * s;
      for (const side of ["left", "right"] as const) {
        b.box(`wall/screen-rib-${side === "left" ? "l" : "r"}${i + 1}`, "screen-stone", V(leaf[side][0], floorY + y, bladeZ + bladeHalfZ + .035 * s), V(leaf[side][1], .075 * s, .045 * s), tone(.36 + jitter(i + 7)), 0, ["blade-wall", "relief"]);
      }
    }

    // ------------------------------------------------------------------- the portal
    // The frame is a pale reveal lining the mouth of the opening; the warmth
    // is carried entirely by the emitter recessed 0.18 s behind it.
    const reveal = tone(.72, .8);
    b.box("portal/left", "emissive-fixture", V(-(apertureX - .045 * s), (apertureTop + apertureSill) * .5, bladeZ + bladeHalfZ - .045 * s), V(.045 * s, (apertureTop - apertureSill) * .5, .05 * s), reveal, .10, ["portal", "fixture", "emissive-surface-only"]);
    b.box("portal/right", "emissive-fixture", V(apertureX - .045 * s, (apertureTop + apertureSill) * .5, bladeZ + bladeHalfZ - .045 * s), V(.045 * s, (apertureTop - apertureSill) * .5, .05 * s), reveal, .10, ["portal", "fixture", "emissive-surface-only"]);
    b.box("portal/top", "emissive-fixture", V(0, apertureTop - .045 * s, bladeZ + bladeHalfZ - .045 * s), V(apertureX, .045 * s, .05 * s), reveal, .10, ["portal", "fixture", "emissive-surface-only"]);
    b.box("portal/aperture", "emissive-portal", V(0, (apertureTop + apertureSill) * .5, bladeZ - bladeHalfZ + .06 * s), V(apertureX - .04 * s, (apertureTop - apertureSill) * .5 - .01 * s, .022 * s), C(1, .58, .28), .9, ["portal", "fixture", "light", "emits-positive-z"]);

    // ------------------------------------------------------- the recessed light slot
    // A low dam-like wall between the portal and the tank, split by a slot at
    // free-surface height. This is the key: it rakes forward, rim-lights the
    // water, and silhouettes the box stack against its own throw.
    const slotZ = -.88 * s, slotHalfZ = .14 * s, slotSpan = 1.58 * s;
    const slotBottom = .155 * s, slotHead = .255 * s, slotWallTop = .46 * s;
    panel("slot/wall-lower", "slot-reveal-stone", 0, slotSpan, slotBottom, slotZ, slotHalfZ, tone(.14), ["light-slot"]);
    b.box("slot/wall-upper", "slot-reveal-stone", V(0, (slotHead + slotWallTop) * .5, slotZ), V(slotSpan, (slotWallTop - slotHead) * .5, slotHalfZ), tone(.21), 0, ["light-slot"]);
    for (const side of [-1, 1]) {
      panel(`slot/pier-${side < 0 ? "left" : "right"}`, "slot-reveal-stone", side * 1.47 * s, .11 * s, slotWallTop, slotZ, slotHalfZ, tone(.165), ["light-slot"]);
    }
    b.box("slot/recess-back", "recess-void-stone", V(0, (slotBottom + slotHead) * .5, slotZ - slotHalfZ), V(1.36 * s, (slotHead - slotBottom) * .5, .02 * s), tone(.035), 0, ["light-slot"]);
    b.box("slot/emitter", "emissive-slot", V(0, (slotBottom + slotHead) * .5, slotZ - slotHalfZ + .045 * s), V(1.30 * s, .042 * s, .012 * s), C(1, .87, .70), 3, ["light-slot", "fixture", "light", "emits-positive-z"]);

    // ------------------------------------------------------ the tank recess kerbing
    // A shallow reveal ring around the container footprint: the water sits in
    // a recess rather than on the floor, which is what makes it read as the
    // exhibit. Held clear of the container volume on every face.
    const kerbHalf = .15 * s, kerbGap = .07 * s, kerbTop = .075 * s;
    const kerbX = tankX + kerbGap + kerbHalf, kerbZ = tankZ + kerbGap + kerbHalf;
    const kerb = tone(.105);
    for (const side of [-1, 1]) {
      stand(`kerb/${side < 0 ? "left" : "right"}`, "gallery-kerb-stone", side * kerbX, 0, kerbHalf, kerbTop * .5, kerbZ + kerbHalf, kerb, ["kerb", "recess"]);
      stand(`kerb/${side < 0 ? "back" : "front"}`, "gallery-kerb-stone", 0, side * kerbZ, kerbX + kerbHalf, kerbTop * .5, kerbHalf, kerb, ["kerb", "recess"]);
    }

    // ---------------------------------------------------------- the cantilevered slab
    // A heavy overhang from the left wall that stops short of the tank, held
    // just clear of the sightline to the portal so it catches that glow along
    // its underside and gives the room a lid to press the event down against.
    const slabTop = 1.86 * s, slabHalfY = .105 * s, slabEdge = .40 * s;
    const slabZ = -.62 * s, slabHalfZ = .58 * s;
    const slab = tone(.085);
    b.box("canopy/slab", "canopy-stone", V((-room.x + slabEdge) * .5, slabTop - slabHalfY, slabZ), V((room.x + slabEdge) * .5, slabHalfY, slabHalfZ), slab, 0, ["canopy", "overhang"]);
    b.box("canopy/fascia", "canopy-stone", V(slabEdge + .06 * s, slabTop - .19 * s, slabZ), V(.06 * s, .19 * s, slabHalfZ), cmul(slab, 1.65), 0, ["canopy", "overhang"]);
    for (let i = 0; i < 2; i++) {
      b.box(`canopy/rib-${i + 1}`, "canopy-stone", V((-1.55 + 1.05 * i) * s, slabTop - 2 * slabHalfY - .07 * s, slabZ), V(.07 * s, .07 * s, slabHalfZ), cmul(slab, .82), 0, ["canopy", "relief"]);
    }

    // -------------------------------------------------------- the low cool wall-washer
    // Second emitter, deliberately low and hard to one side: it counter-rakes
    // the warm slot so every monolith throws two crossing shadows.
    b.box("washer/recess", "recess-void-stone", V(wallX + .015 * s, floorY + .165 * s, 0), V(.015 * s, .095 * s, 1.85 * s), tone(.04), 0, ["washer"]);
    b.box("washer/emitter", "emissive-washer", V(wallX + .046 * s, floorY + .165 * s, 0), V(.014 * s, .07 * s, 1.72 * s), C(.66, .77, .93), 1.4, ["washer", "fixture", "light", "emits-positive-x"]);
    b.box("washer/lip", "washer-reveal-stone", V(wallX + .085 * s, floorY + .30 * s, 0), V(.085 * s, .045 * s, 1.90 * s), tone(.19), 0, ["washer", "relief"]);

    // ------------------------------------------------------------------ the monoliths
    // Chunky white blocks scattered on the floor, rhyming with the rigid box
    // stack the water is about to hit. Hand-authored, so the composition is
    // fixed: verticals gathered behind and left of the tank where the slot
    // throws the longest shadows, low slabs out in the near foreground, and one
    // block on the far side of the slot wall to catch its throw edge-on.
    const monoliths: readonly (readonly [string, number, number, number, number, number, number])[] = [
      ["a", -1.75, -.28, .22, .40, .20, .72],
      ["b", -1.85, -1.00, .24, .46, .20, .64],
      ["c", -2.46, -1.05, .15, .44, .18, .80],
      ["d", -2.35, .35, .20, .15, .24, .50],
      ["e", -1.98, 1.18, .30, .17, .26, .66],
      ["f", -2.40, 1.75, .19, .13, .18, .60],
      ["g", .85, -1.12, .19, .30, .07, .55],
    ];
    for (const [key, x, z, hx, hy, hz, value] of monoliths) {
      stand(`monolith/${key}`, "monolith-stone", x * s, z * s, hx * s, hy * s, hz * s, tone(value), ["monolith"]);
    }
    // One deliberate stack, so the rhyme with the box tower is unmistakable.
    b.box("monolith/stack-cap", "monolith-stone", V(-1.85 * s, floorY + 1.01 * s, -1.00 * s), V(.14 * s, .09 * s, .12 * s), tone(.88), 0, ["monolith", "stack"]);

    // -------------------------------------------------------------- the barrier line
    // The foreground layer: a bollard-and-rope line across the near floor that
    // holds the viewer back from the tank and parallaxes against it.
    const ropeZ = .85 * s, ropeTop = floorY + .32 * s;
    const bollardX = [-1.95, -1.10, -.25, .60];
    bollardX.forEach((x, i) => {
      b.cylinder(`bollard/post-${i + 1}`, "steel-bollard", V(x * s, floorY + .17 * s, ropeZ), .052 * s, .17 * s, tone(.045), 0, ["barrier", "fixture"]);
      b.ellipsoid(`bollard/cap-${i + 1}`, "steel-bollard", V(x * s, floorY + .352 * s, ropeZ), V(.058 * s, .044 * s, .058 * s), tone(.09), 0, ["barrier", "fixture"]);
    });
    // Three chunky segments per span approximate a sag; axis-aligned voxels
    // read a stepped catenary better than a taut line.
    const sag: readonly (readonly [number, number, number])[] = [[0, .28, .045], [.28, .72, .095], [.72, 1, .045]];
    for (let i = 0; i < bollardX.length - 1; i++) {
      const x0 = bollardX[i] * s, x1 = bollardX[i + 1] * s;
      sag.forEach(([t0, t1, drop], j) => {
        const a = x0 + (x1 - x0) * t0, e = x0 + (x1 - x0) * t1;
        b.box(`rope/span-${i + 1}-${j + 1}`, "rope-line", V((a + e) * .5, ropeTop - drop * s, ropeZ), V((e - a) * .5, .022 * s, .022 * s), tone(.55), 0, ["barrier", "rope"]);
      });
    }

    // ------------------------------------------------------------------- the furniture
    // A long cast bench along the left, side-on to the tank, low enough to sit
    // under the wall-washer's throw and cast a hard shadow across the floor.
    const benchX = -1.50 * s, benchZ = .20 * s;
    const bench = tone(.155);
    b.box("bench/seat", "gallery-seat-stone", V(benchX, floorY + .30 * s, benchZ), V(.62 * s, .06 * s, .175 * s), bench, 0, ["bench"]);
    for (const i of [-1, 1]) {
      stand(`bench/leg-${i < 0 ? "left" : "right"}`, "gallery-seat-stone", benchX + .44 * s * i, benchZ, .06 * s, .12 * s, .14 * s, cmul(bench, .78), ["bench"]);
    }

    // A satin cast form on a plinth to the right of the tank, standing between
    // the slot and the camera so it reads as a black silhouette with a hot rim.
    const sculptureX = 1.30 * s, sculptureZ = -.30 * s;
    stand("sculpture/plinth", "stone-plinth", sculptureX, sculptureZ, .21 * s, .18 * s, .21 * s, tone(.12), ["plinth", "sculpture"]);
    b.ellipsoid("sculpture/lower", "sculpture-metal", V(sculptureX, floorY + .64 * s, sculptureZ), V(.21 * s, .31 * s, .175 * s), tone(.075), 0, ["sculpture"]);
    b.ellipsoid("sculpture/upper", "sculpture-metal", V(sculptureX - .12 * s, floorY + .92 * s, sculptureZ + .03 * s), V(.14 * s, .21 * s, .13 * s), tone(.62), 0, ["sculpture"]);

    return shell;
  },
};
