import type { Vec3 } from "./model";
import type { SceneryGraph, SceneryGroupNode, SceneryMaterial, SceneryNode } from "./scenery-graph";
import { alongAxis } from "./voxel-scenery/builder";
import { planOakV2 } from "./voxel-scenery/oak-v2";

// Metres throughout. The park sits on y=0; the flume keeps its own solver geometry.
const white: SceneryMaterial = { colorLinear: [.9, .9, .9], surface: "plaster" };
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
type Point = readonly [number, number, number];
const point = ([x, y, z]: Point) => v(x, y, z);
const yaw = (angle: number) => ({ w: Math.cos(angle / 2), x: 0, y: Math.sin(angle / 2), z: 0 });
const box = (id: string, at: Point, half: Point): SceneryNode =>
  ({ kind: "box", id, place: { position: point(at) }, halfSize: point(half), material: white });
const oval = (id: string, at: Point, radius: Point): SceneryNode =>
  ({ kind: "ellipsoid", id, place: { position: point(at) }, radius: point(radius), material: white });
const rod = (id: string, from: Point, to: Point, radius = .025): SceneryNode =>
  ({ kind: "capsule", id, from: point(from), to: point(to), radius, material: white });
const group = (id: string, at: Point, children: SceneryNode[], angle = 0, scale = 1): SceneryGroupNode =>
  ({ kind: "group", id, tags: [id.split("/")[1]!], place: { units: "metres", position: point(at), orientation: yaw(angle), scale }, children });

/** A shaped limb: taper and overlapping rounded ends avoid bead-like joints. */
function limb(id: string, a: Point, b: Point, ra: number, rb: number): SceneryNode {
  const d = v(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  return { kind: "cone", id, place: { position: v((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2), orientation: alongAxis(d) },
    halfHeight: Math.hypot(d.x, d.y, d.z) / 2, baseRadius: ra, topRadius: rb, roundedEnds: true, material: white };
}

/** Anatomical proportions, relaxed elbows, shoes, swimwear seams and facial planes. */
function visitor(id: string, at: Point, angle: number, pose: "walk" | "watch" | "seated", scale = 1): SceneryNode {
  const sitting = pose === "seated", hip = sitting ? .60 : .93, shoulder = hip + .47;
  const children: SceneryNode[] = [
    oval(`${id}/pelvis`, [0, hip, 0], [.17, .17, .12]),
    oval(`${id}/waist`, [0, hip + .18, 0], [.145, .23, .105]),
    oval(`${id}/ribcage`, [0, hip + .32, -.008], [.205, .23, .12]),
    limb(`${id}/shoulders`, [-.18, shoulder, 0], [.18, shoulder, 0], .09, .09),
    limb(`${id}/neck`, [0, shoulder + .04, 0], [0, shoulder + .14, .005], .065, .055),
    oval(`${id}/skull`, [0, shoulder + .245, .006], [.09, .125, .09]),
    oval(`${id}/jaw`, [0, shoulder + .195, .045], [.073, .069, .058]),
    oval(`${id}/nose`, [0, shoulder + .24, .097], [.018, .027, .028]),
    oval(`${id}/hair`, [0, shoulder + .30, -.006], [.094, .078, .092]),
  ];
  for (const side of [-1, 1]) {
    const x = side * .10, stride = pose === "walk" ? side * .18 : 0;
    const knee: Point = sitting ? [x, .49, .34] : [x, .52, stride];
    const ankle: Point = sitting ? [x, .12, .38] : [x + side * .025, .12, -stride];
    const elbow: Point = sitting ? [side * .24, hip + .12, .18]
      : [side * .25, hip + .22, pose === "walk" ? -stride * .55 : .05];
    const wrist: Point = sitting ? [side * .13, hip - .02, .35]
      : [side * .27, hip - .03, pose === "walk" ? -stride : .14];
    children.push(
      limb(`${id}/${side}/thigh`, [x, hip, 0], knee, .102, .07),
      limb(`${id}/${side}/calf`, knee, ankle, .069, .038),
      oval(`${id}/${side}/shoe`, [ankle[0], .065, ankle[2] + .065], [.06, .06, .13]),
      limb(`${id}/${side}/sleeve`, [side * .19, shoulder - .005, 0], [side * .23, shoulder - .10, .015], .088, .079),
      limb(`${id}/${side}/upper-arm`, [side * .21, shoulder - .08, .01], elbow, .063, .043),
      limb(`${id}/${side}/forearm`, elbow, wrist, .047, .03),
      oval(`${id}/${side}/hand`, [wrist[0], wrist[1] - .045, wrist[2] + .013], [.035, .063, .026]),
      oval(`${id}/${side}/ear`, [side * .09, shoulder + .24, .008], [.016, .03, .021]),
      limb(`${id}/${side}/shorts-hem`, [x, hip - .06, .01], [x, hip - .18, sitting ? .17 : stride * .25], .108, .10),
    );
  }
  return group(id, at, children, angle, scale);
}

function bench(id: string, at: Point, angle: number): SceneryNode {
  const children: SceneryNode[] = [];
  for (let i = 0; i < 5; i++) {
    children.push(box(`${id}/seat-${i}`, [0, .45, -.20 + i * .10], [.83, .027, .042]));
  }
  for (let i = 0; i < 4; i++) {
    children.push(box(`${id}/back-${i}`, [0, .63 + i * .105, -.25 - i * .022], [.83, .04, .025]));
  }
  for (const side of [-1, 1]) {
    const x = side * .63;
    children.push(rod(`${id}/frame-${side}`, [x, .08, .24], [x, .88, -.31], .035));
    for (const z of [-.19, .19]) {
      children.push(box(`${id}/foot-${side}-${z}`, [x, .025, z], [.085, .025, .09]),
        rod(`${id}/leg-${side}-${z}`, [x, .04, z], [x, .43, z], .03));
    }
    children.push(rod(`${id}/arm-${side}`, [x, .68, -.26], [x, .68, .23], .028),
      rod(`${id}/arm-post-${side}`, [x, .46, .19], [x, .68, .19], .025));
    for (const y of [.63, .945]) children.push(oval(`${id}/bolt-${side}-${y}`, [x, y, -.216 - (y - .63) * .21], [.013, .013, .008]));
  }
  return group(id, at, children, angle);
}

/** Two flights with open risers, stringers, balusters and a continuous landing. */
function accessStairs(): SceneryNode {
  const id = "park/stairs", children: SceneryNode[] = [];
  const rise = 2.12 / 15, run = 2.4 / 15;
  for (let flight = 0; flight < 2; flight++) {
    const z = flight === 0 ? -3.0 : -3.85;
    const xAt = (t: number) => flight === 0 ? -2.9 + t * 2.4 : -.5 - t * 2.4;
    for (let step = 0; step < 15; step++) {
      const y = flight * 2.12 + (step + 1) * rise, x = xAt((step + .5) / 15);
      children.push(box(`${id}/${flight}/tread-${step}`, [x, y - .035, z], [run * .51, .035, .35]));
      if (step % 3 === 0) for (const side of [-1, 1]) {
        children.push(rod(`${id}/${flight}/baluster-${step}-${side}`, [x, y, z + side * .34], [x, y + .9, z + side * .34], .019));
      }
    }
    for (const side of [-1, 1]) {
      children.push(rod(`${id}/${flight}/stringer-${side}`, [xAt(0), flight * 2.12, z + side * .29], [xAt(1), (flight + 1) * 2.12, z + side * .29], .055),
        rod(`${id}/${flight}/rail-${side}`, [xAt(0), flight * 2.12 + .96, z + side * .34], [xAt(1), (flight + 1) * 2.12 + .96, z + side * .34], .025));
    }
  }
  children.push(box(`${id}/half-landing`, [-.14, 2.08, -3.425], [.36, .04, .775]),
    box(`${id}/upper-walkway`, [-3.24, 4.20, -2.9], [.34, .04, 1.30]),
    box(`${id}/launch-landing`, [-2.54, 4.20, -1.94], [1.04, .04, .34]));
  for (const [x, z, y] of [[.15, -4.12, 2.12], [.15, -2.73, 2.12], [-3.5, -4.12, 4.24], [-3.5, -1.68, 4.24]]) {
    children.push(rod(`${id}/support-${x}-${z}`, [x, 0, z], [x, y, z], .055));
  }
  for (let i = 0; i <= 6; i++) {
    const z = -4.15 + i * .42;
    children.push(rod(`${id}/upper-post-${i}`, [-3.55, 4.24, z], [-3.55, 5.14, z], .02));
  }
  children.push(rod(`${id}/upper-rail`, [-3.55, 5.14, -4.15], [-3.55, 5.14, -1.6]),
    rod(`${id}/landing-rail`, [-3.55, 5.14, -1.6], [-1.9, 5.14, -1.6]));
  return group(id, [0, 0, 0], children);
}

/** Open serving hatch, deep counter, ribbed canopy and a small display cabinet. */
function kiosk(id: string, at: Point, angle: number, variant: number): SceneryNode {
  const children: SceneryNode[] = [
    box(`${id}/plinth`, [0, .06, 0], [1.2, .06, .8]),
    box(`${id}/back`, [0, 1.23, -.74], [1.15, 1.17, .06]),
    box(`${id}/left`, [-1.09, 1.23, 0], [.06, 1.17, .74]),
    box(`${id}/right`, [1.09, 1.23, 0], [.06, 1.17, .74]),
    box(`${id}/counter-front`, [0, .52, .65], [1.08, .46, .08]),
    box(`${id}/counter`, [0, 1.02, .75], [1.24, .045, .28]),
    box(`${id}/fascia`, [0, 2.25, .76], [1.2, .15, .06]),
    box(`${id}/roof`, [0, 2.44, -.05], [1.24, .06, .88]),
    box(`${id}/menu-frame`, [-.57, 1.66, -.66], [.32, .38, .027]),
    box(`${id}/display-base`, [.43, 1.10, .71], [.35, .045, .22]),
    box(`${id}/display-top`, [.43, 1.51, .71], [.35, .025, .22]),
    box(`${id}/register`, [-.71, 1.16, .62], [.16, .10, .12]),
  ];
  for (let i = 0; i < 17; i++) {
    const x = -1.2 + i * .15;
    children.push(limb(`${id}/awning-rib-${i}`, [x, 2.4, .55], [x, 2.17, 1.27], .055, .055),
      oval(`${id}/awning-scallop-${i}`, [x, 2.12, 1.27], [.072, .10, .025]));
  }
  for (const x of [-1.15, 1.15]) children.push(rod(`${id}/canopy-post-${x}`, [x, 0, 1.22], [x, 2.2, 1.22], .027));
  for (const x of [.12, .74]) for (const z of [.52, .9]) {
    children.push(rod(`${id}/display-upright-${x}-${z}`, [x, 1.12, z], [x, 1.5, z], .012));
  }
  for (let i = 0; i < 5; i++) {
    children.push(box(`${id}/menu-line-${i}`, [-.60, 1.9 - i * .12, -.625], [.22 - (i % 2) * .055, .012, .012]));
    if (variant === 0) {
      children.push({ kind: "cone", id: `${id}/cup-${i}`, place: { position: v(.18 + i * .12, 1.2, .72) },
        baseRadius: .035, topRadius: .045, halfHeight: .075, material: white });
    } else {
      children.push(oval(`${id}/pastry-${i}`, [.18 + i * .12, 1.18, .72], [.052, .047, .08]));
    }
  }
  // Simple raised pictograms on the fascia read without introducing colored signs.
  children.push({ kind: "torus", id: `${id}/sign-ring`, place: { position: v(0, 2.26, .84), orientation: alongAxis(v(0, 0, 1)) },
    majorRadius: .085, minorRadius: .018, material: white });
  return group(id, at, children, angle);
}

function toilets(): SceneryNode {
  const id = "park/toilets", children: SceneryNode[] = [
    box(`${id}/foundation`, [0, .045, 0], [2.15, .045, 1.18]),
    box(`${id}/back`, [0, 1.35, -1.02], [2.04, 1.30, .08]),
    box(`${id}/left`, [-1.96, 1.35, 0], [.08, 1.30, 1.02]),
    box(`${id}/right`, [1.96, 1.35, 0], [.08, 1.30, 1.02]),
    box(`${id}/lintel`, [0, 2.39, 1.02], [2.04, .26, .08]),
    box(`${id}/roof`, [0, 2.7, 0], [2.23, .065, 1.23]),
    box(`${id}/canopy`, [0, 2.43, 1.38], [2.23, .045, .35]),
  ];
  for (const x of [-1.94, -.66, .66, 1.94]) {
    children.push(box(`${id}/pier-${x}`, [x, 1.09, 1.02], [.095, 1.04, .08]));
  }
  for (const [i, x] of [-1.3, 0, 1.3].entries()) {
    children.push(box(`${id}/door-${i}`, [x, 1.04, .93], [.52, .97, .035]),
      box(`${id}/threshold-${i}`, [x, .075, 1.04], [.55, .025, .12]),
      rod(`${id}/handle-${i}`, [x + .36, .96, 1.0], [x + .36, 1.15, 1.0], .018),
      oval(`${id}/symbol-head-${i}`, [x, 2.48, 1.12], [.043, .043, .018]),
      box(`${id}/symbol-body-${i}`, [x, 2.36, 1.12], [.04, .065, .018]));
    for (const side of [-1, 1]) children.push(rod(`${id}/symbol-leg-${i}-${side}`,
      [x + side * .023, 2.30, 1.12], [x + side * .04, 2.23, 1.12], .014));
    for (let j = 0; j < 5; j++) children.push(box(`${id}/door-louver-${i}-${j}`, [x, .28 + j * .075, .974], [.38, .018, .018]));
    children.push(box(`${id}/vent-${i}`, [x, 2.88, -.55], [.21, .13, .21]),
      box(`${id}/vent-cap-${i}`, [x, 3.04, -.55], [.27, .025, .27]));
  }
  // Raised WC letters on the side wall, built from the same white material.
  const letter = (name: string, pts: readonly (readonly [number, number])[]) => {
    for (let i = 1; i < pts.length; i++) children.push(rod(`${id}/WC-${name}-${i}`,
      [-2.055, pts[i - 1]![1], pts[i - 1]![0]], [-2.055, pts[i]![1], pts[i]![0]], .025));
  };
  letter("W", [[.65, 2.12], [.53, 1.7], [.4, 1.96], [.27, 1.7], [.15, 2.12]]);
  letter("C", [[-.48, 2.08], [-.27, 2.12], [-.16, 2.02], [-.16, 1.8], [-.27, 1.7], [-.48, 1.74]]);
  for (const x of [-2.12, 2.12]) children.push(rod(`${id}/downpipe-${x}`, [x, .1, -.9], [x, 2.69, -.9], .032));
  return group(id, [5.5, 0, -5.65], children);
}

function picnicTable(id: string, at: Point): SceneryNode {
  const children: SceneryNode[] = [];
  for (let i = 0; i < 7; i++) children.push(box(`${id}/top-${i}`, [0, .74, -.36 + i * .12], [.82, .027, .052]));
  for (const side of [-1, 1]) {
    for (const x of [-.58, .58]) children.push(rod(`${id}/leg-${side}-${x}`, [x, .05, side * .64], [x, .72, -side * .27], .035));
    children.push(box(`${id}/seat-${side}`, [0, .43, side * .62], [.89, .035, .145]));
  }
  return group(id, at, children);
}

export function themeParkScenery(): SceneryGraph {
  const nodes: SceneryNode[] = [
    { kind: "room-shell", id: "shell", materialModel: "room", faces: [], floor: white, wall: white, ceiling: white },
    group("park/terrace", [0, 0, 0], [box("park/terrace/base", [0, -.13, -.65], [8.4, .13, 7.2])]),
    accessStairs(),
    kiosk("park/drinks-stall", [-4.35, 0, -6.55], 0, 0),
    kiosk("park/snack-stall", [-1.4, 0, -6.55], 0, 1),
    kiosk("park/ice-cream-stall", [-7.05, 0, .0], Math.PI / 2, 1),
    toilets(),
    picnicTable("park/picnic-west", [-2.8, 0, 5.55]),
    picnicTable("park/picnic-centre", [1, 0, 5.55]),
    picnicTable("park/picnic-east", [6, 0, 4.65]),
    visitor("park/visitor-stall", [-1.8, 0, -4.65], Math.PI, "watch"),
    visitor("park/visitor-drinks", [-4.4, 0, -4.6], Math.PI, "watch", .97),
    visitor("park/vendor", [-7.1, .12, .0], Math.PI / 2, "watch"),
    visitor("park/visitor-promenade", [2.2, 0, 5.8], -.7, "walk"),
    visitor("park/visitor-facilities", [6, 0, -3.4], -.5, "walk", .93),
    bench("park/bench-west", [-4.4, 0, .4], Math.PI / 2),
    bench("park/bench-front", [-.5, 0, 3.9], Math.PI),
    bench("park/bench-east", [4.9, 0, 1.8], -Math.PI / 2),
    visitor("park/visitor-walking", [-2.5, 0, 3.65], 1.25, "walk"),
    visitor("park/visitor-child", [-1.85, 0, 3.5], 1.05, "walk", .68),
    visitor("park/visitor-pool", [3.95, 0, 3.55], -2.3, "watch"),
    visitor("park/visitor-friend", [4.75, 0, 3.55], -2.0, "watch", .94),
    visitor("park/visitor-seated", [-.72, 0, 3.9], Math.PI, "seated"),
    visitor("park/visitor-west", [-4.3, 0, -1.05], .7, "watch", .97),
    visitor("park/visitor-queue", [-3.0, 0, -4.55], .7, "walk", .92),
  ];
  // A quiet paving rhythm around the ride, with actual recessed expansion joints.
  const paving: SceneryNode[] = [];
  for (let i = 0; i < 23; i++) {
    paving.push(box(`park/paving/front-${i}`, [-7.7 + i * .7, .012, 6.15], [.342, .012, .26]));
    paving.push(box(`park/paving/back-${i}`, [-7.7 + i * .7, .012, -7.55], [.342, .012, .20]));
  }
  for (let i = 0; i < 19; i++) {
    paving.push(box(`park/paving/east-${i}`, [7.95, .012, -7.2 + i * .7], [.26, .012, .342]));
  }
  nodes.push(group("park/paving", [0, 0, 0], paving));
  // Four distinct, fully branched specimens. Two twig generations retain fine
  // silhouettes while keeping the entire set within a 4,096-primitive budget.
  for (const [index, [x, z, scale, seed]] of [
    [-4.9, -3.0, 3.5, 4258], [2.5, -3.65, 3.7, 1978], [-4.75, 2.65, 2.8, 8731], [4.75, -.85, 3.1, 6382],
  ].entries()) {
    const id = `park/oak-${index}`;
    const tree = planOakV2({ key: id, seed, scale_m: scale, twigDepth: 2, sitesPerBough: 6,
      crownWidth: .8, trunkHeight: 1.2, leafScale: 1.05, bark: white, foliage: white }).node;
    nodes.push({ ...tree, place: { units: "metres", position: v(x, 0, z), orientation: yaw(index * 1.7) } });
    const stones: SceneryNode[] = [];
    for (let i = 0; i < 11; i++) {
      const a = i * 2.39996 + seed, radius = .36 + (i % 4) * .15;
      const size = .11 + (i % 3) * .055;
      stones.push({ kind: "cluster", field: "seeded-lobes", id: `${id}/stone-${i}`,
        place: { position: v(Math.cos(a) * radius, size * .4, Math.sin(a) * radius), orientation: yaw(a) },
        lobe: v(size * 1.4, size, size * 1.15), smoothRadius: .015, seed: seed + i,
        lobeCount: 7, anisotropy: 1.8, lobeSpan: .62, lobeSpanSpread: .15, displacement: .6, material: white });
    }
    nodes.push(group(`${id}/rock-garden`, [x, 0, z], stones));
  }
  for (const [i, [x, z, size]] of [[-3.4, 2.7, .45], [1.2, -3.9, .38], [4.5, -2.4, .5], [-3.1, 2.95, .27]].entries()) {
    nodes.push(group(`park/boulder-${i}`, [x, 0, z], [{
      kind: "cluster", field: "seeded-lobes", id: `park/boulder-${i}/body`, place: { position: v(0, size * .45, 0), orientation: yaw(i * 1.8) },
      lobe: v(size * 1.25, size, size * .85), smoothRadius: .035, seed: 912 + i,
      lobeCount: 9, anisotropy: 2.1, lobeSpan: .64, lobeSpanSpread: .12, displacement: .65, material: white,
    }]));
  }
  // Pool-side service details give the surroundings a human scale.
  const details: SceneryNode[] = [];
  for (const x of [3.0, 3.45]) {
    details.push(rod(`park/pool-ladder/${x}/outside`, [x, 0, 3.32], [x, 1.4, 3.32]),
      rod(`park/pool-ladder/${x}/top`, [x, 1.4, 3.32], [x, 1.4, 2.76]),
      rod(`park/pool-ladder/${x}/inside`, [x, 1.4, 2.76], [x, .12, 2.76]));
  }
  for (let i = 0; i < 4; i++) details.push(rod(`park/pool-ladder/rung-${i}`, [3, .27 + i * .23, 2.76], [3.45, .27 + i * .23, 2.76], .022));
  // Ladder is visual scenery: keep its rungs against the far pool wall and out
  // of the slide's impact hood (x=2.16, z=2.4 in world coordinates).
  nodes.push(group("park/pool-ladder", [0, 0, 0], details));
  for (const [i, [x, z]] of [[-4.45, 1.6], [4.85, 2.9]].entries()) {
    nodes.push(group(`park/bin-${i}`, [x, 0, z], [
      { kind: "cylinder", id: `park/bin-${i}/body`, place: { position: v(0, .34, 0) }, radius: .19, halfHeight: .34, edgeRadius: .025, material: white },
      { kind: "torus", id: `park/bin-${i}/rim`, place: { position: v(0, .7, 0) }, majorRadius: .175, minorRadius: .025, material: white },
      box(`park/bin-${i}/lid`, [0, .77, -.04], [.2, .022, .14]),
    ]));
  }
  return { palettes: {}, nodes };
}
