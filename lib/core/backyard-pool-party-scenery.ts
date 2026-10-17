import type { SceneryGraph, SceneryMaterial, SceneryNode } from "./scenery-graph";
import { planOakV2 } from "./voxel-scenery/oak-v2";
import { planWoodyPlant } from "./voxel-scenery/woody-plants";
import type { Vec3 } from "./model";

// Metres, with the patio at y=2 and the pool recessed below it.
type Point = readonly [number, number, number];
const v = ([x, y, z]: Point): Vec3 => ({ x, y, z });
const material = (colorLinear: Point, surface: SceneryMaterial["surface"]): SceneryMaterial => ({ colorLinear, surface });
const stone = material([.68, .61, .49], "stone");
const wood = material([.24, .12, .055], "wood");
const metal = material([.42, .46, .48], "brushed-metal");
const cream = material([.88, .83, .70], "architectural");
const teal = material([.025, .32, .35], "none");
const coral = material([.73, .16, .09], "none");
const leaf = material([.07, .19, .035], "foliage");
const box = (id: string, at: Point, half: Point, mat = wood): SceneryNode =>
  ({ kind: "box", id, place: { units: "metres", position: v(at) }, halfSize: v(half), material: mat });
const oval = (id: string, at: Point, radius: Point, mat: SceneryMaterial): SceneryNode =>
  ({ kind: "ellipsoid", id, place: { units: "metres", position: v(at) }, radius: v(radius), material: mat });
const rod = (id: string, from: Point, to: Point, radius = .035, mat = metal): SceneryNode =>
  ({ kind: "capsule", id, place: { units: "metres" }, from: v(from), to: v(to), radius, material: mat });
const cylinder = (id: string, at: Point, radius: number, halfHeight: number, mat: SceneryMaterial): SceneryNode =>
  ({ kind: "cylinder", id, place: { units: "metres", position: v(at) }, radius, halfHeight, material: mat });

function lounger(id: string, x: number, z: number, fabric: SceneryMaterial): SceneryNode[] {
  const nodes: SceneryNode[] = [box(`${id}/frame`, [x, 2.36, z], [.38, .06, .98]),
    box(`${id}/cushion`, [x, 2.45, z + .26], [.35, .06, .65], fabric),
    { kind: "box", id: `${id}/backrest`, place: { units: "metres", position: v([x, 2.70, z - .66]),
      orientation: { w: Math.cos(-.48 / 2), x: Math.sin(-.48 / 2), y: 0, z: 0 } },
      halfSize: v([.35, .065, .43]), material: fabric },
    box(`${id}/towel`, [x, 2.52, z + .55], [.31, .025, .32], cream)];
  for (const side of [-1, 1]) for (const end of [-1, 1]) nodes.push(
    rod(`${id}/leg-${side}-${end}`, [x + side * .29, 2.04, z + end * .73], [x + side * .29, 2.36, z + end * .73], .04, wood));
  return nodes;
}

function guest(id: string, x: number, z: number, skin: SceneryMaterial, clothing: SceneryMaterial, angle: number): SceneryNode {
  const nodes: SceneryNode[] = [
    oval(`${id}/torso`, [0, 1.20, 0], [.21, .29, .12], clothing),
    oval(`${id}/shorts`, [0, .91, 0], [.19, .17, .13], clothing),
    rod(`${id}/neck`, [0, 1.42, 0], [0, 1.52, 0], .065, skin),
    oval(`${id}/head`, [0, 1.65, 0], [.10, .14, .105], skin),
    oval(`${id}/hair`, [0, 1.73, -.015], [.105, .075, .105], wood),
    oval(`${id}/nose`, [0, 1.64, .105], [.025, .03, .025], skin),
  ];
  for (const side of [-1, 1]) {
    nodes.push(rod(`${id}/thigh-${side}`, [side * .10, .86, 0], [side * .12, .48, .015], .078, skin),
      rod(`${id}/calf-${side}`, [side * .12, .48, .015], [side * .14, .10, 0], .052, skin),
      oval(`${id}/sandal-${side}`, [side * .14, .045, .06], [.07, .035, .14], wood),
      rod(`${id}/upper-arm-${side}`, [side * .20, 1.37, 0], [side * .28, 1.08, .06], .055, skin),
      rod(`${id}/forearm-${side}`, [side * .28, 1.08, .06], [side * .22, 1.02, .27], .042, skin));
  }
  nodes.push(cylinder(`${id}/drink`, [.22, 1.12, .27], .045, .085, cream));
  return { kind: "group", id, tags: ["guest"], place: { units: "metres", position: v([x, 2, z]),
    orientation: { w: Math.cos(angle / 2), x: 0, y: Math.sin(angle / 2), z: 0 } }, children: nodes };
}

export function backyardPoolPartyScenery(): SceneryGraph {
  const nodes: SceneryNode[] = [{ kind: "room-shell", id: "shell", materialModel: "room", faces: [], floor: cream, wall: cream, ceiling: cream }];
  // A warm stone terrace, divided around the cavity. Fine seams break up its scale.
  for (let x = -8; x < 8; x += 1) for (let z = -6; z < 6; z += 1) {
    if (x >= -4 && x < 4 && z >= -2 && z < 2) continue;
    nodes.push(box(`patio/${x}/${z}`, [x + .5, 2.015, z + .5], [.494, .015, .494], stone));
  }
  // Coping lies outside the physical water cavity; blue tile marks the waterline.
  for (const side of [-1, 1]) {
    nodes.push(box(`pool/coping-x-${side}`, [side * 4.15, 2.07, 0], [.15, .07, 2.3], cream),
      box(`pool/coping-z-${side}`, [0, 2.07, side * 2.15], [4, .07, .15], cream),
      box(`pool/tile-x-${side}`, [side * 3.995, 1.82, 0], [.012, .14, 2], teal),
      box(`pool/tile-z-${side}`, [0, 1.82, side * 1.995], [4, .14, .012], teal));
  }
  // Tall anchored steel support, diagonal braces, ladder and rear handrails.
  for (const side of [-1, 1]) {
    const z = side * .46;
    nodes.push(box(`diving/foot-${side}`, [-5.75, 2.10, z], [.38, .08, .19], metal),
      rod(`diving/upright-${side}`, [-5.75, 2.12, z], [-5.75, 5.18, z], .075),
      rod(`diving/brace-${side}`, [-6.45, 2.10, z], [-5.75, 4.7, z], .045),
      rod(`diving/ladder-${side}`, [-7.05, 2.1, side * .34], [-6.0, 5.35, side * .34], .04),
      rod(`diving/handrail-${side}`, [-6.0, 5.35, z], [-6.0, 6.20, z], .035),
      rod(`diving/rail-${side}`, [-6.0, 6.20, z], [-5.15, 6.12, z], .035));
  }
  for (let i = 0; i < 11; i++) {
    const t = (i + 1) / 11;
    nodes.push(rod(`diving/rung-${i}`, [-7.05 + t * 1.05, 2.10 + t * 3.25, -.34],
      [-7.05 + t * 1.05, 2.10 + t * 3.25, .34], .045));
  }
  nodes.push(box("diving/crossbeam", [-5.75, 5.17, 0], [.16, .08, .56], metal));
  // Pool ladder at the opposite end, away from the ball's landing zone.
  for (const z of [-.4, .4]) {
    nodes.push(rod(`pool/ladder/${z}`, [3.7, .65, z], [3.7, 2.7, z], .035),
      rod(`pool/ladder-top/${z}`, [3.7, 2.7, z], [4.4, 2.7, z], .035),
      rod(`pool/ladder-return/${z}`, [4.4, 2.7, z], [4.4, 2.1, z], .035));
  }
  for (let i = 0; i < 4; i++) nodes.push(rod(`pool/step-${i}`, [3.7, .8 + i * .35, -.4], [3.7, .8 + i * .35, .4], .045));
  nodes.push(...lounger("lounger/one", -.8, -3.65, teal), ...lounger("lounger/two", 1.1, -3.65, coral),
    ...lounger("lounger/three", 3, -3.65, cream));
  // Garden parasol, table, refreshments and a cooler.
  nodes.push(cylinder("parasol/base", [1.1, 2.08, -5.05], .34, .08, stone),
    rod("parasol/pole", [1.1, 2.12, -5.05], [1.1, 4.9, -5.05], .045, wood),
    { kind: "cone", id: "parasol/canopy", place: { units: "metres", position: v([1.1, 4.65, -5.05]) },
      halfHeight: .25, baseRadius: 1.65, topRadius: .045, material: cream },
    box("refreshments/top", [5.65, 2.8, -3.5], [.65, .055, 1.25]),
    box("cooler/body", [5.7, 2.28, -1.6], [.37, .25, .25], teal),
    box("cooler/lid", [5.7, 2.56, -1.6], [.39, .045, .27], cream));
  for (const x of [5.15, 6.15]) for (const z of [-4.5, -2.5]) nodes.push(rod(`table/leg/${x}/${z}`, [x, 2, z], [x, 2.8, z], .055, wood));
  for (let i = 0; i < 5; i++) nodes.push(cylinder(`refreshments/cup-${i}`, [5.4, 2.96, -4.3 + i * .35], .055, .10, i % 2 ? coral : cream));
  nodes.push(cylinder("refreshments/pitcher", [5.85, 3.04, -3.5], .12, .18, teal));
  nodes.push(guest("guest/host", 5.2, 1.2, material([.48, .26, .14], "organic"), coral, -1.2),
    guest("guest/friend", 6.3, .6, material([.24, .11, .055], "organic"), cream, -2.3),
    guest("guest/poolside", 1.5, 3.5, material([.72, .43, .27], "organic"), teal, Math.PI),
    guest("guest/patio", 3, 3.8, material([.38, .19, .09], "organic"), coral, -2.2));
  // Lawn strips, timber privacy fence, clipped shrubs and a rear house facade.
  nodes.push(box("lawn/rear", [0, 1.97, -7], [9, .05, 1], leaf),
    box("lawn/right", [8.75, 1.97, 0], [.75, .05, 8], leaf),
    box("lawn/front", [0, 1.97, 7], [9, .05, 1], leaf),
    box("lawn/left", [-8.75, 1.97, 0], [.75, .05, 8], leaf));
  for (let i = 0; i < 60; i++) nodes.push(box(`fence/picket-${i}`, [-8.85 + i * .3, 2.95, -7.8], [.14, .95, .045]));
  for (let i = 0; i < 50; i++) nodes.push(box(`fence/side-${i}`, [9.4, 2.95, -7.5 + i * .3], [.045, .95, .14]));
  // Use the nursery's actual branching and foliage recipes. The low border
  // leaves the diving board visible; taller specimens frame the outer lawn.
  for (let i = 0; i < 9; i++) {
    const species = (["hebe", "azalea", "juniper", "boxwood"] as const)[i % 4]!;
    const shrub = planWoodyPlant({ key: `garden/${species}-${i}`, species,
      seed: 9411 + i * 137, scale_m: .85 + .1 * (i % 3), bark: wood, foliage: leaf }).node;
    nodes.push({ ...shrub, place: { units: "metres", position: v([-7.2 + i * 1.7, 2.02, -6.8]),
      orientation: { w: Math.cos(i * 1.2), x: 0, y: Math.sin(i * 1.2), z: 0 } } });
  }
  const oak = planOakV2({ key: "garden/oak", seed: 4258, scale_m: 4.1,
    twigDepth: 2, sitesPerBough: 6, crownWidth: .75, trunkHeight: 1.2,
    leafScale: 1.05, bark: wood, foliage: leaf }).node;
  nodes.push({ ...oak, place: { units: "metres", position: v([-7.5, 2.02, -6.9]) } });
  for (const [species, x, z, scale, seed] of [
    ["birch", 7.7, -6.6, 4, 1978], ["pine", 8.65, 4.8, 3.5, 8731],
  ] as const) {
    const tree = planWoodyPlant({ key: `garden/${species}`, species, seed,
      scale_m: scale, bark: wood, foliage: leaf }).node;
    nodes.push({ ...tree, place: { units: "metres", position: v([x, 2.02, z]) } });
  }
  nodes.push(box("house/wall", [-9.8, 3.65, .5], [.25, 1.65, 5.6], cream),
    box("house/eaves", [-9.5, 5.36, .5], [.7, .12, 5.8], wood));
  for (const z of [-2.5, 1, 4]) nodes.push(box(`house/window-${z}`, [-9.53, 3.8, z], [.025, .85, 1.15], material([.065, .13, .17], "architectural")));
  // Festoon lights hang beyond the pool, never across the diving trajectory.
  for (const x of [-7.5, 7.5]) nodes.push(rod(`lights/post-${x}`, [x, 2, 5.4], [x, 5.15, 5.4], .055, wood));
  for (let i = 0; i < 24; i++) {
    const x = -7.5 + i * .625, y = 5.15 - .65 * Math.sin(Math.PI * i / 24);
    const nextY = 5.15 - .65 * Math.sin(Math.PI * (i + 1) / 24);
    nodes.push(rod(`lights/wire-${i}`, [x, y, 5.4], [x + .625, nextY, 5.4], .012, wood));
    if (i % 2 === 0) nodes.push(oval(`lights/bulb-${i}`, [x, y - .09, 5.4], [.055, .075, .055],
      { colorLinear: [1, .68, .27], emission: 1.5, surface: "ceramic" }));
  }
  return { palettes: {}, nodes };
}
