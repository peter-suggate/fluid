import type { SceneryGraph, SceneryMaterial, SceneryNode } from "./scenery-graph";
import { meshAssetHeader } from "./imported-assets/asset-library";
import { planOakV2 } from "./voxel-scenery/oak-v2";
import { planWoodyPlant } from "./voxel-scenery/woody-plants";
import type { Vec3 } from "./model";

// Metres, with the patio at y=2 and the pool recessed below it.
type Point = readonly [number, number, number];
const v = ([x, y, z]: Point): Vec3 => ({ x, y, z });
const material = (colorLinear: Point, surface: SceneryMaterial["surface"]): SceneryMaterial => ({ colorLinear, surface });
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

/** Asset samples are centred in their local field; place their Blender origin
 * on the patio/table while preserving the authored metre scale. */
function imported(id: string, asset: string, anchor: Point, angle = 0): SceneryNode {
  const header = meshAssetHeader(asset);
  const c = header.origin.map((v, i) => v + header.dimensions[i]! * header.cellSize / 2);
  return { kind: "field-program", id,
    program: { ops: [], result: 0, sampledVolume: { asset } },
    place: { units: "metres", position: v([anchor[0] + Math.cos(angle) * c[0]! + Math.sin(angle) * c[2]!,
      anchor[1] + c[1]!, anchor[2] - Math.sin(angle) * c[0]! + Math.cos(angle) * c[2]!]),
      orientation: { w: Math.cos(angle / 2), x: 0, y: Math.sin(angle / 2), z: 0 } },
    material: cream };
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
  // Shared honed stone field: eased edges, mineral pores, UV roughness and
  // source shading normals. Quarter-turns break up the sediment repetition.
  for (let x = -8; x < 8; x += 1) for (let z = -6; z < 6; z += 1) {
    if (x >= -4 && x < 4 && z >= -2 && z < 2) continue;
    const turn = ((x + 8) * 13 + (z + 6) * 7) % 4;
    nodes.push(imported(`patio/${x}/${z}`, "honed-travertine-paver", [x + .5, 2, z + .5], turn * Math.PI / 2));
  }
  // Individually eased coping stones and glazed mosaics retain their native
  // 6.25 mm source sampling, independently of the fluid boundary.
  for (const side of [-1, 1]) {
    for (let i = 0; i < 16; i++) nodes.push(imported(`pool/coping-z-${side}/${i}`,
      "limestone-coping", [-3.75 + i * .5, 2, side * 2.15]));
    for (let i = 0; i < 8; i++) {
      nodes.push(imported(`pool/coping-x-${side}/${i}`, "limestone-coping",
        [side * 4.15, 2, -1.75 + i * .5], Math.PI / 2),
      imported(`pool/tile-z-${side}/${i}`, "ceramic-waterline",
        [-3.5 + i, 1.68, side * 1.995], side < 0 ? 0 : Math.PI));
    }
    for (let i = 0; i < 4; i++) nodes.push(imported(`pool/tile-x-${side}/${i}`,
      "ceramic-waterline", [side * 3.995, 1.68, -1.5 + i], -side * Math.PI / 2));
    for (const end of [-1, 1]) nodes.push(imported(`pool/coping-corner/${side}/${end}`,
      "limestone-coping-corner", [side * 4.15, 2, end * 2.15]));
  }
  // Blender assembly follows the original board's 4.5° slope and encloses
  // its simulation collider; the ladder treads also have SolidWorld support.
  nodes.push(imported("diving/assembly", "diving-board-ladder", [-4.1, 2, 0]),
    imported("pool/stainless-ladder", "pool-stainless-ladder", [4, 2, 0]));
  nodes.push(imported("lounger/one", "teak-lounger", [-.8, 2.03, -3.65], Math.PI),
    imported("lounger/two", "teak-lounger", [1.1, 2.03, -3.65], Math.PI),
    imported("lounger/three", "teak-lounger", [3, 2.03, -3.65], Math.PI),
    imported("parasol", "garden-parasol", [1.1, 2.03, -5.05]),
    imported("cooler", "insulated-cooler", [5.7, 2.03, -1.6]),
    imported("refreshments/service", "refreshment-set", [5.65, 2.855, -3.5]),
    imported("refreshments/table", "teak-serving-table", [5.65, 2.03, -3.5]));
  nodes.push(guest("guest/host", 5.2, 1.2, material([.48, .26, .14], "organic"), coral, -1.2),
    guest("guest/friend", 6.3, .6, material([.24, .11, .055], "organic"), cream, -2.3),
    guest("guest/poolside", 1.5, 3.5, material([.72, .43, .27], "organic"), teal, Math.PI),
    guest("guest/patio", 3, 3.8, material([.38, .19, .09], "organic"), coral, -2.2));
  // Lawn strips, timber privacy fence, clipped shrubs and a rear house facade.
  // Each metre of turf references one shared 3.125 mm source volume.
  // The 50 mm fluid grid never acquires grass blades or texture samples.
  for (let x = -10; x < 10; x++) for (let z = -8; z < 8; z++) {
    if (x >= -8 && x < 8 && z >= -6 && z < 6) continue;
    const turn = ((x + 10) * 7 + (z + 8) * 11) % 4;
    nodes.push(imported(`lawn/turf/${x}/${z}`, "living-turf", [x + .5, 2.01, z + .5], turn * Math.PI / 2));
  }
  for (let i = 0; i < 10; i++) nodes.push(imported(`fence/rear-${i}`,
    "cedar-fence-panel", [-8.1 + i * 1.8, 2, -7.8]));
  for (let i = 0; i < 9; i++) nodes.push(imported(`fence/side-${i}`,
    "cedar-fence-panel", [9, 2, -7.1 + i * 1.8], -Math.PI / 2));
  nodes.push(box("fence/rear-end", [9, 2.94, -7.825], [.0475, .94, .0475]),
    box("fence/side-end", [9.025, 2.94, 8.2], [.0475, .94, .0475]));
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
  nodes.push(box("house/wall", [-9.8, 3.65, .5], [.25, 1.65, 5.25], cream));
  for (const z of [-3, .5, 4]) nodes.push(imported(`house/bay-${z}`,
    "stucco-window-bay", [-9.42, 2, z], Math.PI / 2));
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
