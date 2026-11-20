import type { SceneryGraph, SceneryMaterial, SceneryNode } from "./scenery-graph";
import { meshAssetHeader } from "./imported-assets/asset-library";
import type { Vec3 } from "./model";

// Metres, with the patio at y=2 and the pool recessed below it.
type Point = readonly [number, number, number];
const v = ([x, y, z]: Point): Vec3 => ({ x, y, z });
const material = (colorLinear: Point, surface: SceneryMaterial["surface"]): SceneryMaterial => ({ colorLinear, surface });
const wood = material([.24, .12, .055], "wood");
const cream = material([.88, .83, .70], "architectural");
const box = (id: string, at: Point, half: Point, mat = wood): SceneryNode =>
  ({ kind: "box", id, place: { units: "metres", position: v(at) }, halfSize: v(half), material: mat });
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

export function backyardPoolPartyScenery(cellSize_m: number): SceneryGraph {
  const nodes: SceneryNode[] = [{ kind: "room-shell", id: "shell", materialModel: "room", faces: [], floor: cream, wall: cream, ceiling: cream }];
  // An inset concrete bed and the plaster inside the basin are editable
  // scenery. The paving, coping and mosaics supply its exposed finish.
  const plaster = material([.51, .63, .61], "architectural");
  const mortar = material([.32, .30, .25], "architectural");
  // Put the plaster finish on the wet side of the structural slab. Coincident
  // mortar, plaster and voxel faces fought for depth in the refracted image.
  const finish = .012;
  const floorTop = Math.round(.2 / cellSize_m) * cellSize_m + finish;
  nodes.push(box("pool/plaster-floor", [0, floorTop / 2, 0], [4, floorTop / 2, 2], plaster));
  for (const side of [-1, 1]) {
    nodes.push(box(`pool/plaster-end/${side}`, [side * 6, .93, 0], [2, .93, 6], mortar),
      box(`pool/plaster-side/${side}`, [0, .93, side * 4], [4, .93, 2], mortar),
      box(`pool/finish-end/${side}`, [side * (4 - finish / 2), .935, 0], [finish / 2, .735, 2], plaster),
      box(`pool/finish-side/${side}`, [0, .935, side * (2 - finish / 2)], [4 - finish, .735, finish / 2], plaster));
  }
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
  nodes.push(imported("lounger/one", "teak-lounger", [-1.35, 2.03, -4.25], 2.76),
    imported("lounger/two", "teak-lounger", [.6, 2.03, -3.67], 3.35),
    imported("lounger/three", "teak-lounger", [3.3, 2.03, -4.40], 3.0),
    imported("parasol", "garden-parasol", [1.1, 2.03, -5.05]),
    imported("cooler", "insulated-cooler", [5.7, 2.03, -1.6]),
    imported("refreshments/service", "refreshment-set", [5.65, 2.855, -3.5]),
    imported("refreshments/table", "teak-serving-table", [5.65, 2.03, -3.5]));
  nodes.push(imported("guest/host", "terrace-host", [4.52, 2.03, -3.5], Math.PI / 2),
    imported("guest/reader", "terrace-reader", [1.02, 2.08, -3.68], 1.82));
  // The host grips the narrow ends of a serving board just above the table.
  nodes.push(box("refreshments/serving-board", [5.105, 3.022, -3.5], [.14, .014, .212], wood));
  for (const [i, dx, dz] of [[0, -.052, -.11], [1, .047, -.043], [2, -.025, .060]] as const) {
    nodes.push({ kind: "ellipsoid", id: `refreshments/citrus-${i}`,
      place: { units: "metres", position: v([5.105 + dx, 3.072, -3.5 + dz]) },
      radius: v([.048, .037, .034]), material: material([.83, .48, .043], "organic") });
  }
  // Each metre of turf references one shared 3.125 mm source volume.
  // The fluid lattice remains independent of grass blades and texture samples.
  for (let x = -10; x < 10; x++) for (let z = -8; z < 8; z++) {
    if (x >= -8 && x < 8 && z >= -6 && z < 6) continue;
    const turn = ((x + 10) * 7 + (z + 8) * 11) % 4;
    nodes.push(imported(`lawn/turf/${x}/${z}`, "living-turf", [x + .5, 2.01, z + .5], turn * Math.PI / 2));
  }
  for (let i = 1; i < 10; i++) nodes.push(imported(`fence/rear-${i}`,
    "cedar-fence-panel", [-8.1 + i * 1.8, 2, -7.8]));
  for (let i = 0; i < 9; i++) nodes.push(imported(`fence/side-${i}`,
    "cedar-fence-panel", [9, 2, -7.1 + i * 1.8], -Math.PI / 2));
  nodes.push(box("fence/rear-end", [9, 2.94, -7.825], [.0475, .94, .0475]),
    box("fence/side-end", [9.025, 2.94, 8.2], [.0475, .94, .0475]));
  nodes.push(imported("grounds/valley", "valley-farmland", [0, 2, 0]),
    imported("house/pavilion", "garden-pavilion", [-9.15, 2, .25], Math.PI / 2),
    imported("neighbors/orchard-house", "neighbor-cottage", [1.8, .93, -13.7], -.13),
    imported("neighbors/lane-house", "lane-bungalow", [17.6, 2.05, 1.2], 1.73),
    imported("neighbors/hill-house", "hill-saltbox", [-6.4, 3.18, 17.4], 2.96));
  // Shared native tree volumes retain branch gaps and the source leaf palettes.
  // Positions offset the crowns in depth and leave a view through to the roofs.
  const trees: readonly [string, Point, number][] = [
    ["forest-tree-1", [-7.55, 2, -7.2], 1.15],
    ["forest-alder-1", [8.1, 2, -6.45], -.37],
    ["forest-tree-1", [10.7, 1.94, 7.9], 3.51],
    ["forest-alder-1", [-17.1, 2.05, 7.2], 2.08],
    ["forest-tree-1", [-8.8, .63, -16.6], 2.63],
    ["forest-alder-1", [11.2, .20, -20.2], .68],
    ["forest-tree-1", [-17.1, 2.95, 17.4], .52],
    ["forest-alder-1", [13.2, 3.55, 19.7], 3.12],
  ];
  trees.forEach(([asset, anchor, angle], i) =>
    nodes.push(imported(`garden/canopy-${i}`, asset, anchor, angle)));
  // Shelterbelts around the neighboring plots sit on the authored lawn slope.
  // Leave breaks for the lane, the houses and views of the open hillside.
  const lawnAt = (x:number,z:number) => {
    const amount = Math.min(1, Math.max(0, Math.max(Math.abs(x)-10, Math.abs(z)-8)/10));
    return 1.91 + amount*(.4*Math.sin(x*.16)+.36*Math.cos(z*.27)+.08*z);
  };
  const belt = [[-22, -12], [-23.8,-15.8], [-26.6,-18], [-22,-23],
    [-14,23], [-17.1,23.8], [-19,21.7], [5.6,24.1], [8,24.9],
    [21.8,-12.8], [22.7,-16.7], [26.1,-19.1], [16,-23.7]] as const;
  belt.forEach(([x,z], i) => {
    nodes.push(imported(`neighbors/shelterbelt-${i}`, i % 3 === 0 ? "forest-cedar-0" : i % 3 === 1 ? "forest-tree-1" : "forest-alder-1",
      [x,lawnAt(x,z)-.18,z], i*2.40));
  });
  // Copse roots follow the Blender terrain. Small gaps between
  // groups leave the long fields and their changing ridgeline visible.
  const copses: readonly Point[] = [
    [-37, 1.74, 7], [-40, .88, 6], [-43, 1.13, 9],
    [-36, 1.28, -16], [-38, 1.93, -18], [-41, 2.02, -20],
    [-7, -2.50, -41], [-10, -2.31, -42],
    [-61, 10.51, 10], [-62, 11.12, 13], [-58, 9.14, 12],
    [-6, 13.48, -62], [-3, 12.27, -60], [1, 11.20, -59],
    [32, 3.09, 17], [35, 4.04, 20], [7, 7.55, 38], [10, 8.20, 40],
  ];
  const copseTrees = ["forest-alder-1", "forest-tree-1", "forest-cedar-0", "forest-tree-1"];
  copses.forEach((anchor, i) =>
    nodes.push(imported(`grounds/copse-${i}`, copseTrees[i % copseTrees.length]!, anchor, i * 2.17)));
  // Foundation planting follows the cottage gardens and the bend in the lane.
  const borders=[[5.8,-15.0],[6.5,-13.1],[6.6,-11.4],[-2.6,-15.3],
    [14.2,5.3],[12.4,4.9],[20.7,-2.8],[-11.2,17.1],[-11.6,15.1],[-4.3,21.3]] as const;
  borders.forEach(([x,z],i)=>{
    nodes.push(imported(`neighbors/shrub-${i}`,"forest-broadleaf-0",[x,lawnAt(x,z)-.08,z],i*1.75));
  });
  const lavender: readonly Point[] = [
    [-6.75, 2.02, -6.68], [-5.95, 2.02, -6.55], [-5.18, 2.02, -6.84],
    [-3.25, 2.02, -6.71], [-2.55, 2.02, -6.65], [4.65, 2.02, -6.64],
    [5.37, 2.02, -6.77], [6.35, 2.02, -6.60], [7.98, 2.02, -4.64],
    [8.12, 2.02, 2.98], [8.26, 2.02, 3.74], [8.10, 2.02, 4.57],
  ];
  lavender.forEach((anchor, i) => nodes.push(imported(`garden/lavender-${i}`, "lavender-border", anchor, i * 1.89)));
  // Fronds spread into the shaded planting beds at the pavilion returns.
  for (const [i, anchor] of ([[-9.1, 2.0, -5.8], [-8.65, 2.0, -6.55],
    [-8.9, 2.0, 5.14], [-9.5, 2.0, 5.75], [8.16, 2, -5.70]] as const).entries()) {
    nodes.push(imported(`garden/fern-${i}`, "forest-fern-0", anchor, .8 + i * 1.73));
  }
  for (let i = 0; i < 5; i++) nodes.push(box(`side-path/step-${i}`,
    [-8.05, 2.035, -5.55-i*.64], [.49,.065,.24], cream));
  nodes.push(imported("side-path/gate", "cedar-fence-panel", [-8.96,2,-7.06], .94));
  nodes.push(imported("lights/festoon", "terrace-festoon", [0, 2, -6.12]));
  return { palettes: {}, solidBoundarySurface: "scenery", nodes };
}
