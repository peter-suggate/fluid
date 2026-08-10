import type {
  SceneryGroupNode,
  SceneryMaterial,
  SceneryRecursiveShapeNode,
  SceneryTaperedSweepClusterNode,
} from "./scenery-graph";

const TREE_MATERIAL: SceneryMaterial = { palette: "clay", value: 0.955, surface: "architectural" };
const FOLIAGE_MATERIAL: SceneryMaterial = { palette: "clay", value: 0.975, surface: "foliage" };

type SweepControl = readonly [x: number, y: number, z: number, radius: number];

function structuralSweep(id: string, controls: readonly SweepControl[]): SceneryTaperedSweepClusterNode {
  const xs = controls.map(([x]) => x), ys = controls.map(([, y]) => y), zs = controls.map(([, , z]) => z);
  const centre = {
    x: (Math.min(...xs) + Math.max(...xs)) * 0.5,
    y: (Math.min(...ys) + Math.max(...ys)) * 0.5,
    z: (Math.min(...zs) + Math.max(...zs)) * 0.5,
  };
  const local = controls.map(([x, y, z, radius]) => ({
    position: { x: x - centre.x, y: y - centre.y, z: z - centre.z }, radius,
  }));
  // A spherical envelope is conservative for every control sphere and avoids
  // the reduced radial clearance an anisotropic envelope has near two extrema.
  const envelopeRadius = Math.max(...local.map((point) => Math.hypot(
    point.position.x, point.position.y, point.position.z,
  ) + point.radius + 0.008));
  return {
    kind: "cluster",
    field: "tapered-sweep",
    id,
    group: id,
    tags: ["tree", "structure", "individual-shape"],
    place: { units: "metres", position: centre },
    lobe: { x: envelopeRadius, y: envelopeRadius, z: envelopeRadius },
    smoothRadius: 0.006,
    seed: 0x71_ee,
    points: local,
    material: TREE_MATERIAL,
  };
}

function crown(
  id: string,
  position: readonly [number, number, number],
  radii_m: readonly [number, number, number],
  seed: number,
  pattern: "fan" | "ring" | "cap",
): SceneryRecursiveShapeNode {
  const root: SceneryRecursiveShapeNode = {
    kind: "recursive-shape",
    family: "foliage-pad",
    id,
    seed,
    place: { units: "metres", position: { x: position[0], y: position[1], z: position[2] } },
    form: {
      radii_m,
      flatten: 0.88,
      edgeLobes: 8,
      lobeDepth: 0.52,
      topBias: 0.66,
      undersideCut: 0.58,
      blockJitter: 0.42,
      density: {
        clusterPeriod_m: 0.1550208,
        dotSpacing_m: 0.0275616,
        threshold: 0.5002,
        clusterWeight: 0.72,
        detailWeight: 0.28,
        interiorBias: 0.093,
      },
    },
    split: {
      pattern,
      childCount: 4,
      childScale: 0.62,
      spread: 0.76,
      overlap: 0.34,
      verticalBias: 0.40,
      flattening: 0.32,
      jitter: 0.30,
    },
    material: FOLIAGE_MATERIAL,
  };
  return root;
}

/**
 * The hero's scene-owned cloud tree. Geometry is explicit after construction:
 * every structural sweep and active foliage frontier pad has a stable id and
 * is independently addressable in Shape Lab.
 */
export function heroGardenCloudTree(): SceneryGroupNode {
  const structure: SceneryGroupNode = {
    kind: "group",
    id: "tree/structure",
    children: [
      structuralSweep("tree/structure/trunk", [
        [0, 0.00, 0, 0.052], [-0.018, 0.125, 0.006, 0.047], [0.006, 0.265, 0.012, 0.039], [0.010, 0.382, 0.018, 0.031],
      ]),
      structuralSweep("tree/structure/arm-left", [
        [0.00, 0.24, 0.01, 0.036], [-0.10, 0.34, 0.00, 0.029], [-0.24, 0.43, -0.01, 0.021], [-0.39, 0.48, 0.03, 0.014],
      ]),
      structuralSweep("tree/structure/arm-centre", [
        [0.00, 0.30, 0.01, 0.033], [-0.03, 0.42, 0.02, 0.024], [-0.08, 0.54, 0.02, 0.014],
      ]),
      structuralSweep("tree/structure/arm-right", [
        [0.01, 0.27, 0.02, 0.034], [0.10, 0.375, 0.04, 0.026], [0.23, 0.45, 0.08, 0.019], [0.35, 0.49, 0.10, 0.013],
      ]),
      structuralSweep("tree/structure/arm-back", [
        [0.00, 0.31, 0.02, 0.031], [-0.02, 0.40, 0.12, 0.024], [-0.02, 0.48, 0.24, 0.014],
      ]),
      structuralSweep("tree/structure/arm-front", [
        [0.00, 0.28, 0.00, 0.032], [-0.08, 0.38, -0.11, 0.023], [-0.15, 0.45, -0.22, 0.014],
      ]),
    ],
  };
  const foliage: SceneryGroupNode = {
    kind: "group",
    id: "tree/foliage",
    children: [
      // One loose acceleration domain around the whole crown. Its low-frequency
      // octave is contrast-shaped into density clusters and its high-frequency
      // octave supplies the foliage breakup; neither the domain nor a sphere
      // lattice contributes geometry.
      crown("tree/foliage/canopy", [-0.11, 0.57, 0.02], [0.64, 0.30, 0.50], 0x10a2, "ring"),
    ],
  };
  return {
    kind: "group",
    id: "tree",
    tags: ["tree", "cloud-tree", "shape-lab"],
    place: {
      units: "metres",
      anchor: "terrain",
      ground: [0.53, -0.15],
      position: { x: 0.53, y: 0, z: -0.15 },
    },
    children: [structure, foliage],
  };
}
